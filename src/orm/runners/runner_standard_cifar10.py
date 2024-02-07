from __future__ import annotations

import argparse
from collections import namedtuple
import logging
import random
import sys
from typing import NamedTuple

import numpy as np
import torch
import wandb

from orm.models import get_resnet_embedding
from orm.utils import (
    AverageMeter,
    calc_accuracy,
    get_config_from_file,
)
from orm.utils.dataset import load_dataset

Metrics = namedtuple("Metrics", ["loss", "acc_top1", "acc_top5"])

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ArgparseLogger(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore
        logging.error(message)
        super().error(message)


def get_parser() -> argparse.ArgumentParser:
    parser = ArgparseLogger(description="Standard Training for modified CIFAR10")
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="random seed",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--eta-min",
        type=float,
        default=1e-5,
        help="minimum learning rate for the scheduler",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="weight decay for the optimizer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of epochs to run the model",
    )

    return parser


def setup_logging() -> logging.Logger:
    import os
    import time

    ct = time.strftime("%Y-%d-%h-%H:%M:%S", time.gmtime(time.time()))
    filename = f"logs/{ct}/cifar10_run.log"
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


def train_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> NamedTuple:
    model.train()
    train_losses, train_top1, train_top5 = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    for _, batch in enumerate(train_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        train_prec1, train_prec5 = calc_accuracy(outputs.data, labels.data, topk=(1, 5))
        train_losses.update(loss.item(), images.size(0))
        train_top1.update(train_prec1.item(), images.size(0))
        train_top5.update(train_prec5.item(), images.size(0))
    train_metrics = Metrics(train_losses.avg, train_top1.avg, train_top5.avg)
    return train_metrics


def validate_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> NamedTuple:
    model.eval()
    valid_losses, valid_top1, valid_top5 = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    with torch.no_grad():
        for _, batch in enumerate(valid_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            valid_loss = criterion(outputs, labels)

            valid_prec1, valid_prec5 = calc_accuracy(
                outputs.data, labels.data, topk=(1, 5)
            )
            valid_losses.update(valid_loss.item(), images.size(0))
            valid_top1.update(valid_prec1.item(), images.size(0))
            valid_top5.update(valid_prec5.item(), images.size(0))
    valid_metrics = Metrics(valid_losses.avg, valid_top1.avg, valid_top5.avg)
    return valid_metrics


def test(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> NamedTuple:
    return validate_step(model, criterion, test_loader, device)


def log_on_wandb(
    title: str,
    metrics: NamedTuple,
    epoch: int | None = None,
) -> None:
    log_metrics = {
        f"{title}/epochs": epoch,
        f"{title}/loss": metrics.loss,  # type: ignore
        f"{title}/acc_top1": metrics.acc_top1,  # type: ignore
        f"{title}/acc_top5": metrics.acc_top5,  # type: ignore
    }
    wandb.log(log_metrics)  # type: ignore


def log_metrics(
    logger: logging.Logger, title: str, metrics: NamedTuple, epoch: int | None = None
) -> None:
    msg = "Epoch {} <{}> : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
        epoch,
        title,
        metrics.loss,  # type: ignore
        metrics.acc_top1,  # type: ignore
        metrics.acc_top5,  # type: ignore
    )
    logger.info(msg)


def set_seed(rand_seed: int) -> None:
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)


if __name__ == "__main__":
    logger = setup_logging()
    config = get_config_from_file(logger=logger)
    parser = get_parser()

    try:
        args = parser.parse_args(sys.argv[1:])  # type: ignore
    except:
        print("Error handling arguments")
        raise

    set_seed(args.seed)
    (
        train_queue,
        meta_queue,
        valid_queue,
        outlier_valid_queue,
        test_queue,
        train_transform,
        meta_transform,
        valid_transform,
    ) = load_dataset(config)

    model = get_resnet_embedding().to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )

    def itr_merge(*itrs):  # type: ignore
        for itr in itrs:
            yield from itr

    wandb_log = True

    if wandb_log:
        wandb.init(  # type: ignore
            project="OrmModel", group="standard_model", config=config
        )

    for epoch in range(args.epochs):
        train_metrics = train_step(
            model,
            criterion,
            optimizer,
            itr_merge(train_queue, meta_queue),
            device=DEVICE,
        )
        log_metrics(logger, "Train", train_metrics, epoch)

        valid_metrics = validate_step(
            model,
            criterion,
            valid_queue,
            device=DEVICE,
        )
        log_metrics(logger, "Valid", valid_metrics, epoch)

        outlier_valid_metric = validate_step(
            model,
            criterion,
            outlier_valid_queue,
            device=DEVICE,
        )
        log_metrics(logger, "Outlier Valid", outlier_valid_metric, epoch)

        if wandb_log:
            log_on_wandb("train", train_metrics, epoch)
            log_on_wandb("valid", valid_metrics, epoch)
            log_on_wandb("outlier_valid", outlier_valid_metric, "-")  # type: ignore

        scheduler.step()

    # test queue
    test_metric = test(model, criterion, test_queue, device=DEVICE)
    log_metrics(logger, "Test", test_metric)
