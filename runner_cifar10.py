import argparse
from abc import ABC

import torch
import sys


from betty.configs import Config, EngineConfig
from betty.engine import Engine

from utils import get_config_from_args, get_loaders, AverageMeter
from model import TrainProblem, MetaProblem, get_resnet_embedding, OutlierDetectionModel

config = get_config_from_args()


def get_other_args():
    parser = argparse.ArgumentParser(description="Experiment- CIFAR10")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--eta-min", type=float, default=1e-5, help="minimum learning rate for the scheduler")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="optimizer: weight decay")

    return parser.parse_args()


args = get_other_args()


class MyTrainProblem(TrainProblem, ABC):
    def loss_function(self, labels, outputs):
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return loss_fn(outputs, labels)

    def configure_module(self):
        model = get_resnet_embedding()
        return model

    def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.module.parameters(), lr=0.01, weight_decay=args.weight_decay)
        return optimizer

    def configure_scheduler(self):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                  T_max=100,
                                                                  eta_min=args.eta_min)
        return lr_scheduler


class MyMetaProblem(MetaProblem, ABC):
    def loss_function(self, labels, outputs):
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(outputs, labels)

    def configure_module(self):
        model = OutlierDetectionModel()
        return model

    def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.module.parameters(), lr=0.01, weight_decay=args.weight_decay)
        return optimizer

    def configure_scheduler(self):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                  T_max=100,
                                                                  eta_min=args.eta_min)
        return lr_scheduler


if __name__ == "__main__":
    # Get the dataset and splits
    train_queue, valid_queue, meta_queue, test_queue, train_transform, meta_transform, valid_transform = get_loaders(
        config)

    # prepare meters
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    device = "cpu"
    trainer_config = Config(type="darts", unroll_steps=1)
    trainer_problem = MyTrainProblem(
        name="train_learner",
        train_data_loader=train_queue,
        config=trainer_config,
        device=device,
        loss_meter=train_loss_meter,
    )

    meta_config = Config(type="darts", unroll_steps=1)
    meta_problem = MyMetaProblem(
        name="meta_learner",
        train_data_loader=meta_queue,
        val_data_loader=valid_queue,
        test_data_loader=test_queue,
        config=meta_config,
        device=device,
        loss_meter=val_loss_meter,
    )

    # Setup the engine and dependencies
    engine_config = EngineConfig(train_iters=config.total_steps)

    problems = [trainer_problem, meta_problem]
    u2l = {meta_problem: [trainer_problem]}
    l2u = {trainer_problem: [meta_problem]}
    dependencies = {"l2u": l2u, "u2l": u2l}

    engine = Engine(config=engine_config, problems=problems, dependencies=dependencies)
    engine.run()
