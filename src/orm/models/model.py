from __future__ import annotations

import abc
import logging
from typing import Any

from betty.configs import Config  # type: ignore
from betty.problems import ImplicitProblem  # type: ignore
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import torchvision  # type: ignore
import wandb

from orm.utils import AverageMeter  # type: ignore


class TrainProblem(ImplicitProblem):
    def __init__(
        self,
        name: str,
        config: Config,
        module: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        train_data_loader: DataLoader | None = None,
        device: str | None = None,
        loss_meter: AverageMeter = None,
        is_wandb_log: bool = False,
    ) -> None:
        super().__init__(
            name, config, module, optimizer, scheduler, train_data_loader, device
        )
        self.train_loss_meter = loss_meter
        self.loader_len = len(train_data_loader)  # type: ignore
        self.step_num = 0
        self.epoch_num = 0
        self.is_wandb_log = is_wandb_log

    @abc.abstractmethod
    def loss_function(
        self, labels: torch.Tensor, outputs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Override this function to add the loss functionality"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)  # type: ignore

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self.module.train()
        images, labels = batch
        outputs = self.forward(images)
        weights = self.meta_learner(images, labels)

        # Use categorical loss function type
        loss = self.loss_function(labels, outputs)
        loss = torch.dot(weights.squeeze(-1), loss).mean()
        self.train_loss_meter.update(loss.item())
        self.train_loss_meter.save()
        self.step_num += 1
        if self.step_num % self.loader_len == 0:
            if self.is_wandb_log:
                wandb.log(  # type: ignore
                    {"train/loss_per_epochs": self.train_loss_meter.avg}
                )
                wandb.log({"train/epochs": self.epoch_num})  # type: ignore

            logging.info(
                f"[TrainProblem] Epoch {self.epoch_num}: \
                    Training loss is {self.train_loss_meter.avg}"
            )
            self.epoch_num += 1
        # logging.info(
        #     f"[Train Problem]: train/step: {self.step_num}, train/loss: \
        #              {self.train_loss_meter.avg}"
        # )
        if self.is_wandb_log:
            log_dict = {
                "train/step": self.step_num,
                "train/loss": self.train_loss_meter.avg,
            }
            wandb.log(log_dict)  # type: ignore

        return loss

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        trainable_params = []
        for param in self.module.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def save_chekpoints(self) -> None:
        pass


class MetaProblem(ImplicitProblem):
    def __init__(
        self,
        name: str,
        config: Config,
        module: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        train_data_loader: DataLoader | None = None,
        val_data_loader: DataLoader | None = None,
        test_data_loader: DataLoader | None = None,
        device: str | None = None,
        loss_meter: AverageMeter = AverageMeter(),  # noqa: B008
        is_wandb_log: bool = False,
    ):
        super().__init__(
            name, config, module, optimizer, scheduler, train_data_loader, device
        )
        self.meta_loss_meter = loss_meter
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.val_results: dict[Any, Any] = {}
        self.step_num = 0
        self.epoch_num = 0
        self.loader_len = len(train_data_loader)  # type: ignore
        self.is_wandb_log = is_wandb_log

    @abc.abstractmethod
    def loss_function(
        self, labels: torch.Tensor, outputs: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Override this function to add the loss functionality"
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # There are 2 options
        # -> different n/w (s) for input and label
        # -> stack a 10 channel sparse layers across the input
        # and pass them into one n/w
        return self.module(images, labels)  # type: ignore

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self.module.train()
        images, labels = batch
        output = self.train_learner(images)
        loss = self.loss_function(labels, output)
        self.meta_loss_meter.update(loss.item())
        self.meta_loss_meter.save()
        self.step_num += 1
        if self.step_num % self.loader_len == 0:
            logging.info(
                f"[MetaProblem] Epoch {self.epoch_num}: Training Loss is \
                    {self.meta_loss_meter.avg}"
            )
            if self.is_wandb_log:
                wandb.log(  # type: ignore
                    {"meta/loss_per_epoch": self.meta_loss_meter.avg}
                )
            self.log_validate()
            self.epoch_num += 1
        if self.is_wandb_log:
            log_dict = {
                "meta/step": self.step_num,
                "meta/loss": self.meta_loss_meter.avg,
            }
            wandb.log(log_dict)  # type: ignore
        return loss

    # FIXME Update the validate function to do something better
    def validate(self) -> tuple[float, float]:
        self.train_learner.module.eval()
        assert self.val_data_loader is not None
        with torch.no_grad():
            for batch in self.val_data_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                out = self.train_learner(images)  # noqa: F841
                grid = torchvision.utils.make_grid(images)  # noqa: F841
        return 0, 0

    def test(self) -> None:
        assert self.test_data_loader is not None
        with torch.no_grad():
            for x, target in self.test_data_loader:
                out = self.train_learner(x)
                # TODO plot image also
                print("My predicted output is :", out)
                print("My target label is:", target)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        trainable_params = []
        for param in self.module.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def log_validate(self) -> None:
        if self.step_num % self.loader_len == 0:
            log_dict = {"meta/epochs": self.epoch_num}

            validate = getattr(self, "validate", None)
            if validate is not None and callable(validate):
                val_loss, val_acc = self.validate()
                log_dict.update(
                    {
                        "validate/acc": val_acc,  # type: ignore
                        "validate/loss": val_loss,  # type: ignore
                    }
                )

            logging.info(
                f"[MetaProblem] Epoch {self.epoch_num} -- validation loss is \
                {val_loss},validation accuracy is {val_acc}"
            )
            if self.is_wandb_log:
                wandb.log(log_dict)  # type: ignore


def get_resnet_embedding(
    num_classes: int = 10, freeze_layers: bool = True, hidden_layer_size: int = 512
) -> torch.nn.Module:
    resnet18 = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT
    )

    if freeze_layers:
        for param in resnet18.parameters():
            param.requires_grad = False
    n_in = resnet18.fc.in_features
    resnet18.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=n_in, out_features=hidden_layer_size),
        torch.nn.Linear(in_features=hidden_layer_size, out_features=num_classes),
        torch.nn.Softmax(dim=1),
    )

    for param in resnet18.fc.parameters():
        param.requires_grad = True

    return resnet18  # type: ignore


class OutlierDetectionModel(torch.nn.Module):
    def __init__(
        self,
        n_classes: int = 10,
        freeze_layers: bool = True,
        hidden_layer_size: int = 512,
    ):
        super().__init__()
        self.num_classes = n_classes
        self.image_encoder = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )

        if freeze_layers:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        n_in = self.image_encoder.fc.in_features
        self.image_encoder.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_in, out_features=hidden_layer_size),
        )

        # Enable gradients for back propogation
        for param in self.image_encoder.fc.parameters():
            param.requires_grad = True

        self.label_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_classes, out_features=hidden_layer_size // 2),
            torch.nn.Linear(
                in_features=hidden_layer_size // 2, out_features=hidden_layer_size
            ),
        )

        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hidden_layer_size * 2, out_features=hidden_layer_size
            ),
            torch.nn.Linear(in_features=hidden_layer_size, out_features=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        image_enc = self.image_encoder(images)
        with torch.no_grad():
            labels = torch.nn.functional.one_hot(
                labels, num_classes=self.num_classes
            ).float()
        label_enc = self.label_encoder(labels)
        full_enc = torch.cat((image_enc, label_enc), -1)
        return self.final_layer(full_enc)  # type: ignore


if __name__ == "__main__":
    # Test here
    resnet = get_resnet_embedding()
    outlier_model = OutlierDetectionModel()
    image = torch.rand(10, 3, 32, 32)
    label = torch.randint(low=0, high=9, size=(10,))
    print(label)

    resnet_op = resnet(image)
    loss_fn = torch.nn.CrossEntropyLoss()
    res_loss = loss_fn(resnet_op, label)
    res_loss.backward(retain_graph=True)
    print(res_loss)
    print(resnet_op)

    outlier_prob = outlier_model(image, label)
    outlier_score = torch.rand((10, 1))
    print(outlier_prob)
    outlier_loss = torch.nn.functional.mse_loss(outlier_prob, outlier_score)
    print(outlier_loss)
    outlier_loss.backward(retain_graph=True)
