import torch
import torchvision
from betty.problems import ImplicitProblem
from utils import AverageMeter

import abc


class TrainProblem(ImplicitProblem):
    def __init__(
            self,
            name,
            config,
            module=None,
            optimizer=None,
            scheduler=None,
            train_data_loader=None,
            device=None,
            loss_meter=None,
    ):
        super().__init__(name, config, module, optimizer, scheduler, train_data_loader, device)
        self.train_loss_meter = loss_meter

    @abc.abstractmethod
    def loss_function(self, labels, outputs):
        raise NotImplemented("Override this function to add the loss functionality")

    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        images, labels = batch
        outputs = self.forward(images)
        weights = self.meta_learner(images, labels)

        # Use categorical loss function type
        loss = self.loss_function(labels, outputs)
        loss = torch.dot(weights.squeeze(-1), loss).mean()
        self.train_loss_meter.update(loss.item())
        self.train_loss_meter.save()
        return loss


# You look at the problem from this end
class MetaProblem(ImplicitProblem):
    def __init__(
            self,
            name,
            config,
            module=None,
            optimizer=None,
            scheduler=None,
            train_data_loader=None,
            val_data_loader=None,
            test_data_loader=None,
            device=None,
            loss_meter=AverageMeter(),
    ):
        super().__init__(name, config, module, optimizer, scheduler, train_data_loader, device)
        self.meta_loss_meter = loss_meter
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.val_results = {}

    @abc.abstractmethod
    def loss_function(self, labels, outputs):
        raise NotImplemented("Override this function to add the loss functionality")

    def forward(self, images, labels):
        # There are 2 options
        # -> different n/w (s) for input and label
        # -> stack a 10 channel sparse layers across the input and pass them into one n/w
        return self.module(images, labels)

    def training_step(self, batch):
        images, labels = batch
        output = self.train_learner(images)
        loss = self.loss_function(labels, output)
        self.meta_loss_meter.update(loss.item())
        self.meta_loss_meter.save()
        return loss

    def validate(self):
        with torch.no_grad():
            for batch in self.val_data_loader:
                images, labels = batch
                out = self.train_learner(images)
                grid = torchvision.utils.make_grid(images)

    def test(self):
        with torch.no_grad():
            for x, target in self.test_data_loader:
                out = self.train_learner(x)
                # TODO plot image also
                print("My predicted output is :", out)
                print("My target label is:", target)


def get_resnet_embedding(num_classes=10, freeze_layers=True, hidden_layer_size=512):
    resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    if freeze_layers:
        for param in resnet18.parameters():
            param.requires_grad = False
    n_in = resnet18.fc.in_features
    resnet18.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=n_in, out_features=hidden_layer_size),
        torch.nn.Linear(in_features=hidden_layer_size, out_features=num_classes),
        torch.nn.Softmax(),
    )

    for param in resnet18.fc.parameters():
        param.requires_grad = True

    return resnet18


class OutlierDetectionModel(torch.nn.Module):
    def __init__(self, n_classes=10, freeze_layers=True, hidden_layer_size=512):
        super().__init__()
        self.num_classes = n_classes
        self.image_encoder = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

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
            torch.nn.Linear(in_features=hidden_layer_size // 2, out_features=hidden_layer_size)
        )

        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_layer_size * 2, out_features=hidden_layer_size),
            torch.nn.Linear(in_features=hidden_layer_size, out_features=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, images, labels):
        image_enc = self.image_encoder(images)
        with torch.no_grad():
            labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        label_enc = self.label_encoder(labels)
        full_enc = torch.cat((image_enc, label_enc), -1)
        return self.final_layer(full_enc)


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
    res_loss.backward()
    print(res_loss)
    print(resnet_op)

    outlier_prob = outlier_model(image, label)
    outlier_score = torch.rand((10,1))
    print(outlier_score)
    outlier_loss = torch.nn.functional.mse_loss(outlier_prob, outlier_score)
    print(outlier_loss)
    outlier_loss.backward()
