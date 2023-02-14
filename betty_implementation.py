"""
Implementation with Betty
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from betty.problems import ImplicitProblem
import torch.nn.functional as F
from torch.utils.data import DataLoader

from betty.configs import Config, EngineConfig
from betty.engine import Engine

from dataset import ToyDataset
from utils import OutlierModel, LinearRegressor, AverageMeter

toy_dataset_train = ToyDataset(mode="Train")
toy_dataset_val = ToyDataset(mode="Validate")

# dataloaders
toy_train_loader = DataLoader(dataset=toy_dataset_train, batch_size=16, shuffle=True)
toy_meta_loader = DataLoader(dataset=toy_dataset_val, batch_size=1, shuffle=True)

print(len(toy_train_loader))
print(len(toy_meta_loader))

train_loss_meter = AverageMeter()


# Bilevel optimization Model
class TrainProblem(ImplicitProblem):
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        inputs, labels, _ = batch
        inputs = torch.unsqueeze(inputs, -1)
        labels = torch.unsqueeze(labels, -1)
        outputs = self.forward(inputs)
        weights = self.meta_learner(torch.stack([inputs, labels], dim=-1))
        loss = ((outputs - labels) ** 2).squeeze(-1)
        loss = torch.dot(weights, loss).mean()
        train_loss_meter.update(loss.item())
        train_loss_meter.save()
        return loss

    def configure_module(self):
        return LinearRegressor(in_features=1, out_features=1).double()

    def configure_optimizer(self):
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=1e-4, weight_decay=1e-4)
        return optimizer


meta_loss_meter = AverageMeter()
val_results = {"val_points": [], "val_outputs": []}


class MetaProblem(ImplicitProblem):
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        inputs, labels = batch
        inputs = torch.unsqueeze(inputs, -1)
        labels = torch.unsqueeze(labels, -1)

        loss = F.mse_loss(self.train_learner(inputs), labels)
        meta_loss_meter.update(loss.item())
        meta_loss_meter.save()
        return loss

    def configure_module(self):
        return OutlierModel(in_features=2).double()

    def configure_optimizer(self):
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=1e-4, weight_decay=1e-4)
        return optimizer

    def validate(self):
        with torch.no_grad():
            for batch in self.train_data_loader[0]:
                inputs, labels = batch
                inputs = torch.unsqueeze(inputs, -1)
                out = self.train_learner(torch.tensor(inputs, dtype=torch.float64).unsqueeze(-1))
                val_results["val_points"].append(inputs.item())
                val_results["val_outputs"].append(out.item())

    def test(self, test_ex):
        with torch.no_grad():
            print("Slope of the train lerarner is :", self.train_learner.parameters())
            for x, target in test_ex:
                out = self.train_learner(torch.tensor(x, dtype=torch.float64).unsqueeze(-1))
                print("My predicted output is :", out)
                print("My target label is:", target)


device = "cpu"
trainer_config = Config(type="darts", unroll_steps=1)
trainer_problem = TrainProblem(
    name="train_learner",
    train_data_loader=toy_train_loader,
    config=trainer_config,
    device=device,
)

meta_config = Config(type="darts", unroll_steps=1)
meta_problem = MetaProblem(
    name="meta_learner",
    train_data_loader=toy_meta_loader,
    config=meta_config,
    device=device,
)

engine_config = EngineConfig(train_iters=1000)

problems = [trainer_problem, meta_problem]
u2l = {meta_problem: [trainer_problem]}
l2u = {trainer_problem: [meta_problem]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = Engine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()

plt.plot(range(len(train_loss_meter.history["avg"])), train_loss_meter.history["avg"], label="Average Training Loss")
plt.show()

meta_problem.validate()
plt.plot(val_results["val_points"], val_results["val_points"], label="Validation Results")
plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), c="r", label="Ideal Output")
plt.show()

test_ex = [[5000, 5000], [4780, 4780], [3213, 3213]]
meta_problem.test(test_ex)
