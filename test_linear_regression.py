import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import ToyDataset
from utils import AverageMeter, train, validate, LinearRegressor

random.seed(42)
toy_dataset_train = ToyDataset(mode="Train")
toy_dataset_val = ToyDataset(mode="Validate")

# dataloaders
toy_train_loader = DataLoader(dataset=toy_dataset_train, batch_size=16, shuffle=True)
toy_val_loader = DataLoader(dataset=toy_dataset_val, batch_size=1, shuffle=False)

num_epochs = 50

# Model without Regularization
normal_regressor = LinearRegressor(in_features=1, out_features=1).double()
loss_criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(normal_regressor.parameters(), lr=1e-3, weight_decay=0)
train_loss_meter = AverageMeter()
val_loss_meter = AverageMeter()

val_res = []
train_res = []
for epochs in range(num_epochs):
    train_results = train(train_loader=toy_train_loader, regressor=normal_regressor, loss_criterion=loss_criterion,
                          optimizer=optimizer,
                          train_loss_meter=train_loss_meter)
    val_results = validate(val_loader=toy_val_loader, regressor=normal_regressor, loss_criterion=loss_criterion,
                           val_loss_meter=val_loss_meter)
    val_res.append(val_results)
    train_res.append(train_results)

print("Slope of the line for standard linear regression", [p for p in normal_regressor.parameters()])

# plots
plt.plot(range(num_epochs), train_loss_meter.history["avg"], label="Training Loss")
plt.plot(range(num_epochs), val_loss_meter.history["avg"], label="Validation loss")
plt.legend(loc="upper left")
plt.show()
plt.scatter(val_res[-1]["points"], val_res[-1]["outputs"], label="Predicted Output on Validation Dataset")
plt.scatter(train_res[-1]["points"], train_res[-1]["outputs"], label="Predicted Output on Training Dataset")
plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), c="r", label="Ideal Output")
plt.scatter(train_res[-1]["points"], train_res[-1]["label"], label="Ground Truth labels for Training Dataset")
plt.scatter(val_res[-1]["points"], val_res[-1]["label"], label="Ground Truth labels for Validation Dataset")
plt.legend(loc="upper left")
plt.show()

print("Linear Regression ===================================================")
test_ex = [[5000, 5000], [4780, 4780], [3213, 3213]]
for x, target in test_ex:
    with torch.no_grad():
        out = normal_regressor(torch.tensor(x, dtype=torch.float64).unsqueeze(-1))
        print("My predicted output is :", out)
        print("My target label is:", target)

# Model with reguarization
regularized_regressor = LinearRegressor(in_features=1, out_features=1).double()
regularized_optimizer = torch.optim.SGD(normal_regressor.parameters(), lr=1e-3, weight_decay=1e-4)
loss_criterion_reg = torch.nn.MSELoss()
train_loss_meter_reg = AverageMeter()
val_loss_meter_reg = AverageMeter()

val_reg_res = []
train_reg_res = []
for epochs in range(num_epochs):
    train_results = train(train_loader=toy_train_loader, regressor=regularized_regressor,
                          loss_criterion=loss_criterion_reg,
                          optimizer=regularized_optimizer,
                          train_loss_meter=train_loss_meter_reg)
    val_results = validate(val_loader=toy_val_loader, regressor=regularized_regressor,
                           loss_criterion=loss_criterion_reg,
                           val_loss_meter=val_loss_meter_reg)
    val_reg_res.append(val_results)
    train_reg_res.append(train_results)

print("Slope of the line for regularized linear regression", [p for p in regularized_regressor.parameters()])
# plots
plt.plot(range(num_epochs), train_loss_meter_reg.history["avg"], label="Training Loss")
plt.plot(range(num_epochs), val_loss_meter_reg.history["avg"], label="Validation loss")
plt.legend(loc="upper left")
plt.show()
plt.scatter(val_reg_res[-1]["points"], val_reg_res[-1]["outputs"], label="Predicted Output on Validation Dataset")
plt.scatter(train_reg_res[-1]["points"], train_reg_res[-1]["outputs"], label="Predicted Output on Training Dataset")
plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), c="r", label="Ideal Output")
plt.scatter(train_reg_res[-1]["points"], train_reg_res[-1]["label"], label="Ground Truth labels for Training Dataset")
plt.scatter(val_reg_res[-1]["points"], val_reg_res[-1]["label"], label="Ground Truth labels for Validation Dataset")
plt.legend(loc="upper left")
plt.show()

print("Linear Regression with Regularization ==================================")
for x, target in test_ex:
    with torch.no_grad():
        out = normal_regressor(torch.tensor(x, dtype=torch.float64).unsqueeze(-1))
        print("My predicted output is :", out)
        print("My target label is:", target)
