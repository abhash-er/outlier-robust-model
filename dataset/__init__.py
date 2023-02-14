import random
import numpy as np
import matplotlib as plt
import torch
from torch.utils.data import Dataset, DataLoader

random.seed(42)


def ground_truth(x, m, b):
    return (m * x) + b


def generate_outliers():
    # return random.uniform(0, 1e-4)
    return np.random.normal(20, 10)

# Define toy Dataset
sigma = 10
X = np.linspace(0, 1, 1000)
size = X.size

# apply mask to each point to determine whether its a outlier or not
mask = np.array([1] * int(0.9 * size) + [0] * (size - int(0.9 * size)))
np.random.shuffle(mask)

X_normal = []
y_normal = []

X_outlier = []
y_outlier = []
for i in range(size):
    if mask[i]:
        X_normal.append(X[i])
        y_normal.append(random.gauss(ground_truth(X[i], 1, 0), 0.15))
    else:
        X_outlier.append(X[i])
        y_outlier.append(X[i] + generate_outliers())

# plt.scatter(X, y)
# plt.show()

random.seed(42)
# define splits
val_split_ratio = 0.8
toy_dataset_size = len(X_normal)
toy_dataset_indices = list(range(toy_dataset_size))
random.shuffle(toy_dataset_indices)
val_split_index = int(np.floor(val_split_ratio * toy_dataset_size))
val_idx, train_idx = toy_dataset_indices[val_split_index:], toy_dataset_indices[:val_split_index]

# Note : We also need to identify which are outliers and which are not
X_train = []
y_train = []
is_outlier = []
for idx in train_idx:
    X_train.append(X_normal[idx])
    y_train.append(y_normal[idx])
    is_outlier.append(False)

X_train.extend(X_outlier)
y_train.extend(y_outlier)
is_outlier.extend([True] * len(y_outlier))

X_val = []
y_val = []
for idx in val_idx:
    X_val.append(X_normal[idx])
    y_val.append(y_normal[idx])


# Make torch Dataset
class ToyDataset(Dataset):
    def __init__(self, mode='Train'):
        self.mode = mode
        self.train_points = X_train
        self.train_values = y_train

        self.val_points = X_val
        self.val_values = y_val
        self.outlier_labels = is_outlier

    def __len__(self):
        if self.mode == "Train":
            return len(self.train_points)
        elif self.mode == 'Validate':
            return len(self.val_points)
        else:
            raise AttributeError("Unidentified mode for the dataset")

    def __getitem__(self, idx):
        if self.mode == 'Train':
            return torch.tensor(self.train_points[idx], dtype=torch.float64), torch.tensor(self.train_values[idx],
                                                                                           dtype=torch.float64), \
                   torch.tensor(self.outlier_labels)

        elif self.mode == 'Validate':
            return torch.tensor(self.val_points[idx], dtype=torch.float64), torch.tensor(self.val_values[idx],
                                                                                         dtype=torch.float64)
        else:
            raise AttributeError("Unidentified mode for the dataset")
