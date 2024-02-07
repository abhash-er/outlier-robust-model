from __future__ import annotations

import json
import os
import pickle
import random

from fvcore.common.config import CfgNode  # type: ignore
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets as dset  # type: ignore
from torchvision.transforms import transforms  # type: ignore

from .cifar100_mappings import get_cifar100_mappings_dict
from .modified_cifar100 import CIFAR100

CIFAR100_to_CIFAR10 = {
    "Train": 9,  # Truck
    "Bicycle": 1,  # Automobile
    "Rocket": 0,  # Airplane
    "Lion": 3,  # Cat
    "Camel": 7,  # Horse
    "Kangaroo": 5,  # Dog
}


def get_loaders(config: CfgNode) -> tuple:
    data_path = config.data_path
    dataset = config.dataset
    batch_size = config.batch_size
    train_portion = config.train_portion
    train_meta_fraction = config.train_meta_fraction
    seed = config.seed

    if dataset == "cifar10":
        train_transform, valid_transform = _data_transforms_cifar10(config)
        injection_train_transform, _ = _data_transforms_cifar100(config)
        meta_transform = train_transform
        source_data = dset.CIFAR10(
            root=data_path, train=True, download=True, transform=train_transform
        )
        injection_data = CIFAR100(
            root=data_path,
            train=True,
            transform=injection_train_transform,
            download=True,
            coarse=True,
        )

        dataset_container = OutlierPlusDataset(
            source_data,
            injection_data,
            train_portion=train_portion,
            train_meta_portion=train_meta_fraction,
        )

        train_data = CustomDataset(dataset_container, mode="train")
        meta_data = CustomDataset(dataset_container, mode="meta")
        valid_data = CustomDataset(dataset_container, mode="valid")
        outlier_valid_data = CustomDataset(dataset_container, mode="outlier_valid")
        test_data = dset.CIFAR10(
            root=data_path, train=False, download=True, transform=valid_transform
        )
    else:
        # TODO Add more datasets
        raise ValueError(f"Unknown dataset: {dataset}")

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    meta_queue = torch.utils.data.DataLoader(
        meta_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    outlier_valid_queue = torch.utils.data.DataLoader(
        outlier_valid_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    return (
        train_queue,
        meta_queue,
        valid_queue,
        outlier_valid_queue,
        test_queue,
        train_transform,
        meta_transform,
        valid_transform,
    )


# This should be initialized only once to save memory
class OutlierPlusDataset:
    def __init__(  # noqa: C901, PLR0915
        self,
        source_data: Dataset,
        injection_data: Dataset,
        train_portion: float = 0.7,  # Train - 70%, Meta - 30 %
        outlier_portion: float = 0.3,
        train_meta_portion: float = 0.8,  # Train & Meta - 80 %,  Valid - 20 %
        is_negative_label: bool = True,
        shuffle_portion: float = 0.15,
    ) -> None:
        # Both train and meta should contain outlier examples
        self.train_images = []
        self.train_labels = []

        self.meta_images = []
        self.meta_labels = []

        # This Validation should not contain outliers
        self.valid_images = []
        self.valid_labels = []

        # Validation outlier examples ->
        self.outlier_valid_images = []
        self.outlier_valid_labels = []

        # Create loaders
        num_train = len(source_data)  # type: ignore
        indices = list(range(num_train))
        train_meta_split = int(np.floor(train_meta_portion * num_train))
        injection_split = int(np.floor(outlier_portion * train_meta_split))
        source_loader_train_meta = torch.utils.data.DataLoader(
            source_data,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[:train_meta_split]
            ),
            batch_size=1,
            pin_memory=True,
        )
        source_loader_valid = torch.utils.data.DataLoader(
            source_data,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[train_meta_split:]
            ),
            batch_size=1,
            pin_memory=True,
        )
        injection_loader_train_meta = torch.utils.data.DataLoader(
            injection_data,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[:injection_split]
            ),
            batch_size=1,
            pin_memory=True,
        )

        injection_loader_valid = torch.utils.data.DataLoader(
            injection_data,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[injection_split : injection_split + 1000]
            ),
            batch_size=1,
            pin_memory=True,
        )

        ## Training Data
        # Enter data
        images = []
        labels = []
        for image, label in source_loader_train_meta:
            image = image[0]
            label = label[0]
            images.append(image)
            labels.append(label)

        # Shuffle data
        if is_negative_label:
            shuffle_idx = random.sample(
                list(range(train_meta_split)),
                int(np.floor(shuffle_portion * train_meta_split)),
            )

            for idx in shuffle_idx:
                label = labels[idx].item()
                classes = list(range(10))
                classes.remove(label)
                new_label = random.choice(classes)
                labels[idx] = torch.tensor(new_label)

        # Inject data
        # Note: it is not gauranteed that all the data would be injected
        id_fine, id_coarse = get_cifar100_mappings_dict()
        for image, fine_label, coarse_label in injection_loader_train_meta:
            image = image[0]
            fine_label = fine_label[0]
            coarse_label = coarse_label[0]
            if id_coarse[coarse_label.item()] in [
                "vehicles 1",
                "vehicles 2",
            ] and id_fine[fine_label.item()] not in ["Train", "Bicycle", "Rocket"]:
                continue

            # Almost same but different
            if id_fine[fine_label.item()] in CIFAR100_to_CIFAR10:
                images.append(image)
                labels.append(
                    torch.tensor(CIFAR100_to_CIFAR10[id_fine[fine_label.item()]])
                )
            else:
                images.append(image)
                labels.append(torch.tensor(random.randint(0, 9)))

        # Distribute this data
        split_idx = int(np.floor(train_portion * len(images)))
        images = np.array(images)  # type: ignore
        labels = np.array(labels)  # type: ignore

        idx_to_shuffle = list(range(len(images)))
        random.shuffle(idx_to_shuffle)
        images = images[idx_to_shuffle]  # type: ignore
        labels = labels[idx_to_shuffle]  # type: ignore

        self.train_images = images[:split_idx]
        self.train_labels = labels[:split_idx]

        self.meta_images = images[split_idx:]
        self.meta_labels = labels[split_idx:]

        ## Validation Data
        for image, label in source_loader_valid:
            image = image[0]
            label = label[0]
            self.valid_images.append(image)
            self.valid_labels.append(label)

        for image, fine_label, coarse_label in injection_loader_valid:
            image = image[0]
            fine_label = fine_label[0]
            coarse_label = coarse_label[0]
            if id_coarse[coarse_label.item()] in [
                "vehicles 1",
                "vehicles 2",
            ] and id_fine[fine_label.item()] not in ["Train", "Bicycle", "Rocket"]:
                continue

            # Almost same but different
            if id_fine[fine_label.item()] in CIFAR100_to_CIFAR10:
                self.outlier_valid_images.append(image)
                self.outlier_valid_labels.append(
                    torch.tensor(CIFAR100_to_CIFAR10[id_fine[fine_label.item()]])
                )
            else:
                self.outlier_valid_images.append(image)
                self.outlier_valid_labels.append(torch.tensor(random.randint(0, 9)))

    def get_train_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.train_images, self.train_labels  # type: ignore

    def get_meta_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.meta_images, self.meta_labels  # type: ignore

    def get_valid_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.valid_images, self.valid_labels  # type: ignore

    def get_outlier_valid_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.outlier_valid_images, self.outlier_valid_labels  # type: ignore


class CustomDataset(Dataset):
    def __init__(
        self,
        dataset_container: OutlierPlusDataset,
        mode: str = "train",
    ) -> None:
        assert mode in [
            "train",
            "meta",
            "valid",
            "outlier_valid",
        ], "Please enter a mode among train, meta, val, outlier_val"

        if mode == "train":
            images, labels = dataset_container.get_train_data()

        if mode == "meta":
            images, labels = dataset_container.get_meta_data()

        if mode == "valid":
            images, labels = dataset_container.get_valid_data()

        if mode == "outlier_valid":
            images, labels = dataset_container.get_outlier_valid_data()

        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def _data_transforms_cifar10(
    config: CfgNode,
) -> tuple[transforms.Compose, transforms.Compose]:
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    if hasattr(config, "cutout") and config.cutout:
        train_transform.transforms.append(
            Cutout(config.cutout_length, config.cutout_prob)
        )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_cifar100(
    args: CfgNode,
) -> tuple[transforms.Compose, transforms.Compose]:
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    if hasattr(args, "cutout") and args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


class Cutout:
    def __init__(self, length: int, prob: float = 1.0) -> None:
        self.length = length
        self.prob = prob

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.binomial(1, self.prob):
            h, w = img.size(1), img.size(2)  # type: ignore
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0
            mask = torch.from_numpy(mask)  # type: ignore
            mask = mask.expand_as(img)  # type: ignore
            img *= mask
        return img


def load_dataset(config: CfgNode) -> tuple:
    if os.path.exists("data/cifar10_augmented.pickle"):
        with open("data/cifar10_augmented.pickle", "rb") as f:
            augmented_dataset = pickle.load(f)

        train_queue = augmented_dataset["train_queue"]
        meta_queue = augmented_dataset["meta_queue"]
        valid_queue = augmented_dataset["valid_queue"]
        outlier_valid_queue = augmented_dataset["outlier_valid_queue"]
        test_queue = augmented_dataset["test_queue"]
        train_transform = augmented_dataset["train_transform"]
        meta_transform = augmented_dataset["meta_transform"]
        valid_transform = augmented_dataset["valid_transform"]
    else:
        (
            train_queue,
            meta_queue,
            valid_queue,
            outlier_valid_queue,
            test_queue,
            train_transform,
            meta_transform,
            valid_transform,
        ) = get_loaders(config)

        augmented_dataset = {
            "train_queue": train_queue,
            "meta_queue": meta_queue,
            "valid_queue": valid_queue,
            "outlier_valid_queue": outlier_valid_queue,
            "test_queue": test_queue,
            "train_transform": train_transform,
            "meta_transform": meta_transform,
            "valid_transform": valid_transform,
        }
        with open("data/cifar10_augmented.pickle", "wb") as f:
            pickle.dump(augmented_dataset, f)

    return (
        train_queue,
        meta_queue,
        valid_queue,
        outlier_valid_queue,
        test_queue,
        train_transform,
        meta_transform,
        valid_transform,
    )


if __name__ == "__main__":
    # Sanity Check
    config = {
        "seed": 0,
        "data_path": "data/",
        "dataset": "cifar10",
        "batch_size": 16,
        "train_portion": 0.75,
        "train_meta_fraction": 0.8,
    }
    config = CfgNode.load_cfg(json.dumps(config))
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
    print(len(train_queue))
    print(len(meta_queue))
    print(len(valid_queue))
    print(len(outlier_valid_queue))
    print(len(test_queue))
    for batch in train_queue:
        image, label = batch
        print(image.shape)
        print(label)
        break

    for batch in outlier_valid_queue:
        image, label = batch
        print(image.shape)
        print(label)
        break
    # Do something to check the train val loaders
