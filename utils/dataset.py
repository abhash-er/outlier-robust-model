import numpy as np
import torch
import torchvision.datasets as dset
from torchvision.transforms import transforms
import json
from fvcore.common.config import CfgNode


def get_loaders(config):
    data_path = config.data_path
    dataset = config.dataset
    batch_size = config.batch_size
    train_portion = config.train_portion
    meta_train_fraction = config.meta_train_fraction
    seed = config.seed

    if dataset == "cifar10":
        train_transform, valid_transform = _data_transforms_cifar10(config)
        meta_transform = train_transform
        train_data = dset.CIFAR10(
            root=data_path, train=True, download=True, transform=train_transform
        )
        test_data = dset.CIFAR10(
            root=data_path, train=False, download=True, transform=valid_transform
        )
    elif dataset == "cifar100":
        train_transform, valid_transform = _data_transforms_cifar100(config)
        meta_transform = train_transform
        train_data = dset.CIFAR100(
            root=data_path, train=True, download=True, transform=train_transform
        )
        test_data = dset.CIFAR100(
            root=data_path, train=False, download=True, transform=valid_transform
        )
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

    num_train = len(train_data)
    indices = list(range(num_train))
    first_split = int(np.floor(train_portion * num_train * (1 - meta_train_fraction)))
    second_split = int(np.floor(train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[:first_split]),
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    meta_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[first_split:second_split]),
        pin_memory=True,
        num_workers=0,
        worker_init_fn=np.random.seed(seed),
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[second_split:num_train]),
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

    return train_queue, valid_queue, meta_queue, test_queue, train_transform, meta_transform, valid_transform


def _data_transforms_cifar10(args):
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

    if hasattr(args, 'cutout') and args.cutout:
        train_transform.transforms.append(
            Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
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
    if args.cutout:
        train_transform.transforms.append(
            Cutout(args.cutout_length, args.cutout_prob))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


class Cutout(object):
    def __init__(self, length, prob=1.0):
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.binomial(1, self.prob):
            h, w = img.size(1), img.size(2)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img *= mask
        return img


if __name__ == "__main__":
    # Sanity Check
    config = {
        "seed": 0,
        "data_path": "data/",
        "dataset": "cifar10",
        "batch_size": 16,
        "train_portion": 0.75,
        "meta_train_fraction": 0.2,
    }
    config = CfgNode.load_cfg(json.dumps(config))
    train_queue, valid_queue, meta_queue, test_queue, train_transform, meta_transform, valid_transform = get_loaders(
        config)

    for batch in train_queue:
        image, label = batch
        print(image.shape)
        break
    # Do something to check the train val loaders
