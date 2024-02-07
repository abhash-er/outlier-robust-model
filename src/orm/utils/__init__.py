from __future__ import annotations

from typing import Any, Iterable

import torch

from .arg_parse import get_config_from_file
from .dataset import CustomDataset, get_loaders, load_dataset


def calc_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Iterable = (1,)
) -> list[float]:
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter:
    def __init__(self) -> None:
        self.reset()
        self.history: dict[str, Any] = {"avg": [], "sum": [], "cnt": []}

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def save(self) -> None:
        self.history["avg"].append(self.avg)
        self.history["sum"].append(self.sum)
        self.history["cnt"].append(self.cnt)


__all__ = [
    "get_loaders",
    "get_config_from_file",
    "load_dataset",
    "calc_accuracy",
    "CustomDataset",
]
