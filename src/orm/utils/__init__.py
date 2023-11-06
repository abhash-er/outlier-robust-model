from __future__ import annotations

from typing import Any

from .arg_parse import get_config_from_args, logger
from .dataset import get_loaders


class AverageMeter:
    def __init__(self) -> None:
        self.reset()
        self.history: dict[str, Any] = {"avg": [], "sum": [], "cnt": []}

    def reset(self) -> None:
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def save(self) -> None:
        self.history["avg"].append(self.avg)
        self.history["sum"].append(self.sum)
        self.history["cnt"].append(self.cnt)


__all__ = [
    "get_loaders",
    "get_config_from_args",
    "logger",
]
