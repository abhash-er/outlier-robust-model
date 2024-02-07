from __future__ import annotations

import logging

from fvcore.common.config import CfgNode

logger = logging.getLogger(__name__)


def load_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    return logger


def load_config(path: str) -> CfgNode:
    with open(path) as f:
        config = CfgNode.load_cfg(f)

    return config


def get_config_from_file(
    logger: logging.Logger | None = None,
    config_file_path: str | None = None,
) -> CfgNode:
    """Parses command line arguments and merges them with the defaults
    from the config file.

    Prepares experiment directories.
    :param args: args from a different argument parser than the default one.
    :return:
    """
    if logger is None:
        logger = load_logger()

    if config_file_path is None:  # type: ignore
        config = load_config("experiments/example_config.yaml")
    else:
        config = load_config(config_file_path)

    return config
