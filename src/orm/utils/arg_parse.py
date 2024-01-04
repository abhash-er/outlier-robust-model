from __future__ import annotations

import argparse
import logging
import sys

from fvcore.common.config import CfgNode

logger = logging.getLogger(__name__)


def load_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    return logger


def default_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default=None, metavar="FILE", help="path to config file"
    )
    parser.add_argument("--seed", default=None, help="random seed")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to saved model weights"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--data_path",
        default=None,
        metavar="FILE",
        help="Path to the folder with train/test data folders",
    )
    return parser


DEFAULT_PARSER = default_argument_parser()


def parse_args(
    parser: argparse.ArgumentParser = DEFAULT_PARSER, args: list[str] = sys.argv[1:]
) -> argparse.Namespace:
    if "-f" in args:
        args = args[2:]
    return parser.parse_args(args)


def load_config(path: str) -> CfgNode:
    with open(path) as f:
        config = CfgNode.load_cfg(f)

    return config


def get_config_from_args(
    args: list[str] | None = None, logger: logging.Logger | None = None
) -> CfgNode:
    """Parses command line arguments and merges them with the defaults
    from the config file.

    Prepares experiment directories.
    :param args: args from a different argument parser than the default one.
    :return:
    """
    if logger is None:
        logger = load_logger()
    if args is None:
        args = parse_args()  # type: ignore
    logger.info(f"Command line args: {args}")

    if args.config_file is None:  # type: ignore
        config = load_config("experiments/example_config.yaml")
    else:
        config = load_config(path=args.config_file)  # type: ignore

    try:
        it = iter(args.opts)  # type: ignore
        for arg, value in zip(it, it):
            if "." in arg:
                arg1, arg2 = arg.split(".")
                config[arg1][arg2] = type(config[arg1][arg2])(value)
            else:
                config[arg] = type(config[arg])(value) if arg in config else eval(value)

        config.resume = args.resume  # type: ignore
        config.model_path = args.model_path  # type: ignore

        # load config file
        config.set_new_allowed(True)
        config.merge_from_list(args.opts)  # type: ignore

    except AttributeError:
        it = iter(args)  # type: ignore
        for arg, value in zip(it, it):
            config[arg] = value

    return config
