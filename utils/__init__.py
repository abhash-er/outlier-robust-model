import argparse
import io
import logging
import sys

from fvcore.common.config import CfgNode
from matplotlib import pyplot as plt

from utils.dataset import get_loaders

logger = logging.getLogger(__name__)


class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.history = {"avg": [], "sum": [], "cnt": []}

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def save(self):
        self.history["avg"].append(self.avg)
        self.history["sum"].append(self.sum)
        self.history["cnt"].append(self.cnt)


def default_argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default=None,
                        metavar="FILE", help="path to config file")
    parser.add_argument("--seed", default=None, help="random seed")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Path to saved model weights"
    )
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:8888",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
             "N processes per node, which has N GPUs. This is the "
             "fastest way to use PyTorch for either single node or "
             "multi node data parallel training",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--data_path", default=None, metavar="FILE",
                        help="Path to the folder with train/test data folders")
    return parser


def parse_args(parser=default_argument_parser(), args=sys.argv[1:]):
    if "-f" in args:
        args = args[2:]
    return parser.parse_args(args)

def load_config(path):
    with open(path) as f:
        config = CfgNode.load_cfg(f)

    return config

def get_config_from_args(args=None):
    """
    Parses command line arguments and merges them with the defaults
    from the config file.

    Prepares experiment directories.
    :param args: args from a different argument parser than the default one.
    :return:
    """
    if args is None:
        args = parse_args()
    logger.info("Command line args: {}".format(args))

    if args.config_file is None:
        config = load_config("experiments/example_config.yaml")
    else:
        config = load_config(path=args.config_file)

    try:
        it = iter(args.opts)
        for arg, value in zip(it, it):
            if "." in arg:
                arg1, arg2 = arg.split(".")
                config[arg1][arg2] = type(config[arg1][arg2])(value)
            else:
                config[arg] = type(config[arg])(
                    value) if arg in config else eval(value)

        config.resume = args.resume
        config.model_path = args.model_path

        # load config file
        config.set_new_allowed(True)
        config.merge_from_list(args.opts)

    except AttributeError:
        it = iter(args)
        for arg, value in zip(it, it):
            config[arg] = value

    return config