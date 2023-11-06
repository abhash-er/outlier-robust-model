from __future__ import annotations

import argparse
import os

import yaml


def main(args: argparse.Namespace) -> None:
    folder = os.path.join(
        args.config_root,
        args.experiment,
        args.dataset,
    )

    print(folder)
    os.makedirs(folder, exist_ok=True)
    args.start_seed = int(args.start_seed)

    config = {
        "config_type": args.experiment,
        "seed": args.start_seed,
        "dataset": args.dataset,
        "out_dir": args.out_dir,
        "batch_size": args.batch_size,
        "train_portion": args.train_portion,
        "train_meta_fraction": args.train_meta_fraction,
        "cutout": args.cutout,
        "data_path": args.data_path,
        "total_steps": args.total_steps,
        "cutout_length": args.cutout_length,
        "cutout_prob": args.cutout_prob,
    }

    with open(folder + f"/config_{args.start_seed}.yaml", "w") as fh:
        yaml.dump(config, fh)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_root", type=str, required=True, help="Root config directory"
    )
    parser.add_argument("--start_seed", type=int, default=9000, help="Starting seed")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Dataset directory"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="Which dataset")
    parser.add_argument("--out_dir", type=str, default="run", help="Output directory")
    parser.add_argument("--cutout", type=str, default=True, help="Cutout Attribute")
    parser.add_argument(
        "--total_steps", type=int, default=1000, help="How many steps to take"
    )
    parser.add_argument(
        "--experiment", type=str, default="benchmarks", help="Experiment type"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--cutout_length", type=float, default=16, help="Train portion")
    parser.add_argument("--cutout_prob", type=float, default=1.0, help="Train portion")
    parser.add_argument(
        "--train_meta_fraction",
        type=float,
        default=0.7,
        help="Fraction of train and meta dataset from whole dataset. For example if \
        the value is set to 0.7, 70 %% of examples will belong to train + meta, and \
        the rest would go to validation set",
    )
    parser.add_argument(
        "--train_portion",
        type=float,
        default=0.7,
        help="Train portion from train-meta portion",
    )
    args = parser.parse_args()

    main(args)
