import argparse
import os
import yaml


def main(args):
    folder = os.path.join(
        args.config_root,
        args.experiment,
        args.dataset,
    )

    print(folder)
    os.makedirs(folder, exist_ok=True)
    args.start_seed = int(args.start_seed)

    config = {
        'config_type': args.experiment,
        'seed': args.start_seed,
        'dataset': args.dataset,
        'out_dir': args.out_dir,
        'batch_size': args.batch_size,
        'train_portion': args.train_portion,
        'meta_train_fraction': args.meta_train_fraction,
        'cutout': args.cutout,
        'data_path': args.data_path,
        'total_steps': args.total_steps,
        'cutout_length': args.cutout_length,
        'cutout_prob': args.cutout_prob,
    }

    with open(folder + f'/config_{args.start_seed}.yaml', 'w') as fh:
        yaml.dump(config, fh)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_root", type=str, required=True, help="Root config directory")
    parser.add_argument("--start_seed", type=int, default=9000, help="Starting seed")
    parser.add_argument("--data_path", type=str, required=True, help="Dataset directory")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Which dataset")
    parser.add_argument("--out_dir", type=str, default='run', help="Output directory")
    parser.add_argument("--cutout", type=str, default=True, help="Cutout Attribute")
    parser.add_argument("--total_steps", type=int, default=1000, help="How many steps to take")
    parser.add_argument("--experiment", type=str, default='benchmarks', help="Experiment type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_portion", type=float, default=0.7, help="Train portion")
    parser.add_argument("--cutout_length", type=float, default=16, help="Train portion")
    parser.add_argument("--cutout_prob", type=float, default=1.0, help="Train portion")
    parser.add_argument("--meta_train_fraction", type=float, default=0.7,
                        help="Fraction of meta from the train portoon")
    args = parser.parse_args()

    main(args)
