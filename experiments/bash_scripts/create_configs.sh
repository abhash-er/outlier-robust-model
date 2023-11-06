#!/bin/bash

# search space and datasets:
dataset=$1
start_seed=$2
if [ -z "$start_seed" ]
then
    start_seed=0
fi

if [ -z "$dataset" ]
then
    dataset=cifar10
fi

# folders:
data_path='data'
config_root='configs'
out_dir=run

# other variables:
trials=10
end_seed=$(($start_seed + $trials - 1))

# create config files
for i in $(seq $start_seed $end_seed)
  do
    python experiments/make_configs.py --start_seed $i \
    --out_dir $out_dir --dataset=$dataset \
    --experiment outlier_robust --config_root=$config_root --data_path=$data_path
done