#!/bin/bash
#SBATCH -p relea_gpu-rtx2080 # mlhiwidlc_gpu-rtx2080     # partition (queue)
#SBATCH --gres=gpu:5
#SBATCH -o logs/slurm_logs/%A.%N.out       # STDOUT  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/slurm_logs/%A.%N.err       # STDERR  (the folder log has to exist) %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -J orm_cifar              # sets the job name. 
#SBATCH -a 1-5 # array size
# Print some information about the job to STDOUT

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

source ~/.bashrc
conda activate orm
python src/orm/runners/runner_cifar10.py --seed ${SLURM_ARRAY_TASK_ID}

end=`date +%s`
runtime=$((end-start))

echo "DONE";
echo "Finished at $(date)";