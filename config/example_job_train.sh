#!/bin/bash
#SBATCH --job-name=cslr2_train            # job name
#SBATCH --account=vvh@v100                # project code
#SBATCH -C v100-32g                       # choose nodes with 32G GPU memory
#SBATCH --ntasks=4                        # number of tasks (GPUs)
#SBATCH --ntasks-per-node=4               # number of tasks (GPUs) per node
#SBATCH --gres=gpu:4                      # number of GPUs per node
#SBATCH --qos=qos_gpu-t3                  # (20h) jobs
#SBATCH --cpus-per-task=10                # number of cores per tasks
#SBATCH --hint=nomultithread              # we get physical cores not logical
#SBATCH --time=20:00:00                   # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/cslr2_train%j.out   # output file name
#SBATCH --error=logs/cslr2_train%j.err    # error file name

set -x # echo launched commands

module purge

. ${WORK}/miniconda3/etc/profile.d/conda.sh
conda activate gpu_pytorch_1.12

cd "${WORK}/code/cslr2"

export HYDRA_FULL_ERROR=1 # to get better error messages if job crashes
export WANDB_MODE=offline

echo "Do not forget to set distributed: True in config/cslr2.yaml"

srun python main.py run_name=cslr2

