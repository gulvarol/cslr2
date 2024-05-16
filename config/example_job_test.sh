#!/bin/bash
#SBATCH --job-name=cslr2_test             # job name
#SBATCH --account=vvh@v100                # project code
#SBATCH -C v100-32g                       # choose nodes with 32G GPU memory
#SBATCH --ntasks-per-node=1               # number of MPI tasks per node
#SBATCH --gres=gpu:1                      # number of GPUs per node
#SBATCH --qos=qos_gpu-t3                  # (20h) jobs
#SBATCH --cpus-per-task=8                 # number of cores per tasks
#SBATCH --hint=nomultithread              # we get physical cores not logical
#SBATCH --time=00:30:00                   # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/cslr2_test%j.out    # output file name
#SBATCH --error=logs/cslr2_test%j.err     # error file namea

set -x # echo launched commands

module purge

. ${WORK}/miniconda3/etc/profile.d/conda.sh
conda activate gpu_pytorch_1.12

cd "${WORK}/code/cslr2"

export HYDRA_FULL_ERROR=1 # to get better error messages if job crashes
export WANDB_MODE=offline

# Set folder where to save outputs
export RUN_NAME=runs/cslr2
# Set checkpoint
export PATH_TO_CHECKPOINT=${RUN_NAME}/models/model_best.pth

# 1) Evaluate sentence retrieval
python main.py run_name=cslr2_test checkpoint=${PATH_TO_CHECKPOINT} test=True

# 2) Evaluate CSLR (in two steps)
python extract_for_eval.py checkpoint=${PATH_TO_CHECKPOINT}

python frame_level_evaluation.py prediction_pickle_files=${RUN_NAME}/cslr/eval/nn

