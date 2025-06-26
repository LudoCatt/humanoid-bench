#!/bin/bash
#SBATCH --account=ls_krausea
#SBATCH --job-name=test
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/test_out.txt
#SBATCH --error=logs/test_err.txt

module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.4.1
module load cudnn/9.2.0.82-12

# Environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate humanoidbench

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"

# Runtime flags
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS=--xla_gpu_autotune_level=2

# Evaluate
python test_policy.py --logdir /cluster/scratch/lcattaneo/DreamerV3/h1-walk-v0/seed1
python test_policy.py --logdir /cluster/scratch/lcattaneo/DreamerV3/h1-stand-v0/seed1
python test_policy.py --logdir /cluster/scratch/lcattaneo/DreamerV3/h1-reach-v0/seed1
python test_policy.py --logdir /cluster/scratch/lcattaneo/DreamerV3/h1-sit_simple-v0/seed1
python test_policy.py --logdir /cluster/scratch/lcattaneo/DreamerV3/h1-pole-v0/seed1
python test_policy.py --logdir /cluster/scratch/lcattaneo/DreamerV3/h1-crawl-v0/seed1
