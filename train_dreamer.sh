#!/bin/bash
#SBATCH --account=ls_krausea
#SBATCH --job-name=crawl
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=32G
#SBATCH --output=logs/crawl_out.txt
#SBATCH --error=logs/crawl_err.txt

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

# Experiment parameters
TASK="h1-crawl-v0"
SEED=1
STEPS=10_000_000
SAVE=1800
EVAL=1000000
LOGDIR=/cluster/scratch/lcattaneo/DreamerV3/${TASK}/seed${SEED}
mkdir -p "$LOGDIR"

export WANDB_MODE=online
export WANDB_API_KEY="d1777559d9715b506ab22a26ae90e9196d940a1e"

# Train DreamerV3
srun python -m embodied.agents.dreamerv3.train \
      --configs humanoid_benchmark \
      --task humanoid_${TASK} \
      --method dreamer \
      --seed=${SEED} \
      --logdir=${LOGDIR} \
      --run.num_envs=4 \
      --run.steps=${STEPS} \
      --run.save_every=${SAVE} \
      --run.eval_every=${EVAL} \
      --run.log_video_fps=30 \
      --run.wandb=True \
      --run.wandb_entity=ludocatt-eth-z-rich \
      --run.wandb_project=DreamerV3