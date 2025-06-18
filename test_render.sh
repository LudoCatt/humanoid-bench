#!/bin/bash
#SBATCH --job-name=hb_render
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/out.txt
#SBATCH --error=logs/err.txt

module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.4.1

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false  
export XLA_FLAGS=--xla_gpu_autotune_level=2
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate humanoidbench

python -m humanoid_bench.test_env \
       --env h1-walk-v0 \
       --steps 500 \
       --width 720 --height 480 \
       --save_video /cluster/scratch/lcattaneo/rendered