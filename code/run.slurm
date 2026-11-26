#!/bin/bash
#SBATCH --job-name=5026-anomaly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
root=$1
python=$2
inventory=$3
attempt=$4
category=$5
device=$6
shift 6
if [[ "$category" == array ]]; then
    categories=(bottle grid screw)
    category=${categories[${SLURM_ARRAY_TASK_ID:?}]}
fi

cd "$root"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
date -Is
hostname
if [[ "$device" == cuda ]]; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi
"$python" -B -u code/experiment.py run --category "$category" --device "$device" \
    --data inputs/mvtec --model inputs/dinov2-small --inventory "$inventory" \
    --cache runs/features --out "$attempt/$category" "$@"
date -Is
