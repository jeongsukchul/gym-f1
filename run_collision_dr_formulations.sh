#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_ID TRACK" >&2
  exit 2
fi

gpu_id="$1"
track="$2"
seed=1
python_bin="${PYTHON_BIN:-/home/tjrcjf410/miniconda3/envs/f1/bin/python}"

export CUDA_VISIBLE_DEVICES="$gpu_id"
echo "GPU physical=$gpu_id track=$track seed=$seed constraint_cost=collision safety_bound=0.05"

echo "Starting track=$track formulation=uniform"
"$python_bin" train/jax_sampler_ppo.py \
  --track "$track" \
  --seed "$seed" \
  --dr-profile narrow \
  --sampler uniform \
  --constraint-cost collision \
  --safety-bound 0.05

for formulation in reward cost reward_cost; do
  echo "Starting track=$track formulation=$formulation"
  "$python_bin" train/jax_sampler_ppo.py \
    --track "$track" \
    --seed "$seed" \
    --dr-profile narrow \
    --gmm-formulation "$formulation" \
    --constraint-cost collision \
    --safety-bound 0.05
done
