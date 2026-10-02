#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_ID TRACK" >&2
  exit 2
fi

gpu_id="$1"
track="$2"
seed=1
budget=0.30
eval_horizon=12288
python_bin="${PYTHON_BIN:-/home/tjrcjf410/miniconda3/envs/f1/bin/python}"

export CUDA_VISIBLE_DEVICES="$gpu_id"
echo "GPU physical=$gpu_id track=$track seed=$seed constraint_cost=collision safety_bound=$budget eval_horizon=$eval_horizon"

echo "Starting track=$track formulation=udr"
"$python_bin" train/jax_sampler_ppo.py \
  --track "$track" \
  --seed "$seed" \
  --dr-profile narrow \
  --sampler uniform \
  --constraint-cost collision \
  --safety-bound "$budget" \
  --eval-episode-steps "$eval_horizon"

echo "Starting track=$track formulation=reward_gmmvi"
"$python_bin" train/jax_sampler_ppo.py \
  --track "$track" \
  --seed "$seed" \
  --dr-profile narrow \
  --gmm-formulation reward \
  --constraint-cost collision \
  --safety-bound "$budget" \
  --eval-episode-steps "$eval_horizon"
