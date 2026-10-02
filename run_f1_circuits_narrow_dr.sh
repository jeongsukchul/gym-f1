#!/usr/bin/env bash
# PPO-Lag, one seed per requested F1 circuit, with the named narrow DR profile.

set -euo pipefail

tracks=(Spielberg Catalunya Silverstone) # AUT, ESP, GBR
seed=1
python_bin="${PYTHON_BIN:-/home/tjrcjf410/miniconda3/envs/f1/bin/python}"
# JAX sees only these physical GPUs.  The current single-device JIT trainer
# executes on visible device 0 (physical GPU 2); multi-GPU sharding requires
# an explicit pmap/sharding implementation.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4}"

if [[ ! -x "$python_bin" ]]; then
    echo "Python interpreter not found or not executable: $python_bin" >&2
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for track in "${tracks[@]}"; do
    "$python_bin" train/jax_sampler_ppo.py --track "$track" --seed "$seed" --dr-profile narrow
done
