#!/usr/bin/env bash
# Train one JAX sampler-PPO run for each requested F1 circuit and seed.
# WandB is enabled by default: its project and other settings come unchanged
# from train/config/gym_config.yaml and train/config/rl_config.yaml.

set -euo pipefail

tracks=(Spielberg Catalunya Silverstone) # AUT, ESP, GBR
seed=1 # matches train/config/rl_config.yaml
python_bin="${PYTHON_BIN:-/home/tjrcjf410/miniconda3/envs/f1/bin/python}"

if [[ ! -x "$python_bin" ]]; then
    echo "Python interpreter not found or not executable: $python_bin" >&2
    exit 1
fi

for track in "${tracks[@]}"; do
    "$python_bin" train/jax_sampler_ppo.py --track "$track" --seed "$seed"
done
