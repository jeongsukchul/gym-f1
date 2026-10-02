#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/tjrcjf410/miniconda3/envs/f1/bin/python}"
for scale in 5 20; do
  echo "Starting cost sampler scaling=$scale on GPU 2/3/4"
  "$python_bin" -u run_collision_long_horizon.py \
    --formulations cost \
    --cost-score-scale "$scale" \
    --output-dir "outputs/collision_cost_scaled_${scale}"
done

"$python_bin" -u report_experiments.py \
  outputs/collision_long_horizon_budget005/final.json \
  outputs/collision_cost_scaled_5/final.json \
  outputs/collision_cost_scaled_20/final.json \
  --output-dir outputs/collision_cost_scaling_report \
  --title "Cost scaling versus UDR and reward GMMVI"
