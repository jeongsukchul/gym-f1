#!/usr/bin/env python3
"""Queue a faster cost-dual LR comparison after the existing seed study."""

import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from report_experiments import report_results

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "outputs/cost_scale_multiseed_ema05"
OUT = ROOT / "outputs/cost_beta_lr05"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-controller-pid", type=int, required=True)
    args = parser.parse_args()
    if args.wait_for_controller_pid <= 0:
        raise ValueError("Controller PID must be positive")
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "manifest.json").write_text(json.dumps(dict(
        status="queued", baseline=str(BASELINE), cost_dual_lr=0.5,
        cost_dual_ema_decay=0.5, scales=[20, 40, 80], seeds=[1, 2, 3], new_runs=27,
    ), indent=2))
    print("Waiting for existing three-seed study; GPUs 2/3/4 remain with that study.", flush=True)
    while not (BASELINE / "final.json").exists():
        try:
            os.kill(args.wait_for_controller_pid, 0)
        except ProcessLookupError:
            raise RuntimeError("Baseline controller stopped before final results; queue remains unexecuted")
        time.sleep(30)
    results = json.loads((BASELINE / "final.json").read_text())
    for result in results:
        if result["formulation"] == "cost":
            result.setdefault("cost_dual_lr", 0.1)
    (OUT / "base_rl_config.yaml").write_text((BASELINE / "base_rl_config.yaml").read_text())
    (OUT / "base_gym_config.yaml").write_text((BASELINE / "base_gym_config.yaml").read_text())
    stages = [(scale, seed) for seed in (1, 2, 3) for scale in (20, 40, 80)]
    for index, (scale, seed) in enumerate(stages, 1):
        directory = OUT / f"scale{scale}_seed{seed}"
        print(f"Stage {index}/9: scale={scale} seed={seed} cost_dual_lr=0.5", flush=True)
        subprocess.run([sys.executable, "-u", "run_collision_long_horizon.py",
            "--formulations", "cost", "--seed", str(seed),
            "--base-rl-config", str(OUT / "base_rl_config.yaml"),
            "--base-gym-config", str(OUT / "base_gym_config.yaml"),
            "--cost-dual-ema-decay", "0.5", "--cost-dual-lr", "0.5",
            "--cost-score-scale", str(scale), "--output-dir", str(directory)],
            cwd=ROOT, check=True)
        results.extend(json.loads((directory / "final.json").read_text()))
        (OUT / "results_so_far.json").write_text(json.dumps(results, indent=2))
        report_results(results, OUT / f"report_stage{index:02d}", f"Cost beta LR comparison: stage {index}/9")
    (OUT / "final.json").write_text(json.dumps(results, indent=2))
    report_results(results, OUT / "final_report", "Cost beta LR comparison completed")


if __name__ == "__main__":
    main()
