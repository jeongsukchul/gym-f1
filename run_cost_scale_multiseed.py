#!/usr/bin/env python3
"""Cost scales 20/40/80 and UDR/reward controls over seeds 1, 2, 3."""

import json
import argparse
import os
import time
from pathlib import Path
import subprocess
import sys

from report_experiments import report_results

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/cost_scale_multiseed"


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--cost-dual-ema-decay", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cost_scale_multiseed_ema05")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-stage-pid", type=int, default=None)
    args = parser.parse_args()
    if not 0.0 <= args.cost_dual_ema_decay < 1.0:
        raise ValueError("EMA decay must lie between zero and one")
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=args.resume)
    baseline = json.loads((ROOT / "outputs/collision_long_horizon_budget005/final.json").read_text())
    results = [dict(r, seed=1) for r in baseline if r["formulation"] in {"udr", "reward"}]
    # Freeze the prior scale-20 settings so later workspace edits cannot
    # silently change conditions between stages of this comparison.
    base_rl = OUT / "base_rl_config.yaml"
    base_gym = OUT / "base_gym_config.yaml"
    if not args.resume:
        base_rl.write_text((ROOT / "outputs/collision_cost_scaled_20/Spielberg_cost/rl_config.yaml").read_text())
        base_gym.write_text((ROOT / "outputs/collision_cost_scaled_20/gym_config.yaml").read_text())
    # Early high-scale trials are followed by replication and control seeds.
    stages = [(20, 1, "cost"), (40, 1, "cost"), (80, 1, "cost")]
    for seed in (2, 3):
        stages.extend([(20, seed, "cost"), (40, seed, "cost"), (80, seed, "cost"),
                       (1, seed, "udr"), (1, seed, "reward")])
    (OUT / "manifest.json").write_text(json.dumps(dict(
        seeds=[1, 2, 3], scales=[20, 40, 80], budget=0.05, horizon=12288,
        reused_runs=len(results), new_runs=len(stages) * 3,
        cost_dual_ema_decay=args.cost_dual_ema_decay,
        stages=stages, gpu_track={2: "Spielberg", 3: "Catalunya", 4: "Silverstone"},
    ), indent=2))
    for index, (scale, seed, formulation) in enumerate(stages, 1):
        directory = OUT / f"{formulation}_scale{scale}_seed{seed}"
        if args.resume and (directory / "final.json").exists():
            results.extend(json.loads((directory / "final.json").read_text()))
            print(f"Reusing completed stage {index}/{len(stages)}", flush=True)
            continue
        if args.resume and directory.exists():
            if args.wait_for_stage_pid is None:
                raise RuntimeError(f"Incomplete stage requires --wait-for-stage-pid: {directory}")
            print(f"Waiting for existing stage {index}, PID={args.wait_for_stage_pid}", flush=True)
            while not (directory / "final.json").exists():
                try:
                    os.kill(args.wait_for_stage_pid, 0)
                except ProcessLookupError:
                    raise RuntimeError(f"Existing stage stopped without final results: {directory}")
                time.sleep(5)
            records = json.loads((directory / "final.json").read_text())
            results.extend(records)
            (OUT / "results_so_far.json").write_text(json.dumps(results, indent=2))
            report_results(results, OUT / f"report_stage{index:02d}",
                f"Cost scale multi-seed comparison: stage {index}/{len(stages)}")
            continue
        print(f"Stage {index}/{len(stages)}: formulation={formulation} scale={scale} seed={seed}", flush=True)
        subprocess.run([sys.executable, "-u", "run_collision_long_horizon.py",
            "--formulations", formulation, "--seed", str(seed),
            "--base-rl-config", str(base_rl), "--base-gym-config", str(base_gym),
            "--cost-dual-ema-decay", str(args.cost_dual_ema_decay),
            "--cost-score-scale", str(scale), "--output-dir", str(directory)],
            cwd=ROOT, check=True)
        records = json.loads((directory / "final.json").read_text())
        results.extend(records)
        (OUT / "results_so_far.json").write_text(json.dumps(results, indent=2))
        report_results(results, OUT / f"report_stage{index:02d}",
            f"Cost scale multi-seed comparison: stage {index}/{len(stages)}")
    (OUT / "final.json").write_text(json.dumps(results, indent=2))
    report_results(results, OUT / "final_report", "Cost scale multi-seed comparison completed")
    print(f"Completed comparison of {len(results)} runs across three training seeds.", flush=True)


if __name__ == "__main__":
    main()
