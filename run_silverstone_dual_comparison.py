#!/usr/bin/env python3
"""One-seed Silverstone screening of additive versus multiplicative duals."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

from report_experiments import report_results

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/silverstone_dual_scale10_rollout256"
BASE = ROOT / "outputs/cost_scale_multiseed_ema05"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    modes = [(2, "linear", 0.05), (3, "log", 0.05), (4, "log", 0.2)]
    manifest = dict(status="running", track="Silverstone", seed=1, cost_dual_direction="adversarial",
                    scale=10, cost_ema=0.5, budget=0.05, unroll_length=256,
                    reset_state_on_rollout=False, lagrangian_update_mode="completed_episode",
                    comparisons=[dict(gpu=g, update=m, lr=lr) for g, m, lr in modes])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name in ("rl", "gym"):
        (OUT / f"base_{name}_config.yaml").write_text(
            (BASE / f"base_{name}_config.yaml").read_text())

    def worker(item):
        gpu, mode, lr = item
        directory = OUT / f"{mode}_lr{lr:g}"
        cmd = [sys.executable, "-u", "run_collision_long_horizon.py",
               "--tracks", "Silverstone", "--gpus", str(gpu),
               "--formulations", "cost", "--seed", "1",
               "--unroll-length", "256",
               "--cost-score-scale", "10", "--cost-dual-ema-decay", "0.5",
               "--cost-dual-update", mode, "--cost-dual-lr", str(lr),
               "--base-rl-config", str(OUT / "base_rl_config.yaml"),
               "--base-gym-config", str(OUT / "base_gym_config.yaml"),
               "--output-dir", str(directory)]
        print(f"Starting Silverstone GPU={gpu} update={mode} lr={lr}", flush=True)
        with (OUT / f"{mode}_lr{lr:g}_controller.log").open("w") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads((directory / "final.json").read_text())

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, item) for item in modes]
        for future in as_completed(futures):
            results.extend(future.result())
            (OUT / "results_so_far.json").write_text(json.dumps(results, indent=2))
            report_results(results, OUT / f"report_stage{len(results)}",
                           f"Silverstone dual screening: {len(results)}/3 completed")
    (OUT / "final.json").write_text(json.dumps(results, indent=2))
    report_results(results, OUT / "final_report", "Silverstone dual screening completed (one seed)")
    manifest["status"] = "completed"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
