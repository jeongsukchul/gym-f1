#!/usr/bin/env python3
"""Wait for the cost screen, then run matched UDR/reward controls on Silverstone."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

from report_experiments import report_results

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs/silverstone_dual_scale10_rollout256"
OUT = ROOT / "outputs/silverstone_rollout256_udr_reward"


def controller_running(pid):
    """Avoid mistaking a zombie or reused PID for the original controller."""
    proc = Path("/proc") / str(pid)
    try:
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
        return stat[0] != "Z" and b"run_silverstone_dual_comparison.py" in command
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-controller-pid", required=True, type=int)
    args = parser.parse_args()
    if args.wait_for_controller_pid <= 0:
        parser.error("Controller PID must be positive")
    OUT.mkdir(parents=True, exist_ok=False)
    # Freeze the actual running configuration, not mutable workspace defaults.
    (OUT / "base_rl_config.yaml").write_text(
        (BASE / "log_lr0.05/Silverstone_cost/rl_config.yaml").read_text())
    (OUT / "base_gym_config.yaml").write_text(
        (BASE / "log_lr0.05/gym_config.yaml").read_text())
    manifest = dict(status="queued", prerequisite=str(BASE),
                    prerequisite_pid=args.wait_for_controller_pid,
                    track="Silverstone", seed=1, rollout_length=256,
                    reset_state_on_rollout=False, lagrangian_update_mode="completed_episode",
                    total_timesteps=200000000, budget=0.05, horizon=12288,
                    jobs=[dict(gpu=2, formulation="udr"), dict(gpu=3, formulation="reward")])
    manifest_path = OUT / "manifest.json"

    def save_status(status, error=None):
        manifest["status"] = status
        if error is not None:
            manifest["error"] = str(error)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    save_status("queued")
    print("Waiting for all three cost runs and their controller to finish.", flush=True)
    while controller_running(args.wait_for_controller_pid):
        time.sleep(30)
    if not (BASE / "final.json").exists():
        save_status("blocked", "Cost controller stopped without final results; controls not started")
        raise RuntimeError(manifest["error"])
    cost_results = json.loads((BASE / "final.json").read_text())
    if (len(cost_results) != 3 or any(
            r["track"] != "Silverstone" or r["formulation"] != "cost"
            or r["steps"] < 200000000 for r in cost_results)):
        save_status("blocked", "Incomplete or unexpected cost prerequisite results")
        raise RuntimeError(manifest["error"])
    save_status("running")

    def worker(gpu, formulation):
        directory = OUT / formulation
        cmd = [sys.executable, "-u", "run_collision_long_horizon.py",
               "--tracks", "Silverstone", "--gpus", str(gpu),
               "--formulations", formulation, "--seed", "1",
               "--unroll-length", "256", "--cost-score-scale", "10",
               "--base-rl-config", str(OUT / "base_rl_config.yaml"),
               "--base-gym-config", str(OUT / "base_gym_config.yaml"),
               "--output-dir", str(directory)]
        print(f"Starting {formulation} on GPU {gpu}", flush=True)
        with (OUT / f"{formulation}_controller.log").open("w") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads((directory / "final.json").read_text())

    controls = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, 2, "udr"), pool.submit(worker, 3, "reward")]
            for future in as_completed(futures):
                controls.extend(future.result())
                combined = cost_results + controls
                (OUT / "results_so_far.json").write_text(json.dumps(combined, indent=2))
                report_results(combined, OUT / f"report_stage{len(controls)}",
                               f"Silverstone controls: {len(controls)}/2 completed")
        (OUT / "final.json").write_text(json.dumps(cost_results + controls, indent=2))
        report_results(cost_results + controls, OUT / "final_report",
                       "Silverstone cost / UDR / reward GMMVI comparison completed")
        save_status("completed")
    except Exception as exc:
        save_status("failed", exc)
        raise


if __name__ == "__main__":
    main()
