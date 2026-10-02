#!/usr/bin/env python3
"""Autonomous, bounded lambda search on GPU 2/3/4, followed by confirmation.

Each trial is fresh training with seed 1. Configs/results/logs are retained.
Candidates are selected using long-horizon collision budget excess, with a
reward floor to reject policies that satisfy safety by making no progress.
"""

import concurrent.futures
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml
from report_experiments import report_results

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/lambda_autotune"
TRACKS = [(2, "Spielberg", 350.0), (3, "Catalunya", 800.0), (4, "Silverstone", 800.0)]
BASE = yaml.safe_load((ROOT / "train/config/jax/rl_config.yaml").read_text())


def run_trial(label, mode, rate, initial, updates, sampler, track_info):
    gpu, track, floor = track_info
    trial = OUT / f"{label}_{track}_{sampler}"
    trial.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(BASE)
    config["seed"] = 1
    config["total_timesteps"] = updates * 256 * 2048
    jc = config["jax_sampler_ppo"]
    jc.update(
        constraint_cost_type="collision", safety_bound=0.30,
        lagrangian_update_mode=mode, lagrangian_coef_rate=rate,
        initial_lambda_lagr=initial, lagrangian_max=100.0,
        lagrangian_ema_decay=0.9 if mode != "per_step" else 0.0,
        sampler="uniform" if sampler == "udr" else "reward_cost_gmmvi",
        gmm_formulation="reward", gmm_reward_fraction=1.0,
        num_evals=3 if updates < 382 else 10,
        num_eval_envs=256 if updates < 382 else 1024,
        eval_episodes_per_dynamics=2 if updates < 382 else 10,
        eval_episode_steps=12288,
        eval_render=False, eval_video=False,
    )
    config_path = trial / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    cmd = [sys.executable, "-u", "train/jax_sampler_ppo.py",
           "--rl-config", str(config_path), "--track", track,
           "--dr-profile", "narrow", "--updates", str(updates)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    started = time.time()
    with (trial / "train.log").open("w") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    log_text = (trial / "train.log").read_text()
    ids = re.findall(r"View run at .*?/runs/([a-z0-9]+)", log_text)
    if result.returncode or not ids:
        raise RuntimeError(f"Trial failed: {trial}; exit={result.returncode}")
    run_id = ids[-1]
    files = list((ROOT / "wandb").glob(f"run-*-{run_id}/files/wandb-summary.json"))
    if len(files) != 1:
        raise RuntimeError(f"Missing W&B summary for {run_id}")
    summary = json.loads(files[0].read_text())
    reward = summary["evaluation/reward/episode_return_mean"]
    collision = summary["evaluation/collision/rate"]
    record = dict(label=label, mode=mode, rate=rate, initial=initial,
                  updates=updates, sampler=sampler, track=track, run_id=run_id,
                  reward=reward, collision=collision, reward_floor=floor,
                  cvar10=summary["evaluation/reward/cvar10"],
                  lambda_final=summary["training/constraint/lambda_lagr"],
                  elapsed=time.time()-started)
    (trial / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    with (OUT / "completed_trials.jsonl").open("a") as progress:
        progress.write(json.dumps(record) + "\n")
    return record


def stage(candidates, updates, sampler="udr"):
    def worker(track_info):
        return [run_trial(*candidate, updates, sampler, track_info) for candidate in candidates]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        groups = list(pool.map(worker, TRACKS))
    return [record for group in groups for record in group]


def select(records):
    labels = sorted({r["label"] for r in records})
    def score(label):
        rows = [r for r in records if r["label"] == label]
        return (sum(r["reward"] < r["reward_floor"] for r in rows),
                sum(max(0.0, r["collision"] - 0.30) for r in rows),
                -sum(r["reward"] for r in rows))
    winner = min(labels, key=score)
    row = next(r for r in records if r["label"] == winner)
    return row, {label: score(label) for label in labels}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    screening = stage([
        ("screen_legacy", "per_step", 1.0, 1.0),
        ("screen_slow", "first_episode", 0.1, 1.0),
        ("screen_medium", "first_episode", 0.5, 1.0),
        ("screen_fast", "first_episode", 2.0, 1.0),
    ], 64)
    winner, scores = select(screening)
    (OUT / "screening.json").write_text(json.dumps(dict(results=screening, scores=scores, winner=winner), indent=2))
    # Refine rate around the best screened candidate, and test stronger init.
    rate = winner["rate"]
    mode = winner["mode"]
    refinement = stage([
        ("refine_low", mode, rate / 3.0, 1.0),
        ("refine_center", mode, rate, 1.0),
        ("refine_high", mode, rate * 3.0, 1.0),
        ("refine_initial5", mode, rate, 5.0),
    ], 128)
    winner, scores = select(refinement)
    (OUT / "refinement.json").write_text(json.dumps(dict(results=refinement, scores=scores, winner=winner), indent=2))
    selected_config = copy.deepcopy(BASE)
    selected_config["jax_sampler_ppo"].update(
        lagrangian_update_mode=winner["mode"],
        lagrangian_coef_rate=winner["rate"],
        initial_lambda_lagr=winner["initial"],
        lagrangian_ema_decay=0.9 if winner["mode"] != "per_step" else 0.0,
        lagrangian_max=100.0,
    )
    (OUT / "selected_config.yaml").write_text(yaml.safe_dump(selected_config, sort_keys=False))
    candidate = ("confirm", winner["mode"], winner["rate"], winner["initial"])
    confirmations = stage([candidate], 382, "udr") + stage([candidate], 382, "gmmvi")
    satisfied = all(r["collision"] <= 0.30 and r["reward"] >= r["reward_floor"] for r in confirmations)
    (OUT / "final.json").write_text(json.dumps(dict(
        selected=winner, results=confirmations, all_constraints_satisfied=satisfied,
        note="Search completed; failure to meet budget is not reported as success.",
    ), indent=2))
    print(f"Search completed. All constraints satisfied: {satisfied}", flush=True)
    report_results(confirmations, OUT, "Lambda tuning confirmation completed")


if __name__ == "__main__":
    main()
