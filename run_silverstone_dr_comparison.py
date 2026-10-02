#!/usr/bin/env python3
"""Matched fresh DR studies using the verified nominal PPO-Lag hyperparameters."""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import re
from queue import Queue, Empty
import subprocess
import sys
import time

import yaml
from report_experiments import report_results

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/silverstone_nominal_lambda_autotune"
FORMULATIONS = ("udr", "reward", "cost", "reward_cost")


def comparison_config(base, formulation, *, dr_strength=1., cost_scale=None):
    if formulation not in FORMULATIONS:
        raise ValueError(f"Unknown formulation: {formulation}")
    config = copy.deepcopy(base)
    for key in ("resume_checkpoint", "resume_from_update", "n_steps"):
        config.pop(key, None)
    config.update(seed=1, eval_seed=3000, total_timesteps=400000000,
                  rollout_length=256, batch_size=1024,
                  start_learning_rate=5e-6, end_learning_rate=5e-6)
    config["log_std_schedule"] = dict(init=-1.5, end=-1.5)
    jc = config["jax_sampler_ppo"]
    jc.update(domain_randomization=True, eval_domain_randomization=True,
              domain_randomization_profile="narrow",
              domain_randomization_ranges=copy.deepcopy(jc["domain_randomization_profiles"]["narrow"]),
              sampler="uniform" if formulation == "udr" else "reward_cost_gmmvi",
              gmm_formulation="reward" if formulation == "udr" else formulation,
              gmm_reward_fraction={"udr": 1., "reward": 1., "cost": 0., "reward_cost": .5}[formulation],
              num_envs=96, num_eval_envs=1024, num_evals=11,
              eval_episode_steps=12288, eval_episodes_per_dynamics=10,
              use_ppo_lag=True, safety_bound=.05, constraint_cost_type="collision",
              lagrangian_update_mode="completed_episode", allow_partial_first_episode=False,
              lagrangian_coef_rate=1., lagrangian_ema_decay=.5,
              lagrangian_min_completed_episodes=384, lagrangian_warmup_steps=10000000,
              initial_lambda_lagr=0., lagrangian_max=100., reset_state_on_rollout=False,
              normalize_cost_advantage=True, cost_advantage_std_floor=.001,
              eval_video=False, eval_video_percentiles=False, sampler_plot_samples=0)
    if dr_strength != 1.:
        from reevaluate_stronger_dr import stronger_ranges
        jc["domain_randomization_ranges"] = stronger_ranges(jc["domain_randomization_ranges"], dr_strength)
        jc["domain_randomization_profile"] = "stronger"
    if cost_scale is not None:
        if not math.isfinite(cost_scale) or cost_scale <= 0.:
            raise ValueError("Cost score scale must be finite and positive")
        jc["gmm_cost_score_scale"] = cost_scale
    return config


def comparison_cases(cost_scales=None):
    if cost_scales is None:
        return [(f, f, None) for f in FORMULATIONS]
    scales = list(dict.fromkeys(cost_scales))
    if not scales or any(not math.isfinite(s) or s <= 0. for s in scales):
        raise ValueError("Positive finite cost scales required")
    return [("udr", "udr", None), ("reward", "reward", None)] + [
        (f"{f}_scale{s:g}", f, s) for f in ("cost", "reward_cost") for s in scales]


def child_environment(gpu, output):
    if gpu not in (2, 3, 4):
        raise ValueError("Only GPUs 2/3/4 are allowed")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               JAX_COMPILATION_CACHE_DIR=str(output / "jax_compilation_cache"),
               JAX_COMPILATION_CACHE_MAX_SIZE=str(2 * 1024**3))
    env.pop("WANDB_SERVICE", None)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dr-strength", type=float, default=1.)
    parser.add_argument("--cost-scales", nargs="+", type=float)
    parser.add_argument("--wait-gpu2-config", type=Path,
                        help="Preserve an existing GPU 2 training run; wait for its exact config")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = SOURCE / "g06_screen1/rl_config.yaml"
    base = yaml.safe_load(source.read_text())
    gym = yaml.safe_load((SOURCE / "gym_config.yaml").read_text())
    gym.update(map="Silverstone", max_episode_steps=12288)
    gym_path = output / "gym_config.yaml"
    if gym_path.exists():
        assert yaml.safe_load(gym_path.read_text()) == gym
    else:
        gym_path.write_text(yaml.safe_dump(gym, sort_keys=False))
    cases = comparison_cases(args.cost_scales)
    for label, formulation, scale in cases:
        directory = output / label
        directory.mkdir(exist_ok=True)
        path = directory / "rl_config.yaml"
        config = comparison_config(base, formulation, dr_strength=args.dr_strength, cost_scale=scale)
        if path.exists():
            assert yaml.safe_load(path.read_text()) == config, "Frozen config mismatch"
        else:
            path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = dict(status="prepared", source_config=str(source), track="Silverstone",
                    formulations=list(FORMULATIONS), cases=cases, cost_scales=args.cost_scales,
                    dr_strength=args.dr_strength, training_seed=1, gpus=[2, 3, 4],
                    initialization="fresh; no nominal checkpoint reuse", total_timesteps=400000000,
                    optimizer_lr=5e-6, rollout_length=256, batch_size=1024,
                    domain_randomization=True, eval_domain_randomization=True,
                    dr_profile="narrow" if args.dr_strength == 1. else "stronger", budget=.05, horizon=12288,
                    eval_episodes=10240, results=[])
    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    save()
    if args.prepare_only:
        print(f"Prepared {len(cases)} frozen DR configurations in {output}", flush=True)
        return

    def worker(gpu, assigned):
        for label, formulation, scale in assigned:
            directory = output / label
            result_path = directory / "result.json"
            if result_path.exists():
                yield json.loads(result_path.read_text())
                continue
            log_path = directory / "train.log"
            if log_path.exists():
                raise RuntimeError(f"Existing unfinished run is preserved; inspect {log_path}")
            cmd = [sys.executable, "-u", "train/jax_sampler_ppo.py",
                   "--rl-config", str(directory / "rl_config.yaml"),
                   "--gym-config", str(gym_path), "--track", "Silverstone",
                   "--checkpoint-output", str(directory / "checkpoint.msgpack")]
            print(f"Starting {label} on GPU {gpu}", flush=True)
            with log_path.open("w") as log:
                run = subprocess.run(cmd, cwd=ROOT, env=child_environment(gpu, output),
                                     stdout=log, stderr=subprocess.STDOUT)
            if run.returncode:
                raise RuntimeError(f"{formulation} failed with exit {run.returncode}: {log_path}")
            ids = re.findall(r"View run at .*?/runs/([a-z0-9]+)", log_path.read_text())
            if not ids:
                raise RuntimeError(f"Missing W&B run ID for {formulation}")
            summary = json.loads((directory / "checkpoint.summary.json").read_text())
            expected = math.ceil(400000000 / (96 * 256))
            assert summary["training/progress/env_steps"] == expected * 96 * 256
            assert summary["evaluation/meta/update"] == expected
            assert summary["evaluation/meta/total_episodes"] == 10240
            assert (directory / "checkpoint.msgpack").stat().st_size > 0
            result = dict(track="Silverstone", formulation=formulation, seed=1, run_id=ids[-1],
                          reward=summary["evaluation/reward/episode_return_mean"],
                          cvar10=summary["evaluation/reward/cvar10"],
                          collision_rate=summary["evaluation/collision/rate"],
                          lambda_final=summary["training/constraint/lambda_lagr"],
                          steps=summary["training/progress/env_steps"], budget=.05, horizon=12288,
                          rollout_length=256, ppo_batch_size=1024, reset_state_on_rollout=False,
                          lagrangian_update_mode="completed_episode", domain_randomization=True,
                          eval_domain_randomization=True, dr_profile=manifest["dr_profile"], optimizer_lr=5e-6,
                          dr_strength=args.dr_strength)
            if args.cost_scales is not None:
                result["tuning_label"] = label
            jc = yaml.safe_load((directory / "rl_config.yaml").read_text())["jax_sampler_ppo"]
            result.update(cost_score_scale=jc["gmm_cost_score_scale"],
                          cost_dual_lr=jc["gmm_cost_dual_lr"],
                          cost_dual_update=jc["gmm_cost_dual_update"],
                          cost_dual_ema_decay=jc["gmm_cost_dual_ema_decay"],
                          cost_dual_direction="adversarial")
            result_path.write_text(json.dumps(result, indent=2))
            yield result

    # Three GPU workers; the fourth formulation follows UDR on GPU 2.
    # Completion reporting is serialized so children never inherit report services.
    completed = Queue()
    pending = Queue()
    if args.cost_scales is not None:
        for case in cases:
            pending.put(case)
    def consume(gpu, assigned):
        if gpu == 2 and args.wait_gpu2_config:
            from tune_silverstone_nominal_lagrange import training_pids
            previous = args.wait_gpu2_config.resolve()
            while training_pids(previous):
                print(f"GPU 2 waits for preserved training: {previous}", flush=True)
                time.sleep(30)
        if assigned is None:
            # A waiting GPU does not reserve work; free GPUs take the next case.
            while True:
                try:
                    case = pending.get_nowait()
                except Empty:
                    return
                for result in worker(gpu, [case]):
                    completed.put(result)
            return
        for result in worker(gpu, assigned):
            completed.put(result)
    manifest["status"] = "running"
    save()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            if args.cost_scales is None:
                assignments = [(2, [cases[0], cases[3]]), (3, [cases[1]]), (4, [cases[2]])]
            else:
                assignments = [(3, None), (4, None), (2, None)]
            futures = [pool.submit(consume, gpu, assigned) for gpu, assigned in assignments]
            while len(manifest["results"]) < len(cases):
                try:
                    result = completed.get(timeout=10)
                except Empty:
                    for future in futures:
                        if future.done():
                            future.result()
                    if all(future.done() for future in futures) and completed.empty():
                        raise RuntimeError("Workers ended without all planned results")
                    continue
                manifest["results"].append(result)
                save()
                report_results(manifest["results"], output / f"report_stage{len(manifest['results'])}",
                               "Silverstone tuned PPO-Lag DR comparison: completed runs")
        (output / "final.json").write_text(json.dumps(manifest["results"], indent=2))
        report_results(manifest["results"], output / "final_report", "Silverstone DR comparison completed")
        manifest["status"] = "completed"
        save()
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
