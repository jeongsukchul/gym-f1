#!/usr/bin/env python3
"""Compare UDR, reward GMMVI and cost GMMVI at matched long horizons."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml
from report_experiments import report_results

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/collision_long_horizon_budget005"


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--formulations", nargs="+", choices=("udr", "reward", "cost", "reward_cost"), default=["udr", "reward", "cost"])
    parser.add_argument("--cost-score-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None, help="PPO optimizer minibatch size")
    parser.add_argument("--unroll-length", type=int, default=6144,
                        help="PPO rollout length; short rollouts use state continuation")
    parser.add_argument("--cost-dual-ema-decay", type=float, default=None)
    parser.add_argument("--cost-dual-lr", type=float, default=None)
    parser.add_argument("--cost-dual-update", choices=("linear", "log"), default=None)
    parser.add_argument("--lagrangian-update-mode", choices=("first_episode", "completed_episode"), default="completed_episode")
    parser.add_argument("--allow-partial-first-episode", action="store_true")
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--tracks", nargs="+", choices=("Spielberg", "Catalunya", "Silverstone"), default=["Spielberg", "Catalunya", "Silverstone"])
    parser.add_argument("--gpus", nargs="+", type=int, choices=(2, 3, 4), default=[2, 3, 4])
    parser.add_argument("--base-rl-config", type=Path, default=ROOT / "train/config/jax/rl_config.yaml")
    parser.add_argument("--base-gym-config", type=Path, default=ROOT / "train/config/jax/gym_config.yaml")
    args = parser.parse_args()
    if len(args.tracks) != len(args.gpus):
        parser.error("Provide one GPU per track")
    if args.unroll_length < 1:
        parser.error("Rollout length must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("Batch size must be positive")
    if args.no_domain_randomization and args.formulations != ["udr"]:
        parser.error("Non-DR training uses --formulations udr to disable adaptive samplers")
    if args.lagrangian_update_mode == "first_episode" and args.unroll_length < 6144 and not args.allow_partial_first_episode:
        parser.error("Short first_episode needs explicit --allow-partial-first-episode")
    if not 0.0 < args.cost_score_scale < float("inf"):
        raise ValueError("Cost score scale must be finite and positive")
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    rl = yaml.safe_load(args.base_rl_config.read_text())
    gym = yaml.safe_load(args.base_gym_config.read_text())
    gym["max_episode_steps"] = 12288
    gym_path = OUT / "gym_config.yaml"
    gym_path.write_text(yaml.safe_dump(gym, sort_keys=False))
    rl.pop("n_steps", None)  # Migrate frozen legacy configs before overriding.
    rl["rollout_length"] = args.unroll_length
    if args.batch_size is not None:
        rl["batch_size"] = args.batch_size
    rl["total_timesteps"] = 200000000
    rl["seed"] = args.seed
    rl["jax_sampler_ppo"].update(
        num_envs=96, eval_episode_steps=12288,
        constraint_cost_type="collision", safety_bound=0.05,
        reset_state_on_rollout=False,
        lagrangian_update_mode=args.lagrangian_update_mode,
        allow_partial_first_episode=args.allow_partial_first_episode,
        lagrangian_coef_rate=1.0 / 6.0,
        lagrangian_ema_decay=0.9, initial_lambda_lagr=1.0, lagrangian_max=100.0,
        eval_video=False, eval_render=True,
        num_eval_envs=1024, eval_episodes_per_dynamics=10, num_evals=10,
        gmm_cost_score_scale=args.cost_score_scale,
    )
    if args.cost_dual_ema_decay is not None:
        rl["jax_sampler_ppo"]["gmm_cost_dual_ema_decay"] = args.cost_dual_ema_decay
    if args.no_domain_randomization:
        rl["jax_sampler_ppo"]["domain_randomization"] = False
        rl["jax_sampler_ppo"]["eval_domain_randomization"] = False
    if args.cost_dual_update is not None:
        rl["jax_sampler_ppo"]["gmm_cost_dual_update"] = args.cost_dual_update
    if args.cost_dual_lr is not None:
        if not 0.0 < args.cost_dual_lr < float("inf"):
            raise ValueError("Cost dual learning rate must be finite and positive")
        rl["jax_sampler_ppo"]["gmm_cost_dual_lr"] = args.cost_dual_lr

    def worker(item):
        gpu, track = item
        results = []
        for formulation in args.formulations:
            trial_label = "nominal" if args.no_domain_randomization else formulation
            trial = OUT / f"{track}_{trial_label}"
            trial.mkdir()
            config = copy.deepcopy(rl)
            jc = config["jax_sampler_ppo"]
            jc["sampler"] = "uniform" if formulation == "udr" else "reward_cost_gmmvi"
            jc["gmm_formulation"] = formulation if formulation != "udr" else "reward"
            jc["gmm_reward_fraction"] = {"cost": 0.0, "reward_cost": 0.5}.get(formulation, 1.0)
            path = trial / "rl_config.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            cmd = [sys.executable, "-u", "train/jax_sampler_ppo.py",
                   "--rl-config", str(path), "--gym-config", str(gym_path),
                   "--track", track, "--dr-profile", "narrow"]
            print(f"Starting gpu={gpu} track={track} formulation={formulation}", flush=True)
            with (trial / "train.log").open("w") as log:
                run = subprocess.run(cmd, cwd=ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdout=log, stderr=subprocess.STDOUT)
            if run.returncode:
                raise RuntimeError(f"Failed run {trial}: exit {run.returncode}")
            ids = re.findall(r"View run at .*?/runs/([a-z0-9]+)", (trial / "train.log").read_text())
            if not ids:
                raise RuntimeError(f"Missing W&B run: {trial}")
            run_id = ids[-1]
            files = list((ROOT / "wandb").glob(f"run-*-{run_id}/files/wandb-summary.json"))
            if len(files) != 1:
                raise RuntimeError(f"Missing summary: {trial}")
            summary = json.loads(files[0].read_text())
            result = dict(track=track, formulation="nominal" if args.no_domain_randomization else formulation, run_id=run_id,
                reward=summary["evaluation/reward/episode_return_mean"],
                cvar10=summary["evaluation/reward/cvar10"],
                collision_rate=summary["evaluation/collision/rate"],
                lambda_final=summary["training/constraint/lambda_lagr"],
                budget=0.05, horizon=12288,
                steps=summary["training/progress/env_steps"])
            result["cost_score_scale"] = args.cost_score_scale
            result["seed"] = args.seed
            result["cost_dual_ema_decay"] = jc.get("gmm_cost_dual_ema_decay", jc.get("gmm_dual_ema_decay", 0.9))
            result["cost_dual_lr"] = jc["gmm_cost_dual_lr"]
            result["cost_dual_update"] = jc.get("gmm_cost_dual_update", "linear")
            result["cost_dual_direction"] = "adversarial"
            result["rollout_length"] = args.unroll_length
            result["ppo_batch_size"] = config["batch_size"]
            result["domain_randomization"] = jc.get("domain_randomization", True)
            result["eval_domain_randomization"] = jc.get("eval_domain_randomization", False)
            result["reset_state_on_rollout"] = False
            result["lagrangian_update_mode"] = args.lagrangian_update_mode
            result["allow_partial_first_episode"] = args.allow_partial_first_episode
            (trial / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            results.append(result)
        return results

    with ThreadPoolExecutor(max_workers=3) as pool:
        groups = list(pool.map(worker, zip(args.gpus, args.tracks)))
    results = [r for group in groups for r in group]
    (OUT / "final.json").write_text(json.dumps(results, indent=2))
    report_results(results, OUT, "Matched-horizon collision experiment completed")
    print(f"Completed all {len(results)} matched-horizon runs.", flush=True)


if __name__ == "__main__":
    main()
