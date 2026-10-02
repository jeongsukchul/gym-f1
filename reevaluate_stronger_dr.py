#!/usr/bin/env python3
"""Widen evaluation DR for immutable completed checkpoints; never train."""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/silverstone_dr_tuned_20261002"
FORMULATIONS = ("udr", "reward", "cost")


def stronger_ranges(ranges, strength):
    if not math.isfinite(strength) or strength <= 1.:
        raise ValueError("DR strength must be finite and greater than one")
    result = {}
    for name, (low, high) in ranges.items():
        if not 0. < low <= 1. <= high:
            raise ValueError(f"Expected positive nominal-relative bounds for {name}")
        widened = [1. + strength * (low - 1.), 1. + strength * (high - 1.)]
        if widened[0] <= 0.:
            raise ValueError(f"Strength makes {name} nonpositive")
        result[name] = widened
    return result


def evaluate_checkpoint(formulation, output, strength):
    import jax
    import numpy as np
    from flax.serialization import from_bytes
    from gymkhana.jax_sampler_ppo import SamplerPPOTrainer
    from train.jax_sampler_ppo import _build_env, _build_trainer_config, _make_eval_fn, _metric_dict, _num_updates
    from recover_nominal_trial import final_evaluation_key
    from report_experiments import ENTITY, PROJECT

    directory = SOURCE / formulation
    rl = yaml.safe_load((directory / "rl_config.yaml").read_text())
    gym = yaml.safe_load((SOURCE / "gym_config.yaml").read_text())
    original = json.loads((directory / "result.json").read_text())
    jc = rl["jax_sampler_ppo"]
    bounds = stronger_ranges(jc["domain_randomization_ranges"], strength)
    num_envs = int(jc["num_envs"])
    config = _build_trainer_config(rl, num_envs)
    kwargs = dict(asymmetric_critic=bool(jc.get("asymmetric_critic", False)),
                  action_repeat_steps=config.policy_repeat_steps,
                  constraint_cost_type=config.constraint_cost_type)
    train_env = _build_env(gym, num_envs, int(rl["seed"]), "Silverstone",
                           domain_randomization_ranges=jc["domain_randomization_ranges"], **kwargs)
    trainer = SamplerPPOTrainer(train_env, config)
    template = trainer.init_state(jax.random.PRNGKey(int(rl["seed"])))
    checkpoint_path = directory / "checkpoint.msgpack"
    state = from_bytes(template, checkpoint_path.read_bytes())
    # Evaluation consumes network parameters, not the sampler's serialized
    # Python scalar bookkeeping (which can deserialize with a wider dtype).
    if jax.tree_util.tree_structure(state.params) != jax.tree_util.tree_structure(template.params):
        raise ValueError("Checkpoint network structure mismatch")
    for actual, expected in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(template.params)):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError("Checkpoint array shape or dtype mismatch")
    updates = _num_updates(int(rl["total_timesteps"]), num_envs, config.unroll_length)
    if int(state.update_steps) != updates or int(state.env_steps) != updates * num_envs * config.unroll_length:
        raise ValueError("Expected completed checkpoint")
    eval_env = _build_env(gym, config.num_eval_envs, int(rl["seed"]), "Silverstone",
                          domain_randomization_ranges=bounds, max_episode_steps=config.eval_episode_steps,
                          warn_track_pool=False, **kwargs)
    evaluate = _make_eval_fn(eval_env, trainer, episode_steps=config.eval_episode_steps,
                             episodes_per_dynamics=config.eval_episodes_per_dynamics,
                             randomize_dynamics=True, jit=True)
    key = final_evaluation_key(int(rl["eval_seed"]), updates, config.num_evals,
                               start_update=int(rl.get("resume_from_update", 0)))
    print(f"Evaluating {formulation}: unchanged checkpoint, strength={strength}, bounds={bounds}", flush=True)
    metrics, _, _, _ = evaluate(state.params, key)
    metrics = _metric_dict(metrics)
    assert metrics["evaluation/meta/total_episodes"] == 10240
    metrics.update({"evaluation/meta/dr_strength": strength,
                    "evaluation/meta/checkpoint_env_steps": int(state.env_steps)})
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    effective_rl = copy.deepcopy(rl)
    effective_rl["jax_sampler_ppo"]["domain_randomization_ranges"] = bounds
    effective_rl["jax_sampler_ppo"]["domain_randomization_profile"] = "stronger_evaluation_only"
    (output / "evaluation_config.yaml").write_text(yaml.safe_dump(effective_rl, sort_keys=False))
    import wandb
    with wandb.init(entity=ENTITY, project=PROJECT, job_type="checkpoint-evaluation",
                    name=f"Silverstone_{formulation}_strongerDR_{strength:.3f}_seed1",
                    tags=["evaluation-only", "stronger-dr"],
                    config={"evaluation_rl_config": effective_rl, "gym_config": gym,
                            "source_run_id": original["run_id"], "checkpoint": str(checkpoint_path),
                            "dr_strength": strength, "training_performed": False,
                            "paired_with_original_final_evaluation_key": True},
                    settings={"disable_git": True, "disable_code": True}) as run:
        run.log(metrics)
        run_id = run.id
    result = dict(original, run_id=run_id, source_run_id=original["run_id"],
                  reward=metrics["evaluation/reward/episode_return_mean"],
                  cvar10=metrics["evaluation/reward/cvar10"],
                  collision_rate=metrics["evaluation/collision/rate"],
                  tuning_label=f"{formulation}_stronger_evaluation", dr_strength=strength,
                  training_performed=False, evaluation_ranges=bounds,
                  dr_profile="stronger_evaluation_only")
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=4./3.)
    parser.add_argument("--evaluate", choices=FORMULATIONS)
    parser.add_argument("--collect-only", action="store_true",
                        help="Publish already completed evaluations without running GPU work")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.evaluate:
        evaluate_checkpoint(args.evaluate, output, args.strength)
        return
    from report_experiments import report_results
    base = yaml.safe_load((SOURCE / "udr/rl_config.yaml").read_text())
    bounds = stronger_ranges(base["jax_sampler_ppo"]["domain_randomization_ranges"], args.strength)
    output.mkdir(parents=True, exist_ok=args.collect_only)
    manifest = dict(status="running", formulations=list(FORMULATIONS), gpus=[3, 4],
                    training_performed=False, dr_strength=args.strength, evaluation_ranges=bounds,
                    horizon=12288, evaluation_episodes=10240, results=[])
    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    save()
    def worker(gpu, formulations):
        for formulation in formulations:
            directory = output / formulation
            directory.mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            env.pop("WANDB_SERVICE", None)
            with (directory / "evaluation.log").open("w") as log:
                subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                "--evaluate", formulation, "--strength", str(args.strength),
                                "--output-dir", str(directory)], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        return [json.loads((output / f / "result.json").read_text()) for f in formulations]
    try:
        if args.collect_only:
            for formulation in FORMULATIONS:
                r = json.loads((output / formulation / "result.json").read_text())
                assert r["dr_strength"] == args.strength and not r["training_performed"]
                manifest["results"].append(r)
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker, 3, ["udr", "cost"]), pool.submit(worker, 4, ["reward"])]
                for future in as_completed(futures):
                    manifest["results"].extend(future.result())
                    save()
        baseline = []
        for formulation in FORMULATIONS:
            r = json.loads((SOURCE / formulation / "result.json").read_text())
            baseline.append(dict(r, tuning_label=f"{formulation}_narrow_baseline", dr_strength=1.))
        results = baseline + manifest["results"]
        (output / "final.json").write_text(json.dumps(results, indent=2))
        report_results(results, output / "final_report", "Silverstone stronger DR checkpoint evaluation (no training)")
        manifest["status"] = "completed"
        save()
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
