"""Re-evaluate a saved final policy if its W&B service connection was lost.

Training is not restarted. The original final evaluation key and environment
are replayed, and the original partial W&B run and stdout logs are retained.
"""

import argparse
import json
from pathlib import Path

from train.jax_sampler_ppo import (
    _build_env, _build_trainer_config, _evaluation_schedule, _make_eval_fn,
    _metric_dict, _num_updates,
)
import jax
from flax.serialization import from_bytes
import yaml

from gymkhana.jax_sampler_ppo import SamplerPPOTrainer
from report_experiments import ENTITY, PROJECT


def final_evaluation_key(seed, updates, num_evals, *, start_update=0):
    key = jax.random.PRNGKey(seed)
    initial, scheduled = _evaluation_schedule(updates, num_evals, start_update=start_update)
    count = int(initial) + len(scheduled)
    if not count:
        raise ValueError("No original evaluation to replay")
    for _ in range(count):
        key, evaluation_key, _, _ = jax.random.split(key, 4)
    return evaluation_key


def restore_trial(directory, gym_path):
    rl = yaml.safe_load((directory / "rl_config.yaml").read_text())
    gym = yaml.safe_load(gym_path.read_text())
    jc = rl["jax_sampler_ppo"]
    if jc["domain_randomization"] or jc["eval_domain_randomization"]:
        raise ValueError("Recovery is scoped to nominal Silverstone trials")
    n = int(jc["num_envs"])
    config = _build_trainer_config(rl, n)
    kwargs = dict(domain_randomization_ranges=jc.get("domain_randomization_ranges"),
                  asymmetric_critic=bool(jc.get("asymmetric_critic", False)),
                  action_repeat_steps=config.policy_repeat_steps,
                  constraint_cost_type=config.constraint_cost_type)
    env = _build_env(gym, n, int(rl["seed"]), "Silverstone", **kwargs)
    trainer = SamplerPPOTrainer(env, config)
    template = trainer.init_state(jax.random.PRNGKey(int(rl["seed"])))
    state = from_bytes(template, (directory / "checkpoint.msgpack").read_bytes())
    updates = _num_updates(int(rl["total_timesteps"]), n, config.unroll_length)
    if int(state.update_steps) != updates or int(state.env_steps) != updates*n*config.unroll_length:
        raise ValueError("Checkpoint is not the completed trial state")
    return rl, gym, config, kwargs, trainer, state, updates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--gym-config", type=Path, required=True)
    parser.add_argument("--original-run-id", required=True)
    args = parser.parse_args()
    rl, gym, config, kwargs, trainer, state, updates = restore_trial(args.trial_dir, args.gym_config)
    local_summary = args.trial_dir / "checkpoint.summary.json"
    if local_summary.exists():
        metrics = json.loads(local_summary.read_text())
        if (metrics.get("training/progress/env_steps") != int(state.env_steps)
                or metrics.get("evaluation/meta/update") != updates):
            raise ValueError("Local summary does not match the final checkpoint/evaluation")
        print("Using exact locally saved final metrics", flush=True)
    else:
        eval_env = _build_env(gym, config.num_eval_envs, int(rl["seed"]), "Silverstone",
                              max_episode_steps=config.eval_episode_steps, warn_track_pool=False, **kwargs)
        evaluate = _make_eval_fn(eval_env, trainer, episode_steps=config.eval_episode_steps,
                                episodes_per_dynamics=config.eval_episodes_per_dynamics,
                                randomize_dynamics=False, jit=True)
        key = final_evaluation_key(int(rl["eval_seed"]), updates, config.num_evals,
                                   start_update=int(rl.get("resume_from_update", 0)))
        metrics, _, _, _ = evaluate(state.params, key)
        metrics = _metric_dict(metrics)
    metrics.update({"training/progress/env_steps": int(state.env_steps),
                    "training/progress/update_steps": int(state.update_steps),
                    "training/constraint/lambda_lagr": float(state.lambda_lagr),
                    "evaluation/meta/update": updates,
                    "evaluation/meta/checkpoint_recovery": 1,
                    "evaluation/meta/checkpoint_reevaluated": int(not local_summary.exists())})
    print(json.dumps(metrics), flush=True)
    # Persist exact evaluation even if publication fails; never substitute rounded stdout.
    (args.trial_dir / "recovered_metrics.json").write_text(json.dumps(metrics, indent=2))
    import wandb
    with wandb.init(entity=ENTITY, project=PROJECT,
                    name=args.trial_dir.name + "_checkpoint_evaluation_recovery",
                    job_type="evaluation-recovery", tags=["logging-recovery", "nominal-autotune"],
                    config={"rl_config": rl, "gym_config": gym,
                            "original_run_id": args.original_run_id,
                            "note": "Original W&B connection lost; original logs retained; exact final checkpoint/key re-evaluated."},
                    settings={"disable_git": True, "disable_code": True}) as run:
        run.log(metrics)
        rid = run.id
    (args.trial_dir / "recovered_summary.json").write_text(json.dumps(
        {"metrics": metrics, "run_id": rid, "original_run_id": args.original_run_id}, indent=2))


if __name__ == "__main__":
    main()
