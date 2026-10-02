"""Train F1TENTH with the Gym-Khana JAX sampler PPO backend."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import multiprocessing
import os
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Avoid OOM on limited-memory devices

import jax
import jax.numpy as jnp
import yaml
from train.config.rollout import get_rollout_length

from gymkhana.jax_env import JaxRaceEnv
from gymkhana.jax_sampler_ppo import (
    F1TenthAdvWrapper,
    SamplerPPOConfig,
    SamplerPPOTrainer,
    evaluate_policy,
    make_domain_spec,
    record_policy_trajectory,
)

_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _metric_scalar(value):
    try:
        return float(value)
    except TypeError:
        return value


def _metric_dict(metrics: dict) -> dict:
    return {key: _metric_scalar(value) for key, value in metrics.items()}


def _wandb_run_name(gym_config: dict, rl_config: dict) -> str:
    jax_config = rl_config.get("jax_sampler_ppo", {})
    sampler = str(jax_config.get("sampler", "uniform"))
    if not jax_config.get("domain_randomization", True):
        sampler = "nominal"
    gmm_beta = jax_config.get("gmm_target_beta", "na")
    use_ppo_lag = bool(jax_config.get("use_ppo_lag", False))
    algorithm = "ppo_lag" if use_ppo_lag else "ppo"
    map_name = gym_config.get("map", "unknown_map")
    seed = rl_config.get("seed", "na")
    eval_dr = "evalDR" if jax_config.get("eval_domain_randomization", False) else "evalNominal"
    eval_horizon = int(jax_config.get("eval_episode_steps", 10000))
    lag_label = (
        f"_{jax_config.get('constraint_cost_type', 'edge')}cost"
        f"_sb{jax_config.get('safety_bound', 0):g}"
        f"_li{jax_config.get('initial_lambda_lagr', 0):g}"
        f"_lr{jax_config.get('lagrangian_coef_rate', 0):g}"
        f"_lu{jax_config.get('lagrangian_update_mode', 'per_step')}"
        if use_ppo_lag
        else ""
    )
    dr_profile = jax_config.get("domain_randomization_profile")
    dr_label = f"_dr{dr_profile}" if dr_profile else ""
    if sampler in {"reward_cost_gmmvi", "dual_gmmvi", "rc_gmmvi"}:
        formulation = jax_config.get("gmm_formulation", "reward_cost")
        sampler_label = (
            f"{sampler}_{formulation}_rkl{jax_config.get('gmm_reward_kl_radius', 'na')}"
            f"_cb{jax_config.get('safety_bound', 'na')}"
            f"_cs{jax_config.get('gmm_cost_score_scale', 1):g}"
            f"_ce{jax_config.get('gmm_cost_dual_ema_decay', jax_config.get('gmm_dual_ema_decay', 0.9))}"
            f"_clr{jax_config.get('gmm_cost_dual_lr', 0.01):g}"
            f"_cdu{jax_config.get('gmm_cost_dual_update', 'linear')}"
            "_cdiradversarial"
        )
    elif sampler in {"gmmvi", "gmm"}:
        sampler_label = f"{sampler}_beta_{gmm_beta}"
    else:
        sampler_label = sampler
    return f"{map_name}_{algorithm}{lag_label}{dr_label}_{sampler_label}_s_{seed}_{eval_dr}_eh{eval_horizon}"


def _hidden_layer_sizes_from_config(rl_config: dict, key: str, legacy_key: str) -> tuple[int, ...]:
    value = rl_config.get(key, rl_config.get(legacy_key))
    if value is None:
        raise KeyError(f"Missing required RL config key '{key}'")
    if isinstance(value, int):
        return (int(value), int(value))
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"RL config key '{key}' must be a non-empty list of positive ints, got {value!r}")
    layers = tuple(int(width) for width in value)
    if any(width <= 0 for width in layers):
        raise ValueError(f"RL config key '{key}' must contain only positive ints, got {value!r}")
    return layers


def _init_wandb(
    gym_config: dict,
    rl_config: dict,
    *,
    disabled: bool = False,
    save_code: bool = False,
) -> bool:
    if disabled:
        return False
    try:
        import wandb

        started_here = False
        if wandb.run is None:
            wandb.init(
                project=gym_config.get("project_name", "f1tenth-jax-sampler-ppo"),
                name=_wandb_run_name(gym_config, rl_config),
                config={"rl_config": rl_config, "gym_config": gym_config},
                save_code=save_code,
                settings={"disable_git": not save_code, "disable_code": not save_code},
            )
            started_here = True
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        return started_here
    except Exception as exc:
        print(f"WandB logging disabled: {exc}")
        return False


def _wandb_log(metrics: dict, env_steps: int) -> bool:
    try:
        import wandb

        if wandb.run is None:
            return False
        scalar_metrics = _metric_dict(metrics)
        canonical = {
            key: value
            for key, value in scalar_metrics.items()
            if key.startswith(("training/", "evaluation/"))
        }
        # Prefer a stage-first hierarchy so W&B panels do not mix training
        # and evaluation series. Legacy-only callers still use the fallback.
        payload = canonical or scalar_metrics
        payload["global_step"] = int(env_steps)
        wandb.log(payload, step=int(env_steps))
        return True
    except Exception:
        return False


def _wandb_update_config(config_updates: dict) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.config.update(config_updates, allow_val_change=True)
    except Exception:
        pass


def _wandb_run_id() -> str | None:
    try:
        import wandb

        if wandb.run is None:
            return None
        run_id = getattr(wandb.run, "id", None)
        return str(run_id) if run_id else None
    except Exception:
        return None


def _load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def _resolve_direction(track_direction: str, seed: int) -> bool:
    if track_direction == "normal":
        return False
    if track_direction == "reverse":
        return True
    if track_direction == "random":
        return random.Random(seed).random() < 0.5
    raise ValueError("track_direction must be one of: normal, reverse, random")


def _num_updates(total_timesteps: int, num_envs: int, unroll_length: int) -> int:
    return max(1, math.ceil(total_timesteps / (num_envs * unroll_length)))


def _lr_transition_steps(rl_config: dict, jax_config: dict, num_envs: int) -> int:
    rollout_length = get_rollout_length(rl_config)
    rollout_size = int(num_envs) * rollout_length
    num_rollouts = _num_updates(int(rl_config["total_timesteps"]), num_envs, rollout_length)
    num_minibatches = max(1, rollout_size // int(rl_config["batch_size"]))
    return num_rollouts * int(jax_config["num_epochs"]) * num_minibatches


def _run_output_label() -> str:
    run_id = _wandb_run_id()
    if run_id is not None:
        return f"run_{run_id}"
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def _scoped_output_dir(base_dir: str | os.PathLike[str], *, sampler: str, seed: int, run_label: str) -> str:
    base_path = Path(base_dir)
    return str(base_path.parent / sampler / f"seed_{int(seed)}" / run_label / base_path.name)


def _apply_scoped_output_dirs(
    rl_config: dict,
    run_label: str,
    *,
    scoped_keys: tuple[str, ...] = ("sampler_plot_dir", "eval_render_dir", "eval_video_dir"),
) -> None:
    jax_config = rl_config["jax_sampler_ppo"]
    sampler = str(jax_config.get("sampler", "uniform"))
    seed = int(rl_config["seed"])
    for key in scoped_keys:
        base_dir = jax_config.get(key)
        if not base_dir:
            continue
        jax_config[key] = _scoped_output_dir(base_dir, sampler=sampler, seed=seed, run_label=run_label)


def _build_trainer_config(rl_config: dict, num_envs: int) -> SamplerPPOConfig:
    jax_config = rl_config["jax_sampler_ppo"]
    log_std_schedule = rl_config.get("log_std_schedule")
    init_log_std = float(log_std_schedule["init"]) if log_std_schedule else -1.0
    end_log_std = float(log_std_schedule["end"]) if log_std_schedule else None
    return SamplerPPOConfig(
        sampler=str(jax_config.get("sampler", "uniform")),
        domain_randomization=bool(jax_config.get("domain_randomization", True)),
        num_eval_envs=int(jax_config.get("num_eval_envs", 0)),
        num_evals=int(jax_config.get("num_evals", 1)),
        sampler_plot_samples=int(jax_config.get("sampler_plot_samples", 4096)),
        sampler_plot_grid=int(jax_config.get("sampler_plot_grid", 50)),
        sampler_plot_context_samples=int(jax_config.get("sampler_plot_context_samples", 64)),
        sampler_plot_max_dims=jax_config.get("sampler_plot_max_dims", 5),
        sampler_plot_dir=str(jax_config.get("sampler_plot_dir", "outputs/jax_sampler_ppo/sampler_plots")),
        eval_episode_steps=int(jax_config.get("eval_episode_steps", 10000)),
        eval_episodes_per_dynamics=int(jax_config.get("eval_episodes_per_dynamics", 1)),
        eval_render=bool(jax_config.get("eval_render", False)),
        eval_render_dir=str(jax_config.get("eval_render_dir", "outputs/jax_sampler_ppo/eval_renders")),
        eval_video=bool(jax_config.get("eval_video", False)),
        eval_video_dir=str(jax_config.get("eval_video_dir", "outputs/jax_sampler_ppo/eval_videos")),
        eval_video_fps=int(jax_config.get("eval_video_fps", 30)),
        eval_video_max_frames=int(jax_config.get("eval_video_max_frames", 600)),
        eval_video_final=bool(jax_config.get("eval_video_final", False)),
        eval_domain_randomization=bool(jax_config.get("eval_domain_randomization", False)),
        eval_video_percentiles=bool(jax_config.get("eval_video_percentiles", False)),
        eval_video_percentile_step=int(jax_config.get("eval_video_percentile_step", 5)),
        learning_rate=float(rl_config["start_learning_rate"]),
        end_learning_rate=float(rl_config["end_learning_rate"]),
        learning_rate_transition_steps=_lr_transition_steps(rl_config, jax_config, num_envs),
        total_timesteps=int(rl_config["total_timesteps"]),
        unroll_length=get_rollout_length(rl_config),
        reset_state_on_rollout=bool(jax_config.get("reset_state_on_rollout", True)),
        policy_repeat_steps=int(jax_config.get("policy_repeat_steps", 1)),
        batch_size=int(rl_config["batch_size"]),
        num_epochs=int(jax_config["num_epochs"]),
        discounting=float(jax_config["discounting"]),
        cost_discounting=(float(jax_config["cost_discounting"])
                          if jax_config.get("cost_discounting") is not None else None),
        cost_terminal_at_time_limit=bool(jax_config.get("cost_terminal_at_time_limit", False)),
        gae_lambda=float(jax_config["gae_lambda"]),
        clipping_epsilon=float(jax_config["clipping_epsilon"]),
        entropy_cost=float(jax_config["entropy_cost"]),
        value_cost=float(jax_config["value_cost"]),
        normalize_advantage=bool(jax_config["normalize_advantage"]),
        normalize_cost_advantage=(bool(jax_config["normalize_cost_advantage"])
                                  if jax_config.get("normalize_cost_advantage") is not None else None),
        cost_advantage_std_floor=float(jax_config.get("cost_advantage_std_floor", 0.0)),
        sampler_update_freq=int(jax_config["sampler_update_freq"]),
        gmm_components=int(jax_config["gmm_components"]),
        gmm_target_beta=float(jax_config["gmm_target_beta"]),
        gmm_init_std=float(jax_config["gmm_init_std"]),
        gmm_reward_fraction=float(jax_config.get("gmm_reward_fraction", 0.5)),
        gmm_reward_kl_radius=float(jax_config.get("gmm_reward_kl_radius", 0.1)),
        gmm_reward_dual_lr=float(jax_config.get("gmm_reward_dual_lr", 1e-3)),
        gmm_cost_initial_beta=float(jax_config.get("gmm_cost_initial_beta", 1.0)),
        gmm_cost_dual_lr=float(jax_config.get("gmm_cost_dual_lr", 1e-2)),
        gmm_cost_dual_update=str(jax_config.get("gmm_cost_dual_update", "linear")),
        gmm_cost_score_scale=float(jax_config.get("gmm_cost_score_scale", 1.0)),
        gmm_dual_ema_decay=float(jax_config.get("gmm_dual_ema_decay", 0.9)),
        gmm_cost_dual_ema_decay=(
            None if jax_config.get("gmm_cost_dual_ema_decay") is None
            else float(jax_config["gmm_cost_dual_ema_decay"])
        ),
        gmm_dual_lambda_min=float(jax_config.get("gmm_dual_lambda_min", 1e-3)),
        gmm_dual_lambda_max=float(jax_config.get("gmm_dual_lambda_max", 1e3)),
        gmm_reward_violation_clip=(
            None
            if jax_config.get("gmm_reward_violation_clip") is None
            else float(jax_config["gmm_reward_violation_clip"])
        ),
        gmm_cost_violation_clip=(
            None
            if jax_config.get("gmm_cost_violation_clip") is None
            else float(jax_config["gmm_cost_violation_clip"])
        ),
        policy_hidden_layer_sizes=_hidden_layer_sizes_from_config(rl_config, "actor_layer", "actor_layer_size"),
        value_hidden_layer_sizes=_hidden_layer_sizes_from_config(rl_config, "critic_layer", "critic_layer_size"),
        cost_value_hidden_layer_sizes=_hidden_layer_sizes_from_config(
            jax_config, "cost_critic_layer", "cost_critic_layer"
        ),
        policy_use_layer_norm=bool(rl_config.get("actor_layer_norm", False)),
        value_use_layer_norm=bool(rl_config.get("critic_layer_norm", False)),
        init_log_std=init_log_std,
        end_log_std=end_log_std,
        use_ppo_lag=bool(jax_config.get("use_ppo_lag", False)),
        safety_bound=float(jax_config.get("safety_bound", 0.0)),
        lagrangian_coef_rate=float(jax_config.get("lagrangian_coef_rate", 0.01)),
        initial_lambda_lagr=float(jax_config.get("initial_lambda_lagr", 0.0)),
        constraint_cost_type=str(jax_config.get("constraint_cost_type", "edge")),
        lagrangian_update_mode=str(jax_config.get("lagrangian_update_mode", "per_step")),
        allow_partial_first_episode=bool(jax_config.get("allow_partial_first_episode", False)),
        lagrangian_ema_decay=float(jax_config.get("lagrangian_ema_decay", 0.0)),
        lagrangian_max=float(jax_config.get("lagrangian_max", 100.0)),
        lagrangian_min_completed_episodes=int(jax_config.get("lagrangian_min_completed_episodes", 1)),
        lagrangian_warmup_steps=int(jax_config.get("lagrangian_warmup_steps", 0)),
    )


def _build_env(
    gym_config: dict,
    num_envs: int,
    seed: int,
    track_override: str | None,
    *,
    domain_randomization_ranges: dict | None = None,
    asymmetric_critic: bool = False,
    auto_reset: bool = True,
    max_episode_steps: int | None = None,
    action_repeat_steps: int = 1,
    constraint_cost_type: str = "edge",
    warn_track_pool: bool = True,
) -> F1TenthAdvWrapper:
    if warn_track_pool and gym_config.get("track_pool"):
        print("Note: JAX sampler PPO currently uses gym_config['map']; multi-map track_pool is not ported yet.")

    track_name = track_override or gym_config["map"]
    reversed_track = _resolve_direction(gym_config["track_direction"], seed)
    sensor_noise_psi_std = gym_config.get("sensor_noise_psi_std")
    if sensor_noise_psi_std is None:
        sensor_noise_psi_std = math.radians(float(gym_config.get("sensor_noise_psi_std_deg", 0.0)))
    base_env = JaxRaceEnv.from_track_name(
        track_name,
        reversed=reversed_track,
        batch_size=num_envs,
        timestep=float(gym_config["timestep"]),
        max_episode_steps=int(max_episode_steps or gym_config["max_episode_steps"]),
        progress_gain=float(gym_config["progress_gain"]),
        out_of_bounds_penalty=float(gym_config["out_of_bounds_penalty"]),
        negative_vel_penalty=float(gym_config["negative_vel_penalty"]),
        lookahead_n_points=int(gym_config["lookahead_n_points"]),
        lookahead_ds=float(gym_config["lookahead_ds"]),
        sparse_width_obs=bool(gym_config["sparse_width_obs"]),
        normalize_obs=bool(gym_config["normalize_obs"]),
        mask_track_obs=bool(gym_config.get("mask_track_obs", False)),
        steering_delay_steps=int(gym_config["steering_delay_steps"]),
        slip_reward_enabled=bool(gym_config.get("slip_reward_enabled", False)),
        slip_reward_path_weight=float(gym_config.get("slip_reward_path_weight", 0.3)),
        slip_reward_slip_weight=float(gym_config.get("slip_reward_slip_weight", 0.7)),
        slip_reward_target_deg=float(gym_config.get("slip_reward_target_deg", 45.0)),
        slip_reward_width_deg=float(gym_config.get("slip_reward_width_deg", 20.0)),
        slip_reward_shape=float(gym_config.get("slip_reward_shape", 2.5)),
        edge_penalty_weight=float(gym_config.get("edge_penalty_weight", 1.0)),
        edge_penalty_start_ratio=float(gym_config.get("edge_penalty_start_ratio", 0.8)),
        termination_boundary_margin_ratio=float(gym_config.get("termination_boundary_margin_ratio", 1.0)),
        sensor_noise_enabled=bool(gym_config.get("sensor_noise_enabled", False)),
        sensor_noise_s_std=float(gym_config.get("sensor_noise_s_std", 0.0)),
        sensor_noise_n_std=float(gym_config.get("sensor_noise_n_std", 0.0)),
        sensor_noise_psi_std=float(sensor_noise_psi_std),
    )
    if domain_randomization_ranges is None:
        domain_spec = make_domain_spec(
            sigmas=gym_config.get("domain_randomization", {}),
            clip_k=float(gym_config.get("dr_clip_k", 3.0)),
        )
    else:
        domain_spec = make_domain_spec(ranges=domain_randomization_ranges)
    return F1TenthAdvWrapper(
        base_env,
        domain_spec,
        auto_reset=auto_reset,
        obs_history_len=int(gym_config.get("obs_history_len", 1)),
        obs_delay_min_steps=int(gym_config.get("obs_delay_min_steps", 0)),
        obs_delay_max_steps=int(gym_config.get("obs_delay_max_steps", gym_config.get("obs_delay_min_steps", 0))),
        asymmetric_critic=asymmetric_critic,
        action_repeat_steps=action_repeat_steps,
        constraint_cost_type=constraint_cost_type,
    )


def _resolve_eval_track(gym_config: dict, track_override: str | None) -> str | None:
    if track_override is not None:
        return track_override
    evaluation_track_pool = gym_config.get("evaluation_track_pool")
    if isinstance(evaluation_track_pool, str):
        return evaluation_track_pool
    if isinstance(evaluation_track_pool, list) and evaluation_track_pool:
        if len(evaluation_track_pool) > 1:
            print("Note: JAX evaluation currently uses the first evaluation_track_pool entry.")
        return evaluation_track_pool[0]
    return None


def _make_eval_fn(
    eval_env: F1TenthAdvWrapper,
    trainer: SamplerPPOTrainer,
    *,
    episode_steps: int,
    episodes_per_dynamics: int,
    randomize_dynamics: bool,
    jit: bool,
):
    episodes_per_dynamics = int(episodes_per_dynamics)
    if episodes_per_dynamics < 1:
        raise ValueError(f"episodes_per_dynamics must be >= 1, got {episodes_per_dynamics}")

    def run_eval(params, key):
        key_reset, key_params = jax.random.split(key)
        dynamics_params = (
            eval_env.sample_uniform_params(key_params) if randomize_dynamics else eval_env.nominal_dynamics_params
        )
        policy_episode_steps = max(1, math.ceil(int(episode_steps) / trainer.config.policy_repeat_steps))
        result = evaluate_policy(
            eval_env,
            trainer.make_policy(params, deterministic=True),
            key_reset,
            dynamics_params,
            policy_episode_steps,
            episodes_per_dynamics,
            safety_bound=trainer.config.safety_bound if trainer.config.use_ppo_lag else None,
        )
        if episodes_per_dynamics == 1:
            episode_dynamics_params = dynamics_params
        else:
            episode_dynamics_params = jnp.tile(dynamics_params, (episodes_per_dynamics, 1))
        return result.metrics, result.rewards, result.lengths, episode_dynamics_params

    return jax.jit(run_eval) if jit else run_eval


def _should_save_eval_video(config: SamplerPPOConfig, *, is_final: bool) -> bool:
    return bool(config.eval_video and (not config.eval_video_final or is_final))


def _evaluation_schedule(num_updates: int, num_evals: int, *, start_update: int = 0) -> tuple[bool, set[int]]:
    """Returns whether to eval before training and which completed updates to eval."""
    num_updates = int(num_updates)
    num_evals = int(num_evals)
    if num_evals <= 0:
        return False, set()
    if num_updates <= 0:
        return True, set()
    if num_evals == 1:
        return False, {num_updates}

    eval_updates = {
        max(1, min(num_updates, math.ceil(idx * num_updates / (num_evals - 1))))
        for idx in range(1, num_evals)
    }
    return True, {update for update in eval_updates if update > start_update}


def _restore_training_checkpoint(template, path: Path, *, num_envs: int,
                                 rollout_length: int, target_updates: int):
    """Restore the complete state, rejecting incompatible geometry or counters."""
    from flax.serialization import from_bytes

    state = from_bytes(template, Path(path).read_bytes())
    if jax.tree_util.tree_structure(state) != jax.tree_util.tree_structure(template):
        raise ValueError("Checkpoint training-state structure does not match this trainer")
    for actual, expected in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(template)):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError("Checkpoint array shape/dtype does not match this trainer")
    start_update = int(state.update_steps)
    expected_steps = start_update * num_envs * rollout_length
    if start_update < 0 or int(state.env_steps) != expected_steps:
        raise ValueError("Checkpoint rollout geometry/counters are inconsistent")
    if start_update >= target_updates:
        raise ValueError("Resume target must exceed the checkpoint's completed updates")
    return state


def _save_sampler_plot(
    trainer: SamplerPPOTrainer,
    state,
    key: jax.Array,
    config: SamplerPPOConfig,
    *,
    eval_index: int,
    env_steps: int,
) -> str | None:
    if not config.domain_randomization or config.sampler_plot_samples <= 0 or trainer.env.domain_spec.size < 2:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import warnings

    import matplotlib.pyplot as plt

    from gymkhana.jax_sampler_ppo.gmmvi import utils as gmmvi_utils

    samples, _, _ = trainer.sampler.sample(state.sampler_state, key, int(config.sampler_plot_samples))

    def log_prob_fn(*, sample):
        return trainer.sampler.log_prob(state.sampler_state, sample)

    try:
        fig, _ = gmmvi_utils.visualise_pairwise_2d_marginal(
            log_prob_fn=log_prob_fn,
            dr_range_low=trainer.env.domain_spec.low,
            dr_range_high=trainer.env.domain_spec.high,
            eval_samples=samples,
            max_dims=config.sampler_plot_max_dims,
            marginal_mc_samples=config.sampler_plot_context_samples,
            num_grid=config.sampler_plot_grid,
            show=False,
        )
    except Exception as exc:
        print(f"Warning: failed to save sampler plot at eval={eval_index}, steps={env_steps}: {exc}")
        return None
    fig.suptitle(f"Sampler Heatmap eval={eval_index} steps={env_steps}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()

        plot_dir = Path(config.sampler_plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"sampler_eval_{eval_index:03d}_steps_{env_steps}.png"
        fig.savefig(plot_path, bbox_inches="tight", pad_inches=0.1)
    except Exception as exc:
        print(f"Warning: failed to save sampler plot at eval={eval_index}, steps={env_steps}: {exc}")
        plt.close(fig)
        return None

    try:
        import wandb

        if wandb.run is not None:
            wandb.log({"evaluation/artifacts/sampler_heatmap": wandb.Image(str(plot_path))}, step=env_steps)
    except Exception:
        pass
    finally:
        plt.close(fig)

    return str(plot_path)


def _track_boundary_lines(track) -> tuple:
    import numpy as np

    xs = np.asarray(track.closed_xs)
    ys = np.asarray(track.closed_ys)
    yaws = np.concatenate([np.asarray(track.yaws), np.asarray(track.yaws[:1])])
    widths = np.concatenate([np.asarray(track.widths), np.asarray(track.widths[:1])])
    half_width = 0.5 * widths
    normal_x = -np.sin(yaws)
    normal_y = np.cos(yaws)
    left_x = xs + normal_x * half_width
    left_y = ys + normal_y * half_width
    right_x = xs - normal_x * half_width
    right_y = ys - normal_y * half_width
    return left_x, left_y, right_x, right_y


def _save_eval_render(
    trainer: SamplerPPOTrainer,
    state,
    render_env: F1TenthAdvWrapper,
    key: jax.Array,
    config: SamplerPPOConfig,
    *,
    eval_index: int,
    env_steps: int,
    dynamics_params: jax.Array | None = None,
    render_image: bool | None = None,
    render_video: bool | None = None,
    file_label: str = "eval",
    wandb_image_key: str = "evaluation/artifacts/trajectory",
    wandb_video_key: str = "evaluation/artifacts/video",
    title_label: str = "Eval Trajectory",
    subtitle: str | None = None,
) -> tuple[str | None, str | None]:
    save_image = config.eval_render if render_image is None else bool(render_image)
    save_video = config.eval_video if render_video is None else bool(render_video)
    if not save_image and not save_video:
        return None, None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers

    image_path = None
    video_path = None
    fig = None
    episode_length = max(1, math.ceil(int(config.eval_episode_steps) / config.policy_repeat_steps))
    try:
        if dynamics_params is None:
            key_reset, key_params = jax.random.split(key)
            dynamics_params = (
                render_env.sample_uniform_params(key_params)
                if config.eval_domain_randomization
                else render_env.nominal_dynamics_params
            )
        else:
            key_reset = key
            dynamics_params = jnp.asarray(dynamics_params, dtype=jnp.float32)
            if dynamics_params.ndim == 1:
                dynamics_params = dynamics_params[None, :]
        trajectory = record_policy_trajectory(
            render_env,
            trainer.make_policy(state.params, deterministic=True),
            key_reset,
            dynamics_params,
            episode_length,
        )
        states = np.asarray(trajectory.states[:, 0, :])
        rewards = np.asarray(trajectory.rewards[:, 0])
        done = np.asarray(trajectory.done[:, 0], dtype=bool)
        active = np.asarray(trajectory.active[:, 0], dtype=bool)
        valid_count = int(np.sum(active))
        if np.any(done):
            valid_count = int(np.argmax(done) + 1)
        valid_count = max(1, min(valid_count, states.shape[0]))

        track_x = np.asarray(render_env.env.track.closed_xs)
        track_y = np.asarray(render_env.env.track.closed_ys)
        left_x, left_y, right_x, right_y = _track_boundary_lines(render_env.env.track)
        traj_x = states[:valid_count, 0]
        traj_y = states[:valid_count, 1]
        steer = states[:valid_count, 2]
        speed = states[:valid_count, 3]
        yaw = states[:valid_count, 4]
        beta = states[:valid_count, 6]
        vx = speed * np.cos(beta)
        wheelbase = float(render_env.env.params.lf + render_env.env.params.lr)
        lf = float(render_env.env.params.lf)
        heading_len = max(wheelbase, 0.45)
        steer_len = 0.8 * heading_len
        slip_len = 0.9 * heading_len

        def setup_axis(ax):
            ax.fill(
                np.concatenate([left_x, right_x[::-1]]),
                np.concatenate([left_y, right_y[::-1]]),
                color="0.92",
                alpha=0.9,
                label="track",
                zorder=0,
            )
            ax.plot(left_x, left_y, color="0.45", linewidth=1.1, label="boundary")
            ax.plot(right_x, right_y, color="0.45", linewidth=1.1)
            ax.plot(track_x, track_y, color="0.25", linewidth=1.0, linestyle="--", label="centerline")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.grid(True, alpha=0.25)

        def draw_pose_arrows(ax):
            pose_count = min(valid_count, 24)
            pose_indices = np.unique(np.linspace(0, valid_count - 1, pose_count, dtype=int))
            pose_x = traj_x[pose_indices]
            pose_y = traj_y[pose_indices]
            pose_yaw = yaw[pose_indices]
            pose_steer = steer[pose_indices]
            pose_beta = beta[pose_indices]
            front_x = pose_x + lf * np.cos(pose_yaw)
            front_y = pose_y + lf * np.sin(pose_yaw)
            ax.quiver(
                pose_x,
                pose_y,
                heading_len * np.cos(pose_yaw),
                heading_len * np.sin(pose_yaw),
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="tab:blue",
                width=0.004,
                alpha=0.8,
                label="heading",
            )
            ax.quiver(
                front_x,
                front_y,
                steer_len * np.cos(pose_yaw + pose_steer),
                steer_len * np.sin(pose_yaw + pose_steer),
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="tab:orange",
                width=0.004,
                alpha=0.85,
                label="steer",
            )
            ax.quiver(
                pose_x,
                pose_y,
                slip_len * np.cos(pose_yaw + pose_beta),
                slip_len * np.sin(pose_yaw + pose_beta),
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="tab:purple",
                width=0.003,
                alpha=0.55,
                label="velocity/slip",
            )

        episode_return = float(np.sum(rewards[:valid_count]))
        max_abs_steer = float(np.rad2deg(np.max(np.abs(steer))))
        max_abs_slip = float(np.rad2deg(np.max(np.abs(beta))))
        max_abs_vx = float(np.max(np.abs(vx)))
        extra_title = f"\n{subtitle}" if subtitle else ""
        if save_image:
            fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
            setup_axis(ax)
            ax.plot(traj_x, traj_y, color="tab:red", linewidth=2.0, label="policy")
            draw_pose_arrows(ax)
            ax.scatter(traj_x[0], traj_y[0], color="tab:green", s=36, label="start", zorder=3)
            ax.scatter(traj_x[-1], traj_y[-1], color="tab:red", s=36, label="end", zorder=3)
            ax.set_title(
                f"{title_label} eval={eval_index} steps={env_steps} return={episode_return:.2f}\n"
                f"max|steer|={max_abs_steer:.1f} deg max|slip|={max_abs_slip:.1f} deg "
                f"max|v_x|={max_abs_vx:.2f} m/s{extra_title}"
            )
            ax.legend(loc="best")
            render_dir = Path(config.eval_render_dir)
            render_dir.mkdir(parents=True, exist_ok=True)
            image_path = render_dir / f"eval_render_{file_label}_{eval_index:03d}_steps_{env_steps}.png"
            fig.savefig(image_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            fig = None

        if save_video:
            video_dir = Path(config.eval_video_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            fps = max(1, int(config.eval_video_fps))
            use_ffmpeg = writers.is_available("ffmpeg")
            video_suffix = "mp4" if use_ffmpeg else "gif"
            video_path = video_dir / f"eval_video_{file_label}_{eval_index:03d}_steps_{env_steps}.{video_suffix}"
            frame_count = min(valid_count, max(1, int(config.eval_video_max_frames)))
            frame_indices = np.unique(np.linspace(0, valid_count - 1, frame_count, dtype=int))

            fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
            setup_axis(ax)
            (line,) = ax.plot([], [], color="tab:red", linewidth=2.0, label="policy")
            point = ax.scatter([traj_x[0]], [traj_y[0]], color="tab:red", s=36, zorder=3)
            (heading_line,) = ax.plot([], [], color="tab:blue", linewidth=2.4, label="heading")
            (steer_line,) = ax.plot([], [], color="tab:orange", linewidth=2.4, label="steer")
            (slip_line,) = ax.plot([], [], color="tab:purple", linewidth=2.0, alpha=0.75, label="velocity/slip")
            title = ax.set_title("")
            velocity_text = ax.text(
                0.02,
                0.98,
                "",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
            )
            ax.legend(loc="best")

            def update(frame_idx):
                end = int(frame_idx) + 1
                idx = int(frame_idx)
                center_x = traj_x[idx]
                center_y = traj_y[idx]
                current_yaw = yaw[idx]
                current_steer = steer[idx]
                current_beta = beta[idx]
                current_vx = vx[idx]
                front_x = center_x + lf * np.cos(current_yaw)
                front_y = center_y + lf * np.sin(current_yaw)
                line.set_data(traj_x[:end], traj_y[:end])
                point.set_offsets(np.asarray([[center_x, center_y]]))
                heading_line.set_data(
                    [center_x, center_x + heading_len * np.cos(current_yaw)],
                    [center_y, center_y + heading_len * np.sin(current_yaw)],
                )
                steer_line.set_data(
                    [front_x, front_x + steer_len * np.cos(current_yaw + current_steer)],
                    [front_y, front_y + steer_len * np.sin(current_yaw + current_steer)],
                )
                slip_line.set_data(
                    [center_x, center_x + slip_len * np.cos(current_yaw + current_beta)],
                    [center_y, center_y + slip_len * np.sin(current_yaw + current_beta)],
                )
                title.set_text(
                    f"{title_label} eval={eval_index} frame={end}/{valid_count} "
                    f"return={float(np.sum(rewards[:end])):.2f}\n"
                    f"v_x={current_vx:.2f} m/s "
                    f"steer={np.rad2deg(current_steer):.1f} deg slip={np.rad2deg(current_beta):.1f} deg"
                    f"{extra_title}"
                )
                velocity_text.set_text(f"v_x: {current_vx:.2f} m/s")
                return line, point, heading_line, steer_line, slip_line, title, velocity_text

            animation = FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps)
            if use_ffmpeg:
                writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
                try:
                    animation.save(video_path, writer=writer)
                except Exception as exc:
                    print(f"Warning: failed to save mp4 eval video, falling back to gif: {exc}")
                    video_path = video_dir / f"eval_video_{file_label}_{eval_index:03d}_steps_{env_steps}.gif"
                    animation.save(video_path, writer=PillowWriter(fps=fps))
            else:
                animation.save(video_path, writer=PillowWriter(fps=fps))
    except Exception as exc:
        print(f"Warning: failed to save eval render/video at eval={eval_index}, steps={env_steps}: {exc}")
        return image_path and str(image_path), video_path and str(video_path)
    finally:
        if fig is not None:
            plt.close(fig)

    try:
        import wandb

        if wandb.run is not None:
            payload = {}
            if image_path is not None:
                payload[wandb_image_key] = wandb.Image(str(image_path))
            if video_path is not None:
                payload[wandb_video_key] = wandb.Video(
                    str(video_path),
                    fps=max(1, int(config.eval_video_fps)),
                    format=video_path.suffix.lstrip("."),
                )
            if payload:
                wandb.log(payload, step=env_steps)
    except Exception:
        pass

    return image_path and str(image_path), video_path and str(video_path)


def _eval_video_percentile_values(config: SamplerPPOConfig) -> tuple[int, ...]:
    step = max(1, int(config.eval_video_percentile_step))
    values = list(range(0, 101, step))
    if values[-1] != 100:
        values.append(100)
    return tuple(values)


def _select_reward_percentile_indices(rewards, percentiles: tuple[int, ...]):
    import numpy as np

    rewards = np.asarray(rewards, dtype=np.float32)
    selections = []
    for percentile in percentiles:
        target = float(np.percentile(rewards, percentile))
        index = int(np.argmin(np.abs(rewards - target)))
        selections.append(
            {
                "percentile": int(percentile),
                "index": index,
                "reward": float(rewards[index]),
                "target_reward": target,
            }
        )
    return selections


def _save_eval_percentile_videos(
    trainer: SamplerPPOTrainer,
    state,
    render_env: F1TenthAdvWrapper,
    key: jax.Array,
    config: SamplerPPOConfig,
    *,
    eval_index: int,
    env_steps: int,
    rewards,
    lengths,
    dynamics_params,
) -> tuple[str | None, dict[str, str]]:
    if not config.eval_video or not config.eval_video_percentiles:
        return None, {}

    import csv

    import numpy as np

    percentiles = _eval_video_percentile_values(config)
    selections = _select_reward_percentile_indices(rewards, percentiles)
    rewards = np.asarray(rewards, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    dynamics_params = np.asarray(dynamics_params, dtype=np.float32)
    param_names = tuple(render_env.domain_spec.names)

    video_dir = Path(config.eval_video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    csv_path = video_dir / f"eval_dynamics_percentiles_{eval_index:03d}_steps_{env_steps}.csv"
    columns = [
        "percentile",
        "eval_env_index",
        "selected_reward",
        "target_percentile_reward",
        "episode_length",
        *param_names,
    ]

    rows = []
    for selection in selections:
        index = selection["index"]
        values = dynamics_params[index]
        rows.append(
            [
                selection["percentile"],
                index,
                selection["reward"],
                selection["target_reward"],
                float(lengths[index]),
                *[float(value) for value in values],
            ]
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {"evaluation/artifacts/dynamics_percentile_params": wandb.Table(columns=columns, data=rows)},
                step=env_steps,
            )
    except Exception:
        pass

    video_paths = {}
    render_keys = jax.random.split(key, len(selections))
    for render_key, selection in zip(render_keys, selections):
        percentile = selection["percentile"]
        index = selection["index"]
        label = f"p{percentile}"
        subtitle = (
            f"eval_env={index} selected_reward={selection['reward']:.2f} "
            f"target_p_reward={selection['target_reward']:.2f}"
        )
        _, video_path = _save_eval_render(
            trainer,
            state,
            render_env,
            render_key,
            config,
            eval_index=eval_index,
            env_steps=env_steps,
            dynamics_params=jnp.asarray(dynamics_params[index][None, :], dtype=jnp.float32),
            render_image=False,
            render_video=True,
            file_label=label,
            wandb_video_key=f"evaluation/artifacts/video_{label}",
            title_label=f"Eval Video {label}",
            subtitle=subtitle,
        )
        if video_path is not None:
            video_paths[label] = video_path

    return str(csv_path), video_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-config", type=Path, default=_CONFIG_DIR / "jax" / "rl_config.yaml")
    parser.add_argument("--gym-config", type=Path, default=_CONFIG_DIR / "jax" / "gym_config.yaml")
    parser.add_argument("--no-domain-randomization", action="store_true",
                        help="Use nominal dynamics for both training and evaluation")
    parser.add_argument("--track", default=None, help="Optional explicit map override; default comes from gym_config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override rl_config.yaml seed")
    parser.add_argument(
        "--sampler",
        choices=("uniform", "gmmvi", "reward_cost_gmmvi"),
        default=None,
        help="Override jax_sampler_ppo.sampler",
    )
    parser.add_argument(
        "--gmm-target-beta",
        "--gmm_target_beta",
        dest="gmm_target_beta",
        type=float,
        default=None,
        help="Override jax_sampler_ppo.gmm_target_beta",
    )
    parser.add_argument(
        "--gmm-formulation",
        choices=("reward", "cost", "reward_cost"),
        default=None,
        help="Use reward-only, cost-only, or a 50/50 reward+cost dual GMMVI batch",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="Optional override; default comes from jax_sampler_ppo.num_envs")
    parser.add_argument(
        "--num-eval-envs",
        type=int,
        default=None,
        help="Optional override; 0 disables periodic JAX evaluation",
    )
    parser.add_argument("--num-evals", type=int, default=None, help="Total number of JAX evaluations")
    parser.add_argument(
        "--dr-profile",
        default=None,
        help="Named JAX domain-randomization profile from rl_config.yaml",
    )
    parser.add_argument("--ppo-lag", action="store_true", help="Enable CRAX PPO-Lagrange")
    parser.add_argument(
        "--constraint-cost",
        choices=("edge", "collision"),
        default=None,
        help="PPO-Lag constraint signal: continuous edge penalty or binary terminal collision",
    )
    parser.add_argument("--safety-bound", type=float, default=None, help="PPO-Lag cumulative constraint-cost budget per episode")
    parser.add_argument(
        "--lagrangian-coef-rate",
        type=float,
        default=None,
        help="PPO-Lag Lagrange-multiplier update rate",
    )
    parser.add_argument(
        "--initial-lambda-lagr",
        type=float,
        default=None,
        help="PPO-Lag initial non-negative Lagrange multiplier",
    )
    parser.add_argument("--eval-episode-steps", type=int, default=None, help="Separate JAX eval episode length")
    parser.add_argument(
        "--eval-episodes-per-dynamics",
        type=int,
        default=None,
        help="Number of eval episodes to run for each selected dynamics parameter vector",
    )
    parser.add_argument("--sampler-plot-dir", type=Path, default=None, help="Directory for per-eval sampler plots")
    parser.add_argument("--render-eval", action="store_true", help="Enable per-eval trajectory render PNGs")
    parser.add_argument("--no-render-eval", action="store_true", help="Disable per-eval trajectory render PNGs")
    parser.add_argument("--eval-render-dir", type=Path, default=None, help="Directory for per-eval trajectory renders")
    parser.add_argument("--eval-video", action="store_true", help="Enable per-eval trajectory videos")
    parser.add_argument("--no-eval-video", action="store_true", help="Disable per-eval trajectory videos")
    parser.add_argument("--eval-video-dir", type=Path, default=None, help="Directory for per-eval trajectory videos")
    parser.add_argument("--eval-video-max-frames", type=int, default=None, help="Maximum frames per eval trajectory video")
    parser.add_argument("--updates", type=int, default=None, help="Optional override; default derives from total_timesteps")
    parser.add_argument("--onnx-output", default="", help="Optional path to export the deterministic actor as ONNX")
    parser.add_argument("--deployment-output", type=Path, default=None,
                        help="Export NumPy actor, track and observation/action contract for legacy ROS1")
    parser.add_argument("--checkpoint-output", type=Path, default=None,
                        help="Save final full JAX training state for reproducibility")
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="Restore full state; total_timesteps is the absolute training target")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging for this run")
    parser.add_argument(
        "--wandb-save-code",
        action="store_true",
        help="Enable WandB code saving and git probing for this run",
    )
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args()

    rl_config = _load_yaml(args.rl_config)
    gym_config = _load_yaml(args.gym_config)

    # Make a command-line track override part of the effective experiment
    # configuration before WandB is initialized.  This keeps the run name and
    # logged config aligned with the environment that is actually trained.
    if args.track is not None:
        gym_config["map"] = args.track

    if rl_config.get("use_custom_relu"):
        raise NotImplementedError("JAX sampler PPO currently supports the default tanh MLP only; set use_custom_relu: false.")

    if args.seed is not None:
        rl_config["seed"] = args.seed
    if args.sampler is not None:
        rl_config["jax_sampler_ppo"]["sampler"] = args.sampler
    if args.gmm_target_beta is not None:
        rl_config["jax_sampler_ppo"]["gmm_target_beta"] = args.gmm_target_beta
    if args.gmm_formulation is not None:
        formulation_fraction = {"reward": 1.0, "cost": 0.0, "reward_cost": 0.5}
        rl_config["jax_sampler_ppo"]["sampler"] = "reward_cost_gmmvi"
        rl_config["jax_sampler_ppo"]["gmm_formulation"] = args.gmm_formulation
        rl_config["jax_sampler_ppo"]["gmm_reward_fraction"] = formulation_fraction[args.gmm_formulation]
    if args.no_domain_randomization:
        rl_config["jax_sampler_ppo"]["domain_randomization"] = False
        rl_config["jax_sampler_ppo"]["eval_domain_randomization"] = False
    if args.num_eval_envs is not None:
        rl_config["jax_sampler_ppo"]["num_eval_envs"] = args.num_eval_envs
    if args.num_evals is not None:
        rl_config["jax_sampler_ppo"]["num_evals"] = args.num_evals
    if args.dr_profile is not None:
        profiles = rl_config["jax_sampler_ppo"].get("domain_randomization_profiles", {})
        if args.dr_profile not in profiles:
            available = ", ".join(sorted(profiles)) or "none"
            raise ValueError(f"Unknown DR profile {args.dr_profile!r}; available profiles: {available}")
        rl_config["jax_sampler_ppo"]["domain_randomization_ranges"] = profiles[args.dr_profile]
        rl_config["jax_sampler_ppo"]["domain_randomization_profile"] = args.dr_profile
    if args.ppo_lag:
        rl_config["jax_sampler_ppo"]["use_ppo_lag"] = True
    if args.constraint_cost is not None:
        rl_config["jax_sampler_ppo"]["constraint_cost_type"] = args.constraint_cost
    if args.safety_bound is not None:
        rl_config["jax_sampler_ppo"]["safety_bound"] = args.safety_bound
    if args.lagrangian_coef_rate is not None:
        rl_config["jax_sampler_ppo"]["lagrangian_coef_rate"] = args.lagrangian_coef_rate
    if args.initial_lambda_lagr is not None:
        rl_config["jax_sampler_ppo"]["initial_lambda_lagr"] = args.initial_lambda_lagr
    if args.eval_episode_steps is not None:
        rl_config["jax_sampler_ppo"]["eval_episode_steps"] = args.eval_episode_steps
    if args.eval_episodes_per_dynamics is not None:
        rl_config["jax_sampler_ppo"]["eval_episodes_per_dynamics"] = args.eval_episodes_per_dynamics
    if args.sampler_plot_dir is not None:
        rl_config["jax_sampler_ppo"]["sampler_plot_dir"] = str(args.sampler_plot_dir)
    if args.render_eval:
        rl_config["jax_sampler_ppo"]["eval_render"] = True
    if args.no_render_eval:
        rl_config["jax_sampler_ppo"]["eval_render"] = False
    if args.eval_render_dir is not None:
        rl_config["jax_sampler_ppo"]["eval_render_dir"] = str(args.eval_render_dir)
    if args.eval_video:
        rl_config["jax_sampler_ppo"]["eval_video"] = True
    if args.no_eval_video:
        rl_config["jax_sampler_ppo"]["eval_video"] = False
    if args.eval_video_dir is not None:
        rl_config["jax_sampler_ppo"]["eval_video_dir"] = str(args.eval_video_dir)
    if args.eval_video_max_frames is not None:
        rl_config["jax_sampler_ppo"]["eval_video_max_frames"] = args.eval_video_max_frames
    if args.resume_checkpoint is not None:
        rl_config["resume_checkpoint"] = str(args.resume_checkpoint.resolve())

    wandb_started_here = _init_wandb(
        gym_config,
        rl_config,
        disabled=args.no_wandb,
        save_code=args.wandb_save_code,
    )
    run_output_label = _run_output_label()
    scoped_dir_keys = tuple(
        key
        for key, overridden in (
            ("sampler_plot_dir", args.sampler_plot_dir is not None),
            ("eval_render_dir", args.eval_render_dir is not None),
            ("eval_video_dir", args.eval_video_dir is not None),
        )
        if not overridden
    )
    _apply_scoped_output_dirs(rl_config, run_output_label, scoped_keys=scoped_dir_keys)
    _wandb_update_config({"rl_config": rl_config})
    jax_config = rl_config["jax_sampler_ppo"]
    domain_randomization_ranges = jax_config.get("domain_randomization_ranges")
    asymmetric_critic = bool(jax_config.get("asymmetric_critic", False))
    configured_num_envs = jax_config.get("num_envs")
    num_envs = int(
        args.num_envs
        if args.num_envs is not None
        else configured_num_envs
        if configured_num_envs is not None
        else int(rl_config["core_mult"]) * multiprocessing.cpu_count()
    )
    config = _build_trainer_config(rl_config, num_envs)
    updates = int(
        args.updates
        if args.updates is not None
        else _num_updates(int(rl_config["total_timesteps"]), num_envs, config.unroll_length)
    )
    env = _build_env(
        gym_config,
        num_envs,
        int(rl_config["seed"]),
        args.track,
        domain_randomization_ranges=domain_randomization_ranges,
        asymmetric_critic=asymmetric_critic,
        action_repeat_steps=config.policy_repeat_steps,
        constraint_cost_type=config.constraint_cost_type,
    )

    trainer = SamplerPPOTrainer(env, config)
    state = trainer.init_state(jax.random.PRNGKey(int(rl_config["seed"])))
    resume_path = rl_config.get("resume_checkpoint")
    if resume_path:
        resume_path = Path(resume_path)
        if args.checkpoint_output is not None and args.checkpoint_output.resolve() == resume_path.resolve():
            raise ValueError("Resume output must not overwrite the source checkpoint")
        state = _restore_training_checkpoint(state, resume_path, num_envs=num_envs,
                                             rollout_length=config.unroll_length,
                                             target_updates=updates)
        rl_config["resume_from_update"] = int(state.update_steps)
        _wandb_update_config({"rl_config": rl_config})
        print(f"Restored full training state: updates={int(state.update_steps)} "
              f"steps={int(state.env_steps)} lambda={float(state.lambda_lagr):.6f} "
              f"source={resume_path}")
    start_update = int(state.update_steps)
    step_fn = trainer.training_step if args.no_jit else jax.jit(trainer.training_step)
    eval_key = jax.random.PRNGKey(int(rl_config.get("eval_seed", rl_config["seed"] + 1)))
    eval_fn = None
    render_env = None
    run_initial_eval, eval_updates = _evaluation_schedule(updates, config.num_evals,
                                                        start_update=start_update)
    eval_count = 0
    if config.num_eval_envs > 0:
        eval_env = _build_env(
            gym_config,
            config.num_eval_envs,
            int(rl_config["seed"]),
            _resolve_eval_track(gym_config, args.track),
            domain_randomization_ranges=domain_randomization_ranges,
            asymmetric_critic=asymmetric_critic,
            max_episode_steps=config.eval_episode_steps,
            action_repeat_steps=config.policy_repeat_steps,
            constraint_cost_type=config.constraint_cost_type,
            warn_track_pool=False,
        )
        eval_fn = _make_eval_fn(
            eval_env,
            trainer,
            episode_steps=config.eval_episode_steps,
            episodes_per_dynamics=config.eval_episodes_per_dynamics,
            randomize_dynamics=config.eval_domain_randomization,
            jit=not args.no_jit,
        )
        if config.eval_render or config.eval_video:
            render_env = _build_env(
                gym_config,
                1,
                int(rl_config["seed"]),
                _resolve_eval_track(gym_config, args.track),
                domain_randomization_ranges=domain_randomization_ranges,
                asymmetric_critic=asymmetric_critic,
                auto_reset=False,
                max_episode_steps=config.eval_episode_steps,
                action_repeat_steps=config.policy_repeat_steps,
                constraint_cost_type=config.constraint_cost_type,
                warn_track_pool=False,
            )

    print(f"JAX devices: {jax.devices()}")
    physics_hz = 1.0 / float(gym_config["timestep"])
    policy_hz = physics_hz / float(config.policy_repeat_steps)
    print(
        "Training sampler PPO: "
        f"sampler={config.sampler}, envs={num_envs}, unroll={config.unroll_length}, "
        f"policy_repeat={config.policy_repeat_steps}, batch_size={config.batch_size}, "
        f"updates={updates}, eval_envs={config.num_eval_envs}, "
        f"num_evals={config.num_evals}, eval_episodes_per_dynamics={config.eval_episodes_per_dynamics}, "
        f"constraint_cost={config.constraint_cost_type}, safety_bound={config.safety_bound:g}"
    )
    print(f"Control rate: physics={physics_hz:.1f} Hz, policy={policy_hz:.1f} Hz")
    print(
        "Track direction: "
        f"{'reverse' if _resolve_direction(gym_config['track_direction'], int(rl_config['seed'])) else 'normal'} "
        f"(track_direction={gym_config['track_direction']}, seed={int(rl_config['seed'])})"
    )
    print(
        "Eval dynamics: "
        f"{'randomized over DR bounds' if config.eval_domain_randomization else 'nominal only'}"
    )
    print(
        "Eval percentile videos: "
        f"{'enabled' if config.eval_video and config.eval_video_percentiles else 'disabled'} "
        f"(step={config.eval_video_percentile_step}, "
        f"{'final eval only' if config.eval_video_final else 'every eval'})"
    )
    print(
        "Artifact dirs: "
        f"sampler_plots={config.sampler_plot_dir}, "
        f"eval_renders={config.eval_render_dir}, "
        f"eval_videos={config.eval_video_dir}"
    )
    if eval_fn is not None and run_initial_eval:
        eval_count += 1
        eval_key, key_eval, key_plot, key_render = jax.random.split(eval_key, 4)
        eval_metrics, eval_rewards, eval_lengths, eval_dynamics_params = eval_fn(state.params, key_eval)
        eval_metrics = _metric_dict(eval_metrics)
        eval_metrics["eval/index"] = eval_count
        eval_metrics["eval/update"] = start_update
        eval_metrics["evaluation/meta/index"] = eval_count
        eval_metrics["evaluation/meta/update"] = start_update
        eval_logged = _wandb_log(eval_metrics, int(state.env_steps))
        sampler_plot = _save_sampler_plot(
            trainer,
            state,
            key_plot,
            config,
            eval_index=eval_count,
            env_steps=int(state.env_steps),
        )
        key_render_single, key_render_percentiles = jax.random.split(key_render)
        should_save_eval_video = _should_save_eval_video(config, is_final=False)
        eval_render, eval_video = (
            _save_eval_render(
                trainer,
                state,
                render_env,
                key_render_single,
                config,
                eval_index=eval_count,
                env_steps=int(state.env_steps),
                render_video=should_save_eval_video,
            )
            if render_env is not None
            else (None, None)
        )
        eval_percentile_csv, eval_percentile_videos = (
            _save_eval_percentile_videos(
                trainer,
                state,
                render_env,
                key_render_percentiles,
                config,
                eval_index=eval_count,
                env_steps=int(state.env_steps),
                rewards=eval_rewards,
                lengths=eval_lengths,
                dynamics_params=eval_dynamics_params,
            )
            if render_env is not None and should_save_eval_video
            else (None, {})
        )
        print(
            f"eval index={eval_count} update={start_update} steps={int(state.env_steps)} "
            f"wandb_logged={eval_logged} sampler_plot={sampler_plot} "
            f"eval_render={eval_render} eval_video={eval_video} "
            f"eval_percentile_csv={eval_percentile_csv} "
            f"eval_percentile_videos={len(eval_percentile_videos)}"
        )

    for update in range(start_update, updates):
        update_number = update + 1
        state, metrics = step_fn(state)
        metrics = _metric_dict(metrics)
        env_steps = int(metrics["train/env_steps"])
        train_logged = _wandb_log(metrics, env_steps)
        eval_logged = False
        eval_count_for_message = None
        sampler_plot = None
        if eval_fn is not None and update_number in eval_updates:
            is_final_eval = update_number == updates
            eval_count += 1
            eval_key, key_eval, key_plot, key_render = jax.random.split(eval_key, 4)
            eval_metrics, eval_rewards, eval_lengths, eval_dynamics_params = eval_fn(state.params, key_eval)
            eval_metrics = _metric_dict(eval_metrics)
            eval_metrics["eval/index"] = eval_count
            eval_metrics["eval/update"] = update_number
            eval_metrics["evaluation/meta/index"] = eval_count
            eval_metrics["evaluation/meta/update"] = update_number
            eval_logged = _wandb_log(eval_metrics, env_steps)
            eval_count_for_message = eval_count
            sampler_plot = _save_sampler_plot(
                trainer,
                state,
                key_plot,
                config,
                eval_index=eval_count,
                env_steps=int(state.env_steps),
            )
            key_render_single, key_render_percentiles = jax.random.split(key_render)
            should_save_eval_video = _should_save_eval_video(config, is_final=is_final_eval)
            eval_render, eval_video = (
                _save_eval_render(
                    trainer,
                    state,
                    render_env,
                    key_render_single,
                    config,
                    eval_index=eval_count,
                    env_steps=int(state.env_steps),
                    render_video=should_save_eval_video,
                )
                if render_env is not None
                else (None, None)
            )
            eval_percentile_csv, eval_percentile_videos = (
                _save_eval_percentile_videos(
                    trainer,
                    state,
                    render_env,
                    key_render_percentiles,
                    config,
                    eval_index=eval_count,
                    env_steps=int(state.env_steps),
                    rewards=eval_rewards,
                    lengths=eval_lengths,
                    dynamics_params=eval_dynamics_params,
                )
                if render_env is not None and should_save_eval_video
                else (None, {})
            )
        else:
            eval_render = None
            eval_video = None
            eval_percentile_csv = None
            eval_percentile_videos = {}

        message = (
            f"update={update_number} "
            f"steps={env_steps} "
            f"reward={metrics['training/reward/rollout_return_mean']:.3f} "
            f"cost={metrics['training/cost/rollout_cost_mean']:.4f} "
            f"cost_violation={metrics['training/cost/signed_violation_mean']:.4f} "
            f"cost_satisfied={metrics['training/cost/satisfied_rate']:.3f} "
            f"loss={metrics['loss/total']:.3f} "
            f"wandb_logged={train_logged}"
        )
        if eval_count_for_message is not None:
            message += (
                f" eval_index={eval_count_for_message} "
                f"eval_reward={eval_metrics['evaluation/reward/episode_return_mean']:.3f} "
                f"eval_cost={eval_metrics['evaluation/cost/episode_cost_mean']:.4f} "
            )
            if config.use_ppo_lag:
                message += (
                    f"eval_cost_violation={eval_metrics['evaluation/cost/signed_violation_mean']:.4f} "
                    f"eval_cost_satisfied={eval_metrics['evaluation/cost/satisfied_rate']:.3f} "
                )
            message += (
                f"eval_wandb_logged={eval_logged} "
                f"sampler_plot={sampler_plot} "
                f"eval_render={eval_render} "
                f"eval_video={eval_video} "
                f"eval_percentile_csv={eval_percentile_csv} "
                f"eval_percentile_videos={len(eval_percentile_videos)}"
            )
        print(message)

    if args.checkpoint_output is not None:
        import json
        from flax.serialization import to_bytes
        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint_output.write_bytes(to_bytes(jax.device_get(state)))
        final_metrics = _metric_dict(metrics)
        if eval_fn is not None and (run_initial_eval or eval_updates):
            final_metrics.update(eval_metrics)
        final_metrics.update({
            "training/progress/env_steps": int(state.env_steps),
            "training/progress/update_steps": int(state.update_steps),
            "training/constraint/lambda_lagr": float(state.lambda_lagr),
        })
        args.checkpoint_output.with_suffix(".summary.json").write_text(
            json.dumps(final_metrics, indent=2))
        print(f"Saved final JAX checkpoint: {args.checkpoint_output}")
    if args.onnx_output:
        from gymkhana.jax_sampler_ppo import export_trainer_policy_to_onnx

        export_trainer_policy_to_onnx(trainer, state, args.onnx_output)
        print(f"Exported ONNX policy: {args.onnx_output}")

    if args.deployment_output is not None:
        from gymkhana.jax_sampler_ppo.export_deploy import export_deployment_bundle

        contract = export_deployment_bundle(trainer, state, args.deployment_output)
        print(f"Exported ROS1 bundle: {args.deployment_output} "
              f"(actor_obs={contract['actor_observation_size']}, policy_hz={contract['policy_hz']})")

    if wandb_started_here:
        try:
            import wandb

            completion_text = (
                f"Track={gym_config['map']}, sampler={config.sampler}, "
                f"seed={rl_config['seed']}, steps={int(state.env_steps)}"
            )
            if eval_fn is not None and (run_initial_eval or eval_updates):
                completion_text += (
                    f"\nEvaluation return={eval_metrics['evaluation/reward/episode_return_mean']:.2f}, "
                    f"collision rate={eval_metrics['evaluation/collision/rate']:.2%}, "
                    f"{config.constraint_cost_type} cost budget={config.safety_bound:g}"
                )
            try:
                wandb.alert(title="F1 training completed", text=completion_text, level="INFO", wait_duration=0)
            except Exception as exc:
                print(f"W&B completion alert failed: {exc}")
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
