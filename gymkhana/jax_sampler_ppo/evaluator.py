"""Evaluator and unroll helpers for F1TENTH JAX sampler PPO."""

from __future__ import annotations

import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .wrappers import AdvEnvState, F1TenthAdvWrapper, TransitionWithParams

PolicyFn = Callable[[jax.Array, jax.Array], tuple[jax.Array, dict]]
REWARD_PERCENTILES = tuple(range(5, 100, 5))


class EvalResult(NamedTuple):
    metrics: dict
    rewards: jax.Array
    lengths: jax.Array


class TrajectoryResult(NamedTuple):
    states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    done: jax.Array
    active: jax.Array


class _EvalEpisodeBatch(NamedTuple):
    rewards: jax.Array
    lengths: jax.Array
    slip_angle_abs_deg: jax.Array
    slip_reward: jax.Array


def generate_adv_unroll(
    env: F1TenthAdvWrapper,
    env_state: AdvEnvState,
    dynamics_params: jax.Array,
    policy: PolicyFn,
    key: jax.Array,
    unroll_length: int,
) -> tuple[AdvEnvState, TransitionWithParams]:
    """Collect a fixed-length rollout with explicit dynamics parameters."""
    env_state = env.with_dynamics_params(env_state, dynamics_params)

    def scan_step(carry, _):
        state, current_key = carry
        current_key, key_policy, key_env = jax.random.split(current_key, 3)
        action, policy_extras = policy(state.obs["actor_obs"], key_policy)
        next_state, out = env.step(state, action, key_env)
        transition = TransitionWithParams(
            observation=state.obs,
            dynamics_params=state.dynamics_params,
            action=action,
            raw_action=policy_extras.get("raw_action", action),
            log_prob=policy_extras.get("log_prob", jnp.zeros_like(out.reward)),
            reward=out.reward,
            discount=1.0 - out.done.astype(jnp.float32),
            next_observation=next_state.obs,
            done=out.done,
            extras={
                "policy_extras": policy_extras,
                "state_extras": {
                    "truncation": out.metrics.get("truncated", jnp.zeros_like(out.reward)),
                    "boundary": out.metrics.get("boundary", jnp.zeros_like(out.reward)),
                    "path_reward": out.metrics.get("path_reward", jnp.zeros_like(out.reward)),
                    "slip_angle_abs_deg": out.metrics.get("slip_angle_abs_deg", jnp.zeros_like(out.reward)),
                    "slip_reward_raw": out.metrics.get("slip_reward_raw", jnp.zeros_like(out.reward)),
                    "slip_reward": out.metrics.get("slip_reward", jnp.zeros_like(out.reward)),
                },
            },
        )
        return (next_state, current_key), transition

    (final_state, _), data = jax.lax.scan(scan_step, (env_state, key), None, length=unroll_length)
    return final_state, data


def _build_eval_metrics(
    rewards: jax.Array,
    lengths: jax.Array,
    slip_angle_abs_deg: jax.Array,
    slip_reward: jax.Array,
    *,
    dynamics_count: int,
    episodes_per_dynamics: int,
) -> dict:
    rewards_sorted = jnp.sort(rewards)
    k10 = max(1, int(rewards.shape[0] * 0.1))
    k20 = max(1, int(rewards.shape[0] * 0.2))
    reward_percentiles = {
        f"eval/episode_reward_p{percentile}": jnp.percentile(rewards, percentile)
        for percentile in REWARD_PERCENTILES
    }
    return {
        "eval/episode_reward_mean": jnp.mean(rewards),
        "eval/episode_reward_std": jnp.std(rewards),
        "eval/episode_reward_min": jnp.min(rewards),
        "eval/episode_reward_max": jnp.max(rewards),
        **reward_percentiles,
        "eval/episode_reward_CVaR10": jnp.mean(rewards_sorted[:k10]),
        "eval/episode_reward_CVaR20": jnp.mean(rewards_sorted[:k20]),
        "eval/avg_episode_length": jnp.mean(lengths),
        "eval/std_episode_length": jnp.std(lengths),
        "eval/slip_angle_abs_deg_mean": jnp.mean(slip_angle_abs_deg),
        "eval/slip_reward_raw_mean": jnp.mean(slip_reward),
        "eval/dynamics_count": jnp.asarray(dynamics_count, dtype=jnp.float32),
        "eval/episodes_per_dynamics": jnp.asarray(episodes_per_dynamics, dtype=jnp.float32),
        "eval/total_episodes": jnp.asarray(rewards.shape[0], dtype=jnp.float32),
    }


def _evaluate_policy_episode_batch(
    env: F1TenthAdvWrapper,
    policy: PolicyFn,
    key: jax.Array,
    dynamics_params: jax.Array,
    episode_length: int,
) -> _EvalEpisodeBatch:
    """Run one episode per vectorized env and collect until first done."""
    state = env.reset(key, dynamics_params)
    active = jnp.ones((env.batch_size,), dtype=jnp.float32)
    episode_reward = jnp.zeros((env.batch_size,), dtype=jnp.float32)
    episode_length_acc = jnp.zeros((env.batch_size,), dtype=jnp.float32)
    slip_angle_abs_deg_acc = jnp.zeros((env.batch_size,), dtype=jnp.float32)
    slip_reward_acc = jnp.zeros((env.batch_size,), dtype=jnp.float32)

    def scan_step(carry, _):
        state, active, episode_reward, episode_length_acc, slip_angle_abs_deg_acc, slip_reward_acc, current_key = carry
        current_key, key_policy, key_env = jax.random.split(current_key, 3)
        action, _ = policy(state.obs["actor_obs"], key_policy)
        next_state, out = env.step(state, action, key_env)
        episode_reward = episode_reward + out.reward * active
        episode_length_acc = episode_length_acc + active
        slip_angle_abs_deg_acc = slip_angle_abs_deg_acc + out.metrics["slip_angle_abs_deg"] * active
        slip_reward_acc = slip_reward_acc + out.metrics["slip_reward_raw"] * active
        active = active * (1.0 - out.done.astype(jnp.float32))
        return (
            next_state,
            active,
            episode_reward,
            episode_length_acc,
            slip_angle_abs_deg_acc,
            slip_reward_acc,
            current_key,
        ), None

    (_, _, rewards, lengths, slip_angle_abs_deg_acc, slip_reward_acc, _), _ = jax.lax.scan(
        scan_step,
        (state, active, episode_reward, episode_length_acc, slip_angle_abs_deg_acc, slip_reward_acc, key),
        None,
        length=episode_length,
    )
    safe_lengths = jnp.maximum(lengths, 1.0)
    slip_angle_abs_deg_per_env = slip_angle_abs_deg_acc / safe_lengths
    slip_reward_per_env = slip_reward_acc / safe_lengths
    return _EvalEpisodeBatch(
        rewards=rewards,
        lengths=lengths,
        slip_angle_abs_deg=slip_angle_abs_deg_per_env,
        slip_reward=slip_reward_per_env,
    )


def evaluate_policy(
    env: F1TenthAdvWrapper,
    policy: PolicyFn,
    key: jax.Array,
    dynamics_params: jax.Array,
    episode_length: int,
    episodes_per_dynamics: int = 1,
) -> EvalResult:
    """Run eval rollouts and aggregate episode returns over selected dynamics."""
    episodes_per_dynamics = int(episodes_per_dynamics)
    if episodes_per_dynamics < 1:
        raise ValueError(f"episodes_per_dynamics must be >= 1, got {episodes_per_dynamics}")

    if episodes_per_dynamics == 1:
        batch = _evaluate_policy_episode_batch(env, policy, key, dynamics_params, episode_length)
    else:
        keys = jax.random.split(key, episodes_per_dynamics)

        def scan_episode(_, episode_key):
            return None, _evaluate_policy_episode_batch(
                env,
                policy,
                episode_key,
                dynamics_params,
                episode_length,
            )

        _, batch = jax.lax.scan(scan_episode, None, keys)
        batch = _EvalEpisodeBatch(
            rewards=jnp.reshape(batch.rewards, (-1,)),
            lengths=jnp.reshape(batch.lengths, (-1,)),
            slip_angle_abs_deg=jnp.reshape(batch.slip_angle_abs_deg, (-1,)),
            slip_reward=jnp.reshape(batch.slip_reward, (-1,)),
        )

    metrics = _build_eval_metrics(
        batch.rewards,
        batch.lengths,
        batch.slip_angle_abs_deg,
        batch.slip_reward,
        dynamics_count=env.batch_size,
        episodes_per_dynamics=episodes_per_dynamics,
    )
    rewards = batch.rewards
    lengths = batch.lengths
    return EvalResult(metrics=metrics, rewards=rewards, lengths=lengths)


def record_policy_trajectory(
    env: F1TenthAdvWrapper,
    policy: PolicyFn,
    key: jax.Array,
    dynamics_params: jax.Array,
    episode_length: int,
) -> TrajectoryResult:
    """Record a fixed-length eval rollout as arrays for Python-side rendering."""
    state = env.reset(key, dynamics_params)
    active = jnp.ones((env.batch_size,), dtype=bool)

    def freeze_inactive(new_state: AdvEnvState, old_state: AdvEnvState, mask):
        mask_x = mask[:, None]
        mask_history = mask[:, None, None]
        mask_buffer = mask[:, None]
        env_state = old_state.env_state._replace(
            x=jnp.where(mask_x, new_state.env_state.x, old_state.env_state.x),
            last_s=jnp.where(mask, new_state.env_state.last_s, old_state.env_state.last_s),
            step_count=jnp.where(mask, new_state.env_state.step_count, old_state.env_state.step_count),
            steer_buffer=jnp.where(mask_buffer, new_state.env_state.steer_buffer, old_state.env_state.steer_buffer),
        )
        return AdvEnvState(
            env_state=env_state,
            obs=jax.tree_util.tree_map(lambda new, old: jnp.where(mask_x, new, old), new_state.obs, old_state.obs),
            obs_history=jnp.where(mask_history, new_state.obs_history, old_state.obs_history),
            dynamics_params=old_state.dynamics_params,
            obs_delay_steps=old_state.obs_delay_steps,
        )

    def scan_step(carry, _):
        state, active, current_key = carry
        current_key, key_policy, key_env = jax.random.split(current_key, 3)
        action, _ = policy(state.obs["actor_obs"], key_policy)
        next_state, out = env.step(state, action, key_env)
        next_state = freeze_inactive(next_state, state, active)
        reward = jnp.where(active, out.reward, 0.0)
        done = active & out.done
        next_active = active & ~out.done
        return (next_state, next_active, current_key), (next_state.env_state.x, action, reward, done, active)

    (_, _, _), (states, actions, rewards, done, active) = jax.lax.scan(
        scan_step,
        (state, active, key),
        None,
        length=episode_length,
    )
    return TrajectoryResult(states=states, actions=actions, rewards=rewards, done=done, active=active)


class AdvEvaluator:
    """Small evaluator matching the sampler-DR structure from the source repo."""

    def __init__(
        self,
        eval_env: F1TenthAdvWrapper,
        eval_policy_fn: Callable[[object], PolicyFn],
        key: jax.Array,
        *,
        episode_length: int,
    ):
        self.eval_env = eval_env
        self.eval_policy_fn = eval_policy_fn
        self.key = key
        self.episode_length = int(episode_length)
        self.eval_walltime = 0.0
        self.steps_per_unroll = self.episode_length * self.eval_env.batch_size * self.eval_env.action_repeat_steps

    def run_evaluation(
        self,
        policy_params,
        dynamics_params: jax.Array | None = None,
        training_metrics: dict | None = None,
    ) -> tuple[dict, jax.Array, jax.Array]:
        self.key, key = jax.random.split(self.key)
        if dynamics_params is None:
            dynamics_params = self.eval_env.nominal_dynamics_params
        t0 = time.time()
        result = evaluate_policy(
            self.eval_env,
            self.eval_policy_fn(policy_params),
            key,
            dynamics_params,
            self.episode_length,
        )
        result.rewards.block_until_ready()
        elapsed = time.time() - t0
        self.eval_walltime += elapsed
        metrics = {
            "eval/walltime": self.eval_walltime,
            "eval/epoch_eval_time": elapsed,
            "eval/sps": self.steps_per_unroll / max(elapsed, 1e-9),
            **(training_metrics or {}),
            **result.metrics,
        }
        return metrics, result.rewards, result.lengths
