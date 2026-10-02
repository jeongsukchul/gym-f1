"""F1TENTH JAX wrappers for adversarial domain-randomized training."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from gymkhana.jax_env import EnvState, JaxRaceEnv, StepOutput

from .domain import F1TenthDomainSpec, params_from_vector, sample_uniform


class AdvEnvState(NamedTuple):
    env_state: EnvState
    obs: dict[str, jax.Array]
    obs_history: jax.Array
    dynamics_params: jax.Array
    obs_delay_steps: jax.Array


class TransitionWithParams(NamedTuple):
    observation: dict[str, jax.Array]
    dynamics_params: jax.Array
    action: jax.Array
    raw_action: jax.Array
    log_prob: jax.Array
    reward: jax.Array
    discount: jax.Array
    next_observation: dict[str, jax.Array]
    done: jax.Array
    extras: dict[str, Any]


class F1TenthAdvWrapper:
    """Batched adversarial DR wrapper for :class:`JaxRaceEnv`."""

    def __init__(
        self,
        env: JaxRaceEnv,
        domain_spec: F1TenthDomainSpec,
        *,
        auto_reset: bool = True,
        obs_history_len: int = 1,
        obs_delay_min_steps: int = 0,
        obs_delay_max_steps: int | None = None,
        asymmetric_critic: bool = False,
        action_repeat_steps: int = 1,
        constraint_cost_type: str = "edge",
    ):
        self.env = env
        self.domain_spec = domain_spec
        self.auto_reset = bool(auto_reset)
        self.obs_history_len = int(obs_history_len)
        self.obs_delay_min_steps = int(obs_delay_min_steps)
        self.obs_delay_max_steps = (
            int(obs_delay_max_steps) if obs_delay_max_steps is not None else self.obs_delay_min_steps
        )
        self.obs_history_buffer_len = self.obs_history_len + self.obs_delay_max_steps
        self.asymmetric_critic = bool(asymmetric_critic)
        self.action_repeat_steps = int(action_repeat_steps)
        self.constraint_cost_type = str(constraint_cost_type).lower()
        if self.obs_history_len < 1:
            raise ValueError(f"obs_history_len must be >= 1, got {obs_history_len}")
        if self.obs_delay_min_steps < 0:
            raise ValueError(f"obs_delay_min_steps must be >= 0, got {obs_delay_min_steps}")
        if self.obs_delay_max_steps < self.obs_delay_min_steps:
            raise ValueError(
                f"obs_delay_max_steps must be >= obs_delay_min_steps, got "
                f"{obs_delay_max_steps} < {obs_delay_min_steps}"
            )
        if self.action_repeat_steps < 1:
            raise ValueError(f"action_repeat_steps must be >= 1, got {action_repeat_steps}")
        if self.constraint_cost_type not in {"edge", "collision"}:
            raise ValueError(
                "constraint_cost_type must be one of: edge, collision; "
                f"got {constraint_cost_type!r}"
            )
        self.batch_size = env.batch_size
        self.base_observation_size = 7 + env.lookahead_n_points + (2 if env.sparse_width_obs else env.lookahead_n_points)
        self.actor_observation_size = self.base_observation_size * self.obs_history_len
        self.value_observation_size = (
            self.actor_observation_size + domain_spec.size if self.asymmetric_critic else self.actor_observation_size
        )
        self.observation_size = self.actor_observation_size
        self.action_size = 2
        self.dynamics_param_size = domain_spec.size

    @property
    def constraint_cost_metric(self) -> str:
        return "collision_cost" if self.constraint_cost_type == "collision" else "edge_cost"

    def constraint_cost(self, metrics: dict[str, Any], like: jax.Array) -> jax.Array:
        """Return the configured PPO-Lagrange cost from environment metrics."""
        return metrics.get(self.constraint_cost_metric, jnp.zeros_like(like))

    @property
    def nominal_dynamics_params(self) -> jax.Array:
        return jnp.broadcast_to(self.domain_spec.nominal_vector, (self.batch_size, self.dynamics_param_size))

    def sample_uniform_params(self, key: jax.Array) -> jax.Array:
        return sample_uniform(self.domain_spec, key, self.batch_size)

    def reset(self, key: jax.Array, dynamics_params: jax.Array | None = None) -> AdvEnvState:
        key_reset, key_params, key_delay = jax.random.split(key, 3)
        env_state, base_obs = self.env.reset(key_reset)
        if dynamics_params is None:
            dynamics_params = sample_uniform(self.domain_spec, key_params, self.batch_size)
        obs_delay_steps = self._sample_obs_delay_steps(key_delay)
        obs_history = self._reset_history(base_obs)
        obs = self._make_obs(obs_history, dynamics_params, obs_delay_steps)
        return AdvEnvState(
            env_state=env_state,
            obs=obs,
            obs_history=obs_history,
            dynamics_params=dynamics_params,
            obs_delay_steps=obs_delay_steps,
        )

    def with_dynamics_params(self, state: AdvEnvState, dynamics_params: jax.Array) -> AdvEnvState:
        return AdvEnvState(
            env_state=state.env_state,
            obs=self._make_obs(state.obs_history, dynamics_params, state.obs_delay_steps),
            obs_history=state.obs_history,
            dynamics_params=dynamics_params,
            obs_delay_steps=state.obs_delay_steps,
        )

    def step(self, state: AdvEnvState, action: jax.Array, key: jax.Array | None = None) -> tuple[AdvEnvState, Any]:
        params = params_from_vector(self.domain_spec, state.dynamics_params)
        key_env = key_delay = key
        if key is not None:
            key_env, key_delay = jax.random.split(key)
        if self.action_repeat_steps == 1:
            out = self.env.step(
                state.env_state,
                action,
                key=key_env,
                auto_reset=self.auto_reset,
                params=params,
            )
        else:
            out = self._repeat_env_step(state.env_state, action, key_env, params)
        next_state = self._wrap_step_output(state, out, reset_done_history=key is not None, key=key_delay)
        return next_state, out

    def _repeat_env_step(self, env_state: EnvState, action: jax.Array, key: jax.Array | None, params) -> StepOutput:
        keys = jax.random.split(key, self.action_repeat_steps) if key is not None else [None] * self.action_repeat_steps
        active = jnp.ones((self.batch_size,), dtype=bool)
        reward = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        done = jnp.zeros((self.batch_size,), dtype=bool)
        count = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        boundary = jnp.zeros((self.batch_size,), dtype=bool)
        terminal_boundary = jnp.zeros((self.batch_size,), dtype=bool)
        truncated = jnp.zeros((self.batch_size,), dtype=bool)
        zero_bool = jnp.zeros((self.batch_size,), dtype=bool)
        zero_metric = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        s = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        ey = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        path_reward = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        edge_proximity = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        edge_cost = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        collision_cost = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        slip_angle_abs_deg = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        slip_reward_raw = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        slip_reward = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        obs = None

        for repeat_idx in range(self.action_repeat_steps):
            out = self.env.step(
                env_state,
                action,
                key=keys[repeat_idx],
                auto_reset=self.auto_reset,
                params=params,
            )
            env_state = self._where_env_state(active, out.state, env_state)
            obs = out.obs if obs is None else self._where_batch(active, out.obs, obs)

            active_f = active.astype(jnp.float32)
            reward = reward + out.reward * active_f
            count = count + active_f
            done = done | (active & out.done)
            s = jnp.where(active, out.metrics.get("s", zero_metric), s)
            ey = jnp.where(active, out.metrics.get("ey", zero_metric), ey)
            boundary = boundary | (active & out.metrics.get("boundary", zero_bool))
            terminal_boundary = terminal_boundary | (active & out.metrics.get("terminal_boundary", zero_bool))
            truncated = truncated | (active & out.metrics.get("truncated", zero_bool))
            path_reward = path_reward + out.metrics.get("path_reward", zero_metric) * active_f
            edge_proximity = edge_proximity + out.metrics.get("edge_proximity", zero_metric) * active_f
            edge_cost = edge_cost + out.metrics.get("edge_cost", zero_metric) * active_f
            collision_cost = collision_cost + out.metrics.get("collision_cost", zero_metric) * active_f
            slip_angle_abs_deg = slip_angle_abs_deg + out.metrics.get("slip_angle_abs_deg", zero_metric) * active_f
            slip_reward_raw = slip_reward_raw + out.metrics.get("slip_reward_raw", zero_metric) * active_f
            slip_reward = slip_reward + out.metrics.get("slip_reward", zero_metric) * active_f
            active = active & ~out.done

        safe_count = jnp.maximum(count, 1.0)
        return StepOutput(
            state=env_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics={
                "s": s,
                "ey": ey,
                "boundary": boundary,
                "terminal_boundary": terminal_boundary,
                "truncated": truncated,
                "path_reward": path_reward,
                "edge_proximity": edge_proximity / safe_count,
                "edge_cost": edge_cost,
                "collision_cost": collision_cost,
                "slip_angle_abs_deg": slip_angle_abs_deg / safe_count,
                "slip_reward_raw": slip_reward_raw / safe_count,
                "slip_reward": slip_reward / safe_count,
            },
        )

    def _wrap_step_output(
        self,
        state: AdvEnvState,
        out: StepOutput,
        *,
        reset_done_history: bool,
        key: jax.Array | None,
    ) -> AdvEnvState:
        obs_history = self._append_history(state.obs_history, out.obs)
        obs_delay_steps = state.obs_delay_steps
        if self.auto_reset and reset_done_history:
            reset_history = self._reset_history(out.obs)
            obs_history = jnp.where(out.done[:, None, None], reset_history, obs_history)
            if key is not None:
                reset_delay_steps = self._sample_obs_delay_steps(key)
                obs_delay_steps = jnp.where(out.done, reset_delay_steps, obs_delay_steps)
        obs = self._make_obs(obs_history, state.dynamics_params, obs_delay_steps)
        return AdvEnvState(
            env_state=out.state,
            obs=obs,
            obs_history=obs_history,
            dynamics_params=state.dynamics_params,
            obs_delay_steps=obs_delay_steps,
        )

    @staticmethod
    def _where_batch(mask: jax.Array, new_value: jax.Array, old_value: jax.Array) -> jax.Array:
        while mask.ndim < new_value.ndim:
            mask = mask[..., None]
        return jnp.where(mask, new_value, old_value)

    def _where_env_state(self, mask: jax.Array, new_state: EnvState, old_state: EnvState) -> EnvState:
        return jax.tree_util.tree_map(lambda new, old: self._where_batch(mask, new, old), new_state, old_state)

    def _reset_history(self, obs: jax.Array) -> jax.Array:
        return jnp.repeat(obs[:, None, :], self.obs_history_buffer_len, axis=1)

    def _sample_obs_delay_steps(self, key: jax.Array) -> jax.Array:
        if self.obs_delay_min_steps == self.obs_delay_max_steps:
            return jnp.full((self.batch_size,), self.obs_delay_min_steps, dtype=jnp.int32)
        return jax.random.randint(
            key,
            shape=(self.batch_size,),
            minval=self.obs_delay_min_steps,
            maxval=self.obs_delay_max_steps + 1,
            dtype=jnp.int32,
        )

    @staticmethod
    def _append_history(obs_history: jax.Array, obs: jax.Array) -> jax.Array:
        return jnp.concatenate([obs[:, None, :], obs_history[:, :-1, :]], axis=1)

    @staticmethod
    def _flatten_history(obs_history: jax.Array) -> jax.Array:
        return jnp.reshape(obs_history, (obs_history.shape[0], -1))

    def _make_obs(
        self,
        obs_history: jax.Array,
        dynamics_params: jax.Array,
        obs_delay_steps: jax.Array,
    ) -> dict[str, jax.Array]:
        obs_delay_steps = jnp.asarray(obs_delay_steps, dtype=jnp.int32)
        obs_delay_steps = jnp.clip(obs_delay_steps, 0, self.obs_delay_max_steps)
        obs_delay_steps = jnp.broadcast_to(obs_delay_steps, (obs_history.shape[0],))
        history_offsets = jnp.arange(self.obs_history_len, dtype=jnp.int32)[None, :]
        history_indices = obs_delay_steps[:, None] + history_offsets
        batch_indices = jnp.arange(obs_history.shape[0])[:, None]
        delayed_history = obs_history[batch_indices, history_indices, :]
        actor_obs = self._flatten_history(delayed_history)
        value_obs = jnp.concatenate([actor_obs, dynamics_params], axis=-1) if self.asymmetric_critic else actor_obs
        return {"actor_obs": actor_obs, "value_obs": value_obs}
