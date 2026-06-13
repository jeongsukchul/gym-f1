"""Minimal sampler PPO trainer for the F1TENTH JAX environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict, freeze, unfreeze

from .evaluator import generate_adv_unroll
from .gmmvi_sampler import BoundedGMMVISampler
from .losses import compute_gae, flatten_transition, ppo_loss
from .networks import (
    SamplerPPONetworkParams,
    SamplerPPONetworks,
    init_network_params,
    make_inference_fn,
    make_sampler_ppo_networks,
)
from .uniform_sampler import UniformDRSampler
from .wrappers import AdvEnvState, F1TenthAdvWrapper


@dataclass(frozen=True)
class SamplerPPOConfig:
    sampler: str = "uniform"
    num_eval_envs: int = 0
    num_evals: int = 1
    eval_episode_steps: int = 10000
    eval_episodes_per_dynamics: int = 1
    sampler_plot_samples: int = 4096
    sampler_plot_grid: int = 50
    sampler_plot_context_samples: int = 64
    sampler_plot_max_dims: int | None = 5
    sampler_plot_dir: str = "outputs/jax_sampler_ppo/sampler_plots"
    eval_render: bool = False
    eval_render_dir: str = "outputs/jax_sampler_ppo/eval_renders"
    eval_video: bool = False
    eval_video_dir: str = "outputs/jax_sampler_ppo/eval_videos"
    eval_video_fps: int = 30
    eval_video_max_frames: int = 600
    eval_video_final: bool = False
    eval_domain_randomization: bool = False
    eval_video_percentiles: bool = False
    eval_video_percentile_step: int = 5
    learning_rate: float = 3e-4
    end_learning_rate: float | None = None
    learning_rate_transition_steps: int = 1
    total_timesteps: int = 1
    unroll_length: int = 256
    policy_repeat_steps: int = 1
    batch_size: int = 4096
    num_epochs: int = 4
    discounting: float = 0.99
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.2
    entropy_cost: float = 1e-3
    value_cost: float = 0.5
    normalize_advantage: bool = True
    sampler_update_freq: int = 1
    gmm_components: int = 4
    gmm_target_beta: float = -2.0
    gmm_init_std: float = 0.1
    policy_hidden_layer_sizes: tuple[int, ...] = (64, 64)
    value_hidden_layer_sizes: tuple[int, ...] = (64, 64)
    init_log_std: float = -1.0
    end_log_std: float | None = None


@flax.struct.dataclass
class SamplerPPOTrainingState:
    params: SamplerPPONetworkParams
    opt_state: optax.OptState
    sampler_state: Any
    env_state: AdvEnvState
    key: jax.Array
    env_steps: jax.Array
    update_steps: jax.Array


class SamplerPPOTrainer:
    def __init__(
        self,
        env: F1TenthAdvWrapper,
        config: SamplerPPOConfig | None = None,
        networks: SamplerPPONetworks | None = None,
    ):
        self.env = env
        self.config = config or SamplerPPOConfig()
        if self.config.policy_repeat_steps < 1:
            raise ValueError(f"policy_repeat_steps must be >= 1, got {self.config.policy_repeat_steps}")
        if self.env.action_repeat_steps != self.config.policy_repeat_steps:
            raise ValueError(
                f"env action_repeat_steps ({self.env.action_repeat_steps}) must match "
                f"policy_repeat_steps ({self.config.policy_repeat_steps})"
            )
        self.networks = networks or make_sampler_ppo_networks(
            env.observation_size,
            env.action_size,
            value_observation_size=env.value_observation_size,
            policy_hidden_layer_sizes=self.config.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=self.config.value_hidden_layer_sizes,
            init_log_std=self.config.init_log_std,
        )
        if self.config.end_learning_rate is None:
            learning_rate = self.config.learning_rate
        else:
            learning_rate = optax.linear_schedule(
                init_value=self.config.learning_rate,
                end_value=self.config.end_learning_rate,
                transition_steps=max(1, self.config.learning_rate_transition_steps),
            )
        self.optimizer = optax.adam(learning_rate)
        sampler_name = self.config.sampler.lower()
        if sampler_name in {"uniform", "udr", "uniform_dr"}:
            self.sampler = UniformDRSampler(env.domain_spec.low, env.domain_spec.high)
        elif sampler_name in {"gmmvi", "gmm"}:
            self.sampler = BoundedGMMVISampler(
                env.domain_spec.low,
                env.domain_spec.high,
                num_components=self.config.gmm_components,
                init_std=self.config.gmm_init_std,
                target_beta=self.config.gmm_target_beta,
                num_envs=env.batch_size,
                batch_size=self.config.batch_size,
            )
        else:
            raise ValueError("sampler must be one of: uniform, gmmvi")

    def init_state(self, key: jax.Array) -> SamplerPPOTrainingState:
        key_net, key_sampler, key_params, key_reset, key = jax.random.split(key, 5)
        print("obs size : ", self.env.observation_size)
        print("value obs size : ", self.env.value_observation_size)
        params = init_network_params(self.networks, key_net, self.env.observation_size, self.env.value_observation_size)
        opt_state = self.optimizer.init(params)
        sampler_state = self.sampler.init(key_sampler, self.env.domain_spec.nominal_vector)
        dynamics_params, _, _ = self.sampler.sample(sampler_state, key_params, self.env.batch_size)
        env_state = self.env.reset(key_reset, dynamics_params)
        return SamplerPPOTrainingState(
            params=params,
            opt_state=opt_state,
            sampler_state=sampler_state,
            env_state=env_state,
            key=key,
            env_steps=jnp.asarray(0, dtype=jnp.int32),
            update_steps=jnp.asarray(0, dtype=jnp.int32),
        )

    def make_policy(self, params: SamplerPPONetworkParams, *, deterministic: bool = False):
        return make_inference_fn(self.networks)(params, deterministic=deterministic)

    def training_step(self, state: SamplerPPOTrainingState) -> tuple[SamplerPPOTrainingState, dict]:
        key, key_sample, key_reset, key_rollout, key_update = jax.random.split(state.key, 5)
        dynamics_params, sampler_log_prob, component_ids = self.sampler.sample(
            state.sampler_state,
            key_sample,
            self.env.batch_size,
        )
        env_state = self.env.reset(key_reset, dynamics_params)
        final_env_state, data = generate_adv_unroll(
            self.env,
            env_state,
            dynamics_params,
            self.make_policy(state.params),
            key_rollout,
            self.config.unroll_length,
        )

        values = self.networks.value_network.apply(state.params.value, data.observation["value_obs"])
        bootstrap_value = self.networks.value_network.apply(state.params.value, final_env_state.obs["value_obs"])
        targets, advantages = compute_gae(
            data.reward,
            values,
            bootstrap_value,
            data.discount,
            data.extras["state_extras"]["truncation"],
            gae_lambda=self.config.gae_lambda,
            discounting=self.config.discounting,
        )
        flat_data = flatten_transition(data)
        flat_targets = jnp.reshape(targets, (-1,))
        flat_advantages = jnp.reshape(advantages, (-1,))

        num_samples = flat_targets.shape[0]
        minibatch_size = min(int(self.config.batch_size), int(num_samples))
        num_minibatches = max(1, int(num_samples) // minibatch_size)
        num_used_samples = num_minibatches * minibatch_size
        key_epoch_root, key_sampler_update, key_update = jax.random.split(key_update, 3)
        epoch_keys = jax.random.split(key_epoch_root, self.config.num_epochs)

        def update_epoch(carry, epoch_key):
            params, opt_state = carry

            perm = jax.random.permutation(epoch_key, num_samples)[:num_used_samples]
            minibatches = jnp.reshape(perm, (num_minibatches, minibatch_size))

            def update_minibatch(carry, indices):
                params, opt_state = carry
                batch_data = jax.tree_util.tree_map(lambda x: x[indices], flat_data)
                batch_targets = flat_targets[indices]
                batch_advantages = flat_advantages[indices]

                def loss_fn(p):
                    return ppo_loss(
                        p,
                        self.networks,
                        batch_data,
                        batch_targets,
                        batch_advantages,
                        clipping_epsilon=self.config.clipping_epsilon,
                        entropy_cost=self.config.entropy_cost,
                        value_cost=self.config.value_cost,
                        normalize_advantage=self.config.normalize_advantage,
                    )

                (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                metrics = {**metrics, "loss/epoch_total": loss}
                return (params, opt_state), metrics

            return jax.lax.scan(update_minibatch, (params, opt_state), minibatches)

        (params, opt_state), epoch_metrics = jax.lax.scan(
            update_epoch,
            (state.params, state.opt_state),
            epoch_keys,
        )
        metrics = jax.tree_util.tree_map(lambda x: x[-1, -1], epoch_metrics)
        rollout_returns = jnp.sum(jnp.maximum(data.reward, 0), axis=0)
        sampler_update = self.sampler.update(
            state.sampler_state,
            dynamics_params,
            rollout_returns / float(self.config.unroll_length * self.config.policy_repeat_steps) * 10,
            component_ids,
            key_sampler_update,
        )
        should_update_sampler = (state.update_steps % self.config.sampler_update_freq) == 0
        sampler_state = jax.lax.cond(
            should_update_sampler,
            lambda _: sampler_update.state,
            lambda _: state.sampler_state,
            operand=None,
        )
        sampler_metrics = jax.tree_util.tree_map(
            lambda x: jnp.where(should_update_sampler, x, jnp.zeros_like(x)),
            sampler_update.metrics,
        )

        next_env_steps = state.env_steps + self.config.unroll_length * self.env.batch_size
        params = self._apply_log_std_schedule(params, next_env_steps)

        next_state = SamplerPPOTrainingState(
            params=params,
            opt_state=opt_state,
            sampler_state=sampler_state,
            env_state=final_env_state,
            key=key_update,
            env_steps=next_env_steps,
            update_steps=state.update_steps + 1,
        )
        metrics = {
            **metrics,
            **sampler_metrics,
            "train/env_steps": next_state.env_steps,
            "train/sim_steps": next_state.env_steps * self.config.policy_repeat_steps,
            "train/policy_repeat_steps": jnp.asarray(self.config.policy_repeat_steps, dtype=jnp.float32),
            "train/update_steps": next_state.update_steps,
            "train/rollout_return_mean": jnp.mean(rollout_returns),
            "train/rollout_return_min": jnp.min(rollout_returns),
            "train/rollout_return_max": jnp.max(rollout_returns),
            "train/path_reward_mean": jnp.mean(data.extras["state_extras"]["path_reward"]),
            "train/slip_angle_abs_deg_mean": jnp.mean(data.extras["state_extras"]["slip_angle_abs_deg"]),
            "train/slip_reward_raw_mean": jnp.mean(data.extras["state_extras"]["slip_reward_raw"]),
            "train/slip_reward_mean": jnp.mean(data.extras["state_extras"]["slip_reward"]),
            "sampler/log_prob_mean": jnp.mean(sampler_log_prob),
        }
        return next_state, metrics

    def _apply_log_std_schedule(self, params: SamplerPPONetworkParams, env_steps: jax.Array) -> SamplerPPONetworkParams:
        if self.config.end_log_std is None:
            return params
        progress = jnp.clip(env_steps.astype(jnp.float32) / float(max(1, self.config.total_timesteps)), 0.0, 1.0)
        log_std = self.config.init_log_std + progress * (self.config.end_log_std - self.config.init_log_std)
        policy = unfreeze(params.policy)
        policy["params"]["log_std"] = jnp.full_like(policy["params"]["log_std"], log_std)
        policy = freeze(policy) if isinstance(params.policy, FrozenDict) else policy
        return params.replace(policy=policy)


def train(
    env: F1TenthAdvWrapper,
    *,
    num_updates: int,
    key: jax.Array,
    config: SamplerPPOConfig | None = None,
    metrics_callback: Callable[[int, dict], None] | None = None,
    jit: bool = True,
) -> SamplerPPOTrainingState:
    trainer = SamplerPPOTrainer(env, config=config)
    state = trainer.init_state(key)
    step_fn = jax.jit(trainer.training_step) if jit else trainer.training_step
    for update_idx in range(int(num_updates)):
        state, metrics = step_fn(state)
        if metrics_callback is not None:
            metrics_callback(update_idx, metrics)
    return state
