"""Minimal sampler PPO trainer for the F1TENTH JAX environment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Callable
import warnings

import flax
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict, freeze, unfreeze

from .evaluator import generate_adv_unroll
from .lagrange import completed_episode_costs, first_episode_costs, lagrange_cost_estimate
from .losses import compute_gae, flatten_transition, ppo_lagrange_loss, ppo_loss
from .networks import (
    SamplerPPONetworkParams,
    SamplerPPONetworks,
    init_network_params,
    make_inference_fn,
    make_sampler_ppo_networks,
)
from .uniform_sampler import UniformDRSampler
from .wrappers import AdvEnvState, F1TenthAdvWrapper

if TYPE_CHECKING:
    from .gmmvi_sampler import BoundedGMMVISampler


@dataclass(frozen=True)
class SamplerPPOConfig:
    sampler: str = "uniform"
    domain_randomization: bool = True
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
    reset_state_on_rollout: bool = True  # Legacy behavior; false resamples only DR.
    policy_repeat_steps: int = 1
    batch_size: int = 4096
    num_epochs: int = 4
    discounting: float = 0.99
    cost_discounting: float | None = None  # None preserves the reward discount.
    cost_terminal_at_time_limit: bool = False  # Legacy mask vs finite-episode cost objective.
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.2
    entropy_cost: float = 1e-3
    value_cost: float = 0.5
    normalize_advantage: bool = True
    normalize_cost_advantage: bool | None = None  # Legacy: follows reward normalization.
    cost_advantage_std_floor: float = 0.0  # Zero preserves legacy standardization.
    sampler_update_freq: int = 1
    gmm_components: int = 4
    gmm_target_beta: float = -2.0
    gmm_init_std: float = 0.1
    gmm_reward_fraction: float = 0.5
    gmm_reward_kl_radius: float = 0.1
    gmm_reward_dual_lr: float = 1e-3
    gmm_cost_initial_beta: float = 1.0
    gmm_cost_dual_lr: float = 1e-2
    gmm_cost_dual_update: str = "linear"
    gmm_cost_score_scale: float = 1.0
    gmm_dual_ema_decay: float = 0.9
    gmm_cost_dual_ema_decay: float | None = None
    gmm_dual_lambda_min: float = 1e-3
    gmm_dual_lambda_max: float = 1e3
    gmm_reward_violation_clip: float | None = None
    gmm_cost_violation_clip: float | None = None
    policy_hidden_layer_sizes: tuple[int, ...] = (64, 64)
    value_hidden_layer_sizes: tuple[int, ...] = (64, 64)
    cost_value_hidden_layer_sizes: tuple[int, ...] = (256, 256, 256, 256, 256)
    policy_use_layer_norm: bool = False
    value_use_layer_norm: bool = False
    init_log_std: float = -1.0
    end_log_std: float | None = None
    # CRAX PPO-Lagrange.  ``safety_bound`` is an episodic cumulative edge-cost
    # budget and is converted to a policy-step budget before lambda updates.
    use_ppo_lag: bool = False
    safety_bound: float = 0.0
    lagrangian_coef_rate: float = 0.01
    initial_lambda_lagr: float = 0.0
    constraint_cost_type: str = "edge"
    lagrangian_update_mode: str = "per_step"
    allow_partial_first_episode: bool = False  # Explicit short-window ablation.
    lagrangian_ema_decay: float = 0.0
    lagrangian_max: float = 100.0
    lagrangian_min_completed_episodes: int = 1
    lagrangian_warmup_steps: int = 0


@flax.struct.dataclass
class SamplerPPOTrainingState:
    params: SamplerPPONetworkParams
    opt_state: optax.OptState
    sampler_state: Any
    env_state: AdvEnvState
    key: jax.Array
    env_steps: jax.Array
    update_steps: jax.Array
    lambda_lagr: jax.Array
    lagrangian_cost_ema: jax.Array
    episode_cost_accumulator: jax.Array
    lagrangian_pending_cost_sum: jax.Array
    lagrangian_pending_episode_count: jax.Array


class SamplerPPOTrainer:
    def __init__(
        self,
        env: F1TenthAdvWrapper,
        config: SamplerPPOConfig | None = None,
        networks: SamplerPPONetworks | None = None,
    ):
        self.env = env
        self.config = config or SamplerPPOConfig()
        self.cost_discounting = (self.config.discounting if self.config.cost_discounting is None
                                 else self.config.cost_discounting)
        if not 0.0 < self.cost_discounting <= 1.0:
            raise ValueError("cost_discounting must be in (0, 1]")
        if not math.isfinite(self.config.cost_advantage_std_floor) or self.config.cost_advantage_std_floor < 0:
            raise ValueError("cost_advantage_std_floor must be finite and non-negative")
        if not self.config.domain_randomization and self.config.sampler.lower() not in {"uniform", "udr", "uniform_dr"}:
            raise ValueError("Non-DR training requires sampler=uniform; adaptive domain samplers are disabled")
        if self.config.lagrangian_update_mode not in {"per_step", "rollout", "first_episode", "completed_episode"}:
            raise ValueError("Invalid lagrangian_update_mode")
        if self.config.lagrangian_update_mode == "completed_episode" and self.config.reset_state_on_rollout:
            raise ValueError("completed_episode requires reset_state_on_rollout=false")
        if self.config.lagrangian_min_completed_episodes < 1 or self.config.lagrangian_warmup_steps < 0:
            raise ValueError("Completed episode batch must be positive and warmup non-negative")
        if not 0.0 <= self.config.lagrangian_ema_decay < 1.0:
            raise ValueError("lagrangian_ema_decay must be in [0, 1)")
        if self.config.lagrangian_max <= 0 or self.config.lagrangian_coef_rate < 0:
            raise ValueError("Lagrange maximum must be positive and rate non-negative")
        if self.config.lagrangian_update_mode == "first_episode":
            policy_horizon = (env.env.max_episode_steps + self.config.policy_repeat_steps - 1) // self.config.policy_repeat_steps
            if self.config.unroll_length < policy_horizon:
                if not self.config.allow_partial_first_episode:
                    raise ValueError("first_episode requires a rollout covering the episode horizon; "
                                     "allow_partial_first_episode is an explicit short-window ablation")
                warnings.warn("Partial first_episode: unfinished episodes contribute their observed "
                              "window cost, including zeros. Budget feedback is NOT a full-episode "
                              "collision-probability estimate.", stacklevel=2)
        if self.config.policy_repeat_steps < 1:
            raise ValueError(f"policy_repeat_steps must be >= 1, got {self.config.policy_repeat_steps}")
        if self.env.action_repeat_steps != self.config.policy_repeat_steps:
            raise ValueError(
                f"env action_repeat_steps ({self.env.action_repeat_steps}) must match "
                f"policy_repeat_steps ({self.config.policy_repeat_steps})"
            )
        if self.env.constraint_cost_type != self.config.constraint_cost_type:
            raise ValueError(
                f"env constraint_cost_type ({self.env.constraint_cost_type}) must match "
                f"config constraint_cost_type ({self.config.constraint_cost_type})"
            )
        self.networks = networks or make_sampler_ppo_networks(
            env.observation_size,
            env.action_size,
            value_observation_size=env.value_observation_size,
            policy_hidden_layer_sizes=self.config.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=self.config.value_hidden_layer_sizes,
            cost_value_hidden_layer_sizes=(
                self.config.cost_value_hidden_layer_sizes
                if self.config.use_ppo_lag
                else self.config.value_hidden_layer_sizes
            ),
            policy_use_layer_norm=self.config.policy_use_layer_norm,
            value_use_layer_norm=self.config.value_use_layer_norm,
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
            from .gmmvi_sampler import BoundedGMMVISampler

            self.sampler = BoundedGMMVISampler(
                env.domain_spec.low,
                env.domain_spec.high,
                num_components=self.config.gmm_components,
                init_std=self.config.gmm_init_std,
                target_beta=self.config.gmm_target_beta,
                num_envs=int(0.95 * env.batch_size),
            )
        elif sampler_name in {"reward_cost_gmmvi", "dual_gmmvi", "rc_gmmvi"}:
            from .gmmvi_sampler import RewardCostGMMVISampler

            if self.config.safety_bound <= 0.0:
                raise ValueError("reward_cost_gmmvi requires a positive safety_bound")
            policy_episode_steps = max(
                1,
                (self.env.env.max_episode_steps + self.config.policy_repeat_steps - 1)
                // self.config.policy_repeat_steps,
            )
            rollout_budget = self.config.safety_bound * self.config.unroll_length / float(policy_episode_steps)
            self.sampler = RewardCostGMMVISampler(
                env.domain_spec.low,
                env.domain_spec.high,
                num_components=self.config.gmm_components,
                init_std=self.config.gmm_init_std,
                num_envs=env.batch_size,
                reward_fraction=self.config.gmm_reward_fraction,
                reward_initial_beta=self.config.gmm_target_beta,
                reward_kl_radius=self.config.gmm_reward_kl_radius,
                reward_dual_lr=self.config.gmm_reward_dual_lr,
                cost_initial_beta=self.config.gmm_cost_initial_beta,
                cost_budget=(self.config.safety_bound if self.config.lagrangian_update_mode
                             in {"first_episode", "completed_episode"} else rollout_budget),
                cost_dual_lr=self.config.gmm_cost_dual_lr,
                cost_dual_update=self.config.gmm_cost_dual_update,
                cost_score_scale=self.config.gmm_cost_score_scale,
                dual_ema_decay=self.config.gmm_dual_ema_decay,
                cost_dual_ema_decay=self.config.gmm_cost_dual_ema_decay,
                dual_lambda_min=self.config.gmm_dual_lambda_min,
                dual_lambda_max=self.config.gmm_dual_lambda_max,
                reward_violation_clip=self.config.gmm_reward_violation_clip,
                cost_violation_clip=self.config.gmm_cost_violation_clip,
            )
        else:
            raise ValueError("sampler must be one of: uniform, gmmvi, reward_cost_gmmvi")

    def _sample_training_dynamics(self, sampler_state, key):
        if self.config.domain_randomization:
            return self.sampler.sample(sampler_state, key, self.env.batch_size)
        return (self.env.nominal_dynamics_params,
                jnp.zeros((self.env.batch_size,), dtype=jnp.float32),
                jnp.zeros((self.env.batch_size,), dtype=jnp.int32))

    def init_state(self, key: jax.Array) -> SamplerPPOTrainingState:
        key_net, key_sampler, key_params, key_reset, key = jax.random.split(key, 5)
        print("obs size : ", self.env.observation_size)
        print("value obs size : ", self.env.value_observation_size)
        params = init_network_params(self.networks, key_net, self.env.observation_size, self.env.value_observation_size)
        opt_state = self.optimizer.init(params)
        sampler_state = self.sampler.init(key_sampler, self.env.domain_spec.nominal_vector)
        dynamics_params, _, _ = self._sample_training_dynamics(sampler_state, key_params)
        env_state = self.env.reset(key_reset, dynamics_params)
        return SamplerPPOTrainingState(
            params=params,
            opt_state=opt_state,
            sampler_state=sampler_state,
            env_state=env_state,
            key=key,
            env_steps=jnp.asarray(0, dtype=jnp.int32),
            update_steps=jnp.asarray(0, dtype=jnp.int32),
            lambda_lagr=jnp.asarray(self.config.initial_lambda_lagr, dtype=jnp.float32),
            lagrangian_cost_ema=jnp.asarray(
                self.config.safety_bound / float(max(1, (self.env.env.max_episode_steps + self.config.policy_repeat_steps - 1) // self.config.policy_repeat_steps))
                if self.config.lagrangian_update_mode == "per_step" else self.config.safety_bound,
                dtype=jnp.float32,
            ),
            episode_cost_accumulator=jnp.zeros((self.env.batch_size,), dtype=jnp.float32),
            lagrangian_pending_cost_sum=jnp.asarray(0.0, dtype=jnp.float32),
            lagrangian_pending_episode_count=jnp.asarray(0.0, dtype=jnp.float32),
        )

    def make_policy(self, params: SamplerPPONetworkParams, *, deterministic: bool = False):
        return make_inference_fn(self.networks)(params, deterministic=deterministic)

    def training_step(self, state: SamplerPPOTrainingState) -> tuple[SamplerPPOTrainingState, dict]:
        key, key_sample, key_reset, key_rollout, key_update = jax.random.split(state.key, 5)
        dynamics_params, sampler_log_prob, component_ids = self._sample_training_dynamics(
            state.sampler_state, key_sample,
        )
        env_state = (self.env.reset(key_reset, dynamics_params)
                     if self.config.reset_state_on_rollout
                     else self.env.with_dynamics_params(state.env_state, dynamics_params))
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
        if self.config.use_ppo_lag:
            cost_values = self.networks.cost_value_network.apply(state.params.cost_value, data.observation["value_obs"])
            bootstrap_cost_value = self.networks.cost_value_network.apply(
                state.params.cost_value, final_env_state.obs["value_obs"]
            )
            cost_truncation = data.extras["state_extras"]["truncation"]
            if self.config.cost_terminal_at_time_limit:
                # The constraint budget ends at the time limit. Keep done=1,
                # but do not erase the zero-future-cost terminal TD residual.
                cost_truncation = jnp.zeros_like(cost_truncation)
            cost_targets, cost_advantages = compute_gae(
                data.extras["state_extras"]["cost"],
                cost_values,
                bootstrap_cost_value,
                data.discount,
                cost_truncation,
                gae_lambda=self.config.gae_lambda,
                discounting=self.cost_discounting,
            )
            flat_cost_targets = jnp.reshape(cost_targets, (-1,))
            flat_cost_advantages = jnp.reshape(cost_advantages, (-1,))

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
                    if self.config.use_ppo_lag:
                        batch_cost_targets = flat_cost_targets[indices]
                        batch_cost_advantages = flat_cost_advantages[indices]
                        return ppo_lagrange_loss(
                            p,
                            self.networks,
                            batch_data,
                            batch_targets,
                            batch_advantages,
                            batch_cost_targets,
                            batch_cost_advantages,
                            state.lambda_lagr,
                            clipping_epsilon=self.config.clipping_epsilon,
                            entropy_cost=self.config.entropy_cost,
                            value_cost=self.config.value_cost,
                            normalize_advantage=self.config.normalize_advantage,
                            normalize_cost_advantage=self.config.normalize_cost_advantage,
                            cost_advantage_std_floor=self.config.cost_advantage_std_floor,
                        )
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
        rollout_mean_cost = jnp.mean(data.extras["state_extras"]["cost"])
        episode_cost_accumulator, completed_cost_sums, completed_counts = completed_episode_costs(
            data.extras["state_extras"]["cost"], data.done,
            jnp.zeros_like(state.episode_cost_accumulator) if self.config.reset_state_on_rollout
            else state.episode_cost_accumulator,
        )
        completed_count = jnp.sum(completed_counts)
        completed_cost_mean = jnp.sum(completed_cost_sums) / jnp.maximum(completed_count, 1.0)
        pending_cost = state.lagrangian_pending_cost_sum + jnp.sum(completed_cost_sums)
        pending_count = state.lagrangian_pending_episode_count + completed_count
        batched_episode_mode = self.config.lagrangian_update_mode == "completed_episode"
        batch_ready = pending_count >= self.config.lagrangian_min_completed_episodes
        feedback_valid = batch_ready if batched_episode_mode else jnp.asarray(True)
        feedback_valid = feedback_valid & (state.env_steps >= self.config.lagrangian_warmup_steps)
        feedback_episode_count = pending_count if batched_episode_mode else completed_count
        if self.config.use_ppo_lag:
            policy_episode_steps = max(
                1, (self.env.env.max_episode_steps + self.config.policy_repeat_steps - 1) // self.config.policy_repeat_steps
            )
            per_step_safety_bound = self.config.safety_bound / float(policy_episode_steps)
            # Cost and budget must share units. The legacy mode averages per
            # step; first_episode excludes collisions after early auto-resets.
            if self.config.lagrangian_update_mode == "completed_episode":
                dual_cost = pending_cost / jnp.maximum(pending_count, 1.0)
                dual_budget = self.config.safety_bound
            else:
                dual_cost, dual_budget = lagrange_cost_estimate(
                    data.extras["state_extras"]["cost"], data.done,
                    mode=self.config.lagrangian_update_mode,
                    episode_steps=policy_episode_steps, budget=self.config.safety_bound,
                )
            next_cost_ema = self.config.lagrangian_ema_decay * state.lagrangian_cost_ema + (1.0 - self.config.lagrangian_ema_decay) * dual_cost
            cost_ema = jnp.where(feedback_valid, next_cost_ema, state.lagrangian_cost_ema)
            cost_violation = cost_ema - dual_budget
            lambda_lagr = jnp.clip(
                state.lambda_lagr + self.config.lagrangian_coef_rate * cost_violation,
                0.0, self.config.lagrangian_max,
            )
            lambda_lagr = jnp.where(feedback_valid, lambda_lagr, state.lambda_lagr)
            metrics = {
                **metrics,
                "constraint/mean_cost": rollout_mean_cost,
                "constraint/lambda_lagr": lambda_lagr,
                "constraint/cost_violation": cost_violation,
                "constraint/dual_cost": dual_cost,
                "constraint/dual_cost_ema": cost_ema,
                "constraint/dual_budget": jnp.asarray(dual_budget, dtype=jnp.float32),
                "constraint/per_step_safety_bound": jnp.asarray(per_step_safety_bound, dtype=jnp.float32),
            }
        else:
            lambda_lagr = state.lambda_lagr
            cost_ema = state.lagrangian_cost_ema
        raw_rollout_returns = jnp.sum(data.reward, axis=0)
        rollout_returns = jnp.sum(jnp.maximum(data.reward, 0), axis=0)
        rollout_costs = jnp.sum(data.extras["state_extras"]["cost"], axis=0)
        rollout_edge_costs = jnp.sum(data.extras["state_extras"]["edge_cost"], axis=0)
        rollout_collision_counts = jnp.sum(data.extras["state_extras"]["collision_cost"], axis=0)
        episode_costs = first_episode_costs(data.extras["state_extras"]["cost"], data.done)
        sampler_costs = episode_costs if self.config.lagrangian_update_mode == "first_episode" else rollout_costs
        normalized_returns = rollout_returns / float(self.config.unroll_length * self.config.policy_repeat_steps) * 10
        if self.config.sampler.lower() in {"reward_cost_gmmvi", "dual_gmmvi", "rc_gmmvi"}:
            sampler_feedback = {}
            if self.config.lagrangian_update_mode == "completed_episode":
                split = self.sampler.reward_num_envs
                cost_count = jnp.sum(completed_counts[split:])
                sampler_feedback = dict(
                    cost_constraint_value=jnp.sum(completed_cost_sums[split:]) / jnp.maximum(cost_count, 1.0),
                    cost_constraint_valid=cost_count > 0,
                )
            sampler_update = self.sampler.update(
                state.sampler_state,
                dynamics_params,
                normalized_returns,
                component_ids,
                key_sampler_update,
                cost_scores=sampler_costs,
                **sampler_feedback,
            )
        else:
            topk = int(0.95 * self.env.batch_size)
            score, indices = jax.lax.top_k(rollout_returns, topk)
            sampler_update = self.sampler.update(
                state.sampler_state,
                dynamics_params[indices],
                score / float(self.config.unroll_length * self.config.policy_repeat_steps) * 10,
                component_ids[indices],
                key_sampler_update,
            )
        should_update_sampler = jnp.asarray(self.config.domain_randomization) & (
            (state.update_steps % self.config.sampler_update_freq) == 0)
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
            lambda_lagr=lambda_lagr,
            lagrangian_cost_ema=cost_ema,
            episode_cost_accumulator=episode_cost_accumulator,
            # Flush completed batches even during warmup: do not feed stale warmup policy data later.
            lagrangian_pending_cost_sum=jnp.where(batch_ready, 0.0, pending_cost),
            lagrangian_pending_episode_count=jnp.where(batch_ready, 0.0, pending_count),
        )
        policy_episode_steps_for_budget = max(
            1,
            (self.env.env.max_episode_steps + self.config.policy_repeat_steps - 1)
            // self.config.policy_repeat_steps,
        )
        rollout_budget = (
            self.config.safety_bound * self.config.unroll_length / float(policy_episode_steps_for_budget)
        )
        rollout_cost_violation = rollout_costs - rollout_budget
        canonical_metrics = {
            "training/constraint/cost_terminal_at_time_limit": jnp.asarray(self.config.cost_terminal_at_time_limit, dtype=jnp.float32),
            "training/advantage/reward_std_raw": metrics.get("constraint/reward_advantage_std_raw", jnp.asarray(0.)),
            "training/advantage/cost_std_raw": metrics.get("constraint/cost_advantage_std_raw", jnp.asarray(0.)),
            "training/advantage/normalize_cost": metrics.get("constraint/normalize_cost_advantage", jnp.asarray(0.)),
            "training/advantage/cost_normalization_scale": metrics.get("constraint/cost_normalization_scale", jnp.asarray(1.)),
            "training/advantage/cost_std_floor": jnp.asarray(self.config.cost_advantage_std_floor),
            "training/progress/env_steps": next_state.env_steps,
            "training/progress/sim_steps": next_state.env_steps * self.config.policy_repeat_steps,
            "training/progress/update_steps": next_state.update_steps,
            "training/dynamics/domain_randomization": jnp.asarray(self.config.domain_randomization, dtype=jnp.float32),
            "training/episode/completed_count": completed_count,
            "training/episode/completed_cost_mean": completed_cost_mean,
            "training/episode/cost_feedback_valid": feedback_valid.astype(jnp.float32),
            "training/constraint/feedback_episode_count": feedback_episode_count,
            "training/constraint/pending_episode_count": next_state.lagrangian_pending_episode_count,
            "training/episode/partial_cost_mean": jnp.mean(episode_cost_accumulator),
            "training/progress/reset_state_on_rollout": jnp.asarray(self.config.reset_state_on_rollout, dtype=jnp.float32),
            "training/constraint/partial_first_episode": jnp.asarray(
                self.config.lagrangian_update_mode == "first_episode"
                and self.config.allow_partial_first_episode, dtype=jnp.float32,
            ),
            "training/progress/policy_repeat_steps": jnp.asarray(
                self.config.policy_repeat_steps, dtype=jnp.float32
            ),
            "training/reward/rollout_return_mean": jnp.mean(raw_rollout_returns),
            "training/reward/rollout_return_min": jnp.min(raw_rollout_returns),
            "training/reward/rollout_return_max": jnp.max(raw_rollout_returns),
            "training/reward/rollout_positive_return_mean": jnp.mean(rollout_returns),
            "training/reward/path_reward_step_mean": jnp.mean(data.extras["state_extras"]["path_reward"]),
            "training/cost/rollout_cost_mean": jnp.mean(rollout_costs),
            "training/cost/rollout_cost_min": jnp.min(rollout_costs),
            "training/cost/rollout_cost_max": jnp.max(rollout_costs),
            "training/cost/step_cost_mean": rollout_mean_cost,
            "training/cost/first_episode_cost_mean": jnp.mean(episode_costs),
            "training/cost/first_episode_signed_violation": jnp.mean(episode_costs) - self.config.safety_bound,
            "training/cost/episode_budget": jnp.asarray(self.config.safety_bound, dtype=jnp.float32),
            "training/cost/rollout_budget": jnp.asarray(rollout_budget, dtype=jnp.float32),
            "training/cost/signed_violation_mean": jnp.mean(rollout_cost_violation),
            "training/cost/positive_violation_mean": jnp.mean(jnp.maximum(rollout_cost_violation, 0.0)),
            "training/cost/violation_rate": jnp.mean((rollout_cost_violation > 0.0).astype(jnp.float32)),
            "training/cost/satisfied_rate": jnp.mean((rollout_cost_violation <= 0.0).astype(jnp.float32)),
            "training/cost/mean_budget_margin": rollout_budget - jnp.mean(rollout_costs),
            "training/cost/mean_constraint_satisfied": (
                jnp.mean(rollout_costs) <= rollout_budget
            ).astype(jnp.float32),
            "training/edge/rollout_penalty_mean": jnp.mean(rollout_edge_costs),
            "training/edge/rollout_penalty_max": jnp.max(rollout_edge_costs),
            "training/edge/step_penalty_mean": jnp.mean(data.extras["state_extras"]["edge_cost"]),
            "training/collision/rollout_count_mean": jnp.mean(rollout_collision_counts),
            "training/collision/rollout_count_max": jnp.max(rollout_collision_counts),
            "training/collision/rollout_rate": jnp.mean(
                (rollout_collision_counts > 0.0).astype(jnp.float32)
            ),
            "training/collision/collision_free_rollout_rate": jnp.mean(
                (rollout_collision_counts == 0.0).astype(jnp.float32)
            ),
            "training/behavior/slip_angle_abs_deg_mean": jnp.mean(
                data.extras["state_extras"]["slip_angle_abs_deg"]
            ),
            "training/behavior/slip_reward_raw_mean": jnp.mean(
                data.extras["state_extras"]["slip_reward_raw"]
            ),
            "training/behavior/slip_reward_mean": jnp.mean(data.extras["state_extras"]["slip_reward"]),
            "training/optimization/loss_total": metrics["loss/total"],
            "training/optimization/loss_epoch_total": metrics["loss/epoch_total"],
            "training/optimization/loss_policy": metrics["loss/policy"],
            "training/optimization/loss_value": metrics["loss/value"],
            "training/optimization/loss_entropy": metrics["loss/entropy"],
            "training/optimization/approx_kl": metrics["train/approx_kl"],
            "training/optimization/clip_fraction": metrics["train/clip_fraction"],
            **{f"training/sampler/{key.removeprefix('sampler/')}": value for key, value in sampler_metrics.items()},
            "training/sampler/log_prob_mean": jnp.mean(sampler_log_prob),
        }
        if self.config.lagrangian_update_mode == "completed_episode":
            # These legacy metrics only describe a partial window, not an episode.
            canonical_metrics.pop("training/cost/first_episode_cost_mean")
            canonical_metrics.pop("training/cost/first_episode_signed_violation")
            canonical_metrics["training/episode/completed_budget_violation"] = jnp.where(
                completed_count > 0, completed_cost_mean - self.config.safety_bound, 0.0,
            )
        if self.config.use_ppo_lag:
            canonical_metrics.update(
                {
                    "training/constraint/lambda_lagr": lambda_lagr,
                    "training/constraint/per_step_budget": metrics["constraint/per_step_safety_bound"],
                    "training/constraint/signed_step_violation": rollout_mean_cost - metrics["constraint/per_step_safety_bound"],
                    "training/constraint/dual_cost": metrics["constraint/dual_cost"],
                    "training/constraint/dual_cost_ema": metrics["constraint/dual_cost_ema"],
                    "training/constraint/dual_budget": metrics["constraint/dual_budget"],
                    "training/constraint/dual_violation": metrics["constraint/cost_violation"],
                    "training/critic/cost_value_loss": metrics["loss/cost_value"],
                    "training/critic/cost_value_mean": metrics["constraint/cost_value_mean"],
                    "training/critic/cost_target_mean": metrics["constraint/cost_target_mean"],
                    "training/critic/cost_target_std": metrics["constraint/cost_target_std"],
                    "training/critic/cost_value_explained_variance": metrics[
                        "constraint/cost_value_explained_variance"
                    ],
                }
            )
        metrics = {
            **metrics,
            **sampler_metrics,
            **canonical_metrics,
            "train/env_steps": next_state.env_steps,
            "train/sim_steps": next_state.env_steps * self.config.policy_repeat_steps,
            "train/policy_repeat_steps": jnp.asarray(self.config.policy_repeat_steps, dtype=jnp.float32),
            "train/update_steps": next_state.update_steps,
            "train/rollout_return_mean": jnp.mean(rollout_returns),
            "train/rollout_return_min": jnp.min(rollout_returns),
            "train/rollout_return_max": jnp.max(rollout_returns),
            "train/rollout_cost_mean": jnp.mean(rollout_costs),
            "train/rollout_cost_min": jnp.min(rollout_costs),
            "train/rollout_cost_max": jnp.max(rollout_costs),
            "train/path_reward_mean": jnp.mean(data.extras["state_extras"]["path_reward"]),
            "train/slip_angle_abs_deg_mean": jnp.mean(data.extras["state_extras"]["slip_angle_abs_deg"]),
            "train/slip_reward_raw_mean": jnp.mean(data.extras["state_extras"]["slip_reward_raw"]),
            "train/slip_reward_mean": jnp.mean(data.extras["state_extras"]["slip_reward"]),
            "train/edge_cost_mean": jnp.mean(data.extras["state_extras"]["edge_cost"]),
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
