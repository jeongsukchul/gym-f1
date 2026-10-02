"""Adapter from F1TENTH sampler PPO to the vendored GMMVI runtime."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .dual_optimization import (
    cost_beta_from_dual,
    cost_dual_from_beta,
    estimate_kl_to_uniform,
    exponential_moving_average,
    projected_dual_ascent,
    projected_log_dual_ascent,
    reward_beta_from_dual,
    reward_dual_from_beta,
)
from .gmmvi.network import GMMNetwork, GMMTrainingState, create_gmm_network_and_state

GMMVISamplerState = GMMTrainingState


class GMMVISamplerUpdate(NamedTuple):
    state: object
    metrics: dict


class RewardCostGMMVISamplerState(NamedTuple):
    reward_state: GMMVISamplerState
    cost_state: GMMVISamplerState
    reward_dual_lambda: jax.Array
    reward_kl_ema: jax.Array
    cost_dual_lambda: jax.Array
    cost_ema: jax.Array
    num_updates: jax.Array


class BoundedGMMVISampler:
    """Bounded GMMVI sampler backed by the original sample DB/update stack."""

    def __init__(
        self,
        low: jax.Array,
        high: jax.Array,
        *,
        num_components: int = 4,
        init_std: float = 0.1,
        target_beta: float = -2.0,
        num_envs: int = 1,
        batch_size: int | None = None,
    ):
        self.low = jnp.asarray(low, dtype=jnp.float32)
        self.high = jnp.asarray(high, dtype=jnp.float32)
        self.dim = int(self.low.shape[0])
        self.num_components = int(num_components)
        self.init_std = float(init_std)
        self.target_beta = float(target_beta)
        # ``batch_size`` was used by an earlier adapter API.  Keep it as a
        # compatibility alias for tests and downstream callers.
        self.num_envs = int(num_envs if batch_size is None else batch_size)
        self.gmm_network: GMMNetwork | None = None

    def init(self, key: jax.Array, nominal: jax.Array | None = None) -> GMMVISamplerState:
        if nominal is None:
            nominal = 0.5 * (self.low + self.high)
        prior_mean = self.to_latent(jnp.asarray(nominal, dtype=jnp.float32))
        state, network = create_gmm_network_and_state(
            self.dim,
            self.num_envs,
            key,
            prior_mean=prior_mean,
            prior_scale=self.init_std,
            bound_info=(self.low, self.high),
            max_components=self.num_components,
            num_initial_components=self.num_components,
        )
        self.gmm_network = network
        return state

    def to_latent(self, values: jax.Array) -> jax.Array:
        y = 2.0 * (values - self.low) / (self.high - self.low) - 1.0
        y = jnp.clip(y, -1.0 + 1e-6, 1.0 - 1e-6)
        return 0.5 * (jnp.log1p(y) - jnp.log1p(-y))

    def from_latent(self, latent: jax.Array) -> jax.Array:
        return jnp.tanh(latent) * (self.high - self.low) / 2.0 + (self.low + self.high) / 2.0

    def _network(self) -> GMMNetwork:
        if self.gmm_network is None:
            raise RuntimeError("BoundedGMMVISampler.init must be called before sample/update.")
        return self.gmm_network

    def log_prob(self, state: GMMVISamplerState, values: jax.Array) -> jax.Array:
        network = self._network()
        return jax.vmap(network.model.log_density, in_axes=(None, 0))(state.model_state.gmm_state, values)

    def sample(self, state: GMMVISamplerState, key: jax.Array, num_samples: int) -> tuple[jax.Array, jax.Array, jax.Array]:
        network = self._network()
        samples, component_ids = network.model.sample(state.model_state.gmm_state, key, int(num_samples))
        return samples, self.log_prob(state, samples), component_ids

    def update(
        self,
        state: GMMVISamplerState,
        samples: jax.Array,
        scores: jax.Array,
        component_ids: jax.Array | None = None,
        key: jax.Array | None = None,
        target_beta: jax.Array | float | None = None,
    ) -> GMMVISamplerUpdate:
        """Save rollout scores to the GMMVI DB and run the original MORE-style update."""
        if component_ids is None:
            component_ids = jnp.argmax(
                jax.vmap(self._network().model.component_log_densities, in_axes=(None, 0))(
                    state.model_state.gmm_state,
                    samples,
                ),
                axis=-1,
            )
        if key is None:
            key = jax.random.PRNGKey(0)

        network = self._network()
        beta = self.target_beta if target_beta is None else target_beta
        target_logpdf = jnp.asarray(beta, dtype=jnp.float32) * jnp.asarray(scores, dtype=jnp.float32)
        target_grads = jnp.zeros_like(samples)
        sample_db_state = network.sample_selector.save_samples(
            state.model_state,
            state.sample_db_state,
            samples,
            target_logpdf,
            target_grads,
            component_ids,
        )
        state = state._replace(sample_db_state=sample_db_state)
        next_state = self._gmm_update(state, key)
        metrics = self._metrics(next_state, target_logpdf, scores)
        return GMMVISamplerUpdate(next_state, metrics)

    def _gmm_update(self, state: GMMVISamplerState, key: jax.Array) -> GMMVISamplerState:
        network = self._network()
        samples, mapping, sample_dist_densities, target_lnpdfs, target_lnpdf_grads = (
            network.sample_selector.select_train_datas(state.sample_db_state)
        )
        del mapping
        component_stepsizes = network.component_stepsize_fn(state.model_state)
        model_state = network.model.update_stepsizes(state.model_state, component_stepsizes)
        expected_hessian_neg, expected_grad_neg = network.more_ng_estimator(
            model_state,
            samples,
            sample_dist_densities,
            target_lnpdfs,
            target_lnpdf_grads,
        )
        model_state = network.component_updater(
            model_state,
            expected_hessian_neg,
            expected_grad_neg,
            model_state.stepsizes,
        )
        model_state = network.weight_updater(
            model_state,
            samples,
            sample_dist_densities,
            target_lnpdfs,
            state.weight_stepsize,
        )
        model_state, component_adaptation_state, sample_db_state = network.component_adapter(
            state.component_adaptation_state,
            state.sample_db_state,
            model_state,
            state.num_updates + 1,
            key,
        )
        return GMMTrainingState(
            temperature=state.temperature,
            num_updates=state.num_updates + 1,
            model_state=model_state,
            sample_db_state=sample_db_state,
            component_adaptation_state=component_adaptation_state,
            weight_stepsize=state.weight_stepsize,
        )

    def _metrics(self, state: GMMVISamplerState, target_logpdf: jax.Array, scores: jax.Array) -> dict:
        gmm_state = state.model_state.gmm_state
        log_weights = gmm_state.log_weights
        safe_log_weights = jnp.where(gmm_state.component_mask > 0, log_weights, 0.0)
        weights = jnp.exp(safe_log_weights) * gmm_state.component_mask
        chol_diag = jnp.diagonal(gmm_state.chol_covs, axis1=-2, axis2=-1)
        active_chol_diag = chol_diag * gmm_state.component_mask[:, None]
        active_dims = jnp.maximum(jnp.sum(gmm_state.component_mask) * self.dim, 1.0)
        return {
            "sampler/target_logpdf_mean": jnp.mean(target_logpdf),
            "sampler/target_logpdf_min": jnp.min(target_logpdf),
            "sampler/target_logpdf_max": jnp.max(target_logpdf),
            "sampler/std_mean": jnp.sum(active_chol_diag) / active_dims,
            "sampler/entropy_proxy": -jnp.sum(weights * safe_log_weights),
            "sampler/average_component_entropy": self._network().model.average_entropy(gmm_state),
            "sampler/num_components": gmm_state.num_components,
            "sampler/database_samples_written": state.sample_db_state.num_samples_written[0],
            "sampler/rollout_score_mean": jnp.mean(scores),
            "sampler/rollout_score_min": jnp.min(scores),
            "sampler/rollout_score_p25": jnp.percentile(scores, 25.0),
            "sampler/rollout_score_p75": jnp.percentile(scores, 75.0),
            "sampler/rollout_score_max": jnp.max(scores),
        }


class RewardCostGMMVISampler:
    """Two independent GMMVI samplers for low reward and high safety cost.

    The reward inverse temperature is controlled by a KL-to-uniform dual, as
    in fixed-radius risk-sensitive DR.  The cost inverse temperature is
    controlled by the already-defined PPO-Lag episodic budget: violations
    decrease its reciprocal dual and increase beta for adversarial search.
    This is budget-feedback control, not sampler-cost constraint minimization.
    Half of each
    rollout batch comes from each sampler by default.
    """

    def __init__(
        self,
        low: jax.Array,
        high: jax.Array,
        *,
        num_components: int,
        init_std: float,
        num_envs: int,
        reward_fraction: float = 0.5,
        reward_initial_beta: float = -20.0,
        reward_kl_radius: float = 0.1,
        reward_dual_lr: float = 1e-3,
        cost_initial_beta: float = 1.0,
        cost_budget: float = 2.0,
        cost_dual_lr: float = 1e-2,
        dual_ema_decay: float = 0.9,
        dual_lambda_min: float = 1e-3,
        dual_lambda_max: float = 1e3,
        reward_violation_clip: float | None = None,
        cost_violation_clip: float | None = None,
        cost_score_scale: float = 1.0,
        cost_dual_ema_decay: float | None = None,
        cost_dual_update: str = "linear",
    ):
        if num_envs < 2:
            raise ValueError("reward_cost_gmmvi requires at least two environments")
        if not 0.0 <= reward_fraction <= 1.0:
            raise ValueError("reward_fraction must lie between zero and one")
        if reward_initial_beta >= 0.0:
            raise ValueError("reward_initial_beta must be negative")
        if cost_initial_beta <= 0.0:
            raise ValueError("cost_initial_beta must be positive")
        if not 0.0 < cost_score_scale < float("inf"):
            raise ValueError("cost_score_scale must be finite and positive")
        if reward_kl_radius < 0.0 or cost_budget < 0.0:
            raise ValueError("reward_kl_radius and cost_budget must be non-negative")
        if not 0.0 <= dual_ema_decay < 1.0:
            raise ValueError("dual_ema_decay must be in [0, 1)")
        cost_dual_ema_decay = dual_ema_decay if cost_dual_ema_decay is None else cost_dual_ema_decay
        if not 0.0 <= cost_dual_ema_decay < 1.0:
            raise ValueError("cost_dual_ema_decay must be in [0, 1)")
        if dual_lambda_min <= 0.0 or dual_lambda_max <= dual_lambda_min:
            raise ValueError("dual lambda bounds must be positive and ordered")

        self.low = jnp.asarray(low, dtype=jnp.float32)
        self.high = jnp.asarray(high, dtype=jnp.float32)
        self.num_envs = int(num_envs)
        self.reward_num_envs = int(round(self.num_envs * float(reward_fraction)))
        self.cost_num_envs = self.num_envs - self.reward_num_envs
        self.reward_fraction = self.reward_num_envs / float(self.num_envs)
        self.reward_initial_beta = float(reward_initial_beta)
        self.reward_kl_radius = float(reward_kl_radius)
        self.reward_dual_lr = float(reward_dual_lr)
        self.cost_initial_beta = float(cost_initial_beta)
        self.cost_budget = float(cost_budget)
        self.cost_score_scale = float(cost_score_scale)
        self.cost_dual_lr = float(cost_dual_lr)
        if cost_dual_update not in {"linear", "log"}:
            raise ValueError("cost_dual_update must be linear or log")
        self.cost_dual_update = cost_dual_update
        self._cost_dual_ascent = (projected_log_dual_ascent
                                 if cost_dual_update == "log" else projected_dual_ascent)
        self.dual_ema_decay = float(dual_ema_decay)
        self.cost_dual_ema_decay = float(cost_dual_ema_decay)
        self.dual_lambda_min = float(dual_lambda_min)
        self.dual_lambda_max = float(dual_lambda_max)
        self.reward_violation_clip = reward_violation_clip
        self.cost_violation_clip = cost_violation_clip
        self.num_components = int(num_components)

        self.reward_sampler = BoundedGMMVISampler(
            self.low,
            self.high,
            num_components=num_components,
            init_std=init_std,
            target_beta=reward_initial_beta,
            num_envs=max(1, self.reward_num_envs),
        )
        self.cost_sampler = BoundedGMMVISampler(
            self.low,
            self.high,
            num_components=num_components,
            init_std=init_std,
            target_beta=cost_initial_beta,
            num_envs=max(1, self.cost_num_envs),
        )

    def init(self, key: jax.Array, nominal: jax.Array | None = None) -> RewardCostGMMVISamplerState:
        reward_key, cost_key = jax.random.split(key)
        return RewardCostGMMVISamplerState(
            reward_state=self.reward_sampler.init(reward_key, nominal),
            cost_state=self.cost_sampler.init(cost_key, nominal),
            reward_dual_lambda=reward_dual_from_beta(self.reward_initial_beta).astype(jnp.float32),
            reward_kl_ema=jnp.asarray(self.reward_kl_radius, dtype=jnp.float32),
            cost_dual_lambda=cost_dual_from_beta(self.cost_initial_beta).astype(jnp.float32),
            cost_ema=jnp.asarray(self.cost_budget, dtype=jnp.float32),
            num_updates=jnp.asarray(0, dtype=jnp.int32),
        )

    def sample(self, state, key: jax.Array, num_samples: int):
        reward_count = int(round(int(num_samples) * self.reward_fraction))
        cost_count = int(num_samples) - reward_count
        reward_key, cost_key = jax.random.split(key)
        if reward_count == 0:
            samples, log_prob, component_ids = self.cost_sampler.sample(
                state.cost_state, cost_key, cost_count
            )
            return samples, log_prob, component_ids + self.num_components
        if cost_count == 0:
            return self.reward_sampler.sample(state.reward_state, reward_key, reward_count)
        reward_samples, _, reward_ids = self.reward_sampler.sample(
            state.reward_state, reward_key, reward_count
        )
        cost_samples, _, cost_ids = self.cost_sampler.sample(
            state.cost_state, cost_key, cost_count
        )
        samples = jnp.concatenate([reward_samples, cost_samples], axis=0)
        log_prob = self.log_prob(state, samples)
        source_ids = jnp.concatenate(
            [reward_ids, cost_ids + self.num_components],
            axis=0,
        )
        return samples, log_prob, source_ids

    def log_prob(self, state, values: jax.Array) -> jax.Array:
        reward_logq = self.reward_sampler.log_prob(state.reward_state, values)
        cost_logq = self.cost_sampler.log_prob(state.cost_state, values)
        if self.reward_fraction == 1.0:
            return reward_logq
        if self.reward_fraction == 0.0:
            return cost_logq
        mixture_logits = jnp.stack(
            [
                reward_logq + jnp.log(self.reward_fraction),
                cost_logq + jnp.log(1.0 - self.reward_fraction),
            ],
            axis=0,
        )
        return jax.scipy.special.logsumexp(mixture_logits, axis=0)

    @staticmethod
    def _prefixed(metrics: dict, prefix: str) -> dict:
        return {f"sampler/{prefix}_{key.removeprefix('sampler/')}": value for key, value in metrics.items()}

    def _cost_feedback(self, state, cost_mean, constraint_value, valid):
        value = cost_mean if constraint_value is None else constraint_value
        ema = exponential_moving_average(state.cost_ema, value, self.cost_dual_ema_decay)
        dual, violation = self._cost_dual_ascent(
            state.cost_dual_lambda, ema, self.cost_budget, -self.cost_dual_lr,
            self.dual_lambda_min, self.dual_lambda_max, self.cost_violation_clip,
        )
        return (jnp.where(valid, ema, state.cost_ema),
                jnp.where(valid, dual, state.cost_dual_lambda),
                jnp.where(valid, violation, 0.0))

    def update(
        self,
        state: RewardCostGMMVISamplerState,
        samples: jax.Array,
        reward_scores: jax.Array,
        component_ids: jax.Array | None = None,
        key: jax.Array | None = None,
        *,
        cost_scores: jax.Array,
        cost_constraint_value: jax.Array | None = None,
        cost_constraint_valid: jax.Array | bool = True,
    ) -> GMMVISamplerUpdate:
        if key is None:
            key = jax.random.PRNGKey(0)
        if component_ids is None:
            raise ValueError("reward_cost_gmmvi requires source-aware component ids")
        reward_key, cost_key = jax.random.split(key)
        split = self.reward_num_envs

        if self.cost_num_envs == 0:
            reward_beta = reward_beta_from_dual(state.reward_dual_lambda)
            reward_logq = self.reward_sampler.log_prob(state.reward_state, samples)
            reward_kl = estimate_kl_to_uniform(reward_logq, self.low, self.high)
            reward_kl_ema = exponential_moving_average(
                state.reward_kl_ema, reward_kl, self.dual_ema_decay
            )
            reward_dual, reward_violation = projected_dual_ascent(
                state.reward_dual_lambda,
                reward_kl_ema,
                self.reward_kl_radius,
                self.reward_dual_lr,
                self.dual_lambda_min,
                self.dual_lambda_max,
                self.reward_violation_clip,
            )
            reward_update = self.reward_sampler.update(
                state.reward_state,
                samples,
                reward_scores,
                component_ids,
                reward_key,
                target_beta=reward_beta,
            )
            next_state = RewardCostGMMVISamplerState(
                reward_state=reward_update.state,
                cost_state=state.cost_state,
                reward_dual_lambda=reward_dual,
                reward_kl_ema=reward_kl_ema,
                cost_dual_lambda=state.cost_dual_lambda,
                cost_ema=state.cost_ema,
                num_updates=state.num_updates + 1,
            )
            metrics = {
                **reward_update.metrics,
                **self._prefixed(reward_update.metrics, "reward"),
                "sampler/reward_beta": reward_beta_from_dual(reward_dual),
                "sampler/reward_dual_lambda": reward_dual,
                "sampler/reward_kl_to_uniform": reward_kl,
                "sampler/reward_kl_ema": reward_kl_ema,
                "sampler/reward_kl_radius": jnp.asarray(self.reward_kl_radius, dtype=jnp.float32),
                "sampler/reward_kl_violation": reward_violation,
            }
            return GMMVISamplerUpdate(next_state, metrics)

        if self.reward_num_envs == 0:
            cost_values = jnp.asarray(cost_scores, dtype=jnp.float32)
            cost_beta = cost_beta_from_dual(state.cost_dual_lambda)
            cost_mean = jnp.mean(cost_values)
            cost_ema, cost_dual, cost_violation = self._cost_feedback(
                state, cost_mean, cost_constraint_value, cost_constraint_valid,
            )
            cost_update = self.cost_sampler.update(
                state.cost_state,
                samples,
                cost_values * self.cost_score_scale,
                component_ids - self.num_components,
                cost_key,
                target_beta=cost_beta,
            )
            next_state = RewardCostGMMVISamplerState(
                reward_state=state.reward_state,
                cost_state=cost_update.state,
                reward_dual_lambda=state.reward_dual_lambda,
                reward_kl_ema=state.reward_kl_ema,
                cost_dual_lambda=cost_dual,
                cost_ema=cost_ema,
                num_updates=state.num_updates + 1,
            )
            metrics = {
                **cost_update.metrics,
                **self._prefixed(cost_update.metrics, "cost"),
                "sampler/cost_beta": cost_beta_from_dual(cost_dual),
                "sampler/cost_dual_lambda": cost_dual,
                "sampler/cost_beta_relative_change": state.cost_dual_lambda / cost_dual - 1.0,
                "sampler/cost_log_dual_change": jnp.log(cost_dual / state.cost_dual_lambda),
                "sampler/cost_mean": cost_mean,
                "sampler/cost_constraint_value": cost_mean if cost_constraint_value is None else cost_constraint_value,
                "sampler/cost_constraint_valid": jnp.asarray(cost_constraint_valid, dtype=jnp.float32),
                "sampler/cost_ema": cost_ema,
                "sampler/cost_ema_decay": jnp.asarray(self.cost_dual_ema_decay, dtype=jnp.float32),
                "sampler/cost_budget": jnp.asarray(self.cost_budget, dtype=jnp.float32),
                "sampler/cost_budget_violation": cost_violation,
                "sampler/cost_score_scale": jnp.asarray(self.cost_score_scale, dtype=jnp.float32),
                "sampler/cost_scaled_mean": cost_mean * self.cost_score_scale,
                "sampler/cost_scaled_budget": jnp.asarray(self.cost_budget * self.cost_score_scale, dtype=jnp.float32),
                "sampler/cost_target_beta_used": cost_beta,
                "sampler/cost_effective_raw_beta_used": cost_beta * self.cost_score_scale,
            }
            return GMMVISamplerUpdate(next_state, metrics)

        reward_samples = samples[:split]
        cost_samples = samples[split:]
        reward_ids = component_ids[:split]
        cost_ids = component_ids[split:] - self.num_components
        reward_values = jnp.asarray(reward_scores[:split], dtype=jnp.float32)
        cost_values = jnp.asarray(cost_scores[split:], dtype=jnp.float32)

        reward_beta = reward_beta_from_dual(state.reward_dual_lambda)
        cost_beta = cost_beta_from_dual(state.cost_dual_lambda)
        reward_logq = self.reward_sampler.log_prob(state.reward_state, reward_samples)
        cost_logq = self.cost_sampler.log_prob(state.cost_state, cost_samples)
        reward_kl = estimate_kl_to_uniform(reward_logq, self.low, self.high)
        cost_mean = jnp.mean(cost_values)

        reward_kl_ema = exponential_moving_average(state.reward_kl_ema, reward_kl, self.dual_ema_decay)
        reward_dual, reward_violation = projected_dual_ascent(
            state.reward_dual_lambda,
            reward_kl_ema,
            self.reward_kl_radius,
            self.reward_dual_lr,
            self.dual_lambda_min,
            self.dual_lambda_max,
            self.reward_violation_clip,
        )
        cost_ema, cost_dual, cost_violation = self._cost_feedback(
            state, cost_mean, cost_constraint_value, cost_constraint_valid,
        )

        reward_update = self.reward_sampler.update(
            state.reward_state,
            reward_samples,
            reward_values,
            reward_ids,
            reward_key,
            target_beta=reward_beta,
        )
        cost_update = self.cost_sampler.update(
            state.cost_state,
            cost_samples,
            cost_values * self.cost_score_scale,
            cost_ids,
            cost_key,
            target_beta=cost_beta,
        )
        next_state = RewardCostGMMVISamplerState(
            reward_state=reward_update.state,
            cost_state=cost_update.state,
            reward_dual_lambda=reward_dual,
            reward_kl_ema=reward_kl_ema,
            cost_dual_lambda=cost_dual,
            cost_ema=cost_ema,
            num_updates=state.num_updates + 1,
        )
        metrics = {
            **self._prefixed(reward_update.metrics, "reward"),
            **self._prefixed(cost_update.metrics, "cost"),
            "sampler/reward_beta": reward_beta_from_dual(reward_dual),
            "sampler/reward_dual_lambda": reward_dual,
            "sampler/reward_kl_to_uniform": reward_kl,
            "sampler/reward_kl_ema": reward_kl_ema,
            "sampler/reward_kl_radius": jnp.asarray(self.reward_kl_radius, dtype=jnp.float32),
            "sampler/reward_kl_violation": reward_violation,
            "sampler/cost_beta": cost_beta_from_dual(cost_dual),
            "sampler/cost_dual_lambda": cost_dual,
            "sampler/cost_beta_relative_change": state.cost_dual_lambda / cost_dual - 1.0,
            "sampler/cost_log_dual_change": jnp.log(cost_dual / state.cost_dual_lambda),
            "sampler/cost_mean": cost_mean,
            "sampler/cost_constraint_value": cost_mean if cost_constraint_value is None else cost_constraint_value,
            "sampler/cost_constraint_valid": jnp.asarray(cost_constraint_valid, dtype=jnp.float32),
            "sampler/cost_ema": cost_ema,
            "sampler/cost_ema_decay": jnp.asarray(self.cost_dual_ema_decay, dtype=jnp.float32),
            "sampler/cost_budget": jnp.asarray(self.cost_budget, dtype=jnp.float32),
            "sampler/cost_budget_violation": cost_violation,
            "sampler/cost_score_scale": jnp.asarray(self.cost_score_scale, dtype=jnp.float32),
            "sampler/cost_scaled_mean": cost_mean * self.cost_score_scale,
            "sampler/cost_scaled_budget": jnp.asarray(self.cost_budget * self.cost_score_scale, dtype=jnp.float32),
            "sampler/cost_target_beta_used": cost_beta,
            "sampler/cost_effective_raw_beta_used": cost_beta * self.cost_score_scale,
            "sampler/std_mean": 0.5
            * (reward_update.metrics["sampler/std_mean"] + cost_update.metrics["sampler/std_mean"]),
            "sampler/entropy_proxy": 0.5
            * (
                reward_update.metrics["sampler/entropy_proxy"]
                + cost_update.metrics["sampler/entropy_proxy"]
            ),
            "sampler/average_component_entropy": 0.5
            * (
                reward_update.metrics["sampler/average_component_entropy"]
                + cost_update.metrics["sampler/average_component_entropy"]
            ),
            "sampler/num_components": (
                reward_update.metrics["sampler/num_components"]
                + cost_update.metrics["sampler/num_components"]
            ),
            "sampler/database_samples_written": (
                reward_update.metrics["sampler/database_samples_written"]
                + cost_update.metrics["sampler/database_samples_written"]
            ),
            "sampler/rollout_score_mean": jnp.mean(reward_scores),
            "sampler/target_logpdf_mean": 0.5
            * (
                reward_update.metrics["sampler/target_logpdf_mean"]
                + cost_update.metrics["sampler/target_logpdf_mean"]
            ),
            "sampler/target_logpdf_min": jnp.minimum(
                reward_update.metrics["sampler/target_logpdf_min"],
                cost_update.metrics["sampler/target_logpdf_min"],
            ),
            "sampler/target_logpdf_max": jnp.maximum(
                reward_update.metrics["sampler/target_logpdf_max"],
                cost_update.metrics["sampler/target_logpdf_max"],
            ),
            "sampler/rollout_score_min": jnp.min(reward_scores),
            "sampler/rollout_score_p25": jnp.percentile(reward_scores, 25.0),
            "sampler/rollout_score_p75": jnp.percentile(reward_scores, 75.0),
            "sampler/rollout_score_max": jnp.max(reward_scores),
        }
        return GMMVISamplerUpdate(next_state, metrics)
