"""Adapter from F1TENTH sampler PPO to the vendored GMMVI runtime."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .gmmvi.network import GMMNetwork, GMMTrainingState, create_gmm_network_and_state

GMMVISamplerState = GMMTrainingState


class GMMVISamplerUpdate(NamedTuple):
    state: GMMVISamplerState
    metrics: dict


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
        self.num_envs = int(num_envs)
        self.batch_size = int(batch_size or num_envs)
        self.gmm_network: GMMNetwork | None = None

    def init(self, key: jax.Array, nominal: jax.Array | None = None) -> GMMVISamplerState:
        if nominal is None:
            nominal = 0.5 * (self.low + self.high)
        prior_mean = self.to_latent(jnp.asarray(nominal, dtype=jnp.float32))
        state, network = create_gmm_network_and_state(
            self.dim,
            self.num_envs,
            self.batch_size,
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
        target_logpdf = self.target_beta * jnp.asarray(scores, dtype=jnp.float32)
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
