"""Uniform domain-randomization sampler for the JAX PPO trainer."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class UniformDRSamplerState(NamedTuple):
    num_updates: jax.Array


class UniformDRSamplerUpdate(NamedTuple):
    state: UniformDRSamplerState
    metrics: dict


class UniformDRSampler:
    """Samples dynamics parameters uniformly inside the configured DR box."""

    def __init__(self, low: jax.Array, high: jax.Array):
        self.low = jnp.asarray(low, dtype=jnp.float32)
        self.high = jnp.asarray(high, dtype=jnp.float32)
        self.dim = int(self.low.shape[0])
        self._log_prob = -jnp.sum(jnp.log(self.high - self.low))
        self._std_mean = jnp.mean((self.high - self.low) / jnp.sqrt(12.0))

    def init(self, key: jax.Array, nominal: jax.Array | None = None) -> UniformDRSamplerState:
        del key, nominal
        return UniformDRSamplerState(num_updates=jnp.asarray(0, dtype=jnp.int32))

    def sample(
        self,
        state: UniformDRSamplerState,
        key: jax.Array,
        num_samples: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del state
        samples = jax.random.uniform(
            key,
            shape=(int(num_samples), self.dim),
            minval=self.low,
            maxval=self.high,
        )
        log_prob = jnp.full((int(num_samples),), self._log_prob, dtype=jnp.float32)
        component_ids = jnp.zeros((int(num_samples),), dtype=jnp.int32)
        return samples, log_prob, component_ids

    def log_prob(self, state: UniformDRSamplerState, values: jax.Array) -> jax.Array:
        del state
        return jnp.full((values.shape[0],), self._log_prob, dtype=jnp.float32)

    def update(
        self,
        state: UniformDRSamplerState,
        samples: jax.Array,
        scores: jax.Array,
        component_ids: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> UniformDRSamplerUpdate:
        del component_ids, key
        next_state = UniformDRSamplerState(num_updates=state.num_updates + 1)
        metrics = {
            "sampler/target_logpdf_mean": jnp.asarray(0.0, dtype=jnp.float32),
            "sampler/target_logpdf_min": jnp.asarray(0.0, dtype=jnp.float32),
            "sampler/target_logpdf_max": jnp.asarray(0.0, dtype=jnp.float32),
            "sampler/std_mean": self._std_mean,
            "sampler/entropy_proxy": -self._log_prob,
            "sampler/average_component_entropy": -self._log_prob,
            "sampler/num_components": jnp.asarray(0, dtype=jnp.int32),
            "sampler/database_samples_written": jnp.asarray(samples.shape[0], dtype=jnp.int32),
            "sampler/rollout_score_mean": jnp.mean(scores),
            "sampler/rollout_score_min": jnp.min(scores),
            "sampler/rollout_score_p25": jnp.percentile(scores, 25.0),
            "sampler/rollout_score_p75": jnp.percentile(scores, 75.0),
            "sampler/rollout_score_max": jnp.max(scores),
        }
        return UniformDRSamplerUpdate(next_state, metrics)
