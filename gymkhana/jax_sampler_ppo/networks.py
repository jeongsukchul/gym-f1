"""Flax policy/value networks for sampler PPO."""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import flax
import jax
import jax.numpy as jnp
from flax import linen as nn


@flax.struct.dataclass
class SamplerPPONetworkParams:
    policy: object
    value: object
    cost_value: object


class GaussianPolicy(nn.Module):
    action_size: int
    hidden_layer_sizes: Sequence[int] = (64, 64)
    activation: Callable = nn.tanh
    init_log_std: float = -1.0
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, obs):
        x = obs
        for size in self.hidden_layer_sizes:
            x = nn.Dense(size)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation(x)
        mean = nn.Dense(self.action_size)(x)
        log_std = self.param(
            "log_std",
            lambda key, shape: jnp.full(shape, self.init_log_std, dtype=jnp.float32),
            (self.action_size,),
        )
        log_std = jnp.clip(log_std, -5.0, 2.0)
        return mean, log_std


class ValueNetwork(nn.Module):
    hidden_layer_sizes: Sequence[int] = (64, 64)
    activation: Callable = nn.tanh
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, obs):
        x = obs
        for size in self.hidden_layer_sizes:
            x = nn.Dense(size)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation(x)
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


class SamplerPPONetworks(NamedTuple):
    policy_network: GaussianPolicy
    value_network: ValueNetwork
    cost_value_network: ValueNetwork


def normal_tanh_log_prob(mean: jax.Array, log_std: jax.Array, raw_action: jax.Array) -> jax.Array:
    std = jnp.exp(log_std)
    z = (raw_action - mean) / std
    normal_log_prob = -0.5 * jnp.sum(z**2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1)
    squashed = jnp.tanh(raw_action)
    correction = jnp.sum(jnp.log(jnp.clip(1.0 - squashed**2, 1e-6)), axis=-1)
    return normal_log_prob - correction


def normal_entropy(log_std: jax.Array) -> jax.Array:
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


def sample_action(
    network: GaussianPolicy,
    params,
    obs: jax.Array,
    key: jax.Array,
    *,
    deterministic: bool = False,
) -> tuple[jax.Array, dict]:
    mean, log_std = network.apply(params, obs)
    raw_action = jnp.where(
        deterministic,
        mean,
        mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape),
    )
    action = jnp.tanh(raw_action)
    log_prob = normal_tanh_log_prob(mean, log_std, raw_action)
    return action, {"raw_action": raw_action, "log_prob": log_prob}


def make_sampler_ppo_networks(
    observation_size: int,
    action_size: int,
    *,
    value_observation_size: int | None = None,
    policy_hidden_layer_sizes: Sequence[int] = (64, 64),
    value_hidden_layer_sizes: Sequence[int] = (64, 64),
    cost_value_hidden_layer_sizes: Sequence[int] | None = None,
    policy_use_layer_norm: bool = False,
    value_use_layer_norm: bool = False,
    init_log_std: float = -1.0,
) -> SamplerPPONetworks:
    del observation_size, value_observation_size
    if cost_value_hidden_layer_sizes is None:
        cost_value_hidden_layer_sizes = value_hidden_layer_sizes
    return SamplerPPONetworks(
        policy_network=GaussianPolicy(
            action_size=action_size,
            hidden_layer_sizes=tuple(policy_hidden_layer_sizes),
            use_layer_norm=policy_use_layer_norm,
            init_log_std=init_log_std,
        ),
        value_network=ValueNetwork(
            hidden_layer_sizes=tuple(value_hidden_layer_sizes),
            use_layer_norm=value_use_layer_norm,
        ),
        cost_value_network=ValueNetwork(
            hidden_layer_sizes=tuple(cost_value_hidden_layer_sizes),
            use_layer_norm=value_use_layer_norm,
        ),
    )


def init_network_params(
    networks: SamplerPPONetworks,
    key: jax.Array,
    actor_observation_size: int,
    value_observation_size: int | None = None,
) -> SamplerPPONetworkParams:
    key_policy, key_value, key_cost_value = jax.random.split(key, 3)
    if value_observation_size is None:
        value_observation_size = actor_observation_size
    dummy_actor_obs = jnp.zeros((1, actor_observation_size), dtype=jnp.float32)
    dummy_value_obs = jnp.zeros((1, value_observation_size), dtype=jnp.float32)
    return SamplerPPONetworkParams(
        policy=networks.policy_network.init(key_policy, dummy_actor_obs),
        value=networks.value_network.init(key_value, dummy_value_obs),
        cost_value=networks.cost_value_network.init(key_cost_value, dummy_value_obs),
    )


def make_inference_fn(networks: SamplerPPONetworks):
    def make_policy(params: SamplerPPONetworkParams, deterministic: bool = False):
        def policy(obs: jax.Array, key: jax.Array):
            return sample_action(networks.policy_network, params.policy, obs, key, deterministic=deterministic)

        return policy

    return make_policy
