"""PPO losses for the F1TENTH JAX sampler PPO trainer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .networks import SamplerPPONetworkParams, SamplerPPONetworks, normal_entropy, normal_tanh_log_prob
from .wrappers import TransitionWithParams


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_value: jax.Array,
    discounts: jax.Array,
    truncation: jax.Array,
    *,
    gae_lambda: float = 0.95,
    discounting: float = 0.99,
) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation for arrays shaped ``[T, B]``."""
    truncation_mask = 1.0 - truncation.astype(jnp.float32)
    values_t_plus_1 = jnp.concatenate([values[1:], bootstrap_value[None, :]], axis=0)
    termination = 1.0 - discounts
    deltas = rewards + discounting * (1.0 - termination) * values_t_plus_1 - values
    deltas = deltas * truncation_mask

    def scan_step(acc, inputs):
        delta, discount, trunc = inputs
        acc = delta + discounting * discount * trunc * gae_lambda * acc
        return acc, acc

    _, advantages = jax.lax.scan(
        scan_step,
        jnp.zeros_like(bootstrap_value),
        (deltas, discounts, truncation_mask),
        reverse=True,
    )
    targets = advantages + values
    return jax.lax.stop_gradient(targets), jax.lax.stop_gradient(advantages)


def flatten_transition(data: TransitionWithParams) -> TransitionWithParams:
    return jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)


def ppo_loss(
    params: SamplerPPONetworkParams,
    networks: SamplerPPONetworks,
    data: TransitionWithParams,
    targets: jax.Array,
    advantages: jax.Array,
    *,
    clipping_epsilon: float = 0.2,
    entropy_cost: float = 1e-3,
    value_cost: float = 0.5,
    normalize_advantage: bool = True,
) -> tuple[jax.Array, dict]:
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    mean, log_std = networks.policy_network.apply(params.policy, data.observation["actor_obs"])
    new_log_prob = normal_tanh_log_prob(mean, log_std, data.raw_action)
    ratio = jnp.exp(new_log_prob - data.log_prob)
    surrogate1 = ratio * advantages
    surrogate2 = jnp.clip(ratio, 1.0 - clipping_epsilon, 1.0 + clipping_epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    values = networks.value_network.apply(params.value, data.observation["value_obs"])
    value_loss = 0.5 * jnp.mean((targets - values) ** 2)
    entropy_loss = -entropy_cost * normal_entropy(log_std)
    total_loss = policy_loss + value_cost * value_loss + entropy_loss

    metrics = {
        "loss/total": total_loss,
        "loss/policy": policy_loss,
        "loss/value": value_loss,
        "loss/entropy": entropy_loss,
        "train/approx_kl": jnp.mean(data.log_prob - new_log_prob),
        "train/clip_fraction": jnp.mean((jnp.abs(ratio - 1.0) > clipping_epsilon).astype(jnp.float32)),
    }
    return total_loss, metrics
