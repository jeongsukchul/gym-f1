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


def ppo_lagrange_loss(
    params: SamplerPPONetworkParams,
    networks: SamplerPPONetworks,
    data: TransitionWithParams,
    targets: jax.Array,
    advantages: jax.Array,
    cost_targets: jax.Array,
    cost_advantages: jax.Array,
    lambda_lagr: jax.Array,
    *,
    clipping_epsilon: float = 0.2,
    entropy_cost: float = 1e-3,
    value_cost: float = 0.5,
    normalize_advantage: bool = True,
    normalize_cost_advantage: bool | None = None,
    cost_advantage_std_floor: float = 0.0,
) -> tuple[jax.Array, dict]:
    """CRAX PPO-Lagrange objective for one minibatch.

    The policy surrogate uses ``A_r - lambda * A_c`` and trains separate
    reward and cost critics.  The multiplier update is performed by the
    trainer after the PPO update, matching CRAX's ``post_step_fn``.
    """
    reward_advantage_std = advantages.std()
    cost_advantage_std = cost_advantages.std()
    if normalize_cost_advantage is None:
        normalize_cost_advantage = normalize_advantage
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    cost_normalization_scale = jnp.asarray(1., dtype=cost_advantages.dtype)
    if normalize_cost_advantage:
        cost_normalization_scale = jnp.maximum(cost_advantage_std, cost_advantage_std_floor) + 1e-8
        cost_advantages = (cost_advantages - cost_advantages.mean()) / cost_normalization_scale

    modified_advantages = advantages - lambda_lagr * cost_advantages
    mean, log_std = networks.policy_network.apply(params.policy, data.observation["actor_obs"])
    new_log_prob = normal_tanh_log_prob(mean, log_std, data.raw_action)
    ratio = jnp.exp(new_log_prob - data.log_prob)
    surrogate1 = ratio * modified_advantages
    surrogate2 = jnp.clip(ratio, 1.0 - clipping_epsilon, 1.0 + clipping_epsilon) * modified_advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    values = networks.value_network.apply(params.value, data.observation["value_obs"])
    cost_values = networks.cost_value_network.apply(params.cost_value, data.observation["value_obs"])
    value_error = targets - values
    cost_value_error = cost_targets - cost_values
    value_loss = 0.5 * jnp.mean(value_error**2)
    cost_value_loss = 0.5 * jnp.mean(cost_value_error**2)
    entropy_loss = -entropy_cost * normal_entropy(log_std)
    total_loss = policy_loss + value_cost * value_loss + value_cost * cost_value_loss + entropy_loss

    costs = data.extras["state_extras"]["cost"]
    return total_loss, {
        "loss/total": total_loss,
        "loss/policy": policy_loss,
        "loss/value": value_loss,
        "loss/cost_value": cost_value_loss,
        "loss/entropy": entropy_loss,
        "constraint/mean_cost": jnp.mean(costs),
        "constraint/reward_advantage_std_raw": reward_advantage_std,
        "constraint/cost_advantage_std_raw": cost_advantage_std,
        "constraint/cost_normalization_scale": cost_normalization_scale,
        "constraint/normalize_cost_advantage": jnp.asarray(normalize_cost_advantage, dtype=jnp.float32),
        "constraint/cost_value_mean": jnp.mean(cost_values),
        "constraint/cost_target_mean": jnp.mean(cost_targets),
        "constraint/cost_target_std": jnp.std(cost_targets),
        "constraint/cost_value_explained_variance": 1.0
        - jnp.var(cost_value_error) / (jnp.var(cost_targets) + 1e-8),
        "train/approx_kl": jnp.mean(data.log_prob - new_log_prob),
        "train/clip_fraction": jnp.mean((jnp.abs(ratio - 1.0) > clipping_epsilon).astype(jnp.float32)),
    }
