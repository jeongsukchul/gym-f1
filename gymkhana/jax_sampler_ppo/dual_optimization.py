"""JAX-safe dual updates for adaptive risk-sensitive GMM targets.

The reward sampler constrains its KL distance to uniform.  The cost sampler
uses the PPO-Lag episodic safety budget as its constraint.  Their inverse
temperatures have opposite signs because reward GMMVI searches for low-return
domains while cost GMMVI searches for high-cost domains.
"""

from __future__ import annotations

import jax.numpy as jnp


def exponential_moving_average(previous_value, current_value, decay):
    decay = jnp.asarray(decay, dtype=jnp.float32)
    return decay * jnp.asarray(previous_value) + (1.0 - decay) * jnp.asarray(current_value)


def clipped_constraint_violation(value, limit, max_abs_violation=None):
    """Returns ``value - limit``, optionally with symmetric clipping."""
    violation = jnp.asarray(value) - jnp.asarray(limit)
    if max_abs_violation is not None:
        clip = jnp.asarray(max_abs_violation)
        violation = jnp.clip(violation, -clip, clip)
    return violation


def estimate_kl_to_uniform(log_q, low, high):
    """Monte-Carlo estimate of KL(q || Uniform([low, high]))."""
    log_volume = jnp.sum(jnp.log(jnp.asarray(high) - jnp.asarray(low)))
    return jnp.mean(jnp.asarray(log_q)) + log_volume


def projected_dual_ascent(
    dual_lambda,
    constraint_value,
    constraint_limit,
    learning_rate,
    min_dual,
    max_dual,
    max_abs_violation=None,
):
    """Projected ascent for a positive dual using ``value - limit``."""
    violation = clipped_constraint_violation(
        constraint_value,
        constraint_limit,
        max_abs_violation,
    )
    updated = jnp.clip(
        jnp.asarray(dual_lambda) + jnp.asarray(learning_rate) * violation,
        jnp.asarray(min_dual),
        jnp.asarray(max_dual),
    )
    return updated, violation


def projected_log_dual_ascent(
    dual_lambda, constraint_value, constraint_limit, learning_rate,
    min_dual, max_dual, max_abs_violation=None,
):
    """Multiplicative (mirror) ascent, not Euclidean ascent in log coordinates.

    Away from bounds, beta_new / beta_old = exp(-lr * violation).
    The original budget and positive reciprocal-beta relation are preserved.
    """
    violation = clipped_constraint_violation(
        constraint_value, constraint_limit, max_abs_violation,
    )
    log_min, log_max = jnp.log(min_dual), jnp.log(max_dual)
    log_dual = jnp.log(jnp.clip(jnp.asarray(dual_lambda), min_dual, max_dual))
    updated = jnp.clip(jnp.exp(jnp.clip(
        log_dual + jnp.asarray(learning_rate) * violation, log_min, log_max,
    )), min_dual, max_dual)
    return updated, violation


def reward_beta_from_dual(dual_lambda):
    """Negative inverse temperature for low-return domain search."""
    return -jnp.reciprocal(jnp.asarray(dual_lambda))


def cost_beta_from_dual(dual_lambda):
    """Positive inverse temperature for high-cost domain search."""
    return jnp.reciprocal(jnp.asarray(dual_lambda))


def reward_dual_from_beta(beta):
    beta = jnp.asarray(beta)
    return -jnp.reciprocal(beta)


def cost_dual_from_beta(beta):
    beta = jnp.asarray(beta)
    return jnp.reciprocal(beta)
