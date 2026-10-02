"""Budget-aligned Lagrange updates for finite-horizon rollouts."""

import jax.numpy as jnp


def first_episode_costs(costs, dones):
    """Cumulative per-env cost up to and including the first terminal step."""
    previous_done = jnp.concatenate([jnp.zeros_like(dones[:1]), dones[:-1]], axis=0)
    active = jnp.cumsum(previous_done.astype(jnp.int32), axis=0) == 0
    return jnp.sum(costs * active, axis=0)


def completed_episode_costs(costs, dones, previous_cost):
    """Carry partial episode costs; return per-env completed cost sums/counts."""
    import jax

    def step(carry, item):
        cost, done = item
        total = carry + cost
        completed = jnp.where(done, total, 0.0)
        return jnp.where(done, 0.0, total), (completed, done.astype(jnp.float32))

    remaining, (totals, counts) = jax.lax.scan(step, previous_cost, (costs, dones))
    return remaining, jnp.sum(totals, axis=0), jnp.sum(counts, axis=0)


def lagrange_cost_estimate(costs, dones, *, mode, episode_steps, budget):
    """Return cost and budget in the same units.

    first_episode counts only each env's first episode in this rollout.
    With binary terminal collision it estimates collision probability up to
    the rollout horizon; early auto-resets cannot add extra collisions.
    """
    if mode == "per_step":
        return jnp.mean(costs), budget / float(episode_steps)
    if mode == "rollout":
        return jnp.mean(jnp.sum(costs, axis=0)), budget * costs.shape[0] / float(episode_steps)
    if mode == "first_episode":
        return jnp.mean(first_episode_costs(costs, dones)), budget
    raise ValueError(f"Unknown Lagrange update mode: {mode}")
