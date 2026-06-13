"""Domain-parameter helpers for F1TENTH JAX sampler PPO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp

from gymkhana.jax_env import JaxVehicleParams, load_vehicle_params

_FORBIDDEN_PARAMS = {"lr": "lf", "s_min": "s_max", "sv_min": "sv_max"}


@dataclass(frozen=True)
class F1TenthDomainSpec:
    """Bounded domain randomization vector for JAX F1TENTH dynamics."""

    names: tuple[str, ...]
    low: jax.Array
    high: jax.Array
    nominal_vector: jax.Array
    nominal_params: JaxVehicleParams
    wheelbase: float

    @property
    def size(self) -> int:
        return len(self.names)


def make_domain_spec(
    ranges: Mapping[str, Sequence[float]] | None = None,
    *,
    sigmas: Mapping[str, float] | None = None,
    clip_k: float = 3.0,
    param_name: str = "f1tenth_std",
    names: Sequence[str] | None = None,
) -> F1TenthDomainSpec:
    """Build actual-value DR bounds from multiplicative range config.

    The JAX training path uses explicit multiplicative factors per parameter:
    the two endpoints are ``nominal * range[0]`` and ``nominal * range[1]``.
    They are sorted into numeric low/high bounds so negative nominal parameters
    still sample correctly. Coupled fields are derived later by
    :func:`params_from_vector`.

    ``sigmas`` is kept only as a compatibility shim for older direct callers.
    """
    nominal_params = load_vehicle_params(param_name)
    if ranges is not None and sigmas is not None:
        raise ValueError("Pass either domain ranges or legacy sigmas, not both.")
    if ranges is None:
        if not sigmas:
            raise ValueError("JAX domain randomization requires a non-empty domain_randomization_ranges config.")
        ranges = {}
        for name, sigma_value in sigmas.items():
            sigma = float(sigma_value)
            if sigma <= 0.0:
                raise ValueError(f"domain sigma for {name!r} must be positive, got {sigma}")
            delta = float(clip_k) * sigma
            ranges[name] = (1.0 - delta, 1.0 + delta)

    ranges = dict(ranges)
    if not ranges:
        raise ValueError("JAX domain randomization requires a non-empty domain_randomization_ranges config.")
    names = tuple(ranges.keys() if names is None else names)

    low = []
    high = []
    nominal = []
    for name in names:
        canonical = _FORBIDDEN_PARAMS.get(name)
        if canonical is not None:
            raise ValueError(f"Randomize {canonical!r}; {name!r} is derived from it.")
        if not hasattr(nominal_params, name):
            raise ValueError(f"{name!r} is not a JAX vehicle parameter.")
        if name not in ranges:
            raise ValueError(f"missing domain range for {name!r}")

        range_values = tuple(float(value) for value in ranges[name])
        if len(range_values) != 2:
            raise ValueError(f"domain range for {name!r} must be [low_factor, high_factor]")
        low_factor, high_factor = range_values
        if not (math.isfinite(low_factor) and math.isfinite(high_factor)):
            raise ValueError(f"domain range for {name!r} must be finite, got {range_values}")
        if low_factor <= 0.0 or high_factor <= 0.0:
            raise ValueError(f"domain range for {name!r} must use positive factors, got {range_values}")
        if low_factor >= high_factor:
            raise ValueError(f"domain range for {name!r} must satisfy low < high, got {range_values}")

        center = float(getattr(nominal_params, name))
        lo = center * low_factor
        hi = center * high_factor
        low.append(min(lo, hi))
        high.append(max(lo, hi))
        nominal.append(center)

    return F1TenthDomainSpec(
        names=names,
        low=jnp.asarray(low, dtype=jnp.float32),
        high=jnp.asarray(high, dtype=jnp.float32),
        nominal_vector=jnp.asarray(nominal, dtype=jnp.float32),
        nominal_params=nominal_params,
        wheelbase=float(nominal_params.lf + nominal_params.lr),
    )


def params_from_vector(spec: F1TenthDomainSpec, values: jax.Array) -> JaxVehicleParams:
    """Convert ``(..., D)`` actual DR values into a batched params pytree."""
    values = jnp.asarray(values, dtype=jnp.float32)
    updates = {name: values[..., idx] for idx, name in enumerate(spec.names)}

    if "lf" in updates:
        updates["lr"] = spec.wheelbase - updates["lf"]
    if "s_max" in updates:
        updates["s_min"] = -updates["s_max"]
    if "sv_max" in updates:
        updates["sv_min"] = -updates["sv_max"]

    return spec.nominal_params._replace(**updates)


def sample_uniform(spec: F1TenthDomainSpec, key: jax.Array, batch_size: int) -> jax.Array:
    return jax.random.uniform(key, (batch_size, spec.size), minval=spec.low, maxval=spec.high)


def clip_to_domain(spec: F1TenthDomainSpec, values: jax.Array) -> jax.Array:
    return jnp.clip(values, spec.low, spec.high)
