"""Sampler PPO utilities for the Gym-Khana JAX backend.

Optional features such as ONNX export and the GMMVI sampler are loaded lazily
so the base JAX trainer path does not require their extra dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "F1TenthDomainSpec": (".domain", "F1TenthDomainSpec"),
    "make_domain_spec": (".domain", "make_domain_spec"),
    "params_from_vector": (".domain", "params_from_vector"),
    "AdvEvaluator": (".evaluator", "AdvEvaluator"),
    "EvalResult": (".evaluator", "EvalResult"),
    "TrajectoryResult": (".evaluator", "TrajectoryResult"),
    "evaluate_policy": (".evaluator", "evaluate_policy"),
    "generate_adv_unroll": (".evaluator", "generate_adv_unroll"),
    "record_policy_trajectory": (".evaluator", "record_policy_trajectory"),
    "export_policy_to_onnx": (".export_onnx", "export_policy_to_onnx"),
    "export_trainer_policy_to_onnx": (".export_onnx", "export_trainer_policy_to_onnx"),
    "validate_export": (".export_onnx", "validate_export"),
    "BoundedGMMVISampler": (".gmmvi_sampler", "BoundedGMMVISampler"),
    "GMMVISamplerState": (".gmmvi_sampler", "GMMVISamplerState"),
    "RewardCostGMMVISampler": (".gmmvi_sampler", "RewardCostGMMVISampler"),
    "RewardCostGMMVISamplerState": (".gmmvi_sampler", "RewardCostGMMVISamplerState"),
    "SamplerPPONetworkParams": (".networks", "SamplerPPONetworkParams"),
    "SamplerPPONetworks": (".networks", "SamplerPPONetworks"),
    "make_inference_fn": (".networks", "make_inference_fn"),
    "make_sampler_ppo_networks": (".networks", "make_sampler_ppo_networks"),
    "SamplerPPOConfig": (".sampler_ppo", "SamplerPPOConfig"),
    "SamplerPPOTrainer": (".sampler_ppo", "SamplerPPOTrainer"),
    "SamplerPPOTrainingState": (".sampler_ppo", "SamplerPPOTrainingState"),
    "train": (".sampler_ppo", "train"),
    "UniformDRSampler": (".uniform_sampler", "UniformDRSampler"),
    "UniformDRSamplerState": (".uniform_sampler", "UniformDRSamplerState"),
    "AdvEnvState": (".wrappers", "AdvEnvState"),
    "F1TenthAdvWrapper": (".wrappers", "F1TenthAdvWrapper"),
    "TransitionWithParams": (".wrappers", "TransitionWithParams"),
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:
    from .domain import F1TenthDomainSpec, make_domain_spec, params_from_vector
    from .evaluator import (
        AdvEvaluator,
        EvalResult,
        TrajectoryResult,
        evaluate_policy,
        generate_adv_unroll,
        record_policy_trajectory,
    )
    from .export_onnx import export_policy_to_onnx, export_trainer_policy_to_onnx, validate_export
    from .gmmvi_sampler import (
        BoundedGMMVISampler,
        GMMVISamplerState,
        RewardCostGMMVISampler,
        RewardCostGMMVISamplerState,
    )
    from .networks import SamplerPPONetworkParams, SamplerPPONetworks, make_inference_fn, make_sampler_ppo_networks
    from .sampler_ppo import SamplerPPOConfig, SamplerPPOTrainer, SamplerPPOTrainingState, train
    from .uniform_sampler import UniformDRSampler, UniformDRSamplerState
    from .wrappers import AdvEnvState, F1TenthAdvWrapper, TransitionWithParams


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return list(__all__)
