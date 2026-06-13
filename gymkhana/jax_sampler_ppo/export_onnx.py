"""Export JAX sampler PPO policies to the existing Gym-Khana ONNX format."""

from __future__ import annotations

import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from flax.core import unfreeze

from .networks import SamplerPPONetworkParams, SamplerPPONetworks, sample_action


def _dense_layer_names(policy_params: dict) -> list[str]:
    names = [name for name in policy_params if name.startswith("Dense_")]
    return sorted(names, key=lambda name: int(name.split("_", maxsplit=1)[1]))


def _activation_name(network: nn.Module) -> str:
    activation = getattr(network, "activation", None)
    name = getattr(activation, "__name__", "")
    if name in {"tanh", "swish"}:
        return name
    raise ValueError(f"Unsupported JAX policy activation for ONNX export: {activation!r}")


class TorchDeterministicJaxPolicy(nn.Module):
    """Torch actor mirror used only to trace a Flax policy into ONNX."""

    def __init__(self, policy_params, activation: str):
        super().__init__()
        params = unfreeze(policy_params)["params"]
        dense_names = _dense_layer_names(params)
        if not dense_names:
            raise ValueError("No Dense layers found in JAX policy params.")

        self.layers = nn.ModuleList()
        for name in dense_names:
            kernel = np.asarray(params[name]["kernel"], dtype=np.float32)
            bias = np.asarray(params[name]["bias"], dtype=np.float32)
            layer = nn.Linear(kernel.shape[0], kernel.shape[1])
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(kernel.T.copy()))
                layer.bias.copy_(torch.from_numpy(bias.copy()))
            self.layers.append(layer)

        self.activation = activation

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "swish":
            return x * torch.sigmoid(x)
        raise RuntimeError(f"Unsupported activation: {self.activation}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for layer in self.layers[:-1]:
            x = self._activate(layer(x))
        mean = self.layers[-1](x)
        return torch.tanh(mean)


def export_policy_to_onnx(
    params: SamplerPPONetworkParams,
    networks: SamplerPPONetworks,
    output_path: str,
    *,
    observation_size: int,
    validation_obs: np.ndarray | None = None,
    atol: float = 5e-4,
) -> None:
    """Export a deterministic JAX actor to ONNX.

    The exported model matches the existing Gymnasium/SB3 ONNX contract:
    input ``obs`` with shape ``(batch, obs_dim)`` and output ``action`` with
    shape ``(batch, act_dim)``.
    """
    activation = _activation_name(networks.policy_network)
    torch_policy = TorchDeterministicJaxPolicy(params.policy, activation)
    torch_policy.eval()

    dummy = torch.randn(1, observation_size, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*legacy TorchScript-based.*")
        torch.onnx.export(
            torch_policy,
            dummy,
            output_path,
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=14,
            dynamo=False,
        )

    external_data = output_path + ".data"
    if os.path.exists(external_data):
        os.remove(external_data)

    if validation_obs is None:
        validation_obs = np.random.default_rng(0).standard_normal((5, observation_size)).astype(np.float32)
    validate_export(params, networks, output_path, validation_obs=validation_obs, atol=atol)


def validate_export(
    params: SamplerPPONetworkParams,
    networks: SamplerPPONetworks,
    onnx_path: str,
    *,
    validation_obs: np.ndarray,
    atol: float = 5e-4,
) -> float:
    """Compare exported ONNX deterministic actions against the JAX policy."""
    obs = np.asarray(validation_obs, dtype=np.float32)
    jax_actions, _ = sample_action(
        networks.policy_network,
        params.policy,
        jnp.asarray(obs),
        jax.random.PRNGKey(0),
        deterministic=True,
    )
    jax_actions = np.asarray(jax_actions)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_actions = session.run(["action"], {"obs": obs})[0]
    max_diff = float(np.max(np.abs(jax_actions - onnx_actions)))
    if not np.allclose(jax_actions, onnx_actions, atol=atol):
        raise RuntimeError(f"ONNX output differs from JAX policy (max diff: {max_diff:.2e}).")
    return max_diff


def export_trainer_policy_to_onnx(
    trainer,
    training_state,
    output_path: str,
    *,
    validation_obs: np.ndarray | None = None,
    atol: float = 5e-4,
) -> None:
    """Convenience wrapper for :class:`SamplerPPOTrainer` instances."""
    export_policy_to_onnx(
        training_state.params,
        trainer.networks,
        output_path,
        observation_size=trainer.env.observation_size,
        validation_obs=validation_obs,
        atol=atol,
    )
