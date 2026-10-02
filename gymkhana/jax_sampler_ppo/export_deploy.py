"""Export a deterministic JAX actor and its complete legacy-ROS contract."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import unfreeze

from .networks import sample_action


def export_deployment_bundle(trainer, training_state, output_dir, *, obs_delay_steps=None):
    """Export NumPy MLP weights, track geometry and fixed training normalization.

    Delays are in policy steps, not physics steps. The default uses the lower
    training delay; override after measuring latency to avoid adding latency twice.
    This exports only the actor; privileged critic DR parameters stay on the PC.
    """
    wrapper = trainer.env
    env = wrapper.env
    network = trainer.networks.policy_network
    activation = getattr(network.activation, "__name__", "")
    if activation not in ("tanh", "swish"):
        raise ValueError("Unsupported actor activation: " + activation)
    if network.use_layer_norm:
        raise ValueError("Legacy NumPy actor export does not support actor LayerNorm")
    delay = wrapper.obs_delay_min_steps if obs_delay_steps is None else int(obs_delay_steps)
    if delay < 0 or delay > wrapper.obs_delay_max_steps:
        raise ValueError("Deployment delay must be between zero and the maximum training delay")
    params = unfreeze(training_state.params.policy)["params"]
    names = sorted((name for name in params if name.startswith("Dense_")), key=lambda name: int(name.split("_")[1]))
    if not names or any(name not in names and name != "log_std" for name in params):
        raise ValueError("Unexpected actor parameter structure")
    weights = {"layer_count": np.asarray(len(names), dtype=np.int32), "activation": np.asarray(activation)}
    size = wrapper.observation_size
    for i, name in enumerate(names):
        kernel = np.asarray(params[name]["kernel"], dtype=np.float32)
        bias = np.asarray(params[name]["bias"], dtype=np.float32)
        if kernel.ndim != 2 or kernel.shape[0] != size or bias.shape != (kernel.shape[1],):
            raise ValueError("Actor shape mismatch")
        if not np.isfinite(kernel).all() or not np.isfinite(bias).all():
            raise ValueError("Non-finite actor weights")
        weights["kernel_%d" % i], weights["bias_%d" % i] = kernel, bias
        size = kernel.shape[1]
    if size != 2:
        raise ValueError("Actor must output [steering, speed]")

    observations = np.random.default_rng(0).uniform(-1, 1, (32, wrapper.observation_size)).astype(np.float32)
    result = observations
    for i in range(len(names)):
        result = result @ weights["kernel_%d" % i] + weights["bias_%d" % i]
        result = np.tanh(result) if i == len(names)-1 or activation == "tanh" else result / (1+np.exp(-result))
    expected, _ = sample_action(network, training_state.params.policy, jnp.asarray(observations),
                                jax.random.PRNGKey(0), deterministic=True)
    max_error = float(np.max(np.abs(result-np.asarray(expected))))
    if not np.allclose(result, expected, atol=5e-5, rtol=1e-5):
        raise ValueError("NumPy/JAX actor disagreement: %g" % max_error)

    p = env.params
    widths = 2 if env.sparse_width_obs else env.lookahead_n_points
    config = {
        "contract_version": 1,
        "action_order": ["steering", "speed"],
        "action_params": {"s_max": float(p.s_max), "v_min": float(p.v_min), "v_max": float(p.v_max)},
        "physics_timestep": env.timestep,
        "policy_repeat_steps": wrapper.action_repeat_steps,
        "policy_hz": 1.0 / (env.timestep * wrapper.action_repeat_steps),
        "lookahead_n_points": env.lookahead_n_points,
        "lookahead_ds": env.lookahead_ds,
        "sparse_width_obs": env.sparse_width_obs,
        "normalize_obs": env.normalize_obs,
        "mask_track_obs": env.mask_track_obs,
        "obs_history_len": wrapper.obs_history_len,
        "training_obs_delay_min_steps": wrapper.obs_delay_min_steps,
        "training_obs_delay_max_steps": wrapper.obs_delay_max_steps,
        "deployment_obs_delay_steps": delay,
        "actor_observation_size": wrapper.observation_size,
        "features": ["vx", "vy", "heading_error", "lateral_error", "yaw_rate", "beta", "wheel_omega",
                     "lookahead_curvatures", "lookahead_widths"],
        "coordinate_origin": "vehicle_cog",
        "lateral_error_sign": "left_positive",
        "history_order": "newest_to_oldest",
        "beta_speed_floor": 0.05,
        "norm_lows": [float(p.v_min), -0.5*float(p.v_max), -np.pi, -1.1, -5.0, -np.pi/3, 0.0]
                     + [-1.95]*env.lookahead_n_points + [1.2]*widths,
        "norm_highs": [float(p.v_max), 0.5*float(p.v_max), np.pi, 1.1, 5.0, np.pi/3,
                       float(p.v_max/p.R_w*6.4)] + [1.95]*env.lookahead_n_points + [2.2]*widths,
        "actor_validation_max_error": max_error,
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Refuse to replace an existing model bundle accidentally.
    for name in ("actor.npz", "track.npz", "contract.json"):
        if (directory/name).exists():
            raise FileExistsError("Bundle already exists: " + str(directory/name))
    np.savez(directory/"actor.npz", **weights)
    track_arrays = {name: np.asarray(getattr(env.track, name), dtype=np.float32)
                    for name in ("xs", "ys", "ss", "yaws", "curvatures", "widths")}
    np.savez(directory/"track.npz", length=np.asarray(env.track.length, dtype=np.float32), **track_arrays)
    config["sha256"] = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                        for name in ("actor.npz", "track.npz")}
    (directory/"contract.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config
