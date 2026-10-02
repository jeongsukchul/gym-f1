"""Export an existing JAX msgpack checkpoint without resuming training."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml
from flax.serialization import msgpack_restore

from gymkhana.jax_sampler_ppo.export_deploy import export_deployment_bundle
from gymkhana.jax_sampler_ppo.networks import SamplerPPONetworkParams, make_sampler_ppo_networks
from train.jax_sampler_ppo import _build_env, _hidden_layer_sizes_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gym-config", type=Path, required=True, help="Effective saved gym config for this checkpoint")
    parser.add_argument("--rl-config", type=Path, required=True, help="Effective saved RL config for this checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--obs-delay-steps", type=int, default=None)
    args = parser.parse_args()
    gym_config = yaml.safe_load(args.gym_config.read_text(encoding="utf-8"))
    rl_config = yaml.safe_load(args.rl_config.read_text(encoding="utf-8"))
    settings = rl_config["jax_sampler_ppo"]
    wrapper = _build_env(gym_config, 1, int(rl_config["seed"]), None,
                         domain_randomization_ranges=settings.get("domain_randomization_ranges"),
                         asymmetric_critic=bool(settings.get("asymmetric_critic", False)),
                         action_repeat_steps=int(settings.get("policy_repeat_steps", 1)))
    networks = make_sampler_ppo_networks(
        wrapper.observation_size, 2,
        policy_hidden_layer_sizes=_hidden_layer_sizes_from_config(rl_config, "actor_layer", "actor_layer_size"),
        policy_use_layer_norm=bool(rl_config.get("actor_layer_norm", False)),
    )
    raw = msgpack_restore(args.checkpoint.read_bytes())
    saved_params = raw["params"]
    state = SimpleNamespace(params=SamplerPPONetworkParams(policy=saved_params["policy"],
                                                           value=saved_params["value"],
                                                           cost_value=saved_params["cost_value"]))
    trainer = SimpleNamespace(env=wrapper, networks=networks)
    contract = export_deployment_bundle(trainer, state, args.output, obs_delay_steps=args.obs_delay_steps)
    print("Exported %s: actor_obs=%d, policy_hz=%g, JAX/NumPy max_error=%g" %
          (args.output, contract["actor_observation_size"], contract["policy_hz"], contract["actor_validation_max_error"]))


if __name__ == "__main__":
    main()
