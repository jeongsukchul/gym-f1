"""Backend config boundaries without importing heavy training dependencies."""

import ast
from pathlib import Path
import unittest

import yaml
from train.config.rollout import get_rollout_length

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "train/config"


class TestConfigSplit(unittest.TestCase):
    def load(self, backend, filename):
        return yaml.safe_load((CONFIG / backend / filename).read_text())

    def test_rl_values_preserved_and_backend_keys_separated(self):
        legacy = yaml.safe_load((CONFIG / "rl_config.yaml").read_text())
        jax = self.load("jax", "rl_config.yaml")
        ppo = self.load("ppo", "rl_config.yaml")
        self.assertEqual(jax["rollout_length"], 256)
        self.assertEqual(jax["batch_size"], 1024)
        self.assertNotIn("jax_sampler_ppo", ppo)
        for key in ("core_mult", "additional_timesteps", "transfer_reset_critic",
                    "transfer_reset_log_std", "act_func_neg_slope", "use_custom_relu"):
            self.assertNotIn(key, jax)
            self.assertIn(key, ppo)
        for config in (jax, ppo):
            for key, value in config.items():
                if key in {"rollout_length", "batch_size"} and config is jax:
                    continue  # JAX rollout/minibatch are independently tuned after the split.
                if key == "jax_sampler_ppo":
                    expected = dict(legacy[key], reset_state_on_rollout=False,
                                    lagrangian_update_mode="first_episode", allow_partial_first_episode=True)
                    self.assertEqual(value, expected)
                    continue
                self.assertEqual(value, legacy["n_steps" if key == "rollout_length" else key], key)

    def test_rollout_length_legacy_compatibility(self):
        self.assertEqual(get_rollout_length({"rollout_length": 1024}), 1024)
        self.assertEqual(get_rollout_length({"n_steps": 6144}), 6144)
        self.assertEqual(get_rollout_length({"n_steps": 1024, "rollout_length": 1024}), 1024)
        with self.assertRaises(ValueError):
            get_rollout_length({"n_steps": 6144, "rollout_length": 1024})
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                get_rollout_length({"rollout_length": value})

    def test_environment_keys_and_values(self):
        legacy = yaml.safe_load((CONFIG / "gym_config.yaml").read_text())
        jax = self.load("jax", "gym_config.yaml")
        ppo = self.load("ppo", "gym_config.yaml")
        for key in ("domain_randomization", "curriculum", "recovery_project_name",
                    "ckpt_save_freq", "num_beams"):
            self.assertNotIn(key, jax)
            self.assertIn(key, ppo)
        for key in ("obs_history_len", "obs_delay_min_steps", "mask_track_obs",
                    "sensor_noise_enabled", "edge_penalty_weight"):
            self.assertIn(key, jax)
            self.assertNotIn(key, ppo)
        self.assertIsNone(ppo["track_pool"])
        for config in (jax, ppo):
            for key, value in config.items():
                if key != "track_pool":
                    self.assertEqual(value, legacy[key], key)

    def test_sb3_loader_required_keys_exist(self):
        tree = ast.parse((CONFIG / "env_config.py").read_text())
        configs = {"_config": self.load("ppo", "gym_config.yaml"),
                   "_rl_config": self.load("ppo", "rl_config.yaml")}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id in configs and isinstance(node.slice, ast.Constant)):
                self.assertIn(node.slice.value, configs[node.value.id])


if __name__ == "__main__":
    unittest.main()
