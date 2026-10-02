"""Full-state checkpoint continuation, not policy-only warm starts."""

from pathlib import Path
import tempfile
import unittest

from flax.serialization import to_bytes
import jax
import jax.numpy as jnp
import numpy as np

from gymkhana.jax_env import JaxRaceEnv
from gymkhana.jax_sampler_ppo.domain import make_domain_spec
from gymkhana.jax_sampler_ppo.wrappers import F1TenthAdvWrapper
from gymkhana.jax_sampler_ppo.sampler_ppo import SamplerPPOConfig, SamplerPPOTrainer
from train.jax_sampler_ppo import _restore_training_checkpoint, _evaluation_schedule
from recover_nominal_trial import final_evaluation_key


class TestTrainingResume(unittest.TestCase):
    def test_roundtrip_preserves_all_state_and_next_update(self):
        env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=32)
        wrapper = F1TenthAdvWrapper(env, make_domain_spec(ranges={"m": [.8, 1.2]}),
                                  constraint_cost_type="collision")
        trainer = SamplerPPOTrainer(wrapper, SamplerPPOConfig(
            domain_randomization=False, unroll_length=2, batch_size=8, num_epochs=1,
            use_ppo_lag=True, constraint_cost_type="collision", reset_state_on_rollout=False,
            lagrangian_update_mode="completed_episode", lagrangian_min_completed_episodes=384,
            init_log_std=-1.5, end_log_std=-1.5,
            policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,),
            cost_value_hidden_layer_sizes=(8,),
        ))
        initial = trainer.init_state(jax.random.PRNGKey(1))
        update = jax.jit(trainer.training_step)
        saved, _ = update(initial)
        saved = saved.replace(lambda_lagr=jnp.asarray(7.2),
                              lagrangian_pending_cost_sum=jnp.asarray(4.),
                              lagrangian_pending_episode_count=jnp.asarray(10.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.msgpack"
            path.write_bytes(to_bytes(jax.device_get(saved)))
            restored = _restore_training_checkpoint(initial, path, num_envs=4,
                                                     rollout_length=2, target_updates=3)
            for actual, expected in zip(jax.tree_util.tree_leaves(restored),
                                        jax.tree_util.tree_leaves(saved)):
                np.testing.assert_array_equal(actual, expected)
            continued, _ = update(restored)
            uninterrupted, _ = update(saved)
            for actual, expected in zip(jax.tree_util.tree_leaves(continued),
                                        jax.tree_util.tree_leaves(uninterrupted)):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(int(continued.update_steps), 2)
            self.assertEqual(int(continued.env_steps), 16)
            with self.assertRaisesRegex(ValueError, "target"):
                _restore_training_checkpoint(initial, path, num_envs=4,
                                             rollout_length=2, target_updates=1)
            with self.assertRaisesRegex(ValueError, "geometry"):
                _restore_training_checkpoint(initial, path, num_envs=8,
                                             rollout_length=2, target_updates=3)

    def test_resume_schedule_uses_absolute_updates_and_replay_key(self):
        initial, scheduled = _evaluation_schedule(100, 11, start_update=50)
        self.assertTrue(initial)
        self.assertEqual(scheduled, {60, 70, 80, 90, 100})
        key = jax.random.PRNGKey(3000)
        for _ in range(6):
            key, evaluation_key, _, _ = jax.random.split(key, 4)
        np.testing.assert_array_equal(final_evaluation_key(3000, 100, 11, start_update=50),
                                      evaluation_key)


if __name__ == "__main__":
    unittest.main()
