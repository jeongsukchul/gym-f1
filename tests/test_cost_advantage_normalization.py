"""Independent cost normalization keeps the historical objective as default."""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from gymkhana.jax_sampler_ppo.losses import ppo_lagrange_loss
from gymkhana.jax_sampler_ppo.networks import normal_tanh_log_prob


class TestCostAdvantageNormalization(unittest.TestCase):
    def loss(self, theta, cost_normalization, cost_targets=None, std_floor=0.):
        raw = jnp.array([[-.5], [.5]])
        networks = SimpleNamespace(
            policy_network=SimpleNamespace(apply=lambda p, obs: (jnp.full((2, 1), p), jnp.zeros(1))),
            value_network=SimpleNamespace(apply=lambda p, obs: jnp.zeros(2)),
            cost_value_network=SimpleNamespace(apply=lambda p, obs: jnp.zeros(2)),
        )
        data = SimpleNamespace(observation={"actor_obs": jnp.zeros((2, 1)), "value_obs": jnp.zeros((2, 1))},
                               raw_action=raw, log_prob=normal_tanh_log_prob(jnp.zeros((2, 1)), jnp.zeros(1), raw),
                               extras={"state_extras": {"cost": jnp.zeros(2)}})
        return ppo_lagrange_loss(SimpleNamespace(policy=theta, value=None, cost_value=None),
                                networks, data, jnp.zeros(2), jnp.array([-1., 1.]),
                                jnp.zeros(2) if cost_targets is None else cost_targets,
                                jnp.array([-.001, .001]), jnp.asarray(2.),
                                normalize_advantage=True, normalize_cost_advantage=cost_normalization,
                                cost_advantage_std_floor=std_floor,
                                entropy_cost=0., value_cost=0.)

    def test_default_matches_historical_independent_standardization(self):
        for flag in (None, True):
            grad = jax.grad(lambda theta: self.loss(theta, flag)[0])(jnp.asarray(0.))
            self.assertGreater(float(grad), 0.)
        np.testing.assert_allclose(jax.grad(lambda t: self.loss(t, None)[0])(jnp.asarray(0.)),
                                   jax.grad(lambda t: self.loss(t, True)[0])(jnp.asarray(0.)))

    def test_raw_cost_does_not_amplify_tiny_cost_advantage(self):
        grad = jax.grad(lambda theta: self.loss(theta, False)[0])(jnp.asarray(0.))
        self.assertLess(float(grad), 0.)
        _, metrics = self.loss(jnp.asarray(0.), False)
        np.testing.assert_allclose(metrics["constraint/cost_advantage_std_raw"], .001)
        self.assertEqual(float(metrics["constraint/normalize_cost_advantage"]), 0.)

    def test_cost_critic_diagnostics_use_actual_targets(self):
        _, metrics = self.loss(jnp.asarray(0.), False, jnp.array([.2, .8]))
        self.assertEqual(float(metrics["constraint/cost_value_mean"]), 0.)
        np.testing.assert_allclose(metrics["constraint/cost_target_mean"], .5)
        np.testing.assert_allclose(metrics["constraint/cost_target_std"], .3)

    def test_std_floor_limits_tiny_signal_amplification_without_disabling_normalization(self):
        grad = jax.grad(lambda theta: self.loss(theta, True, std_floor=.01)[0])(jnp.asarray(0.))
        self.assertLess(float(grad), 0.)
        _, metrics = self.loss(jnp.asarray(0.), True, std_floor=.01)
        np.testing.assert_allclose(metrics["constraint/cost_normalization_scale"], .01+1e-8)
        np.testing.assert_allclose(metrics["constraint/reward_advantage_std_raw"], 1.)
        self.assertEqual(float(metrics["constraint/normalize_cost_advantage"]), 1.)
        _, raw_metrics = self.loss(jnp.asarray(0.), False, std_floor=.01)
        self.assertEqual(float(raw_metrics["constraint/cost_normalization_scale"]), 1.)


if __name__ == "__main__":
    unittest.main()
