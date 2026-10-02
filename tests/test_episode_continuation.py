"""Regression checks for state-preserving rollout boundaries and episode cost."""

import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np

from gymkhana.jax_env import JaxRaceEnv
from gymkhana.jax_sampler_ppo.domain import make_domain_spec
from gymkhana.jax_sampler_ppo.wrappers import F1TenthAdvWrapper
from gymkhana.jax_sampler_ppo.lagrange import completed_episode_costs, lagrange_cost_estimate
from gymkhana.jax_sampler_ppo.gmmvi_sampler import RewardCostGMMVISampler
from gymkhana.jax_sampler_ppo.sampler_ppo import SamplerPPOConfig, SamplerPPOTrainer
from gymkhana.jax_sampler_ppo.losses import compute_gae


class TestEpisodeContinuation(unittest.TestCase):
    def test_finite_episode_cost_gae_learns_safe_timeout(self):
        costs = jnp.zeros((3, 1))
        values = jnp.ones_like(costs)
        discounts = jnp.array([[1.], [1.], [0.]])
        truncation = jnp.array([[0.], [0.], [1.]])
        legacy_targets, _ = compute_gae(costs, values, jnp.ones(1), discounts,
                                        truncation, gae_lambda=1., discounting=1.)
        targets, advantages = compute_gae(costs, values, jnp.ones(1), discounts,
                                          jnp.zeros_like(truncation),
                                          gae_lambda=1., discounting=1.)
        np.testing.assert_array_equal(legacy_targets, jnp.ones_like(costs))
        np.testing.assert_array_equal(targets, costs)
        np.testing.assert_array_equal(advantages, -values)
        collision_targets, _ = compute_gae(costs.at[-1, 0].set(1.), values,
                                           jnp.ones(1), discounts,
                                           jnp.zeros_like(truncation),
                                           gae_lambda=1., discounting=1.)
        np.testing.assert_array_equal(collision_targets, jnp.ones_like(costs))

    def test_time_limit_flag_changes_only_cost_gae_mask(self):
        env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=1)
        wrapper = F1TenthAdvWrapper(env, make_domain_spec(ranges={"m": [0.8, 1.2]}),
                                  constraint_cost_type="collision")
        self.assertFalse(SamplerPPOConfig().cost_terminal_at_time_limit)
        for enabled in (False, True):
            trainer = SamplerPPOTrainer(wrapper, SamplerPPOConfig(
                domain_randomization=False, unroll_length=2, batch_size=8,
                num_epochs=1, use_ppo_lag=True, cost_discounting=1.,
                constraint_cost_type="collision",
                cost_advantage_std_floor=.01,
                cost_terminal_at_time_limit=enabled,
                policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,),
                cost_value_hidden_layer_sizes=(8,),
            ))
            masks = []

            def capture_gae(*args, **kwargs):
                jax.debug.callback(lambda mask: masks.append(np.asarray(mask)),
                                   args[4], ordered=True)
                return compute_gae(*args, **kwargs)

            with patch("gymkhana.jax_sampler_ppo.sampler_ppo.compute_gae",
                       side_effect=capture_gae):
                state, metrics = jax.jit(trainer.training_step)(
                    trainer.init_state(jax.random.PRNGKey(0)))
                jax.block_until_ready((state, metrics))
            self.assertEqual(len(masks), 2)
            self.assertTrue(np.any(masks[0]))
            np.testing.assert_array_equal(masks[1],
                                          np.zeros_like(masks[0]) if enabled else masks[0])
            self.assertEqual(float(metrics["training/constraint/cost_terminal_at_time_limit"]),
                             float(enabled))
            for name in ("cost_value_mean", "cost_target_mean", "cost_target_std"):
                self.assertTrue(np.isfinite(float(metrics[f"training/critic/{name}"])))
                self.assertEqual(float(metrics[f"training/critic/{name}"]),
                                 float(metrics[f"constraint/{name}"]))
            np.testing.assert_allclose(metrics["training/advantage/cost_std_floor"], .01)
            self.assertGreaterEqual(float(metrics["training/advantage/cost_normalization_scale"]), .01)

    def test_log_std_schedule_does_not_recompile_after_initial_step(self):
        env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=32)
        wrapper = F1TenthAdvWrapper(env, make_domain_spec(ranges={"m": [0.8, 1.2]}))
        trainer = SamplerPPOTrainer(wrapper, SamplerPPOConfig(
            domain_randomization=False, unroll_length=2, batch_size=8, num_epochs=1,
            init_log_std=-.4, end_log_std=-1.5,
            policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,),
            cost_value_hidden_layer_sizes=(8,),
        ))
        state = trainer.init_state(jax.random.PRNGKey(0))
        self.assertFalse(state.params.policy["params"]["log_std"].weak_type)
        update = jax.jit(trainer.training_step)
        state, _ = update(state)
        jax.block_until_ready(state)
        state, _ = update(state)
        jax.block_until_ready(state)
        self.assertEqual(update._cache_size(), 1)

    def test_cost_discount_can_change_without_reward_discount(self):
        env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=32)
        wrapper = F1TenthAdvWrapper(env, make_domain_spec(ranges={"m": [0.8, 1.2]}))
        for gamma in (None, .999, 1.):
            config = SamplerPPOConfig(discounting=.99, cost_discounting=gamma,
                                      policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,),
                                      cost_value_hidden_layer_sizes=(8,))
            trainer = SamplerPPOTrainer(wrapper, config)
            self.assertEqual(trainer.config.discounting, .99)
            self.assertEqual(trainer.cost_discounting, .99 if gamma is None else gamma)
        with self.assertRaisesRegex(ValueError, "cost_discounting"):
            SamplerPPOTrainer(wrapper, SamplerPPOConfig(cost_discounting=1.1))
        for floor in (-.01, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "cost_advantage_std_floor"):
                SamplerPPOTrainer(wrapper, SamplerPPOConfig(cost_advantage_std_floor=floor))

    def test_lambda_waits_for_episode_batch_and_warmup(self):
        env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=32)
        wrapper = F1TenthAdvWrapper(env, make_domain_spec(ranges={"m": [0.8, 1.2]}),
                                  constraint_cost_type="collision")
        trainer = SamplerPPOTrainer(wrapper, SamplerPPOConfig(
            domain_randomization=False, unroll_length=2, batch_size=8, num_epochs=1,
            reset_state_on_rollout=False, use_ppo_lag=True,
            constraint_cost_type="collision", safety_bound=0.05,
            lagrangian_update_mode="completed_episode", initial_lambda_lagr=0.,
            lagrangian_coef_rate=1., lagrangian_ema_decay=0.,
            lagrangian_min_completed_episodes=384, lagrangian_warmup_steps=100,
            cost_discounting=1.,
            policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,),
            cost_value_hidden_layer_sizes=(8,),
        ))
        state = trainer.init_state(jax.random.PRNGKey(0))
        update = jax.jit(trainer.training_step)
        with patch("gymkhana.jax_sampler_ppo.sampler_ppo.compute_gae", wraps=compute_gae) as gae:
            waiting, metrics = update(state.replace(env_steps=jnp.asarray(100)))
        self.assertEqual([call.kwargs["discounting"] for call in gae.call_args_list], [.99, 1.])
        self.assertEqual(float(waiting.lambda_lagr), 0.)
        self.assertEqual(float(metrics["training/episode/cost_feedback_valid"]), 0.)
        ready = state.replace(lagrangian_pending_cost_sum=jnp.asarray(384.),
                              lagrangian_pending_episode_count=jnp.asarray(384.))
        warming, metrics = update(ready)
        self.assertEqual(float(warming.lambda_lagr), 0.)
        self.assertEqual(float(warming.lagrangian_pending_episode_count), 0.)
        updated, metrics = update(ready.replace(env_steps=jnp.asarray(100)))
        self.assertGreater(float(updated.lambda_lagr), 0.9)
        self.assertEqual(float(metrics["training/episode/cost_feedback_valid"]), 1.)
        self.assertEqual(float(updated.lagrangian_pending_episode_count), 0.)

    def test_nominal_training_disables_sampling_without_changing_obs_shape(self):
        env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=32)
        wrapper = F1TenthAdvWrapper(env, make_domain_spec(ranges={"m": [0.8, 1.2]}),
                                  asymmetric_critic=True)
        trainer = SamplerPPOTrainer(wrapper, SamplerPPOConfig(
            domain_randomization=False, policy_hidden_layer_sizes=(8,),
            value_hidden_layer_sizes=(8,), cost_value_hidden_layer_sizes=(8,),
        ))
        sampler_state = trainer.sampler.init(jax.random.PRNGKey(0))
        for key in (1, 2):
            params, log_prob, _ = jax.jit(trainer._sample_training_dynamics)(
                sampler_state, jax.random.PRNGKey(key))
            np.testing.assert_array_equal(params, wrapper.nominal_dynamics_params)
            np.testing.assert_array_equal(log_prob, jnp.zeros(4))

    def test_first_episode_uses_all_envs_and_excludes_later_resets(self):
        costs = jnp.array([[1., 0., 0., 0.], [1., 0., 0., 0.]])
        dones = costs > 0
        mean, budget = jax.jit(lambda c, d: lagrange_cost_estimate(
            c, d, mode="first_episode", episode_steps=6144, budget=0.05,
        ))(costs, dones)
        np.testing.assert_allclose(mean, 0.25)  # one first collision / four envs
        np.testing.assert_allclose(budget, 0.05)

    def test_partial_cost_and_multiple_completions(self):
        update = jax.jit(completed_episode_costs)
        partial, sums, counts = update(jnp.array([[0.2], [0.3]]),
                                      jnp.array([[False], [False]]), jnp.zeros(1))
        np.testing.assert_allclose(partial, [0.5])
        np.testing.assert_allclose(sums, [0.0])
        np.testing.assert_allclose(counts, [0.0])
        partial, sums, counts = update(jnp.array([[0.5], [0.25], [0.1]]),
                                      jnp.array([[True], [True], [False]]), partial)
        np.testing.assert_allclose(partial, [0.1])
        np.testing.assert_allclose(sums, [1.25])
        np.testing.assert_allclose(counts, [2.0])

    def test_dr_change_preserves_full_state_and_history(self):
        env = JaxRaceEnv.from_track_name("Drift", batch_size=4, max_episode_steps=32)
        wrapper = F1TenthAdvWrapper(env, make_domain_spec(ranges={"m": [0.8, 1.2]}),
                                  obs_history_len=2, asymmetric_critic=True)
        state = wrapper.reset(jax.random.PRNGKey(1))
        state, _ = wrapper.step(state, jnp.zeros((4, 2)), jax.random.PRNGKey(2))
        changed = jax.jit(wrapper.with_dynamics_params)(state, state.dynamics_params * 1.01)
        for old, new in zip(jax.tree_util.tree_leaves(state.env_state),
                            jax.tree_util.tree_leaves(changed.env_state)):
            np.testing.assert_array_equal(old, new)
        np.testing.assert_array_equal(state.obs_history, changed.obs_history)
        np.testing.assert_array_equal(state.obs_delay_steps, changed.obs_delay_steps)
        np.testing.assert_array_equal(state.obs["actor_obs"], changed.obs["actor_obs"])
        self.assertFalse(np.array_equal(state.obs["value_obs"], changed.obs["value_obs"]))

    def test_cost_beta_waits_for_completed_episode(self):
        for fraction in (0.0, 0.5):
            sampler = RewardCostGMMVISampler(
                jnp.array([0.5, 0.5]), jnp.array([1.5, 1.5]),
                num_components=2, init_std=0.1, num_envs=4,
                reward_fraction=fraction, cost_budget=0.05, cost_dual_update="log",
            )
            state = sampler.init(jax.random.PRNGKey(1))
            x, _, ids = sampler.sample(state, jax.random.PRNGKey(2), 4)
            result = jax.jit(lambda st: sampler.update(
                st, x, jnp.ones(4), ids, jax.random.PRNGKey(3),
                cost_scores=jnp.zeros(4), cost_constraint_value=jnp.asarray(0.0),
                cost_constraint_valid=False,
            ))(state)
            np.testing.assert_array_equal(result.state.cost_dual_lambda, state.cost_dual_lambda)
            np.testing.assert_array_equal(result.state.cost_ema, state.cost_ema)
            self.assertEqual(int(result.state.num_updates), 1)


if __name__ == "__main__":
    unittest.main()
