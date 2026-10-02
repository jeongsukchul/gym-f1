import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from run_silverstone_dr_comparison import SOURCE, FORMULATIONS, comparison_config, child_environment, comparison_cases


class TestDRComparison(unittest.TestCase):
    def test_wider_dr_scales_5_10_20_have_eight_matched_cases(self):
        cases = comparison_cases([5., 10., 20.])
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({label for label, _, _ in cases}), 8)
        base = yaml.safe_load((SOURCE / "g06_screen1/rl_config.yaml").read_text())
        for _, formulation, scale in cases:
            c = comparison_config(base, formulation, dr_strength=(4. / 3.) * 1.1,
                                  cost_scale=scale)
            jc = c["jax_sampler_ppo"]
            for value, expected in zip(jc["domain_randomization_ranges"]["m"], [.78, 1.22]):
                self.assertAlmostEqual(value, expected)
            for value, expected in zip(jc["domain_randomization_ranges"]["tire_p_dx1"],
                                       [1. - 1.1 / 3., 1. + 1.1 / 3.]):
                self.assertAlmostEqual(value, expected)
            self.assertEqual(jc["gmm_cost_score_scale"], scale if scale is not None else 10.)
            self.assertEqual(jc["safety_bound"], .05)
            self.assertEqual(c["seed"], 1)
            self.assertEqual(c["total_timesteps"], 400000000)

    def test_stronger_dr_scale_comparison_has_six_matched_cases(self):
        cases = comparison_cases([10., 20.])
        self.assertEqual(len(cases), 6)
        self.assertEqual(len({label for label, _, _ in cases}), 6)
        self.assertEqual([(f, s) for _, f, s in cases],
                         [("udr", None), ("reward", None), ("cost", 10.), ("cost", 20.),
                          ("reward_cost", 10.), ("reward_cost", 20.)])
        base = yaml.safe_load((SOURCE / "g06_screen1/rl_config.yaml").read_text())
        for _, f, scale in cases:
            c = comparison_config(base, f, dr_strength=4./3., cost_scale=scale)
            jc = c["jax_sampler_ppo"]
            self.assertEqual(jc["domain_randomization_profile"], "stronger")
            self.assertAlmostEqual(jc["domain_randomization_ranges"]["m"][0], .8)
            self.assertAlmostEqual(jc["domain_randomization_ranges"]["tire_p_dx1"][1], 4./3.)
            self.assertEqual(jc["gmm_cost_score_scale"], scale or 10.)
            self.assertEqual(jc["safety_bound"], .05)
            self.assertEqual(jc["lagrangian_coef_rate"], 1.)
            self.assertNotIn("resume_checkpoint", c)
        self.assertEqual(len(comparison_cases([10., 10., 20.])), 6)
        with self.assertRaises(ValueError):
            comparison_cases([0.])

    def test_matched_fresh_configs_keep_winning_ppo_lag_settings(self):
        base = yaml.safe_load((SOURCE / "g06_screen1/rl_config.yaml").read_text())
        for formulation in FORMULATIONS:
            c = comparison_config(base, formulation)
            self.assertNotIn("resume_checkpoint", c)
            self.assertNotIn("resume_from_update", c)
            self.assertEqual((c["seed"], c["rollout_length"], c["batch_size"]), (1, 256, 1024))
            self.assertEqual(c["total_timesteps"], 400000000)
            self.assertEqual(c["start_learning_rate"], 5e-6)
            jc = c["jax_sampler_ppo"]
            self.assertTrue(jc["domain_randomization"] and jc["eval_domain_randomization"])
            self.assertEqual(jc["domain_randomization_ranges"], jc["domain_randomization_profiles"]["narrow"])
            self.assertEqual(jc["lagrangian_update_mode"], "completed_episode")
            self.assertEqual(jc["lagrangian_min_completed_episodes"], 384)
            self.assertEqual(jc["safety_bound"], .05)
            self.assertFalse(jc["reset_state_on_rollout"])
            self.assertEqual(jc["cost_advantage_std_floor"], .001)
            self.assertEqual(jc["eval_episode_steps"], 12288)
            self.assertEqual(jc["num_eval_envs"] * jc["eval_episodes_per_dynamics"], 10240)
            self.assertEqual(jc["sampler"], "uniform" if formulation == "udr" else "reward_cost_gmmvi")
        self.assertIn("resume_checkpoint", base)
        self.assertFalse(base["jax_sampler_ppo"]["domain_randomization"])

    def test_child_gpu_and_wandb_service_isolation(self):
        with patch.dict("os.environ", {"WANDB_SERVICE": "parent"}):
            for gpu in (2, 3, 4):
                e = child_environment(gpu, Path("fixture"))
                self.assertNotIn("WANDB_SERVICE", e)
                self.assertEqual(e["CUDA_VISIBLE_DEVICES"], str(gpu))
        with self.assertRaises(ValueError):
            child_environment(0, Path("fixture"))


if __name__ == "__main__":
    unittest.main()
