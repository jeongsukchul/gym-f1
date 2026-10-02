"""Exact final metric recovery, without restarting training or publishing tests."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import recover_nominal_trial as recovery
from train.jax_sampler_ppo import _should_save_eval_video


class TestNominalRecovery(unittest.TestCase):
    def test_disabled_video_stays_disabled_at_final_evaluation(self):
        for enabled in (False, True):
            for final_only in (False, True):
                config = SimpleNamespace(eval_video=enabled, eval_video_final=final_only)
                for final in (False, True):
                    self.assertEqual(_should_save_eval_video(config, is_final=final),
                                     enabled and (not final_only or final))

    def exercise(self, directory, *, local_summary):
        metrics = {"evaluation/reward/episode_return_mean": 1700.123456,
                   "evaluation/collision/rate": 0.041234567,
                   "evaluation/reward/cvar10": 1200.123456,
                   "evaluation/meta/update": 2442,
                   "training/progress/env_steps": 60014592}
        if local_summary:
            (directory / "checkpoint.summary.json").write_text(json.dumps(metrics))
        state = SimpleNamespace(env_steps=60014592, update_steps=2442, lambda_lagr=.7, params=object())
        config = SimpleNamespace(num_eval_envs=1024, eval_episode_steps=12288,
                                 eval_episodes_per_dynamics=10, num_evals=6)
        restored = ({"seed": 1, "eval_seed": 1000}, {}, config, {}, object(), state, 2442)
        fake_wandb = MagicMock()
        fake_wandb.init.return_value.__enter__.return_value.id = "test-recovery"
        evaluate = MagicMock(return_value=(metrics, None, None, None))
        with patch.object(recovery, "restore_trial", return_value=restored), \
             patch.object(recovery, "_build_env") as build_env, \
             patch.object(recovery, "_make_eval_fn", return_value=evaluate), \
             patch.dict(sys.modules, {"wandb": fake_wandb}), \
             patch.object(sys, "argv", ["recover", "--trial-dir", str(directory),
                                       "--gym-config", str(directory / "gym.yaml"),
                                       "--original-run-id", "original-test"]):
            recovery.main()
        result = json.loads((directory / "recovered_summary.json").read_text())
        self.assertEqual(result["metrics"]["evaluation/collision/rate"], 0.041234567)
        self.assertEqual(result["metrics"]["evaluation/reward/episode_return_mean"], 1700.123456)
        self.assertEqual(result["original_run_id"], "original-test")
        self.assertEqual(result["run_id"], "test-recovery")
        self.assertEqual(result["metrics"]["evaluation/meta/checkpoint_reevaluated"], int(not local_summary))
        if local_summary:
            build_env.assert_not_called()
            evaluate.assert_not_called()
        else:
            evaluate.assert_called_once()
        fake_wandb.init.return_value.__enter__.return_value.log.assert_called_once()

    def test_saved_exact_summary_skips_reevaluation(self):
        with TemporaryDirectory() as tmp:
            self.exercise(Path(tmp), local_summary=True)

    def test_missing_summary_reevaluates_checkpoint(self):
        with TemporaryDirectory() as tmp:
            self.exercise(Path(tmp), local_summary=False)


if __name__ == "__main__":
    unittest.main()
