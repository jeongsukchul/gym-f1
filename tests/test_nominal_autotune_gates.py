"""The automatic search must not confuse idle safety with successful racing."""

import unittest
import os
from unittest.mock import patch

from tune_silverstone_nominal_lagrange import (BUDGET, REWARD_FLOOR, OUT, SCREEN_STEPS, CONFIRM_STEPS,
    eligible, wilson_upper, refinement_center, training_environment, confirmation_jobs, confirmed_success,
    finite_episode_candidates, next_screen_job, credit_horizon_candidates,
    search_parameters, continuation_job, continuation_for_gpu, stabilization_candidates)


class TestNominalAutotuneGates(unittest.TestCase):
    def test_near_feasible_long_checkpoint_continues_without_cold_start(self):
        record = dict(steps=200024064, reward=3436., collision_rate=.063,
                      tuning_label="fixture", candidate=dict(lr=.5, std_end=-1.5))
        job = continuation_job(record)
        self.assertEqual(job["steps"], 400000000)
        self.assertEqual(job["gpu"], 2)
        self.assertEqual(job["eval_seed"], 3000)
        self.assertEqual(job["candidate"]["resume_from_update"], 8139)
        self.assertEqual(job["candidate"]["std_start"], -1.5)
        self.assertEqual(job["candidate"]["fixed_optimizer_lr"], 5e-5)
        self.assertEqual(search_parameters(job["candidate"]), record["candidate"])
        self.assertIsNone(continuation_job(dict(record, reward=10.)))
        self.assertIsNone(continuation_job(dict(record, collision_rate=.2)))
        self.assertIsNone(continuation_job(dict(record, steps=60014592)))
        self.assertFalse(any("resume_checkpoint" in c for c in credit_horizon_candidates(job["candidate"])))
        jobs = [continuation_for_gpu(record, gpu) for gpu in (2, 3, 4)]
        self.assertEqual([j["gpu"] for j in jobs], [2, 3, 4])
        self.assertEqual([j["candidate"]["lr"] for j in jobs], [.5, .1, 1.])
        self.assertEqual(len({j["label"] for j in jobs}), 3)
        self.assertEqual(len({j["candidate"]["resume_checkpoint"] for j in jobs}), 1)
        self.assertTrue(all(j["steps"] == 400000000 and j["eval_seed"] == 3000 for j in jobs))
        self.assertEqual(record["candidate"]["lr"], .5)
        with self.assertRaisesRegex(ValueError, "GPUs"):
            continuation_for_gpu(record, 1)
        later = continuation_job(dict(record, steps=400023552))
        self.assertEqual(later["steps"], 600000000)
        floors = stabilization_candidates(record)
        self.assertEqual([c["cost_advantage_std_floor"] for c in floors], [.001, .01, .03])
        self.assertTrue(all(c["normalize_cost_advantage"] and c["lr"] == 1. for c in floors))
        self.assertTrue(all(c["continuation_target_steps"] == 400000000 for c in floors))
        self.assertNotIn("continuation_target_steps", search_parameters(floors[0]))
        self.assertEqual(len({c["resume_checkpoint"] for c in floors}), 1)

    def test_credit_horizon_search_preserves_budget_gate_and_raw_units(self):
        center = dict(lr=.5, initial=0., ema=.5, episode_batch=384,
                      warmup=10000000, std_end=-1.5)
        candidates = credit_horizon_candidates(center)
        self.assertEqual([c["cost_discounting"] for c in candidates], [.99, .999, 1.])
        for candidate in candidates:
            self.assertTrue(candidate["cost_terminal_at_time_limit"])
            self.assertFalse(candidate["normalize_cost_advantage"])
            self.assertEqual(candidate["max_lambda"], 1000.)
            self.assertEqual(candidate["lr"], 2.5)
            self.assertEqual(candidate["warmup"], 30000000)
            self.assertEqual(candidate["episode_batch"], center["episode_batch"])
            self.assertEqual(candidate["std_end"], center["std_end"])
        self.assertNotIn("max_lambda", center)
        self.assertEqual(center["warmup"], 10000000)
        self.assertEqual(BUDGET, .05)

    def test_freed_gpus_run_persistable_next_generation_specs(self):
        candidates = finite_episode_candidates(dict(lr=.5, initial=0., ema=.5))
        self.assertEqual([c["lr"] for c in candidates], [.5, 2.5, 5.])
        self.assertTrue(all(c["cost_terminal_at_time_limit"] for c in candidates))
        self.assertTrue(all(c["cost_discounting"] == 1. and not c["normalize_cost_advantage"]
                            for c in candidates))
        for gpu in (2, 3, 4):
            job = next_screen_job(2, gpu, candidates)
            self.assertEqual(job["candidate"], candidates[gpu-2])
            self.assertEqual(job["label"], f"g03_screen{gpu-2}")
            self.assertEqual(job["gpu"], gpu)
            self.assertEqual(job["steps"], SCREEN_STEPS)
            self.assertEqual(job["eval_seed"], 1000)
        credit_candidates = credit_horizon_candidates(candidates[0])
        for gpu in (2, 3, 4):
            job = next_screen_job(3, gpu, credit_candidates)
            self.assertEqual(job["label"], f"g04_screen{gpu-2}")
            self.assertEqual(job["candidate"], credit_candidates[gpu-2])
            self.assertEqual(job["gpu"], gpu)
            self.assertEqual(job["steps"], SCREEN_STEPS)
        self.assertIsNone(next_screen_job(4, 3, candidates))

    def test_long_study_uses_all_three_gpus_without_changing_success_gate(self):
        record = dict(reward=2683., collision_rate=.34, candidate={"lr": .5, "std_end": -1.5})
        jobs = confirmation_jobs(2, [record], [record])
        self.assertEqual([j["gpu"] for j in jobs], [2, 3, 4])
        self.assertEqual([j["steps"] for j in jobs], [CONFIRM_STEPS, SCREEN_STEPS, SCREEN_STEPS])
        self.assertEqual(len({j["label"] for j in jobs}), 3)
        self.assertFalse(jobs[1]["candidate"]["normalize_cost_advantage"])
        self.assertEqual(jobs[2]["candidate"]["cost_discounting"], 1.)
        short = dict(reward=2000., collision_rate=.04, stable_last_two=True,
                     collision_wilson_upper95=.045, steps=SCREEN_STEPS)
        self.assertFalse(confirmed_success(short))
        self.assertTrue(confirmed_success(dict(short, steps=CONFIRM_STEPS)))

    def test_feasible_candidates_confirm_in_parallel(self):
        records = [dict(reward=2000.+i, collision_rate=.04, candidate={"lr": .1*(i+1)}) for i in range(3)]
        jobs = confirmation_jobs(1, records, records)
        self.assertEqual([j["gpu"] for j in jobs], [2, 3, 4])
        self.assertTrue(all(j["steps"] == CONFIRM_STEPS for j in jobs))
        self.assertEqual(confirmation_jobs(1, [], records), [])

    def test_warm_long_studies_do_not_trigger_redundant_cold_confirmation(self):
        source = dict(reward=3436., collision_rate=.063, steps=200024064,
                      candidate=dict(lr=.5, std_end=-1.5))
        screened = [dict(reward=2500., collision_rate=.4, steps=400023552,
                         candidate=dict(lr=1., resume_checkpoint="fixture.msgpack"))]
        self.assertEqual(confirmation_jobs(5, screened, [source]+screened), [])
        feasible = dict(screened[0], collision_rate=.04)
        self.assertEqual(confirmation_jobs(5, [feasible], [source, feasible]), [])

    def test_training_gpu_isolation_and_shared_cache(self):
        with patch.dict(os.environ, {"WANDB_SERVICE": "parent-service-connection"}):
            for gpu in (2, 3, 4):
                env = training_environment(gpu)
                self.assertNotIn("WANDB_SERVICE", env)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], str(gpu))
                self.assertEqual(env["JAX_COMPILATION_CACHE_DIR"], str(OUT / "jax_compilation_cache"))
                self.assertEqual(int(env["JAX_COMPILATION_CACHE_MAX_SIZE"]), 2 * 1024**3)

    def test_idle_and_unsafe_policies_fail_joint_gate(self):
        self.assertFalse(eligible(dict(reward=2., collision_rate=0.)))
        self.assertFalse(eligible(dict(reward=2500., collision_rate=0.5)))
        self.assertTrue(eligible(dict(reward=REWARD_FLOOR, collision_rate=BUDGET)))

    def test_confirmation_confidence_is_stricter_than_observed_budget(self):
        self.assertGreater(wilson_upper(BUDGET, 10240), BUDGET)
        self.assertLess(wilson_upper(0.04, 10240), BUDGET)
        self.assertGreater(wilson_upper(0., 10240), 0.)

    def test_refinement_does_not_center_on_idle_safety(self):
        idle = dict(reward=2., collision_rate=0.)
        learning = dict(reward=900., collision_rate=0.98)
        self.assertIs(refinement_center([idle, learning]), learning)
        racer = dict(reward=1800., collision_rate=0.1)
        self.assertIs(refinement_center([idle, learning, racer]), racer)
        safe_racer = dict(reward=1600., collision_rate=0.04)
        self.assertIs(refinement_center([racer, safe_racer]), safe_racer)


if __name__ == "__main__":
    unittest.main()
