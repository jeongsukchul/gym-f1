"""Offline report checks: test fixtures are not training results."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from report_experiments import report_results


class TestExperimentReport(unittest.TestCase):
    def test_tuning_parameters_and_joint_gate_are_visible_without_publication(self):
        parameters = dict(lr=2.5, max_lambda=1000., warmup=30000000,
                          cost_discounting=.999, normalize_cost_advantage=False)
        record = dict(track="fixture", formulation="nominal", seed=1, reward=10.,
                      cvar10=0., collision_rate=.5, budget=.05, lambda_final=4.,
                      run_id="fixture_not_a_run", tuning_label="fixture", candidate=parameters,
                      eligible=False, steps=60014592)
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(report_results([record], directory, "Offline fixture", publish=False))
            with (Path(directory) / "report.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(json.loads(row["tuning_parameters"]), parameters)
            self.assertEqual(row["joint_goal_satisfied"], "False")
            self.assertEqual(row["steps"], "60014592")
            self.assertIn("max_lambda", (Path(directory) / "report.txt").read_text())
            self.assertFalse((Path(directory) / "report_publication.json").exists())

    def test_non_tuning_records_do_not_invent_goal_or_hyperparameters(self):
        record = dict(track="fixture", sampler="uniform", reward=10., cvar10=0.,
                      collision_rate=.5, budget=.05, lambda_final=4., run_id="fixture_not_a_run")
        with tempfile.TemporaryDirectory() as directory:
            report_results([record], directory, "Offline fixture", publish=False)
            with (Path(directory) / "report.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["tuning_parameters"], "")
            self.assertEqual(row["joint_goal_satisfied"], "")


if __name__ == "__main__":
    unittest.main()
