from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_experiment_registry import compare_suite_metrics, upsert_registry_entry


class ModelExperimentRegistryTests(unittest.TestCase):
    def test_compare_suite_metrics_flags_regression_for_higher_error_and_lower_hit_rate(self) -> None:
        baseline = {
            "screen_relative_error_mean": 0.0100,
            "max_corner_error_mean": 0.0200,
            "perspective_tilt_error_mean": 0.3000,
            "quad_inset_ratio_abs_mean": 0.0200,
            "point_le_0_01_ratio": 0.8500,
        }
        candidate = {
            "screen_relative_error_mean": 0.0145,
            "max_corner_error_mean": 0.0280,
            "perspective_tilt_error_mean": 0.9000,
            "quad_inset_ratio_abs_mean": 0.0600,
            "point_le_0_01_ratio": 0.7800,
        }

        comparison = compare_suite_metrics(candidate, baseline)

        self.assertTrue(comparison["has_regression"])
        self.assertIn("screen_relative_error_mean", comparison["regressions"])
        self.assertIn("point_le_0_01_ratio", comparison["regressions"])

    def test_upsert_registry_entry_preserves_history_and_replaces_same_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            first = {"experiment_id": "exp-a", "stage": "global", "status": "accepted"}
            second = {"experiment_id": "exp-b", "stage": "global", "status": "rejected"}
            replacement = {"experiment_id": "exp-a", "stage": "global", "status": "archived"}

            upsert_registry_entry(registry_path, first)
            upsert_registry_entry(registry_path, second)
            upsert_registry_entry(registry_path, replacement)

            data = json.loads(registry_path.read_text(encoding="utf-8"))

        self.assertEqual([item["experiment_id"] for item in data["experiments"]], ["exp-a", "exp-b"])
        self.assertEqual(data["experiments"][0]["status"], "archived")
        self.assertEqual(data["experiments"][1]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
