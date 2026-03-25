from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from global_corner_bootstrap_weights import (
    bucketed_adaptive_weight,
    compute_bootstrap_hardness,
    continuous_adaptive_weight,
    rebalance_project_mean_weights,
)


class GlobalCornerBootstrapWeightsTests(unittest.TestCase):
    def test_compute_bootstrap_hardness_uses_worst_metric_ratio(self) -> None:
        metrics = {
            "screen_relative_point_error": 0.012,
            "max_corner_error": 0.018,
            "quad_inset_ratio": 0.095,
        }

        hardness = compute_bootstrap_hardness(metrics)

        self.assertAlmostEqual(hardness, 0.095 / 0.03, places=6)

    def test_bucketed_adaptive_weight_caps_hardness(self) -> None:
        easy = bucketed_adaptive_weight(1.0)
        medium = bucketed_adaptive_weight(1.7)
        very_hard = bucketed_adaptive_weight(8.0)

        self.assertEqual(easy, 1.0)
        self.assertGreater(medium, easy)
        self.assertEqual(very_hard, 2.05)

    def test_continuous_adaptive_weight_can_exceed_bucketed_cap(self) -> None:
        continuous = continuous_adaptive_weight(8.0)
        bucketed = bucketed_adaptive_weight(8.0)

        self.assertGreater(continuous, bucketed)
        self.assertEqual(continuous, 3.5)

    def test_rebalance_project_mean_weights_caps_project_average(self) -> None:
        rows = [
            {"project_name": "A", "adaptive_weight": 2.05},
            {"project_name": "A", "adaptive_weight": 1.75},
            {"project_name": "A", "adaptive_weight": 1.45},
            {"project_name": "B", "adaptive_weight": 1.0},
            {"project_name": "B", "adaptive_weight": 1.2},
        ]

        adjusted = rebalance_project_mean_weights(rows, max_project_mean=1.35)

        project_a = [float(row["adaptive_weight"]) for row in adjusted if row["project_name"] == "A"]
        project_b = [float(row["adaptive_weight"]) for row in adjusted if row["project_name"] == "B"]
        self.assertLessEqual(sum(project_a) / len(project_a), 1.35 + 1e-6)
        self.assertAlmostEqual(sum(project_b) / len(project_b), 1.1, places=6)
        self.assertGreaterEqual(min(project_a), 1.0)


if __name__ == "__main__":
    unittest.main()
