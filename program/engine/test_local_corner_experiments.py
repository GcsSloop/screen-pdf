from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_experiments import compute_targeted_adaptive_weight, summarize_project_errors


class LocalCornerExperimentsTests(unittest.TestCase):
    def test_summarize_project_errors_sorts_by_failures_then_mean(self) -> None:
        rows = [
            {"project_name": "A", "page_error": 0.031},
            {"project_name": "A", "page_error": 0.029},
            {"project_name": "A", "page_error": 0.004},
            {"project_name": "B", "page_error": 0.041},
            {"project_name": "B", "page_error": 0.011},
            {"project_name": "C", "page_error": 0.033},
            {"project_name": "C", "page_error": 0.032},
        ]

        summary = summarize_project_errors(rows)

        self.assertEqual([item["project_name"] for item in summary[:3]], ["C", "B", "A"])
        self.assertEqual(summary[0]["fail_gt_0_03"], 2)
        self.assertAlmostEqual(summary[1]["mean_page_error"], 0.026, places=4)

    def test_compute_targeted_adaptive_weight_focuses_hard_pages_and_caps_value(self) -> None:
        base_row = {
            "project_name": "融合赋能引领智控新纪元",
            "corner_index": 2,
            "target_residual_norm": [0.1, 0.25],
            "adaptive_weight": 1.8,
        }

        boosted = compute_targeted_adaptive_weight(
            row=base_row,
            page_error=0.082,
            corner_error=0.094,
            focus_projects={"融合赋能引领智控新纪元"},
        )
        easy = compute_targeted_adaptive_weight(
            row={**base_row, "project_name": "稳定项目", "corner_index": 0, "target_residual_norm": [0.0, 0.01], "adaptive_weight": 1.0},
            page_error=0.006,
            corner_error=0.005,
            focus_projects={"融合赋能引领智控新纪元"},
        )

        self.assertGreater(boosted, 2.0)
        self.assertLessEqual(boosted, 2.35)
        self.assertAlmostEqual(easy, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
