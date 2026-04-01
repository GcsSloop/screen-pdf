from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_runtime_failure_layers import (
    classify_failure_layer,
    select_best_metric_record,
    summarize_diagnostic_rows,
)


class AnalyzeRuntimeFailureLayersTests(unittest.TestCase):
    def test_select_best_metric_record_prefers_threshold_hit_before_lower_error(self) -> None:
        records = [
            {
                "source": "runtime",
                "metrics": {"point_error": 0.0098, "max_corner_error": 0.0312},
            },
            {
                "source": "opencv",
                "metrics": {"point_error": 0.0104, "max_corner_error": 0.0281},
            },
        ]

        best = select_best_metric_record(records)

        self.assertEqual(best["source"], "opencv")

    def test_classify_failure_layer_marks_runtime_candidate_recoverable(self) -> None:
        category = classify_failure_layer(
            baseline_metrics={"point_error": 0.0180, "max_corner_error": 0.0600},
            runtime_oracle_metrics={"point_error": 0.0090, "max_corner_error": 0.0250},
            opencv_oracle_metrics={"point_error": 0.0200, "max_corner_error": 0.0550},
        )

        self.assertEqual(category, "runtime_candidate_recoverable")

    def test_classify_failure_layer_marks_opencv_recoverable_when_runtime_oracle_still_bad(self) -> None:
        category = classify_failure_layer(
            baseline_metrics={"point_error": 0.0180, "max_corner_error": 0.0600},
            runtime_oracle_metrics={"point_error": 0.0150, "max_corner_error": 0.0410},
            opencv_oracle_metrics={"point_error": 0.0090, "max_corner_error": 0.0260},
        )

        self.assertEqual(category, "opencv_recoverable")

    def test_classify_failure_layer_marks_hard_when_both_oracles_fail(self) -> None:
        category = classify_failure_layer(
            baseline_metrics={"point_error": 0.0180, "max_corner_error": 0.0600},
            runtime_oracle_metrics={"point_error": 0.0160, "max_corner_error": 0.0450},
            opencv_oracle_metrics={"point_error": 0.0130, "max_corner_error": 0.0380},
        )

        self.assertEqual(category, "hard_both_fail")

    def test_summarize_diagnostic_rows_counts_categories_and_pass_rates(self) -> None:
        rows = [
            {
                "category": "baseline_ok",
                "baseline_metrics": {"point_error": 0.0080, "max_corner_error": 0.0200},
                "runtime_oracle_metrics": {"point_error": 0.0080, "max_corner_error": 0.0200},
                "opencv_oracle_metrics": {"point_error": 0.0120, "max_corner_error": 0.0400},
                "union_oracle_metrics": {"point_error": 0.0080, "max_corner_error": 0.0200},
            },
            {
                "category": "runtime_candidate_recoverable",
                "baseline_metrics": {"point_error": 0.0180, "max_corner_error": 0.0500},
                "runtime_oracle_metrics": {"point_error": 0.0080, "max_corner_error": 0.0200},
                "opencv_oracle_metrics": {"point_error": 0.0140, "max_corner_error": 0.0400},
                "union_oracle_metrics": {"point_error": 0.0080, "max_corner_error": 0.0200},
            },
            {
                "category": "opencv_recoverable",
                "baseline_metrics": {"point_error": 0.0180, "max_corner_error": 0.0500},
                "runtime_oracle_metrics": {"point_error": 0.0150, "max_corner_error": 0.0410},
                "opencv_oracle_metrics": {"point_error": 0.0080, "max_corner_error": 0.0250},
                "union_oracle_metrics": {"point_error": 0.0080, "max_corner_error": 0.0250},
            },
            {
                "category": "hard_both_fail",
                "baseline_metrics": {"point_error": 0.0220, "max_corner_error": 0.0610},
                "runtime_oracle_metrics": {"point_error": 0.0160, "max_corner_error": 0.0450},
                "opencv_oracle_metrics": {"point_error": 0.0130, "max_corner_error": 0.0380},
                "union_oracle_metrics": {"point_error": 0.0130, "max_corner_error": 0.0380},
            },
        ]

        summary = summarize_diagnostic_rows(rows)

        self.assertEqual(summary["pages"], 4)
        self.assertEqual(summary["category_counts"]["baseline_ok"], 1)
        self.assertEqual(summary["category_counts"]["runtime_candidate_recoverable"], 1)
        self.assertEqual(summary["category_counts"]["opencv_recoverable"], 1)
        self.assertEqual(summary["category_counts"]["hard_both_fail"], 1)
        self.assertEqual(summary["baseline_strict_ratio"], 0.25)
        self.assertEqual(summary["runtime_oracle_strict_ratio"], 0.5)
        self.assertEqual(summary["opencv_oracle_strict_ratio"], 0.25)
        self.assertEqual(summary["union_oracle_strict_ratio"], 0.75)


if __name__ == "__main__":
    unittest.main()
