from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_candidate_expand_selector import (
    SelectorTrainingRow,
    _build_project_sample_weights,
    _candidate_training_utility,
    _row_pairwise_preferences,
    _select_training_label,
    train_linear_selector,
)


class TrainCandidateExpandSelectorTests(unittest.TestCase):
    def test_build_project_sample_weights_upweights_sparse_projects(self) -> None:
        rows = [
            SelectorTrainingRow(
                features=np.zeros((2, 3), dtype=np.float32),
                label_index=0,
                candidate_metrics=[],
                candidate_errors=[0.01, 0.02],
                page_id="a-1",
                project_name="project-a",
                baseline_index=0,
            ),
            SelectorTrainingRow(
                features=np.zeros((2, 3), dtype=np.float32),
                label_index=0,
                candidate_metrics=[],
                candidate_errors=[0.01, 0.02],
                page_id="a-2",
                project_name="project-a",
                baseline_index=0,
            ),
            SelectorTrainingRow(
                features=np.zeros((2, 3), dtype=np.float32),
                label_index=0,
                candidate_metrics=[],
                candidate_errors=[0.01, 0.02],
                page_id="a-3",
                project_name="project-a",
                baseline_index=0,
            ),
            SelectorTrainingRow(
                features=np.zeros((2, 3), dtype=np.float32),
                label_index=0,
                candidate_metrics=[],
                candidate_errors=[0.01, 0.02],
                page_id="b-1",
                project_name="project-b",
                baseline_index=0,
            ),
        ]

        weights = _build_project_sample_weights(rows, balance_power=1.0)

        self.assertLess(weights["project-a"], 1.0)
        self.assertGreater(weights["project-b"], 1.0)
        self.assertGreater(weights["project-b"], weights["project-a"])

    def test_select_training_label_keeps_baseline_when_gain_is_small_in_same_bucket(self) -> None:
        candidate_metrics = [
            {"point_error": 0.0099, "max_corner_error": 0.0285},
            {"point_error": 0.0091, "max_corner_error": 0.0277},
        ]

        label_index = _select_training_label(
            candidate_metrics,
            baseline_index=0,
            preserve_point_margin=0.0015,
            preserve_max_margin=0.0020,
        )

        self.assertEqual(label_index, 0)

    def test_select_training_label_switches_when_candidate_crosses_guardrail_bucket(self) -> None:
        candidate_metrics = [
            {"point_error": 0.0115, "max_corner_error": 0.0340},
            {"point_error": 0.0107, "max_corner_error": 0.0290},
        ]

        label_index = _select_training_label(
            candidate_metrics,
            baseline_index=0,
            preserve_point_margin=0.0020,
            preserve_max_margin=0.0030,
        )

        self.assertEqual(label_index, 1)

    def test_select_training_label_keeps_baseline_when_same_bucket_only_improves_one_metric(self) -> None:
        candidate_metrics = [
            {"point_error": 0.0200, "max_corner_error": 0.0500},
            {"point_error": 0.0150, "max_corner_error": 0.0490},
        ]

        label_index = _select_training_label(
            candidate_metrics,
            baseline_index=0,
            preserve_point_margin=0.0015,
            preserve_max_margin=0.0020,
        )

        self.assertEqual(label_index, 0)

    def test_train_linear_selector_persists_balance_and_switch_metadata(self) -> None:
        rows = [
            SelectorTrainingRow(
                features=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                label_index=0,
                candidate_metrics=[
                    {"point_error": 0.0090, "max_corner_error": 0.0240},
                    {"point_error": 0.0110, "max_corner_error": 0.0330},
                ],
                candidate_errors=[0.0090, 0.0110],
                page_id="a-1",
                project_name="project-a",
                baseline_index=1,
            ),
            SelectorTrainingRow(
                features=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
                label_index=1,
                candidate_metrics=[
                    {"point_error": 0.0120, "max_corner_error": 0.0310},
                    {"point_error": 0.0090, "max_corner_error": 0.0240},
                ],
                candidate_errors=[0.0120, 0.0090],
                page_id="b-1",
                project_name="project-b",
                baseline_index=0,
            ),
        ]

        result = train_linear_selector(
            rows,
            epochs=2,
            learning_rate=0.05,
            weight_decay=1e-4,
            balance_power=0.75,
            switch_margin=0.12,
        )

        self.assertEqual(result["balance_power"], 0.75)
        self.assertEqual(result["switch_margin"], 0.12)
        self.assertEqual(result["project_counts"]["project-a"], 1)
        self.assertEqual(result["project_counts"]["project-b"], 1)

    def test_candidate_training_utility_prioritizes_guardrail_crossing(self) -> None:
        baseline_utility = _candidate_training_utility(
            {"point_error": 0.0105, "max_corner_error": 0.0310},
            baseline_metrics={"point_error": 0.0105, "max_corner_error": 0.0310},
        )
        improved_utility = _candidate_training_utility(
            {"point_error": 0.0108, "max_corner_error": 0.0285},
            baseline_metrics={"point_error": 0.0105, "max_corner_error": 0.0310},
        )

        self.assertGreater(improved_utility, baseline_utility)

    def test_train_linear_selector_pairwise_mode_records_objective_metadata(self) -> None:
        rows = [
            SelectorTrainingRow(
                features=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                label_index=1,
                candidate_metrics=[
                    {"point_error": 0.0108, "max_corner_error": 0.0310},
                    {"point_error": 0.0098, "max_corner_error": 0.0280},
                ],
                candidate_errors=[0.0108, 0.0098],
                page_id="pairwise-a",
                project_name="project-a",
                baseline_index=0,
            ),
        ]

        result = train_linear_selector(
            rows,
            epochs=10,
            learning_rate=0.05,
            weight_decay=1e-4,
            objective_mode="pairwise",
        )

        self.assertEqual(result["objective_mode"], "pairwise")
        self.assertGreaterEqual(result["train_pairwise_preference_accuracy"], 0.0)

    def test_train_linear_selector_pairwise_mode_can_learn_simple_switch(self) -> None:
        rows = [
            SelectorTrainingRow(
                features=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                label_index=1,
                candidate_metrics=[
                    {"point_error": 0.0120, "max_corner_error": 0.0340},
                    {"point_error": 0.0080, "max_corner_error": 0.0240},
                ],
                candidate_errors=[0.0120, 0.0080],
                page_id="pairwise-b",
                project_name="project-a",
                baseline_index=0,
            ),
            SelectorTrainingRow(
                features=np.array([[0.0, 0.0], [1.2, 1.1]], dtype=np.float32),
                label_index=1,
                candidate_metrics=[
                    {"point_error": 0.0110, "max_corner_error": 0.0320},
                    {"point_error": 0.0085, "max_corner_error": 0.0250},
                ],
                candidate_errors=[0.0110, 0.0085],
                page_id="pairwise-c",
                project_name="project-b",
                baseline_index=0,
            ),
        ]

        result = train_linear_selector(
            rows,
            epochs=60,
            learning_rate=0.05,
            weight_decay=1e-4,
            objective_mode="pairwise",
        )

        self.assertGreaterEqual(result["train_candidate_accuracy"], 1.0)

    def test_train_linear_selector_hybrid_mode_records_pairwise_weight(self) -> None:
        rows = [
            SelectorTrainingRow(
                features=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                label_index=1,
                candidate_metrics=[
                    {"point_error": 0.0120, "max_corner_error": 0.0340},
                    {"point_error": 0.0080, "max_corner_error": 0.0240},
                ],
                candidate_errors=[0.0120, 0.0080],
                page_id="hybrid-a",
                project_name="project-a",
                baseline_index=0,
            ),
        ]

        result = train_linear_selector(
            rows,
            epochs=10,
            learning_rate=0.05,
            weight_decay=1e-4,
            objective_mode="hybrid",
            pairwise_weight=0.3,
        )

        self.assertEqual(result["objective_mode"], "hybrid")
        self.assertEqual(result["pairwise_weight"], 0.3)

    def test_row_pairwise_preferences_only_compares_selected_label_against_other_candidates(self) -> None:
        row = SelectorTrainingRow(
            features=np.array([[0.0, 0.0], [0.4, 0.4], [0.9, 0.9]], dtype=np.float32),
            label_index=2,
            candidate_metrics=[
                {"point_error": 0.0140, "max_corner_error": 0.0400},
                {"point_error": 0.0115, "max_corner_error": 0.0330},
                {"point_error": 0.0085, "max_corner_error": 0.0240},
            ],
            candidate_errors=[0.0140, 0.0115, 0.0085],
            page_id="pairwise-label-only",
            project_name="project-a",
            baseline_index=0,
        )

        pairs = _row_pairwise_preferences(row)

        self.assertEqual([(winner, loser) for winner, loser, _ in pairs], [(2, 0), (2, 1)])


if __name__ == "__main__":
    unittest.main()
