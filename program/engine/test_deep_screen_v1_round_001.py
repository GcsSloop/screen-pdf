from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deep_screen_v1_round_001 import (
    _artifact_name,
    _build_round_comparison,
    build_round_paths,
    partition_rows_by_split,
    resolve_teacher_model_paths,
)


class DeepScreenV1Round001Tests(unittest.TestCase):
    def test_build_round_paths_uses_round_local_directories(self) -> None:
        paths = build_round_paths(Path("/tmp/work/training/runs/deep_screen_v1/round_001"))

        self.assertEqual(paths.round_root, Path("/tmp/work/training/runs/deep_screen_v1/round_001"))
        self.assertEqual(paths.data_root, Path("/tmp/work/training/runs/deep_screen_v1/round_001/data"))
        self.assertEqual(paths.checkpoints_root, Path("/tmp/work/training/runs/deep_screen_v1/round_001/checkpoints"))
        self.assertEqual(paths.reports_root, Path("/tmp/work/training/runs/deep_screen_v1/round_001/reports"))
        self.assertEqual(paths.artifacts_root, Path("/tmp/work/training/runs/deep_screen_v1/round_001/artifacts"))
        self.assertEqual(paths.logs_root, Path("/tmp/work/training/runs/deep_screen_v1/round_001/logs"))

    def test_resolve_teacher_model_paths_maps_frozen_aliases_to_runtime_files(self) -> None:
        repo_root = Path("/tmp/work")
        mapping = resolve_teacher_model_paths(
            repo_root,
            {
                "r3": "global_corner_model.pt",
                "v28": "local_corner_moe_coord_model.pt",
            },
        )

        self.assertEqual(mapping["r3"], repo_root / "models/runtime/global_corner_model.pt")
        self.assertEqual(mapping["v28"], repo_root / "models/runtime/local_corner_moe_coord_model.pt")

    def test_partition_rows_by_split_uses_project_membership(self) -> None:
        rows = [
            {"page_id": "p1", "project_slug": "project_001"},
            {"page_id": "p2", "project_slug": "project_002"},
            {"page_id": "p3", "project_slug": "project_003"},
            {"page_id": "p4", "project_slug": "project_999"},
        ]
        split_payload = {
            "train_projects": ["project_001"],
            "val_projects": ["project_003"],
            "holdout_projects": ["project_002"],
        }

        partitioned = partition_rows_by_split(rows, split_payload)

        self.assertEqual([row["page_id"] for row in partitioned["train"]], ["p1"])
        self.assertEqual([row["page_id"] for row in partitioned["val"]], ["p3"])
        self.assertEqual([row["page_id"] for row in partitioned["holdout"]], ["p2"])

    def test_artifact_name_uses_round_identifier(self) -> None:
        name = _artifact_name({"public_name": "deep_screen_v1", "round": "round_002"}, "student", ".pt")

        self.assertEqual(name, "deep_screen_v1_round_002_student.pt")

    def test_build_round_comparison_includes_holdout_decision(self) -> None:
        comparison = _build_round_comparison(
            {"point_error_mean": 0.0416, "point_le_0_01_ratio": 0.0, "avg_page_infer_ms": 3.31},
            {"point_error_mean": 0.0055, "point_le_0_01_ratio": 0.9342},
            {"point_error_mean": 0.007, "point_le_0_01_ratio": 0.8882},
        )

        self.assertEqual(comparison["decision"], "continue")
        self.assertIn("far from target", comparison["decision_reason"])
        self.assertAlmostEqual(comparison["delta_to_teacher"]["point_error_mean"], 0.0361, places=4)


if __name__ == "__main__":
    unittest.main()
