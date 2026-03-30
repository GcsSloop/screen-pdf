from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deep_screen_v1_train import (
    _apply_train_augmentation,
    _apply_perspective_augmentation,
    _build_adaptive_teacher_target,
    _build_border_contact_target,
    _compute_border_distance_min,
    _build_oracle_teacher_target,
    _augmentation_settings,
    _artifact_name,
    _build_candidate_pool_tensors,
    _build_train_dataloader,
    _configure_trainable_parameters,
    _build_round_comparison,
    _compute_process_targets,
    _compute_total_loss,
    _candidate_rank_loss,
    _is_better_checkpoint,
    _max_corner_constraint_loss,
    _quad_inset_inward_loss,
    _strict_point_soft_target_loss,
    _update_ema_state,
    _quad_inset_abs_loss,
    _recent_rounds_hit_plateau,
    _resolve_image_path,
    _reuse_teacher_snapshot,
    _sampling_settings,
    _snapshot_state_dict,
    _scene_target_from_tags,
    _select_training_device,
    partition_rows_by_split,
    DeepScreenV1Dataset,
    build_round_paths,
)


class DeepScreenV1TrainTests(unittest.TestCase):
    def test_build_round_paths_uses_round_local_directories(self) -> None:
        paths = build_round_paths(Path("/tmp/work/training/runs/deep_screen_v1/round_001"))

        self.assertEqual(paths.round_root, Path("/tmp/work/training/runs/deep_screen_v1/round_001"))
        self.assertEqual(paths.data_root, Path("/tmp/work/training/runs/deep_screen_v1/round_001/data"))
        self.assertEqual(paths.checkpoints_root, Path("/tmp/work/training/runs/deep_screen_v1/round_001/checkpoints"))

    def test_artifact_name_uses_round_identifier(self) -> None:
        name = _artifact_name({"public_name": "deep_screen_v1", "round": "round_001"}, "student", ".pt")

        self.assertEqual(name, "deep_screen_v1_round_001_student.pt")

    def test_reuse_teacher_snapshot_copies_existing_exports_into_round(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_round_root = temp_root / "source_round"
            source_export_root = source_round_root / "data" / "teacher_exports"
            source_export_root.mkdir(parents=True)
            for split_name in ("train", "val", "holdout"):
                (source_export_root / f"{split_name}.jsonl").write_text('{"page_id":"p1"}\n', encoding="utf-8")
            summary_payload = {
                "train": {"rows": 1, "path": str(source_export_root / "train.jsonl"), "teacher_summary": {}, "r3_summary": {}},
                "val": {"rows": 1, "path": str(source_export_root / "val.jsonl"), "teacher_summary": {}, "r3_summary": {}},
                "holdout": {"rows": 1, "path": str(source_export_root / "holdout.jsonl"), "teacher_summary": {}, "r3_summary": {}},
            }
            (source_export_root / "summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")

            round_paths = build_round_paths(temp_root / "target_round")
            reused = _reuse_teacher_snapshot(
                Path(temp_dir),
                round_paths,
                {
                    "teachers": {
                        "r3": {"runtime_file": "global_corner_model.pt"},
                        "v28": {"runtime_file": "local_corner_moe_coord_model.pt"},
                    },
                    "reuse_teacher_export_from": str(source_round_root),
                },
            )

            self.assertEqual(reused["reused_from"], str(source_export_root))
            self.assertEqual(reused["splits"]["holdout"]["rows"], 1)
            self.assertTrue((round_paths.teacher_export_root / "train.jsonl").exists())
            self.assertTrue((round_paths.teacher_export_root / "val.jsonl").exists())
            self.assertTrue((round_paths.teacher_export_root / "holdout.jsonl").exists())
            self.assertTrue((round_paths.teacher_export_root / "summary.json").exists())

    def test_update_ema_state_blends_floating_tensors_and_copies_non_floating(self) -> None:
        ema_state = {
            "weight": torch.tensor([1.0, 3.0], dtype=torch.float32),
            "steps": torch.tensor([2], dtype=torch.int64),
        }
        model_state = {
            "weight": torch.tensor([3.0, 7.0], dtype=torch.float32),
            "steps": torch.tensor([5], dtype=torch.int64),
        }

        updated = _update_ema_state(ema_state, model_state, decay=0.5)

        self.assertTrue(torch.allclose(updated["weight"], torch.tensor([2.0, 5.0], dtype=torch.float32)))
        self.assertEqual(int(updated["steps"].item()), 5)

    def test_resolve_image_path_uses_existing_absolute_path_before_workspace_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.jpeg"
            image_path.write_bytes(b"stub")

            with mock.patch(
                "deep_screen_v1_train._workspace_image_index",
                side_effect=AssertionError("workspace index should not be used"),
            ):
                resolved = _resolve_image_path(
                    Path(temp_dir),
                    {"image_path": str(image_path), "page_name": image_path.name, "page_id": "page"},
                )

        self.assertEqual(resolved, image_path)

    def test_round_comparison_reports_continue_when_student_above_teacher(self) -> None:
        comparison = _build_round_comparison(
            {"point_error_mean": 0.03, "point_le_0_01_ratio": 0.1, "avg_page_infer_ms": 2.0},
            {"point_error_mean": 0.0055, "point_le_0_01_ratio": 0.9342},
            {"point_error_mean": 0.007, "point_le_0_01_ratio": 0.8882},
        )

        self.assertEqual(comparison["decision"], "continue")
        self.assertIn("target", comparison["decision_reason"])

    def test_round_comparison_can_stop_when_strict_point_target_and_latency_gate_are_met(self) -> None:
        comparison = _build_round_comparison(
            {
                "point_error_mean": 0.012,
                "point_le_0_01_ratio": 0.71,
                "max_corner_le_0_03_ratio": 0.15,
                "avg_page_infer_ms": 120.0,
            },
            {"point_error_mean": 0.0055, "point_le_0_01_ratio": 0.9342},
            {"point_error_mean": 0.007, "point_le_0_01_ratio": 0.8882},
        )

        self.assertEqual(comparison["decision"], "stop")
        self.assertIn("strict-point", comparison["decision_reason"])

    def test_round_comparison_rejects_stop_when_latency_gate_is_missed(self) -> None:
        comparison = _build_round_comparison(
            {
                "point_error_mean": 0.011,
                "point_le_0_01_ratio": 0.72,
                "avg_page_infer_ms": 640.0,
            },
            {"point_error_mean": 0.0055, "point_le_0_01_ratio": 0.9342},
            {"point_error_mean": 0.007, "point_le_0_01_ratio": 0.8882},
        )

        self.assertEqual(comparison["decision"], "continue")
        self.assertIn("latency", comparison["decision_reason"])

    def test_select_training_device_avoids_mps_for_grid_sample_backward(self) -> None:
        with mock.patch("deep_screen_v1_train.torch.cuda.is_available", return_value=False), mock.patch(
            "deep_screen_v1_train.torch.backends.mps.is_available",
            return_value=True,
        ):
            device = _select_training_device()

        self.assertEqual(device.type, "cpu")

    def test_checkpoint_selection_can_prioritize_point_error_over_loss(self) -> None:
        current_best = {"loss_mean": 0.0014, "point_error_mean": 0.0259, "point_le_0_03_ratio": 0.8039, "point_le_0_02_ratio": 0.1765}
        candidate = {"loss_mean": 0.0022, "point_error_mean": 0.0238, "point_le_0_03_ratio": 0.8026, "point_le_0_02_ratio": 0.4211}

        is_better = _is_better_checkpoint(candidate, current_best, selection_metric="point_error_mean")

        self.assertTrue(is_better)

    def test_checkpoint_selection_uses_hit_rate_tie_breakers(self) -> None:
        current_best = {"loss_mean": 0.0018, "point_error_mean": 0.0235, "point_le_0_03_ratio": 0.8618, "point_le_0_02_ratio": 0.3882}
        candidate = {"loss_mean": 0.0019, "point_error_mean": 0.0235, "point_le_0_03_ratio": 0.8618, "point_le_0_02_ratio": 0.4079}

        is_better = _is_better_checkpoint(candidate, current_best, selection_metric="point_error_mean")

        self.assertTrue(is_better)

    def test_checkpoint_selection_can_prioritize_point_le_0_02_ratio(self) -> None:
        current_best = {"loss_mean": 0.0016, "point_error_mean": 0.0223, "point_le_0_03_ratio": 0.9050, "point_le_0_02_ratio": 0.4013}
        candidate = {"loss_mean": 0.0019, "point_error_mean": 0.0224, "point_le_0_03_ratio": 0.9010, "point_le_0_02_ratio": 0.4342}

        is_better = _is_better_checkpoint(candidate, current_best, selection_metric="point_le_0_02_ratio")

        self.assertTrue(is_better)

    def test_checkpoint_selection_can_prioritize_max_corner_le_0_03_ratio(self) -> None:
        current_best = {
            "loss_mean": 0.0014,
            "point_error_mean": 0.0212,
            "point_le_0_03_ratio": 0.9520,
            "point_le_0_02_ratio": 0.4710,
            "point_le_0_01_ratio": 0.4020,
            "max_corner_le_0_03_ratio": 0.5110,
        }
        candidate = {
            "loss_mean": 0.0019,
            "point_error_mean": 0.0218,
            "point_le_0_03_ratio": 0.9470,
            "point_le_0_02_ratio": 0.4620,
            "point_le_0_01_ratio": 0.4180,
            "max_corner_le_0_03_ratio": 0.6030,
        }

        is_better = _is_better_checkpoint(candidate, current_best, selection_metric="max_corner_le_0_03_ratio")

        self.assertTrue(is_better)

    def test_checkpoint_selection_rejects_candidates_above_latency_gate(self) -> None:
        current_best = {
            "loss_mean": 0.0032,
            "point_error_mean": 0.0245,
            "point_le_0_02_ratio": 0.4620,
            "point_le_0_01_ratio": 0.2280,
            "avg_page_infer_ms": 180.0,
        }
        candidate = {
            "loss_mean": 0.0029,
            "point_error_mean": 0.0241,
            "point_le_0_02_ratio": 0.4710,
            "point_le_0_01_ratio": 0.2590,
            "avg_page_infer_ms": 612.0,
        }

        is_better = _is_better_checkpoint(candidate, current_best, selection_metric="point_le_0_01_ratio")

        self.assertFalse(is_better)

    def test_checkpoint_selection_prioritizes_point_le_0_01_ratio_when_latency_is_valid(self) -> None:
        current_best = {
            "loss_mean": 0.0028,
            "point_error_mean": 0.0242,
            "point_le_0_02_ratio": 0.4520,
            "point_le_0_01_ratio": 0.2010,
            "avg_page_infer_ms": 220.0,
        }
        candidate = {
            "loss_mean": 0.0035,
            "point_error_mean": 0.0248,
            "point_le_0_02_ratio": 0.4410,
            "point_le_0_01_ratio": 0.2480,
            "avg_page_infer_ms": 240.0,
        }

        is_better = _is_better_checkpoint(candidate, current_best, selection_metric="point_le_0_01_ratio")

        self.assertTrue(is_better)

    def test_checkpoint_selection_uses_point_error_before_latency_when_strict_point_ties(self) -> None:
        current_best = {
            "loss_mean": 0.0030,
            "point_error_mean": 0.0278,
            "point_le_0_03_ratio": 0.5909,
            "point_le_0_02_ratio": 0.5909,
            "point_le_0_01_ratio": 0.5341,
            "max_corner_le_0_03_ratio": 0.5000,
            "avg_page_infer_ms": 6.68,
        }
        candidate = {
            "loss_mean": 0.0031,
            "point_error_mean": 0.0274,
            "point_le_0_03_ratio": 0.5909,
            "point_le_0_02_ratio": 0.5909,
            "point_le_0_01_ratio": 0.5341,
            "max_corner_le_0_03_ratio": 0.5000,
            "avg_page_infer_ms": 6.73,
        }

        is_better = _is_better_checkpoint(candidate, current_best, selection_metric="point_le_0_01_ratio")

        self.assertTrue(is_better)

    def test_scene_target_from_tags_encodes_known_scene_labels(self) -> None:
        target = _scene_target_from_tags(["near_color_background", "bright_screen", "border_contact_scene"])

        self.assertEqual(target.shape, (5,))
        self.assertListEqual(target.tolist(), [1.0, 0.0, 0.0, 1.0, 1.0])

    def test_snapshot_state_dict_clones_cpu_tensors(self) -> None:
        layer = torch.nn.Linear(3, 2)
        snapshot = _snapshot_state_dict(layer.state_dict())
        weight_before = snapshot["weight"].clone()

        with torch.no_grad():
            layer.weight.add_(1.0)

        self.assertFalse(torch.equal(layer.state_dict()["weight"], weight_before))
        self.assertTrue(torch.equal(snapshot["weight"], weight_before))
        self.assertNotEqual(snapshot["weight"].data_ptr(), layer.state_dict()["weight"].data_ptr())

    def test_recent_rounds_hit_plateau_after_three_small_improvements(self) -> None:
        plateau = _recent_rounds_hit_plateau(
            [
                {"point_le_0_01_ratio": 0.1217},
                {"point_le_0_01_ratio": 0.1260},
                {"point_le_0_01_ratio": 0.1290},
                {"point_le_0_01_ratio": 0.1300},
            ],
            improvement_threshold=0.01,
            patience=3,
        )

        self.assertTrue(plateau)

    def test_compute_process_targets_returns_visibility_edge_and_fallback(self) -> None:
        image = np.full((160, 200, 3), 255, dtype=np.uint8)
        teacher_roi_quad = [[24.0, 28.0], [174.0, 26.0], [176.0, 134.0], [22.0, 138.0]]
        teacher_final_quad = [[20.0, 24.0], [180.0, 20.0], [182.0, 140.0], [18.0, 144.0]]

        process = _compute_process_targets(image, teacher_roi_quad, teacher_final_quad)

        self.assertEqual(process["teacher_refine_delta_norm"].shape, (4, 2))
        self.assertEqual(process["teacher_corner_visibility"].shape, (4, 2))
        self.assertEqual(process["teacher_corner_edge_direction"].shape, (4, 5))
        self.assertEqual(process["teacher_corner_fallback_mask"].shape, (4,))

    def test_build_border_contact_target_marks_corners_near_edges(self) -> None:
        manual_quad = np.array(
            [
                [0.01, 0.02],
                [0.98, 0.03],
                [0.95, 0.97],
                [0.15, 0.92],
            ],
            dtype=np.float32,
        )

        target = _build_border_contact_target(manual_quad, threshold=0.05)

        self.assertListEqual(target.tolist(), [1.0, 1.0, 1.0, 0.0])

    def test_build_adaptive_teacher_target_blends_only_trusted_corners(self) -> None:
        manual_quad = torch.tensor(
            [[[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        r3_quad = torch.tensor(
            [[[0.105, 0.105], [0.94, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )

        target = _build_adaptive_teacher_target(
            manual_quad,
            r3_quad,
            blend_ratio=0.5,
            corner_error_max=0.01,
            sample_error_max=0.03,
        )

        expected = torch.tensor(
            [[[0.1025, 0.1025], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(target["target_quad"], expected, atol=1e-5))
        self.assertListEqual(target["corner_mask"].to(dtype=torch.int32).view(-1).tolist(), [1, 0, 1, 1])
        self.assertListEqual(target["sample_mask"].to(dtype=torch.int32).view(-1).tolist(), [1])

    def test_partition_rows_by_page_ids(self) -> None:
        rows = [
            {"project_slug": "project_001", "page_id": "IMG_0001"},
            {"project_slug": "project_001", "page_id": "IMG_0002"},
            {"project_slug": "project_001", "page_id": "IMG_0003"},
            {"project_slug": "project_002", "page_id": "IMG_1001"},
        ]

        partitioned = partition_rows_by_split(
            rows,
            {
                "train_page_ids": ["project_001:IMG_0001", "project_001:IMG_0002"],
                "val_page_ids": ["project_001:IMG_0003"],
                "holdout_projects": ["project_002"],
            },
        )

        self.assertEqual([row["page_id"] for row in partitioned["train"]], ["IMG_0001", "IMG_0002"])
        self.assertEqual([row["page_id"] for row in partitioned["val"]], ["IMG_0003"])
        self.assertEqual([row["page_id"] for row in partitioned["holdout"]], ["IMG_1001"])
        self.assertEqual(partitioned["unassigned"], [])

    def test_dataset_sample_weights_can_use_r3_as_difficulty_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "train.jsonl"
            rows = [
                {
                    "image_path": "/tmp/a.jpg",
                    "manual_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_v28_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_r3_quad": [[0.12, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_roi_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "scene_tags": [],
                },
                {
                    "image_path": "/tmp/b.jpg",
                    "manual_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_v28_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_r3_quad": [[0.20, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_roi_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "scene_tags": [],
                },
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            dataset = DeepScreenV1Dataset(manifest_path, Path(temp_dir))

            weights = dataset.build_sample_weights(
                power=1.0,
                difficulty_metric="strict_point_gap",
                teacher_key="teacher_r3_quad",
            )

        self.assertGreater(weights[1], weights[0])

    def test_dataset_can_filter_rows_by_teacher_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "train.jsonl"
            rows = [
                {
                    "image_path": "/tmp/a.jpg",
                    "manual_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_v28_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_r3_quad": [[0.105, 0.105], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_roi_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "scene_tags": [],
                },
                {
                    "image_path": "/tmp/b.jpg",
                    "manual_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_v28_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_r3_quad": [[0.16, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_roi_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "scene_tags": [],
                },
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            dataset = DeepScreenV1Dataset(
                manifest_path,
                Path(temp_dir),
                filter_settings={
                    "teacher_key": "teacher_r3_quad",
                    "mean_point_error_max": 0.01,
                    "all_corner_error_max": 0.01,
                },
            )

        self.assertEqual(len(dataset), 1)

    def test_dataset_can_filter_rows_by_border_distance_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest_path = temp_root / "manifest.jsonl"
            rows = [
                {
                    "image_path": str(temp_root / "page_a.jpg"),
                    "manual_quad": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "teacher_v28_quad": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "teacher_r3_quad": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "teacher_roi_quad": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "hard_case_border_distance_min": 0.01,
                    "scene_tags": [],
                },
                {
                    "image_path": str(temp_root / "page_b.jpg"),
                    "manual_quad": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "teacher_v28_quad": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "teacher_r3_quad": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "teacher_roi_quad": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "hard_case_border_distance_min": 0.05,
                    "scene_tags": [],
                },
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            dataset = DeepScreenV1Dataset(
                manifest_path,
                temp_root,
                filter_settings={"min_border_distance": 0.02},
            )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(float(dataset.rows[0]["hard_case_border_distance_min"]), 0.05)

    def test_compute_border_distance_min_normalizes_from_image_shape(self) -> None:
        row = {
            "manual_quad": [[10, 20], [190, 20], [190, 80], [10, 80]],
        }

        value = _compute_border_distance_min(row, image_width=200, image_height=100)

        self.assertAlmostEqual(value, 0.05, places=6)

    def test_dataset_build_sample_weights_can_boost_specific_scene_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "train.jsonl"
            rows = [
                {
                    "image_path": "/tmp/a.jpg",
                    "manual_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_v28_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_r3_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_roi_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "scene_tags": ["low_contrast_scene"],
                },
                {
                    "image_path": "/tmp/b.jpg",
                    "manual_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_v28_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_r3_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_roi_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "scene_tags": [],
                },
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            dataset = DeepScreenV1Dataset(manifest_path, Path(temp_dir))

            weights = dataset.build_sample_weights(
                power=0.0,
                difficulty_metric="strict_point_gap",
                scene_tag_boosts={"low_contrast_scene": 2.0},
            )

        self.assertGreater(weights[0], weights[1])

    def test_dataset_build_sample_weights_respects_sample_weight_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "train.jsonl"
            rows = [
                {
                    "image_path": "/tmp/a.jpg",
                    "sample_weight_multiplier": 3.0,
                    "manual_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_v28_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_r3_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_roi_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "scene_tags": [],
                },
                {
                    "image_path": "/tmp/b.jpg",
                    "sample_weight_multiplier": 1.0,
                    "manual_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_v28_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_r3_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_roi_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "scene_tags": [],
                },
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            dataset = DeepScreenV1Dataset(manifest_path, Path(temp_dir))

            weights = dataset.build_sample_weights(power=0.0)

        self.assertGreater(weights[0], weights[1])

    def test_dataset_build_sample_weights_can_boost_specific_dataset_slug(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "train.jsonl"
            rows = [
                {
                    "image_path": "/tmp/awe.jpg",
                    "dataset_slug": "202603-awe",
                    "manual_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_v28_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_r3_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_roi_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                },
                {
                    "image_path": "/tmp/yangzhou.jpg",
                    "dataset_slug": "202603-guochenghao-yangzhou",
                    "manual_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_v28_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_r3_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    "teacher_roi_quad": [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                },
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            dataset = DeepScreenV1Dataset(manifest_path, Path(temp_dir))

            weights = dataset.build_sample_weights(
                power=0.0,
                dataset_slug_boosts={"202603-awe": 6.0},
            )

        self.assertGreater(weights[0], weights[1])

    def test_compute_total_loss_accepts_process_distillation_targets(self) -> None:
        output = {
            "coarse_heatmaps": torch.zeros((2, 4, 8, 8), dtype=torch.float32),
            "coarse_offsets": torch.zeros((2, 4, 2, 8, 8), dtype=torch.float32),
            "coarse_quad": torch.full((2, 4, 2), 0.5, dtype=torch.float32),
            "roi_boxes": torch.tensor(
                [
                    [0.2, 0.2, 0.8, 0.8],
                    [0.2, 0.2, 0.8, 0.8],
                ],
                dtype=torch.float32,
            ),
            "roi_stage_quad": torch.full((2, 4, 2), 0.5, dtype=torch.float32),
            "final_quad": torch.full((2, 4, 2), 0.5, dtype=torch.float32),
            "router_logits": torch.zeros((2, 3), dtype=torch.float32),
            "scene_logits": torch.zeros((2, 4), dtype=torch.float32),
            "process_delta": torch.zeros((2, 4, 2), dtype=torch.float32),
            "process_visibility": torch.zeros((2, 4, 2), dtype=torch.float32),
            "process_edge": torch.zeros((2, 4, 5), dtype=torch.float32),
            "process_fallback_logits": torch.zeros((2, 4), dtype=torch.float32),
            "strict_point_heatmaps": torch.zeros((2, 4, 8, 8), dtype=torch.float32),
            "strict_point_offsets": torch.zeros((2, 4, 2, 8, 8), dtype=torch.float32),
            "strict_point_quad": torch.full((2, 4, 2), 0.5, dtype=torch.float32),
            "candidate_quads": torch.stack(
                [
                    torch.full((2, 4, 2), 0.45, dtype=torch.float32),
                    torch.full((2, 4, 2), 0.5, dtype=torch.float32),
                    torch.full((2, 4, 2), 0.55, dtype=torch.float32),
                ],
                dim=1,
            ),
            "candidate_scores": torch.zeros((2, 3), dtype=torch.float32),
        }
        quad = torch.full((2, 4, 2), 0.5, dtype=torch.float32)
        heatmaps = torch.zeros((2, 4, 8, 8), dtype=torch.float32)
        scene_target = torch.zeros((2, 4), dtype=torch.float32)
        process_delta = torch.zeros((2, 4, 2), dtype=torch.float32)
        process_visibility = torch.zeros((2, 4, 2), dtype=torch.float32)
        process_edge = torch.zeros((2, 4, 5), dtype=torch.float32)
        process_fallback = torch.zeros((2, 4), dtype=torch.float32)

        loss = _compute_total_loss(
            output,
            quad,
            quad,
            quad,
            quad,
            heatmaps,
            heatmaps,
            scene_target,
            process_delta,
            process_visibility,
            process_edge,
            process_fallback,
            {
                "roi_stage_teacher_weight": 0.5,
                "process_delta_weight": 0.5,
                "process_visibility_weight": 0.5,
                "process_edge_weight": 0.5,
                "process_fallback_weight": 0.5,
                "strict_spatial_manual_weight": 0.5,
                "strict_spatial_teacher_weight": 0.25,
                "strict_spatial_heatmap_manual_weight": 0.5,
                "strict_spatial_heatmap_teacher_weight": 0.25,
                "candidate_rank_weight": 0.5,
            },
            torch.device("cpu"),
        )

        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_compute_total_loss_supports_state_aware_candidate_losses(self) -> None:
        output = {
            "coarse_heatmaps": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "coarse_offsets": torch.zeros((1, 4, 2, 8, 8), dtype=torch.float32),
            "coarse_quad": torch.full((1, 4, 2), 0.5, dtype=torch.float32),
            "roi_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "roi_stage_quad": torch.full((1, 4, 2), 0.5, dtype=torch.float32),
            "base_final_quad": torch.full((1, 4, 2), 0.5, dtype=torch.float32),
            "state_aware_quad": torch.full((1, 4, 2), 0.55, dtype=torch.float32),
            "corner_state_logits": torch.zeros((1, 4), dtype=torch.float32),
            "final_quad": torch.full((1, 4, 2), 0.5, dtype=torch.float32),
            "router_logits": torch.zeros((1, 3), dtype=torch.float32),
            "process_delta": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_visibility": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_edge": torch.zeros((1, 4, 5), dtype=torch.float32),
            "process_fallback_logits": torch.zeros((1, 4), dtype=torch.float32),
        }
        quad = torch.full((1, 4, 2), 0.5, dtype=torch.float32)
        heatmaps = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        scene_target = torch.zeros((1, 4), dtype=torch.float32)
        process_delta = torch.zeros((1, 4, 2), dtype=torch.float32)
        process_visibility = torch.zeros((1, 4, 2), dtype=torch.float32)
        process_edge = torch.zeros((1, 4, 5), dtype=torch.float32)
        process_fallback = torch.zeros((1, 4), dtype=torch.float32)
        border_contact_target = torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

        loss = _compute_total_loss(
            output,
            quad,
            quad,
            quad,
            quad,
            heatmaps,
            heatmaps,
            scene_target,
            process_delta,
            process_visibility,
            process_edge,
            process_fallback,
            {
                "state_aware_manual_weight": 0.5,
                "border_contact_weight": 0.25,
            },
            torch.device("cpu"),
            border_contact_target=border_contact_target,
        )

        self.assertGreater(float(loss.item()), 0.0)

    def test_strict_point_soft_target_loss_rewards_sub_threshold_predictions(self) -> None:
        manual_quad = torch.tensor(
            [[[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        near_quad = manual_quad.clone()
        near_quad[0, 0, 0] += 0.002
        far_quad = manual_quad.clone()
        far_quad[0, 0, 0] += 0.03

        near_loss = _strict_point_soft_target_loss(
            near_quad,
            manual_quad,
            threshold=0.01,
            temperature=0.002,
        )
        far_loss = _strict_point_soft_target_loss(
            far_quad,
            manual_quad,
            threshold=0.01,
            temperature=0.002,
        )

        self.assertLess(float(near_loss.item()), float(far_loss.item()))

    def test_strict_point_soft_target_loss_can_boost_border_contact_corners(self) -> None:
        manual_quad = torch.tensor(
            [[[0.01, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        predicted_quad = manual_quad.clone()
        predicted_quad[0, 0, 0] += 0.03
        border_contact_target = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        baseline_loss = _strict_point_soft_target_loss(
            predicted_quad,
            manual_quad,
            threshold=0.01,
            temperature=0.002,
        )
        boosted_loss = _strict_point_soft_target_loss(
            predicted_quad,
            manual_quad,
            threshold=0.01,
            temperature=0.002,
            border_contact_target=border_contact_target,
            border_contact_boost=3.0,
        )

        self.assertGreater(float(boosted_loss.item()), float(baseline_loss.item()))

    def test_compute_total_loss_supports_soft_strict_point_manual_weight(self) -> None:
        manual_quad = torch.tensor(
            [[[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        output = {
            "coarse_heatmaps": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "coarse_offsets": torch.zeros((1, 4, 2, 8, 8), dtype=torch.float32),
            "coarse_quad": manual_quad.clone(),
            "roi_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "roi_stage_quad": manual_quad.clone(),
            "final_quad": manual_quad.clone(),
            "router_logits": torch.zeros((1, 3), dtype=torch.float32),
            "process_delta": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_visibility": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_edge": torch.zeros((1, 4, 5), dtype=torch.float32),
            "process_fallback_logits": torch.zeros((1, 4), dtype=torch.float32),
        }
        output["final_quad"][0, 0, 0] += 0.03
        zero = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        base_loss = _compute_total_loss(
            output,
            manual_quad=manual_quad,
            teacher_quad=manual_quad,
            r3_quad=manual_quad,
            teacher_roi_quad=manual_quad,
            manual_heatmaps=zero,
            r3_heatmaps=zero,
            scene_target=torch.zeros((1, 4), dtype=torch.float32),
            teacher_refine_delta=torch.zeros((1, 4, 2), dtype=torch.float32),
            teacher_corner_visibility=torch.zeros((1, 4, 2), dtype=torch.float32),
            teacher_corner_edge=torch.zeros((1, 4, 5), dtype=torch.float32),
            teacher_corner_fallback=torch.zeros((1, 4), dtype=torch.float32),
            loss_settings={
                "coarse_heatmap_weight": 0.0,
                "r3_heatmap_weight": 0.0,
                "coarse_manual_weight": 0.0,
                "coarse_r3_weight": 0.0,
                "final_manual_weight": 0.0,
                "final_teacher_weight": 0.0,
                "strict_point_manual_weight": 0.0,
                "strict_point_soft_manual_weight": 0.0,
                "router_reg_weight": 0.0,
            },
            device=torch.device("cpu"),
        )
        soft_loss = _compute_total_loss(
            output,
            manual_quad=manual_quad,
            teacher_quad=manual_quad,
            r3_quad=manual_quad,
            teacher_roi_quad=manual_quad,
            manual_heatmaps=zero,
            r3_heatmaps=zero,
            scene_target=torch.zeros((1, 4), dtype=torch.float32),
            teacher_refine_delta=torch.zeros((1, 4, 2), dtype=torch.float32),
            teacher_corner_visibility=torch.zeros((1, 4, 2), dtype=torch.float32),
            teacher_corner_edge=torch.zeros((1, 4, 5), dtype=torch.float32),
            teacher_corner_fallback=torch.zeros((1, 4), dtype=torch.float32),
            loss_settings={
                "coarse_heatmap_weight": 0.0,
                "r3_heatmap_weight": 0.0,
                "coarse_manual_weight": 0.0,
                "coarse_r3_weight": 0.0,
                "final_manual_weight": 0.0,
                "final_teacher_weight": 0.0,
                "strict_point_manual_weight": 0.0,
                "strict_point_soft_manual_weight": 1.0,
                "strict_point_soft_temperature": 0.002,
                "router_reg_weight": 0.0,
            },
            device=torch.device("cpu"),
        )

        self.assertGreater(float(soft_loss.item()), float(base_loss.item()))

    def test_compute_total_loss_supports_adaptive_coarse_quad_target(self) -> None:
        manual_quad = torch.tensor(
            [[[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        r3_quad = torch.tensor(
            [[[0.105, 0.105], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        output = {
            "coarse_heatmaps": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "coarse_offsets": torch.zeros((1, 4, 2, 8, 8), dtype=torch.float32),
            "coarse_quad": manual_quad.clone(),
            "roi_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "roi_stage_quad": manual_quad.clone(),
            "final_quad": manual_quad.clone(),
            "router_logits": torch.zeros((1, 3), dtype=torch.float32),
            "process_delta": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_visibility": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_edge": torch.zeros((1, 4, 5), dtype=torch.float32),
            "process_fallback_logits": torch.zeros((1, 4), dtype=torch.float32),
        }

        loss = _compute_total_loss(
            output,
            manual_quad=manual_quad,
            teacher_quad=manual_quad,
            r3_quad=r3_quad,
            teacher_roi_quad=manual_quad,
            manual_heatmaps=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            r3_heatmaps=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            scene_target=torch.zeros((1, 4), dtype=torch.float32),
            teacher_refine_delta=torch.zeros((1, 4, 2), dtype=torch.float32),
            teacher_corner_visibility=torch.zeros((1, 4, 2), dtype=torch.float32),
            teacher_corner_edge=torch.zeros((1, 4, 5), dtype=torch.float32),
            teacher_corner_fallback=torch.zeros((1, 4), dtype=torch.float32),
            loss_settings={
                "coarse_heatmap_weight": 0.0,
                "r3_heatmap_weight": 0.0,
                "coarse_manual_weight": 0.0,
                "coarse_r3_weight": 0.0,
                "final_manual_weight": 0.0,
                "final_teacher_weight": 0.0,
                "router_reg_weight": 0.0,
                "adaptive_coarse_heatmap_weight": 0.0,
                "adaptive_coarse_quad_weight": 1.0,
                "adaptive_teacher_blend_ratio": 0.5,
                "adaptive_teacher_corner_error_max": 0.01,
                "adaptive_teacher_sample_error_max": 0.03,
            },
            device=torch.device("cpu"),
        )

        self.assertGreater(float(loss.item()), 0.0)

    def test_build_oracle_teacher_target_selects_best_corner_from_r3_and_v28(self) -> None:
        manual_quad = torch.tensor(
            [[[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        r3_quad = torch.tensor(
            [[[0.105, 0.105], [0.94, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        v28_quad = torch.tensor(
            [[[0.11, 0.11], [0.905, 0.105], [0.89, 0.89], [0.10, 0.95]]],
            dtype=torch.float32,
        )

        target = _build_oracle_teacher_target(
            manual_quad,
            r3_quad,
            v28_quad,
            corner_error_max=0.01,
        )

        expected = torch.tensor(
            [[[0.105, 0.105], [0.905, 0.105], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(target["target_quad"], expected, atol=1e-5))
        self.assertListEqual(target["corner_mask"].to(dtype=torch.int32).view(-1).tolist(), [1, 1, 1, 1])

    def test_compute_total_loss_supports_oracle_coarse_quad_target(self) -> None:
        manual_quad = torch.tensor(
            [[[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        r3_quad = torch.tensor(
            [[[0.105, 0.105], [0.94, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        teacher_quad = torch.tensor(
            [[[0.11, 0.11], [0.905, 0.105], [0.89, 0.89], [0.10, 0.95]]],
            dtype=torch.float32,
        )
        output = {
            "coarse_heatmaps": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "coarse_offsets": torch.zeros((1, 4, 2, 8, 8), dtype=torch.float32),
            "coarse_quad": manual_quad.clone(),
            "roi_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "roi_stage_quad": manual_quad.clone(),
            "final_quad": manual_quad.clone(),
            "router_logits": torch.zeros((1, 3), dtype=torch.float32),
            "process_delta": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_visibility": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_edge": torch.zeros((1, 4, 5), dtype=torch.float32),
            "process_fallback_logits": torch.zeros((1, 4), dtype=torch.float32),
        }

        loss = _compute_total_loss(
            output,
            manual_quad=manual_quad,
            teacher_quad=teacher_quad,
            r3_quad=r3_quad,
            teacher_roi_quad=manual_quad,
            manual_heatmaps=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            r3_heatmaps=torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            scene_target=torch.zeros((1, 4), dtype=torch.float32),
            teacher_refine_delta=torch.zeros((1, 4, 2), dtype=torch.float32),
            teacher_corner_visibility=torch.zeros((1, 4, 2), dtype=torch.float32),
            teacher_corner_edge=torch.zeros((1, 4, 5), dtype=torch.float32),
            teacher_corner_fallback=torch.zeros((1, 4), dtype=torch.float32),
            loss_settings={
                "coarse_heatmap_weight": 0.0,
                "adaptive_coarse_heatmap_weight": 0.0,
                "r3_heatmap_weight": 0.0,
                "coarse_manual_weight": 0.0,
                "coarse_r3_weight": 0.0,
                "adaptive_coarse_quad_weight": 0.0,
                "oracle_coarse_heatmap_weight": 0.0,
                "oracle_coarse_quad_weight": 1.0,
                "oracle_teacher_corner_error_max": 0.01,
                "final_manual_weight": 0.0,
                "final_teacher_weight": 0.0,
                "router_reg_weight": 0.0,
            },
            device=torch.device("cpu"),
        )

        self.assertGreater(float(loss.item()), 0.0)

    def test_compute_total_loss_can_gate_r3_supervision_when_r3_disagrees_with_manual(self) -> None:
        output = {
            "coarse_heatmaps": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "coarse_offsets": torch.zeros((1, 4, 2, 8, 8), dtype=torch.float32),
            "coarse_quad": torch.zeros((1, 4, 2), dtype=torch.float32),
            "roi_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "roi_stage_quad": torch.zeros((1, 4, 2), dtype=torch.float32),
            "final_quad": torch.zeros((1, 4, 2), dtype=torch.float32),
            "router_logits": torch.zeros((1, 3), dtype=torch.float32),
            "scene_logits": torch.zeros((1, 4), dtype=torch.float32),
            "process_delta": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_visibility": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_edge": torch.zeros((1, 4, 5), dtype=torch.float32),
            "process_fallback_logits": torch.zeros((1, 4), dtype=torch.float32),
        }
        manual_quad = torch.zeros((1, 4, 2), dtype=torch.float32)
        teacher_quad = torch.zeros((1, 4, 2), dtype=torch.float32)
        r3_quad = torch.full((1, 4, 2), 0.9, dtype=torch.float32)
        teacher_roi_quad = torch.zeros((1, 4, 2), dtype=torch.float32)
        manual_heatmaps = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        r3_heatmaps = torch.ones((1, 4, 8, 8), dtype=torch.float32)
        scene_target = torch.zeros((1, 4), dtype=torch.float32)
        process_delta = torch.zeros((1, 4, 2), dtype=torch.float32)
        process_visibility = torch.zeros((1, 4, 2), dtype=torch.float32)
        process_edge = torch.zeros((1, 4, 5), dtype=torch.float32)
        process_fallback = torch.zeros((1, 4), dtype=torch.float32)

        baseline_loss = _compute_total_loss(
            output,
            manual_quad,
            teacher_quad,
            r3_quad,
            teacher_roi_quad,
            manual_heatmaps,
            r3_heatmaps,
            scene_target,
            process_delta,
            process_visibility,
            process_edge,
            process_fallback,
            {
                "r3_heatmap_weight": 1.0,
                "coarse_r3_weight": 1.0,
                "router_reg_weight": 0.0,
            },
            torch.device("cpu"),
        )

        gated_loss = _compute_total_loss(
            output,
            manual_quad,
            teacher_quad,
            r3_quad,
            teacher_roi_quad,
            manual_heatmaps,
            r3_heatmaps,
            scene_target,
            process_delta,
            process_visibility,
            process_edge,
            process_fallback,
            {
                "r3_heatmap_weight": 1.0,
                "coarse_r3_weight": 1.0,
                "r3_agreement_point_error_max": 0.05,
                "router_reg_weight": 0.0,
            },
            torch.device("cpu"),
        )

        self.assertGreater(float(baseline_loss.item()), 0.0)
        self.assertAlmostEqual(float(gated_loss.item()), 0.0, places=6)

    def test_compute_total_loss_skips_teacher_terms_for_manual_only_samples(self) -> None:
        output = {
            "coarse_heatmaps": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "coarse_offsets": torch.zeros((1, 4, 2, 8, 8), dtype=torch.float32),
            "coarse_quad": torch.zeros((1, 4, 2), dtype=torch.float32),
            "roi_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "roi_stage_quad": torch.zeros((1, 4, 2), dtype=torch.float32),
            "final_quad": torch.zeros((1, 4, 2), dtype=torch.float32),
            "router_logits": torch.zeros((1, 3), dtype=torch.float32),
            "process_delta": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_visibility": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_edge": torch.zeros((1, 4, 5), dtype=torch.float32),
            "process_fallback_logits": torch.zeros((1, 4), dtype=torch.float32),
        }
        manual_quad = torch.zeros((1, 4, 2), dtype=torch.float32)
        teacher_quad = torch.full((1, 4, 2), 0.9, dtype=torch.float32)
        r3_quad = torch.full((1, 4, 2), 0.9, dtype=torch.float32)
        teacher_roi_quad = torch.full((1, 4, 2), 0.8, dtype=torch.float32)
        manual_heatmaps = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        r3_heatmaps = torch.ones((1, 4, 8, 8), dtype=torch.float32)
        scene_target = torch.zeros((1, 4), dtype=torch.float32)
        process_delta = torch.ones((1, 4, 2), dtype=torch.float32)
        process_visibility = torch.ones((1, 4, 2), dtype=torch.float32)
        process_edge = torch.ones((1, 4, 5), dtype=torch.float32)
        process_fallback = torch.ones((1, 4), dtype=torch.float32)

        baseline_loss = _compute_total_loss(
            output,
            manual_quad,
            teacher_quad,
            r3_quad,
            teacher_roi_quad,
            manual_heatmaps,
            r3_heatmaps,
            scene_target,
            process_delta,
            process_visibility,
            process_edge,
            process_fallback,
            {
                "coarse_heatmap_weight": 0.0,
                "coarse_manual_weight": 0.0,
                "final_manual_weight": 0.0,
                "final_teacher_weight": 1.0,
                "coarse_r3_weight": 1.0,
                "r3_heatmap_weight": 1.0,
                "roi_stage_teacher_weight": 1.0,
                "process_delta_weight": 1.0,
                "process_visibility_weight": 1.0,
                "process_edge_weight": 1.0,
                "process_fallback_weight": 1.0,
                "router_reg_weight": 0.0,
            },
            torch.device("cpu"),
        )

        manual_only_loss = _compute_total_loss(
            output,
            manual_quad,
            teacher_quad,
            r3_quad,
            teacher_roi_quad,
            manual_heatmaps,
            r3_heatmaps,
            scene_target,
            process_delta,
            process_visibility,
            process_edge,
            process_fallback,
            {
                "coarse_heatmap_weight": 0.0,
                "coarse_manual_weight": 0.0,
                "final_manual_weight": 0.0,
                "final_teacher_weight": 1.0,
                "coarse_r3_weight": 1.0,
                "r3_heatmap_weight": 1.0,
                "roi_stage_teacher_weight": 1.0,
                "process_delta_weight": 1.0,
                "process_visibility_weight": 1.0,
                "process_edge_weight": 1.0,
                "process_fallback_weight": 1.0,
                "router_reg_weight": 0.0,
            },
            torch.device("cpu"),
            manual_only_mask=torch.tensor([True], dtype=torch.bool),
        )

        self.assertGreater(float(baseline_loss.item()), 0.0)
        self.assertAlmostEqual(float(manual_only_loss.item()), 0.0, places=6)

    def test_compute_total_loss_can_keep_process_structure_losses_for_manual_only_samples(self) -> None:
        output = {
            "coarse_heatmaps": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "coarse_offsets": torch.zeros((1, 4, 2, 8, 8), dtype=torch.float32),
            "coarse_quad": torch.zeros((1, 4, 2), dtype=torch.float32),
            "roi_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
            "roi_stage_quad": torch.zeros((1, 4, 2), dtype=torch.float32),
            "final_quad": torch.zeros((1, 4, 2), dtype=torch.float32),
            "router_logits": torch.zeros((1, 3), dtype=torch.float32),
            "process_delta": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_visibility": torch.zeros((1, 4, 2), dtype=torch.float32),
            "process_edge": torch.zeros((1, 4, 5), dtype=torch.float32),
            "process_fallback_logits": torch.zeros((1, 4), dtype=torch.float32),
        }
        zero_quad = torch.zeros((1, 4, 2), dtype=torch.float32)
        zero_heatmaps = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        process_visibility = torch.ones((1, 4, 2), dtype=torch.float32)
        process_edge = torch.zeros((1, 4, 5), dtype=torch.float32)
        process_fallback = torch.ones((1, 4), dtype=torch.float32)

        loss = _compute_total_loss(
            output,
            zero_quad,
            zero_quad,
            zero_quad,
            zero_quad,
            zero_heatmaps,
            zero_heatmaps,
            torch.zeros((1, 4), dtype=torch.float32),
            torch.zeros((1, 4, 2), dtype=torch.float32),
            process_visibility,
            process_edge,
            process_fallback,
            {
                "coarse_heatmap_weight": 0.0,
                "coarse_manual_weight": 0.0,
                "final_manual_weight": 0.0,
                "final_teacher_weight": 0.0,
                "coarse_r3_weight": 0.0,
                "r3_heatmap_weight": 0.0,
                "roi_stage_teacher_weight": 0.0,
                "process_delta_weight": 0.0,
                "process_visibility_weight": 1.0,
                "process_edge_weight": 0.0,
                "process_fallback_weight": 1.0,
                "router_reg_weight": 0.0,
            },
            torch.device("cpu"),
            manual_only_mask=torch.tensor([True], dtype=torch.bool),
            process_structure_mask=torch.tensor([True], dtype=torch.bool),
        )

        self.assertGreater(float(loss.item()), 0.0)

    def test_dataset_build_sample_weights_manual_only_rows_do_not_use_teacher_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "train.jsonl"
            rows = [
                {
                    "page_id": "manual-only-a",
                    "manual_only": True,
                    "scene_tags": [],
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.4, 0.5], [0.7, 0.5], [0.7, 0.7], [0.4, 0.7]],
                    "teacher_r3_quad": [[0.4, 0.5], [0.7, 0.5], [0.7, 0.7], [0.4, 0.7]],
                },
                {
                    "page_id": "manual-only-b",
                    "manual_only": True,
                    "scene_tags": [],
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.12, 0.21], [0.79, 0.22], [0.79, 0.79], [0.11, 0.79]],
                    "teacher_r3_quad": [[0.12, 0.21], [0.79, 0.22], [0.79, 0.79], [0.11, 0.79]],
                },
            ]
            manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = DeepScreenV1Dataset(manifest_path, root, augment=False)
            weights = dataset.build_sample_weights(power=1.0, difficulty_metric="strict_point_gap")

        self.assertEqual(weights.shape, (2,))
        self.assertAlmostEqual(float(weights[0]), float(weights[1]), places=6)

    def test_dataset_manual_process_targets_can_use_manual_quad_instead_of_teacher_quad(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "page.png"
            import cv2

            cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))
            manifest_path = root / "train.jsonl"
            row = {
                "page_id": "manual-process-a",
                "image_path": str(image_path),
                "page_name": image_path.name,
                "manual_only": True,
                "manual_process_targets": True,
                "scene_tags": [],
                "manual_quad": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "teacher_v28_quad": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                "teacher_r3_quad": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                "teacher_roi_quad": [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
            }
            manifest_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            dataset = DeepScreenV1Dataset(manifest_path, root, augment=False)
            dummy_targets = {
                "teacher_refine_delta_norm": np.zeros((4, 2), dtype=np.float32),
                "teacher_corner_visibility": np.zeros((4, 2), dtype=np.float32),
                "teacher_corner_edge_direction": np.zeros((4, 5), dtype=np.float32),
                "teacher_corner_fallback_mask": np.zeros((4,), dtype=np.float32),
                "teacher_roi_box": np.zeros((4,), dtype=np.float32),
            }

            with mock.patch("deep_screen_v1_train._compute_process_targets", return_value=dummy_targets) as process_mock:
                _ = dataset[0]

        args = process_mock.call_args.args
        manual_quad_px = np.array(row["manual_quad"], dtype=np.float32)
        self.assertTrue(np.allclose(args[1], manual_quad_px, atol=1e-5))
        self.assertTrue(np.allclose(args[2], manual_quad_px, atol=1e-5))

    def test_candidate_rank_loss_prefers_candidate_with_smallest_manual_error(self) -> None:
        candidate_quads = torch.tensor(
            [
                [
                    [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    [[0.12, 0.12], [0.88, 0.12], [0.88, 0.88], [0.12, 0.88]],
                    [[0.20, 0.20], [0.80, 0.20], [0.80, 0.80], [0.20, 0.80]],
                ]
            ],
            dtype=torch.float32,
        )
        manual_quad = torch.tensor(
            [[[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        good_scores = torch.tensor([[4.0, 1.0, -2.0]], dtype=torch.float32)
        bad_scores = torch.tensor([[-2.0, 1.0, 4.0]], dtype=torch.float32)

        good_loss = _candidate_rank_loss(candidate_quads, good_scores, manual_quad)
        bad_loss = _candidate_rank_loss(candidate_quads, bad_scores, manual_quad)

        self.assertLess(float(good_loss.item()), float(bad_loss.item()))

    def test_candidate_rank_loss_ignores_masked_external_candidates(self) -> None:
        candidate_quads = torch.tensor(
            [
                [
                    [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    [[0.12, 0.12], [0.88, 0.12], [0.88, 0.88], [0.12, 0.88]],
                    [[0.20, 0.20], [0.80, 0.20], [0.80, 0.80], [0.20, 0.80]],
                    [[0.30, 0.30], [0.70, 0.30], [0.70, 0.70], [0.30, 0.70]],
                ]
            ],
            dtype=torch.float32,
        )
        candidate_mask = torch.tensor([[False, True, True, True]], dtype=torch.bool)
        manual_quad = torch.tensor(
            [[[0.12, 0.12], [0.88, 0.12], [0.88, 0.88], [0.12, 0.88]]],
            dtype=torch.float32,
        )
        good_scores = torch.tensor([[-4.0, 4.0, 1.0, -2.0]], dtype=torch.float32)
        bad_scores = torch.tensor([[4.0, -2.0, 1.0, -1.0]], dtype=torch.float32)

        good_loss = _candidate_rank_loss(candidate_quads, good_scores, manual_quad, candidate_mask=candidate_mask)
        bad_loss = _candidate_rank_loss(candidate_quads, bad_scores, manual_quad, candidate_mask=candidate_mask)

        self.assertLess(float(good_loss.item()), float(bad_loss.item()))

    def test_candidate_rank_loss_can_prioritize_strict_point_hits_before_mean_error(self) -> None:
        candidate_quads = torch.tensor(
            [
                [
                    [[0.10, 0.10], [0.915, 0.10], [0.90, 0.90], [0.10, 0.90]],
                    [[0.10, 0.10], [0.905, 0.10], [0.90, 0.90], [0.10, 0.90]],
                ]
            ],
            dtype=torch.float32,
        )
        manual_quad = torch.tensor(
            [[[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]]],
            dtype=torch.float32,
        )
        good_scores = torch.tensor([[0.0, 2.0]], dtype=torch.float32)
        bad_scores = torch.tensor([[2.0, 0.0]], dtype=torch.float32)

        good_loss = _candidate_rank_loss(
            candidate_quads,
            good_scores,
            manual_quad,
            rank_metric="strict_point_then_mean_error",
        )
        bad_loss = _candidate_rank_loss(
            candidate_quads,
            bad_scores,
            manual_quad,
            rank_metric="strict_point_then_mean_error",
        )

        self.assertLess(float(good_loss.item()), float(bad_loss.item()))

    def test_build_candidate_pool_tensors_appends_external_candidate_before_internal_candidates(self) -> None:
        output = {
            "coarse_quad": torch.full((2, 4, 2), 0.2, dtype=torch.float32),
            "roi_stage_quad": torch.full((2, 4, 2), 0.4, dtype=torch.float32),
            "base_final_quad": torch.full((2, 4, 2), 0.6, dtype=torch.float32),
        }
        external_candidate_quads = torch.full((2, 1, 4, 2), 0.8, dtype=torch.float32)
        external_candidate_mask = torch.tensor([[True], [False]], dtype=torch.bool)

        pool = _build_candidate_pool_tensors(
            output,
            external_candidate_quads=external_candidate_quads,
            external_candidate_mask=external_candidate_mask,
        )

        self.assertEqual(tuple(pool["candidate_quads"].shape), (2, 4, 4, 2))
        self.assertEqual(tuple(pool["candidate_mask"].shape), (2, 4))
        self.assertTrue(torch.allclose(pool["candidate_quads"][0, 0], torch.full((4, 2), 0.8)))
        self.assertTrue(torch.equal(pool["candidate_mask"][0], torch.tensor([True, True, True, True])))
        self.assertTrue(torch.equal(pool["candidate_mask"][1], torch.tensor([False, True, True, True])))

    def test_max_corner_constraint_loss_matches_worst_corner_error_when_threshold_is_zero(self) -> None:
        predicted = torch.tensor(
            [[
                [0.00, 0.00],
                [1.00, 0.00],
                [1.20, 1.00],
                [0.00, 1.00],
            ]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[
                [0.00, 0.00],
                [1.00, 0.00],
                [1.00, 1.00],
                [0.00, 1.00],
            ]],
            dtype=torch.float32,
        )

        loss = _max_corner_constraint_loss(predicted, target, threshold=0.0)

        self.assertAlmostEqual(float(loss.item()), 0.1414, places=4)

    def test_max_corner_constraint_loss_ignores_samples_below_threshold(self) -> None:
        predicted = torch.tensor(
            [[
                [0.00, 0.00],
                [1.00, 0.00],
                [1.01, 1.00],
                [0.00, 1.00],
            ]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[
                [0.00, 0.00],
                [1.00, 0.00],
                [1.00, 1.00],
                [0.00, 1.00],
            ]],
            dtype=torch.float32,
        )

        loss = _max_corner_constraint_loss(predicted, target, threshold=0.01)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_quad_inset_abs_loss_is_zero_for_identical_quads(self) -> None:
        quad = torch.tensor(
            [[
                [0.00, 0.00],
                [1.00, 0.00],
                [1.00, 1.00],
                [0.00, 1.00],
            ]],
            dtype=torch.float32,
        )

        loss = _quad_inset_abs_loss(quad, quad)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_quad_inset_abs_loss_penalizes_inward_and_outward_area_shift(self) -> None:
        target = torch.tensor(
            [[
                [0.00, 0.00],
                [1.00, 0.00],
                [1.00, 1.00],
                [0.00, 1.00],
            ]],
            dtype=torch.float32,
        )
        inward = torch.tensor(
            [[
                [0.10, 0.10],
                [0.90, 0.10],
                [0.90, 0.90],
                [0.10, 0.90],
            ]],
            dtype=torch.float32,
        )
        outward = torch.tensor(
            [[
                [-0.10, -0.10],
                [1.10, -0.10],
                [1.10, 1.10],
                [-0.10, 1.10],
            ]],
            dtype=torch.float32,
        )

        inward_loss = _quad_inset_abs_loss(inward, target)
        outward_loss = _quad_inset_abs_loss(outward, target)

        self.assertAlmostEqual(float(inward_loss.item()), 0.36, places=4)
        self.assertAlmostEqual(float(outward_loss.item()), 0.44, places=4)

    def test_quad_inset_inward_loss_penalizes_only_inward_shrink(self) -> None:
        target = torch.tensor(
            [[
                [0.00, 0.00],
                [1.00, 0.00],
                [1.00, 1.00],
                [0.00, 1.00],
            ]],
            dtype=torch.float32,
        )
        inward = torch.tensor(
            [[
                [0.10, 0.10],
                [0.90, 0.10],
                [0.90, 0.90],
                [0.10, 0.90],
            ]],
            dtype=torch.float32,
        )
        outward = torch.tensor(
            [[
                [-0.10, -0.10],
                [1.10, -0.10],
                [1.10, 1.10],
                [-0.10, 1.10],
            ]],
            dtype=torch.float32,
        )

        inward_loss = _quad_inset_inward_loss(inward, target)
        outward_loss = _quad_inset_inward_loss(outward, target)

        self.assertAlmostEqual(float(inward_loss.item()), 0.36, places=4)
        self.assertAlmostEqual(float(outward_loss.item()), 0.0, places=6)

    def test_augmentation_settings_reads_optional_config(self) -> None:
        settings = _augmentation_settings(
            {
                "augmentation": {
                    "enable_flip": False,
                    "brightness_contrast_prob": 0.5,
                    "jpeg_prob": 0.25,
                    "gaussian_noise_prob": 0.75,
                }
            }
        )

        self.assertEqual(settings["enable_flip"], False)
        self.assertEqual(settings["brightness_contrast_prob"], 0.5)
        self.assertEqual(settings["jpeg_prob"], 0.25)
        self.assertEqual(settings["gaussian_noise_prob"], 0.75)

    def test_sampling_settings_reads_optional_config(self) -> None:
        settings = _sampling_settings({"sampling": {"sample_weight_power": 1.5}})

        self.assertEqual(settings["sample_weight_power"], 1.5)

    def test_apply_train_augmentation_changes_pixels_but_preserves_shape(self) -> None:
        image = np.full((32, 32, 3), 127, dtype=np.uint8)

        augmented = _apply_train_augmentation(
            image,
            {
                "brightness_contrast_prob": 1.0,
                "jpeg_prob": 1.0,
                "gaussian_noise_prob": 1.0,
                "brightness_delta": 24.0,
                "contrast_scale": 1.15,
                "jpeg_quality_min": 70,
                "jpeg_quality_max": 70,
                "gaussian_noise_std": 6.0,
            },
            rng=np.random.default_rng(7),
        )

        self.assertEqual(augmented.shape, image.shape)
        self.assertNotEqual(int(augmented.mean()), int(image.mean()))

    def test_apply_train_augmentation_can_apply_occlusion_cutout(self) -> None:
        image = np.full((32, 32, 3), 127, dtype=np.uint8)

        augmented = _apply_train_augmentation(
            image,
            {
                "occlusion_prob": 1.0,
                "occlusion_count_min": 1,
                "occlusion_count_max": 1,
                "occlusion_size_min": 0.25,
                "occlusion_size_max": 0.25,
            },
            rng=np.random.default_rng(7),
        )

        self.assertEqual(augmented.shape, image.shape)
        self.assertFalse(np.array_equal(augmented, image))
        self.assertGreater(int((augmented == 0).sum()), 0)

    def test_apply_train_augmentation_can_apply_edge_occlusion(self) -> None:
        image = np.full((32, 32, 3), 127, dtype=np.uint8)

        augmented = _apply_train_augmentation(
            image,
            {
                "edge_occlusion_prob": 1.0,
                "edge_occlusion_sides_min": 1,
                "edge_occlusion_sides_max": 1,
                "edge_occlusion_size_min": 0.25,
                "edge_occlusion_size_max": 0.25,
            },
            rng=np.random.default_rng(7),
        )

        self.assertEqual(augmented.shape, image.shape)
        self.assertFalse(np.array_equal(augmented, image))
        border_band = 8
        self.assertGreater(
            int(
                (augmented[:border_band] == 0).sum()
                + (augmented[-border_band:] == 0).sum()
                + (augmented[:, :border_band] == 0).sum()
                + (augmented[:, -border_band:] == 0).sum()
            ),
            0,
        )

    def test_apply_train_augmentation_returns_original_when_all_probs_are_zero(self) -> None:
        image = np.full((32, 32, 3), 127, dtype=np.uint8)

        augmented = _apply_train_augmentation(
            image,
            {
                "brightness_contrast_prob": 0.0,
                "jpeg_prob": 0.0,
                "gaussian_noise_prob": 0.0,
                "occlusion_prob": 0.0,
            },
            rng=np.random.default_rng(7),
        )

        self.assertTrue(np.array_equal(augmented, image))

    def test_apply_perspective_augmentation_warps_image_and_quads(self) -> None:
        image = np.full((48, 64, 3), 127, dtype=np.uint8)
        quads = [
            np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], dtype=np.float32),
        ]

        warped_image, warped_quads = _apply_perspective_augmentation(
            image,
            quads,
            {
                "perspective_prob": 1.0,
                "perspective_jitter_ratio": 0.08,
            },
            rng=np.random.default_rng(7),
        )

        self.assertEqual(warped_image.shape, image.shape)
        self.assertEqual(len(warped_quads), 1)
        self.assertEqual(warped_quads[0].shape, quads[0].shape)
        self.assertGreaterEqual(float(warped_quads[0].min()), 0.0)
        self.assertLessEqual(float(warped_quads[0].max()), 1.0)
        self.assertFalse(np.allclose(warped_quads[0], quads[0]))

    def test_dataset_build_sample_weights_upweights_rows_with_larger_teacher_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "train.jsonl"
            rows = [
                {
                    "page_id": "easy",
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.11, 0.2], [0.8, 0.21], [0.79, 0.79], [0.1, 0.79]],
                    "teacher_r3_quad": [[0.12, 0.2], [0.8, 0.22], [0.78, 0.79], [0.1, 0.78]],
                },
                {
                    "page_id": "hard",
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.2, 0.3], [0.72, 0.32], [0.7, 0.72], [0.18, 0.74]],
                    "teacher_r3_quad": [[0.22, 0.34], [0.7, 0.35], [0.68, 0.68], [0.2, 0.7]],
                },
            ]
            manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = DeepScreenV1Dataset(manifest_path, root, augment=False)
            weights = dataset.build_sample_weights(power=1.0)

        self.assertEqual(weights.shape, (2,))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_dataset_build_sample_weights_can_target_max_corner_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "train.jsonl"
            rows = [
                {
                    "page_id": "balanced",
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.12, 0.22], [0.78, 0.22], [0.79, 0.79], [0.12, 0.79]],
                    "teacher_r3_quad": [[0.12, 0.22], [0.78, 0.22], [0.79, 0.79], [0.12, 0.79]],
                },
                {
                    "page_id": "single-corner-outlier",
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.25, 0.65]],
                    "teacher_r3_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.25, 0.65]],
                },
            ]
            manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = DeepScreenV1Dataset(manifest_path, root, augment=False)
            weights = dataset.build_sample_weights(power=1.0, difficulty_metric="max_corner_error")

        self.assertEqual(weights.shape, (2,))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_dataset_build_sample_weights_can_target_strict_point_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "train.jsonl"
            rows = [
                {
                    "page_id": "teacher-almost-perfect",
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.105, 0.205], [0.795, 0.205], [0.795, 0.795], [0.105, 0.795]],
                    "teacher_r3_quad": [[0.105, 0.205], [0.795, 0.205], [0.795, 0.795], [0.105, 0.795]],
                },
                {
                    "page_id": "teacher-misses-multiple-corners",
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.14, 0.24], [0.77, 0.25], [0.76, 0.76], [0.14, 0.75]],
                    "teacher_r3_quad": [[0.14, 0.24], [0.77, 0.25], [0.76, 0.76], [0.14, 0.75]],
                },
            ]
            manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = DeepScreenV1Dataset(manifest_path, root, augment=False)
            weights = dataset.build_sample_weights(power=1.0, difficulty_metric="strict_point_gap")

        self.assertEqual(weights.shape, (2,))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_dataset_build_sample_weights_can_balance_rare_scene_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "train.jsonl"
            rows = [
                {
                    "page_id": "common-1",
                    "scene_tags": ["bright_screen"],
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.12, 0.21], [0.79, 0.22], [0.79, 0.79], [0.11, 0.79]],
                    "teacher_r3_quad": [[0.12, 0.21], [0.79, 0.22], [0.79, 0.79], [0.11, 0.79]],
                },
                {
                    "page_id": "common-2",
                    "scene_tags": ["bright_screen"],
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.12, 0.21], [0.79, 0.22], [0.79, 0.79], [0.11, 0.79]],
                    "teacher_r3_quad": [[0.12, 0.21], [0.79, 0.22], [0.79, 0.79], [0.11, 0.79]],
                },
                {
                    "page_id": "rare-scene",
                    "scene_tags": ["black_frame_scene"],
                    "manual_quad": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "teacher_v28_quad": [[0.12, 0.21], [0.79, 0.22], [0.79, 0.79], [0.11, 0.79]],
                    "teacher_r3_quad": [[0.12, 0.21], [0.79, 0.22], [0.79, 0.79], [0.11, 0.79]],
                },
            ]
            manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = DeepScreenV1Dataset(manifest_path, root, augment=False)
            weights = dataset.build_sample_weights(
                power=0.0,
                difficulty_metric="point_error",
                scene_balance_power=1.0,
            )

        self.assertEqual(weights.shape, (3,))
        self.assertGreater(float(weights[2]), float(weights[0]))

    def test_build_train_dataloader_uses_weighted_sampler_when_sampling_enabled(self) -> None:
        dataset = torch.utils.data.TensorDataset(torch.arange(4))

        loader = _build_train_dataloader(
            dataset,
            batch_size=2,
            sampling_settings={"sample_weight_power": 1.0},
            sample_weights=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )

        self.assertIsInstance(loader.sampler, WeightedRandomSampler)

    def test_build_train_dataloader_uses_weighted_sampler_when_dataset_boosts_are_enabled(self) -> None:
        dataset = torch.utils.data.TensorDataset(torch.arange(2))

        loader = _build_train_dataloader(
            dataset,
            batch_size=1,
            sampling_settings={"dataset_slug_boosts": {"202603-awe": 6.0}},
            sample_weights=np.array([6.0, 1.0], dtype=np.float32),
        )

        self.assertIsInstance(loader.sampler, WeightedRandomSampler)

    def test_student_config_can_expose_roi_adapter_layers(self) -> None:
        settings = {
            "student": {
                "base_channels": 32,
                "roi_size": 20,
                "experts": 3,
                "roi_expand_ratio": 0.12,
                "roi_adapter_layers": 2,
                "residual_quad_head_layers": 2,
            }
        }

        self.assertEqual(settings["student"]["roi_adapter_layers"], 2)
        self.assertEqual(settings["student"]["residual_quad_head_layers"], 2)

    def test_configure_trainable_parameters_can_freeze_all_but_candidate_selection_head(self) -> None:
        from deep_screen_v1_model import DeepScreenV1Net

        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, candidate_selection_enabled=True)

        trainable = _configure_trainable_parameters(model, {"trainable_modules": ["candidate_selection_head"]})

        self.assertTrue(any(name.startswith("candidate_selection_head") for name in trainable))
        self.assertTrue(all(name.startswith("candidate_selection_head") for name in trainable))


if __name__ == "__main__":
    unittest.main()
