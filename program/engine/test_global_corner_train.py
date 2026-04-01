from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from corner_train import CornerHeatmapNet
from global_corner_train import (
    GlobalCornerDataset,
    build_adaptive_teacher_target,
    build_multi_teacher_target,
    apply_global_perspective_augmentation,
    bootstrap_inward_geometry_scale,
    build_global_feature_tensor,
    build_effective_split_dir,
    compute_edge_supervision_loss,
    compute_corner_weighted_coord_loss,
    compute_global_geometry_loss,
    denormalize_corners,
    export_global_corner_split,
    initialize_global_model_from_checkpoint,
    is_bootstrap_inward_hard_page,
    is_geometry_priority_low_contrast_scene,
    is_legacy_dark_muted_scene,
    reframe_image_and_corners,
    train_global_corner_model,
)


class GlobalCornerTrainTests(unittest.TestCase):
    def test_reframe_image_and_corners_shrinks_and_shifts_quad(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :] = (25, 50, 75)
        corners = np.array(
            [[0.10, 0.20], [0.90, 0.20], [0.90, 0.80], [0.10, 0.80]],
            dtype=np.float32,
        )

        reframed_image, reframed_corners = reframe_image_and_corners(
            image,
            corners,
            scale=0.8,
            shift_x=0.1,
            shift_y=-0.05,
        )

        self.assertEqual(reframed_image.shape, image.shape)
        np.testing.assert_allclose(
            reframed_corners,
            np.array(
                [[0.28, 0.21], [0.92, 0.21], [0.92, 0.69], [0.28, 0.69]],
                dtype=np.float32,
            ),
            atol=1e-4,
        )

    def test_apply_global_perspective_augmentation_warps_quad_geometry(self) -> None:
        image = np.full((100, 200, 3), 180, dtype=np.uint8)
        corners = np.array(
            [[0.10, 0.20], [0.90, 0.20], [0.90, 0.80], [0.10, 0.80]],
            dtype=np.float32,
        )
        jitter = np.array(
            [[0.0, -8.0], [0.0, 4.0], [0.0, 0.0], [0.0, 10.0]],
            dtype=np.float32,
        )

        warped_image, warped_corners = apply_global_perspective_augmentation(image, corners, jitter=jitter)

        self.assertEqual(warped_image.shape, image.shape)
        self.assertEqual(warped_corners.shape, corners.shape)
        self.assertGreater(float(warped_corners[1, 1]), float(warped_corners[0, 1]))
        self.assertGreater(float(warped_corners[3, 1]), float(warped_corners[0, 1]))

    def test_denormalize_corners_restores_image_coordinates(self) -> None:
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

        restored = denormalize_corners(corners, 400, 300)

        self.assertEqual(restored.tolist(), [[0.0, 0.0], [400.0, 0.0], [400.0, 300.0], [0.0, 300.0]])

    def test_global_corner_dataset_returns_image_and_heatmaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((120, 200, 3), dtype=np.uint8)
            cv2.rectangle(image, (20, 20), (180, 100), (240, 240, 240), -1)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=False)
            item = dataset[0]

        self.assertEqual(tuple(item["image"].shape), (3, 128, 128))
        self.assertEqual(tuple(item["heatmaps"].shape), (4, 32, 32))
        self.assertEqual(tuple(item["corners"].shape), (4, 2))

    def test_global_corner_dataset_supports_rgb_gray_border_feature_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((120, 200, 3), dtype=np.uint8)
            image[:, :] = (30, 90, 150)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(
                manifest,
                input_size=128,
                output_size=32,
                augment=False,
                feature_mode="rgb_gray_border",
            )
            item = dataset[0]

        self.assertEqual(tuple(item["image"].shape), (5, 128, 128))
        self.assertGreater(float(item["image"][3].mean()), 0.0)
        self.assertGreater(float(item["image"][4][:8, :].mean()), 0.0)
        self.assertAlmostEqual(float(item["image"][4][64, 64]), 0.0, places=5)

    def test_build_sample_weights_supports_failure_layer_category_boosts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "train.jsonl"
            rows = [
                {"page_id": "base", "image_path": "/tmp/base.jpg", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                {
                    "page_id": "runtime",
                    "image_path": "/tmp/runtime.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "failure_layer_category": "runtime_candidate_recoverable",
                },
                {
                    "page_id": "opencv",
                    "image_path": "/tmp/opencv.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "failure_layer_category": "opencv_recoverable",
                },
                {
                    "page_id": "hard",
                    "image_path": "/tmp/hard.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "failure_layer_category": "hard_both_fail",
                },
            ]
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, augment=False)
            weights = dataset.build_sample_weights(
                failure_layer_runtime_boost=1.2,
                failure_layer_opencv_boost=1.5,
                failure_layer_hard_boost=1.8,
            )

        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertGreater(float(weights[2]), float(weights[1]))
        self.assertGreater(float(weights[3]), float(weights[2]))

    def test_build_sample_weights_supports_failure_layer_gain_power(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "train.jsonl"
            rows = [
                {
                    "page_id": "low-gain",
                    "image_path": "/tmp/low.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "failure_layer_category": "opencv_recoverable",
                    "failure_layer_union_gain": 0.02,
                },
                {
                    "page_id": "high-gain",
                    "image_path": "/tmp/high.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "failure_layer_category": "opencv_recoverable",
                    "failure_layer_union_gain": 0.20,
                },
            ]
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, augment=False)
            weights = dataset.build_sample_weights(
                failure_layer_opencv_boost=1.0,
                failure_layer_gain_power=1.0,
            )

        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_build_sample_weights_can_project_balance_failure_layer_gains(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "train.jsonl"
            rows = [
                {
                    "page_id": "hz-1",
                    "project_name": "杭州A",
                    "image_path": "/tmp/hz1.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "failure_layer_category": "opencv_recoverable",
                    "failure_layer_union_gain": 0.20,
                },
                {
                    "page_id": "hz-2",
                    "project_name": "杭州A",
                    "image_path": "/tmp/hz2.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "failure_layer_category": "opencv_recoverable",
                    "failure_layer_union_gain": 0.20,
                },
                {
                    "page_id": "jy-1",
                    "project_name": "金溢B",
                    "image_path": "/tmp/jy1.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "failure_layer_category": "opencv_recoverable",
                    "failure_layer_union_gain": 0.02,
                },
            ]
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, augment=False)
            unbalanced = dataset.build_sample_weights(
                failure_layer_gain_power=1.0,
            )
            balanced = dataset.build_sample_weights(
                failure_layer_gain_power=1.0,
                failure_layer_project_balance=True,
            )

        self.assertGreater(float(unbalanced[0]), float(unbalanced[2]))
        self.assertLess(float(balanced[0] - balanced[2]), float(unbalanced[0] - unbalanced[2]))

    def test_build_sample_weights_can_balance_projects_by_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "train.jsonl"
            rows = [
                {
                    "page_id": "major-1",
                    "project_name": "主项目",
                    "image_path": "/tmp/major1.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                },
                {
                    "page_id": "major-2",
                    "project_name": "主项目",
                    "image_path": "/tmp/major2.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                },
                {
                    "page_id": "major-3",
                    "project_name": "主项目",
                    "image_path": "/tmp/major3.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                },
                {
                    "page_id": "minor-1",
                    "project_name": "次项目",
                    "image_path": "/tmp/minor1.jpg",
                    "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                },
            ]
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, augment=False)
            unbalanced = dataset.build_sample_weights()
            balanced = dataset.build_sample_weights(project_balance_power=1.0)

        self.assertAlmostEqual(float(unbalanced[0]), float(unbalanced[3]), places=6)
        self.assertGreater(float(balanced[3]), float(balanced[0]))

    def test_build_global_feature_tensor_adds_border_gray_channels(self) -> None:
        image = np.full((10, 12, 3), 128, dtype=np.uint8)

        feature = build_global_feature_tensor(image, feature_mode="rgb_gray_border")

        self.assertEqual(feature.shape, (5, 10, 12))
        self.assertTrue(np.allclose(feature[3], 128.0 / 255.0, atol=1e-6))
        self.assertGreater(float(feature[4, 0, 0]), 0.0)
        self.assertGreater(float(feature[4, -1, -1]), 0.0)
        self.assertAlmostEqual(float(feature[4, 5, 6]), 0.0, places=6)

    def test_initialize_global_model_from_checkpoint_inflates_stem_input_channels(self) -> None:
        source_model = CornerHeatmapNet(in_channels=3, channels=16, output_channels=4)
        target_model = CornerHeatmapNet(in_channels=5, channels=16, output_channels=4)
        checkpoint = {"state_dict": source_model.state_dict()}

        initialize_global_model_from_checkpoint(target_model, checkpoint)

        target_weight = target_model.state_dict()["stem.block.0.weight"]
        source_weight = checkpoint["state_dict"]["stem.block.0.weight"]
        np.testing.assert_allclose(
            target_weight[:, :3].detach().cpu().numpy(),
            source_weight.detach().cpu().numpy(),
            atol=1e-6,
        )
        expected_fill = source_weight.mean(dim=1, keepdim=True).detach().cpu().numpy()
        np.testing.assert_allclose(
            target_weight[:, 3:4].detach().cpu().numpy(),
            expected_fill,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            target_weight[:, 4:5].detach().cpu().numpy(),
            expected_fill,
            atol=1e-6,
        )

    def test_compute_global_geometry_loss_is_zero_for_identical_quads(self) -> None:
        corners = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )

        loss, parts = compute_global_geometry_loss(corners, corners)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertAlmostEqual(float(parts["max_corner"]), 0.0, places=6)
        self.assertAlmostEqual(float(parts["edge_direction"]), 0.0, places=6)
        self.assertAlmostEqual(float(parts["edge_line_offset"]), 0.0, places=6)
        self.assertAlmostEqual(float(parts["edge_length_ratio"]), 0.0, places=6)
        self.assertAlmostEqual(float(parts["corner_line"]), 0.0, places=6)
        self.assertAlmostEqual(float(parts["corner_angle"]), 0.0, places=6)
        self.assertAlmostEqual(float(parts["inset"]), 0.0, places=6)

    def test_compute_global_geometry_loss_penalizes_inset_and_tilt(self) -> None:
        target = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )
        inset_tilt = torch.tensor(
            [[[0.2, 0.18], [0.8, 0.08], [0.72, 0.7], [0.18, 0.74]]],
            dtype=torch.float32,
        )

        loss, parts = compute_global_geometry_loss(inset_tilt, target)

        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(float(parts["max_corner"]), 0.0)
        self.assertGreater(float(parts["edge_direction"]), 0.0)
        self.assertGreater(float(parts["edge_line_offset"]), 0.0)
        self.assertGreater(float(parts["edge_length_ratio"]), 0.0)
        self.assertGreater(float(parts["corner_line"]), 0.0)
        self.assertGreater(float(parts["corner_angle"]), 0.0)
        self.assertGreater(float(parts["inset"]), 0.0)

    def test_compute_global_geometry_loss_respects_inset_weight(self) -> None:
        target = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )
        inset_only = torch.tensor(
            [[[0.2, 0.15], [0.8, 0.15], [0.8, 0.75], [0.2, 0.75]]],
            dtype=torch.float32,
        )

        low_loss, low_parts = compute_global_geometry_loss(inset_only, target, inset_weight=0.1)
        high_loss, high_parts = compute_global_geometry_loss(inset_only, target, inset_weight=1.0)

        self.assertGreater(float(low_parts["inset"]), 0.0)
        self.assertGreater(float(high_parts["inset"]), 0.0)
        self.assertGreater(float(high_loss.item()), float(low_loss.item()))

    def test_compute_global_geometry_loss_exposes_inward_boundary_penalty(self) -> None:
        target = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )
        inward_only = torch.tensor(
            [[[0.16, 0.16], [0.84, 0.16], [0.84, 0.74], [0.16, 0.74]]],
            dtype=torch.float32,
        )

        low_loss, low_parts = compute_global_geometry_loss(inward_only, target, inward_boundary_weight=0.1)
        high_loss, high_parts = compute_global_geometry_loss(inward_only, target, inward_boundary_weight=1.0)

        self.assertIn("inward_boundary", low_parts)
        self.assertGreater(float(low_parts["inward_boundary"]), 0.0)
        self.assertGreater(float(high_parts["inward_boundary"]), 0.0)
        self.assertGreater(float(high_loss.item()), float(low_loss.item()))

    def test_compute_global_geometry_loss_inward_boundary_margin_ignores_tiny_shrink(self) -> None:
        target = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )
        tiny_inward = torch.tensor(
            [[[0.105, 0.105], [0.895, 0.105], [0.895, 0.795], [0.105, 0.795]]],
            dtype=torch.float32,
        )

        _, zero_parts = compute_global_geometry_loss(
            tiny_inward,
            target,
            inward_boundary_weight=1.0,
            inward_boundary_margin=0.01,
        )
        _, active_parts = compute_global_geometry_loss(
            tiny_inward,
            target,
            inward_boundary_weight=1.0,
            inward_boundary_margin=0.0,
        )

        self.assertAlmostEqual(float(zero_parts["inward_boundary"]), 0.0, places=6)
        self.assertGreater(float(active_parts["inward_boundary"]), 0.0)

    def test_compute_global_geometry_loss_exposes_quad_mask_penalty(self) -> None:
        target = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )
        inward_only = torch.tensor(
            [[[0.18, 0.16], [0.82, 0.16], [0.82, 0.74], [0.18, 0.74]]],
            dtype=torch.float32,
        )

        low_loss, low_parts = compute_global_geometry_loss(inward_only, target, quad_mask_weight=0.1)
        high_loss, high_parts = compute_global_geometry_loss(inward_only, target, quad_mask_weight=0.8)

        self.assertIn("quad_mask", low_parts)
        self.assertGreater(float(low_parts["quad_mask"]), 0.0)
        self.assertGreater(float(high_parts["quad_mask"]), 0.0)
        self.assertGreater(float(high_loss.item()), float(low_loss.item()))

    def test_compute_global_geometry_loss_exposes_inner_boundary_band_penalty(self) -> None:
        target = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )
        inward_only = torch.tensor(
            [[[0.16, 0.14], [0.84, 0.14], [0.84, 0.76], [0.16, 0.76]]],
            dtype=torch.float32,
        )

        low_loss, low_parts = compute_global_geometry_loss(inward_only, target, inner_boundary_band_weight=0.05)
        high_loss, high_parts = compute_global_geometry_loss(inward_only, target, inner_boundary_band_weight=0.4)

        self.assertIn("inner_boundary_band", low_parts)
        self.assertGreater(float(low_parts["inner_boundary_band"]), 0.0)
        self.assertGreater(float(high_parts["inner_boundary_band"]), 0.0)
        self.assertGreater(float(high_loss.item()), float(low_loss.item()))

    def test_compute_global_geometry_loss_respects_edge_line_weight(self) -> None:
        target = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )
        edge_line_offset_only = torch.tensor(
            [[[0.1, 0.18], [0.9, 0.18], [0.9, 0.72], [0.1, 0.72]]],
            dtype=torch.float32,
        )

        low_loss, low_parts = compute_global_geometry_loss(edge_line_offset_only, target, edge_line_weight=0.2)
        high_loss, high_parts = compute_global_geometry_loss(edge_line_offset_only, target, edge_line_weight=1.5)

        self.assertGreater(float(low_parts["edge_line_offset"]), 0.0)
        self.assertGreater(float(high_parts["edge_line_offset"]), 0.0)
        self.assertGreater(float(high_loss.item()), float(low_loss.item()))

    def test_compute_global_geometry_loss_exposes_and_penalizes_edge_collapse(self) -> None:
        target = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )
        mildly_short = torch.tensor(
            [[[0.1, 0.1], [0.78, 0.12], [0.86, 0.78], [0.12, 0.82]]],
            dtype=torch.float32,
        )
        collapsed = torch.tensor(
            [[[0.1, 0.1], [0.18, 0.11], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )

        mild_loss, mild_parts = compute_global_geometry_loss(mildly_short, target)
        collapsed_loss, collapsed_parts = compute_global_geometry_loss(collapsed, target)

        self.assertIn("edge_collapse", mild_parts)
        self.assertGreater(float(collapsed_parts["edge_collapse"]), float(mild_parts["edge_collapse"]))
        self.assertGreater(float(collapsed_loss.item()), float(mild_loss.item()))

    def test_compute_edge_supervision_loss_is_near_zero_for_identical_quads(self) -> None:
        corners = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )

        loss = compute_edge_supervision_loss(corners, corners)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=5)

    def test_compute_edge_supervision_loss_penalizes_shifted_left_edge(self) -> None:
        target = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )
        shifted = torch.tensor(
            [[[0.22, 0.14], [0.9, 0.1], [0.9, 0.8], [0.22, 0.84]]],
            dtype=torch.float32,
        )

        loss = compute_edge_supervision_loss(shifted, target)

        self.assertGreater(float(loss.item()), 0.01)

    def test_global_corner_dataset_build_sample_weights_uses_adaptive_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((120, 200, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            rows = [
                {
                    "page_id": "easy",
                    "image_path": str(image_path),
                    "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                    "adaptive_weight": 1.0,
                },
                {
                    "page_id": "hard",
                    "image_path": str(image_path),
                    "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                    "adaptive_weight": 2.0,
                },
            ]
            manifest = root / "train.jsonl"
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=False)
            weights = dataset.build_sample_weights(power=1.0)

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_global_corner_dataset_build_sample_weights_boosts_disagreement_and_geometry_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((120, 200, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            rows = [
                {
                    "page_id": "baseline",
                    "image_path": str(image_path),
                    "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                    "adaptive_weight": 1.0,
                    "teacher_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                    "teacher_r3_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                },
                {
                    "page_id": "disagreement",
                    "image_path": str(image_path),
                    "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                    "adaptive_weight": 1.0,
                    "teacher_quad": [[30, 28], [172, 22], [176, 96], [24, 98]],
                    "teacher_r3_quad": [[16, 16], [184, 16], [184, 104], [16, 104]],
                },
                {
                    "page_id": "geometry",
                    "image_path": str(image_path),
                    "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                    "adaptive_weight": 1.0,
                    "scene_tags": ["near_color_background", "low_contrast_scene", "black_frame_scene"],
                    "scene_profile": {"lab_distance": 19.0, "luma_delta": 0.06, "edge_strength": 0.42},
                },
            ]
            manifest = root / "train.jsonl"
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=False)
            weights = dataset.build_sample_weights(
                power=0.0,
                disagreement_floor=0.01,
                disagreement_boost=1.5,
                geometry_priority_boost=1.25,
            )

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[1]), float(weights[0]))
        self.assertGreater(float(weights[2]), float(weights[0]))

    def test_global_corner_dataset_build_sample_weights_boosts_high_hardness_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((120, 200, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            rows = [
                {
                    "page_id": "easy",
                    "image_path": str(image_path),
                    "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                    "adaptive_weight": 1.0,
                    "hardness_score": 0.5,
                },
                {
                    "page_id": "hard",
                    "image_path": str(image_path),
                    "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                    "adaptive_weight": 1.0,
                    "hardness_score": 8.0,
                },
            ]
            manifest = root / "train.jsonl"
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=False)
            weights = dataset.build_sample_weights(power=0.0, hardness_score_power=0.5)

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_global_corner_dataset_reads_scene_tags_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 128, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                "scene_tags": ["near_color_background", "low_contrast_scene"],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=False)
            item = dataset[0]

        self.assertEqual(dataset.rows[0]["scene_tags"], ["near_color_background", "low_contrast_scene"])
        self.assertGreater(float(item["geometry_scale"]), 1.0)

    def test_geometry_priority_low_contrast_scene_uses_profile_even_without_tags(self) -> None:
        row = {
            "scene_tags": [],
            "scene_profile": {
                "lab_distance": 21.2,
                "luma_delta": 0.077,
                "edge_strength": 0.49,
                "inner_border_contrast": 0.01,
            },
        }

        self.assertTrue(is_geometry_priority_low_contrast_scene(row))

    def test_geometry_priority_low_contrast_scene_rejects_high_separation_profile(self) -> None:
        row = {
            "scene_tags": [],
            "scene_profile": {
                "lab_distance": 47.0,
                "luma_delta": 0.10,
                "edge_strength": 0.38,
                "inner_border_contrast": 0.04,
            },
        }

        self.assertFalse(is_geometry_priority_low_contrast_scene(row))

    def test_global_corner_dataset_geometry_scale_is_lower_for_untagged_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 128, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=False)
            item = dataset[0]

        self.assertLess(float(item["geometry_scale"]), 1.0)

    def test_bootstrap_inward_geometry_scale_boosts_only_positive_inset_failures(self) -> None:
        self.assertAlmostEqual(
            bootstrap_inward_geometry_scale(
                {
                    "bootstrap_metrics": {
                        "quad_inset_ratio": 0.0,
                        "screen_relative_point_error": 0.02,
                        "max_corner_error": 0.08,
                    }
                }
            ),
            1.0,
            places=6,
        )
        boosted = bootstrap_inward_geometry_scale(
            {
                "bootstrap_metrics": {
                    "quad_inset_ratio": 0.12,
                    "screen_relative_point_error": 0.05,
                    "max_corner_error": 0.16,
                }
            }
        )
        self.assertGreater(boosted, 1.3)

    def test_global_corner_dataset_geometry_scale_uses_bootstrap_inward_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 128, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                "bootstrap_metrics": {
                    "quad_inset_ratio": 0.10,
                    "screen_relative_point_error": 0.04,
                    "max_corner_error": 0.12,
                },
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=False)
            item = dataset[0]

        self.assertGreater(float(item["geometry_scale"]), 1.2)

    def test_build_adaptive_teacher_target_blends_only_reliable_teacher_corners(self) -> None:
        manual = np.array(
            [[0.10, 0.10], [0.90, 0.10], [0.90, 0.80], [0.10, 0.80]],
            dtype=np.float32,
        )
        teacher = np.array(
            [[0.12, 0.11], [0.88, 0.11], [0.70, 0.68], [0.09, 0.80]],
            dtype=np.float32,
        )

        target, guidance_scale = build_adaptive_teacher_target(
            manual,
            teacher,
            blend_ratio=0.5,
            corner_error_max=0.04,
            sample_error_max=0.08,
        )

        np.testing.assert_allclose(target[0], np.array([0.11, 0.105], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(target[1], np.array([0.89, 0.105], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(target[2], manual[2], atol=1e-6)
        np.testing.assert_allclose(target[3], np.array([0.095, 0.80], dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(guidance_scale, 0.75, places=6)

    def test_build_multi_teacher_target_selects_best_corner_per_source(self) -> None:
        manual = np.array(
            [[0.10, 0.10], [0.90, 0.10], [0.90, 0.80], [0.10, 0.80]],
            dtype=np.float32,
        )
        candidate_quads = np.array(
            [
                [[0.105, 0.105], [0.96, 0.12], [0.92, 0.83], [0.15, 0.83]],
                [[0.13, 0.12], [0.905, 0.105], [0.89, 0.79], [0.10, 0.96]],
                [[0.12, 0.11], [0.92, 0.11], [0.902, 0.801], [0.098, 0.802]],
            ],
            dtype=np.float32,
        )

        target, guidance_scale, selected_index = build_multi_teacher_target(
            manual,
            candidate_quads,
            blend_ratio=1.0,
            corner_error_max=0.03,
        )

        expected = np.array(
            [[0.105, 0.105], [0.905, 0.105], [0.902, 0.801], [0.098, 0.802]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(target, expected, atol=1e-6)
        self.assertAlmostEqual(guidance_scale, 1.0, places=6)
        self.assertEqual(selected_index.tolist(), [0, 1, 2, 2])

    def test_build_multi_teacher_target_ignores_masked_candidates(self) -> None:
        manual = np.array(
            [[0.10, 0.10], [0.90, 0.10], [0.90, 0.80], [0.10, 0.80]],
            dtype=np.float32,
        )
        candidate_quads = np.array(
            [
                [[0.1005, 0.1005], [0.9005, 0.1005], [0.9005, 0.8005], [0.1005, 0.8005]],
                [[0.11, 0.11], [0.91, 0.11], [0.91, 0.81], [0.11, 0.81]],
            ],
            dtype=np.float32,
        )
        candidate_mask = np.array([False, True], dtype=bool)

        target, guidance_scale, selected_index = build_multi_teacher_target(
            manual,
            candidate_quads,
            blend_ratio=1.0,
            candidate_mask=candidate_mask,
            corner_error_max=0.03,
        )

        np.testing.assert_allclose(target, candidate_quads[1], atol=1e-6)
        self.assertAlmostEqual(guidance_scale, 1.0, places=6)
        self.assertEqual(selected_index.tolist(), [1, 1, 1, 1])

    def test_global_corner_dataset_exposes_teacher_target_and_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 128, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                "teacher_quad": [[24, 21], [176, 21], [180, 100], [20, 100]],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(
                manifest,
                input_size=128,
                output_size=32,
                augment=False,
                teacher_blend_ratio=0.5,
                teacher_corner_error_max=0.03,
                teacher_sample_error_max=0.03,
            )
            item = dataset[0]

        self.assertIn("teacher_target", item)
        self.assertIn("teacher_guidance_scale", item)
        self.assertGreater(float(item["teacher_guidance_scale"]), 0.0)
        np.testing.assert_allclose(
            item["teacher_target"][0].numpy(),
            np.array([0.11, 0.17083333], dtype=np.float32),
            atol=1e-6,
        )

    def test_global_corner_dataset_can_build_oracle_teacher_target_from_multiple_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 128, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                "teacher_quad": [[24, 24], [184, 24], [180, 100], [20, 100]],
                "teacher_r3_quad": [[21, 21], [192, 23], [182, 103], [28, 99]],
                "teacher_roi_quad": [[23, 22], [181, 21], [180.5, 100.5], [19.5, 100.5]],
                "opencv_best_quad": [[20.2, 20.2], [180.2, 20.2], [180.2, 100.2], [20.2, 100.2]],
                "opencv_best_score": 0.62,
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(
                manifest,
                input_size=128,
                output_size=32,
                augment=False,
                teacher_blend_ratio=1.0,
                teacher_corner_error_max=0.03,
                teacher_sample_error_max=0.03,
                teacher_target_mode="oracle",
                teacher_candidate_sources=("teacher", "r3", "roi", "opencv"),
                teacher_opencv_score_min=0.5,
            )
            item = dataset[0]

        self.assertIn("teacher_target", item)
        self.assertIn("teacher_guidance_scale", item)
        self.assertGreater(float(item["teacher_guidance_scale"]), 0.0)
        np.testing.assert_allclose(
            item["teacher_target"].numpy(),
            np.array(
                [
                    [0.101, 0.16833334],
                    [0.901, 0.16833334],
                    [0.9, 0.8333333],
                    [0.1, 0.8333333],
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_global_corner_dataset_disables_teacher_guidance_when_teacher_r3_disagreement_is_small(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 128, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[20, 20], [180, 20], [180, 100], [20, 100]],
                "teacher_quad": [[21, 20], [181, 20], [180, 100], [20, 100]],
                "teacher_r3_quad": [[21.2, 20.1], [181.1, 20.2], [180.1, 100.2], [20.1, 100.1]],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(
                manifest,
                input_size=128,
                output_size=32,
                augment=False,
                teacher_blend_ratio=1.0,
                teacher_corner_error_max=0.03,
                teacher_sample_error_max=0.03,
                teacher_target_mode="oracle",
                teacher_candidate_sources=("teacher", "r3"),
                teacher_activation_min_disagreement=0.01,
            )
            item = dataset[0]

        self.assertAlmostEqual(float(item["teacher_guidance_scale"]), 0.0, places=6)

    def test_is_bootstrap_inward_hard_page_requires_positive_inset_failure(self) -> None:
        self.assertFalse(
            is_bootstrap_inward_hard_page(
                {
                    "bootstrap_metrics": {
                        "quad_inset_ratio": -0.12,
                        "screen_relative_point_error": 0.09,
                        "max_corner_error": 0.18,
                    }
                }
            )
        )
        self.assertTrue(
            is_bootstrap_inward_hard_page(
                {
                    "bootstrap_metrics": {
                        "quad_inset_ratio": 0.12,
                        "screen_relative_point_error": 0.09,
                        "max_corner_error": 0.18,
                    }
                }
            )
        )

    def test_global_corner_dataset_edge_scale_is_higher_for_narrow_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 128, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            rows = [
                {
                    "image_path": str(image_path),
                    "manual_quad": [[10, 10], [190, 10], [190, 100], [10, 100]],
                },
                {
                    "image_path": str(image_path),
                    "manual_quad": [[40, 10], [160, 10], [160, 100], [40, 100]],
                },
            ]
            manifest = root / "train.jsonl"
            manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=False)
            wide_item = dataset[0]
            narrow_item = dataset[1]

        self.assertGreater(float(narrow_item["edge_scale"]), float(wide_item["edge_scale"]))

    def test_global_corner_dataset_legacy_r3_profile_disables_scene_scaling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 128, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[40, 10], [160, 10], [160, 100], [40, 100]],
                "scene_tags": ["near_color_background", "low_contrast_scene", "black_frame_scene"],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            default_dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=False)
            legacy_dataset = GlobalCornerDataset(
                manifest,
                input_size=128,
                output_size=32,
                augment=False,
                training_profile="legacy_r3",
            )
            default_item = default_dataset[0]
            legacy_item = legacy_dataset[0]

        self.assertGreater(float(default_item["geometry_scale"]), 1.0)
        self.assertGreater(float(default_item["edge_scale"]), 1.0)
        self.assertAlmostEqual(float(legacy_item["geometry_scale"]), 1.0, places=6)
        self.assertAlmostEqual(float(legacy_item["edge_scale"]), 1.0, places=6)

    def test_global_corner_dataset_augment_can_synthesize_narrower_screen_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 180, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[10, 12], [190, 10], [188, 108], [12, 110]],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=True)
            original_corners = np.array(
                [[0.05, 0.10], [0.95, 0.08333334], [0.94, 0.90], [0.06, 0.9166667]],
                dtype=np.float32,
            )
            row = {"scene_tags": []}

            with (
                mock.patch("global_corner_train.random.random", side_effect=[0.9, 0.1, 0.9, 0.9]),
                mock.patch(
                    "global_corner_train.random.uniform",
                    side_effect=[0.86, 0.06, -0.02, 1.0, 0.0],
                ),
            ):
                _, augmented_corners = dataset._augment(image.copy(), original_corners.copy(), row)

        original_width = float(original_corners[:, 0].max() - original_corners[:, 0].min())
        augmented_width = float(augmented_corners[:, 0].max() - augmented_corners[:, 0].min())
        self.assertLess(augmented_width, original_width)
        self.assertGreater(float(augmented_corners[:, 0].min()), float(original_corners[:, 0].min()))

    def test_global_corner_dataset_augment_can_shift_centered_screen_sideways(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 180, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[30, 12], [170, 10], [168, 108], [32, 110]],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=True)
            original_corners = np.array(
                [[0.15, 0.10], [0.85, 0.08333334], [0.84, 0.90], [0.16, 0.9166667]],
                dtype=np.float32,
            )
            row = {"scene_tags": []}

            with (
                mock.patch("global_corner_train.random.random", side_effect=[0.9, 0.1, 0.9, 0.9]),
                mock.patch(
                    "global_corner_train.random.uniform",
                    side_effect=[0.94, 0.06, 0.01, 1.0, 0.0],
                ),
            ):
                _, augmented_corners = dataset._augment(image.copy(), original_corners.copy(), row)

        original_center = float(original_corners[:, 0].mean())
        augmented_center = float(augmented_corners[:, 0].mean())
        self.assertGreater(augmented_center, original_center + 0.02)

    def test_global_corner_dataset_legacy_r3_augment_can_synthesize_small_screen_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 180, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[30, 16], [170, 14], [168, 108], [32, 110]],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(
                manifest,
                input_size=128,
                output_size=32,
                augment=True,
                training_profile="legacy_r3",
            )
            original_corners = np.array(
                [[0.15, 0.13333334], [0.85, 0.11666667], [0.84, 0.90], [0.16, 0.9166667]],
                dtype=np.float32,
            )
            row = {"scene_tags": []}

            with (
                mock.patch("global_corner_train.random.random", side_effect=[0.9, 0.1, 0.9]),
                mock.patch(
                    "global_corner_train.random.uniform",
                    side_effect=[0.76, 0.02, -0.03, 1.0, 0.0],
                ),
            ):
                _, augmented_corners = dataset._augment(image.copy(), original_corners.copy(), row)

        original_width = float(original_corners[:, 0].max() - original_corners[:, 0].min())
        augmented_width = float(augmented_corners[:, 0].max() - augmented_corners[:, 0].min())
        self.assertLess(augmented_width, original_width - 0.08)

    def test_is_legacy_dark_muted_scene_detects_dark_low_saturation_image(self) -> None:
        image = np.full((80, 120, 3), (88, 84, 96), dtype=np.uint8)

        self.assertTrue(is_legacy_dark_muted_scene(image))

    def test_is_legacy_dark_muted_scene_rejects_bright_image(self) -> None:
        image = np.full((80, 120, 3), (170, 160, 190), dtype=np.uint8)

        self.assertFalse(is_legacy_dark_muted_scene(image))

    def test_global_corner_dataset_legacy_r3_augment_can_apply_dark_muted_scene_perturbation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), (88, 84, 96), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[30, 16], [170, 14], [168, 108], [32, 110]],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(
                manifest,
                input_size=128,
                output_size=32,
                augment=True,
                training_profile="legacy_r3",
            )
            corners = np.array(
                [[0.15, 0.13333334], [0.85, 0.11666667], [0.84, 0.90], [0.16, 0.9166667]],
                dtype=np.float32,
            )
            row = {"scene_tags": []}

            with (
                mock.patch("global_corner_train.random.random", side_effect=[0.9, 0.9, 0.9, 0.1, 0.2]),
                mock.patch("global_corner_train.random.uniform", side_effect=[1.0, 0.0]),
            ):
                augmented_image, augmented_corners = dataset._augment(image.copy(), corners.copy(), row)

        self.assertGreater(float(np.mean(np.abs(augmented_image.astype(np.float32) - image.astype(np.float32)))), 1.0)
        np.testing.assert_allclose(augmented_corners, corners, atol=1e-6)

    def test_global_corner_dataset_legacy_r3_augment_can_synthesize_distant_small_screen_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), (88, 84, 96), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[20, 16], [180, 14], [176, 108], [24, 110]],
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(
                manifest,
                input_size=128,
                output_size=32,
                augment=True,
                training_profile="legacy_r3",
            )
            original_corners = np.array(
                [[0.10, 0.13333334], [0.90, 0.11666667], [0.88, 0.90], [0.12, 0.9166667]],
                dtype=np.float32,
            )
            row = {"scene_tags": []}

            with (
                mock.patch("global_corner_train.random.random", side_effect=[0.9, 0.05, 0.2, 0.9]),
                mock.patch(
                    "global_corner_train.random.uniform",
                    side_effect=[0.56, 0.0, -0.06, 0.34, 0.82, 0.22, 0.36, 1.0, 0.0, 0.0],
                ),
            ):
                _, augmented_corners = dataset._augment(image.copy(), original_corners.copy(), row)

        original_width = float(original_corners[:, 0].max() - original_corners[:, 0].min())
        augmented_width = float(augmented_corners[:, 0].max() - augmented_corners[:, 0].min())
        self.assertLess(augmented_width, 0.55)
        self.assertLess(augmented_width, original_width - 0.18)
        self.assertLess(float(augmented_corners[:, 1].mean()), float(original_corners[:, 1].mean()))

    def test_global_corner_dataset_bootstrap_inward_hard_page_can_synthesize_distant_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((120, 200, 3), 150, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            sample = {
                "image_path": str(image_path),
                "manual_quad": [[20, 16], [180, 14], [176, 108], [24, 110]],
                "bootstrap_metrics": {
                    "quad_inset_ratio": 0.18,
                    "screen_relative_point_error": 0.09,
                    "max_corner_error": 0.22,
                },
            }
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = GlobalCornerDataset(manifest, input_size=128, output_size=32, augment=True)
            original_corners = np.array(
                [[0.10, 0.13333334], [0.90, 0.11666667], [0.88, 0.90], [0.12, 0.9166667]],
                dtype=np.float32,
            )
            row = sample

            with (
                mock.patch("global_corner_train.random.random", side_effect=[0.9, 0.18, 0.9, 0.9]),
                mock.patch(
                    "global_corner_train.random.uniform",
                    side_effect=[0.82, 0.01, -0.05, 1.0, 0.0],
                ),
            ):
                _, augmented_corners = dataset._augment(image.copy(), original_corners.copy(), row)

        original_width = float(original_corners[:, 0].max() - original_corners[:, 0].min())
        augmented_width = float(augmented_corners[:, 0].max() - augmented_corners[:, 0].min())
        self.assertLess(augmented_width, original_width - 0.10)
        self.assertLess(float(augmented_corners[:, 1].mean()), float(original_corners[:, 1].mean()))

    def test_compute_corner_weighted_coord_loss_emphasizes_bottom_corners(self) -> None:
        target = torch.tensor(
            [[[0.1, 0.1], [0.9, 0.1], [0.9, 0.8], [0.1, 0.8]]],
            dtype=torch.float32,
        )
        predicted = target.clone()
        predicted[0, 2, 1] += 0.10
        predicted[0, 3, 1] += 0.10

        base_loss = compute_corner_weighted_coord_loss(predicted, target)
        weighted_loss = compute_corner_weighted_coord_loss(
            predicted,
            target,
            torch.tensor([1.0, 1.0, 1.5, 1.5], dtype=torch.float32),
        )

        self.assertGreater(float(weighted_loss), float(base_loss))

    def test_export_global_corner_split_writes_focus_split_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            project_a = dataset_root / "focus-project"
            project_b = dataset_root / "base-project"
            project_a.mkdir(parents=True)
            project_b.mkdir(parents=True)

            image = np.zeros((80, 120, 3), dtype=np.uint8)
            for project_dir, prefix in ((project_a, "a"), (project_b, "b")):
                pages = []
                for index in range(4):
                    image_path = project_dir / f"{prefix}-{index}.jpg"
                    cv2.imwrite(str(image_path), image)
                    pages.append(
                        {
                            "id": f"{prefix}{index}",
                            "path": image_path.name,
                            "manualQuad": [[10, 10], [110, 10], [110, 70], [10, 70]],
                        }
                    )
                (project_dir / "screen-pdf-project.json").write_text(
                    json.dumps({"pages": pages}, ensure_ascii=False),
                    encoding="utf-8",
                )

            output_dir = root / "split"
            summary = export_global_corner_split(
                dataset_root,
                output_dir,
                seed=5,
                test_ratio=0.25,
                focus_projects=["focus-project"],
                focus_test_ratio=0.5,
            )

            focus_train_path = output_dir / "focus_train.jsonl"
            focus_test_path = output_dir / "focus_test.jsonl"

            self.assertTrue(focus_train_path.exists())
            self.assertTrue(focus_test_path.exists())
            self.assertEqual(summary["focus_project_count"], 1)
            self.assertEqual(summary["focus_test_pages"], 2)
            self.assertEqual(summary["focus_train_pages"], 2)

    def test_export_global_corner_split_writes_holdout_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            for project_name in ("focus-project", "holdout-project", "base-project"):
                project_dir = dataset_root / project_name
                project_dir.mkdir(parents=True)
                image = np.zeros((80, 120, 3), dtype=np.uint8)
                pages = []
                for index in range(2):
                    image_path = project_dir / f"{project_name}-{index}.jpg"
                    cv2.imwrite(str(image_path), image)
                    pages.append(
                        {
                            "id": f"{project_name}-{index}",
                            "path": image_path.name,
                            "manualQuad": [[10, 10], [110, 10], [110, 70], [10, 70]],
                        }
                    )
                (project_dir / "screen-pdf-project.json").write_text(
                    json.dumps({"pages": pages}, ensure_ascii=False),
                    encoding="utf-8",
                )

            output_dir = root / "split"
            summary = export_global_corner_split(
                dataset_root,
                output_dir,
                seed=5,
                test_ratio=0.25,
                focus_projects=["focus-project"],
                focus_test_ratio=0.5,
                holdout_projects=["holdout-project"],
            )

            self.assertTrue((output_dir / "holdout.jsonl").exists())
            self.assertEqual(summary["holdout_pages"], 2)

    def test_build_effective_split_dir_can_merge_focus_rows_into_train_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_dir = root / "split"
            split_dir.mkdir()
            train_row = {"page_id": "train-base", "image_path": "/tmp/base.jpg", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            test_row = {"page_id": "test-base", "image_path": "/tmp/base-test.jpg", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            focus_train_row = {"page_id": "focus-train", "image_path": "/tmp/focus.jpg", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            focus_test_row = {"page_id": "focus-test", "image_path": "/tmp/focus-test.jpg", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            (split_dir / "train.jsonl").write_text(json.dumps(train_row, ensure_ascii=False) + "\n", encoding="utf-8")
            (split_dir / "test.jsonl").write_text(json.dumps(test_row, ensure_ascii=False) + "\n", encoding="utf-8")
            (split_dir / "focus_train.jsonl").write_text(json.dumps(focus_train_row, ensure_ascii=False) + "\n", encoding="utf-8")
            (split_dir / "focus_test.jsonl").write_text(json.dumps(focus_test_row, ensure_ascii=False) + "\n", encoding="utf-8")

            effective_dir = build_effective_split_dir(
                split_dir=split_dir,
                merge_focus_train=True,
                focus_train_repeat=3,
                merge_focus_test=True,
            )

            train_rows = [
                json.loads(line)
                for line in (effective_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            test_rows = [
                json.loads(line)
                for line in (effective_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual([row["page_id"] for row in train_rows], ["train-base", "focus-train", "focus-train", "focus-train"])
        self.assertEqual([row["page_id"] for row in test_rows], ["test-base", "focus-test"])

    def test_build_effective_split_dir_returns_original_split_when_focus_merge_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_dir = root / "split"
            split_dir.mkdir()
            row = {"page_id": "train-base", "image_path": "/tmp/base.jpg", "manual_quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}
            (split_dir / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            (split_dir / "test.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            effective_dir = build_effective_split_dir(split_dir=split_dir)

        self.assertEqual(effective_dir, split_dir)

    def test_train_global_corner_model_legacy_r3_profile_saves_epoch_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((96, 160, 3), 200, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            row = {
                "page_id": "sample",
                "project_name": "demo",
                "image_path": str(image_path),
                "manual_quad": [[16, 12], [144, 12], [144, 84], [16, 84]],
                "scene_tags": ["near_color_background"],
            }
            split_dir = root / "split"
            split_dir.mkdir()
            payload = json.dumps(row, ensure_ascii=False) + "\n"
            for name in ("train", "test"):
                (split_dir / f"{name}.jsonl").write_text(payload, encoding="utf-8")

            output_dir = root / "output"
            result = train_global_corner_model(
                dataset_root=root,
                output_dir=output_dir,
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                input_size=64,
                output_size=16,
                channels=8,
                split_dir=split_dir,
                training_profile="legacy_r3",
                save_epoch_checkpoints=True,
            )

            checkpoint_path = output_dir / "checkpoints" / "epoch_001.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = torch.load(result.model_path, map_location="cpu")
            self.assertEqual(checkpoint["training_profile"], "legacy_r3")
            self.assertAlmostEqual(float(checkpoint["edge_supervision_weight"]), 0.0, places=6)

    def test_train_global_corner_model_saves_inset_weight_in_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((96, 160, 3), 200, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            rows = [
                {
                    "page_id": "sample-a",
                    "project_name": "demo",
                    "image_path": str(image_path),
                    "manual_quad": [[16, 12], [144, 12], [144, 84], [16, 84]],
                },
                {
                    "page_id": "sample-b",
                    "project_name": "demo",
                    "image_path": str(image_path),
                    "manual_quad": [[18, 14], [142, 10], [146, 82], [14, 86]],
                },
            ]
            split_dir = root / "split"
            split_dir.mkdir()
            (split_dir / "train.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            (split_dir / "test.jsonl").write_text(
                json.dumps(rows[0], ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            result = train_global_corner_model(
                dataset_root=root,
                output_dir=root / "output",
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                input_size=64,
                output_size=16,
                channels=8,
                split_dir=split_dir,
                inset_weight=0.75,
            )

            checkpoint = torch.load(result.model_path, map_location="cpu")

        self.assertAlmostEqual(float(checkpoint["inset_weight"]), 0.75, places=6)

    def test_train_global_corner_model_saves_geometry_component_weights_in_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((96, 160, 3), 200, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            row = {
                "page_id": "sample",
                "project_name": "demo",
                "image_path": str(image_path),
                "manual_quad": [[16, 12], [144, 12], [144, 84], [16, 84]],
            }
            split_dir = root / "split"
            split_dir.mkdir()
            payload = json.dumps(row, ensure_ascii=False) + "\n"
            for name in ("train", "test"):
                (split_dir / f"{name}.jsonl").write_text(payload, encoding="utf-8")

            result = train_global_corner_model(
                dataset_root=root,
                output_dir=root / "output",
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                input_size=64,
                output_size=16,
                channels=8,
                split_dir=split_dir,
                max_corner_weight=1.3,
                edge_weight=0.4,
                edge_line_weight=1.1,
                edge_length_weight=0.6,
                corner_line_weight=0.9,
                corner_angle_weight=0.35,
            )

            checkpoint = torch.load(result.model_path, map_location="cpu")

        self.assertAlmostEqual(float(checkpoint["max_corner_weight"]), 1.3, places=6)
        self.assertAlmostEqual(float(checkpoint["edge_weight"]), 0.4, places=6)
        self.assertAlmostEqual(float(checkpoint["edge_line_weight"]), 1.1, places=6)
        self.assertAlmostEqual(float(checkpoint["edge_length_weight"]), 0.6, places=6)
        self.assertAlmostEqual(float(checkpoint["corner_line_weight"]), 0.9, places=6)
        self.assertAlmostEqual(float(checkpoint["corner_angle_weight"]), 0.35, places=6)

    def test_train_global_corner_model_saves_teacher_guidance_metadata_in_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((96, 160, 3), 200, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            row = {
                "page_id": "sample",
                "project_name": "demo",
                "image_path": str(image_path),
                "manual_quad": [[16, 12], [144, 12], [144, 84], [16, 84]],
                "teacher_quad": [[18, 13], [142, 13], [144, 84], [16, 84]],
            }
            split_dir = root / "split"
            split_dir.mkdir()
            payload = json.dumps(row, ensure_ascii=False) + "\n"
            for name in ("train", "test"):
                (split_dir / f"{name}.jsonl").write_text(payload, encoding="utf-8")

            result = train_global_corner_model(
                dataset_root=root,
                output_dir=root / "output",
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                input_size=64,
                output_size=16,
                channels=8,
                split_dir=split_dir,
                teacher_guidance_weight=0.35,
                teacher_blend_ratio=0.5,
                teacher_corner_error_max=0.03,
                teacher_sample_error_max=0.03,
                teacher_activation_min_disagreement=0.01,
            )

            checkpoint = torch.load(result.model_path, map_location="cpu")

        self.assertAlmostEqual(float(checkpoint["teacher_guidance_weight"]), 0.35, places=6)
        self.assertAlmostEqual(float(checkpoint["teacher_blend_ratio"]), 0.5, places=6)
        self.assertAlmostEqual(float(checkpoint["teacher_corner_error_max"]), 0.03, places=6)
        self.assertAlmostEqual(float(checkpoint["teacher_sample_error_max"]), 0.03, places=6)
        self.assertAlmostEqual(float(checkpoint["teacher_activation_min_disagreement"]), 0.01, places=6)

    def test_train_global_corner_model_saves_sample_reweight_metadata_in_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((96, 160, 3), 200, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            row = {
                "page_id": "sample",
                "project_name": "demo",
                "image_path": str(image_path),
                "manual_quad": [[16, 12], [144, 12], [144, 84], [16, 84]],
                "teacher_quad": [[18, 13], [142, 13], [144, 84], [16, 84]],
                "teacher_r3_quad": [[14, 11], [146, 11], [146, 86], [14, 86]],
                "scene_tags": ["near_color_background", "low_contrast_scene", "black_frame_scene"],
                "scene_profile": {"lab_distance": 18.5, "luma_delta": 0.058, "edge_strength": 0.44},
            }
            split_dir = root / "split"
            split_dir.mkdir()
            payload = json.dumps(row, ensure_ascii=False) + "\n"
            for name in ("train", "test"):
                (split_dir / f"{name}.jsonl").write_text(payload, encoding="utf-8")

            result = train_global_corner_model(
                dataset_root=root,
                output_dir=root / "output",
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                input_size=64,
                output_size=16,
                channels=8,
                split_dir=split_dir,
                disagreement_sample_weight_floor=0.01,
                disagreement_sample_weight_boost=1.6,
                geometry_priority_sample_weight_boost=1.2,
            )

            checkpoint = torch.load(result.model_path, map_location="cpu")

        self.assertAlmostEqual(float(checkpoint["disagreement_sample_weight_floor"]), 0.01, places=6)
        self.assertAlmostEqual(float(checkpoint["disagreement_sample_weight_boost"]), 1.6, places=6)
        self.assertAlmostEqual(float(checkpoint["geometry_priority_sample_weight_boost"]), 1.2, places=6)

    def test_train_global_corner_model_saves_hardness_sample_weight_power_in_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((96, 160, 3), 200, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            row = {
                "page_id": "sample",
                "project_name": "demo",
                "image_path": str(image_path),
                "manual_quad": [[16, 12], [144, 12], [144, 84], [16, 84]],
                "hardness_score": 6.0,
            }
            split_dir = root / "split"
            split_dir.mkdir()
            payload = json.dumps(row, ensure_ascii=False) + "\n"
            for name in ("train", "test"):
                (split_dir / f"{name}.jsonl").write_text(payload, encoding="utf-8")

            result = train_global_corner_model(
                dataset_root=root,
                output_dir=root / "output",
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                input_size=64,
                output_size=16,
                channels=8,
                split_dir=split_dir,
                hardness_sample_weight_power=0.4,
            )

            checkpoint = torch.load(result.model_path, map_location="cpu")

        self.assertAlmostEqual(float(checkpoint["hardness_sample_weight_power"]), 0.4, places=6)

    def test_train_global_corner_model_saves_quad_mask_weight_in_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((96, 160, 3), 200, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            row = {
                "page_id": "sample",
                "project_name": "demo",
                "image_path": str(image_path),
                "manual_quad": [[16, 12], [144, 12], [144, 84], [16, 84]],
            }
            split_dir = root / "split"
            split_dir.mkdir()
            payload = json.dumps(row, ensure_ascii=False) + "\n"
            for name in ("train", "test"):
                (split_dir / f"{name}.jsonl").write_text(payload, encoding="utf-8")

            result = train_global_corner_model(
                dataset_root=root,
                output_dir=root / "output",
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                input_size=64,
                output_size=16,
                channels=8,
                split_dir=split_dir,
                quad_mask_weight=0.3,
            )

            checkpoint = torch.load(result.model_path, map_location="cpu")

        self.assertAlmostEqual(float(checkpoint["quad_mask_weight"]), 0.3, places=6)

    def test_train_global_corner_model_saves_inner_boundary_band_weight_in_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((96, 160, 3), 200, dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            row = {
                "page_id": "sample",
                "project_name": "demo",
                "image_path": str(image_path),
                "manual_quad": [[16, 12], [144, 12], [144, 84], [16, 84]],
            }
            split_dir = root / "split"
            split_dir.mkdir()
            payload = json.dumps(row, ensure_ascii=False) + "\n"
            for name in ("train", "test"):
                (split_dir / f"{name}.jsonl").write_text(payload, encoding="utf-8")

            result = train_global_corner_model(
                dataset_root=root,
                output_dir=root / "output",
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                input_size=64,
                output_size=16,
                channels=8,
                split_dir=split_dir,
                inner_boundary_band_weight=0.2,
            )

            checkpoint = torch.load(result.model_path, map_location="cpu")

        self.assertAlmostEqual(float(checkpoint["inner_boundary_band_weight"]), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
