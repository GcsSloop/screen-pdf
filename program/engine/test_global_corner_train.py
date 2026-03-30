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

from global_corner_train import (
    GlobalCornerDataset,
    build_effective_split_dir,
    compute_edge_supervision_loss,
    compute_corner_weighted_coord_loss,
    compute_global_geometry_loss,
    denormalize_corners,
    export_global_corner_split,
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


if __name__ == "__main__":
    unittest.main()
