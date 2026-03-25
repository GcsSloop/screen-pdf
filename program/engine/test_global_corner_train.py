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
    compute_edge_supervision_loss,
    compute_corner_weighted_coord_loss,
    compute_global_geometry_loss,
    denormalize_corners,
    export_global_corner_split,
    reframe_image_and_corners,
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


if __name__ == "__main__":
    unittest.main()
