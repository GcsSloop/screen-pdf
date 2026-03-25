from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_heatmap import (
    LocalCornerHeatmapDataset,
    LocalCornerHeatmapNet,
    build_corner_direction_target,
    build_corner_edge_maps,
    build_corner_visibility_target,
    cleanup_patch_for_special_cases,
    compute_local_corner_sample_weight,
    build_local_corner_heatmaps,
    decode_local_corner_heatmaps,
)


class LocalCornerHeatmapTests(unittest.TestCase):
    def test_build_local_corner_heatmaps_creates_peak_at_target(self) -> None:
        heatmaps = build_local_corner_heatmaps([[0.25, 0.75]], output_size=16, sigma=1.5)

        self.assertEqual(heatmaps.shape, (1, 16, 16))
        peak_y, peak_x = np.unravel_index(int(np.argmax(heatmaps[0])), heatmaps[0].shape)
        self.assertEqual((peak_x, peak_y), (4, 11))

    def test_local_corner_heatmap_dataset_returns_tensor_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patch_dir = root / "patches" / "train"
            patch_dir.mkdir(parents=True)
            patch = np.zeros((96, 96, 3), dtype=np.uint8)
            cv2.line(patch, (2, 2), (2, 94), (255, 255, 255), 4)
            cv2.line(patch, (2, 2), (94, 2), (255, 255, 255), 4)
            patch_path = patch_dir / "sample.png"
            cv2.imwrite(str(patch_path), patch)
            row = {
                "page_id": "page-1",
                "corner_index": 0,
                "patch_path": "patches/train/sample.png",
                "target_point_norm": [0.6, 0.4],
                "patch": {"x": 0, "y": 0, "size": 96},
                "manual_quad": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
            }
            (root / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = LocalCornerHeatmapDataset(root / "train.jsonl", root, input_size=64, output_size=16, augment=False)
            item = dataset[0]

        self.assertEqual(tuple(item["image"].shape), (10, 64, 64))
        self.assertEqual(tuple(item["heatmaps"].shape), (1, 16, 16))
        self.assertEqual(tuple(item["target"].shape), (2,))
        self.assertIn("metadata", item)
        self.assertIn("edge_target", item)
        self.assertIn("edge_maps", item)
        self.assertIn("visibility_target", item)
        self.assertIn("sample_weight", item)
        self.assertGreaterEqual(int(item["metadata"].shape[0]), 8)
        self.assertEqual(tuple(item["edge_target"].shape), (5,))
        self.assertEqual(tuple(item["edge_maps"].shape), (2, 16, 16))
        self.assertEqual(tuple(item["visibility_target"].shape), (2,))
        self.assertGreater(float(item["visibility_target"][0]), 0.5)
        self.assertGreater(float(item["visibility_target"][1]), 0.5)
        self.assertGreater(float(item["sample_weight"]), 0.0)

    def test_local_corner_heatmap_dataset_can_emit_structure_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patch_dir = root / "patches" / "train"
            patch_dir.mkdir(parents=True)
            patch = np.zeros((96, 96, 3), dtype=np.uint8)
            cv2.line(patch, (10, 10), (10, 80), (255, 255, 255), 2)
            cv2.line(patch, (10, 10), (80, 10), (255, 255, 255), 2)
            patch_path = patch_dir / "sample.png"
            cv2.imwrite(str(patch_path), patch)
            row = {
                "page_id": "page-1",
                "corner_index": 0,
                "patch_path": "patches/train/sample.png",
                "target_point_norm": [0.6, 0.4],
                "manual_quad": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
            }
            (root / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = LocalCornerHeatmapDataset(
                root / "train.jsonl",
                root,
                input_size=64,
                output_size=16,
                augment=False,
                input_channels=13,
            )
            item = dataset[0]

        self.assertEqual(tuple(item["image"].shape), (13, 64, 64))

    def test_local_corner_heatmap_dataset_can_disable_flip_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patch_dir = root / "patches" / "train"
            patch_dir.mkdir(parents=True)
            patch = np.zeros((96, 96, 3), dtype=np.uint8)
            patch_path = patch_dir / "sample.png"
            cv2.imwrite(str(patch_path), patch)
            row = {
                "page_id": "page-1",
                "corner_index": 0,
                "patch_path": "patches/train/sample.png",
                "target_point_norm": [0.6, 0.4],
                "manual_quad": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
            }
            (root / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = LocalCornerHeatmapDataset(
                root / "train.jsonl",
                root,
                input_size=64,
                output_size=16,
                augment=True,
                allow_flips=False,
            )
            with mock.patch("local_corner_heatmap.random.random", return_value=0.0):
                item = dataset[0]

        self.assertAlmostEqual(float(item["target"][0]), 0.6, places=4)
        self.assertAlmostEqual(float(item["target"][1]), 0.4, places=4)

    def test_local_corner_heatmap_dataset_supports_partial_flip_probability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patch_dir = root / "patches" / "train"
            patch_dir.mkdir(parents=True)
            patch = np.zeros((96, 96, 3), dtype=np.uint8)
            patch_path = patch_dir / "sample.png"
            cv2.imwrite(str(patch_path), patch)
            row = {
                "page_id": "page-1",
                "corner_index": 0,
                "patch_path": "patches/train/sample.png",
                "target_point_norm": [0.6, 0.4],
                "manual_quad": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
            }
            (root / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = LocalCornerHeatmapDataset(
                root / "train.jsonl",
                root,
                input_size=64,
                output_size=16,
                augment=True,
                flip_prob=0.25,
            )
            with mock.patch("local_corner_heatmap.random.random", side_effect=[0.2, 0.3]):
                item = dataset[0]

        self.assertAlmostEqual(float(item["target"][0]), 0.4, places=4)
        self.assertAlmostEqual(float(item["target"][1]), 0.4, places=4)

    def test_local_corner_heatmap_dataset_applies_adaptive_sample_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patch_dir = root / "patches" / "train"
            patch_dir.mkdir(parents=True)
            patch = np.zeros((96, 96, 3), dtype=np.uint8)
            patch_path = patch_dir / "sample.png"
            cv2.imwrite(str(patch_path), patch)
            row = {
                "page_id": "page-1",
                "corner_index": 2,
                "patch_path": "patches/train/sample.png",
                "target_point_norm": [0.6, 0.4],
                "manual_quad": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
                "adaptive_weight": 1.5,
            }
            (root / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = LocalCornerHeatmapDataset(root / "train.jsonl", root, input_size=64, output_size=16, augment=False)
            item = dataset[0]

        self.assertGreater(float(item["sample_weight"]), 1.5)

    def test_compute_local_corner_sample_weight_boosts_bl_large_residual_samples(self) -> None:
        base_row = {
            "target_point_norm": [0.65, 0.45],
            "patch": {"x": 0, "y": 0, "size": 100},
            "predicted_point": [40.0, 40.0],
            "predicted_quad": [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
            "target_residual_norm": [0.18, 0.20],
            "adaptive_weight": 1.0,
        }

        br_weight = compute_local_corner_sample_weight({**base_row, "corner_index": 2})
        bl_weight = compute_local_corner_sample_weight({**base_row, "corner_index": 3})

        self.assertGreater(bl_weight, br_weight)

    def test_build_corner_direction_target_uses_manual_quad_geometry(self) -> None:
        row = {
            "corner_index": 0,
            "manual_quad": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]],
        }

        target = build_corner_direction_target(row)

        self.assertAlmostEqual(float(target[0]), 0.0, places=3)
        self.assertAlmostEqual(float(target[1]), 1.0, places=3)
        self.assertAlmostEqual(float(target[2]), 1.0, places=3)
        self.assertAlmostEqual(float(target[3]), 0.0, places=3)
        self.assertAlmostEqual(float(target[4]), 0.5, places=3)

    def test_build_corner_edge_maps_draws_prev_and_next_edges(self) -> None:
        row = {
            "corner_index": 0,
            "patch": {"x": 0, "y": 0, "size": 64},
            "manual_quad": [[0.0, 0.0], [48.0, 0.0], [48.0, 48.0], [0.0, 48.0]],
        }

        edge_maps = build_corner_edge_maps(row, output_size=16)

        self.assertEqual(edge_maps.shape, (2, 16, 16))
        self.assertGreater(float(edge_maps[0].sum()), 0.0)
        self.assertGreater(float(edge_maps[1].sum()), 0.0)
        self.assertGreater(float(edge_maps[:, 0, 0].max()), 0.0)

    def test_build_corner_visibility_target_detects_missing_branch(self) -> None:
        patch = np.zeros((96, 96, 3), dtype=np.uint8)
        cv2.line(patch, (4, 4), (4, 92), (255, 255, 255), 4)
        row = {
            "corner_index": 0,
            "patch": {"x": 0, "y": 0, "size": 96},
            "manual_quad": [[0.0, 0.0], [92.0, 0.0], [92.0, 92.0], [0.0, 92.0]],
        }

        visibility = build_corner_visibility_target(row, patch)

        self.assertEqual(tuple(visibility.shape), (2,))
        self.assertGreater(float(visibility[0]), 0.5)
        self.assertLess(float(visibility[1]), 0.25)

    def test_local_corner_heatmap_net_outputs_single_heatmap_channel(self) -> None:
        model = LocalCornerHeatmapNet(channels=16)
        batch = torch.randn(2, 10, 64, 64)

        output = model(batch)

        self.assertEqual(tuple(output.shape), (2, 1, 16, 16))

    def test_decode_local_corner_heatmaps_recovers_peak_coordinate(self) -> None:
        heatmaps = torch.full((1, 1, 5, 5), -8.0, dtype=torch.float32)
        heatmaps[0, 0, 3, 1] = 8.0

        coords = decode_local_corner_heatmaps(heatmaps)

        self.assertAlmostEqual(float(coords[0, 0, 0]), 0.25, places=2)
        self.assertAlmostEqual(float(coords[0, 0, 1]), 0.75, places=2)

    def test_cleanup_patch_for_special_cases_removes_top_right_overlay_region(self) -> None:
        patch = np.full((100, 100, 3), 255, dtype=np.uint8)
        patch[0:12, 78:96] = 0
        row = {
            "project_name": "全球化重构下道路照明的出海之路",
            "corner_index": 1,
        }

        cleaned = cleanup_patch_for_special_cases(patch, row, augment=False)

        self.assertGreater(float(cleaned[5, 85].mean()), 10.0)


if __name__ == "__main__":
    unittest.main()
