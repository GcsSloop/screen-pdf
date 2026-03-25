from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corner_train import (
    CornerHeatmapNet,
    CornerSampleDataset,
    build_corner_heatmaps,
    decode_heatmaps,
    decode_heatmaps_with_offsets,
    freeze_model_backbone_for_offset_tuning,
    initialize_model_from_checkpoint,
    remap_legacy_head_state_dict,
)


class CornerTrainTests(unittest.TestCase):
    def test_build_corner_heatmaps_creates_one_peak_per_corner(self) -> None:
        corners = [[0.1, 0.2], [0.8, 0.2], [0.78, 0.82], [0.12, 0.84]]

        heatmaps = build_corner_heatmaps(corners, output_size=32, sigma=1.5)

        self.assertEqual(heatmaps.shape, (4, 32, 32))
        for channel in range(4):
            self.assertGreater(float(heatmaps[channel].max()), 0.9)

    def test_corner_sample_dataset_returns_tensor_batch_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            roi_dir = root / "roi" / "train"
            roi_dir.mkdir(parents=True)
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.rectangle(image, (20, 20), (140, 100), (240, 240, 240), -1)
            image_path = roi_dir / "sample.png"
            cv2.imwrite(str(image_path), image)
            sample = {
                "split": "train",
                "page_id": "page-1",
                "roi_path": "roi/train/sample.png",
                "corner_norm": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                "coarse_quad_norm": [[0.12, 0.22], [0.78, 0.2], [0.79, 0.78], [0.11, 0.81]],
            }
            (root / "train.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = CornerSampleDataset(root / "train.jsonl", root, input_size=128, output_size=32, augment=False)
            item = dataset[0]

        self.assertEqual(tuple(item["image"].shape), (4, 128, 128))
        self.assertEqual(tuple(item["heatmaps"].shape), (4, 32, 32))
        self.assertEqual(tuple(item["corners"].shape), (4, 2))

    def test_corner_sample_dataset_build_sample_weights_upweights_harder_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            roi_dir = root / "roi" / "train"
            roi_dir.mkdir(parents=True)
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            image_path = roi_dir / "sample.png"
            cv2.imwrite(str(image_path), image)
            rows = [
                {
                    "split": "train",
                    "page_id": "easy",
                    "roi_path": "roi/train/sample.png",
                    "corner_norm": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "coarse_quad_norm": [[0.11, 0.21], [0.79, 0.2], [0.79, 0.79], [0.11, 0.79]],
                },
                {
                    "split": "train",
                    "page_id": "hard",
                    "roi_path": "roi/train/sample.png",
                    "corner_norm": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                    "coarse_quad_norm": [[0.2, 0.3], [0.7, 0.3], [0.7, 0.7], [0.2, 0.7]],
                },
            ]
            (root / "train.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

            dataset = CornerSampleDataset(root / "train.jsonl", root, input_size=128, output_size=32, augment=False)
            weights = dataset.build_sample_weights(power=1.0)

        self.assertEqual(weights.shape, (2,))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_corner_sample_dataset_can_serve_cached_image_after_source_file_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            roi_dir = root / "roi" / "train"
            roi_dir.mkdir(parents=True)
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            cv2.rectangle(image, (20, 20), (140, 100), (240, 240, 240), -1)
            image_path = roi_dir / "sample.png"
            cv2.imwrite(str(image_path), image)
            sample = {
                "split": "train",
                "page_id": "page-1",
                "roi_path": "roi/train/sample.png",
                "corner_norm": [[0.1, 0.2], [0.8, 0.2], [0.8, 0.8], [0.1, 0.8]],
                "coarse_quad_norm": [[0.12, 0.22], [0.78, 0.2], [0.79, 0.78], [0.11, 0.81]],
            }
            (root / "train.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = CornerSampleDataset(
                root / "train.jsonl",
                root,
                input_size=128,
                output_size=32,
                augment=False,
                cache_images=True,
            )
            image_path.unlink()
            item = dataset[0]

        self.assertEqual(tuple(item["image"].shape), (4, 128, 128))

    def test_corner_heatmap_net_outputs_expected_shape(self) -> None:
        model = CornerHeatmapNet(in_channels=4, channels=24, output_channels=4)
        batch = torch.randn(2, 4, 128, 128)

        output = model(batch)

        self.assertEqual(tuple(output.shape), (2, 4, 32, 32))

    def test_decode_heatmaps_returns_continuous_coordinates(self) -> None:
        heatmaps = torch.full((1, 1, 5, 5), -8.0, dtype=torch.float32)
        heatmaps[0, 0, 2, 3] = 8.0

        coords = decode_heatmaps(heatmaps)

        self.assertAlmostEqual(float(coords[0, 0, 0]), 0.75, places=2)
        self.assertAlmostEqual(float(coords[0, 0, 1]), 0.5, places=2)

    def test_corner_heatmap_net_offset_head_outputs_heatmaps_and_offsets(self) -> None:
        model = CornerHeatmapNet(in_channels=4, channels=24, output_channels=4, head_mode="heatmap_offset")
        batch = torch.randn(2, 4, 128, 128)

        heatmaps, offsets = model(batch)

        self.assertEqual(tuple(heatmaps.shape), (2, 4, 32, 32))
        self.assertEqual(tuple(offsets.shape), (2, 4, 2, 32, 32))

    def test_decode_heatmaps_with_offsets_applies_residual_shift(self) -> None:
        heatmaps = torch.full((1, 1, 5, 5), -8.0, dtype=torch.float32)
        heatmaps[0, 0, 2, 2] = 8.0
        offsets = torch.zeros((1, 1, 2, 5, 5), dtype=torch.float32)
        offsets[0, 0, 0, 2, 2] = 0.2
        offsets[0, 0, 1, 2, 2] = -0.1

        coords = decode_heatmaps_with_offsets(heatmaps, offsets)

        self.assertAlmostEqual(float(coords[0, 0, 0]), 0.55, places=2)
        self.assertAlmostEqual(float(coords[0, 0, 1]), 0.475, places=2)

    def test_remap_legacy_head_state_dict_maps_old_head_keys(self) -> None:
        state_dict = {
            "head.0.weight": torch.randn(2, 2, 3, 3),
            "head.0.bias": torch.randn(2),
            "head.2.weight": torch.randn(4, 2, 1, 1),
            "head.2.bias": torch.randn(4),
        }

        remapped = remap_legacy_head_state_dict(state_dict)

        self.assertIn("heatmap_head.0.weight", remapped)
        self.assertIn("heatmap_head.2.bias", remapped)
        self.assertNotIn("head.0.weight", remapped)

    def test_initialize_model_from_checkpoint_copies_heatmap_weights_into_offset_model(self) -> None:
        source = CornerHeatmapNet(in_channels=4, channels=24, output_channels=4)
        target = CornerHeatmapNet(in_channels=4, channels=24, output_channels=4, head_mode="heatmap_offset")
        source_state = source.state_dict()

        initialize_model_from_checkpoint(target, {"state_dict": source_state})

        self.assertTrue(torch.equal(target.state_dict()["stem.block.0.weight"], source_state["stem.block.0.weight"]))
        self.assertTrue(torch.equal(target.state_dict()["heatmap_head.0.weight"], source_state["heatmap_head.0.weight"]))

    def test_freeze_model_backbone_for_offset_tuning_only_keeps_offset_head_trainable(self) -> None:
        model = CornerHeatmapNet(in_channels=4, channels=24, output_channels=4, head_mode="heatmap_offset")

        freeze_model_backbone_for_offset_tuning(model)

        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("offset_head.") for name in trainable))


if __name__ == "__main__":
    unittest.main()
