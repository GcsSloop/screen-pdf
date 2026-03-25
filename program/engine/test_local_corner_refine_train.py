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

from local_corner_refine_train import LocalCornerRefineDataset, LocalCornerRefineNet


class LocalCornerRefineTrainTests(unittest.TestCase):
    def test_local_corner_refine_dataset_returns_feature_tensor_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patch_dir = root / "patches" / "train"
            patch_dir.mkdir(parents=True)
            patch = np.zeros((96, 96, 3), dtype=np.uint8)
            cv2.line(patch, (12, 12), (12, 84), (255, 255, 255), 2)
            cv2.line(patch, (12, 12), (84, 12), (255, 255, 255), 2)
            patch_path = patch_dir / "sample.png"
            cv2.imwrite(str(patch_path), patch)
            row = {
                "page_id": "page-1",
                "corner_index": 0,
                "patch_path": "patches/train/sample.png",
                "target_residual_norm": [0.1, -0.05],
            }
            (root / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = LocalCornerRefineDataset(root / "train.jsonl", root, input_size=64, augment=False)
            item = dataset[0]

        self.assertEqual(tuple(item["image"].shape), (10, 64, 64))
        self.assertEqual(tuple(item["target"].shape), (2,))

    def test_local_corner_refine_net_outputs_residual_vector(self) -> None:
        model = LocalCornerRefineNet(channels=16)
        batch = torch.randn(2, 10, 64, 64)

        output = model(batch)

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertLessEqual(float(output.abs().max()), 1.0)


if __name__ == "__main__":
    unittest.main()
