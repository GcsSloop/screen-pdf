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

from corner_patch_train import CornerPatchDataset, CornerPatchNet


class CornerPatchTrainTests(unittest.TestCase):
    def test_corner_patch_dataset_returns_tensor_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patch_dir = root / "patches" / "train"
            patch_dir.mkdir(parents=True)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            path = patch_dir / "sample.png"
            cv2.imwrite(str(path), image)
            row = {
                "patch_path": "patches/train/sample.png",
                "corner_index": 0,
                "target_norm": [0.4, 0.6],
            }
            (root / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = CornerPatchDataset(root / "train.jsonl", root, input_size=64, augment=False)
            item = dataset[0]

        self.assertEqual(tuple(item["image"].shape), (7, 64, 64))
        self.assertEqual(tuple(item["target"].shape), (2,))

    def test_corner_patch_net_outputs_xy(self) -> None:
        model = CornerPatchNet(channels=16)
        batch = torch.randn(2, 7, 64, 64)

        output = model(batch)

        self.assertEqual(tuple(output.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
