from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corner_infer import _decode_from_checkpoint, denormalize_corners_to_image


class CornerInferTests(unittest.TestCase):
    def test_denormalize_corners_to_image_restores_absolute_points(self) -> None:
        roi = {"x": 100, "y": 50, "width": 400, "height": 300}
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

        restored = denormalize_corners_to_image(corners, roi)

        self.assertEqual(restored.tolist(), [[100.0, 50.0], [500.0, 50.0], [500.0, 350.0], [100.0, 350.0]])

    def test_eval_result_includes_decode_mode_and_device(self) -> None:
        sample = {
            "pages": 1,
            "device": "mps",
            "decode_mode": "soft_argmax",
        }
        self.assertEqual(sample["device"], "mps")
        self.assertEqual(sample["decode_mode"], "soft_argmax")

    def test_decode_from_checkpoint_supports_heatmap_offset_head(self) -> None:
        heatmaps = torch.full((1, 1, 5, 5), -8.0, dtype=torch.float32)
        heatmaps[0, 0, 2, 2] = 8.0
        offsets = torch.zeros((1, 1, 2, 5, 5), dtype=torch.float32)
        offsets[0, 0, 0, 2, 2] = 0.2
        offsets[0, 0, 1, 2, 2] = -0.1

        coords = _decode_from_checkpoint(
            (heatmaps, offsets),
            {
                "head_mode": "heatmap_offset",
                "decode_mode": "soft_argmax_offset",
            },
        )

        self.assertAlmostEqual(float(coords[0, 0, 0]), 0.55, places=2)
        self.assertAlmostEqual(float(coords[0, 0, 1]), 0.475, places=2)


if __name__ == "__main__":
    unittest.main()
