from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corner_train import select_torch_device, soft_argmax_2d


class SoftArgmaxTests(unittest.TestCase):
    def test_soft_argmax_recovers_peak_coordinate(self) -> None:
        heatmaps = torch.full((1, 1, 5, 5), -8.0, dtype=torch.float32)
        heatmaps[0, 0, 2, 3] = 8.0

        coords = soft_argmax_2d(heatmaps)

        self.assertAlmostEqual(float(coords[0, 0, 0]), 0.75, places=2)
        self.assertAlmostEqual(float(coords[0, 0, 1]), 0.5, places=2)

    def test_select_torch_device_prefers_mps_when_available(self) -> None:
        with (
            mock.patch("corner_train.torch.cuda.is_available", return_value=False),
            mock.patch("corner_train.torch.backends.mps.is_available", return_value=True),
        ):
            device = select_torch_device()

        self.assertEqual(device.type, "mps")


if __name__ == "__main__":
    unittest.main()
