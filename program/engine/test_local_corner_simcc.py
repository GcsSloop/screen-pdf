from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_simcc import (
    LocalCornerSimCCNet,
    _soft_ce_loss,
    build_simcc_target,
    decode_simcc_logits,
)


class LocalCornerSimCCTests(unittest.TestCase):
    def test_build_simcc_target_creates_peaks_near_coordinate_bins(self) -> None:
        target_x, target_y = build_simcc_target(np.array([0.25, 0.75], dtype=np.float32), bins=20, sigma=1.2)

        self.assertEqual(target_x.shape, (20,))
        self.assertEqual(target_y.shape, (20,))
        self.assertEqual(int(np.argmax(target_x)), 5)
        self.assertEqual(int(np.argmax(target_y)), 14)

    def test_decode_simcc_logits_recovers_expected_coordinate(self) -> None:
        x_logits = torch.full((1, 8), -8.0, dtype=torch.float32)
        y_logits = torch.full((1, 8), -8.0, dtype=torch.float32)
        x_logits[0, 2] = 8.0
        y_logits[0, 5] = 8.0

        coords = decode_simcc_logits(x_logits, y_logits)

        self.assertAlmostEqual(float(coords[0, 0]), 2 / 7, places=2)
        self.assertAlmostEqual(float(coords[0, 1]), 5 / 7, places=2)

    def test_local_corner_simcc_net_outputs_x_y_logits(self) -> None:
        model = LocalCornerSimCCNet(channels=16, coord_bins=64, metadata_dim=14)
        batch = torch.randn(2, 10, 64, 64)
        metadata = torch.randn(2, 14)

        x_logits, y_logits = model(batch, metadata)

        self.assertEqual(tuple(x_logits.shape), (2, 64))
        self.assertEqual(tuple(y_logits.shape), (2, 64))

    def test_soft_ce_loss_is_finite_for_sparse_soft_targets(self) -> None:
        logits = torch.zeros((1, 8), dtype=torch.float32)
        target = torch.tensor([[0.0, 0.0, 0.2, 0.6, 0.2, 0.0, 0.0, 0.0]], dtype=torch.float32)

        loss = _soft_ce_loss(logits, target)

        self.assertTrue(torch.isfinite(loss).all())


if __name__ == "__main__":
    unittest.main()
