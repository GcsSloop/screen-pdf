from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_moe_coord import LocalCornerMoECoordNet, decode_moe_coord_output


class LocalCornerMoECoordTests(unittest.TestCase):
    def test_local_corner_moe_coord_net_outputs_expected_tensors(self) -> None:
        model = LocalCornerMoECoordNet(channels=16, experts=3, metadata_dim=14, input_channels=13)
        batch = torch.randn(2, 13, 64, 64)
        metadata = torch.randn(2, 14)

        heatmaps, offsets, coords, edge_dirs, edge_maps, visibility, gates = model(batch, metadata)

        self.assertEqual(tuple(heatmaps.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(offsets.shape), (2, 2, 16, 16))
        self.assertEqual(tuple(coords.shape), (2, 2))
        self.assertEqual(tuple(edge_dirs.shape), (2, 5))
        self.assertEqual(tuple(edge_maps.shape), (2, 2, 16, 16))
        self.assertEqual(tuple(visibility.shape), (2, 2))
        self.assertEqual(tuple(gates.shape), (2, 3))

    def test_decode_moe_coord_output_blends_heatmap_and_coord_head(self) -> None:
        heatmaps = torch.full((1, 1, 5, 5), -8.0, dtype=torch.float32)
        heatmaps[0, 0, 2, 2] = 8.0
        offsets = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
        coord_head = torch.tensor([[0.8, 0.2]], dtype=torch.float32)

        coords = decode_moe_coord_output(heatmaps, offsets, coord_head, coord_mix=0.25)

        self.assertAlmostEqual(float(coords[0, 0]), 0.575, places=2)
        self.assertAlmostEqual(float(coords[0, 1]), 0.425, places=2)


if __name__ == "__main__":
    unittest.main()
