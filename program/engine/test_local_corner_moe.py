from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_moe import LocalCornerMoENet, decode_moe_output, remap_legacy_moe_state_dict


class LocalCornerMoETests(unittest.TestCase):
    def test_local_corner_moe_net_outputs_heatmap_offset_and_gate(self) -> None:
        model = LocalCornerMoENet(channels=16, experts=3, metadata_dim=14)
        batch = torch.randn(2, 10, 64, 64)
        metadata = torch.randn(2, 14)

        heatmaps, offsets, gates = model(batch, metadata)

        self.assertEqual(tuple(heatmaps.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(offsets.shape), (2, 2, 16, 16))
        self.assertEqual(tuple(gates.shape), (2, 3))
        self.assertTrue(torch.allclose(gates.sum(dim=-1), torch.ones(2), atol=1e-5))

    def test_decode_moe_output_applies_offset(self) -> None:
        heatmaps = torch.full((1, 1, 5, 5), -8.0, dtype=torch.float32)
        heatmaps[0, 0, 2, 2] = 8.0
        offsets = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
        offsets[0, 0, 2, 2] = 0.2
        offsets[0, 1, 2, 2] = -0.1

        coords = decode_moe_output(heatmaps, offsets)

        self.assertAlmostEqual(float(coords[0, 0]), 0.55, places=2)
        self.assertAlmostEqual(float(coords[0, 1]), 0.475, places=2)

    def test_remap_legacy_moe_state_dict_rewrites_gate_head_keys(self) -> None:
        state_dict = {
            "gate_head.1.weight": torch.randn(32, 64),
            "gate_head.1.bias": torch.randn(32),
            "gate_head.3.weight": torch.randn(3, 32),
            "gate_head.3.bias": torch.randn(3),
        }

        remapped = remap_legacy_moe_state_dict(state_dict)

        self.assertIn("gate_head.0.weight", remapped)
        self.assertIn("gate_head.2.weight", remapped)


if __name__ == "__main__":
    unittest.main()
