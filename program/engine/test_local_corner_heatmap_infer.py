from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_heatmap_infer import apply_patch_points_to_quad


class LocalCornerHeatmapInferTests(unittest.TestCase):
    def test_apply_patch_points_to_quad_restores_absolute_coordinates(self) -> None:
        patch_samples = [
            {"patch": {"x": 20, "y": 10, "size": 80}},
            {"patch": {"x": 100, "y": 10, "size": 80}},
            {"patch": {"x": 100, "y": 90, "size": 80}},
            {"patch": {"x": 20, "y": 90, "size": 80}},
        ]
        point_norms = np.array([[0.5, 0.5], [0.5, 0.4], [0.4, 0.5], [0.5, 0.4]], dtype=np.float32)

        quad = apply_patch_points_to_quad(patch_samples, point_norms)

        self.assertEqual(quad[0], [60.0, 50.0])
        self.assertEqual(quad[1], [140.0, 42.0])


if __name__ == "__main__":
    unittest.main()
