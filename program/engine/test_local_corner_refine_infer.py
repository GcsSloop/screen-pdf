from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_refine_infer import apply_residuals_to_quad


class LocalCornerRefineInferTests(unittest.TestCase):
    def test_apply_residuals_to_quad_updates_each_corner_by_patch_size(self) -> None:
        quad = np.array([[60, 40], [240, 42], [238, 160], [58, 158]], dtype=np.float32)
        residuals = np.array([[0.1, 0.05], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

        refined = apply_residuals_to_quad(quad, residuals, patch_size=80)

        self.assertEqual(refined[0], [68.0, 44.0])
        self.assertEqual(refined[1], [240.0, 42.0])


if __name__ == "__main__":
    unittest.main()
