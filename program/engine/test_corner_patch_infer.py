from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corner_patch_infer import refine_quad_with_patch_predictor


class CornerPatchInferTests(unittest.TestCase):
    def test_refine_quad_with_patch_predictor_updates_each_corner_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.jpg"
            cv2.imwrite(str(image_path), np.zeros((300, 400, 3), dtype=np.uint8))
            quad = np.array([[100, 80], [300, 82], [298, 220], [102, 218]], dtype=np.float32)

            def patch_predictor(sample: dict[str, object]) -> np.ndarray:
                corner_index = int(sample["corner_index"])
                mapping = {
                    0: [0.75, 0.7],
                    1: [0.2, 0.75],
                    2: [0.25, 0.25],
                    3: [0.8, 0.3],
                }
                return np.array(mapping[corner_index], dtype=np.float32)

            refined = refine_quad_with_patch_predictor(
                image_path=image_path,
                predicted_quad=quad,
                patch_predictor=patch_predictor,
                patch_scale=0.2,
            )

        expected = [[110.0, 88.0], [288.0, 92.0], [288.0, 210.0], [114.0, 210.0]]
        for actual_point, expected_point in zip(refined, expected):
            self.assertAlmostEqual(actual_point[0], expected_point[0], places=3)
            self.assertAlmostEqual(actual_point[1], expected_point[1], places=3)


if __name__ == "__main__":
    unittest.main()
