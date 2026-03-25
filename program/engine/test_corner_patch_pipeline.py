from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corner_patch_pipeline import build_corner_patch_sample, denormalize_point


class CornerPatchPipelineTests(unittest.TestCase):
    def test_denormalize_point_restores_patch_coordinates(self) -> None:
        patch = {"x": 100, "y": 50, "size": 80}

        restored = denormalize_point([0.25, 0.75], patch)

        self.assertEqual(restored, [120.0, 110.0])

    def test_build_corner_patch_sample_crops_square_patch_and_normalizes_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((300, 400, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)

            predicted_quad = np.array([[100, 80], [300, 82], [298, 220], [102, 218]], dtype=np.float32)
            manual_quad = np.array([[110, 88], [292, 90], [290, 212], [108, 210]], dtype=np.float32)

            sample = build_corner_patch_sample(
                image_path=image_path,
                page_id="page-1",
                corner_index=0,
                predicted_quad=predicted_quad,
                manual_quad=manual_quad,
                patch_scale=0.2,
            )

        self.assertEqual(sample["corner_index"], 0)
        self.assertEqual(sample["page_id"], "page-1")
        self.assertEqual(tuple(sample["patch_image"].shape[:2]), (40, 40))
        self.assertEqual(sample["patch"], {"x": 80, "y": 60, "size": 40})
        self.assertEqual(sample["target_norm"], [0.75, 0.7])


if __name__ == "__main__":
    unittest.main()
