from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detect_frame


class DetectFrameTests(unittest.TestCase):
    def test_local_model_path_prefers_coord_model_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            models = root / "models"
            models.mkdir()
            legacy = models / "local_corner_moe_model.pt"
            coord = models / "local_corner_moe_coord_model.pt"
            legacy.write_bytes(b"legacy")
            coord.write_bytes(b"coord")

            with mock.patch("detect_frame._engine_root", return_value=root):
                model_path = detect_frame.local_model_path()

        self.assertEqual(model_path, coord)

    def test_build_detect_payload_prefers_model_result_and_keeps_opencv_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.jpg"
            cv2.imwrite(str(image_path), np.zeros((100, 160, 3), dtype=np.uint8))
            image = cv2.imread(str(image_path))

            opencv_result = {
                "best": {
                    "method": "contour_quad",
                    "score": 0.22,
                    "confidence": 0.03,
                    "quad": [[10.0, 10.0], [150.0, 10.0], [150.0, 90.0], [10.0, 90.0]],
                },
                "candidates": [
                    {
                        "method": "contour_quad",
                        "score": 0.22,
                        "metrics": {"coverage_score": 0.6},
                        "quad": [[10.0, 10.0], [150.0, 10.0], [150.0, 90.0], [10.0, 90.0]],
                    }
                ],
            }
            model_result = {
                "method": "model_two_stage",
                "score": 0.96,
                "confidence": 0.08,
                "quad": [[12.0, 8.0], [148.0, 11.0], [146.0, 92.0], [14.0, 89.0]],
                "metrics": {"model": 1.0},
            }

            with (
                mock.patch("detect_frame.detect_best_candidate", return_value=opencv_result),
                mock.patch("detect_frame.run_model_detection", return_value=model_result),
            ):
                payload = detect_frame.build_detect_payload(str(image_path), image)

        self.assertEqual(payload["best"]["method"], "model_two_stage")
        self.assertEqual(payload["best"]["quad"], model_result["quad"])
        self.assertEqual(payload["candidates"][0]["method"], "model_two_stage")
        self.assertEqual(payload["candidates"][1]["method"], "contour_quad")

    def test_build_detect_payload_falls_back_to_opencv_when_model_detection_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "page.jpg"
            cv2.imwrite(str(image_path), np.zeros((100, 160, 3), dtype=np.uint8))
            image = cv2.imread(str(image_path))
            opencv_result = {
                "best": {
                    "method": "contour_quad",
                    "score": 0.22,
                    "confidence": 0.03,
                    "quad": [[10.0, 10.0], [150.0, 10.0], [150.0, 90.0], [10.0, 90.0]],
                },
                "candidates": [
                    {
                        "method": "contour_quad",
                        "score": 0.22,
                        "metrics": {"coverage_score": 0.6},
                        "quad": [[10.0, 10.0], [150.0, 10.0], [150.0, 90.0], [10.0, 90.0]],
                    }
                ],
            }

            with (
                mock.patch("detect_frame.detect_best_candidate", return_value=opencv_result),
                mock.patch("detect_frame.run_model_detection", return_value=None),
            ):
                payload = detect_frame.build_detect_payload(str(image_path), image)

        self.assertEqual(payload["best"]["method"], "contour_quad")
        self.assertEqual(len(payload["candidates"]), 1)
        self.assertEqual(payload["candidates"][0]["method"], "contour_quad")


if __name__ == "__main__":
    unittest.main()
