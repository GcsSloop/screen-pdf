from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from two_stage_corner_pipeline import LocalCornerMoEPredictor, apply_roi_prediction, build_refine_request, predict_two_stage


class TwoStageCornerPipelineTests(unittest.TestCase):
    def test_local_corner_predictor_prefers_coord_model_checkpoint(self) -> None:
        checkpoint = {
            "channels": 24,
            "experts": 4,
            "metadata_dim": 14,
            "input_size": 96,
            "coord_mix": 0.25,
            "state_dict": {},
        }

        class FakeModel:
            def __init__(self, *args, **kwargs):
                self.loaded = None

            def load_state_dict(self, state_dict, strict=False):
                self.loaded = (state_dict, strict)
                return ([], [])

            def to(self, device):
                return self

            def eval(self):
                return self

        with (
            mock.patch("two_stage_corner_pipeline.select_torch_device", return_value=torch.device("cpu")),
            mock.patch("two_stage_corner_pipeline.torch.load", return_value=checkpoint),
            mock.patch("local_corner_moe_coord.LocalCornerMoECoordNet", FakeModel),
        ):
            predictor = LocalCornerMoEPredictor(Path("/tmp/local_corner_moe_coord_model.pt"))

        self.assertEqual(predictor.coord_mix, 0.25)

    def test_build_refine_request_normalizes_coarse_quad_to_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((200, 300, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            coarse_quad = np.array([[60, 40], [240, 42], [238, 160], [58, 158]], dtype=np.float32)

            request = build_refine_request(image_path=image_path, coarse_quad=coarse_quad, page_id="page-1")

        self.assertEqual(request["page_id"], "page-1")
        self.assertEqual(tuple(request["roi_image"].shape[:2]), (144, 212))
        self.assertEqual(request["roi"], {"x": 43, "y": 28, "width": 212, "height": 144})
        self.assertEqual(request["coarse_quad_norm"], [[0.080189, 0.083333], [0.929245, 0.097222], [0.919811, 0.916667], [0.070755, 0.902778]])

    def test_apply_roi_prediction_restores_absolute_points(self) -> None:
        roi = {"x": 40, "y": 20, "width": 200, "height": 100}
        pred_norm = np.array([[0.1, 0.1], [0.9, 0.1], [0.92, 0.88], [0.08, 0.9]], dtype=np.float32)

        restored = apply_roi_prediction(pred_norm, roi, image_shape=(180, 320, 3))

        self.assertEqual(restored.tolist(), [[60.0, 30.0], [220.0, 30.0], [224.0, 108.0], [56.0, 110.0]])

    def test_predict_two_stage_runs_global_then_roi_predictor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)

            coarse_quad = np.array([[50, 30], [270, 35], [260, 150], [55, 145]], dtype=np.float32)

            def fake_global_predictor(path: Path) -> np.ndarray:
                self.assertEqual(path, image_path)
                return coarse_quad

            def fake_roi_predictor(request: dict[str, object]) -> np.ndarray:
                self.assertEqual(request["page_id"], "sample")
                self.assertEqual(request["image_path"], str(image_path))
                return np.array([[0.1, 0.1], [0.9, 0.1], [0.92, 0.9], [0.08, 0.88]], dtype=np.float32)

            result = predict_two_stage(
                image_path=image_path,
                global_predictor=fake_global_predictor,
                roi_predictor=fake_roi_predictor,
            )

        self.assertEqual(result["coarse_quad"], coarse_quad.tolist())
        self.assertEqual(result["final_quad"], [[57.6, 32.4], [262.4, 32.4], [267.52, 147.6], [52.48, 144.72]])

    def test_predict_two_stage_applies_optional_local_predictor_after_roi_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)

            coarse_quad = np.array([[50, 30], [270, 35], [260, 150], [55, 145]], dtype=np.float32)
            roi_quad = np.array([[57.6, 32.4], [262.4, 32.4], [267.52, 147.6], [52.48, 144.72]], dtype=np.float32)
            refined_quad = np.array([[56.0, 31.0], [264.0, 31.5], [268.0, 146.0], [51.0, 143.5]], dtype=np.float32)

            def fake_global_predictor(path: Path) -> np.ndarray:
                self.assertEqual(path, image_path)
                return coarse_quad

            def fake_roi_predictor(request: dict[str, object]) -> np.ndarray:
                return np.array([[0.1, 0.1], [0.9, 0.1], [0.92, 0.9], [0.08, 0.88]], dtype=np.float32)

            def fake_local_predictor(path: Path, quad: np.ndarray) -> np.ndarray:
                self.assertEqual(path, image_path)
                np.testing.assert_allclose(quad, roi_quad, atol=1e-4)
                return refined_quad

            result = predict_two_stage(
                image_path=image_path,
                global_predictor=fake_global_predictor,
                roi_predictor=fake_roi_predictor,
                local_predictor=fake_local_predictor,
            )

        self.assertEqual(result["coarse_quad"], coarse_quad.tolist())
        self.assertEqual(result["roi_quad"], [[57.6, 32.4], [262.4, 32.4], [267.52, 147.6], [52.48, 144.72]])
        self.assertEqual(result["final_quad"], [[56.0, 31.0], [264.0, 31.5], [268.0, 146.0], [51.0, 143.5]])


if __name__ == "__main__":
    unittest.main()
