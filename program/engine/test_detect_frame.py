from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys
import types

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detect_frame


class DetectFrameTests(unittest.TestCase):
    def tearDown(self) -> None:
        detect_frame._RUNTIME_RELEASE_MODEL_ID = None

    def test_runtime_release_model_id_prefers_promoted_model_release_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = Path(temp_dir)
            (model_root / "deep_screen_r1_2026_03_28.json").write_text(
                '{"public_name":"deep_screen_r1_2026_03_28","model_release_id":"model-20260330-153045-ab12cd34","status":"promoted"}',
                encoding="utf-8",
            )
            with mock.patch("detect_frame._model_root", return_value=model_root):
                self.assertEqual(detect_frame.runtime_release_model_id(), "model-20260330-153045-ab12cd34")

    def test_run_teacher_detection_uses_runtime_release_name_for_model_id(self) -> None:
        fake_module = types.SimpleNamespace(
            predict_two_stage=mock.Mock(
                return_value={
                    "final_quad": [[12.0, 8.0], [148.0, 11.0], [146.0, 92.0], [14.0, 89.0]]
                }
            )
        )
        with (
            mock.patch("detect_frame._get_model_runtime", return_value={"global_predictor": object(), "roi_predictor": object(), "local_predictor": object()}),
            mock.patch("detect_frame.runtime_release_model_id", return_value="deep_screen_r1_2026_03_28"),
            mock.patch.dict(sys.modules, {"two_stage_corner_pipeline": fake_module}),
        ):
            result = detect_frame.run_teacher_detection("/tmp/page.jpg", image=np.zeros((100, 160, 3), dtype=np.uint8))

        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "teacher_current")
        self.assertEqual(result["model_id"], "deep_screen_r1_2026_03_28")

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
                "method": "teacher_current",
                "score": 0.96,
                "confidence": 0.08,
                "quad": [[12.0, 8.0], [148.0, 11.0], [146.0, 92.0], [14.0, 89.0]],
                "metrics": {"model": 1.0},
                "source": "runtime_teacher",
                "model_id": "teacher_current",
                "debug_only": False,
            }

            with (
                mock.patch("detect_frame.detect_best_candidate", return_value=opencv_result),
                mock.patch("detect_frame.run_teacher_detection", return_value=model_result),
                mock.patch("detect_frame.run_deep_screen_v1_detection", return_value=None),
            ):
                payload = detect_frame.build_detect_payload(str(image_path), image)

        self.assertEqual(payload["best"]["method"], "teacher_current")
        self.assertEqual(payload["best"]["quad"], model_result["quad"])
        self.assertEqual(payload["best"]["source"], "runtime_teacher")
        self.assertEqual(payload["candidates"][0]["method"], "teacher_current")
        self.assertEqual(payload["candidates"][0]["source"], "runtime_teacher")
        self.assertEqual(payload["candidates"][0]["modelId"], "teacher_current")
        self.assertEqual(payload["candidates"][0]["debugOnly"], False)
        self.assertEqual(payload["candidates"][1]["method"], "contour_quad")
        self.assertEqual(payload["candidates"][1]["source"], "opencv")

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
                mock.patch("detect_frame.run_teacher_detection", return_value=None),
                mock.patch("detect_frame.run_deep_screen_v1_detection", return_value=None),
            ):
                payload = detect_frame.build_detect_payload(str(image_path), image)

        self.assertEqual(payload["best"]["method"], "contour_quad")
        self.assertEqual(len(payload["candidates"]), 1)
        self.assertEqual(payload["candidates"][0]["method"], "contour_quad")
        self.assertEqual(payload["candidates"][0]["source"], "opencv")

    def test_build_detect_payload_adds_debug_student_candidate_when_debug_mode_enabled(self) -> None:
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
                "candidates": [],
            }
            teacher_result = {
                "method": "teacher_current",
                "score": 0.96,
                "confidence": 0.08,
                "quad": [[12.0, 8.0], [148.0, 11.0], [146.0, 92.0], [14.0, 89.0]],
                "metrics": {"model": 1.0},
                "source": "runtime_teacher",
                "model_id": "teacher_current",
                "debug_only": False,
            }
            student_result = {
                "method": "deep_screen_v1_best",
                "score": 0.95,
                "confidence": 0.06,
                "quad": [[11.0, 9.0], [149.0, 10.0], [147.0, 91.0], [13.0, 88.0]],
                "metrics": {"model": 1.0},
                "source": "runtime_student",
                "model_id": "deep_screen_v1_round_022",
                "debug_only": True,
            }

            with (
                mock.patch.dict("os.environ", {"SCREEN_PDF_DEBUG_DUAL_MODEL": "1"}, clear=False),
                mock.patch("detect_frame.detect_best_candidate", return_value=opencv_result),
                mock.patch("detect_frame.run_teacher_detection", return_value=teacher_result),
                mock.patch("detect_frame.run_deep_screen_v1_detection", return_value=student_result),
            ):
                payload = detect_frame.build_detect_payload(str(image_path), image)

        self.assertEqual(payload["best"]["method"], "teacher_current")
        self.assertEqual(len(payload["candidates"]), 2)
        self.assertEqual(payload["candidates"][0]["method"], "teacher_current")
        self.assertEqual(payload["candidates"][1]["method"], "deep_screen_v1_best")
        self.assertEqual(payload["candidates"][1]["source"], "runtime_student")
        self.assertEqual(payload["candidates"][1]["modelId"], "deep_screen_v1_round_022")
        self.assertEqual(payload["candidates"][1]["debugOnly"], True)

    def test_run_deep_screen_v1_detection_can_use_opencv_candidate_pool(self) -> None:
        image = np.zeros((100, 160, 3), dtype=np.uint8)
        model = mock.Mock()
        runtime = {
            "model": model,
            "input_size": 256,
            "torch": torch,
            "model_path": Path("/tmp/deep_screen_v1_round_054_student.pt"),
            "opencv_candidate_selection_enabled": True,
        }
        model.return_value = {
            "final_quad": torch.tensor([[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]], dtype=torch.float32)
        }
        model.select_candidate_pool.return_value = {
            "selected_quad": torch.tensor([[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]], dtype=torch.float32)
        }
        opencv_result = {
            "best": {
                "quad": [[20.0, 20.0], [140.0, 20.0], [140.0, 80.0], [20.0, 80.0]],
            }
        }

        with (
            mock.patch("detect_frame._get_deep_screen_v1_runtime", return_value=runtime),
            mock.patch("detect_frame.detect_best_candidate", return_value=opencv_result),
            mock.patch("detect_frame.cv2.resize", return_value=np.zeros((256, 256, 3), dtype=np.uint8)),
            mock.patch("detect_frame.cv2.cvtColor", side_effect=lambda img, _: img),
        ):
            result = detect_frame.run_deep_screen_v1_detection("/tmp/page.jpg", image=image)

        model.select_candidate_pool.assert_called_once()
        expected = [[32.0, 20.0], [128.0, 20.0], [128.0, 80.0], [32.0, 80.0]]
        for actual_point, expected_point in zip(result["quad"], expected):
            self.assertAlmostEqual(actual_point[0], expected_point[0], places=4)
            self.assertAlmostEqual(actual_point[1], expected_point[1], places=4)

    def test_get_deep_screen_v1_runtime_passes_final_output_mode_from_checkpoint(self) -> None:
        checkpoint = {
            "state_dict": {},
            "base_channels": 32,
            "roi_size": 16,
            "experts": 3,
            "roi_expand_ratio": 0.08,
            "roi_adapter_layers": 0,
            "spatial_refine_layers": 0,
            "residual_quad_head_layers": 0,
            "strict_spatial_refine_layers": 0,
            "candidate_selection_enabled": False,
            "final_output_mode": "coarse",
            "scene_classes": 4,
            "scene_embedding_dim": 8,
            "input_size": 256,
        }
        fake_model = mock.Mock()

        with (
            mock.patch("detect_frame.deep_screen_v1_model_path", return_value=Path("/tmp/deep_screen_v1.pt")),
            mock.patch("torch.load", return_value=checkpoint),
            mock.patch("deep_screen_v1_model.DeepScreenV1Net", return_value=fake_model) as model_cls,
            mock.patch("deep_screen_v1_model.load_compatible_state_dict", return_value=[]),
        ):
            detect_frame._DEEP_SCREEN_V1_RUNTIME = None
            runtime = detect_frame._get_deep_screen_v1_runtime()

        self.assertIsNotNone(runtime)
        model_cls.assert_called_once()
        self.assertEqual(model_cls.call_args.kwargs["final_output_mode"], "coarse")


if __name__ == "__main__":
    unittest.main()
