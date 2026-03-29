from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perspective_detect import (
    DEFAULT_SCORING_PROFILE,
    Candidate,
    _collect_base_candidates,
    _build_lsd_v2_profile,
    _clamp_quad_bottom,
    _detect_bright_screen,
    _detect_document_quad,
    _detect_hough_screen,
    _detect_line_fusion_quad,
    _detect_lsd_grabcut_quad,
    _detect_roi_guided_quad,
    _refine_quad_by_edge_alignment,
    _should_enable_lsd_v2,
    _select_problem_scene_candidate,
    combine_score_from_metrics,
    detect_best_candidate_with_profile,
    load_opencv_profile,
    load_scoring_profile,
)
from train_scoring_profile import evaluate_profile_on_samples, load_training_samples
from tune_opencv_profile import objective_score


class PerspectiveDetectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8)
        quad = np.array(
            [[180, 120], [1090, 150], [1110, 580], [170, 560]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(self.image, quad, (242, 242, 242))
        cv2.polylines(self.image, [quad], True, (255, 255, 255), 6)

    def test_detect_bright_screen_finds_quad(self) -> None:
        quad = _detect_bright_screen(self.image)
        self.assertIsNotNone(quad)
        assert quad is not None
        self.assertEqual(quad.shape, (4, 2))

    def test_detect_hough_screen_finds_quad(self) -> None:
        quad = _detect_hough_screen(self.image)
        self.assertIsNotNone(quad)
        assert quad is not None
        self.assertEqual(quad.shape, (4, 2))

    def test_detect_document_quad_finds_white_projection_surface(self) -> None:
        image = np.full((900, 1400, 3), 28, dtype=np.uint8)
        image[:, :, 1] = 36
        quad = np.array([[210, 170], [1200, 130], [1240, 690], [170, 720]], dtype=np.int32)
        cv2.fillConvexPoly(image, quad, (248, 248, 248))
        cv2.polylines(image, [quad], True, (254, 254, 254), 12)
        cv2.line(image, (0, 770), (1400, 820), (85, 72, 60), 120)

        detected = _detect_document_quad(image)

        self.assertIsNotNone(detected)
        assert detected is not None
        self.assertEqual(detected.shape, (4, 2))

    def test_detect_line_fusion_quad_finds_line_defined_screen(self) -> None:
        image = np.zeros((880, 1460, 3), dtype=np.uint8)
        quad = np.array([[260, 200], [1180, 170], [1235, 645], [225, 690]], dtype=np.int32)
        cv2.line(image, tuple(quad[0]), tuple(quad[1]), (255, 255, 255), 10)
        cv2.line(image, tuple(quad[1]), tuple(quad[2]), (255, 255, 255), 10)
        cv2.line(image, tuple(quad[2]), tuple(quad[3]), (255, 255, 255), 10)
        cv2.line(image, tuple(quad[3]), tuple(quad[0]), (255, 255, 255), 10)
        cv2.line(image, (0, 780), (1460, 840), (70, 60, 50), 110)

        detected = _detect_line_fusion_quad(image)

        self.assertIsNotNone(detected)
        assert detected is not None
        self.assertEqual(detected.shape, (4, 2))

    def test_detect_roi_guided_quad_prefers_central_stage_screen(self) -> None:
        image = np.full((960, 1560, 3), 30, dtype=np.uint8)
        main_quad = np.array([[310, 210], [1260, 180], [1310, 720], [280, 760]], dtype=np.int32)
        side_quad = np.array([[40, 140], [250, 150], [265, 410], [30, 400]], dtype=np.int32)
        cv2.fillConvexPoly(image, main_quad, (236, 236, 236))
        cv2.polylines(image, [main_quad], True, (255, 255, 255), 8)
        cv2.fillConvexPoly(image, side_quad, (215, 215, 215))
        cv2.polylines(image, [side_quad], True, (230, 230, 230), 6)
        cv2.line(image, (0, 850), (1560, 900), (88, 75, 62), 120)

        detected = _detect_roi_guided_quad(image)

        self.assertIsNotNone(detected)
        assert detected is not None
        center_x = float(np.mean(detected[:, 0]))
        self.assertGreater(center_x, 650.0)

    def test_detect_lsd_grabcut_quad_finds_main_projection_surface(self) -> None:
        image = np.full((980, 1580, 3), 28, dtype=np.uint8)
        main_quad = np.array([[290, 190], [1290, 165], [1345, 735], [255, 780]], dtype=np.int32)
        cv2.fillConvexPoly(image, main_quad, (240, 240, 240))
        cv2.polylines(image, [main_quad], True, (255, 255, 255), 8)
        cv2.line(image, (0, 860), (1580, 930), (86, 73, 60), 130)
        cv2.rectangle(image, (40, 180), (230, 430), (205, 205, 205), -1)

        detected = _detect_lsd_grabcut_quad(image)

        self.assertIsNotNone(detected)
        assert detected is not None
        center_x = float(np.mean(detected[:, 0]))
        self.assertGreater(center_x, 700.0)

    def test_collect_base_candidates_only_runs_bright_screen_and_quad_detectors(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        lsd_quad = np.array([[80, 60], [560, 60], [560, 420], [80, 420]], dtype=np.float32)
        document_quad = np.array([[90, 55], [550, 65], [545, 415], [88, 410]], dtype=np.float32)
        line_quad = np.array([[100, 58], [548, 72], [540, 408], [95, 400]], dtype=np.float32)
        roi_quad = np.array([[108, 62], [540, 78], [535, 402], [104, 396]], dtype=np.float32)
        contour_quad = np.array([[118, 66], [532, 84], [528, 396], [114, 390]], dtype=np.float32)
        bright_quad = np.array([[128, 70], [524, 90], [520, 390], [124, 384]], dtype=np.float32)

        with (
            mock.patch("perspective_detect._detect_lsd_grabcut_quad", return_value=lsd_quad) as lsd_mock,
            mock.patch("perspective_detect._detect_document_quad", return_value=document_quad) as document_mock,
            mock.patch("perspective_detect._detect_line_fusion_quad", return_value=line_quad) as line_fusion_mock,
            mock.patch("perspective_detect._detect_roi_guided_quad", return_value=roi_quad) as roi_mock,
            mock.patch("perspective_detect._detect_quad_by_contours", return_value=contour_quad) as contour_mock,
            mock.patch("perspective_detect._detect_bright_screen", return_value=bright_quad) as bright_mock,
            mock.patch("perspective_detect._detect_hough_screen", return_value=bright_quad) as hough_mock,
            mock.patch("perspective_detect._detect_colored_screen", return_value=bright_quad) as color_mock,
            mock.patch("perspective_detect._detect_refined_edges", return_value=bright_quad) as refined_mock,
        ):
            candidates = _collect_base_candidates(image, load_opencv_profile())

        methods = [method for method, _ in candidates]
        self.assertGreater(len(methods), 0)
        self.assertTrue(all(method in {"document_quad", "contour_quad"} for method in methods))
        self.assertEqual(lsd_mock.call_count, 0)
        self.assertEqual(document_mock.call_count, 1)
        self.assertEqual(line_fusion_mock.call_count, 0)
        self.assertEqual(roi_mock.call_count, 0)
        self.assertEqual(contour_mock.call_count, 1)
        self.assertEqual(bright_mock.call_count, 0)
        self.assertEqual(hough_mock.call_count, 0)
        self.assertEqual(color_mock.call_count, 0)
        self.assertEqual(refined_mock.call_count, 0)

    def test_detect_lsd_grabcut_quad_limits_bottom_floor_spill(self) -> None:
        image = np.full((980, 1580, 3), 26, dtype=np.uint8)
        main_quad = np.array([[310, 180], [1280, 170], [1330, 700], [280, 720]], dtype=np.int32)
        cv2.fillConvexPoly(image, main_quad, (238, 238, 238))
        cv2.polylines(image, [main_quad], True, (255, 255, 255), 7)
        cv2.rectangle(image, (0, 760), (1580, 980), (120, 104, 86), -1)
        cv2.line(image, (0, 745), (1580, 785), (150, 132, 112), 18)

        detected = _detect_lsd_grabcut_quad(image)

        self.assertIsNotNone(detected)
        assert detected is not None
        bottom_y = float(max(detected[2][1], detected[3][1]))
        self.assertLess(bottom_y, 800.0)

    def test_refine_quad_by_edge_alignment_reduces_point_error(self) -> None:
        image = np.zeros((600, 900, 3), dtype=np.uint8)
        truth = np.array([[150, 110], [760, 95], [780, 470], [140, 500]], dtype=np.int32)
        cv2.fillConvexPoly(image, truth, (235, 235, 235))
        cv2.polylines(image, [truth], True, (255, 255, 255), 8)

        coarse = np.array([[170, 130], [735, 120], [750, 445], [170, 470]], dtype=np.float32)
        refined = _refine_quad_by_edge_alignment(image, coarse)

        truth_f = truth.astype(np.float32)
        coarse_err = float(np.mean(np.linalg.norm(coarse - truth_f, axis=1)))
        refined_err = float(np.mean(np.linalg.norm(refined - truth_f, axis=1)))

        self.assertLess(refined_err, coarse_err)

    def test_clamp_quad_bottom_reduces_floor_spill(self) -> None:
        quad = np.array(
            [[300.0, 180.0], [1280.0, 175.0], [1360.0, 860.0], [260.0, 900.0]],
            dtype=np.float32,
        )
        clamped = _clamp_quad_bottom(quad, 760.0)

        self.assertLessEqual(float(clamped[2][1]), 760.0)
        self.assertLessEqual(float(clamped[3][1]), 760.0)

    def test_load_scoring_profile_falls_back_to_default(self) -> None:
        profile = load_scoring_profile(Path("/tmp/does-not-exist-screen-pdf-profile.json"))
        self.assertEqual(profile["weights"], DEFAULT_SCORING_PROFILE["weights"])
        self.assertEqual(profile["method_bias"], DEFAULT_SCORING_PROFILE["method_bias"])

    def test_load_scoring_profile_uses_file_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "weights": {
                            "aspect_score": 0.2,
                            "edge_score": 0.15,
                        },
                        "method_bias": {"refined_edges": 0.08},
                        "base_method_bias": {"refined_edges": 0.03},
                        "selector_mode": "score_only",
                        "enable_lsd_v2": False,
                    }
                ),
                encoding="utf-8",
            )
            profile = load_scoring_profile(profile_path)
        self.assertEqual(profile["weights"]["aspect_score"], 0.2)
        self.assertEqual(profile["weights"]["edge_score"], 0.15)
        self.assertEqual(profile["method_bias"]["refined_edges"], 0.08)
        self.assertEqual(profile["base_method_bias"]["refined_edges"], 0.03)
        self.assertEqual(profile["selector_mode"], "score_only")
        self.assertFalse(profile["enable_lsd_v2"])
        self.assertIn("parallel_score", profile["weights"])

    def test_load_opencv_profile_falls_back_to_default(self) -> None:
        profile = load_opencv_profile(Path("/tmp/does-not-exist-screen-pdf-opencv-profile.json"))
        self.assertIn("lsd_scale", profile)
        self.assertIn("grabcut_iters", profile)
        self.assertGreater(profile["roi_expand_ratio"], 0.0)

    def test_load_opencv_profile_uses_file_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "opencv-profile.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "lsd_scale": 0.7,
                        "grabcut_iters": 4,
                        "roi_expand_ratio": 0.16,
                    }
                ),
                encoding="utf-8",
            )
            profile = load_opencv_profile(profile_path)
        self.assertEqual(profile["lsd_scale"], 0.7)
        self.assertEqual(profile["grabcut_iters"], 4)
        self.assertEqual(profile["roi_expand_ratio"], 0.16)

    def test_build_lsd_v2_profile_uses_tighter_mask_variant(self) -> None:
        profile = _build_lsd_v2_profile(load_opencv_profile())

        self.assertEqual(profile["grabcut_iters"], 4)
        self.assertEqual(profile["roi_expand_ratio"], 0.1)
        self.assertEqual(profile["mask_close_kernel"], 7)
        self.assertEqual(profile["mask_open_kernel"], 3)

    def test_should_enable_lsd_v2_for_low_confidence_floor_spill_scene(self) -> None:
        candidate = Candidate(
            method="bright_screen",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "floor_penalty": 0.74,
                "spill_penalty": 0.16,
            },
            score=0.81,
        )

        self.assertTrue(_should_enable_lsd_v2([candidate], confidence=0.018))

    def test_should_not_enable_lsd_v2_for_confident_clean_scene(self) -> None:
        candidate = Candidate(
            method="document_quad",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "floor_penalty": 0.12,
                "spill_penalty": 0.03,
            },
            score=0.89,
        )

        self.assertFalse(_should_enable_lsd_v2([candidate], confidence=0.14))

    def test_select_problem_scene_candidate_prefers_bright_screen_for_distorted_refined_edges(self) -> None:
        best = Candidate(
            method="refined_edges",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.52,
                "floor_penalty": 0.84,
                "spill_penalty": 0.18,
                "coverage_score": 0.91,
            },
            score=0.76,
        )
        bright = Candidate(
            method="bright_screen",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.69,
                "floor_penalty": 0.51,
                "spill_penalty": 0.07,
                "coverage_score": 0.95,
            },
            score=0.75,
        )
        other = Candidate(
            method="color_box_cleanup",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.72,
                "floor_penalty": 0.73,
                "spill_penalty": 0.05,
                "coverage_score": 0.79,
            },
            score=0.74,
        )

        selected = _select_problem_scene_candidate([best, bright, other], confidence=0.01)

        self.assertEqual(selected.method, "bright_screen")

    def test_select_problem_scene_candidate_prefers_roi_for_overfilled_bright_screen(self) -> None:
        best = Candidate(
            method="bright_screen",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.69,
                "floor_penalty": 0.63,
                "spill_penalty": 0.02,
                "coverage_score": 0.96,
            },
            score=0.80,
        )
        roi = Candidate(
            method="roi_guided_quad",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.96,
                "floor_penalty": 0.56,
                "spill_penalty": 0.14,
                "coverage_score": 0.33,
            },
            score=0.76,
        )

        selected = _select_problem_scene_candidate([best, roi], confidence=0.03)

        self.assertEqual(selected.method, "roi_guided_quad")

    def test_select_problem_scene_candidate_keeps_best_when_rescue_not_qualified(self) -> None:
        best = Candidate(
            method="bright_screen",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.68,
                "floor_penalty": 0.57,
                "spill_penalty": 0.06,
                "coverage_score": 0.95,
            },
            score=0.75,
        )
        bad_alt = Candidate(
            method="roi_guided_quad",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.95,
                "floor_penalty": 0.71,
                "spill_penalty": 0.15,
                "coverage_score": 0.24,
            },
            score=0.73,
        )

        selected = _select_problem_scene_candidate([best, bad_alt], confidence=0.01)

        self.assertEqual(selected.method, "bright_screen")

    def test_select_problem_scene_candidate_prefers_lsd_v2_for_overfilled_bright_screen(self) -> None:
        best = Candidate(
            method="bright_screen",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.73,
                "floor_penalty": 0.88,
                "spill_penalty": 0.0,
                "coverage_score": 0.0,
            },
            score=0.56,
        )
        lsd_v2 = Candidate(
            method="lsd_grabcut_quad_v2",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.79,
                "floor_penalty": 0.09,
                "spill_penalty": 0.08,
                "coverage_score": 0.15,
            },
            score=0.56,
        )

        selected = _select_problem_scene_candidate([best, lsd_v2], confidence=0.0)

        self.assertEqual(selected.method, "lsd_grabcut_quad_v2")

    def test_select_problem_scene_candidate_prefers_lsd_for_distorted_refined_edges(self) -> None:
        best = Candidate(
            method="refined_edges",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.56,
                "floor_penalty": 0.56,
                "spill_penalty": 0.0,
                "coverage_score": 0.6,
            },
            score=0.82,
        )
        lsd = Candidate(
            method="lsd_grabcut_quad",
            quad=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            metrics={
                "aspect_score": 0.88,
                "floor_penalty": 0.42,
                "spill_penalty": 0.0,
                "coverage_score": 0.0,
            },
            score=0.69,
        )

        selected = _select_problem_scene_candidate([best, lsd], confidence=0.08)

        self.assertEqual(selected.method, "lsd_grabcut_quad")

    def test_detect_best_candidate_does_not_add_removed_candidates_for_problem_scene(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        base_quad = np.array([[80, 60], [560, 60], [560, 420], [80, 420]], dtype=np.float32)
        alt_quad = np.array([[78, 55], [562, 62], [561, 418], [76, 419]], dtype=np.float32)
        v2_quad = np.array([[100, 70], [540, 80], [535, 380], [105, 375]], dtype=np.float32)

        base_candidates = [("document_quad", base_quad), ("contour_quad", alt_quad)]

        def fake_score_candidate(
            _image: np.ndarray,
            quad: np.ndarray,
            method: str,
            _profile: dict[str, dict[str, float]] | None = None,
            base_method: str | None = None,
        ) -> tuple[float, dict[str, float]]:
            if method == "document_quad":
                return 0.81, {"floor_penalty": 0.72, "spill_penalty": 0.15}
            if method == "contour_quad":
                return 0.78, {"floor_penalty": 0.7, "spill_penalty": 0.14}
            return 0.2, {"floor_penalty": 0.1, "spill_penalty": 0.02}

        with (
            mock.patch("perspective_detect._collect_base_candidates", return_value=base_candidates),
            mock.patch("perspective_detect._score_candidate", side_effect=fake_score_candidate),
            mock.patch("perspective_detect._detect_lsd_grabcut_quad", return_value=v2_quad) as detect_mock,
            mock.patch(
                "perspective_detect.load_scoring_profile",
                return_value={
                    "weights": dict(DEFAULT_SCORING_PROFILE["weights"]),
                    "method_bias": {},
                    "base_method_bias": {},
                    "selector_mode": "legacy_rescue",
                    "enable_lsd_v2": True,
                },
            ),
        ):
            result = detect_best_candidate_with_profile(image, load_opencv_profile())

        self.assertIsNotNone(result)
        assert result is not None
        methods = [item["method"] for item in result["candidates"]]
        self.assertNotIn("lsd_grabcut_quad_v2", methods)
        self.assertEqual(detect_mock.call_count, 0)

    def test_detect_best_candidate_skips_problem_scene_rescue_in_score_only_mode(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        base_quad = np.array([[80, 60], [560, 60], [560, 420], [80, 420]], dtype=np.float32)

        def fake_score_candidate(
            _image: np.ndarray,
            _quad: np.ndarray,
            method: str,
            _profile: dict[str, dict[str, float]] | None = None,
            base_method: str | None = None,
        ) -> tuple[float, dict[str, float]]:
            if method == "bright_screen":
                return 0.83, {"edge_score": 0.6}
            if method == "document_quad":
                return 0.81, {"edge_score": 0.5}
            return 0.2, {"edge_score": 0.2}

        with (
            mock.patch(
                "perspective_detect.load_scoring_profile",
                return_value={
                    "weights": dict(DEFAULT_SCORING_PROFILE["weights"]),
                    "method_bias": {},
                    "base_method_bias": {},
                    "selector_mode": "score_only",
                    "enable_lsd_v2": False,
                },
            ),
            mock.patch(
                "perspective_detect._collect_base_candidates",
                return_value=[("document_quad", base_quad), ("contour_quad", base_quad)],
            ),
            mock.patch("perspective_detect._score_candidate", side_effect=fake_score_candidate),
            mock.patch("perspective_detect._select_problem_scene_candidate") as select_mock,
        ):
            result = detect_best_candidate_with_profile(image, load_opencv_profile())

        self.assertIsNotNone(result)
        assert result is not None
        methods = [item["method"] for item in result["candidates"]]
        self.assertNotIn("refined_edges", methods)
        self.assertEqual(result["best"]["method"], "document_quad")
        select_mock.assert_not_called()

    def test_detect_best_candidate_adds_edge_refined_variant_candidate(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        base_quad = np.array([[80, 60], [560, 60], [560, 420], [80, 420]], dtype=np.float32)
        refined_quad = np.array([[76, 56], [564, 59], [562, 424], [78, 421]], dtype=np.float32)

        def fake_score_candidate(
            _image: np.ndarray,
            _quad: np.ndarray,
            method: str,
            _profile: dict[str, dict[str, float]] | None = None,
            *,
            base_method: str | None = None,
        ) -> tuple[float, dict[str, float]]:
            if method == "contour_quad_edge":
                return 0.92, {"edge_score": 0.9}
            if method == "contour_quad":
                return 0.84, {"edge_score": 0.6}
            return 0.2, {"edge_score": 0.2}

        with (
            mock.patch("perspective_detect._collect_base_candidates", return_value=[("contour_quad", base_quad)]),
            mock.patch("perspective_detect._refine_quad_by_edge_alignment", return_value=refined_quad),
            mock.patch("perspective_detect._score_candidate", side_effect=fake_score_candidate),
            mock.patch("perspective_detect._should_enable_lsd_v2", return_value=False),
        ):
            result = detect_best_candidate_with_profile(image, load_opencv_profile())

        self.assertIsNotNone(result)
        assert result is not None
        methods = [item["method"] for item in result["candidates"]]
        self.assertIn("contour_quad_edge", methods)
        self.assertEqual(result["best"]["method"], "contour_quad_edge")

    def test_combine_score_uses_profile_weights_and_method_bias(self) -> None:
        base_metrics = {
            "aspect_score": 0.8,
            "symmetry_score": 0.7,
            "parallel_score": 0.6,
            "rectangularity_score": 0.5,
            "center_score": 0.4,
            "edge_score": 0.9,
            "coverage_score": 0.65,
            "blue_penalty": 0.2,
            "top_dark_penalty": 0.1,
            "floor_penalty": 0.3,
            "spill_penalty": 0.25,
        }
        profile = {
            "weights": {
                "aspect_score": 0.1,
                "symmetry_score": 0.0,
                "parallel_score": 0.0,
                "rectangularity_score": 0.0,
                "center_score": 0.0,
                "edge_score": 0.2,
                "coverage_score": 0.0,
                "blue_penalty": -0.3,
                "top_dark_penalty": 0.0,
                "floor_penalty": -0.4,
                "spill_penalty": 0.0,
            },
            "method_bias": {
                "refined_edges": 0.15,
                "bright_screen": -0.1,
            },
            "base_method_bias": {
                "refined_edges": 0.02,
            },
        }

        refined_score = combine_score_from_metrics(
            base_metrics,
            profile,
            "refined_edges",
            base_method="refined_edges",
        )
        bright_score = combine_score_from_metrics(base_metrics, profile, "bright_screen")

        self.assertGreater(refined_score, bright_score)
        self.assertAlmostEqual(refined_score, 0.25, places=4)

    def test_load_training_samples_uses_manual_quad_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "screen-pdf-project.json"
            project_path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "id": "page-1",
                                "manualQuad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                "selectedCandidateIndex": 1,
                                "candidates": [
                                    {
                                        "method": "bright_screen",
                                        "score": 0.5,
                                        "quad": [[0, 0], [90, 0], [90, 70], [0, 70]],
                                        "metrics": {
                                            "aspect_score": 0.5,
                                            "symmetry_score": 0.5,
                                            "parallel_score": 0.5,
                                            "rectangularity_score": 0.5,
                                            "center_score": 0.5,
                                            "edge_score": 0.5,
                                            "coverage_score": 0.5,
                                            "blue_penalty": 0.2,
                                            "top_dark_penalty": 0.1,
                                            "floor_penalty": 0.2,
                                            "spill_penalty": 0.2
                                        }
                                    },
                                    {
                                        "method": "refined_edges",
                                        "score": 0.4,
                                        "quad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                        "metrics": {
                                            "aspect_score": 0.7,
                                            "symmetry_score": 0.7,
                                            "parallel_score": 0.7,
                                            "rectangularity_score": 0.7,
                                            "center_score": 0.7,
                                            "edge_score": 0.7,
                                            "coverage_score": 0.7,
                                            "blue_penalty": 0.0,
                                            "top_dark_penalty": 0.0,
                                            "floor_penalty": 0.0,
                                            "spill_penalty": 0.0
                                        }
                                    }
                                ]
                            },
                            {
                                "id": "page-2",
                                "manualQuad": None,
                                "candidates": []
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            samples = load_training_samples(Path(temp_dir))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["page_id"], "page-1")
        self.assertEqual(samples[0]["selected_method"], "refined_edges")
        self.assertEqual(len(samples[0]["candidates"]), 2)

    def test_load_training_samples_uses_selected_candidate_manual_quad_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "screen-pdf-project.json"
            project_path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "id": "page-1",
                                "manualQuad": None,
                                "status": "reviewed",
                                "activeQuad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                "selectedCandidateIndex": 1,
                                "candidates": [
                                    {
                                        "method": "bright_screen",
                                        "score": 0.5,
                                        "quad": [[0, 0], [90, 0], [90, 70], [0, 70]],
                                        "metrics": {},
                                    },
                                    {
                                        "method": "refined_edges",
                                        "source": "runtime_teacher",
                                        "modelId": "r66",
                                        "score": 0.4,
                                        "manualQuad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                        "quad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                        "metrics": {},
                                    },
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            samples = load_training_samples(Path(temp_dir))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["page_id"], "page-1")
        self.assertEqual(samples[0]["selected_method"], "refined_edges")

    def test_load_training_samples_can_rerun_detector_from_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            quad = np.array([[180, 120], [1090, 150], [1110, 580], [170, 560]], dtype=np.int32)
            cv2.fillConvexPoly(image, quad, (242, 242, 242))
            cv2.polylines(image, [quad], True, (255, 255, 255), 6)
            image_path = temp_path / "page-1.jpg"
            cv2.imwrite(str(image_path), image)

            project_path = temp_path / "screen-pdf-project.json"
            project_path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "id": "page-1",
                                "path": "page-1.jpg",
                                "manualQuad": [[180, 120], [1090, 150], [1110, 580], [170, 560]],
                                "selectedCandidateIndex": 0,
                                "candidates": []
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            samples = load_training_samples(temp_path, rerun_detection=True)

        self.assertEqual(len(samples), 1)
        self.assertGreater(len(samples[0]["candidates"]), 0)

    def test_load_training_samples_resolves_stale_absolute_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            quad = np.array([[180, 120], [1090, 150], [1110, 580], [170, 560]], dtype=np.int32)
            cv2.fillConvexPoly(image, quad, (242, 242, 242))
            cv2.polylines(image, [quad], True, (255, 255, 255), 6)
            image_path = temp_path / "IMG_1001.jpeg"
            cv2.imwrite(str(image_path), image)

            project_path = temp_path / "screen-pdf-project.json"
            project_path.write_text(
                json.dumps(
                    {
                        "sourceDir": str(temp_path / "legacy-source"),
                        "pages": [
                            {
                                "id": "page-1",
                                "path": str(temp_path / "legacy-source" / "IMG_1001.jpeg"),
                                "manualQuad": [[180, 120], [1090, 150], [1110, 580], [170, 560]],
                                "selectedCandidateIndex": 0,
                                "candidates": []
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            samples = load_training_samples(temp_path, rerun_detection=True)

        self.assertEqual(len(samples), 1)
        self.assertGreater(len(samples[0]["candidates"]), 0)

    def test_evaluate_profile_returns_expected_summary(self) -> None:
        samples = [
            {
                "project_path": "/tmp/project-a/screen-pdf-project.json",
                "page_id": "page-1",
                "selected_index": 1,
                "selected_method": "refined_edges",
                "candidates": [
                    {
                        "method": "bright_screen",
                        "metrics": {
                            "aspect_score": 0.4,
                            "symmetry_score": 0.4,
                            "parallel_score": 0.4,
                            "rectangularity_score": 0.4,
                            "center_score": 0.4,
                            "edge_score": 0.4,
                            "coverage_score": 0.4,
                            "blue_penalty": 0.2,
                            "top_dark_penalty": 0.1,
                            "floor_penalty": 0.2,
                            "spill_penalty": 0.2,
                        },
                        "iou": 0.52,
                    },
                    {
                        "method": "refined_edges",
                        "metrics": {
                            "aspect_score": 0.8,
                            "symmetry_score": 0.8,
                            "parallel_score": 0.8,
                            "rectangularity_score": 0.8,
                            "center_score": 0.8,
                            "edge_score": 0.8,
                            "coverage_score": 0.8,
                            "blue_penalty": 0.0,
                            "top_dark_penalty": 0.0,
                            "floor_penalty": 0.0,
                            "spill_penalty": 0.0,
                        },
                        "iou": 0.94,
                    },
                ],
            }
        ]
        summary = evaluate_profile_on_samples(DEFAULT_SCORING_PROFILE, samples)

        self.assertEqual(summary["pages"], 1)
        self.assertAlmostEqual(summary["top1_mean_iou"], 0.94, places=4)
        self.assertAlmostEqual(summary["oracle_mean_iou"], 0.94, places=4)
        self.assertEqual(summary["top1_ge_0_9"], 1)

    def test_objective_score_penalizes_global_regression(self) -> None:
        candidate = {
            "focus": {
                "top1_mean_iou": 0.82,
                "oracle_mean_iou": 0.9,
                "top1_ge_0_8": 10,
                "pages": 20,
            },
            "global": {
                "top1_mean_iou": 0.78,
                "oracle_mean_iou": 0.88,
                "top1_ge_0_8": 100,
                "pages": 200,
            },
            "project_floor": 0.62,
        }
        safer = {
            "focus": {
                "top1_mean_iou": 0.8,
                "oracle_mean_iou": 0.88,
                "top1_ge_0_8": 9,
                "pages": 20,
            },
            "global": {
                "top1_mean_iou": 0.86,
                "oracle_mean_iou": 0.9,
                "top1_ge_0_8": 150,
                "pages": 200,
            },
            "project_floor": 0.75,
        }
        baseline_global_top1 = 0.85

        self.assertLess(
            objective_score(candidate, baseline_global_top1),
            objective_score(safer, baseline_global_top1),
        )


if __name__ == "__main__":
    unittest.main()
