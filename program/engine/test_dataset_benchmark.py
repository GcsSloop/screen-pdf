from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_benchmark import (
    build_split,
    build_project_aware_split,
    compute_scene_profile,
    load_manual_pages,
    max_corner_error,
    normalized_point_error,
    perspective_tilt_error,
    point_success_ratio,
    quad_geometry_metrics,
    quad_inset_ratio,
    screen_relative_point_error,
    summarize_geometry_metric_rows,
)


class DatasetBenchmarkTests(unittest.TestCase):
    def test_normalized_point_error_is_zero_for_identical_quads(self) -> None:
        quad = [[0, 0], [100, 0], [100, 80], [0, 80]]

        error = normalized_point_error(quad, quad)

        self.assertEqual(error, 0.0)

    def test_point_success_ratio_uses_threshold(self) -> None:
        manual = [[0, 0], [100, 0], [100, 100], [0, 100]]
        close = [[0.2, 0.2], [99.8, 0.2], [100.1, 99.8], [0.2, 100.0]]
        far = [[5, 5], [95, 5], [95, 95], [5, 95]]

        ratio = point_success_ratio([normalized_point_error(manual, close), normalized_point_error(manual, far)], 0.01)

        self.assertEqual(ratio, 0.5)

    def test_screen_relative_metrics_capture_single_corner_outlier(self) -> None:
        manual = [[0, 0], [200, 0], [200, 100], [0, 100]]
        predicted = [[0, 0], [200, 0], [170, 100], [0, 100]]

        mean_error = screen_relative_point_error(manual, predicted)
        max_error = max_corner_error(manual, predicted)

        self.assertGreater(max_error, mean_error)
        self.assertGreater(max_error, 0.15)

    def test_perspective_tilt_error_is_zero_for_identical_quads(self) -> None:
        quad = [[10, 20], [210, 30], [205, 120], [5, 110]]

        tilt = perspective_tilt_error(quad, quad)

        self.assertAlmostEqual(tilt, 0.0, places=6)

    def test_quad_inset_ratio_is_positive_for_inward_quad(self) -> None:
        manual = [[0, 0], [200, 0], [200, 100], [0, 100]]
        inward = [[10, 5], [190, 5], [190, 95], [10, 95]]
        outward = [[-10, -5], [210, -5], [210, 105], [-10, 105]]

        self.assertGreater(quad_inset_ratio(manual, inward), 0.0)
        self.assertLess(quad_inset_ratio(manual, outward), 0.0)

    def test_quad_geometry_metrics_bundle_new_shape_metrics(self) -> None:
        manual = [[0, 0], [200, 0], [200, 100], [0, 100]]
        predicted = [[5, 0], [195, 8], [190, 95], [10, 92]]

        metrics = quad_geometry_metrics(manual, predicted)

        self.assertIn("screen_relative_point_error", metrics)
        self.assertIn("max_corner_error", metrics)
        self.assertIn("perspective_tilt_error", metrics)
        self.assertIn("quad_inset_ratio", metrics)
        self.assertGreater(metrics["perspective_tilt_error"], 0.0)

    def test_summary_uses_per_corner_point_le_0_01_ratio_and_max_corner_le_0_03_ratio(self) -> None:
        manual = [[0, 0], [100, 0], [100, 100], [0, 100]]
        almost_good = [[0, 0], [100, 0], [102, 100], [0, 100]]
        one_corner_too_far = [[0, 0], [100, 0], [105, 100], [0, 100]]

        summary = summarize_geometry_metric_rows(
            [
                quad_geometry_metrics(manual, almost_good),
                quad_geometry_metrics(manual, one_corner_too_far),
            ]
        )

        self.assertEqual(summary["point_le_0_01_ratio"], 0.75)
        self.assertEqual(summary["max_corner_le_0_03_ratio"], 0.5)

    def test_compute_scene_profile_detects_near_color_background(self) -> None:
        image = np.full((120, 180, 3), 140, dtype=np.uint8)
        cv2.rectangle(image, (30, 20), (150, 100), (150, 145, 145), -1)
        quad = [[30, 20], [150, 20], [150, 100], [30, 100]]

        profile = compute_scene_profile(image, quad)

        self.assertTrue(profile["near_color_background"])
        self.assertIn("near_color_background", profile["scene_tags"])

    def test_compute_scene_profile_keeps_high_contrast_scene_out_of_near_color(self) -> None:
        image = np.full((120, 180, 3), 30, dtype=np.uint8)
        cv2.rectangle(image, (30, 20), (150, 100), (240, 240, 240), -1)
        quad = [[30, 20], [150, 20], [150, 100], [30, 100]]

        profile = compute_scene_profile(image, quad)

        self.assertFalse(profile["near_color_background"])
        self.assertGreater(profile["lab_distance"], 18.0)

    def test_compute_scene_profile_detects_black_frame_scene(self) -> None:
        image = np.full((140, 220, 3), 18, dtype=np.uint8)
        cv2.rectangle(image, (40, 25), (180, 115), (35, 35, 35), -1)
        cv2.rectangle(image, (52, 35), (168, 105), (230, 230, 235), -1)
        quad = [[40, 25], [180, 25], [180, 115], [40, 115]]

        profile = compute_scene_profile(image, quad)

        self.assertTrue(profile["black_frame_scene"])
        self.assertIn("black_frame_scene", profile["scene_tags"])

    def test_load_manual_pages_skips_missing_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir = root / "project-a"
            project_dir.mkdir()
            image = np.zeros((100, 120, 3), dtype=np.uint8)
            image_path = project_dir / "page-1.jpg"
            cv2.imwrite(str(image_path), image)
            project_path = project_dir / "screen-pdf-project.json"
            project_path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "id": "page-1",
                                "path": "page-1.jpg",
                                "manualQuad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                            },
                            {
                                "id": "page-2",
                                "path": "missing.jpg",
                                "manualQuad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pages = load_manual_pages(root)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["page_id"], "page-1")

    def test_load_manual_pages_includes_reviewed_non_model_active_quad_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir = root / "project-a"
            project_dir.mkdir()
            image = np.zeros((100, 120, 3), dtype=np.uint8)
            image_path = project_dir / "page-1.jpg"
            cv2.imwrite(str(image_path), image)
            project_path = project_dir / "screen-pdf-project.json"
            project_path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "id": "page-1",
                                "path": "page-1.jpg",
                                "status": "reviewed",
                                "manualQuad": None,
                                "activeQuad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                "selectedCandidateIndex": 0,
                                "candidates": [
                                    {
                                        "method": "document_quad",
                                        "source": "opencv",
                                        "quad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pages = load_manual_pages(root)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["manual_quad"], [[0, 0], [100, 0], [100, 80], [0, 80]])

    def test_load_manual_pages_includes_selected_candidate_manual_quad_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir = root / "project-a"
            project_dir.mkdir()
            image = np.zeros((100, 120, 3), dtype=np.uint8)
            image_path = project_dir / "page-1.jpg"
            cv2.imwrite(str(image_path), image)
            project_path = project_dir / "screen-pdf-project.json"
            project_path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "id": "page-1",
                                "path": "page-1.jpg",
                                "status": "reviewed",
                                "manualQuad": None,
                                "activeQuad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                "selectedCandidateIndex": 0,
                                "candidates": [
                                    {
                                        "method": "teacher_current",
                                        "source": "runtime_teacher",
                                        "modelId": "r66",
                                        "manualQuad": [[1, 1], [99, 1], [99, 79], [1, 79]],
                                        "quad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pages = load_manual_pages(root)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["manual_quad"], [[1, 1], [99, 1], [99, 79], [1, 79]])

    def test_load_manual_pages_excludes_model_selected_active_quad_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir = root / "project-a"
            project_dir.mkdir()
            image = np.zeros((100, 120, 3), dtype=np.uint8)
            image_path = project_dir / "page-1.jpg"
            cv2.imwrite(str(image_path), image)
            project_path = project_dir / "screen-pdf-project.json"
            project_path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "id": "page-1",
                                "path": "page-1.jpg",
                                "status": "reviewed",
                                "manualQuad": None,
                                "activeQuad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                "selectedCandidateIndex": 0,
                                "candidates": [
                                    {
                                        "method": "teacher_current",
                                        "source": "runtime_teacher",
                                        "modelId": "r57e001",
                                        "quad": [[0, 0], [100, 0], [100, 80], [0, 80]],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pages = load_manual_pages(root)

        self.assertEqual(pages, [])

    def test_build_split_keeps_every_project_in_test_when_possible(self) -> None:
        pages = [
            {"project_path": "/tmp/a/screen-pdf-project.json", "page_id": "a1"},
            {"project_path": "/tmp/a/screen-pdf-project.json", "page_id": "a2"},
            {"project_path": "/tmp/a/screen-pdf-project.json", "page_id": "a3"},
            {"project_path": "/tmp/b/screen-pdf-project.json", "page_id": "b1"},
            {"project_path": "/tmp/b/screen-pdf-project.json", "page_id": "b2"},
            {"project_path": "/tmp/c/screen-pdf-project.json", "page_id": "c1"},
        ]

        split = build_split(pages, test_ratio=0.34, seed=7)

        self.assertEqual(len(split["test"]), 3)
        test_projects = {item["project_path"] for item in split["test"]}
        self.assertSetEqual(
            test_projects,
            {
                "/tmp/a/screen-pdf-project.json",
                "/tmp/b/screen-pdf-project.json",
            },
        )
        train_projects = {item["project_path"] for item in split["train"]}
        self.assertIn("/tmp/c/screen-pdf-project.json", train_projects)

    def test_build_project_aware_split_reserves_focus_project_test_pages(self) -> None:
        pages = [
            {"project_path": "/tmp/a/screen-pdf-project.json", "project_name": "A", "page_id": "a1"},
            {"project_path": "/tmp/a/screen-pdf-project.json", "project_name": "A", "page_id": "a2"},
            {"project_path": "/tmp/a/screen-pdf-project.json", "project_name": "A", "page_id": "a3"},
            {"project_path": "/tmp/a/screen-pdf-project.json", "project_name": "A", "page_id": "a4"},
            {"project_path": "/tmp/b/screen-pdf-project.json", "project_name": "B", "page_id": "b1"},
            {"project_path": "/tmp/b/screen-pdf-project.json", "project_name": "B", "page_id": "b2"},
            {"project_path": "/tmp/c/screen-pdf-project.json", "project_name": "C", "page_id": "c1"},
        ]

        split = build_project_aware_split(
            pages,
            focus_projects={"/tmp/a/screen-pdf-project.json"},
            test_ratio=0.25,
            focus_test_ratio=0.5,
            seed=7,
        )

        focus_test_ids = {item["page_id"] for item in split["focus_test"]}
        focus_train_ids = {item["page_id"] for item in split["focus_train"]}
        train_ids = {item["page_id"] for item in split["train"]}

        self.assertSetEqual(focus_test_ids, {"a2", "a4"})
        self.assertSetEqual(focus_train_ids, {"a1", "a3"})
        self.assertTrue(focus_test_ids.isdisjoint(train_ids))
        self.assertTrue(focus_test_ids.isdisjoint(focus_train_ids))
        self.assertTrue({"a1", "a3"}.issubset(train_ids))

    def test_build_project_aware_split_tracks_focus_metadata(self) -> None:
        pages = [
            {"project_path": "/tmp/focus/screen-pdf-project.json", "project_name": "Focus", "page_id": "p1"},
            {"project_path": "/tmp/focus/screen-pdf-project.json", "project_name": "Focus", "page_id": "p2"},
            {"project_path": "/tmp/base/screen-pdf-project.json", "project_name": "Base", "page_id": "b1"},
            {"project_path": "/tmp/base/screen-pdf-project.json", "project_name": "Base", "page_id": "b2"},
        ]

        split = build_project_aware_split(
            pages,
            focus_projects={"Focus"},
            test_ratio=0.5,
            focus_test_ratio=0.5,
            seed=11,
        )

        metadata = split["metadata"]

        self.assertEqual(metadata["focus_project_count"], 1)
        self.assertEqual(metadata["focus_test_pages"], 1)
        self.assertEqual(metadata["focus_train_pages"], 1)
        self.assertIn("Focus", metadata["focus_project_names"])

    def test_build_project_aware_split_excludes_holdout_projects_from_training(self) -> None:
        pages = [
            {"project_path": "/tmp/focus/screen-pdf-project.json", "project_name": "Focus", "page_id": "f1"},
            {"project_path": "/tmp/focus/screen-pdf-project.json", "project_name": "Focus", "page_id": "f2"},
            {"project_path": "/tmp/holdout/screen-pdf-project.json", "project_name": "Holdout", "page_id": "h1"},
            {"project_path": "/tmp/holdout/screen-pdf-project.json", "project_name": "Holdout", "page_id": "h2"},
            {"project_path": "/tmp/base/screen-pdf-project.json", "project_name": "Base", "page_id": "b1"},
            {"project_path": "/tmp/base/screen-pdf-project.json", "project_name": "Base", "page_id": "b2"},
        ]

        split = build_project_aware_split(
            pages,
            focus_projects={"Focus"},
            holdout_projects={"Holdout"},
            test_ratio=0.5,
            focus_test_ratio=0.5,
            seed=3,
        )

        train_ids = {item["page_id"] for item in split["train"]}
        holdout_ids = {item["page_id"] for item in split["holdout"]}

        self.assertTrue({"h1", "h2"}.isdisjoint(train_ids))
        self.assertSetEqual(holdout_ids, {"h1", "h2"})
        self.assertEqual(split["metadata"]["holdout_pages"], 2)


if __name__ == "__main__":
    unittest.main()
