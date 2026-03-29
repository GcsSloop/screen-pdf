from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_corner_benchmark import benchmark_project


class ProjectCornerBenchmarkTests(unittest.TestCase):
    def test_benchmark_project_reports_active_and_candidate_method_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "IMG_0001.jpg"
            image_path.write_bytes(b"fake")
            project_path = root / "screen-pdf-project.json"
            project = {
                "pages": [
                    {
                        "id": "IMG_0001",
                        "path": str(image_path),
                        "manualQuad": [[10, 10], [110, 10], [110, 90], [10, 90]],
                        "activeQuad": [[20, 20], [120, 20], [120, 100], [20, 100]],
                        "bestMethod": "model_three_stage_local_moe",
                        "candidates": [
                            {
                                "method": "contour_quad_edge",
                                "quad": [[10, 10], [110, 10], [110, 90], [10, 90]],
                            },
                            {
                                "method": "document_quad_edge",
                                "quad": [[18, 18], [118, 18], [118, 98], [18, 98]],
                            },
                        ],
                    }
                ]
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

            result = benchmark_project(project_path, candidate_methods=["contour_quad_edge", "document_quad_edge"])

        self.assertEqual(result["pages"], 1)
        self.assertIn("active", result["summaries"])
        self.assertIn("contour_quad_edge", result["summaries"])
        self.assertLess(
            result["summaries"]["contour_quad_edge"]["point_error_mean"],
            result["summaries"]["active"]["point_error_mean"],
        )
        self.assertEqual(result["summaries"]["contour_quad_edge"]["point_le_0_01_ratio"], 1.0)
        self.assertIn("screen_relative_error_mean", result["summaries"]["active"])
        self.assertIn("max_corner_error_mean", result["summaries"]["active"])
        self.assertIn("perspective_tilt_error_mean", result["summaries"]["active"])
        self.assertIn("quad_inset_ratio_mean", result["summaries"]["active"])

    def test_benchmark_project_uses_selected_candidate_manual_quad_in_new_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "IMG_0002.jpg"
            image_path.write_bytes(b"fake")
            project_path = root / "screen-pdf-project.json"
            project = {
                "pages": [
                    {
                        "id": "IMG_0002",
                        "path": str(image_path),
                        "manualQuad": None,
                        "status": "reviewed",
                        "selectedCandidateIndex": 0,
                        "activeQuad": [[20, 20], [120, 20], [120, 100], [20, 100]],
                        "bestMethod": "teacher_current",
                        "candidates": [
                            {
                                "method": "teacher_current",
                                "source": "runtime_teacher",
                                "modelId": "r66",
                                "manualQuad": [[10, 10], [110, 10], [110, 90], [10, 90]],
                                "quad": [[20, 20], [120, 20], [120, 100], [20, 100]],
                            }
                        ],
                    }
                ]
            }
            project_path.write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

            result = benchmark_project(project_path, runtime_runner=None)

        self.assertEqual(result["pages"], 1)
        self.assertIn("active", result["summaries"])


if __name__ == "__main__":
    unittest.main()
