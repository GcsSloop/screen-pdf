from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corner_model_pipeline import build_infer_request, export_corner_dataset


class CornerModelPipelineTests(unittest.TestCase):
    def test_build_infer_request_uses_coarse_quad_and_normalizes_corners(self) -> None:
        image_shape = (800, 1200, 3)
        coarse_quad = np.array(
            [[200, 120], [980, 130], [1010, 620], [180, 640]],
            dtype=np.float32,
        )

        request = build_infer_request("page-1", "/tmp/a.jpg", image_shape, coarse_quad)

        self.assertEqual(request["page_id"], "page-1")
        self.assertEqual(request["image_path"], "/tmp/a.jpg")
        self.assertIn("roi", request)
        self.assertIn("coarse_quad_norm", request)
        self.assertEqual(len(request["coarse_quad_norm"]), 4)
        for x, y in request["coarse_quad_norm"]:
            self.assertGreaterEqual(x, 0.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertLessEqual(y, 1.0)

    def test_export_corner_dataset_writes_manifests_and_roi_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            project_dir = root / "project-a"
            project_dir.mkdir(parents=True)
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            quad = np.array([[180, 110], [1080, 130], [1100, 580], [160, 600]], dtype=np.int32)
            cv2.fillConvexPoly(image, quad, (240, 240, 240))
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
                                "manualQuad": [[180, 110], [1080, 130], [1100, 580], [160, 600]],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_dir = Path(temp_dir) / "exported"

            report = export_corner_dataset(root, output_dir, seed=7, test_ratio=0.25)

            self.assertEqual(report["pages"], 1)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "train.jsonl").exists())
            self.assertTrue((output_dir / "test.jsonl").exists())

            lines = (output_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            sample = json.loads(lines[0])
            self.assertEqual(sample["page_id"], "page-1")
            self.assertTrue((output_dir / sample["roi_path"]).exists())
            self.assertEqual(sample["split"], "train")
            self.assertEqual(len(sample["corner_norm"]), 4)
            for x, y in sample["corner_norm"]:
                self.assertGreaterEqual(x, 0.0)
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(x, 1.0)
                self.assertLessEqual(y, 1.0)

    def test_export_corner_dataset_uses_detector_coarse_roi_not_manual_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            project_dir = root / "project-a"
            project_dir.mkdir(parents=True)
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
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
                                "manualQuad": [[300, 200], [900, 200], [900, 500], [300, 500]],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output_dir = Path(temp_dir) / "exported"

            from unittest import mock

            coarse_quad = np.array([[100, 80], [1100, 100], [1080, 620], [120, 600]], dtype=np.float32)
            with mock.patch(
                "corner_model_pipeline.detect_best_candidate_with_profile",
                return_value={"best": {"quad": coarse_quad.tolist()}},
            ):
                export_corner_dataset(root, output_dir, seed=7, test_ratio=0.25)

            sample = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8").strip())
            self.assertLessEqual(sample["roi"]["x"], 120)
            self.assertGreaterEqual(sample["roi"]["width"], 980)


if __name__ == "__main__":
    unittest.main()
