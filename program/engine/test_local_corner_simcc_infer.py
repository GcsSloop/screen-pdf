from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_simcc_infer import evaluate_local_corner_simcc


class LocalCornerSimCCInferTests(unittest.TestCase):
    def test_evaluate_local_corner_simcc_reports_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            row = {
                "project_path": "/tmp/project.json",
                "page_id": "page-1",
                "corner_index": 0,
                "patch": {"x": 0, "y": 0, "size": 96},
                "predicted_quad": [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]],
                "manual_quad": [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]],
                "image_path": "/tmp/fake.png",
                "predicted_point": [10.0, 10.0],
                "target_point_norm": [0.5, 0.5],
            }
            rows = []
            for corner_index in range(4):
                entry = dict(row)
                entry["corner_index"] = corner_index
                rows.append(entry)
            (root / "test.jsonl").write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")

            class FakePredictor:
                device = type("D", (), {"type": "cpu"})()

                def __call__(self, sample, predicted_quad):
                    return np.array([0.5, 0.5], dtype=np.float32)

            with mock.patch("local_corner_simcc_infer.LocalCornerSimCCPredictor", return_value=FakePredictor()), mock.patch(
                "local_corner_simcc_infer.build_local_corner_patch_sample",
                side_effect=lambda **kwargs: {
                    "patch": {"x": 0, "y": 0, "size": 96},
                    "predicted_point": [48.0, 48.0],
                    "patch_image": np.zeros((96, 96, 3), dtype=np.uint8),
                    "corner_index": kwargs["corner_index"],
                },
            ):
                result = evaluate_local_corner_simcc(Path("/tmp/model.pt"), root, split="test")

        self.assertIn("avg_page_infer_ms", result)
        self.assertIn("avg_corner_infer_ms", result)
        self.assertGreaterEqual(float(result["avg_page_infer_ms"]), 0.0)


if __name__ == "__main__":
    unittest.main()
