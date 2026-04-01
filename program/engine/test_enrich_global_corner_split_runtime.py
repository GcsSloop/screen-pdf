from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enrich_global_corner_split_runtime import load_failure_layer_index, merge_failure_layer_fields


class EnrichGlobalCornerSplitRuntimeTests(unittest.TestCase):
    def test_load_failure_layer_index_uses_page_id_and_image_path_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "diagnostics.json"
            payload = {
                "runtime_examples": [
                    {
                        "page_id": "p1",
                        "image_path": "/tmp/a.jpg",
                        "category": "runtime_candidate_recoverable",
                        "baseline_metrics": {"max_corner_error": 0.12},
                        "runtime_oracle_metrics": {"max_corner_error": 0.02},
                        "opencv_oracle_metrics": {"max_corner_error": 0.11},
                        "union_oracle_metrics": {"max_corner_error": 0.02},
                    }
                ],
                "opencv_examples": [],
                "hard_examples": [],
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            index = load_failure_layer_index(path)

        self.assertIn(("p1", "/tmp/a.jpg"), index)
        self.assertEqual(index[("p1", "/tmp/a.jpg")]["category"], "runtime_candidate_recoverable")

    def test_merge_failure_layer_fields_writes_category_and_gain_fields(self) -> None:
        row = {
            "page_id": "p2",
            "image_path": "/tmp/b.jpg",
            "manual_quad": [[1, 1], [2, 1], [2, 2], [1, 2]],
        }
        diagnostic = {
            "category": "opencv_recoverable",
            "baseline_metrics": {"max_corner_error": 0.24},
            "runtime_oracle_metrics": {"max_corner_error": 0.10},
            "opencv_oracle_metrics": {"max_corner_error": 0.01},
            "union_oracle_metrics": {"max_corner_error": 0.01},
        }

        merged = merge_failure_layer_fields(row, diagnostic)

        self.assertEqual(merged["failure_layer_category"], "opencv_recoverable")
        self.assertFalse(merged["failure_layer_baseline_strict_ok"])
        self.assertFalse(merged["failure_layer_runtime_strict_ok"])
        self.assertTrue(merged["failure_layer_opencv_strict_ok"])
        self.assertTrue(merged["failure_layer_union_strict_ok"])
        self.assertAlmostEqual(merged["failure_layer_runtime_gain"], 0.14, places=6)
        self.assertAlmostEqual(merged["failure_layer_opencv_gain"], 0.23, places=6)
        self.assertAlmostEqual(merged["failure_layer_union_gain"], 0.23, places=6)

    def test_merge_failure_layer_fields_keeps_row_when_diagnostic_missing(self) -> None:
        row = {"page_id": "p3", "image_path": "/tmp/c.jpg", "adaptive_weight": 1.2}

        merged = merge_failure_layer_fields(row, None)

        self.assertEqual(merged, row)


if __name__ == "__main__":
    unittest.main()
