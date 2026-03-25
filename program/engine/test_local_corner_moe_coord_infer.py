from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_moe_coord_infer import blend_structure_aware_point, evaluate_local_corner_moe_coord


class LocalCornerMoECoordInferTests(unittest.TestCase):
    def test_blend_structure_aware_point_prefers_coord_head_when_visibility_is_high(self) -> None:
        decoded = torch.tensor([[0.45, 0.70]], dtype=torch.float32)
        coord = torch.tensor([[0.60, 0.84]], dtype=torch.float32)
        visibility = torch.tensor([[0.95, 0.90]], dtype=torch.float32)

        point = blend_structure_aware_point(decoded, coord, visibility, base_coord_mix=0.25)

        self.assertGreater(float(point[0, 0]), 0.50)
        self.assertGreater(float(point[0, 1]), 0.75)

    def test_blend_structure_aware_point_prefers_decoded_point_when_visibility_is_low(self) -> None:
        decoded = torch.tensor([[0.45, 0.70]], dtype=torch.float32)
        coord = torch.tensor([[0.60, 0.84]], dtype=torch.float32)
        visibility = torch.tensor([[0.10, 0.08]], dtype=torch.float32)

        point = blend_structure_aware_point(decoded, coord, visibility, base_coord_mix=0.25)

        self.assertLess(float(point[0, 0]), 0.50)
        self.assertLess(float(point[0, 1]), 0.75)

    def test_blend_structure_aware_point_can_fallback_to_legacy_fixed_mix(self) -> None:
        decoded = torch.tensor([[0.45, 0.70]], dtype=torch.float32)
        coord = torch.tensor([[0.60, 0.84]], dtype=torch.float32)
        visibility = torch.tensor([[0.99, 0.99]], dtype=torch.float32)

        point = blend_structure_aware_point(
            decoded,
            coord,
            visibility,
            base_coord_mix=0.25,
            use_visibility=False,
        )

        self.assertAlmostEqual(float(point[0, 0]), 0.4875, places=4)
        self.assertAlmostEqual(float(point[0, 1]), 0.735, places=4)

    def test_evaluate_local_corner_moe_coord_reports_timing(self) -> None:
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

            with mock.patch("local_corner_moe_coord_infer.LocalCornerMoECoordPredictor", return_value=FakePredictor()), mock.patch(
                "local_corner_moe_coord_infer.build_local_corner_patch_sample",
                side_effect=lambda **kwargs: {
                    "patch": {"x": 0, "y": 0, "size": 96},
                    "predicted_point": [48.0, 48.0],
                    "patch_image": np.zeros((96, 96, 3), dtype=np.uint8),
                    "corner_index": kwargs["corner_index"],
                },
            ):
                result = evaluate_local_corner_moe_coord(Path("/tmp/model.pt"), root, split="test")

        self.assertIn("avg_page_infer_ms", result)
        self.assertIn("avg_corner_infer_ms", result)
        self.assertIn("all_corners_le_0_01_ratio", result)
        self.assertIn("screen_relative_point_error_mean", result)
        self.assertIn("bl_corner_error_mean", result)

    def test_evaluate_local_corner_moe_coord_passes_patch_bias_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for corner_index in range(4):
                rows.append(
                    {
                        "project_path": "/tmp/project.json",
                        "page_id": "page-1",
                        "corner_index": corner_index,
                        "patch": {"x": 0, "y": 0, "size": 96, "bottom_vertical_bias": 0.2},
                        "predicted_quad": [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]],
                        "manual_quad": [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]],
                        "image_path": "/tmp/fake.png",
                        "predicted_point": [10.0, 10.0],
                        "target_point_norm": [0.5, 0.5],
                    }
                )
            (root / "test.jsonl").write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")

            class FakePredictor:
                device = type("D", (), {"type": "cpu"})()

                def __call__(self, sample, predicted_quad):
                    return np.array([0.5, 0.5], dtype=np.float32)

            seen_biases: list[float] = []

            def fake_build_local_corner_patch_sample(**kwargs):
                seen_biases.append(float(kwargs.get("bottom_vertical_bias", 0.0)))
                return {
                    "patch": {"x": 0, "y": 0, "size": 96},
                    "predicted_point": [48.0, 48.0],
                    "patch_image": np.zeros((96, 96, 3), dtype=np.uint8),
                    "corner_index": kwargs["corner_index"],
                }

            with mock.patch("local_corner_moe_coord_infer.LocalCornerMoECoordPredictor", return_value=FakePredictor()), mock.patch(
                "local_corner_moe_coord_infer.build_local_corner_patch_sample",
                side_effect=fake_build_local_corner_patch_sample,
            ):
                evaluate_local_corner_moe_coord(Path("/tmp/model.pt"), root, split="test")

        self.assertEqual(seen_biases, [0.2, 0.2, 0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
