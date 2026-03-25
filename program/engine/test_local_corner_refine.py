from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_refine import (
    build_local_corner_patch_sample,
    build_patch_features,
    export_local_corner_patch_dataset,
    refine_quad_with_residual_predictor,
)


class LocalCornerRefineTests(unittest.TestCase):
    def test_build_local_corner_patch_sample_returns_centered_residual_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((200, 300, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            predicted_quad = np.array([[60, 40], [240, 42], [238, 160], [58, 158]], dtype=np.float32)
            manual_quad = np.array([[66, 46], [240, 42], [238, 160], [58, 158]], dtype=np.float32)

            sample = build_local_corner_patch_sample(
                image_path=image_path,
                page_id="page-1",
                corner_index=0,
                predicted_quad=predicted_quad,
                manual_quad=manual_quad,
                patch_size=64,
            )

        self.assertEqual(sample["patch"]["size"], 64)
        self.assertAlmostEqual(sample["target_residual_norm"][0], 6 / 64, places=4)
        self.assertAlmostEqual(sample["target_residual_norm"][1], 6 / 64, places=4)

    def test_build_patch_features_contains_edge_planes(self) -> None:
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.line(patch, (8, 8), (8, 56), (255, 255, 255), 2)
        cv2.line(patch, (8, 8), (56, 8), (255, 255, 255), 2)

        features = build_patch_features(patch, corner_index=0, input_size=64)

        self.assertEqual(features.shape, (10, 64, 64))
        self.assertGreater(float(features[4].max()), 0.1)
        self.assertGreater(float(features[7].max()), 0.1)

    def test_build_patch_features_can_include_structure_planes(self) -> None:
        patch = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.line(patch, (8, 8), (8, 56), (255, 255, 255), 2)
        cv2.line(patch, (8, 8), (56, 8), (255, 255, 255), 2)

        features = build_patch_features(patch, corner_index=0, input_size=64, input_channels=13)

        self.assertEqual(features.shape, (13, 64, 64))
        self.assertGreater(float(features[10].max()), 0.01)
        self.assertGreater(float(features[11].max()), 0.01)

    def test_refine_quad_with_residual_predictor_updates_corners_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((200, 300, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            predicted_quad = np.array([[60, 40], [240, 42], [238, 160], [58, 158]], dtype=np.float32)

            def fake_predictor(sample: dict[str, object]) -> np.ndarray:
                corner_index = int(sample["corner_index"])
                if corner_index == 0:
                    return np.array([0.1, 0.05], dtype=np.float32)
                return np.array([0.0, 0.0], dtype=np.float32)

            refined = refine_quad_with_residual_predictor(
                image_path=image_path,
                predicted_quad=predicted_quad,
                residual_predictor=fake_predictor,
                patch_size=80,
            )

        self.assertAlmostEqual(refined[0][0], 68.0, places=4)
        self.assertAlmostEqual(refined[0][1], 44.0, places=4)
        self.assertEqual(refined[1], [240.0, 42.0])

    def test_export_local_corner_patch_dataset_uses_active_quad_from_project(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir = root / "demo"
            project_dir.mkdir(parents=True)
            image = np.zeros((120, 180, 3), dtype=np.uint8)
            image_path = project_dir / "IMG_0001.jpg"
            cv2.imwrite(str(image_path), image)
            project = {
                "pages": [
                    {
                        "id": "IMG_0001",
                        "path": str(image_path),
                        "activeQuad": [[20, 20], [160, 20], [160, 100], [20, 100]],
                        "manualQuad": [[24, 24], [160, 20], [160, 100], [20, 100]],
                    }
                ]
            }
            (project_dir / "screen-pdf-project.json").write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
            output_dir = root / "local-patch"

            summary = export_local_corner_patch_dataset(
                dataset_root=root,
                output_dir=output_dir,
                seed=7,
                test_ratio=0.0,
                patch_size=64,
            )

            lines = (output_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
            sample = json.loads(lines[0])

        self.assertEqual(summary["page_count"], 1)
        self.assertEqual(summary["pages"], 4)
        self.assertEqual(summary["train_pages"], 1)
        self.assertEqual(summary["test_pages"], 0)
        self.assertEqual(sample["page_id"], "IMG_0001")
        self.assertEqual(sample["predicted_quad"][0], [20.0, 20.0])
        self.assertIn("target_point_norm", sample)

    def test_export_local_corner_patch_dataset_computes_exact_target_point_norm_when_patch_is_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_dir = root / "demo"
            project_dir.mkdir(parents=True)
            image = np.zeros((160, 220, 3), dtype=np.uint8)
            image_path = project_dir / "IMG_0002.jpg"
            cv2.imwrite(str(image_path), image)
            project = {
                "pages": [
                    {
                        "id": "IMG_0002",
                        "path": str(image_path),
                        "activeQuad": [[10, 12], [180, 18], [178, 120], [16, 118]],
                        "manualQuad": [[18, 20], [180, 18], [178, 120], [16, 118]],
                    }
                ]
            }
            (project_dir / "screen-pdf-project.json").write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")
            output_dir = root / "local-patch"

            export_local_corner_patch_dataset(
                dataset_root=root,
                output_dir=output_dir,
                seed=7,
                test_ratio=0.0,
                patch_size=64,
            )

            row = json.loads((output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(row["patch"]["x"], 0)
        self.assertEqual(row["patch"]["y"], 0)
        self.assertEqual(row["target_point_norm"], [18 / 64, 20 / 64])

    def test_build_local_corner_patch_sample_allows_large_residual_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            predicted_quad = np.array([[60, 40], [240, 42], [238, 160], [58, 158]], dtype=np.float32)
            manual_quad = np.array([[132, 50], [240, 42], [238, 160], [58, 158]], dtype=np.float32)

            sample = build_local_corner_patch_sample(
                image_path=image_path,
                page_id="page-1",
                corner_index=0,
                predicted_quad=predicted_quad,
                manual_quad=manual_quad,
                patch_size=96,
            )

        self.assertAlmostEqual(sample["target_residual_norm"][0], 0.75, places=4)
        self.assertAlmostEqual(sample["target_residual_norm"][1], 10 / 96, places=4)

    def test_build_local_corner_patch_sample_supports_dynamic_patch_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((800, 1200, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            predicted_quad = np.array([[100, 100], [1100, 120], [1080, 700], [120, 680]], dtype=np.float32)
            manual_quad = predicted_quad.copy()

            sample = build_local_corner_patch_sample(
                image_path=image_path,
                page_id="page-1",
                corner_index=0,
                predicted_quad=predicted_quad,
                manual_quad=manual_quad,
                patch_size=None,
                patch_scale=0.25,
                patch_min=96,
                patch_max=256,
            )

        self.assertEqual(sample["patch"]["size"], 150)

    def test_build_local_corner_patch_sample_biases_bottom_corner_patch_upward(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((800, 1200, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            predicted_quad = np.array([[100, 100], [1100, 120], [1080, 700], [120, 680]], dtype=np.float32)
            manual_quad = predicted_quad.copy()

            centered = build_local_corner_patch_sample(
                image_path=image_path,
                page_id="page-1",
                corner_index=2,
                predicted_quad=predicted_quad,
                manual_quad=manual_quad,
                patch_size=200,
            )
            shifted = build_local_corner_patch_sample(
                image_path=image_path,
                page_id="page-1",
                corner_index=2,
                predicted_quad=predicted_quad,
                manual_quad=manual_quad,
                patch_size=200,
                bottom_vertical_bias=0.2,
            )

        self.assertEqual(centered["patch"]["size"], 200)
        self.assertEqual(shifted["patch"]["size"], 200)
        self.assertEqual(centered["patch"]["x"], shifted["patch"]["x"])
        self.assertEqual(shifted["patch"]["y"], centered["patch"]["y"] - 40)

    def test_build_local_corner_patch_sample_can_expand_and_shift_bl_patch_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((900, 1400, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            predicted_quad = np.array([[120, 120], [1280, 140], [1240, 760], [160, 740]], dtype=np.float32)
            manual_quad = predicted_quad.copy()

            baseline = build_local_corner_patch_sample(
                image_path=image_path,
                page_id="page-1",
                corner_index=3,
                predicted_quad=predicted_quad,
                manual_quad=manual_quad,
                patch_size=None,
                patch_scale=0.2,
                patch_min=96,
                patch_max=256,
            )
            boosted = build_local_corner_patch_sample(
                image_path=image_path,
                page_id="page-1",
                corner_index=3,
                predicted_quad=predicted_quad,
                manual_quad=manual_quad,
                patch_size=None,
                patch_scale=0.2,
                patch_min=96,
                patch_max=256,
                bl_patch_scale_multiplier=1.25,
                bl_bottom_vertical_bias=0.12,
            )

        self.assertGreater(boosted["patch"]["size"], baseline["patch"]["size"])
        self.assertLess(boosted["patch"]["y"], baseline["patch"]["y"])

    def test_build_local_corner_patch_sample_can_reuse_preloaded_image(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        predicted_quad = np.array([[60, 40], [240, 42], [238, 160], [58, 158]], dtype=np.float32)
        manual_quad = np.array([[66, 46], [240, 42], [238, 160], [58, 158]], dtype=np.float32)

        sample = build_local_corner_patch_sample(
            image_path=Path("/tmp/non-existent.jpg"),
            image=image,
            page_id="page-1",
            corner_index=0,
            predicted_quad=predicted_quad,
            manual_quad=manual_quad,
            patch_size=64,
        )

        self.assertEqual(sample["patch"]["size"], 64)
        self.assertEqual(tuple(sample["patch_image"].shape[:2]), (64, 64))

    def test_export_local_corner_patch_dataset_supports_project_aware_focus_and_holdout_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            project_specs = [
                ("focus", "主项目", ["IMG_0001", "IMG_0002", "IMG_0003", "IMG_0004"]),
                ("broad", "泛化项目", ["IMG_0011", "IMG_0012", "IMG_0013", "IMG_0014"]),
                ("holdout", "留出项目", ["IMG_0021", "IMG_0022"]),
            ]
            for folder_name, project_name, page_ids in project_specs:
                project_dir = root / folder_name / project_name
                project_dir.mkdir(parents=True)
                pages = []
                for index, page_id in enumerate(page_ids):
                    image = np.zeros((120, 180, 3), dtype=np.uint8)
                    image_path = project_dir / f"{page_id}.jpg"
                    cv2.imwrite(str(image_path), image)
                    pages.append(
                        {
                            "id": page_id,
                            "path": str(image_path),
                            "activeQuad": [[20, 20], [160, 20], [160, 100], [20, 100]],
                            "manualQuad": [[24 + index, 24], [160, 20], [160, 100], [20, 100]],
                        }
                    )
                (project_dir / "screen-pdf-project.json").write_text(
                    json.dumps({"pages": pages}, ensure_ascii=False),
                    encoding="utf-8",
                )

            output_dir = root / "local-patch"
            summary = export_local_corner_patch_dataset(
                dataset_root=root,
                output_dir=output_dir,
                seed=7,
                test_ratio=0.25,
                focus_projects=["主项目"],
                holdout_projects=["留出项目"],
                focus_test_ratio=0.5,
                patch_size=64,
            )

            focus_rows = [json.loads(line) for line in (output_dir / "focus_test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            broad_rows = [json.loads(line) for line in (output_dir / "broad_test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            holdout_rows = [json.loads(line) for line in (output_dir / "holdout.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            train_rows = [json.loads(line) for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            test_rows = [json.loads(line) for line in (output_dir / "test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(summary["focus_test_pages"], 2)
        self.assertEqual(summary["holdout_pages"], 2)
        self.assertEqual(len(focus_rows), 8)
        self.assertEqual(len(broad_rows), 4)
        self.assertEqual(len(holdout_rows), 8)
        self.assertEqual(len(test_rows), 12)
        focus_page_ids = {row["page_id"] for row in focus_rows}
        broad_page_ids = {row["page_id"] for row in broad_rows}
        holdout_page_ids = {row["page_id"] for row in holdout_rows}
        train_page_ids = {row["page_id"] for row in train_rows}
        self.assertTrue(focus_page_ids.isdisjoint(train_page_ids))
        self.assertTrue(holdout_page_ids.isdisjoint(train_page_ids))
        self.assertEqual(len(focus_page_ids), 2)
        self.assertEqual(len(broad_page_ids), 1)
        self.assertTrue(focus_page_ids.issubset({"IMG_0001", "IMG_0002", "IMG_0003", "IMG_0004"}))
        self.assertTrue(broad_page_ids.issubset({"IMG_0011", "IMG_0012", "IMG_0013", "IMG_0014"}))


if __name__ == "__main__":
    unittest.main()
