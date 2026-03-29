from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from training.build_manual_scene_dataset import build_manual_scene_dataset


class BuildManualSceneDatasetTests(unittest.TestCase):
    def test_build_manual_scene_dataset_writes_clean_annotation_rows_and_page_level_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            project_root = temp_root / "source_project"
            project_root.mkdir()
            project_file = project_root / "screen-pdf-project.json"

            pages = []
            for index in range(20):
                page_id = f"IMG_{index:04d}"
                failure_tags: list[str] = []
                difficulty_bucket = None
                if index < 3:
                    failure_tags = ["corner_out_of_frame", "strong_perspective"]
                    difficulty_bucket = "hard"
                elif index < 8:
                    failure_tags = ["strong_perspective"]
                image_path = project_root / f"{page_id}.jpeg"
                image_path.write_bytes(b"stub")
                pages.append(
                    {
                        "id": page_id,
                        "name": f"{page_id}.jpeg",
                        "path": str(image_path),
                        "thumbPath": f"/tmp/{page_id}.thumb.jpg",
                        "previewPath": f"/tmp/{page_id}.preview.png",
                        "status": "reviewed",
                        "confidence": 0.08,
                        "bestMethod": "teacher_current",
                        "selectedCandidateIndex": 0,
                        "candidates": [{"method": "teacher_current"}],
                        "manualQuad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "activeQuad": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "details": {"width": 100, "height": 100},
                        "failureTags": failure_tags,
                        "difficultyBucket": difficulty_bucket,
                        "reviewTags": ["auto"] if failure_tags else [],
                    }
                )

            project_file.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "name": "2026-03-06 中交机电",
                        "sourceDir": str(project_root),
                        "projectPath": str(project_file),
                        "pages": pages,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            summary = build_manual_scene_dataset(
                project_file=project_file,
                dataset_slug="202603-zhongjiao-jidian",
                repo_root=temp_root,
                val_count=4,
                holdout_count=4,
            )

            self.assertEqual(summary["train_pages"], 12)
            self.assertEqual(summary["val_pages"], 4)
            self.assertEqual(summary["holdout_pages"], 4)

            annotation_path = temp_root / "data" / "curated" / "annotations" / "202603-zhongjiao-jidian_pages.jsonl"
            annotation_rows = [json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(annotation_rows), 20)
            self.assertNotIn("failureTags", annotation_rows[0])
            self.assertNotIn("difficultyBucket", annotation_rows[0])
            self.assertNotIn("reviewTags", annotation_rows[0])

            split_path = temp_root / "data" / "splits" / "cross_project" / "202603-zhongjiao-jidian_split_manual_only_v1.json"
            split_payload = json.loads(split_path.read_text(encoding="utf-8"))
            self.assertEqual(len(split_payload["train_page_ids"]), 12)
            self.assertEqual(len(split_payload["val_page_ids"]), 4)
            self.assertEqual(len(split_payload["holdout_page_ids"]), 4)

            registry_path = temp_root / "training" / "registry" / "202603-zhongjiao-jidian.json"
            registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(registry_payload["dataset_slug"], "202603-zhongjiao-jidian")
            self.assertEqual(registry_payload["split_file"], str(split_path))

    def test_build_manual_scene_dataset_uses_reviewed_non_model_active_quad_as_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            project_root = temp_root / "source_project"
            project_root.mkdir()
            project_file = project_root / "screen-pdf-project.json"

            pages = []
            for index in range(6):
                page_id = f"IMG_{index:04d}"
                image_path = project_root / f"{page_id}.jpeg"
                image_path.write_bytes(b"stub")
                pages.append(
                    {
                        "id": page_id,
                        "name": f"{page_id}.jpeg",
                        "path": str(image_path),
                        "status": "reviewed",
                        "selectedCandidateIndex": 0,
                        "candidates": [
                            {
                                "method": "document_quad",
                                "source": "opencv",
                                "quad": [[index, 0], [10 + index, 0], [10 + index, 10], [index, 10]],
                            }
                        ],
                        "manualQuad": None,
                        "activeQuad": [[index, 0], [10 + index, 0], [10 + index, 10], [index, 10]],
                    }
                )

            project_file.write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")

            summary = build_manual_scene_dataset(
                project_file=project_file,
                dataset_slug="202603-fuzhou-opencv-supervision",
                repo_root=temp_root,
                val_count=2,
                holdout_count=1,
            )

            self.assertEqual(summary["train_pages"], 3)
            self.assertEqual(summary["val_pages"], 2)
            self.assertEqual(summary["holdout_pages"], 1)
            self.assertEqual(summary["manual_pages"], 6)

            annotation_path = temp_root / "data" / "curated" / "annotations" / "202603-fuzhou-opencv-supervision_pages.jsonl"
            annotation_rows = [json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(annotation_rows[0]["supervision_source"], "accepted_active_quad")
            self.assertEqual(annotation_rows[0]["selected_candidate_source"], "opencv")
            self.assertEqual(annotation_rows[0]["selected_candidate_method"], "document_quad")
            self.assertEqual(annotation_rows[0]["manual_quad"], annotation_rows[0]["active_quad"])
            self.assertFalse(annotation_rows[0]["has_manual_quad"])

    def test_build_manual_scene_dataset_excludes_model_selected_active_quad_without_manual_quad(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            project_root = temp_root / "source_project"
            project_root.mkdir()
            project_file = project_root / "screen-pdf-project.json"

            pages = []
            for index in range(5):
                page_id = f"IMG_{index:04d}"
                image_path = project_root / f"{page_id}.jpeg"
                image_path.write_bytes(b"stub")
                pages.append(
                    {
                        "id": page_id,
                        "name": f"{page_id}.jpeg",
                        "path": str(image_path),
                        "status": "reviewed",
                        "selectedCandidateIndex": 0,
                        "candidates": [
                            {
                                "method": "teacher_current",
                                "source": "runtime_teacher",
                                "modelId": "r57e001",
                                "quad": [[index, 0], [10 + index, 0], [10 + index, 10], [index, 10]],
                            }
                        ],
                        "manualQuad": None,
                        "activeQuad": [[index, 0], [10 + index, 0], [10 + index, 10], [index, 10]],
                    }
                )

            project_file.write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "at least 2 manual pages"):
                build_manual_scene_dataset(
                    project_file=project_file,
                    dataset_slug="202603-model-only-active-quad",
                    repo_root=temp_root,
                    val_count=1,
                    holdout_count=1,
                )

    def test_build_manual_scene_dataset_uses_selected_candidate_manual_quad_in_new_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            project_root = temp_root / "source_project"
            project_root.mkdir()
            project_file = project_root / "screen-pdf-project.json"

            pages = []
            for index in range(4):
                page_id = f"IMG_{index:04d}"
                image_path = project_root / f"{page_id}.jpeg"
                image_path.write_bytes(b"stub")
                pages.append(
                    {
                        "id": page_id,
                        "name": f"{page_id}.jpeg",
                        "path": str(image_path),
                        "status": "reviewed",
                        "selectedCandidateIndex": 0,
                        "manualQuad": None,
                        "activeQuad": [[index, 0], [10 + index, 0], [10 + index, 10], [index, 10]],
                        "candidates": [
                            {
                                "method": "teacher_current",
                                "source": "runtime_teacher",
                                "modelId": "r66",
                                "quad": [[index, 0], [10 + index, 0], [10 + index, 10], [index, 10]],
                                "manualQuad": [[index + 1, 1], [10 + index, 1], [10 + index, 10], [index + 1, 10]],
                            }
                        ],
                    }
                )

            project_file.write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=2), encoding="utf-8")

            summary = build_manual_scene_dataset(
                project_file=project_file,
                dataset_slug="202603-new-structure-manual",
                repo_root=temp_root,
                val_count=1,
                holdout_count=1,
            )

            self.assertEqual(summary["manual_pages"], 4)
            annotation_path = temp_root / "data" / "curated" / "annotations" / "202603-new-structure-manual_pages.jsonl"
            annotation_rows = [json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(annotation_rows[0]["supervision_source"], "selected_candidate_manual_quad")
            self.assertEqual(annotation_rows[0]["manual_source"], "selected_candidate_manual_quad")
            self.assertTrue(annotation_rows[0]["has_manual_quad"])
            self.assertEqual(annotation_rows[0]["manual_quad"], [[1, 1], [10, 1], [10, 10], [1, 10]])


if __name__ == "__main__":
    unittest.main()
