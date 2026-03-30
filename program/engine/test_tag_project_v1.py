import json
import tempfile
import unittest
from pathlib import Path

from tag_project_v1 import tag_project, write_tagged_project


def sample_project() -> dict:
    return {
        "version": 1,
        "name": "Demo Talk",
        "sourceDir": "/tmp/demo",
        "projectPath": "/tmp/demo/screen-pdf-project.json",
        "selectedPageId": "page-1",
        "pages": [
            {
                "id": "page-1",
                "name": "page-1.jpg",
                "path": "/tmp/demo/page-1.jpg",
                "createdAt": "2026-03-26 10:00:00",
                "status": "reviewed",
                "confidence": 0.91,
                "bestMethod": "contour_quad",
                "selectedCandidateIndex": 0,
                "activeQuad": [[10, 20], [990, 22], [988, 760], [8, 758]],
                "manualQuad": None,
                "previewPath": None,
                "details": {
                    "width": 1000,
                    "height": 800,
                    "fileSizeBytes": 1024,
                    "capturedAt": None,
                    "createdAt": "2026-03-26 10:00:00",
                    "modifiedAt": "2026-03-26 10:00:00",
                },
                "candidates": [
                    {
                        "method": "contour_quad",
                        "score": 0.81,
                        "quad": [[10, 20], [990, 22], [988, 760], [8, 758]],
                        "metrics": {
                            "spill_penalty": 0.0,
                            "blue_penalty": 0.02,
                            "parallel_score": 0.93,
                            "edge_score": 0.95,
                            "coverage_score": 0.52,
                        },
                    },
                    {
                        "method": "hough_screen",
                        "score": 0.8,
                        "quad": [[13, 22], [991, 20], [987, 759], [10, 760]],
                        "metrics": {},
                    },
                ],
            },
            {
                "id": "page-2",
                "name": "page-2.jpg",
                "path": "/tmp/demo/page-2.jpg",
                "createdAt": "2026-03-26 10:01:00",
                "status": "needs_review",
                "confidence": 0.28,
                "bestMethod": "contour_quad",
                "selectedCandidateIndex": 0,
                "activeQuad": [[-30, 10], [1020, 30], [960, 810], [4, 792]],
                "manualQuad": None,
                "previewPath": None,
                "details": {
                    "width": 1000,
                    "height": 800,
                    "fileSizeBytes": 1024,
                    "capturedAt": None,
                    "createdAt": "2026-03-26 10:01:00",
                    "modifiedAt": "2026-03-26 10:01:00",
                },
                "candidates": [
                    {
                        "method": "contour_quad",
                        "score": 0.55,
                        "quad": [[-30, 10], [1020, 30], [960, 810], [4, 792]],
                        "metrics": {
                            "spill_penalty": 0.22,
                            "blue_penalty": 0.4,
                            "parallel_score": 0.58,
                            "edge_score": 0.36,
                            "coverage_score": 0.04,
                        },
                    },
                    {
                        "method": "hough_screen",
                        "score": 0.44,
                        "quad": [[70, 85], [920, 66], [930, 700], [60, 720]],
                        "metrics": {},
                    },
                ],
            },
        ],
    }


class TagProjectV1Test(unittest.TestCase):
    def test_tag_project_adds_event_and_page_bucket_fields(self) -> None:
        project_dir = Path(
            "/tmp/202603 中国智慧道路照明大会/2026-03-19 Demo Talk/Demo Talk"
        )
        tagged = tag_project(project_dir / "screen-pdf-project.json", sample_project())

        self.assertEqual(tagged["eventSlug"], "2026-03-19-demo-talk")
        self.assertEqual(tagged["eventName"], "2026-03-19 Demo Talk")
        self.assertEqual(tagged["tagVersion"], 1)
        self.assertIn("tagSummary", tagged)

        clean_page = tagged["pages"][0]
        self.assertEqual(clean_page["difficultyBucket"], "clean")
        self.assertEqual(clean_page["failureTags"], [])
        self.assertEqual(clean_page["eventSlug"], "2026-03-19-demo-talk")

        abnormal_page = tagged["pages"][1]
        self.assertEqual(abnormal_page["difficultyBucket"], "abnormal")
        self.assertIn("corner_out_of_frame", abnormal_page["failureTags"])
        self.assertIn("large_spill", abnormal_page["failureTags"])
        self.assertIn("candidate_disagreement", abnormal_page["failureTags"])
        self.assertEqual(abnormal_page["reviewTags"], ["auto"])

    def test_write_tagged_project_writes_parallel_v1_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_dir = Path(temp_dir) / "2026-03-19 Demo Talk" / "Demo Talk"
            project_dir.mkdir(parents=True)
            source_path = project_dir / "screen-pdf-project.json"
            source_path.write_text(json.dumps(sample_project(), ensure_ascii=False), encoding="utf-8")

            target_path = write_tagged_project(source_path)

            self.assertEqual(target_path.name, "screen-pdf-project_v1.json")
            self.assertTrue(target_path.exists())
            payload = json.loads(target_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["projectPath"], str(target_path))
            self.assertEqual(payload["pages"][0]["tagVersion"], 1)


if __name__ == "__main__":
    unittest.main()
