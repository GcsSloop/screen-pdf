from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from supervision_utils import (
    CURRENT_DATA_STRUCTURE_VERSION,
    resolve_data_structure_version,
    resolve_manual_quad,
    resolve_supervision_quad,
)


class SupervisionUtilsTests(unittest.TestCase):
    def test_resolve_manual_quad_prefers_selected_candidate_manual_quad(self) -> None:
        page = {
            "selectedCandidateIndex": 1,
            "manualQuad": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "candidates": [
                {"method": "document_quad", "quad": [[1, 1], [9, 1], [9, 9], [1, 9]]},
                {
                    "method": "teacher_current",
                    "quad": [[2, 2], [8, 2], [8, 8], [2, 8]],
                    "manualQuad": [[3, 3], [7, 3], [7, 7], [3, 7]],
                },
            ],
        }

        manual_quad, source = resolve_manual_quad(page)

        self.assertEqual(manual_quad, [[3, 3], [7, 3], [7, 7], [3, 7]])
        self.assertEqual(source, "selected_candidate_manual_quad")

    def test_resolve_supervision_quad_falls_back_to_legacy_page_manual_quad(self) -> None:
        page = {
            "manualQuad": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "activeQuad": [[1, 1], [9, 1], [9, 9], [1, 9]],
            "status": "reviewed",
            "selectedCandidateIndex": 0,
            "candidates": [{"method": "teacher_current", "source": "runtime_teacher"}],
        }

        manual_quad, source = resolve_supervision_quad(page)

        self.assertEqual(manual_quad, [[0, 0], [10, 0], [10, 10], [0, 10]])
        self.assertEqual(source, "manual_quad")

    def test_resolve_supervision_quad_uses_candidate_manual_quad_even_for_model_candidate(self) -> None:
        page = {
            "status": "reviewed",
            "activeQuad": [[2, 2], [8, 2], [8, 8], [2, 8]],
            "selectedCandidateIndex": 0,
            "candidates": [
                {
                    "method": "teacher_current",
                    "source": "runtime_teacher",
                    "modelId": "r66",
                    "manualQuad": [[1, 1], [9, 1], [9, 9], [1, 9]],
                }
            ],
        }

        supervision_quad, source = resolve_supervision_quad(page)

        self.assertEqual(supervision_quad, [[1, 1], [9, 1], [9, 9], [1, 9]])
        self.assertEqual(source, "selected_candidate_manual_quad")

    def test_resolve_data_structure_version_defaults_to_legacy(self) -> None:
        self.assertEqual(resolve_data_structure_version({}), 1)
        self.assertEqual(resolve_data_structure_version(None), 1)

    def test_resolve_data_structure_version_reads_current_version(self) -> None:
        self.assertEqual(
            resolve_data_structure_version({"dataStructureVersion": CURRENT_DATA_STRUCTURE_VERSION}),
            CURRENT_DATA_STRUCTURE_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
