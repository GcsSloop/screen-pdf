from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from balanced_generalization_eval import build_runtime_runner


class BalancedGeneralizationEvalTests(unittest.TestCase):
    def test_build_runtime_runner_passes_selector_and_expand_config_to_pipeline(self) -> None:
        fake_selector = object()
        fake_global = object()
        fake_roi = object()
        fake_local = object()

        with tempfile.TemporaryDirectory() as temp_dir:
            selector_path = Path(temp_dir) / "candidate_expand_selector.json"
            selector_path.write_text("{}", encoding="utf-8")

            with (
                mock.patch("balanced_generalization_eval.GlobalCornerPredictor", return_value=fake_global),
                mock.patch("balanced_generalization_eval.RoiCornerPredictor", return_value=fake_roi),
                mock.patch("balanced_generalization_eval.LocalCornerMoEPredictor", return_value=fake_local),
                mock.patch(
                    "balanced_generalization_eval.LinearCandidateExpandSelector.from_json",
                    return_value=fake_selector,
                ) as selector_loader,
                mock.patch(
                    "balanced_generalization_eval.predict_two_stage",
                    return_value={"final_quad": [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]},
                ) as predict_mock,
            ):
                runner = build_runtime_runner(
                    global_model=Path("/tmp/global.pt"),
                    roi_model=Path("/tmp/roi.pt"),
                    local_model=Path("/tmp/local.pt"),
                    candidate_selector_path=selector_path,
                    candidate_expand_ratios=[0.04, 0.08, 0.12],
                    candidate_baseline_gate=0.41,
                    candidate_min_score_gain=0.02,
                )
                result = runner("/tmp/page.jpg", image=np.zeros((80, 120, 3), dtype=np.uint8))

        selector_loader.assert_called_once_with(selector_path)
        predict_mock.assert_called_once()
        self.assertIs(predict_mock.call_args.kwargs["candidate_selector"], fake_selector)
        self.assertEqual(predict_mock.call_args.kwargs["candidate_expand_ratios"], [0.04, 0.08, 0.12])
        self.assertEqual(predict_mock.call_args.kwargs["candidate_baseline_gate"], 0.41)
        self.assertEqual(predict_mock.call_args.kwargs["candidate_min_score_gain"], 0.02)
        self.assertEqual(result["method"], "custom_model_three_stage")
        self.assertEqual(result["quad"], [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
