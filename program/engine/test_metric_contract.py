from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_benchmark import checkpoint_selection_key, normalized_quad_geometry_metrics, quad_geometry_metrics, summarize_geometry_metric_rows, summarize_point_hits
from global_corner_train import _evaluate
from analyze_runtime_failure_layers import classify_failure_layer
from two_stage_corner_pipeline import predict_two_stage
from balanced_generalization_eval import infer_runtime_rows, evaluate_runtime_pages


class MetricContractTests(unittest.TestCase):
    def test_point_only_ties_ignore_every_other_measure(self):
        self.assertEqual(checkpoint_selection_key({'point_le_0_01_ratio': .75, 'quad_le_0_01_ratio': 0}),
                         checkpoint_selection_key({'point_le_0_01_ratio': .75, 'quad_le_0_01_ratio': 1, 'point_error_mean': 0}))

    def test_point_report_has_only_one_accuracy_measure_and_counts_failures(self):
        manual = [[0, 0], [600, 0], [600, 400], [0, 400]]
        pred = np.array(manual, dtype=float)
        pred[2, 1] += 10
        result = summarize_point_hits([manual]*3, [pred, None, np.full((4, 2), np.nan)])
        self.assertEqual(result, {'pages': 3, 'corners': 12, 'hit_corners': 3, 'point_le_0_01_ratio': .25})

    def test_checkpoint_selection_prioritizes_strict_hits_over_mean_error(self):
        sharp = {'point_le_0_01': .8, 'quad_le_0_01': .5, 'point_error_mean': .02, 'max_corner_error_mean': .04}
        smooth = {**sharp, 'point_le_0_01': .7, 'point_error_mean': .01}
        self.assertLess(checkpoint_selection_key(sharp), checkpoint_selection_key(smooth))

    def test_normalized_validation_matches_pixel_runtime_for_non_square_image(self):
        size = np.array([1600, 900], dtype=np.float32)
        manual = np.array([[200, 100], [1400, 120], [1380, 800], [220, 780]], dtype=np.float32)
        predicted = manual + np.array([[0, 0], [0, 0], [20, 0], [0, 0]])
        pixel = quad_geometry_metrics(manual, predicted)
        restored = normalized_quad_geometry_metrics(manual / size, predicted / size, size)
        for key in pixel:
            self.assertAlmostEqual(pixel[key], restored[key], places=4, msg=key)

    def test_single_corner_and_all_four_use_identical_scale(self):
        manual = np.array([[0, 0], [600, 0], [600, 400], [0, 400]])
        predicted = manual.copy()
        predicted[2, 1] += 10
        summary = summarize_geometry_metric_rows([quad_geometry_metrics(manual, predicted)])
        self.assertEqual(summary['point_le_0_01_ratio'], .75)
        self.assertEqual(summary['quad_le_0_01_ratio'], 0)
        self.assertEqual(summary['point_error_px_mean'], 2.5)

    def test_training_evaluator_restores_original_size(self):
        corners = torch.tensor([[[.2, .3], [.8, .3], [.8, .7], [.2, .7]]])
        predicted = corners.clone()
        predicted[0, 2, 1] += .01
        from unittest.mock import patch
        class Model(torch.nn.Module):
            def forward(self, image):
                return torch.zeros(1, 4, 4, 4)
        batch = {'image': torch.zeros(1, 3, 16, 16), 'heatmaps': torch.zeros(1, 4, 4, 4),
                 'corners': corners, 'image_size': torch.tensor([[1000, 1000]])}
        with patch('global_corner_train.decode_heatmaps', return_value=predicted):
            metrics = _evaluate(Model(), [batch], torch.device('cpu'))
        self.assertEqual(metrics['point_le_0_01'], .75)
        self.assertEqual(metrics['quad_le_0_01'], 0)

    def test_invalid_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            normalized_quad_geometry_metrics(np.zeros((4, 2)), np.zeros((4, 2)), [0, 900])

    def test_zero_error_is_not_classified_as_a_failure(self):
        perfect = {'point_error': 0., 'max_corner_error': 0.}
        self.assertEqual(classify_failure_layer(baseline_metrics=perfect, runtime_oracle_metrics=perfect,
                                               opencv_oracle_metrics=perfect), 'baseline_ok')

    def test_missing_image_counts_as_failure_instead_of_disappearing(self):
        from unittest.mock import patch
        pages = [{'image_path': '/missing.jpg', 'manual_quad': [[0, 0], [100, 0], [100, 80], [0, 80]]}]
        with patch('balanced_generalization_eval.cv2.imread', return_value=None):
            rows = infer_runtime_rows(pages, lambda *args, **kwargs: None)
        summary = evaluate_runtime_pages(pages, rows)['summary']
        self.assertEqual(summary['pages'], 1)
        self.assertEqual(summary['point_le_0_01_ratio'], 0.)
        self.assertEqual(summary['quad_le_0_01_ratio'], 0.)
        self.assertEqual(summary['pixel_metric_pages'], 0)

    def test_diagnostics_do_not_change_runtime_output(self):
        image = np.zeros((120, 200, 3), dtype=np.uint8)
        coarse = lambda _: np.array([[20, 20], [180, 20], [180, 100], [20, 100]])
        roi = lambda _: np.array([[.1, .1], [.9, .1], [.9, .9], [.1, .9]])
        plain = predict_two_stage(Path('/unused.jpg'), coarse, roi, image=image)
        diagnostic = predict_two_stage(Path('/unused.jpg'), coarse, roi, image=image, return_diagnostics=True)
        self.assertEqual(plain, {k: v for k, v in diagnostic.items() if k != 'diagnostics'})
        np.testing.assert_allclose(diagnostic['diagnostics']['baseline_local_quad'], diagnostic['final_quad'], atol=1e-4)


if __name__ == '__main__':
    unittest.main()
