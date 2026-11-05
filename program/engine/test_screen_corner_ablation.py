import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / 'training'))
from corner_train import CornerHeatmapNet, decode_heatmaps
from screen_corner_ablation import ExperimentPredictor, local_peak_decode
from two_stage_corner_pipeline import GlobalCornerPredictor


class ScreenCornerAblationTests(unittest.TestCase):
    def test_local_decode_does_not_average_two_distant_modes(self):
        logits = torch.zeros(1, 1, 32, 32)
        logits[0, 0, 4, 4] = 10.
        logits[0, 0, 26, 26] = 10.
        self.assertGreater(float(decode_heatmaps(logits)[0, 0, 0]), .4)
        self.assertLess(float(local_peak_decode(logits)[0, 0, 0]), .2)

    def test_experimental_legacy_checkpoint_has_runtime_prediction_parity(self):
        model = CornerHeatmapNet(in_channels=5, channels=24, output_channels=4)
        checkpoint = {'state_dict': model.state_dict(), 'channels': 24, 'input_channels': 5,
                      'input_size': 256, 'feature_mode': 'rgb_gray_border', 'decode_mode': 'soft_argmax'}
        image = np.random.default_rng(7).integers(0, 256, (90, 160, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / 'legacy.pt'
            experimental = Path(directory) / 'experimental.pt'
            torch.save(checkpoint, legacy)
            torch.save({**checkpoint, 'condition': 'legacy_new'}, experimental)
            expected = GlobalCornerPredictor(legacy).predict_image(image)
            actual = ExperimentPredictor(experimental).predict_image(image)
        np.testing.assert_allclose(expected, actual, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
