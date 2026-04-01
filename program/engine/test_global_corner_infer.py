from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from global_corner_infer import _load_image_tensor, _load_model


class GlobalCornerInferTests(unittest.TestCase):
    def test_load_image_tensor_supports_rgb_gray_border_feature_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.jpg"
            image = np.full((48, 64, 3), 96, dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            tensor, image_size = _load_image_tensor(image_path, 32, feature_mode="rgb_gray_border")

        self.assertEqual(tuple(tensor.shape), (1, 5, 32, 32))
        self.assertEqual(image_size, (64, 48))
        self.assertGreater(float(tensor[0, 4, 0, 0]), 0.0)
        self.assertAlmostEqual(float(tensor[0, 4, 16, 16]), 0.0, places=6)

    def test_load_model_uses_checkpoint_input_channels(self) -> None:
        checkpoint = {
            "channels": 16,
            "input_channels": 5,
            "feature_mode": "rgb_gray_border",
            "state_dict": {},
        }
        captured: dict[str, int] = {}

        class FakeModel:
            def __init__(self, *, in_channels, channels, output_channels, head_mode="heatmap"):
                captured["in_channels"] = int(in_channels)
                captured["channels"] = int(channels)
                captured["output_channels"] = int(output_channels)

            def load_state_dict(self, state_dict):
                return self

            def to(self, device):
                return self

            def eval(self):
                return self

        with (
            mock.patch("global_corner_infer.torch.load", return_value=checkpoint),
            mock.patch("global_corner_infer.CornerHeatmapNet", FakeModel),
        ):
            model, loaded = _load_model(Path("/tmp/global_corner_model.pt"), torch.device("cpu"))

        self.assertIsInstance(model, FakeModel)
        self.assertEqual(captured["in_channels"], 5)
        self.assertEqual(loaded["feature_mode"], "rgb_gray_border")
