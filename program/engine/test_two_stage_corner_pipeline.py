from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from two_stage_corner_pipeline import (
    GlobalCornerPredictor,
    LinearCandidateExpandSelector,
    LocalCornerMoEPredictor,
    apply_roi_prediction,
    build_refine_request,
    export_refine_dataset_from_global_predictions,
    predict_two_stage,
)


class TwoStageCornerPipelineTests(unittest.TestCase):
    def test_global_corner_predictor_uses_checkpoint_feature_mode_and_input_channels(self) -> None:
        checkpoint = {
            "channels": 24,
            "input_size": 96,
            "input_channels": 5,
            "feature_mode": "rgb_gray_border",
            "decode_mode": "argmax",
            "head_mode": "heatmap",
            "state_dict": {},
        }
        model_args: dict[str, int] = {}
        feature_calls: list[str] = []

        class FakeModel:
            def __init__(self, *, in_channels, channels, output_channels, head_mode="heatmap"):
                model_args["in_channels"] = int(in_channels)
                model_args["channels"] = int(channels)
                model_args["output_channels"] = int(output_channels)
                model_args["head_mode"] = str(head_mode)

            def load_state_dict(self, state_dict, strict=False):
                return self

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, tensor):
                self.last_shape = tuple(tensor.shape)
                return torch.zeros((1, 4, 24, 24), dtype=torch.float32)

        def fake_build_global_feature_tensor(image_rgb, feature_mode="rgb"):
            feature_calls.append(str(feature_mode))
            return np.zeros((5, 96, 96), dtype=np.float32)

        with (
            mock.patch("two_stage_corner_pipeline.select_torch_device", return_value=torch.device("cpu")),
            mock.patch("two_stage_corner_pipeline.torch.load", return_value=checkpoint),
            mock.patch("two_stage_corner_pipeline.CornerHeatmapNet", FakeModel),
            mock.patch("two_stage_corner_pipeline.remap_legacy_head_state_dict", side_effect=lambda state_dict: state_dict),
            mock.patch("two_stage_corner_pipeline.build_global_feature_tensor", side_effect=fake_build_global_feature_tensor),
            mock.patch(
                "two_stage_corner_pipeline.decode_model_output",
                return_value=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
            ),
        ):
            predictor = GlobalCornerPredictor(Path("/tmp/global_corner_model.pt"))
            quad = predictor.predict_image(np.zeros((120, 200, 3), dtype=np.uint8))

        self.assertEqual(model_args["in_channels"], 5)
        self.assertEqual(predictor.input_channels, 5)
        self.assertEqual(predictor.feature_mode, "rgb_gray_border")
        self.assertEqual(feature_calls, ["rgb_gray_border"])
        np.testing.assert_allclose(
            quad,
            np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 120.0], [0.0, 120.0]], dtype=np.float32),
            atol=1e-5,
        )

    def test_local_corner_predictor_prefers_coord_model_checkpoint(self) -> None:
        checkpoint = {
            "channels": 24,
            "experts": 4,
            "metadata_dim": 14,
            "input_size": 96,
            "coord_mix": 0.25,
            "state_dict": {},
        }

        class FakeModel:
            def __init__(self, *args, **kwargs):
                self.loaded = None

            def load_state_dict(self, state_dict, strict=False):
                self.loaded = (state_dict, strict)
                return ([], [])

            def to(self, device):
                return self

            def eval(self):
                return self

        with (
            mock.patch("two_stage_corner_pipeline.select_torch_device", return_value=torch.device("cpu")),
            mock.patch("two_stage_corner_pipeline.torch.load", return_value=checkpoint),
            mock.patch("local_corner_moe_coord.LocalCornerMoECoordNet", FakeModel),
        ):
            predictor = LocalCornerMoEPredictor(Path("/tmp/local_corner_moe_coord_model.pt"))

        self.assertEqual(predictor.coord_mix, 0.25)

    def test_local_corner_predictor_uses_checkpoint_input_channels_for_coord_model_features(self) -> None:
        checkpoint = {
            "channels": 24,
            "experts": 4,
            "metadata_dim": 14,
            "input_size": 96,
            "input_channels": 13,
            "coord_mix": 0.15,
            "state_dict": {"visibility_heads.0.weight": torch.zeros(1)},
        }

        class FakeModel:
            def load_state_dict(self, state_dict, strict=False):
                return ([], [])

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, tensor, metadata):
                heatmaps = torch.zeros((1, 1, 24, 24), dtype=torch.float32)
                offsets = torch.zeros((1, 1, 2, 24, 24), dtype=torch.float32)
                coord_head = torch.full((1, 2), 0.5, dtype=torch.float32)
                visibility = torch.full((1, 2), 0.5, dtype=torch.float32)
                gates = torch.ones((1, 4), dtype=torch.float32) / 4.0
                edge = torch.zeros((1, 5), dtype=torch.float32)
                edge_map = torch.zeros((1, 2, 24, 24), dtype=torch.float32)
                return heatmaps, offsets, coord_head, edge, edge_map, visibility, gates

        patch_feature_calls: list[int] = []

        def fake_build_patch_features(patch_image, corner_index, input_size=96, input_channels=10):
            patch_feature_calls.append(int(input_channels))
            return np.zeros((input_channels, input_size, input_size), dtype=np.float32)

        with (
            mock.patch("two_stage_corner_pipeline.select_torch_device", return_value=torch.device("cpu")),
            mock.patch("two_stage_corner_pipeline.torch.load", return_value=checkpoint),
            mock.patch("local_corner_moe_coord.LocalCornerMoECoordNet", return_value=FakeModel()),
            mock.patch("local_corner_heatmap.build_patch_metadata", return_value=np.zeros((14,), dtype=np.float32)),
            mock.patch(
                "local_corner_refine.build_local_corner_patch_sample",
                return_value={
                    "patch_image": np.zeros((96, 96, 3), dtype=np.uint8),
                    "patch": {"x": 0, "y": 0, "size": 96},
                    "predicted_point": [48.0, 48.0],
                },
            ),
            mock.patch("local_corner_refine.build_patch_features", side_effect=fake_build_patch_features),
            mock.patch("two_stage_corner_pipeline.order_points", side_effect=lambda quad: np.array(quad, dtype=np.float32)),
            mock.patch(
                "local_corner_moe_coord.decode_moe_coord_output",
                return_value=torch.full((1, 2), 0.5, dtype=torch.float32),
            ),
        ):
            predictor = LocalCornerMoEPredictor(Path("/tmp/local_corner_moe_coord_model.pt"))
            predictor(Path("/tmp/fake.jpg"), np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32), image=np.zeros((120, 120, 3), dtype=np.uint8))

        self.assertEqual(predictor.input_channels, 13)
        self.assertEqual(patch_feature_calls, [13, 13, 13, 13])

    def test_local_corner_predictor_uses_checkpoint_local_patch_config(self) -> None:
        checkpoint = {
            "channels": 24,
            "experts": 4,
            "metadata_dim": 14,
            "input_size": 96,
            "input_channels": 10,
            "coord_mix": 0.15,
            "state_dict": {"visibility_heads.0.weight": torch.zeros(1)},
            "local_patch_config": {
                "patch_scale": 0.22,
                "patch_min": 112,
                "patch_max": 320,
                "bottom_vertical_bias": 0.04,
                "bl_patch_scale_multiplier": 1.1,
                "bl_bottom_vertical_bias": 0.08,
            },
        }

        class FakeModel:
            def load_state_dict(self, state_dict, strict=False):
                return ([], [])

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, tensor, metadata):
                heatmaps = torch.zeros((1, 1, 24, 24), dtype=torch.float32)
                offsets = torch.zeros((1, 1, 2, 24, 24), dtype=torch.float32)
                coord_head = torch.full((1, 2), 0.5, dtype=torch.float32)
                visibility = torch.full((1, 2), 0.5, dtype=torch.float32)
                gates = torch.ones((1, 4), dtype=torch.float32) / 4.0
                edge = torch.zeros((1, 5), dtype=torch.float32)
                edge_map = torch.zeros((1, 2, 24, 24), dtype=torch.float32)
                return heatmaps, offsets, coord_head, edge, edge_map, visibility, gates

        patch_calls: list[dict[str, float | int | None]] = []

        def fake_build_local_corner_patch_sample(**kwargs):
            patch_calls.append(
                {
                    "patch_size": kwargs.get("patch_size"),
                    "patch_scale": kwargs.get("patch_scale"),
                    "patch_min": kwargs.get("patch_min"),
                    "patch_max": kwargs.get("patch_max"),
                    "bottom_vertical_bias": kwargs.get("bottom_vertical_bias"),
                    "bl_patch_scale_multiplier": kwargs.get("bl_patch_scale_multiplier"),
                    "bl_bottom_vertical_bias": kwargs.get("bl_bottom_vertical_bias"),
                }
            )
            return {
                "patch_image": np.zeros((96, 96, 3), dtype=np.uint8),
                "patch": {"x": 0, "y": 0, "size": 96},
                "predicted_point": [48.0, 48.0],
            }

        with (
            mock.patch("two_stage_corner_pipeline.select_torch_device", return_value=torch.device("cpu")),
            mock.patch("two_stage_corner_pipeline.torch.load", return_value=checkpoint),
            mock.patch("local_corner_moe_coord.LocalCornerMoECoordNet", return_value=FakeModel()),
            mock.patch("local_corner_heatmap.build_patch_metadata", return_value=np.zeros((14,), dtype=np.float32)),
            mock.patch("local_corner_refine.build_local_corner_patch_sample", side_effect=fake_build_local_corner_patch_sample),
            mock.patch("local_corner_refine.build_patch_features", return_value=np.zeros((10, 96, 96), dtype=np.float32)),
            mock.patch("two_stage_corner_pipeline.order_points", side_effect=lambda quad: np.array(quad, dtype=np.float32)),
            mock.patch(
                "local_corner_moe_coord.decode_moe_coord_output",
                return_value=torch.full((1, 2), 0.5, dtype=torch.float32),
            ),
        ):
            predictor = LocalCornerMoEPredictor(Path("/tmp/local_corner_moe_coord_model.pt"))
            predictor(
                Path("/tmp/fake.jpg"),
                np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
                image=np.zeros((120, 120, 3), dtype=np.uint8),
            )

        self.assertEqual(
            predictor.local_patch_config,
            {
                "patch_scale": 0.22,
                "patch_min": 112,
                "patch_max": 320,
                "bottom_vertical_bias": 0.04,
                "bl_patch_scale_multiplier": 1.1,
                "bl_bottom_vertical_bias": 0.08,
            },
        )
        self.assertEqual(len(patch_calls), 4)
        self.assertTrue(all(call["patch_size"] is None for call in patch_calls))
        self.assertTrue(all(call["patch_scale"] == 0.22 for call in patch_calls))
        self.assertTrue(all(call["patch_min"] == 112 for call in patch_calls))
        self.assertTrue(all(call["patch_max"] == 320 for call in patch_calls))
        self.assertTrue(all(call["bottom_vertical_bias"] == 0.04 for call in patch_calls))
        self.assertTrue(all(call["bl_patch_scale_multiplier"] == 1.1 for call in patch_calls))
        self.assertTrue(all(call["bl_bottom_vertical_bias"] == 0.08 for call in patch_calls))

    def test_local_corner_predictor_blends_low_trust_corners_toward_roi(self) -> None:
        checkpoint = {
            "channels": 24,
            "experts": 4,
            "metadata_dim": 14,
            "input_size": 96,
            "input_channels": 10,
            "coord_mix": 0.15,
            "state_dict": {"visibility_heads.0.weight": torch.zeros(1)},
            "local_blend_config": {
                "enabled": True,
                "visibility_scale": 1.0,
                "visibility_pow": 1.0,
                "gate_pow": 0.0,
                "displacement_weight": 0.0,
                "max_trust": 1.0,
                "min_trust": 0.0,
            },
        }

        class FakeModel:
            def load_state_dict(self, state_dict, strict=False):
                return ([], [])

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, tensor, metadata):
                heatmaps = torch.zeros((1, 1, 24, 24), dtype=torch.float32)
                offsets = torch.zeros((1, 1, 2, 24, 24), dtype=torch.float32)
                coord_head = torch.full((1, 2), 0.9, dtype=torch.float32)
                visibility = torch.full((1, 2), 0.2, dtype=torch.float32)
                gates = torch.ones((1, 4), dtype=torch.float32) / 4.0
                edge = torch.zeros((1, 5), dtype=torch.float32)
                edge_map = torch.zeros((1, 2, 24, 24), dtype=torch.float32)
                return heatmaps, offsets, coord_head, edge, edge_map, visibility, gates

        with (
            mock.patch("two_stage_corner_pipeline.select_torch_device", return_value=torch.device("cpu")),
            mock.patch("two_stage_corner_pipeline.torch.load", return_value=checkpoint),
            mock.patch("local_corner_moe_coord.LocalCornerMoECoordNet", return_value=FakeModel()),
            mock.patch("local_corner_heatmap.build_patch_metadata", return_value=np.zeros((14,), dtype=np.float32)),
            mock.patch(
                "local_corner_refine.build_local_corner_patch_sample",
                return_value={
                    "patch_image": np.zeros((96, 96, 3), dtype=np.uint8),
                    "patch": {"x": 0, "y": 0, "size": 100},
                    "predicted_point": [50.0, 50.0],
                },
            ),
            mock.patch("local_corner_refine.build_patch_features", return_value=np.zeros((10, 96, 96), dtype=np.float32)),
            mock.patch("two_stage_corner_pipeline.order_points", side_effect=lambda quad: np.array(quad, dtype=np.float32)),
            mock.patch(
                "local_corner_moe_coord.decode_moe_coord_output",
                return_value=torch.full((1, 2), 0.9, dtype=torch.float32),
            ),
        ):
            predictor = LocalCornerMoEPredictor(Path("/tmp/local_corner_moe_coord_model.pt"))
            result = predictor.predict_with_details(
                Path("/tmp/fake.jpg"),
                np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]], dtype=np.float32),
                image=np.zeros((120, 120, 3), dtype=np.uint8),
            )

        expected = np.array([[26.0, 26.0], [34.0, 26.0], [34.0, 34.0], [26.0, 34.0]], dtype=np.float32)
        np.testing.assert_allclose(result["quad"], expected, atol=1e-4)
        self.assertTrue(all(abs(item["trust"] - 0.2) < 1e-6 for item in result["corner_details"]))

    def test_local_corner_predictor_can_disable_trust_blend(self) -> None:
        checkpoint = {
            "channels": 24,
            "experts": 4,
            "metadata_dim": 14,
            "input_size": 96,
            "input_channels": 10,
            "coord_mix": 0.15,
            "state_dict": {"visibility_heads.0.weight": torch.zeros(1)},
            "local_blend_config": {"enabled": False},
        }

        class FakeModel:
            def load_state_dict(self, state_dict, strict=False):
                return ([], [])

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, tensor, metadata):
                heatmaps = torch.zeros((1, 1, 24, 24), dtype=torch.float32)
                offsets = torch.zeros((1, 1, 2, 24, 24), dtype=torch.float32)
                coord_head = torch.full((1, 2), 0.9, dtype=torch.float32)
                visibility = torch.full((1, 2), 0.1, dtype=torch.float32)
                gates = torch.ones((1, 4), dtype=torch.float32) / 4.0
                edge = torch.zeros((1, 5), dtype=torch.float32)
                edge_map = torch.zeros((1, 2, 24, 24), dtype=torch.float32)
                return heatmaps, offsets, coord_head, edge, edge_map, visibility, gates

        with (
            mock.patch("two_stage_corner_pipeline.select_torch_device", return_value=torch.device("cpu")),
            mock.patch("two_stage_corner_pipeline.torch.load", return_value=checkpoint),
            mock.patch("local_corner_moe_coord.LocalCornerMoECoordNet", return_value=FakeModel()),
            mock.patch("local_corner_heatmap.build_patch_metadata", return_value=np.zeros((14,), dtype=np.float32)),
            mock.patch(
                "local_corner_refine.build_local_corner_patch_sample",
                return_value={
                    "patch_image": np.zeros((96, 96, 3), dtype=np.uint8),
                    "patch": {"x": 0, "y": 0, "size": 100},
                    "predicted_point": [50.0, 50.0],
                },
            ),
            mock.patch("local_corner_refine.build_patch_features", return_value=np.zeros((10, 96, 96), dtype=np.float32)),
            mock.patch("two_stage_corner_pipeline.order_points", side_effect=lambda quad: np.array(quad, dtype=np.float32)),
            mock.patch(
                "local_corner_moe_coord.decode_moe_coord_output",
                return_value=torch.full((1, 2), 0.9, dtype=torch.float32),
            ),
        ):
            predictor = LocalCornerMoEPredictor(Path("/tmp/local_corner_moe_coord_model.pt"))
            result = predictor.predict_with_details(
                Path("/tmp/fake.jpg"),
                np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]], dtype=np.float32),
                image=np.zeros((120, 120, 3), dtype=np.uint8),
            )

        expected = np.array([[90.0, 90.0], [90.0, 90.0], [90.0, 90.0], [90.0, 90.0]], dtype=np.float32)
        np.testing.assert_allclose(result["quad"], expected, atol=1e-4)
        self.assertTrue(all(abs(item["trust"] - 1.0) < 1e-6 for item in result["corner_details"]))

    def test_local_corner_predictor_can_fallback_entire_page_to_roi_when_visibility_too_low(self) -> None:
        checkpoint = {
            "channels": 24,
            "experts": 4,
            "metadata_dim": 14,
            "input_size": 96,
            "input_channels": 10,
            "coord_mix": 0.15,
            "state_dict": {"visibility_heads.0.weight": torch.zeros(1)},
            "local_blend_config": {
                "enabled": True,
                "visibility_scale": 1.0,
                "visibility_pow": 1.0,
                "gate_pow": 0.0,
                "displacement_weight": 0.0,
                "max_trust": 1.0,
                "min_trust": 0.0,
                "page_fallback_visibility_min": 0.2,
            },
        }

        class FakeModel:
            def load_state_dict(self, state_dict, strict=False):
                return ([], [])

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, tensor, metadata):
                heatmaps = torch.zeros((1, 1, 24, 24), dtype=torch.float32)
                offsets = torch.zeros((1, 1, 2, 24, 24), dtype=torch.float32)
                coord_head = torch.full((1, 2), 0.9, dtype=torch.float32)
                visibility = torch.full((1, 2), 0.1, dtype=torch.float32)
                gates = torch.ones((1, 4), dtype=torch.float32) / 4.0
                edge = torch.zeros((1, 5), dtype=torch.float32)
                edge_map = torch.zeros((1, 2, 24, 24), dtype=torch.float32)
                return heatmaps, offsets, coord_head, edge, edge_map, visibility, gates

        roi_quad = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]], dtype=np.float32)

        with (
            mock.patch("two_stage_corner_pipeline.select_torch_device", return_value=torch.device("cpu")),
            mock.patch("two_stage_corner_pipeline.torch.load", return_value=checkpoint),
            mock.patch("local_corner_moe_coord.LocalCornerMoECoordNet", return_value=FakeModel()),
            mock.patch("local_corner_heatmap.build_patch_metadata", return_value=np.zeros((14,), dtype=np.float32)),
            mock.patch(
                "local_corner_refine.build_local_corner_patch_sample",
                return_value={
                    "patch_image": np.zeros((96, 96, 3), dtype=np.uint8),
                    "patch": {"x": 0, "y": 0, "size": 100},
                    "predicted_point": [50.0, 50.0],
                },
            ),
            mock.patch("local_corner_refine.build_patch_features", return_value=np.zeros((10, 96, 96), dtype=np.float32)),
            mock.patch("two_stage_corner_pipeline.order_points", side_effect=lambda quad: np.array(quad, dtype=np.float32)),
            mock.patch(
                "local_corner_moe_coord.decode_moe_coord_output",
                return_value=torch.full((1, 2), 0.9, dtype=torch.float32),
            ),
        ):
            predictor = LocalCornerMoEPredictor(Path("/tmp/local_corner_moe_coord_model.pt"))
            result = predictor.predict_with_details(
                Path("/tmp/fake.jpg"),
                roi_quad,
                image=np.zeros((120, 120, 3), dtype=np.uint8),
            )

        np.testing.assert_allclose(result["quad"], roi_quad, atol=1e-4)

    def test_build_refine_request_normalizes_coarse_quad_to_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((200, 300, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)
            coarse_quad = np.array([[60, 40], [240, 42], [238, 160], [58, 158]], dtype=np.float32)

            request = build_refine_request(image_path=image_path, coarse_quad=coarse_quad, page_id="page-1")

        self.assertEqual(request["page_id"], "page-1")
        self.assertEqual(tuple(request["roi_image"].shape[:2]), (144, 212))
        self.assertEqual(request["roi"], {"x": 43, "y": 28, "width": 212, "height": 144})
        self.assertEqual(request["coarse_quad_norm"], [[0.080189, 0.083333], [0.929245, 0.097222], [0.919811, 0.916667], [0.070755, 0.902778]])

    def test_apply_roi_prediction_restores_absolute_points(self) -> None:
        roi = {"x": 40, "y": 20, "width": 200, "height": 100}
        pred_norm = np.array([[0.1, 0.1], [0.9, 0.1], [0.92, 0.88], [0.08, 0.9]], dtype=np.float32)

        restored = apply_roi_prediction(pred_norm, roi, image_shape=(180, 320, 3))

        self.assertEqual(restored.tolist(), [[60.0, 30.0], [220.0, 30.0], [224.0, 108.0], [56.0, 110.0]])

    def test_predict_two_stage_runs_global_then_roi_predictor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)

            coarse_quad = np.array([[50, 30], [270, 35], [260, 150], [55, 145]], dtype=np.float32)

            def fake_global_predictor(path: Path) -> np.ndarray:
                self.assertEqual(path, image_path)
                return coarse_quad

            def fake_roi_predictor(request: dict[str, object]) -> np.ndarray:
                self.assertEqual(request["page_id"], "sample")
                self.assertEqual(request["image_path"], str(image_path))
                return np.array([[0.1, 0.1], [0.9, 0.1], [0.92, 0.9], [0.08, 0.88]], dtype=np.float32)

            result = predict_two_stage(
                image_path=image_path,
                global_predictor=fake_global_predictor,
                roi_predictor=fake_roi_predictor,
            )

        self.assertEqual(result["coarse_quad"], coarse_quad.tolist())
        self.assertEqual(result["final_quad"], [[57.6, 32.4], [262.4, 32.4], [267.52, 147.6], [52.48, 144.72]])

    def test_predict_two_stage_applies_optional_local_predictor_after_roi_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)

            coarse_quad = np.array([[50, 30], [270, 35], [260, 150], [55, 145]], dtype=np.float32)
            roi_quad = np.array([[57.6, 32.4], [262.4, 32.4], [267.52, 147.6], [52.48, 144.72]], dtype=np.float32)
            refined_quad = np.array([[56.0, 31.0], [264.0, 31.5], [268.0, 146.0], [51.0, 143.5]], dtype=np.float32)

            def fake_global_predictor(path: Path) -> np.ndarray:
                self.assertEqual(path, image_path)
                return coarse_quad

            def fake_roi_predictor(request: dict[str, object]) -> np.ndarray:
                return np.array([[0.1, 0.1], [0.9, 0.1], [0.92, 0.9], [0.08, 0.88]], dtype=np.float32)

            def fake_local_predictor(path: Path, quad: np.ndarray) -> np.ndarray:
                self.assertEqual(path, image_path)
                np.testing.assert_allclose(quad, roi_quad, atol=1e-4)
                return refined_quad

            result = predict_two_stage(
                image_path=image_path,
                global_predictor=fake_global_predictor,
                roi_predictor=fake_roi_predictor,
                local_predictor=fake_local_predictor,
            )

        self.assertEqual(result["coarse_quad"], coarse_quad.tolist())
        self.assertEqual(result["roi_quad"], [[57.6, 32.4], [262.4, 32.4], [267.52, 147.6], [52.48, 144.72]])
        self.assertEqual(result["final_quad"], [[56.0, 31.0], [264.0, 31.5], [268.0, 146.0], [51.0, 143.5]])

    def test_export_refine_dataset_uses_fast_png_write_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_dir = root / "split"
            split_dir.mkdir(parents=True, exist_ok=True)
            output_dir = root / "output"
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), np.zeros((120, 160, 3), dtype=np.uint8))
            row = {
                "page_id": "sample",
                "project_name": "demo",
                "image_path": str(image_path),
                "manual_quad": [[20.0, 20.0], [140.0, 20.0], [140.0, 100.0], [20.0, 100.0]],
            }
            for split_name in ("train", "test"):
                (split_dir / f"{split_name}.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            fake_request = {
                "roi_image": np.zeros((64, 64, 3), dtype=np.uint8),
                "roi": {"x": 10, "y": 10, "width": 80, "height": 60},
                "coarse_quad": [[22.0, 22.0], [138.0, 22.0], [138.0, 98.0], [22.0, 98.0]],
                "coarse_quad_norm": [[0.15, 0.2], [0.85, 0.2], [0.85, 0.8], [0.15, 0.8]],
                "image_shape": (120, 160, 3),
            }

            with (
                mock.patch("two_stage_corner_pipeline.GlobalCornerPredictor", return_value=mock.Mock(return_value=np.array(row["manual_quad"], dtype=np.float32))),
                mock.patch("two_stage_corner_pipeline.build_refine_request", return_value=fake_request),
                mock.patch("two_stage_corner_pipeline.cv2.imwrite", return_value=True) as mock_imwrite,
            ):
                export_refine_dataset_from_global_predictions(
                    global_model_path=Path("/tmp/global.pt"),
                    split_dir=split_dir,
                    output_dir=output_dir,
                )

        write_args = mock_imwrite.call_args_list[0].args
        self.assertEqual(write_args[2], [cv2.IMWRITE_PNG_COMPRESSION, 1])

    def test_export_refine_dataset_reuses_cached_requests_for_duplicate_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_dir = root / "split"
            split_dir.mkdir(parents=True, exist_ok=True)
            output_dir = root / "output"
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), np.zeros((120, 160, 3), dtype=np.uint8))
            row = {
                "page_id": "sample",
                "project_name": "demo",
                "image_path": str(image_path),
                "manual_quad": [[20.0, 20.0], [140.0, 20.0], [140.0, 100.0], [20.0, 100.0]],
            }
            train_payload = "\n".join(json.dumps(row, ensure_ascii=False) for _ in range(2)) + "\n"
            (split_dir / "train.jsonl").write_text(train_payload, encoding="utf-8")
            (split_dir / "test.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            fake_request = {
                "roi_image": np.zeros((64, 64, 3), dtype=np.uint8),
                "roi": {"x": 10, "y": 10, "width": 80, "height": 60},
                "coarse_quad": [[22.0, 22.0], [138.0, 22.0], [138.0, 98.0], [22.0, 98.0]],
                "coarse_quad_norm": [[0.15, 0.2], [0.85, 0.2], [0.85, 0.8], [0.15, 0.8]],
                "image_shape": (120, 160, 3),
            }

            with (
                mock.patch("two_stage_corner_pipeline.GlobalCornerPredictor", return_value=mock.Mock(return_value=np.array(row["manual_quad"], dtype=np.float32))),
                mock.patch("two_stage_corner_pipeline.build_refine_request", return_value=fake_request) as mock_build_request,
                mock.patch("two_stage_corner_pipeline.cv2.imwrite", return_value=True) as mock_imwrite,
            ):
                export_refine_dataset_from_global_predictions(
                    global_model_path=Path("/tmp/global.pt"),
                    split_dir=split_dir,
                    output_dir=output_dir,
                )

        self.assertEqual(mock_build_request.call_count, 2)
        self.assertEqual(mock_imwrite.call_count, 2)

    def test_predict_two_stage_uses_multi_expand_candidate_for_low_confidence_baseline(self) -> None:
        test_case = self
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)

            coarse_quad = np.array([[50, 30], [270, 35], [260, 150], [55, 145]], dtype=np.float32)
            baseline_quad = np.array([[60.0, 35.0], [260.0, 35.0], [260.0, 145.0], [60.0, 145.0]], dtype=np.float32)
            alt_quad = np.array([[58.0, 33.0], [262.0, 33.0], [262.0, 147.0], [58.0, 147.0]], dtype=np.float32)

            class FakeLocalPredictor:
                def __init__(self) -> None:
                    self.calls = 0

                def predict_with_details(self, path: Path, predicted_quad: np.ndarray, image: np.ndarray | None = None) -> dict[str, object]:
                    del image
                    test_case.assertEqual(path, image_path)
                    self.calls += 1
                    if self.calls in {1, 3}:
                        return {
                            "quad": baseline_quad,
                            "corner_details": [
                                {"trust": 0.30, "patch_size": 100.0, "displacement": 20.0},
                                {"trust": 0.31, "patch_size": 100.0, "displacement": 18.0},
                                {"trust": 0.32, "patch_size": 100.0, "displacement": 17.0},
                                {"trust": 0.30, "patch_size": 100.0, "displacement": 19.0},
                            ],
                        }
                    return {
                        "quad": alt_quad,
                        "corner_details": [
                            {"trust": 0.62, "patch_size": 100.0, "displacement": 8.0},
                            {"trust": 0.61, "patch_size": 100.0, "displacement": 7.0},
                            {"trust": 0.60, "patch_size": 100.0, "displacement": 9.0},
                            {"trust": 0.63, "patch_size": 100.0, "displacement": 8.0},
                        ],
                    }

                def __call__(self, path: Path, quad: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
                    return self.predict_with_details(path, quad, image=image)["quad"]

            def fake_global_predictor(path: Path) -> np.ndarray:
                self.assertEqual(path, image_path)
                return coarse_quad

            def fake_roi_predictor(request: dict[str, object]) -> np.ndarray:
                roi = request["roi"]
                if int(roi["x"]) > 840:
                    return np.array([[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]], dtype=np.float32)
                return np.array([[0.16, 0.12], [0.86, 0.12], [0.86, 0.88], [0.16, 0.88]], dtype=np.float32)

            result = predict_two_stage(
                image_path=image_path,
                global_predictor=fake_global_predictor,
                roi_predictor=fake_roi_predictor,
                local_predictor=FakeLocalPredictor(),
                expand_ratio=0.08,
                candidate_expand_ratios=[0.04, 0.08],
                candidate_baseline_gate=0.45,
                candidate_min_score_gain=0.03,
            )

        self.assertEqual(result["selected_expand_ratio"], 0.04)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["final_quad"], [[58.0, 33.0], [262.0, 33.0], [262.0, 147.0], [58.0, 147.0]])

    def test_predict_two_stage_keeps_baseline_when_baseline_confidence_is_high(self) -> None:
        test_case = self
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)

            coarse_quad = np.array([[50, 30], [270, 35], [260, 150], [55, 145]], dtype=np.float32)
            baseline_quad = np.array([[60.0, 35.0], [260.0, 35.0], [260.0, 145.0], [60.0, 145.0]], dtype=np.float32)
            alt_quad = np.array([[58.0, 33.0], [262.0, 33.0], [262.0, 147.0], [58.0, 147.0]], dtype=np.float32)

            class FakeLocalPredictor:
                def __init__(self) -> None:
                    self.calls = 0

                def predict_with_details(self, path: Path, predicted_quad: np.ndarray, image: np.ndarray | None = None) -> dict[str, object]:
                    del image
                    test_case.assertEqual(path, image_path)
                    self.calls += 1
                    if self.calls in {1, 3}:
                        return {
                            "quad": baseline_quad,
                            "corner_details": [
                                {"trust": 0.58, "patch_size": 100.0, "displacement": 8.0},
                                {"trust": 0.57, "patch_size": 100.0, "displacement": 7.0},
                                {"trust": 0.59, "patch_size": 100.0, "displacement": 8.0},
                                {"trust": 0.58, "patch_size": 100.0, "displacement": 9.0},
                            ],
                        }
                    return {
                        "quad": alt_quad,
                        "corner_details": [
                            {"trust": 0.62, "patch_size": 100.0, "displacement": 8.0},
                            {"trust": 0.61, "patch_size": 100.0, "displacement": 7.0},
                            {"trust": 0.60, "patch_size": 100.0, "displacement": 9.0},
                            {"trust": 0.63, "patch_size": 100.0, "displacement": 8.0},
                        ],
                    }

                def __call__(self, path: Path, quad: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
                    return self.predict_with_details(path, quad, image=image)["quad"]

            def fake_global_predictor(path: Path) -> np.ndarray:
                self.assertEqual(path, image_path)
                return coarse_quad

            def fake_roi_predictor(request: dict[str, object]) -> np.ndarray:
                roi = request["roi"]
                if int(roi["x"]) > 840:
                    return np.array([[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]], dtype=np.float32)
                return np.array([[0.16, 0.12], [0.86, 0.12], [0.86, 0.88], [0.16, 0.88]], dtype=np.float32)

            result = predict_two_stage(
                image_path=image_path,
                global_predictor=fake_global_predictor,
                roi_predictor=fake_roi_predictor,
                local_predictor=FakeLocalPredictor(),
                expand_ratio=0.08,
                candidate_expand_ratios=[0.04, 0.08],
                candidate_baseline_gate=0.45,
                candidate_min_score_gain=0.03,
            )

        self.assertEqual(result["selected_expand_ratio"], 0.08)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["final_quad"], [[60.0, 35.0], [260.0, 35.0], [260.0, 145.0], [60.0, 145.0]])

    def test_predict_two_stage_uses_candidate_selector_when_available(self) -> None:
        test_case = self
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            image_path = root / "sample.jpg"
            cv2.imwrite(str(image_path), image)

            coarse_quad = np.array([[50, 30], [270, 35], [260, 150], [55, 145]], dtype=np.float32)
            baseline_quad = np.array([[60.0, 35.0], [260.0, 35.0], [260.0, 145.0], [60.0, 145.0]], dtype=np.float32)
            alt_quad = np.array([[58.0, 33.0], [262.0, 33.0], [262.0, 147.0], [58.0, 147.0]], dtype=np.float32)

            class FakeLocalPredictor:
                def __init__(self) -> None:
                    self.calls = 0

                def predict_with_details(self, path: Path, predicted_quad: np.ndarray, image: np.ndarray | None = None) -> dict[str, object]:
                    del image, predicted_quad
                    test_case.assertEqual(path, image_path)
                    self.calls += 1
                    if self.calls in {1, 3}:
                        return {
                            "quad": baseline_quad,
                            "corner_details": [
                                {"trust": 0.58, "patch_size": 100.0, "displacement": 8.0},
                                {"trust": 0.57, "patch_size": 100.0, "displacement": 7.0},
                                {"trust": 0.59, "patch_size": 100.0, "displacement": 8.0},
                                {"trust": 0.58, "patch_size": 100.0, "displacement": 9.0},
                            ],
                        }
                    return {
                        "quad": alt_quad,
                        "corner_details": [
                            {"trust": 0.55, "patch_size": 100.0, "displacement": 8.0},
                            {"trust": 0.54, "patch_size": 100.0, "displacement": 7.0},
                            {"trust": 0.53, "patch_size": 100.0, "displacement": 9.0},
                            {"trust": 0.56, "patch_size": 100.0, "displacement": 8.0},
                        ],
                    }

                def __call__(self, path: Path, quad: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
                    return self.predict_with_details(path, quad, image=image)["quad"]

            class FakeSelector:
                def select_candidate(self, candidates: list[dict[str, object]], *, baseline_expand_ratio: float) -> dict[str, object] | None:
                    test_case.assertEqual(round(float(baseline_expand_ratio), 2), 0.08)
                    return next(item for item in candidates if abs(float(item["expand_ratio"]) - 0.04) < 1e-9)

            def fake_global_predictor(path: Path) -> np.ndarray:
                self.assertEqual(path, image_path)
                return coarse_quad

            def fake_roi_predictor(request: dict[str, object]) -> np.ndarray:
                roi = request["roi"]
                if int(roi["x"]) > 840:
                    return np.array([[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]], dtype=np.float32)
                return np.array([[0.16, 0.12], [0.86, 0.12], [0.86, 0.88], [0.16, 0.88]], dtype=np.float32)

            result = predict_two_stage(
                image_path=image_path,
                global_predictor=fake_global_predictor,
                roi_predictor=fake_roi_predictor,
                local_predictor=FakeLocalPredictor(),
                expand_ratio=0.08,
                candidate_expand_ratios=[0.04, 0.08],
                candidate_baseline_gate=0.45,
                candidate_min_score_gain=0.03,
                candidate_selector=FakeSelector(),
            )

        self.assertEqual(result["selected_expand_ratio"], 0.04)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["final_quad"], [[58.0, 33.0], [262.0, 33.0], [262.0, 147.0], [58.0, 147.0]])

    def test_linear_candidate_expand_selector_keeps_baseline_without_clear_margin(self) -> None:
        selector = LinearCandidateExpandSelector(
            weights=[1.0],
            bias=0.0,
            feature_mean=[0.0],
            feature_std=[1.0],
            feature_names=["score"],
            switch_margin=0.1,
        )

        chosen = selector.select_candidate(
            [
                {"expand_ratio": 0.08, "selector_features": [0.50]},
                {"expand_ratio": 0.04, "selector_features": [0.56]},
            ],
            baseline_expand_ratio=0.08,
        )

        self.assertIsNotNone(chosen)
        self.assertEqual(float(chosen["expand_ratio"]), 0.08)


if __name__ == "__main__":
    unittest.main()
