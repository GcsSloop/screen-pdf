from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deep_screen_v1_model import (
    DeepScreenV1Net,
    apply_visibility_guided_process_delta,
    build_roi_boxes_from_quads,
    load_compatible_state_dict,
)


class DeepScreenV1ModelTests(unittest.TestCase):
    def test_build_roi_boxes_from_quads_returns_normalized_boxes(self) -> None:
        quads = torch.tensor(
            [
                [[0.2, 0.2], [0.8, 0.2], [0.82, 0.78], [0.18, 0.8]],
            ],
            dtype=torch.float32,
        )

        boxes = build_roi_boxes_from_quads(quads, expand_ratio=0.1)

        self.assertEqual(tuple(boxes.shape), (1, 4))
        self.assertGreaterEqual(float(boxes.min()), 0.0)
        self.assertLessEqual(float(boxes.max()), 1.0)

    def test_model_forward_returns_coarse_and_refine_outputs(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        output = model(image)

        self.assertEqual(tuple(output["coarse_heatmaps"].shape), (2, 4, 64, 64))
        self.assertEqual(tuple(output["coarse_offsets"].shape), (2, 4, 2, 64, 64))
        self.assertEqual(tuple(output["coarse_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["roi_stage_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["final_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["router_logits"].shape), (2, 2))
        self.assertEqual(tuple(output["process_delta"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["process_visibility"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["process_edge"].shape), (2, 4, 5))
        self.assertEqual(tuple(output["process_fallback_logits"].shape), (2, 4))

    def test_model_forward_supports_roi_adapter_layers(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, roi_adapter_layers=2)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        output = model(image)

        self.assertEqual(tuple(output["final_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["router_logits"].shape), (2, 2))

    def test_model_forward_returns_scene_logits_for_scene_aware_routing(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        output = model(image)

        self.assertIn("scene_logits", output)
        self.assertEqual(tuple(output["scene_logits"].shape), (2, 4))

    def test_roi_adapter_initializes_close_to_identity(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, roi_adapter_layers=2)
        roi_features = torch.rand(2, 32, 16, 16, dtype=torch.float32)

        with torch.no_grad():
            adapted = model.roi_adapter(roi_features)

        self.assertLess(float((adapted - roi_features).abs().mean()), 1e-4)

    def test_model_forward_supports_spatial_refine_layers(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, spatial_refine_layers=2)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        output = model(image)

        self.assertEqual(tuple(output["final_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["base_final_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["spatial_quad"].shape), (2, 4, 2))

    def test_spatial_refine_head_initializes_close_to_identity(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, spatial_refine_layers=2)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertLess(float((output["final_quad"] - output["base_final_quad"]).abs().mean()), 1e-4)

    def test_model_forward_supports_residual_quad_head(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, residual_quad_head_layers=2)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        output = model(image)

        self.assertEqual(tuple(output["residual_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["residual_blend_weight"].shape), (2, 1, 1))
        self.assertEqual(tuple(output["base_final_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["final_quad"].shape), (2, 4, 2))

    def test_residual_quad_head_initializes_close_to_identity(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, residual_quad_head_layers=2)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertLess(float(output["residual_blend_weight"].mean()), 0.2)
        self.assertLess(float((output["residual_quad"] - output["base_final_quad"]).abs().mean()), 1e-4)
        self.assertLess(float((output["final_quad"] - output["base_final_quad"]).abs().mean()), 1e-4)

    def test_model_forward_supports_strict_spatial_refine_head(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, strict_spatial_refine_layers=2)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        output = model(image)

        self.assertEqual(tuple(output["strict_point_heatmaps"].shape), (2, 4, 16, 16))
        self.assertEqual(tuple(output["strict_point_offsets"].shape), (2, 4, 2, 16, 16))
        self.assertEqual(tuple(output["strict_point_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["strict_point_blend_weight"].shape), (2, 4, 1))
        self.assertEqual(tuple(output["final_quad"].shape), (2, 4, 2))

    def test_model_forward_supports_candidate_selection_head(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, candidate_selection_enabled=True)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        output = model(image)

        self.assertEqual(tuple(output["candidate_quads"].shape), (2, 3, 4, 2))
        self.assertEqual(tuple(output["candidate_scores"].shape), (2, 3))
        self.assertEqual(tuple(output["candidate_selected_index"].shape), (2,))
        self.assertEqual(tuple(output["final_quad"].shape), (2, 4, 2))

    def test_model_can_limit_internal_candidate_pool(self) -> None:
        model = DeepScreenV1Net(
            base_channels=16,
            roi_size=16,
            experts=2,
            candidate_selection_enabled=True,
            internal_candidate_names=["coarse_quad"],
        )
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertEqual(tuple(output["candidate_quads"].shape), (2, 1, 4, 2))
        self.assertEqual(tuple(output["candidate_scores"].shape), (2, 1))
        self.assertLess(float((output["final_quad"] - output["coarse_quad"]).abs().mean()), 1e-6)

    def test_model_forward_can_expose_state_aware_candidate(self) -> None:
        model = DeepScreenV1Net(
            base_channels=16,
            roi_size=16,
            experts=2,
            candidate_selection_enabled=True,
            state_aware_candidate_enabled=True,
        )
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertIn("state_aware_quad", output)
        self.assertIn("corner_state_logits", output)
        self.assertEqual(tuple(output["state_aware_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["corner_state_logits"].shape), (2, 4))
        self.assertEqual(tuple(output["candidate_quads"].shape), (2, 3, 4, 2))

    def test_model_can_include_state_aware_candidate_in_internal_candidate_pool(self) -> None:
        model = DeepScreenV1Net(
            base_channels=16,
            roi_size=16,
            experts=2,
            candidate_selection_enabled=True,
            state_aware_candidate_enabled=True,
            internal_candidate_names=["coarse_quad", "roi_stage_quad", "state_aware_quad", "base_final_quad"],
        )
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertEqual(tuple(output["candidate_quads"].shape), (2, 4, 4, 2))
        self.assertTrue(torch.equal(output["candidate_selected_index"], torch.full((2,), 3, dtype=torch.long)))
        self.assertLess(float((output["final_quad"] - output["base_final_quad"]).abs().mean()), 1e-6)

    def test_candidate_selection_initializes_to_base_final_candidate(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, candidate_selection_enabled=True)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertTrue(torch.equal(output["candidate_selected_index"], torch.full((2,), 2, dtype=torch.long)))
        self.assertLess(float((output["final_quad"] - output["base_final_quad"]).abs().mean()), 1e-6)

    def test_state_aware_candidate_initializes_close_to_coarse_and_keeps_base_final_as_default(self) -> None:
        model = DeepScreenV1Net(
            base_channels=16,
            roi_size=16,
            experts=2,
            candidate_selection_enabled=True,
            state_aware_candidate_enabled=True,
        )
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertTrue(torch.equal(output["candidate_selected_index"], torch.full((2,), 2, dtype=torch.long)))
        self.assertLess(float((output["state_aware_quad"] - output["coarse_quad"]).abs().mean()), 1e-6)
        self.assertLess(float((output["final_quad"] - output["base_final_quad"]).abs().mean()), 1e-6)

    def test_model_can_force_final_output_to_coarse_quad(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, final_output_mode="coarse")
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertLess(float((output["final_quad"] - output["coarse_quad"]).abs().mean()), 1e-6)

    def test_model_can_force_final_output_to_coarse_strict_refine(self) -> None:
        model = DeepScreenV1Net(
            base_channels=16,
            roi_size=16,
            experts=2,
            strict_spatial_refine_layers=2,
            final_output_mode="coarse_strict",
        )
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertIn("strict_point_base_quad", output)
        self.assertLess(float((output["strict_point_base_quad"] - output["coarse_quad"]).abs().mean()), 1e-6)
        self.assertLess(float((output["final_quad"] - output["coarse_quad"]).abs().mean()), 1e-4)

    def test_model_forward_supports_coarse_residual_head(self) -> None:
        model = DeepScreenV1Net(
            base_channels=16,
            roi_size=16,
            experts=2,
            coarse_residual_head_layers=2,
            final_output_mode="coarse_residual",
        )
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertIn("coarse_residual_quad", output)
        self.assertIn("coarse_residual_gate", output)
        self.assertEqual(tuple(output["coarse_residual_quad"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["coarse_residual_gate"].shape), (2, 4, 1))
        self.assertLess(float((output["final_quad"] - output["coarse_residual_quad"]).abs().mean()), 1e-6)

    def test_coarse_residual_head_initializes_close_to_identity(self) -> None:
        model = DeepScreenV1Net(
            base_channels=16,
            roi_size=16,
            experts=2,
            coarse_residual_head_layers=2,
            final_output_mode="coarse_residual",
        )
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertGreaterEqual(float(output["coarse_residual_gate"].min()), 0.0)
        self.assertLessEqual(float(output["coarse_residual_gate"].max()), 1.0)
        self.assertLess(float((output["coarse_residual_quad"] - output["coarse_quad"]).abs().mean()), 1e-4)

    def test_model_can_force_final_output_to_state_aware_candidate(self) -> None:
        model = DeepScreenV1Net(
            base_channels=16,
            roi_size=16,
            experts=2,
            final_output_mode="state_aware",
            state_aware_candidate_enabled=True,
        )
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertLess(float((output["final_quad"] - output["state_aware_quad"]).abs().mean()), 1e-6)

    def test_model_forward_exposes_identity_coarse_scene_adapter(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, final_output_mode="coarse")
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertIn("p2", output)
        self.assertIn("coarse_p2", output)
        self.assertLess(float((output["coarse_p2"] - output["p2"]).abs().mean()), 1e-6)

    def test_coarse_scene_adapter_initializes_to_identity(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, final_output_mode="coarse")
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertLess(float((output["coarse_quad"] - output["final_quad"]).abs().mean()), 1e-6)
        self.assertLess(float((output["coarse_p2"] - output["p2"]).abs().mean()), 1e-6)

    def test_apply_visibility_guided_process_delta_only_adjusts_low_visibility_fallback_corners(self) -> None:
        coarse_quad = torch.tensor(
            [
                [[0.10, 0.10], [0.90, 0.10], [0.90, 0.90], [0.10, 0.90]],
            ],
            dtype=torch.float32,
        )
        roi_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
        process_delta = torch.tensor(
            [
                [[0.08, -0.04], [0.05, 0.05], [0.05, 0.05], [0.05, 0.05]],
            ],
            dtype=torch.float32,
        )
        process_visibility = torch.tensor(
            [
                [[0.10, 0.10], [0.95, 0.95], [0.95, 0.95], [0.95, 0.95]],
            ],
            dtype=torch.float32,
        )
        process_fallback_logits = torch.tensor([[8.0, -8.0, -8.0, -8.0]], dtype=torch.float32)

        refined_quad, refine_gate = apply_visibility_guided_process_delta(
            coarse_quad,
            roi_boxes,
            process_delta,
            process_visibility,
            process_fallback_logits,
            refine_scale=torch.tensor(1.0, dtype=torch.float32),
        )

        self.assertGreater(float(refine_gate[0, 0]), 0.9)
        self.assertLess(float(refine_gate[0, 1]), 1e-4)
        self.assertLess(float(refine_gate[0, 2]), 1e-4)
        self.assertLess(float(refine_gate[0, 3]), 1e-4)
        self.assertGreater(float((refined_quad[0, 0] - coarse_quad[0, 0]).abs().mean()), 1e-3)
        self.assertLess(float((refined_quad[0, 1:] - coarse_quad[0, 1:]).abs().mean()), 1e-6)

    def test_visibility_guided_coarse_refine_initializes_to_identity(self) -> None:
        model = DeepScreenV1Net(
            base_channels=16,
            roi_size=16,
            experts=2,
            final_output_mode="coarse",
            coarse_visibility_refine_enabled=True,
        )
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertIn("visibility_refined_quad", output)
        self.assertIn("visibility_refine_gate", output)
        self.assertGreaterEqual(float(output["visibility_refine_gate"].min()), 0.0)
        self.assertLessEqual(float(output["visibility_refine_gate"].max()), 1.0)
        self.assertLess(float((output["visibility_refined_quad"] - output["coarse_quad"]).abs().mean()), 1e-6)
        self.assertLess(float((output["final_quad"] - output["coarse_quad"]).abs().mean()), 1e-6)

    def test_candidate_selection_head_can_score_external_candidate_pool(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, candidate_selection_enabled=True)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)
            external_candidates = output["base_final_quad"][:, None, :, :].clone()
            candidate_pool = model.assemble_candidate_pool(output, external_candidate_quads=external_candidates)
            scored = model.score_candidate_pool(output, candidate_pool["candidate_quads"], candidate_mask=candidate_pool["candidate_mask"])

        self.assertEqual(tuple(candidate_pool["candidate_quads"].shape), (2, 4, 4, 2))
        self.assertEqual(tuple(candidate_pool["candidate_mask"].shape), (2, 4))
        self.assertEqual(tuple(scored.shape), (2, 4))

    def test_strict_spatial_refine_head_initializes_to_identity(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2, strict_spatial_refine_layers=2)
        image = torch.rand(2, 3, 256, 256, dtype=torch.float32)

        with torch.no_grad():
            output = model(image)

        self.assertGreaterEqual(float(output["strict_point_blend_weight"].min()), 0.0)
        self.assertLessEqual(float(output["strict_point_blend_weight"].max()), 1.0)
        self.assertLess(float((output["strict_point_quad"] - output["base_final_quad"]).abs().mean()), 1e-4)
        self.assertLess(float((output["final_quad"] - output["base_final_quad"]).abs().mean()), 1e-4)

    def test_load_compatible_state_dict_skips_mismatched_router_shapes(self) -> None:
        model = DeepScreenV1Net(base_channels=16, roi_size=16, experts=2)
        state_dict = model.state_dict()
        state_dict["local_refine_head.router.0.weight"] = torch.rand(48, 68)

        loaded_keys = load_compatible_state_dict(model, state_dict)

        self.assertNotIn("local_refine_head.router.0.weight", loaded_keys)


if __name__ == "__main__":
    unittest.main()
