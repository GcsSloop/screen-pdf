from __future__ import annotations

import unittest
from pathlib import Path
import sys
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_corner_moe_coord import LocalCornerMoECoordNet, decode_moe_coord_output, train_local_corner_moe_coord_model
from local_corner_moe_coord import load_local_corner_moe_coord_init_state


class LocalCornerMoECoordTests(unittest.TestCase):
    def test_local_corner_moe_coord_net_outputs_expected_tensors(self) -> None:
        model = LocalCornerMoECoordNet(channels=16, experts=3, metadata_dim=14, input_channels=13)
        batch = torch.randn(2, 13, 64, 64)
        metadata = torch.randn(2, 14)

        heatmaps, offsets, coords, edge_dirs, edge_maps, visibility, gates = model(batch, metadata)

        self.assertEqual(tuple(heatmaps.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(offsets.shape), (2, 2, 16, 16))
        self.assertEqual(tuple(coords.shape), (2, 2))
        self.assertEqual(tuple(edge_dirs.shape), (2, 5))
        self.assertEqual(tuple(edge_maps.shape), (2, 2, 16, 16))
        self.assertEqual(tuple(visibility.shape), (2, 2))
        self.assertEqual(tuple(gates.shape), (2, 3))

    def test_decode_moe_coord_output_blends_heatmap_and_coord_head(self) -> None:
        heatmaps = torch.full((1, 1, 5, 5), -8.0, dtype=torch.float32)
        heatmaps[0, 0, 2, 2] = 8.0
        offsets = torch.zeros((1, 2, 5, 5), dtype=torch.float32)
        coord_head = torch.tensor([[0.8, 0.2]], dtype=torch.float32)

        coords = decode_moe_coord_output(heatmaps, offsets, coord_head, coord_mix=0.25)

        self.assertAlmostEqual(float(coords[0, 0]), 0.575, places=2)
        self.assertAlmostEqual(float(coords[0, 1]), 0.425, places=2)

    def test_load_local_corner_moe_coord_init_state_supports_partial_load(self) -> None:
        source_model = LocalCornerMoECoordNet(channels=16, experts=3, metadata_dim=14, input_channels=13)
        target_model = LocalCornerMoECoordNet(channels=16, experts=3, metadata_dim=14, input_channels=10)
        checkpoint_path = Path(self.id().replace("/", "_") + ".pt")
        try:
            torch.save({"state_dict": source_model.state_dict()}, checkpoint_path)
            info = load_local_corner_moe_coord_init_state(target_model, checkpoint_path)
        finally:
            checkpoint_path.unlink(missing_ok=True)

        self.assertIn("stem.0.weight", info["missing"])
        self.assertIn("stem.0.weight", info["skipped_shape"])
        self.assertEqual(info["unexpected"], [])
        self.assertEqual(info["checkpoint_path"], str(checkpoint_path))

    def test_train_local_corner_moe_coord_model_saves_local_patch_config(self) -> None:
        class FakeDataset:
            def __len__(self):
                return 0

        class FakeLoader(list):
            pass

        class FakeOptimizer:
            def zero_grad(self, set_to_none=True):
                return None

            def step(self):
                return None

        class FakeModel:
            def __init__(self):
                self._state = {"weight": torch.tensor([1.0], dtype=torch.float32)}

            def to(self, device):
                return self

            def train(self):
                return self

            def eval(self):
                return self

            def state_dict(self):
                return self._state

            def parameters(self):
                return []

        saved_payload: dict[str, object] = {}

        def fake_torch_save(obj, path):
            saved_payload["obj"] = obj
            saved_payload["path"] = str(path)

        with mock.patch("local_corner_moe_coord.LocalCornerHeatmapDataset", side_effect=[FakeDataset(), FakeDataset()]), mock.patch(
            "local_corner_moe_coord.DataLoader", side_effect=lambda dataset, **kwargs: FakeLoader()
        ), mock.patch("local_corner_moe_coord.select_torch_device", return_value=torch.device("cpu")), mock.patch(
            "local_corner_moe_coord.LocalCornerMoECoordNet", return_value=FakeModel()
        ), mock.patch(
            "local_corner_moe_coord._evaluate", return_value={"loss": 0.1, "point_error": 0.02}
        ), mock.patch(
            "local_corner_moe_coord.torch.optim.Adam", return_value=FakeOptimizer()
        ), mock.patch(
            "local_corner_moe_coord.torch.save", side_effect=fake_torch_save
        ):
            result = train_local_corner_moe_coord_model(
                dataset_dir=Path("/tmp/fake-local-dataset"),
                output_dir=Path("/tmp/fake-local-output"),
                epochs=1,
                local_patch_config={
                    "patch_scale": 0.22,
                    "patch_min": 112,
                    "patch_max": 320,
                    "bottom_vertical_bias": 0.04,
                    "bl_patch_scale_multiplier": 1.1,
                    "bl_bottom_vertical_bias": 0.08,
                },
            )

        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(saved_payload["obj"]["local_patch_config"], {
            "patch_scale": 0.22,
            "patch_min": 112,
            "patch_max": 320,
            "bottom_vertical_bias": 0.04,
            "bl_patch_scale_multiplier": 1.1,
            "bl_bottom_vertical_bias": 0.08,
        })


if __name__ == "__main__":
    unittest.main()
