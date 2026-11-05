from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from dataset_benchmark import summarize_point_hits
from screen_corner_single import (ScreenCornerSingle, SingleModelPredictor, decode_points,
    patch_boxes, sample_corner_patches)
from training.train_screen_corner_single import hit_loss
import detect_frame


class TinyEncoder(nn.Module):
    def forward(self, x):
        scalar = x.mean(1, keepdim=True)
        return (F.interpolate(scalar, [32, 32]).repeat(1, 24, 1, 1),
                F.interpolate(scalar, [16, 16]).repeat(1, 48, 1, 1),
                F.interpolate(scalar, [8, 8]).repeat(1, 576, 1, 1))


def make_model():
    encoder = torch.jit.trace(TinyEncoder(), torch.rand(1, 3, 256, 256))
    return ScreenCornerSingle(encoder).eval()


def test_single_file_reloads_without_checkpoints_or_torchvision():
    torch.set_num_threads(2)
    model = make_model()
    scripted = torch.jit.script(model)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'only_file.pt'
        scripted.save(str(path))
        with mock.patch('torch.load', side_effect=AssertionError('External checkpoint read')):
            loaded = torch.jit.load(str(path))
            for shape in [(1, 3, 160, 280), (2, 3, 300, 180)]:
                image = torch.rand(shape)
                with torch.no_grad():
                    expected, actual = model(image), loaded(image)
                for key in ['coarse_quad', 'final_quad']:
                    torch.testing.assert_close(expected[key], actual[key])
                    assert actual[key].shape == (shape[0], 4, 2)
                assert torch.isfinite(actual['final_quad']).all()


def test_cached_training_refinement_matches_complete_graph():
    model = make_model()
    image = torch.rand(1, 3, 180, 320)
    with torch.no_grad():
        small = F.interpolate(image, [256, 256], mode='bilinear', align_corners=False)
        logits, _, features = model.global_head(model.encoder((small-model.mean)/model.std))
        boxes = patch_boxes(decode_points(logits))
        relative, _ = model.local_head(sample_corner_patches(image, boxes, 64),
                                       sample_corner_patches(features, boxes, 16))
        final = (boxes[..., :2] + relative.view(-1, 4, 2)*(boxes[..., 2:]-boxes[..., :2])).clamp(0., 1.)
        torch.testing.assert_close(model(image)['final_quad'], final)


def test_point_surrogate_restores_original_pixel_aspect_ratio():
    target = torch.tensor([[[.1, .1], [.9, .1], [.9, .9], [.1, .9]]])
    pred_x, pred_y = target.clone(), target.clone()
    pred_x[0, 0, 0] += .02
    pred_y[0, 0, 1] += .02
    size = torch.tensor([[2000., 500.]])
    assert hit_loss(pred_x, target, size) > hit_loss(pred_y, target, size)
    pred_x.requires_grad_(True)
    hit_loss(pred_x, target, size).backward()
    assert torch.isfinite(pred_x.grad).all()
    assert pred_x.grad[0, 0, 0] > 0


def test_runtime_uses_single_graph_without_legacy_models():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'screen_corner_single.pt'
        torch.jit.script(make_model()).save(str(path))
        with (mock.patch.dict('os.environ', {'SCREEN_PDF_SINGLE_MODEL': str(path)}),
              mock.patch('detect_frame.run_teacher_detection', side_effect=AssertionError('legacy pipeline used')),
              mock.patch('detect_frame.detect_best_candidate', side_effect=AssertionError('external candidate used'))):
            try:
                payload = detect_frame.build_detect_payload('/unused.jpg', np.zeros((100, 180, 3), np.uint8))
                assert payload['best']['method'] == 'screen_corner_single'
                assert len(payload['candidates']) == 1
                assert len(payload['best']['quad']) == 4
            finally:
                detect_frame._SINGLE_MODEL_RUNTIME = None
