import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from screen_corner_region import region_to_quad, RegionGuidedScreenCornerSingle
from screen_corner_single import SingleModelPredictor
from test_screen_corner_single import TinyEncoder


def test_region_fit_preserves_perspective_instead_of_bbox():
    truth = np.array([[.17, .28], [.82, .15], [.72, .82], [.25, .72]])
    mask = np.zeros((256, 256), np.float32)
    cv2.fillConvexPoly(mask, np.round(truth * 255).astype(np.int32), 1.)
    result = region_to_quad(torch.from_numpy(mask)[None, None]).numpy()[0]
    np.testing.assert_allclose(result, truth, atol=.006)
    assert abs(result[0, 1] - result[1, 1]) > .1


def test_interior_occlusion_does_not_become_a_new_side():
    mask = torch.zeros(1, 1, 128, 128)
    mask[:, :, 20:108, 24:104] = 1.
    expected = region_to_quad(mask)
    mask[:, :, 65:100, 47:75] = 0.
    torch.testing.assert_close(region_to_quad(mask), expected)


def test_empty_region_is_finite_and_region_fit_has_finite_gradients():
    assert torch.isfinite(region_to_quad(torch.zeros(2, 1, 64, 64))).all()
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    result = region_to_quad(logits.sigmoid())
    result.square().sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_region_variant_exports_a_self_contained_file():
    torch.set_num_threads(2)
    encoder = torch.jit.trace(TinyEncoder(), torch.zeros(1, 3, 256, 256))
    model = RegionGuidedScreenCornerSingle(encoder).eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'only_model.pt'
        torch.jit.script(model).save(str(path))
        predictor = SingleModelPredictor(path)
        result = predictor.predict(np.zeros((200, 300, 3), np.uint8))
        assert result['coarse_quad'].shape == (4, 2)
        assert result['final_quad'].shape == (4, 2)
        assert np.isfinite(result['coarse_quad']).all()
