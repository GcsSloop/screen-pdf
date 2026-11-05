from pathlib import Path
import tempfile
from unittest import mock

import numpy as np
import torch

from screen_corner_all_history import AllHistoryScreenCorner, PortableHardSigmoid, PortableHardSwish
from screen_corner_single import SingleModelPredictor
from test_screen_corner_single import TinyEncoder
from training.build_all_history_dataset import protected_roles


def test_portable_activations_match_pretrained_operations_and_gradients():
    x = torch.linspace(-5, 5, 31, requires_grad=True)
    for portable, standard in [(PortableHardSigmoid(), torch.nn.Hardsigmoid()), (PortableHardSwish(), torch.nn.Hardswish())]:
        torch.testing.assert_close(portable(x), standard(x))
        assert torch.isfinite(torch.autograd.grad(portable(x).sum(), x, retain_graph=True)[0]).all()


def test_known_evaluation_roles_can_never_enter_training():
    assert protected_roles(['train', 'validation']) == 'validation'
    assert protected_roles(['train', 'holdout']) == 'holdout'
    assert protected_roles(['validation', 'holdout']) == 'holdout'


def test_region_prediction_influences_coarse_and_both_heads_receive_gradients():
    torch.set_num_threads(2)
    model = AllHistoryScreenCorner(TinyEncoder())
    image = torch.rand(2, 3, 180, 320)
    coarse, _, boundary, _ = model.global_prediction(image)
    (coarse.square().sum() + boundary.sigmoid().square().mean()).backward()
    assert model.global_head.pyramid.boundary.weight.grad.abs().sum() > 0
    assert model.global_head.coarse[0].weight.grad.abs().sum() > 0
    assert torch.isfinite(coarse).all()


def test_new_model_exports_and_loads_without_task_checkpoints():
    torch.set_num_threads(2)
    model = AllHistoryScreenCorner(TinyEncoder()).eval()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'screen_corner_single.pt'
        torch.jit.script(model).save(str(path))
        with mock.patch('torch.load', side_effect=AssertionError('External weights loaded')):
            predictor = SingleModelPredictor(path)
            for h, w in [(160, 280), (240, 120)]:
                output = predictor.predict(np.zeros((h, w, 3), np.uint8))
                assert all(x.shape == (4, 2) and np.isfinite(x).all() for x in output.values())
