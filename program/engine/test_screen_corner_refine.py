from pathlib import Path
import tempfile

import torch

from screen_corner_all_history import AllHistoryScreenCorner
from test_screen_corner_single import TinyEncoder
from training.train_all_history_refine import load_frozen_base


def test_warm_start_copies_encoder_and_coarse_and_keeps_local_head_fresh():
    torch.set_num_threads(2)
    base = AllHistoryScreenCorner(TinyEncoder())
    reference = {k: v.detach().clone() for k, v in base.state_dict().items()}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'base.pt'
        torch.jit.script(base.eval()).save(str(path))
        model = AllHistoryScreenCorner(TinyEncoder())
        fresh_local = {k: v.clone() for k, v in model.local_head.state_dict().items()}
        loaded = load_frozen_base(model, path)
    for key, value in loaded.items():
        if key.startswith('local_head.'):
            continue
        assert torch.equal(model.state_dict()[key], reference[key]), key
        assert torch.equal(model.state_dict()[key].cpu(), value.cpu()), key
    for key, value in model.local_head.state_dict().items():
        assert torch.equal(value, fresh_local[key]), key
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.global_head.parameters())
    assert all(p.requires_grad for p in model.local_head.parameters())


def test_frozen_encoder_and_coarse_survive_a_local_head_optimizer_step():
    torch.set_num_threads(2)
    base = AllHistoryScreenCorner(TinyEncoder())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'base.pt'
        torch.jit.script(base.eval()).save(str(path))
        model = AllHistoryScreenCorner(TinyEncoder())
        load_frozen_base(model, path)
    frozen = {k: v.detach().clone() for k, v in model.state_dict().items() if not k.startswith('local_head.')}
    optimizer = torch.optim.Adam(model.local_head.parameters(), lr=3e-4)
    image = torch.rand(2, 3, 180, 320)
    coarse, _, _, feature = model.global_prediction(image)
    from screen_corner_single import patch_boxes, sample_corner_patches
    boxes = patch_boxes(coarse)
    relative, logits = model.local_head(sample_corner_patches(image, boxes, 64),
                                        sample_corner_patches(feature, boxes, 16))
    (relative.square().sum() + logits.square().mean()).backward()
    optimizer.step()
    for key, value in frozen.items():
        assert torch.equal(model.state_dict()[key], value), key
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.local_head.parameters())
