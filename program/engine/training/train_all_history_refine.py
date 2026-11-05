"""Local-refinement specialization on a frozen region+coarse base model.

The encoder and the region/coarse head are copied from a frozen all-history
candidate and never updated here. Only the local point head is trained, on
crops taken from the frozen model's own coarse predictions. This isolates the
one component that the round_030/031 comparison showed was under-trained.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset_benchmark import METRIC_CONTRACT
from screen_corner_all_history import AllHistoryScreenCorner, make_generic_encoder
from screen_corner_single import patch_boxes, sample_corner_patches
from training.screen_corner_ablation import fingerprint, read_rows, write_json
from training.train_screen_corner_single import hit_loss, score, training_order

REPO = Path(__file__).resolve().parents[3]


def now():
    return datetime.now(timezone.utc).isoformat()


def state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def load_frozen_base(model, artifact):
    """Copy encoder/coarse weights from a frozen artifact; keep a fresh local head."""
    base = torch.jit.load(str(artifact), map_location='cpu').state_dict()
    warm = {k: v for k, v in base.items()
            if k.startswith(('encoder.', 'global_head.')) or k in ('mean', 'std')}
    missing, unexpected = model.load_state_dict(warm, strict=False)
    if unexpected:
        raise ValueError('Unexpected base keys: ' + ', '.join(sorted(unexpected)))
    if not missing or any(not k.startswith('local_head.') for k in missing):
        raise ValueError('Only local_head parameters may be re-initialized')
    for module in (model.encoder, model.global_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return base


def configure(root, base):
    if (root / 'manifest.json').exists():
        return json.loads((root / 'manifest.json').read_text())
    dataset = json.loads((root / 'data/manifest.json').read_text())
    base = Path(base).resolve()
    pin = root / 'artifacts/pinned'
    pin.mkdir(parents=True, exist_ok=True)
    runtime = REPO / 'models/runtime/screen_corner_single.pt'
    shutil.copyfile(runtime, pin / 'current_runtime.pt')
    shutil.copyfile(REPO / 'models/runtime/screen_corner_single.json', pin / 'current_runtime.json')
    shutil.copyfile(base, root / 'artifacts/base_frozen.pt')
    train = read_rows(root / 'data/train.jsonl')
    cfg = {
        'round': root.name, 'architecture': 'screen_corner_local_refine_v1',
        'metric_contract': METRIC_CONTRACT,
        'purpose': 'refine-only continuation of the all-history candidate; targets the under-trained local head',
        'base_artifact': str(base), 'base_sha256': fingerprint(base),
        'base_model_id': 'screen_corner_single_' + fingerprint(base)[:12],
        'frozen_components': ['encoder', 'global_head'],
        'local_head_initialization': 'fresh LocalPointHead; base local head discarded',
        'seed': 7, 'batch_size': 32, 'local_epochs': 96, 'local_lr': .0003,
        'warmup_epochs': 5, 'local_gradient_clip_norm': 1.,
        'lr_schedule': 'linear warmup over 5 epochs to 3e-4, then constant',
        'local_loss': {'heatmap_mse': 1., 'relative_smooth_l1': 3.5, 'point_hit_surrogate': .1},
        'local_training_crops': 'frozen base coarse predictions on full-resolution source photographs; never manual-centered',
        'crop_geometry': 'patch_boxes span*0.15 unchanged; exported graph interface identical to round_030/031',
        'dataset': dataset, 'training_unique_pages': len(train),
        'training_samples_per_epoch': len(read_rows(root / 'data/train_sampling.jsonl')),
        'selection': 'pooled development point_le_0_01_ratio only; earliest epoch on ties; encoder and coarse frozen',
        'comparison_runtime': str(pin / 'current_runtime.pt'), 'comparison_runtime_sha256': fingerprint(runtime),
        'runtime_before': {p.name: fingerprint(p) for p in (REPO / 'models/runtime').glob('screen_corner_single.*')},
        'input_hashes': {str(p): fingerprint(p) for p in (root / 'data').glob('*.jsonl')},
        'status': 'configured', 'created_at': now(),
        'limitations': ['single seed', 'encoder and coarse inherited from a frozen candidate instead of retrained here',
            'historical validation and regression already observed',
            'refine-only round cannot improve region or coarse errors',
            'fixed training budget is not proof of convergence',
            'no automatic runtime promotion or packaging'],
    }
    write_json(root / 'manifest.json', cfg)
    for path in [Path(__file__), REPO / 'program/engine/screen_corner_all_history.py',
                 REPO / 'program/engine/screen_corner_single.py']:
        shutil.copyfile(path, root / 'artifacts' / path.name)
    return cfg


def build_cache(model, rows, device):
    """Crops come from the frozen model's own coarse predictions, never manual centers.

    The frozen model must be in eval mode: batch-norm statistics from a
    single-image training-mode pass would move the coarse points and the local
    head would learn on wrongly centred crops.
    """
    cache = {}
    model.eval()
    with torch.no_grad():
        for i, row in enumerate(rows):
            image = cv2.imread(row['image_path'])
            if image is None:
                raise FileNotFoundError(row['image_path'])
            rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())[None].float() / 255.
            small = F.interpolate(rgb, [256, 256], mode='bilinear', align_corners=False).to(device)
            coarse, _, _, feature = model.global_head(model.encoder((small - model.mean) / model.std))
            boxes = patch_boxes(coarse).cpu()
            patches = sample_corner_patches(rgb, boxes, 64)
            context = sample_corner_patches(feature.cpu(), boxes, 16)
            payload = {'boxes': boxes[0], 'coarse': coarse[0].cpu(), 'rgb': patches, 'context': context,
                       'target': torch.tensor(row['manual_quad'], dtype=torch.float32) / torch.tensor([row['image_width'], row['image_height']], dtype=torch.float32),
                       'size': torch.tensor([row['image_width'], row['image_height']], dtype=torch.float32)}
            if not all(torch.isfinite(payload[k]).all() for k in ['boxes', 'coarse', 'rgb', 'context', 'target']):
                raise FloatingPointError('Nonfinite predicted local cache: ' + row['image_path'])
            cache[row['image_path']] = payload
            if i % 200 == 0:
                print(f'predicted local crop cache {i+1}/{len(rows)}', flush=True)
    return cache


def local_score(model, rows, cache):
    model.local_head.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            entries = [cache[r['image_path']] for r in rows[start:start+32]]
            rgb = torch.cat([e['rgb'] for e in entries])
            context = torch.cat([e['context'] for e in entries])
            boxes = torch.stack([e['boxes'] for e in entries])
            relative, _ = model.local_head(rgb, context)
            output = (boxes[..., :2] + relative.view(-1, 4, 2) * (boxes[..., 2:] - boxes[..., :2])).clamp(0., 1.)
            predictions.extend(output.numpy())
    return score(rows, predictions)


def train(root):
    torch.set_num_threads(4)
    cfg = json.loads((root / 'manifest.json').read_text())
    if (root / 'reports/frozen_selection.json').exists():
        raise FileExistsError('Round frozen; use a new round to retrain')
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    rows = {k: read_rows(root / f'data/{k}.jsonl') for k in ['train', 'train_sampling', 'validation']}
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = AllHistoryScreenCorner(make_generic_encoder()).to(device)
    fresh_local = state(model.local_head)
    base = load_frozen_base(model, cfg['base_artifact'])
    base_digest = fingerprint(cfg['base_artifact'])
    if base_digest != cfg['base_sha256']:
        raise ValueError('Base artifact changed since configure')
    # The warm start must be exact on the frozen parts and untouched on local.
    for key, value in base.items():
        if key.startswith('local_head.'):
            continue
        assert torch.equal(model.state_dict()[key].cpu(), value.cpu()), key
    for key, value in model.local_head.state_dict().items():
        assert torch.equal(value.cpu(), fresh_local[key]), key
    torch.save({'base_sha256': base_digest, 'local_head_fresh_state': fresh_local,
                'frozen_encoder_and_coarse': True}, root / 'artifacts/warmstart_audit.pt')
    cache = build_cache(model, rows['train'] + rows['validation'], device)
    del base
    gc.collect()
    torch.mps.empty_cache() if device.type == 'mps' else None
    # Identity check: the freshly initialized local head must reproduce the
    # frozen coarse points. A train-mode batch-norm pass or mis-centred crops
    # break this invariant, so refuse to train when it drifts.
    model.cpu().eval()
    coarse_only = score(rows['validation'], [cache[r['image_path']]['coarse'].numpy() for r in rows['validation']])
    identity = local_score(model, rows['validation'], cache)
    if abs(identity['point_le_0_01_ratio'] - coarse_only['point_le_0_01_ratio']) > .02:
        raise AssertionError(f"Fresh local head does not reproduce coarse points: "
                             f"{identity['hit_corners']}/{identity['corners']} vs {coarse_only['hit_corners']}/{coarse_only['corners']}")
    write_json(root / 'reports/local_cache_identity.json', {'identity': identity, 'coarse': coarse_only,
        'policy': 'fresh local head must reproduce the frozen coarse points; tolerance 2 percentage points'})
    # MPS grid-sampler backward is numerically unstable on this cached workload;
    # the local head trains on CPU, exactly like round_030/031.
    model.cpu().eval()
    model.local_head.to('cpu')
    for module in (model.encoder, model.global_head):
        module.eval()
    optimizer = torch.optim.Adam(model.local_head.parameters(), lr=cfg['local_lr'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    summary = local_score(model, rows['validation'], cache)
    best, best_epoch, best_state = summary['point_le_0_01_ratio'], 0, state(model.local_head)
    history = [{'epoch': 0, 'validation': summary, 'lr': cfg['local_lr']}]
    latest = root / 'checkpoints/local_latest.pt'
    latest.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    if latest.exists():
        saved = torch.load(latest, map_location='cpu')
        model.local_head.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator']); best, best_epoch = saved['best'], saved['best_epoch']
        best_state, history = saved['best_state'], saved['history']; start_epoch = saved['epoch'] + 1
        print('RESUME', start_epoch, flush=True)
    axis = torch.linspace(0., 1., 16); gy, gx = torch.meshgrid(axis, axis, indexing='ij')
    for epoch in range(start_epoch, cfg['local_epochs'] + 1):
        started = time.perf_counter()
        lr = cfg['local_lr'] * min(1., epoch / cfg['warmup_epochs'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.local_head.train(); losses = []
        order = training_order(len(rows['train_sampling']), generator)
        for start in range(0, len(order), 32):
            entries = [cache[rows['train_sampling'][i]['image_path']] for i in order[start:start+32]]
            rgb = torch.cat([e['rgb'] for e in entries]); context = torch.cat([e['context'] for e in entries])
            target, sizes, boxes = [torch.stack([e[k] for e in entries]) for k in ['target', 'size', 'boxes']]
            relative, logits = model.local_head(rgb, context); span = boxes[..., 2:] - boxes[..., :2]
            target_relative = (target - boxes[..., :2]) / span
            inside = ((target_relative >= 0.) & (target_relative <= 1.)).all(-1).flatten()
            t = target_relative.reshape(-1, 2)
            distance = (gx[None] - t[:, 0, None, None]).square() + (gy[None] - t[:, 1, None, None]).square()
            heatmaps = torch.exp(-distance / (2 * (1.5 / 15)**2))[:, None]
            losses_hm = (logits - heatmaps).square().flatten(1).mean(1)
            heatmap_loss = (losses_hm * inside).sum() / inside.sum().clamp(min=1.)
            prediction = (boxes[..., :2] + relative.view(-1, 4, 2) * span).clamp(0., 1.)
            loss = heatmap_loss + 3.5 * F.smooth_l1_loss(relative, t.clamp(0., 1.)) + .1 * hit_loss(prediction, target, sizes)
            optimizer.zero_grad(); loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.local_head.parameters(), cfg['local_gradient_clip_norm'])
            if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f'Nonfinite local loss/gradient at epoch {epoch}; update skipped')
            optimizer.step(); losses.append(float(loss.detach()))
        summary = local_score(model, rows['validation'], cache)
        if summary['point_le_0_01_ratio'] > best:
            best, best_epoch, best_state = summary['point_le_0_01_ratio'], epoch, state(model.local_head)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'lr': lr, 'validation': summary,
                        'elapsed_seconds': time.perf_counter() - started})
        write_json(root / 'reports/local_history.json', history)
        torch.save({'model': state(model.local_head), 'optimizer': optimizer.state_dict(), 'generator': generator.get_state(),
            'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'best_state': best_state, 'history': history}, latest)
        cfg.update(status='training_local', latest_epoch=epoch); write_json(root / 'manifest.json', cfg)
        print(f'local {epoch}/{cfg["local_epochs"]} final={summary["hit_corners"]}/{summary["corners"]} '
              f'lr={lr:.2e} seconds={time.perf_counter()-started:.1f}', flush=True)
    model.local_head.load_state_dict(best_state); model.cpu().eval()
    artifact = root / 'checkpoints/screen_corner_single.pt'
    torch.jit.script(model).save(str(artifact))
    digest = fingerprint(artifact)
    write_json(root / 'reports/frozen_selection.json', {'artifact': str(artifact), 'sha256': digest,
        'model_id': 'screen_corner_single_' + digest[:12], 'coarse_epoch': None, 'local_epoch': best_epoch,
        'validation': history[best_epoch]['validation'], 'metric_contract': METRIC_CONTRACT,
        'base_artifact': cfg['base_artifact'], 'base_sha256': cfg['base_sha256'], 'base_model_id': cfg['base_model_id'],
        'encoder_and_coarse_frozen': True, 'local_head_reinitialized': True,
        'local_epochs': cfg['local_epochs'], 'local_lr': cfg['local_lr'], 'warmup_epochs': cfg['warmup_epochs'],
        'regression_used': False, 'frozen_at': now()})
    cfg.update(status='trained_frozen_awaiting_comparison', model_id='screen_corner_single_' + digest[:12],
               artifact_sha256=digest)
    write_json(root / 'manifest.json', cfg)
    print('FROZEN', cfg['model_id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['configure', 'train'])
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--base', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'configure':
        if not args.base:
            parser.error('configure requires --base')
        configure(root, args.base)
    else:
        train(root)
