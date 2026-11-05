"""Fresh all-history region/coarse/local training and current-runtime comparison.

No ScreenPDF task weights are read by training. Comparison is performed only
after the new artifact and its development selection have been frozen.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataset_benchmark import METRIC_CONTRACT, summarize_point_hits
from screen_corner_all_history import AllHistoryScreenCorner, make_generic_encoder
from screen_corner_single import patch_boxes, sample_corner_patches, SingleModelPredictor
from training.screen_corner_ablation import read_rows, write_rows, write_json, fingerprint
from training.train_screen_corner_single import hit_loss, training_order, score

REPO = Path(__file__).resolve().parents[3]


def now():
    return datetime.now(timezone.utc).isoformat()


def state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def configure(root):
    if (root / 'manifest.json').exists():
        return json.loads((root / 'manifest.json').read_text())
    dataset = json.loads((root / 'data/manifest.json').read_text())
    train = read_rows(root / 'data/train.jsonl')
    counts = Counter(r['project_key'] for r in train)
    # Every unique training photo appears every epoch, with a capped replay
    # floor for small sources. No one recent session gets a special multiplier.
    replay = [r for r in train for _ in range(min(4, max(1, math.ceil(24 / counts[r['project_key']]))))]
    write_rows(root / 'data/train_sampling.jsonl', replay)
    pin = root / 'artifacts/pinned'; pin.mkdir(parents=True, exist_ok=True)
    runtime = REPO / 'models/runtime/screen_corner_single.pt'
    shutil.copyfile(runtime, pin / 'current_runtime.pt')
    shutil.copyfile(REPO / 'models/runtime/screen_corner_single.json', pin / 'current_runtime.json')
    generic = Path.home() / '.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
    cfg = {'round': root.name, 'architecture': 'screen_corner_region_coarse_local_fresh_v1',
        'metric_contract': METRIC_CONTRACT, 'initialization': 'ImageNet MobileNetV3-small only; all task heads freshly randomized',
        'task_checkpoint_initialization': None, 'teacher_models': None, 'encoder_trainable': True,
        'generic_pretrain_path': str(generic), 'generic_pretrain_sha256': fingerprint(generic),
        'seed': 7, 'batch_size': 32, 'region_epochs': 12, 'coarse_epochs': 80, 'local_epochs': 48,
        'global_head_lr': .0008, 'encoder_lr': .00008, 'local_lr': .0003,
        'training_unique_pages': len(train), 'training_samples_per_epoch': len(replay), 'dataset': dataset,
        'selection': 'pooled development point_le_0_01_ratio only; earliest epoch on ties; final artifact after local training',
        'region_phase_policy': 'fixed-budget supervised initialization; not candidate selection',
        'coarse_loss': {'soft_heatmap_cross_entropy': 1., 'coordinate_smooth_l1': 5., 'complete_region_bce': .5, 'complete_region_dice': .5, 'boundary_bce': .1, 'point_hit_surrogate': .1},
        'local_loss': {'heatmap_mse': 1., 'relative_smooth_l1': 3.5, 'point_hit_surrogate': .1},
        'augmentation': 'none; preserve all manual full-region targets and occlusion',
        'local_training_crops': 'only new model predictions on full-resolution source photographs; never manual-centered',
        'comparison_runtime': str(pin / 'current_runtime.pt'), 'comparison_runtime_sha256': fingerprint(runtime),
        'runtime_before': {p.name: fingerprint(p) for p in (REPO / 'models/runtime').glob('screen_corner_single.*')},
        'input_hashes': {str(p): fingerprint(p) for p in (root / 'data').glob('*.jsonl')},
        'status': 'configured', 'created_at': now(),
        'limitations': ['single seed', 'historical validation and regression already observed',
            'within-project near-duplicate quarantine; cross-project recompressed near duplicates not exhaustively resolved',
            'joint change of data and architecture; comparison is product-level, not causal architecture ablation',
            'fixed training budget is not proof of convergence', 'no automatic runtime promotion or packaging']}
    write_json(root / 'manifest.json', cfg)
    for path in [Path(__file__), REPO / 'program/engine/screen_corner_all_history.py',
                 REPO / 'program/engine/screen_corner_single.py', REPO / 'program/engine/training/build_all_history_dataset.py']:
        shutil.copyfile(path, root / 'artifacts' / path.name)
    return cfg


def region_loss(boundary, truth):
    probability = boundary[:, :1].sigmoid(); region = truth[:, :1]
    dice = 1. - (2. * (probability * region).sum((1, 2, 3)) + 1.) / ((probability + region).sum((1, 2, 3)) + 1.)
    return (.5 * F.binary_cross_entropy_with_logits(boundary[:, :1], region) + .5 * dice.mean()
            + .1 * F.binary_cross_entropy_with_logits(boundary[:, 1:], truth[:, 1:]))


def cache_global(rows):
    cache = {}
    axis = torch.linspace(0., 1., 64)
    gy, gx = torch.meshgrid(axis, axis, indexing='ij')
    for i, row in enumerate(rows):
        path = row['image_path']; image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]
        rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())[None].float() / 255.
        rgb = F.interpolate(rgb, [256, 256], mode='bilinear', align_corners=False)[0]
        target = torch.tensor(row['manual_quad'], dtype=torch.float32) / torch.tensor([w, h])
        boundary = np.zeros((2, 64, 64), np.float32)
        polygon = np.round(target.numpy() * 63).astype(np.int32)
        cv2.fillConvexPoly(boundary[0], polygon, 1.); cv2.polylines(boundary[1], [polygon], True, 1., 2)
        distance = (gx[None] - target[:, 0, None, None]).square() + (gy[None] - target[:, 1, None, None]).square()
        maps = torch.exp(-distance / (2 * (1.25 / 63)**2))
        maps = maps / maps.flatten(1).sum(-1)[:, None, None].clamp(min=1e-8)
        cache[path] = {'rgb_global': rgb, 'target': target, 'size': torch.tensor([w, h]),
                       'boundary': torch.from_numpy(boundary), 'heatmap': maps}
        if i % 200 == 0:
            print(f'global image cache {i+1}/{len(rows)}', flush=True)
    return cache


def global_score(model, rows, cache, device):
    model.eval(); predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            image = torch.stack([cache[r['image_path']]['rgb_global'] for r in rows[start:start+32]]).to(device)
            prediction, _, _, _ = model.global_prediction(image)
            predictions.extend(prediction.cpu().numpy())
    return score(rows, predictions)


def train(root):
    torch.set_num_threads(4)
    cfg = configure(root)
    if (root / 'reports/frozen_selection.json').exists():
        raise FileExistsError('Round frozen; use a new round to retrain')
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    rows = {k: read_rows(root / f'data/{k}.jsonl') for k in ['train', 'train_sampling', 'validation']}
    cache = cache_global(rows['train'] + rows['validation'])
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = AllHistoryScreenCorner(make_generic_encoder()).to(device)
    initial_encoder = state(model.encoder)
    initial_task = state(model.global_head)
    torch.save({'encoder': initial_encoder, 'global_head': initial_task, 'local_head': state(model.local_head)}, root / 'artifacts/fresh_initialization.pt')
    optimizer = torch.optim.Adam([{'params': model.encoder.parameters(), 'lr': cfg['encoder_lr']},
                                 {'params': model.global_head.parameters(), 'lr': cfg['global_head_lr']}])
    generator = torch.Generator().manual_seed(cfg['seed'])
    latest = root / 'checkpoints/global_latest.pt'; latest.parent.mkdir(exist_ok=True)
    histories = {'region': [], 'coarse': []}; coarse_best = -1.; best_epoch = 0
    resume_phase, resume_epoch = 'region', 0
    if latest.exists():
        saved = torch.load(latest, map_location='cpu')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator']); histories = saved['histories']
        resume_phase, resume_epoch = saved['phase'], saved['epoch']
        coarse_best, best_epoch = saved['coarse_best'], saved['best_epoch']
        print('RESUME', resume_phase, resume_epoch, flush=True)
    for phase in ['region', 'coarse']:
        epochs = cfg[phase + '_epochs']
        if phase == 'region' and resume_phase == 'coarse':
            continue
        start_epoch = resume_epoch + 1 if phase == resume_phase else 1
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter(); model.train(); losses = []
            if phase == 'coarse':
                factor = .2 + .8 * .5 * (1. + math.cos(math.pi * (epoch - 1) / epochs))
                optimizer.param_groups[0]['lr'] = cfg['encoder_lr'] * factor
                optimizer.param_groups[1]['lr'] = cfg['global_head_lr'] * factor
            order = training_order(len(rows['train_sampling']), generator)
            for start in range(0, len(order), cfg['batch_size']):
                entries = [cache[rows['train_sampling'][i]['image_path']] for i in order[start:start+cfg['batch_size']]]
                image, target, sizes, maps, boundary = [torch.stack([e[k] for e in entries]).to(device)
                    for k in ['rgb_global', 'target', 'size', 'heatmap', 'boundary']]
                coarse, logits, predicted_boundary, _ = model.global_prediction(image)
                loss = region_loss(predicted_boundary, boundary)
                if phase == 'coarse':
                    loss = loss - (maps.flatten(2) * F.log_softmax(logits.flatten(2), -1)).sum(-1).mean()
                    loss = loss + 5. * F.smooth_l1_loss(coarse, target) + .1 * hit_loss(coarse, target, sizes)
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            summary = global_score(model, rows['validation'], cache, device)
            histories[phase].append({'epoch': epoch, 'loss': float(np.mean(losses)), 'validation_coarse': summary,
                                     'elapsed_seconds': time.perf_counter() - started})
            if phase == 'coarse' and summary['point_le_0_01_ratio'] > coarse_best:
                coarse_best, best_epoch = summary['point_le_0_01_ratio'], epoch
                torch.save({'model': state(model), 'best_epoch': epoch, 'validation': summary}, root / 'checkpoints/global_best.pt')
            write_json(root / f'reports/{phase}_history.json', histories[phase])
            torch.save({'model': state(model), 'optimizer': optimizer.state_dict(), 'generator': generator.get_state(),
                'histories': histories, 'phase': phase, 'epoch': epoch, 'coarse_best': coarse_best, 'best_epoch': best_epoch}, latest)
            cfg.update(status='training_' + phase, latest_epoch=epoch); write_json(root / 'manifest.json', cfg)
            print(f'{phase} {epoch}/{epochs} coarse={summary["hit_corners"]}/{summary["corners"]} seconds={time.perf_counter()-started:.1f}', flush=True)
    best = torch.load(root / 'checkpoints/global_best.pt', map_location='cpu')
    model.load_state_dict(best['model']); model.eval()
    encoder_changed = any(not torch.equal(v.cpu(), initial_encoder[k]) for k, v in model.encoder.state_dict().items())
    assert encoder_changed
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.global_head.parameters():
        p.requires_grad_(False)
    write_json(root / 'reports/coarse_frozen_selection.json', {'best_epoch': best_epoch, 'validation': best['validation'],
        'selection': cfg['selection'], 'encoder_retrained': True, 'frozen_at': now(), 'regression_used': False})
    # Use full resolution only for predicted local crops. Global input cache
    # exactly matches the graph's bilinear resizing, including camera aspect.
    with torch.no_grad():
        for i, row in enumerate(rows['train'] + rows['validation']):
            entry = cache[row['image_path']]
            coarse, _, _, feature = model.global_prediction(entry['rgb_global'][None].to(device))
            boxes = patch_boxes(coarse)
            image = cv2.imread(row['image_path'])
            rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())[None].float() / 255.
            entry.update(boxes=boxes[0].cpu(), coarse=coarse[0].cpu(), rgb=sample_corner_patches(rgb, boxes.cpu(), 64),
                         context=sample_corner_patches(feature, boxes, 16).cpu())
            if not all(torch.isfinite(entry[k]).all() for k in ['boxes', 'coarse', 'rgb', 'context']):
                raise FloatingPointError('Nonfinite predicted local cache: ' + row['image_path'])
            for key in ['rgb_global', 'heatmap', 'boundary']:
                del entry[key]
            if i % 200 == 0:
                print(f'predicted local crop cache {i+1}/{len(cache)}', flush=True)
    gc.collect(); torch.mps.empty_cache() if device.type == 'mps' else None
    # MPS grid-sampler backward is unstable on this cached local-crop workload;
    # keep the frozen global result and train the local head on CPU instead.
    local_device = torch.device('cpu')
    model.local_head.to(local_device)
    def local_score():
        model.local_head.eval(); predictions = []
        with torch.no_grad():
            for start in range(0, len(rows['validation']), 32):
                entries = [cache[r['image_path']] for r in rows['validation'][start:start+32]]
                rgb = torch.cat([e['rgb'] for e in entries]).to(local_device)
                context = torch.cat([e['context'] for e in entries]).to(local_device)
                relative, _ = model.local_head(rgb, context)
                boxes = torch.stack([e['boxes'] for e in entries]).to(local_device)
                output = (boxes[..., :2] + relative.view(-1, 4, 2) * (boxes[..., 2:] - boxes[..., :2])).clamp(0., 1.)
                predictions.extend(output.cpu().numpy())
        return score(rows['validation'], predictions)
    optimizer = torch.optim.Adam(model.local_head.parameters(), lr=cfg['local_lr'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    summary = local_score(); local_best = summary['point_le_0_01_ratio']; local_epoch = 0
    local_state = state(model.local_head); history = [{'epoch': 0, 'validation': summary}]
    latest_local = root / 'checkpoints/local_latest.pt'; start_epoch = 1
    if latest_local.exists():
        saved = torch.load(latest_local, map_location='cpu')
        model.local_head.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator']); local_best, local_epoch = saved['best'], saved['best_epoch']
        local_state, history = saved['best_state'], saved['history']; start_epoch = saved['epoch'] + 1
    axis = torch.linspace(0., 1., 16, device=local_device); gy, gx = torch.meshgrid(axis, axis, indexing='ij')
    for epoch in range(start_epoch, cfg['local_epochs'] + 1):
        started = time.perf_counter(); model.local_head.train(); losses = []
        order = training_order(len(rows['train_sampling']), generator)
        for start in range(0, len(order), 32):
            entries = [cache[rows['train_sampling'][i]['image_path']] for i in order[start:start+32]]
            rgb = torch.cat([e['rgb'] for e in entries]).to(local_device); context = torch.cat([e['context'] for e in entries]).to(local_device)
            target, sizes, boxes = [torch.stack([e[k] for e in entries]).to(local_device) for k in ['target', 'size', 'boxes']]
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
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.local_head.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f'Nonfinite local loss/gradient at epoch {epoch}; update skipped')
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        summary = local_score()
        if summary['point_le_0_01_ratio'] > local_best:
            local_best, local_epoch, local_state = summary['point_le_0_01_ratio'], epoch, state(model.local_head)
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'validation': summary,
                        'elapsed_seconds': time.perf_counter() - started})
        write_json(root / 'reports/local_history.json', history)
        torch.save({'model': state(model.local_head), 'optimizer': optimizer.state_dict(), 'generator': generator.get_state(),
            'epoch': epoch, 'best': local_best, 'best_epoch': local_epoch, 'best_state': local_state, 'history': history}, latest_local)
        cfg.update(status='training_local', latest_epoch=epoch); write_json(root / 'manifest.json', cfg)
        print(f'local {epoch}/{cfg["local_epochs"]} final={summary["hit_corners"]}/{summary["corners"]} seconds={time.perf_counter()-started:.1f}', flush=True)
    model.local_head.load_state_dict(local_state); model.cpu().eval()
    artifact = root / 'checkpoints/screen_corner_single.pt'; torch.jit.script(model).save(str(artifact))
    digest = fingerprint(artifact)
    write_json(root / 'reports/frozen_selection.json', {'artifact': str(artifact), 'sha256': digest,
        'model_id': 'screen_corner_single_' + digest[:12], 'coarse_epoch': best_epoch, 'local_epoch': local_epoch,
        'validation': history[local_epoch]['validation'], 'metric_contract': METRIC_CONTRACT,
        'encoder_retrained': encoder_changed, 'all_task_heads_random_initialization': True,
        'teacher_or_current_weights_used': False, 'regression_used': False, 'frozen_at': now()})
    cfg.update(status='trained_frozen_awaiting_comparison', model_id='screen_corner_single_' + digest[:12], artifact_sha256=digest)
    write_json(root / 'manifest.json', cfg)
    print('FROZEN', cfg['model_id'], flush=True)


def evaluate_artifact(path, rows):
    model = SingleModelPredictor(path); records = []; times = []
    for i, row in enumerate(rows):
        image = cv2.imread(row['image_path']); started = time.perf_counter()
        if image is None:
            prediction = {'coarse_quad': None, 'final_quad': None}
        else:
            output = model.predict(image); prediction = {k: v.tolist() for k, v in output.items()}
        times.append(time.perf_counter() - started)
        records.append(dict(row, **prediction))
        if i % 200 == 0:
            print(f'{path.parent.name} full graph {i+1}/{len(rows)}', flush=True)
    return records, {'mean_ms': float(np.mean(times) * 1000), 'device': 'cpu', 'pages': len(rows)}


def summarize(records):
    return {stage: summarize_point_hits([r['manual_quad'] for r in records], [r.get(stage + '_quad') for r in records])
            for stage in ['coarse', 'final']}


def collection_key(row):
    if row.get('collection_id'):
        return row['collection_id']
    path = row['image_path']
    if '南京-第十届未来网络发展大会' in path:
        return '202610-nanjing-future-network'
    if '杭州-隧道照明与安全论坛' in path:
        return '202610-hangzhou-tunnel'
    return row.get('dataset_slug', 'unassigned')


def compare(root):
    torch.set_num_threads(4)
    cfg = json.loads((root / 'manifest.json').read_text()); frozen = json.loads((root / 'reports/frozen_selection.json').read_text())
    artifact = Path(frozen['artifact']); assert fingerprint(artifact) == frozen['sha256']
    assert fingerprint(cfg['comparison_runtime']) == cfg['comparison_runtime_sha256']
    # All historical manual pages: includes fitting diagnostics, never counted
    # as a fresh held-out test. Roles are reported separately below.
    allrows = read_rows(root / 'data/all_manual.jsonl'); roles = {r['image_sha256']: role for role in ['train', 'validation', 'holdout']
        for r in read_rows(root / f'data/{role}.jsonl')}
    for row in allrows:
        row['new_dataset_role'] = roles.get(row['image_sha256'], 'quarantined_train')
    report = {'metric_contract': METRIC_CONTRACT, 'frozen': frozen, 'models': {}, 'comparison_current_model': cfg['comparison_runtime_sha256'],
              'interpretation': 'same images and labels; all-manual aggregate includes training and is diagnostic only'}
    for label, path in [('current_runtime_round027', Path(cfg['comparison_runtime'])), ('new_fresh_all_history', artifact)]:
        output = root / f'reports/{label}_predictions.json'
        if output.exists():
            records = json.loads(output.read_text()); timing = None
        else:
            records, timing = evaluate_artifact(path, allrows); write_json(output, records)
        result = {'all_manual_diagnostic': summarize(records), 'roles': {}, 'projects': {}, 'collections': {}, 'latency_observation': timing}
        for role in ['train', 'validation', 'holdout', 'quarantined_train']:
            result['roles'][role] = summarize([r for r in records if r['new_dataset_role'] == role])
        for key in ['project_key', 'collection_id']:
            group = defaultdict(list)
            for record in records:
                group[collection_key(record) if key == 'collection_id' else record['project_key']].append(record)
            for name, entries in group.items():
                result['projects' if key == 'project_key' else 'collections'][name] = {
                    'all_manual_diagnostic': summarize(entries),
                    'roles': {role: summarize([r for r in entries if r['new_dataset_role'] == role]) for role in ['train', 'validation', 'holdout']}}
        frozen_pages = read_rows(REPO / 'data/curated/collections/conference_v1/pages.jsonl')
        frozen_hashes = {r['image_sha256'] for r in frozen_pages}
        subset = [r for r in records if r['image_sha256'] in frozen_hashes]
        # One frozen page carries a quad 0.77 px outside the photograph, so the
        # dataset revision excludes it before training. Score it here with the
        # frozen official quad: the 1808-page diagnostic keeps the same pages,
        # denominator and metric as earlier rounds. Selection is already frozen.
        presented = {r['image_sha256'] for r in records}
        missing = [r for r in frozen_pages if r['image_sha256'] not in presented]
        if missing:
            subset = subset + evaluate_artifact(path, missing)[0]
        assert len(subset) == 1808
        result['frozen_conference1808_diagnostic'] = summarize(subset)
        result['frozen_subset_not_in_dataset_revision'] = [r['image_path'] for r in missing]
        previous_data = REPO / 'training/runs/screen_corner_single/round_029/data/with_computing'
        previous_validation = {str(Path(r['image_path']).resolve()) for r in read_rows(previous_data / 'validation.jsonl')}
        result['previous428_development'] = summarize([r for r in records if r['image_path'] in previous_validation])
        report['models'][label] = result
    assert report['models']['new_fresh_all_history']['roles']['validation']['final'] == frozen['validation']
    for p, digest in cfg['dataset']['inputs'].items():
        assert fingerprint(p) == digest, p
    for p, digest in cfg['input_hashes'].items():
        assert fingerprint(p) == digest, p
    assert all(fingerprint(REPO / 'models/runtime' / n) == h for n, h in cfg['runtime_before'].items())
    report.update(completed_at=now(), runtime_unchanged=True, status='compared_candidate_not_promoted', limitations=cfg['limitations'])
    write_json(root / 'reports/comparison.json', report)
    write_json(REPO / f'training/registry/screen_corner_single_v1_{root.name.replace("_", "")}.json', report)
    cfg.update(status=report['status'], completed_at=report['completed_at']); write_json(root / 'manifest.json', cfg)
    print('COMPARISON COMPLETE', json.dumps({k: v['roles']['validation'] for k, v in report['models'].items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['configure', 'train', 'compare'])
    parser.add_argument('--root', type=Path, required=True); args = parser.parse_args()
    {'configure': configure, 'train': train, 'compare': compare}[args.action](args.root.resolve())
