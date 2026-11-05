"""Reproducible, isolated screen-corner ablations; never promotes runtime weights."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.mps
from torch import nn
import torch.nn.functional as F

ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parents[1]
sys.path.insert(0, str(ENGINE))
from corner_train import CornerHeatmapNet, build_corner_heatmaps, decode_heatmaps, select_torch_device
from dataset_benchmark import METRIC_CONTRACT, load_manual_pages, quad_geometry_metrics, summarize_geometry_metric_rows
from global_corner_train import build_global_feature_tensor
from two_stage_corner_pipeline import GlobalCornerPredictor, RoiCornerPredictor, LocalCornerMoEPredictor, predict_two_stage
from detect_frame import TEACHER_MULTI_EXPAND_RATIOS

NEW_SOURCE = Path('/Users/gcssloop/WorkSpace/AIGC/展会/2026/202610 追光卅载铸根基，AI赋能启新篇')
SLUG = '202610-zhuiguang-ai'
BASE_SPLIT = REPO / 'training/runs/global_corner_r144_202604_mall_curriculum_stage2_from_stage1_v1/split'
BASE_MODEL = next((REPO / 'training/runs').glob('global_corner_r142_*/output/global_corner_model.pt'),
                  REPO / 'models/runtime/global_corner_model.pt')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def unique_rows(rows):
    return list({str(Path(row['image_path']).resolve()): row for row in rows}.values())


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    if (root / 'data/manifest.json').exists():
        raise FileExistsError('Data snapshot already exists; use its manifests without re-splitting.')
    pages = load_manual_pages(NEW_SOURCE)
    if not pages:
        raise ValueError('No human supervision found')
    # Hash the rectified screen, not the surrounding camera frame. Group near duplicates.
    hashes = []
    for page in pages:
        image = cv2.imread(page['image_path'])
        h, w = image.shape[:2]
        quad = np.asarray(page['manual_quad'], np.float32)
        dst = np.array([[0, 0], [127, 0], [127, 71], [0, 71]], np.float32)
        crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(quad, dst), (128, 72))
        gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (9, 8))
        hashes.append(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes())
        page.update(image_width=w, image_height=h, dataset_slug=SLUG,
                    annotation_source='explicit_manual_quad', image_sha256=fingerprint(page['image_path']))
    parents = list(range(len(pages)))
    def find(i):
        while parents[i] != i:
            i = parents[i]
        return i
    for i in range(len(pages)):
        for j in range(i):
            if sum((a ^ b).bit_count() for a, b in zip(hashes[i], hashes[j])) <= 5:
                parents[find(i)] = find(j)
    groups = defaultdict(list)
    for i, page in enumerate(pages):
        groups[find(i)].append(page)
    ordered = list(groups.values())
    random.Random(20261017).shuffle(ordered)
    partitions = {'holdout': [], 'validation': [], 'train': []}
    for group in ordered:
        name = 'holdout' if len(partitions['holdout']) < 5 else 'validation' if len(partitions['validation']) < 5 else 'train'
        gid = hashlib.sha256('|'.join(sorted(p['image_sha256'] for p in group)).encode()).hexdigest()[:12]
        for page in group:
            page.update(split=name, capture_group=gid)
        partitions[name].extend(group)
    if any(not value for value in partitions.values()):
        raise ValueError('Too few independent groups for train/validation/holdout')
    old_train = unique_rows(read_rows(BASE_SPLIT / 'train.jsonl') + read_rows(BASE_SPLIT / 'focus_train.jsonl'))
    old_val = unique_rows(read_rows(BASE_SPLIT / 'test.jsonl') + read_rows(BASE_SPLIT / 'focus_test.jsonl'))
    old_holdout = read_rows(BASE_SPLIT / 'holdout.jsonl')
    files = {'train_old': old_train, 'train': old_train + partitions['train'],
             'validation': old_val + partitions['validation'], 'new_train': partitions['train'],
             'new_validation': partitions['validation'], 'holdout': partitions['holdout'],
             'old_validation': old_val, 'mall_holdout': old_holdout}
    train_ids = {str(Path(r['image_path']).resolve()) for r in files['train']}
    for name in ['validation', 'holdout', 'mall_holdout']:
        if train_ids & {str(Path(r['image_path']).resolve()) for r in files[name]}:
            raise ValueError(f'Train overlap with {name}')
    for name, rows in files.items():
        write_rows(root / f'data/{name}.jsonl', rows)
    annotation_file = REPO / f'data/curated/annotations/{SLUG}_pages.jsonl'
    write_rows(annotation_file, pages)
    alias = REPO / f'data/raw/{SLUG}'
    if not alias.exists():
        alias.symlink_to(NEW_SOURCE, target_is_directory=True)
    manifest = {'dataset_slug': SLUG, 'source_root': str(NEW_SOURCE), 'raw_alias': str(alias),
                'source_project_sha256': fingerprint(NEW_SOURCE / 'screen-pdf-project.json'),
                'annotation_file': str(annotation_file), 'page_count': len(pages),
                'split_counts': {k: len(v) for k, v in partitions.items()}, 'similarity_groups': len(groups),
                'split_policy': 'rectified-screen dHash <=5 grouped; fixed seed 20261017; same-event evaluation only',
                'metric_contract': METRIC_CONTRACT, 'files': {k: len(v) for k, v in files.items()},
                'source_base_split': str(BASE_SPLIT), 'baseline_model': str(BASE_MODEL),
                'baseline_sha256': fingerprint(BASE_MODEL), 'new_holdout_opened': False}
    write_json(root / 'data/manifest.json', manifest)
    write_json(REPO / f'data/splits/cross_project/{SLUG}_v1.json', {'metadata': manifest, **partitions})
    write_json(REPO / f'training/registry/{SLUG}.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def stage_metrics(manual, result):
    debug = result['diagnostics']
    quads = {'coarse': result['coarse_quad'], 'roi': debug['baseline_roi_quad'],
             'local': debug['baseline_local_quad'], 'selected': result['final_quad']}
    pool = [result['coarse_quad']] + [q for c in debug['candidates'] for q in [c['roi_quad'], c['final_quad']]]
    scores = [quad_geometry_metrics(manual, q) for q in pool]
    def rank(m):
        errors = [m[f'corner_error_{i:02d}'] for i in range(4)]
        return (-sum(e <= .01 for e in errors), max(errors), m['point_error'])
    metrics = {k: quad_geometry_metrics(manual, q) for k, q in quads.items()}
    metrics['candidate_oracle'] = min(scores, key=rank)
    return metrics, quads


def local_peak_decode(logits, radius=3):
    batch, channels, height, width = logits.shape
    flat = logits.reshape(batch, channels, -1)
    peaks = flat.argmax(-1)
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device), torch.arange(width, device=logits.device), indexing='ij')
    mask = ((xx.flatten()[None, None] - (peaks % width)[..., None]).abs() <= radius) & (
        (yy.flatten()[None, None] - (peaks // width)[..., None]).abs() <= radius)
    weights = torch.softmax((flat * 12).masked_fill(~mask, -1e9), -1)
    return torch.stack([(weights * xx.flatten()).sum(-1)/(width-1),
                        (weights * yy.flatten()).sum(-1)/(height-1)], -1)


class ExperimentPredictor:
    def __init__(self, path, decode='soft_argmax'):
        self.device = select_torch_device()
        checkpoint = torch.load(path, map_location='cpu')
        self.decode = decode
        self.encoder = None
        if 'condition' not in checkpoint:
            self.legacy = GlobalCornerPredictor(path)
            self.model = self.legacy.model
        elif checkpoint['condition'].startswith('legacy'):
            self.legacy = None
            self.model = CornerHeatmapNet(in_channels=5, channels=24, output_channels=4)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.legacy = None
            self.encoder = make_encoder(False).cpu()
            self.encoder.load_state_dict(checkpoint['encoder_state'])
            self.model = PyramidCornerHead()
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device).eval()
        if self.encoder is not None:
            self.encoder.eval()

    def predict_image(self, image):
        height, width = image.shape[:2]
        if self.legacy is not None and self.decode == 'soft_argmax':
            return self.legacy.predict_image(image)
        image = cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            if self.encoder is None:
                tensor = torch.from_numpy(build_global_feature_tensor(image, 'rgb_gray_border')[None]).to(self.device)
                logits = self.model(tensor)
            else:
                tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()[None]).float()/255
                mean = tensor.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
                std = tensor.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
                logits, _ = self.model([v.to(self.device) for v in self.encoder((tensor-mean)/std).values()])
            decoder = local_peak_decode if self.decode == 'local_peak' else decode_heatmaps
            return decoder(logits).cpu().numpy()[0] * np.array([width, height])


def diagnose(root, model_path=BASE_MODEL, label='r142', include_holdout=False, decode='soft_argmax', holdout_only=False):
    paths = ['new_train', 'new_validation', 'old_validation', 'mall_holdout']
    if include_holdout:
        paths.append('holdout')
    if holdout_only:
        paths = ['holdout']
    pages = unique_rows([r for name in paths for r in read_rows(root / f'data/{name}.jsonl')])
    global_model = ExperimentPredictor(model_path, decode)
    roi = RoiCornerPredictor(REPO / 'models/runtime/corner_heatmap_model.pt')
    local = LocalCornerMoEPredictor(REPO / 'models/runtime/local_corner_moe_coord_model.pt')
    records = []
    for index, page in enumerate(pages):
        image = cv2.imread(page['image_path'])
        if image is None:
            raise FileNotFoundError(page['image_path'])
        start = time.perf_counter()
        result = predict_two_stage(Path(page['image_path']), global_model, roi, local,
                                   image=image, candidate_expand_ratios=TEACHER_MULTI_EXPAND_RATIOS,
                                   return_diagnostics=True)
        metrics, quads = stage_metrics(page['manual_quad'], result)
        records.append({'image_path': page['image_path'], 'page_id': page['page_id'],
                        'project_name': page['project_name'], 'elapsed_ms': (time.perf_counter()-start)*1000,
                        'stages': metrics, 'quads': quads, 'manual_quad': page['manual_quad']})
        if index % 25 == 0:
            print(f'{label} diagnostic {index+1}/{len(pages)}', flush=True)
    sections = {}
    for name in paths:
        ids = {r['image_path'] for r in read_rows(root / f'data/{name}.jsonl')}
        subset = [r for r in records if r['image_path'] in ids]
        sections[name] = {stage: summarize_geometry_metric_rows([r['stages'][stage] for r in subset])
                          for stage in ['coarse', 'roi', 'local', 'selected', 'candidate_oracle']}
    write_json(root / f'reports/{label}_stage_diagnostics.json',
               {'metric_contract': METRIC_CONTRACT, 'model_path': str(model_path),
                'model_sha256': fingerprint(model_path), 'candidate_expand_ratios': TEACHER_MULTI_EXPAND_RATIOS,
                'device': global_model.device.type, 'decode_mode': decode, 'torch_threads': torch.get_num_threads(),
                'stage_definitions': {'roi': 'baseline expansion 0.08 before local', 'local': 'baseline expansion 0.08 after local',
                                      'selected': 'actual runtime multi-expansion selection',
                                      'candidate_oracle': 'offline only: best of coarse and every ROI/local proposal'},
                'timing_scope': 'diagnostic wall time; not an isolated performance certification',
                'sections': sections, 'pages': records})
    print(label, json.dumps({name: {stage: {k: value[k] for k in ['point_le_0_01_ratio', 'quad_le_0_01_ratio', 'point_error_mean']} for stage, value in stages.items()} for name, stages in sections.items()}, ensure_ascii=False), flush=True)


class PyramidCornerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, 32, 1) for c in [24, 48, 576]])
        self.fuse = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.corners = nn.Conv2d(32, 4, 1)
        # Both ablation cells construct this branch to keep initialization identical.
        self.boundary = nn.Conv2d(32, 2, 1)

    def forward(self, features):
        fused = sum(F.interpolate(layer(feature), size=(64, 64), mode='bilinear', align_corners=False)
                    for layer, feature in zip(self.laterals, features))
        fused = self.fuse(fused)
        return self.corners(fused), self.boundary(fused)


def make_encoder(pretrained):
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    encoder = create_feature_extractor(model.features, return_nodes={'3': 'low', '8': 'mid', '12': 'high'})
    return encoder.eval().requires_grad_(False)


def collect_cache(rows):
    cache = {}
    for index, row in enumerate(rows):
        image = cv2.imread(row['image_path'])
        if image is None:
            raise FileNotFoundError(row['image_path'])
        height, width = image.shape[:2]
        image = cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_BGR2RGB)
        corners = np.array(row['manual_quad'], np.float32) / [width, height]
        target = np.zeros((2, 64, 64), np.float32)
        points = np.round(corners * 63).astype(np.int32)
        cv2.fillConvexPoly(target[0], points, 1.)
        cv2.polylines(target[1], [points], True, 1., 2)
        cache[row['image_path']] = {'rgb': torch.from_numpy(image.transpose(2, 0, 1).copy()).float()/255,
                                   'legacy': torch.from_numpy(build_global_feature_tensor(image, 'rgb_gray_border')),
                                   'corners': torch.tensor(corners, dtype=torch.float32),
                                   'heatmap': torch.from_numpy(build_corner_heatmaps(corners)),
                                   'boundary': torch.from_numpy(target), 'size': np.array([width, height])}
        if index % 200 == 0:
            print(f'image cache {index+1}/{len(rows)}', flush=True)
    return cache


def extract_features(encoder, rows, cache, device, batch_size=32):
    # PyTorch 2.0 MPS lacks MobileNet's hardsigmoid; frozen extraction is on CPU.
    # Trainable heads still use the requested accelerator. This is identical in
    # the pretrained/random/boundary cells, without global operator fallbacks.
    device = torch.device('cpu')
    encoder = encoder.to(device)
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
    features = {}
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset+batch_size]
            images = torch.stack([cache[r['image_path']]['rgb'] for r in batch]).to(device)
            output = list(encoder((images-mean)/std).values())
            for i, row in enumerate(batch):
                features[row['image_path']] = [value[i].cpu() for value in output]
    return features


def model_forward(model, paths, cache, features, device):
    if features is None:
        return model(torch.stack([cache[p]['legacy'] for p in paths]).to(device)), None
    batch = [torch.stack([features[p][i] for p in paths]).to(device) for i in range(3)]
    return model(batch)


def evaluate_cell(model, rows, cache, features, device):
    model.eval()
    records = []
    with torch.no_grad():
        for offset in range(0, len(rows), 32):
            batch = rows[offset:offset+32]
            paths = [r['image_path'] for r in batch]
            logits, _ = model_forward(model, paths, cache, features, device)
            predicted = decode_heatmaps(logits).cpu().numpy()
            for row, pred in zip(batch, predicted):
                metrics = quad_geometry_metrics(row['manual_quad'], pred * cache[row['image_path']]['size'])
                records.append({'project_name': row['project_name'], 'image_path': row['image_path'], 'metrics': metrics})
    projects = defaultdict(list)
    for item in records:
        projects[item['project_name']].append(item['metrics'])
    summaries = {key: summarize_geometry_metric_rows(values) for key, values in projects.items()}
    return {'summary': summarize_geometry_metric_rows([r['metrics'] for r in records]), 'projects': summaries,
            'macro_point_hit': float(np.mean([s['point_le_0_01_ratio'] for s in summaries.values()])),
            'macro_quad_hit': float(np.mean([s['quad_le_0_01_ratio'] for s in summaries.values()]))}


def train_experiments(root, seeds, epochs, conditions=None):
    allowed = {'legacy_old', 'legacy_new', 'pyramid_random', 'pyramid_pretrained', 'pyramid_boundary'}
    if epochs < 1 or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Use positive epochs and distinct seeds')
    if conditions is not None and (not conditions or len(conditions) != len(set(conditions)) or set(conditions)-allowed):
        raise ValueError(f'Conditions must be distinct members of {sorted(allowed)}')
    device = select_torch_device()
    train_old = read_rows(root / 'data/train_old.jsonl')
    train_new = read_rows(root / 'data/train.jsonl')
    validation = read_rows(root / 'data/validation.jsonl')
    rows = unique_rows(train_new + validation)
    cache = collect_cache(rows)
    settings = {'seeds': seeds, 'epochs': epochs, 'batch_size': 32, 'image_size': 256,
                'heatmap_size': 64, 'augmentation': 'none (controlled screening)',
                'loss': 'heatmap MSE + 2.5 * coordinate SmoothL1; optional 0.1 * boundary BCE',
                'teacher_guidance': False, 'metric_contract': METRIC_CONTRACT,
                'selection': 'pooled validation point_le_0_01_ratio only; keep earlier checkpoint on ties',
                'device': str(device), 'torch': torch.__version__, 'threads': torch.get_num_threads(),
                'data_hashes': {name: fingerprint(root / f'data/{name}.jsonl') for name in ['train_old', 'train', 'validation', 'holdout']},
                'conditions': conditions or ['legacy_old', 'legacy_new', 'pyramid_random', 'pyramid_pretrained', 'pyramid_boundary'],
                'interpretation': 'fixed-budget screening; random/pretrained pyramids share architecture; boundary pair shares exact initialization; architecture-vs-legacy changes more than pretraining alone'}
    settings_path = root / 'manifest.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('Existing experiment settings differ; use a new run directory')
    write_json(settings_path, settings)
    completed = []
    for seed in seeds:
        for condition in settings['conditions']:
            cell = root / f'cells/{condition}/seed_{seed:03d}'
            if (cell / 'result.json').exists():
                completed.append(json.loads((cell / 'result.json').read_text()))
                continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            is_legacy = condition.startswith('legacy')
            pretrained = condition in ['pyramid_pretrained', 'pyramid_boundary']
            boundary_weight = .1 if condition == 'pyramid_boundary' else 0.
            encoder = None
            features = None
            if is_legacy:
                ckpt = torch.load(BASE_MODEL, map_location='cpu')
                model = CornerHeatmapNet(in_channels=5, channels=24, output_channels=4)
                model.load_state_dict(ckpt['state_dict'])
                learning_rate = 2e-5
            else:
                # Reset seeds before constructing the head so pretrained-vs-random
                # differs only in the frozen encoder, and boundary pairs match.
                encoder = make_encoder(pretrained)
                torch.manual_seed(seed)
                model = PyramidCornerHead()
                features = extract_features(encoder, rows, cache, device)
                encoder.cpu()
                learning_rate = 1e-3
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            train_rows = train_old if condition == 'legacy_old' else train_new
            generator = torch.Generator().manual_seed(seed)
            history, best_key, best_eval, best_epoch = [], None, None, 0
            cell.mkdir(parents=True, exist_ok=True)
            initial = evaluate_cell(model, validation, cache, features, device)
            start = time.perf_counter()
            for epoch in range(1, epochs+1):
                model.train()
                losses = []
                order = torch.randperm(len(train_rows), generator=generator).tolist()
                for offset in range(0, len(order), 32):
                    paths = [train_rows[i]['image_path'] for i in order[offset:offset+32]]
                    logits, boundaries = model_forward(model, paths, cache, features, device)
                    targets = torch.stack([cache[p]['heatmap'] for p in paths]).to(device)
                    corners = torch.stack([cache[p]['corners'] for p in paths]).to(device)
                    pred = decode_heatmaps(logits)
                    loss = F.mse_loss(logits, targets) + 2.5 * F.smooth_l1_loss(pred, corners)
                    if boundary_weight:
                        boundary_targets = torch.stack([cache[p]['boundary'] for p in paths]).to(device)
                        loss = loss + boundary_weight * F.binary_cross_entropy_with_logits(boundaries, boundary_targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                evaluation = evaluate_cell(model, validation, cache, features, device)
                key = (evaluation['summary']['point_le_0_01_ratio'],)
                history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation': evaluation})
                if best_key is None or key > best_key:
                    best_key, best_eval, best_epoch = key, evaluation, epoch
                    payload = {'condition': condition, 'seed': seed, 'epoch': epoch,
                               'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                               'metric_contract': METRIC_CONTRACT, 'input_size': 256,
                               'encoder_state': encoder.state_dict() if encoder is not None else None,
                               'decode_mode': 'soft_argmax', 'learning_rate': learning_rate}
                    torch.save(payload, cell / 'model.pt')
                write_json(cell / 'history.json', history)
                print(f'{condition} seed={seed} epoch={epoch}/{epochs} point_hit={key[0]:.4f}', flush=True)
            result = {'condition': condition, 'seed': seed, 'best_epoch': best_epoch, 'epochs': epochs,
                      'learning_rate': learning_rate, 'train_pages': len(train_rows),
                      'seconds': time.perf_counter()-start, 'initial_validation': initial,
                      'validation': best_eval, 'model_path': str(cell / 'model.pt'), 'model_sha256': fingerprint(cell / 'model.pt')}
            write_json(cell / 'result.json', result)
            completed.append(result)
            del model, features, encoder
            if device.type == 'mps':
                torch.mps.empty_cache()
    write_json(root / 'reports/ablation_results.json', {'settings': settings, 'cells': completed})


def finish(root, additional_root=None):
    roots = [root] + ([additional_root] if additional_root else [])
    cells = []
    for run in roots:
        report = run / 'reports/ablation_results.json'
        if not report.exists():
            raise RuntimeError(f'Training is not complete: {run}')
        cells.extend(json.loads(report.read_text())['cells'])
    new_name = NEW_SOURCE.name
    groups = defaultdict(list)
    for cell in cells:
        projects = cell['validation']['projects']
        old = [value for key, value in projects.items() if key != new_name]
        cell['old_point_hit'] = sum(s['pages']*s['point_le_0_01_ratio'] for s in old)/sum(s['pages'] for s in old)
        cell['new_point_hit'] = projects[new_name]['point_le_0_01_ratio']
        groups[(cell['condition'], cell['epochs'])].append(cell)
    aggregates = []
    for (condition, epochs), values in groups.items():
        aggregates.append({'condition': condition, 'epochs': epochs, 'seeds': [x['seed'] for x in values],
                           'point_mean': float(np.mean([x['validation']['summary']['point_le_0_01_ratio'] for x in values])),
                           'macro_point_std': float(np.std([x['validation']['macro_point_hit'] for x in values])),
                           'new_point_mean': float(np.mean([x['new_point_hit'] for x in values])),
                           'old_point_mean': float(np.mean([x['old_point_hit'] for x in values]))})
    pool = [a for a in aggregates if a['condition'] != 'legacy_old']
    winner = max(pool, key=lambda a: a['point_mean'])
    candidates = groups[(winner['condition'], winner['epochs'])]
    candidate = min(candidates, key=lambda c: abs(c['validation']['summary']['point_le_0_01_ratio']-winner['point_mean']))
    selection = {'policy': 'pooled validation point hit only; stable condition order on ties; representative seed closest to group mean of that same metric',
                 'metric_contract': METRIC_CONTRACT, 'aggregates': aggregates, 'nominated': candidate,
                 'new_holdout_used_for_selection': False,
                 'baseline_final_holdout_comparisons': ['r130', 'r142', 'nominated'],
                 'runtime_promotion': False}
    frozen_path = root / 'reports/frozen_selection.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != selection:
        raise ValueError('Selection already frozen; cannot tune it after opening holdout')
    write_json(frozen_path, selection)
    # No holdout is evaluated before the selection file is committed to disk.
    candidate_path = Path(candidate['model_path'])
    diagnose(root, candidate_path, 'nominated')
    for label, model in [('r130', REPO / 'models/runtime/global_corner_model.pt'), ('r142', BASE_MODEL), ('nominated', candidate_path)]:
        diagnose(root, model, f'{label}_final_holdout', holdout_only=True)
    for run in roots:
        manifest = json.loads((run / 'data/manifest.json').read_text())
        manifest['new_holdout_opened'] = True
        manifest['frozen_selection_sha256'] = fingerprint(frozen_path)
        write_json(run / 'data/manifest.json', manifest)
    write_json(REPO / f'training/registry/{SLUG}.json', manifest)
    split_path = REPO / f'data/splits/cross_project/{SLUG}_v1.json'
    split = json.loads(split_path.read_text())
    split['metadata'].update(new_holdout_opened=True, frozen_selection_sha256=fingerprint(frozen_path))
    write_json(split_path, split)
    write_json(root / 'reports/aggregate_results.json', {'cells': cells, 'selection': selection})


def profile(root):
    selection = json.loads((root / 'reports/frozen_selection.json').read_text())
    pages = read_rows(root / 'data/old_validation.jsonl')[:20] + read_rows(root / 'data/new_validation.jsonl') + read_rows(root / 'data/holdout.jsonl')
    output = {'scope': 'isolated batch-1 complete recognition; includes preprocessing and candidate selection; excludes disk JPEG loading, OCR and PDF export',
              'threads': torch.get_num_threads(), 'pages': len(pages), 'models': {}}
    for label, path in [('r130', REPO / 'models/runtime/global_corner_model.pt'), ('r142', BASE_MODEL),
                        ('nominated', Path(selection['nominated']['model_path']))]:
        predictor = ExperimentPredictor(path)
        roi = RoiCornerPredictor(REPO / 'models/runtime/corner_heatmap_model.pt')
        local = LocalCornerMoEPredictor(REPO / 'models/runtime/local_corner_moe_coord_model.pt')
        image = cv2.imread(pages[0]['image_path'])
        for _ in range(2):
            predict_two_stage(Path(pages[0]['image_path']), predictor, roi, local, image=image,
                              candidate_expand_ratios=TEACHER_MULTI_EXPAND_RATIOS)
        elapsed = []
        for page in pages:
            image = cv2.imread(page['image_path'])
            if predictor.device.type == 'mps':
                torch.mps.synchronize()
            start = time.perf_counter()
            predict_two_stage(Path(page['image_path']), predictor, roi, local, image=image,
                              candidate_expand_ratios=TEACHER_MULTI_EXPAND_RATIOS)
            if predictor.device.type == 'mps':
                torch.mps.synchronize()
            elapsed.append((time.perf_counter()-start)*1000)
        output['models'][label] = {'mean_ms': float(np.mean(elapsed)), 'p95_ms': float(np.percentile(elapsed, 95)),
                                   'max_ms': max(elapsed), 'device': str(predictor.device),
                                   'frozen_encoder_device': 'cpu' if predictor.encoder is not None else None,
                                   'model_sha256': fingerprint(path)}
    write_json(root / 'reports/isolated_latency.json', output)
    print(json.dumps(output, ensure_ascii=False), flush=True)


def render_holdout(root):
    baseline = json.loads((root / 'reports/r130_final_holdout_stage_diagnostics.json').read_text())
    nominated = json.loads((root / 'reports/nominated_final_holdout_stage_diagnostics.json').read_text())
    lookup = {p['image_path']: p for p in nominated['pages']}
    width, height, title_height = 640, 480, 54
    canvas = np.full(((height+title_height)*len(baseline['pages']), width*2, 3), 245, np.uint8)
    for index, original in enumerate(baseline['pages']):
        for column, (label, row) in enumerate([('Released r130', original), ('Experimental', lookup[original['image_path']])]):
            image = cv2.imread(row['image_path'])
            scale = np.array([width/image.shape[1], height/image.shape[0]])
            image = cv2.resize(image, (width, height))
            cv2.polylines(image, [np.round(np.asarray(row['manual_quad'])*scale).astype(np.int32)], True, (60, 210, 60), 2)
            cv2.polylines(image, [np.round(np.asarray(row['quads']['selected'])*scale).astype(np.int32)], True, (0, 130, 255), 2)
            y, x = index*(height+title_height), column*width
            canvas[y+title_height:y+title_height+height, x:x+width] = image
            metrics = row['stages']['selected']
            hit = sum(metrics[f'corner_error_{i:02d}'] <= .01 for i in range(4))
            cv2.putText(canvas, f'{row["page_id"]} | {label} | hit {hit}/4', (x+12, y+22), cv2.FONT_HERSHEY_SIMPLEX, .57, (30,30,30), 1, cv2.LINE_AA)
            cv2.putText(canvas, 'Green: human   Orange: prediction', (x+12, y+44), cv2.FONT_HERSHEY_SIMPLEX, .5, (30,30,30), 1, cv2.LINE_AA)
    path = root / 'artifacts/new_holdout_comparison.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f'Unable to render {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'diagnose', 'train', 'finish', 'profile', 'render'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=BASE_MODEL)
    parser.add_argument('--label', default='r142')
    parser.add_argument('--decode', choices=['soft_argmax', 'local_peak'], default='soft_argmax')
    parser.add_argument('--include-holdout', action='store_true')
    parser.add_argument('--seeds', default='7,17,27')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--conditions', help='Comma-separated cells; use a separate root for a different budget')
    parser.add_argument('--additional-root', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if args.action == 'prepare':
        prepare(args.root)
    elif args.action == 'diagnose':
        diagnose(args.root, args.model, args.label, args.include_holdout, args.decode)
    elif args.action == 'finish':
        finish(args.root, args.additional_root)
    elif args.action == 'profile':
        profile(args.root)
    elif args.action == 'render':
        render_holdout(args.root)
    else:
        train_experiments(args.root, [int(s) for s in args.seeds.split(',')], args.epochs,
                          args.conditions.split(',') if args.conditions else None)


if __name__ == '__main__':
    main()
