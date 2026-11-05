"""Build an append-only all-history manual dataset with inherited evaluation roles."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from training.screen_corner_ablation import read_rows, write_rows, write_json, fingerprint

REPO = Path(__file__).resolve().parents[3]


def legacy_roles(registry, inputs):
    """Old page-id splits supply roles only; old annotation coordinates are unused."""
    roles = {}
    for p in sorted(registry.glob('*.json')):
        meta = json.loads(p.read_text())
        if not isinstance(meta, dict) or not meta.get('annotation_file'):
            continue
        annotation = Path(meta['annotation_file'])
        slug = meta.get('dataset_slug', '')
        manual = REPO / f'data/splits/cross_project/{slug}_split_manual_only_v1.json'
        split = manual if manual.exists() else Path(meta.get('split_file', '/nonexistent'))
        if not annotation.exists() or not split.is_file():
            continue
        spec = json.loads(split.read_text())
        if 'train_page_ids' not in spec:
            continue
        inputs[str(split)] = fingerprint(split)
        inputs[str(annotation)] = fingerprint(annotation)
        maps = {k: set(spec.get(key, [])) for k, key in
                [('train', 'train_page_ids'), ('validation', 'val_page_ids'), ('holdout', 'holdout_page_ids')]}
        for row in read_rows(annotation):
            key = f'{slug}:{row.get("project_slug", "project_001")}:{row["page_id"]}'
            for role, ids in maps.items():
                if key in ids:
                    roles[str(Path(row['image_path']).resolve())] = role
                    roles['dataset:' + slug + ':' + row['page_id']] = role
                    # Some legacy raw aliases are ordinary copied folders.
                    # Match migrated source paths by exact bytes in that case.
                    if Path(row['image_path']).is_file():
                        roles['sha256:' + fingerprint(row['image_path'])] = role
    return roles


def protected_roles(claims):
    """Never make a known development/regression photo a training example."""
    if 'holdout' in claims:
        return 'holdout'
    if 'validation' in claims:
        return 'validation'
    return 'train'


def content_hashes(image, quad):
    destination = np.array([[0, 0], [255, 0], [255, 143], [0, 143]], np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(np.asarray(quad, np.float32), destination), (256, 144))
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dh = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), 'big')
    freq = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    ph = int.from_bytes(np.packbits(freq > np.median(freq.flatten()[1:])).tobytes(), 'big')
    return dh, ph


def build(root):
    root = Path(root)
    if (root / 'manifest.json').exists() or (root / 'data/manifest.json').exists():
        raise FileExistsError('Dataset revision already exists')
    inputs = {}; records = {}; claims = defaultdict(set); role_sources = defaultdict(list)
    frozen = REPO / 'data/curated/collections/conference_v1'
    for row in read_rows(frozen / 'pages.jsonl'):
        path = str(Path(row['image_path']).resolve())
        records[path] = dict(row, image_path=path)
        role = {'observed_regression': 'holdout'}.get(row['current_role'], row['current_role'])
        if role in ['train', 'validation', 'holdout']:
            claims[path].add(role); role_sources[path].append('conference_v1:' + role)
    inputs[str(frozen / 'pages.jsonl')] = fingerprint(frozen / 'pages.jsonl')
    frozen_paths = set(records)
    # Current selected revisions, including training pages previously left out
    # by losing dataset-addition arms (e.g. Liu Qi / Sun Quanming).
    chosen_splits = set()
    for p in sorted((REPO / 'training/registry').glob('*.json')):
        meta = json.loads(p.read_text())
        if isinstance(meta, dict) and meta.get('split_file'):
            split = Path(meta['split_file'])
            if split.is_file():
                chosen_splits.add(split)
    chosen_splits.add(REPO / 'data/splits/cross_project/202610-zhuiguang-ai_v2.json')
    label_revisions = []
    for p in sorted(chosen_splits):
        spec = json.loads(p.read_text())
        if not all(isinstance(spec.get(k), list) and all(isinstance(r, dict) for r in spec[k])
                   for k in ['train', 'validation', 'holdout']):
            continue
        inputs[str(p)] = fingerprint(p)
        for role in ['train', 'validation', 'holdout']:
            for source in spec[role]:
                row = dict(source); path = str(Path(row['image_path']).resolve()); row['image_path'] = path
                if path in records and not np.allclose(records[path]['manual_quad'], row['manual_quad'], atol=.001, rtol=0):
                    label_revisions.append({'image_path': path, 'selected_revision': str(p), 'previous_quad': records[path]['manual_quad'], 'quad': row['manual_quad']})
                # An explicit latest scene revision supersedes the old frozen
                # label in THIS new dataset, never in conference_v1 itself.
                records[path] = dict(records.get(path, {}), **row)
                claims[path].add(role); role_sources[path].append(str(p) + ':' + role)
    # Active snapshot provides role evidence, not extra/stale manual labels.
    active = REPO / 'training/runs/screen_corner_single/round_029/data/with_computing'
    for role, files in [('train', [active / 'train.jsonl']), ('validation', [active / 'validation.jsonl']),
                        ('holdout', sorted(active.glob('*holdout.jsonl')))]:
        for p in files:
            inputs[str(p)] = fingerprint(p)
            for row in read_rows(p):
                path = str(Path(row['image_path']).resolve())
                if path in records:
                    claims[path].add(role); role_sources[path].append(str(p) + ':' + role)
    legacy = legacy_roles(REPO / 'training/registry', inputs)
    old_collection_datasets = {'202603-awe': '202603-awe', '202603-yangzhou': '202603-guochenghao-yangzhou'}
    for path in records:
        if not claims[path] and path in legacy:
            claims[path].add(legacy[path]); role_sources[path].append('legacy_manual_page_id_split:' + legacy[path])
        if not claims[path]:
            role = legacy.get('sha256:' + fingerprint(path))
            if role:
                claims[path].add(role); role_sources[path].append('legacy_manual_page_id_split_exact_bytes:' + role)
        if not claims[path]:
            row = records[path]
            slug = old_collection_datasets.get(row.get('collection_id'))
            role = legacy.get(f'dataset:{slug}:{row["page_id"]}') if slug else None
            if role:
                claims[path].add(role); role_sources[path].append('explicit_legacy_collection_page_id_mapping:' + slug + ':' + role)
        if not claims[path]:
            raise ValueError(f'Unresolved historical role: {path}')
    conflicts = json.loads((frozen / 'duplicate_conflicts.json').read_text())
    excluded_sha = {r['image_sha256'] for r in conflicts}
    exclusions = []; by_hash = {}; label_conflicts = set()
    for i, (path, row) in enumerate(sorted(records.items())):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]; q = np.asarray(row['manual_quad'], np.float64)
        digest = fingerprint(path)
        if (digest in excluded_sha or q.shape != (4, 2) or not np.isfinite(q).all() or
                q.min() < -.01 or (q > np.array([w, h]) + .01).any() or not cv2.isContourConvex(q.astype(np.float32)) or abs(cv2.contourArea(q.astype(np.float32))) < 16):
            exclusions.append({'image_path': path, 'reason': 'invalid_or_previously_conflicting_manual_quad'})
            continue
        row.update(image_sha256=digest, image_width=w, image_height=h, split=protected_roles(claims[path]),
                   role_evidence=role_sources[path], project_key=row.get('project_path', str(Path(path).parent)))
        if digest in by_hash:
            old = by_hash[digest]
            if not np.allclose(q, old['manual_quad'], atol=.001, rtol=0):
                label_conflicts.add(digest)
            old['split'] = protected_roles([old['split'], row['split']])
            old['role_evidence'] += row['role_evidence']
            exclusions.append({'image_path': path, 'reason': 'exact_duplicate_photo', 'canonical': old['image_path']})
            continue
        dh, ph = content_hashes(image, q)
        row.update(screen_dhash=f'{dh:016x}', screen_phash=f'{ph:016x}')
        by_hash[digest] = row
        if i % 200 == 0:
            print(f'audit images {i+1}/{len(records)}', flush=True)
    for digest in label_conflicts:
        exclusions.append({'image_path': by_hash[digest]['image_path'], 'reason': 'new_exact_duplicate_label_conflict'})
        del by_hash[digest]
    pages = list(by_hash.values())
    # Within each source project, preserve evaluation roles and quarantine
    # training photos near an evaluation slide. No historical roles rewritten.
    parents = list(range(len(pages)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    groups = defaultdict(list)
    for i, row in enumerate(pages):
        groups[row['project_key']].append(i)
    for indices in groups.values():
        for offset, i in enumerate(indices):
            for j in indices[:offset]:
                a, b = pages[i], pages[j]
                if ((int(a['screen_dhash'], 16) ^ int(b['screen_dhash'], 16)).bit_count() <= 8 or
                        (int(a['screen_phash'], 16) ^ int(b['screen_phash'], 16)).bit_count() <= 8):
                    parents[find(i)] = find(j)
    groups = defaultdict(list)
    for i, row in enumerate(pages):
        groups[find(i)].append(row)
    quarantined = []; selected = []
    for group in groups.values():
        eval_roles = {r['split'] for r in group} - {'train'}
        gid = hashlib.sha256('|'.join(sorted(r['image_sha256'] for r in group)).encode()).hexdigest()[:16]
        for row in group:
            row['content_group'] = gid
            if row['split'] == 'train' and eval_roles:
                quarantined.append(dict(row, quarantine_reason='near_duplicate_of_inherited_evaluation_photo'))
            else:
                selected.append(row)
    partitions = {role: [r for r in selected if r['split'] == role] for role in ['train', 'validation', 'holdout']}
    for role, rows in partitions.items():
        write_rows(root / 'data' / f'{role}.jsonl', rows)
    write_rows(root / 'data/all_manual.jsonl', pages)
    write_rows(root / 'data/quarantined_train.jsonl', quarantined)
    write_json(root / 'reports/dataset_audit.json', {'exclusions': exclusions, 'label_revision_changes': label_revisions,
        'role_conflicts': [{'image_path': p, 'claims': sorted(c), 'selected': protected_roles(c)} for p, c in claims.items() if len(c) > 1],
        'near_duplicate_quarantined_train': len(quarantined), 'near_duplicate_scope': 'within source project; dHash <=8 OR pHash <=8, transitive',
        'cross_project_near_duplicate_limitation': 'exact photos deduplicated globally; recompressed/cross-talk near duplicates not exhaustively resolved'})
    manifest = {'revision': 'all_history_v1', 'metric_contract': 'screen_point_hit_pixel_v3', 'manual_pages': len(pages),
        'source_projects': len({r['project_key'] for r in pages}), 'frozen_conference_pages': len(frozen_paths),
        'split_counts': {k: len(v) for k, v in partitions.items()}, 'quarantined_train': len(quarantined),
        'per_project': {p: dict(Counter(r['split'] for r in selected if r['project_key'] == p)) for p in sorted({r['project_key'] for r in pages})},
        'inputs': inputs, 'roles_policy': 'preserve known evaluation exposure; holdout > validation > train; quarantine near-eval training slides',
        'interpretation': 'all historical manual sources; inherited validation/regression previously observed, not blind',
        'automatic_candidates_used_as_truth': False, 'sources_untouched': True}
    write_json(root / 'data/manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['per_project', 'inputs']}, ensure_ascii=False), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    build(parser.parse_args().root)
