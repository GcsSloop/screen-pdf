"""Group read-only source projects into conference/session collections, preserving run roles.

No labels, historical splits or model weights are modified. The source projects'
explicit manualQuad is the benchmark truth. Old automatic/accepted quads do not
become human annotations merely because they appeared in a training snapshot.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from project_file import project_files, resolve_image_path
from supervision_utils import resolve_manual_quad
from perspective_detect import order_points
from dataset_benchmark import METRIC_CONTRACT

CONFIG = Path(__file__).with_name('conference_collections.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classify(project, source_root, config):
    parts = project.relative_to(source_root).parts
    matches = [c for c in config['collections'] if c['directory'] in parts and
               (not c.get('project_prefix') or any(p.startswith(c['project_prefix']) for p in parts))]
    if len(matches) != 1:
        raise ValueError(f'Need exactly one explicit conference assignment: {project}, matches={matches}')
    return matches[0]


def active_roles(data):
    roles = defaultdict(set)
    examples = {}
    files = [data/'train.jsonl', data/'validation.jsonl'] + sorted(data.glob('*holdout.jsonl'))
    for path in files:
        role = 'train' if path.stem == 'train' else 'validation' if path.stem == 'validation' else 'observed_regression'
        for row in read_rows(path):
            key = str(Path(row['image_path']).resolve())
            roles[key].add(role)
            examples[key] = row
    conflicts = {p: sorted(v) for p, v in roles.items() if len(v) > 1}
    if conflicts:
        raise ValueError(f'Current run split conflict: {conflicts}')
    return {k: next(iter(v)) for k, v in roles.items()}, examples, files


def build(output, *, config_path=CONFIG, data=None):
    if output.exists():
        raise FileExistsError('Use a new collection revision; do not overwrite frozen benchmark inputs')
    config = json.loads(config_path.read_text())
    source = Path(config['source_root'])
    roles, old_rows, active_files = active_roles(data) if data else ({}, {}, [])
    projects = project_files(source)
    available = {}; excluded = []; inventory = []; changes = []; duplicate_aliases = []
    source_digests = {str(p): digest(p) for p in projects}
    by_image_hash = defaultdict(list)
    conflicts = []
    for project in projects:
        group = classify(project, source, config)
        payload = json.loads(project.read_text())
        project_id = hashlib.sha256(str(project).encode()).hexdigest()[:16]
        counts = Counter()
        for page in payload.get('pages', []):
            counts['source_pages'] += 1
            quad, annotation_source = resolve_manual_quad(page)
            image = resolve_image_path(project, page, payload)
            reason = 'missing_explicit_manual_quad' if quad is None else 'missing_image' if image is None else None
            if reason:
                counts[reason] += 1
                excluded.append(dict(collection_id=group['id'], project_path=str(project), page_id=page.get('id'), reason=reason))
                continue
            image = image.resolve()
            key = str(image)
            # Read dimensions without guessing orientation; inference uses this same cv2 decode.
            pixels = cv2.imread(key)
            if pixels is None:
                counts['unreadable_image'] += 1
                excluded.append(dict(collection_id=group['id'], project_path=str(project), page_id=page.get('id'), reason='unreadable_image'))
                continue
            height, width = pixels.shape[:2]
            q = order_points(np.asarray(quad, dtype=np.float32))
            valid_quad = not (q.shape != (4, 2) or not np.isfinite(q).all() or not cv2.isContourConvex(q)
                    or abs(cv2.contourArea(q)) < 100 or (q.min(0) < [-1, -1]).any()
                    or (q.max(0) > [width+1, height+1]).any())
            if not valid_quad:
                counts['invalid_manual_quad'] += 1
            image_hash = digest(image)
            row = dict(image_path=key, image_sha256=image_hash, manual_quad=q.tolist(), image_width=width,
                       image_height=height, project_path=str(project), project_name=project.parent.name,
                       project_id=project_id, page_id=page.get('id'), collection_id=group['id'],
                       collection_name=group['name'], annotation_source=annotation_source,
                       current_role=roles.get(key, 'outside_current_snapshot'),
                       scene_tags=old_rows.get(key, {}).get('scene_tags', []),
                       historical_scene_dataset=old_rows.get(key, {}).get('dataset_slug'),
                       valid_manual_quad=valid_quad)
            if key in old_rows and not np.allclose(q, order_points(np.asarray(old_rows[key]['manual_quad'], dtype=np.float32)), atol=.001, rtol=0):
                changes.append(dict(image_path=key, collection_id=group['id'], current_role=row['current_role'],
                                    snapshot_quad=old_rows[key]['manual_quad'], current_source_manual_quad=q.tolist()))
            by_image_hash[image_hash].append(row)
            counts['candidate_manual_pages'] += 1
        inventory.append(dict(project_id=project_id, project_path=str(project), project_name=project.parent.name,
                              collection_id=group['id'], counts=dict(counts)))
    for image_hash, variants in by_image_hash.items():
        prior = variants[0]
        label_conflict = any(not np.allclose(prior['manual_quad'], r['manual_quad'], atol=.001, rtol=0) for r in variants[1:])
        role_conflict = len({r['current_role'] for r in variants}) > 1
        if label_conflict or role_conflict:
            conflicts.append(dict(image_sha256=image_hash, label_conflict=label_conflict,
                                  role_conflict=role_conflict, variants=variants))
            excluded.extend(dict(collection_id=r['collection_id'], project_path=r['project_path'],
                                 page_id=r['page_id'], reason='duplicate_label_or_role_conflict') for r in variants)
            continue
        if not prior['valid_manual_quad']:
            excluded.extend(dict(collection_id=r['collection_id'], project_path=r['project_path'],
                                 page_id=r['page_id'], reason='invalid_manual_quad', variant=r) for r in variants)
            continue
        available[prior['image_path']] = prior
        duplicate_aliases.extend(dict(image_path=r['image_path'], canonical=prior['image_path'], project_path=r['project_path']) for r in variants[1:])
    usable_by_project = Counter(r['project_id'] for r in available.values())
    for project in inventory:
        project['counts']['usable_manual_pages'] = usable_by_project[project['project_id']]
    rows = sorted(available.values(), key=lambda r: (r['collection_id'], r['project_path'], str(r['page_id'])))
    role_counts = dict(Counter(r['current_role'] for r in rows))
    manifest = dict(schema_version=1, collection_revision=output.name, metric_contract=METRIC_CONTRACT,
                    source_root=str(source), config_path=str(config_path.resolve()), config_sha256=digest(config_path),
                    active_data_snapshot=str(data), source_projects=len(projects), source_pages=sum(p['counts']['source_pages'] for p in inventory),
                    usable_manual_pages=len(rows), excluded_pages=len(excluded), exclusion_reasons=dict(Counter(r['reason'] for r in excluded)),
                    duplicate_image_aliases=duplicate_aliases, conflicting_image_groups=len(conflicts), role_counts=role_counts,
                    source_project_sha256=source_digests, active_snapshot_sha256={str(p): digest(p) for p in active_files},
                    label_authority='current explicit human manualQuad in original source project; never activeQuad or model output',
                    role_policy='current round_010 snapshot membership retained as annotation; no split assignment for other data; no training or split rewrite',
                    historical_exposure='outside_current_snapshot means absence from this snapshot, not proof of never used in older model training',
                    changed_current_labels=len(changes), duplicate_label_tolerance_pixels=.001, blind_test_claim=False,
                    collections=[])
    for group in config['collections']:
        members = [r for r in rows if r['collection_id'] == group['id']]
        projs = [p for p in inventory if p['collection_id'] == group['id']]
        folder = output / group['id']
        item = dict(group, projects=len(projs), source_pages=sum(p['counts']['source_pages'] for p in projs),
                    usable_manual_pages=len(members), roles=dict(Counter(r['current_role'] for r in members)),
                    member_projects=projs, path=str(folder.resolve()))
        write_rows(folder/'pages.jsonl', members)
        for role in ['train', 'validation', 'observed_regression', 'outside_current_snapshot']:
            write_rows(folder/f'{role}.jsonl', [r for r in members if r['current_role'] == role])
        write_json(folder/'manifest.json', item)
        manifest['collections'].append(item)
        lines = [f'# {group["name"]}', '', group.get('target', group.get('note', '按会议场次合并查看，演讲与原项目保留来源。')), '',
                 f'源项目 {len(projs)} 个，源页面 {item["source_pages"]}，可测人工标注 {len(members)}。', '',
                 '| 演讲 / 项目 | 可测人工页 | 源页面 |', '| --- | ---: | ---: |']
        lines += [f'| {p["project_name"]} | {p["counts"].get("usable_manual_pages", 0)} | {p["counts"]["source_pages"]} |' for p in projs]
        lines += ['', 'pages.jsonl 为去重人工标注全集；分用途文件仅继承当前训练快照成员关系，不重新划分数据。',
                  '不在当前快照中的页面不代表独立盲测；历史训练暴露未完全核实。基准结果统一查看上级 BASELINE.md。', '']
        (folder/'README.md').write_text('\n'.join(lines))
    write_rows(output/'pages.jsonl', rows)
    write_json(output/'index.json', manifest)
    write_json(output/'source_inventory.json', inventory)
    write_json(output/'excluded_pages.json', excluded)
    write_json(output/'annotation_changes.json', changes)
    write_json(output/'duplicate_conflicts.json', conflicts)
    unrepresented = sorted(set(roles) - set(available))
    write_json(output/'snapshot_pages_without_current_manual.json', unrepresented)
    assert all(digest(p) == sha for p, sha in source_digests.items())
    write_json(output/'preservation_audit.json', dict(passed=True, source_project_count=len(projects),
               source_project_bytes_unchanged=True, current_snapshot_pages=len(roles),
               snapshot_pages_without_current_manual=len(unrepresented), source_and_snapshot_sha256_in='index.json'))
    print(json.dumps({k: manifest[k] for k in ['source_projects', 'source_pages', 'usable_manual_pages', 'excluded_pages', 'role_counts', 'changed_current_labels']}, ensure_ascii=False), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--active-data', type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve(), config_path=args.config, data=args.active_data.resolve())
