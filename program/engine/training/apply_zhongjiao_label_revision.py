"""Apply the re-corrected Zhongjiao corner labels and refresh the training snapshot.

Out-of-frame convention: several physical screen corners are outside the photo.
The human points are the intersections of the screen edges with the image
border, not the true physical corners. They are intentionally kept at the
frame edge and must not be rejected as invalid quads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from project_file import resolve_image_path
from supervision_utils import resolve_manual_quad
from perspective_detect import order_points
from training.screen_corner_ablation import fingerprint, read_rows, write_json, write_rows

REPO = Path(__file__).resolve().parents[3]
PROJECT = Path('/Users/gcssloop/WorkSpace/AIGC/展会/2026/202503 杭州-第十五届智能交通市场年会/2026-03-06 中交机电/screen-pdf-project.json')
SLUG = '202603-zhongjiao-jidian'
BASE = REPO / 'training/datasets/screen_corner_single_all_manual_3_20fine_v1'
SNAPSHOT = REPO / 'training/datasets/screen_corner_single_all_manual_3_20fine_zhongjiao_v1'
ANNOTATION = REPO / f'data/curated/annotations/{SLUG}_pages_v2.jsonl'
SPLIT = REPO / f'data/splits/cross_project/{SLUG}_v2.json'
SCENE = REPO / f'data/curated/scenes/{SLUG}_v1'
REGISTRY = REPO / f'training/registry/{SLUG}.json'
OLD_ANNOTATION = REPO / f'data/curated/annotations/{SLUG}_pages.jsonl'
OLD_SPLIT = REPO / f'data/splits/cross_project/{SLUG}_split_manual_only_v1.json'
EXCLUDED = 'IMG_7018'
NO_MANUAL = 'IMG_7022'
NEW_PAGE_REPLAY = 4


def content_hashes(image: np.ndarray, quad: list[list[float]]) -> tuple[int, int]:
    destination = np.array([[0, 0], [255, 0], [255, 143], [0, 143]], np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(np.asarray(quad, np.float32), destination), (256, 144))
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dh = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), 'big')
    freq = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    ph = int.from_bytes(np.packbits(freq > np.median(freq.flatten()[1:])).tobytes(), 'big')
    return dh, ph


def diag(quad: np.ndarray) -> float:
    return float(np.linalg.norm(quad.max(0) - quad.min(0)))


def build(project_file: Path = PROJECT, base: Path = BASE) -> dict:
    project_file = project_file.resolve()
    source_root = project_file.parent
    project = json.loads(project_file.read_text())
    old_rows = {r['page_id']: r for r in read_rows(OLD_ANNOTATION)}
    old_split = json.loads(OLD_SPLIT.read_text())
    old_roles = {}
    for role, key in [('train', 'train_page_ids'), ('validation', 'val_page_ids'), ('holdout', 'holdout_page_ids')]:
        for item in old_split.get(key, []):
            old_roles[item.rsplit(':', 1)[-1]] = role
    base_rows = [r for r in read_rows(base / 'all_manual.jsonl')
                 if str(Path(r.get('project_path', ''))) == str(project_file)]
    base_by_page = {r['page_id']: r for r in base_rows}
    template = base_rows[0]

    pages = {p['id']: p for p in project['pages']}
    records = []
    for page_id, page in pages.items():
        image_path = resolve_image_path(project_file, page, project)
        if image_path is None:
            raise FileNotFoundError(page.get('path') or page.get('name'))
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        h, w = image.shape[:2]
        quad, source = resolve_manual_quad(page)
        manual = bool(quad)
        if manual:
            points = order_points(np.asarray(quad, np.float32))
            assert points.shape == (4, 2) and np.isfinite(points).all()
            assert points.min() >= -0.01 and (points <= np.array([w, h]) + 0.01).all()
            assert cv2.isContourConvex(points) and abs(cv2.contourArea(points)) >= 100
            quad = points.tolist()
            dh, ph = content_hashes(image, quad)
            border = [i for i, (x, y) in enumerate(quad)
                      if x <= 1 or y <= 1 or x >= w - 1 or y >= h - 1]
        else:
            quad = None
            dh = ph = None
            border = []
        if page_id == EXCLUDED:
            role = None
            reason = 'exact_duplicate_conflict_with_202603-white-mix:IMG_7018'
        elif not manual:
            role = None
            reason = 'no_manual_quad'
        elif page_id in base_by_page:
            role = base_by_page[page_id]['split']
            reason = None
        elif page_id == 'IMG_7021':
            # Same screen-content group as IMG_7020, which is train in the
            # current base. Keeping 7021 as holdout would leak; use train.
            role = 'train'
            reason = None
        elif page_id in old_roles:
            role = old_roles[page_id]
            reason = None
        else:
            role = 'train'
            reason = None
        adjustment = None
        candidate = page.get('candidates') or []
        if manual and candidate:
            selected = candidate[int(page.get('selectedCandidateIndex') or 0)]
            original = selected.get('originalQuad') or selected.get('quad')
            if original:
                adjustment = float(np.linalg.norm(np.asarray(quad) - np.asarray(original, float), axis=1).max() / diag(np.asarray(quad)))
        records.append({
            'dataset_slug': SLUG,
            'project_slug': 'project_001',
            'project_name': source_root.name,
            'project_path': str(project_file),
            'page_id': page_id,
            'page_name': page.get('name'),
            'image_path': str(image_path),
            'image_width': w,
            'image_height': h,
            'image_sha256': fingerprint(image_path),
            'manual_quad': quad,
            'annotation_source': source,
            'annotation_revision': 2,
            'split': role,
            'training_eligible': role is not None,
            'exclusion_reason': reason,
            'out_of_frame_corners': border,
            'edge_intersection': bool(border),
            'previous_manual_quad': old_rows.get(page_id, {}).get('manual_quad'),
            'previous_role': old_roles.get(page_id),
            'label_quality': 'runtime_candidate_seeded_manual',
            'fine_corner_truth': False,
            'screen_dhash': f'{dh:016x}' if dh is not None else None,
            'screen_phash': f'{ph:016x}' if ph is not None else None,
            'status': page.get('status'),
            'adjustment_max_ratio': adjustment,
        })

    # Duplicate audit against every other annotation and the frozen collection.
    others = []
    for path in sorted((REPO / 'data/curated/annotations').glob('*.jsonl')):
        if path == ANNOTATION or path.name == OLD_ANNOTATION.name or 'deep_screen_v1' in path.name:
            continue
        for row in read_rows(path):
            row['_file'] = path.name
            others.append(row)
    for row in read_rows(REPO / 'data/curated/collections/conference_v1/pages.jsonl'):
        row['_file'] = 'conference_v1/pages.jsonl'
        others.append(row)
    exact, near = [], []
    for row in records:
        if not row['manual_quad']:
            continue
        for other in others:
            same_project = (str(Path(other.get('project_path', ''))) == str(project_file)
                            or other.get('project_name') == source_root.name)
            if same_project:
                continue
            if other.get('image_sha256') and row['image_sha256'] == other['image_sha256']:
                exact.append({'new': row['page_id'], 'old': other.get('page_id'), 'old_file': other['_file']})
                continue
            if not other.get('screen_dhash') or not other.get('screen_phash') or not row['screen_dhash']:
                continue
            dh = (int(row['screen_dhash'], 16) ^ int(other['screen_dhash'], 16)).bit_count()
            ph = (int(row['screen_phash'], 16) ^ int(other['screen_phash'], 16)).bit_count()
            if dh <= 8 or ph <= 8:
                near.append({'new': row['page_id'], 'old': other.get('page_id'), 'old_file': other['_file'], 'dh': dh, 'ph': ph})
    if EXCLUDED not in {x['new'] for x in exact}:
        exact.append({'new': EXCLUDED, 'old': EXCLUDED, 'old_file': '202603-white-mix_pages.jsonl'})
    assert {x['new'] for x in exact} <= {EXCLUDED}, exact
    assert not [x for x in near if x['new'] != EXCLUDED], near
    # 7020 / 7021 are the same slide group; both are train.
    assert {r['split'] for r in records if r['page_id'] in {'IMG_7020', 'IMG_7021'}} == {'train'}

    active = [r for r in records if r['training_eligible']]
    split_counts = dict(Counter(r['split'] for r in active))
    assert split_counts == {'train': 16, 'validation': 2}, split_counts
    assert not [r for r in active if r['split'] == 'holdout']
    adjustments = [r['adjustment_max_ratio'] for r in records if r['training_eligible'] and r['adjustment_max_ratio'] is not None]
    out_pages = [{'page_id': r['page_id'], 'corners': r['out_of_frame_corners']} for r in active if r['edge_intersection']]
    previous_roles = {r['page_id']: r['previous_role'] for r in records if r['previous_role']}
    meta = {
        'dataset_slug': SLUG,
        'source_root': str(source_root),
        'source_project': str(project_file),
        'source_project_sha256': fingerprint(project_file),
        'annotation_file': str(ANNOTATION),
        'split_file': str(SPLIT),
        'page_count': len(records),
        'manual_page_count': sum(bool(r['manual_quad']) for r in records),
        'active_training_pages': len(active),
        'split_counts': split_counts,
        'excluded_pages': {EXCLUDED: 'exact_duplicate_conflict_with_202603-white-mix:IMG_7018'},
        'no_manual_pages': [NO_MANUAL],
        'annotation_revision': 2,
        'previous_annotation_file': str(OLD_ANNOTATION),
        'previous_split_file': str(OLD_SPLIT),
        'previous_split_counts': {'train': 6, 'validation': 2, 'holdout': 4},
        'role_policy': ('preserve current base roles; 7031/7035 old train; 7021 train because it is the same content group as train 7020; '
                        '7 newly labeled pages train; 7018 excluded duplicate conflict; no holdout remains'),
        'role_changes': previous_roles,
        'out_of_frame_pages': out_pages,
        'out_of_frame_note': ('8 pages have one corner outside the captured frame; the stored point is the intersection of the screen edge '
                              'with the image border, not the physical corner. Predictions are clamped to the image; the 1% metric uses this '
                              'boundary intersection as truth. Points are not invalid.'),
        'label_quality': 'runtime_candidate_seeded_manual',
        'label_quality_note': ('manual re-correction of runtime-seeded labels; 12 previously labeled pages were adjusted, 7 pages newly labeled; '
                               'large adjustments on many pages correct runtime mis-localization'),
        'fine_corner_truth': False,
        'annotation_independence': False,
        'adjustment_max_ratio': max(adjustments),
        'adjustment_mean_ratio': float(np.mean(adjustments)),
        'adjustments_by_page': {r['page_id']: r['adjustment_max_ratio'] for r in records if r['adjustment_max_ratio'] is not None},
        'cross_session_duplicate_audit': {
            'exact_byte_duplicates': len(exact),
            'exact_duplicate_pages': sorted({x['new'] for x in exact}),
            'near_duplicates': [x for x in near if x['new'] != EXCLUDED],
            'verdict': 'IMG_7018 is the same photo as 202603-white-mix IMG_7018 with conflicting label/role; excluded. No other exact or pHash/dHash <=8 duplicate.',
        },
        'status': 'registered_not_trained',
        'training_status': 'not_started',
        'training_run': None,
        'source_untouched': True,
        'objective': 'apply user re-corrected Zhongjiao corner labels, including frame-edge intersections for out-of-frame screen corners',
    }

    for path in [ANNOTATION, SPLIT, SNAPSHOT]:
        if path.exists():
            raise FileExistsError(path)

    write_rows(ANNOTATION, records)
    write_json(SPLIT, {'metadata': meta,
                       'train': [r for r in active if r['split'] == 'train'],
                       'validation': [r for r in active if r['split'] == 'validation'],
                       'holdout': []})
    if REGISTRY.exists():
        (SCENE / 'reports').mkdir(parents=True, exist_ok=True)
        (SCENE / 'reports/registry_before_v2.json').write_bytes(REGISTRY.read_bytes())
    write_json(REGISTRY, meta)

    # Snapshot patch
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    for name in ['holdout', 'new_train', 'new_validation', 'new_holdout', 'quarantined_train']:
        (SNAPSHOT / f'{name}.jsonl').write_bytes((base / f'{name}.jsonl').read_bytes())
    updates = {r['page_id']: r for r in active}
    new_train = [r for r in active if r['split'] == 'train' and r['page_id'] not in base_by_page]

    def patch(rows, add_new, replay=1):
        touched = 0; output = list(rows)
        for i, row in enumerate(output):
            if str(Path(row.get('project_path', ''))) != str(project_file):
                continue
            update = updates.get(row['page_id'])
            if not update:
                continue
            output[i] = dict(row, manual_quad=update['manual_quad'], annotation_revision=2,
                             out_of_frame_corners=update['out_of_frame_corners'],
                             edge_intersection=update['edge_intersection'])
            touched += 1
        if add_new:
            for row in new_train:
                for _ in range(replay):
                    item = dict(template)
                    item.update(image_path=row['image_path'], image_sha256=row['image_sha256'],
                                manual_quad=row['manual_quad'], image_width=row['image_width'],
                                image_height=row['image_height'], page_id=row['page_id'],
                                page_name=row['page_name'],
                                split='train', current_role='train', annotation_source=row['annotation_source'],
                                annotation_revision=2, out_of_frame_corners=row['out_of_frame_corners'],
                                edge_intersection=row['edge_intersection'],
                                role_evidence=['202603-zhongjiao-jidian_v2:train'],
                                screen_dhash=row['screen_dhash'], screen_phash=row['screen_phash'])
                    item['content_group'] = (base_by_page['IMG_7020']['content_group']
                                             if row['page_id'] == 'IMG_7021'
                                             else hashlib.sha256(row['image_sha256'].encode()).hexdigest()[:16])
                    output.append(item)
        return output, touched

    train, changed_train = patch(read_rows(base / 'train.jsonl'), True, NEW_PAGE_REPLAY)
    validation, changed_validation = patch(read_rows(base / 'validation.jsonl'), False)
    all_manual, changed_all = patch(read_rows(base / 'all_manual.jsonl'), True, 1)
    assert len(train) == 5575, len(train)
    assert len(all_manual) == 2286, len(all_manual)
    assert len({r['image_path'] for r in all_manual}) == len(all_manual)
    write_rows(SNAPSHOT / 'train.jsonl', train)
    write_rows(SNAPSHOT / 'validation.jsonl', validation)
    write_rows(SNAPSHOT / 'all_manual.jsonl', all_manual)
    write_rows(SNAPSHOT / 'quarantined_train.jsonl', read_rows(base / 'quarantined_train.jsonl'))
    train_paths = {r['image_path'] for r in train}
    for name in ['validation', 'holdout', 'new_validation', 'new_holdout']:
        assert not train_paths & {r['image_path'] for r in read_rows(SNAPSHOT / f'{name}.jsonl')}, name

    audit = {
        'revision': 'zhongjiao_corner_revision_v2',
        'base_snapshot': str(base),
        'source_project': str(project_file),
        'source_project_sha256': fingerprint(project_file),
        'annotation_file': str(ANNOTATION),
        'split_file': str(SPLIT),
        'page_count': len(records),
        'manual_page_count': sum(bool(r['manual_quad']) for r in records),
        'active_training_pages': len(active),
        'split_counts': split_counts,
        'previous_split_counts': meta['previous_split_counts'],
        'excluded_pages': meta['excluded_pages'],
        'no_manual_pages': meta['no_manual_pages'],
        'out_of_frame_pages': out_pages,
        'out_of_frame_note': meta['out_of_frame_note'],
        'adjustment_max_ratio': meta['adjustment_max_ratio'],
        'adjustment_mean_ratio': meta['adjustment_mean_ratio'],
        'changes': [{'page_id': r['page_id'], 'previous': r['previous_manual_quad'], 'current': r['manual_quad'],
                     'max_fraction_of_diagonal': r['adjustment_max_ratio'], 'out_of_frame_corners': r['out_of_frame_corners']}
                    for r in records if r['previous_manual_quad']],
        'changed_existing_pages': sum(1 for r in records if r['previous_manual_quad']),
        'newly_labeled_pages': [r['page_id'] for r in records if r['manual_quad'] and not r['previous_manual_quad']],
        'duplicate_audit': meta['cross_session_duplicate_audit'],
        'entry_counts': {name: {'before': len(read_rows(base / f'{name}.jsonl')), 'after': len(read_rows(SNAPSHOT / f'{name}.jsonl'))}
                         for name in ['train', 'validation', 'holdout', 'all_manual', 'quarantined_train']},
        'changed_entries': {'train': changed_train, 'validation': changed_validation, 'all_manual': changed_all,
                            'new_train_entries': len(new_train) * NEW_PAGE_REPLAY},
        'protected_inputs_sha256': {
            'base_train': fingerprint(base / 'train.jsonl'),
            'base_validation': fingerprint(base / 'validation.jsonl'),
            'base_all_manual': fingerprint(base / 'all_manual.jsonl'),
            'conference_pages': fingerprint(REPO / 'data/curated/collections/conference_v1/pages.jsonl'),
            'source_project': fingerprint(project_file),
        },
        'status': 'registered_not_trained',
        'notes': [
            'IMG_7018 is the same photo as 202603-white-mix IMG_7018 with contradictory labels/roles; excluded from training.',
            'IMG_7021 is the same content group as train IMG_7020; it is train, so this revision has no holdout.',
            '7031 and 7035 were invalid in the old split due points outside the image; the re-corrected points are on the frame border and valid.',
            'Frame-edge points are intentional intersections, not true physical corners.',
        ],
    }
    write_json(SNAPSHOT / 'audit.json', audit)
    write_json(SNAPSHOT / 'summary.json', {
        'revision': 'zhongjiao_corner_revision_v2',
        'base_snapshot': str(base),
        'train_entries': len(train),
        'validation_pages': len(validation),
        'holdout_pages': len(read_rows(SNAPSHOT / 'holdout.jsonl')),
        'all_manual_pages': len(all_manual),
        'quarantined_train_entries': len(read_rows(SNAPSHOT / 'quarantined_train.jsonl')),
        'active_training_pages': len(active),
        'split_counts': split_counts,
        'out_of_frame_pages': out_pages,
        'excluded_pages': meta['excluded_pages'],
        'newly_labeled_pages': audit['newly_labeled_pages'],
        'status': 'registered_not_trained',
    })
    (SNAPSHOT / 'README.md').write_text(
        '# 中交机电点位重矫正训练快照（v2）\n\n'
        f'基线：`{base}`。本快照替换中交机电 8 个既有训练/开发页的人工四角，并加入 10 个新页（7021/7031/7035 + 7 个新标注页），'
        '`train` 5535→5575 条（新页 4 倍回放，共 40 条），`all_manual` 2276→2286 页，`validation` 483 页，`holdout` 266 页。\n\n'
        '**出框约定**：8 页有一个屏幕角超出照片范围；人工点取屏幕边与图像边界的交点，位于画面边框上，不是真实物理角点。'
        '这些点有效，不应按越界框过滤；预测被钳制在图像内，1% 指标以边界交点为真值。\n\n'
        '`IMG_7018` 与 `202603-white-mix/IMG_7018` 是同一张照片但标签/用途冲突，继续排除。`IMG_7022` 无人工框。'
        '`IMG_7021` 与训练页 `IMG_7020` 同内容组，因此为训练页，本 revision 没有 holdout。状态 `registered_not_trained`，未训练、未改 runtime、未打包。\n',
        encoding='utf-8',
    )

    SCENE.mkdir(parents=True, exist_ok=True)
    write_json(SCENE / 'manifest.json', dict(meta, scene_slug=SCENE.name))
    for name in ['train', 'validation', 'holdout']:
        write_rows(SCENE / f'{name}.jsonl', [r for r in active if r['split'] == name] if name != 'holdout' else [])
    imgs = []
    for row in active:
        image = cv2.imread(row['image_path'])
        q = np.asarray(row['manual_quad'], np.float32).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [q], True, (0, 220, 0), 7)
        thumb = cv2.resize(image, (480, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"{row['page_id']} {row['split'][0].upper()}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        imgs.append(thumb)
    while len(imgs) % 4:
        imgs.append(np.zeros_like(imgs[0]))
    cv2.imwrite(str(SCENE / 'contact.jpg'), np.vstack([np.hstack(imgs[i:i + 4]) for i in range(0, len(imgs), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    (SCENE / 'reports').mkdir(exist_ok=True)
    (SCENE / 'reports/zhongjiao_corner_revision.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'pages': len(records), 'manual': meta['manual_page_count'], 'split': split_counts,
                      'out_of_frame_pages': len(out_pages), 'max_adjustment': meta['adjustment_max_ratio'],
                      'snapshot': str(SNAPSHOT)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    build(args.project, args.base)
