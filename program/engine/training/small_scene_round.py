"""Bounded replay of a small, distinctive scene; one frozen dataset-addition test."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import torch

ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parents[1]
sys.path.insert(0, str(ENGINE))
from dataset_benchmark import METRIC_CONTRACT
from training.metaverse_round import evaluate
from training.screen_corner_ablation import read_rows, write_rows, write_json, fingerprint
from training.train_screen_corner_single import run

SLUG = '20260929-xr-zhouqunhua'
BASE = REPO / 'training/runs/screen_corner_single/round_002'
INITIAL = BASE / 'checkpoints/screen_corner_single.pt'
TAGS = ['black_main_screen', 'blue_side_panels', 'low_contrast', 'possible_corner_occlusion']


def audit_splits(splits, old):
    """Audit unique source rows before intentional training-only replay."""
    pairs = [('train', 'validation'), ('train', 'holdout'), ('validation', 'holdout')]
    for key in ['image_path', 'image_sha256', 'capture_group']:
        sets = {name: {r[key] for r in rows} for name, rows in splits.items()}
        for a, b in pairs:
            if sets[a] & sets[b]:
                raise ValueError(f'New scene overlap: {key}: {a}/{b}')
    old_paths = {str(Path(r['image_path']).resolve()) for r in old}
    old_hashes = {fingerprint(p) for p in old_paths}
    new = [r for rows in splits.values() for r in rows]
    if old_paths & {str(Path(r['image_path']).resolve()) for r in new}:
        raise ValueError('New scene overlaps previous paths')
    if old_hashes & {r['image_sha256'] for r in new}:
        raise ValueError('New scene duplicates previous image bytes')
    return {'path_overlap': 0, 'exact_image_overlap': 0, 'new_group_overlap': 0}


def prepare(root, *, slug=SLUG, base=BASE, base_data=None, prefix='xr', replay=4,
            contact_sheet=None, scene_tags=None, scene_description=None):
    if root.exists():
        raise FileExistsError('Use a new, append-only round directory')
    if replay < 1:
        raise ValueError('Replay factor must be positive')
    if not prefix.isidentifier():
        raise ValueError('Scene prefix must be an identifier')
    initial = base / 'checkpoints/screen_corner_single.pt'
    dataset_path = REPO / f'data/splits/cross_project/{slug}_v1.json'
    dataset = json.loads(dataset_path.read_text())
    splits = {k: dataset[k] for k in ['train', 'validation', 'holdout']}
    old_data = base_data or base / 'data/with_metaverse'
    old = list({r['image_path']: r for path in old_data.glob('*.jsonl')
               for r in read_rows(path)}.values())
    audit = audit_splits(splits, old)
    tags = list(scene_tags) if scene_tags is not None else TAGS.copy()
    metadata = dataset['metadata']
    metadata.update(scene_tags=tags, scene_tags_scope='project-level; not per-corner visibility labels',
                    objective='expose model to this small scene; no small-scene target score',
                    holdout_previously_observed=True, holdout_opened=True,
                    observation_reason=f'all {metadata["page_count"]} source images visually audited before training; regression only')
    if scene_description:
        metadata['scene_description']=scene_description
    for rows in splits.values():
        for row in rows:
            row['scene_tags'] = tags.copy()
    write_json(dataset_path, dataset)
    write_json(REPO / f'training/registry/{slug}.json', metadata)
    write_rows(Path(metadata['annotation_file']), [r for rows in splits.values() for r in rows])
    root.mkdir(parents=True)
    condition_name = f'with_{prefix}'
    for condition in ['control', condition_name]:
        target = root / 'data' / condition
        shutil.copytree(old_data, target)
        for name, rows in splits.items():
            write_rows(target / f'{prefix}_{name}.jsonl', rows)
        write_rows(target / 'validation.jsonl', read_rows(target / 'validation.jsonl') + splits['validation'])
        if condition == condition_name:
            write_rows(target / 'train.jsonl', read_rows(target / 'train.jsonl') + splits['train'] * replay)
        # No old or new regression pages may become training samples.
        train_paths = {r['image_path'] for r in read_rows(target / 'train.jsonl')}
        for path in target.glob('*.jsonl'):
            if path.stem == 'validation' or path.stem.endswith('_validation') or path.stem.endswith('holdout'):
                if train_paths & {r['image_path'] for r in read_rows(path)}:
                    raise ValueError(f'Training/regression overlap: {path.stem}')
    training = read_rows(root / 'data' / condition_name / 'train.jsonl')
    settings = dict(architecture='screen_corner_single_v1', metric_contract=METRIC_CONTRACT,
                    initial_model=str(initial), initial_sha256=fingerprint(initial), dataset=metadata,
                    base_run=str(base), base_data=str(old_data), data_prefix=prefix,
                    seed=7, coarse_epochs=4, local_epochs=8, steps_per_epoch=math.ceil(len(training)/32), batch_size=32,
                    replay_factor=replay, unique_new_train_pages=len(splits['train']),
                    train_entries={'control':len(read_rows(old_data/'train.jsonl')),
                                   condition_name:len(training)},
                    validation_pages=len(read_rows(root/'data'/condition_name/'validation.jsonl')),
                    conditions=['control',condition_name], loss_condition='point_hit',
                    selection=f'pooled development point_le_0_01_ratio only; initial, control, {condition_name} order on ties',
                    controlled_variable=f'add {len(splits["train"])} unique scene training photos with fixed {replay}x replay; same initial weights, loss, seed, optimizer steps and development set',
                    qualitative_review=(scene_description or 'inspect full main-screen boundary, blue-side confusion and occluded corners')+'; observations only, no second score or selection rule',
                    no_score_target=True, regression_used_for_selection=False)
    write_json(root/'manifest.json', settings)
    write_json(root/'reports/data_audit.json', dict(audit, unique_new_pages=metadata['page_count'],
                splits=metadata['split_counts'], groups=metadata['content_groups'],
                source_project_sha256=metadata['source_project_sha256']))
    (root/'artifacts').mkdir()
    shutil.copyfile(contact_sheet or REPO/'.build/xr-scene-contact.jpg',root/'artifacts/source_contact.jpg')
    for path in [Path(__file__), ENGINE/'training/train_screen_corner_single.py',
                 ENGINE/'screen_corner_single.py', ENGINE/'training/import_screen_project.py']:
        shutil.copyfile(path,root/'artifacts'/path.name)
    print(json.dumps(settings,ensure_ascii=False),flush=True)


def train(root):
    settings=json.loads((root/'manifest.json').read_text())
    initial = Path(settings['initial_model'])
    if fingerprint(initial) != settings['initial_sha256']:
        raise ValueError('Initial checkpoint changed')
    for condition in settings['conditions']:
        cell=root/'cells'/condition
        if (cell/'reports/frozen_selection.json').exists():
            continue
        run(cell,settings['coarse_epochs'],settings['local_epochs'],seed=settings['seed'],
            data_source=root/'data'/condition,initial_model=initial,
            steps_per_epoch=settings['steps_per_epoch'],conditions=['point_hit'],skip_evaluation=True)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    finish(root)


def finish(root):
    torch.set_num_threads(4)
    settings=json.loads((root/'manifest.json').read_text())
    data=root/'data'/settings['conditions'][-1]
    prefix=settings.get('data_prefix', 'xr')
    paths={'initial':Path(settings['initial_model']),**{c:root/f'cells/{c}/checkpoints/screen_corner_single.pt' for c in settings['conditions']}}
    validation=read_rows(data/'validation.jsonl')
    dev={}; records={}
    for label,path in paths.items():
        dev[label],records[label]=evaluate(path,validation)
        write_json(root/f'reports/{label}_validation_predictions.json',records[label])
    selected=max(paths,key=lambda k:dev[k]['final']['point_le_0_01_ratio'])
    frozen=dict(metric_contract=METRIC_CONTRACT,selected=selected,policy=settings['selection'],
                models={k:dict(path=str(p),sha256=fingerprint(p),validation=dev[k]['final']) for k,p in paths.items()},
                regression_used_for_selection=False)
    frozen_path=root/'reports/frozen_selection.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError('Selection already frozen; do not change after regression access')
    write_json(frozen_path,frozen)
    (root/'checkpoints').mkdir(exist_ok=True)
    shutil.copyfile(paths[selected],root/'checkpoints/screen_corner_single.pt')
    results={k:{} for k in paths}
    for label,path in paths.items():
        lookup={r['image_path']:r for r in records[label]}
        from dataset_benchmark import summarize_point_hits
        for name in sorted(p.stem for p in data.glob('*_validation.jsonl')):
            rows=read_rows(data/f'{name}.jsonl')
            results[label][name]={'final':summarize_point_hits([r['manual_quad'] for r in rows],
                                      [lookup[r['image_path']].get('final_quad') for r in rows])}
        diagnostic_names={p.stem for p in data.glob('*holdout.jsonl')}
        diagnostic_names.update([f'{prefix}_train', 'xr_train'])
        for name in sorted(diagnostic_names):
            if not (data/f'{name}.jsonl').exists():
                continue
            results[label][name],pred=evaluate(path,read_rows(data/f'{name}.jsonl'))
            write_json(root/f'reports/{label}_{name}_predictions.json',pred)
        print(label,json.dumps(results[label]),flush=True)
    write_json(root/'reports/evaluation.json',dict(metric_contract=METRIC_CONTRACT,selection=frozen,
                results=results,artifact=str(root/'checkpoints/screen_corner_single.pt'),
                artifact_sha256=fingerprint(root/'checkpoints/screen_corner_single.pt'),
                limitations=settings.get('limitations', ['one seed; small correlated single-talk scene','all new source images visually audited before training; regression is observed',
                             'replayed entries are not independent training images','no per-corner visibility annotations']),
                status='trained_candidate_not_promoted'))
    print('FINISHED',selected,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','train','finish'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--slug',default=SLUG)
    parser.add_argument('--base-run',type=Path,default=BASE)
    parser.add_argument('--base-data',type=Path)
    parser.add_argument('--prefix',default='xr')
    parser.add_argument('--replay',type=int,default=4)
    parser.add_argument('--contact-sheet',type=Path)
    parser.add_argument('--scene-tags',nargs='+')
    parser.add_argument('--scene-description')
    args=parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root.resolve(),slug=args.slug,base=args.base_run.resolve(),
                base_data=args.base_data.resolve() if args.base_data else None,
                prefix=args.prefix,replay=args.replay,contact_sheet=args.contact_sheet,
                scene_tags=args.scene_tags,scene_description=args.scene_description)
    else:
        globals()[args.action](args.root.resolve())
