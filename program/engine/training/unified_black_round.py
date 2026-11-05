"""Retrain with a consolidated black-background scene and uniform replay."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import sys

ENGINE=Path(__file__).resolve().parents[1]
REPO=ENGINE.parents[1]
sys.path.insert(0,str(ENGINE))
from dataset_benchmark import METRIC_CONTRACT
from training.build_black_scene_dataset import SLUG
from training.screen_corner_ablation import read_rows,write_rows,write_json,fingerprint
from training.small_scene_round import train,finish
import json


def compose_training(previous,scene_rows,replay,include_new):
    """Replace source-specific black replay; never append duplicate weighting."""
    black={r['image_sha256'] for r in scene_rows}
    paths={str(Path(r['image_path']).resolve()) for r in scene_rows}
    ordinary=[r for r in previous if str(Path(r['image_path']).resolve()) not in paths
              and r.get('image_sha256') not in black]
    eligible=[r for r in scene_rows if include_new or not r['new_source']]
    return ordinary+eligible*replay


def prepare(root):
    if root.exists():raise FileExistsError('Round directory already exists')
    base=REPO/'training/runs/screen_corner_single/round_004'
    initial=base/'checkpoints/screen_corner_single.pt'
    old=base/'data/with_immersive'
    dataset=json.loads((REPO/f'data/splits/cross_project/{SLUG}_v1.json').read_text())
    replay=6
    previous=read_rows(old/'train.jsonl')
    root.mkdir(parents=True)
    for condition in ['control','with_black']:
        target=root/'data'/condition;shutil.copytree(old,target)
        for role in ['train','validation','holdout']:
            write_rows(target/f'black_{role}.jsonl',dataset[role])
        training=compose_training(previous,dataset['train'],replay,condition=='with_black')
        write_rows(target/'train.jsonl',training)
        # Existing black development pages are already in pooled validation.
        validation=read_rows(target/'validation.jsonl')
        known={str(Path(r['image_path']).resolve()) for r in validation}
        validation += [r for r in dataset['validation'] if str(Path(r['image_path']).resolve()) not in known]
        write_rows(target/'validation.jsonl',validation)
        train_paths={str(Path(r['image_path']).resolve()) for r in training}
        for path in target.glob('*.jsonl'):
            if path.stem=='validation' or path.stem.endswith('_validation') or path.stem.endswith('holdout'):
                if train_paths & {str(Path(r['image_path']).resolve()) for r in read_rows(path)}:
                    raise ValueError(f'Training/evaluation overlap: {path.name}')
    full=read_rows(root/'data/with_black/train.jsonl')
    settings=dict(architecture='screen_corner_single_v1',metric_contract=METRIC_CONTRACT,
                  initial_model=str(initial),initial_sha256=fingerprint(initial),base_run=str(base),
                  dataset=dataset['metadata'],data_prefix='black',seed=7,coarse_epochs=4,local_epochs=8,
                  steps_per_epoch=math.ceil(len(full)/32),batch_size=32,replay_factor=replay,
                  conditions=['control','with_black'],loss_condition='point_hit',
                  train_entries={c:len(read_rows(root/f'data/{c}/train.jsonl')) for c in ['control','with_black']},
                  unique_training_pages=len({r['image_path'] for r in full}),
                  validation_pages=len(read_rows(root/'data/with_black/validation.jsonl')),
                  black_train_unique=len(dataset['train']),
                  new_black_train_unique=sum(r['new_source'] for r in dataset['train']),
                  selection='pooled development point_le_0_01_ratio only; initial, control, with_black order on ties',
                  controlled_variable='both arms uniformly replay old consolidated black-scene training pages 6x; only with_black includes fresh black training pages at the same 6x rate',
                  no_score_target=True,regression_used_for_selection=False,
                  limitations=['one seed; multiple talks but same venue and similar camera views',
                               'all black-scene pictures visually reviewed; regression is observed',
                               'old split exposure is preserved; replayed entries are not independent images'])
    write_json(root/'manifest.json',settings)
    write_json(root/'reports/data_audit.json',dataset['metadata'])
    (root/'artifacts').mkdir()
    shutil.copyfile(REPO/f'data/curated/scenes/{SLUG}_v1/contact.jpg',root/'artifacts/black_scene_contact.jpg')
    for file in [Path(__file__),ENGINE/'training/build_black_scene_dataset.py',
                 ENGINE/'training/small_scene_round.py',ENGINE/'training/import_screen_project.py',
                 ENGINE/'training/train_screen_corner_single.py',ENGINE/'screen_corner_single.py']:
        shutil.copyfile(file,root/'artifacts'/file.name)
    print(json.dumps(settings,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','finish'])
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    globals()[a.action](a.root.resolve())
