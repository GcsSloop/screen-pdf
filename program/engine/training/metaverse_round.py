"""Controlled dataset-addition experiment; fixed one-point metric and one-file model."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
import torch.mps

ENGINE=Path(__file__).resolve().parents[1]
REPO=ENGINE.parents[1]
sys.path.insert(0,str(ENGINE))
from dataset_benchmark import METRIC_CONTRACT,summarize_point_hits
from screen_corner_single import SingleModelPredictor
from training.train_screen_corner_single import run
from training.screen_corner_ablation import read_rows,write_rows,write_json,fingerprint

DATASET='20260929-metaverse-xingjie'
BASE=REPO/'training/runs/screen_corner_single/round_001'
INITIAL=BASE/'checkpoints/screen_corner_single.pt'


def prepare(root):
    if (root/'manifest.json').exists():raise FileExistsError('Experiment already frozen')
    root.mkdir(parents=True,exist_ok=True)
    dataset=json.loads((REPO/f'data/splits/cross_project/{DATASET}_v2.json').read_text())
    if (root/'data').exists():
        archive=root/'artifacts/pretraining_split_v1';archive.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(root/'data',archive)
    update={r['image_path']:r for r in read_rows(REPO/'data/curated/annotations/202610-zhuiguang-ai_pages_v2.jsonl')}
    for condition in ['control','with_metaverse']:
        target=root/'data'/condition;shutil.copytree(BASE/'data',target)
        for name in ['train','validation','old_validation','new_validation','holdout']:
            rows=read_rows(target/f'{name}.jsonl')
            for row in rows:
                if row['image_path'] in update:row['manual_quad']=update[row['image_path']]['manual_quad']
            write_rows(target/f'{name}.jsonl',rows)
        write_rows(target/'validation.jsonl',read_rows(target/'validation.jsonl')+dataset['validation'])
        if condition=='with_metaverse':write_rows(target/'train.jsonl',read_rows(target/'train.jsonl')+dataset['train'])
        for name in ['train','validation','holdout']:write_rows(target/f'metaverse_{name}.jsonl',dataset[name])
        write_json(target/'metaverse_manifest.json',dataset['metadata'])
    train=read_rows(root/'data/with_metaverse/train.jsonl')
    validation=read_rows(root/'data/with_metaverse/validation.jsonl')
    paths=lambda rows:{str(Path(r['image_path']).resolve()) for r in rows}
    intersections={}
    for name in ['validation','holdout','mall_holdout','metaverse_holdout']:
        other=read_rows(root/f'data/with_metaverse/{name}.jsonl')
        intersections[name]=len(paths(train)&paths(other))
        if intersections[name]:raise ValueError(f'Train path overlap: {name}')
    # Match exact bytes even when a photo was copied/renamed into the new project.
    all_old={r['image_path']:r for name in ['train','validation','holdout','mall_holdout']
             for r in read_rows(BASE/f'data/{name}.jsonl')}
    old_hashes={fingerprint(path) for path in all_old}
    new_hashes={r['image_sha256'] for name in ['train','validation','holdout'] for r in dataset[name]}
    if old_hashes&new_hashes:raise ValueError('New project duplicates existing benchmark/training bytes')
    for key in ['image_sha256','capture_group','image_path']:
        sets={n:{r[key] for r in dataset[n]} for n in ['train','validation','holdout']}
        if any(sets[a]&sets[b] for a,b in [('train','validation'),('train','holdout'),('validation','holdout')]):
            raise ValueError(f'New split overlap: {key}')
    settings={'architecture':'screen_corner_single_v1','metric_contract':METRIC_CONTRACT,
              'initial_model':str(INITIAL),'initial_sha256':fingerprint(INITIAL),
              'dataset':dataset['metadata'],'seed':7,'coarse_epochs':8,'local_epochs':24,
              'steps_per_epoch':math.ceil(len(train)/32),'batch_size':32,
              'training':{'control':1097,'with_metaverse':len(train)},'validation_pages':len(validation),
              'conditions':['control','with_metaverse'],'loss_condition':'point_hit',
              'selection':'pooled development point_le_0_01_ratio only; initialization first on ties, then listed condition order',
              'controlled_variable':'add new project training pages; both arms use same initial global/local weights, seed, losses, optimizer steps, validation and budgets',
              'holdout_used_for_selection':False,'fresh_holdout_opened':False}
    write_json(root/'manifest.json',settings)
    write_json(root/'reports/data_audit.json',{'new_pages':dataset['metadata']['page_count'],
               'groups':dataset['metadata']['content_groups'],'split_counts':dataset['metadata']['split_counts'],
               'train_path_intersections':intersections,'exact_duplicates_with_any_old_split':0,
               'new_content_group_intersections':0,'annotation_revision':2,
               'source_project_sha256':dataset['metadata']['source_project_sha256']})
    art=root/'artifacts';art.mkdir(exist_ok=True)
    shutil.copyfile(REPO/'.build/metaverse-contact.jpg',art/'screen_contact.jpg')
    print(json.dumps(settings,ensure_ascii=False),flush=True)


def train(root):
    settings=json.loads((root/'manifest.json').read_text())
    for condition in settings['conditions']:
        cell=root/'cells'/condition
        if (cell/'reports/frozen_selection.json').exists():continue
        run(cell,settings['coarse_epochs'],settings['local_epochs'],seed=settings['seed'],
            data_source=root/'data'/condition,initial_model=INITIAL,steps_per_epoch=settings['steps_per_epoch'],
            conditions=['point_hit'],skip_evaluation=True)
        gc.collect()
        if torch.backends.mps.is_available():torch.mps.empty_cache()
    finish(root)


def evaluate(path,rows):
    model=SingleModelPredictor(path);records=[];final=[];coarse=[]
    for i,row in enumerate(rows):
        image=cv2.imread(row['image_path'])
        if image is None:
            final.append(None);coarse.append(None)
            records.append({'image_path':row['image_path'],'error':'unreadable'})
            continue
        output=model.predict(image)
        final.append(output['final_quad']);coarse.append(output['coarse_quad'])
        records.append({'image_path':row['image_path'],'manual_quad':row['manual_quad'],
                        'final_quad':output['final_quad'].tolist(),'coarse_quad':output['coarse_quad'].tolist()})
        if i%100==0:print(f'{path.parent.name} evaluation {i+1}/{len(rows)}',flush=True)
    manuals=[r['manual_quad'] for r in rows]
    return {'final':summarize_point_hits(manuals,final),'coarse':summarize_point_hits(manuals,coarse)},records


def finish(root):
    torch.set_num_threads(4)
    settings=json.loads((root/'manifest.json').read_text())
    data=root/'data/with_metaverse'
    paths={'initial':INITIAL,**{c:root/f'cells/{c}/checkpoints/screen_corner_single.pt' for c in settings['conditions']}}
    validation=read_rows(data/'validation.jsonl')
    evaluated={};raw={}
    for label,path in paths.items():
        evaluated[label],raw[label]=evaluate(path,validation)
        write_json(root/f'reports/{label}_validation_predictions.json',raw[label])
    best=max(paths,key=lambda label:evaluated[label]['final']['point_le_0_01_ratio'])
    frozen={'metric_contract':METRIC_CONTRACT,'policy':settings['selection'],'selected':best,
            'models':{label:{'path':str(path),'sha256':fingerprint(path),'validation':evaluated[label]['final']} for label,path in paths.items()},
            'fresh_holdout_used_for_selection':False}
    frozen_path=root/'reports/frozen_selection.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text())!=frozen:
        raise ValueError('Selection was already frozen; do not tune after test access')
    write_json(frozen_path,frozen)
    checkpoints=root/'checkpoints';checkpoints.mkdir(exist_ok=True)
    shutil.copyfile(paths[best],checkpoints/'screen_corner_single.pt')
    # Mark access BEFORE opening the fresh test, even if inference later fails.
    settings['fresh_holdout_opened']=True
    settings['frozen_selection_sha256']=fingerprint(frozen_path)
    write_json(root/'manifest.json',settings)
    ds=json.loads((REPO/f'data/splits/cross_project/{DATASET}_v2.json').read_text())
    ds['metadata'].update(holdout_opened=True,holdout_opened_by=str(root),frozen_selection_sha256=fingerprint(frozen_path))
    write_json(REPO/f'data/splits/cross_project/{DATASET}_v2.json',ds)
    write_json(REPO/f'training/registry/{DATASET}.json',ds['metadata'])
    results={label:{} for label in paths}
    for label in paths:
        records={r['image_path']:r for r in raw[label]}
        for name in ['old_validation','new_validation','metaverse_validation']:
            split=read_rows(data/f'{name}.jsonl')
            results[label][name]={'final':summarize_point_hits([r['manual_quad'] for r in split],[records[r['image_path']].get('final_quad') for r in split]),
                                 'coarse':summarize_point_hits([r['manual_quad'] for r in split],[records[r['image_path']].get('coarse_quad') for r in split])}
        for name in ['mall_holdout','holdout','metaverse_holdout']:
            results[label][name],predictions=evaluate(paths[label],read_rows(data/f'{name}.jsonl'))
            write_json(root/f'reports/{label}_{name}_predictions.json',predictions)
        print(label,json.dumps(results[label]),flush=True)
    write_json(root/'reports/evaluation.json',{'metric_contract':METRIC_CONTRACT,'selection':frozen,
               'results':results,'artifact':str(checkpoints/'screen_corner_single.pt'),
               'artifact_sha256':fingerprint(checkpoints/'screen_corner_single.pt'),
               'status':'trained_candidate_not_promoted',
               'limitations':['one seed','new talk shares screen and camera across subsets','development sets reused for selection',
                              'old holdouts are previously observed; new metaverse holdout opened only after all selection was frozen']})
    print('FINISHED',best,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','finish']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.root.resolve())
