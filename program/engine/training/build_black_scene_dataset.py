"""One black-background scene across talks, preserving historical split exposure."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ENGINE=Path(__file__).resolve().parents[1]
REPO=ENGINE.parents[1]
sys.path.insert(0,str(ENGINE))
from dataset_benchmark import METRIC_CONTRACT
from training.import_screen_project import content_groups,extract_project
from training.screen_corner_ablation import read_rows,write_rows,write_json,fingerprint

SLUG='20260929-black-background-unified'
NEW_SOURCE=Path('/Users/gcssloop/WorkSpace/AIGC/展会/2026/202609 上海-元宇宙产业嘉年华/2026-09-29 黑色背景')
PREVIOUS=REPO/'training/runs/screen_corner_single/round_004/data/with_immersive'
EXCLUSIONS={
    '20260929-xr-zhouqunhua':{'IMG_1610.jpeg':'bright full-photo page',
        'IMG_1619.jpeg':'photo-dominant page; retained in ordinary dataset',
        'IMG_1621.jpeg':'photo-dominant page; retained in ordinary dataset'},
    '20260929-immersive-lilinlin':{'IMG_1630.jpeg':'light background', 'IMG_1633.jpeg':'light background'}}


def deduplicate(rows):
    """Keep all provenance; exact duplicate labels/split roles must agree."""
    result={}
    for row in rows:
        key=row['image_sha256']
        origin={k:row.get(k) for k in ['source_dataset_slug','source_page_id','project_path','image_path','historical_split']}
        if key not in result:
            result[key]=dict(row,provenance=[origin])
            continue
        existing=result[key]
        if not np.allclose(existing['manual_quad'],row['manual_quad'],atol=.5,rtol=0):
            raise ValueError('Conflicting human labels on duplicate image')
        roles={r for r in [existing.get('historical_split'),row.get('historical_split')] if r}
        if len(roles)>1:
            raise ValueError('Duplicate image has conflicting historical splits')
        existing['historical_split']=next(iter(roles),None)
        existing['new_source']=bool(existing.get('new_source') and row.get('new_source'))
        existing['provenance'].append(origin)
    return list(result.values())


def assign_groups(groups,seed=20261017):
    """Old exposure is locked; never reshuffle old evaluation into training."""
    result={k:[] for k in ['train','validation','holdout']}
    pending=[];fresh_counts=Counter()
    fresh_total=sum(not row.get('historical_split') for group in groups for row in group)
    def assign(group,role):
        gid=hashlib.sha256('|'.join(sorted(r['image_sha256'] for r in group)).encode()).hexdigest()[:16]
        for row in group:
            if not row.get('historical_split'):fresh_counts[role]+=1
            row.update(split=role,capture_group=gid)
        result[role].extend(group)
    for group in groups:
        locks={r['historical_split'] for r in group if r.get('historical_split')}
        if len(locks)>1:
            raise ValueError('Near-duplicate group conflicts with historical split exposure')
        if locks:assign(group,next(iter(locks)))
        else:pending.append(group)
    pending.sort(key=lambda g:sorted(r['image_sha256'] for r in g))
    random.Random(seed).shuffle(pending)
    weights={'train':.6,'validation':.2,'holdout':.2}
    targets={k:max(fresh_total*w,1) for k,w in weights.items()}
    for group in pending:
        role=max(targets,key=lambda k:(targets[k]-fresh_counts[k])/targets[k])
        assign(group,role)
    return result


def historical_exposure(data):
    paths={}
    for file in data.glob('*.jsonl'):
        name=file.stem
        if name=='train':role='train'
        elif name=='validation':role='validation'
        elif name.endswith('holdout'):role='holdout'
        else:continue
        for row in read_rows(file):paths.setdefault(str(Path(row['image_path']).resolve()),set()).add(role)
    result={}
    for path,roles in paths.items():result.setdefault(fingerprint(path),set()).update(roles)
    if any(len(roles)>1 for roles in result.values()):
        raise ValueError('Historical training/evaluation overlap must be resolved explicitly')
    return {digest:next(iter(roles)) for digest,roles in result.items()}


def build():
    annotation=REPO/f'data/curated/annotations/{SLUG}_pages_v1.jsonl'
    split_path=REPO/f'data/splits/cross_project/{SLUG}_v1.json'
    if annotation.exists() or split_path.exists():raise FileExistsError('Frozen dataset revision already exists')
    rows=[];sources=[];excluded=[]
    for slug,names in EXCLUSIONS.items():
        source_rows=read_rows(REPO/f'data/curated/annotations/{slug}_pages_v1.jsonl')
        project=Path(source_rows[0]['project_path'])
        sources.append(dict(dataset_slug=slug,project_path=str(project),project_sha256=fingerprint(project),
                            annotation_path=str(REPO/f'data/curated/annotations/{slug}_pages_v1.jsonl')))
        for row in source_rows:
            name=Path(row['image_path']).name
            if name in names:
                excluded.append(dict(source=slug,page_id=row['page_id'],name=name,reason=names[name]));continue
            row=dict(row,source_dataset_slug=slug,source_page_id=row['page_id'],
                     historical_split=row['split'],source_capture_group=row['capture_group'],new_source=False)
            rows.append(row)
    new_slug='20260929-black-background-extra'
    new,missing,_,digest=extract_project(NEW_SOURCE,new_slug)
    if missing:raise ValueError('New black-scene project contains pages without human labels')
    sources.append(dict(dataset_slug=new_slug,project_path=str(NEW_SOURCE/'screen-pdf-project.json'),project_sha256=digest))
    history=historical_exposure(PREVIOUS)
    for row in rows:
        if fingerprint(row['image_path'])!=row['image_sha256']:raise ValueError('Previously curated image changed')
        if history.get(row['image_sha256'])!=row['historical_split']:raise ValueError('Historical exposure changed')
    for row in new:
        rows.append(dict(row,source_dataset_slug=new_slug,source_page_id=row['page_id'],
                         historical_split=history.get(row['image_sha256']),new_source=True))
    before=len(rows);rows=deduplicate(rows)
    for row in rows:
        row.update(dataset_slug=SLUG,page_id=row['source_dataset_slug']+':'+row['source_page_id'],
                   scene_tags=['black_background','blue_side_panels','same_venue'],
                   scene_selection='human visual review; predominantly dark main-screen background')
    groups=content_groups(rows);split=assign_groups(groups)
    for key in ['image_path','image_sha256','capture_group']:
        seen=set()
        for role,part in split.items():
            current={r[key] for r in part}
            if seen&current:raise ValueError(f'Merged scene split overlap: {key}')
            seen.update(current)
    metadata=dict(dataset_slug=SLUG,scene_name='black_background_blue_sides',sources=sources,
                  annotation_file=str(annotation),split_file=str(split_path),page_count=len(rows),
                  source_counts=dict(Counter(r['source_dataset_slug'] for r in rows)),
                  excluded=excluded,exact_duplicate_rows_removed=before-len(rows),content_groups=len(groups),
                  group_sizes=sorted(map(len,groups),reverse=True),split_counts={k:len(v) for k,v in split.items()},
                  new_split_counts={k:sum(r['new_source'] for r in v) for k,v in split.items()},
                  seed=20261017,annotation_revision=1,metric_contract=METRIC_CONTRACT,
                  grouping='screen dHash <=8 OR pHash <=8 OR exact bytes; whole connected groups stay together',
                  split_policy='historical exposure locked; fresh groups approximately 60/20/20',
                  holdout_opened=True,holdout_previously_observed=True,
                  interpretation='multiple talks in the same venue; all pictures visually audited; not a blind or cross-venue evaluation',
                  objective='one black-scene dataset with uniform training sampling; no small-sample score target')
    alias=REPO/'data/raw'/new_slug
    if not alias.exists():alias.symlink_to(NEW_SOURCE,target_is_directory=True)
    write_rows(annotation,[r for part in split.values() for r in part])
    write_json(split_path,dict(metadata=metadata,**split))
    write_json(REPO/f'training/registry/{SLUG}.json',metadata)
    scene=REPO/f'data/curated/scenes/{SLUG}_v1'
    scene.mkdir(parents=True,exist_ok=False)
    write_json(scene/'manifest.json',metadata)
    for row in rows:
        link=scene/'images'/row['source_dataset_slug']/Path(row['image_path']).name
        link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(Path(row['image_path']))
    width=4*400;height=((len(rows)+3)//4)*300
    canvas=np.full((height,width,3),240,np.uint8)
    for i,row in enumerate(sorted(rows,key=lambda r:r['page_id'])):
        image=cv2.imread(row['image_path']);h,w=image.shape[:2];small=cv2.resize(image,(390,260))
        points=(np.asarray(row['manual_quad'])*[390/w,260/h]).astype(np.int32)
        cv2.polylines(small,[points],True,(0,220,0),1)
        x=i%4*400;y=i//4*300;canvas[y:y+260,x:x+390]=small
        cv2.putText(canvas,Path(row['image_path']).name+' '+row['split'],(x+4,y+282),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
    cv2.imwrite(str(scene/'contact.jpg'),canvas)
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
