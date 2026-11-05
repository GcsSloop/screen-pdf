"""Import explicit human quadrilaterals; deterministic screen-content group split."""
from __future__ import annotations
import argparse
from collections import defaultdict
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
from perspective_detect import order_points
from supervision_utils import resolve_manual_quad
from project_file import resolve_image_path
from dataset_benchmark import METRIC_CONTRACT


def content_groups(rows, threshold=8):
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,a in enumerate(rows):
        for j,b in enumerate(rows[:i]):
            exact=a['image_sha256']==b['image_sha256']
            similar=(int(a['screen_dhash'],16)^int(b['screen_dhash'],16)).bit_count()<=threshold
            perceptual = (bool(a.get('screen_phash')) and bool(b.get('screen_phash'))
                          and (int(a['screen_phash'],16)^int(b['screen_phash'],16)).bit_count()<=8)
            if exact or similar or perceptual:parents[find(i)]=find(j)
    result=defaultdict(list)
    for i,row in enumerate(rows):result[find(i)].append(row)
    return list(result.values())


def split_groups(groups,seed=20261017):
    # All members of a connected near-duplicate group stay in the same split.
    ordered=sorted(groups,key=lambda group:sorted(r['page_id'] for r in group))
    random.Random(seed).shuffle(ordered)
    if len(ordered)<3:raise ValueError('Need at least three screen-content groups')
    count=sum(map(len,ordered));targets={'train':count*.6,'validation':count*.2,'holdout':count*.2}
    splits={k:[] for k in targets}
    for group in ordered:
        name=max(targets,key=lambda k:(targets[k]-len(splits[k]))/max(targets[k],1))
        gid=hashlib.sha256('|'.join(sorted(r['image_sha256'] for r in group)).encode()).hexdigest()[:16]
        for row in group:row.update(capture_group=gid,split=name)
        splits[name].extend(group)
    if any(not rows for rows in splits.values()):raise ValueError('Unbalanced groups prevent three independent subsets')
    return splits


def extract_project(source,slug):
    """Read human labels without assigning a new split or writing source files."""
    project=source/'screen-pdf-project.json';raw=project.read_bytes();payload=json.loads(raw)
    rows=[];excluded=[];manual_sources=defaultdict(int)
    for p in payload['pages']:
        quad,kind=resolve_manual_quad(p)
        if quad is None:
            excluded.append({'page_id':p['id'],'reason':'no explicit manual quadrilateral'});continue
        image_path=resolve_image_path(project,p,payload)
        if image_path is None:raise FileNotFoundError(p.get('path') or p.get('name'))
        image=cv2.imread(str(image_path))
        if image is None:raise FileNotFoundError(image_path)
        height,width=image.shape[:2]
        points=order_points(np.array(quad,np.float32))
        if not np.isfinite(points).all() or not cv2.isContourConvex(points) or abs(cv2.contourArea(points))<100:
            raise ValueError(f'Invalid human quadrilateral: {p["name"]}')
        if (points.min(0)<[-1,-1]).any() or (points.max(0)>[width+1,height+1]).any():
            raise ValueError(f'Human point outside image: {p["name"]}')
        dst=np.array([[0,0],[255,0],[255,143],[0,143]],np.float32)
        crop=cv2.warpPerspective(image,cv2.getPerspectiveTransform(points,dst),(256,144))
        blurred=cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),(7,7),0)
        gray=cv2.resize(blurred,(9,8),interpolation=cv2.INTER_AREA)
        dhash=np.packbits(gray[:,1:]>gray[:,:-1]).tobytes().hex()
        frequency=cv2.dct(cv2.resize(blurred,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32))[:8,:8]
        frequency[0,0]=0
        phash=np.packbits(frequency>np.median(frequency)).tobytes().hex()
        rows.append({'project_path':str(project),'project_name':source.name,'page_id':p['id'],
                     'image_path':str(image_path.resolve()),'manual_quad':points.tolist(),
                     'image_width':width,'image_height':height,'dataset_slug':slug,
                     'annotation_source':kind,'image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
                     'screen_dhash':dhash,'screen_phash':phash,'scene_tags':[]})
        manual_sources[kind]+=1
    return rows,excluded,dict(manual_sources),hashlib.sha256(raw).hexdigest()


def import_project(source,slug,revision=1):
    rows,excluded,manual_sources,source_digest=extract_project(source,slug)
    groups=content_groups(rows)
    split=split_groups(groups)
    annotation=REPO/f'data/curated/annotations/{slug}_pages_v{revision}.jsonl'
    split_path=REPO/f'data/splits/cross_project/{slug}_v{revision}.json'
    if annotation.exists() or split_path.exists():raise FileExistsError('Dataset revision already exists')
    alias=REPO/'data/raw'/slug
    if not alias.exists():alias.symlink_to(source,target_is_directory=True)
    metadata={'dataset_slug':slug,'source_root':str(source),'source_project_sha256':source_digest,
              'raw_alias':str(alias),'annotation_file':str(annotation),'split_file':str(split_path),
              'page_count':len(rows),'excluded':excluded,'annotation_sources':dict(manual_sources),
              'content_groups':len(groups),'group_sizes':sorted(map(len,groups),reverse=True),
              'split_counts':{k:len(v) for k,v in split.items()},'seed':20261017,
              'grouping':'rectified 256x144 screen, Gaussian blur, area resize; dHash <=8 OR pHash <=8, transitive closure, plus exact file duplicates',
              'annotation_revision':revision,
              'split_policy':'approximately 60/20/20; whole content groups stay together',
              'metric_contract':METRIC_CONTRACT,'holdout_opened':False,
              'interpretation':'same talk/screen/camera; not cross-event generalization'}
    annotation.parent.mkdir(parents=True,exist_ok=True)
    annotation.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    split_path.write_text(json.dumps({'metadata':metadata,**split},ensure_ascii=False,indent=2)+'\n')
    (REPO/f'training/registry/{slug}.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))
    return split

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--slug',required=True)
    p.add_argument('--revision',type=int,default=1)
    a=p.parse_args();import_project(a.source.resolve(),a.slug,a.revision)
