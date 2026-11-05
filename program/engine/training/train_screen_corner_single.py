"""Point-only, single-file coarse/local training, controlled loss comparison.

Run directories are append-only experiments. Test/regression pages are evaluated
only after the development-selected artifact has been frozen and hashed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import torch.mps
import torch.nn.functional as F
from torch import nn

ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parents[1]
sys.path.insert(0, str(ENGINE))
from corner_train import build_corner_heatmaps
from dataset_benchmark import METRIC_CONTRACT, summarize_point_hits
from screen_corner_single import (ScreenCornerSingle, GlobalPyramid, LocalPointHead,
    decode_points, patch_boxes, sample_corner_patches, SingleModelPredictor)
from training.screen_corner_ablation import make_encoder, read_rows, write_json, fingerprint

PREVIOUS = REPO / 'training/runs/screen_corner_unified_20261017'
WARM = REPO / 'training/runs/screen_corner_unified_20261017_longhead/cells/pyramid_boundary/seed_007/model.pt'

class TupleEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, x):
        d = self.encoder(x)
        return d['low'], d['mid'], d['high']


def hit_loss(pred, target, size):
    """Differentiable per-corner 1% objective, measured in original pixels."""
    pixel_target = target * size[:, None]
    diag = (pixel_target.amax(1) - pixel_target.amin(1)).norm(dim=-1).clamp(min=1.)
    distance = (((pred-target) * size[:, None]).square().sum(-1) + 1e-10).sqrt() / diag[:, None]
    return torch.sigmoid((distance - .01) / .003).mean()


def teacher_point_errors(prediction, target, size):
    """Per semantic channel; all inputs normalized except original image size."""
    diag = np.linalg.norm(np.ptp(target, axis=0) * size)
    return np.linalg.norm((prediction-target) * size, axis=1) / max(diag, 1.)


def positive_guard_loss(pred, target, teacher, mask, sizes):
    """Protect teacher-correct training points without pulling back improvements.

    This is a GT-directed regression adaptation, not the classification PCT loss.
    A teacher mistake or a student closer to GT contributes exactly zero.
    """
    student_distance = (((pred-target) * sizes[:, None]).square().sum(-1) + 1e-10).sqrt()
    teacher_distance = (((teacher-target) * sizes[:, None]).square().sum(-1) + 1e-10).sqrt().detach()
    diag = ((target.amax(1)-target.amin(1))*sizes).norm(dim=-1).clamp(min=1.)[:, None]
    return (F.relu(student_distance/diag-teacher_distance/diag)*mask).sum() / mask.sum().clamp(min=1.)


def score(rows, predictions):
    manual = [r['manual_quad'] for r in rows]
    pixels = [p * [r['image_width'], r['image_height']] for r, p in zip(rows, predictions)]
    return summarize_point_hits(manual, pixels)


def evaluate_global(head, rows, cache, device):
    head.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            group = rows[start:start+32]
            features = tuple(torch.stack([cache[r['image_path']]['features'][i] for r in group]).to(device) for i in range(3))
            logits, _, _ = head(features)
            output.extend(decode_points(logits).cpu().numpy())
    return score(rows, output)


def prepare(root, data_source=None):
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(data_source or PREVIOUS / 'data', root / 'data')
    rows = {name: read_rows(root / f'data/{name}.jsonl') for name in
            ['train', 'validation', 'old_validation', 'new_validation', 'mall_holdout', 'holdout']}
    train = {str(Path(r['image_path']).resolve()) for r in rows['train']}
    for name in ['validation', 'mall_holdout', 'holdout']:
        assert not train & {str(Path(r['image_path']).resolve()) for r in rows[name]}, name
    return rows


def training_order(count, generator, steps=None, batch_size=32):
    order = torch.randperm(count, generator=generator).tolist()
    if steps is None:
        return order
    required = steps * batch_size
    while len(order) < required:
        order.extend(torch.randperm(count, generator=generator).tolist())
    return order[:required]


def run(root, coarse_epochs=8, local_epochs=24, seed=7, *, data_source=None,
        initial_model=None, teacher_model=None, teacher_guard_weight=0.5,
        steps_per_epoch=None, conditions=None, skip_evaluation=False):
    if conditions and 'teacher_guard' in conditions and teacher_model is None:
        raise ValueError('teacher_guard requires an explicitly pinned teacher model')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    rows = prepare(root, data_source)
    settings = {'architecture': 'screen_corner_single_v1', 'artifact_format': 'torchscript',
                'metric_contract': METRIC_CONTRACT, 'selection': 'pooled validation point_le_0_01_ratio only; earliest checkpoint on ties',
                'seed': seed, 'coarse_epochs': coarse_epochs, 'local_epochs': local_epochs,
                'batch_size': 32, 'coarse_lr': 1e-4, 'local_lr': 3e-4,
                'conditions': conditions or ['coordinate', 'point_hit'],
                'controlled_variable': ('local positive guard loss; all data and global training shared'
                    if conditions and 'teacher_guard' in conditions else
                    'local hit-loss weight: 0 versus 0.1' if not conditions else 'external dataset comparison; loss fixed'),
                'global_init': str(initial_model or WARM), 'global_init_sha256': fingerprint(initial_model or WARM),
                'teacher_model': str(teacher_model) if teacher_model else None,
                'teacher_sha256': fingerprint(teacher_model) if teacher_model else None,
                'teacher_guard_weight': teacher_guard_weight if teacher_model else 0.0,
                'steps_per_epoch': steps_per_epoch, 'data_source': str(data_source or PREVIOUS / 'data'),
                'encoder_frozen': True, 'global_frozen_in_local_stage': True,
                'training_pages': len(rows['train']), 'validation_pages': len(rows['validation']),
                'regression_only': ['mall_holdout', 'holdout'], 'test_used_for_selection': False,
                'new_activity_train_pages': 17, 'augmentation': 'none, fixed controlled comparison',
                'device_training_heads': str(device), 'device_encoder_and_export': 'cpu',
                'environment': {'torch': torch.__version__, 'threads': 4},
                'data_hashes': {k: fingerprint(root / f'data/{k}.jsonl') for k in rows}}
    write_json(root / 'manifest.json', settings)
    initialized_local = None
    if initial_model is not None:
        initial = torch.jit.load(str(initial_model), map_location='cpu').eval()
        encoder = initial.encoder
        initialized_local = {k:v.detach().cpu().clone() for k,v in initial.local_head.state_dict().items()}
        global_state = initial.global_head.state_dict()
    else:
        checkpoint = torch.load(WARM, map_location='cpu')
        encoder = make_encoder(False)
        encoder.load_state_dict(checkpoint['encoder_state'])
        encoder = torch.jit.trace(TupleEncoder(encoder), torch.zeros(1, 3, 256, 256)).eval()
        global_state = checkpoint['state_dict']
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.save(str(root / 'encoder_training_cache.pt'))
    head = GlobalPyramid()
    head.load_state_dict(global_state)
    head.to(device)
    mean = torch.tensor([.485,.456,.406])[None,:,None,None]
    std = torch.tensor([.229,.224,.225])[None,:,None,None]
    allrows = {r['image_path']: r for r in rows['train'] + rows['validation']}
    cache = {}
    with torch.no_grad():
        for index, (path, row) in enumerate(allrows.items()):
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(path)
            h,w = image.shape[:2]
            row.update(image_width=w, image_height=h)
            rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2,0,1).copy())[None].float() / 255.
            small = F.interpolate(rgb, size=[256,256], mode='bilinear', align_corners=False)
            target = torch.tensor(row['manual_quad'], dtype=torch.float32) / torch.tensor([w,h])
            boundary = np.zeros((2,64,64), np.float32)
            pts = np.round(target.numpy()*63).astype(np.int32)
            cv2.fillConvexPoly(boundary[0], pts, 1.)
            cv2.polylines(boundary[1], [pts], True, 1., 2)
            cache[path] = {'features': tuple(f[0] for f in encoder((small-mean)/std)),
                           'target': target, 'size': torch.tensor([w,h]),
                           'heatmap': torch.from_numpy(build_corner_heatmaps(target.numpy())),
                           'boundary': torch.from_numpy(boundary)}
            if index % 100 == 0:
                print(f'global cache {index+1}/{len(allrows)}', flush=True)
    # Optional positive-congruent teacher targets.  The teacher is used only
    # during training and only on old training pages where it is within the
    # existing 1% point-hit rule.  New pages and teacher failures are masked.
    if teacher_model is not None:
        teacher = SingleModelPredictor(teacher_model)
        old_train_paths = {r['image_path'] for r in read_rows(root / 'data/train_old.jsonl')}
        teacher_ok = 0
        teacher_total = 0
        for index, row in enumerate(rows['train']):
            path = row['image_path']
            if path not in old_train_paths:
                continue
            image = cv2.imread(path)
            prediction = teacher.predict(image)['final_quad'] / np.array([image.shape[1], image.shape[0]], np.float32)
            target = cache[path]['target'].numpy()
            size = cache[path]['size'].numpy()
            teacher_error = teacher_point_errors(prediction, target, size)
            cache[path]['teacher_target'] = torch.from_numpy(prediction.astype(np.float32))
            cache[path]['teacher_ok'] = torch.from_numpy((teacher_error <= .01).astype(np.float32))
            teacher_ok += int((teacher_error <= .01).sum())
            teacher_total += 4
            if index % 100 == 0:
                print(f'teacher cache {index+1}/{len(rows["train"])}', flush=True)
        settings['teacher_old_train_corners'] = teacher_total
        settings['teacher_old_train_hits'] = teacher_ok
        settings['teacher_guard_scope'] = 'train_old.jsonl membership, original 1080-page replay subset; per original semantic corner channel'
        if teacher_ok == 0:
            raise ValueError('No teacher-correct training corners: inspect coordinates and mask before training')
        write_json(root / 'manifest.json', settings)
    # Ensure all row copies, including split subsets, use source pixel dimensions.
    for split in rows.values():
        for row in split:
            if row['image_path'] in cache:
                w,h = cache[row['image_path']]['size'].tolist()
                row.update(image_width=w, image_height=h)
    initial = evaluate_global(head, rows['validation'], cache, device)
    best_score = initial['point_le_0_01_ratio']
    best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    best_epoch = 0
    optimizer = torch.optim.Adam(head.parameters(), lr=settings['coarse_lr'])
    generator = torch.Generator().manual_seed(seed)
    history = [{'epoch':0,'validation': initial}]
    for epoch in range(1, coarse_epochs+1):
        head.train()
        order = training_order(len(rows['train']), generator, steps_per_epoch)
        losses = []
        for start in range(0,len(order),32):
            group = [rows['train'][i] for i in order[start:start+32]]
            entries = [cache[r['image_path']] for r in group]
            feat = tuple(torch.stack([e['features'][i] for e in entries]).to(device) for i in range(3))
            target = torch.stack([e['target'] for e in entries]).to(device)
            sizes = torch.stack([e['size'] for e in entries]).to(device)
            logits,boundary,_ = head(feat)
            loss = F.mse_loss(logits,torch.stack([e['heatmap'] for e in entries]).to(device))
            loss = loss + 2.5*F.smooth_l1_loss(decode_points(logits),target)
            loss = loss + .1*F.binary_cross_entropy_with_logits(boundary,torch.stack([e['boundary'] for e in entries]).to(device))
            loss = loss + .1*hit_loss(decode_points(logits),target,sizes)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate_global(head, rows['validation'], cache, device)
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)), 'validation':validation})
        if validation['point_le_0_01_ratio'] > best_score:
            best_score,best_epoch = validation['point_le_0_01_ratio'],epoch
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        write_json(root/'reports/coarse_history.json',history)
        print(f'coarse {epoch}/{coarse_epochs} point_hit={validation["point_le_0_01_ratio"]:.5f}',flush=True)
    head.load_state_dict(best_state)
    head.eval()
    torch.save({'state_dict':best_state,'best_epoch':best_epoch,'metric_contract':METRIC_CONTRACT}, root/'coarse_training_cache.pt')
    # Cache model-proposed crops, never human-centered crops, using the exact graph operations.
    with torch.no_grad():
        for index,(path,row) in enumerate(allrows.items()):
            item=cache[path]
            logits,_,feature = head(tuple(f[None].to(device) for f in item['features']))
            coarse=decode_points(logits).cpu()
            boxes=patch_boxes(coarse)
            image=cv2.imread(path)
            rgb=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy())[None].float()/255.
            item.update(coarse=coarse[0], boxes=boxes[0],
                        rgb=sample_corner_patches(rgb,boxes,64),
                        context=sample_corner_patches(feature.cpu(),boxes,16))
            del item['features'],item['boundary'],item['heatmap']
            if index%100==0:
                print(f'local cache {index+1}/{len(allrows)}',flush=True)
    torch.manual_seed(seed)
    initial_local = initialized_local if initialized_local is not None else LocalPointHead().state_dict()
    results=[]
    for condition in settings['conditions']:
        local=LocalPointHead().to(device)
        local.load_state_dict(initial_local)
        optimizer=torch.optim.Adam(local.parameters(),lr=settings['local_lr'])
        generator=torch.Generator().manual_seed(seed)
        cell=root/'cells'/condition
        cell.mkdir(parents=True)
        def evaluate():
            local.eval()
            predictions=[]
            with torch.no_grad():
                for start in range(0,len(rows['validation']),32):
                    group=rows['validation'][start:start+32]
                    entries=[cache[r['image_path']] for r in group]
                    rgb=torch.cat([e['rgb'] for e in entries]).to(device)
                    context=torch.cat([e['context'] for e in entries]).to(device)
                    rel,_=local(rgb,context)
                    boxes=torch.stack([e['boxes'] for e in entries]).to(device)
                    pred=(boxes[...,:2]+rel.view(-1,4,2)*(boxes[...,2:]-boxes[...,:2])).clamp(0.,1.)
                    predictions.extend(pred.cpu().numpy())
            return score(rows['validation'],predictions)
        initial=evaluate()
        best=initial['point_le_0_01_ratio']
        best_epoch=0
        state={k:v.detach().cpu().clone() for k,v in local.state_dict().items()}
        history=[{'epoch':0,'validation':initial}]
        for epoch in range(1,local_epochs+1):
            local.train()
            losses=[]
            order=training_order(len(rows['train']),generator,steps_per_epoch)
            for start in range(0,len(order),32):
                entries=[cache[rows['train'][i]['image_path']] for i in order[start:start+32]]
                rgb=torch.cat([e['rgb'] for e in entries]).to(device)
                context=torch.cat([e['context'] for e in entries]).to(device)
                target=torch.stack([e['target'] for e in entries]).to(device)
                sizes=torch.stack([e['size'] for e in entries]).to(device)
                boxes=torch.stack([e['boxes'] for e in entries]).to(device)
                relative,logits=local(rgb,context)
                span=boxes[...,2:]-boxes[...,:2]
                target_relative=(target-boxes[...,:2])/span
                inside=((target_relative>=0)&(target_relative<=1)).all(-1).flatten()
                axis=torch.linspace(0.,1.,16,device=device)
                gy,gx=torch.meshgrid(axis,axis,indexing='ij')
                distance=(gx[None]-target_relative.reshape(-1,2)[:,0,None,None]).square()+(gy[None]-target_relative.reshape(-1,2)[:,1,None,None]).square()
                heatmap_target=torch.exp(-distance/(2*(1.5/15)**2))[:,None]
                heatmap_loss=(logits-heatmap_target).square().flatten(1).mean(1)
                heatmap_loss=(heatmap_loss*inside).sum()/inside.sum().clamp(min=1)
                relative_loss=F.smooth_l1_loss(relative,target_relative.reshape(-1,2).clamp(0.,1.))
                pred=(boxes[...,:2]+relative.view(-1,4,2)*span).clamp(0.,1.)
                loss=heatmap_loss+3.5*relative_loss
                if condition=='point_hit':
                    loss=loss+.1*hit_loss(pred,target,sizes)
                if condition=='teacher_guard':
                    loss=loss+.1*hit_loss(pred,target,sizes)
                    teacher_targets = torch.stack([e.get('teacher_target', torch.zeros_like(e['target'])) for e in entries]).to(device)
                    teacher_mask = torch.stack([e.get('teacher_ok', torch.zeros(4)) for e in entries]).to(device)
                    loss = loss + settings['teacher_guard_weight'] * positive_guard_loss(pred,target,teacher_targets,teacher_mask,sizes)
                optimizer.zero_grad();loss.backward();optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation=evaluate()
            history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation':validation})
            if validation['point_le_0_01_ratio']>best:
                best,best_epoch=validation['point_le_0_01_ratio'],epoch
                state={k:v.detach().cpu().clone() for k,v in local.state_dict().items()}
            write_json(cell/'history.json',history)
            print(f'{condition} {epoch}/{local_epochs} point_hit={validation["point_le_0_01_ratio"]:.5f}',flush=True)
        model=ScreenCornerSingle(encoder)
        model.global_head.load_state_dict(best_state)
        model.local_head.load_state_dict(state)
        model.eval()
        artifact=cell/'screen_corner_single.pt'
        scripted=torch.jit.script(model)
        scripted.save(str(artifact))
        result={'condition':condition,'best_epoch':best_epoch,'validation':{'point_le_0_01_ratio':best},
                'model':str(artifact),'sha256':fingerprint(artifact),'bytes':artifact.stat().st_size}
        results.append(result)
        write_json(cell/'result.json',result)
    # Stable max retains first condition for an exact tie, no second metric.
    chosen=max(results,key=lambda r:r['validation']['point_le_0_01_ratio'])
    final=root/'checkpoints/screen_corner_single.pt'
    final.parent.mkdir()
    shutil.copyfile(chosen['model'],final)
    write_json(root/'reports/frozen_selection.json',{'metric_contract':METRIC_CONTRACT,'cells':results,'chosen':chosen,
               'artifact':str(final),'sha256':fingerprint(final),'test_used_for_selection':False})
    print('FROZEN',str(final),flush=True)
    if not skip_evaluation:
        evaluate_artifact(root)


def evaluate_artifact(root):
    torch.set_num_threads(4)
    frozen=json.loads((root/'reports/frozen_selection.json').read_text())
    artifact=root/'checkpoints/screen_corner_single.pt'
    assert fingerprint(artifact)==frozen['sha256']
    predictor=SingleModelPredictor(artifact)
    reports={}
    for name in ['old_validation','new_validation','mall_holdout','holdout']:
        rows=read_rows(root/f'data/{name}.jsonl')
        predictions=[]; coarse=[]; records=[]; times=[]
        for index,row in enumerate(rows):
            image=cv2.imread(row['image_path'])
            if image is None:
                predictions.append(None);coarse.append(None)
                records.append({'image_path':row['image_path'],'error':'image unreadable'})
                continue
            t=time.perf_counter()
            result=predictor.predict(image)
            times.append((time.perf_counter()-t)*1000)
            predictions.append(result['final_quad']);coarse.append(result['coarse_quad'])
            records.append({'image_path':row['image_path'],'manual_quad':row['manual_quad'],
                            'coarse_quad':result['coarse_quad'].tolist(),'final_quad':result['final_quad'].tolist()})
            if index%100==0:
                print(f'eval {name} {index+1}/{len(rows)}',flush=True)
        reports[name]={'coarse':summarize_point_hits([r['manual_quad'] for r in rows],coarse),
                       'final':summarize_point_hits([r['manual_quad'] for r in rows],predictions)}
        write_json(root/f'reports/{name}_predictions.json',records)
        print(name,json.dumps(reports[name]),flush=True)
    # Compare exactly the same images against saved previous full-pipeline predictions.
    comparisons={}
    for label in ['r130','r142','nominated']:
        previous=json.loads((PREVIOUS/f'reports/{label}_stage_diagnostics.json').read_text())
        by_path={p['image_path']:p for p in previous['pages']}
        previous_test=json.loads((PREVIOUS/f'reports/{label}_final_holdout_stage_diagnostics.json').read_text())
        by_path.update({p['image_path']:p for p in previous_test['pages']})
        comparisons[label]={}
        for name in reports:
            split=read_rows(root/f'data/{name}.jsonl')
            comparisons[label][name]=summarize_point_hits([r['manual_quad'] for r in split],
                [by_path[r['image_path']]['quads']['selected'] for r in split])
    # Warmup + isolated CPU model graph, includes resizing/crops; excludes file IO.
    latency=[]
    warm=read_rows(root/'data/old_validation.jsonl')[:2]
    for row in warm:
        predictor.predict(cv2.imread(row['image_path']))
    profile_rows=read_rows(root/'data/old_validation.jsonl')[:20]+read_rows(root/'data/new_validation.jsonl')+read_rows(root/'data/holdout.jsonl')
    for row in profile_rows:
        image=cv2.imread(row['image_path'])
        start=time.perf_counter();predictor.predict(image)
        latency.append((time.perf_counter()-start)*1000)
    report={'metric_contract':METRIC_CONTRACT,'model':str(artifact),'sha256':frozen['sha256'],
            'selection':frozen['chosen'],'splits':reports,'previous_full_pipeline':comparisons,
            'latency_observation':{'pages':len(latency),'mean_ms':float(np.mean(latency)),
                                  'p95_ms':float(np.percentile(latency,95)),'max_ms':max(latency),'device':'cpu',
                                  'scope':'BGR to tensor + graph including all stages; excludes JPEG read, OCR, PDF'},
            'limitations':['new-event validation only 5 photos','previous holdout already observed; regression only',
                          'single seed controlled experiment','old/new validation has been used for previous model selection'],
            'status':'trained_candidate_not_promoted'}
    write_json(root/'reports/evaluation.json',report)
    print('DONE',json.dumps({'scores':reports,'latency':report['latency_observation']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','evaluate'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--coarse-epochs',type=int,default=8)
    p.add_argument('--local-epochs',type=int,default=24)
    p.add_argument('--data-source',type=Path)
    p.add_argument('--initial-model',type=Path)
    p.add_argument('--teacher-model',type=Path)
    p.add_argument('--teacher-guard-weight',type=float,default=0.5)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--steps-per-epoch',type=int)
    p.add_argument('--conditions',nargs='+',choices=['coordinate','point_hit','teacher_guard'])
    p.add_argument('--skip-evaluation',action='store_true')
    a=p.parse_args()
    if a.action=='train':
        if a.coarse_epochs<1 or a.local_epochs<1:
            p.error('epoch budgets must be positive')
        run(a.root.resolve(),a.coarse_epochs,a.local_epochs,seed=a.seed,data_source=a.data_source,
            initial_model=a.initial_model,teacher_model=a.teacher_model,
            teacher_guard_weight=a.teacher_guard_weight,steps_per_epoch=a.steps_per_epoch,
            conditions=a.conditions,skip_evaluation=a.skip_evaluation)
    else:
        evaluate_artifact(a.root.resolve())
