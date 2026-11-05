"""One frozen, full-source benchmark, aggregated by conference rather than talk.

Weights are pinned before reading results. This benchmark never selects or trains
models and never promotes runtime weights. All human pages are measured; training
and historically exposed pages are always separated from current development.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from dataset_benchmark import METRIC_CONTRACT, summarize_point_hits
from screen_corner_single import SingleModelPredictor
from training.build_conference_collections import digest, read_rows, write_rows, write_json

ROLES = {'train':'当前快照训练页（拟合诊断）', 'validation':'当前开发页',
         'observed_regression':'已查看回归页', 'outside_current_snapshot':'不在当前快照（历史暴露未核实）'}


def summarize(rows, predictions):
    return summarize_point_hits([r['manual_quad'] for r in rows],
                                [predictions[r['image_path']].get('final_quad') for r in rows])


def predict_one(model, row):
    start = time.perf_counter()
    result = dict(image_path=row['image_path'], manual_quad=row['manual_quad'],
                  collection_id=row['collection_id'], project_id=row['project_id'], current_role=row['current_role'])
    try:
        pixels = cv2.imread(row['image_path'])
        if pixels is None:
            raise ValueError('unreadable image')
        q = np.asarray(model.predict(pixels)['final_quad'], dtype=float)
        if q.shape != (4, 2) or not np.isfinite(q).all():
            raise ValueError('invalid model quadrilateral')
        result['final_quad'] = q.tolist()
        truth = np.asarray(row['manual_quad'])
        error = np.linalg.norm(q-truth, axis=1) / np.linalg.norm(truth.max(0)-truth.min(0))
        result['corner_hits'] = (error <= .01).tolist()
    except Exception as exc:
        result.update(final_quad=None, corner_hits=[False]*4, error=f'{type(exc).__name__}: {exc}')
    result['elapsed_ms_observation'] = (time.perf_counter()-start)*1000
    return result


def run(collections, output, models):
    if output.exists():
        raise FileExistsError('Use a new benchmark directory')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    index = json.loads((collections/'index.json').read_text())
    rows = read_rows(collections/'pages.jsonl')
    manifest = dict(metric_contract=METRIC_CONTRACT, collections=str(collections),
                    collection_index_sha256=digest(collections/'index.json'),
                    input_pages_sha256=digest(collections/'pages.jsonl'), pages=len(rows),
                    models={name:dict(path=str(path.resolve()),sha256=digest(path)) for name,path in models.items()},
                    policy='evaluate every usable explicit human annotation once per model; no checkpoint or threshold selection; no training; no promotion',
                    failure_policy='unreadable image, exception or invalid prediction counts as four missed corners',
                    role_policy=ROLES, blind_test_claim=False, torch_version=torch.__version__,
                    changed_current_labels=index['changed_current_labels'], status='frozen_before_inference')
    write_json(output/'manifest.json', manifest)
    (output/'data').mkdir()
    shutil.copyfile(collections/'pages.jsonl', output/'data/pages.jsonl')
    shutil.copyfile(collections/'index.json', output/'data/collection_index.json')
    results = {}
    for name, path in models.items():
        assert digest(path) == manifest['models'][name]['sha256']
        predictor = SingleModelPredictor(path)
        predictions = []
        with (output/f'{name}_predictions.jsonl').open('w') as handle:
            for i, row in enumerate(rows):
                record = predict_one(predictor, row)
                predictions.append(record)
                handle.write(json.dumps(record,ensure_ascii=False)+'\n')
                if (i+1) % 100 == 0 or i == 0 or i+1 == len(rows):
                    handle.flush()
                    print(f'{name}: {i+1}/{len(rows)}',flush=True)
        lookup = {r['image_path']:r for r in predictions}
        result = dict(all_manual_pages_diagnostic=summarize(rows, lookup),
                      failed_pages=sum(r['final_quad'] is None for r in predictions), roles={}, collections={})
        for role in ROLES:
            sub = [r for r in rows if r['current_role']==role]
            result['roles'][role] = summarize(sub,lookup) if sub else None
        for group in index['collections']:
            members = [r for r in rows if r['collection_id']==group['id']]
            entry = dict(all_manual_pages_diagnostic=summarize(members,lookup) if members else None,
                         roles={}, projects={})
            for role in ROLES:
                sub = [r for r in members if r['current_role']==role]
                entry['roles'][role] = summarize(sub,lookup) if sub else None
            for project in group['member_projects']:
                sub=[r for r in members if r['project_id']==project['project_id']]
                entry['projects'][project['project_id']] = summarize(sub,lookup) if sub else None
            result['collections'][group['id']] = entry
        results[name] = result
        write_json(output/f'{name}_summary.json',result)
        del predictor
    for path,sha in index['source_project_sha256'].items():
        assert digest(path)==sha, f'Original project changed while benchmarking: {path}'
    for path,sha in index['active_snapshot_sha256'].items():
        assert digest(path)==sha, f'Historical data snapshot changed: {path}'
    assert digest(collections/'pages.jsonl') == manifest['input_pages_sha256']
    manifest['status']='completed';manifest['source_projects_and_training_snapshots_unchanged']=True
    manifest['results']={k:v['roles'] for k,v in results.items()}
    write_json(output/'manifest.json',manifest)
    write_json(output/'evaluation.json',dict(metric_contract=METRIC_CONTRACT,models=manifest['models'],results=results,model_selection_performed=False))
    report(collections,output,index,manifest,results)
    print('COMPLETED',str(output),flush=True)


def fmt(score):
    return '—' if score is None else f'{score["point_le_0_01_ratio"]*100:.2f}%（{score["hit_corners"]}/{score["corners"]}）'


def report(collections,output,index,manifest,results):
    names=list(results)
    desktop_counts = {r['collection_id']: r['pages'] for r in json.loads((collections/'desktop-projects.json').read_text())['projects']} if (collections/'desktop-projects.json').exists() else {}
    lines=['# 按会议集合的全量基准测试','',
           f'全量只读扫描 {index["source_projects"]} 个源项目、{index["source_pages"]} 页；{index["usable_manual_pages"]} 页明确人工标注进入实测，{index["excluded_pages"]} 页排除。', '',
           '本次重新运行全部可测页面，没有拼接旧分数。主口径为原图像素误差 / 人工框包围盒对角线 <=1% 的角点占比；失败四点全记未命中。模型在测试前冻结，不训练、不改阈值、不根据本次结果重新选择候选。', '',
           f'对比模型：{", ".join(names)}。这里的“当前训练页”继承冻结集合记录的用途；它描述数据快照成员关系，不等同于每个模型实际训练过所有标记为训练页的资料。历史训练暴露并未由当前快照穷尽，所有结果均不宣称独立盲测。', '',
           f'真值来自当前源项目显式 manualQuad；相对训练快照有 {index["changed_current_labels"]} 页点位不同，详见 annotation_changes.json；本报告只在同一份新冻结真值上比较两模型，不能与历史报告分母或分数直接拼接。', '',
           '## 分用途结果','', '| 用途 | '+' | '.join(names)+' |','| --- | '+' | '.join(['---:']*len(names))+' |']
    for role,cn in ROLES.items():lines.append('| '+cn+' | '+' | '.join(fmt(results[n]['roles'][role]) for n in names)+' |')
    lines += ['', '全集混合值仅用于全量覆盖/拟合诊断，包含训练页，不能当作泛化准确率。各集合依次展示，演讲明细仅作下钻。', '',
              '## 会议与场次总览','', '| 集合 | 项目 | 可测页 | 开发页 | 已查看回归页 | 其他快照外页 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for g in index['collections']:
        roles=g['roles'];lines.append(f'| [{g["name"]}]({g["id"]}/README.md) | {g["projects"]} | {g["usable_manual_pages"]} | {roles.get("validation",0)} | {roles.get("observed_regression",0)} | {roles.get("outside_current_snapshot",0)} |')
    flat=[]
    for role,cn in ROLES.items():
        lines += ['',f'## {cn}：按会议集合比较','', '| 集合 | '+' | '.join(names)+' |','| --- | '+' | '.join(['---:']*len(names))+' |']
        for g in index['collections']:
            scores=[results[n]['collections'][g['id']]['roles'][role] for n in names]
            if not any(scores):continue
            lines.append('| '+g['name']+' | '+' | '.join(fmt(s) for s in scores)+' |')
            for name,score in zip(names,scores):
                if score:flat.append(dict(collection_id=g['id'],collection_name=g['name'],role=role,model=name,**score))
    lines += ['', '## 全部人工页：覆盖与拟合诊断（包含训练页）','', '| 集合 | '+' | '.join(names)+' |','| --- | '+' | '.join(['---:']*len(names))+' |']
    for g in index['collections']:lines.append('| '+g['name']+' | '+' | '.join(fmt(results[n]['collections'][g['id']]['all_manual_pages_diagnostic']) for n in names)+' |')
    lines += ['', '## 复现与边界','',
              f'基准目录：`{output}`。逐页预测、失败原因、模型摘要、输入快照与结构化评价均保留。',
           '未标注页、无效人工框及缺失图片逐条列在 excluded_pages.json。未标注页不是模型检测失败，不进入人工准确率分母；已纳入测试但推理失败则留在分母。',
              '[人工框冲突与异常校正清单](conflict-review/README.md) 包含同图异框的来源、坐标、叠框、角点放大图及可编辑副本。全部冲突版本已从此次基准排除；修正前不自动挑选一份。',
              '原始项目、训练快照、拆分和正式模型未改写。今后新增会议先更新集合配置并生成新 revision，再做可复现的相同页面对比。', '']
    text='\n'.join(lines)
    (collections/'BASELINE.md').write_text(text)
    (output/'README.md').write_text(text.replace('](', ']('+str(collections)+'/'))
    with (collections/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=['collection_id','collection_name','role','model','pages','corners','hit_corners','point_le_0_01_ratio'])
        writer.writeheader();writer.writerows(flat)
    write_json(collections/'benchmark.json',dict(benchmark_directory=str(output),manifest_sha256=digest(output/'manifest.json'),results=results))
    readme=['# 会议与场次项目集合','',
            f'{len(index["collections"])} 个大集合，{index["source_projects"]} 个源项目，{index["usable_manual_pages"]} 张可测人工标注。', '',
            '[统一基准对比](BASELINE.md) · [表格数据](comparison.csv) · [人工框校正](conflict-review/README.md) · [屏幕场景下钻](SCENES.md) · [机器索引](index.json)', '',
            '日常按会议集合查看，原演讲作为成员保留。中国智慧道路照明大会依据用户的场次分类，分为 3 月 19 日、20 日；专项测试单独标明，不伪装成会议。', '',
            '| 集合 | 打开合并项目 | 源项目 | 可测人工标注 | 源页面 | 集合页面 |', '| --- | --- | ---: | ---: | ---: | ---: |']
    readme += [f'| [{g["name"]}]({g["id"]}/README.md) | [ScreenPDF 项目](<desktop-projects/{g["name"]}/screen-pdf-project.json>) | {g["projects"]} | {g["usable_manual_pages"]} | {g["source_pages"]} | {desktop_counts.get(g["id"], g["source_pages"])} |' for g in index['collections']]
    readme += ['', '原图与源项目不移动、不覆盖。聚合 pages.jsonl 每图仅计一次；train/validation/observed_regression 文件继承既有用途，快照外页面保留为 outside_current_snapshot，不自动进入训练。',
               f'合并项目保留未标注页及既有候选；冲突和无效框页面作为异常排除，详情保留在 conflict-review/。当前集合 {sum(desktop_counts.values()) if desktop_counts else "以项目清单为准"} 页，源资料 {index["source_pages"]} 页；原始资料和冻结基准不改写。图片是独立副本，来源映射保存在各项目的 source-provenance.json。',
               '统一黑底等横向场景索引保留为诊断视图，同一图不会因有多个标签被重复计入会议总数。', '']
    (collections/'README.md').write_text('\n'.join(readme))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collections',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',action='append',required=True,help='label=/absolute/path/to/model.pt')
    a=p.parse_args();models={name:Path(path) for name,path in (value.split('=',1) for value in a.model)}
    run(a.collections.resolve(),a.output.resolve(),models)
