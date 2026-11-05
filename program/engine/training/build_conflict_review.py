"""Prepare source-linked manual correction packets; never choose or edit a label."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from training.build_conference_collections import digest, write_json
from training.materialize_conference_projects import independent_copy

COLORS = [(0, 200, 255), (255, 80, 220), (80, 230, 60), (255, 170, 0)]


def draw(pixels, variants, origin=(0,0), scale=1):
    canvas = pixels.copy()
    for i, row in enumerate(variants):
        q = ((np.asarray(row['manual_quad'])-origin)*scale).round().astype(np.int32)
        color = COLORS[i % len(COLORS)]
        cv2.polylines(canvas, [q], True, color, 2, cv2.LINE_AA)
        for k, (x,y) in enumerate(q):
            cv2.circle(canvas, (x,y), 5, color, 2)
            cv2.putText(canvas, f'V{i+1} {("TL","TR","BR","BL")[k]}', (x+8,y-8+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
    return canvas


def prepare(collections):
    output=collections/'conflict-review'
    if output.exists():raise FileExistsError('Preserve pending manual corrections; use a new revision')
    output.mkdir()
    conflicts=json.loads((collections/'duplicate_conflicts.json').read_text())
    invalid=[r['variant'] for r in json.loads((collections/'excluded_pages.json').read_text())
             if r['reason']=='invalid_manual_quad']
    groups=[dict(id=f'C{i+1:03}',kind='重复照片人工框冲突',variants=g['variants'],image_sha256=g['image_sha256'])
            for i,g in enumerate(conflicts)]
    groups += [dict(id=f'I{i+1:03}',kind='人工框越界或无效',variants=[r],image_sha256=r['image_sha256']) for i,r in enumerate(invalid)]
    project_folder=output/'待人工校正'
    project_folder.mkdir()
    pages=[]; provenance=[]; csv_rows=[]
    lines=['# 人工框冲突与异常校正清单','',
           f'发现 **{len(conflicts)} 张重复照片存在人工框冲突，涉及 {sum(len(g["variants"]) for g in conflicts)} 个项目页面版本**；另有 **{len(invalid)} 页人工框无效**。重复照片按文件内容 SHA256 识别，比较规范化 TL/TR/BR/BL 四角，容差 0.001 原图像素；不把改名当新图，也不将不同照片按文件名误合并。', '',
           '[打开校正项目](待人工校正/screen-pdf-project.json) · [坐标清单](corrections.csv) · [全部来源](review.json)', '',
           '校正项目含全部冲突版本及无效框，每页名称对应下方编号。可以在 ScreenPDF 打开、调整并保存；图片为独立副本，原项目不会被覆盖。source-provenance.json 保留原项目、原页面及原始摘要，便于后续核对和回写。', '',
           '本次基准已排除这些页面；没有自动挑选某份框。完成校正后需明确采用的点位、同步冲突来源并生成新版本基准，当前冻结报告保留。', '']
    copied={}
    for group in groups:
        ident=group['id'];variants=group['variants']
        image=Path(variants[0]['image_path']);pixels=cv2.imread(str(image));height,width=pixels.shape[:2]
        small=cv2.resize(pixels,(1400,round(height*1400/width)))
        cv2.imwrite(str(output/f'{ident}-overlay.jpg'),draw(small,variants,scale=1400/width))
        lines += [f'## {ident} · {image.name} · {group["kind"]}', '', f'![{ident} 叠框]({ident}-overlay.jpg)', '',
                  'V1 为黄色，V2 为紫色。点序均为 TL（左上）、TR（右上）、BR（右下）、BL（左下）。', '',
                  '| 版本 | 来源项目 | 页面 | 用途 | 四角坐标（原图像素） |','| --- | --- | --- | --- | --- |']
        if len(variants)>1:
            q=np.asarray([r['manual_quad'] for r in variants]);dist=np.linalg.norm(q-q[0],axis=2).max(0)
            group['max_corner_difference_pixels']=dist.tolist()
            lines += [f'| 差异 | 相对 V1 最大角点距离 | — | — | '+', '.join(f'{k} {d:.2f}px' for k,d in zip(['TL','TR','BR','BL'],dist))+' |']
            panels=[]
            for k,label in enumerate(['TL','TR','BR','BL']):
                center=q[:,k].mean(0);x=max(0,min(width-480,int(center[0])-240));y=max(0,min(height-400,int(center[1])-200))
                panel=draw(pixels[y:y+400,x:x+480],variants,origin=(x,y))
                cv2.putText(panel,label,(12,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                panels.append(panel)
            cv2.imwrite(str(output/f'{ident}-corners.jpg'),np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:])]))
        for i,row in enumerate(variants):
            vid=f'{ident}-V{i+1}';source=Path(row['project_path']);payload=json.loads(source.read_text())
            original=next(p for p in payload['pages'] if p.get('id')==row['page_id'])
            sha=row['image_sha256']
            if sha not in copied:
                rel=Path('images')/(sha+Path(row['image_path']).suffix)
                independent_copy(Path(row['image_path']),project_folder/rel)
                copied[sha]=rel.as_posix()
            item=copy.deepcopy(original);item.update(id=vid,name=f'{vid} / {row["project_name"]} / {Path(row["image_path"]).name}',
                       path=copied[sha],thumbPath=None,previewPath=None)
            pages.append(item)
            record=dict(review_id=vid,issue=group['kind'],image_sha256=sha,source_project=row['project_path'],source_page_id=row['page_id'],
                        source_image=row['image_path'],source_project_sha256=digest(source),original_manual_quad=row['manual_quad'])
            provenance.append(record)
            csv_rows.append(dict(review_id=vid,issue=group['kind'],source_project=row['project_path'],source_page_id=row['page_id'],image_sha256=sha,
                                 original_quad=json.dumps(row['manual_quad']),correction_status='待人工校正',corrected_quad='',notes=''))
            coords='; '.join(f'{k}=({x:.3f}, {y:.3f})' for k,(x,y) in zip(['TL','TR','BR','BL'],row['manual_quad']))
            lines.append(f'| {vid} | [{row["project_name"]}](<{row["project_path"]}>) | {row["page_id"]} | {row["current_role"]} | {coords} |')
        if len(variants)>1:lines += ['',f'![{ident} 四角原尺寸放大]({ident}-corners.jpg)']
        else:
            lines += ['',f'图片尺寸：{width}×{height}。人工点超出图片或几何无效，保留原值，未裁切到边缘。']
        lines += ['']
    write_json(project_folder/'screen-pdf-project.json',dict(version=2,dataStructureVersion=2,sourceDir='.',projectPath='screen-pdf-project.json',
               selectedPageId=pages[0]['id'] if pages else None,pages=pages))
    write_json(project_folder/'source-provenance.json',provenance)
    write_json(output/'review.json',dict(conflicting_photos=len(conflicts),conflicting_versions=sum(len(g['variants']) for g in conflicts),invalid_pages=len(invalid),groups=groups))
    with (output/'corrections.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=['review_id','issue','source_project','source_page_id','image_sha256','original_quad','correction_status','corrected_quad','notes']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(csv_rows)
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(conflicts=len(conflicts),invalid=len(invalid),review_pages=len(pages)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collections',required=True,type=Path)
    prepare(p.parse_args().collections.resolve())
