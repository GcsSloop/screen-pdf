"""Source-backed overlapping scene views of a frozen conference benchmark."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from training.build_conference_collections import digest, read_rows, write_json
from dataset_benchmark import summarize_point_hits

LABELS={'strong_perspective':'强透视','white_ppt':'白底 PPT','colorful_ppt':'彩色 PPT',
        'black_background':'黑色背景','low_contrast_edge':'低对比边缘','complex_background':'复杂背景',
        'near_color_background':'近色背景','black_border':'外围黑边','corner_occlusion':'角点遮挡',
        'ui_overlay':'界面叠层','led_screen':'LED 屏','wide_led_screen_partial_ppt_target':'超宽屏内局部 PPT',
        'left_fixed_information_panel_excluded':'排除左侧固定信息区','led_capture_stripes':'拍摄条纹',
        'dark_physical_frame':'深色实体边框','foreground_people':'前景人物','blue_side_panels':'蓝色侧屏',
        'mixed_light_dark_backgrounds':'亮暗混合页面','complex_multicolor_backgrounds':'复杂花色背景',
        'bottom_edge_interference':'下边缘干扰','floor_reflection':'地面反射','unclassified':'尚无既有场景标签'}


def render(collections, benchmark, scene_root):
    index=json.loads((collections/'index.json').read_text());rows=read_rows(collections/'pages.jsonl')
    names=list(json.loads((benchmark/'evaluation.json').read_text())['results'])
    predictions={name:{r['image_path']:r['final_quad'] for r in read_rows(benchmark/f'{name}_predictions.jsonl')} for name in names}
    manual_maps=sorted(scene_root.glob('*project_scene_map.manual.json'))
    sources={str(p.resolve()):digest(p) for p in manual_maps}
    groups=defaultdict(list)
    for row in rows:
        tags=set(row['scene_tags'])
        provenance=['frozen current training snapshot'] if tags else []
        for p in manual_maps:
            # Only existing manually curated project tags, never inferred labels.
            if 'china_smart_road_lighting' in p.name and row['collection_id'] not in ['20260319-smart-road-lighting','20260320-smart-road-lighting']:continue
            for item in json.loads(p.read_text()).get('projects',[]):
                expected=item['project_name'];actual=row['project_name']
                if actual==expected or actual.removesuffix('PPT')==expected or actual.endswith(' '+expected):
                    tags.update(item.get('scene_tags',[]));provenance.append(str(p.resolve()))
        if not tags:tags={'unclassified'}
        for tag in tags:groups[(row['collection_id'],tag)].append(row)
    entries=[]
    lines=['# 会议集合内的屏幕场景下钻','',
           '沿用训练快照的既有场景标签及原人工维护场景表；没有标签的资料单列“尚无既有场景标签”，不凭会议主题补造外观标签。标签多为项目级可能因素，不能当作逐页 / 逐角点可见性标注。', '',
           '同图可同时属于黑底、遮挡、条纹等多种视图，以下行不能相加作为会议总数。结果复用本次完整实测逐页预测，模型、真值、阈值均未改变。', '',
           '[返回会议集合](README.md) · [统一基准](BASELINE.md) · [机器视图](scene-views.json)', '']
    roles={'validation':'开发','observed_regression':'已查看回归','outside_current_snapshot':'快照外','train':'训练诊断'}
    for collection in index['collections']:
        lines += ['## '+collection['name'],'',
                  '| 场景 | 页数 | 用途 | '+' | '.join(names)+' |','| --- | ---: | --- | '+' | '.join(['---:']*len(names))+' |']
        for (cid,tag),members in sorted(groups.items()):
            if cid!=collection['id']:continue
            entry=dict(collection_id=cid,scene_tag=tag,pages=len(members),projects=sorted({r['project_name'] for r in members}),results={})
            for role,cn in roles.items():
                sub=[r for r in members if r['current_role']==role]
                if not sub:continue
                scores={n:summarize_point_hits([r['manual_quad'] for r in sub],[predictions[n][r['image_path']] for r in sub]) for n in names}
                entry['results'][role]=scores
                formatted=[f'{s["point_le_0_01_ratio"]*100:.2f}%（{s["hit_corners"]}/{s["corners"]}）' for s in scores.values()]
                lines.append(f'| {LABELS.get(tag,tag)} | {len(sub)} | {cn} | '+' | '.join(formatted)+' |')
            entries.append(entry)
        lines += ['', '成员演讲见本场次 [项目清单]('+collection['id']+'/README.md)。','']
    write_json(collections/'scene-views.json',dict(overlapping=True,annotation_policy='existing snapshot tags and manual project scene maps; no new inferred labels',
               manual_scene_map_sha256=sources,benchmark_directory=str(benchmark.resolve()),entries=entries))
    (collections/'SCENES.md').write_text('\n'.join(lines))
