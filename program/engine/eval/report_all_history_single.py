"""Report frozen fresh-model comparisons and deterministic region illustrations."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset_benchmark import normalized_corner_errors
from screen_corner_single import SingleModelPredictor
from training.screen_corner_ablation import read_rows, write_json
from training.train_all_history_single import collection_key

REPO = Path(__file__).resolve().parents[3]


def hits(record, stage):
    prediction = record.get(stage + '_quad')
    if prediction is None or not np.isfinite(prediction).all():
        return np.zeros(4, bool)
    return normalized_corner_errors(record['manual_quad'], prediction) <= .01


def diagnostic(root, report):
    records = {label: json.loads((root / f'reports/{label}_predictions.json').read_text()) for label in report['models']}
    old = {r['image_sha256']: r for r in records['current_runtime_round027']}
    new = records['new_fresh_all_history']; point_report = {}; failures = []
    for role in ['train', 'validation', 'holdout', 'quarantined_train']:
        current = [r for r in new if r['new_dataset_role'] == role]
        point_report[role] = {}
        for stage in ['coarse', 'final']:
            before = np.asarray([hits(old[r['image_sha256']], stage) for r in current])
            after = np.asarray([hits(r, stage) for r in current])
            point_report[role][stage] = {'gained': int((~before & after).sum()), 'lost': int((before & ~after).sum()),
                'before_by_corner': dict(zip(['TL', 'TR', 'BR', 'BL'], before.sum(0).tolist())),
                'after_by_corner': dict(zip(['TL', 'TR', 'BR', 'BL'], after.sum(0).tolist())),
                'corners_per_position': len(current)}
    for r in new:
        if r['new_dataset_role'] not in ['validation', 'holdout']:
            continue
        for stage in ['coarse', 'final']:
            correct = hits(r, stage)
            if not correct.all():
                failures.append({'image_path': r['image_path'], 'role': r['new_dataset_role'], 'stage': stage,
                    'missed_corners': [name for name, hit in zip(['TL', 'TR', 'BR', 'BL'], correct) if not hit]})
    write_json(root / 'reports/point_changes.json', point_report)
    write_json(root / 'reports/remaining_evaluation_failures.json', failures)
    # Determine the pages BEFORE examining predictions: lexicographically
    # first two inherited development paths in each meeting/collection.
    groups = defaultdict(list)
    for r in new:
        if r['new_dataset_role'] == 'validation':
            groups[collection_key(r)].append(r)
    chosen = [(name, r) for name, group in sorted(groups.items()) for r in sorted(group, key=lambda x: x['image_path'])[:2]]
    write_json(root / 'reports/visual_sample_policy.json', {'policy': 'first two lexicographic development paths per meeting; no selection by prediction success',
        'pages': [{'collection': name, 'image_path': row['image_path']} for name, row in chosen]})
    predictor = SingleModelPredictor(Path(report['frozen']['artifact'])); width, height = 400, 330
    image_files = []
    for start in range(0, len(chosen), 6):
        group = chosen[start:start+6]
        canvas = np.full((len(group) * height, width * 4, 3), 245, np.uint8)
        for i, (name, row) in enumerate(group):
            image = cv2.imread(row['image_path']); source = image.copy()
            rgb = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())[None].float() / 255.
            with torch.inference_mode():
                _, _, boundary, _ = predictor.model.global_prediction(rgb)
            mask = cv2.resize(boundary[0, 0].sigmoid().numpy(), (image.shape[1], image.shape[0]))
            colored = image.copy(); colored[:, :, 2] = np.clip(colored[:, :, 2].astype(float) + mask * 90., 0, 255).astype(np.uint8)
            tiles = [colored, source.copy(), source.copy(), source.copy()]
            descriptions = [name[:35] + ' region', 'current final', 'new coarse', 'new final']
            predictions = [None, old[row['image_sha256']].get('final_quad'), row.get('coarse_quad'), row.get('final_quad')]
            for column, tile in enumerate(tiles):
                cv2.polylines(tile, [np.round(row['manual_quad']).astype(np.int32)], True, (0, 255, 0), 7)
                if predictions[column] is not None:
                    cv2.polylines(tile, [np.round(predictions[column]).astype(np.int32)], True, (0, 140, 255), 7)
                tile = cv2.resize(tile, (width, 300), interpolation=cv2.INTER_AREA)
                x, y = column * width, i * height; canvas[y:y+300, x:x+width] = tile
                cv2.putText(canvas, descriptions[column], (x+4, y+320), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1)
        path = root / f'artifacts/all_history_comparison_part{start//6+1}.jpg'
        cv2.imwrite(str(path), canvas); image_files.append(path)
    return point_report, image_files


def render(root):
    torch.set_num_threads(4)
    comparison = json.loads((root / 'reports/comparison.json').read_text())
    cfg = json.loads((root / 'manifest.json').read_text())
    audit = json.loads((root / 'reports/final_verification.json').read_text())
    changes, images = diagnostic(root, comparison)
    current, new = [comparison['models'][key] for key in ['current_runtime_round027', 'new_fresh_all_history']]
    frozen = comparison['frozen']
    def value(result):
        return f'{result["hit_corners"]}/{result["corners"]}（{result["point_le_0_01_ratio"]:.2%}）'
    def row(name, before, after):
        return f'| {name} | {value(before)} | {value(after)} | {after["hit_corners"]-before["hit_corners"]:+d} |'
    def table(before, after):
        return '\n'.join(['| 阶段 | 正式 round_027 | 新独立模型 | 命中变化 |', '| --- | ---: | ---: | ---: |'] +
            [row(stage, before[stage], after[stage]) for stage in ['coarse', 'final']])
    before, after = current['roles']['validation']['final'], new['roles']['validation']['final']
    delta = after['hit_corners'] - before['hit_corners']
    direction = '提高' if delta > 0 else '下降' if delta < 0 else '持平'
    body = f'''# 2026-10-22 全历史数据独立单体模型 round_030

已按用户要求重新训练区域识别 → 粗定位 → 精细矫正单体模型，并与正式 round_027 在同图、同标签、同用途上重新推理比较。新候选 `{frozen['model_id']}` 在 432 页开发的最终角点命中为 {value(after)}，正式模型为 {value(before)}，{direction} {abs(delta)} 点。新权重未替换正式 runtime，未打包。

## 这次训练的边界

只加载通用 ImageNet MobileNetV3-small 权重。没有加载任何 ScreenPDF 任务 checkpoint 初始化，没有使用当前模型教师、蒸馏或旧正确点保护；区域、粗定位、局部头均随机初始化。视觉骨干在区域和粗定位阶段参与训练，精修阶段冻结骨干和全局头。独立验收确认骨干 {audit['artifact']['encoder_parameter_tensors_changed']} 个参数张量、全局 {audit['artifact']['global_parameter_tensors_changed']} 个参数张量、局部 {audit['artifact']['local_parameter_tensors_changed']} 个参数张量相对初始改变。

该实验从通用视觉特征开始重新学习屏幕任务，不是从随机像素特征开始，也不是沿用当前业务模型继续微调。正式 round_027 在训练前已复制固定并记录 SHA256，仅在新模型冻结后用于比较。

## 历史资料覆盖和隔离

本轮汇总冻结 conference_v1 的 1808 页有效人工框及当前注册的各新增场次修订，覆盖 62 个来源项目、2275 张有效人工标注；所有 62 个来源均有训练页。仅使用明确人工四角，另 1 页旧训练快照内无效框排除，既有异常 /冲突页不重新引入。旧冻结集合与源图片 /标签 /用途不覆盖。近期旧快照有 121 张照片不在当前有效人工资料集合中，包含 115 张缺少显式人工框及异常 /冲突页，逐项原因见 `reports/legacy_snapshot_exclusions.json`；不能只凭历史条目存在 manual_quad 字段就当成人工真值。

| 本轮用途 | 唯一照片 |
| --- | ---: |
| 训练 | 1461 |
| 开发选优 | 432 |
| 已查看回归 | 216 |
| 与评估内容近重复而隔离的训练页 | 166 |
| 人工资料总计 | 2275 |

继承历史用途；已知开发和回归页不能转入训练。当前注册场次修订提供新增训练页，包括早先因对照未胜而未被选中采样的刘奇 /孙泉明等。快照外的较早资料按其已有人工 page-id 拆分补齐；AWE /扬州的旧 raw 副本通过显式集合与 page-id 映射继承用途，当前照片与人工点仍以冻结来源为准。

全局照片按字节校验和去重，项目内经矫正的 PPT 内容采用 dHash /pHash 阈值 8 的传递分组，将与开发或回归近重复的训练照片隔离，保留评估用途。实际训练与评估之间的路径、字节和内容分组交集均为 0，开发与回归之间内容分组交集也为 0。跨项目重压缩 /裁切近重复尚未穷尽识别，历史开发 /回归早已被查看，不能称独立盲测。

## 模型与训练流程

1. 区域阶段：由人工四边形生成完整 PPT 区域与边界监督，包含页内空白、黑底及遮挡部分，不沿人头、椅背或倒计时收边。
2. 粗定位阶段：区域概率调制全局特征，保留未掩掉的全局上下文、区域 /边界概率及位置编码，交给可学习的四角热图头。通过概率分布得到四角，保留透视，不强制 16:9，也不要求硬阈值掩码有四条完整边。
3. 精修阶段：由本模型预测的粗框，在原图和全局特征中图内取四角局部样本，再训练局部角点矫正。训练裁剪不由人工四角居中，缓存不属于推理依赖。

最终导出一个 TorchScript 文件，包括缩放 /归一化、骨干、区域、粗定位、局部采样和精修。程序不需外部权重、torchvision 或网络。

轮数在开始前固定：seed=7、batch=32，12 轮区域监督初始化、80 轮区域 +粗定位共同学习、48 轮精修。每张训练照片每轮至少使用一次；小来源统一补至采样下限，最多 4 倍，共 2222 条 /轮，没有最近场次专属倍率。全局头学习率 8e-4、骨干 8e-5，粗阶段余弦降到 20%。精修最初学习率 3e-4；首轮 MPS 反向出现非有限值且无有效精修 checkpoint 保存，随后降低为 1e-4、梯度范数裁剪 1.0，并把局部训练转至 CPU 后从相同初始局部头重新开始。该实现修正写入 protocol_amendments，完整失败日志保留；未重训或改选全局 checkpoint、数据、48 轮精修预算、唯一选优指标和回归隔离均不变。没有图像增强、测试后调参或额外追分轮次。

区域监督用完整区域 BCE /Dice 和边界 BCE；粗定位加入软热图交叉熵、坐标 smooth-L1、角点命中 surrogate；局部使用热图 MSE、相对坐标 smooth-L1 和命中 surrogate。训练 loss 仅用于优化，所有 checkpoint 均只按 pooled 开发角点 `point_le_0_01_ratio` 选取，平局保留较早 epoch。区域初始化末尾作为固定初始化，不按 IoU 或其他指标选模型。粗定位保留第 {frozen['coarse_epoch']} 轮，精修保留第 {frozen['local_epoch']} 轮。

## 同集比较

唯一准确率是原图像素角点误差 /人工框轴对齐包围盒对角线 <=1% 的角点占比，失败页仍计四个未命中。每个模型重新运行完整单文件推理，未拼接历史分数。先冻结新候选和摘要，再评回归。

冻结 1808 页诊断中有 1 页（杭州智能交通 `IMG_7035`）的人工框下边超出照片 0.77 px，数据集修订在训练前已将其排除。对照时仍按冻结官方人工框为该页补算两个模型，因此 1808 页、7232 个角点、阈值与汇总口径都与历轮一致；补算只影响这条含训练资料的诊断，不参与训练、选优或晋升判断，记录见 `reports/comparison.json` 的 `frozen_subset_not_in_dataset_revision` 与 `manifest.json` 的 `comparison_amendments`。

432 页开发：

{table(current['roles']['validation'], new['roles']['validation'])}

216 页已查看回归：

{table(current['roles']['holdout'], new['roles']['holdout'])}

原有 428 页开发中的当前有效人工子集（26 张不在当前有效人工集合，不从分母隐去；不等同完整 428 页重测）：

{table(current['previous428_development'], new['previous428_development'])}

1461 页训练诊断：

{table(current['roles']['train'], new['roles']['train'])}

166 页近重复隔离训练诊断：

{table(current['roles']['quarantined_train'], new['roles']['quarantined_train'])}

冻结 1808 页会议集合诊断（含训练页，非新盲测）：

{table(current['frozen_conference1808_diagnostic'], new['frozen_conference1808_diagnostic'])}

全部 2275 页诊断（含训练页）：

{table(current['all_manual_diagnostic'], new['all_manual_diagnostic'])}

开发最终角点变化：新增命中 {changes['validation']['final']['gained']}，原命中回退 {changes['validation']['final']['lost']}；已查看回归新增 {changes['holdout']['final']['gained']}、回退 {changes['holdout']['final']['lost']}。逐角位置和残留失败见 `reports/point_changes.json`、`reports/remaining_evaluation_failures.json`。

## 按会议 /集合看开发结果

| 会议 /集合 | 正式最终命中 | 新模型最终命中 | 变化 |
| --- | ---: | ---: | ---: |
'''
    for key in sorted(new['collections']):
        a, b = current['collections'][key]['roles']['validation']['final'], new['collections'][key]['roles']['validation']['final']
        body += row(key, a, b) + '\n'
    body += '''
上述会议聚合与来源项目明细均保存于机器报告；同图只计一次。较早 frozen conference_v1 的会议用途与本轮新增训练覆盖不同，两个模型先前暴露也不同，所以这是整体产品比较，不能把全部变化归因于区域分支。

## 可视化与验收

每个会议选路径排序最前的两张开发页，抽样与预测成败无关；包含改善与失败页。四列依次为新区域概率、正式最终框、新粗框、新最终框。绿为人工，橙为模型，区域红色叠层只用于展示。

'''
    for path in images:
        body += f'![{path.stem}](../../{path.relative_to(REPO)})\n\n'
    body += f'''源图和全部历史输入摘要核查通过；训练、开发、回归隔离核查通过。23 项模型 /指标 /导入相关测试通过。新模型在临时目录仅保留一个权重，禁止读取外部 checkpoint 和旧链路后，三张不同来源实际检测均返回 runtime_single；图内局部采样复算与完整图一致。

单文件候选：[screen_corner_single.pt](../../{Path(frozen['artifact']).relative_to(REPO)})。SHA256：`{frozen['sha256']}`。正式对照 SHA256：`{cfg['comparison_runtime_sha256']}`。

保留单 seed、历史开发 /回归已查看、近重复识别范围、固定预算不等于收敛证明、数据和结构共同改变等限制。整体分数不能掩盖会议或底角局部回退。耗时仅作观测，没有参与选择。未自动晋升、未重新打包，正式 runtime 仍 round_027。

[运行约定](../../{root.relative_to(REPO)}/README.md) · [机器比较](../../{root.relative_to(REPO)}/reports/comparison.json) · [最终验收](../../{root.relative_to(REPO)}/reports/final_verification.json) · [seed 复验](2026-10-22-all-history-seed-replication.md)。复现时新建 round 保留本轮，依次运行 `build_all_history_dataset.py`、`train_all_history_single.py configure /train /compare`、`verify_all_history_single.py`、`report_all_history_single.py`，参数入口见各脚本 `--help`。
'''
    path = REPO / 'research/experiments/2026-10-22-all-history-region-coarse-local-training.md'; path.write_text(body)
    short = f'''round_030 已根据全部历史可用人工资料独立训练区域识别 → 粗定位 → 精修单体模型。62 个来源、2275 页，1461 训练 /432 开发 /216 已查看回归，166 张近评估重复训练页隔离。仅通用 ImageNet 初始化，骨干和三个任务阶段重新训练，未继承 ScreenPDF 业务权重或教师。候选 `{frozen['model_id']}`；开发最终正式 {value(before)} → 新模型 {value(after)}（{delta:+d} 点），已查看回归正式 {value(current['roles']['holdout']['final'])} → 新模型 {value(new['roles']['holdout']['final'])}。南京开发 +33、InfoComm +7，openatom -18、福州高速 -12 等局部回退保留。12 /80 /48 轮固定预算，按 pooled 开发唯一角点指标冻结，先冻结再回归；含训练资料的 1808 页诊断（1 页数据集修订排除页按冻结人工框补算）/2275 页诊断与逐会议变化另报。seed=11 复验（round_031）开发 1522、已查看回归 770，粗定位 1496/1497 复现，final 增益带约 ±0.8 个百分点种子波动；精修净贡献仅 +12/+25（成熟模型 +191），openatom /福州高速 /smart-road /metaverse /商场测试集回退两 seed 复现，故建议先修精修阶段再谈晋升。23 项测试和隔离单文件实际检测通过；两 seed 复验后，固定预算、历史评估已查看等限制仍保留。未晋升 /打包，正式 runtime 仍 round_027。'''
    for rel, prefix, prepend in [('docs/status/current-status.md', '../../', True), ('training/README.md', '../', False), ('AGENTS.md', '', False), ('README.md', '', False)]:
        p = REPO / rel; text = p.read_text(); start, end = '<!-- all-history-round030:start -->', '<!-- all-history-round030:end -->'
        block = start + '\n' + ('## 2026-10-22 全历史独立单体模型 round_030\n\n' if prepend else '') + short + f'\n\n[完整报告]({prefix}{path.relative_to(REPO)}) · [seed 复验与综合指标]({prefix}research/experiments/2026-10-22-all-history-seed-replication.md)。\n' + end
        if start in text:
            a, b = text.index(start), text.index(end) + len(end); text = text[:a] + block + text[b:]
        elif prepend:
            title, rest = text.split('\n', 1); text = title + '\n\n' + block + '\n' + rest
        else:
            text = text.rstrip() + '\n\n' + block + '\n'
        p.write_text(text)
    print(path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    render(parser.parse_args().root.resolve())
