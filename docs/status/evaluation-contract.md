# 单角点命中率评估契约

当前版本：`screen_point_hit_pixel_v3`，2026-10-17 按用户最新要求调整。历史 v2 的像素坐标修复仍有效，准确率目标缩减为一项。

## 唯一准确率

`point_le_0_01_ratio = hit_corners / corners`。

人工点和预测点统一到原图像素坐标，按 TL、TR、BR、BL 对应。逐点欧氏距离除以人工框轴对齐包围盒对角线；归一化误差 `<=0.01` 的角点记为命中。非正方形图像必须分别恢复原始宽高，不能在 0～1 坐标直接计算欧氏误差。

每页有四个人工角点。未读到图片、未检测出框或预测无效时，这页计四次未命中，不能从分母中删除。摘要只保留准确率和核对所需的页数、角点数、命中数。

## 唯一选优规则

训练 checkpoint 和对照实验只最大化开发验证集的 `point_le_0_01_ratio`；完全相同则保留较早 checkpoint / 预先列在前面的条件。没有四点全中率、平均误差、最差点或耗时平局条件。

合并数据集按总角点数加权。各场景仍分别报告同一个指标，不以项目宏平均替代主指标。不自动设立新的单点 80% 目标；旧四点全中率和平均误差门槛取消。耗时、模型大小单独观测。

历史 `quad_geometry_metrics`、`summarize_geometry_metric_rows` 保留兼容诊断字段，供历史工具读取；新报告使用 `summarize_point_hits`，这些历史字段不参与当前决策。

## 单文件模型

`ScreenCornerSingle` 的一个前向计算包含：图内缩放与归一化 → MobileNetV3 全局特征 → FPN 粗四角点 → 图内局部图像和特征采样 → 局部角点头 → 最终四角点。

导出为一个 TorchScript `screen_corner_single.pt`，含全部权重和常量。推理只需 PyTorch 和输入图片；不读取旧 global / ROI / local 模型、不下载预训练文件，也不依赖 torchvision。训练阶段可以缓存冻结特征，但缓存不属于最终推理依赖。

程序入口 `detect_frame.py` 支持 `SCREEN_PDF_SINGLE_MODEL`，或读取模型目录内的 `screen_corner_single.pt`。一旦单文件存在即优先运行；坏路径和坏文件应暴露错误，不能悄悄切换旧链路。

## 数据纪律

追光活动 27 张人工修正图片：固定 17 训练、5 开发验证、5 已查看留出回归。原始数据别名：`data/raw/202610-zhuiguang-ai`，标注和分组拆分见 `data/curated/annotations/202610-zhuiguang-ai_pages.jsonl`、`data/splits/cross_project/202610-zhuiguang-ai_v1.json`。

人工真值从选中候选的 `manualQuad` 解析，不把自动候选当人工点。沿用屏幕内容 dHash 近重复分组，保持训练与验证路径不相交。新增活动内部拆分不能代表跨活动泛化。

先冻结候选和文件摘要，再评估回归集。旧域 265 张验证、新活动 5 张验证、商场 29 张回归、新活动 5 张已查看留出分别报告。后两组在此前实验已经查看，不能再作为独立盲测。

## 复现入口

- 新训练：`program/engine/training/train_screen_corner_single.py`
- 单文件模型与加载：`program/engine/screen_corner_single.py`
- 每轮资产：`training/runs/screen_corner_single/round_xxx/`
- 历史多指标实验：`training/runs/screen_corner_unified_20261017/` 与 `_longhead/`，历史产物不回写。

本轮已完成结果见 [单指标单体模型报告](../../research/experiments/2026-10-17-single-point-single-model-results.md)。追光人工标注当前修订为 `202610-zhuiguang-ai_pages_v2.jsonl`，拆分版本 `202610-zhuiguang-ai_v2.json`；仅一个已查看回归角点改变，训练与验证标签保持不变。旧快照和评估不覆盖，最新重计记录在本轮 `reports/annotation_revision_comparison.json`。
