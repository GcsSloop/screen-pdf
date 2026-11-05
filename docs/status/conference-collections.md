# 会议 / 场次项目集合

2026-10-17：日常入口改为 [会议集合总览](../../data/curated/collections/conference_v1/README.md)，原演讲作为集合成员保留，不再依靠一串逐演讲报告比较。48 个原项目共 1950 页，排除 4 个异常页面后，13 个可打开的 ScreenPDF 合并项目共 1946 页；1808 页明确且有效、无冲突的人工标注用于 [全量基准](../../data/curated/collections/conference_v1/BASELINE.md)。

## 查看与校正

总览每行提供合并项目。打开该 `screen-pdf-project.json`，即可集中查看同场所有演讲；页面名称带原项目名，`source-provenance.json` 对应原项目、页面 ID、原图及内容摘要。13 个合并项目保留 1946 页及原候选；未标注页继续保留，4 个冲突 / 越界异常页面按用户要求排除。杭州智能交通年会合并项目由 96 页变为 92 页；原始 1950 页资料不动。图片通过独立 APFS 写时复制或普通复制放在项目内部，符合 v2 相对路径规范；没有指向原图的符号链接或硬链接。

合集里的修改只保存在合集副本，不自动回写原始项目、训练标签或冻结基准。要将合集校正纳入后续数据，先按来源映射核对，再建立新标注修订和新集合版本；人工点位不能被自动候选覆盖。

[人工校正包](../../data/curated/collections/conference_v1/conflict-review/README.md) 单列 1 张重复照片的 2 个冲突版本，以及另 2 页越界人工框。包含原项目链接、所有四角坐标、彩色叠框、角点原尺寸放大、CSV 和 4 页可编辑校正项目。用户已将这 4 个页面确认为异常，从日常合并项目排除，无需等待人工校正；此包仅保留溯源对照。基准原本已排除它们，输入、预测与分数不变。排除和前后页数见 `excluded-anomaly-pages.json`、`collection-exclusion-audit.json`。

## 如何归组

主集合是会议 / 场次，屏幕外观标签用于集合内下钻，演讲 / 项目作为来源。配置为 `program/engine/training/conference_collections.json`，按源目录显式映射，无法唯一归组时停止并要求补配置，避免静默丢项目。

中国智慧道路照明大会按用户的 `展会/2026/数据集场次分类.txt` 分为 3 月 19 日、20 日两个集合。杭州智能交通年会、福州高速公路信息化大会合并各自全部演讲。商场、AWE、扬州保留为专项测试；杭州隧道照明集合包含刘奇与涂耘，共 61 页，目标是超宽屏右侧 PPT。

杭州智能交通年会源目录前缀为 `202503`、演讲目录为 `2026-03`，此次保留原名，没有擅自统一活动年份。黑底、实体边框、条纹、遮挡等既有交叉标签保留为 [屏幕场景诊断](../../data/curated/collections/conference_v1/SCENES.md)，同图多标签不重复计入会议总数；没有标签的资料明确单列，不按会议主题猜外观。

## 真值、冲突和用途

真值仅从原项目显式人工 `manualQuad` 解析，选中候选人工框优先，其次页面人工框。自动 `activeQuad`、自动候选和“已查看”状态本身不构成人工真值。138 页无显式人工框、2 页框无效、2 个同图冲突版本，共排除 142 页；原因在 `excluded_pages.json`。缺标注与模型失败是不同概念：前者不进入人工准确率，后者四个点全计未命中。

照片按文件内容 SHA256 识别，四角统一 TL/TR/BR/BL 后以 0.001 原图像素容差比较；同名不同照片不误合并，改名但字节相同的照片可识别。当前不是重压缩 / 裁切图片的近重复识别工具。相同照片人工框不一致或跨用途时，全部版本隔离，详见 `duplicate_conflicts.json`；有效且一致的重复照片在基准仅计一次。无效框也参与冲突检查，不能先过滤再漏掉另一版本。

用途继承 `round_010/data/with_tunnel_focus` 实验快照，保留 1166 页训练计划、305 页开发、91 页已查看回归，另 246 页不在当前快照中。这里记录的是快照成员用途：round_010 最终选中 control，与 round_009 都没有用刘奇新增的 21 张训练计划页训练。快照外资料的历史训练暴露未穷尽核查，也不代表独立盲测。原训练采样、开发、回归拆分均不重写。

<!-- smart-road-3-20-fine-labels:start -->
2026-10-24：3 月 20 日「智慧城市道路与景观照明融合的守正创新实践」源项目人工点位精细修订，58 页 232 点全部微调（最大 0.645% 对角线），`IMG_8243` 新增显式人工框。新修订写入 `annotations/20260320-smart-road-lighting_pages_v2.jsonl`、`splits/cross_project/20260320-smart-road-lighting_v2.json` 和训练快照；本文件描述的 `conference_v1` 冻结集合、1808 页基准及旧用途不回写，3.20 集合仍显示旧 58 页人工框。详见 [报告](../../research/experiments/2026-10-24-smart-road-lighting-fine-labels.md)。
<!-- smart-road-3-20-fine-labels:end -->

## 本次基准与复现

模型在推理前冻结：包内 round_009 `6fc6378701c4` 与先前已选中的 round_010 control `d9d6ad842126`。逐页重新运行全部 1808 页，不拼接历史分数。只报告 `point_le_0_01_ratio`：原图像素角点误差 / 人工框轴对齐包围盒对角线 <=1%；按总命中角点 / 总人工角点汇总。训练 / 开发 / 已查看回归 / 快照外分别展示，包含训练页的全集分数只作覆盖与拟合诊断。耗时只作观测，不参与准确率排序。

完整资产为 `research/experiments/conference-benchmark/20261017_v1/`，含冻结模型摘要、输入快照、两个模型逐页预测、失败原因、分集合与成员统计。源项目和既有训练快照完成前后摘要核查；正式 runtime、安装包、模型权重和阈值没有改变。本次基准不重新选候选、不训练、不晋升。

新增 / 校正后使用新的 revision 和新基准目录，保留旧报告和待处理校正。仓库根目录运行：

```bash
PYTHONPATH=program/engine program/engine/vendor/macos/bin/python3 program/engine/training/build_conference_collections.py \
  --output data/curated/collections/conference_v2 \
  --active-data training/runs/screen_corner_single/round_010/data/with_tunnel_focus

PYTHONPATH=program/engine program/engine/vendor/macos/bin/python3 program/engine/training/materialize_conference_projects.py \
  --collections data/curated/collections/conference_v2

PYTHONPATH=program/engine program/engine/vendor/macos/bin/python3 program/engine/training/build_conflict_review.py \
  --collections data/curated/collections/conference_v2

PYTHONPATH=program/engine program/engine/vendor/macos/bin/python3 program/engine/eval/benchmark_conference_collections.py \
  --collections data/curated/collections/conference_v2 \
  --output research/experiments/conference-benchmark/20261017_v2 \
  --model package_0_2_16=training/runs/screen_corner_single/round_009/checkpoints/screen_corner_single.pt \
  --model candidate_round010_control=training/runs/screen_corner_single/round_010/checkpoints/screen_corner_single.pt
```

既有人工场景视图从冻结逐页预测派生，不增加推理、调参或数据选择：

```bash
PYTHONPATH=program/engine program/engine/vendor/macos/bin/python3 -c "from pathlib import Path; from eval.conference_scene_views import render; render(Path('data/curated/collections/conference_v2'), Path('research/experiments/conference-benchmark/20261017_v2'), Path('data/curated/scenes'))"
```
