# Data And Training Layout

## 目标

把“原始资料”“可训练数据”“训练产物”“运行时模型”四层明确拆开，避免互相污染。

## 顶层新增结构

```text
screen-pdf/
├── data/
│   ├── raw/
│   ├── staging/
│   ├── curated/
│   ├── splits/
│   ├── derived/
│   └── benchmarks/
├── training/
│   ├── configs/
│   ├── runs/
│   ├── checkpoints/
│   ├── reports/
│   ├── registry/
│   └── promoted/
├── models/
│   └── runtime/
└── program/
```

## 分层边界

### `data/raw`

- 真实原始项目资料
- 保留用户原始目录习惯
- 不做覆盖式改写

### 会议场次集合

`data/curated/collections/conference_v1/` 是当前统一入口：会议 / 场次作为主集合，演讲保留来源，黑底、边框和遮挡等多标签作为交叉诊断。`desktop-projects/` 提供可编辑 v2 根项目及独立图像副本；`pages.jsonl` 提供按照片内容去重的显式人工真值。查看项目保留未标注页，冲突 / 无效框页面作为异常排除并保留溯源对照；基准仅使用真值明确且无冲突的页面。

同图异框或跨用途冲突先进入 `conflict-review/`，不自动挑选一份。数据修订、人工校正、既有用途及模型暴露的边界见 [集合规则](../status/conference-collections.md)。基准大文件放 `research/experiments/conference-benchmark/`，文档只保留入口；输入与模型先冻结，完成后检查原项目和训练快照摘要。

### `data/curated`

- 结构化后的项目、场景和标注
- 为训练和评估提供稳定输入

### `data/splits`

- 正式训练 / 验证 / holdout 拆分
- 必须可追溯、可复现

### `data/derived`

- 面向具体任务的派生数据
- 允许按模型任务组织，但不直接替代原始标注

### `training`

- 训练配置、训练日志、checkpoint、报告、模型注册
- 与桌面程序完全分离

### `models/runtime`

- 程序当前实际加载的模型
- 只接受经过验证后晋升的版本

## 推荐组织原则

- 场景分类应该存在，但放在 `data/curated/scenes`，而不是直接打散原始项目目录。
- 模型任务分类应该存在，但放在 `data/derived` 和 `training/configs`。
- 训练时优先使用项目级拆分和 holdout，而不是随机抽样拆分。
- 所有“用户手动修正过”的标注，都应尽量在 `curated/annotations` 里保留来源字段。
