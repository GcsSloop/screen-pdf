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
