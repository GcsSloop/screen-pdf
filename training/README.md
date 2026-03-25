# Training

`training` 只存放训练资产，不存放桌面端程序。

## 目录约定

- `configs/`
  - 训练配置、实验配置、数据拆分配置。
  - 建议按模型族区分，例如 `global_corner/`、`local_corner/`、`ocr_overlay/`。
- `runs/`
  - 每次训练的工作目录。
  - 包含日志、临时评测结果、可视化样例、命令行快照。
- `checkpoints/`
  - 训练过程中产出的权重。
  - 默认不直接给桌面程序使用。
- `reports/`
  - 训练总结、对比报告、回归结论。
- `registry/`
  - 记录每一代数据集、模型版本、发布名、教师模型、关键指标。
  - 数据集记录继续沿用 `dataset_slug.json`。
  - 统一模型发布记录建议使用 `deep_screen_v{major}.json`。
  - 字段约定见 `registry/model-release-template.json` 和 `docs/status/model-release-convention.md`。
- `promoted/`
  - 经过验证后，准备晋升为运行时或候选运行时的模型与说明。

## 统一蒸馏主线

当前统一蒸馏主线已锁定为：

- 对外版本：`deep_screen_v1`
- 内部版本：`ds_corner_unified_distill_v1`
- teacher：`r3`、`v28`

对应的演进根目录：

- `runs/deep_screen_v1/`

每一轮演进都必须使用独立 round 目录，例如：

- `runs/deep_screen_v1/round_001/`
- `runs/deep_screen_v1/round_002/`

round 目录的详细约定见：

- [docs/status/distillation-run-convention.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/distillation-run-convention.md)

## 与 `models/runtime` 的关系

- `training/checkpoints` 是训练产物。
- `models/runtime` 是程序实际加载的已晋升模型。
- 任何模型进入 `models/runtime` 前，都应该在 `training/reports` 和 `training/registry` 里留下依据。

## 最小发布流程

1. 在 `training/runs` 产出训练结果
2. 在 `training/reports` 写清楚训练集、验证集、指标和结论
3. 在 `training/registry` 记录模型版本、发布名、教师模型和来源
4. 只有通过门槛的模型，才复制到 `models/runtime`
