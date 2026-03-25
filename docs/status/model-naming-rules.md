# Model Naming Rules

这份文档定义统一模型的对外发布名和内部技术名，避免后续在训练、注册、发布和文档里各写各的。

## 命名原则

1. 对外发布名要短，容易记，适合传播。
2. 内部技术名要自描述，能看出它属于哪条模型线。
3. 对外名和内部名必须一一映射。
4. 版本号只表示正式发布序列，不表示训练轮次。
5. 只有通过独立 holdout 和速度门槛的模型，才允许晋升为新版本。

## 对外发布名

格式：

```text
deep_screen_v{major}.pt
```

示例：

- `deep_screen_v1.pt`
- `deep_screen_v2.pt`

规则：

- 只保留 `major` 版本号。
- 不写 `beta`、`final`、`best` 之类状态词。
- 只有模型达到正式发布标准，才递增版本号。

用途：

- runtime 发布包
- 对外说明
- 用户可见的默认模型名
- 下载和引用链接

## 内部技术名

格式：

```text
ds_corner_unified_distill_v{major}
```

示例：

- `ds_corner_unified_distill_v1`
- `ds_corner_unified_distill_v2`

规则：

- `ds` 表示 deep screen。
- `corner_unified_distill` 表示统一角点蒸馏模型。
- 内部版本号默认与对外版本号一致。
- 如果只是 ablation 或临时试验，必须加后缀，但不能冒充正式版本。

可选试验后缀示例：

- `ds_corner_unified_distill_v1_exp_flip_split`
- `ds_corner_unified_distill_v1_exp_roi_warp`
- `ds_corner_unified_distill_v1_candidate_a`

## 文件名约定

建议按用途分层：

- 训练权重：`ds_corner_unified_distill_v1.pt`
- 发布权重：`deep_screen_v1.pt`
- 训练报告：`ds_corner_unified_distill_v1_report.md`
- 指标记录：`ds_corner_unified_distill_v1_metrics.json`
- Registry 记录：`ds_corner_unified_distill_v1.json`

## 升级规则

只有满足下面条件，才允许从 `v{major}` 升到 `v{major+1}`：

1. 独立 holdout 不回退。
2. 速度满足当前门槛。
3. 关键几何指标不回退。
4. 模型行为有明确、可重复的改进。

如果只是调整：

- 学习率
- loss 权重
- 数据增强
- ablation 试验

但还没达到正式发布门槛，就不要升主版本号，只加实验后缀。

## 记录要求

每个正式版本都必须同时记录：

- `public_name`
- `internal_name`
- `teacher_models`
- `training_dataset`
- `validation_summary`
- `status`
- `promoted_at`

建议写入训练注册表和发布说明。

## 当前约定

- 对外发布名：`deep_screen_v{major}.pt`
- 内部技术名：`ds_corner_unified_distill_v{major}`
- 当前先从 `v1` 开始

