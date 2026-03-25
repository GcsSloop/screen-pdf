# Distillation Run Convention

这份文档定义统一蒸馏模型的演进目录结构。

目标很明确：

1. 锁定当前 teacher 版本。
2. 每一轮测试、训练、演进都放在独立目录。
3. 不同轮次的数据、报告、checkpoint 不互相覆盖。
4. 后续可以直接从目录判断某一轮属于哪一次模型演进。

## 锁定对象

当前已锁定的蒸馏主线为：

- 对外发布名：`deep_screen_v1`
- 内部技术名：`ds_corner_unified_distill_v1`

当前 teacher 冻结为：

- `r3`
- `v28`

## 目录根

统一蒸馏项目建议使用下面的根目录：

```text
training/runs/deep_screen_v1/
```

这里代表 `deep_screen_v1` 的整个演进主线。

## 每轮目录

每一轮演进必须单独使用一个 round 目录，例如：

```text
training/runs/deep_screen_v1/round_001/
training/runs/deep_screen_v1/round_002/
training/runs/deep_screen_v1/round_003/
```

轮次目录必须是自包含的，不允许把不同 round 的数据混在同一个目录里。

## 推荐子目录

每个 round 目录建议包含以下子目录：

```text
round_001/
├── README.md
├── manifest.json
├── data/
├── checkpoints/
├── reports/
├── artifacts/
└── logs/
```

### `data/`

放这一轮专属的数据快照和派生数据。

建议存放：

- teacher 预测结果
- round 专属训练样本
- round 专属验证样本
- round 专属 holdout/eval 结果
- 这一轮生成的任何中间数据

### `checkpoints/`

放这一轮的权重和中间 checkpoint。

### `reports/`

放这一轮的评估报告、对比报告和结论。

### `artifacts/`

放这一轮生成的可视化、样例图、曲线图和其他辅助产物。

### `logs/`

放训练日志、命令行记录和运行摘要。

## round 记录要求

每个 round 至少要能回答下面的问题：

1. 这轮用了哪些 teacher？
2. 这轮对应哪个对外版本和内部版本？
3. 这轮的数据来自哪里？
4. 这轮的评估指标是多少？
5. 这轮是否晋升、保留还是淘汰？

因此每个 round 目录必须有一个 `manifest.json`，至少记录：

- `public_name`
- `internal_name`
- `round`
- `teacher_models`
- `training_dataset`
- `status`

## 命名规则

- round 目录统一用 `round_001` 这种三位数形式。
- 目录顺序必须单调递增。
- 一轮一目录，不复用旧目录。
- 评估时优先引用 round 目录里的数据和报告，而不是散落在别处的临时文件。

## 当前落地

当前已经建立的首轮目录应当是：

```text
training/runs/deep_screen_v1/round_001/
```

