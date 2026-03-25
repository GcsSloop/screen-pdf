# Model Release Convention

这份文档把“统一蒸馏模型”从训练记录、发布命名到 runtime 落盘方式统一起来，避免后续只改一层而其他层继续沿用旧约定。

## 关系说明

- `model-naming-rules.md` 负责定义名字本身。
- 本文负责定义这些名字怎么进入训练注册表、怎么落到 runtime 目录、怎么被程序消费。

## Registry 字段

每个正式发布版本都应该在 `training/registry/` 下拥有一条记录，字段建议如下：

| 字段 | 含义 |
| --- | --- |
| `public_name` | 对外发布名，例如 `deep_screen_v1` |
| `internal_name` | 内部技术名，例如 `ds_corner_unified_distill_v1` |
| `status` | 生命周期状态，建议取值：`candidate`、`promoted`、`rejected`、`deprecated` |
| `teacher_models` | 蒸馏教师模型列表 |
| `training_dataset` | 训练数据集标识 |
| `validation_summary` | 验证指标摘要 |
| `runtime_files` | 要进入 `models/runtime` 的文件列表 |
| `promoted_at` | 晋升时间 |
| `notes` | 备注 |

## Registry 记录原则

1. `public_name` 和 `internal_name` 必须一一对应。
2. `runtime_files` 里记录的是最终发布时真正放进 runtime 目录的文件名。
3. 只有 `status=promoted` 的模型，才允许进入 runtime 默认集合。
4. `candidate` 只能留在训练/实验侧，不应作为默认发布版本。
5. `rejected` 和 `deprecated` 只保留历史记录，不进入默认加载路径。

## Runtime 目录约定

当前 `models/runtime` 仍承担两类内容：

1. 旧的兼容运行时组件。
2. 未来统一蒸馏模型的正式发布文件。

### 现阶段

- 旧链路文件继续保留在 `models/runtime` 平铺目录中。
- 当前默认加载逻辑仍兼容旧文件名。

### 统一模型发布后

统一蒸馏模型正式发布时，运行时目录应优先使用对外发布名：

```text
models/runtime/deep_screen_v{major}.pt
```

如果需要保留旧组件用于回退或对照，可以继续共存，但它们必须明确标记为兼容资产，而不是正式发布入口。

## 推荐映射

正式记录建议按下面方式对齐：

- 对外名：`deep_screen_v1`
- 内部名：`ds_corner_unified_distill_v1`
- runtime 文件：`models/runtime/deep_screen_v1.pt`
- registry 文件：`training/registry/deep_screen_v1.json`

## 现阶段过渡原则

1. 在统一模型真正接管默认链路之前，不要删除当前旧 runtime 文件。
2. 新的统一模型只要晋升，就必须同时更新：
   - `training/registry`
   - `models/runtime`
   - `docs/status/current-status.md`
3. 如果同一版本既有内部权重名又有对外发布名，发布名优先用于 runtime 和用户入口，内部名优先用于训练与审计。

