# Model Release Convention

这份文档定义 runtime manifest、训练 registry、git tag 和桌面程序之间如何共享同一个模型发布语义。

## 目标

- 保留现有内部训练命名体系，不打断训练流程。
- 对外只暴露一个统一模型发布编号。
- 程序版本更新和模型版本更新彻底解耦。

## 核心字段

每个正式 runtime 发布都应在 `models/runtime/*.json` 和 `training/registry/*.json` 中记录：

| 字段 | 含义 |
| --- | --- |
| `public_name` | 人类可读别名，例如 `deep_screen_r1_2026_03_28` |
| `model_release_id` | 对外统一模型发布编号，例如 `model-20260330-153045-ab12cd34` |
| `runtime_digest` | 基于 runtime 三阶段模型清单计算出的统一摘要 |
| `teacher_models` | 三阶段组合来源 |
| `validation_summary` | 关键指标摘要 |
| `runtime_files` | 运行时依赖文件 |
| `status` | 生命周期状态，例如 `promoted` |
| `promoted_at` / `released_at` | 晋升或发布时间 |

## 发布边界

### 程序发布

- git tag 形式：`v*`
- 会同步桌面端版本号
- 会触发多平台程序打包
- 会生成并发布 `latest.json`
- 会触发桌面端自动更新链路

### 模型发布

- git tag 形式：`model-*`
- 只校验当前 promoted runtime manifest
- 只发布模型元数据
- 不触发桌面程序编译

## Runtime 目录约定

当前 `models/runtime` 仍保留三阶段平铺文件名以保证兼容：

- `global_corner_model.pt`
- `corner_heatmap_model.pt`
- `local_corner_moe_coord_model.pt`

统一发布信息放在 manifest 中，而不是强行改掉兼容文件名。

## 程序消费顺序

程序显示和日志输出按下面顺序读取模型编号：

1. `model_release_id`
2. `public_name`
3. 兼容回退值

## 当前固定版本

- `public_name = deep_screen_r1_2026_04_01_r130`
- `model_release_id = model-20260401-060636-035e5e08`
- app 版本线：`v0.2.7`

## 晋升动作

当新的 runtime 组合被晋升时，至少要同步：

1. 更新 `models/runtime/*.json`
2. 更新 `training/registry/*.json`
3. 生成新的 `model_release_id`
4. 记录新的 `runtime_digest`
5. 视需要推送 `model-*` tag
