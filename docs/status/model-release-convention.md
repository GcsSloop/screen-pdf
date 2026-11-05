# Model Release Convention

## 2026-10-17 单文件补充（当前优先）

新主线为 `screen_corner_single/round_xxx`，不覆盖旧蒸馏轮次。最终交付一个 `screen_corner_single.pt`（TorchScript），包含粗定位、局部采样、精修全部参数与计算。新的 manifest 使用 `architecture=screen_corner_single_v1`，`runtime_files` 只列这一个权重文件；`runtime_digest` 基于该文件摘要计算。元数据可记录初始化来源，但不作为运行依赖。对外正式发布编号继续使用 `model-*`，候选不伪造正式发布编号。

唯一准确率和选优规则为 `point_le_0_01_ratio`；旧三阶段约定和多指标门槛仅用于历史溯源。下文记载旧发布兼容行为，当前正式单体 runtime 为 round_038 张凯源 /三雄极光精细定位。


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
| `runtime_digest` | 基于 runtime manifest 中实际运行时文件计算出的统一摘要；单文件模型使用该权重摘要 |
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

当前 `models/runtime` 以已晋升的单文件模型为默认入口，并保留三阶段平铺文件名以保证兼容：

- `screen_corner_single.pt`
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

- `public_name = screen_corner_single_round042_qujingjing_xuhui`
- `model_release_id = model-20261029-180536-9db7ad220b21`
- `round = round_042`
- 旧兼容链：`r130 + c22 + r69probe`
- 正式 runtime 为 `screen_corner_single_9db7ad220b21`；桌面版本 0.2.28，晋升审计与安装包验收见 [当前状态](current-status.md)。

## 晋升动作

当新的 runtime 组合被晋升时，至少要同步：

1. 更新 `models/runtime/*.json`
2. 更新 `training/registry/*.json`
3. 生成新的 `model_release_id`
4. 记录新的 `runtime_digest`
5. 视需要推送 `model-*` tag
