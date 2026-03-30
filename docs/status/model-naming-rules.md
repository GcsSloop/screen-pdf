# Model Naming Rules

这份文档定义模型的内部训练命名、对外统一发布编号，以及程序版本和模型版本的边界。

## 命名原则

1. 程序版本和模型版本必须分离。
2. 内部训练名继续服务训练、评估和审计，不直接暴露给外部发布流程。
3. 对外模型发布编号必须稳定、唯一、可追溯。
4. 同一组 runtime 三阶段模型一旦晋升，就只对应一个统一的 `model_release_id`。

## 程序版本

程序版本使用 semver，并通过 git tag 发布：

```text
v0.2.1
```

用途：

- 桌面端 `package.json`
- `tauri.conf.json`
- `Cargo.toml`
- GitHub Release
- 自动更新 `latest.json`

## 模型统一发布编号

对外模型版本不再使用 `deep_screen_v{major}` 这种主版本号形式，改为：

```text
model-YYYYMMDD-HHMMSS-<runtime_digest_8>
```

示例：

- `model-20260330-153045-ab12cd34`
- `model-20260402-091530-4ef091aa`

规则：

- 时间使用 UTC 发布时间。
- digest 使用当前 runtime manifest 里三阶段模型摘要的前 8 位。
- 该编号只表示一次正式外部模型发布，不表示训练轮次。

用途：

- `models/runtime/*.json` 中的 `model_release_id`
- Git tag：`model-*`
- 程序展示给用户的模型统一编号
- 模型元数据发布记录

## 内部训练命名

内部训练继续沿用当前方案，例如：

- `r85`
- `c21`
- `r64probe_mt065`
- `deep_screen_r1_2026_03_28`
- `training/runs/.../round_xxx`

这些名字继续用于：

- run 目录
- checkpoint
- 评估报告
- 实验比对
- registry 审计

## 记录要求

每个正式 runtime 发布至少要同时记录：

- `public_name`
- `model_release_id`
- `runtime_digest`
- `teacher_models`
- `validation_summary`
- `status`
- `promoted_at` 或 `released_at`

## 当前约定

- 当前程序版本线使用 `v*`
- 当前模型发布线使用 `model-*`
- 程序显示模型名时优先读取 `model_release_id`
- 如果缺失，再回退到 `public_name`
