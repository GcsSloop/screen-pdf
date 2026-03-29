# Runtime

这个目录只放程序默认会加载的运行时模型。

## 现阶段兼容文件

- `global_corner_model.pt`
- `corner_heatmap_model.pt`
- `local_corner_moe_coord_model.pt`
- `local_corner_moe_model.pt`
- `deep_screen_r1_2026_03_28.json`

## 当前固定发布

- `deep_screen_r1_2026_03_28`
- app 版本：`0.2.0`
- coarse：`r66`
- roi：`c12`
- local：`r48`

## 统一蒸馏模型

未来统一模型正式发布后，建议直接使用对外发布名落盘：

- `deep_screen_v{major}.pt`

例如：

- `deep_screen_v1.pt`

## 约定

1. 这里不放历史训练 run 产物。
2. 这里不放实验中间件。
3. 这里只放默认运行时会加载的正式模型，或其兼容资产。
4. 新的统一模型晋升后，应同步更新 `training/registry` 和 `docs/status`。
