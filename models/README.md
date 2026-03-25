# Models

## runtime

详细的 runtime 目录约定见：

- [runtime/README.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/models/runtime/README.md)

程序默认使用的模型目录：

- `runtime/global_corner_model.pt`
- `runtime/corner_heatmap_model.pt`
- `runtime/local_corner_moe_coord_model.pt`
- `runtime/local_corner_moe_model.pt`
- 统一蒸馏模型正式发布后，优先采用 `runtime/deep_screen_v{major}.pt`

说明：

- 这里只放运行时被程序直接加载的模型。
- 历史训练 run 产物不放这里，统一保留在 `../research/experiments`。
