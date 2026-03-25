# Scene Map V1 Summary

## Current Import Status

- Source dataset: `conference_202603_china_smart_road_lighting`
- Newly discovered projects since the last baseline import: `0`
- Existing tracked projects: `22`

说明：

- 这次没有发现新的 `screen-pdf-project.json` 项目目录。
- 我仍然按当前基线重跑了导入流程，确保目录、拆分和场景映射保持同步。

## Manual Scene Map Status

- Manual scene map file:
  - `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/curated/scenes/conference_202603_china_smart_road_lighting_project_scene_map.manual.json`
- Canonical merged scene map:
  - `/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/curated/scenes/conference_202603_china_smart_road_lighting_project_scene_map.json`

当前结果：

- `unknown` 标签项目数：`0`
- `needs_manual_review=true` 项目数：`0`

## Tag Distribution

- `white_ppt`: `13`
- `colorful_ppt`: `7`
- `near_color_background`: `4`
- `low_contrast_edge`: `10`
- `complex_background`: `2`
- `floor_reflection`: `3`
- `bottom_edge_interference`: `1`
- `black_border`: `2`
- `led_screen`: `1`
- `corner_occlusion`: `1`
- `ui_overlay`: `1`
- `lens_distortion_sensitive`: `2`
- `strong_perspective`: `22`

## Why This Helps

- 后续训练可以按 `white_ppt`、`near_color_background`、`floor_reflection`、`black_border` 等关键场景拆指标，而不是只看全量平均值。
- 像 `project_002`、`project_019`、`project_020`、`project_022` 这类历史 hard case，现在已经能稳定落入更准确的场景桶。
- 以后新增项目时，如果没有人工 scene map，脚本会先给出启发式标签；已有人工标签不会被覆盖。

## Next Recommended Step

基于这份 scene map，下一步最合适的是：

1. 生成按场景聚合的 benchmark 汇总
2. 在训练报告里同时输出全量指标和分场景指标
3. 先盯 `near_color_background`、`low_contrast_edge`、`floor_reflection` 这三组的回归表现
