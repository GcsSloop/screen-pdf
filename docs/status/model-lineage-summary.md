# Model Lineage Summary

这份表把当前项目里“模型怎么调用”“历史模型有哪些”“各自跑出了什么结果”“是否已经淘汰”统一放在一起，避免信息散落在多个实验报告里。

## 当前调用顺序

运行时入口在 [program/engine/detect_frame.py](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/engine/detect_frame.py) 和 [program/engine/two_stage_corner_pipeline.py](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/engine/two_stage_corner_pipeline.py)。

实际调用顺序如下：

1. 先走规则/几何候选：`perspective_detect.detect_best_candidate(image)`。
2. 如果没有显式关闭模型且 runtime 模型都存在，则加载 coarse/global 模型：`global_corner_model.pt`。
3. 用 coarse 结果构造 ROI，再调用 `corner_heatmap_model.pt` 做 ROI refine。
4. 如果存在 local 模型，则再调用 `local_corner_moe_coord_model.pt`，否则回退到 `local_corner_moe_model.pt`。
5. `build_detect_payload()` 会把模型结果放到候选列表最前面，并把模型结果作为 `best`。

所以，当前可理解为：

- 有 local 模型时：`规则候选 -> global -> ROI refine -> local refine`
- 没有 local 模型时：`规则候选 -> global -> ROI refine`

其中 `local_corner_moe_coord_model.pt` 的优先级高于 `local_corner_moe_model.pt`。

## 历史模型总数

按当前高信号文档里能够明确找到“版本号 + 结果 + 结论”的模型统计：

- global / coarse 线：`11` 个
- local-corner 线：`9` 个
- 合计：`20` 个

如果把 `runtime old` 这个旧基线也算进去，则是 `21` 个。

说明：

- 这里没有把 `research/experiments` 里的重复跑、临时产物和中间态反复计数。
- 下面的表只收录当前已经形成明确判断的版本。

## Global / Coarse 线

| 模型 | 关键结果 | 结论 | 当前状态 | 主要来源 |
| --- | --- | --- | --- | --- |
| `v12` | `jinjiang_global_holdout point_error_mean=0.0475`, `point_le_0_01_ratio=0.0` | 晋江集显著退化，不能替代运行时 | `rejected` | [model-experiment-registry.json](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/model-experiment-registry.json) |
| `v13` | `jinjiang_global_holdout point_error_mean=0.0092`, `point_le_0_01_ratio=0.913` | 作为 candidate 有明显改善 | `candidate` | [model-experiment-registry.json](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/model-experiment-registry.json) |
| `v14` | `new_project_holdout point_error_mean=0.0077`, `avg_page_infer_ms=402.5` | 泛化与速度都更稳 | `candidate` | [model-experiment-registry.json](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/model-experiment-registry.json) |
| `v17` | `<=1%=0.8665`, `point_error_mean=0.0072`, `avg=30.71 ms` | 全量回归盘表现稳定 | 历史对比模型 | [broad-global-model-validation-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-24-broad-global-model-validation-report.md) |
| `v18` | `<=1%=0.8825`, `point_error_mean=0.0069`, `avg=30.67 ms` | 全量命中率最高，但平均几何不是最优 | 历史对比模型 | [broad-global-model-validation-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-24-broad-global-model-validation-report.md) |
| `v19` | `<=1%=0.8765`, `point_error_mean=0.0064`, `avg=30.53 ms` | 平均几何最好，也是当前更值得推进的主线 | 历史对比模型 | [broad-global-model-validation-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-24-broad-global-model-validation-report.md) |
| `v21` | `focus_test<=1%=0.1111`, `broad_non_focus<=1%=0.8645`, `holdout2=0.9565` | 仍保留在对比序列里，但不作为当前主线 | 历史对比模型 | [model-compare-balanced-2026-03-25/summary.json](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/model-compare-balanced-2026-03-25/summary.json) |
| `r1` | `focus_test<=1%=0.3889`, `broad_non_focus<=1%=0.8566`, `holdout2=1.0` | 作为 baseline | 历史基线 | [balanced-generalization-r3-r5-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-balanced-generalization-r3-r5-report.md) |
| `r3` | `focus_test<=1%=0.5000`, `broad_non_focus<=1%=0.8486`, `holdout2=0.9565` | 当前 coarse/global 主干 | 当前运行时主线 | [docs/status/current-status.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/current-status.md), [balanced-generalization-r3-r5-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-balanced-generalization-r3-r5-report.md) |
| `r4` | `focus_test<=1%=0.4444`, `broad_non_focus<=1%=0.8486`, `holdout2=1.0` | 训练面更好，但真实 focus 回退 | `淘汰` | [balanced-generalization-r3-r5-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-balanced-generalization-r3-r5-report.md) |
| `r5` | `focus_test<=1%=0.3333`, `broad_non_focus<=1%=0.8526`, `holdout2=1.0` | 对 focus_test 破坏明显 | `淘汰` | [balanced-generalization-r3-r5-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-balanced-generalization-r3-r5-report.md) |

## Local-corner 线

| 模型 | 关键结果 | 结论 | 当前状态 | 主要来源 |
| --- | --- | --- | --- | --- |
| `runtime old` | `focus_test point_error_mean=0.0109`, `broad_test all_corners_<1%=0.2889`, `holdout=0.1034` | 旧基线，可作为对照 | 旧基线 | [2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-local-corner-bl-structure-experiments-report.md) |
| `v26a` | `focus_test all_corners_<1%=0.7500`, `broad_test=0.2222`, `holdout=0.4655` | 强结构监督有效，但 broad 回落 | 历史候选 | [2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-local-corner-bl-structure-experiments-report.md) |
| `v26b` | `focus_test all_corners_<1%=0.7500`, `broad_test=0.2222`, `holdout=0.4310` | 没有比 v26a 更好 | 历史候选 | [2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-local-corner-bl-structure-experiments-report.md) |
| `v27a` | `focus_test all_corners_<1%=0.7500`, `broad_test=0.1852`, `holdout=0.5172` | holdout 明显提升，但 broad 仍偏弱 | 历史候选 | [2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-local-corner-bl-structure-experiments-report.md) |
| `v27b` | `focus_test all_corners_<1%=0.7500`, `broad_test=0.1704`, `holdout=0.5517` | 比 v27a 更稳一点，但 broad 没恢复 | 历史候选 | [2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-local-corner-bl-structure-experiments-report.md) |
| `v28` | `focus_test all_corners_<1%=0.7500`, `broad_test=0.1926`, `holdout=0.5862` | 当前 local-corner 最稳候选 | 当前运行时候选 | [docs/status/current-status.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/current-status.md), [2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-local-corner-bl-structure-experiments-report.md) |
| `v31` | `focus_test all_corners_<1%=0.7500`, `broad_test=0.1407`, `holdout=0.3621` | 单独加 BL 权重，专项偏置加重 | `淘汰` | [2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-local-corner-bl-structure-experiments-report.md) |
| `v32` | `focus_test all_corners_<1%=0.7500`, `broad_test=0.1556`, `holdout=0.5862` | holdout 回到 v28 水平，但 broad 仍弱 | 历史候选 | [2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-local-corner-bl-structure-experiments-report.md) |
| `v33` | `focus_test all_corners_<1%=0.6250`, `broad_test=0.3481`, `holdout=0.0862` | broad 提升明显，但 holdout 崩掉 | `淘汰` | [2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-local-corner-bl-structure-experiments-report.md) |
| `v34` | `focus_test all_corners_<1%=0.7500`, `broad_test=0.1704`, `holdout=0.0690` | 统一下调 flip_prob 不是正确收口方式 | `淘汰` | [2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/2026-03-25-local-corner-bl-structure-experiments-report.md) |

## 结论

1. 当前真正的运行时主线是 `r3 -> v28`，外加 `global -> ROI refine -> local refine` 这条三段式链路。
2. global 线里，`r3` 是当前主线；`r4` 和 `r5` 已明确淘汰。
3. local 线里，`v28` 仍是当前最稳候选；`v31`、`v33`、`v34` 明确淘汰。
4. 如果只看历史对比，`v19` 是 coarse/global 线里平均几何最好的版本，但当前仓库运行时共识仍以 `r3` 为准。
5. 当前最重要的训练问题仍然是：
   - 不破坏泛化的前提下提升角点精度
   - 避免翻转增强与专项权重带来的过拟合

## 相关来源

- [docs/status/current-status.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/current-status.md)
- [docs/status/key-metrics.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/key-metrics.md)
- [program/engine/detect_frame.py](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/engine/detect_frame.py)
- [program/engine/two_stage_corner_pipeline.py](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/engine/two_stage_corner_pipeline.py)
- [research/experiments/model-experiment-registry.json](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/research/experiments/model-experiment-registry.json)
