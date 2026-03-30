# Current Status

## 仓库状态

新仓库已建立：

```text
/Users/gcssloop/WorkSpace/AIGC/screen-pdf
```

已迁入内容：

- 程序源码
- Python 引擎
- 数据目录骨架
- 训练资产目录骨架
- 运行时模型
- 历史训练与实验资料
- 当前高信号文档入口
- `engine/runtime / training / eval / tests` 分组视图

未迁入内容：

- `node_modules`
- `dist`
- `src-tauri/target`
- `__pycache__`
- 其他明显构建缓存

## 当前运行时模型共识

- coarse/global 主干：`r3`
- local-corner 候选：`v28`

`2026-03-28` 当前固定 runtime 发布版本：

- `deep_screen_r1_2026_03_28`
- `model_release_id = model-20260330-153045-e60e199b`
- app 版本：`0.2.1`
- runtime 组合：`r85 + c21 + r64probe(mt065)`

当前发布流程约定：

- 程序版本 tag：`v*`
- 模型版本 tag：`model-*`
- `v*` 触发桌面端自动构建、产物收集、`latest.json` 更新与 GitHub Release
- `model-*` 只校验当前 promoted runtime manifest，并发布模型元数据，不触发桌面端编译

组合原则：

- `r3` 负责先把轮廓找对。
- `v28` 只做局部角点精修。
- `v28` 不应替代 `r3`。

## 命名规则

统一模型命名已正式记录，后续新增统一蒸馏模型时必须遵守：

- 对外发布名：`deep_screen_v{major}.pt`
- 内部技术名：`ds_corner_unified_distill_v{major}`

详细规则见：

- [model-naming-rules.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/model-naming-rules.md)
- [model-release-convention.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/model-release-convention.md)

## 蒸馏主线

当前已锁定的蒸馏主线：

- `deep_screen_v1`
- 内部名：`ds_corner_unified_distill_v1`
- teacher：`r3`、`v28`
- 新架构：`shared_backbone_fpn_coarse_to_fine_local_moe`

当前轮次入口：

- [training/runs/deep_screen_v1/round_026/README.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/training/runs/deep_screen_v1/round_026/README.md)

轮次目录约定：

- 每一轮测试、演进、评估都必须放在独立 round 目录。
- round 目录下的数据、报告、checkpoint、日志不能和其他 round 混放。
- round 的记录以 `manifest.json` 为准。

详细规则见：

- [distillation-run-convention.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/distillation-run-convention.md)

当前 `V1` 已完成架构重置：

- 旧 `heatmap_offset` 学生线已从主线文档中移除
- 旧 `training/configs/deep_screen_v1/round_*` 已清理
- 旧 `training/runs/deep_screen_v1/round_*` 已清理
- 新 `V1` 已从新的 `round_001` 重新开始训练

当前新的训练目标：

- 共享 backbone 提升跨场景泛化
- coarse-to-fine 统一建模
- 只在 local refine 模块引入 MoE
- 按 round 持续训练并记录每次微调

当前新主线已完成 `round_001` 到 `round_041`。

`2026-03-26` 新增 strict-point `split_v2` 验证线：

- split 文件：`data/splits/cross_project/conference_202603_china_smart_road_lighting_split_v2_strict_point.json`
- 口径调整：
  - train `397` manual 页
  - val `60` manual 页
  - holdout `134` manual 页
  - 更强调 `model_two_stage / model_three_stage_local_moe / refined_edges / edge-heavy` 的项目混合
- 注意：
  - `round_042` 起使用新的 `split_v2`
  - 这些轮次与 `round_001` 到 `round_041` 的 `split_v1` 结果不能直接横向比较

`round_001` baseline：

- teacher 导出页数：train `388` / val `51` / holdout `152`
- student holdout：`point_error_mean=0.0365`
- student holdout：`point_le_0_05_ratio=0.7368`
- student holdout：`point_le_0_01_ratio=0.0`
- student 平均单页推理：`8.16 ms`
- 结论：新架构第一轮已跑通，但仍显著落后 teacher，继续训练

`round_032` 到 `round_041` 的阶段结果：

- `round_032`：
  - 新增 process distillation：`roi_stage_quad + refine_delta + visibility / edge / fallback`
  - 保存结果：`point_error_mean=0.0225`
  - 保存结果：`point_le_0_01_ratio=0.0773`
  - 保存结果：`avg_page_infer_ms=12.34`
  - 结论：重过程、重结构监督没有带来 strict-point 提升，说明结构 cue 监督噪声较大
- `round_033`：
  - 收紧为几何过程蒸馏，弱化 visibility / edge / fallback
  - 保存结果：`point_error_mean=0.0221`
  - 保存结果：`point_le_0_01_ratio=0.0938`
  - 保存结果：`avg_page_infer_ms=11.93`
  - 结论：较 `round_032` 和 `round_031` 有恢复，但仍明显低于 `round_025 = 0.1217`
- `round_034`：
  - 从 `round_033` 热启动，继续收紧为几何过程蒸馏 continuation
  - 保存结果：`point_error_mean=0.0222`
  - 保存结果：`point_le_0_01_ratio=0.0921`
  - 保存结果：`avg_page_infer_ms=9.88`
  - 结论：时延改善，但 strict-point 没有延续提升
- `round_035`：
  - 只保留 ROI-stage 中间框监督，移除 process-delta 训练项
  - 保存结果：`point_error_mean=0.0222`
  - 保存结果：`point_le_0_01_ratio=0.0855`
  - 保存结果：`avg_page_infer_ms=9.18`
  - 结论：更干净的 ROI-stage-only 蒸馏仍然回退，teacher-process 线应视为阶段性瓶颈
- `round_036`：
  - 从 `round_025 epoch_002` 热启动，回到 max-corner 主线并叠轻量 strict-point
  - 保存结果：`point_error_mean=0.0217`
  - 保存结果：`point_le_0_01_ratio=0.0888`
  - 保存结果：`avg_page_infer_ms=9.48`
  - 结论：没有守住历史 strict-point 高点，hybrid 方案无效
- `round_037`：
  - 从 `round_025 epoch_002` 热启动，近似复刻 `round_025` 配方，只把 checkpoint 选择改成 strict-point
  - 保存结果：`point_error_mean=0.0216`
  - 保存结果：`point_le_0_01_ratio=0.1102`
  - 保存结果：`avg_page_infer_ms=9.04`
  - 结论：这是 `round_025` 之后最接近历史峰值的一轮，但仍低于 `epoch_002 = 0.1250`
- `round_038`：
  - 继续保守 strict-point 线，进一步收软增强
  - 保存结果：`point_error_mean=0.0217`
  - 保存结果：`point_le_0_01_ratio=0.0954`
  - 保存结果：`avg_page_infer_ms=9.09`
  - 结论：收软增强没有保住峰值，反而进一步回退
- `round_039`：
  - 从 `round_037` 做 1 个极短低增强收敛轮
  - 保存结果：`point_error_mean=0.0222`
  - 保存结果：`point_le_0_01_ratio=0.0954`
  - 保存结果：`avg_page_infer_ms=7.55`
  - 结论：时延更低，但 strict-point 无提升
- `round_040`：
  - 切到 strict-point 难例与场景重平衡采样
  - 保存结果：`point_error_mean=0.0221`
  - 保存结果：`point_le_0_01_ratio=0.0987`
  - 保存结果：`avg_page_infer_ms=8.97`
  - 结论：数据/采样重分布也没有超过 `round_037`
- `round_041`：
  - 从 `round_040` 做 1 个低增强 replay
  - 保存结果：`point_error_mean=0.0229`
  - 保存结果：`point_le_0_01_ratio=0.0806`
  - 保存结果：`avg_page_infer_ms=9.24`
  - 结论：这条数据/采样验证线确认失败

当前对 process distillation 的结论：

- 仅做最终 quad 蒸馏不够
- 但把 teacher 的局部结构 cue 全量压进 student 也没有奏效
- 目前更可信的信号仍是几何过程，而不是 visibility / edge / fallback 的强监督
- 但几何过程蒸馏继续追到 `round_034`、`round_035` 也没有超过 `round_033`
- 严格回到 `round_025` 历史峰值继续追，也只能恢复到 `round_037 = 0.1102`
- 再继续做同家族短轮次微调，`round_038`、`round_039` 也都没有超过 `round_037`
- 数据/采样重分布快速验证到 `round_040`、`round_041` 也没有超过 `round_037`
- 下一步若继续沿 unified 方向推进，应优先做：
  - 真正重做 strict-point 数据集与 split，而不是继续调采样权重
  - 更可靠的 validation 口径
  - 或单独把 `round_025 epoch_002` 作为当前 strict-point 最佳候选冻结

`round_042` 到 `round_050` 的 `split_v2` 结果：

- `round_042`：
  - 首次切到 strict-point `split_v2`
  - `best epoch=3`
  - holdout：`point_error_mean=0.0235`
  - holdout：`point_le_0_01_ratio=0.0896`
  - holdout：`max_corner_le_0_03_ratio=0.1418`
  - holdout：`avg_page_infer_ms=8.91`
  - 结论：新口径更苛刻，validation 与 holdout 的 strict-point 相关性仍然不足
- `round_043`：
  - 从 `round_042 epoch_001` 做 1 轮低学习率、低增强 replay
  - `best epoch=1`
  - holdout：`point_error_mean=0.0249`
  - holdout：`point_le_0_01_ratio=0.1455`
  - holdout：`max_corner_le_0_03_ratio=0.0522`
  - holdout：`avg_page_infer_ms=9.58`
  - 结论：主指标显著上升，但伴随明显 inward shrink，整体几何质量同步退化
- `round_044`：
  - 从 `round_043` 加入 `quad_inset_abs_weight=2.0` 做定向修正
  - `best epoch=1`
  - holdout：`point_error_mean=0.0245`
  - holdout：`point_le_0_01_ratio=0.1343`
  - holdout：`max_corner_le_0_03_ratio=0.0746`
  - holdout：`avg_page_infer_ms=9.28`
  - 结论：anti-inset 有部分修复，但没有同时超过 `round_043` 的 strict-point 峰值
- `round_045`：
  - 从 `round_042 epoch_001` 回到更稳几何基线，再叠中等强度 `strict_point + anti-inset`
  - `best epoch=1`
  - holdout：`point_error_mean=0.0244`
  - holdout：`point_le_0_01_ratio=0.1306`
  - holdout：`max_corner_le_0_03_ratio=0.0746`
  - holdout：`avg_page_infer_ms=9.05`
  - 结论：比 `round_044` 的几何略稳、时延更低，但 strict-point 仍低于 `round_043`
- `round_046`：
  - 新增 `quad_inset_inward_weight`，只惩罚 inward shrink
  - `best epoch=1`
  - holdout：`point_error_mean=0.0244`
  - holdout：`point_le_0_01_ratio=0.1325`
  - holdout：`max_corner_le_0_03_ratio=0.0970`
  - holdout：`avg_page_infer_ms=9.06`
  - 结论：新 inward-only 约束有效，几何比 `round_045` 更稳，但 strict-point 仍低于 `round_043`
- `round_047`：
  - 从 `round_046` 继续叠轻量 `strict_point_manual_weight`
  - `best epoch=1`
  - holdout：`point_error_mean=0.0240`
  - holdout：`point_le_0_01_ratio=0.1213`
  - holdout：`max_corner_le_0_03_ratio=0.1119`
  - holdout：`avg_page_infer_ms=8.99`
  - 结论：几何和最差角点继续改善，时延继续下降，但主指标继续回吐
- `round_048`：
  - 首次启用 `residual_quad_head`
  - 从 `round_043` strict-point 最优 checkpoint 出发，只验证结构改造本身
  - `best epoch=1`
  - holdout：`point_error_mean=0.0233`
  - holdout：`point_le_0_01_ratio=0.0784`
  - holdout：`max_corner_le_0_03_ratio=0.1791`
  - holdout：`avg_page_infer_ms=9.21`
  - 结论：结构 head 对几何稳定极有效，但会明显压掉 strict-point
- `round_049`：
  - 从 `round_046` 稳定几何基线启用 `residual_quad_head + light strict-point`
  - `best epoch=1`
  - holdout：`point_error_mean=0.0228`
  - holdout：`point_le_0_01_ratio=0.0821`
  - holdout：`max_corner_le_0_03_ratio=0.1418`
  - holdout：`avg_page_infer_ms=9.33`
  - 结论：几何继续明显改善，但主指标仍然远低于 `round_043`
- `round_050`：
  - 把 `residual_quad_head` 改成保守 blend 结构，再从 `round_046` 稳定几何基线继续验证
  - `best epoch=1`
  - holdout：`point_error_mean=0.0228`
  - holdout：`point_le_0_01_ratio=0.0784`
  - holdout：`max_corner_le_0_03_ratio=0.1493`
  - holdout：`avg_page_infer_ms=9.61`
  - 结论：blend 结构比原始 residual 结构更稳，但仍没有把 strict-point 拉回可接受区间
- `round_051`：
  - 新增 full-ROI strict spatial head，尝试直接在 ROI 上重解码 strict quad
  - `best epoch=3`
  - holdout：`point_error_mean=0.0244`
  - holdout：`point_le_0_02_ratio=0.4030`
  - holdout：`point_le_0_01_ratio=0.0709`
  - holdout：`max_corner_le_0_03_ratio=0.1567`
  - holdout：`avg_page_infer_ms=9.53`
  - 结论：宽阈值和几何较 `round_043` 更稳，但 strict-point 明显塌陷，这条整块 ROI strict 重解码路线失败
- `round_052`：
  - strict spatial head 改为 per-corner patch residual，尝试更贴近 teacher 的局部角点搜索
  - `best epoch=3`
  - holdout：`point_error_mean=0.0370`
  - holdout：`point_le_0_02_ratio=0.0149`
  - holdout：`point_le_0_01_ratio=0.0224`
  - holdout：`max_corner_le_0_03_ratio=0.0075`
  - holdout：`avg_page_infer_ms=12.36`
  - 结论：局部 patch residual strict-head 进一步退化，说明当前 unified student 内继续堆 strict-head / strict-loss 已进入明确瓶颈
- `round_053`：
  - 新增 candidate selection head，只在 student 内部 `coarse / roi_stage / base_final` 三候选之间做排序
  - `best epoch=3`
  - holdout：`point_error_mean=0.0232`
  - holdout：`point_le_0_02_ratio=0.4328`
  - holdout：`point_le_0_01_ratio=0.0728`
  - holdout：`max_corner_le_0_03_ratio=0.1567`
  - holdout：`avg_page_infer_ms=9.36`
  - 结论：候选排序能明显改善几何和宽阈值命中，但 strict-point 仍远低于 `round_043`，说明只做 student 内部三候选选择还不够
- `round_054`：
  - 把 OpenCV `best candidate` 接入统一排序，验证 richer candidate pool
  - `best epoch=1`
  - holdout：`point_error_mean=0.0232`
  - holdout：`point_le_0_02_ratio=0.4552`
  - holdout：`point_le_0_01_ratio=0.0653`
  - holdout：`max_corner_le_0_03_ratio=0.1791`
  - holdout：`avg_page_infer_ms=9.47`
  - 结论：外部候选池继续改善几何和宽阈值，但 strict-point 进一步回退，说明“候选更丰富”本身仍不足以解决主指标
- `round_055`：
  - 冻结主干与 refine，只训练 `candidate_selection_head`
  - `best epoch=6`
  - holdout：`point_error_mean=0.0231`
  - holdout：`point_le_0_02_ratio=0.4478`
  - holdout：`point_le_0_01_ratio=0.0765`
  - holdout：`max_corner_le_0_03_ratio=0.1642`
  - holdout：`avg_page_infer_ms=6.80`
  - 结论：选择头定向微调只带来小幅恢复，仍无法学会稳定选到真正强的候选
- `round_056`：
  - 新增 `final_output_mode=coarse`，把最终输出直接锁到 `coarse_quad`
  - 只训练 coarse heads，验证 coarse-preserving 输出策略
  - `best epoch=2`
  - holdout：`point_error_mean=0.0097`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7127`
  - holdout：`max_corner_le_0_03_ratio=0.7612`
  - holdout：`avg_page_infer_ms=6.25`
  - 结论：保存件首次稳定越过 `70%` strict-point 目标，说明真正有效的是保留强 coarse 候选，而不是继续把结果交给 downstream refine
- `round_057`：
  - 在 `round_056` 上低学习率放开 shared backbone + coarse heads
  - `best epoch=2`
  - holdout：`point_error_mean=0.0096`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7127`
  - holdout：`max_corner_le_0_03_ratio=0.7687`
  - holdout：`avg_page_infer_ms=6.97`
  - 结论：主指标与 `round_056` 持平，只在几何与最差角点上小幅改善，说明 coarse-preserving 主线已开始接近平台
- `round_058`：
  - 基于 `round_057` 做离线候选拆解，确认 `base_final` 在 holdout 上仅 `3/134` 页 `point_error` 胜过 `coarse`
  - 训练上移除 `v28` 驱动的 `final_teacher / strict_point_teacher / max_corner_teacher` 冲突监督
  - 收紧为 `manual + r3` 主导的 coarse-only distillation
  - `best epoch=2`
  - holdout：`point_error_mean=0.0095`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7164`
  - holdout：`max_corner_le_0_03_ratio=0.7687`
  - holdout：`avg_page_infer_ms=6.62`
  - 结论：这证明 coarse 训练目标里确实存在冲突 teacher 信号；切掉后主指标还能再抬一小步
- `round_059`：
  - 保留 `round_058` 的 corrected coarse loss
  - 低学习率放开 shared backbone + FPN + coarse heads
  - `best epoch=2`
  - holdout：`point_error_mean=0.0095`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7164`
  - holdout：`max_corner_le_0_03_ratio=0.7836`
  - holdout：`avg_page_infer_ms=6.92`
  - 结论：主指标没有继续提升，只换来最差角点和几何的轻微改善；按当前规则可视为新的平台确认轮
- `round_060`：
  - 提高输入分辨率：`input_size 256 -> 320`，`output_size 64 -> 80`
  - 保持 corrected coarse loss，不回到旧 refine 线
  - `best epoch=2`
  - holdout：`point_error_mean=0.0111`
  - holdout：`point_le_0_02_ratio=0.8507`
  - holdout：`point_le_0_01_ratio=0.6754`
  - holdout：`max_corner_le_0_03_ratio=0.6493`
  - holdout：`avg_page_infer_ms=9.08`
  - 结论：更高输入分辨率会明显伤害 strict-point 和几何，说明当前瓶颈不是 `256` 输入上限
- `round_061`：
  - 回到 `256` 输入
  - 保持 corrected coarse output，只提高 `r3` 蒸馏强度
  - `best epoch=2`
  - holdout：`point_error_mean=0.0095`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7164`
  - holdout：`max_corner_le_0_03_ratio=0.7761`
  - holdout：`avg_page_infer_ms=6.56`
  - 结论：更强 `r3` 蒸馏只能打平 `round_058`，没有继续突破主指标
- `round_062`：
  - 在训练代码里新增 `r3` agreement gate
  - 只在 `r3` 与人工标注的逐样本点位误差足够小的时候，才启用 `r3` heatmap / quad 蒸馏
  - `best epoch=2`
  - holdout：`point_error_mean=0.0095`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7164`
  - holdout：`max_corner_le_0_03_ratio=0.7761`
  - holdout：`avg_page_infer_ms=6.61`
  - 结论：teacher-tail 噪声门控没有继续拉高主指标，当前平台仍然成立
- `round_063`：
  - 新增逐角点 `adaptive manual/r3` coarse target
  - 同时把训练采样从 `teacher_v28_quad` 难度切到 `teacher_r3_quad` 难度
  - `best epoch=2`
  - holdout：`point_error_mean=0.0095`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7090`
  - holdout：`max_corner_le_0_03_ratio=0.7761`
  - holdout：`avg_page_infer_ms=6.49`
  - 结论：adaptive blended coarse supervision 会直接压低主指标，这条新 coarse target 设计当前无效
- `round_064`：
  - 回到 `round_061` 的 corrected coarse loss
  - 只保留 `teacher_r3_quad` 难度采样，不再使用 adaptive blended target
  - `best epoch=2`
  - holdout：`point_error_mean=0.0095`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7090`
  - holdout：`max_corner_le_0_03_ratio=0.7761`
  - holdout：`avg_page_infer_ms=5.72`
  - 结论：把采样口径改成 `r3` 难度并不能解锁 `0.7164` 平台，只换来更低时延
- `round_065`：
  - 首次引入高可信 coarse 课程：训练集按 `teacher_r3_quad` 质量做过滤预热
  - 过滤阈值：`mean_point_error <= 0.008` 且四角都 `<= 0.012`
  - `best epoch=1`
  - holdout：`point_error_mean=0.0096`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7127`
  - holdout：`max_corner_le_0_03_ratio=0.7687`
  - holdout：`avg_page_infer_ms=5.73`
  - 结论：单独做高可信子集预热会回到 `round_056` 水平，不能直接突破主指标
- `round_066`：
  - 从 `round_065` 结果回全量数据继续收敛
  - `best epoch=2`
  - holdout：`point_error_mean=0.0095`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7164`
  - holdout：`max_corner_le_0_03_ratio=0.7836`
  - holdout：`avg_page_infer_ms=6.71`
  - 结论：两阶段课程能把主指标恢复到 `round_061` 水平，但还没有形成新的提升
- `round_067`：
  - 在 `round_066` 基础上低学习率放开 shared backbone + FPN + coarse heads
  - `best epoch=2`
  - holdout：`point_error_mean=0.0095`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7239`
  - holdout：`max_corner_le_0_03_ratio=0.7836`
  - holdout：`avg_page_infer_ms=6.89`
  - 结论：这是当前第一次真正突破 `0.7164` 平台，说明“高可信课程 + 低学习率 reopening”是有效方向
- `round_068`：
  - 从 `round_067` 回收到 coarse-head-only，做低学习率稳定收敛
  - `best epoch=2`
  - holdout：`point_error_mean=0.0095`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7239`
  - holdout：`max_corner_le_0_03_ratio=0.7836`
  - holdout：`avg_page_infer_ms=6.45`
  - 结论：保住 `round_067` 的 strict-point 提升，同时把时延压到更低，当前应视为新的 split_v2 最佳保存件
- `round_069`：
  - 从 `round_068` 做 1 个极短、极低学习率 replay，验证是否还能继续抬高主指标
  - `best epoch=1`
  - holdout：`point_error_mean=0.0093`
  - holdout：`point_le_0_02_ratio=0.9403`
  - holdout：`point_le_0_01_ratio=0.6978`
  - holdout：`max_corner_le_0_03_ratio=0.7537`
  - holdout：`avg_page_infer_ms=6.48`
  - 结论：短 replay 会明显伤害 strict-point，说明 `round_068` 已是这条课程线的局部峰值
- `round_070`：
  - 从 `round_068` 重新放开 shared backbone/FPN，并对 `low_contrast_scene`、`near_color_background` 做定向采样 boost
  - `best epoch=2`
  - holdout：`point_error_mean=0.0092`
  - holdout：`point_le_0_02_ratio=0.9254`
  - holdout：`point_le_0_01_ratio=0.7071`
  - holdout：`max_corner_le_0_03_ratio=0.7985`
  - holdout：`avg_page_infer_ms=6.72`
  - 结论：弱场景 boost 会把收益推到宽阈值几何，而不是 strict-point，不能作为当前主线
- `round_071`：
  - 从 `round_068` 做高层窄 reopen，只打开 `stage3 + lat2 + lat3 + coarse heads`
  - `best epoch=2`
  - holdout：`point_error_mean=0.0094`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7239`
  - holdout：`max_corner_le_0_03_ratio=0.7836`
  - holdout：`avg_page_infer_ms=6.67`
  - 结论：主指标与最差角点都只打平 `round_068`，时延略差，说明更窄 reopen 也没有新增益
- `round_072`：
  - 基于 holdout 归因结果，只保留高可信 `r3` 训练页并做 1 个 coarse-head-only exactness polish
  - `best epoch=1`
  - holdout：`point_error_mean=0.0103`
  - holdout：`point_le_0_02_ratio=0.8657`
  - holdout：`point_le_0_01_ratio=0.7127`
  - holdout：`max_corner_le_0_03_ratio=0.7687`
  - holdout：`avg_page_infer_ms=6.48`
  - 结论：easy-page exactness rehearse 会明显伤害 holdout 泛化，这条线确认失败
- `round_073`：
  - 把 adaptive coarse target 切到 `r3-priority` per-corner switch
  - `best epoch=2`
  - holdout：`point_error_mean=0.0096`
  - holdout：`point_le_0_02_ratio=0.8806`
  - holdout：`point_le_0_01_ratio=0.7201`
  - holdout：`max_corner_le_0_03_ratio=0.7910`
  - holdout：`avg_page_infer_ms=6.49`
  - 结论：接近打平，但 strict-point 仍略低于 `round_068`，单独使用这条 target 不足以形成新 best
- `round_074`：
  - 新增 `oracle(r3,v28)` per-corner coarse target，只训练 coarse heads
  - `best epoch=2`
  - holdout：`point_error_mean=0.0095`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7239`
  - holdout：`max_corner_le_0_03_ratio=0.7836`
  - holdout：`avg_page_infer_ms=6.27`
  - 结论：主指标打平 `round_068`，但时延更低，说明新的 coarse teacher target 设计是有效的
- `round_075`：
  - 在 `round_074` 基础上叠高层 reopen，验证 oracle teacher target 是否还能继续抬主指标
  - `best epoch=2`
  - holdout：`point_error_mean=0.0094`
  - holdout：`point_le_0_02_ratio=0.8881`
  - holdout：`point_le_0_01_ratio=0.7239`
  - holdout：`max_corner_le_0_03_ratio=0.7836`
  - holdout：`avg_page_infer_ms=6.24`
  - 结论：主指标继续打平，但平均误差和时延继续下降，当前可视为同主指标下更优的 split_v2 保存件

当前对 `split_v2` 线的阶段结论：

- 如果只盯 `point_le_0_01_ratio + avg_page_infer_ms`，当前 `split_v2` 最好应更新为 `round_075 = 0.7239 @ 6.24ms`
- `round_043` 这种旧 strict-point 拉升方式已经被 coarse-preserving 主线彻底超越，而且它本身还伴随明显 inward shrink
- `round_044` 证明只靠一层 `quad_inset_abs_weight` 还不够把几何质量和 strict-point 同时守住
- `round_045` 进一步证明：即使回到更稳的 `round_042 epoch_001` 基线，再加温和 `strict_point + anti-inset`，也只能落在 `round_043` 和 `round_044` 之间
- `round_046` 与 `round_047` 进一步证明：
  - 新的 inward-only 正则比旧的 `quad_inset_abs` 更合理
  - 但当前 shared-backbone + local-MoE 结构下，只靠 loss 级别修正，仍然是在“strict-point”与“几何稳定”之间做 trade-off
- `round_048` 与 `round_049` 进一步证明：
  - 新的 `residual_quad_head` 结构方向是有效的，因为它能系统性改善 `point_error_mean / point_le_0_02_ratio / max_corner_le_0_03_ratio / quad_inset_ratio_mean`
  - 但当前接法会明显牺牲 `point_le_0_01_ratio`
  - 也就是说，当前结构改造还没有解决“strict-point 与几何稳定并存”的问题，只是把平衡点进一步往几何稳定一侧推
- `round_050` 进一步确认：
  - 即使把 residual 结构改成更保守的 blend 形式，主指标仍然卡在 `0.08` 左右
  - 这说明当前结构线也已经进入平台，不适合继续做邻域微调
- `round_051` 与 `round_052` 进一步确认：
  - 不管是 full-ROI strict spatial head，还是 per-corner patch residual strict-head，都没有把 `point_le_0_01_ratio` 拉回 `round_043`
  - 说明问题不再是“再换一种局部 strict head 就行”，而是当前 unified student 的 supervision / decode 表述本身不对
- `round_053` 进一步确认：
  - 即使把 student 改成“内部多候选 + 排序”，如果候选集合本身仍然只来自 student 内部三阶段输出，strict-point 依然起不来
  - 这说明下一步需要更贴近 teacher 的候选来源，而不是只在 student 现有输出之间重排
- `round_054` 与 `round_055` 进一步确认：
  - richer candidate pool 和 head-only selector tuning 都没有把最终输出拉回 strict-point 目标区间
  - 但对候选拆解后发现，student 自己的 `coarse_quad` 实际上已经在 holdout 上达到约 `0.7034` 到 `0.7052` 的 `point_le_0_01_ratio`
  - 当前真正的瓶颈不再是“coarse 点位学不会”，而是“最终输出链路把已经很强的 coarse 候选覆盖掉了”
- `round_056` 到 `round_075` 进一步确认：
  - 一旦 runtime / 训练都切到 coarse-preserving 输出策略，保存件会直接进入 `0.7127` 的 strict-point 新区间
  - 仅切输出模式还不够，coarse loss 里混入 `v28` teacher 监督也会拖偏训练方向
  - `round_058` 证明把 coarse 蒸馏重新收紧到 `manual + r3` 后，主指标还能从 `0.7127` 抬到 `0.7164`
  - `round_059` 证明 corrected coarse loss 下再放开 backbone/FPN，主指标不再前进
  - `round_060` 证明更高输入分辨率不是当前主瓶颈
  - `round_061` 证明更强 `r3` 蒸馏也不能继续把主指标推过 `0.7164`
  - `round_062` 证明简单的 `r3` 可靠性门控也不能继续把主指标推过 `0.7164`
  - `round_063` 证明逐角点 adaptive `manual/r3` blended target 会直接拖低 strict-point
  - `round_064` 证明单独把难例采样切到 `r3` 口径也不能继续推高主指标
  - `round_065` 到 `round_066` 证明：高可信 coarse 课程本身不会直接提升主指标，但可以作为后续 reopening 的准备步骤
  - `round_067` 证明：在课程线之后再低学习率放开 backbone/FPN，能够把主指标从 `0.7164` 推到 `0.7239`
  - `round_068` 证明：把 `round_067` 的收益回收到 coarse heads 后，能在保住 `0.7239` 的同时把时延压低到 `6.45 ms`
  - `round_069` 证明：继续做极短 replay 会重新伤害 strict-point
  - `round_070` 证明：弱场景 boost 只会把收益推向宽阈值几何
  - `round_071` 证明：更窄的高层 reopen 也不能带来新的 strict-point 增益
  - `round_072` 证明：高可信 `r3` easy-page polish 会重新伤害 holdout 泛化
  - `round_073` 证明：单独把 target 切到 `r3-priority switch` 还不能直接超过当前 best
  - `round_074` 证明：新的 `oracle(r3,v28)` coarse teacher target 是有效的，因为它在不伤 strict-point 的前提下继续压低时延
  - `round_075` 证明：oracle teacher target 再叠高层 reopen 后，主指标仍未继续上升，但可以把当前同主指标下的保存件进一步压到更低时延
- 因此 `split_v2` 线上当前已进入新的 trade-off 瓶颈：
  - 旧的 local-refine / selector 主线已经确认失败
  - 新的 coarse-preserving 主线已经再次被推高，并且当前最佳保存件更新为 `round_075`
  - `round_067`、`round_068` 已经打破旧平台，`round_069` 到 `round_073` 说明旧邻域微调基本失效，但 `round_074`、`round_075` 说明新的 coarse teacher target 仍有价值
  - 下一步不应回到旧的 local-refine / selector 邻域继续同类训练；应以 `oracle teacher target` 为新基线，继续尝试更大的 coarse-only supervision / residual 方案

`round_002` 结果：

- student holdout：`point_error_mean=0.0374`
- student holdout：`point_le_0_05_ratio=0.7039`
- student holdout：`point_le_0_01_ratio=0.0`
- student 平均单页推理：`9.05 ms`
- 结论：validation 改善，但 holdout 退化，说明 local refine 泛化仍不稳

`round_003` 结果：

- student holdout：`point_error_mean=0.0280`
- student holdout：`point_le_0_05_ratio=0.8947`
- student holdout：`point_le_0_03_ratio=0.6579`
- student holdout：`point_le_0_01_ratio=0.0`
- student 平均单页推理：`8.82 ms`
- 结论：当前是重置后最优 round，但距离最终目标仍有明显差距

`round_004` 结果：

- student holdout：`point_error_mean=0.0261`
- student holdout：`point_le_0_05_ratio=0.9474`
- student holdout：`point_le_0_03_ratio=0.7368`
- student holdout：`point_le_0_02_ratio=0.3750`
- student holdout：`point_le_0_01_ratio=0.0`
- student 平均单页推理：`8.84 ms`
- 结论：当前保存结果继续改善，但同轮 `epoch 3` 的 holdout 已优于最终保存的 `epoch 4`

`round_005` 到 `round_008` 的阶段结果：

- `round_005`：
  - `point_error_mean=0.0271`
  - 结论：简单收紧 epoch 数无效
- `round_006`：
  - `point_error_mean=0.0233`
  - `point_le_0_05_ratio=0.9737`
  - `point_le_0_03_ratio=0.8355`
  - `point_le_0_02_ratio=0.4145`
  - 结论：当前最佳保存轮次
- `round_007`：
  - `point_error_mean=0.0235`
  - `point_le_0_05_ratio=0.9803`
  - `point_le_0_03_ratio=0.8618`
  - 结论：宽阈值命中率有改善，但主指标未继续下降
- `round_008`：
  - `point_error_mean=0.0236`
  - `point_le_0_05_ratio=0.9803`
  - `point_le_0_03_ratio=0.8618`
  - 结论：继续缩小学习率仍未压低主指标
- `round_009`：
  - `point_error_mean=0.0244`
  - `point_le_0_05_ratio=0.9803`
  - `point_le_0_03_ratio=0.7697`
  - `point_le_0_02_ratio=0.4145`
  - `selection_metric=point_error_mean`
  - 结论：新 checkpoint 逻辑与更大 ROI context 已验证，但仍未超过 `round_006`
- `round_010`：
  - `point_error_mean=0.0245`
  - `point_le_0_05_ratio=0.9737`
  - `point_le_0_03_ratio=0.8158`
  - `point_le_0_02_ratio=0.3882`
  - `selection_metric=point_error_mean`
  - 结论：轻量数据增强线已验证，但仍未超过 `round_006`
- `round_011`：
  - `point_error_mean=0.0222`
  - `point_le_0_05_ratio=1.0000`
  - `point_le_0_03_ratio=0.9145`
  - `point_le_0_02_ratio=0.4079`
  - `selection_metric=point_error_mean`
  - 结论：更强几何增强首次打破旧平台，当前最佳轮次更新到 `round_011`
- `round_012`：
  - `point_error_mean=0.0232`
  - `point_le_0_05_ratio=0.9868`
  - `point_le_0_03_ratio=0.8684`
  - `point_le_0_02_ratio=0.4079`
  - `selection_metric=point_error_mean`
  - 结论：从 `round_011` 继续以更低学习率和更软增强热启动，没有延续收益
- `round_013`：
  - `point_error_mean=0.0231`
  - `point_le_0_05_ratio=0.9868`
  - `point_le_0_03_ratio=0.8816`
  - `point_le_0_02_ratio=0.3882`
  - `selection_metric=point_error_mean`
  - 结论：回到 `round_006` 锚点并继续加重 perspective 扰动，仍未超过 `round_011`
- `round_014`：
  - `point_error_mean=0.0224`
  - `point_le_0_05_ratio=0.9868`
  - `point_le_0_03_ratio=0.9079`
  - `point_le_0_02_ratio=0.4013`
  - `selection_metric=point_error_mean`
  - 结论：loss reweight 线已明显逼近 `round_011`，同轮 `epoch 2` holdout 到 `0.0221`，但正式保存结果仍未稳定刷新最优
- `round_015`：
  - `point_error_mean=0.0229`
  - `point_le_0_05_ratio=0.9868`
  - `point_le_0_03_ratio=0.8816`
  - `point_le_0_02_ratio=0.3816`
  - `selection_metric=point_le_0_03_ratio`
  - 结论：从 `round_014 epoch_002` 热启动并调整 checkpoint 口径后，仍未复现 `0.0221` 的潜在峰值
- `round_016`：
  - `point_error_mean=0.0220`
  - `point_le_0_05_ratio=0.9868`
  - `point_le_0_03_ratio=0.9211`
  - `point_le_0_02_ratio=0.3947`
  - `selection_metric=point_error_mean`
  - 结论：`loss reweight + hard-example sampling` 首次带来新的正式最优保存轮次，当前最佳更新到 `round_016`
- `round_017`：
  - `point_error_mean=0.0226`
  - `point_le_0_05_ratio=0.9737`
  - `point_le_0_03_ratio=0.8618`
  - `point_le_0_02_ratio=0.4013`
  - `selection_metric=point_error_mean`
  - 结论：继续提高采样强度后正式保存结果回退，说明当前 hard-example 线已接近新的局部瓶颈
- `round_018`：
  - `point_error_mean=0.2973`
  - `point_le_0_05_ratio=0.0000`
  - `point_le_0_03_ratio=0.0000`
  - `point_le_0_02_ratio=0.0000`
  - `selection_metric=point_error_mean`
  - 结论：首版 ROI adapter 直接插入热启动模型后失稳，原始随机初始化结构应判定为失败实验，不可继续沿用
- `round_019`：
  - `point_error_mean=0.0210`
  - `point_le_0_05_ratio=0.9934`
  - `point_le_0_03_ratio=0.9539`
  - `point_le_0_02_ratio=0.4737`
  - `selection_metric=point_error_mean`
  - 结论：残差式零初始化 ROI adapter 成功恢复稳定热启动，并刷新当前最佳保存轮次
- `round_020`：
  - `point_error_mean=0.0217`
  - `point_le_0_05_ratio=0.9868`
  - `point_le_0_03_ratio=0.9474`
  - `point_le_0_02_ratio=0.4342`
  - `selection_metric=point_error_mean`
  - 结论：从 `round_019` 继续降低学习率后没有继续改善，说明当前 ROI adapter 线已接近新的局部瓶颈
- `round_021`：
  - `point_error_mean=0.0219`
  - `point_le_0_05_ratio=0.9737`
  - `point_le_0_03_ratio=0.9474`
  - `point_le_0_02_ratio=0.4276`
  - `selection_metric=point_le_0_02_ratio`
  - 结论：precision-oriented follow-up 没有带来收益，说明仅靠 loss / sampling / checkpoint 联动已无法继续推动当前结构线
- `round_022`：
  - 口径切换：`point_le_0_01_ratio` 改为逐角点 `< 0.01` 命中率
  - 新增：`max_corner_le_0_03_ratio`
  - holdout：`point_error_mean=0.0213`
  - holdout：`point_le_0_01_ratio=0.0905`
  - holdout：`max_corner_le_0_03_ratio=0.2039`
  - `selection_metric=max_corner_le_0_03_ratio`
  - 结论：strict-corner 训练线首次把最差角点命中率推过 `20%`
- `round_023`：
  - holdout：`point_error_mean=0.0210`
  - holdout：`point_le_0_01_ratio=0.0822`
  - holdout：`max_corner_le_0_03_ratio=0.1908`
  - `selection_metric=max_corner_le_0_03_ratio`
  - 结论：低学习率延长没有继续抬升 strict-corner 指标，说明 `round_022` 已是当前新口径下的局部峰值
- `round_024`：
  - holdout：`point_error_mean=0.0226`
  - holdout：`point_le_0_01_ratio=0.0921`
  - holdout：`max_corner_le_0_03_ratio=0.1382`
  - `selection_metric=max_corner_le_0_03_ratio`
  - 结论：spatial residual refine 结构尝试显著拉低 strict-corner 主指标，应判定为失败结构线
- `round_025`：
  - holdout：`point_error_mean=0.0219`
  - holdout：`point_le_0_05_ratio=0.9803`
  - holdout：`point_le_0_01_ratio=0.1217`
  - holdout：`max_corner_le_0_03_ratio=0.1118`
  - holdout：`quad_inset_ratio_mean=0.0177`
  - `selection_metric=max_corner_le_0_03_ratio`
  - 结论：显式 `max_corner` loss 能推高逐点 `<1%` 命中，但会把 quad 往内缩，导致整页最差角点命中率明显恶化
- `round_026`：
  - holdout：`point_error_mean=0.0210`
  - holdout：`point_le_0_05_ratio=0.9803`
  - holdout：`point_le_0_01_ratio=0.0839`
  - holdout：`max_corner_le_0_03_ratio=0.1974`
  - holdout：`quad_inset_ratio_mean=-0.0048`
  - `selection_metric=max_corner_le_0_03_ratio`
  - 结论：降低 strict-corner loss 并加上 inset 约束后，内缩偏置被修正，但最终仍未超过 `round_022`
- `round_027`：
  - holdout：`point_error_mean=0.0212`
  - holdout：`point_le_0_05_ratio=0.9934`
  - holdout：`point_le_0_01_ratio=0.0789`
  - holdout：`max_corner_le_0_03_ratio=0.1645`
  - `selection_metric=max_corner_le_0_03_ratio`
  - 结论：切换到“只看 max-corner 的评判口径”并收软 hard-example 采样后，没有延续 `round_022` 的 strict-corner 收益
- `round_028`：
  - 训练目标切回：`point_le_0_01_ratio`
  - 保留硬门槛：`avg_page_infer_ms <= 500`
  - holdout：`point_error_mean=0.0217`
  - holdout：`point_le_0_05_ratio=0.9868`
  - holdout：`point_le_0_01_ratio=0.0839`
  - holdout：`max_corner_le_0_03_ratio=0.2105`
  - holdout：`avg_page_infer_ms=10.4`
  - `selection_metric=point_le_0_01_ratio`
  - 结论：strict-point 评判口径已恢复，但以 `round_025` 热启动并追加 strict-point loss / sampling 的方案没有守住 `round_025` 的逐点 `<1%` 高点
- `round_029`：
  - 首次引入 scene-aware MoE 路由与 scene tag 监督
  - holdout：`point_error_mean=0.0280`
  - holdout：`point_le_0_01_ratio=0.0905`
  - holdout：`max_corner_le_0_03_ratio=0.1118`
  - holdout：`avg_page_infer_ms=10.97`
  - `selection_metric=point_le_0_01_ratio`
  - 结论：首次 scene-aware 版本暴露出 router / expert 热启动不兼容，前两轮训练几乎崩掉，最终未超过 `round_025`
- `round_030`：
  - 场景注入改为 residual adapter，但仍保留了错误的 hidden width
  - holdout：`point_error_mean=0.0506`
  - holdout：`point_le_0_01_ratio=0.0280`
  - holdout：`max_corner_le_0_03_ratio=0.0000`
  - holdout：`avg_page_infer_ms=11.89`
  - `selection_metric=point_le_0_01_ratio`
  - 结论：虽然 scene-aware 输入已兼容，但 hidden width 仍导致旧 router/expert 无法完整热启动，这轮判定为失败
- `round_031`：
  - scene-aware 路由修正为兼容旧 router/expert 全量热启动，scene 只通过零初始化残差适配注入
  - holdout：`point_error_mean=0.0219`
  - holdout：`point_le_0_01_ratio=0.0872`
  - holdout：`max_corner_le_0_03_ratio=0.1776`
  - holdout：`avg_page_infer_ms=9.63`
  - `selection_metric=point_le_0_01_ratio`
  - 结论：兼容热启动修复后恢复稳定，但仍没有超过 `round_029=0.0905`，更没有接近 `round_025=0.1217`

当前阶段判断：

- 旧口径平均误差最佳保存轮次：`round_019`
- 新口径 strict-corner 最佳保存轮次：`round_058`
- `round_012` 与 `round_013` 已分别验证 softer geometry follow-up 和 stronger geometry continuation
- 两轮都未超过 `round_011`，说明当前几何增强主线已在 `round_011` 附近进入新的阶段瓶颈
- `round_014` 与 `round_015` 说明 loss reweight 线存在正向信号，但当前仍不稳定，尚不足以替代 `round_011`
- `round_016` 证明 hard-example sampling 是当前阶段最有效的新策略
- `round_017` 没有继续延长这条收益曲线，当前可视为在 `round_016` 附近进入新的局部瓶颈
- `round_018` 证明结构改动不能直接随机接入热启动模型，否则会导致灾难性退化
- `round_019` 证明残差式 ROI adapter 是当前阶段最有效的新结构策略
- `round_020` 没能延续 `round_019` 的收益，表明当前结构线已进入新的平台区
- `round_021` 进一步确认当前平台并非单纯学习率问题，精细阈值导向微调也没有突破 `round_019`
- `round_022` 证明切换到 strict-corner 采样和 checkpoint 口径后，最差角点指标仍有可见提升
- `round_023` 说明这条 strict-corner 热启动延长线也已出现第一次明确回落
- `round_024` 证明新增 spatial residual refine 分支不是当前有效方向
- `round_025` 与 `round_026` 证明显式 strict-corner 监督存在可学信号，但当前这条 loss 线会在逐点精度和整页最差角点之间形成明显拉扯，尚未越过 `round_022`
- 当前可判定：基于 `round_022` 的同结构监督微调已经进入新的清晰瓶颈，下一阶段不应继续做同类 loss 微调
- `round_027` 进一步说明：即使把 `point_le_0_01_ratio` 降级为纯观测指标，当前这条同结构训练线仍未自动恢复 strict-corner 增益
- `round_028` 进一步说明：把 `point_le_0_01_ratio` 恢复为主评判指标是必要的，但当前 strict-point continuation 配方仍会把 `round_025` 的高点重新拉回平台区
- `round_029` 到 `round_031` 说明：场景识别 + MoE 适配这条线在工程上已经跑通，但即使修复兼容热启动，当前主线也只回到 `0.0872`，没有提供超过 strict-point 旧高点的新增益

## `2026-03-26` 新数据 manual-only continuation

目标：

- 引入 `202603-awe`
- 引入 `202603-guochenghao-yangzhou`
- 新数据训练阶段不使用 teacher 监督，只保留人工标注监督
- teacher 仅保留 frozen 导出，作为验证参考

本阶段关键 round：

- `round_076`：
  - 方案：仅新数据、全量 `manual_only`
  - 新数据 holdout：`point_le_0_01_ratio=0.5000`
  - 新数据 holdout：`avg_page_infer_ms=7.64`
  - 旧 `split_v2` holdout 复测：`point_le_0_01_ratio=0.6810`
  - 结论：新模式学到了，但旧域遗忘明显
- `round_079`：
  - 方案：延续 `round_076`，只对新数据做 `scene/border-contact` 难例加权
  - 新数据 holdout：`point_le_0_01_ratio=0.5000`
  - 新数据 holdout：`avg_page_infer_ms=6.91`
  - 旧 `split_v2` holdout 复测：`point_le_0_01_ratio=0.6810`
  - 结论：几何和时延更稳，但主指标没有超过 `round_076`
- `round_080`：
  - 方案：从 `round_079` 出发，加入旧主线 `manual_only` 低权重 replay，只训练 coarse heads
  - 新数据 holdout：`point_le_0_01_ratio=0.4125`
  - 新数据 holdout：`avg_page_infer_ms=7.01`
  - 旧 `split_v2` holdout 复测：`point_le_0_01_ratio=0.6978`
  - 结论：旧域部分恢复，但新域再次明显回落

当前结论：

- 当前 `manual_only` continuation 家族已经形成新的明确 trade-off：
  - 要保住新数据 `0.5000`，旧 `split_v2` 只能停在 `0.6810`
  - 要把旧 `split_v2` 拉回到 `0.6978` 附近，新数据会掉回 `0.4125`
- 因此截至 `round_080`，当前家族可判定进入瓶颈，不应继续沿同一配方族做小幅微调
- 这一阶段的新数据最佳保存件应视为 `round_076`
- 当前全局默认运行时最佳保存件仍然是 `round_075`

后续又验证了 `coarse scene adapter` 新家族：

- `round_081`：
  - 方案：新增 zero-init `coarse_scene_adapter`，从 `round_075` 出发，带 old low-weight replay
  - 新数据 holdout：`point_le_0_01_ratio=0.4250`
  - 旧 `split_v2` holdout 复测：`point_le_0_01_ratio=0.7108`
  - 结论：能更好保住旧域，但明显伤害新域 strict-point
- `round_082`：
  - 方案：`coarse_scene_adapter` + new-only continuation，从 `round_075` 出发
  - 新数据 holdout：`point_le_0_01_ratio=0.5000`
  - 旧 `split_v2` holdout 复测：`point_le_0_01_ratio=0.6866`
  - 结论：只能打平 `round_076`，没有形成新增益
- `round_083`：
  - 方案：`coarse_scene_adapter` + new-only continuation，从 `round_076` 新域最佳件出发
  - 新数据 holdout：`point_le_0_01_ratio=0.4750`
  - 新数据 val：`point_le_0_01_ratio=0.5114`
  - 旧 `split_v2` holdout 复测：`point_le_0_01_ratio=0.6847`
  - 结论：新域 val 有轻微提升，但 holdout 没有突破 `0.5000`，不能判定为有效突破
- `round_084`：
  - 方案：在 `coarse_scene_adapter` 基础上新增 `border_contact_scene` 显式标签，并对这类样本做轻量 boost
  - 新数据 holdout：`point_le_0_01_ratio=0.4875`
  - 新数据 val：`point_le_0_01_ratio=0.5227`
  - 旧 `split_v2` holdout 复测：`point_le_0_01_ratio=0.6828`
  - 结论：显式贴边标签提升了 new val，但 new holdout 仍未超过 `0.5000`
- `round_085`：
  - 方案：把 `visibility/fallback` 接入 coarse 输出链路，新增 `visibility_refined_quad`
  - 新数据 holdout：`point_le_0_01_ratio=0.4875`，但 epoch_004 曾短暂回到 `0.5000`
  - 新数据 val：`point_le_0_01_ratio=0.5227`
  - 结论：这条路不是完全无效，但当时的 hard gate 让 `visibility_refine_gate` 实际始终为 `0`
- `round_086`：
  - 方案：从 `round_085 epoch_004` 出发，仅训练 `process_head`
  - 新数据 holdout：`point_le_0_01_ratio=0.4875`
  - 新数据 val：`point_le_0_01_ratio=0.5227`
  - 结论：冻结主干后仍无突破，说明问题不只是 backbone 扰动
- `round_087`：
  - 方案：把 gate 改成可导 soft gate，从 `round_076` 重开，只训练 `scene_context_head + process_head`
  - 新数据 holdout：`point_le_0_01_ratio=0.4875`
  - 新数据 val：`point_le_0_01_ratio=0.5227`
  - 结论：gate 已经真正打开，但几乎恒定在 `0.25`，`visibility/fallback` 头没有学出有区分度的修正
- `round_088`：
  - 方案：manual-only 样本启用 `manual_process_targets_for_manual_only`，直接用人工标注监督 `visibility/fallback`
  - 新数据 holdout：`point_le_0_01_ratio=0.4875`
  - 新数据 val：`point_le_0_01_ratio=0.5227`
  - 结论：即使加入 manual process supervision，这条 visibility-aware coarse refine 家族仍未突破 `0.5000`
- `round_089`：
  - 方案：修复 manual-only process structure mask，继续 `visibility-aware coarse refine`
  - 新数据 holdout：`point_le_0_01_ratio=0.4875`
  - 新数据 val：`point_le_0_01_ratio=0.5227`
  - 结论：mask 修复后过程分支仍近似常数，说明这一 family 的真实问题是“缺少有效修正信号”，不是单纯 wiring bug
- `round_090`：
  - 方案：新增 `state-aware candidate`，并把 `opencv / coarse / state-aware / base_final` 接进统一 selector
  - 新数据 holdout：`point_le_0_01_ratio=0.0625`
  - 新数据 val：`point_le_0_01_ratio=0.0227`
  - 结论：selector 直接塌到默认偏置，始终选回最差 `base_final`，这一实现方式不可继续
- `round_091`：
  - 方案：绕开 selector，直接让 `state-aware` 头作为最终输出
  - 新数据 holdout：`point_le_0_01_ratio=0.5000`
  - 新数据 holdout：`avg_page_infer_ms=6.93`
  - 结论：修复死分支后，这个候选本身只能打平 `round_076`，未形成 strict-point 新增益，但时延更低
- `round_092`：
  - 方案：对 `state-aware` 头改成 strict-oriented loss，并放开 `stage3/lat2/lat3` 做高层 reopen
  - 新数据 holdout：`point_le_0_01_ratio=0.4625`
  - 新数据 holdout：`point_error_mean=0.0252`
  - 新数据 holdout：`max_corner_le_0_03_ratio=0.4500`
  - 结论：这条线主要改善的是平均误差和几何，不是 `<0.01` strict-point 命中
- `round_093`：
  - 方案：只保留 `opencv + coarse` 二选一候选池，训练 `candidate_selection_head`
  - 新数据 holdout：`point_le_0_01_ratio=0.4750`
  - 新数据 holdout：`avg_page_infer_ms=7.13`
  - 补充诊断：training split 中 `opencv` 实际有 `60/165` 页优于 `coarse`，holdout 也有 `3/20` 页优于 `coarse`
  - 补充诊断：但最终 selector 在 holdout `20/20` 仍全部选回 `coarse`
  - 结论：当前 selector 训练强度下，这条路连已有 oracle 空间都无法兑现
- `round_094`：
  - 方案：保持 `opencv + coarse` 二选一不变，仅把 selector 学习率和 epoch 数拉高做高强度复验
  - 新数据 val：`point_le_0_01_ratio=0.5795`
  - 新数据 holdout：`point_le_0_01_ratio=0.4375`
  - 新数据 holdout：`avg_page_infer_ms=7.27`
  - 补充诊断：selector 已经学动，holdout 上有 `4/20` 页切到了 `opencv`
  - 结论：这条线出现明显 `val -> holdout` 错位，继续训练只会放大过拟合，不会稳定提升 strict-point
- `round_095`：
  - 方案：彻底回到 `coarse` 主输出，对新数据 train manifest 里的贴边/疑似越界 hard-case 直接写入 `sample_weight_multiplier`
  - 加权摘要：train `165` 页中，`103` 页 `min_border<=0.03`，`121` 页权重 `>=2x`
  - 新数据 holdout：`point_le_0_01_ratio=0.4750`
  - 新数据 holdout：`point_error_mean=0.0267`
  - 新数据 holdout：`avg_page_infer_ms=7.15`
  - 结论：hard-case weighting 明显改善几何和宽阈值指标，但 strict-point 仍未超过 `0.5000`
- `round_096`：
  - 方案：从 `round_095 epoch_001` 做短 replay，收软增强，专门保 early strict-point 信号
  - 新数据 holdout：`point_le_0_01_ratio=0.5000`
  - 新数据 holdout：`point_error_mean=0.0274`
  - 新数据 holdout：`avg_page_infer_ms=6.67`
  - 结论：把 weighted coarse 线重新拉回 `0.5000`，并且时延优于 `round_091 = 6.93ms`，当前新数据线同分条件下的最优保存件可更新为 `round_096`

因此到 `round_096` 为止，`manual-only new-data continuation`、`coarse scene adapter`、`border_contact_scene` 显式标签、`visibility-aware coarse refine`、`state-aware candidate`、`opencv + coarse` selector、以及 `coarse hard-case weighting` 这七条线都没有打破当前新数据 strict-point 上限 `0.5000`。当前可以正式判定：这一阶段已经进入新的结构瓶颈，不能再靠同家族邻域微调继续推进。

补充判断：

- `round_091` 与 `round_076` 在新数据 holdout `point_le_0_01_ratio` 上打平，但 `round_091` 时延更低，因此当前新数据 strict-point 持平条件下的更优保存件可更新为 `round_091`
- `round_092` 说明：若继续沿 `state-aware` family 推进，收益会优先落到 `point_error_mean / max_corner_le_0_03_ratio`，而不是当前最重要的 strict-point
- `round_093` 说明：当前 `opencv + coarse` selector 的第一性问题不是“候选池里没有更优页”，而是 head 在现有训练强度下根本学不跨初始化偏置
- `round_094` 说明：即使强行把 selector 学动，当前 val 口径也无法把这种路由泛化正确筛出来，说明这条路已经不只是优化问题，而是验证与目标表述一起失配
- `round_095` 说明：当前新数据的主要可学习增益仍然先体现在几何稳定性，而不是 `<0.01` strict-point，本轮把 hard-case 权重抬高后，改善的是 `point_error_mean / point_le_0_03_ratio / point_le_0_02_ratio`
- `round_096` 说明：weighted coarse family 的最优使用方式是“短 replay 保 early strict-point”，而不是长训；但即使这样，也只能把 strict-point 拉回 `0.5000`，不能形成新的台阶
- `round_100` 说明：当前 manual-only 新数据线唯一被验证有效的新增益，仍然是基于 student 自身误差做一次适度 self-hard-mining replay；这把 holdout `point_le_0_01_ratio` 从 `0.5000` 推到了 `0.5250`
- `round_101` 到 `round_111` 说明：在真实 `round_100 epoch_001` 基线上继续沿 corrected self-hard-mining、curriculum、process/gated refine、`coarse -> local strict patch refine`、checkpoint soup、`EMA` 这些方向推进，都没有超过 `0.5250`
- `round_112` 说明：把 `<1%` 命中率直接改写成 `soft strict-hit` supervision 后，holdout strict-point 仍回退到 `0.4875`
- `round_113` 说明：新增零初始化 `coarse_residual_head` 并只训练这个 coarse-side 小残差头，holdout strict-point 依旧只有 `0.4875`
- `round_114` 说明：把 old split_v1 的 train replay 收紧到“非极端贴边”子集后，holdout strict-point 进一步掉到 `0.4375`，说明旧口径上的分布修正也没有打开新空间
- `round_115` 说明：旧新数据 holdout 完全没有 `202603-awe`，因此新建 `split_v2_balanced_eval` 是必要的；但在新口径下直接叠 `dataset boost` 后，mixed holdout 只有 `0.4643`，`202603-guochenghao-yangzhou` 子集也没有收益
- `round_116` 说明：在相同 balanced-eval split 下去掉 `dataset boost` 后，mixed holdout 恢复到 `0.4762`，`yangzhou-only holdout = 0.4875`，同时 `AWE` 单页 holdout 仍保有可用输出；因此当前 balanced-eval 线的更优对照应更新为 `round_116`
- `round_117` 说明：把 balanced-eval 线也切到 student 自举 `self-hard-mining replay` 后，mixed holdout 仍只打平 `0.4762`，而且时延升到 `7.15ms`
- `round_118` 说明：把 balanced-eval 线切到 `EMA` 动态平滑后，mixed holdout 仍只打平 `0.4762`，时延也没有优于 `round_116`
- 因此当前 manual-only 新数据 continuation 可以进一步正式判定为：旧 split_v1 上不仅采样/selector/process/loss/head 家族已经见顶，连 non-border replay 也失败；新的 balanced-eval 口径已经落地，但 `round_116` 到 `round_118` 三轮没有形成新的 strict-point 台阶。旧口径最佳保存件仍是 `round_100 epoch_001`，新 balanced-eval 口径当前最佳对照仍是 `round_116`

## 当前关键问题

1. coarse-preserving 主线已经满足当前硬门槛：
   - 平均点位偏差 `< 0.5%`
   - `point_le_0_01_ratio > 70%`
   - 单张识别耗时 `< 500 ms`
   - 当前最佳保存件：`round_068`
2. 当前更高优先级的问题已经切换为：
   - 如何在保持 `0.7239` strict-point 的前提下继续逼近 teacher
   - 是否可以在 runtime 上进一步短路无效 refine，继续降低耗时
3. local refine MoE 旧主线已经确认不是当前默认输出的正确方向：
   - `round_043` 到 `round_055` 证明 downstream refine / selector 会覆盖掉更强的 coarse 候选
   - 若继续回到这条线，需要新的问题表述，而不是继续邻域微调
4. teacher 当前仍遵循“优先利用角落 T / L 结构，识别不到时再退回边缘推算”的两阶段运行时路径；student 训练应尽量逼近这条路径的逐点精度，而不是只做宽松几何拟合。
5. 训练与验证必须持续保持严格拆分。
6. 若 coarse-preserving 主线连续 `3` 轮都不能把 `point_le_0_01_ratio` 再提高至少 `+0.01`，则这条新主线也应判定为进入瓶颈。
7. 当前 `round_056` 到 `round_069` 已证明：
   - 通过重构最终输出策略，可以直接跨过历史 `split_v2` 瓶颈
   - `round_058` 还能通过清理冲突监督把主指标抬到 `0.7164`
   - `round_059`、`round_060`、`round_061`、`round_062` 说明：当前 corrected coarse line 不会被简单的 backbone reopening、输入分辨率放大、更高 `r3` 权重或简单 teacher-agreement gating 继续推高
   - `round_063`、`round_064` 进一步说明：当前平台也不会被 adaptive coarse blended target 或 `r3` 难度采样单独打破
   - `round_065` 到 `round_068` 再进一步说明：高可信 coarse 课程配合低学习率 reopening，可以把 strict-point 推到新的 `0.7239`
   - `round_069` 说明：这条课程线继续做超短 replay 已经开始回退，当前局部峰值应视为 `round_068`

关键指标入口：

- [key-metrics.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/key-metrics.md)
- [data-and-training-layout.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/architecture/data-and-training-layout.md)

## 下一步建议

1. 继续保留 `point_le_0_01_ratio + latency gate` 作为晋升规则。
2. 当前优先建议不要继续沿同一 coarse-preserving 配置族做邻域微调：
   - `round_060`、`round_061`、`round_062` 已满足“连续 3 轮没有 `+0.01` 提升”的平台判定
   - `round_063`、`round_064` 对更换 coarse target / sampling 口径的验证也没有突破主指标
   - `round_067`、`round_068` 已经通过“课程 + reopening”打出新的正收益，但 `round_069` 已开始回退
   - 若继续推进，应从 `round_068` 出发，优先尝试新的更大改变量，而不是继续做同配方 replay
3. 若继续推进，应切到更大改变量的 coarse 主线：
   - 新的 coarse teacher 口径
   - 更强但不冲突的 coarse-only supervision
   - 新的训练数据或更宽的场景覆盖
4. 当前不建议回到 `round_043 -> round_055` 这条旧 split_v2 同家族训练线：
   - `round_044` 到 `round_050` 已证明邻域 loss / residual 结构调参无效
   - `round_051` 已证明 full-ROI strict spatial head 无效
   - `round_052` 已证明 per-corner patch residual strict-head 仍无效
   - `round_053` 到 `round_055` 已证明 internal selector、external candidate pool 和 head-only selector tuning 也无效
5. `round_056` 到 `round_069` 已经说明：
   - 当前最有效的策略不是“更强 refine”
   - 而是“不要覆盖已经足够强的 coarse 输出，并避免让冲突 teacher 信号拖偏 coarse 学习”
   - 在此基础上，再叠“高可信 coarse 课程 + 低学习率 reopening + coarse-head consolidation”可以把 strict-point 从 `0.7164` 继续推到 `0.7239`
