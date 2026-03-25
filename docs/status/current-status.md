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

## 当前关键问题

1. local-corner 仍未满足最终目标：
   - 平均点位偏差 `< 0.5%`
   - 四点全部 `< 1%` 命中率 `> 80%`
   - 单张识别耗时 `< 500 ms`
2. 数据增强策略对泛化影响较大，尤其是翻转增强。
3. 训练与验证必须持续保持严格拆分。

关键指标入口：

- [key-metrics.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/key-metrics.md)
- [data-and-training-layout.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/architecture/data-and-training-layout.md)

## 最近实验结论

- `v31`：淘汰
- `v32`：holdout 恢复，但 broad 不够
- `v33`：禁用 flip 后 broad 大涨，但 holdout 崩
- `v34`：统一降低 flip 概率仍不行

结论：

- `v28` 仍是当前 local-corner 最稳候选
- 下一步更应该拆分水平/垂直翻转，而不是继续只调统一 `flip_prob`

## 下一步建议

1. 在新仓库跑最小验证
2. 确认桌面程序在新结构下可正常读取 `models/runtime`
3. 再继续训练时，优先试：
   - horizontal / vertical flip 分离
   - 受限 local refine 回退策略
