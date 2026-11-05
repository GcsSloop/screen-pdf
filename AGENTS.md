# Agents Guide

## 目标

让进入本仓库的代理快速判断：

- 程序入口在哪里
- 模型放在哪里
- 历史实验放在哪里
- 当前最重要的问题是什么
- 下一步应该从哪份文档开始

## 首先看什么

1. [README.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/README.md)
2. [docs/current-status.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/current-status.md)
3. [docs/status/model-naming-rules.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/model-naming-rules.md)
4. [docs/status/model-release-convention.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/model-release-convention.md)
5. [docs/status/distillation-run-convention.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/distillation-run-convention.md)
6. [docs/repository-layout.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/architecture/repository-layout.md)
7. [docs/key-metrics.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/key-metrics.md)

## 数据查看与基准入口

先读 [会议场次集合约定](docs/status/conference-collections.md) 和 `data/curated/collections/conference_v1/README.md`。当前 13 个会议 / 场次 / 专项集合覆盖 48 源项目、1950 页，排除 4 个异常页面后提供共 1946 页的可打开合并项目；1808 页明确人工框用于冻结模型的全量同集比较。后续先按会议 / 场次归组，再用屏幕场景标签下钻，不再把逐演讲报告作为唯一总入口。

数据集数量与完整清单先看 [去重索引](docs/status/dataset-index.md)：当前 56 个数据集单元、2,764 张唯一人工图像；会议合集与场次场景存在同图，索引中已注明重叠。

重复照片按内容校验，不同人工框 / 用途冲突隔离统计；`conflict-review/` 提供 1 张冲突照片的 2 个版本及另 2 页无效框保留溯源对照；用户已将这 4 页确认为异常并从日常集合排除，无需等待校正。原项目 / 原标签不覆盖，基准原本已排除这些页面，输入与分数不变。合并项目图片为独立副本，修改后不自动成为原项目或冻结基准的新真值。用途继承 round_010 定向实验快照，控制模型未训练刘奇新增页，快照外资料也不等于盲测。新增数据 / 修正均创建新 revision，已有训练、用途和基准不回写。

<!-- white-mix-corner-round040:start -->
2026-10-27 杭州智能交通年会「白屏混合」42 页人工四角已按用户精修重新登记：164/168 个点有位移（`IMG_6941` 未变），平均 0.213%、P90 0.365%、最大 0.868% 对角线，**全部在 1% 命中半径内**；页面成员与用途完全不变（33 训练 / 9 开发 / 0 回归，`IMG_7018` 与中交机电同图继续排除），新快照 `training/datasets/screen_corner_single_all_manual_3_20fine_whitemix_v1/` 的 `train` 仍 5575 条、`all_manual` 2286 页。round_040 从正式 round_038 做 control（旧标注）/ refined（精修标注）固定预算对照，两组 5575 条、12 轮粗定位 + 16 轮精修、每轮 175 次更新、seed=7，共用同一套评估真值。最终三组并列 33/36，选中 `initial`：**checkpoint 与正式 runtime 字节相同，没有新候选**；粗定位阶段精修标注有 +2 信号（31→33）但被精修阶段抹平，原域 474 页两组同时 -14（control 与 refined 净 0），说明是重训漂移而非标注贡献。剩余 3 个开发失配全是右下角（12.68% / 3.50% / 2.00%）。正式 runtime、0.2.26 包、冻结 conference_v1 与旧快照不变；未晋升、未打包。详见 [报告](research/experiments/2026-10-27-white-mix-corner-refinement.md)。
<!-- white-mix-corner-round040:end -->

<!-- smart-road-3-20-fine-labels:start -->
2026-10-24 中国智慧道路照明大会 3 月 20 日 59 页人工四角已按用户精细修订更新训练数据：58 页 232 个点全部微调（平均 0.168%、最大 0.645% 对角线），`IMG_8243` 从无显式人工框变为训练页；用途保持 46 训练（1 张近重复隔离）/12 开发，新增后 47/12。新标注、拆分、注册和新快照 `training/datasets/screen_corner_single_all_manual_3_20fine_v1/` 已生成；`train` 5534→5535 条、`all_manual` 2275→2276 页，其他成员、回放和 sidecar 继承 round_038。冻结 conference_v1、旧快照、正式 runtime 和 0.2.26 包不变；未训练、未开 round。当前 runtime 在 59 页旧 /新标签下均 234/236 命中、0 翻转，修订主要改善训练回归坐标。详见 [报告](research/experiments/2026-10-24-smart-road-lighting-fine-labels.md)。
<!-- smart-road-3-20-fine-labels:end -->

<!-- qinghui-round039:start -->
2026-10-24 2025-12-08「清晖 PgMP 同学会」116 页已接入独立会议室投影屏场景 `202512-qinghui-alumni_v1`（69 训练 / 24 开发 / 23 已查看回归，92 个内容组；3 张无项目页照片不纳入）。人工框由当前 runtime 候选微调（51/116 页 >1%、6 页 >10%，最大 78.99%），登记 `label_quality=runtime_candidate_seeded_manual`、`fine_corner_truth=false`，不是独立真值。round_039 从正式 round_038 出发，base 用含 3.20 精细标注的 `screen_corner_single_all_manual_3_20fine_v1`；control 5535 条、with_new 加入 69 页 ×24 共 7191 条，12 轮粗定位 + 16 轮精修、每轮 225 次更新、seed=7，只按清晖 24 页开发选 epoch 并先冻结再回归。选中 with_new：清晖开发 84→90/96、已查看回归 77→85/92、全部 161→175/188；原 483 页开发 1671→1665/1932，合并开发 1755/2028 持平，合并已查看回归 1007→1013/1156。候选 `screen_corner_single_049cd9cdf0de` 未晋升，正式 runtime 与 0.2.26 包不变；未重跑 1808 页冻结审计。详见 [报告](research/experiments/2026-10-24-qinghui-alumni-training.md)。
<!-- qinghui-round039:end -->

<!-- openatom-202511-dataset:start -->
2026-10-25 2025-11-03「开放原子开源基金」6 页已接入独立会议室投影屏场景 `202511-openatom-open-source-fund_v1`（4 训练 / 1 开发 / 1 已查看回归，5 个内容组）。明显前景遮挡：观众头部 /肩部沿下沿遮住屏幕，桌面与椅背占据下半部；人工四角按屏幕物理边框外推到被遮挡底边，不沿人头收边。6 张框由当前 runtime 候选微调 1.05%–5.85%，登记 `runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`，不是独立真值；无重复标注。开发 / holdout 各仅 1 页，无法支撑可靠选 epoch，本 revision 只登记训练数据，未训练、未开 round、未改正式 runtime 或 0.2.26 包。去重索引更新为 53 个单元 / 2,649 张唯一人工图像。详见 [场景说明](data/curated/scenes/202511-openatom-open-source-fund_v1/README.md)。
<!-- openatom-202511-dataset:end -->

<!-- zhongjiao-corner-revision:start -->
2026-10-25 中交机电点位重矫正 v2：源 20 页 /19 人工框，18 有效页 16 训练 /2 开发 /0 回归；旧 12 页点位变化 0.13%–0.80%，7 页新标注；7031/7035 修正后有效，7021 与训练页 7020 同内容组改训练，7018 与 202603-white-mix 同图冲突继续排除，7022 无人工框。**8 页屏幕角超出照片范围**（7021/7023/7026/7028/7031/7032/7035 右下、7038 右上），人工点取屏幕边与图像边界的交点，位于画面边框上，不是真实物理角点；这些点有效，不能按越界框过滤。新快照 `screen_corner_single_all_manual_3_20fine_zhongjiao_v1`：train 5535→5575、all_manual 2276→2286，validation/holdout 不变。未训练、未开 round、未改正式 runtime 或 0.2.26 包；去重索引 53 单元 / 3,703 单元内人工页 / 2,649 唯一人工图像。详见 [场景说明](data/curated/scenes/202603-zhongjiao-jidian_v1/README.md)。
<!-- zhongjiao-corner-revision:end -->

## 代码地图

- `program/desktop`
  - 桌面端 UI、Tauri 壳、打包配置。
- `program/engine`
  - Python 运行时检测、导出、训练、评估和测试。
- `data`
  - 原始数据、清洗数据、训练/验证拆分和任务派生数据。
- `training`
  - 训练配置、训练 runs、checkpoint、报告与模型注册。
- `models/runtime`
  - 当前程序默认加载的模型。
- `research/experiments`
  - 历史数据集、训练 run、评估报告、对比结果、实验记录。
- `docs`
  - 高信号说明文档，优先读这里，不要先在 `research` 里盲找。

## 当前共识

<!-- qujingjing-xuhui-round042:start -->
2026-10-29 上海开源大数据创新论坛同场次「AsterHarness：事实即本体-瞿晶晶」与「Rust 编译器开源社区治理-徐辉」各 27 页已接入独立会议室投影屏场景（两场各 15 训练 / 6 开发 / 6 已查看回归，25 /24 个内容组；瞿晶晶 `IMG_2791` 与张立文 `IMG_2777` 的 dHash=8 碰撞目视为不同幻灯片）。人工框由当前 runtime `c2409691b054` 候选微调 0.23%–2.06% / 0.29%–6.14%，登记 `runtime_candidate_seeded_manual`、`fine_corner_truth=false`，不是独立真值。训练前两场合计粗定位 110/216、最终 198/216，开发 42/48。round_042 从正式 round_041 冻结 encoder 与粗定位头、只训精修头：两组都回放快照 5575 + 张立文 36 页 ×24 = 6439 条，with_new 再加两场 30 页 ×24 共 7159 条；24 轮、每轮 224 次更新、seed=7，只按两场 12 页开发选 epoch 并先冻结再回归。选中 with_new 第 19 轮：两场开发 42→47/48、已查看回归 45→46/48、24 页 87→93/96；原 483 页开发 1668→1688（+20）、266 页回归 924→936（+12）；张立文重放 49→50/46→47；合并开发 1759→1785、合并回归 1015→1029。control 只回放也改善原域 +11/+7、两场开发 +2，部分收益来自精修头再拟合；with_new 相对 control 多 +9/+5 原域、+3/0 两场。同图 2605 页 9897→9953（+56），冻结 1808 页 6855→6881（+26，8 升 /0 退 /5 平，无单页丢 3 角）。候选 `screen_corner_single_9db7ad220b21` 已按用户要求晋升为正式单文件 runtime，发布编号 `model-20261029-180536-9db7ad220b21`，旧权重备份 `screen_corner_single.pre_round041.pt`；并重新打包 0.2.28（DMG SHA256 `ceadc2d942ab2498a5f48ce310ef35313ad92f55e57b2ddc011ac0952bf4ef5c`），Finder 布局与 0.2.16 一致、ad-hoc 签名、包内唯一权重摘要一致、真实检测返回新模型，本地 ad-hoc、未公证、无 updater 产物。残余 `IMG_2817` 右下 1.28%、`IMG_2818` 左下 1.77%、`IMG_2826` 右上 1.15%。详见 [报告](research/experiments/2026-10-29-qujingjing-xuhui.md)。
<!-- qujingjing-xuhui-round042:end -->

<!-- open-bigdata-zhangliwen-round041:start -->
2026-10-29 上海开源大数据创新论坛「金融领域的开源大语言模型-张立文」61 页已接入独立会议室投影屏场景 `202610-open-source-bigdata-zhangliwen`（36 训练 / 13 开发 / 12 已查看回归，54 个内容组；`IMG_2788` 未进项目排除）。人工框由当前 runtime `screen_corner_single_31adf6eec801` 候选微调 0.93%–30.56%（中位数 1.72%），登记 `runtime_candidate_seeded_manual`、`fine_corner_truth=false`，不是独立真值。训练前该场景粗定位 116/244、最终 144/244，开发 30/52。round_041 从正式 round_038 冻结 encoder 与粗定位头、只训精修头：control 5575 条、with_new 6439 条（36 页 ×24），24 轮、每轮 202 次更新、seed=7，只按本场 13 页开发选 epoch 并先冻结再回归。选中 with_new：开发 30→49/52、已查看回归 31→46/48、25 页 61→95/100、61 页 144→238/244；原 483 页开发 1671→1668、266 页回归 930→924，合并开发 1701→1717、合并回归 961→970；同图 2551 页 9605→9699（+94），冻结 1808 页 6861→6855（-6，4 升 /6 退 /3 平，无单页丢 3 角）。候选 `screen_corner_single_c2409691b054` 已按用户要求晋升为正式单文件 runtime，发布编号 `model-20261029-152403-c2409691b054`，旧权重备份 `screen_corner_single.pre_round038.pt`；并重新打包 0.2.27（DMG SHA256 `df7ba0eaeff08f2718d1613c85430a1081bba7e6bf81f997f22adfe718ec7b6c`），Finder 布局与 0.2.16 一致、ad-hoc 签名、包内唯一权重摘要一致、真实检测返回新模型，本地 ad-hoc、未公证、无 updater 产物。残余 `IMG_2738` 左上 2.43%、`IMG_2727` 左下 1.77%、`IMG_2790` 右下 2.19%，`IMG_2776` 因粗定位冻结仍错 17.85%。详见 [报告](research/experiments/2026-10-29-open-source-bigdata-zhangliwen.md)。
<!-- open-bigdata-zhangliwen-round041:end -->

<!-- nanjing-pm-round028:start -->
南京主论坛（下）40 页显式人工框已加入 `202610-nanjing-future-network-main-pm_v1`（24 训练 /8 开发 /8 已查看回归），中央 PPT 为目标，排除两侧人物直播区，保留摄像摇臂 /观众遮挡；另 5 张未列入项目照片不纳入。上下半场同族索引共 82 页，旧用途不回写。round_028 从正式 round_027 继续，control=3351、加入组=3927 条采样（新 24 页各 24 倍），相同 seed=7、每轮 123 次更新，先 12 轮粗定位再 16 轮精修。加入组粗定位 pooled 未胜初始，保留 epoch=0；新增页仍实际参与粗定位训练与最终精修，精修第 10 轮选中 `with_nanjing_pm`，候选 `screen_corner_single_a09836ddb464`。414 页开发 initial=1496/1656、control=1500/1656、加入组=1501/1656。下半场开发 10→14/32、已查看回归 8→11/32；上半场开发 15→17/36、回归 19→21/36。原 265 页开发少 2 点，旧新 174 页回归 624→622/696（旧 166 页少 5 点）；收益较小且残留粗定位 /摇臂干扰限制保留。29 项测试、组件冻结与缓存 /完整图一致性、隔离单权重真实检测通过。候选未晋升，未重跑全量晋升审计，正式 runtime 和安装包继续 round_027 /0.2.21。

[训练报告与叠框](research/experiments/2026-10-22-nanjing-main-pm-training.md)。
<!-- nanjing-pm-round028:end -->

<!-- round027-promotion-v0221:start -->
已按用户“整体退化不严重则晋升并重新打包”的授权，将 round_027 晋升为正式单文件 runtime `screen_corner_single_eef7e8e646aa`，模型发布编号 `model-20261022-020033-eef7e8e6`。与正式 round_024 重新推理冻结 1808 页：6697→6686/7232，下降约 0.15 个百分点（改善 72、回退 83 点）；当前 406 页开发 1468→1486/1624，166 页已查看回归 601→616/664，原 397 页开发 1466→1471/1588。快照外少 12/1068，杭州隧道集合少 7/244、北京 InfoComm 少 5/80，局部回退继续保留；全集含训练页，仅作覆盖/拟合诊断。南京开发 2→15/36、已查看回归 7→19/36（相对正式 round_024）。0.2.21 macOS Apple Silicon 本地安装包已重新生成，最终 DMG 的严格签名、Finder 布局与 Applications 拖放入口、唯一权重摘要及包内真实检测均通过；本地 ad-hoc 签名，未公证或外部发布，不含 updater 产物。

[整体回归审计](training/runs/screen_corner_single/round_027/reports/promotion-audit/README.md) · [下载 0.2.21（macOS Apple Silicon）](release-assets/local-single-20261022-v0221-round027/0.2.21/macos/ScreenPDF_0.2.21_aarch64.dmg)。DMG SHA256：`31fe5df6c1169d3e94497e6ecf42bb0d2e782bd38e0bb5a85364e4dd8fe65ee2`。
<!-- round027-promotion-v0221:end -->

<!-- nanjing-staged-round026-027:start -->
南京中央 PPT 按用户要求先粗后细：round_026 冻结 encoder /local head，12 轮粗定位对照选中 coordinate_guard，pooled 粗定位 1302→1312/1624，南京开发 8→11/36，真实角点进入局部裁剪 19→23/36；空间交叉熵分支退化未采用。round_027 冻结 encoder /global head，16 轮精修对照选中 point_hit，第 4 轮单文件候选 `screen_corner_single_eef7e8e646aa`。南京最终开发 12→15/36、已查看回归 15→19/36；合并开发 1478→1486/1624，原 397 页开发 1466→1471/1588、原 265 页 981→992/1060。两阶段分别冻结选择后才评回归；保留远景上角误吸吊顶、中央区域定位失败、单 seed 与已查看回归限制。32 项测试、组件冻结摘要和隔离单权重真实检测通过，训练完成时尚未重跑 1808 页冻结全量基准或晋升；后续已按用户授权完成审计并晋升 round_027，见最新晋升记录。

[训练报告与分阶段叠图](research/experiments/2026-10-22-nanjing-coarse-then-fine-training.md)。
<!-- nanjing-staged-round026-027:end -->

<!-- nanjing-round025:start -->
南京第十届未来网络大会主论坛（上）42 页显式人工框已接入独立中央 PPT 场景 `202610-nanjing-future-network-main-am_v1`（24 训练 /9 开发 /9 已查看回归），排除两侧人物直播区；另 4 张未列入源项目的照片不纳入。round_025 从 round_024 做固定预算 teacher_guard 对照，2775 /2919 条采样，共享 406 页开发，选中 `with_nanjing`：`screen_corner_single_c7f83295c7e7`。pooled 开发 initial=1468/1624、control=1477/1624、加入组=1478/1624；南京开发 2→12/36、已查看回归 7→15/36。旧 397 页开发同为 1466/1588，原始 265 页少 9 点；远景上角仍误吸吊顶、中央边界与遮挡失败保留。单 seed，本轮未重跑冻结全量基准，候选未晋升，正式 runtime 仍为 round_024。

[训练报告](research/experiments/2026-10-22-nanjing-main-forum-training.md)。
<!-- nanjing-round025:end -->

- 2026-10-19 正式单文件 runtime 已按用户要求晋升 round_024 马张翼翔 / 工业具身智能：`screen_corner_single_7f2257153f0c`，发布编号 `model-20261018-173619-7f225715`，摘要见 `models/runtime/screen_corner_single.json`。397 页 pooled 开发为 1466/1588；马张翼翔开发 / 留出 39/40、40/40，工业具身智能开发 / 留出 21/24、20/20；本次未重跑 1808 页冻结全量基准，也未生成桌面安装包。

- 涂耘资料目录已更名为 `/Users/gcssloop/WorkSpace/AIGC/展会/2026/202610 杭州-隧道照明与安全论坛/2026-10-16 隧道照明与绿色能源的深度融合之路-涂耘`，同名 PDF/导出目录、训练快照及原图链接同步迁移；数据标识、26 张人工标签、拆分、模型字节不变；当前 0.2.17 安装包仍使用 round_009。历史执行日志及原始打包验证中的旧路径按 `research/experiments/path-migrations/20261017-tuyun-date/README.md` 映射。

<!-- tuyun-round009:start -->
- 涂耘隧道照明资料 26 张已按独立超宽屏局部 PPT 场景 `202610-tunnel-lighting-tuyun_v1` 接入：15 训练 / 6 开发 / 5 已查看回归。目标仅为右侧 PPT，排除左侧固定信息区；保留 LED 拍摄条纹、右上角倒计时/红色超时叠层与左下角人头遮挡，沿用原人工四角，不沿人头或计时框收边。 round_009 固定预算对照完成，合并开发 initial=1180/1292、control=1179/1292、with_tuyun=1194/1292，选中 with_tuyun；未晋升单文件候选 `6fc6378701c4`。新场景开发 8/24→13/24、已查看回归 6/20→11/20。 左下角仍失败（开发 0/6、回归 0/5），蓝色结束页 0/4；右上角仅部分改善，不能称遮挡/计时干扰已解决。 旧资料完整回放，不追小样本分数；已打入本地 0.2.17 安装包，0.2.15 仍为 round_008，正式 runtime 未晋升。详见 `research/experiments/2026-10-17-tuyun-partial-screen-training.md`。
<!-- tuyun-round009:end -->

- 2026-10-17 用户调整：唯一准确率指标为 `point_le_0_01_ratio`，统计命中角点数 / 全部人工标注角点数。
- 原图像素距离除以人工框轴对齐包围盒对角线，误差 `<=1%` 计入命中；失败页面计四个未命中。
- checkpoint、对照实验和候选选择只最大化该指标；平局保留较早保存件。平均误差、四点全中率不再作为目标、门槛或平局条件。
- 耗时只作运行观测，不能混入准确率排序。
- 最终交付必须是一个模型文件，图内完成全局定位、局部取样和角点精修；不能依赖三个外部权重文件。
- 新主线：`program/engine/screen_corner_single.py`、`program/engine/training/train_screen_corner_single.py`，每轮放 `training/runs/screen_corner_single/round_xxx/`。
- 旧三阶段 runtime `r130 + c22 + r69probe` 仍保留作兼容；当前正式单文件 runtime 为 round_042 瞿晶晶 /徐辉精细定位（`screen_corner_single_9db7ad220b21`，发布编号 `model-20261029-180536-9db7ad220b21`）。该轮两场开发 42→47/48、已查看回归 45→46/48，原 483 页开发 1668→1688、266 页回归 925→936、合并开发 1759→1785、冻结 1808 页诊断 6855→6881（8 升 /0 退 /5 平，逐页无丢 3 角）；2605 页粗定位输出与上一版逐位相同，增益全部来自精修头，control 分解显示部分收益来自再拟合。上一版 round_041 保留在发布元数据与晋升审计中。
- 新活动 27 张人工标注已纳入数据：17 训练 / 5 开发验证 / 5 已查看留出回归，不能重新宣称盲测。
- 元宇宙邢杰演讲 98 张人工标注已接入 v2 拆分：58 训练 / 20 开发验证 / 20 已查看留出；round_002 完成同预算数据加入对照，单体候选 `3e8f4df71c2a` 已注册并打入本地 0.2.9 安装包，未正式晋升。
- XR 周群华演讲 17 张人工标注已接入：10 训练 / 4 开发验证 / 3 已查看回归。round_003 固定轻量对照完成，未晋升候选 `8f8fa2f691c5` 已打入本地 0.2.10 安装包；按用户要求先接触黑底蓝侧屏场景，不设小样本达标分数。部分边界改善，严重遮挡/点序仍未解决，详见 `research/experiments/2026-10-17-xr-small-scene-training.md`。
- 李琳琳媒体沉浸娱乐资料 7 张（5 黑、2 亮）已接入：3 训练 / 2 开发验证 / 2 已查看回归。round_004 继续回放 XR 和旧资料，完成一次固定预算对照；按用户要求不设小样本目标、不根据结果追分。未晋升候选 `5b2c52ff8450` 有部分边界改善，也保留少文字黑底漏边、角点落进蓝侧屏的失败；已打入本地 0.2.11 安装包。详见 `research/experiments/2026-10-17-immersive-black-scene-training.md`。
- 蓝侧屏黑底资料统一为 `20260929-black-background-unified_v1`：XR 14 + 李琳琳 5 + 新增 5，共 24 张（14 训练 / 5 开发 / 5 已查看回归）。历史用途锁定，亮底/大幅照片仍在普通集；统一黑底每图 6 倍采样，替换旧按演讲的权重。round_005 候选 `cf3cb63c6abe` 已打入本地 0.2.12 安装包。继续训练使用这份统一集，不把两场完整演讲都作为黑底，也不把旧黑底条目再次叠加。
- 董志国交通资料 26 张已接入独立的深色实体边框场景 `20260929-transport-dongzhiguo_v1`：15 训练 / 6 开发 / 5 已查看回归。round_006 固定预算对照完成，未晋升单文件候选 `39af695be2f4`；新场景开发 9/24→16/24，回归 6/20→14/20，旧蓝侧屏黑底命中数保持，追光开发少 1 点。已打入本地 0.2.13 安装包，0.2.12 仍用 round_005；正式 runtime 未晋升。场景总索引 `data/curated/scenes/black-background-index.json`；不套用蓝侧屏标签、不追小样本分数，详见 `research/experiments/2026-10-17-transport-low-contrast-scene-training.md`。
- 李阳 City Walk 资料 29 张已按独立黑白混合场景 `20260929-citywalk-liyang_v1` 接入：17 训练 / 6 开发 / 6 已查看回归，各拆分保留亮暗页面，不整场并入黑底集。round_007 固定预算对照完成，未晋升单文件候选 `db1dfebdcbd5`；新场景开发 16/24→19/24、回归 19/24→20/24，但两项及亮暗分组与同预算对照同分，不能全归因于新增资料；旧 XR 开发、追光回归各少 1 点。完整旧资料回放，不设小样本目标或追加追分。已打入本地 0.2.14 安装包，0.2.13 仍使用 round_006，正式 runtime 未晋升。详见 `research/experiments/2026-10-17-citywalk-mixed-scene-training.md`。
- 罗瀛漕河泾更新实验资料 38 张已按独立复杂花色背景场景 `20260929-urban-renewal-luoying_v1` 接入：22 训练 / 8 开发 / 8 已查看回归，保留蓝色纹理、照片、地图图表和混排，不整场并入黑底集。round_008 固定预算对照完成，未晋升单文件候选 `23fbbb5f04da`；合并开发 initial=1169/1268、control=1171/1268、with_luoying=1172/1268，新场景开发 28/32→30/32、回归 29/32→30/32。旧资料完整回放，不追加追分；已打入本地 0.2.15 安装包，0.2.14 仍使用 round_007，正式 runtime 未晋升。详见 `research/experiments/2026-10-17-luoying-complex-background-training.md`。
- 元宇宙原资料父目录已改为 `/Users/gcssloop/WorkSpace/AIGC/展会/2026/202609 上海-元宇宙产业嘉年华`；训练与数据集路径及原图链接同步迁移，数据标识和拆分不变。迁移审计、前后摘要和原文件备份在 `research/experiments/path-migrations/20261017-shanghai-metaverse/`。历史执行日志、签名安装包及原始打包验证保留当时的旧路径，查阅时使用迁移映射。
- 项目 JSON 打开兼容旧格式且不改写原文件，保存/导出统一写根目录 v2 相对路径格式；名称取当前目录名，详见 `docs/architecture/project-file-format.md`。
- 继续训练先读 `docs/status/evaluation-contract.md`；下文历史实验入口仅供溯源。

## 改动原则

- 运行时代码优先放在 `program/engine`。
- 新的运行时模型只放 `models/runtime`。
- 大体量实验结果不要直接堆进 `docs`，放 `research/experiments`。
- `docs` 只保留高信号入口、当前状态、架构与执行计划。

## 如果你要继续训练

先看：

1. [docs/current-status.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/current-status.md)
2. [docs/2026-03-25-local-corner-bl-structure-experiments-report.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/plans/2026-03-25-local-corner-bl-structure-experiments-report.md)
3. [docs/key-metrics.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/key-metrics.md)
4. [data/README.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/data/README.md)
5. [training/README.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/training/README.md)
6. `program/engine/*train*.py`
7. `program/engine/*infer*.py`

## 如果你要继续桌面程序

先看：

1. `program/desktop/src`
2. `program/desktop/src-tauri`
3. `program/engine/detect_frame.py`

如果任务涉及构建或发布安装包，继续阅读 [桌面程序打包流程](docs/runbooks/desktop-packaging.md)，按本机候选构建和正式多平台发布的不同步骤执行。

## 如果你要改模型路径或打包

关注：

- [detect_frame.py](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/engine/detect_frame.py)
- [tauri.conf.json](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/desktop/src-tauri/tauri.conf.json)


<!-- liuqi-round010:start -->
刘奇资料 35 张已接入（21 训练 / 7 开发 / 7 已查看回归），与涂耘组成 61 张超宽屏右侧 PPT 场景族，旧拆分保留。round_010 定向组两场训练页均 12 倍采样，固定预算对照完成；合并开发 initial=1206/1320、control=1212/1320、with_tunnel_focus=1206/1320，选中 control；单文件候选 `d9d6ad842126` 未晋升。同场景族开发 48.08%（25/52）→59.62%（31/52），已查看回归 60.42%（29/48）→85.42%（41/48）。定向组未胜过 control，选中候选未采用新增刘奇训练页的权重；该轮完成时正式 runtime 未变，0.2.17 安装包仍使用 round_009，之后由 round_011 晋升更新正式 runtime。详见 [训练报告](research/experiments/2026-10-17-liuqi-tunnel-family-training.md)。
<!-- liuqi-round010:end -->

<!-- sunquanming-round012:start -->
- 孙泉明资料 29 张已接入独立场景 `202610-tunnel-lighting-sunquanming_v1`：17 训练 / 6 开发 / 6 已查看回归。目标为杭州隧道照明宽幅 LED 屏内右侧 PPT，排除左侧蓝色固定区；多数物理屏幕背景为黑色，保留页内白色图表/照片、LED 拍摄条纹、右上角倒计时/红色超时叠层和底部观众遮挡，人工四角不沿遮挡轮廓收边。
- round_012 从正式 round_011 继续，旧 1665 条训练采样保持不变，新增 17 张训练页各 6 倍为 1767 条。合并开发 initial=1223/1344、control=1218/1344、with_sunquanming=1231/1344，选择候选 `a1f4bec7b047`；孙泉明开发 8/24→16/24、已查看回归 2/24→15/24，旧开发回退 7 点，因此未晋升正式 runtime。完整报告见 `research/experiments/2026-10-18-sunquanming-black-screen-training.md`。
<!-- sunquanming-round012:end -->

<!-- liwei-round013:start -->
- 李薇资料 24 张已接入独立场景 `202610-tunnel-lighting-liwei_v1`：14 训练 / 5 开发 / 5 已查看回归。目标为杭州隧道照明宽幅 LED 屏内右侧 PPT，排除左侧蓝色固定区；保留黑色屏幕背景、页内亮色内容、LED 拍摄条纹、右上角倒计时/红色超时叠层和底部观众遮挡，人工四角不沿遮挡轮廓收边。
- round_013 从正式 round_011 继续，旧 1665 条训练采样保持不变，新增 14 张训练页各 6 倍为 1749 条。合并开发 initial=1230/1340、control=1227/1340、with_liwei=1236/1340，选择候选 `5ba8b4a37a5d`；李薇开发 15/20→16/20、已查看回归 14/20→17/20。冻结全量基准中 13 个集合有 9 个上升、3 个回退、1 个持平，累计上升 40 点、回退 6 点，已晋升正式 runtime。完整报告见 `research/experiments/2026-10-18-liwei-partial-screen-training.md`。
<!-- liwei-round013:end -->

<!-- lihuabin-round014:start -->
- 李华斌资料 17 张已接入独立场景 `202610-tunnel-lighting-lihuabin_v1`：9 训练 / 4 开发 / 4 已查看回归。目标为杭州隧道照明宽幅 LED 屏内右侧 PPT，排除左侧固定蓝色信息区；保留蓝色和深色屏幕背景、LED 拍摄条纹、右上角倒计时叠层和底部观众遮挡，人工四角不沿遮挡轮廓收边。
- round_014 从正式 round_013 继续，旧 1749 条训练采样保持不变，新增 9 张训练页各 6 倍为 1803 条。合并开发 initial=1240/1356、control=1240/1356、with_lihuabin=1241/1356，选择候选 `2343521e5471`；李华斌开发 4/16→5/16、已查看回归 4/16→9/16，旧验证回退 3 点，候选未晋升正式 runtime。详见 `research/experiments/2026-10-18-lihuabin-partial-screen-training.md`。
<!-- lihuabin-round014:end -->

<!-- liusongrong-round015:start -->
- 刘松荣资料 15 张已接入独立场景 `202610-tunnel-lighting-liusongrong_v1`：9 训练 / 3 开发 / 3 已查看回归。目标为杭州隧道照明宽幅 LED 屏内右侧 PPT，排除左侧固定蓝色信息区；保留白底/蓝底页面、密集表格图表、LED 拍摄条纹、右上角倒计时叠层和底部观众遮挡，人工四角不沿遮挡轮廓收边。
- round_015 沿用 round_014 同场训练快照，旧 1803 条训练采样保持不变，新增 9 张训练页各 6 倍为 1857 条。合并开发 initial=1247/1368、control=1251/1368、with_liusongrong=1251/1368，按平局规则选择 control；刘松荣开发 6/12→8/12、已查看回归 8/12→9/12，旧验证回退 3 点，候选未晋升正式 runtime。详见 `research/experiments/2026-10-18-liusongrong-partial-screen-training.md`。
<!-- liusongrong-round015:end -->

<!-- konglingmin-round021:start -->
- 孔令敏 AI + PLC 隧道照明资料 39 张已接入 `202610-tunnel-lighting-konglingmin_v1`（23 训练 / 8 开发 / 8 已查看回归）；3 段 MOV 不进图片训练。目标为完整右侧 PPT，排除左侧固定信息区，保留深蓝底、密集图表/软件截图、LED 条纹、倒计时和观众；人工点不自动改写。
- round_021 从正式 round_018 做同 teacher_guard 固定预算对照，control=1947 条、新增组=2085 条（23 页各 6 倍），两组 4/8 轮、每轮 66 次更新。合并开发 initial=1292/1420、control=1292/1420、with_konglingmin=1303/1420，冻结候选 `7279848d9388`。本场开发 13→23/32，已查看回归 11→24/32；右下角开发仅 1/8、回归 3/8，旧 265 页开发回退 3 点，不能称精修或遮挡问题解决。未重跑全量晋升基准，随后按用户明确要求晋升，正式 runtime 为 round_021，配套本地版本 0.2.20。场景族新修订 v2 共 210 页，旧用途/v1/冻结会议集不回写。详见 `research/experiments/2026-10-18-konglingmin-fine-corner-training.md`。
<!-- konglingmin-round021:end -->

<!-- zhangwenju-round022:start -->
- 张文居资料 48 张已接入 `202610-tunnel-lighting-zhangwenju_v1`（28 训练 / 10 开发 / 10 已查看回归），目标为宽幅 LED 屏内完整右侧 PPT，保留白底/蓝底、地图图表、工程照片、LED 条纹、倒计时与观众，原人工点不改写。round_022 从正式 round_021 做相同 teacher_guard 的固定预算对照，旧 2085 条采样保留，新增 28 页各 6 倍共 2253 条；两组均 4/8 轮、每轮 71 次更新。合并开发 initial=1335/1460, control=1351/1460, with_zhangwenju=1349/1460，选中 `control`，单文件候选 `c2267f9e0d71`。选中候选未使用张文居新增训练页，加入组权重另保留。本场开发 32→35/40、已查看回归 33→33/40；既有 355 页开发净变化 +13 点，原 265 页开发 +3 点。场景族 v3 共 258 页，旧用途/v1/v2/冻结会议集不回写；小样本、单 seed 与残留失败保留。正式 runtime 继续 round_021，候选未晋升。详见 [训练报告](research/experiments/2026-10-18-zhangwenju-partial-screen-training.md)。
<!-- zhangwenju-round022:end -->

<!-- duchao-dataset-v1:start -->
杜超资料初次登记为 8 训练 / 3 开发 / 3 已查看回归，原登记快照保留；随后已在 round_023 的 with_hangzhou 与 family_focus 两组执行训练。选中加入两场新资料的 with_hangzhou；提高同族采样比例的分支未超过它。 详见 [完整报告](research/experiments/2026-10-18-gengchiyuan-family-focus-training.md)。
<!-- duchao-dataset-v1:end -->

<!-- gengchiyuan-round023:start -->
耿驰远 36 张已接入（22 训练 / 7 开发 / 7 已查看回归），杜超 14 张沿用 8 / 3 / 3。round_023 从 round_022 候选 c2267f9e0d71 初始化，完成 control / with_hangzhou / family_focus 三组固定预算对照；各组 seed=7、4/8 轮、每轮 103 次更新，训练采样 2253 / 2433 / 3291 条。合并开发 381 页：initial=1397/1524, control=1404/1524, with_hangzhou=1408/1524, family_focus=1404/1524，候选 `8e554a8d68b7`。选中加入两场新资料的 with_hangzhou；提高同族采样比例的分支未超过它。同族开发 210/256 → 228/256，已查看回归 209/252 → 230/252；原 265 页开发 993/1060 → 988/1060。场景族 v5 共 11 场、308 页；143 张实际采样的同族训练页在定向组提高至 12 倍，刘奇/孙泉明未采样训练页没有借此引入。旧用途、旧 revision、冻结会议集及源人工点保留；单种子、反复使用开发集和残留失败的限制继续保留。正式 runtime 已按用户要求晋升为 round_023：`screen_corner_single_8e554a8d68b7`，发布编号 `model-20261018-154105-8e554a8d`，SHA256 `8e554a8d68b7b7651c52f3050a0c7b2e2383a156a2e65f835f1a902683ab6033`。未重跑 1808 页全量基准；旧 265 页开发少 5 点，右上与遮挡右下失败仍保留。

0.2.20 本地 macOS Apple Silicon 包已重新生成并验收，DMG SHA256 `0475e4e7dbada0ce4246b59db7b08e8b8f45b80e79b09005e7d5cf66444f197a`；包内真实检测来源 `runtime_single`，本地 ad-hoc 签名，未公证或外部发布。详见 `release-assets/local-single-20261018-v0220-round023/`。

[完整报告](research/experiments/2026-10-18-gengchiyuan-family-focus-training.md)。
<!-- gengchiyuan-round023:end -->


<!-- round024-mazhangyixiang-industrial:start -->
- 马张翼翔资料 47 张按 27 训练 / 10 开发 / 10 已查看留出加入杭州宽屏场景族；工业具身智能资料 26 张按 15 / 6 / 5 加入独立新场景。round_024 四组固定预算对照选择 `with_both`，pooled 开发 1466/1588；工业最终开发 / 留出 21/24、20/20，正式 runtime 为 `screen_corner_single_7f2257153f0c`，发布编号 `model-20261018-173619-7f225715`。详见 `research/experiments/2026-10-19-mazhangyixiang-industrial-training.md`。
<!-- round024-mazhangyixiang-industrial:end -->

<!-- computing-round029:start -->
智算网络创新技术论坛 66 页已接入（39 训练 /14 开发 /13 已查看回归），保留完整中央 PPT 人工四角和底部观众 /椅背遮挡。round_029 从 round_028 候选做相同 20 轮、每轮 182 次更新的粗定位对照，选中直接加入组第 16 轮 `screen_corner_single_c495b67dcac4`；本轮按智算开发粗定位选优，encoder /local head 冻结。新场景开发粗定位 17→45/56、底部 5→19/28，已查看回归粗定位 9→33/52、最终 30→42/52。区域拟合分支开发为 34/56，旧域明显回退，未采用。选中候选旧 414 页开发粗定位 -21 点、最终 -12 点，pooled 最终 1535→1534/1712；左下仅 6/14，冻结精修会重新拉偏部分正确粗点。南京场景族 v2 共 148 页，旧用途 /revision /冻结会议集保留。33 项测试和四个隔离单文件实际检测通过；未晋升 /打包，正式 runtime /桌面包继续 round_027 /0.2.21。

[完整报告](research/experiments/2026-10-22-computing-forum-region-coarse-training.md)。
<!-- computing-round029:end -->

<!-- all-history-round030:start -->
round_030 已根据全部历史可用人工资料独立训练区域识别 → 粗定位 → 精修单体模型。62 个来源、2275 页，1461 训练 /432 开发 /216 已查看回归，166 张近评估重复训练页隔离。仅通用 ImageNet 初始化，骨干和三个任务阶段重新训练，未继承 ScreenPDF 业务权重或教师。候选 `screen_corner_single_500bb4820b86`；开发最终正式 1496/1728（86.57%） → 新模型 1508/1728（87.27%）（+12 点），已查看回归正式 715/864（82.75%） → 新模型 762/864（88.19%）。南京开发 +33、InfoComm +7，openatom -18、福州高速 -12 等局部回退保留。12 /80 /48 轮固定预算，按 pooled 开发唯一角点指标冻结，先冻结再回归；含训练资料的 1808 页诊断（1 页数据集修订排除页按冻结人工框补算）/2275 页诊断与逐会议变化另报。seed=11 复验（round_031）开发 1522、已查看回归 770，粗定位 1496/1497 复现，final 增益带约 ±0.8 个百分点种子波动；精修净贡献仅 +12/+25（成熟模型 +191），openatom /福州高速 /smart-road /metaverse /商场测试集回退两 seed 复现，故建议先修精修阶段再谈晋升。23 项测试和隔离单文件实际检测通过；两 seed 复验后，固定预算、历史评估已查看等限制仍保留。未晋升 /打包，正式 runtime 仍 round_027。

[完整报告](research/experiments/2026-10-22-all-history-region-coarse-local-training.md) · [seed 复验与综合指标](research/experiments/2026-10-22-all-history-seed-replication.md)。
<!-- all-history-round030:end -->

<!-- round031-promotion-v0222:start -->
## 2026-10-22 round_031 全历史模型已晋升，0.2.22 本地包已验收

按用户要求把全历史独立训练的单文件模型晋升为正式 runtime `screen_corner_single_052594290a1d`（round_031，seed=11），发布编号 `model-20261022-131533-05259429`，旧权重备份 `models/runtime/screen_corner_single.pre_round027.pt`。与正式 round_027 同图重建推理：432 页开发 1496→1522（86.57%→88.08%），216 页已查看回归 715→770（82.75%→89.12%），原 428 页子集 1423→1432，冻结 1808 页诊断 6686→6882，全部 2275 页诊断 8182→8640；平均归一化误差开发 1.91%→0.76%、回归 2.16%→0.73%。冻结诊断仍有三处回退并写入发布元数据：`202604-mall` -59（疑似把屏幕外边黑框纳入目标）、`202312-openatom` -30、`202603-fuzhou-highway` -13。同轮精修专项 round_032（冻结该基座、只重训局部头 96 轮）开发 1519、回归 773，未超过基座，说明精修瓶颈不是训练轮数，未采用。0.2.22 macOS Apple Silicon 安装包已生成并验收：Finder 布局与 0.2.16 逐字节一致、版本 0.2.22、ad-hoc 严格签名、包内真实检测来源 `runtime_single` 且模型摘要一致；本地 ad-hoc 签名、未公证、未对外发布，包内含本次构建的 updater 签名产物但没有 latest.json，不构成更新通道。

[晋升审计](training/runs/screen_corner_single/round_031/reports/promotion-audit/README.md) · [决策记录](training/runs/screen_corner_single/round_031/reports/promotion_decision.json) · [下载 0.2.22（macOS Apple Silicon）](release-assets/local-single-20261022-v0222-round031/0.2.22/macos/ScreenPDF_0.2.22_aarch64.dmg)。DMG SHA256：`8c35eb39c9709ac14e04b84da73e3a9b9845ff7c42fd4abf9008eed45d355773`。
<!-- round031-promotion-v0222:end -->

<!-- nanjing-edge-dataset-v1:start -->
- 南京边缘算力与泛在智能创新发展论坛 67 页已接入 `202610-nanjing-edge-computing-forum_v1`（39 训练 /13 开发 /15 已查看回归），中央 PPT 幻灯片四边形为目标，排除左右固定信息区与演讲人标题 /头像区，保留底角观众 /椅背遮挡；源项目 1 张未列入页面照片不纳入。状态 registered_not_trained：本次仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包；南京中央 PPT 场景族 v3 共 215 页（126 训练 /44 开发 /45 已查看回归）。跨场次无精确字节重复，5 对 pHash 标记逐对目视为不同幻灯片；旧用途 /revision 与冻结会议集不回写。详见 [场景说明](data/curated/scenes/202610-nanjing-edge-computing-forum_v1/README.md)。
<!-- nanjing-edge-dataset-v1:end -->

<!-- embodied-ai-chain-dataset-v1:start -->
- 上海 IN³C 2026 场次「AI铸链 向实而生」（2026-10-14，京东工业）4 页显式人工框已接入 `202610-embodied-intelligence-ai-chain`（2 训练 /1 开发 /1 已查看回归），目标为宽幅 LED 舞台屏内完整中央 PPT 幻灯片，排除左右品牌固定区与底部 IN³C 横幅，保留观众头部遮挡。状态 registered_not_trained：本次仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包。按场次建立场景族 v1（10-13 工业具身智能 26 页 + 10-14 AI铸链 4 页，共 30 页）；工业场次原 split、round_024 快照与冻结会议集不回写。详见 [场景说明](data/curated/scenes/202610-embodied-intelligence-ai-chain_v1/README.md) 与 [场景族 v1](data/curated/scenes/202610-shanghai-embodied-intelligence_v1/README.md)。
<!-- embodied-ai-chain-dataset-v1:end -->

<!-- embodied-occluded-dataset-v1:start -->
- 上海 IN³C 2026 两场 2026-10-13 资料已接入：华为 AI 6 页（3 训练 /2 开发 /1 已查看回归，`202610-embodied-intelligence-huawei-ai`）与「工业控制系统从自动化向自主化的演进-龚涛」40 页（24 /8 /8，`202610-embodied-intelligence-gongtao`）。用户指出本批遮挡较大：讲者位于屏幕左前缘，讲台、观众头部与举手拍摄遮挡下沿和左下部；华为场屏幕占画面比例小、摩尔纹重。目标仍为 LED 屏完整四边形（含页脚条），人工四角外推到被遮挡边缘，不沿遮挡轮廓收边。状态 registered_not_trained：本次仅接入训练数据；上海 IN³C 场景族 v2 共 76 页（44 训练 /17 开发 /15 已查看回归），工业具身智能 split、round_024 快照与冻结会议集不回写。详见 [华为 AI](data/curated/scenes/202610-embodied-intelligence-huawei-ai_v1/README.md)、[龚涛](data/curated/scenes/202610-embodied-intelligence-gongtao_v1/README.md)、[场景族 v2](data/curated/scenes/202610-shanghai-embodied-intelligence_v2/README.md)。
<!-- embodied-occluded-dataset-v1:end -->

<!-- embodied-occlusion-round033:start -->
- 上海 IN³C 重遮挡两场（华为 AI 6 页 + 龚涛 40 页，27 页训练）已在 round_033 从正式 round_031 做固定预算对照：control 回放 2222 条、with_occluded 加入 27 页×24 共 2870 条，粗定位 12 轮 + 精修 16 轮、每轮 90 次更新，只按 10 页重遮挡开发唯一角点指标选 epoch 并先冻结再回归。选中 with_occluded（加入 27 页），重遮挡开发最终 32/40（80.00%）；候选 `screen_corner_single_4a6904df2cd6` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、10 页开发、人工点外推遮挡边缘与已查看回归限制保留。详见 [训练报告](research/experiments/2026-10-22-embodied-occlusion-training.md)。
<!-- embodied-occlusion-round033:end -->

<!-- pm-cube-round034:start -->
- 2024「新任产品经理管理魔方」15 页已接入（9 训练 / 3 开发 / 3 已查看回归，投影白底幻灯片，参考模型在这批页面的角点误差约 1–5%、命中仅 16/60）。round_034 从正式 round_031 冻结 encoder 与粗定位头，只训练本地精修头 24 轮做固定预算对照：control 回放 2222 条、with_pm_cube 加入 9 页×24 共 2438 条，只按 3 页精细定位开发唯一角点指标选 epoch 并先冻结再回归。选中 with_pm_cube（加入 9 页），开发最终 11/12（91.67%）；候选 `screen_corner_single_9eb8bc25c145` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、3 页开发、只训本地头的限制保留。详见 [训练报告](research/experiments/2026-10-22-pm-cube-fine-localization.md)。
<!-- pm-cube-round034:end -->

<!-- combined-new-data-round035:start -->
- 五批新增资料（南京边缘算力 67 页、AI铸链 4 页、华为 AI 6 页、龚涛 40 页、2024 PM 管理魔方 15 页，合计 132 页 / 77 训练页）已在 round_035 从正式 round_031 合并做固定预算对照：control 回放 2222 条、with_new_all 加入 77 页×24 共 4070 条，粗定位 12 轮 + 精修 16 轮、每轮 128 次更新，只按 27 页合并新场景开发唯一角点指标选 epoch 并先冻结再回归。选中 with_new_all（加入 77 页），合并新场景开发最终 94/108（87.04%）；候选 `screen_corner_single_5be1a6da1dd2` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、27 页开发、五场景混合采样与已查看回归限制保留。详见 [训练报告](research/experiments/2026-10-23-combined-new-data-training.md)。
<!-- combined-new-data-round035:end -->

<!-- candidate-scene-comparison-20261023:start -->
- 四个单文件模型（正式 round_031、round_033 遮挡、round_034 PM 精修、round_035 五批合并）已在同一 2407 页语料（旧 2275 + 新增 132）重新推理做分场景对比：全部页 8951 /8996 /8984 /**9067** 命中；旧 2275 页 8640 /8618 /8632 /8600；新增 132 页 311 /378 /352 /**467**；已查看回归 244 页 836 /832 /**850 /851**。相对正式：round_033 旧 -22 /新 +67 /回归 -4；round_034 旧 -8 /新 +41 /回归 +14；round_035 旧 -40 /新 +156 /回归 +15。要一个 runtime 覆盖全部新增场景选 round_035（`screen_corner_single_5be1a6da1dd2`，需重跑 1808 页冻结审计并重新打包）；要旧域风险最小选 round_034（`9eb8bc25c145`）；round_033 保留为遮挡对照；正式 round_031 仍是旧场景最佳。均未晋升。详见 [对比报告](research/experiments/2026-10-23-candidate-scene-comparison.md)。
<!-- candidate-scene-comparison-20261023:end -->

<!-- round035-promotion-v0223:start -->
- 按用户要求把 round_035（五批新增资料合并加入）晋升为正式单文件 runtime `screen_corner_single_5be1a6da1dd2`，发布编号 `model-20261023-041021-5be1a6da`，旧权重备份 `models/runtime/screen_corner_single.pre_round031.pt`。晋升审计同图重跑正式 round_031 与候选：459 页开发 1587→1581/1836（-6），244 页已查看回归 836→851/976（+15），冻结 1808 页诊断 6882→6853/7232（-29），全部 2275 页诊断 8640→8600/9100（-40），新增 132 页 311→467/528（+156）。冻结集合回退：福州高速 -12、开放原子 -9、扬州 -3、智慧道路照明 3 月 20 日 -3、元宇宙 -2、杭州隧道集合 -2、追光 -2、杭州智能交通 -2；改善：上海大屏 +2、3 月 19 日 +2、商场 +1、AWE +1，全部写入发布元数据。0.2.23 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `98f42ca7a06321cf8b7246ba51c2c6d12fc770b0dcf13691d4db1f9e4482c8c0`，Finder 布局与 0.2.16 逐字节一致、版本 0.2.23、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_5be1a6da1dd2`；本地 ad-hoc 签名、未公证、未对外发布，包内含 updater 签名产物但没有 latest.json，不构成更新通道。详见 [晋升审计](training/runs/screen_corner_single/round_035/reports/promotion-audit/README.md) 与 [下载 0.2.23](release-assets/local-single-20261023-v0223-round035/0.2.23/macos/ScreenPDF_0.2.23_aarch64.dmg)。
<!-- round035-promotion-v0223:end -->

<!-- qintong-dataset-v1:start -->
- 上海 IN³C 2026 场次「面向开放世界的具身智能三维探索、移动操作与全场景仿真-秦通」33 页已接入 `202610-embodied-intelligence-qintong`（19 训练 /7 开发 /7 已查看回归，32 个内容组），目标为宽幅 LED 舞台屏内完整白色幻灯片内容区，排除两侧品牌固定区与底部 IN³C 会议横幅，保留讲者 /观众遮挡、LED 摩尔纹与轻微透视。状态 registered_not_trained：本次仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包。上海 IN³C 场景族由 v2 追加为 v3，五场共 109 页（63 训练 /24 开发 /22 已查看回归）；与既有标注无精确字节 /强 dHash 重复，2 对 pHash 标记逐对目视为不同幻灯片；旧 revision、旧用途与冻结会议集不回写。详见 [场景说明](data/curated/scenes/202610-embodied-intelligence-qintong_v1/README.md) 与 [场景族 v3](data/curated/scenes/202610-shanghai-embodied-intelligence_v3/README.md)。
<!-- qintong-dataset-v1:end -->

<!-- zhufeng-luchen-dataset-v1:start -->
- 上海 IN³C 2026 两场 2026-10-13 资料已接入：「智能无人装备安全标准研究进展-朱峰」4 页（2 训练 /1 开发 /1 已查看回归，`202610-embodied-intelligence-zhufeng`）与「西门子工业边缘及人工智能赋能制造业数字化转型-陆晨」10 页（6 /2 /2，`202610-embodied-intelligence-luchen`），目标为宽幅 LED 舞台屏完整幻灯片 /屏幕四边形，排除两侧品牌固定区，保留讲者与观众遮挡、LED 摩尔纹与轻微透视。状态 registered_not_trained：仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包。上海 IN³C 场景族由 v3 追加为 v4，七场共 123 页（71 训练 /27 开发 /25 已查看回归）；无精确字节 /强 dHash 重复，2 对 pHash 标记（朱峰 IMG_2099 vs 龚涛 IMG_2039、陆晨 IMG_2091 vs 华为 IMG_2003）逐对目视为不同幻灯片。同目录「工业领或具身智能洞察与实践-于青民」6 张照片没有 ScreenPDF 项目文件与人工框，本批未纳入，需标注后再接入。详见 [朱峰](data/curated/scenes/202610-embodied-intelligence-zhufeng_v1/README.md)、[陆晨](data/curated/scenes/202610-embodied-intelligence-luchen_v1/README.md) 与 [场景族 v4](data/curated/scenes/202610-shanghai-embodied-intelligence_v4/README.md)。
<!-- zhufeng-luchen-dataset-v1:end -->

<!-- occlusion-round036:start -->
- 秦通 33 /朱峰 4 /陆晨 10 页已接入；round_036 从正式 round_035 做遮挡定向对照：control 回放 4070 条、with_new 加入 27 页×24 共 4718 条、occlusion_focus 再把 54 张遮挡相关页加 24 倍共 6014 条，粗定位 12 轮 + 精修 16 轮、每轮 188 次更新，只按 20 页遮挡开发唯一角点指标选 epoch 并先冻结再回归。选中 with_new（加入 27 页），遮挡开发最终 62/80（77.50%）；候选 `screen_corner_single_7237adbcaa54` 未晋升，正式 runtime 与 0.2.23 安装包不变。单 seed、20 页遮挡开发、加倍采样不是独立样本与已查看回归限制保留。详见 [训练报告](research/experiments/2026-10-24-occlusion-focus-training.md)。
<!-- occlusion-round036:end -->

<!-- regression-compare-round036:start -->
- round_036 回归验证：正式 round_035（5be1a6da1dd2）对比候选 `screen_corner_single_7237adbcaa54`，同一 2454 页（旧 2275 + 新增 179）各自重跑推理。全部页 9177→9250（+73），新增 179 页 577→651（+74，80.59%→90.92%），旧 2275 页 8600→8599（-1），冻结 1808 页诊断 6853→6846（-7），469 页开发 1604→1616（+12），254 页已查看回归 874→886（+12）。新增场景：秦通全部页 +37、龚涛 +18、陆晨 +11、朱峰 +5，新场景开发 +5、回归 +9；旧域 13 个集合 5 升 /7 降 /1 平，最差 -3，逐页没有少 3 角的页面。结论：候选达到可晋升量级（新场景 +74、旧域 -1、无整页崩坏）；晋升仍需重跑 1808 页冻结审计并重新打包，正式 runtime 暂不变。详见 [回归验证报告](research/experiments/2026-10-24-round035-vs-round036-regression/README.md)。
<!-- regression-compare-round036:end -->

<!-- round036-promotion-v0224:start -->
- 按用户要求把 round_036 遮挡定向候选晋升为正式单文件 runtime `screen_corner_single_7237adbcaa54`，发布编号 `model-20261023-165533-7237adbc`，旧权重备份 `models/runtime/screen_corner_single.pre_round035.pt`。同图重跑正式 round_035 与候选：469 页开发 1604→1616（+12），254 页已查看回归 874→886（+12），冻结 1808 页诊断 6853→6846（-7），全部 2275 页诊断 8600→8599（-1），新增 179 页 577→651（+74）。冻结集合回退：InfoComm /商场 /开放原子各 -3，元宇宙 -2，AWE /上海大屏 /道路照明 3.19 各 -1；改善：扬州 +3，道路照明 3.20 /智能交通年会 /福州高速 /追光各 +1；逐页没有少 3 角的页面。0.2.24 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `80103f2d3765ae7d28dd25e70ebbe2a1bcf050b01011de52985cc160ddb4b010`，Finder 布局与 0.2.16 参考一致（同一 .DS_Store / 卷图标摘要）、版本 0.2.24、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_7237adbcaa54`；本地 ad-hoc 签名、未公证、未对外发布，包内含 updater 签名产物但没有 latest.json，不构成更新通道。详见 [晋升审计](training/runs/screen_corner_single/round_036/reports/promotion-audit/README.md) 与 [下载 0.2.24](release-assets/local-single-20261024-v0224-round036/0.2.24/macos/ScreenPDF_0.2.24_aarch64.dmg)。
<!-- round036-promotion-v0224:end -->

<!-- embodied-v5-dataset:start -->
- 上海 IN³C 2026 四场 2026-10-13 资料已接入：「从软硬件工程到应用的AI应用方案-周浩杰」6 页（3 训练 /2 开发 /1 已查看回归，`202610-embodied-intelligence-zhouhaojie`）、「智能机器人全域全行业应用分享-姚志强」4 页（2 /1 /1，`202610-embodied-intelligence-yaozhiqiang`）、「边缘为枢，云网为基，赋能具身智能走向规模-马战凯」15 页（9 /3 /3，`202610-embodied-intelligence-mazhankai`；项目内 `IMG_2139` 为 needs_review、无人工框，排除）、「人工智能财能工业提质增效-刘意杨」14 页（8 /3 /3，`202610-embodied-intelligence-liuyiyang`），共 39 页。目标为宽幅 LED 舞台屏完整屏幕四边形，排除两侧品牌固定区。用户指出这批人工框"基本定位相对准确，但还不够精细"：四场标注均由当前 runtime 候选 `screen_corner_single_7237adbcaa54` 微调而来，登记 `label_quality=coarse_runtime_seeded_manual`、`fine_corner_truth=false`，训练用途按既有拆分保留，但细角点精度只能按近似值解释，不当作精细真值或盲测。状态 registered_not_trained：本次仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包。场景族由 v4 追加为 v5，11 场共 162 页（93 训练 /36 开发 /33 已查看回归）；与既有标注无精确字节 /路径 /强 dHash 重复，无 pHash ≤8 标记。详见 [场景族 v5](data/curated/scenes/202610-shanghai-embodied-intelligence_v5/README.md) 与 [周浩杰](data/curated/scenes/202610-embodied-intelligence-zhouhaojie_v1/README.md)、[姚志强](data/curated/scenes/202610-embodied-intelligence-yaozhiqiang_v1/README.md)、[马战凯](data/curated/scenes/202610-embodied-intelligence-mazhankai_v1/README.md)、[刘意杨](data/curated/scenes/202610-embodied-intelligence-liuyiyang_v1/README.md)。
<!-- embodied-v5-dataset:end -->

<!-- embodied-v5-round037:start -->
- 上海 IN³C 四场 39 页（22 训练页）已在 round_037 从正式 round_036 训练：control 回放 4718 条、with_new 加入 22 页×24 共 5246 条（每轮 164 次更新，12 轮粗定位 + 16 轮精修，seed=7），只按 9 页新场景开发的唯一角点指标选 epoch 并先冻结再回归。选中 with_new，候选 `screen_corner_single_9b9e9c200c49` 未晋升：新场景开发 24→30/36、已查看回归 27→28/32；旧 432 页开发 1494→1499、216 页已查看回归 773→773；冻结 1808 页诊断 6846→6865（+19），逐页只有 +1/0/-1，唯一回退集合为中国智慧道路照明大会 3 月 19 日 -1；2493 页全量 9362→9413（+51，旧域 +15）。正式 runtime 与 0.2.24 安装包不变，未重跑晋升审计或打包。限制：四场人工框由当前 runtime 候选微调、基本定位对但角点不够精细（`coarse_runtime_seeded_manual`），9 页开发仅 36 角点，单 seed，精修阶段在新页面无净增益，收益主要来自粗定位头，不能宣称精细精度已解决。详见 [综合报告](research/experiments/2026-10-24-embodied-v5-training.md) 与 [回归对比](research/experiments/2026-10-24-embodied-v5-round037-regression/README.md)。
<!-- embodied-v5-round037:end -->

<!-- round037-promotion-v0225:start -->
- 按用户要求把 round_037 四场加入候选晋升为正式单文件 runtime `screen_corner_single_9b9e9c200c49`，发布编号 `model-20261023-182931-9b9e9c200c49`，旧权重备份 `models/runtime/screen_corner_single.pre_round036.pt`。同图重跑正式 round_036 与候选：478 页开发 1640→1650（+10），262 页已查看回归 913→917（+4），冻结 1808 页诊断 6846→6865（+19），全部 2275 页诊断 8599→8614（+15），新增 218 页 763→799（+36）；四场 9 页开发 24→30/36、8 页已查看回归 27→28/32。冻结 13 个集合 8 升 /4 平 /1 降，唯一回退为中国智慧道路照明大会 3 月 19 日 -1，逐页只有 +1/0/-1，无单页丢 3 角。0.2.25 macOS Apple Silicon 本地安装包已验收：DMG SHA256 `719e91c177a499adade72e253157b85dd9fb03b921cd7eb90519ef385ad430e8`，Finder 布局与 0.2.16 基准一致、版本 0.2.25、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_9b9e9c200c49`；本地 ad-hoc、未公证、未对外发布，包内无 updater 产物，不构成更新通道。限制保留：四场框由 runtime 候选微调、粗精度，9 页开发仅 36 角点，单 seed，精修在新页面无净增益。详见 [晋升审计](training/runs/screen_corner_single/round_037/reports/promotion-audit/README.md) 与 [综合报告](research/experiments/2026-10-24-embodied-v5-training.md)。
<!-- round037-promotion-v0225:end -->

<!-- zhangkaiyuan-dataset-v1:start -->
- 上海 IN³C 2026 场次「多模态大模型与云边协同赋能油气机器人智能巡检-张凯源」11 页已接入 `202610-embodied-intelligence-zhangkaiyuan`（6 训练 /3 开发 /2 已查看回归，9 个内容组），目标为宽幅 LED 舞台屏完整屏幕四边形，排除两侧品牌固定区。**关键限制**：11 张人工框全部由当前正式 runtime `screen_corner_single_9b9e9c200c49` 候选微调，逐页最大调整量仅 0.40%–1.08% 对角线，登记 `label_quality=runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`；它们不是独立人工真值，当前模型在这批页面的 1% 命中不能当盲测或精细精度证据。与既有全部人工标注无精确字节 /路径 /强 dHash /pHash ≤8 重复。状态 registered_not_trained：仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包；场景族由 v5 追加为 v6，12 场共 173 页（99 训练 /39 开发 /35 已查看回归）。详见 [场景说明](data/curated/scenes/202610-embodied-intelligence-zhangkaiyuan_v1/README.md) 与 [场景族 v6](data/curated/scenes/202610-shanghai-embodied-intelligence_v6/README.md)。
<!-- zhangkaiyuan-dataset-v1:end -->

<!-- sanxiong-dataset-v1:start -->
- 2026-06-10 光亚展「三雄极光」10 页已作为独立展台 LED 屏场景接入 `202606-guangya-sanxiong`（6 训练 /2 开发 /2 已查看回归，9 个内容组），目标为展台 LED 屏完整四边形，排除屏幕外展台结构，顶部品牌条与页脚按人工点保留。标注由当前正式 runtime `screen_corner_single_9b9e9c200c49` 候选微调，逐页调整 1.22%–5.56% 对角线，登记 `label_quality=runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`；包含人工修正但仍非独立真值。与既有全部人工标注无精确字节 /路径 /强 dHash /pHash ≤8 重复。状态 registered_not_trained；不加入上海 IN³C 场景族。详见 [场景说明](data/curated/scenes/202606-guangya-sanxiong_v1/README.md)。
<!-- sanxiong-dataset-v1:end -->

<!-- fine-local-round038:start -->
- 张凯源 11 页与三雄极光 10 页接入后，round_038 按“粗定位基本可以、精细定位略差”从正式 round_037 冻结 encoder/粗定位头，只训练精修头：control 5246 条、with_new 追加 12 页×24 共 5534 条，24 轮、每轮 173 次更新、seed=7，只按 5 页新场景开发（20 角点）选 epoch 并先冻结再回归。新场景开发 13→18/20、已查看回归 9→13/16；三雄极光开发 1→6/8、回归 1→5/8，张凯源保持 20/20；旧 432 页开发 +5、216 页回归持平；2514 页全量 +24，冻结 1808 页 -4（3 升 /5 平 /5 降，最大 -2，逐页无丢 3 角）。候选 `screen_corner_single_31adf6eec801` 未晋升，正式 runtime 与 0.2.25 安装包不变。限制：5 页开发仅 20 角点、单 seed、标签为 runtime 候选微调（三雄 1.22%–5.56%、张凯源 0.40%–1.08%），三雄残余最大单角误差 2.82%。详见 [综合报告](research/experiments/2026-10-24-fine-local-round038.md) 与 [回归对比](research/experiments/2026-10-24-fine-local-round038-regression/README.md)。
<!-- fine-local-round038:end -->

<!-- round038-promotion-v0226:start -->
- 按用户要求把 round_038 精细定位候选晋升为正式单文件 runtime `screen_corner_single_31adf6eec801`，发布编号 `model-20261023-191338-31adf6eec801`，旧权重备份 `models/runtime/screen_corner_single.pre_round037.pt`。同图：483 页开发 1663→1671（+8）、266 页已查看回归 926→930（+4）、新增 239 页 +22、旧域 2275 页 +2、冻结 1808 页 -4（3 升 /5 平 /5 降，最大 -2，逐页无丢 3 角）；本次两场开发 13→18/20、回归 9→13/16（三雄 1→6/8、1→5/8；张凯源保持 20/20）。0.2.26 macOS Apple Silicon DMG SHA256 `d3933635dc9566afcdac0b76bf06edf1a4a4de10b368d68c1358097b4657acd3`，Finder 布局与 0.2.16 一致、ad-hoc 签名、包内权重摘要一致、真实检测返回新模型；本地 ad-hoc、未公证、无 updater 产物。限制：5 页开发仅 20 角点、单 seed、标签为 runtime 候选微调，三雄残余最大单角误差 2.82%。详见 [晋升审计](training/runs/screen_corner_single/round_038/reports/promotion-audit/README.md) 与 [综合报告](research/experiments/2026-10-24-fine-local-round038.md)。
<!-- round038-promotion-v0226:end -->
