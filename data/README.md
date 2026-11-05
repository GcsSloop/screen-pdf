# Data

`data` 负责数据整理，不放程序或运行时模型。

数据集总览：[去重索引](../docs/status/dataset-index.md) 当前为 **53 个数据集单元**（1 会议合集 + 39 独立场景 + 11 历史专项 + 1 组合场景 + 1 修订场景），单元内共 3,703 条人工标注页，按图像 SHA / 会议页映射去重后为 **2,649 张唯一人工图像**；跨单元重叠 1,054 页。

## 从会议场次集合开始

主入口：[conference_v1](curated/collections/conference_v1/README.md)。覆盖展会目录当前 48 个源项目、1950 页，生成 13 个较大的 ScreenPDF 项目；道路照明大会 3 月 19 日、20 日依用户分类分开，商场 / AWE / 扬州保留为专项测试。合并项目排除 4 个冲突 / 越界异常页面后共 1946 页，保留其余原页面及原候选，图片是独立副本，原资料保持原位。

1808 页有可用显式人工框，进入同一冻结输入的 [全量基准](curated/collections/conference_v1/BASELINE.md)。138 页缺少显式人工框、2 页人工框无效、1 张重复照片的 2 份框冲突，逐项排除并记录；[人工校正包](curated/collections/conference_v1/conflict-review/README.md) 含叠框、原尺寸角点放大、来源和可编辑项目。

合集用于查看；机器标注与基准继续以原项目显式人工框为准。编辑合集不会自动回写源项目或冻结报告，后续采用合集校正时需核对来源并生成新修订。用途继承 round_010 定向实验数据快照，不重分训练 / 开发 / 已查看回归。快照外页面不等于盲测，也不会自动纳入训练。

<!-- smart-road-3-20-fine-labels:start -->
中国智慧道路照明大会 3 月 20 日 59 页已做精细标注修订：新标注 `curated/annotations/20260320-smart-road-lighting_pages_v2.jsonl`、新拆分 `splits/cross_project/20260320-smart-road-lighting_v2.json`，并生成训练快照 `training/datasets/screen_corner_single_all_manual_3_20fine_v1/`（train 5534→5535、all_manual 2275→2276）。冻结 `conference_v1`、旧快照和用途不回写；状态 registered_not_trained。详见 [报告](../research/experiments/2026-10-24-smart-road-lighting-fine-labels.md)。
<!-- smart-road-3-20-fine-labels:end -->

<!-- qinghui-round039:start -->
2025-12-08 清晖 PgMP 同学会 116 页已接入独立会议室投影屏场景 `202512-qinghui-alumni_v1`（69 训练 / 24 开发 / 23 已查看回归，92 个内容组；`IMG_5999`、`IMG_6001`、`IMG_6006` 无项目页不纳入）。人工框由当前 runtime 候选微调，非独立真值。round_039 已在 base `screen_corner_single_all_manual_3_20fine_v1` 上完成训练：清晖开发 84→90/96、已查看回归 77→85/92、全部 161→175/188；原 483 页开发 1671→1665/1932，合并开发持平。候选 `screen_corner_single_049cd9cdf0de` 未晋升。详见 [报告](../research/experiments/2026-10-24-qinghui-alumni-training.md)。
<!-- qinghui-round039:end -->

<!-- openatom-202511-dataset:start -->
2025-11-03 开放原子开源基金 6 页已接入独立会议室投影屏场景 `202511-openatom-open-source-fund_v1`（4 训练 / 1 开发 / 1 已查看回归，5 个内容组）。观众头部 /肩部沿屏幕下沿遮挡，桌面与椅背占据下半部；人工四角按屏幕物理边框外推到被遮挡底边，不沿人头收边。6 张框由当前 runtime 候选微调 1.05%–5.85%，非独立真值；无重复标注。开发 / holdout 各仅 1 页，本 revision 只登记训练数据，未训练。去重索引更新为 53 个单元 / 2,642 张唯一人工图像。
<!-- openatom-202511-dataset:end -->

<!-- zhongjiao-corner-revision:start -->
2026-03-06 中交机电点位重矫正 v2：20 页 /19 人工框，18 有效页 16 训练 /2 开发 /0 回归；旧 12 页点位变化 0.13%–0.80%，7 页新标注。**8 页屏幕角超出镜头范围**，人工点取屏幕边与图像边界交点，位于画面边框上，不是物理角点；不能按越界框过滤。`IMG_7018` 与 `202603-white-mix` 同图冲突继续排除，`IMG_7022` 无人工框。新快照 `training/datasets/screen_corner_single_all_manual_3_20fine_zhongjiao_v1`：train 5535→5575、all_manual 2276→2286，validation/holdout 不变；未训练。去重索引 53 单元 / 3,703 单元内人工页 / 2,649 唯一人工图像。
<!-- zhongjiao-corner-revision:end -->

屏幕外观、黑底、边框、遮挡等交叉场景继续在 `curated/scenes/` 管理，它们是诊断视图，同一照片不因多标签在会议总数中重复计数。新增资料、冲突处理、建集和评测命令见 [集合约定](../docs/status/conference-collections.md)。历史逐场接入与固定拆分入口见 [训练记录](../training/README.md) 和 [状态](../docs/status/current-status.md)，不再在此重复维护各轮分数。

## 目录约定

- `raw/`
  - 原始数据入口。
  - 保留你的使用习惯，不强制改目录层级，不强制重命名。
  - 建议直接按你现有项目目录结构放入，例如会议名 / 日期 / 主题 / 图片与项目文件。
  - 这里的数据默认视为只读源数据，不做覆盖式清洗。
- `staging/`
  - 原始数据导入后的中间整理区。
  - 用于补齐 manifest、抽取元信息、识别哪些页面有人手工修正过。
- `curated/`
  - 可复用的结构化数据。
  - `collections/` 按会议 / 场次聚合，是日常查看和比较的主入口。
  - `projects/` 按项目聚合。
  - `scenes/` 按场景聚合，例如白底、近色背景、人物遮挡、黑边、反射、强透视。
  - `annotations/` 放标准化标注、点位、可见性、来源说明。
- `splits/`
  - 训练集 / 验证集 / holdout 的正式拆分。
  - `holdout/` 保留严格不进训练的验证数据。
  - `cross_project/` 保留跨项目泛化验证拆分。
- `derived/`
  - 由原始或清洗数据派生出的任务数据。
  - `global_corner/` 全局四角点任务数据。
  - `local_corner/` 局部角点精修任务数据。
  - `ocr_overlay/` OCR 透明文字叠层相关数据。
  - `export_preview/` 导出预览、可视化回归样例。
- `benchmarks/`
  - 固定评测集及其汇总结果索引。

## 原始数据原则

`raw/` 的核心目标不是“整齐”，而是“忠实保留你真实使用和标注后的原貌”。

这意味着：

- 原始项目名、会议目录名、中文目录名都可以直接保留。
- 如果同一批图片里有你手动修正后的项目文件，也保留在原目录旁边。
- 训练脚本应从 `raw/` 读取，再把标准化结果写入 `staging/` 或 `curated/`，不要反写原始目录。

## 推荐数据流

1. 新增会议或项目资料，先放到 `raw/`
2. 导入脚本扫描 `raw/`，把元信息整理到 `staging/`
3. 标注抽取与清洗后写入 `curated/`
4. 训练/验证拆分写入 `splits/`
5. 任务派生数据写入 `derived/`
6. 固定回归评测结果写入 `benchmarks/`


## 孔令敏资料新接入

`curated/scenes/202610-tunnel-lighting-konglingmin_v1/` 固定 39 张显式人工框（23 训练 / 8 开发 / 8 已查看回归），对应 round_021。杭州右侧 PPT 场景族新修订 `202610-hangzhou-tunnel-partial-ppt_v2/` 共 210 页；既有拆分及 conference_v1 冻结基准保留，孔令敏不包含在该旧基准中。

<!-- zhangwenju-round022:start -->
张文居 48 张资料已按 28 训练 / 10 开发 / 10 已查看回归接入，[场景族 v3](curated/scenes/202610-hangzhou-tunnel-partial-ppt_v3/manifest.json)现共 9 场、258 页。旧场景用途与冻结会议集保留，训练对照见[round_022 报告](../research/experiments/2026-10-18-zhangwenju-partial-screen-training.md)。
<!-- zhangwenju-round022:end -->

<!-- duchao-dataset-v1:start -->
杜超资料初次登记为 8 训练 / 3 开发 / 3 已查看回归，原登记快照保留；随后已在 round_023 的 with_hangzhou 与 family_focus 两组执行训练。选中加入两场新资料的 with_hangzhou；提高同族采样比例的分支未超过它。 详见 [完整报告](../research/experiments/2026-10-18-gengchiyuan-family-focus-training.md)。
<!-- duchao-dataset-v1:end -->

<!-- gengchiyuan-round023:start -->
## 2026-10-18 耿驰远 / 杜超 round_023 已完成

耿驰远 36 张已接入（22 训练 / 7 开发 / 7 已查看回归），杜超 14 张沿用 8 / 3 / 3。round_023 从 round_022 候选 c2267f9e0d71 初始化，完成 control / with_hangzhou / family_focus 三组固定预算对照；各组 seed=7、4/8 轮、每轮 103 次更新，训练采样 2253 / 2433 / 3291 条。合并开发 381 页：initial=1397/1524, control=1404/1524, with_hangzhou=1408/1524, family_focus=1404/1524，候选 `8e554a8d68b7`。选中加入两场新资料的 with_hangzhou；提高同族采样比例的分支未超过它。同族开发 210/256 → 228/256，已查看回归 209/252 → 230/252；原 265 页开发 993/1060 → 988/1060。场景族 v5 共 11 场、308 页；143 张实际采样的同族训练页在定向组提高至 12 倍，刘奇/孙泉明未采样训练页没有借此引入。旧用途、旧 revision、冻结会议集及源人工点保留；单种子、反复使用开发集和残留失败的限制继续保留。正式 runtime 保持 round_021，未晋升或打包。

[完整报告](../research/experiments/2026-10-18-gengchiyuan-family-focus-training.md)。
<!-- gengchiyuan-round023:end -->

<!-- nanjing-round025:start -->
## 2026-10-22 南京主论坛（上）round_025 已完成，候选未晋升

南京第十届未来网络大会主论坛（上）42 页显式人工框已接入独立中央 PPT 场景 `202610-nanjing-future-network-main-am_v1`（24 训练 /9 开发 /9 已查看回归），排除两侧人物直播区；另 4 张未列入源项目的照片不纳入。round_025 从 round_024 做固定预算 teacher_guard 对照，2775 /2919 条采样，共享 406 页开发，选中 `with_nanjing`：`screen_corner_single_c7f83295c7e7`。pooled 开发 initial=1468/1624、control=1477/1624、加入组=1478/1624；南京开发 2→12/36、已查看回归 7→15/36。旧 397 页开发同为 1466/1588，原始 265 页少 9 点；远景上角仍误吸吊顶、中央边界与遮挡失败保留。单 seed，本轮未重跑冻结全量基准，候选未晋升，正式 runtime 仍为 round_024。

[训练报告](../research/experiments/2026-10-22-nanjing-main-forum-training.md)。
<!-- nanjing-round025:end -->

<!-- nanjing-staged-round026-027:start -->
## 2026-10-22 南京场景先粗定位、再精修已完成

南京中央 PPT 按用户要求先粗后细：round_026 冻结 encoder /local head，12 轮粗定位对照选中 coordinate_guard，pooled 粗定位 1302→1312/1624，南京开发 8→11/36，真实角点进入局部裁剪 19→23/36；空间交叉熵分支退化未采用。round_027 冻结 encoder /global head，16 轮精修对照选中 point_hit，第 4 轮单文件候选 `screen_corner_single_eef7e8e646aa`。南京最终开发 12→15/36、已查看回归 15→19/36；合并开发 1478→1486/1624，原 397 页开发 1466→1471/1588、原 265 页 981→992/1060。两阶段分别冻结选择后才评回归；保留远景上角误吸吊顶、中央区域定位失败、单 seed 与已查看回归限制。32 项测试、组件冻结摘要和隔离单权重真实检测通过，训练完成时尚未重跑 1808 页冻结全量基准或晋升；后续已按用户授权完成审计并晋升 round_027，见最新晋升记录。

[训练报告与分阶段叠图](../research/experiments/2026-10-22-nanjing-coarse-then-fine-training.md)。
<!-- nanjing-staged-round026-027:end -->

<!-- round027-promotion-v0221:start -->
## 2026-10-22 round_027 已晋升，0.2.21 本地安装包已验收

已按用户“整体退化不严重则晋升并重新打包”的授权，将 round_027 晋升为正式单文件 runtime `screen_corner_single_eef7e8e646aa`，模型发布编号 `model-20261022-020033-eef7e8e6`。与正式 round_024 重新推理冻结 1808 页：6697→6686/7232，下降约 0.15 个百分点（改善 72、回退 83 点）；当前 406 页开发 1468→1486/1624，166 页已查看回归 601→616/664，原 397 页开发 1466→1471/1588。快照外少 12/1068，杭州隧道集合少 7/244、北京 InfoComm 少 5/80，局部回退继续保留；全集含训练页，仅作覆盖/拟合诊断。南京开发 2→15/36、已查看回归 7→19/36（相对正式 round_024）。0.2.21 macOS Apple Silicon 本地安装包已重新生成，最终 DMG 的严格签名、Finder 布局与 Applications 拖放入口、唯一权重摘要及包内真实检测均通过；本地 ad-hoc 签名，未公证或外部发布，不含 updater 产物。

[整体回归审计](../training/runs/screen_corner_single/round_027/reports/promotion-audit/README.md) · [下载 0.2.21（macOS Apple Silicon）](../release-assets/local-single-20261022-v0221-round027/0.2.21/macos/ScreenPDF_0.2.21_aarch64.dmg)。DMG SHA256：`31fe5df6c1169d3e94497e6ecf42bb0d2e782bd38e0bb5a85364e4dd8fe65ee2`。
<!-- round027-promotion-v0221:end -->

<!-- nanjing-pm-round028:start -->
## 2026-10-22 南京主论坛（下）round_028 已完成，候选未晋升

南京主论坛（下）40 页显式人工框已加入 `202610-nanjing-future-network-main-pm_v1`（24 训练 /8 开发 /8 已查看回归），中央 PPT 为目标，排除两侧人物直播区，保留摄像摇臂 /观众遮挡；另 5 张未列入项目照片不纳入。上下半场同族索引共 82 页，旧用途不回写。round_028 从正式 round_027 继续，control=3351、加入组=3927 条采样（新 24 页各 24 倍），相同 seed=7、每轮 123 次更新，先 12 轮粗定位再 16 轮精修。加入组粗定位 pooled 未胜初始，保留 epoch=0；新增页仍实际参与粗定位训练与最终精修，精修第 10 轮选中 `with_nanjing_pm`，候选 `screen_corner_single_a09836ddb464`。414 页开发 initial=1496/1656、control=1500/1656、加入组=1501/1656。下半场开发 10→14/32、已查看回归 8→11/32；上半场开发 15→17/36、回归 19→21/36。原 265 页开发少 2 点，旧新 174 页回归 624→622/696（旧 166 页少 5 点）；收益较小且残留粗定位 /摇臂干扰限制保留。29 项测试、组件冻结与缓存 /完整图一致性、隔离单权重真实检测通过。候选未晋升，未重跑全量晋升审计，正式 runtime 和安装包继续 round_027 /0.2.21。

[训练报告与叠框](../research/experiments/2026-10-22-nanjing-main-pm-training.md)。
<!-- nanjing-pm-round028:end -->

<!-- computing-round029:start -->
南京中央 PPT [场景族 v2](curated/scenes/202610-nanjing-future-network-central-ppt_v2/README.md) 追加智算论坛 66 页，共三场 148 页（87 训练 /31 开发 /30 已查看回归）。继承原用途，完整人工四角与底角遮挡保留；旧 v1 和冻结会议集不回写。
<!-- computing-round029:end -->

<!-- nanjing-edge-dataset-v1:start -->
南京中央 PPT [场景族 v3](curated/scenes/202610-nanjing-future-network-central-ppt_v3/README.md) 追加边缘算力与泛在智能创新发展论坛 67 页，共四场 215 页（126 训练 /44 开发 /45 已查看回归）。该场状态 registered_not_trained，本次仅接入数据；完整人工四角、底角遮挡与固定拆分保留，旧 v1 /v2 和冻结会议集不回写。
<!-- nanjing-edge-dataset-v1:end -->

<!-- embodied-ai-chain-dataset-v1:start -->
上海 IN³C 2026 具身智能与边缘计算[场景族 v1](curated/scenes/202610-shanghai-embodied-intelligence_v1/README.md) 登记两个场次共 30 页：10-13 工业具身智能 26 页（已随 round_024 训练）+ 10-14 AI铸链 向实而生 4 页（registered_not_trained）。各自拆分保留，旧用法与冻结会议集不回写。
<!-- embodied-ai-chain-dataset-v1:end -->

<!-- embodied-occluded-dataset-v1:start -->
上海 IN³C 2026 [场景族 v2](curated/scenes/202610-shanghai-embodied-intelligence_v2/README.md) 追加华为 AI 6 页与龚涛 40 页，四场共 76 页（44 训练 /17 开发 /15 已查看回归）。两场遮挡较大，人工四角外推到被遮挡的完整屏幕、不沿讲者/人头/讲台收边；均为 registered_not_trained。旧 v1、工业具身智能 split、round_024 快照与冻结会议集不回写。
<!-- embodied-occluded-dataset-v1:end -->

<!-- pm-cube-dataset-v1:start -->
2024「新任产品经理管理魔方」15 页已接入（9 训练 / 3 开发 / 3 已查看回归，`202404-pm-management-cube`），目标为投影白底幻灯片内容区，参考模型误差约 1–5%；round_034 仅训练本地精修头做固定预算对照，本场开发 4/12→11/12、回归 2/12→10/12，候选 `screen_corner_single_9eb8bc25c145` 未晋升，正式 runtime 与 0.2.22 安装包不变。详见 [场景与审计](curated/scenes/202404-pm-management-cube_v1/README.md) 与 [训练报告](../research/experiments/2026-10-22-pm-cube-fine-localization.md)。
<!-- pm-cube-dataset-v1:end -->

<!-- combined-new-data-round035:start -->
五批新增数据（南京边缘算力 67 页、AI铸链 4 页、华为 AI 6 页、龚涛 40 页、2024 PM 管理魔方 15 页，132 页 / 77 训练页）已在 round_035 合并做固定预算对照：新场景开发 65→94/108、回归 66→81/112，旧 432 页开发 -35 角点；候选 `screen_corner_single_5be1a6da1dd2` 未晋升，正式 runtime 与 0.2.22 安装包不变。详见 [训练报告](../research/experiments/2026-10-23-combined-new-data-training.md)。
<!-- combined-new-data-round035:end -->

<!-- qintong-dataset-v1:start -->
上海 IN³C 2026 [场景族 v3](curated/scenes/202610-shanghai-embodied-intelligence_v3/README.md) 追加秦通场 33 页，五场共 109 页（63 训练 /24 开发 /22 已查看回归）。该场为 registered_not_trained，仅接入数据；旧 v1 /v2、工业具身智能 split 与冻结会议集不回写。
<!-- qintong-dataset-v1:end -->

<!-- zhufeng-luchen-dataset-v1:start -->
上海 IN³C 2026 [场景族 v4](curated/scenes/202610-shanghai-embodied-intelligence_v4/README.md) 追加朱峰 4 页与陆晨 10 页，七场共 123 页（71 训练 /27 开发 /25 已查看回归）。两场均为 registered_not_trained；于青民 6 张无人工框未纳入。旧 revision、旧用途与冻结会议集不回写。
<!-- zhufeng-luchen-dataset-v1:end -->

<!-- embodied-v5-dataset:start -->
上海 IN³C 2026 [场景族 v5](curated/scenes/202610-shanghai-embodied-intelligence_v5/README.md) 追加周浩杰 6 页（3 /2 /1）、姚志强 4 页（2 /1 /1）、马战凯 15 页（9 /3 /3，1 张 needs_review 无人工框排除）、刘意杨 14 页（8 /3 /3），11 场共 162 页（93 训练 /36 开发 /33 已查看回归）。四场均为 registered_not_trained；人工框由当前 runtime 候选微调，登记 `coarse_runtime_seeded_manual`（基本定位准确、角点不够精细），训练用途按既有拆分保留，但细角点精度按近似值解释，不当作精细真值或盲测。旧 revision、旧用途与冻结会议集不回写。
<!-- embodied-v5-dataset:end -->

<!-- zhangkaiyuan-dataset-v1:start -->
上海 IN³C 2026 [场景族 v6](curated/scenes/202610-shanghai-embodied-intelligence_v6/README.md) 追加张凯源「多模态大模型与云边协同赋能油气机器人智能巡检」11 页（6 训练 /3 开发 /2 已查看回归，9 个内容组），12 场共 173 页（99/39/35）。该场为 registered_not_trained；11 张框全部由当前正式 runtime `screen_corner_single_9b9e9c200c49` 候选微调（逐页最大调整 0.40%–1.08% 对角线），登记 `annotation_independence=false`、`fine_corner_truth=false`，不是独立人工真值，不能作为盲测或精细精度证据。旧 revision、旧用途与冻结会议集不回写。
<!-- zhangkaiyuan-dataset-v1:end -->

<!-- sanxiong-dataset-v1:start -->
2026-06-10 光亚展「三雄极光」10 页作为独立展台 LED 屏场景接入 `202606-guangya-sanxiong`（6 训练 /2 开发 /2 已查看回归，9 个内容组），registered_not_trained；不加入上海 IN³C 场景族。标注由当前 runtime 候选微调（1.22%–5.56% 对角线），登记 `annotation_independence=false`、`fine_corner_truth=false`，不是独立人工真值。详见 [场景说明](curated/scenes/202606-guangya-sanxiong_v1/README.md)。
<!-- sanxiong-dataset-v1:end -->
