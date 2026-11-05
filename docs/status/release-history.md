# 更新日志归档

原 README 底部逐轮累积的训练 / 数据接入记录已移到这里，按 round 顺序排列（round_012 → round_038），内容与数值保持当时记录、不回填。

- 更早的三阶段 runtime、round_001–round_020 与各场景首次接入记录散见 [research/experiments](../../research/experiments/)；一页速查见根目录 [AGENTS.md](../../AGENTS.md)。
- round_038 之后的模型晋升、候选回落与数据接入按时间倒序整理在 [current-status.md](current-status.md)，那里也是后续追加的主入口。
- 下列链接指向的 `research/experiments`、`training/runs`、`release-assets` 属于本地资产、不进版本库，在 GitHub 上打不开，但路径仍可用于本机溯源。
- 所有分数都是当时实测值：口径以 [评估契约](evaluation-contract.md) 的 `point_le_0_01_ratio` 为准，已查看回归不等于盲测。

---

孙泉明 29 张黑底宽屏 PPT 资料已接入 round_012（17 训练 / 6 开发 / 6 已查看回归），候选 `screen_corner_single_a1f4bec7b047` 在合并开发达到 1231/1344，尚未晋升；保留 LED 条纹、倒计时 / 超时叠层和底部观众遮挡，详见 [训练报告](../../research/experiments/2026-10-18-sunquanming-black-screen-training.md)。

李薇 24 张宽幅 LED 右侧 PPT 资料已接入 round_013（14 训练 / 5 开发 / 5 已查看回归），候选 `screen_corner_single_5ba8b4a37a5d` 在合并开发达到 1236/1340，已晋升；保留黑色屏幕背景、LED 条纹、倒计时 / 超时叠层和底部观众遮挡，详见 [训练报告](../../research/experiments/2026-10-18-liwei-partial-screen-training.md)。

李华斌 17 张宽幅 LED 右侧 PPT 资料已接入 round_014（9 训练 / 4 开发 / 4 已查看回归），候选 `screen_corner_single_2343521e5471` 在合并开发达到 1241/1356，尚未晋升；保留蓝色和深色屏幕背景、LED 条纹、倒计时叠层和底部观众遮挡，详见 [训练报告](../../research/experiments/2026-10-18-lihuabin-partial-screen-training.md)。

刘松荣 15 张宽幅 LED 右侧 PPT 资料已接入 round_015（9 训练 / 3 开发 / 3 已查看回归），`with_liusongrong` 与 control 同为 1251/1368，按平局规则选择 control，候选未晋升；新增白底、蓝底、密集表格图表和地图页面，详见 [训练报告](../../research/experiments/2026-10-18-liusongrong-partial-screen-training.md)。

涂耘资料目录已更名为 `2026-10-16 隧道照明与绿色能源的深度融合之路-涂耘`，同名导出资料及训练/原图引用同步更新，原始人工点位、拆分和模型不变。见[更名审计](../../research/experiments/path-migrations/20261017-tuyun-date/README.md)。

## 孔令敏资料 round_021

39 张已接入（23 训练 / 8 开发 / 8 已查看回归），相同 teacher_guard 的固定预算对照选中单文件候选 `7279848d9388`。合并开发 1292→1303/1420，本场开发 13→23/32、已查看回归 11→24/32；右下角仍失败较多，旧 265 页开发少 3 点，训练完成后按用户要求晋升，正式 runtime 为 round_021。详见[训练报告](../../research/experiments/2026-10-18-konglingmin-fine-corner-training.md)。

[下载本地 0.2.20（macOS Apple Silicon）](../../release-assets/local-single-20261018-v0220-round021/0.2.20/macos/ScreenPDF_0.2.20_aarch64.dmg)：round_021 正式模型，单权重与真实检测、签名、Finder 拖放布局均已验收；本地 ad-hoc 签名，未公证或外部发布。

<!-- zhangwenju-round022:start -->
## 张文居资料 round_022

48 张已接入（28 训练 / 10 开发 / 10 已查看回归）；相同预算对照选中 `control`，候选 `c2267f9e0d71`。合并开发 initial=1335/1460, control=1351/1460, with_zhangwenju=1349/1460；本场开发 32→35/40、已查看回归 33→33/40。正式 runtime 保持 round_021。详见[训练报告](../../research/experiments/2026-10-18-zhangwenju-partial-screen-training.md)。
<!-- zhangwenju-round022:end -->

<!-- duchao-dataset-v1:start -->
杜超资料初次登记为 8 训练 / 3 开发 / 3 已查看回归，原登记快照保留；随后已在 round_023 的 with_hangzhou 与 family_focus 两组执行训练。选中加入两场新资料的 with_hangzhou；提高同族采样比例的分支未超过它。 详见 [完整报告](../../research/experiments/2026-10-18-gengchiyuan-family-focus-training.md)。
<!-- duchao-dataset-v1:end -->

<!-- gengchiyuan-round023:start -->
## 2026-10-18 耿驰远 / 杜超 round_023 已完成

耿驰远 36 张已接入（22 训练 / 7 开发 / 7 已查看回归），杜超 14 张沿用 8 / 3 / 3。round_023 从 round_022 候选 c2267f9e0d71 初始化，完成 control / with_hangzhou / family_focus 三组固定预算对照；各组 seed=7、4/8 轮、每轮 103 次更新，训练采样 2253 / 2433 / 3291 条。合并开发 381 页：initial=1397/1524, control=1404/1524, with_hangzhou=1408/1524, family_focus=1404/1524，候选 `8e554a8d68b7`。选中加入两场新资料的 with_hangzhou；提高同族采样比例的分支未超过它。同族开发 210/256 → 228/256，已查看回归 209/252 → 230/252；原 265 页开发 993/1060 → 988/1060。场景族 v5 共 11 场、308 页；143 张实际采样的同族训练页在定向组提高至 12 倍，刘奇/孙泉明未采样训练页没有借此引入。旧用途、旧 revision、冻结会议集及源人工点保留；单种子、反复使用开发集和残留失败的限制继续保留。正式 runtime 已按用户要求晋升为 round_023：`screen_corner_single_8e554a8d68b7`，发布编号 `model-20261018-154105-8e554a8d`，SHA256 `8e554a8d68b7b7651c52f3050a0c7b2e2383a156a2e65f835f1a902683ab6033`。未重跑 1808 页全量基准；旧 265 页开发少 5 点，右上与遮挡右下失败仍保留。

[完整报告](../../research/experiments/2026-10-18-gengchiyuan-family-focus-training.md)。

[下载本地 0.2.20（macOS Apple Silicon）](../../release-assets/local-single-20261018-v0220-round023/0.2.20/macos/ScreenPDF_0.2.20_aarch64.dmg)：round_023 正式模型，单权重、真实检测、签名和 Finder 拖放布局均已验收；本地 ad-hoc 签名，未公证或外部发布。
<!-- gengchiyuan-round023:end -->


<!-- round024-mazhangyixiang-industrial:start -->
## 2026-10-19 马张翼翔 / 工业具身智能 round_024

马张翼翔 47 张（27 训练 / 10 开发 / 10 已查看留出）加入杭州宽屏场景族；工业具身智能 26 张（15 / 6 / 5）作为独立新场景加入。四组固定预算对照选择 `with_both`，pooled 开发 1466/1588；马张翼翔 39/40、40/40，工业具身智能 21/24、20/20。正式 runtime 为 `screen_corner_single_7f2257153f0c`，发布编号 `model-20261018-173619-7f225715`。完整报告见 [训练报告](../../research/experiments/2026-10-19-mazhangyixiang-industrial-training.md)。
<!-- round024-mazhangyixiang-industrial:end -->

<!-- nanjing-round025:start -->
## 2026-10-22 南京主论坛（上）round_025 已完成，候选未晋升

南京第十届未来网络大会主论坛（上）42 页显式人工框已接入独立中央 PPT 场景 `202610-nanjing-future-network-main-am_v1`（24 训练 /9 开发 /9 已查看回归），排除两侧人物直播区；另 4 张未列入源项目的照片不纳入。round_025 从 round_024 做固定预算 teacher_guard 对照，2775 /2919 条采样，共享 406 页开发，选中 `with_nanjing`：`screen_corner_single_c7f83295c7e7`。pooled 开发 initial=1468/1624、control=1477/1624、加入组=1478/1624；南京开发 2→12/36、已查看回归 7→15/36。旧 397 页开发同为 1466/1588，原始 265 页少 9 点；远景上角仍误吸吊顶、中央边界与遮挡失败保留。单 seed，本轮未重跑冻结全量基准，候选未晋升，正式 runtime 仍为 round_024。

[训练报告](../../research/experiments/2026-10-22-nanjing-main-forum-training.md)。
<!-- nanjing-round025:end -->

<!-- nanjing-staged-round026-027:start -->
## 2026-10-22 南京场景先粗定位、再精修已完成

南京中央 PPT 按用户要求先粗后细：round_026 冻结 encoder /local head，12 轮粗定位对照选中 coordinate_guard，pooled 粗定位 1302→1312/1624，南京开发 8→11/36，真实角点进入局部裁剪 19→23/36；空间交叉熵分支退化未采用。round_027 冻结 encoder /global head，16 轮精修对照选中 point_hit，第 4 轮单文件候选 `screen_corner_single_eef7e8e646aa`。南京最终开发 12→15/36、已查看回归 15→19/36；合并开发 1478→1486/1624，原 397 页开发 1466→1471/1588、原 265 页 981→992/1060。两阶段分别冻结选择后才评回归；保留远景上角误吸吊顶、中央区域定位失败、单 seed 与已查看回归限制。32 项测试、组件冻结摘要和隔离单权重真实检测通过，训练完成时尚未重跑 1808 页冻结全量基准或晋升；后续已按用户授权完成审计并晋升 round_027，见最新晋升记录。

[训练报告与分阶段叠图](../../research/experiments/2026-10-22-nanjing-coarse-then-fine-training.md)。
<!-- nanjing-staged-round026-027:end -->

<!-- round027-promotion-v0221:start -->
## 2026-10-22 round_027 已晋升，0.2.21 本地安装包已验收

已按用户“整体退化不严重则晋升并重新打包”的授权，将 round_027 晋升为正式单文件 runtime `screen_corner_single_eef7e8e646aa`，模型发布编号 `model-20261022-020033-eef7e8e6`。与正式 round_024 重新推理冻结 1808 页：6697→6686/7232，下降约 0.15 个百分点（改善 72、回退 83 点）；当前 406 页开发 1468→1486/1624，166 页已查看回归 601→616/664，原 397 页开发 1466→1471/1588。快照外少 12/1068，杭州隧道集合少 7/244、北京 InfoComm 少 5/80，局部回退继续保留；全集含训练页，仅作覆盖/拟合诊断。南京开发 2→15/36、已查看回归 7→19/36（相对正式 round_024）。0.2.21 macOS Apple Silicon 本地安装包已重新生成，最终 DMG 的严格签名、Finder 布局与 Applications 拖放入口、唯一权重摘要及包内真实检测均通过；本地 ad-hoc 签名，未公证或外部发布，不含 updater 产物。

[整体回归审计](../../training/runs/screen_corner_single/round_027/reports/promotion-audit/README.md) · [下载 0.2.21（macOS Apple Silicon）](../../release-assets/local-single-20261022-v0221-round027/0.2.21/macos/ScreenPDF_0.2.21_aarch64.dmg)。DMG SHA256：`31fe5df6c1169d3e94497e6ecf42bb0d2e782bd38e0bb5a85364e4dd8fe65ee2`。
<!-- round027-promotion-v0221:end -->

<!-- nanjing-pm-round028:start -->
## 2026-10-22 南京主论坛（下）round_028 已完成，候选未晋升

南京主论坛（下）40 页显式人工框已加入 `202610-nanjing-future-network-main-pm_v1`（24 训练 /8 开发 /8 已查看回归），中央 PPT 为目标，排除两侧人物直播区，保留摄像摇臂 /观众遮挡；另 5 张未列入项目照片不纳入。上下半场同族索引共 82 页，旧用途不回写。round_028 从正式 round_027 继续，control=3351、加入组=3927 条采样（新 24 页各 24 倍），相同 seed=7、每轮 123 次更新，先 12 轮粗定位再 16 轮精修。加入组粗定位 pooled 未胜初始，保留 epoch=0；新增页仍实际参与粗定位训练与最终精修，精修第 10 轮选中 `with_nanjing_pm`，候选 `screen_corner_single_a09836ddb464`。414 页开发 initial=1496/1656、control=1500/1656、加入组=1501/1656。下半场开发 10→14/32、已查看回归 8→11/32；上半场开发 15→17/36、回归 19→21/36。原 265 页开发少 2 点，旧新 174 页回归 624→622/696（旧 166 页少 5 点）；收益较小且残留粗定位 /摇臂干扰限制保留。29 项测试、组件冻结与缓存 /完整图一致性、隔离单权重真实检测通过。候选未晋升，未重跑全量晋升审计，正式 runtime 和安装包继续 round_027 /0.2.21。

[训练报告与叠框](../../research/experiments/2026-10-22-nanjing-main-pm-training.md)。
<!-- nanjing-pm-round028:end -->

<!-- computing-round029:start -->
智算论坛 66 页已加入 round_029，直接加入组开发粗定位 17→45/56；区域引导实验未胜过直接分支，旧域与精修适配限制保留。候选未晋升，正式模型 /安装包继续 round_027 /0.2.21。[训练报告](../../research/experiments/2026-10-22-computing-forum-region-coarse-training.md)。
<!-- computing-round029:end -->

<!-- all-history-round030:start -->
round_030 已根据全部历史可用人工资料独立训练区域识别 → 粗定位 → 精修单体模型。62 个来源、2275 页，1461 训练 /432 开发 /216 已查看回归，166 张近评估重复训练页隔离。仅通用 ImageNet 初始化，骨干和三个任务阶段重新训练，未继承 ScreenPDF 业务权重或教师。候选 `screen_corner_single_500bb4820b86`；开发最终正式 1496/1728（86.57%） → 新模型 1508/1728（87.27%）（+12 点），已查看回归正式 715/864（82.75%） → 新模型 762/864（88.19%）。南京开发 +33、InfoComm +7，openatom -18、福州高速 -12 等局部回退保留。12 /80 /48 轮固定预算，按 pooled 开发唯一角点指标冻结，先冻结再回归；含训练资料的 1808 页诊断（1 页数据集修订排除页按冻结人工框补算）/2275 页诊断与逐会议变化另报。seed=11 复验（round_031）开发 1522、已查看回归 770，粗定位 1496/1497 复现，final 增益带约 ±0.8 个百分点种子波动；精修净贡献仅 +12/+25（成熟模型 +191），openatom /福州高速 /smart-road /metaverse /商场测试集回退两 seed 复现，故建议先修精修阶段再谈晋升。23 项测试和隔离单文件实际检测通过；两 seed 复验后，固定预算、历史评估已查看等限制仍保留。未晋升 /打包，正式 runtime 仍 round_027。

[完整报告](../../research/experiments/2026-10-22-all-history-region-coarse-local-training.md) · [seed 复验与综合指标](../../research/experiments/2026-10-22-all-history-seed-replication.md)。
<!-- all-history-round030:end -->

<!-- round031-promotion-v0222:start -->
## 2026-10-22 round_031 全历史模型已晋升，0.2.22 本地包已验收

按用户要求把全历史独立训练的单文件模型晋升为正式 runtime `screen_corner_single_052594290a1d`（round_031，seed=11），发布编号 `model-20261022-131533-05259429`，旧权重备份 `models/runtime/screen_corner_single.pre_round027.pt`。与正式 round_027 同图重建推理：432 页开发 1496→1522（86.57%→88.08%），216 页已查看回归 715→770（82.75%→89.12%），原 428 页子集 1423→1432，冻结 1808 页诊断 6686→6882，全部 2275 页诊断 8182→8640；平均归一化误差开发 1.91%→0.76%、回归 2.16%→0.73%。冻结诊断仍有三处回退并写入发布元数据：`202604-mall` -59（疑似把屏幕外边黑框纳入目标）、`202312-openatom` -30、`202603-fuzhou-highway` -13。同轮精修专项 round_032（冻结该基座、只重训局部头 96 轮）开发 1519、回归 773，未超过基座，说明精修瓶颈不是训练轮数，未采用。0.2.22 macOS Apple Silicon 安装包已生成并验收：Finder 布局与 0.2.16 逐字节一致、版本 0.2.22、ad-hoc 严格签名、包内真实检测来源 `runtime_single` 且模型摘要一致；本地 ad-hoc 签名、未公证、未对外发布，包内含本次构建的 updater 签名产物但没有 latest.json，不构成更新通道。

[晋升审计](../../training/runs/screen_corner_single/round_031/reports/promotion-audit/README.md) · [决策记录](../../training/runs/screen_corner_single/round_031/reports/promotion_decision.json) · [下载 0.2.22（macOS Apple Silicon）](../../release-assets/local-single-20261022-v0222-round031/0.2.22/macos/ScreenPDF_0.2.22_aarch64.dmg)。DMG SHA256：`8c35eb39c9709ac14e04b84da73e3a9b9845ff7c42fd4abf9008eed45d355773`。
<!-- round031-promotion-v0222:end -->

<!-- nanjing-edge-dataset-v1:start -->
南京边缘算力与泛在智能创新发展论坛 67 页已按 39 训练 /13 开发 /15 已查看回归接入训练数据（registered_not_trained，尚未启动训练），南京中央 PPT 场景族 v3 共 215 页。见[数据入口](../../data/curated/scenes/202610-nanjing-edge-computing-forum_v1/README.md)。
<!-- nanjing-edge-dataset-v1:end -->

<!-- embodied-ai-chain-dataset-v1:start -->
上海 IN³C 2026 场次「AI铸链 向实而生」4 页已按 2 训练 /1 开发 /1 已查看回归接入训练数据（registered_not_trained，尚未启动训练），与 2026-10-13 工业具身智能 26 页组成[场景族 v1](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v1/README.md)，共 30 页。见[数据入口](../../data/curated/scenes/202610-embodied-intelligence-ai-chain_v1/README.md)。
<!-- embodied-ai-chain-dataset-v1:end -->

<!-- embodied-occluded-dataset-v1:start -->
上海 IN³C 2026 再接入两场（遮挡较大）：华为 AI 6 页（3 /2 /1）与龚涛「工业控制系统从自动化向自主化的演进」40 页（24 /8 /8），均为 registered_not_trained；与工业具身智能 26 页、AI铸链 4 页组成[场景族 v2](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v2/README.md)，共 76 页。见[华为 AI](../../data/curated/scenes/202610-embodied-intelligence-huawei-ai_v1/README.md)、[龚涛](../../data/curated/scenes/202610-embodied-intelligence-gongtao_v1/README.md) 数据入口。
<!-- embodied-occluded-dataset-v1:end -->

<!-- embodied-occlusion-round033:start -->
上海 IN³C 重遮挡两场（华为 AI 6 页 + 龚涛 40 页，27 页训练）已在 round_033 从正式 round_031 做固定预算对照：control 回放 2222 条、with_occluded 加入 27 页×24 共 2870 条，粗定位 12 轮 + 精修 16 轮、每轮 90 次更新，只按 10 页重遮挡开发唯一角点指标选 epoch 并先冻结再回归。选中 with_occluded（加入 27 页），重遮挡开发最终 32/40（80.00%）；候选 `screen_corner_single_4a6904df2cd6` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、10 页开发、人工点外推遮挡边缘与已查看回归限制保留。见[训练报告](../../research/experiments/2026-10-22-embodied-occlusion-training.md)。
<!-- embodied-occlusion-round033:end -->

<!-- pm-cube-round034:start -->
2024「新任产品经理管理魔方」15 页已接入（9 训练 / 3 开发 / 3 已查看回归，投影白底幻灯片，参考模型在这批页面的角点误差约 1–5%、命中仅 16/60）。round_034 从正式 round_031 冻结 encoder 与粗定位头，只训练本地精修头 24 轮做固定预算对照：control 回放 2222 条、with_pm_cube 加入 9 页×24 共 2438 条，只按 3 页精细定位开发唯一角点指标选 epoch 并先冻结再回归。选中 with_pm_cube（加入 9 页），开发最终 11/12（91.67%）；候选 `screen_corner_single_9eb8bc25c145` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、3 页开发、只训本地头的限制保留。见[训练报告](../../research/experiments/2026-10-22-pm-cube-fine-localization.md)。
<!-- pm-cube-round034:end -->

<!-- combined-new-data-round035:start -->
五批新增资料（南京边缘算力 67 页、AI铸链 4 页、华为 AI 6 页、龚涛 40 页、2024 PM 管理魔方 15 页，合计 132 页 / 77 训练页）已在 round_035 从正式 round_031 合并做固定预算对照：control 回放 2222 条、with_new_all 加入 77 页×24 共 4070 条，粗定位 12 轮 + 精修 16 轮、每轮 128 次更新，只按 27 页合并新场景开发唯一角点指标选 epoch 并先冻结再回归。选中 with_new_all（加入 77 页），合并新场景开发最终 94/108（87.04%）；候选 `screen_corner_single_5be1a6da1dd2` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、27 页开发、五场景混合采样与已查看回归限制保留。见[训练报告](../../research/experiments/2026-10-23-combined-new-data-training.md)。
<!-- combined-new-data-round035:end -->

<!-- candidate-scene-comparison-20261023:start -->
四个单文件模型（正式 round_031、round_033 遮挡、round_034 PM 精修、round_035 五批合并）已在同一 2407 页语料（旧 2275 + 新增 132）重新推理做分场景对比：全部页 8951 /8996 /8984 /**9067** 命中；旧 2275 页 8640 /8618 /8632 /8600；新增 132 页 311 /378 /352 /**467**；已查看回归 244 页 836 /832 /**850 /851**。相对正式：round_033 旧 -22 /新 +67 /回归 -4；round_034 旧 -8 /新 +41 /回归 +14；round_035 旧 -40 /新 +156 /回归 +15。要一个 runtime 覆盖全部新增场景选 round_035（`screen_corner_single_5be1a6da1dd2`，需重跑 1808 页冻结审计并重新打包）；要旧域风险最小选 round_034（`9eb8bc25c145`）；round_033 保留为遮挡对照；正式 round_031 仍是旧场景最佳。均未晋升。见[对比报告](../../research/experiments/2026-10-23-candidate-scene-comparison.md)。
<!-- candidate-scene-comparison-20261023:end -->

<!-- round035-promotion-v0223:start -->
按用户要求把 round_035（五批新增资料合并加入）晋升为正式单文件 runtime `screen_corner_single_5be1a6da1dd2`，发布编号 `model-20261023-041021-5be1a6da`，旧权重备份 `models/runtime/screen_corner_single.pre_round031.pt`。晋升审计同图重跑正式 round_031 与候选：459 页开发 1587→1581/1836（-6），244 页已查看回归 836→851/976（+15），冻结 1808 页诊断 6882→6853/7232（-29），全部 2275 页诊断 8640→8600/9100（-40），新增 132 页 311→467/528（+156）。冻结集合回退：福州高速 -12、开放原子 -9、扬州 -3、智慧道路照明 3 月 20 日 -3、元宇宙 -2、杭州隧道集合 -2、追光 -2、杭州智能交通 -2；改善：上海大屏 +2、3 月 19 日 +2、商场 +1、AWE +1，全部写入发布元数据。0.2.23 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `98f42ca7a06321cf8b7246ba51c2c6d12fc770b0dcf13691d4db1f9e4482c8c0`，Finder 布局与 0.2.16 逐字节一致、版本 0.2.23、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_5be1a6da1dd2`；本地 ad-hoc 签名、未公证、未对外发布，包内含 updater 签名产物但没有 latest.json，不构成更新通道。见[晋升审计](../../training/runs/screen_corner_single/round_035/reports/promotion-audit/README.md)与[下载 0.2.23](../../release-assets/local-single-20261023-v0223-round035/0.2.23/macos/ScreenPDF_0.2.23_aarch64.dmg)。
<!-- round035-promotion-v0223:end -->

<!-- qintong-dataset-v1:start -->
上海 IN³C 2026 秦通场 33 页已接入训练数据（19 /7 /7，registered_not_trained），上海 IN³C 场景族 v3 共 109 页。见[场景说明](../../data/curated/scenes/202610-embodied-intelligence-qintong_v1/README.md) 与[场景族 v3](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v3/README.md)。
<!-- qintong-dataset-v1:end -->

<!-- zhufeng-luchen-dataset-v1:start -->
上海 IN³C 2026 朱峰 4 页（2 /1 /1）与陆晨 10 页（6 /2 /2）已接入训练数据（registered_not_trained），场景族 v4 共 123 页；于青民 6 张无人工框未纳入。见[朱峰](../../data/curated/scenes/202610-embodied-intelligence-zhufeng_v1/README.md)、[陆晨](../../data/curated/scenes/202610-embodied-intelligence-luchen_v1/README.md) 与[场景族 v4](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v4/README.md)。
<!-- zhufeng-luchen-dataset-v1:end -->

<!-- occlusion-round036:start -->
秦通 33 /朱峰 4 /陆晨 10 页已接入；round_036 从正式 round_035 做遮挡定向对照：control 回放 4070 条、with_new 加入 27 页×24 共 4718 条、occlusion_focus 再把 54 张遮挡相关页加 24 倍共 6014 条，粗定位 12 轮 + 精修 16 轮、每轮 188 次更新，只按 20 页遮挡开发唯一角点指标选 epoch 并先冻结再回归。选中 with_new（加入 27 页），遮挡开发最终 62/80（77.50%）；候选 `screen_corner_single_7237adbcaa54` 未晋升，正式 runtime 与 0.2.23 安装包不变。单 seed、20 页遮挡开发、加倍采样不是独立样本与已查看回归限制保留。见[训练报告](../../research/experiments/2026-10-24-occlusion-focus-training.md)。
<!-- occlusion-round036:end -->

<!-- regression-compare-round036:start -->
round_036 回归验证：正式 round_035（5be1a6da1dd2）对比候选 `screen_corner_single_7237adbcaa54`，同一 2454 页（旧 2275 + 新增 179）各自重跑推理。全部页 9177→9250（+73），新增 179 页 577→651（+74，80.59%→90.92%），旧 2275 页 8600→8599（-1），冻结 1808 页诊断 6853→6846（-7），469 页开发 1604→1616（+12），254 页已查看回归 874→886（+12）。新增场景：秦通全部页 +37、龚涛 +18、陆晨 +11、朱峰 +5，新场景开发 +5、回归 +9；旧域 13 个集合 5 升 /7 降 /1 平，最差 -3，逐页没有少 3 角的页面。结论：候选达到可晋升量级（新场景 +74、旧域 -1、无整页崩坏）；晋升仍需重跑 1808 页冻结审计并重新打包，正式 runtime 暂不变。见[回归验证报告](../../research/experiments/2026-10-24-round035-vs-round036-regression/README.md)。
<!-- regression-compare-round036:end -->

<!-- round036-promotion-v0224:start -->
按用户要求把 round_036 遮挡定向候选晋升为正式单文件 runtime `screen_corner_single_7237adbcaa54`，发布编号 `model-20261023-165533-7237adbc`，旧权重备份 `models/runtime/screen_corner_single.pre_round035.pt`。同图重跑正式 round_035 与候选：469 页开发 1604→1616（+12），254 页已查看回归 874→886（+12），冻结 1808 页诊断 6853→6846（-7），全部 2275 页诊断 8600→8599（-1），新增 179 页 577→651（+74）。冻结集合回退：InfoComm /商场 /开放原子各 -3，元宇宙 -2，AWE /上海大屏 /道路照明 3.19 各 -1；改善：扬州 +3，道路照明 3.20 /智能交通年会 /福州高速 /追光各 +1；逐页没有少 3 角的页面。0.2.24 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `80103f2d3765ae7d28dd25e70ebbe2a1bcf050b01011de52985cc160ddb4b010`，Finder 布局与 0.2.16 参考一致（同一 .DS_Store / 卷图标摘要）、版本 0.2.24、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_7237adbcaa54`；本地 ad-hoc 签名、未公证、未对外发布，包内含 updater 签名产物但没有 latest.json，不构成更新通道。见[晋升审计](../../training/runs/screen_corner_single/round_036/reports/promotion-audit/README.md)与[下载 0.2.24](../../release-assets/local-single-20261024-v0224-round036/0.2.24/macos/ScreenPDF_0.2.24_aarch64.dmg)。
<!-- round036-promotion-v0224:end -->

<!-- embodied-v5-dataset:start -->
上海 IN³C 2026 四场 2026-10-13 资料已接入训练数据（registered_not_trained）：周浩杰 6 页（3 /2 /1）、姚志强 4 页（2 /1 /1）、马战凯 15 页（9 /3 /3，另 1 张无人工框排除）、刘意杨 14 页（8 /3 /3），共 39 页，场景族 v5 共 162 页。四场人工框均由当前 runtime 候选 `screen_corner_single_7237adbcaa54` 微调，登记 `coarse_runtime_seeded_manual`：基本定位相对准确但角点不够精细，训练用途按既有拆分保留，但细角点精度按近似值解释，不当作精细真值或盲测。见[场景族 v5](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v5/README.md) 与[周浩杰](../../data/curated/scenes/202610-embodied-intelligence-zhouhaojie_v1/README.md)、[姚志强](../../data/curated/scenes/202610-embodied-intelligence-yaozhiqiang_v1/README.md)、[马战凯](../../data/curated/scenes/202610-embodied-intelligence-mazhankai_v1/README.md)、[刘意杨](../../data/curated/scenes/202610-embodied-intelligence-liuyiyang_v1/README.md)。
<!-- embodied-v5-dataset:end -->

<!-- embodied-v5-round037:start -->
上海 IN³C 四场 39 页（22 训练页）已在 round_037 从正式 round_036 训练，选中 with_new：新场景开发 24→30/36、已查看回归 27→28/32；旧 432 页开发 +5、216 页回归持平；冻结 1808 页 +19，逐页无 ±2；2493 页全量 +51。候选 `screen_corner_single_9b9e9c200c49` 未晋升，正式 runtime 与 0.2.24 安装包不变。标签为 runtime 候选微调的粗精度框，9 页开发仅 36 角点，精修在新页面无净增益。见[综合报告](../../research/experiments/2026-10-24-embodied-v5-training.md) 与[回归对比](../../research/experiments/2026-10-24-embodied-v5-round037-regression/README.md)。
<!-- embodied-v5-round037:end -->

<!-- round037-promotion-v0225:start -->
按用户要求把 round_037 四场加入候选晋升为正式 runtime `screen_corner_single_9b9e9c200c49`，发布编号 `model-20261023-182931-9b9e9c200c49`。同图：478 页开发 +10、262 页回归 +4、冻结 1808 +19、全部 2275 +15、新增 218 +36；四场开发 24→30、回归 27→28；冻结 8 升 /4 平 /1 降（3.19 -1），逐页无丢 3 角。0.2.25 macOS Apple Silicon DMG 已验收，SHA256 `719e91c177a499adade72e253157b85dd9fb03b921cd7eb90519ef385ad430e8`，本地 ad-hoc、未公证、无 updater 产物。见[晋升审计](../../training/runs/screen_corner_single/round_037/reports/promotion-audit/README.md)与[下载 0.2.25](../../release-assets/local-single-20261024-v0225-round037/0.2.25/macos/ScreenPDF_0.2.25_aarch64.dmg)。
<!-- round037-promotion-v0225:end -->

<!-- zhangkaiyuan-dataset-v1:start -->
上海 IN³C 2026 张凯源「多模态大模型与云边协同赋能油气机器人智能巡检」11 页已接入训练数据（6 /3 /2，registered_not_trained），场景族 v6 共 173 页。该批 11 张框全部由当前正式 runtime `screen_corner_single_9b9e9c200c49` 候选微调（逐页最大调整 0.40%–1.08% 对角线），不是独立人工真值，不能据此宣称盲测或精细精度改善。见[场景说明](../../data/curated/scenes/202610-embodied-intelligence-zhangkaiyuan_v1/README.md) 与[场景族 v6](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v6/README.md)。
<!-- zhangkaiyuan-dataset-v1:end -->

<!-- sanxiong-dataset-v1:start -->
2026 光亚展「三雄极光」10 页已作为独立展台 LED 屏场景接入训练数据（6 /2 /2，registered_not_trained），标注为当前 runtime 候选微调（1.22%–5.56% 对角线），非独立人工真值。见[场景说明](../../data/curated/scenes/202606-guangya-sanxiong_v1/README.md)。
<!-- sanxiong-dataset-v1:end -->

<!-- fine-local-round038:start -->
round_038 从正式 round_037 冻结 encoder/粗定位头、只训精修头，加入张凯源 6 页与三雄极光 6 页：新场景开发 13→18/20、已查看回归 9→13/16（三雄 1→6/8、1→5/8）；旧 432 页开发 +5、216 页回归持平；2514 页全量 +24，冻结 1808 -4（逐页无丢 3 角）。候选 `screen_corner_single_31adf6eec801` 未晋升，正式 runtime 与 0.2.25 不变。见[综合报告](../../research/experiments/2026-10-24-fine-local-round038.md) 与[回归对比](../../research/experiments/2026-10-24-fine-local-round038-regression/README.md)。
<!-- fine-local-round038:end -->

<!-- round038-promotion-v0226:start -->
按用户要求把 round_038 精细定位候选晋升为正式 runtime `screen_corner_single_31adf6eec801`，发布编号 `model-20261023-191338-31adf6eec801`。同图：483 页开发 +8、266 页回归 +4、新增 239 页 +22、旧域 +2、冻结 1808 -4（逐页无丢 3 角）；本次两场开发 13→18/20、回归 9→13/16。0.2.26 macOS Apple Silicon DMG 已验收，SHA256 `d3933635dc9566afcdac0b76bf06edf1a4a4de10b368d68c1358097b4657acd3`，本地 ad-hoc、未公证、无 updater 产物。见[晋升审计](../../training/runs/screen_corner_single/round_038/reports/promotion-audit/README.md)与[下载 0.2.26](../../release-assets/local-single-20261024-v0226-round038/0.2.26/macos/ScreenPDF_0.2.26_aarch64.dmg)。
<!-- round038-promotion-v0226:end -->
