# Current Status

<!-- round042-promotion-v0228:start -->
## 2026-10-30 round_042 精细定位已晋升，0.2.28 本地安装包已验收

按用户要求把 round_042 精细定位候选晋升为正式单文件 runtime `screen_corner_single_9db7ad220b21`，发布编号 `model-20261029-180536-9db7ad220b21`，旧权重备份 `models/runtime/screen_corner_single.pre_round041.pt`。同图重跑正式 round_041 与候选：两场 12 页开发 42→47/48、12 页已查看回归 45→46/48、54 页诊断 198→213/216；原 483 页开发 1668→1688（+20）、266 页回归 925→936（+11），张立文重放 238→240/244；同图 2605 页 9897→9953（+56），新增场景 246 页 +24，冻结 1808 页诊断 6855→6881（+26，13 个集合 8 升 /0 退 /5 平，逐页无丢 3 角）。持平集合：AWE、上海大屏、3 月 20 日、智能交通年会、追光；改善：商场 +8、开放原子 +5、3 月 19 日 +4、元宇宙 +4、福州高速 +2，InfoComm /扬州 /杭州隧道各 +1。精修头是唯一变量，2605 页粗定位输出与旧版逐位相同；但 control（只回放再训）也改善原域 +11/+7，说明部分收益来自精修头再拟合，新增 30 页的独立增量是两场开发 +3、回归 0、原域 +9/+5。

0.2.28 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `ceadc2d942ab2498a5f48ce310ef35313ad92f55e57b2ddc011ac0952bf4ef5c`（19,577,472 字节），版本 0.2.28、Finder 布局与 0.2.16 基准逐字节一致（`.DS_Store` / `.VolumeIcon.icns` 摘要相同）、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_9db7ad220b21`；前端 55 项测试、cargo check、4 项打包脚本测试通过。本地 ad-hoc 签名、未公证、未对外发布，按本地配置关闭 updater 产物，包内无 `latest.json`，不构成更新通道。限制保留：单 seed、两场开发 12 页 /48 角点；标注由 runtime 候选微调，不是独立真值或盲测；残余 `IMG_2817` 右下 1.28%、`IMG_2818` 左下 1.77%、`IMG_2826` 右上 1.15%。

[晋升审计](../../training/runs/screen_corner_single/round_042/reports/promotion-audit/README.md) · [决策记录](../../training/runs/screen_corner_single/round_042/reports/promotion_decision.json) · [综合报告](../../research/experiments/2026-10-29-qujingjing-xuhui.md) · [下载 0.2.28](../../release-assets/local-single-20261030-v0228-round042/0.2.28/macos/ScreenPDF_0.2.28_aarch64.dmg)。
<!-- round042-promotion-v0228:end -->

<!-- qujingjing-xuhui-round042:start -->
## 2026-10-29 瞿晶晶 /徐辉两场已接入并完成 round_042，候选未晋升

2026-10-29 上海开源大数据创新论坛同场次「AsterHarness：事实即本体-瞿晶晶」与「Rust 编译器开源社区治理-徐辉」各 27 页人工四角已接入：两场均 **15 训练 / 6 开发 / 6 已查看回归**（瞿晶晶 25 个内容组、徐辉 24 个），源目录图片全部列入项目。目标为会议室投影屏完整四边形，排除木纹墙、天花板、讲者与观众；瞿晶晶为白底 AsterHarness 幻灯片，徐辉为蓝底 Rust 社区治理幻灯片。54 张框由当前正式 runtime `screen_corner_single_c2409691b054` 候选微调（瞿晶晶 0.23%–2.06%、徐辉 0.29%–6.14%，中位数 0.51%/0.70%），登记 `runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`；瞿晶晶 `IMG_2791` 与张立文 `IMG_2777` 的 dHash=8 碰撞已目视确认为不同幻灯片。

训练前正式 runtime 在两场合计粗定位 110/216、最终 198/216；开发 42/48、已查看回归 45/48。round_042 按"冻结 encoder 与粗定位头、只训精修头"执行：两组都回放快照 5575 + 张立文 36 页 ×24 = 6439 条，with_new 再加瞿晶晶 15 页与徐辉 15 页 ×24 共 7159 条，24 轮、每轮 224 次更新、seed=7，只按两场 12 页开发选 epoch 并先冻结再回归。选中 with_new 第 19 轮：**两场开发 42→47/48、已查看回归 45→46/48、24 页 87→93/96**；原 483 页开发 1668→1688（+20）、266 页回归 924→936（+12）；张立文重放 49→50/46→47；合并开发 1759→1785、合并回归 1015→1029。control 只回放也改善原域 +11/+7、两场开发 +2，说明部分收益来自精修头再拟合；with_new 相对 control 仍多 +9/+5 原域、+3/0 两场。同图全量 2605 页 9897→9953（+56），新增场景 246 页 +24，冻结 1808 页 6855→6881（+26，13 个集合 8 升 /0 退 /5 平，无单页丢 3 角）。残余 `IMG_2817` 右下 1.28%、`IMG_2818` 左下 1.77%、`IMG_2826` 右上 1.15%。候选 `screen_corner_single_9db7ad220b21` 随后已按用户要求晋升为正式 runtime 并重新打包 0.2.28（见最新晋升记录）；训练时旧快照与冻结 `conference_v1` 未回写。

[训练报告](../../research/experiments/2026-10-29-qujingjing-xuhui.md) · [round_042](../../training/runs/screen_corner_single/round_042/README.md) · [回归对比](../../research/experiments/2026-10-29-qujingjing-xuhui-regression/README.md) · [瞿晶晶场景](../../data/curated/scenes/202610-open-source-bigdata-qujingjing_v1/README.md) · [徐辉场景](../../data/curated/scenes/202610-open-source-bigdata-xuhui_v1/README.md)。
<!-- qujingjing-xuhui-round042:end -->

<!-- round041-promotion-v0227:start -->
## 2026-10-29 round_041 精细定位已晋升，0.2.27 本地安装包已验收

按用户要求把 round_041 精细定位候选晋升为正式单文件 runtime `screen_corner_single_c2409691b054`，发布编号 `model-20261029-152403-c2409691b054`，旧权重备份 `models/runtime/screen_corner_single.pre_round038.pt`。同图重跑正式 round_038 与候选：本场 13 页开发 30→49/52（+19），12 页已查看回归 31→46/48（+15），本场 61 页诊断 144→238/244；原 483 页开发 1671→1668（-3），266 页已查看回归 930→924（finish）/925（审计，`IMG_2563` 左下 1.000% 阈值角点在 CPU/MPS 后端间翻转 1 点），合并开发 1701→1717（+16），合并已查看回归 961→970（+9）；同图 2551 页 9605→9699（+94），新增场景 192 页 569→669，冻结 1808 页诊断 6861→6855（-6，13 个集合 4 升 /6 退 /3 平，逐页无丢 3 角）。冻结集合回退：商场 -7、扬州 -2，3 月 19 日 /杭州隧道 /福州高速 /追光各 -1；改善：开放原子 +3、AWE +2、智能交通年会 +1、InfoComm +1。精修头是唯一变量：2551 页粗定位输出与 round_038 逐位相同。

0.2.27 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `df7ba0eaeff08f2718d1613c85430a1081bba7e6bf81f997f22adfe718ec7b6c`（19,574,994 字节），版本 0.2.27、Finder 布局与 0.2.16 基准逐字节一致（`.DS_Store` / `.VolumeIcon.icns` 摘要相同）、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_c2409691b054`；前端 55 项测试、cargo check、4 项打包脚本测试通过。本地 ad-hoc 签名、未公证、未对外发布，按本地配置关闭 updater 产物，包内无 `latest.json`，不构成更新通道。限制保留：单 seed、13 页开发 /52 角点；标注由 runtime 候选微调，不是独立真值或盲测；旧域小幅回退；`IMG_2776` 因粗定位冻结仍错左上 17.85%。

[晋升审计](../../training/runs/screen_corner_single/round_041/reports/promotion-audit/README.md) · [决策记录](../../training/runs/screen_corner_single/round_041/reports/promotion_decision.json) · [综合报告](../../research/experiments/2026-10-29-open-source-bigdata-zhangliwen.md) · [下载 0.2.27](../../release-assets/local-single-20261029-v0227-round041/0.2.27/macos/ScreenPDF_0.2.27_aarch64.dmg)。
<!-- round041-promotion-v0227:end -->

<!-- open-bigdata-zhangliwen-round041:start -->
## 2026-10-29 金融领域开源大语言模型 61 页已接入，round_041 精细定位候选未晋升

2026-10-29 上海开源大数据创新论坛「金融领域的开源大语言模型-张立文」61 页人工四角已作为独立会议室投影屏场景接入 `202610-open-source-bigdata-zhangliwen`：**36 训练 / 13 开发 / 12 已查看回归**，54 个屏幕内容组；源目录 62 张 JPG，`IMG_2788.jpeg` 未进项目排除。目标为投影屏完整四边形（含白色幻灯片与屏幕边框），排除木纹墙、天花板、讲者与观众。61 张框由当前正式 runtime `screen_corner_single_31adf6eec801` 候选微调，逐页最大调整 **0.93%–30.56%**（中位数 1.72%、平均 2.23%；57 页 >1%、16 页 >2%、1 页 >5%），登记 `runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`，不是独立真值。

训练前正式 runtime 在本场为粗定位 116/244、最终 144/244；开发 30/52、已查看回归 31/48，多数页面只差 1–2 个角。round_041 按"冻结 encoder 与粗定位头、只训精修头"执行：control 回放 5575 条，with_new 加入 36 页 ×24 共 6439 条，24 轮、每轮 202 次更新、seed=7，只按本场 13 页开发（52 角点）最终指标选 epoch 并先冻结再回归。选中 with_new 第 4 轮：**开发 30→49/52、已查看回归 31→46/48、25 页合计 61→95/100、61 页全部 144→238/244**；原 483 页开发 1671→1668、266 页回归 930→924，合并开发 1701→1717、合并回归 961→970。同图全量 2551 页 9605→9699（+94），冻结 1808 页 6861→6855（-6，4 升 /6 退 /3 平，无单页丢 3 角）；本场 56 页改善、0 页回退，残余 `IMG_2738` 左上 2.43%、`IMG_2727` 左下 1.77%、`IMG_2790` 右下 2.19%，`IMG_2776` 因粗定位冻结仍错左上 17.85%。候选 `screen_corner_single_c2409691b054` 随后已按用户要求晋升为正式 runtime 并重新打包 0.2.27（见最新晋升记录）；训练时旧快照与冻结 `conference_v1` 未回写。

[训练报告](../../research/experiments/2026-10-29-open-source-bigdata-zhangliwen.md) · [round_041](../../training/runs/screen_corner_single/round_041/README.md) · [回归对比](../../research/experiments/2026-10-29-open-source-bigdata-zhangliwen-regression/README.md) · [场景说明](../../data/curated/scenes/202610-open-source-bigdata-zhangliwen_v1/README.md)。
<!-- open-bigdata-zhangliwen-round041:end -->

<!-- white-mix-corner-round040:start -->
## 2026-10-27 白屏混合角点精修已接入并完成 round_040，候选回落到正式 runtime

2026-03-06 杭州智能交通年会「白屏混合」的角点已按用户精修重新登记：42 页 168 个角点，164 个点有位移（`IMG_6941` 未变），平均 0.213%、中位数 0.199%、P90 0.365%、最大 0.868% 对角线，**没有任何角点超过 1% 命中半径**。页面成员完全不变（33 训练 / 9 开发 / 0 回归，`IMG_7018` 与中交机电同图继续排除），新快照 `training/datasets/screen_corner_single_all_manual_3_20fine_whitemix_v1` 的 `train` 仍 5575 条、`all_manual` 2286 页、`validation` 483 页、`holdout` 266 页、隔离训练页 166 条，只替换这 42 页的 `manual_quad`（及其 dHash/pHash）。

先用正式 runtime 在同一批 42 页各打分一次：旧 / 新真值都是 165/168，逐页 0 翻转；9 页开发最终 33/36 不变，只有粗定位从 33/36 掉到 31/36，说明 1% 指标对亚 1% 修订不敏感。

round_040 从正式 round_038（`screen_corner_single_31adf6eec801`）出发，control=旧标注、refined=精修标注，两组都是 5575 条、12 轮粗定位 + 16 轮精修、每轮 175 次更新、seed=7，共用同一套评估真值，只按白屏混合 9 页开发选 epoch 并先冻结再回归。最终三组并列 33/36，按平局规则选中 `initial`：**checkpoint 与 `models/runtime/screen_corner_single.pt` 字节相同，本轮没有新候选**。粗定位阶段精修标注确实有信号（新真值下 31 → 32（control）/33（refined）），但精修阶段抹平；原域 474 页开发两组同时 -14 点，control 与 refined 之间净 0（6 得 /6 失），因此该退化是固定预算重训漂移而非标注修订贡献。剩余 3 个开发失配全是右下角（`IMG_7039` 12.68%、`IMG_7016` 3.50%、`IMG_7013` 2.00%），亚 1% 的真值微调在原理上修不了。正式 runtime、0.2.26 安装包、冻结 `conference_v1`、旧 annotation 与旧快照均未回写；未晋升、未打包、未重跑 1808 页冻结审计。

[报告](../../research/experiments/2026-10-27-white-mix-corner-refinement.md) · [round_040](../../training/runs/screen_corner_single/round_040/README.md) · [新标注](../../data/curated/annotations/202603-white-mix_pages_v2.jsonl) · [新快照审计](../../training/datasets/screen_corner_single_all_manual_3_20fine_whitemix_v1/audit.json)。
<!-- white-mix-corner-round040:end -->

<!-- zhongjiao-corner-revision:start -->
## 2026-10-25 中交机电点位重矫正 v2，训练快照已更新，尚未训练

2026-03-06 杭州智能交通年会「中交机电」的点位已按用户重新矫正。源项目 20 页、19 页人工框；本 revision 18 个有效页：**16 训练 / 2 开发 / 0 已查看回归**。旧标注 12 个手工页重新矫正（相对旧点变化 0.13%–0.80% 对角线），7 页本次新标注；`IMG_7031`/`IMG_7035` 旧点越界、修正后有效加入训练，`IMG_7021` 与训练页 `IMG_7020` 同内容组改训练，`IMG_7018` 与 `202603-white-mix/IMG_7018` 同图但标签/用途冲突继续排除，`IMG_7022` 无人工框。旧拆分 6/2/4 改为 16/2/0，本 revision 没有干净 holdout。

**关键出框口径**：8 页（`IMG_7021`、`IMG_7023`、`IMG_7026`、`IMG_7028`、`IMG_7031`、`IMG_7032`、`IMG_7035` 右下，`IMG_7038` 右上）的屏幕角超出照片范围；人工点取屏幕边与图像边界的交点，位于画面边框上，不是真实物理角点。这些点有效，不能按越界框过滤；预测被钳制在图像内，1% 指标以边界交点为真值。标注仍由 runtime 候选微调，登记 `runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`。

新训练快照 `training/datasets/screen_corner_single_all_manual_3_20fine_zhongjiao_v1/`：`train` 5535→5575 条（10 个新训练页 ×4 回放），`all_manual` 2276→2286 页，`validation` 483 页、`holdout` 266 页不变。当前 runtime 在 18 个有效页为 72/72（最大 0.85%，与点位同源，仅覆盖诊断）。状态 `registered_not_trained`；未训练、未开 round、未改正式 runtime 或 0.2.26 安装包。去重索引更新为 53 个单元、3,703 条单元内人工页、2,649 张唯一人工图像。

[场景说明](../../data/curated/scenes/202603-zhongjiao-jidian_v1/README.md) · [新标注](../../data/curated/annotations/202603-zhongjiao-jidian_pages_v2.jsonl) · [新拆分](../../data/splits/cross_project/202603-zhongjiao-jidian_v2.json) · [快照审计](../../training/datasets/screen_corner_single_all_manual_3_20fine_zhongjiao_v1/audit.json)。
<!-- zhongjiao-corner-revision:end -->

<!-- openatom-202511-dataset:start -->
## 2025-11 开放原子开源基金 6 页已接入，registered_not_trained

2025-11-03 开放原子开源基金讨论 6 页选中候选人工框已作为独立会议室投影屏场景接入 `202511-openatom-open-source-fund`：4 训练 / 1 开发 / 1 已查看回归（5 个屏幕内容组；`IMG_5137`/`IMG_5140` 同组）。本场存在明显前景遮挡：观众头部、肩部沿屏幕下沿遮住部分屏幕，桌面与椅背占据下半部；目标为完整投影屏，人工四角外推到被遮挡的屏幕底边，不沿人头轮廓收边。6 张框由当前 runtime `screen_corner_single_31adf6eec801` 候选微调 1.05%–5.85% 对角线，登记 `label_quality=runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`，不是独立真值。与既有全部标注及 `conference_v1` 比对无精确字节 / 路径 / 强 dHash / pHash ≤8 重复。开发 / holdout 各仅 1 页，无法支撑可靠 epoch 选择，本 revision 只登记训练数据，未训练、未创建 round、未改正式 runtime 或 0.2.26 安装包。当前正式 runtime 在这 6 页为 18/24，6 页均是右下角未命中（1.05%–5.85%），说明遮挡主要影响屏幕右下边界。数据集索引更新为 53 个单元、2,642 张唯一人工图像。

[场景说明](../../data/curated/scenes/202511-openatom-open-source-fund_v1/README.md) · [标注](../../data/curated/annotations/202511-openatom-open-source-fund_pages_v1.jsonl) · [拆分](../../data/splits/cross_project/202511-openatom-open-source-fund_v1.json)。
<!-- openatom-202511-dataset:end -->

<!-- qinghui-round039:start -->
## 2026-10-24 清晖同学会 116 页已接入并完成 round_039 训练，候选未晋升

2025-12-08「清晖 PgMP 同学会」116 页人工框已作为独立会议室投影屏场景接入 `202512-qinghui-alumni`：69 训练 / 24 开发 / 23 已查看回归，92 个屏幕内容组；3 张无项目页照片不纳入。人工框由当前 runtime `screen_corner_single_31adf6eec801` 候选微调，登记 `runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`；51/116 页调整 >1%，6 页 >10%，最大 78.99%，不是独立真值。

round_039 从正式 round_038 出发，base 使用含 3 月 20 日精细标注的 `screen_corner_single_all_manual_3_20fine_v1`；control 回放 5535 条，with_new 加入清晖 69 页 ×24 共 7191 条，两组 12 轮粗定位 + 16 轮精修、每轮 225 次更新、seed=7，只按清晖 24 页开发（96 角点）最终指标选 epoch 并先冻结再回归。选中 with_new：清晖开发 84→90/96、已查看回归 77→85/92、全部 161→175/188；原 483 页开发 1671→1665/1932，合并开发 1755/2028 持平，合并已查看回归 1007→1013/1156；粗定位清晖全部 148→167，最终精修头在本场开发未超过初始局部头（epoch 0 选中）。候选 `screen_corner_single_049cd9cdf0de` 未晋升；正式 runtime 仍 `screen_corner_single_31adf6eec801`，0.2.26 安装包不变。保留失败：清晖开发 6 个左下/右下角点，最大左下 5.35%（`IMG_5906`）；已查看回归 `IMG_6009` 1/4、`IMG_6012` 2/4。限制：单 seed、24 页开发仅 96 角点、标签非独立真值、holdout 已查看、同场同相机、69 页 ×24 非独立样本；未做晋升审计或重新打包。

[训练报告](../../research/experiments/2026-10-24-qinghui-alumni-training.md) · [round_039](../../training/runs/screen_corner_single/round_039/README.md) · [场景说明](../../data/curated/scenes/202512-qinghui-alumni_v1/README.md)。
<!-- qinghui-round039:end -->

<!-- smart-road-3-20-fine-labels:start -->
## 2026-10-24 中国智慧道路照明大会 3 月 20 日 59 页精细标注已更新训练数据，尚未训练

用户更新了「智慧城市道路与景观照明融合的守正创新实践」源项目的修正点位。58 个既有页面的 232 个角点全部移动，但都在 1% 对角线以内（平均 0.168%、P90 0.320%、最大 0.645%）；`IMG_8243` 从无显式人工框变为有框。按冻结 `conference_v1` 用途保留 46 训练（其中 `IMG_8277` 仍近重复隔离）/12 开发，新增页按原项目级 legacy split 记为训练，修订后 47 训练 / 12 开发。新标注 `data/curated/annotations/20260320-smart-road-lighting_pages_v2.jsonl`、新拆分 `data/splits/cross_project/20260320-smart-road-lighting_v2.json`、新注册 `training/registry/20260320-smart-road-lighting_v2.json` 和新训练快照 `training/datasets/screen_corner_single_all_manual_3_20fine_v1/` 已生成；`train` 5534→5535 条、`validation` 483 页、`all_manual` 2275→2276 页，成员、用途、24 倍回放和 sidecar 其余部分继承 round_038。当前正式 runtime（round_038，`screen_corner_single_31adf6eec801`）在这批 59 页旧 /新标签下均 234/236 命中，逐页 0 翻转，仍只有 `IMG_8256`、`IMG_8264` 各错 1 个右下角；因此不能把 0 翻转当作模型提升。冻结 conference_v1 /旧快照 / 基准 / 正式 runtime / 0.2.26 安装包均未回写；未训练、未创建新 round、未晋升或打包。

[标注修订报告](../../research/experiments/2026-10-24-smart-road-lighting-fine-labels.md) · [训练快照](../../training/datasets/screen_corner_single_all_manual_3_20fine_v1/README.md)。
<!-- smart-road-3-20-fine-labels:end -->

<!-- round038-promotion-v0226:start -->
## 2026-10-24 round_038 已晋升，0.2.26 本地安装包已验收

按用户要求把 round_038 精细定位候选晋升为正式单文件 runtime `screen_corner_single_31adf6eec801`，发布编号 `model-20261023-191338-31adf6eec801`，旧权重备份 `models/runtime/screen_corner_single.pre_round037.pt`。同图重跑正式 round_037 与候选：483 页开发 1663→1671（+8），266 页已查看回归 926→930（+4），新增 239 页 853→875（+22），旧域 2275 页 8614→8616（+2），冻结 1808 页诊断 6865→6861（-4）；本次两场开发 13→18/20、已查看回归 9→13/16（三雄极光 1→6/8、1→5/8；张凯源保持 20/20）。冻结 13 个集合 3 升 /5 平 /5 降（元宇宙 -2、InfoComm -2，扬州 /智能交通 /福州各 -1），逐页没有丢 3 角。0.2.26 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `d3933635dc9566afcdac0b76bf06edf1a4a4de10b368d68c1358097b4657acd3`，Finder 布局与 0.2.16 基准一致、版本 0.2.26、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_31adf6eec801`；本地 ad-hoc、未公证、未对外发布，按本地配置关闭 updater 产物，不构成更新通道。限制保留：5 页开发仅 20 角点、单 seed；两场标签由当前 runtime 候选微调，不是独立人工真值；三雄残余最大单角误差 2.82%。

[晋升审计](../../training/runs/screen_corner_single/round_038/reports/promotion-audit/README.md) · [综合报告](../../research/experiments/2026-10-24-fine-local-round038.md) · [下载 0.2.26（macOS Apple Silicon）](../../release-assets/local-single-20261024-v0226-round038/0.2.26/macos/ScreenPDF_0.2.26_aarch64.dmg)。DMG SHA256：`d3933635dc9566afcdac0b76bf06edf1a4a4de10b368d68c1358097b4657acd3`。
<!-- round038-promotion-v0226:end -->

<!-- fine-local-round038:start -->
## 2026-10-24 round_038 精细定位训练完成，候选未晋升

按“粗定位基本可以、精细定位略差”的判断，round_038 从正式 round_037 冻结 encoder 与粗定位头，只训练精修头：control 回放 5246 条、with_new 追加张凯源 6 页与三雄极光 6 页×24 共 5534 条；24 轮、每轮 173 次更新，seed=7，只按 5 页新场景开发（20 角点）的最终角点指标选 epoch 并先冻结再回归。新场景开发 13→18/20（65.00%→90.00%），已查看回归 9→13/16；三雄极光开发 1→6/8、回归 1→5/8，张凯源保持 20/20。旧 432 页开发 1499→1504、216 页已查看回归 773→773；池化开发 1663→1671、回归 926→930。同图 2514 页全量 9467→9491（+24，新增 +22、旧域 +2）；冻结 1808 页诊断 6865→6861（-4，3 升 /5 平 /5 降，最大 -2），逐页无丢 3 角。候选 `screen_corner_single_31adf6eec801` 未晋升；正式 runtime 与 0.2.25 安装包不变。限制：5 页开发仅 20 角点、单 seed；两场标签均由当前 runtime 候选微调（三雄 1.22%–5.56%、张凯源 0.40%–1.08%），不是独立人工真值，三雄残余最大单角误差 2.82%，不能宣称精细精度已彻底解决。

[综合报告](../../research/experiments/2026-10-24-fine-local-round038.md) · [回归对比](../../research/experiments/2026-10-24-fine-local-round038-regression/README.md)。
<!-- fine-local-round038:end -->

<!-- sanxiong-dataset-v1:start -->
## 2026-10-24 2026 光亚展三雄极光 10 页已接入训练数据，尚未训练

2026-06-10 光亚展「三雄极光」10 页已作为独立展台 LED 屏场景接入 `202606-guangya-sanxiong`（6 训练 /2 开发 /2 已查看回归，9 个内容组），目标为展台 LED 屏完整四边形，排除屏幕外展台结构，顶部品牌条与页脚按原人工点保留。

标注由当前正式 runtime `screen_corner_single_9b9e9c200c49` 候选微调，逐页最大调整量 1.22%–5.56% 对角线；登记 `label_quality=runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`，包含人工修正但仍非独立人工真值，1% 命中不当作盲测或精细精度证据。与既有全部人工标注比对无精确字节 /路径 /强 dHash /pHash ≤8 重复。状态 registered_not_trained；不加入上海 IN³C 场景族。该场与张凯源已作为 round_038 精细定位训练的两场新资料。

[场景说明](../../data/curated/scenes/202606-guangya-sanxiong_v1/README.md)。
<!-- sanxiong-dataset-v1:end -->

<!-- zhangkaiyuan-dataset-v1:start -->
## 2026-10-24 上海 IN³C 张凯源 11 页已接入训练数据，尚未训练

上海 IN³C 2026 场次「多模态大模型与云边协同赋能油气机器人智能巡检-张凯源」11 页已接入 `202610-embodied-intelligence-zhangkaiyuan`（6 训练 /3 开发 /2 已查看回归，9 个内容组），目标为宽幅 LED 舞台屏完整屏幕四边形，排除两侧品牌固定区，蓝色页脚条按原人工点保留。

**标注独立性限制**：11 张人工框全部由当前正式 runtime `screen_corner_single_9b9e9c200c49` 的候选微调而来，逐页最大调整量只有 0.40%–1.08% 对角线；登记 `label_quality=runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`。它们与当前模型高度同源，不是独立人工真值；用当前模型在这批页面的 1% 命中不能当作盲测或精细精度证据。与既有全部人工标注比对无精确字节 /路径 /强 dHash /pHash ≤8 重复。状态 registered_not_trained：仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包；上海 IN³C 场景族由 v5 追加为 v6，12 场共 173 页（99 训练 /39 开发 /35 已查看回归）。

[场景说明](../../data/curated/scenes/202610-embodied-intelligence-zhangkaiyuan_v1/README.md) · [场景族 v6](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v6/README.md)。
<!-- zhangkaiyuan-dataset-v1:end -->

<!-- round037-promotion-v0225:start -->
## 2026-10-24 round_037 已晋升，0.2.25 本地安装包已验收

按用户要求把 round_037 四场加入候选晋升为正式单文件 runtime `screen_corner_single_9b9e9c200c49`，发布编号 `model-20261023-182931-9b9e9c200c49`，旧权重备份 `models/runtime/screen_corner_single.pre_round036.pt`。同图重跑正式 round_036 与候选：478 页开发 1640→1650（+10），262 页已查看回归 913→917（+4），冻结 1808 页诊断 6846→6865（+19），全部 2275 页诊断 8599→8614（+15），新增 218 页 763→799（+36）；本次四场 9 页开发 24→30/36、8 页已查看回归 27→28/32。冻结集合 8 升 /4 平 /1 降，唯一回退为中国智慧道路照明大会 3 月 19 日 -1；逐页只有 +1/0/-1，没有单页丢 3 角。0.2.25 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `719e91c177a499adade72e253157b85dd9fb03b921cd7eb90519ef385ad430e8`，Finder 布局与 0.2.16 基准一致、版本 0.2.25、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_9b9e9c200c49`；本地 ad-hoc 签名、未公证、未对外发布，包内无 updater 产物，不构成更新通道。限制保留：四场框由 runtime 候选微调、粗精度，9 页开发仅 36 角点，单 seed，精修阶段在新页面无净增益。

[晋升审计](../../training/runs/screen_corner_single/round_037/reports/promotion-audit/README.md) · [决策记录](../../training/runs/screen_corner_single/round_037/reports/promotion_decision.json) · [下载 0.2.25（macOS Apple Silicon）](../../release-assets/local-single-20261024-v0225-round037/0.2.25/macos/ScreenPDF_0.2.25_aarch64.dmg)。DMG SHA256：`719e91c177a499adade72e253157b85dd9fb03b921cd7eb90519ef385ad430e8`。
<!-- round037-promotion-v0225:end -->

<!-- embodied-v5-round037:start -->
## 2026-10-24 round_037 四场加入训练完成，候选未晋升

周浩杰 6 /姚志强 4 /马战凯 15 /刘意杨 14 页（22 训练页）已在 round_037 从正式 round_036（`screen_corner_single_7237adbcaa54`）出发训练：control 回放 4718 条、with_new 加入 22 页×24 共 5246 条，粗定位 12 轮 + 精修 16 轮、每轮 164 次更新，seed=7，只按 9 页新场景开发的唯一角点指标选 epoch 并先冻结再回归。选中 with_new，新场景开发 24→30/36（66.67%→83.33%），已查看回归 27→28/32；旧 432 页开发 1494→1499、216 页已查看回归 773→773；冻结 1808 页诊断 6846→6865（+19），逐页只有 +1/0/-1，唯一回退集合为中国智慧道路照明大会 3 月 19 日 -1；2493 页全量 9362→9413（+51，旧域 +15）。候选 `screen_corner_single_9b9e9c200c49` 未晋升，正式 runtime 与 0.2.24 安装包不变；未重跑晋升审计或打包。限制：四场人工框由当前 runtime 候选微调、角点不够精细，9 页开发仅 36 角点，精修阶段在新页面无净增益，收益主要来自粗定位头，不能据此宣称精细精度已解决。

[综合报告](../../research/experiments/2026-10-24-embodied-v5-training.md) · [回归对比](../../research/experiments/2026-10-24-embodied-v5-round037-regression/README.md)。
<!-- embodied-v5-round037:end -->

<!-- embodied-v5-dataset:start -->
## 2026-10-24 上海 IN³C 四场已接入训练数据，尚未训练

上海 IN³C 2026 四场 2026-10-13 资料已接入：「从软硬件工程到应用的AI应用方案-周浩杰」6 页（3 训练 /2 开发 /1 已查看回归，`202610-embodied-intelligence-zhouhaojie`）、「智能机器人全域全行业应用分享-姚志强」4 页（2 /1 /1，`202610-embodied-intelligence-yaozhiqiang`）、「边缘为枢，云网为基，赋能具身智能走向规模-马战凯」15 页（9 /3 /3，`202610-embodied-intelligence-mazhankai`；项目内 `IMG_2139` 为 needs_review、无人工框，排除）、「人工智能财能工业提质增效-刘意杨」14 页（8 /3 /3，`202610-embodied-intelligence-liuyiyang`），共 39 页。目标为宽幅 LED 舞台屏完整屏幕四边形，排除两侧品牌固定区；屏幕自带的标题 /页脚条按原人工点保留，讲台、讲者与透视不沿遮挡轮廓收边。

用户指出这批人工框"基本定位相对准确，但还不够精细"：四场标注均由当前 runtime 候选 `screen_corner_single_7237adbcaa54` 微调而来，登记 `label_quality=coarse_runtime_seeded_manual`、`fine_corner_truth=false`，训练用途按既有拆分保留，但细角点精度只能按近似值解释，不当作精细真值或盲测。状态 registered_not_trained：本次仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包。上海 IN³C 场景族由 v4 追加为 v5，11 场共 162 页（93 训练 /36 开发 /33 已查看回归）；与既有全部人工标注比对无精确字节 /路径 /强 dHash 重复，无 pHash ≤8 标记；旧 revision、旧用途与冻结 conference_v1 不回写。

[场景族 v5](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v5/README.md) · [周浩杰](../../data/curated/scenes/202610-embodied-intelligence-zhouhaojie_v1/README.md) · [姚志强](../../data/curated/scenes/202610-embodied-intelligence-yaozhiqiang_v1/README.md) · [马战凯](../../data/curated/scenes/202610-embodied-intelligence-mazhankai_v1/README.md) · [刘意杨](../../data/curated/scenes/202610-embodied-intelligence-liuyiyang_v1/README.md)。
<!-- embodied-v5-dataset:end -->

<!-- round036-promotion-v0224:start -->
## 2026-10-24 round_036 已晋升，0.2.24 本地安装包已验收

按用户要求把 round_036 遮挡定向候选晋升为正式单文件 runtime `screen_corner_single_7237adbcaa54`，发布编号 `model-20261023-165533-7237adbc`，旧权重备份 `models/runtime/screen_corner_single.pre_round035.pt`。同图重跑正式 round_035 与候选：469 页开发 1604→1616（+12），254 页已查看回归 874→886（+12），冻结 1808 页诊断 6853→6846（-7），全部 2275 页诊断 8600→8599（-1），新增 179 页 577→651（+74）。冻结集合回退：InfoComm /商场 /开放原子各 -3，元宇宙 -2，AWE /上海大屏 /道路照明 3.19 各 -1；改善：扬州 +3，道路照明 3.20 /智能交通年会 /福州高速 /追光各 +1；逐页没有少 3 角的页面。0.2.24 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `80103f2d3765ae7d28dd25e70ebbe2a1bcf050b01011de52985cc160ddb4b010`，Finder 布局与 0.2.16 参考一致（同一 .DS_Store / 卷图标摘要）、版本 0.2.24、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_7237adbcaa54`；本地 ad-hoc 签名、未公证、未对外发布，包内含 updater 签名产物但没有 latest.json，不构成更新通道。

[晋升审计](../../training/runs/screen_corner_single/round_036/reports/promotion-audit/README.md) · [决策记录](../../training/runs/screen_corner_single/round_036/reports/promotion_decision.json) · [回归验证](../../research/experiments/2026-10-24-round035-vs-round036-regression/README.md) · [下载 0.2.24（macOS Apple Silicon）](../../release-assets/local-single-20261024-v0224-round036/0.2.24/macos/ScreenPDF_0.2.24_aarch64.dmg)。
<!-- round036-promotion-v0224:end -->

<!-- regression-compare-round036:start -->
## 2026-10-24 round_035 vs round_036 回归验证报告

round_036 回归验证：正式 round_035（5be1a6da1dd2）对比候选 `screen_corner_single_7237adbcaa54`，同一 2454 页（旧 2275 + 新增 179）各自重跑推理。全部页 9177→9250（+73），新增 179 页 577→651（+74，80.59%→90.92%），旧 2275 页 8600→8599（-1），冻结 1808 页诊断 6853→6846（-7），469 页开发 1604→1616（+12），254 页已查看回归 874→886（+12）。新增场景：秦通全部页 +37、龚涛 +18、陆晨 +11、朱峰 +5，新场景开发 +5、回归 +9；旧域 13 个集合 5 升 /7 降 /1 平，最差 -3，逐页没有少 3 角的页面。结论：候选达到可晋升量级（新场景 +74、旧域 -1、无整页崩坏）；晋升仍需重跑 1808 页冻结审计并重新打包，正式 runtime 暂不变。

[回归验证报告](../../research/experiments/2026-10-24-round035-vs-round036-regression/README.md)。
<!-- regression-compare-round036:end -->

<!-- occlusion-round036:start -->
## 2026-10-24 round_036 遮挡场景定向训练完成，候选未晋升

秦通 33 /朱峰 4 /陆晨 10 页已接入；round_036 从正式 round_035 做遮挡定向对照：control 回放 4070 条、with_new 加入 27 页×24 共 4718 条、occlusion_focus 再把 54 张遮挡相关页加 24 倍共 6014 条，粗定位 12 轮 + 精修 16 轮、每轮 188 次更新，只按 20 页遮挡开发唯一角点指标选 epoch 并先冻结再回归。选中 with_new（加入 27 页），遮挡开发最终 62/80（77.50%）；候选 `screen_corner_single_7237adbcaa54` 未晋升，正式 runtime 与 0.2.23 安装包不变。单 seed、20 页遮挡开发、加倍采样不是独立样本与已查看回归限制保留。

[训练报告](../../research/experiments/2026-10-24-occlusion-focus-training.md)。
<!-- occlusion-round036:end -->

<!-- zhufeng-luchen-dataset-v1:start -->
## 2026-10-23 上海 IN³C 朱峰 4 页与陆晨 10 页已接入训练数据，尚未训练

上海 IN³C 2026 两场 2026-10-13 资料已接入：「智能无人装备安全标准研究进展-朱峰」4 页（2 训练 /1 开发 /1 已查看回归，`202610-embodied-intelligence-zhufeng`）与「西门子工业边缘及人工智能赋能制造业数字化转型-陆晨」10 页（6 /2 /2，`202610-embodied-intelligence-luchen`），目标为宽幅 LED 舞台屏完整幻灯片 /屏幕四边形，排除两侧品牌固定区，保留讲者与观众遮挡、LED 摩尔纹与轻微透视。状态 registered_not_trained：仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包。上海 IN³C 场景族由 v3 追加为 v4，七场共 123 页（71 训练 /27 开发 /25 已查看回归）；无精确字节 /强 dHash 重复，2 对 pHash 标记（朱峰 IMG_2099 vs 龚涛 IMG_2039、陆晨 IMG_2091 vs 华为 IMG_2003）逐对目视为不同幻灯片。同目录「工业领或具身智能洞察与实践-于青民」6 张照片没有 ScreenPDF 项目文件与人工框，本批未纳入，需标注后再接入。

[朱峰场景](../../data/curated/scenes/202610-embodied-intelligence-zhufeng_v1/README.md) · [陆晨场景](../../data/curated/scenes/202610-embodied-intelligence-luchen_v1/README.md) · [场景族 v4](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v4/README.md)。
<!-- zhufeng-luchen-dataset-v1:end -->

<!-- qintong-dataset-v1:start -->
## 2026-10-23 上海 IN³C 秦通场 33 页已接入训练数据，尚未训练

上海 IN³C 2026 场次「面向开放世界的具身智能三维探索、移动操作与全场景仿真-秦通」33 页已接入 `202610-embodied-intelligence-qintong`（19 训练 /7 开发 /7 已查看回归，32 个内容组），目标为宽幅 LED 舞台屏内完整白色幻灯片内容区，排除两侧品牌固定区与底部 IN³C 会议横幅，保留讲者 /观众遮挡、LED 摩尔纹与轻微透视。状态 registered_not_trained：本次仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包。上海 IN³C 场景族由 v2 追加为 v3，五场共 109 页（63 训练 /24 开发 /22 已查看回归）；与既有标注无精确字节 /强 dHash 重复，2 对 pHash 标记逐对目视为不同幻灯片；旧 revision、旧用途与冻结会议集不回写。

[场景与数据审计](../../data/curated/scenes/202610-embodied-intelligence-qintong_v1/README.md) · [场景族 v3](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v3/README.md)。
<!-- qintong-dataset-v1:end -->

<!-- round035-promotion-v0223:start -->
## 2026-10-23 round_035 已晋升，0.2.23 本地安装包已验收

按用户要求把 round_035（五批新增资料合并加入）晋升为正式单文件 runtime `screen_corner_single_5be1a6da1dd2`，发布编号 `model-20261023-041021-5be1a6da`，旧权重备份 `models/runtime/screen_corner_single.pre_round031.pt`。晋升审计同图重跑正式 round_031 与候选：459 页开发 1587→1581/1836（-6），244 页已查看回归 836→851/976（+15），冻结 1808 页诊断 6882→6853/7232（-29），全部 2275 页诊断 8640→8600/9100（-40），新增 132 页 311→467/528（+156）。冻结集合回退：福州高速 -12、开放原子 -9、扬州 -3、智慧道路照明 3 月 20 日 -3、元宇宙 -2、杭州隧道集合 -2、追光 -2、杭州智能交通 -2；改善：上海大屏 +2、3 月 19 日 +2、商场 +1、AWE +1，全部写入发布元数据。0.2.23 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `98f42ca7a06321cf8b7246ba51c2c6d12fc770b0dcf13691d4db1f9e4482c8c0`，Finder 布局与 0.2.16 逐字节一致、版本 0.2.23、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_5be1a6da1dd2`；本地 ad-hoc 签名、未公证、未对外发布，包内含 updater 签名产物但没有 latest.json，不构成更新通道。

[晋升审计](../../training/runs/screen_corner_single/round_035/reports/promotion-audit/README.md) · [决策记录](../../training/runs/screen_corner_single/round_035/reports/promotion_decision.json) · [下载 0.2.23（macOS Apple Silicon）](../../release-assets/local-single-20261023-v0223-round035/0.2.23/macos/ScreenPDF_0.2.23_aarch64.dmg)。
<!-- round035-promotion-v0223:end -->

<!-- candidate-scene-comparison-20261023:start -->
## 2026-10-23 四个候选在全部场景上的横向对比

四个单文件模型（正式 round_031、round_033 遮挡、round_034 PM 精修、round_035 五批合并）已在同一 2407 页语料（旧 2275 + 新增 132）重新推理做分场景对比：全部页 8951 /8996 /8984 /**9067** 命中；旧 2275 页 8640 /8618 /8632 /8600；新增 132 页 311 /378 /352 /**467**；已查看回归 244 页 836 /832 /**850 /851**。相对正式：round_033 旧 -22 /新 +67 /回归 -4；round_034 旧 -8 /新 +41 /回归 +14；round_035 旧 -40 /新 +156 /回归 +15。要一个 runtime 覆盖全部新增场景选 round_035（`screen_corner_single_5be1a6da1dd2`，需重跑 1808 页冻结审计并重新打包）；要旧域风险最小选 round_034（`9eb8bc25c145`）；round_033 保留为遮挡对照；正式 round_031 仍是旧场景最佳。均未晋升。

[对比报告（含分场景 dev /回归 /全量表）](../../research/experiments/2026-10-23-candidate-scene-comparison.md)。
<!-- candidate-scene-comparison-20261023:end -->

<!-- combined-new-data-round035:start -->
## 2026-10-23 五批新增资料合并加入 round_035 训练完成

五批新增资料（南京边缘算力 67 页、AI铸链 4 页、华为 AI 6 页、龚涛 40 页、2024 PM 管理魔方 15 页，合计 132 页 / 77 训练页）已在 round_035 从正式 round_031 合并做固定预算对照：control 回放 2222 条、with_new_all 加入 77 页×24 共 4070 条，粗定位 12 轮 + 精修 16 轮、每轮 128 次更新，只按 27 页合并新场景开发唯一角点指标选 epoch 并先冻结再回归。选中 with_new_all（加入 77 页），合并新场景开发最终 94/108（87.04%）；候选 `screen_corner_single_5be1a6da1dd2` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、27 页开发、五场景混合采样与已查看回归限制保留。

[训练报告](../../research/experiments/2026-10-23-combined-new-data-training.md)。
<!-- combined-new-data-round035:end -->

<!-- pm-cube-round034:start -->
## 2026-10-22 2024 新任产品经理管理魔方 round_034 仅本地精细定位训练完成

2024「新任产品经理管理魔方」15 页已接入（9 训练 / 3 开发 / 3 已查看回归，投影白底幻灯片，参考模型在这批页面的角点误差约 1–5%、命中仅 16/60）。round_034 从正式 round_031 冻结 encoder 与粗定位头，只训练本地精修头 24 轮做固定预算对照：control 回放 2222 条、with_pm_cube 加入 9 页×24 共 2438 条，只按 3 页精细定位开发唯一角点指标选 epoch 并先冻结再回归。选中 with_pm_cube（加入 9 页），开发最终 11/12（91.67%）；候选 `screen_corner_single_9eb8bc25c145` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、3 页开发、只训本地头的限制保留。

[训练报告](../../research/experiments/2026-10-22-pm-cube-fine-localization.md)。
<!-- pm-cube-round034:end -->

<!-- embodied-occlusion-round033:start -->
## 2026-10-22 上海 IN³C 重遮挡两场 round_033 训练完成，候选未晋升

上海 IN³C 重遮挡两场（华为 AI 6 页 + 龚涛 40 页，27 页训练）已在 round_033 从正式 round_031 做固定预算对照：control 回放 2222 条、with_occluded 加入 27 页×24 共 2870 条，粗定位 12 轮 + 精修 16 轮、每轮 90 次更新，只按 10 页重遮挡开发唯一角点指标选 epoch 并先冻结再回归。选中 with_occluded（加入 27 页），重遮挡开发最终 32/40（80.00%）；候选 `screen_corner_single_4a6904df2cd6` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、10 页开发、人工点外推遮挡边缘与已查看回归限制保留。

[训练报告](../../research/experiments/2026-10-22-embodied-occlusion-training.md)。
<!-- embodied-occlusion-round033:end -->

<!-- embodied-occluded-dataset-v1:start -->
## 2026-10-22 上海 IN³C 华为 AI 6 页与龚涛 40 页已接入训练数据，尚未训练

上海 2026 国际工业互联网大会（IN³C）同峰会两场 2026-10-13 资料已纳入：华为 AI 6 张（**3 训练 / 2 开发 / 1 已查看回归**，6 个内容组）、「工业控制系统从自动化向自主化的演进-龚涛」40 张（**24 / 8 / 8**，26 个内容组），共 46 页显式人工框。用户指出本批遮挡较大：讲者站在屏幕左前缘，讲师台、观众头部与举手拍摄遮挡屏幕下沿和左下部；华为 AI 场屏幕在画面中占比小、摩尔纹重。目标仍为 LED 屏完整四边形（含底部页脚条），人工四角外推到被遮挡的屏幕边缘，不沿人头、讲者或讲台收边。已写入 `severe_audience_head_occlusion`、`speaker_in_front_of_screen`、`large_occluded_screen_area` 等场景标签，但不代表逐角可见性。

两场源目录照片全部列入项目，无未纳入照片。与既有全部人工标注（含同峰会工业具身智能、AI铸链）比对：无精确字节重复、无 dHash /pHash 阈值 8 内近重复、无路径冲突；与 round_031 全历史 2275 页 `audit_splits` 预检同样 0 重叠。龚涛场 26 个内容组中包含按相近版式合并的组，分组偏保守、整组同用途，不造成训练 /评估泄漏。

场景族由 v1 追加为 v2：四场共 76 页（44 训练 / 17 开发 / 15 已查看回归）。v1、工业具身智能 split、round_024 快照与冻结 conference_v1 不回写；两场均为 registered_not_trained，尚未进入任何训练轮次。

本次未启动训练、未创建新 round、未修改正式 runtime 或安装包，正式模型继续 round_031 / 0.2.22。8 页与 1 页回归均为已查看回归，不是盲测。

[华为 AI（遮挡 / 摩尔纹）](../../data/curated/scenes/202610-embodied-intelligence-huawei-ai_v1/README.md) · [龚涛（遮挡较大）](../../data/curated/scenes/202610-embodied-intelligence-gongtao_v1/README.md) · [场景族 v2](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v2/README.md)。
<!-- embodied-occluded-dataset-v1:end -->

<!-- embodied-ai-chain-dataset-v1:start -->
## 2026-10-22 上海 IN³C「AI铸链 向实而生」4 页已接入训练数据，尚未训练

上海 2026 国际工业互联网大会（IN³C）期间「具身智能•边缘计算赋能新质智造峰会」的 2026-10-14 场次「AI铸链 向实而生」（京东工业）4 张显式人工框已纳入 `202610-embodied-intelligence-ai-chain`：**2 训练 / 1 开发 / 1 已查看回归**（4 个内容组，60/20/20 拆分，seed 20261017）。目标为宽幅 LED 舞台屏内完整中央 PPT 幻灯片四边形，排除左右品牌固定区与底部「2026国际工业互联网大会（IN³C）」会议横幅；保留浅色幻灯片细边框、蓝色标题页、透视、LED 拍摄纹理与底部观众头部遮挡。4 张在接入前已经查看，1 张只作已查看回归，不能宣称盲测。

与既有全部人工标注（含同峰会 2026-10-13 工业具身智能 26 页）比对：无精确字节重复、无 dHash /pHash 阈值 8 内近重复、无路径冲突；与 round_031 全历史 2275 页做 `audit_splits` 预检同样 0 重叠。按会议 /场次建立场景族 v1：10-13 工业具身智能 26 页（15 /6 /5，round_024 已训练）+ 10-14 AI铸链 4 页，共 30 页（17 训练 /7 开发 /6 已查看回归）。工业场次人工点、用途与 round_024 快照不回写，本次仅为其补建场景目录与联系表。

本次未启动训练、未创建新 round、未修改正式 runtime 或安装包，正式模型继续 round_031 / 0.2.22。

[场景与数据审计](../../data/curated/scenes/202610-embodied-intelligence-ai-chain_v1/README.md) · [场景族 v1](../../data/curated/scenes/202610-shanghai-embodied-intelligence_v1/README.md)。
<!-- embodied-ai-chain-dataset-v1:end -->

<!-- nanjing-edge-dataset-v1:start -->
## 2026-10-22 南京边缘算力论坛 67 页已接入训练数据，尚未训练

南京第十届未来网络大会「边缘算力与泛在智能创新发展论坛」（2026-10-15 下午场）67 页显式人工框已纳入 `202610-nanjing-edge-computing-forum_v1`：**39 训练 / 13 开发 / 15 已查看回归**（按矫正 PPT 内容分组的 60/20/20 拆分，seed 20261017）。目标为宽幅 LED 屏内完整 PPT 幻灯片四边形，排除左右固定信息区与演讲人标题 /头像区；保留白底密集表格 /图表、深色页面、透视、LED 拍摄纹理和底部观众 /椅背遮挡，人工点不沿遮挡轮廓收边。源项目 68 张照片中 1 张未列入项目页面，且与其余照片非精确重复，不补伪标签、不进入训练与准确率分母。

标注、固定 split、registry 与场景 v1 均已生成；南京中央 PPT 场景族追加为 v3，共 215 页（126 训练用途 /44 开发 /45 已查看回归）。跨场次与主论坛（上 /下）、智算论坛检查：无精确字节重复、无路径冲突；dHash /pHash 阈值 8 下 5 对 pHash 标记经逐对目视均为不同幻灯片（dHash 21–31），未改变既有用途。组内 63 个内容组中两组合并了实际不同页面，分组偏保守、不造成训练 /评估泄漏。旧 revision、旧用途与冻结 conference_v1 不回写。

本次未启动训练、未创建新 round、未修改正式 runtime 或安装包，正式模型继续 round_031 /0.2.22。67 页在接入前已查看，15 页回归仍为已查看回归而非盲测。

[场景与数据审计](../../data/curated/scenes/202610-nanjing-edge-computing-forum_v1/README.md) · [场景族 v3](../../data/curated/scenes/202610-nanjing-future-network-central-ppt_v3/README.md)。
<!-- nanjing-edge-dataset-v1:end -->

<!-- round031-promotion-v0222:start -->
## 2026-10-22 round_031 全历史模型已晋升，0.2.22 本地包已验收

按用户要求把全历史独立训练的单文件模型晋升为正式 runtime `screen_corner_single_052594290a1d`（round_031，seed=11），发布编号 `model-20261022-131533-05259429`，旧权重备份 `models/runtime/screen_corner_single.pre_round027.pt`。与正式 round_027 同图重建推理：432 页开发 1496→1522（86.57%→88.08%），216 页已查看回归 715→770（82.75%→89.12%），原 428 页子集 1423→1432，冻结 1808 页诊断 6686→6882，全部 2275 页诊断 8182→8640；平均归一化误差开发 1.91%→0.76%、回归 2.16%→0.73%。冻结诊断仍有三处回退并写入发布元数据：`202604-mall` -59（疑似把屏幕外边黑框纳入目标）、`202312-openatom` -30、`202603-fuzhou-highway` -13。同轮精修专项 round_032（冻结该基座、只重训局部头 96 轮）开发 1519、回归 773，未超过基座，说明精修瓶颈不是训练轮数，未采用。0.2.22 macOS Apple Silicon 安装包已生成并验收：Finder 布局与 0.2.16 逐字节一致、版本 0.2.22、ad-hoc 严格签名、包内真实检测来源 `runtime_single` 且模型摘要一致；本地 ad-hoc 签名、未公证、未对外发布，包内含本次构建的 updater 签名产物但没有 latest.json，不构成更新通道。

[晋升审计](../../training/runs/screen_corner_single/round_031/reports/promotion-audit/README.md) · [决策记录](../../training/runs/screen_corner_single/round_031/reports/promotion_decision.json) · [下载 0.2.22（macOS Apple Silicon）](../../release-assets/local-single-20261022-v0222-round031/0.2.22/macos/ScreenPDF_0.2.22_aarch64.dmg)。DMG SHA256：`8c35eb39c9709ac14e04b84da73e3a9b9845ff7c42fd4abf9008eed45d355773`。
<!-- round031-promotion-v0222:end -->


<!-- all-history-round030:start -->
## 2026-10-22 全历史独立单体模型 round_030

round_030 已根据全部历史可用人工资料独立训练区域识别 → 粗定位 → 精修单体模型。62 个来源、2275 页，1461 训练 /432 开发 /216 已查看回归，166 张近评估重复训练页隔离。仅通用 ImageNet 初始化，骨干和三个任务阶段重新训练，未继承 ScreenPDF 业务权重或教师。候选 `screen_corner_single_500bb4820b86`；开发最终正式 1496/1728（86.57%） → 新模型 1508/1728（87.27%）（+12 点），已查看回归正式 715/864（82.75%） → 新模型 762/864（88.19%）。南京开发 +33、InfoComm +7，openatom -18、福州高速 -12 等局部回退保留。12 /80 /48 轮固定预算，按 pooled 开发唯一角点指标冻结，先冻结再回归；含训练资料的 1808 页诊断（1 页数据集修订排除页按冻结人工框补算）/2275 页诊断与逐会议变化另报。seed=11 复验（round_031）开发 1522、已查看回归 770，粗定位 1496/1497 复现，final 增益带约 ±0.8 个百分点种子波动；精修净贡献仅 +12/+25（成熟模型 +191），openatom /福州高速 /smart-road /metaverse /商场测试集回退两 seed 复现，故建议先修精修阶段再谈晋升。23 项测试和隔离单文件实际检测通过；两 seed 复验后，固定预算、历史评估已查看等限制仍保留。未晋升 /打包，正式 runtime 仍 round_027。

[完整报告](../../research/experiments/2026-10-22-all-history-region-coarse-local-training.md) · [seed 复验与综合指标](../../research/experiments/2026-10-22-all-history-seed-replication.md)。
<!-- all-history-round030:end -->

<!-- computing-round029:start -->
## 2026-10-22 智算论坛 round_029 粗定位训练完成

智算网络创新技术论坛 66 页已接入（39 训练 /14 开发 /13 已查看回归），保留完整中央 PPT 人工四角和底部观众 /椅背遮挡。round_029 从 round_028 候选做相同 20 轮、每轮 182 次更新的粗定位对照，选中直接加入组第 16 轮 `screen_corner_single_c495b67dcac4`；本轮按智算开发粗定位选优，encoder /local head 冻结。新场景开发粗定位 17→45/56、底部 5→19/28，已查看回归粗定位 9→33/52、最终 30→42/52。区域拟合分支开发为 34/56，旧域明显回退，未采用。选中候选旧 414 页开发粗定位 -21 点、最终 -12 点，pooled 最终 1535→1534/1712；左下仅 6/14，冻结精修会重新拉偏部分正确粗点。南京场景族 v2 共 148 页，旧用途 /revision /冻结会议集保留。33 项测试和四个隔离单文件实际检测通过；未晋升 /打包，正式 runtime /桌面包继续 round_027 /0.2.21。

[完整报告](../../research/experiments/2026-10-22-computing-forum-region-coarse-training.md)。
<!-- computing-round029:end -->

<!-- nanjing-pm-round028:start -->
## 2026-10-22 南京主论坛（下）round_028 已完成，候选未晋升

南京主论坛（下）40 页显式人工框已加入 `202610-nanjing-future-network-main-pm_v1`（24 训练 /8 开发 /8 已查看回归），中央 PPT 为目标，排除两侧人物直播区，保留摄像摇臂 /观众遮挡；另 5 张未列入项目照片不纳入。上下半场同族索引共 82 页，旧用途不回写。round_028 从正式 round_027 继续，control=3351、加入组=3927 条采样（新 24 页各 24 倍），相同 seed=7、每轮 123 次更新，先 12 轮粗定位再 16 轮精修。加入组粗定位 pooled 未胜初始，保留 epoch=0；新增页仍实际参与粗定位训练与最终精修，精修第 10 轮选中 `with_nanjing_pm`，候选 `screen_corner_single_a09836ddb464`。414 页开发 initial=1496/1656、control=1500/1656、加入组=1501/1656。下半场开发 10→14/32、已查看回归 8→11/32；上半场开发 15→17/36、回归 19→21/36。原 265 页开发少 2 点，旧新 174 页回归 624→622/696（旧 166 页少 5 点）；收益较小且残留粗定位 /摇臂干扰限制保留。29 项测试、组件冻结与缓存 /完整图一致性、隔离单权重真实检测通过。候选未晋升，未重跑全量晋升审计，正式 runtime 和安装包继续 round_027 /0.2.21。

[训练报告与叠框](../../research/experiments/2026-10-22-nanjing-main-pm-training.md)。
<!-- nanjing-pm-round028:end -->

<!-- round027-promotion-v0221:start -->
## 2026-10-22 round_027 已晋升，0.2.21 本地安装包已验收

已按用户“整体退化不严重则晋升并重新打包”的授权，将 round_027 晋升为正式单文件 runtime `screen_corner_single_eef7e8e646aa`，模型发布编号 `model-20261022-020033-eef7e8e6`。与正式 round_024 重新推理冻结 1808 页：6697→6686/7232，下降约 0.15 个百分点（改善 72、回退 83 点）；当前 406 页开发 1468→1486/1624，166 页已查看回归 601→616/664，原 397 页开发 1466→1471/1588。快照外少 12/1068，杭州隧道集合少 7/244、北京 InfoComm 少 5/80，局部回退继续保留；全集含训练页，仅作覆盖/拟合诊断。南京开发 2→15/36、已查看回归 7→19/36（相对正式 round_024）。0.2.21 macOS Apple Silicon 本地安装包已重新生成，最终 DMG 的严格签名、Finder 布局与 Applications 拖放入口、唯一权重摘要及包内真实检测均通过；本地 ad-hoc 签名，未公证或外部发布，不含 updater 产物。

[整体回归审计](../../training/runs/screen_corner_single/round_027/reports/promotion-audit/README.md) · [下载 0.2.21（macOS Apple Silicon）](../../release-assets/local-single-20261022-v0221-round027/0.2.21/macos/ScreenPDF_0.2.21_aarch64.dmg)。DMG SHA256：`31fe5df6c1169d3e94497e6ecf42bb0d2e782bd38e0bb5a85364e4dd8fe65ee2`。
<!-- round027-promotion-v0221:end -->

<!-- nanjing-staged-round026-027:start -->
## 2026-10-22 南京场景先粗定位、再精修已完成

南京中央 PPT 按用户要求先粗后细：round_026 冻结 encoder /local head，12 轮粗定位对照选中 coordinate_guard，pooled 粗定位 1302→1312/1624，南京开发 8→11/36，真实角点进入局部裁剪 19→23/36；空间交叉熵分支退化未采用。round_027 冻结 encoder /global head，16 轮精修对照选中 point_hit，第 4 轮单文件候选 `screen_corner_single_eef7e8e646aa`。南京最终开发 12→15/36、已查看回归 15→19/36；合并开发 1478→1486/1624，原 397 页开发 1466→1471/1588、原 265 页 981→992/1060。两阶段分别冻结选择后才评回归；保留远景上角误吸吊顶、中央区域定位失败、单 seed 与已查看回归限制。32 项测试、组件冻结摘要和隔离单权重真实检测通过，训练完成时尚未重跑 1808 页冻结全量基准或晋升；后续已按用户授权完成审计并晋升 round_027，见最新晋升记录。

[训练报告与分阶段叠图](../../research/experiments/2026-10-22-nanjing-coarse-then-fine-training.md)。
<!-- nanjing-staged-round026-027:end -->

<!-- nanjing-round025:start -->
## 2026-10-22 南京主论坛（上）round_025 已完成，候选未晋升

南京第十届未来网络大会主论坛（上）42 页显式人工框已接入独立中央 PPT 场景 `202610-nanjing-future-network-main-am_v1`（24 训练 /9 开发 /9 已查看回归），排除两侧人物直播区；另 4 张未列入源项目的照片不纳入。round_025 从 round_024 做固定预算 teacher_guard 对照，2775 /2919 条采样，共享 406 页开发，选中 `with_nanjing`：`screen_corner_single_c7f83295c7e7`。pooled 开发 initial=1468/1624、control=1477/1624、加入组=1478/1624；南京开发 2→12/36、已查看回归 7→15/36。旧 397 页开发同为 1466/1588，原始 265 页少 9 点；远景上角仍误吸吊顶、中央边界与遮挡失败保留。单 seed，本轮未重跑冻结全量基准，候选未晋升，正式 runtime 仍为 round_024。

[训练报告](../../research/experiments/2026-10-22-nanjing-main-forum-training.md)。
<!-- nanjing-round025:end -->

<!-- round024-mazhangyixiang-industrial:start -->
## 2026-10-19 round_024 马张翼翔 / 工业具身智能已完成并晋升

马张翼翔 47 页已接入杭州宽屏场景族（27 训练 / 10 开发 / 10 已查看留出）；工业具身智能 26 页作为独立新场景接入（15 训练 / 6 开发 / 5 已查看留出）。四组从 round_023 正式权重做固定预算 teacher_guard 对照，新增训练页分别按 6 倍和 12 倍回放，397 页 pooled 开发只按 `point_le_0_01_ratio` 选优。

initial=1448/1588、control=1450/1588、with_mazhang=1464/1588、with_industrial=1460/1588、with_both=1466/1588，选择并晋升 `with_both`：`screen_corner_single_7f2257153f0c`，发布编号 `model-20261018-173619-7f225715`。马张翼翔开发 / 留出为 39/40、40/40；工业具身智能开发 / 留出为 21/24、20/20。工业场景粗定位从 3/24、1/20 提升到 12/24、16/20；最终单文件输出仍有 3 个开发角点未命中，限制保留。

本轮未重跑 1808 页冻结会议全量基准。随后已重新生成 0.2.20 macOS Apple Silicon 本地包并完成 DMG、包内单权重、真实检测、ad-hoc 签名和 Finder 拖放布局验收；DMG SHA256 为 `45b00811e947fb4dc93f4e55072969de232cf77805c575771bcc1ced7f9c31b8`。本机没有 Tauri updater 私钥，因此未生成可发布的 updater 签名产物，也未做 Apple 公证。完整报告：[2026-10-19-mazhangyixiang-industrial-training.md](../../research/experiments/2026-10-19-mazhangyixiang-industrial-training.md)。
<!-- round024-mazhangyixiang-industrial:end -->

<!-- gengchiyuan-round023:start -->
## 2026-10-18 耿驰远 / 杜超 round_023 已完成

耿驰远 36 张已接入（22 训练 / 7 开发 / 7 已查看回归），杜超 14 张沿用 8 / 3 / 3。round_023 从 round_022 候选 c2267f9e0d71 初始化，完成 control / with_hangzhou / family_focus 三组固定预算对照；各组 seed=7、4/8 轮、每轮 103 次更新，训练采样 2253 / 2433 / 3291 条。合并开发 381 页：initial=1397/1524, control=1404/1524, with_hangzhou=1408/1524, family_focus=1404/1524，候选 `8e554a8d68b7`。选中加入两场新资料的 with_hangzhou；提高同族采样比例的分支未超过它。同族开发 210/256 → 228/256，已查看回归 209/252 → 230/252；原 265 页开发 993/1060 → 988/1060。场景族 v5 共 11 场、308 页；143 张实际采样的同族训练页在定向组提高至 12 倍，刘奇/孙泉明未采样训练页没有借此引入。旧用途、旧 revision、冻结会议集及源人工点保留；单种子、反复使用开发集和残留失败的限制继续保留。正式 runtime 已按用户要求晋升为 round_023：`screen_corner_single_8e554a8d68b7`，发布编号 `model-20261018-154105-8e554a8d`，SHA256 `8e554a8d68b7b7651c52f3050a0c7b2e2383a156a2e65f835f1a902683ab6033`。未重跑 1808 页全量基准；旧 265 页开发少 5 点，右上与遮挡右下失败仍保留。

[完整报告](../../research/experiments/2026-10-18-gengchiyuan-family-focus-training.md)。
<!-- gengchiyuan-round023:end -->

<!-- duchao-dataset-v1:start -->
杜超资料初次登记为 8 训练 / 3 开发 / 3 已查看回归，原登记快照保留；随后已在 round_023 的 with_hangzhou 与 family_focus 两组执行训练。选中加入两场新资料的 with_hangzhou；提高同族采样比例的分支未超过它。 详见 [完整报告](../../research/experiments/2026-10-18-gengchiyuan-family-focus-training.md)。
<!-- duchao-dataset-v1:end -->

<!-- zhangwenju-round022:start -->
## 2026-10-18 张文居资料 round_022 已完成

张文居资料 48 张已接入 `202610-tunnel-lighting-zhangwenju_v1`（28 训练 / 10 开发 / 10 已查看回归），目标为宽幅 LED 屏内完整右侧 PPT，保留白底/蓝底、地图图表、工程照片、LED 条纹、倒计时与观众，原人工点不改写。round_022 从正式 round_021 做相同 teacher_guard 的固定预算对照，旧 2085 条采样保留，新增 28 页各 6 倍共 2253 条；两组均 4/8 轮、每轮 71 次更新。合并开发 initial=1335/1460, control=1351/1460, with_zhangwenju=1349/1460，选中 `control`，单文件候选 `c2267f9e0d71`。选中候选未使用张文居新增训练页，加入组权重另保留。本场开发 32→35/40、已查看回归 33→33/40；既有 355 页开发净变化 +13 点，原 265 页开发 +3 点。场景族 v3 共 258 页，旧用途/v1/v2/冻结会议集不回写；小样本、单 seed 与残留失败保留。正式 runtime 继续 round_021，候选未晋升。

[完整报告](../../research/experiments/2026-10-18-zhangwenju-partial-screen-training.md) · [训练配置](../../training/runs/screen_corner_single/round_022/manifest.json)。
<!-- zhangwenju-round022:end -->


## 2026-10-18 round_023 已按用户要求晋升，0.2.20 本地包已验收

正式 runtime 为 `screen_corner_single_8e554a8d68b7`，发布编号 `model-20261018-154105-8e554a8d`，SHA256 `8e554a8d68b7b7651c52f3050a0c7b2e2383a156a2e65f835f1a902683ab6033`。候选选择为 `with_hangzhou`，合并开发 1408/1524（92.388%），杭州同类场景开发 228/256（89.063%），已查看回归 230/252（91.270%）。

本地 macOS Apple Silicon 包已重新生成并完成验收：

- [ScreenPDF_0.2.20_aarch64.dmg](../../release-assets/local-single-20261018-v0220-round023/0.2.20/macos/ScreenPDF_0.2.20_aarch64.dmg)，SHA256 `0475e4e7dbada0ce4246b59db7b08e8b8f45b80e79b09005e7d5cf66444f197a`。
- 包内只有 `screen_corner_single.pt`，模型 ID、SHA1、SHA256 与正式 runtime 一致；使用耿驰远 `IMG_2489.jpeg` 的真实检测返回 `runtime_single` / `screen_corner_single`。
- Finder 布局与 0.2.16 基准逐字节一致，含 `Applications -> /Applications`；版本、严格签名和拖放入口均通过检查。
- 本地 ad-hoc 签名，未做 Apple 公证或外部发布。
- [包内验证](../../release-assets/local-single-20261018-v0220-round023/0.2.20/macos/bundle-verification.json) · [DMG 验证](../../release-assets/local-single-20261018-v0220-round023/0.2.20/macos/dmg-verification.json) · [SHA256SUMS](../../release-assets/local-single-20261018-v0220-round023/0.2.20/macos/SHA256SUMS.txt)。
## 2026-10-18 孔令敏资料 round_021 已完成

孔令敏 AI + PLC 隧道照明资料 39 张已接入杭州宽幅 LED 右侧 PPT 场景：23 训练 / 8 开发 / 8 已查看回归，3 段视频不进入图片训练。保留完整人工边界、LED 条纹、倒计时及观众遮挡；场景族使用追加成员的 v2，旧拆分和冻结集合不回写。

从正式 round_018 做固定预算对照，两组均用同一 teacher_guard、seed=7、4 轮粗定位 / 8 轮精修、每轮 66 次更新；control 1947 条采样，加入组新增 23 页各 6 倍，共 2085 条。合并开发 initial=1292/1420、control=1292/1420、with_konglingmin=1303/1420，冻结候选 `screen_corner_single_7279848d9388`。本场开发 13/32→23/32，已查看回归 11/32→24/32；上方两角和左下改善，右下开发仍仅 1/8、回归 3/8。既有 347 页开发净增 1 点，但原 265 页开发回退 3 点；单 seed、同场小样本且未重跑全量晋升基准，训练完成时仅注册候选；随后按用户要求晋升 round_021，见上方晋升记录。

详情：[训练报告](../../research/experiments/2026-10-18-konglingmin-fine-corner-training.md) · [固定拆分](../../data/splits/cross_project/202610-tunnel-lighting-konglingmin_v1.json)。

## 2026-10-18 round_018 教师保护训练已完成并晋升

按瓶颈研究建议，以正式 round_013 作为训练期教师，只对教师原本命中的旧训练角点增加正向保护；教师错误点不蒸馏，学生更接近人工真值时不拉回教师。round_018 使用 round_016 林燕丹快照、相同预算和 seed=7，教师掩码为 4206/4320 个旧训练角点。

合并开发普通对照为 1274/1388，教师保护为 1279/1388。冻结 1808 页全量基准中，round_013 为 6650/7232，round_018 教师保护为 6668/7232（93 个改善、75 个回退、净增 18）；当前开发页 1123→1130，快照外页 654→661，已查看回归页 351→351。按用户要求已晋升为正式 runtime：模型 ID `screen_corner_single_5814ce182305`，SHA256 `5814ce18230542653ff76d5d2ebcf0499d2b7b33a8f01741fdac4cb6e9524cc4`，发布编号 `model-20261018-124518-5814ce18`。seed=8/9 的开发复核分别为 -3/0 个角点，稳定性限制保留并安排后续复核。完整结果、回退图册和逐点审计见 [round_018 报告](../../research/experiments/2026-10-18-teacher-guard-training.md) 与 [全量基准](../../research/experiments/conference-benchmark/20261018_round018/README.md)。

额外 seed=8/9 的同预算开发复核分别为教师保护 −3/0 个角点（seed=7 为 +5），方向不稳定；本次按用户要求晋升冻结的 seed=7 权重，不将教师保护方法本身表述为已稳定改善。

## 2026-10-18 0.2.19 本地安装包已按 round_018 重建并验收

- macOS Apple Silicon：`release-assets/local-single-20261018-v0219-round018/0.2.19/macos/ScreenPDF_0.2.19_aarch64.dmg`。
- 包内唯一权重为 `screen_corner_single_5814ce182305`，SHA256 为 `5814ce18230542653ff76d5d2ebcf0499d2b7b33a8f01741fdac4cb6e9524cc4`，实际冒烟检测来源为 `runtime_single`。
- DMG SHA256：`11b56ed4beb9d313504052129b0e3f794900fbc1b64a345e732a6580b0155de5`；Finder 布局与 0.2.16 基准一致，`Applications -> /Applications`、版本、ad-hoc 签名和包内模型摘要均已核验。
- 包验证记录：[mounted-bundle-verification.json](../../release-assets/local-single-20261018-v0219-round018/0.2.19/macos/mounted-bundle-verification.json) · [dmg-verification.json](../../release-assets/local-single-20261018-v0219-round018/0.2.19/macos/dmg-verification.json) · [SHA256SUMS.txt](../../release-assets/local-single-20261018-v0219-round018/0.2.19/macos/SHA256SUMS.txt)。本地 ad-hoc 签名，未做 Apple 公证或外部发布。

## 2026-10-18 林燕丹资料 round_016 已完成

林燕丹资料 25 张已接入杭州隧道照明右侧 PPT 场景族：15 训练 / 5 开发 / 5 已查看回归。round_016 从 round_015 control 继续，旧 1857 条训练采样保持不变，新增 15 张训练页各 6 倍为 1947 条；合并开发选中 `screen_corner_single_155173d8bd62`（1274/1388，91.787%）。林燕丹开发 15/20→17/20→17/20，已查看回归 14/20→14/20→14/20；杭州场景族开发 40/52→39/52→43/52，回归 42/48→42/48→45/48。

冻结全量基准相对正式 round_013：当前开发页 1123/1220→1127/1220，已查看回归 351/364→355/364，快照外页面 654/984→648/984。新增候选改善同场景和已查看回归，但快照外旧域回退 6 点，未晋升，正式 runtime 继续使用 `screen_corner_single_5ba8b4a37a5d`。详见[训练报告](../../research/experiments/2026-10-18-linyandan-partial-screen-training.md)和[全量基准](../../research/experiments/conference-benchmark/20261018_round016/README.md)。

## 2026-10-18 round_015 全量冻结基准已完成

对 1808 页有效人工标注、7232 个角点重新运行全量基准，统一比较正式 round_013、李华斌候选、round_015 control 和刘松荣候选。结果按 `point_le_0_01_ratio` 汇总：

| 模型 | 全部人工页（诊断） | 当前开发页 | 已查看回归页 | 快照外页 |
| --- | ---: | ---: | ---: | ---: |
| round_013 | 6650/7232（91.95%） | 1123/1220（92.05%） | 351/364（96.43%） | 654/984（66.46%） |
| round_014 李华斌 | 6624/7232（91.59%） | 1123/1220（92.05%） | 348/364（95.60%） | 633/984（64.33%） |
| round_015 control | 6645/7232（91.88%） | 1125/1220（92.21%） | 350/364（96.15%） | 652/984（66.26%） |
| round_015 刘松荣 | 6646/7232（91.90%） | 1124/1220（92.13%） | 347/364（95.33%） | 647/984（65.75%） |

round_015 control 在当前开发页比 round_013 多 2 个命中，但已查看回归少 1 个、快照外少 2 个；刘松荣候选的局部杭州场景提升没有转化为全量泛化收益。李华斌候选在快照外回退 21 个角点。全量结果不支持晋升，正式 runtime 继续使用 `screen_corner_single_5ba8b4a37a5d`。完整逐页预测、分集合统计和模型摘要见[全量基准报告](../../research/experiments/conference-benchmark/20261018_round015/README.md)。

## 2026-10-18 李华斌宽幅 LED 右侧 PPT round_014 已完成

李华斌资料 17 张已接入：9 训练 / 4 开发 / 4 已查看回归。它与涂耘、刘奇、孙泉明、李薇组成同场 `202610-hangzhou-tunnel-partial-ppt` 场景族，目标为右侧 PPT，排除左侧蓝色固定信息区；保留蓝色和深色屏幕背景、LED 拍摄条纹、右上角倒计时叠层和底部观众遮挡，人工四角不沿遮挡轮廓收边。

round_014 从正式 round_013 `screen_corner_single_5ba8b4a37a5d` 做固定预算对照：旧 1749 条训练采样保持不变，新增 9 张训练页各 6 倍，`with_lihuabin` 共 1803 条。合并开发 initial=1240/1356（91.445%）、control=1240/1356（91.445%）、with_lihuabin=1241/1356（91.519%），选择候选 `screen_corner_single_2343521e5471`；李华斌开发 4/16→5/16、已查看回归 4/16→9/16，旧验证回退 3 点，候选未晋升，正式 runtime 保持 round_013。

详情：[训练报告](../../research/experiments/2026-10-18-lihuabin-partial-screen-training.md)。数据入口：[场景目录](../../data/curated/scenes/202610-tunnel-lighting-lihuabin_v1/) · [固定拆分](../../data/splits/cross_project/202610-tunnel-lighting-lihuabin_v1.json)。

## 2026-10-18 刘松荣宽幅 LED 右侧 PPT round_015 已完成

刘松荣资料 15 张已接入：9 训练 / 3 开发 / 3 已查看回归。它与涂耘、刘奇、孙泉明、李薇、李华斌组成同场 `202610-hangzhou-tunnel-partial-ppt` 场景族，目标为右侧 PPT，排除左侧固定蓝色信息区；保留白底/蓝底页面、密集表格图表、LED 拍摄条纹、右上角倒计时叠层和底部观众遮挡，人工四角不沿遮挡轮廓收边。

round_015 沿用 round_014 同场训练快照：旧 1803 条训练采样保持不变，新增 9 张训练页各 6 倍，`with_liusongrong` 共 1857 条。合并开发 initial=1247/1368（91.155%）、control=1251/1368（91.447%）、with_liusongrong=1251/1368（91.447%），按平局规则选择 control；刘松荣开发 6/12→8/12、已查看回归 8/12→9/12，旧验证回退 3 点，候选未晋升，正式 runtime 保持 round_013。

详情：[训练报告](../../research/experiments/2026-10-18-liusongrong-partial-screen-training.md)。数据入口：[场景目录](../../data/curated/scenes/202610-tunnel-lighting-liusongrong_v1/) · [固定拆分](../../data/splits/cross_project/202610-tunnel-lighting-liusongrong_v1.json)。

<!-- sunquanming-round012:start -->
## 2026-10-18 孙泉明黑底宽屏 PPT round_012 已完成

孙泉明资料 29 张已接入：17 训练 / 6 开发 / 6 已查看回归。它与涂耘、刘奇组成同场 `202610-hangzhou-tunnel-partial-ppt` 场景族，目标为右侧 PPT，排除左侧蓝色固定信息区；保留黑色屏幕背景、LED 拍摄条纹、右上角倒计时 / 红色超时叠层和底部观众遮挡，人工四角不沿遮挡轮廓收边。

round_012 从正式 round_011 `screen_corner_single_585aaa418750` 做固定预算对照：旧 1665 条训练采样保持不变，新增 17 张训练页各 6 倍，`with_sunquanming` 共 1767 条。合并开发 1231/1344（91.592%），初始 1223/1344、control 1218/1344；孙泉明开发 8/24→16/24，已查看回归 2/24→15/24。选择 `with_sunquanming`，候选 `screen_corner_single_a1f4bec7b047` 未晋升；旧开发回退 7 个角点，正式 runtime 在该轮仍为 round_011。

数据入口：[场景目录](../../data/curated/scenes/202610-tunnel-lighting-sunquanming_v1/) · [固定拆分](../../data/splits/cross_project/202610-tunnel-lighting-sunquanming_v1.json) · [训练报告](../../research/experiments/2026-10-18-sunquanming-black-screen-training.md)。
<!-- sunquanming-round012:end -->

<!-- liwei-round013:start -->
## 2026-10-18 李薇宽幅 LED 右侧 PPT round_013 已完成

李薇资料 24 张已接入：14 训练 / 5 开发 / 5 已查看回归。它与涂耘、刘奇、孙泉明组成同场 `202610-hangzhou-tunnel-partial-ppt` 场景族，目标为右侧 PPT，排除左侧蓝色固定信息区；保留黑色屏幕背景、页内亮色内容、LED 拍摄条纹、右上角倒计时 / 红色超时叠层和底部观众遮挡，人工四角不沿遮挡轮廓收边。

round_013 从正式 round_011 `screen_corner_single_585aaa418750` 做固定预算对照：旧 1665 条训练采样保持不变，新增 14 张训练页各 6 倍，`with_liwei` 共 1749 条。合并开发 initial=1230/1340（91.791%）、control=1227/1340（91.567%）、with_liwei=1236/1340（92.239%），选择 `with_liwei`；候选 `screen_corner_single_5ba8b4a37a5d` 已晋升。李薇开发 15/20→16/20，已查看回归 14/20→17/20；旧开发相对 initial 上升 1 点，正式 runtime 已更新为 round_013。

数据入口：[场景目录](../../data/curated/scenes/202610-tunnel-lighting-liwei_v1/) · [固定拆分](../../data/splits/cross_project/202610-tunnel-lighting-liwei_v1.json) · [训练报告](../../research/experiments/2026-10-18-liwei-partial-screen-training.md)。

冻结全量基准相对 round_011：13 个集合中 9 个上升、3 个回退、1 个持平，累计上升 40 个角点、回退 6 个；已按晋升规则更新正式 runtime。全量对比：[round013 基准](../../research/experiments/conference-benchmark/20261018_round013/README.md)。
<!-- liwei-round013:end -->

## 2026-10-18 最新：0.2.18 本地安装包已按 round_013 重建并验收

- macOS Apple Silicon：`release-assets/local-single-20261018-v0218-round013/0.2.18/macos/ScreenPDF_0.2.18_aarch64.dmg`。
- 包内唯一权重为已晋升的 round_013 单文件模型 `screen_corner_single_5ba8b4a37a5d`；SHA256：`5ba8b4a37a5d3dc7a8fc891bd1215b34c1328e41c5cf3722bd09ab5c044a2e26`。
- 最终 DMG SHA256：`d684a029b94136a4abd5b18fc826820a67ccdaf867bd828390f2dc6d8d65649d`；app tar.gz SHA256：`cae230e0a5daeb67f3e6c56b70b5deb143af797935b4c4e7c23c58e26b546809`。
- 前端构建、Rust/Tauri 构建、ad-hoc 签名、DMG 挂载后的版本核对和包内实际识别均通过。DMG 已与 0.2.16 完全复用 Finder 布局元数据：左侧 `ScreenPDF.app`、右侧带箭头的 `Applications -> /Applications`，可直接拖入“应用程序”；冒烟识别返回 `source=runtime_single`、模型 ID `screen_corner_single_5ba8b4a37a5d`。
- 包验证记录：[mounted-bundle-verification.json](../../release-assets/local-single-20261018-v0218-round013/0.2.18/macos/mounted-bundle-verification.json) · [dmg-verification.json](../../release-assets/local-single-20261018-v0218-round013/0.2.18/macos/dmg-verification.json) · [SHA256SUMS.txt](../../release-assets/local-single-20261018-v0218-round013/0.2.18/macos/SHA256SUMS.txt)。本地 ad-hoc 签名，未做 Apple 公证或外部发布。

## 2026-10-18 round_011 已按新标准晋升

round010 control 未晋升的旧原因是旧流程把个别场景回退作为否决因素；它相对 round009 虽然总分上升，但旧开发、董志国开发和商场回归各有少量回退。按本次新标准，从 `candidate_round010_control` 继续重训得到 round011 候选 `screen_corner_single_585aaa418750`，在冻结 1808 页集合上相对正式 round009：13 个集合中 10 个上升、3 个回退，累计上升 109 个角点、回退 7 个；排除训练页后为 6 个上升、5 个回退、2 个持平，累计上升 37、回退 13。采用“上升集合严格占多数，回退角点不超过上升的一半”的可复现解释，round011 **已晋升**。

- 单文件 runtime：`models/runtime/screen_corner_single.pt`，SHA256 `585aaa4187501d8faa3c3c80468f38bfa97948df005b77aec58aa82ad37387ee`。
- 模型发布信息：[screen_corner_single.json](../../models/runtime/screen_corner_single.json)；晋升审计：[promotion_decision.json](../../training/runs/screen_corner_single/round_011/reports/promotion_decision.json)。
- 1808 页全量对比：[round011 benchmark](../../research/experiments/conference-benchmark/20261018_round011/README.md)。round011 开发页为 1118/1220（91.64%），已查看回归为 345/364（94.78%），快照外为 646/984（65.65%）；训练页仅作拟合诊断。

## 2026-10-18 最新：0.2.17 本地安装包已生成

- macOS Apple Silicon：`release-assets/local-single-20261017-v0217/0.2.17/macos/ScreenPDF_0.2.17_aarch64.dmg`，18,427,682 字节。
- 包内唯一权重仍为 round_009 单文件候选 `screen_corner_single_6fc6378701c4`；模型 SHA256：`6fc6378701c4a8545c3a689aa268356969cfe2ce6b7d8c6664d757d58107b2db`。本次只升级程序版本，未把 round_010 control 或之后晋升的 round_011 重新打入包；正式 runtime 的晋升记录见本页顶部。
- 已完成 Cargo 检查、前端生产构建和 55 项前端测试；包内模型摘要及实际识别入口检查通过。DMG 挂载后版本、ad-hoc 签名及真实图片识别检查通过。
- DMG SHA256：`f1d32d7cf7382e5ccb13ee7e17e48b6e569e13a72e9106ec9137b7a77dd243af`。未做 Apple 公证或外部发布。
- [构建信息](../../release-assets/local-single-20261017-v0217/0.2.17/macos/build-info.json) · [DMG 校验](../../release-assets/local-single-20261017-v0217/0.2.17/macos/dmg-verification.json)
- 后续本地构建和正式发布按 [桌面程序打包流程](../runbooks/desktop-packaging.md) 执行。

## 2026-10-17 会议场次集合与全量基准已完成

主入口：[13 个会议 / 专项集合](../../data/curated/collections/conference_v1/README.md)，覆盖 48 源项目、1950 页；按用户要求排除 4 个冲突 / 越界异常页面后，13 个可直接打开的 ScreenPDF 合并项目共 1946 页，其余原标签和候选保留，图片是独立副本。基准重新运行 1808 页明确人工标注，未标注 138 页、越界框 2 页、同图异框 2 个版本排除。

| 用途 | 0.2.17 包内 round_009（基准模型不变） | round_010 control |
| --- | ---: | ---: |
| 开发 305 页 | 90.90%（1109/1220） | 91.39%（1115/1220） |
| 已查看回归 91 页 | 92.03%（335/364） | 95.88%（349/364） |
| 当前快照外 246 页 | 65.14%（641/984） | 65.45%（644/984） |

仅使用角点命中率；训练计划页另作拟合诊断，快照外历史暴露未完全核实，不宣称盲测。按场次仍有回退：商场回归少 2 点，上海百家讲坛及 AWE 快照外分别少 2、1 点；没有根据这次结果重新选模、训练或晋升，模型与 0.2.16 安装包不变。[当前程序全量报告](../../research/experiments/conference-benchmark/20261017_v1/current-program-baseline-report.md) · [完整双模型基准](../../data/curated/collections/conference_v1/BASELINE.md) · [屏幕场景下钻](../../data/curated/collections/conference_v1/SCENES.md) · [使用约定](conference-collections.md)。

[人工校正包](../../data/curated/collections/conference_v1/conflict-review/README.md) 包含 IMG_7018 在中交机电 / 白屏混合中的两份人工框（BL 相差 30.58 像素），以及 IMG_7021 / IMG_7031 的越界框；4 页可编辑项目、坐标清单、叠框和角点放大均已生成。原项目和人工标签未覆盖；用户已将这 4 页作为异常从日常集合排除，无需等待校正；杭州智能交通年会集合由 96 页减至 92 页，冻结基准未变。

<!-- liuqi-round010:start -->
## 2026-10-17 刘奇与同场隧道照明定向 round_010 已完成

刘奇资料 35 张已接入（21 训练 / 7 开发 / 7 已查看回归），与涂耘组成 61 张超宽屏右侧 PPT 场景族，旧拆分保留。round_010 定向组两场训练页均 12 倍采样，固定预算对照完成；合并开发 initial=1206/1320、control=1212/1320、with_tunnel_focus=1206/1320，选中 control；单文件候选 `d9d6ad842126` 未晋升。同场景族开发 48.08%（25/52）→59.62%（31/52），已查看回归 60.42%（29/48）→85.42%（41/48）。定向组未胜过 control，选中候选未采用新增刘奇训练图的权重；正式 runtime 与 0.2.16 安装包未变。

详情：[训练报告](../../research/experiments/2026-10-17-liuqi-tunnel-family-training.md)。
<!-- liuqi-round010:end -->

## 2026-10-17 涂耘资料日期更名

- 当前目录：`/Users/gcssloop/WorkSpace/AIGC/展会/2026/202610 杭州-隧道照明与安全论坛/2026-10-16 隧道照明与绿色能源的深度融合之路-涂耘`。同名 PDF 和导出目录同步改名；35 份训练/数据/执行脚本引用、27 个原图入口和两份 JSONL 摘要记录已修复。
- 原项目、26 张照片、人工四角、用途拆分、模型与 0.2.16 安装包字节不变；两份派生导出 JSON 仅改路径和项目名。18 项相关测试通过，新路径实际识别结果与更名前一致。历史执行日志和原始打包记录保留，见[迁移审计](../../research/experiments/path-migrations/20261017-tuyun-date/README.md)。

## 2026-10-17 最新：0.2.16 本地安装包已生成

- macOS Apple Silicon：`release-assets/local-single-20261017-v0216/0.2.16/macos/ScreenPDF_0.2.16_aarch64.dmg`，18,432,239 字节（约 18 MB）。
- 包内唯一权重为 round_009 涂耘超宽屏局部 PPT 候选 `screen_corner_single_6fc6378701c4`；SHA256：`6fc6378701c4a8545c3a689aa268356969cfe2ce6b7d8c6664d757d58107b2db`。旧项目读取、v2 保存与目录移动规则继续保留，正式 runtime 未晋升；0.2.15 仍使用 round_008。
- 11 项打包相关测试、9 项模型/项目文件测试、55 项前端测试及打包脚本测试通过。挂载最终 DMG 后核对版本、签名、唯一权重、SHA1/SHA256 和包内源文件，并以倒计时/人头、红色超时、蓝色结束页三张照片验证实际识别入口。这里验证包内入口与模型一致，不表示角点准确率达标；左下角和蓝色结束页的残留失败仍见训练报告。
- 本地 ad-hoc 签名，未 Apple 公证或外部发布。配置、模型信息、测试、构建和最终 DMG 验证日志随产物保存；DMG SHA256：`c9b71eba12a2f9a7dab279a3cca2c763f0545a581dcd95c7c3719b445139369a`。


<!-- tuyun-round009:start -->
## 2026-10-17 涂耘超宽屏局部 PPT round_009 已完成

- 涂耘隧道照明资料 26 张已按独立超宽屏局部 PPT 场景 `202610-tunnel-lighting-tuyun_v1` 接入：15 训练 / 6 开发 / 5 已查看回归。目标仅为右侧 PPT，排除左侧固定信息区；保留 LED 拍摄条纹、右上角倒计时/红色超时叠层与左下角人头遮挡，沿用原人工四角，不沿人头或计时框收边。
- round_009 固定预算对照完成，合并开发 initial=1180/1292、control=1179/1292、with_tuyun=1194/1292，选中 with_tuyun；未晋升单文件候选 `6fc6378701c4`。新场景开发 8/24→13/24、已查看回归 6/20→11/20。 左下角仍失败（开发 0/6、回归 0/5），蓝色结束页 0/4；右上角仅部分改善，不能称遮挡/计时干扰已解决。
- 旧 1575 条训练采样完整保留，新图固定 6 倍，共 1665 条、1240 张独立训练图；两组各 4 轮全局、8 轮局部，每轮 53 次更新，共用 323 张开发集，只按合并角点命中率选优。
- 40 项相关测试和训练/评估路径、文件字节隔离审计通过。单文件独立加载、实际程序调用和源文件保全检查通过；[完整结果](../../research/experiments/2026-10-17-tuyun-partial-screen-training.md)包含 TR/BL 诊断、旧场景变化及全开发/回归叠图。0.2.15 仍使用 round_008，正式 runtime 未晋升。
<!-- tuyun-round009:end -->

## 2026-10-17 最新：0.2.15 本地安装包已生成

- macOS Apple Silicon：`release-assets/local-single-20261017-v0215/0.2.15/macos/ScreenPDF_0.2.15_aarch64.dmg`，18,431,643 字节（约 18 MB）。
- 包内唯一权重为 round_008 罗瀛复杂花色背景候选 `screen_corner_single_23fbbb5f04da`；SHA256：`23fbbb5f04da9635499158fe8b11860271767b5ca8dd404caf803d863bf084eb`。保留旧项目读取、新格式保存和目录移动规则，正式 runtime 未晋升。
- 20 项打包相关测试、9 项模型/项目文件测试、55 项前端测试及打包脚本测试通过。挂载最终 DMG 后核对 0.2.15 版本、签名、唯一模型、SHA1/SHA256 和打包源文件，并用亮底图表、混排、蓝色纹理、整幅照片四类图片通过实际识别入口检查；训练报告中的残留误差仍存在。
- 本地 ad-hoc 签名，未 Apple 公证或外部发布。配置、模型信息、测试、构建信息和最终 DMG 验证日志随产物保存；DMG SHA256：`61605accaae4dcbf2fed792b26f8b5f92753f7ddc95f3a85d4ebd62e0f2b3fbd`。


## 2026-10-17 最新训练：罗瀛复杂花色背景 round_008 已完成

- 38 张明确人工角点，36 个内容组，固定 22 训练 / 8 开发 / 8 已查看回归。蓝色纹理、整幅照片、地图图表和照片文字混排作为独立复杂花色场景保留，不整场并入黑底集。原项目与照片摘要不变。
- 从 round_007 继续，旧 1443 条训练采样逐条保留，22 张新训练图固定各 6 倍，共 1575 条、1225 张独立训练图。两组各 4 轮全局、8 轮局部、每轮 50 次更新，共用 317 张开发集；不设小样本目标、不追加追分。
- 唯一合并开发指标选中 with_luoying（1172/1268），初始 1169/1268，同预算对照 1171/1268。新场景开发 28/32→30/32、已查看回归 29/32→30/32；City Walk 开发 19/24→21/24、回归 20/24→22/24；旧统一黑底开发 18/20→18/20、回归 17/20→17/20。旧 265 张开发 989/1060→985/1060，整幅照片回归 8/8→7/8；背景子集只作诊断，完整变化与残留错误见报告。
- 最新未晋升单文件候选 `screen_corner_single_23fbbb5f04da`：`training/runs/screen_corner_single/round_008/checkpoints/screen_corner_single.pt`。40 项相关测试、数据隔离和旧采样保留审计、独立单文件动态尺寸及实际程序四类背景识别检查通过；已打入本地 0.2.15 安装包，0.2.14 仍使用 round_007。
- 数据入口 `data/curated/scenes/20260929-urban-renewal-luoying_v1/`；[复杂花色背景训练报告](../../research/experiments/2026-10-17-luoying-complex-background-training.md)保留分背景同指标结果和完整开发/回归叠图。小样本同演讲结果不代表跨活动泛化。


## 2026-10-17 最新：0.2.14 本地安装包已生成

- macOS Apple Silicon：`release-assets/local-single-20261017-v0214/0.2.14/macos/ScreenPDF_0.2.14_aarch64.dmg`，18,429,379 字节（约 18 MB）。
- 包内唯一权重为 round_007 李阳 City Walk 混合场景候选 `screen_corner_single_db1dfebdcbd5`；SHA256：`db1dfebdcbd540295b3bfd657bae95cb0ed8ce016bca52336dc82a843af23bfa`。保留旧项目读取、新格式保存和目录移动规则，正式 runtime 未晋升。
- 11 项打包相关测试、9 项模型/项目文件测试、55 项前端测试及打包脚本测试通过。挂载最终 DMG 后核对 0.2.14 版本、签名、唯一模型、SHA1/SHA256 及打包源文件，并使用混排、亮底和暗底各一张图片验证实际识别入口；模型仍保留训练报告列出的失败例。
- 本地 ad-hoc 签名，未 Apple 公证或外部发布。配置、模型信息、测试、构建信息和最终 DMG 验证日志随产物保存；DMG SHA256：`2edb23c63d3b3acb6e51d9f017b92c4c5bfbe56460b7cb0063eea0085255a3ce`。


## 2026-10-17 最新训练：李阳 City Walk 混合场景 round_007 已完成

- 29 张明确人工角点，28 个内容组，固定 17 训练 / 6 开发验证 / 6 已查看回归。整场保留白底、黑底/深色照片、蓝色封面和混排，为独立混合场景；不并入旧黑底专用集、不继承蓝侧屏标签。各拆分均含亮暗页面，原项目与源图摘要不变。
- 从 round_006 继续，旧 1341 条训练采样逐条保留；新增 17 张训练图固定 6 倍采样，共 1443 条、1203 张独立训练图。两组各 4 轮全局、8 轮局部、每轮 46 次更新，共用 309 张开发集；不设小样本目标、不追加追分。
- 唯一合并开发指标选中 with_citywalk（1141/1236），初始为 1136/1236，同预算对照为 1132/1236。新场景开发 16/24→19/24、回归 19/24→20/24；旧统一黑底开发 18/20→18/20、回归 17/20→17/20；董志国开发 16/24→18/24、回归 14/20→15/20。全部按同一个角点命中率报告，背景子集不参与选优。新场景及亮暗分组均与同预算对照同分，不能全归因于加数据；旧 XR 开发少 1 点、追光回归少 1 点，已记录。
- 最新未晋升单文件候选 `screen_corner_single_db1dfebdcbd5`：`training/runs/screen_corner_single/round_007/checkpoints/screen_corner_single.pt`。40 项相关测试、训练/评估隔离、独立单文件动态尺寸与实际程序亮/暗/混排识别检查通过。已打入下述 0.2.14 安装包，0.2.13 仍使用 round_006。
- 详细分背景结果、旧场景变化、前后叠图和失败例：[混合场景训练报告](../../research/experiments/2026-10-17-citywalk-mixed-scene-training.md)。数据入口 `data/curated/scenes/20260929-citywalk-liyang_v1/`。


## 2026-10-17 最新：0.2.13 本地安装包已生成

- macOS Apple Silicon：`release-assets/local-single-20261017-v0213/0.2.13/macos/ScreenPDF_0.2.13_aarch64.dmg`，18,430,474 字节（约 18 MB）。
- 包内唯一权重为 round_006 董志国低对比度场景候选 `screen_corner_single_39af695be2f4`；SHA256 为 `39af695be2f44537ad05152a9902d88e5583a51f55ad2b0cc67284813030c9dc`。保留旧项目兼容读取、新格式保存及目录移动/改名规则，正式 runtime 未晋升。
- 20 项打包相关测试、9 项模型/项目文件测试、55 项前端测试及打包脚本测试通过。挂载最终 DMG 后核对版本、签名、唯一权重、模型摘要和打包源文件，并使用董志国场景图片通过实际识别调用检查；模型仍保留训练报告列出的失败例。
- 本地 ad-hoc 签名，未做 Apple 公证或外部发布。配置、模型信息、构建信息和验证日志随产物保存；DMG SHA256：`91c0415ada7752bf296e931d18e1e84e2bdced4f60136515630c290e0eb57dab`。

## 2026-10-17 最新：元宇宙资料父目录已迁移

- 原资料目录现为 `/Users/gcssloop/WorkSpace/AIGC/展会/2026/202609 上海-元宇宙产业嘉年华`。5 个 raw 入口、50 个场景原图链接，以及数据集、训练快照、预测报告和构建数据集脚本中的相关路径已同步调整。
- 只更改路径及相应 JSONL 字节摘要；人工标签、用途拆分、数值评估和模型字节不变，原始项目 JSON 与图片也未改写。历史执行日志、签名安装包及原始打包验证保留当时内容，查阅时使用新旧根路径映射。
- 完整迁移清单、前后摘要与修改前备份见 [迁移审计](../../research/experiments/path-migrations/20261017-shanghai-metaverse/README.md)。
- 迁移验证通过：检查 3483 份数据/训练 JSON 文件，相关路径无缺失或旧根路径残留；5 个源项目的 153 页图片可读取，320 份源文件字节不变。40 项相关测试通过，新路径下的实际引擎识别与迁移前点位一致。

## 2026-10-17 最新训练：董志国低对比度场景 round_006 已完成

- 26 张人工标注按 23 个内容组固定为 15 训练 / 6 开发 / 5 已查看回归。深色实体边框、反射与前景人物作为独立物理场景，未继承蓝侧屏标签；黑底总索引 `data/curated/scenes/black-background-index.json` 同时保留旧 24 张场景。
- 从 round_005 单体权重继续，旧 1251 条训练采样完整回放，新图固定 6 倍采样。两组各 4 轮全局、8 轮局部、每轮 42 次更新，303 张开发集一致；不设小样本达标线、不追加追分。
- 唯一合并开发指标选中 with_transport（1120/1212），高于初始 1108/1212 和同预算对照 1106/1212。新场景开发 9/24→16/24、已查看回归 6/20→14/20；部分塌陷/交叉框恢复外轮廓，仍有严重错位、内缩和遮挡失败。旧蓝侧屏黑底命中数保持，追光开发相对初始少 1 点。
- 最新未晋升单文件候选 `screen_corner_single_39af695be2f4`：`training/runs/screen_corner_single/round_006/checkpoints/screen_corner_single.pt`。40 项相关测试、单文件独立加载及实际引擎调用通过。已打入上述 0.2.13 本地安装包；0.2.12 仍用 round_005。
- 完整数据、对照与成功/失败叠图：[低对比度场景训练报告](../../research/experiments/2026-10-17-transport-low-contrast-scene-training.md)。

## 2026-10-17 最新：0.2.12 本地安装包已生成

- macOS Apple Silicon：`release-assets/local-single-20261017-v0212/0.2.12/macos/ScreenPDF_0.2.12_aarch64.dmg`，约 18 MB。
- 包内唯一权重为 round_005 统一黑底数据集候选 `screen_corner_single_cf3cb63c6abe`；保留旧项目兼容读取、新格式保存及目录移动/改名规则。正式 runtime 未晋升。
- 25 项打包相关测试及打包脚本测试通过；挂载最终 DMG 后核对版本、签名、唯一权重及摘要、引擎源码，并通过黑底图片识别调用检查。模型残留失败仍见下方 round_005 报告。
- 本地 ad-hoc 签名，未做 Apple 公证或外部发布；配置、模型信息、测试和验证日志随产物保存。

## 2026-10-17 最新：统一黑底数据集与 round_005 已完成

- 按用户要求，XR 黑底 14 张、李琳琳黑底 5 张、新增黑底 5 张合为一份场景数据集：`data/curated/scenes/20260929-black-background-unified_v1/`。24 张、18 内容组，固定 14 训练 / 5 开发验证 / 5 已查看回归；图片入口只链接原图，原项目不改写。
- 历史用途不重排；新 5 张为 3/1/1。XR 的 3 张亮底/大幅照片及李琳琳 2 张亮底留在普通数据，不计入黑底子集。没有按模型难易删图。
- 黑底训练图改为统一 6 倍采样，替换此前按演讲 4 倍/8 倍的条目，不重复叠加。对照也统一旧黑底采样，仅完整组加入新三张训练图；同 round_004 初始化、种子与损失，各 4 轮全局、8 轮局部、每轮 40 次更新。
- 297 张合并开发唯一点指标选中 with_black（1099/1188）；黑底开发 13/20→18/20，已查看回归 15/20→17/20。少文字页面与部分变形框改善，重遮挡和部分左下角仍失败。商场回归相对初始少 3 点，已记录；不设小样本达标线或追加追分实验。
- 最新未晋升单文件候选 `screen_corner_single_cf3cb63c6abe`：`training/runs/screen_corner_single/round_005/checkpoints/screen_corner_single.pt`。39 项测试、单文件独立加载及真实程序调用通过。已打入上述 0.2.12 本地安装包，0.2.11 仍使用 round_004。
- 完整数据筛选、对照与叠图：[统一黑底数据集重训练报告](../../research/experiments/2026-10-17-unified-black-dataset-training.md)。

## 2026-10-17 最新：0.2.11 本地安装包已生成

- macOS Apple Silicon：`release-assets/local-single-20261017-v0211/0.2.11/macos/ScreenPDF_0.2.11_aarch64.dmg`，约 18 MB。
- 包内唯一权重为 round_004 黑底场景候选 `screen_corner_single_5b2c52ff8450`，包含上一组 XR 与本次李琳琳资料的训练结果。保留旧项目兼容读取、新格式保存和目录移动/改名规则；正式 runtime 未晋升。
- 25 项打包相关测试及打包脚本测试通过；挂载最终 DMG 后，版本、签名、唯一权重及摘要、引擎源码核对通过，李琳琳黑底图片识别调用通过。模型的残留失败仍见本轮报告。
- 本地 ad-hoc 签名，未做 Apple 公证或外部发布；配置、模型信息与验证日志随产物保存。

## 2026-10-17 最新：李琳琳黑底资料已接入 round_004

- 7 张明确人工标注（5 张偏黑、2 张偏亮），7 个内容组，固定 3 训练 / 2 开发验证 / 2 已查看回归。原 JSON 和 7 张图片未改写。
- 遵循用户“不盲目追求数值”：从 round_003 出发，保留上一组 XR 和旧资料，只做一次固定预算对照；新图固定 8 倍采样，两组各 4 轮全局、8 轮局部、每轮 39 次更新。不设小样本目标，也未根据结果追加训练。
- 新场景开发单点命中初始/对照为 3/8，新增资料为 4/8；回归为 5/8→6/8。XR 开发为 9/16→10/16，回归相对初始 8/12→10/12，但与对照同为 10/12，不能全归因于新资料。
- 唯一合并开发分数选中 with_immersive（1082/1184），比同预算对照多命中 2 点。部分边界改善，少文字黑底页仍会漏左侧，训练图也有角点落进蓝侧屏的失败。追光/邢杰开发的少量回退同样记录。
- 最新未晋升单文件候选：`training/runs/screen_corner_single/round_004/checkpoints/screen_corner_single.pt`，标识 `screen_corner_single_5b2c52ff8450`。31 项相关测试、独立单文件和真实引擎调用通过，已打入上述 0.2.11 本地安装包；0.2.10 安装包仍使用 round_003。
- 详情与前后叠图：[李琳琳黑底场景训练报告](../../research/experiments/2026-10-17-immersive-black-scene-training.md)。

## 2026-10-17 最新：0.2.10 本地安装包已生成

- macOS Apple Silicon：`release-assets/local-single-20261017-v0210/0.2.10/macos/ScreenPDF_0.2.10_aarch64.dmg`，约 18 MB。
- 包内唯一权重更新为 round_003 的 XR 场景候选 `screen_corner_single_8f8fa2f691c5`；保留旧格式读取、新格式保存、相对路径及按当前目录名导出的功能。正式 runtime 晋升状态未改动。
- 25 项引擎/发布/打包相关测试及打包脚本测试通过；挂载最终 DMG 后，核对版本、签名、权重摘要、唯一模型文件及引擎源码，并用 XR 图片实际识别通过。
- 本地 ad-hoc 签名，未做 Apple 公证或外部发布；配置、模型信息、测试和安装包验证日志随产物保存。模型的遮挡和点序限制仍见下方 round_003 报告。

## 2026-10-17 最新：XR 黑底蓝侧屏小样本已接入 round_003

- 周群华 XR 文旅演讲 17 张人工标注，11 个内容相似组，固定 10 训练 / 4 开发验证 / 3 已查看回归；原项目与图片未改写。
- 按用户要求先接触场景，不设该小样本达标分数。保留旧 1155 张训练图，新 10 张以固定 4 倍采样加入，两组均从 round_002 继续，固定 4 轮全局、8 轮局部、每轮 38 次更新，不根据结果追加实验。
- 新场景开发单点命中率：初始及同预算对照均 37.5%（6/16），加入 XR 为 56.25%（9/16）；已查看回归为 33.33%（4/12）→66.67%（8/12）。合并开发唯一分数 90.99%（1070/1176），按既定规则选中 with_xr。
- 叠图显示部分黑底主画面边界改善，但严重遮挡、内容干扰和点序问题仍在；尚不能认为稳定识别。商场、追光回归相对初始分别少 3 点、1 点，已如实记录，无隐藏选优门槛。
- 单文件候选 `screen_corner_single_8f8fa2f691c5`：`training/runs/screen_corner_single/round_003/checkpoints/screen_corner_single.pt`。29 项相关测试及单文件独立加载、真实引擎调用通过；已注册未晋升，已打入上述 0.2.10 本地安装包；0.2.9 安装包仍使用 round_002。
- 完整对照与成功/失败叠图：[XR 小样本训练报告](../../research/experiments/2026-10-17-xr-small-scene-training.md)。

## 2026-10-17 最新：0.2.9 本地安装包已生成

- macOS Apple Silicon 安装包：`release-assets/local-single-20261017-v029/0.2.9/macos/ScreenPDF_0.2.9_aarch64.dmg`。
- 包含根目录唯一 JSON、相对路径及按当前目录名导出的规则；打开旧格式不改写原文件，保存或导出时统一写 v2，保留人工角点与页面状态。
- 包内唯一模型更新为加入元宇宙数据训练的 `screen_corner_single_3e8f4df71c2a`（round_002）；正式 runtime 晋升状态未变。
- 36 项桌面壳、55 项前端、25 项引擎与发布测试及打包脚本测试通过；已挂载最终 DMG，核对 0.2.9 版本、签名、单文件模型摘要和路径解析模块，并用元宇宙图片完成实际识别。构建时另用追光图片验证识别入口。
- 采用本地 ad-hoc 签名，未做 Apple 公证或外部发布；构建信息与验证日志随产物保存。

## 2026-10-17：项目 JSON 存储规范更新

- 新保存格式 `version=2`：唯一目标为项目根目录的 `screen-pdf-project.json`；导出前也只保存根文件。
- 磁盘路径全部相对根目录，缓存路径清空；加载时恢复运行时路径，旧绝对路径兼容重定位并保留人工角点。
- 项目名从当前目录名派生，不再保存固定 `name`；导出 PDF 与派生目录使用当前名称，保留中文、空格和点号。外部移动/改名后重新打开即可使用新名称。
- 规范与 schema：[项目文件规范](../architecture/project-file-format.md)。训练读取也已接入统一解析；现有训练快照和原项目未批量改写。本项已包含在上述 0.2.9 本地安装包中。


## 2026-10-17 最新：元宇宙数据已接入并完成 round_002

- 新项目“2026-09-29 元宇宙2.0四大变革力量-邢杰”：98 张明确人工四角标注，统一点序后按 89 个内容相似组固定为 58 训练 / 20 开发验证 / 20 留出；v2 为正式拆分，初步 v1 没有参与训练。
- 同初始单体模型、种子与更新次数做数据加入对照：原 1097 张 vs 混合 1155 张；两组各 8 轮全局头、24 轮局部头，每轮 37 次更新。
- 元宇宙开发验证单点命中：上一轮 78.75%、同预算对照 82.5%、新增数据 92.5%；留出分别 83.75%、91.25%、96.25%。新增数据相对同预算对照的留出收益为 +5 个百分点。
- 唯一合并开发分数 90.86%（1054/1160），选中加入元宇宙数据的单体候选；先冻结选择，后评估新留出。旧验证相对对照少命中 2 点，仍高于上一轮，详见完整对照。
- 文件：`training/runs/screen_corner_single/round_002/checkpoints/screen_corner_single.pt`，模型标识 `screen_corner_single_3e8f4df71c2a`，约 4.46 MiB；95 项相关测试及单文件独立加载、真实检测入口均通过。
- 原项目、正式 runtime 和上次安装包未改动。新候选已注册，未晋升或重新打包。新留出已经打开，后续属于已查看回归；本轮是同演讲场景验证及单个随机种子实验。

完整记录：[元宇宙数据训练验证报告](../../research/experiments/2026-10-17-metaverse-dataset-training-validation.md)。


## 2026-10-17：本地桌面包已重新构建

- macOS Apple Silicon，程序版本保持 `0.2.8`，安装包：`release-assets/local-single-20261017/0.2.8/macos/ScreenPDF_0.2.8_aarch64.dmg`。
- 此本地包默认使用单体候选 `screen_corner_single_597e05cdea27`；包内只有一个 `.pt`。打包配置通过 `SCREEN_PDF_TAURI_CONFIG` 覆盖资源清单，仓库正式模型与 registry 晋升状态没有改动，也未执行外部发布。
- 52 项前端、20 项引擎/打包测试及打包脚本测试通过；挂载最终 DMG 后，完成签名核对、模型摘要校验和追光图片实际识别。
- 安装包采用本地 ad-hoc 签名，未进行 Apple 公证，也未生成自动更新签名产物。本次使用现有本机运行环境。构建记录与校验结果放在产物目录。


## 2026-10-17 最新：单角点指标、单文件模型已完成重训

- 用户将目标缩减为唯一 `point_le_0_01_ratio`。按原图像素、人工框包围盒对角线、误差 `<=1%` 统计；旧平均误差/四点全中率门槛取消，平局不看其他指标。详见 [评估契约](evaluation-contract.md)。
- 新主线 `screen_corner_single/round_001`：1097 张训练（含追光 17 张）、270 张开发验证；全局头 8 轮、两种局部损失条件各 24 轮，种子 7。加入单点命中损失后，唯一开发分数由对照 83.89% 提升至 87.87%（949/1080）。
- 最终单文件为 `training/runs/screen_corner_single/round_001/checkpoints/screen_corner_single.pt`，4,678,214 字节。一个计算图内包含全局特征、粗定位、局部原图/特征采样与角点精修，不依赖旧权重或 torchvision。
- 按场景单点命中率：旧验证 87.83%、追光验证 90%、商场回归 94.83%、追光回归 80%（最新 v2 人工标注）。v1 的追光回归是 75%；原项目只修订了一个回归角点，训练/验证标签不变，模型未重选。v1 与 v2 结果均保留。
- 单文件 CPU 抽样平均 39.62 ms、P95 55.52 ms；267 项相关测试通过，另完成 24 项运行时聚焦测试与独立单文件加载、真实引擎检测检查。
- 已接入 `SCREEN_PDF_SINGLE_MODEL` 试用入口；候选已保存及注册，未执行正式晋升、桌面构建或发布。正式旧 runtime 仍是 r130 + c22 + r69probe。
- 两个追光评估子集均仅 5 张，回归集已经查看，不能作为跨活动独立盲测。

详情：[单指标与单体模型结果](../../research/experiments/2026-10-17-single-point-single-model-results.md)。下述同日旧多指标实验与更早阶段为历史记录，当前规则以本节及评估契约为准。


## 2026-10-17：统一评估与新场景实验

- 评估契约更新为 `screen_quad_pixel_v2`，修复训练验证归一化坐标与运行时像素坐标不一致的问题；新增同尺度四角全中率与像素误差。见 [evaluation-contract.md](evaluation-contract.md)。
- 接入 `202610 追光卅载铸根基，AI赋能启新篇`：27 张人工修正图片，固定 17 训练 / 5 验证 / 5 封存留出，训练与各验证集合无路径交集。
- 完成 24 次训练、3 个随机种子，以及解码、各阶段输出、旧域回归和独立时延对照。冻结预训练 MobileNetV3-Small + FPN + 视觉边界监督是本轮最有价值的结构方向。
- 封存 5 张的最终逐角点命中：正式 r130 `35%`，候选 `70%`；四角全中仍为 `20%`。候选平均识别 `212.44 ms`、P95 `224.47 ms`（30 张抽样，含完整识别，排除磁盘读取、OCR 和 PDF）。
- 旧域验证逐角点命中从 r130 `84.91%` 提升到 `86.51%`，但商场相对 r142 有回退，且新留出平均误差仍高于目标。因此候选保留，不晋升；正式 runtime 仍为 r130 + c22 + r69probe。
- 候选仅由实验脚本加载，尚未适配桌面模型加载和打包。下一阶段优先研究后续阶段覆盖正确 coarse、候选误选屏幕内部内容及坏角点；已打开的 5 张留出不能再当下一轮盲测集。

结果与复现入口：[2026-10-17 实验报告](../../research/experiments/2026-10-17-unified-evaluation-ablation-results.md)。下文保留历史阶段记录，解释旧指标时应先看新契约。

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

`2026-04-01` 当前固定 runtime 发布版本：

- `deep_screen_r1_2026_04_01_r130`
- `model_release_id = model-20260401-060636-035e5e08`
- app 版本：`0.2.7`
- runtime 组合：`r130 + c22 + r69probe`

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

`2026-04-01` failure-layer 诊断后的当前共识补充：

- 当前 `r96 + c22 + r69probe` 链路在金溢、杭州、OpenAtom 上的主要瓶颈已不是继续扫 selector 参数。
- OpenAtom 仍有明显 `runtime_candidate_recoverable` 空间，说明候选常常已经出现过，但这条线的 oracle 上限也已接近天花板。
- 杭州特别是 `2026-03-05 中控交安` 以 `hard_both_fail` 为主，说明 coarse 对这类几何与边界分布没有学会，不能再靠 selector 或场景特例推进。
- 因此下一主线改为：把 `failure-layer` 诊断写回 coarse 训练 split，并对 `runtime_candidate_recoverable / opencv_recoverable / hard_both_fail` 做显式重采样与 teacher-source 控制。
- 当前阶段不再扩展新的场景特例路由，也不恢复记忆依赖。

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
