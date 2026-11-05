# Training

`training` 只存放训练资产，不存放桌面端程序。

## 2026-10 单指标、单文件主线

当前唯一准确率指标和选优规则见 [评估契约](../docs/status/evaluation-contract.md)。新的训练入口：`program/engine/training/train_screen_corner_single.py`；单体模型：`program/engine/screen_corner_single.py`。

每轮使用 `runs/screen_corner_single/round_xxx/`，包含数据快照、配置、全局和局部训练历史、单变量损失对照、冻结选择、评估、源码快照及单个 TorchScript 成品。冻结全局特征与局部训练缓存仅服务训练，推理只需要 `checkpoints/screen_corner_single.pt`。

全局定位、图内取样和局部精修均属于同一个模块的前向计算；导出的单个文件不引用其他模型。正式 runtime 与候选分离，候选可通过 `SCREEN_PDF_SINGLE_MODEL` 测试。

## 全量会议集合基准（2026-10-17）

数据组织和日常查看统一使用 [会议 / 场次集合](../data/curated/collections/conference_v1/README.md)，比较 0.2.17 包内 round_009 与已冻结 round_010 control 的 [同集全量基准](../data/curated/collections/conference_v1/BASELINE.md)。这次整理不训练、不调整权重、不根据回归重新选择候选；该基准不改变当时的正式 runtime 或已生成安装包，后续 round_011 晋升见 [当前状态](../docs/status/current-status.md)。

机器用途继承 round_010 定向实验快照；选中 control 和 round_009 都未训练刘奇新增的 21 张训练计划页，不能把“train”成员关系当作各模型的实际暴露证明。已查看回归与快照外资料都不宣称盲测；冲突人工框先隔离，人工修正后生成新数据 / 报告修订，不能覆盖已冻结结果。见 [使用与复现](../docs/status/conference-collections.md)。

## 历史：2026-10 统一评估与单变量实验

入口：`program/engine/training/screen_corner_ablation.py`。数据、配置和种子固定，`runs/screen_corner_unified_20261017` 保存 8 轮初筛，`runs/screen_corner_unified_20261017_longhead` 保存 40 轮结构对照。每组 3 个随机种子，共 24 次训练。

统一评估口径见 [评估契约](../docs/status/evaluation-contract.md)，完整结果见 [实验报告](../research/experiments/2026-10-17-unified-evaluation-ablation-results.md)。当前实验候选没有晋升为 runtime，也不属于旧 `deep_screen_v1/round_*` 蒸馏主线。

## 目录约定

- `configs/`
  - 训练配置、实验配置、数据拆分配置。
  - 建议按模型族区分，例如 `global_corner/`、`local_corner/`、`ocr_overlay/`。
- `runs/`
  - 每次训练的工作目录。
  - 包含日志、临时评测结果、可视化样例、命令行快照。
- `checkpoints/`
  - 训练过程中产出的权重。
  - 默认不直接给桌面程序使用。
- `reports/`
  - 训练总结、对比报告、回归结论。
- `registry/`
  - 记录每一代数据集、模型版本、发布名、教师模型、关键指标。
  - 数据集记录继续沿用 `dataset_slug.json`。
  - 统一模型发布记录建议使用 `deep_screen_v{major}.json`。
  - 字段约定见 `registry/model-release-template.json` 和 `docs/status/model-release-convention.md`。
- `promoted/`
  - 经过验证后，准备晋升为运行时或候选运行时的模型与说明。

## 历史统一蒸馏主线

原统一蒸馏主线记录为：

- 对外版本：`deep_screen_v1`
- 内部版本：`ds_corner_unified_distill_v1`
- teacher：`r3`、`v28`

对应的演进根目录：

- `runs/deep_screen_v1/`

每一轮演进都必须使用独立 round 目录，例如：

- `runs/deep_screen_v1/round_001/`
- `runs/deep_screen_v1/round_002/`

round 目录的详细约定见：

- [docs/status/distillation-run-convention.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/distillation-run-convention.md)

## 与 `models/runtime` 的关系

- `training/checkpoints` 是训练产物。
- `models/runtime` 是程序实际加载的已晋升模型。
- 任何模型进入 `models/runtime` 前，都应该在 `training/reports` 和 `training/registry` 里留下依据。

## 最小发布流程

1. 在 `training/runs` 产出训练结果
2. 在 `training/reports` 写清楚训练集、验证集、指标和结论
3. 在 `training/registry` 记录模型版本、发布名、教师模型和来源
4. 只有通过门槛的模型，才复制到 `models/runtime`

## 元宇宙 round_002

`runs/screen_corner_single/round_002` 保存新增 58 张元宇宙训练图的同预算对照，两组均从 round_001 的完整单体权重继续训练，开发集固定为 290 张，统一每轮 37 次更新。复现入口：`program/engine/training/metaverse_round.py`；导入入口：`program/engine/training/import_screen_project.py`。新的训练选项支持 `--data-source`、`--initial-model`、`--steps-per-epoch`，便于固定对照。

候选 `screen_corner_single_3e8f4df71c2a` 仍为一个 TorchScript 文件。逐轮记录在 `registry/screen_corner_single_v1_round002.json`，旧 registry 快照保存在本轮 artifacts 中；family registry 当前指向 round_010 control 候选。新增数据的 20 张留出已在最终选择冻结后评估，后续按已查看回归使用。

## XR 小样本 round_003

`runs/screen_corner_single/round_003` 保存从 round_002 出发的固定预算对照。17 张黑底蓝侧屏场景按内容组固定拆分为 10 训练 / 4 开发验证 / 3 已查看回归。旧 1155 张训练资料继续回放，新增训练图固定 4 倍采样，不能当成新增 40 张独立图片。两组各 4 轮全局、8 轮局部、每轮 38 次更新；唯一变量为加入新场景数据。

用户要求先接触场景，不设该小样本分数目标，不根据结果追加训练。唯一开发点命中率选中单文件候选 `screen_corner_single_8f8fa2f691c5`；逐轮 registry 与之前各轮快照均保留。定性观察有改善和残留失败，部分旧回归下降也已记录。入口：`program/engine/training/small_scene_round.py`；详见 [完整报告及叠图](../research/experiments/2026-10-17-xr-small-scene-training.md)。该候选已打入本地 0.2.10 安装包，正式 runtime 未晋升。

## 李琳琳黑底补充 round_004

`runs/screen_corner_single/round_004` 保存 7 张李琳琳资料的固定对照：3 训练 / 2 开发验证 / 2 已查看回归。继续回放上一组 XR 的 10 张训练图（4 倍采样）和旧 1155 张训练资料，本次 3 张图固定 8 倍采样。两组均从 round_003 继续，固定 4 轮全局、8 轮局部、每轮 39 次更新，开发集 296 张。

不设小样本达标线，不根据结果追加实验。候选 `screen_corner_single_5b2c52ff8450` 仍为单文件；逐轮记录为 `registry/screen_corner_single_v1_round004.json`。有部分边界改善，也有少文字黑底漏边和蓝侧屏误定位，详见 [报告与叠图](../research/experiments/2026-10-17-immersive-black-scene-training.md)。本轮候选已打入本地 0.2.11，0.2.10 安装包仍使用 round_003。

训练入口 `program/engine/training/small_scene_round.py` 已支持 `--slug`、`--base-run`、`--base-data`、`--prefix`、`--replay` 和 `--contact-sheet`；本轮配置和源码快照保存在 run 中。相同开发分数保留初始/更早条件，回归不能改变选择。

## 统一黑底 round_005（0.2.12 包内候选）

统一黑底数据为 `20260929-black-background-unified_v1`：三处原项目中的 24 张黑底图片，14 训练 / 5 开发验证 / 5 已查看回归，历史用途不重排。训练入口为 `program/engine/training/unified_black_round.py`，建集入口为 `program/engine/training/build_black_scene_dataset.py`。

从 round_004 出发，先移除旧按演讲采样的黑底条目，再统一每图 6 倍采样，其他资料继续回放。两组均执行 4 轮全局、8 轮局部、每轮 40 次更新；对照使用旧 11 张黑底训练图，完整组使用全 14 张。297 张合并开发只按单点指标选中 `screen_corner_single_cf3cb63c6abe`，逐轮 registry 保留本轮未晋升候选。未追加追分；该候选已打入本地 0.2.12 安装包。结果、失败例及旧场景回退见 [统一黑底重训练报告](../research/experiments/2026-10-17-unified-black-dataset-training.md)。

## 深色实体边框 round_006（0.2.13 包内候选）

董志国 26 张人工标注独立为另一种黑底物理场景，15 训练 / 6 开发 / 5 已查看回归。`runs/screen_corner_single/round_006` 从 round_005 出发，完整保留旧 1251 条训练采样，新图固定 6 倍采样后共 1341 条；两组各 4 轮全局、8 轮局部，每轮 42 次更新，303 张开发集一致。只比较是否加入本组资料，不追加追分。

合并开发唯一单点分数选中 `screen_corner_single_39af695be2f4`（1120/1212）。新场景开发 9/24→16/24、已查看回归 6/20→14/20，旧蓝侧屏黑底命中数不变；追光开发少 1 点。40 项相关测试、独立单文件及实际引擎调用通过，保留错位和遮挡失败例。round_006 逐轮记录 `registry/screen_corner_single_v1_round006.json`；已打入本地 0.2.13 安装包但未正式晋升，0.2.12 仍使用 round_005。详见 [报告与叠图](../research/experiments/2026-10-17-transport-low-contrast-scene-training.md)。

2026-10-17 元宇宙源资料父目录已增加“上海-”。既有训练快照、预测报告和源码中相关路径同步迁移，JSONL 字节摘要已刷新；原始日志、安装包验证与模型字节保留。完整映射与修改前备份见 [迁移审计](../research/experiments/path-migrations/20261017-shanghai-metaverse/README.md)。

## 李阳 City Walk 黑白混合场景 round_007（0.2.14 包内候选）

29 张人工角点按 28 内容组固定为 17 训练 / 6 开发 / 6 已查看回归，各部分都有亮暗页面，作为独立混合场景保留全部外观。`runs/screen_corner_single/round_007` 从 round_006 继续，旧 1341 条训练采样逐条保留，新图固定 6 倍后共 1443 条、1203 张独立图片。两组各 4 轮全局、8 轮局部、每轮 46 次更新，共用 309 张开发集；不根据结果追加训练。

唯一合并开发指标选中 with_citywalk（1141/1236），候选 `screen_corner_single_db1dfebdcbd5`；新场景开发 16/24→19/24、已查看回归 19/24→20/24。新场景及亮暗分组与同预算对照同分，不能全归因于新增资料；旧 XR 开发、追光回归各少 1 点。round_007 逐轮记录 `registry/screen_corner_single_v1_round007.json`，旧 registry 快照保留。40 项相关测试、数据隔离和实际单文件引擎调用通过；正式 runtime 未晋升，已打入本地 0.2.14 安装包，0.2.13 仍用 round_006。完整结果与失败例见 [混合场景训练报告](../research/experiments/2026-10-17-citywalk-mixed-scene-training.md)。

## 罗瀛复杂花色背景 round_008

38 张人工标注按 36 内容组固定为 22 训练 / 8 开发 / 8 已查看回归；同一物理屏幕包含蓝色纹理、照片、地图图表及复杂混排，独立于黑底专用集。`runs/screen_corner_single/round_008` 从 round_007 继续，旧 1443 条训练采样完整保留，新图固定 6 倍后共 1575 条、1225 张独立训练图。两组各 4 轮全局、8 轮局部、每轮 50 次更新，共用 317 张开发集，不追加追分。

唯一合并开发指标选中 with_luoying（1172/1268），初始 1169/1268、同预算对照 1171/1268；新场景开发 28/32→30/32、已查看回归 29/32→30/32。候选 `screen_corner_single_23fbbb5f04da` 未晋升，family registry 已由 round_010 更新，逐轮记录 `registry/screen_corner_single_v1_round008.json` 保留；前一轮 registry 有快照。40 项相关测试、隔离/回放审计与实际单文件入口验证通过；已打入本地 0.2.15 安装包，0.2.14 仍使用 round_007。完整分背景结果、旧场景回退和失败例见 [报告](../research/experiments/2026-10-17-luoying-complex-background-training.md)。

<!-- tuyun-round009:start -->
## 涂耘局部屏幕 round_009（2026-10-17）

涂耘隧道照明资料 26 张已按独立超宽屏局部 PPT 场景 `202610-tunnel-lighting-tuyun_v1` 接入：15 训练 / 6 开发 / 5 已查看回归。目标仅为右侧 PPT，排除左侧固定信息区；保留 LED 拍摄条纹、右上角倒计时/红色超时叠层与左下角人头遮挡，沿用原人工四角，不沿人头或计时框收边。 round_009 固定预算对照完成，合并开发 initial=1180/1292、control=1179/1292、with_tuyun=1194/1292，选中 with_tuyun；未晋升单文件候选 `6fc6378701c4`。新场景开发 8/24→13/24、已查看回归 6/20→11/20。 左下角仍失败（开发 0/6、回归 0/5），蓝色结束页 0/4；右上角仅部分改善，不能称遮挡/计时干扰已解决。

`runs/screen_corner_single/round_009` 从 round_008 继续，旧 1575 条训练采样完整保留，新 15 张训练图各 6 倍，共 1665 条、1240 张独立训练图；两组同预算 4 轮全局、8 轮局部、每轮 53 次更新，同一 323 张开发集只按 `point_le_0_01_ratio` 选优，冻结后再回归，不追加追分。逐轮 registry 为 `registry/screen_corner_single_v1_round009.json`，family registry 当前已由 round_010 更新，本轮快照保留；[完整报告](../research/experiments/2026-10-17-tuyun-partial-screen-training.md)。当前已打入本地 0.2.17 安装包，0.2.15 安装包仍为 round_008。
<!-- tuyun-round009:end -->


<!-- liuqi-round010:start -->
刘奇资料 35 张已接入（21 训练 / 7 开发 / 7 已查看回归），与涂耘组成 61 张超宽屏右侧 PPT 场景族，旧拆分保留。round_010 定向组两场训练页均 12 倍采样，固定预算对照完成；合并开发 initial=1206/1320、control=1212/1320、with_tunnel_focus=1206/1320，选中 control；单文件候选 `d9d6ad842126` 未晋升。同场景族开发 48.08%（25/52）→59.62%（31/52），已查看回归 60.42%（29/48）→85.42%（41/48）。定向组未胜过 control，选中候选未采用新增刘奇训练页的权重；该轮完成时正式 runtime 未变，0.2.17 安装包仍使用 round_009，之后由 round_011 晋升更新正式 runtime。详见 [训练报告](../research/experiments/2026-10-17-liuqi-tunnel-family-training.md)。
<!-- liuqi-round010:end -->

## 孙泉明黑底宽屏 PPT round_012（2026-10-18）

孙泉明资料 29 张按 17 训练 / 6 开发 / 6 已查看回归接入，与涂耘、刘奇同属杭州隧道照明右侧 PPT 场景族。物理屏幕以黑色为主，但保留页内白色图表和照片；训练输入保留 LED 条纹、右上角倒计时 / 超时叠层和底部观众遮挡。

`runs/screen_corner_single/round_012` 从 round_011 继续，control 保留 1665 条旧采样，完整条件追加 17 张训练图各 6 倍为 1767 条。固定 4 轮全局 + 8 轮局部、每轮 56 次更新；合并开发唯一指标选中 `screen_corner_single_a1f4bec7b047`（1231/1344），候选未晋升，正式 runtime 在该轮仍为 round_011。孙泉明开发 8/24→16/24，已查看回归 2/24→15/24；旧开发回退 7 个角点，回归只作诊断。

完整记录：[孙泉明训练报告](../research/experiments/2026-10-18-sunquanming-black-screen-training.md)。

## 李薇隧道照明局部屏幕 round_013（2026-10-18）

李薇资料 24 张按 14 训练 / 5 开发 / 5 已查看回归接入，与涂耘、刘奇、孙泉明同属杭州隧道照明右侧 PPT 场景族。目标仍排除左侧蓝色固定信息区；保留黑色屏幕背景、页内亮色内容、LED 条纹、倒计时 / 超时叠层和底部观众遮挡。

`runs/screen_corner_single/round_013` 从 round_011 继续，control 保留 1665 条旧采样，完整条件追加 14 张训练图各 6 倍为 1749 条。固定 4 轮全局 + 8 轮局部、每轮 55 次更新；合并开发唯一指标选中 `screen_corner_single_5ba8b4a37a5d`（1236/1340），候选已晋升为正式 runtime round_013。李薇开发 15/20→16/20，已查看回归 14/20→17/20；旧开发相对 initial 上升 1 点，回归只作诊断。

完整记录：[李薇训练报告](../research/experiments/2026-10-18-liwei-partial-screen-training.md)。

## 李华斌隧道照明局部屏幕 round_014（2026-10-18）

李华斌资料 17 张按 9 训练 / 4 开发 / 4 已查看回归接入，与前四场同属杭州隧道照明右侧 PPT 场景族。目标仍排除左侧蓝色固定信息区；保留蓝色和深色屏幕背景、LED 条纹、右上角倒计时叠层和底部观众遮挡。

`runs/screen_corner_single/round_014` 从 round_013 继续，control 保留 1749 条旧采样，完整条件追加 9 张训练图各 6 倍为 1803 条。固定 4 轮全局 + 8 轮局部、每轮 57 次更新；合并开发唯一指标选中候选 `screen_corner_single_2343521e5471`（1241/1356）。李华斌开发 4/16→5/16，已查看回归 4/16→9/16；旧验证回退 3 点，候选未晋升，正式 runtime 继续为 round_013。

完整记录：[李华斌训练报告](../research/experiments/2026-10-18-lihuabin-partial-screen-training.md)。

## 刘松荣隧道照明局部屏幕 round_015（2026-10-18）

刘松荣资料 15 张按 9 训练 / 3 开发 / 3 已查看回归接入，与前五场同属杭州隧道照明右侧 PPT 场景族。目标仍排除左侧蓝色固定信息区；新增白底、蓝底、密集表格图表和地图页面，同时保留 LED 条纹、倒计时叠层和底部观众遮挡。

`runs/screen_corner_single/round_015` 沿用 round_014 的同场训练快照，control 保留 1803 条旧采样，完整条件追加 9 张训练图各 6 倍为 1857 条。固定 4 轮全局 + 8 轮局部、每轮 59 次更新；合并开发选中 control（1251/1368），`with_liusongrong` 同分。刘松荣开发 6/12→8/12，已查看回归 8/12→9/12；旧验证仍有回退，候选未晋升，正式 runtime 继续为 round_013。

完整记录：[刘松荣训练报告](../research/experiments/2026-10-18-liusongrong-partial-screen-training.md)。

## 林燕丹隧道照明局部屏幕 round_016（2026-10-18）

林燕丹资料 25 张按 15 训练 / 5 开发 / 5 已查看回归接入，与前六场同属杭州隧道照明右侧 PPT 场景族。目标仍排除左侧蓝色固定信息区；保留动态光场页面、LED 条纹、叠加内容和底部观众遮挡。

`runs/screen_corner_single/round_016` 沿用 round_015 的同场训练快照，control 保留 1857 条旧采样，完整条件追加 15 张训练图各 6 倍为 1947 条。固定 4 轮全局 + 8 轮局部、每轮 61 次更新；合并开发选中 `with_linyandan`（1274/1388）。林燕丹开发 15/20→17/20→17/20，已查看回归 14/20→14/20→14/20；杭州场景族开发 40/52→39/52→43/52，回归 42/48→42/48→45/48。

冻结全量基准中，林燕丹候选相对正式 round_013 的当前开发页增加 4 个角点、已查看回归增加 4 个角点，但快照外页面减少 6 个角点，未满足晋升条件；正式 runtime 继续使用 round_013。完整记录：[林燕丹训练报告](../research/experiments/2026-10-18-linyandan-partial-screen-training.md) · [round_016 全量基准](../research/experiments/conference-benchmark/20261018_round016/README.md)。

## round_018 教师保护训练（2026-10-18）

round_018 以 round_013 作为训练期教师，仅保护教师原本命中的旧训练角点，不蒸馏教师错误点。教师保护候选在合并开发达到 1279/1388，高于普通对照 1274/1388；冻结 1808 页全量基准为 6668/7232，相对 round_013 有 93 个改善、75 个回退，净增 18。按用户要求已晋升为正式 runtime `screen_corner_single_5814ce182305`，发布编号为 `model-20261018-124518-5814ce18`。seed=8/9 开发复核为 -3/0，稳定性限制已记录。完整结果见 [训练报告](../research/experiments/2026-10-18-teacher-guard-training.md) 与 [round_018 全量基准](../research/experiments/conference-benchmark/20261018_round018/README.md)。

## 孔令敏资料 round_021

39 张已接入（23 训练 / 8 开发 / 8 已查看回归），相同 teacher_guard 的固定预算对照选中单文件候选 `7279848d9388`。合并开发 1292→1303/1420，本场开发 13→23/32、已查看回归 11→24/32；右下角仍失败较多，旧 265 页开发少 3 点。随后按用户要求晋升为正式 round_021，配套本地版本 0.2.20。详见[训练报告](../research/experiments/2026-10-18-konglingmin-fine-corner-training.md)。

round_021 已按用户要求晋升，0.2.20 本地 macOS Apple Silicon 包验收完成；包内唯一权重为 `7279848d9388`，实测点位与冻结预测一致。见 [包验收记录](../release-assets/local-single-20261018-v0220-round021/0.2.20/macos/build-verification.json)。

<!-- zhangwenju-round022:start -->
## 张文居隧道安全与节能技术 round_022（2026-10-18）

张文居资料 48 张已接入 `202610-tunnel-lighting-zhangwenju_v1`（28 训练 / 10 开发 / 10 已查看回归），目标为宽幅 LED 屏内完整右侧 PPT，保留白底/蓝底、地图图表、工程照片、LED 条纹、倒计时与观众，原人工点不改写。round_022 从正式 round_021 做相同 teacher_guard 的固定预算对照，旧 2085 条采样保留，新增 28 页各 6 倍共 2253 条；两组均 4/8 轮、每轮 71 次更新。合并开发 initial=1335/1460, control=1351/1460, with_zhangwenju=1349/1460，选中 `control`，单文件候选 `c2267f9e0d71`。选中候选未使用张文居新增训练页，加入组权重另保留。本场开发 32→35/40、已查看回归 33→33/40；既有 355 页开发净变化 +13 点，原 265 页开发 +3 点。场景族 v3 共 258 页，旧用途/v1/v2/冻结会议集不回写；小样本、单 seed 与残留失败保留。正式 runtime 继续 round_021，候选未晋升。

[完整报告](../research/experiments/2026-10-18-zhangwenju-partial-screen-training.md)。复现入口 `runs/screen_corner_single/round_022/artifacts/run_round022.py`。
<!-- zhangwenju-round022:end -->

<!-- duchao-dataset-v1:start -->
杜超资料初次登记为 8 训练 / 3 开发 / 3 已查看回归，原登记快照保留；随后已在 round_023 的 with_hangzhou 与 family_focus 两组执行训练。选中加入两场新资料的 with_hangzhou；提高同族采样比例的分支未超过它。 详见 [完整报告](../research/experiments/2026-10-18-gengchiyuan-family-focus-training.md)。
<!-- duchao-dataset-v1:end -->

<!-- gengchiyuan-round023:start -->
## 2026-10-18 耿驰远 / 杜超 round_023 已完成

耿驰远 36 张已接入（22 训练 / 7 开发 / 7 已查看回归），杜超 14 张沿用 8 / 3 / 3。round_023 从 round_022 候选 c2267f9e0d71 初始化，完成 control / with_hangzhou / family_focus 三组固定预算对照；各组 seed=7、4/8 轮、每轮 103 次更新，训练采样 2253 / 2433 / 3291 条。合并开发 381 页：initial=1397/1524, control=1404/1524, with_hangzhou=1408/1524, family_focus=1404/1524，候选 `8e554a8d68b7`。选中加入两场新资料的 with_hangzhou；提高同族采样比例的分支未超过它。同族开发 210/256 → 228/256，已查看回归 209/252 → 230/252；原 265 页开发 993/1060 → 988/1060。场景族 v5 共 11 场、308 页；143 张实际采样的同族训练页在定向组提高至 12 倍，刘奇/孙泉明未采样训练页没有借此引入。旧用途、旧 revision、冻结会议集及源人工点保留；单种子、反复使用开发集和残留失败的限制继续保留。

合并开发唯一指标选中 `with_hangzhou`，候选 `screen_corner_single_8e554a8d68b7` 已按用户要求晋升为正式 runtime；旧 265 页开发少 5 点，右上与遮挡右下失败仍保留，本轮未重跑 1808 页全量基准。0.2.20 本地 macOS Apple Silicon 包已重新生成并验收，DMG SHA256 为 `0475e4e7dbada0ce4246b59db7b08e8b8f45b80e79b09005e7d5cf66444f197a`。

[完整报告](../research/experiments/2026-10-18-gengchiyuan-family-focus-training.md) · [包内验证](../release-assets/local-single-20261018-v0220-round023/0.2.20/macos/bundle-verification.json) · [DMG 验证](../release-assets/local-single-20261018-v0220-round023/0.2.20/macos/dmg-verification.json)。
<!-- gengchiyuan-round023:end -->


<!-- round024-mazhangyixiang-industrial:start -->
## 2026-10-19 马张翼翔 / 工业具身智能 round_024

马张翼翔 47 页按 27/10/10 接入杭州宽屏场景族；工业具身智能 26 页按 15/6/5 接入独立新场景。control、with_mazhang、with_industrial、with_both 四组从 round_023 正式权重做相同 seed=7、teacher_guard、4/8 轮和每轮 103 次更新；新增页分别 6 倍和 12 倍回放。pooled 开发分别为 1450/1588、1464/1588、1460/1588、1466/1588，选择并晋升 `with_both`。工业场景最终开发 / 留出为 21/24、20/20，粗定位为 12/24、16/20。候选 `screen_corner_single_7f2257153f0c` 已登记为正式 runtime。

详见 [训练报告](../research/experiments/2026-10-19-mazhangyixiang-industrial-training.md) · `runs/screen_corner_single/round_024/manifest.json`。
<!-- round024-mazhangyixiang-industrial:end -->

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
## 2026-10-22 智算论坛 round_029 粗定位训练完成

智算网络创新技术论坛 66 页已接入（39 训练 /14 开发 /13 已查看回归），保留完整中央 PPT 人工四角和底部观众 /椅背遮挡。round_029 从 round_028 候选做相同 20 轮、每轮 182 次更新的粗定位对照，选中直接加入组第 16 轮 `screen_corner_single_c495b67dcac4`；本轮按智算开发粗定位选优，encoder /local head 冻结。新场景开发粗定位 17→45/56、底部 5→19/28，已查看回归粗定位 9→33/52、最终 30→42/52。区域拟合分支开发为 34/56，旧域明显回退，未采用。选中候选旧 414 页开发粗定位 -21 点、最终 -12 点，pooled 最终 1535→1534/1712；左下仅 6/14，冻结精修会重新拉偏部分正确粗点。南京场景族 v2 共 148 页，旧用途 /revision /冻结会议集保留。33 项测试和四个隔离单文件实际检测通过；未晋升 /打包，正式 runtime /桌面包继续 round_027 /0.2.21。

[完整报告](../research/experiments/2026-10-22-computing-forum-region-coarse-training.md)。
<!-- computing-round029:end -->

<!-- all-history-round030:start -->
round_030 已根据全部历史可用人工资料独立训练区域识别 → 粗定位 → 精修单体模型。62 个来源、2275 页，1461 训练 /432 开发 /216 已查看回归，166 张近评估重复训练页隔离。仅通用 ImageNet 初始化，骨干和三个任务阶段重新训练，未继承 ScreenPDF 业务权重或教师。候选 `screen_corner_single_500bb4820b86`；开发最终正式 1496/1728（86.57%） → 新模型 1508/1728（87.27%）（+12 点），已查看回归正式 715/864（82.75%） → 新模型 762/864（88.19%）。南京开发 +33、InfoComm +7，openatom -18、福州高速 -12 等局部回退保留。12 /80 /48 轮固定预算，按 pooled 开发唯一角点指标冻结，先冻结再回归；含训练资料的 1808 页诊断（1 页数据集修订排除页按冻结人工框补算）/2275 页诊断与逐会议变化另报。seed=11 复验（round_031）开发 1522、已查看回归 770，粗定位 1496/1497 复现，final 增益带约 ±0.8 个百分点种子波动；精修净贡献仅 +12/+25（成熟模型 +191），openatom /福州高速 /smart-road /metaverse /商场测试集回退两 seed 复现，故建议先修精修阶段再谈晋升。23 项测试和隔离单文件实际检测通过；两 seed 复验后，固定预算、历史评估已查看等限制仍保留。未晋升 /打包，正式 runtime 仍 round_027。

[完整报告](../research/experiments/2026-10-22-all-history-region-coarse-local-training.md) · [seed 复验与综合指标](../research/experiments/2026-10-22-all-history-seed-replication.md)。
<!-- all-history-round030:end -->

<!-- round031-promotion-v0222:start -->
## 2026-10-22 round_031 全历史模型已晋升，0.2.22 本地包已验收

按用户要求把全历史独立训练的单文件模型晋升为正式 runtime `screen_corner_single_052594290a1d`（round_031，seed=11），发布编号 `model-20261022-131533-05259429`，旧权重备份 `models/runtime/screen_corner_single.pre_round027.pt`。与正式 round_027 同图重建推理：432 页开发 1496→1522（86.57%→88.08%），216 页已查看回归 715→770（82.75%→89.12%），原 428 页子集 1423→1432，冻结 1808 页诊断 6686→6882，全部 2275 页诊断 8182→8640；平均归一化误差开发 1.91%→0.76%、回归 2.16%→0.73%。冻结诊断仍有三处回退并写入发布元数据：`202604-mall` -59（疑似把屏幕外边黑框纳入目标）、`202312-openatom` -30、`202603-fuzhou-highway` -13。同轮精修专项 round_032（冻结该基座、只重训局部头 96 轮）开发 1519、回归 773，未超过基座，说明精修瓶颈不是训练轮数，未采用。0.2.22 macOS Apple Silicon 安装包已生成并验收：Finder 布局与 0.2.16 逐字节一致、版本 0.2.22、ad-hoc 严格签名、包内真实检测来源 `runtime_single` 且模型摘要一致；本地 ad-hoc 签名、未公证、未对外发布，包内含本次构建的 updater 签名产物但没有 latest.json，不构成更新通道。

[晋升审计](../training/runs/screen_corner_single/round_031/reports/promotion-audit/README.md) · [决策记录](../training/runs/screen_corner_single/round_031/reports/promotion_decision.json) · [下载 0.2.22（macOS Apple Silicon）](../release-assets/local-single-20261022-v0222-round031/0.2.22/macos/ScreenPDF_0.2.22_aarch64.dmg)。DMG SHA256：`8c35eb39c9709ac14e04b84da73e3a9b9845ff7c42fd4abf9008eed45d355773`。
<!-- round031-promotion-v0222:end -->

<!-- nanjing-edge-dataset-v1:start -->
## 待训练数据：南京边缘算力与泛在智能创新发展论坛

南京边缘算力与泛在智能创新发展论坛 67 页已按 39 训练 /13 开发 /15 已查看回归接入 `202610-nanjing-edge-computing-forum_v1`，状态 registered_not_trained，尚未启动训练。目标为宽幅 LED 屏内完整 PPT 幻灯片四边形，保留底角观众 /椅背遮挡与 LED 拍摄纹理；后续在新的 round 中冻结预算与采样后执行，旧训练快照、场景族 v1 /v2 与正式 runtime 不回写。

[固定用途与后续训练入口](../data/curated/scenes/202610-nanjing-edge-computing-forum_v1/README.md)。
<!-- nanjing-edge-dataset-v1:end -->

<!-- embodied-ai-chain-dataset-v1:start -->
## 待训练数据：上海 IN³C「AI铸链 向实而生」

2026-10-14 上海 IN³C 场次「AI铸链 向实而生」（京东工业）4 页已按 2 训练 /1 开发 /1 已查看回归接入 `202610-embodied-intelligence-ai-chain`，状态 registered_not_trained，尚未启动训练。目标为宽幅 LED 舞台屏内完整中央 PPT 幻灯片，排除左右品牌固定区与底部 IN³C 横幅，保留观众头部遮挡；后续在新的 round 中冻结预算与采样后执行，同峰会工业具身智能 26 页的既有用途、round_024 快照与正式 runtime 不回写。

[固定用途与后续训练入口](../data/curated/scenes/202610-embodied-intelligence-ai-chain_v1/README.md)。
<!-- embodied-ai-chain-dataset-v1:end -->

<!-- embodied-occluded-dataset-v1:start -->
## 待训练数据：上海 IN³C 华为 AI 与龚涛（遮挡较大）

2026-10-13 华为 AI 6 页（3 /2 /1）与龚涛「工业控制系统从自动化向自主化的演进」40 页（24 /8 /8）已接入 `202610-embodied-intelligence-huawei-ai`、`202610-embodied-intelligence-gongtao`，状态 registered_not_trained，尚未启动训练。两场遮挡较大：讲者位于屏幕左前缘，讲师台、观众头部与举手遮挡下沿和左下部；华为场屏幕占画面比例小、摩尔纹重。目标为 LED 屏完整四边形（含页脚条），人工点外推到被遮挡边缘；后续在新的 round 中冻结预算与采样后执行，同峰会已有用途、round_024 快照与正式 runtime 不回写。

[华为 AI 入口](../data/curated/scenes/202610-embodied-intelligence-huawei-ai_v1/README.md) · [龚涛入口](../data/curated/scenes/202610-embodied-intelligence-gongtao_v1/README.md)。
<!-- embodied-occluded-dataset-v1:end -->

<!-- embodied-occlusion-round033:start -->
## round_033 上海 IN³C 重遮挡两场

上海 IN³C 重遮挡两场（华为 AI 6 页 + 龚涛 40 页，27 页训练）已在 round_033 从正式 round_031 做固定预算对照：control 回放 2222 条、with_occluded 加入 27 页×24 共 2870 条，粗定位 12 轮 + 精修 16 轮、每轮 90 次更新，只按 10 页重遮挡开发唯一角点指标选 epoch 并先冻结再回归。选中 with_occluded（加入 27 页），重遮挡开发最终 32/40（80.00%）；候选 `screen_corner_single_4a6904df2cd6` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、10 页开发、人工点外推遮挡边缘与已查看回归限制保留。

[训练报告](../research/experiments/2026-10-22-embodied-occlusion-training.md)。
<!-- embodied-occlusion-round033:end -->

<!-- pm-cube-round034:start -->
## round_034 2024 新任产品经理管理魔方

2024「新任产品经理管理魔方」15 页已接入（9 训练 / 3 开发 / 3 已查看回归，投影白底幻灯片，参考模型在这批页面的角点误差约 1–5%、命中仅 16/60）。round_034 从正式 round_031 冻结 encoder 与粗定位头，只训练本地精修头 24 轮做固定预算对照：control 回放 2222 条、with_pm_cube 加入 9 页×24 共 2438 条，只按 3 页精细定位开发唯一角点指标选 epoch 并先冻结再回归。选中 with_pm_cube（加入 9 页），开发最终 11/12（91.67%）；候选 `screen_corner_single_9eb8bc25c145` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、3 页开发、只训本地头的限制保留。

[训练报告](../research/experiments/2026-10-22-pm-cube-fine-localization.md)。
<!-- pm-cube-round034:end -->

<!-- combined-new-data-round035:start -->
## round_035 五批新增资料合并加入

五批新增资料（南京边缘算力 67 页、AI铸链 4 页、华为 AI 6 页、龚涛 40 页、2024 PM 管理魔方 15 页，合计 132 页 / 77 训练页）已在 round_035 从正式 round_031 合并做固定预算对照：control 回放 2222 条、with_new_all 加入 77 页×24 共 4070 条，粗定位 12 轮 + 精修 16 轮、每轮 128 次更新，只按 27 页合并新场景开发唯一角点指标选 epoch 并先冻结再回归。选中 with_new_all（加入 77 页），合并新场景开发最终 94/108（87.04%）；候选 `screen_corner_single_5be1a6da1dd2` 未晋升，正式 runtime 与 0.2.22 安装包不变。单 seed、27 页开发、五场景混合采样与已查看回归限制保留。

[训练报告](../research/experiments/2026-10-23-combined-new-data-training.md)。
<!-- combined-new-data-round035:end -->

<!-- candidate-scene-comparison-20261023:start -->
## 候选横向对比（2026-10-23）

四个单文件模型（正式 round_031、round_033 遮挡、round_034 PM 精修、round_035 五批合并）已在同一 2407 页语料（旧 2275 + 新增 132）重新推理做分场景对比：全部页 8951 /8996 /8984 /**9067** 命中；旧 2275 页 8640 /8618 /8632 /8600；新增 132 页 311 /378 /352 /**467**；已查看回归 244 页 836 /832 /**850 /851**。相对正式：round_033 旧 -22 /新 +67 /回归 -4；round_034 旧 -8 /新 +41 /回归 +14；round_035 旧 -40 /新 +156 /回归 +15。要一个 runtime 覆盖全部新增场景选 round_035（`screen_corner_single_5be1a6da1dd2`，需重跑 1808 页冻结审计并重新打包）；要旧域风险最小选 round_034（`9eb8bc25c145`）；round_033 保留为遮挡对照；正式 round_031 仍是旧场景最佳。均未晋升。

[对比报告](../research/experiments/2026-10-23-candidate-scene-comparison.md)。
<!-- candidate-scene-comparison-20261023:end -->

<!-- round035-promotion-v0223:start -->
## round_035 已晋升（0.2.23）

按用户要求把 round_035（五批新增资料合并加入）晋升为正式单文件 runtime `screen_corner_single_5be1a6da1dd2`，发布编号 `model-20261023-041021-5be1a6da`，旧权重备份 `models/runtime/screen_corner_single.pre_round031.pt`。晋升审计同图重跑正式 round_031 与候选：459 页开发 1587→1581/1836（-6），244 页已查看回归 836→851/976（+15），冻结 1808 页诊断 6882→6853/7232（-29），全部 2275 页诊断 8640→8600/9100（-40），新增 132 页 311→467/528（+156）。冻结集合回退：福州高速 -12、开放原子 -9、扬州 -3、智慧道路照明 3 月 20 日 -3、元宇宙 -2、杭州隧道集合 -2、追光 -2、杭州智能交通 -2；改善：上海大屏 +2、3 月 19 日 +2、商场 +1、AWE +1，全部写入发布元数据。0.2.23 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `98f42ca7a06321cf8b7246ba51c2c6d12fc770b0dcf13691d4db1f9e4482c8c0`，Finder 布局与 0.2.16 逐字节一致、版本 0.2.23、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_5be1a6da1dd2`；本地 ad-hoc 签名、未公证、未对外发布，包内含 updater 签名产物但没有 latest.json，不构成更新通道。

[晋升审计](../training/runs/screen_corner_single/round_035/reports/promotion-audit/README.md)。
<!-- round035-promotion-v0223:end -->

<!-- qintong-dataset-v1:start -->
## 待训练数据：上海 IN³C 秦通场

上海 IN³C 2026 场次「面向开放世界的具身智能三维探索、移动操作与全场景仿真-秦通」33 页已接入 `202610-embodied-intelligence-qintong`（19 训练 /7 开发 /7 已查看回归，32 个内容组），目标为宽幅 LED 舞台屏内完整白色幻灯片内容区，排除两侧品牌固定区与底部 IN³C 会议横幅，保留讲者 /观众遮挡、LED 摩尔纹与轻微透视。状态 registered_not_trained：本次仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包。上海 IN³C 场景族由 v2 追加为 v3，五场共 109 页（63 训练 /24 开发 /22 已查看回归）；与既有标注无精确字节 /强 dHash 重复，2 对 pHash 标记逐对目视为不同幻灯片；旧 revision、旧用途与冻结会议集不回写。

[场景说明](../data/curated/scenes/202610-embodied-intelligence-qintong_v1/README.md)。
<!-- qintong-dataset-v1:end -->

<!-- zhufeng-luchen-dataset-v1:start -->
## 待训练数据：上海 IN³C 朱峰与陆晨

上海 IN³C 2026 两场 2026-10-13 资料已接入：「智能无人装备安全标准研究进展-朱峰」4 页（2 训练 /1 开发 /1 已查看回归，`202610-embodied-intelligence-zhufeng`）与「西门子工业边缘及人工智能赋能制造业数字化转型-陆晨」10 页（6 /2 /2，`202610-embodied-intelligence-luchen`），目标为宽幅 LED 舞台屏完整幻灯片 /屏幕四边形，排除两侧品牌固定区，保留讲者与观众遮挡、LED 摩尔纹与轻微透视。状态 registered_not_trained：仅接入训练数据，未训练、未创建新 round、未改正式 runtime 或安装包。上海 IN³C 场景族由 v3 追加为 v4，七场共 123 页（71 训练 /27 开发 /25 已查看回归）；无精确字节 /强 dHash 重复，2 对 pHash 标记（朱峰 IMG_2099 vs 龚涛 IMG_2039、陆晨 IMG_2091 vs 华为 IMG_2003）逐对目视为不同幻灯片。同目录「工业领或具身智能洞察与实践-于青民」6 张照片没有 ScreenPDF 项目文件与人工框，本批未纳入，需标注后再接入。

[朱峰](../data/curated/scenes/202610-embodied-intelligence-zhufeng_v1/README.md) · [陆晨](../data/curated/scenes/202610-embodied-intelligence-luchen_v1/README.md)。
<!-- zhufeng-luchen-dataset-v1:end -->

<!-- occlusion-round036:start -->
## round_036 遮挡场景定向训练

秦通 33 /朱峰 4 /陆晨 10 页已接入；round_036 从正式 round_035 做遮挡定向对照：control 回放 4070 条、with_new 加入 27 页×24 共 4718 条、occlusion_focus 再把 54 张遮挡相关页加 24 倍共 6014 条，粗定位 12 轮 + 精修 16 轮、每轮 188 次更新，只按 20 页遮挡开发唯一角点指标选 epoch 并先冻结再回归。选中 with_new（加入 27 页），遮挡开发最终 62/80（77.50%）；候选 `screen_corner_single_7237adbcaa54` 未晋升，正式 runtime 与 0.2.23 安装包不变。单 seed、20 页遮挡开发、加倍采样不是独立样本与已查看回归限制保留。

[训练报告](../research/experiments/2026-10-24-occlusion-focus-training.md)。
<!-- occlusion-round036:end -->

<!-- regression-compare-round036:start -->
## round_035 vs round_036 回归验证

round_036 回归验证：正式 round_035（5be1a6da1dd2）对比候选 `screen_corner_single_7237adbcaa54`，同一 2454 页（旧 2275 + 新增 179）各自重跑推理。全部页 9177→9250（+73），新增 179 页 577→651（+74，80.59%→90.92%），旧 2275 页 8600→8599（-1），冻结 1808 页诊断 6853→6846（-7），469 页开发 1604→1616（+12），254 页已查看回归 874→886（+12）。新增场景：秦通全部页 +37、龚涛 +18、陆晨 +11、朱峰 +5，新场景开发 +5、回归 +9；旧域 13 个集合 5 升 /7 降 /1 平，最差 -3，逐页没有少 3 角的页面。结论：候选达到可晋升量级（新场景 +74、旧域 -1、无整页崩坏）；晋升仍需重跑 1808 页冻结审计并重新打包，正式 runtime 暂不变。

[回归验证报告](../research/experiments/2026-10-24-round035-vs-round036-regression/README.md)。
<!-- regression-compare-round036:end -->

<!-- round036-promotion-v0224:start -->
## round_036 已晋升（0.2.24）

按用户要求把 round_036 遮挡定向候选晋升为正式单文件 runtime `screen_corner_single_7237adbcaa54`，发布编号 `model-20261023-165533-7237adbc`，旧权重备份 `models/runtime/screen_corner_single.pre_round035.pt`。同图重跑正式 round_035 与候选：469 页开发 1604→1616（+12），254 页已查看回归 874→886（+12），冻结 1808 页诊断 6853→6846（-7），全部 2275 页诊断 8600→8599（-1），新增 179 页 577→651（+74）。冻结集合回退：InfoComm /商场 /开放原子各 -3，元宇宙 -2，AWE /上海大屏 /道路照明 3.19 各 -1；改善：扬州 +3，道路照明 3.20 /智能交通年会 /福州高速 /追光各 +1；逐页没有少 3 角的页面。0.2.24 macOS Apple Silicon 本地安装包已生成并验收：DMG SHA256 `80103f2d3765ae7d28dd25e70ebbe2a1bcf050b01011de52985cc160ddb4b010`，Finder 布局与 0.2.16 参考一致（同一 .DS_Store / 卷图标摘要）、版本 0.2.24、ad-hoc 严格签名、包内唯一权重摘要与 runtime 一致、真实检测返回 `runtime_single` / `screen_corner_single_7237adbcaa54`；本地 ad-hoc 签名、未公证、未对外发布，包内含 updater 签名产物但没有 latest.json，不构成更新通道。

[晋升审计](../training/runs/screen_corner_single/round_036/reports/promotion-audit/README.md)。
<!-- round036-promotion-v0224:end -->

<!-- embodied-v5-dataset:start -->
## 待训练数据：上海 IN³C 周浩杰 /姚志强 /马战凯 /刘意杨

上海 IN³C 2026 四场 2026-10-13 资料已接入：周浩杰 6 页（3 训练 /2 开发 /1 已查看回归，`202610-embodied-intelligence-zhouhaojie`）、姚志强 4 页（2 /1 /1，`202610-embodied-intelligence-yaozhiqiang`）、马战凯 15 页（9 /3 /3，`202610-embodied-intelligence-mazhankai`；1 张 needs_review 无人工框排除）、刘意杨 14 页（8 /3 /3，`202610-embodied-intelligence-liuyiyang`），状态 registered_not_trained，尚未启动训练。目标为宽幅 LED 舞台屏完整屏幕四边形，排除两侧品牌固定区。

这批标注由当前 runtime 候选 `screen_corner_single_7237adbcaa54` 微调而来，按用户 2026-10-24 说明"基本定位相对准确但还不够精细"，登记 `label_quality=coarse_runtime_seeded_manual`、`fine_corner_truth=false`；训练用途按既有拆分保留，但细角点精度只能按近似值解释，不能当作精细真值或盲测。后续在新的 round 中冻结预算与采样后执行，旧 revision、旧用途与正式 runtime 不回写。

[场景族 v5](../data/curated/scenes/202610-shanghai-embodied-intelligence_v5/README.md) · [周浩杰](../data/curated/scenes/202610-embodied-intelligence-zhouhaojie_v1/README.md) · [姚志强](../data/curated/scenes/202610-embodied-intelligence-yaozhiqiang_v1/README.md) · [马战凯](../data/curated/scenes/202610-embodied-intelligence-mazhankai_v1/README.md) · [刘意杨](../data/curated/scenes/202610-embodied-intelligence-liuyiyang_v1/README.md)。
<!-- embodied-v5-dataset:end -->

<!-- embodied-v5-round037:start -->
## round_037 上海 IN³C 四场加入训练

周浩杰 6 /姚志强 4 /马战凯 15 /刘意杨 14 页（22 训练页）在 round_037 从正式 round_036 训练：control 回放 4718 条、with_new 加入 22 页×24 共 5246 条，粗定位 12 轮 + 精修 16 轮、每轮 164 次更新，seed=7；只按 9 页新场景开发的唯一角点指标选 epoch，先冻结再评回归。选中 with_new：新场景开发 24→30/36（66.67%→83.33%），已查看回归 27→28/32；旧 432 页开发 1494→1499、216 页已查看回归 773→773；冻结 1808 页诊断 +19，2493 页全量 +51。候选 `screen_corner_single_9b9e9c200c49` 未晋升，正式 runtime 与 0.2.24 安装包不变。四场标签为 runtime 候选微调的粗精度框，9 页开发仅 36 角点，精修阶段在新页面无净增益。

[综合报告](../research/experiments/2026-10-24-embodied-v5-training.md) · [回归对比](../research/experiments/2026-10-24-embodied-v5-round037-regression/README.md) · [训练 run](runs/screen_corner_single/round_037/README.md)。
<!-- embodied-v5-round037:end -->

<!-- round037-promotion-v0225:start -->
## round_037 已晋升（0.2.25）

按用户要求把 round_037 四场加入候选晋升为正式单文件 runtime `screen_corner_single_9b9e9c200c49`，发布编号 `model-20261023-182931-9b9e9c200c49`，旧权重备份 `models/runtime/screen_corner_single.pre_round036.pt`。同图重跑：478 页开发 1640→1650（+10）、262 页已查看回归 913→917（+4）、冻结 1808 页诊断 6846→6865（+19）、全部 2275 页诊断 8599→8614（+15）、新增 218 页 763→799（+36）。冻结 13 个集合 8 升 /4 平 /1 降，唯一回退 3 月 19 日道路照明 -1；逐页只有 +1/0/-1。0.2.25 macOS Apple Silicon DMG 已生成并验收，SHA256 `719e91c177a499adade72e253157b85dd9fb03b921cd7eb90519ef385ad430e8`；本地 ad-hoc、未公证、未对外发布，包内无 updater 产物。

[晋升审计](runs/screen_corner_single/round_037/reports/promotion-audit/README.md) · [下载 0.2.25](../release-assets/local-single-20261024-v0225-round037/0.2.25/macos/ScreenPDF_0.2.25_aarch64.dmg)。
<!-- round037-promotion-v0225:end -->

<!-- zhangkaiyuan-dataset-v1:start -->
## 待训练数据：上海 IN³C 张凯源

上海 IN³C 2026 场次「多模态大模型与云边协同赋能油气机器人智能巡检-张凯源」11 页已按 6 训练 /3 开发 /2 已查看回归接入 `202610-embodied-intelligence-zhangkaiyuan`，状态 registered_not_trained，尚未启动训练。目标为宽幅 LED 舞台屏完整屏幕四边形，排除两侧品牌固定区。

**标注限制**：11 张框全部由当前正式 runtime `screen_corner_single_9b9e9c200c49` 候选微调，逐页最大调整量仅 0.40%–1.08% 对角线；`label_quality=runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`。不是独立人工真值；训练或评估时不能把 1% 命中当作盲测或精细精度证据。后续若训练，应在新的 round 中冻结预算与采样，旧 revision、旧用途与正式 runtime 不回写。

[场景说明](../data/curated/scenes/202610-embodied-intelligence-zhangkaiyuan_v1/README.md) · [场景族 v6](../data/curated/scenes/202610-shanghai-embodied-intelligence_v6/README.md)。
<!-- zhangkaiyuan-dataset-v1:end -->

<!-- sanxiong-dataset-v1:start -->
## 待训练数据：2026 光亚展三雄极光

2026-06-10 光亚展「三雄极光」10 页已按 6 训练 /2 开发 /2 已查看回归接入 `202606-guangya-sanxiong`，状态 registered_not_trained。目标为展台 LED 屏完整四边形，排除屏幕外展台结构，顶部品牌条与页脚按人工点保留。

标注由当前正式 runtime `screen_corner_single_9b9e9c200c49` 候选微调，逐页调整 1.22%–5.56% 对角线；`label_quality=runtime_candidate_seeded_manual`、`annotation_independence=false`、`fine_corner_truth=false`，不是独立人工真值。该场不加入上海 IN³C 场景族；后续在 round_038 做精细定位训练，拆分与人工点不改写。

[场景说明](../data/curated/scenes/202606-guangya-sanxiong_v1/README.md)。
<!-- sanxiong-dataset-v1:end -->

<!-- fine-local-round038:start -->
## round_038 张凯源 /三雄极光精细定位训练

张凯源 11 页（6/3/2）与三雄极光 10 页（6/2/2）在 round_038 从正式 round_037 冻结 encoder/粗定位头，只训练精修头：control 回放 5246 条、with_new 追加 12 页×24 共 5534 条，24 轮、每轮 173 次更新，seed=7；只按 5 页新场景开发选 epoch，先冻结再评回归。选中 with_new：新场景开发 13→18/20（65.00%→90.00%），已查看回归 9→13/16；三雄极光开发 1→6/8、回归 1→5/8，张凯源保持 20/20；旧 432 页开发 +5、216 页回归持平；2514 页全量 +24，冻结 1808 -4（3 升 /5 平 /5 降，最大 -2，逐页无丢 3 角）。候选 `screen_corner_single_31adf6eec801` 未晋升，正式 runtime 与 0.2.25 安装包不变。限制：5 页开发仅 20 角点、单 seed、标签为 runtime 候选微调。

[综合报告](../research/experiments/2026-10-24-fine-local-round038.md) · [回归对比](../research/experiments/2026-10-24-fine-local-round038-regression/README.md) · [训练 run](runs/screen_corner_single/round_038/README.md)。
<!-- fine-local-round038:end -->

<!-- round038-promotion-v0226:start -->
## round_038 已晋升（0.2.26）

按用户要求把 round_038 精细定位候选晋升为正式单文件 runtime `screen_corner_single_31adf6eec801`，发布编号 `model-20261023-191338-31adf6eec801`，旧权重备份 `models/runtime/screen_corner_single.pre_round037.pt`。同图：483 页开发 1663→1671（+8）、266 页已查看回归 926→930（+4）、新增 239 页 +22、旧域 2275 页 +2、冻结 1808 页 -4（3 升 /5 平 /5 降，最大 -2，逐页无丢 3 角）。0.2.26 macOS Apple Silicon DMG 已验收，SHA256 `d3933635dc9566afcdac0b76bf06edf1a4a4de10b368d68c1358097b4657acd3`；本地 ad-hoc、未公证、无 updater 产物。

[晋升审计](runs/screen_corner_single/round_038/reports/promotion-audit/README.md) · [下载 0.2.26](../release-assets/local-single-20261024-v0226-round038/0.2.26/macos/ScreenPDF_0.2.26_aarch64.dmg)。
<!-- round038-promotion-v0226:end -->

<!-- qinghui-round039:start -->
## round_039 清晖同学会加入训练

清晖 116 页（69 训练 /24 开发 /23 已查看回归）从正式 round_038 出发，在含 3.20 精细标注的 base 快照上做固定预算对照：control 5535 条、with_new 加入 69 页 ×24 共 7191 条，12 轮粗定位 + 16 轮精修、每轮 225 次更新、seed=7，只按清晖 24 页开发选 epoch，先冻结再回归。清晖开发 84→90/96、已查看回归 77→85/92、全部 161→175/188；原 483 页开发 1671→1665/1932，合并开发 1755/2028 持平，合并已查看回归 1007→1013/1156。候选 `screen_corner_single_049cd9cdf0de` 未晋升，正式 runtime 与 0.2.26 安装包不变。限制：单 seed、24 页开发仅 96 角点、标签由 runtime 候选微调（非独立真值）、holdout 已查看、69 页 ×24 非独立样本。

[综合报告](../research/experiments/2026-10-24-qinghui-alumni-training.md) · [训练 run](runs/screen_corner_single/round_039/README.md) · [场景说明](../data/curated/scenes/202512-qinghui-alumni_v1/README.md)。
<!-- qinghui-round039:end -->
<!-- white-mix-round040:start -->
## round_040 白屏混合角点精修标签对照

白屏混合 42 页（33 训练 /9 开发 /0 回归）的精修人工四角接入后，从正式 round_038 做一次纯标签对照：control=旧标注、refined=精修标注，两组都是 5575 条、12 轮粗定位 + 16 轮精修、每轮 175 次更新、seed=7，共用同一套评估真值，只按白屏混合 9 页开发选 epoch，先冻结再回归。最终三组并列 33/36，按平局规则选中 `initial`，**checkpoint 与正式 runtime 字节相同（`screen_corner_single_31adf6eec801`），本轮没有新候选**。粗定位阶段精修标注有信号（新真值下 31→32 /33），但被精修阶段抹平；原域 474 页两组同时 -14 点、control 与 refined 之间净 0，说明是重训漂移而非标注贡献。剩余 3 个开发失配全是右下角（12.68% /3.50% /2.00%）。正式 runtime、0.2.26 安装包、冻结 `conference_v1` 与旧快照不变。限制：单 seed、9 页开发仅 36 角点、修订幅度全部 <0.87% 对角线（指标不敏感）、开发集历史反复使用。

[综合报告](../research/experiments/2026-10-27-white-mix-corner-refinement.md) · [训练 run](runs/screen_corner_single/round_040/README.md) · [新快照](datasets/screen_corner_single_all_manual_3_20fine_whitemix_v1/README.md)。
<!-- white-mix-round040:end -->


<!-- open-bigdata-zhangliwen-round041:start -->
## round_041 金融领域开源大语言模型（张立文）精细定位

2026-10-29 上海开源大数据创新论坛张立文 61 页（36 训练 /13 开发 /12 已查看回归）接入后，按"粗定位还可以、精细定位有偏差"从正式 round_038 冻结 encoder 与粗定位头，只训练精修头：control 回放 5575 条、with_new 加入 36 页 ×24 共 6439 条，24 轮、每轮 202 次更新、seed=7，只按本场 13 页开发选 epoch 并先冻结再回归。选中 with_new：开发 30→49/52、已查看回归 31→46/48、61 页 144→238/244；原 483 页开发 1671→1668、266 页回归 930→924；合并开发 1701→1717、合并回归 961→970。同图 2551 页 9605→9699（+94），冻结 1808 页 6861→6855（-6，4 升 /6 退 /3 平，无单页丢 3 角）；2551 页粗定位输出与初始逐位相同，增益全部来自精修头。候选 `screen_corner_single_c2409691b054` 已按用户要求晋升为正式 runtime（发布编号 `model-20261029-152403-c2409691b054`），并重新打包 0.2.27（DMG SHA256 `df7ba0eaeff08f2718d1613c85430a1081bba7e6bf81f997f22adfe718ec7b6c`，Finder 布局与 0.2.16 一致、ad-hoc 签名、包内权重摘要一致、真实检测返回新模型，本地 ad-hoc、未公证、无 updater 产物）。限制：单 seed、13 页开发仅 52 角点、标注由 runtime 候选微调（0.93%–30.56%，非独立真值）、holdout 已查看、36 页 ×24 非独立样本；`IMG_2776` 因粗定位冻结仍错左上 17.85%。

[综合报告](../research/experiments/2026-10-29-open-source-bigdata-zhangliwen.md) · [训练 run](runs/screen_corner_single/round_041/README.md) · [回归对比](../research/experiments/2026-10-29-open-source-bigdata-zhangliwen-regression/README.md) · [场景说明](../data/curated/scenes/202610-open-source-bigdata-zhangliwen_v1/README.md)。
<!-- open-bigdata-zhangliwen-round041:end -->

<!-- qujingjing-xuhui-round042:start -->
## round_042 瞿晶晶 /徐辉精细定位

2026-10-29 上海开源大数据创新论坛同场次瞿晶晶、徐辉各 27 页接入后，从正式 round_041 冻结 encoder 与粗定位头，只训练精修头：两组都回放快照 5575 + 张立文 36 页 ×24 = 6439 条，with_new 再加两场 30 页 ×24 共 7159 条；24 轮、每轮 224 次更新、seed=7，只按两场 12 页开发选 epoch 并先冻结再回归。选中 with_new：两场开发 42→47/48、已查看回归 45→46/48、24 页 87→93/96；原 483 页开发 1668→1688（+20）、266 页回归 924→936（+12）；张立文重放 49→50/46→47；合并开发 1759→1785、合并回归 1015→1029。同图 2605 页 9897→9953（+56），冻结 1808 页 6855→6881（+26，8 升 /0 退 /5 平，无单页丢 3 角）。control 只回放也改善原域 +11/+7、两场开发 +2，部分收益来自精修头再拟合；with_new 相对 control 多 +9/+5 原域、+3/0 两场。候选 `screen_corner_single_9db7ad220b21` 已按用户要求晋升为正式 runtime（发布编号 `model-20261029-180536-9db7ad220b21`），并重新打包 0.2.28（DMG SHA256 `ceadc2d942ab2498a5f48ce310ef35313ad92f55e57b2ddc011ac0952bf4ef5c`，Finder 布局与 0.2.16 一致、ad-hoc 签名、包内权重摘要一致、真实检测返回新模型，本地 ad-hoc、未公证、无 updater 产物）。限制：单 seed、两场开发 12 页 /48 角点、标注由 runtime 候选微调（非独立真值）、holdout 已查看、30 页 ×24 非独立样本。

[综合报告](../research/experiments/2026-10-29-qujingjing-xuhui.md) · [训练 run](runs/screen_corner_single/round_042/README.md) · [回归对比](../research/experiments/2026-10-29-qujingjing-xuhui-regression/README.md) · [瞿晶晶场景](../data/curated/scenes/202610-open-source-bigdata-qujingjing_v1/README.md) · [徐辉场景](../data/curated/scenes/202610-open-source-bigdata-xuhui_v1/README.md)。
<!-- qujingjing-xuhui-round042:end -->
