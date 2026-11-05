# Runtime

这个目录只放程序默认会加载的运行时模型。

## 现阶段兼容文件

当前默认单文件模型：

- `screen_corner_single.pt`（round042 瞿晶晶 /徐辉精细定位，`screen_corner_single_9db7ad220b21`）
- `screen_corner_single.json`

旧三阶段文件继续保留作兼容和回溯；检测入口发现单文件模型时优先加载它。

- `global_corner_model.pt`
- `corner_heatmap_model.pt`
- `local_corner_moe_coord_model.pt`
- `local_corner_moe_model.pt`
- `deep_screen_r1_2026_03_28.json`

## 当前固定发布

- 单文件：`screen_corner_single_round042_qujingjing_xuhui` (`screen_corner_single_9db7ad220b21`)
- 模型发布：`model-20261029-180536-9db7ad220b21`
- round：`round_042`（从正式 round_041 初始化，冻结 encoder/粗定位头，只训练精修头；回放张立文 36 页，加入瞿晶晶 /徐辉各 15 个训练页）
- 旧兼容链：`r130 + c22 + r69probe`

round_042 已按用户要求晋升：两场开发 42→47/48（+5）、已查看回归 45→46/48（+1）、54 页诊断 198→213/216；原 483 页开发 1668→1688（+20）、266 页回归 925→936（+11）、张立文重放 238→240/244；合并开发 1759→1785、合并回归 1015→1029；冻结 1808 页诊断 6855→6881（+26，8 升 /0 退 /5 平，逐页无丢 3 角）。control 分解显示部分收益来自精修头再拟合，新增 30 页的独立增量是两场开发 +3、回归 0、原域 +9/+5。逐项见 `training/runs/screen_corner_single/round_042/reports/promotion-audit/README.md` 与 `research/experiments/2026-10-29-qujingjing-xuhui-regression/README.md`；上一版权重备份为 `screen_corner_single.pre_round041.pt`。桌面版本为 0.2.28（本地构建）：[下载](../../release-assets/local-single-20261030-v0228-round042/0.2.28/macos/ScreenPDF_0.2.28_aarch64.dmg)，DMG SHA256 `ceadc2d942ab2498a5f48ce310ef35313ad92f55e57b2ddc011ac0952bf4ef5c`。

## 统一蒸馏模型

未来统一模型正式发布后，建议直接使用对外发布名落盘：

- `deep_screen_v{major}.pt`

例如：

- `deep_screen_v1.pt`

## 约定

1. 这里不放历史训练 run 产物。
2. 这里不放实验中间件。
3. 这里只放默认运行时会加载的正式模型，或其兼容资产。
4. 新的统一模型晋升后，应同步更新 `training/registry` 和 `docs/status`。
