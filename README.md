# Screen PDF

桌面版半自动图片矫正、压缩、OCR 叠层与 PDF 导出工具。

本仓库已经按“程序 / 模型 / 研究资产 / 文档”拆分，目标是让运行时、训练试验和文档沉淀分层清晰，便于持续迭代和代理协作。

## 目录

- `program/desktop`: Tauri + Vite 桌面程序源码。
- `program/engine`: Python 检测、OCR、导出、训练与评估代码。
  - `runtime`: 运行时主链路
  - `training`: 训练与调参
  - `eval`: 评估与回归
- `data`: 原始数据、清洗数据、拆分数据、派生任务数据与 benchmark。
- `training`: 训练配置、runs、checkpoint、报告与模型注册。
- `models/runtime`: 当前运行时使用的模型。
- `research/experiments`: 历史训练、评估、实验产物和大体量资料。
- `docs`: 架构说明、执行计划、当前状态、代理协作入口。

兼容入口：

- `docs/plans` 是到 `research/experiments` 的兼容链接，旧脚本仍可继续访问原有实验目录。

## 快速开始

### 1. 前端和桌面壳

```bash
cd /Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/desktop
pnpm install
pnpm tauri dev
```

### 2. Python 引擎

```bash
cd /Users/gcssloop/WorkSpace/AIGC/screen-pdf
PYTHONPATH=program/engine python program/engine/detect_frame.py --help
```

### 3. 运行时模型

默认从下面目录读取：

```text
/Users/gcssloop/WorkSpace/AIGC/screen-pdf/models/runtime
```

也可以通过环境变量覆盖：

```bash
export SCREEN_PDF_MODEL_DIR=/absolute/path/to/models/runtime
```

## 桌面打包

当前推荐使用仓库根目录的一键打包入口：

```bash
cd /Users/gcssloop/WorkSpace/AIGC/screen-pdf
make package-desktop
```

或直接执行：

```bash
bash scripts/desktop/package_local_release.sh
```

这套打包框架会执行：

- 准备平台运行时到 `program/engine/vendor/<platform>/`
- 运行 Tauri 桌面构建
- 收集 release 产物到 `release-assets/<version>/<platform>/`
- 校验包内 Python、OCR 依赖和三阶段模型文件

运行时依赖契约：

- 包内自带 Python 运行时
- 包内自带 `tesseract`
- 包内自带 `ghostscript`
- 包内自带 `models/runtime`

运行时准备默认由平台脚本负责：

- `scripts/desktop/prepare_runtime_macos.sh`
- `scripts/desktop/prepare_runtime_windows.ps1`
- `scripts/desktop/prepare_runtime_linux.sh`

当前默认要求通过 `SCREEN_PDF_RUNTIME_SOURCE_DIR` 提供已经准备好的平台运行时树，布局如下：

```text
<runtime-source>/
  macos/bin/python3
  macos/bin/tesseract
  macos/bin/gs
  windows/bin/python.exe
  windows/bin/tesseract.exe
  windows/bin/gswin64c.exe
  linux/bin/python3
  linux/bin/tesseract
  linux/bin/gs
```

桌面 Python 运行时依赖清单见：

- [requirements-runtime.txt](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/engine/requirements-runtime.txt)

本地打包产物会收集到：

```text
/Users/gcssloop/WorkSpace/AIGC/screen-pdf/release-assets/<version>/<platform>
```

GitHub Actions 会复用同一套脚本合同执行多平台打包。

## 版本与发布

程序版本和模型版本已经分流：

- 程序版本使用 git tag，例如 `v0.2.1`
- 模型发布使用独立 tag，例如 `model-20260330-153045-ab12cd34`

规则：

- `v*` 触发桌面端多平台打包和 GitHub Release
- `model-*` 只校验并发布模型元数据，不触发桌面程序编译
- 桌面端自动更新固定读取 GitHub Release 中的 `latest.json`

同步桌面端版本号：

```bash
bash scripts/release/sync_release_metadata.sh --tag v0.2.1
```

生成 updater manifest：

```bash
bash scripts/release/generate_updater_manifest.sh \
  --tag v0.2.1 \
  --assets-root /absolute/path/to/release-assets \
  --output /absolute/path/to/release-assets/latest.json
```

## 当前运行时链路

1. `global coarse` 先给出全局四边形。
2. `roi refine` 在粗框基础上收紧 ROI。
3. `local corner refine` 在 ROI 结果上做四角精修。
4. OCR、压缩、PDF 导出在导出链路中执行。

当前建议的运行时组合：

- coarse/global：`r3`
- local corner refine：当前保留候选 `v28`

说明：

- `r3` 负责把轮廓先找对。
- `v28` 只适合作为局部角点精修候选，不应单独主导全局结果。

## 关键指标

当前必须持续盯住的目标：

1. 平均点位偏差 `< 0.5%`
2. 四点全部 `< 1%` 命中率 `> 80%`
3. 单张识别耗时 `< 500 ms`

详见：

- [docs/status/key-metrics.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/key-metrics.md)
- [docs/architecture/data-and-training-layout.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/architecture/data-and-training-layout.md)

## 阅读顺序

1. [AGENTS.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/AGENTS.md)
2. [docs/current-status.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/current-status.md)
3. [docs/status/model-naming-rules.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/model-naming-rules.md)
4. [docs/status/model-release-convention.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/model-release-convention.md)
5. [docs/status/distillation-run-convention.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/distillation-run-convention.md)
6. [docs/repository-layout.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/architecture/repository-layout.md)
7. [docs/repository-migration-and-bootstrap.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/exec-plans/active/2026-03-25-repository-migration-and-bootstrap.md)

## 说明

- 这次迁移默认保留源码、运行时模型、训练与实验资料。
- 构建缓存未迁入新仓库，例如 `node_modules`、`dist`、`src-tauri/target`、`__pycache__`。
- 旧仓库仍保留，便于校验和回滚；后续确认无误后再清理。
