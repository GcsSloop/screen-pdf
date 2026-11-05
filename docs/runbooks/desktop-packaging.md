# 桌面程序打包流程

本文是 ScreenPDF 本地构建、候选模型试装和正式程序发布的操作入口。统一打包实现位于 `scripts/desktop/package_local_release.sh`；CI 的 macOS、Windows、Linux 构建也调用该脚本。修改打包行为后，应同步更新本文和对应检查。

**2026-10-18 修复记录：** round_013 的本地 DMG 曾被裸 `hdiutil` 重建，导致镜像根目录缺少“应用程序”拖放入口。现已恢复到 0.2.16 的 Finder 布局，并在 `collect_release_assets.sh` 收集前后调用 `verify_macos_dmg_layout.py` 与 0.2.16 基准 DMG 比对 `.DS_Store`、卷图标、根目录项目和 `Applications -> /Applications`；任何差异都会中止收集，不能再把布局回退的镜像当作可交付安装包。

## 先区分三种操作

| 操作 | 用途 | 版本 / 模型行为 |
| --- | --- | --- |
| 本地程序包 | 验证当前源码并生成本机安装包 | 使用仓库当前程序版本；默认打入 `models/runtime` 中正式模型 |
| 候选模型试装包 | 在不晋升正式模型的情况下验证候选权重 | 通过独立 Tauri 配置映射候选权重；使用独立产物目录 |
| 正式程序发布 | 生成面向用户的多平台程序版本和更新清单 | 使用 `v<major>.<minor>.<patch>`；推送 `v*` 标签触发 `.github/workflows/desktop-release.yml` |

模型正式发布使用独立的 `model-*` 标签，不会打包桌面程序。打入安装包不等于模型晋升；候选比较和正式 runtime 晋升仍按 [模型发布约定](../status/model-release-convention.md) 执行。

## 打包前检查

1. **确认版本。** 桌面版本记录在 `program/desktop/package.json`、`program/desktop/src-tauri/tauri.conf.json` 和 `program/desktop/src-tauri/Cargo.toml`。发布新程序版本时，用带标签的版本同步命令更新三处；只构建当前版本时不要传 `--tag`。

   ```bash
   bash scripts/release/sync_release_metadata.sh --tag v0.2.18
   python3 scripts/desktop/release_metadata.py version --repo-root .
   ```

   `package_local_release.sh` 内部执行不带参数的版本同步命令：它校验前两处版本并写出发布元数据，不会修改版本号。版本同步命令会改写三个版本文件，先确认工作区中已有改动并保留它们。

2. **使用 CI 相同的工具链。** 锁文件为 pnpm 7.33.7；GitHub Actions 使用 Node.js 22、Python 3.11 和 Rust stable。安装前端依赖时使用 frozen lockfile，避免安装过程改写依赖锁文件。

   ```bash
   pnpm --version  # 应为 7.33.7
   pnpm --dir program/desktop install --frozen-lockfile
   ```

3. **准备目标平台运行时。** 设置 `SCREEN_PDF_RUNTIME_SOURCE_DIR` 指向下列目录结构。平台准备脚本会将对应二进制复制到 `program/engine/vendor/<platform>/bin/`；该目录是本地构建输入，不应提交生成的运行时二进制。

   ```text
   <runtime-source>/
     macos/bin/python3, tesseract, gs
     windows/bin/python.exe, tesseract.exe, gswin64c.exe
     linux/bin/python3, tesseract, gs
   ```

   macOS 和 Linux 的 Python、Tesseract、Ghostscript 需要是目标系统可运行的版本。Windows 使用 PowerShell 环境准备运行时。具体系统构建依赖以 `.github/workflows/desktop-release.yml` 对应平台步骤为准。

4. **确认模型输入。** 正式程序包使用已晋升的 `models/runtime`。如果要试装训练候选，先确认候选权重、模型 manifest 和加载器版本相符；使用配置覆盖，不要把候选复制进正式 runtime。对比报告和当前选择规则见 [评估契约](../status/evaluation-contract.md)。

5. **运行构建前检查。** 打包脚本负责构建和收集，不会替代前端、Rust、发布脚本测试。

   ```bash
   pnpm --dir program/desktop exec vitest run
   pnpm --dir program/desktop build
   cargo check --manifest-path program/desktop/src-tauri/Cargo.toml
   bash scripts/test/desktop_updater_config_test.sh
   bash scripts/test/generate_updater_manifest_test.sh
   bash scripts/test/desktop_release_workflow_publish_test.sh
   bash scripts/test/package_local_release_test.sh
   ```

## 本地构建

在目标操作系统构建。平台可自动识别；需要明确平台时使用对应 `make` 目标。

```bash
# 自动选择当前平台
make package-desktop

# 明确构建 macOS 或 Linux
make package-desktop-macos
make package-desktop-linux
```

这几个入口都会调用 `scripts/desktop/package_local_release.sh`。Windows 可在 Windows 的 Bash 环境中执行该脚本并设置 `RELEASE_PLATFORM=windows`。

打包脚本按顺序准备运行时、校验版本并写元数据、调用 Tauri 构建、对 macOS app 签名/尝试公证、收集产物并验证包内运行时。通过 `SCREEN_PDF_SMOKE_IMAGE` 可让 macOS 包内验证实际运行检测器。macOS 产物收集还会挂载最终 DMG，强制检查根目录含 `ScreenPDF.app`、`Applications -> /Applications` 和 Finder 的 `.DS_Store` 布局元数据，避免生成只能打开、不能直接拖入“应用程序”或图标未按安装引导排列的镜像。

默认产物位置：

```text
release-assets/<version>/<platform>/
```

`RELEASE_ASSET_DIR` 是该目录树的根。每次候选试装包都指定唯一目录，避免覆盖正式版本或混合不同模型的安装包：

```bash
RELEASE_PLATFORM=macos \
SCREEN_PDF_RUNTIME_SOURCE_DIR=/absolute/path/to/runtime-source \
SCREEN_PDF_SMOKE_IMAGE=/absolute/path/to/test-image.jpg \
RELEASE_ASSET_DIR="$PWD/release-assets/local-candidate-<date>-<model-id>" \
bash scripts/desktop/package_local_release.sh
```

候选模型通过 `SCREEN_PDF_TAURI_CONFIG=/absolute/path/to/candidate-tauri-config.json` 提供 Tauri 配置覆盖。覆盖配置应将候选权重及其 manifest 放入包内 `models/runtime/`，manifest 使用 `architecture=screen_corner_single_v1`、`status=packaged_candidate`，并记录实际权重的 SHA1 和 SHA256。单文件模型的 Python 加载器必须与候选权重一起打包。macOS 验证器会检查加载器、唯一权重和摘要；设置 `SCREEN_PDF_SMOKE_IMAGE` 后还会检查检测结果确实来自包内模型。

`SKIP_RUNTIME_PREPARE=1` 只适用于目标平台的 `program/engine/vendor/<platform>/bin/` 已由本次构建正确准备好的情况。缺少运行时源目录时，不要用跳过选项掩盖问题。

本地配置关闭 `createUpdaterArtifacts` 时，还要检查构建缓存中的旧 `.app.tar.gz` / `.sig`：收集器会扫描现存文件，独立的输出目录不能防止旧 updater 产物被带入。先把本次验收的新 app 和 DMG 放入独立的 bundle 暂存目录，通过 `RELEASE_BUNDLE_DIR` 再收集，核对最终 `release-assets.json` 和 `SHA256SUMS.txt` 只包含本次产物。旧文件隔离留档，不能把不匹配的旧签名作为新版更新产物交付。

## 最终产物验收

不要只依据 Tauri 构建成功判断可交付。macOS 至少完成以下检查：

1. 确认 `.dmg` 文件名中的版本和架构正确，且 `SHA256SUMS.txt` 中有该文件的摘要。
2. 只读挂载最终 DMG，确认根目录可见项目只有 `ScreenPDF.app` 和 `Applications`，后者是指向 `/Applications` 的符号链接，并存在 `.DS_Store`。Finder 窗口应与 0.2.16 完全一致：白色背景，左侧是 `ScreenPDF.app`，右侧是带箭头的 `Applications`。收集脚本默认使用 `release-assets/local-single-20261017-v0216/0.2.16/macos/ScreenPDF_0.2.16_aarch64.dmg` 作为布局基准，逐字节比较 `.DS_Store` 和卷图标。不要使用裸 `hdiutil create -srcfolder` 替代 Tauri 的 DMG 生成器；需要手工重建时必须使用 `bundle_dmg.sh` 的 `--app-drop-link`、`--window-size` 和两个 `--icon` 参数，或从 0.2.16 DMG 保留 Finder 元数据。
3. 检查 `ScreenPDF.app/Contents/Info.plist` 里的版本与本次构建一致。
4. 对挂载后的 app 执行 `codesign --verify --deep --strict --verbose=2`。
5. 对挂载后的 app 运行包验证器；加入测试图像以执行真实检测：

   ```bash
   python3 scripts/desktop/verify_desktop_bundle.py \
     --bundle /path/to/mounted/ScreenPDF.app \
     --platform macos \
     --smoke-image /absolute/path/to/test-image.jpg
   ```

6. 核对验证结果中的模型摘要、模型 ID 与本次预期一致，再记录最终 DMG 的 SHA256、拖放链接、签名、公证状态和验证结果。收集脚本会把这些信息写入 `dmg-verification.json`。

`package_local_release.sh` 当前只对 macOS app bundle 执行自动包验证；Windows 和 Linux 构建会明确提示跳过此验证，不能把构建成功写成包内验收通过。最终 DMG 挂载验证也应以最终 `.dmg` 为对象重新执行，不能仅复用构建目录下 `.app` 的结果。

默认 macOS 签名是 ad-hoc，适合本地验证，不等同于 Developer ID 签名或 Apple 公证。需要对外分发时，应提供有效的 `APPLE_SIGNING_IDENTITY`、`APPLE_API_KEY_PATH`、`APPLE_API_KEY_ID`、`APPLE_API_ISSUER`，并设置 `REQUIRE_MACOS_NOTARIZATION=1`；未完成公证时必须如实注明。正式自动更新发布还需要按 GitHub Release 工作流提供 updater 签名密钥。

自动收集的 `release-assets.json` 和 `SHA256SUMS.txt` 位于版本/平台产物目录；设置 `RELEASE_ASSET_DIR` 时，`release-metadata.json` 写在该目录根部；不设置时，默认写入 `release-assets/<version>/meta/`。包验证器结果输出到终端；作为交付证据时应保存到产物目录，并记录构建命令、源码 revision、程序版本、模型标识/摘要、实际测试、签名与公证状态。

## 常见问题定位

| 现象 | 检查项 |
| --- | --- |
| 版本校验失败 | 检查 `package.json` 和 `tauri.conf.json` 是否相同；确认需要改版本时用 `sync_release_metadata.sh --tag vX.Y.Z`，并复核 `Cargo.toml` |
| 缺少运行时输入 | 检查 `SCREEN_PDF_RUNTIME_SOURCE_DIR` 和目标平台 `bin/` 文件名、权限及架构 |
| 模型摘要不匹配 | 检查 Tauri 覆盖配置是否指向预期 checkpoint，manifest 的 SHA1/SHA256 是否针对最终文件生成 |
| 包内找不到模型或加载器 | 检查 Tauri 资源映射目标路径是否在 `Contents/Resources/models/runtime` 与 `Contents/Resources/engine` 中 |
| 检测冒烟失败 | 先用同一测试图运行未打包模型入口，再检查包内 Python 依赖和图片路径；确认结果来源是 `runtime_*` |
| pnpm 版本不匹配导致 `pnpm install` 中止 | 锁文件由 pnpm 7.33.7 生成；本机 pnpm 11 会报 `ERR_PNPM_ABORTED_REMOVE_MODULES_DIR_NO_TTY`。用 `PATH` 前置一个 `exec corepack pnpm@7.33.7 "$@"` 的包装脚本再打包，或安装锁文件对应版本，不要用新版本重写锁文件。 |
| `bundle_dmg.sh` 失败 | 先确认没有旧的 ScreenPDF 卷挂载（`hdiutil info` 后 `hdiutil detach`）；磁盘映像创建还需要当前进程有磁盘操作权限。 |
| macOS 公证被跳过 | 检查 Developer ID 身份和四个 Apple API 凭据；要求必须公证时设置 `REQUIRE_MACOS_NOTARIZATION=1` 让缺失凭据直接失败 |
| Windows/Linux 没有包验证报告 | 当前脚本尚未实现这两平台的自动 bundle 验证；单独记录此限制并在目标系统手工验收 |

## 正式发布

正式多平台程序发布由 `.github/workflows/desktop-release.yml` 管理。发布前准备并审阅版本变更，提交后推送 `vX.Y.Z` 标签；工作流会校验发布脚本、分别构建平台包、生成 `latest.json` 并发布 GitHub Release。手动运行 workflow 时也要选择 `vX.Y.Z` 标签作为运行 ref；它不是本地候选构建入口。

模型发布仍按独立 `model-*` 流程进行。程序版本升级、候选模型试装和模型晋升是三个不同决策，分别记录，不以“已打入安装包”代替“已晋升”或“已发布”。
