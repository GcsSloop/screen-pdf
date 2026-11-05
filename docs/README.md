# Docs

高信号文档入口。

数据查看、人工校正和同集比较先读 [会议场次集合](status/conference-collections.md)。
数据集数量与去重清单见 [数据集索引](status/dataset-index.md)。

打包桌面程序时，按 [桌面程序打包流程](runbooks/desktop-packaging.md) 检查版本、工具链和平台运行时，并完成安装包验收。

README 里使用的示意图由 [scripts/docs/generate_readme_screenshots.py](../scripts/docs/generate_readme_screenshots.py) 生成（见 [docs/assets/screenshots](assets/screenshots/README.md)）。

## 阅读顺序

建议阅读顺序：

1. [current-status.md](status/current-status.md)
2. [model-naming-rules.md](status/model-naming-rules.md)
3. [model-release-convention.md](status/model-release-convention.md)
4. [distillation-run-convention.md](status/distillation-run-convention.md)
5. [repository-layout.md](architecture/repository-layout.md)
6. [key-metrics.md](status/key-metrics.md)
7. [data-and-training-layout.md](architecture/data-and-training-layout.md)
8. [repository-migration-and-bootstrap.md](exec-plans/active/2026-03-25-repository-migration-and-bootstrap.md)
9. [release-history.md](status/release-history.md)：原 README 的逐轮更新记录（round_012 → round_038）

兼容说明：

- `plans` 是到 `../research/experiments` 的兼容链接。
