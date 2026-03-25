# Harness Engineering Notes

本仓库的重构借鉴了“把高信号入口、执行计划、代理协作说明、程序代码、模型资产、研究沉淀分层管理”的思路。

落到本项目上的具体做法是：

- 用 `docs` 承担高信号入口
- 用 `AGENTS.md` 约束代理的阅读路径
- 用 `program` 承担可执行源码
- 用 `models/runtime` 承担当前运行时模型
- 用 `research/experiments` 承担历史实验和大体量资产

这样可以减少新代理进入仓库后的搜索成本，也能避免训练资产继续侵占程序目录。
