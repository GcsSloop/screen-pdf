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
3. [docs/repository-layout.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/architecture/repository-layout.md)
4. [docs/key-metrics.md](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/docs/status/key-metrics.md)

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

- coarse/global 主干仍以 `r3` 为核心。
- local-corner 当前最稳候选仍是 `v28`。
- `v28` 不能替代 `r3`，只能作为后续局部角点精修候选。
- 本仓库当前的关键问题是：
  - 在不破坏泛化的前提下继续提升角点精度
  - 梳理训练增强策略，避免过拟合专项场景
- 本仓库当前必须守住的关键指标是：
  - 平均点位偏差 `< 0.5%`
  - 四点全部 `< 1%` 命中率 `> 80%`
  - 单张识别耗时 `< 500 ms`

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

## 如果你要改模型路径或打包

关注：

- [detect_frame.py](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/engine/detect_frame.py)
- [tauri.conf.json](/Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/desktop/src-tauri/tauri.conf.json)
