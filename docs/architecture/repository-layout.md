# Repository Layout

## 设计目标

这版结构强调四件事：

1. 程序源码和运行时模型分开。
2. 训练与实验资产独立，不污染主程序目录。
3. 文档分成“高信号入口”与“历史实验沉淀”两层。
4. 保留旧路径兼容能力，降低迁移成本。

## 顶层结构

```text
screen-pdf/
├── AGENTS.md
├── README.md
├── data/
├── docs/
│   ├── architecture/
│   ├── exec-plans/
│   ├── status/
│   └── plans -> ../research/experiments
├── models/
│   └── runtime/
├── program/
│   ├── desktop/
│   └── engine/
├── training/
└── research/
    └── experiments/
```

## 分层说明

### `program/desktop`

- Tauri 壳
- 前端页面
- 打包配置
- 图标和静态资源

### `program/engine`

- Python 运行时检测
- OCR / PDF 导出
- 训练与评估脚本
- 自动化测试
- 分组视图：
  - `runtime`
  - `training`
  - `eval`
  - `tests`

### `data`

- `raw` 保留用户原始项目目录习惯
- `staging` 放导入中间态
- `curated` 放结构化项目、场景与标注
- `splits` 放训练 / 验证 / holdout 正式拆分
- `derived` 放 global-corner、local-corner、OCR overlay 等任务数据
- `benchmarks` 放固定评测集和结果索引

### `training`

- 训练配置
- 训练工作目录
- checkpoint
- 报告
- 模型注册与晋升记录

### `models/runtime`

- 当前程序默认加载的模型
- 与训练历史分离
- 便于单独替换、发版、回退

### `research/experiments`

- 原始实验目录
- 训练数据导出
- 训练 run
- 各代评估 json / md
- 较大的训练资料和实验产物

### `docs`

- 新代理或新维护者应优先阅读的内容
- 当前状态
- 架构说明
- 执行计划
- 高信号总结

## 运行时模型解析

`program/engine/detect_frame.py` 现在按以下顺序查找模型：

1. `SCREEN_PDF_MODEL_DIR`
2. 仓库级 `models/runtime`
3. 兼容旧结构的 `program/engine/models`

这样可以同时兼容新仓库和旧脚本。
