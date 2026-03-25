# Repository Migration And Bootstrap

## 背景

原仓库长期在单目录里同时堆放：

- 桌面程序
- Python 引擎
- 运行时模型
- 训练数据
- 历史实验
- 报告与计划文档
- 大量构建缓存

这会导致：

- 代理进入仓库后难以判断“该先读哪里”
- 程序和模型边界不清
- 训练资产和运行时代码混放
- 打包配置和运行时查模路径耦合过深

## 本次目标

1. 迁移到 `/Users/gcssloop/WorkSpace/AIGC/screen-pdf`
2. 程序与模型分离
3. 形成高信号文档入口
4. 保留对旧实验路径的兼容

## 已完成

- 新仓库目录已创建
- 桌面程序源码迁入 `program/desktop`
- Python 引擎迁入 `program/engine`
- 运行时模型迁入 `models/runtime`
- 历史实验资产迁入 `research/experiments`
- `docs/plans` 已建立兼容链接
- 运行时模型查找逻辑已改成优先读 `models/runtime`
- Tauri 打包资源路径已改到新结构
- README / AGENTS / 状态 / 架构文档已补齐

## 待完成

- 在新仓库跑一次最小化验证
- 如需继续发版，在新目录重新安装前端依赖并重新构建
- 视情况清理旧目录

## 最小验证建议

### Python

```bash
cd /Users/gcssloop/WorkSpace/AIGC/screen-pdf
PYTHONPATH=program/engine python -m unittest program/engine/test_detect_frame.py -v
```

### 前端 / 桌面壳

```bash
cd /Users/gcssloop/WorkSpace/AIGC/screen-pdf/program/desktop
pnpm install
pnpm tauri dev
```

## 迁移原则

- 先保证结构清晰，再做算法迭代。
- 运行时模型只保留一份权威目录。
- 历史训练资产全部保留，但不再让它淹没主入口。
