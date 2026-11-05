# README 截图

本目录的所有 `*.jpg` 都由脚本生成，不要手动替换：

```bash
make readme-screenshots
```

- 脚本：`scripts/docs/generate_readme_screenshots.py`
- 数据源：`data/curated/scenes/*/holdout.jsonl`（已查看回归页面，图片本身不进版本库）
- 模型：`models/runtime/screen_corner_single.pt`，可用 `SCREEN_PDF_SINGLE_MODEL` 覆盖
- 图注里的命中数与平均误差是脚本实测值，正式 runtime 换代后重跑即可同步

| 文件 | 内容 |
| --- | --- |
| `hero.jpg` | 会议室投影：现场照片（带检测四角）+ 矫正结果 |
| `accuracy.jpg` | 人工框 vs 模型输出，四角放大并画出 1% 容差圈 |
| `example-*.jpg` | 宽幅 LED、多屏、舞台、展台、遮挡、白底投影 六个场景 |
| `export-pdf.jpg` | 已导出 PDF 的第 11 页（含 OCR 文字层） |
