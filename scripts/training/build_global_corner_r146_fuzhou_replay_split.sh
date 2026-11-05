#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SOURCE_SPLIT_DIR="${SOURCE_SPLIT_DIR:-training/runs/global_corner_r144_202604_mall_curriculum_stage2_from_stage1_v1/split}"
OUTPUT_SPLIT_DIR="${OUTPUT_SPLIT_DIR:-training/runs/global_corner_r146_202604_r142_fuzhou_replay_mall_retention_v1/split}"

python - <<'PY'
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

repo = Path("/Users/gcssloop/WorkSpace/AIGC/screen-pdf")
source_dir = repo / "training/runs/global_corner_r144_202604_mall_curriculum_stage2_from_stage1_v1/split"
output_dir = repo / "training/runs/global_corner_r146_202604_r142_fuzhou_replay_mall_retention_v1/split"
output_dir.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


train_rows = read_jsonl(source_dir / "train.jsonl")
test_rows = read_jsonl(source_dir / "test.jsonl")
focus_train_rows = read_jsonl(source_dir / "focus_train.jsonl")
focus_test_rows = read_jsonl(source_dir / "focus_test.jsonl")
holdout_rows = read_jsonl(source_dir / "holdout.jsonl")

fuzhou_prefix = "/Users/gcssloop/WorkSpace/AIGC/展会/2026/202603 福州-第二十八届高速公路信息化大会/"
fuzhou_train_rows = [row for row in train_rows if str(row.get("image_path", "")).startswith(fuzhou_prefix)]
fuzhou_non_jinyi_rows = [row for row in fuzhou_train_rows if str(row.get("project_name", "")) != "2026-03-26 金溢大模型"]

merged_train_rows = list(train_rows) + list(fuzhou_train_rows)
merged_focus_train_rows = list(focus_train_rows) + list(fuzhou_non_jinyi_rows)

for name, rows in {
    "train.jsonl": merged_train_rows,
    "test.jsonl": test_rows,
    "focus_train.jsonl": merged_focus_train_rows,
    "focus_test.jsonl": focus_test_rows,
    "holdout.jsonl": holdout_rows,
}.items():
    write_jsonl(output_dir / name, rows)

summary = {
    "source_split_dir": str(source_dir),
    "train_rows": len(merged_train_rows),
    "test_rows": len(test_rows),
    "focus_train_rows": len(merged_focus_train_rows),
    "focus_test_rows": len(focus_test_rows),
    "holdout_rows": len(holdout_rows),
    "fuzhou_replay_rows": len(fuzhou_train_rows),
    "fuzhou_non_jinyi_focus_rows": len(fuzhou_non_jinyi_rows),
    "train_project_counts_top20": Counter(str(row.get("project_name", "")) for row in merged_train_rows).most_common(20),
    "focus_train_project_counts_top20": Counter(str(row.get("project_name", "")) for row in merged_focus_train_rows).most_common(20),
}
(output_dir / "summary.merged.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
