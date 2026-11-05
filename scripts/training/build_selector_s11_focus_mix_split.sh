#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SOURCE_SPLIT_DIR="${SOURCE_SPLIT_DIR:-training/runs/global_corner_r144_202604_mall_curriculum_stage2_from_stage1_v1/split}"
OUTPUT_SPLIT_DIR="${OUTPUT_SPLIT_DIR:-training/runs/selector_expand_s11_202604_r142c22r69_opencv_focusmix_v1/split}"

python - <<'PY'
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

repo = Path('/Users/gcssloop/WorkSpace/AIGC/screen-pdf')
source_dir = repo / 'training/runs/global_corner_r144_202604_mall_curriculum_stage2_from_stage1_v1/split'
output_dir = repo / 'training/runs/selector_expand_s11_202604_r142c22r69_opencv_focusmix_v1/split'
output_dir.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + ('\n' if rows else ''), encoding='utf-8')

train_rows = read_jsonl(source_dir / 'train.jsonl')
focus_train_rows = read_jsonl(source_dir / 'focus_train.jsonl')
mixed_train_rows = list(train_rows) + list(focus_train_rows) + list(focus_train_rows)

write_jsonl(output_dir / 'train.jsonl', mixed_train_rows)
summary = {
    'source_split_dir': str(source_dir),
    'train_rows': len(mixed_train_rows),
    'base_train_rows': len(train_rows),
    'focus_train_rows': len(focus_train_rows),
    'focus_repeat': 2,
    'train_project_counts_top20': Counter(str(row.get('project_name', '')) for row in mixed_train_rows).most_common(20),
}
(output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
