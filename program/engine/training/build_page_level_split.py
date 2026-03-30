#!/usr/bin/env python3
"""Build deterministic page-level splits for single-project or mixed datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _page_split_key(row: dict[str, Any]) -> str:
    return f"{str(row.get('dataset_slug') or '').strip()}:{str(row.get('project_slug') or '').strip()}:{str(row.get('page_id') or '').strip()}"


def _stable_order_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _resolve_label(row: dict[str, Any], stratify_labels: dict[str, str] | None) -> str:
    if not stratify_labels:
        return "__default__"
    full_key = _page_split_key(row)
    short_key = f"{str(row.get('project_slug') or '').strip()}:{str(row.get('page_id') or '').strip()}"
    return str(stratify_labels.get(full_key) or stratify_labels.get(short_key) or "__default__")


def _resolve_target_counts(
    total: int,
    val_ratio: float,
    holdout_ratio: float,
    train_val_only: bool,
    target_counts: dict[str, int] | None,
) -> tuple[int, int, int]:
    if target_counts is not None:
        train_count = int(target_counts.get("train", 0))
        val_count = int(target_counts.get("val", 0))
        holdout_count = int(target_counts.get("holdout", 0))
        if train_val_only and holdout_count != 0:
            raise ValueError("train_val_only split cannot include holdout pages")
        if min(train_count, val_count, holdout_count) < 0:
            raise ValueError("target_counts cannot be negative")
        if train_count + val_count + holdout_count != total:
            raise ValueError("target_counts must sum to the number of manual pages")
        if train_count <= 0:
            raise ValueError("page-level split leaves no train pages")
        return train_count, val_count, holdout_count

    holdout_count = 0 if train_val_only else min(max(int(round(total * max(holdout_ratio, 0.0))), 1 if total >= 10 else 0), max(total - 2, 0))
    val_count = min(max(int(round(total * max(val_ratio, 0.0))), 1), max(total - holdout_count - 1, 1))
    train_count = total - val_count - holdout_count
    if train_count <= 0:
        raise ValueError("page-level split leaves no train pages")
    return train_count, val_count, holdout_count


def _allocate_counts(group_rows: dict[str, list[dict[str, Any]]], target_total: int) -> dict[str, int]:
    if target_total <= 0:
        return {label: 0 for label in group_rows}
    total_rows = sum(len(rows) for rows in group_rows.values())
    if target_total > total_rows:
        raise ValueError("target_total exceeds available rows")

    allocated = {label: 0 for label in group_rows}
    remainders: list[tuple[float, str]] = []
    for label, rows in group_rows.items():
        ideal = (len(rows) * target_total) / total_rows if total_rows else 0.0
        base = min(int(math.floor(ideal)), len(rows))
        allocated[label] = base
        remainders.append((ideal - base, label))

    remaining = target_total - sum(allocated.values())
    for _, label in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if allocated[label] >= len(group_rows[label]):
            continue
        allocated[label] += 1
        remaining -= 1

    if remaining > 0:
        for label in sorted(group_rows):
            while remaining > 0 and allocated[label] < len(group_rows[label]):
                allocated[label] += 1
                remaining -= 1

    return allocated


def build_page_level_split(
    rows: list[dict[str, Any]],
    dataset_slug: str,
    val_ratio: float = 0.1,
    holdout_ratio: float = 0.1,
    train_val_only: bool = False,
    target_counts: dict[str, int] | None = None,
    stratify_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    manual_rows = [row for row in rows if row.get("manual_quad")]
    ranked_rows = sorted(manual_rows, key=lambda row: _stable_order_key(_page_split_key(row)))
    total = len(ranked_rows)
    if total < 2:
        raise ValueError("page-level split requires at least 2 manual pages")

    train_count, val_count, holdout_count = _resolve_target_counts(
        total=total,
        val_ratio=val_ratio,
        holdout_ratio=holdout_ratio,
        train_val_only=train_val_only,
        target_counts=target_counts,
    )

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        grouped_rows[_resolve_label(row, stratify_labels)].append(row)

    holdout_allocations = _allocate_counts(grouped_rows, holdout_count)
    holdout_rows: list[dict[str, Any]] = []
    remaining_after_holdout: dict[str, list[dict[str, Any]]] = {}
    for label in sorted(grouped_rows):
        rows_for_label = grouped_rows[label]
        take = holdout_allocations[label]
        holdout_rows.extend(rows_for_label[:take])
        remaining_after_holdout[label] = rows_for_label[take:]

    val_allocations = _allocate_counts(remaining_after_holdout, val_count)
    val_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for label in sorted(remaining_after_holdout):
        rows_for_label = remaining_after_holdout[label]
        take = val_allocations[label]
        val_rows.extend(rows_for_label[:take])
        train_rows.extend(rows_for_label[take:])

    return {
        "dataset_slug": dataset_slug,
        "strategy": "deterministic_page_hash_stratified_v1" if stratify_labels or target_counts is not None else "deterministic_page_hash_v1",
        "train_projects": [],
        "val_projects": [],
        "holdout_projects": [],
        "train_page_ids": [_page_split_key(row) for row in train_rows],
        "val_page_ids": [_page_split_key(row) for row in val_rows],
        "holdout_page_ids": [_page_split_key(row) for row in holdout_rows],
        "metadata": {
            "mode": "train_val_only" if train_val_only else "train_val_holdout",
            "manual_pages": total,
            "train_pages": len(train_rows),
            "val_pages": len(val_rows),
            "holdout_pages": len(holdout_rows),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-rows", required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--holdout-ratio", type=float, default=0.1)
    parser.add_argument("--train-val-only", action="store_true")
    parser.add_argument("--target-train-count", type=int, default=None)
    parser.add_argument("--target-val-count", type=int, default=None)
    parser.add_argument("--target-holdout-count", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_jsonl(Path(args.curated_rows))
    split = build_page_level_split(
        rows,
        dataset_slug=args.dataset_slug,
        val_ratio=args.val_ratio,
        holdout_ratio=args.holdout_ratio,
        train_val_only=bool(args.train_val_only),
        target_counts=None
        if args.target_train_count is None and args.target_val_count is None and args.target_holdout_count is None
        else {
            "train": int(args.target_train_count or 0),
            "val": int(args.target_val_count or 0),
            "holdout": int(args.target_holdout_count or 0),
        },
    )
    _write_json(Path(args.output), split)
    print(json.dumps(split["metadata"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
