from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def summarize_project_errors(page_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in page_rows:
        grouped[str(row["project_name"])].append(float(row["page_error"]))
    summary: list[dict[str, Any]] = []
    for project_name, values in grouped.items():
        arr = np.array(values, dtype=np.float32)
        summary.append(
            {
                "project_name": project_name,
                "pages": int(arr.size),
                "mean_page_error": round(float(arr.mean()), 4),
                "p75_page_error": round(float(np.percentile(arr, 75)), 4),
                "fail_gt_0_02": int((arr > 0.02).sum()),
                "fail_gt_0_03": int((arr > 0.03).sum()),
                "hit_le_0_01": round(float((arr <= 0.01).mean()), 4),
            }
        )
    summary.sort(key=lambda item: (int(item["fail_gt_0_03"]), float(item["mean_page_error"])), reverse=True)
    return summary


def compute_targeted_adaptive_weight(
    row: dict[str, Any],
    page_error: float,
    corner_error: float,
    focus_projects: set[str],
    max_weight: float = 2.35,
) -> float:
    weight = float(max(row.get("adaptive_weight", 1.0), 0.1))
    project_name = str(row.get("project_name", ""))
    corner_index = int(row.get("corner_index", 0))
    residual_norm = np.array(row.get("target_residual_norm", [0.0, 0.0]), dtype=np.float32)

    if page_error > 0.08:
        weight *= 1.28
    elif page_error > 0.05:
        weight *= 1.22
    elif page_error > 0.03:
        weight *= 1.15
    elif page_error > 0.02:
        weight *= 1.08

    if corner_error > 0.10:
        weight *= 1.25
    elif corner_error > 0.07:
        weight *= 1.18
    elif corner_error > 0.04:
        weight *= 1.10
    elif corner_error > 0.025:
        weight *= 1.05

    if project_name in focus_projects and page_error > 0.02:
        weight *= 1.10

    if corner_index in {2, 3} and corner_error > 0.025:
        weight *= 1.08
        if abs(float(residual_norm[1])) > 0.12:
            weight *= 1.05
    elif corner_index == 1 and page_error > 0.04:
        weight *= 1.04

    return round(float(np.clip(weight, 1.0, max_weight)), 4)
