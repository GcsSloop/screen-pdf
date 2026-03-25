from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from dataset_benchmark import quad_geometry_metrics
from two_stage_corner_pipeline import GlobalCornerPredictor


def compute_bootstrap_hardness(
    metrics: dict[str, float],
    *,
    screen_threshold: float = 0.010,
    max_corner_threshold: float = 0.020,
    inset_threshold: float = 0.030,
) -> float:
    screen = float(metrics["screen_relative_point_error"]) / max(screen_threshold, 1e-6)
    max_corner = float(metrics["max_corner_error"]) / max(max_corner_threshold, 1e-6)
    inset = abs(float(metrics["quad_inset_ratio"])) / max(inset_threshold, 1e-6)
    return max(screen, max_corner, inset)


def continuous_adaptive_weight(hardness: float, *, max_boost: float = 2.5) -> float:
    return 1.0 + min(max_boost, max(0.0, float(hardness) - 1.0))


def bucketed_adaptive_weight(
    hardness: float,
    *,
    bucket_thresholds: tuple[float, ...] = (1.15, 1.6, 2.2, 3.2),
    bucket_weights: tuple[float, ...] = (1.0, 1.2, 1.45, 1.75, 2.05),
) -> float:
    if len(bucket_weights) != len(bucket_thresholds) + 1:
        raise ValueError("bucket_weights must be exactly one longer than bucket_thresholds")
    value = float(hardness)
    for threshold, weight in zip(bucket_thresholds, bucket_weights[:-1], strict=True):
        if value <= threshold:
            return float(weight)
    return float(bucket_weights[-1])


def rebalance_project_mean_weights(rows: list[dict[str, Any]], max_project_mean: float | None = None) -> list[dict[str, Any]]:
    if max_project_mean is None:
        return rows
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["project_name"])].append(row)
    for project_rows in grouped.values():
        current_mean = float(np.mean([float(item["adaptive_weight"]) for item in project_rows]))
        if current_mean <= max_project_mean:
            continue
        adjustable = [max(float(item["adaptive_weight"]) - 1.0, 0.0) for item in project_rows]
        total_adjustable = float(sum(adjustable))
        if total_adjustable <= 1e-6:
            continue
        target_extra = (max_project_mean - 1.0) * len(project_rows)
        scale = max(target_extra / total_adjustable, 0.0)
        for item, extra in zip(project_rows, adjustable, strict=True):
            item["adaptive_weight"] = round(1.0 + extra * scale, 6)
    return rows


def summarize_weighted_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = np.array([float(row["adaptive_weight"]) for row in rows], dtype=np.float32)
    top_rows = sorted(
        rows,
        key=lambda row: (float(row["adaptive_weight"]), float(row["bootstrap_metrics"]["screen_relative_point_error"])),
        reverse=True,
    )[:12]
    by_project: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_project[str(row["project_name"])].append(float(row["adaptive_weight"]))
    projects = [
        {
            "project_name": name,
            "pages": len(project_weights),
            "mean_weight": round(float(np.mean(project_weights)), 4),
            "max_weight": round(float(np.max(project_weights)), 4),
            "gt1_5_count": int(sum(weight >= 1.5 for weight in project_weights)),
            "gt2_count": int(sum(weight >= 2.0 for weight in project_weights)),
        }
        for name, project_weights in by_project.items()
    ]
    projects.sort(key=lambda item: (item["mean_weight"], item["max_weight"]), reverse=True)
    return {
        "rows": len(rows),
        "adaptive_weight_mean": round(float(weights.mean()), 4),
        "adaptive_weight_p95": round(float(np.percentile(weights, 95)), 4),
        "adaptive_weight_max": round(float(weights.max()), 4),
        "top12": [
            {
                "page_id": row["page_id"],
                "project_name": row["project_name"],
                "adaptive_weight": round(float(row["adaptive_weight"]), 4),
                "hardness_score": round(float(row["hardness_score"]), 4),
                "screen_relative_error": round(float(row["bootstrap_metrics"]["screen_relative_point_error"]), 4),
                "max_corner_error": round(float(row["bootstrap_metrics"]["max_corner_error"]), 4),
                "quad_inset_abs": round(abs(float(row["bootstrap_metrics"]["quad_inset_ratio"])), 4),
            }
            for row in top_rows
        ],
        "top_projects": projects[:15],
    }


def annotate_manifest(
    manifest_path: Path,
    model_path: Path,
    *,
    mode: str = "bucketed",
    project_mean_cap: float | None = None,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictor = GlobalCornerPredictor(model_path)
    weighted_rows: list[dict[str, Any]] = []
    for row in rows:
        pred_quad = predictor(Path(row["image_path"]))
        metrics = quad_geometry_metrics(row["manual_quad"], pred_quad)
        hardness = compute_bootstrap_hardness(metrics)
        if mode == "continuous":
            adaptive_weight = continuous_adaptive_weight(hardness)
        elif mode == "bucketed":
            adaptive_weight = bucketed_adaptive_weight(hardness)
        else:
            raise ValueError(f"unsupported mode: {mode}")
        updated = dict(row)
        updated["bootstrap_metrics"] = {key: round(float(value), 6) for key, value in metrics.items()}
        updated["hardness_score"] = round(float(hardness), 6)
        updated["adaptive_weight"] = round(float(adaptive_weight), 6)
        weighted_rows.append(updated)
    rebalance_project_mean_weights(weighted_rows, max_project_mean=project_mean_cap)
    manifest_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in weighted_rows) + "\n", encoding="utf-8")
    return summarize_weighted_rows(weighted_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate global-corner training manifest with bootstrap weights")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", default="bucketed", choices=["continuous", "bucketed"])
    parser.add_argument("--project-mean-cap", type=float)
    parser.add_argument("--summary-out")
    args = parser.parse_args()

    summary = annotate_manifest(
        Path(args.manifest),
        Path(args.model),
        mode=str(args.mode),
        project_mean_cap=float(args.project_mean_cap) if args.project_mean_cap is not None else None,
    )
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
