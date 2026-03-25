from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from dataset_benchmark import (
    build_project_aware_split,
    load_manual_pages,
    quad_geometry_metrics,
    summarize_geometry_metric_rows,
)
from detect_frame import run_model_detection
from project_corner_benchmark import benchmark_project
from two_stage_corner_pipeline import GlobalCornerPredictor, LocalCornerMoEPredictor, RoiCornerPredictor, predict_two_stage


def build_runtime_runner(
    global_model: Path | None,
    roi_model: Path | None,
    local_model: Path | None,
):
    if global_model is None or roi_model is None:
        return run_model_detection

    global_predictor = GlobalCornerPredictor(global_model)
    roi_predictor = RoiCornerPredictor(roi_model)
    local_predictor = LocalCornerMoEPredictor(local_model) if local_model is not None else None

    def _runner(image_path: str, image: Any | None = None) -> dict[str, Any] | None:
        result = predict_two_stage(
            image_path=Path(image_path),
            global_predictor=global_predictor,
            roi_predictor=roi_predictor,
            local_predictor=local_predictor,
            page_id=Path(image_path).stem,
            image=image,
        )
        return {
            "quad": result["final_quad"],
            "method": "custom_model_three_stage" if local_predictor is not None else "custom_model_two_stage",
        }

    return _runner


def infer_runtime_rows(pages: list[dict[str, Any]], runtime_runner) -> dict[str, dict[str, Any]]:
    inferred: dict[str, dict[str, Any]] = {}
    for page in pages:
        image_path = str(page["image_path"])
        image = cv2.imread(image_path)
        if image is None:
            continue
        started = time.perf_counter()
        result = runtime_runner(image_path, image=image)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result is None or result.get("quad") is None:
            metrics = {
                "point_error": 1.0,
                "screen_relative_point_error": 1.0,
                "max_corner_error": 1.0,
                "perspective_tilt_error": 90.0,
                "quad_inset_ratio": 1.0,
            }
            method = "none"
        else:
            metrics = quad_geometry_metrics(page["manual_quad"], result["quad"])
            method = str(result.get("method", "runtime_model"))
        inferred[image_path] = {
            "project_name": page.get("project_name"),
            "page_id": page.get("page_id"),
            "image_path": image_path,
            "method": method,
            "elapsed_ms": elapsed_ms,
            "metrics": metrics,
        }
    return inferred


def evaluate_runtime_pages(
    pages: list[dict[str, Any]],
    inferred_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metric_rows: list[dict[str, float]] = []
    elapsed_ms: list[float] = []
    method_counts: Counter[str] = Counter()
    worst_rows: list[dict[str, Any]] = []

    for page in pages:
        cached = inferred_rows.get(str(page["image_path"]))
        if cached is None:
            continue
        metrics = cached["metrics"]
        method = str(cached["method"])
        metric_rows.append(metrics)
        elapsed_ms.append(float(cached["elapsed_ms"]))
        method_counts[method] += 1
        worst_rows.append(
            {
                "project_name": cached.get("project_name"),
                "page_id": cached.get("page_id"),
                "image_path": cached.get("image_path"),
                "method": method,
                "point_error": round(float(metrics["point_error"]), 4),
                "screen_relative_error": round(float(metrics["screen_relative_point_error"]), 4),
                "max_corner_error": round(float(metrics["max_corner_error"]), 4),
            }
        )

    summary = summarize_geometry_metric_rows(metric_rows)
    summary["avg_page_infer_ms"] = round(sum(elapsed_ms) / len(elapsed_ms), 2) if elapsed_ms else 0.0
    summary["p95_page_infer_ms"] = round(sorted(elapsed_ms)[int(max(len(elapsed_ms) * 0.95 - 1, 0))], 2) if elapsed_ms else 0.0
    summary["method_counts"] = dict(sorted(method_counts.items()))
    return {
        "summary": summary,
        "worst_pages": sorted(
            worst_rows,
            key=lambda row: (float(row["point_error"]), float(row["max_corner_error"])),
            reverse=True,
        )[:10],
    }


def _floor_check(summary: dict[str, Any], floor: float) -> dict[str, Any]:
    ratio = float(summary.get("point_le_0_01_ratio", 0.0))
    return {
        "floor": floor,
        "ratio": round(ratio, 4),
        "passed": ratio >= floor,
    }


def _holdout_section(project_path: Path, runtime_runner) -> dict[str, Any]:
    result = benchmark_project(project_path, runtime_runner=runtime_runner)
    runtime_summary = (result.get("summaries") or {}).get("runtime_model", {})
    runtime_worst = (result.get("worst_pages") or {}).get("runtime_model", [])
    return {
        "project_path": str(project_path),
        "project_name": result.get("project_name"),
        "pages": result.get("pages", 0),
        "summary": runtime_summary,
        "worst_pages": runtime_worst,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate balanced generalization with project-aware split")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--focus-project", action="append", dest="focus_projects", default=[])
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--focus-test-ratio", type=float, default=0.25)
    parser.add_argument("--holdout-project", action="append", dest="holdout_projects", default=[])
    parser.add_argument("--floor", type=float, default=0.80)
    parser.add_argument("--global-model")
    parser.add_argument("--roi-model")
    parser.add_argument("--local-model")
    parser.add_argument("--output")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    pages = load_manual_pages(dataset_root)
    runtime_runner = build_runtime_runner(
        Path(args.global_model) if args.global_model else None,
        Path(args.roi_model) if args.roi_model else None,
        Path(args.local_model) if args.local_model else None,
    )
    split = build_project_aware_split(
        pages,
        focus_projects=list(args.focus_projects),
        test_ratio=args.test_ratio,
        focus_test_ratio=args.focus_test_ratio,
        seed=7,
    )
    inferred_rows = infer_runtime_rows(pages, runtime_runner)

    focus_paths = {str(item) for item in args.focus_projects}
    broad_non_focus_pages = [
        page for page in pages if str(page.get("project_path")) not in focus_paths and str(page.get("project_name")) not in focus_paths
    ]

    report = {
        "dataset_root": str(dataset_root),
        "pages": len(pages),
        "metadata": split.get("metadata", {}),
        "broad_all": evaluate_runtime_pages(pages, inferred_rows),
        "broad_non_focus": evaluate_runtime_pages(broad_non_focus_pages, inferred_rows),
        "train": evaluate_runtime_pages(split["train"], inferred_rows),
        "test": evaluate_runtime_pages(split["test"], inferred_rows),
        "focus_train": evaluate_runtime_pages(split["focus_train"], inferred_rows),
        "focus_test": evaluate_runtime_pages(split["focus_test"], inferred_rows),
        "holdouts": {},
    }
    for raw_path in args.holdout_projects:
        holdout_path = Path(raw_path)
        report["holdouts"][holdout_path.parent.name] = _holdout_section(holdout_path, runtime_runner)

    report["guardrails"] = {
        "broad_all": _floor_check(report["broad_all"]["summary"], args.floor),
        "broad_non_focus": _floor_check(report["broad_non_focus"]["summary"], args.floor),
        "focus_test": _floor_check(report["focus_test"]["summary"], args.floor),
        "holdouts": {
            name: _floor_check(section["summary"], args.floor)
            for name, section in report["holdouts"].items()
        },
    }

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
