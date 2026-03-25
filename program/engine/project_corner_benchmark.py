from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2

from dataset_benchmark import quad_geometry_metrics, summarize_geometry_metric_rows
from detect_frame import run_model_detection


MethodRunner = Callable[[str], dict[str, Any] | None]


def _summary_from_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return summarize_geometry_metric_rows(rows)


def _candidate_metrics(page: dict[str, Any], method: str) -> list[dict[str, float]]:
    manual_quad = page.get("manualQuad")
    if not manual_quad:
        return []
    metrics_list: list[dict[str, float]] = []
    for candidate in page.get("candidates") or []:
        quad = candidate.get("quad")
        if candidate.get("method") != method or not quad:
            continue
        metrics_list.append(quad_geometry_metrics(manual_quad, quad))
    return metrics_list


def benchmark_project(
    project_path: Path,
    candidate_methods: list[str] | None = None,
    runtime_runner: MethodRunner | None = None,
) -> dict[str, Any]:
    data = json.loads(project_path.read_text(encoding="utf-8"))
    candidate_methods = candidate_methods or []
    runtime_runner = runtime_runner or run_model_detection
    method_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    pages = 0
    for page in data.get("pages", []):
        manual_quad = page.get("manualQuad")
        image_path = page.get("path")
        if not manual_quad or not image_path:
            continue
        pages += 1
        page_id = str(page.get("id") or Path(image_path).stem)

        active_quad = page.get("activeQuad")
        if active_quad:
            metrics = quad_geometry_metrics(manual_quad, active_quad)
            method_metrics["active"].append(metrics)
            method_rows["active"].append(
                {
                    "page_id": page_id,
                    "image_path": image_path,
                    "method": page.get("bestMethod") or "active",
                    "point_error": round(float(metrics["point_error"]), 4),
                    "screen_relative_error": round(float(metrics["screen_relative_point_error"]), 4),
                    "max_corner_error": round(float(metrics["max_corner_error"]), 4),
                    "perspective_tilt_error": round(float(metrics["perspective_tilt_error"]), 4),
                    "quad_inset_ratio": round(float(metrics["quad_inset_ratio"]), 4),
                }
            )

        for method in candidate_methods:
            metrics_list = _candidate_metrics(page, method)
            if not metrics_list:
                continue
            best_metrics = min(metrics_list, key=lambda item: (float(item["screen_relative_point_error"]), float(item["max_corner_error"])))
            method_metrics[method].append(best_metrics)
            method_rows[method].append(
                {
                    "page_id": page_id,
                    "image_path": image_path,
                    "method": method,
                    "point_error": round(float(best_metrics["point_error"]), 4),
                    "screen_relative_error": round(float(best_metrics["screen_relative_point_error"]), 4),
                    "max_corner_error": round(float(best_metrics["max_corner_error"]), 4),
                    "perspective_tilt_error": round(float(best_metrics["perspective_tilt_error"]), 4),
                    "quad_inset_ratio": round(float(best_metrics["quad_inset_ratio"]), 4),
                }
            )

        runtime_result = runtime_runner(str(image_path)) if runtime_runner is not None else None
        if runtime_result is not None and runtime_result.get("quad") is not None:
            metrics = quad_geometry_metrics(manual_quad, runtime_result["quad"])
            method_metrics["runtime_model"].append(metrics)
            method_rows["runtime_model"].append(
                {
                    "page_id": page_id,
                    "image_path": image_path,
                    "method": runtime_result.get("method", "runtime_model"),
                    "point_error": round(float(metrics["point_error"]), 4),
                    "screen_relative_error": round(float(metrics["screen_relative_point_error"]), 4),
                    "max_corner_error": round(float(metrics["max_corner_error"]), 4),
                    "perspective_tilt_error": round(float(metrics["perspective_tilt_error"]), 4),
                    "quad_inset_ratio": round(float(metrics["quad_inset_ratio"]), 4),
                }
            )

    summaries = {method: _summary_from_metrics(rows) for method, rows in sorted(method_metrics.items())}
    worst_pages = {
        method: sorted(rows, key=lambda item: float(item["point_error"]), reverse=True)[:10]
        for method, rows in sorted(method_rows.items())
    }
    return {
        "project_path": str(project_path),
        "project_name": project_path.parent.name,
        "pages": pages,
        "summaries": summaries,
        "worst_pages": worst_pages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark one screen-pdf project by candidate method")
    parser.add_argument("--project-path", required=True)
    parser.add_argument(
        "--candidate-method",
        action="append",
        dest="candidate_methods",
        default=[],
        help="Candidate method to summarize. Repeat to compare multiple methods.",
    )
    parser.add_argument("--disable-runtime-model", action="store_true")
    args = parser.parse_args()
    runtime_runner = None if args.disable_runtime_model else run_model_detection
    result = benchmark_project(
        Path(args.project_path),
        candidate_methods=list(args.candidate_methods),
        runtime_runner=runtime_runner,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
