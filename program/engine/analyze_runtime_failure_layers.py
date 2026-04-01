from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from dataset_benchmark import _resolve_image_path, quad_geometry_metrics, summarize_geometry_metric_rows
from detect_frame import (
    TEACHER_CANDIDATE_BASELINE_GATE,
    TEACHER_CANDIDATE_MIN_SCORE_GAIN,
    TEACHER_MULTI_EXPAND_RATIOS,
)
from perspective_detect import detect_best_candidate
from supervision_utils import resolve_supervision_quad
from two_stage_corner_pipeline import (
    GlobalCornerPredictor,
    LocalCornerMoEPredictor,
    RoiCornerPredictor,
    _run_two_stage_candidate,
    predict_two_stage,
)


STRICT_POINT_THRESHOLD = 0.01
STRICT_MAX_CORNER_THRESHOLD = 0.03
_DEFAULT_METRIC_ROW = {
    "point_error": 1.0,
    "corner_error_00": 1.0,
    "corner_error_01": 1.0,
    "corner_error_02": 1.0,
    "corner_error_03": 1.0,
    "screen_relative_point_error": 1.0,
    "max_corner_error": 1.0,
    "perspective_tilt_error": 90.0,
    "quad_inset_ratio": 1.0,
}


def _passes_strict(metrics: dict[str, float], *, point_threshold: float, max_corner_threshold: float) -> bool:
    return float(metrics.get("point_error", 1.0) or 1.0) <= float(point_threshold) and float(
        metrics.get("max_corner_error", 1.0) or 1.0
    ) <= float(max_corner_threshold)


def _metric_row(metrics: dict[str, float]) -> dict[str, float]:
    merged = dict(_DEFAULT_METRIC_ROW)
    merged.update({str(key): float(value) for key, value in metrics.items()})
    return merged


def _oracle_rank_key(metrics: dict[str, float]) -> tuple[int, int, float, float]:
    point_error = float(metrics.get("point_error", 1.0) or 1.0)
    max_corner_error = float(metrics.get("max_corner_error", 1.0) or 1.0)
    return (
        int(max_corner_error <= STRICT_MAX_CORNER_THRESHOLD),
        int(point_error <= STRICT_POINT_THRESHOLD),
        -point_error,
        -max_corner_error,
    )


def select_best_metric_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records is empty")
    return max(records, key=lambda item: _oracle_rank_key(item["metrics"]))


def classify_failure_layer(
    *,
    baseline_metrics: dict[str, float],
    runtime_oracle_metrics: dict[str, float],
    opencv_oracle_metrics: dict[str, float],
    point_threshold: float = STRICT_POINT_THRESHOLD,
    max_corner_threshold: float = STRICT_MAX_CORNER_THRESHOLD,
) -> str:
    if _passes_strict(baseline_metrics, point_threshold=point_threshold, max_corner_threshold=max_corner_threshold):
        return "baseline_ok"
    if _passes_strict(runtime_oracle_metrics, point_threshold=point_threshold, max_corner_threshold=max_corner_threshold):
        return "runtime_candidate_recoverable"
    if _passes_strict(opencv_oracle_metrics, point_threshold=point_threshold, max_corner_threshold=max_corner_threshold):
        return "opencv_recoverable"
    return "hard_both_fail"


def summarize_diagnostic_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(str(row["category"]) for row in rows)
    baseline_metrics = [_metric_row(row["baseline_metrics"]) for row in rows]
    runtime_oracle_metrics = [_metric_row(row["runtime_oracle_metrics"]) for row in rows]
    opencv_oracle_metrics = [_metric_row(row["opencv_oracle_metrics"]) for row in rows]
    union_oracle_metrics = [_metric_row(row["union_oracle_metrics"]) for row in rows]
    total = max(len(rows), 1)
    return {
        "pages": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "baseline_summary": summarize_geometry_metric_rows(baseline_metrics),
        "runtime_oracle_summary": summarize_geometry_metric_rows(runtime_oracle_metrics),
        "opencv_oracle_summary": summarize_geometry_metric_rows(opencv_oracle_metrics),
        "union_oracle_summary": summarize_geometry_metric_rows(union_oracle_metrics),
        "baseline_strict_ratio": round(category_counts.get("baseline_ok", 0) / total, 4),
        "runtime_oracle_strict_ratio": round(
            sum(
                _passes_strict(metrics, point_threshold=STRICT_POINT_THRESHOLD, max_corner_threshold=STRICT_MAX_CORNER_THRESHOLD)
                for metrics in runtime_oracle_metrics
            )
            / total,
            4,
        ),
        "opencv_oracle_strict_ratio": round(
            sum(
                _passes_strict(metrics, point_threshold=STRICT_POINT_THRESHOLD, max_corner_threshold=STRICT_MAX_CORNER_THRESHOLD)
                for metrics in opencv_oracle_metrics
            )
            / total,
            4,
        ),
        "union_oracle_strict_ratio": round(
            sum(
                _passes_strict(metrics, point_threshold=STRICT_POINT_THRESHOLD, max_corner_threshold=STRICT_MAX_CORNER_THRESHOLD)
                for metrics in union_oracle_metrics
            )
            / total,
            4,
        ),
    }


def _iter_manual_pages(dataset_root: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for project_path in sorted(dataset_root.rglob("screen-pdf-project.json")):
        data = json.loads(project_path.read_text(encoding="utf-8"))
        for page in data.get("pages", []):
            manual_quad, _ = resolve_supervision_quad(page)
            if not manual_quad:
                continue
            image_path = _resolve_image_path(project_path, page)
            if image_path is None:
                continue
            pages.append(
                {
                    "project_name": project_path.parent.name,
                    "project_path": str(project_path),
                    "page_id": str(page.get("id") or image_path.stem),
                    "image_path": str(image_path),
                    "manual_quad": manual_quad,
                }
            )
    return pages


def _opencv_oracle_records(image: Any, manual_quad: list[list[float]]) -> list[dict[str, Any]]:
    result = detect_best_candidate(image)
    records: list[dict[str, Any]] = []
    for candidate in (result or {}).get("candidates") or []:
        quad = candidate.get("quad")
        if quad is None:
            continue
        records.append(
            {
                "source": str(candidate.get("method", "opencv")),
                "metrics": quad_geometry_metrics(manual_quad, quad),
                "quad": quad,
            }
        )
    return records


def _runtime_candidate_records(
    *,
    image_path: Path,
    image: Any,
    manual_quad: list[list[float]],
    global_predictor: GlobalCornerPredictor,
    roi_predictor: RoiCornerPredictor,
    local_predictor: LocalCornerMoEPredictor | None,
    candidate_expand_ratios: list[float],
) -> list[dict[str, Any]]:
    coarse_quad = global_predictor.predict_image(image)
    records: list[dict[str, Any]] = []
    for expand_ratio in candidate_expand_ratios:
        candidate = _run_two_stage_candidate(
            image_path=image_path,
            page_id=image_path.stem,
            image=image,
            coarse_quad=coarse_quad,
            roi_predictor=roi_predictor,
            local_predictor=local_predictor,
            expand_ratio=float(expand_ratio),
        )
        records.append(
            {
                "source": f"expand_{expand_ratio:.2f}",
                "expand_ratio": float(expand_ratio),
                "metrics": quad_geometry_metrics(manual_quad, candidate["final_quad"]),
                "quad": candidate["final_quad"],
            }
        )
    return records


def analyze_dataset(
    *,
    dataset_root: Path,
    global_model: Path,
    roi_model: Path,
    local_model: Path | None,
    candidate_expand_ratios: list[float] | None = None,
    candidate_baseline_gate: float = TEACHER_CANDIDATE_BASELINE_GATE,
    candidate_min_score_gain: float = TEACHER_CANDIDATE_MIN_SCORE_GAIN,
) -> dict[str, Any]:
    del candidate_baseline_gate, candidate_min_score_gain
    expand_ratios = list(candidate_expand_ratios or TEACHER_MULTI_EXPAND_RATIOS)
    baseline_expand_ratio = 0.08
    global_predictor = GlobalCornerPredictor(global_model)
    roi_predictor = RoiCornerPredictor(roi_model)
    local_predictor = LocalCornerMoEPredictor(local_model) if local_model is not None else None

    rows: list[dict[str, Any]] = []
    for page in _iter_manual_pages(dataset_root):
        image_path = Path(page["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        baseline_result = predict_two_stage(
            image_path=image_path,
            image=image,
            global_predictor=global_predictor,
            roi_predictor=roi_predictor,
            local_predictor=local_predictor,
            page_id=str(page["page_id"]),
            candidate_expand_ratios=expand_ratios,
        )
        baseline_metrics = quad_geometry_metrics(page["manual_quad"], baseline_result["final_quad"])
        runtime_records = _runtime_candidate_records(
            image_path=image_path,
            image=image,
            manual_quad=page["manual_quad"],
            global_predictor=global_predictor,
            roi_predictor=roi_predictor,
            local_predictor=local_predictor,
            candidate_expand_ratios=expand_ratios,
        )
        opencv_records = _opencv_oracle_records(image, page["manual_quad"])
        runtime_oracle = select_best_metric_record(runtime_records) if runtime_records else {"source": "none", "metrics": {"point_error": 1.0, "max_corner_error": 1.0}}
        opencv_oracle = select_best_metric_record(opencv_records) if opencv_records else {"source": "none", "metrics": {"point_error": 1.0, "max_corner_error": 1.0}}
        union_oracle = select_best_metric_record([runtime_oracle, opencv_oracle])
        category = classify_failure_layer(
            baseline_metrics=baseline_metrics,
            runtime_oracle_metrics=runtime_oracle["metrics"],
            opencv_oracle_metrics=opencv_oracle["metrics"],
        )
        rows.append(
            {
                "project_name": page["project_name"],
                "page_id": page["page_id"],
                "image_path": page["image_path"],
                "category": category,
                "baseline_selected_expand_ratio": float(baseline_result["selected_expand_ratio"]),
                "runtime_candidate_count": len(runtime_records),
                "opencv_candidate_count": len(opencv_records),
                "baseline_metrics": baseline_metrics,
                "runtime_oracle_metrics": runtime_oracle["metrics"],
                "opencv_oracle_metrics": opencv_oracle["metrics"],
                "union_oracle_metrics": union_oracle["metrics"],
                "runtime_oracle_source": str(runtime_oracle["source"]),
                "opencv_oracle_source": str(opencv_oracle["source"]),
                "union_oracle_source": str(union_oracle["source"]),
            }
        )
    summary = summarize_diagnostic_rows(rows)
    by_project: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_project.setdefault(str(row["project_name"]), []).append(row)
    project_summaries = {
        project_name: summarize_diagnostic_rows(project_rows)
        for project_name, project_rows in sorted(by_project.items())
    }
    return {
        "dataset_root": str(dataset_root),
        "pages": len(rows),
        "summary": summary,
        "project_summaries": project_summaries,
        "hard_examples": sorted(
            [row for row in rows if row["category"] == "hard_both_fail"],
            key=lambda item: (
                float(item["union_oracle_metrics"]["point_error"]),
                float(item["union_oracle_metrics"]["max_corner_error"]),
            ),
            reverse=True,
        )[:20],
        "selector_examples": sorted(
            [row for row in rows if row["category"] == "runtime_candidate_recoverable"],
            key=lambda item: float(item["baseline_metrics"]["point_error"]) - float(item["runtime_oracle_metrics"]["point_error"]),
            reverse=True,
        )[:20],
        "opencv_examples": sorted(
            [row for row in rows if row["category"] == "opencv_recoverable"],
            key=lambda item: float(item["baseline_metrics"]["point_error"]) - float(item["opencv_oracle_metrics"]["point_error"]),
            reverse=True,
        )[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze runtime failure layers by page")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--global-model", default="models/runtime/global_corner_model.pt")
    parser.add_argument("--roi-model", default="models/runtime/corner_heatmap_model.pt")
    parser.add_argument("--local-model", default="models/runtime/local_corner_moe_coord_model.pt")
    parser.add_argument("--output")
    parser.add_argument("--candidate-expand-ratios", default="0.02,0.04,0.06,0.08,0.10,0.12")
    args = parser.parse_args()
    candidate_expand_ratios = [float(part.strip()) for part in str(args.candidate_expand_ratios).split(",") if part.strip()]
    report = analyze_dataset(
        dataset_root=Path(args.dataset_root),
        global_model=Path(args.global_model),
        roi_model=Path(args.roi_model),
        local_model=Path(args.local_model) if str(args.local_model).strip() else None,
        candidate_expand_ratios=candidate_expand_ratios,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
