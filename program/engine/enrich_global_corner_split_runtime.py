from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from perspective_detect import detect_best_candidate
    from two_stage_corner_pipeline import GlobalCornerPredictor, LocalCornerMoEPredictor, RoiCornerPredictor, predict_two_stage
except ModuleNotFoundError:
    from engine.perspective_detect import detect_best_candidate
    from engine.two_stage_corner_pipeline import (
        GlobalCornerPredictor,
        LocalCornerMoEPredictor,
        RoiCornerPredictor,
        predict_two_stage,
    )


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(_json_ready(row), ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _strict_ok(metrics: dict[str, Any] | None, threshold: float = 0.03) -> bool:
    if not isinstance(metrics, dict):
        return False
    return float(metrics.get("max_corner_error", 1.0) or 1.0) <= float(threshold)


def load_failure_layer_index(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for key in ("runtime_examples", "opencv_examples", "hard_examples"):
        rows.extend(payload.get(key) or [])
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        page_id = str(row.get("page_id") or "").strip()
        image_path = str(row.get("image_path") or "").strip()
        if not page_id or not image_path:
            continue
        index[(page_id, image_path)] = row
    return index


def merge_failure_layer_fields(row: dict[str, Any], diagnostic: dict[str, Any] | None) -> dict[str, Any]:
    if diagnostic is None:
        return row
    merged = dict(row)
    baseline_metrics = diagnostic.get("baseline_metrics") or {}
    runtime_metrics = diagnostic.get("runtime_oracle_metrics") or {}
    opencv_metrics = diagnostic.get("opencv_oracle_metrics") or {}
    union_metrics = diagnostic.get("union_oracle_metrics") or {}
    baseline_error = float(baseline_metrics.get("max_corner_error", 1.0) or 1.0)
    runtime_error = float(runtime_metrics.get("max_corner_error", baseline_error) or baseline_error)
    opencv_error = float(opencv_metrics.get("max_corner_error", baseline_error) or baseline_error)
    union_error = float(union_metrics.get("max_corner_error", baseline_error) or baseline_error)
    merged.update(
        {
            "failure_layer_category": str(diagnostic.get("category") or "").strip(),
            "failure_layer_baseline_strict_ok": _strict_ok(baseline_metrics),
            "failure_layer_runtime_strict_ok": _strict_ok(runtime_metrics),
            "failure_layer_opencv_strict_ok": _strict_ok(opencv_metrics),
            "failure_layer_union_strict_ok": _strict_ok(union_metrics),
            "failure_layer_runtime_gain": max(0.0, baseline_error - runtime_error),
            "failure_layer_opencv_gain": max(0.0, baseline_error - opencv_error),
            "failure_layer_union_gain": max(0.0, baseline_error - union_error),
        }
    )
    return merged


def _round_quad(quad: np.ndarray | list[list[float]]) -> list[list[float]]:
    ordered = np.array(quad, dtype=np.float32)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in ordered]


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def enrich_row(
    row: dict[str, Any],
    *,
    global_predictor: GlobalCornerPredictor,
    roi_predictor: RoiCornerPredictor,
    local_predictor: LocalCornerMoEPredictor | None,
    candidate_expand_ratios: list[float] | None,
    candidate_baseline_gate: float,
    candidate_min_score_gain: float,
    failure_layer_index: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    image_path = Path(row["image_path"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    coarse_quad = global_predictor.predict_image(image)
    runtime_result = predict_two_stage(
        image_path=image_path,
        image=image,
        global_predictor=global_predictor,
        roi_predictor=roi_predictor,
        local_predictor=local_predictor,
        page_id=str(row.get("page_id") or image_path.stem),
        candidate_expand_ratios=candidate_expand_ratios,
        candidate_baseline_gate=float(candidate_baseline_gate),
        candidate_min_score_gain=float(candidate_min_score_gain),
    )
    opencv_result = detect_best_candidate(image)
    enriched = dict(row)
    enriched["teacher_r3_quad"] = _round_quad(coarse_quad)
    enriched["teacher_roi_quad"] = runtime_result["roi_quad"]
    enriched["teacher_quad"] = runtime_result["final_quad"]
    enriched["teacher_selected_expand_ratio"] = float(runtime_result["selected_expand_ratio"])
    enriched["teacher_candidate_count"] = int(runtime_result["candidate_count"])
    if opencv_result is not None and opencv_result.get("best") is not None:
        best = opencv_result["best"]
        enriched["opencv_best_quad"] = best["quad"]
        enriched["opencv_best_score"] = float(best.get("score", 0.0) or 0.0)
        enriched["opencv_best_method"] = str(best.get("method", ""))
    diagnostic = None
    if failure_layer_index:
        key = (str(row.get("page_id") or "").strip(), str(row.get("image_path") or "").strip())
        diagnostic = failure_layer_index.get(key)
    return merge_failure_layer_fields(enriched, diagnostic)


def enrich_split_dir(
    *,
    input_split_dir: Path,
    output_split_dir: Path,
    global_model: Path,
    roi_model: Path,
    local_model: Path | None,
    candidate_expand_ratios: list[float] | None,
    candidate_baseline_gate: float,
    candidate_min_score_gain: float,
    diagnostics_json: Path | None = None,
) -> dict[str, Any]:
    output_split_dir.mkdir(parents=True, exist_ok=True)
    global_predictor = GlobalCornerPredictor(global_model)
    roi_predictor = RoiCornerPredictor(roi_model)
    local_predictor = LocalCornerMoEPredictor(local_model) if local_model is not None else None
    failure_layer_index = load_failure_layer_index(diagnostics_json)
    summary: dict[str, Any] = {
        "input_split_dir": str(input_split_dir),
        "output_split_dir": str(output_split_dir),
        "global_model": str(global_model),
        "roi_model": str(roi_model),
        "local_model": str(local_model) if local_model is not None else None,
        "diagnostics_json": str(diagnostics_json) if diagnostics_json is not None else None,
        "candidate_expand_ratios": candidate_expand_ratios or [],
        "candidate_baseline_gate": float(candidate_baseline_gate),
        "candidate_min_score_gain": float(candidate_min_score_gain),
        "splits": {},
    }
    split_names = ["train", "test", "focus_train", "focus_test", "holdout"]
    for split_name in split_names:
        input_path = input_split_dir / f"{split_name}.jsonl"
        if not input_path.exists():
            continue
        rows = _read_jsonl_rows(input_path)
        enriched_rows = [
            enrich_row(
                row,
                global_predictor=global_predictor,
                roi_predictor=roi_predictor,
                local_predictor=local_predictor,
                candidate_expand_ratios=candidate_expand_ratios,
                candidate_baseline_gate=candidate_baseline_gate,
                candidate_min_score_gain=candidate_min_score_gain,
                failure_layer_index=failure_layer_index,
            )
            for row in rows
        ]
        _write_jsonl_rows(output_split_dir / f"{split_name}.jsonl", enriched_rows)
        summary["splits"][split_name] = {"rows": len(enriched_rows)}
    summary_path = input_split_dir / "summary.json"
    if summary_path.exists():
        (output_split_dir / "summary.source.json").write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_split_dir / "summary.runtime_enriched.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich global corner split rows with runtime candidate quads")
    parser.add_argument("--input-split-dir", required=True)
    parser.add_argument("--output-split-dir", required=True)
    parser.add_argument("--global-model", default="models/runtime/global_corner_model.pt")
    parser.add_argument("--roi-model", default="models/runtime/corner_heatmap_model.pt")
    parser.add_argument("--local-model", default="models/runtime/local_corner_moe_coord_model.pt")
    parser.add_argument(
        "--candidate-expand-ratios",
        default="",
        help="Comma-separated expand ratios used by runtime multi-expand candidate selection.",
    )
    parser.add_argument("--diagnostics-json")
    parser.add_argument("--candidate-baseline-gate", type=float, default=0.45)
    parser.add_argument("--candidate-min-score-gain", type=float, default=0.03)
    args = parser.parse_args()

    candidate_expand_ratios = [
        float(part.strip())
        for part in str(args.candidate_expand_ratios).split(",")
        if part.strip()
    ]
    result = enrich_split_dir(
        input_split_dir=Path(args.input_split_dir),
        output_split_dir=Path(args.output_split_dir),
        global_model=Path(args.global_model),
        roi_model=Path(args.roi_model),
        local_model=Path(args.local_model) if str(args.local_model).strip() else None,
        candidate_expand_ratios=candidate_expand_ratios or None,
        candidate_baseline_gate=float(args.candidate_baseline_gate),
        candidate_min_score_gain=float(args.candidate_min_score_gain),
        diagnostics_json=Path(args.diagnostics_json) if args.diagnostics_json else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
