from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perspective_detect import (
    DEFAULT_SCORING_PROFILE,
    combine_score_from_metrics,
    detect_best_candidate_with_profile,
)


MetricProfile = dict[str, dict[str, float]]
TrainingSample = dict[str, Any]

METRIC_KEYS = tuple(DEFAULT_SCORING_PROFILE["weights"].keys())


def _order_points(points: list[list[float]] | np.ndarray) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def quad_iou(quad_a: list[list[float]] | np.ndarray, quad_b: list[list[float]] | np.ndarray) -> float:
    qa = _order_points(quad_a)
    qb = _order_points(quad_b)
    all_pts = np.vstack([qa, qb])
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)
    width = max(float(max_x - min_x), 1.0)
    height = max(float(max_y - min_y), 1.0)
    canvas = (1000, 1000)
    scale = min((canvas[1] - 4) / width, (canvas[0] - 4) / height)
    offset = np.array([2 - min_x * scale, 2 - min_y * scale], dtype=np.float32)

    mask_a = np.zeros(canvas, dtype=np.uint8)
    mask_b = np.zeros(canvas, dtype=np.uint8)
    cv2.fillConvexPoly(mask_a, np.round(qa * scale + offset).astype(np.int32), 1)
    cv2.fillConvexPoly(mask_b, np.round(qb * scale + offset).astype(np.int32), 1)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    normalized = {key: 0.0 for key in METRIC_KEYS}
    for key in METRIC_KEYS:
        value = metrics.get(key, 0.0)
        normalized[key] = float(value) if isinstance(value, (int, float)) else 0.0
    return normalized


def _candidate_entries_from_detector(
    image_path: Path,
    opencv_profile: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    result = detect_best_candidate_with_profile(image, opencv_profile)
    if result is None:
        return []
    return result["candidates"]


def _resolve_image_path(project_path: Path, project_data: dict[str, Any], page: dict[str, Any]) -> Path | None:
    raw_path = page.get("path")
    source_dir = project_data.get("sourceDir")

    candidates: list[Path] = []
    if raw_path:
        page_path = Path(raw_path)
        candidates.append(page_path)
        candidates.append(project_path.parent / raw_path)
        candidates.append(project_path.parent / page_path.name)
        if source_dir:
            candidates.append(Path(source_dir) / page_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_training_samples(
    dataset_root: Path,
    rerun_detection: bool = False,
    opencv_profile: dict[str, float] | None = None,
) -> list[TrainingSample]:
    samples: list[TrainingSample] = []
    for project_path in sorted(dataset_root.rglob("screen-pdf-project.json")):
        data = json.loads(project_path.read_text(encoding="utf-8"))
        for page in data.get("pages", []):
            manual_quad = page.get("manualQuad")
            if rerun_detection:
                image_path = _resolve_image_path(project_path, data, page)
                candidates = (
                    _candidate_entries_from_detector(image_path, opencv_profile)
                    if image_path
                    else []
                )
            else:
                candidates = page.get("candidates") or []
            if not manual_quad or not candidates:
                continue

            selected_index = int(page.get("selectedCandidateIndex", 0))
            processed_candidates = []
            for candidate in candidates:
                quad = candidate.get("quad")
                metrics = candidate.get("metrics") or {}
                method = candidate.get("method", "unknown")
                if quad is None or len(quad) != 4:
                    continue
                processed_candidates.append(
                    {
                        "method": method,
                        "metrics": _normalize_metrics(metrics),
                        "iou": quad_iou(manual_quad, quad),
                    }
                )
            if not processed_candidates:
                continue

            selected_method = None
            if not rerun_detection and 0 <= selected_index < len(candidates):
                selected_method = candidates[selected_index].get("method")
            if rerun_detection:
                selected_index = -1

            samples.append(
                {
                    "project_path": str(project_path),
                    "page_id": page.get("id", ""),
                    "selected_index": selected_index,
                    "selected_method": selected_method,
                    "candidates": processed_candidates,
                }
            )
    return samples


def evaluate_profile_on_samples(profile: MetricProfile, samples: list[TrainingSample]) -> dict[str, Any]:
    if not samples:
        return {
            "pages": 0,
            "top1_mean_iou": 0.0,
            "oracle_mean_iou": 0.0,
            "selected_mean_iou": 0.0,
            "top1_ge_0_8": 0,
            "top1_ge_0_9": 0,
            "selected_match_count": 0,
            "selected_match_rate": 0.0,
            "method_counts": {},
        }

    top1_ious = []
    oracle_ious = []
    selected_ious = []
    selected_match_count = 0
    method_counts: dict[str, int] = {}

    for sample in samples:
        scored = sorted(
            sample["candidates"],
            key=lambda candidate: combine_score_from_metrics(
                candidate["metrics"], profile, candidate["method"]
            ),
            reverse=True,
        )
        best = scored[0]
        top1_ious.append(float(best["iou"]))
        oracle_ious.append(max(float(candidate["iou"]) for candidate in sample["candidates"]))
        method_counts[best["method"]] = method_counts.get(best["method"], 0) + 1

        selected_method = sample.get("selected_method")
        selected_index = sample.get("selected_index", -1)
        if (
            selected_method
            and isinstance(selected_index, int)
            and 0 <= selected_index < len(sample["candidates"])
        ):
            selected_ious.append(float(sample["candidates"][selected_index]["iou"]))
            if best["method"] == selected_method:
                selected_match_count += 1

    pages = len(samples)
    return {
        "pages": pages,
        "top1_mean_iou": round(float(np.mean(top1_ious)), 4),
        "oracle_mean_iou": round(float(np.mean(oracle_ious)), 4),
        "selected_mean_iou": round(float(np.mean(selected_ious)) if selected_ious else 0.0, 4),
        "top1_ge_0_8": int(sum(value >= 0.8 for value in top1_ious)),
        "top1_ge_0_9": int(sum(value >= 0.9 for value in top1_ious)),
        "selected_match_count": int(selected_match_count),
        "selected_match_rate": round(selected_match_count / pages, 4),
        "method_counts": dict(sorted(method_counts.items())),
    }


def _objective(summary: dict[str, Any]) -> float:
    pages = max(int(summary["pages"]), 1)
    return (
        float(summary["top1_mean_iou"]) * 1000.0
        + float(summary["top1_ge_0_8"]) / pages * 120.0
        + float(summary["top1_ge_0_9"]) / pages * 180.0
        + float(summary["selected_match_rate"]) * 60.0
    )


def optimize_profile(
    samples: list[TrainingSample],
    *,
    iterations: int = 2500,
    seed: int = 7,
) -> tuple[MetricProfile, dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    best_profile: MetricProfile = {
        "weights": dict(DEFAULT_SCORING_PROFILE["weights"]),
        "method_bias": dict(DEFAULT_SCORING_PROFILE["method_bias"]),
    }
    baseline_summary = evaluate_profile_on_samples(best_profile, samples)
    best_summary = baseline_summary
    best_score = _objective(best_summary)

    methods = sorted(
        {
            candidate["method"]
            for sample in samples
            for candidate in sample["candidates"]
        }
    )

    for _ in range(iterations):
        candidate_profile: MetricProfile = {
            "weights": dict(best_profile["weights"]),
            "method_bias": dict(best_profile["method_bias"]),
        }

        for key, value in list(candidate_profile["weights"].items()):
            step = 0.03 if "penalty" not in key else 0.025
            candidate_profile["weights"][key] = float(value + rng.normal(0.0, step))

        for method in methods:
            current = float(candidate_profile["method_bias"].get(method, 0.0))
            candidate_profile["method_bias"][method] = float(
                np.clip(current + rng.normal(0.0, 0.02), -0.22, 0.22)
            )

        summary = evaluate_profile_on_samples(candidate_profile, samples)
        score = _objective(summary)
        if score > best_score:
            best_score = score
            best_summary = summary
            best_profile = candidate_profile

    best_profile["weights"] = {
        key: round(float(value), 4) for key, value in best_profile["weights"].items()
    }
    best_profile["method_bias"] = {
        key: round(float(value), 4)
        for key, value in sorted(best_profile["method_bias"].items())
        if abs(float(value)) >= 0.005
    }
    best_summary = evaluate_profile_on_samples(best_profile, samples)
    return best_profile, baseline_summary, best_summary


def _build_report(samples: list[TrainingSample], baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    hard_cases = []
    profile = {
        "weights": dict(DEFAULT_SCORING_PROFILE["weights"]),
        "method_bias": dict(DEFAULT_SCORING_PROFILE["method_bias"]),
    }
    for sample in samples:
        ranked = sorted(
            sample["candidates"],
            key=lambda candidate: combine_score_from_metrics(
                candidate["metrics"], profile, candidate["method"]
            ),
            reverse=True,
        )
        oracle_iou = max(float(candidate["iou"]) for candidate in sample["candidates"])
        if oracle_iou < 0.8:
            hard_cases.append(
                {
                    "project_path": sample["project_path"],
                    "page_id": sample["page_id"],
                    "oracle_iou": round(oracle_iou, 4),
                    "baseline_top1_method": ranked[0]["method"],
                    "candidate_methods": [candidate["method"] for candidate in sample["candidates"]],
                }
            )
    hard_cases = sorted(hard_cases, key=lambda item: item["oracle_iou"])[:15]
    return {
        "sample_count": len(samples),
        "baseline": baseline,
        "optimized": optimized,
        "hard_cases": hard_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train screen detection scoring profile")
    parser.add_argument("--dataset-root", required=True, help="Directory containing screen-pdf-project.json files")
    parser.add_argument("--output", required=True, help="Output JSON path for the trained profile")
    parser.add_argument("--report", help="Optional JSON report path")
    parser.add_argument("--iterations", type=int, default=2500, help="Random search iterations")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--rerun-detection",
        action="store_true",
        help="Recompute candidates from source images instead of reusing stored project candidates",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    samples = load_training_samples(dataset_root, rerun_detection=args.rerun_detection)
    profile, baseline_summary, optimized_summary = optimize_profile(
        samples, iterations=args.iterations, seed=args.seed
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = _build_report(samples, baseline_summary, optimized_summary)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
