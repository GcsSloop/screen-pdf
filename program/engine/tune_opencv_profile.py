from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import itertools
import random

from perspective_detect import DEFAULT_OPENCV_PROFILE, load_scoring_profile
from train_scoring_profile import evaluate_profile_on_samples, load_training_samples


def objective_score(metrics: dict[str, Any], baseline_global_top1: float) -> float:
    focus = metrics["focus"]
    global_summary = metrics["global"]
    focus_pages = max(int(focus["pages"]), 1)
    global_pages = max(int(global_summary["pages"]), 1)

    score = 0.0
    score += float(focus["top1_mean_iou"]) * 700.0
    score += float(focus["oracle_mean_iou"]) * 900.0
    score += float(focus["top1_ge_0_8"]) / focus_pages * 120.0
    score += float(global_summary["top1_mean_iou"]) * 260.0
    score += float(global_summary["oracle_mean_iou"]) * 220.0
    score += float(metrics["project_floor"]) * 180.0

    global_drop = max(0.0, baseline_global_top1 - float(global_summary["top1_mean_iou"]))
    score -= global_drop * 4000.0
    return float(score)


def _group_project_floor(samples: list[dict[str, Any]], scoring_profile: dict[str, dict[str, float]]) -> float:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["project_path"], []).append(sample)
    if not grouped:
        return 0.0
    project_scores = []
    for entries in grouped.values():
        summary = evaluate_profile_on_samples(scoring_profile, entries)
        project_scores.append(float(summary["top1_mean_iou"]))
    return float(min(project_scores))


def evaluate_candidate_profile(
    dataset_root: Path,
    focus_root: Path,
    opencv_profile: dict[str, float],
    scoring_profile: dict[str, dict[str, float]],
) -> dict[str, Any]:
    focus_samples = load_training_samples(focus_root, rerun_detection=True, opencv_profile=opencv_profile)
    global_samples = load_training_samples(dataset_root, rerun_detection=True, opencv_profile=opencv_profile)
    return {
        "focus": evaluate_profile_on_samples(scoring_profile, focus_samples),
        "global": evaluate_profile_on_samples(scoring_profile, global_samples),
        "project_floor": round(_group_project_floor(global_samples, scoring_profile), 4),
    }


def _project_roots(dataset_root: Path) -> list[Path]:
    return sorted({project.parent for project in dataset_root.rglob("screen-pdf-project.json")})


def _manual_count(project_root: Path) -> int:
    project_path = project_root / "screen-pdf-project.json"
    if not project_path.exists():
        return 0
    data = json.loads(project_path.read_text(encoding="utf-8"))
    return sum(1 for page in data.get("pages", []) if page.get("manualQuad"))


def _evaluate_roots(
    roots: list[Path],
    focus_root: Path,
    opencv_profile: dict[str, float],
    scoring_profile: dict[str, dict[str, float]],
) -> dict[str, Any]:
    focus_samples = load_training_samples(focus_root, rerun_detection=True, opencv_profile=opencv_profile)
    all_samples: list[dict[str, Any]] = []
    for root in roots:
        all_samples.extend(load_training_samples(root, rerun_detection=True, opencv_profile=opencv_profile))
    return {
        "focus": evaluate_profile_on_samples(scoring_profile, focus_samples),
        "global": evaluate_profile_on_samples(scoring_profile, all_samples),
        "project_floor": round(_group_project_floor(all_samples, scoring_profile), 4),
    }


def _candidate_profiles(seed: int, iterations: int) -> list[dict[str, float]]:
    choices = {
        "clahe_clip_limit": [1.8, 2.1, 2.4, 2.8, 3.2],
        "lsd_scale": [0.6, 0.8, 1.0],
        "lsd_sigma_scale": [0.5, 0.6, 0.8],
        "lsd_quant": [1.5, 2.0, 2.5],
        "lsd_ang_th": [18.0, 22.5, 28.0],
        "roi_expand_ratio": [0.08, 0.12, 0.16, 0.2],
        "grabcut_iters": [2, 3, 4],
        "mask_close_kernel": [9, 11, 13],
        "mask_open_kernel": [3, 5, 7],
    }
    grid = []
    for values in itertools.product(*choices.values()):
        candidate = dict(DEFAULT_OPENCV_PROFILE)
        for key, value in zip(choices.keys(), values):
            candidate[key] = value
        grid.append(candidate)
    rng = random.Random(seed)
    rng.shuffle(grid)
    picked = [dict(DEFAULT_OPENCV_PROFILE)]
    picked.extend(grid[: max(0, iterations - 1)])
    return picked


def optimize_opencv_profile(
    dataset_root: Path,
    focus_root: Path,
    *,
    iterations: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    scoring_profile = load_scoring_profile()
    baseline_profile = dict(DEFAULT_OPENCV_PROFILE)
    print("evaluating baseline on full dataset...", flush=True)
    baseline_metrics = evaluate_candidate_profile(dataset_root, focus_root, baseline_profile, scoring_profile)
    baseline_global_top1 = float(baseline_metrics["global"]["top1_mean_iou"])
    ranked: list[tuple[float, dict[str, float]]] = []

    for index, candidate in enumerate(_candidate_profiles(seed, iterations), start=1):
        print(f"search candidate {index}/{iterations}", flush=True)
        metrics = _evaluate_roots([focus_root], focus_root, candidate, scoring_profile)
        focus_only = {
            "focus": metrics["focus"],
            "global": metrics["focus"],
            "project_floor": metrics["project_floor"],
        }
        value = objective_score(focus_only, baseline_metrics["focus"]["top1_mean_iou"])
        ranked.append((value, candidate))

    ranked.sort(key=lambda item: item[0], reverse=True)
    search_best = ranked[0][1] if ranked else baseline_profile
    finalists = [baseline_profile, search_best]
    best_profile = baseline_profile
    best_metrics = baseline_metrics
    best_value = objective_score(best_metrics, baseline_global_top1)
    for idx, candidate in enumerate(finalists, start=1):
        print(f"full validation {idx}/{len(finalists)}", flush=True)
        metrics = evaluate_candidate_profile(dataset_root, focus_root, candidate, scoring_profile)
        value = objective_score(metrics, baseline_global_top1)
        if value > best_value:
            best_value = value
            best_profile = candidate
            best_metrics = metrics
    return best_profile, baseline_metrics, best_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune OpenCV profile")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--focus-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    focus_root = Path(args.focus_root)
    best_profile, baseline_metrics, best_metrics = optimize_opencv_profile(
        dataset_root,
        focus_root,
        iterations=args.iterations,
        seed=args.seed,
    )

    output = Path(args.output)
    output.write_text(json.dumps(best_profile, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {
        "baseline": baseline_metrics,
        "optimized": best_metrics,
        "profile": best_profile,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
