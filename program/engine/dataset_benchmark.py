from __future__ import annotations

import argparse
import json
from collections import defaultdict
import math
from pathlib import Path
import random
import time
from typing import Any

import cv2
import numpy as np

from perspective_detect import detect_best_candidate_with_profile, load_opencv_profile
from perspective_detect import order_points
from train_scoring_profile import quad_iou


PageRecord = dict[str, Any]


def _ordered_quad(quad: list[list[float]] | np.ndarray) -> np.ndarray:
    return order_points(np.array(quad, dtype=np.float32))


def _quad_mask(image_shape: tuple[int, int] | tuple[int, int, int], quad: list[list[float]] | np.ndarray) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(_ordered_quad(quad)).astype(np.int32), 255)
    return mask


def compute_scene_profile(
    image: np.ndarray,
    manual_quad: list[list[float]] | np.ndarray,
) -> dict[str, Any]:
    quad_mask = _quad_mask(image.shape, manual_quad)
    kernel = np.ones((21, 21), dtype=np.uint8)
    outer_ring = cv2.dilate(quad_mask, kernel, iterations=1)
    outer_ring = cv2.subtract(outer_ring, quad_mask)
    edge_ring = cv2.morphologyEx(quad_mask, cv2.MORPH_GRADIENT, np.ones((9, 9), dtype=np.uint8))
    inner_border = cv2.subtract(quad_mask, cv2.erode(quad_mask, np.ones((11, 11), dtype=np.uint8), iterations=1))
    inner_core = cv2.erode(quad_mask, np.ones((31, 31), dtype=np.uint8), iterations=1)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(sobel_x, sobel_y)

    inner_pixels = lab[quad_mask > 0]
    ring_pixels = lab[outer_ring > 0]
    edge_pixels = grad_mag[edge_ring > 0]
    border_pixels = lab[inner_border > 0]
    core_pixels = lab[inner_core > 0]
    if len(inner_pixels) == 0 or len(ring_pixels) == 0:
        return {
            "lab_distance": 0.0,
            "luma_delta": 0.0,
            "edge_strength": 0.0,
            "screen_brightness": 0.0,
            "outer_brightness": 0.0,
            "inner_border_brightness": 0.0,
            "inner_core_brightness": 0.0,
            "inner_border_contrast": 0.0,
            "near_color_background": False,
            "low_contrast_scene": False,
            "black_frame_scene": False,
            "scene_tags": [],
        }

    inner_mean = inner_pixels.mean(axis=0)
    ring_mean = ring_pixels.mean(axis=0)
    lab_distance = float(np.linalg.norm(inner_mean - ring_mean))
    luma_delta = float(abs(inner_mean[0] - ring_mean[0]) / 255.0)
    edge_strength = float(edge_pixels.mean() / 255.0) if len(edge_pixels) else 0.0
    screen_brightness = float(inner_mean[0] / 255.0)
    outer_brightness = float(ring_mean[0] / 255.0)
    inner_border_brightness = float(border_pixels[:, 0].mean() / 255.0) if len(border_pixels) else screen_brightness
    inner_core_brightness = float(core_pixels[:, 0].mean() / 255.0) if len(core_pixels) else screen_brightness
    inner_border_contrast = float(max(inner_core_brightness - inner_border_brightness, 0.0))
    near_color = lab_distance < 18.0 and luma_delta < 0.08
    low_contrast = near_color or edge_strength < 0.12
    black_frame_scene = inner_border_contrast > 0.18 and inner_border_brightness < 0.28 and inner_core_brightness > 0.45
    tags: list[str] = []
    if near_color:
        tags.append("near_color_background")
    if low_contrast:
        tags.append("low_contrast_scene")
    if black_frame_scene:
        tags.append("black_frame_scene")
    if screen_brightness > 0.72:
        tags.append("bright_screen")
    return {
        "lab_distance": round(lab_distance, 4),
        "luma_delta": round(luma_delta, 4),
        "edge_strength": round(edge_strength, 4),
        "screen_brightness": round(screen_brightness, 4),
        "outer_brightness": round(outer_brightness, 4),
        "inner_border_brightness": round(inner_border_brightness, 4),
        "inner_core_brightness": round(inner_core_brightness, 4),
        "inner_border_contrast": round(inner_border_contrast, 4),
        "near_color_background": near_color,
        "low_contrast_scene": low_contrast,
        "black_frame_scene": black_frame_scene,
        "scene_tags": tags,
    }


def _manual_scale(manual_quad: list[list[float]] | np.ndarray) -> float:
    manual = _ordered_quad(manual_quad)
    area = abs(float(cv2.contourArea(manual)))
    if area > 1e-6:
        return max(math.sqrt(area), 1.0)
    bbox = np.ptp(manual, axis=0)
    return float(np.hypot(max(float(bbox[0]), 1.0), max(float(bbox[1]), 1.0)))


def _edge_angles_deg(quad: np.ndarray) -> np.ndarray:
    edges = np.roll(quad, -1, axis=0) - quad
    return np.degrees(np.arctan2(edges[:, 1], edges[:, 0]))


def _angular_delta_deg(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 180.0
    return min(delta, 180.0 - delta)


def normalized_point_error(
    manual_quad: list[list[float]] | np.ndarray,
    predicted_quad: list[list[float]] | np.ndarray,
) -> float:
    manual = _ordered_quad(manual_quad)
    predicted = _ordered_quad(predicted_quad)
    bbox = np.ptp(manual, axis=0)
    diag = float(np.hypot(max(float(bbox[0]), 1.0), max(float(bbox[1]), 1.0)))
    if diag <= 1e-6:
        return 0.0
    distances = np.linalg.norm(predicted - manual, axis=1)
    return float(np.mean(distances) / diag)


def screen_relative_point_error(
    manual_quad: list[list[float]] | np.ndarray,
    predicted_quad: list[list[float]] | np.ndarray,
) -> float:
    manual = _ordered_quad(manual_quad)
    predicted = _ordered_quad(predicted_quad)
    scale = _manual_scale(manual)
    distances = np.linalg.norm(predicted - manual, axis=1)
    return float(np.mean(distances) / max(scale, 1e-6))


def max_corner_error(
    manual_quad: list[list[float]] | np.ndarray,
    predicted_quad: list[list[float]] | np.ndarray,
) -> float:
    manual = _ordered_quad(manual_quad)
    predicted = _ordered_quad(predicted_quad)
    scale = _manual_scale(manual)
    distances = np.linalg.norm(predicted - manual, axis=1)
    return float(np.max(distances) / max(scale, 1e-6))


def perspective_tilt_error(
    manual_quad: list[list[float]] | np.ndarray,
    predicted_quad: list[list[float]] | np.ndarray,
) -> float:
    manual = _ordered_quad(manual_quad)
    predicted = _ordered_quad(predicted_quad)
    manual_angles = _edge_angles_deg(manual)
    predicted_angles = _edge_angles_deg(predicted)
    deltas = [_angular_delta_deg(a, b) for a, b in zip(manual_angles, predicted_angles, strict=True)]
    return float(np.mean(deltas))


def quad_inset_ratio(
    manual_quad: list[list[float]] | np.ndarray,
    predicted_quad: list[list[float]] | np.ndarray,
) -> float:
    manual_area = abs(float(cv2.contourArea(_ordered_quad(manual_quad))))
    predicted_area = abs(float(cv2.contourArea(_ordered_quad(predicted_quad))))
    if manual_area <= 1e-6:
        return 0.0
    return float((manual_area - predicted_area) / manual_area)


def quad_geometry_metrics(
    manual_quad: list[list[float]] | np.ndarray,
    predicted_quad: list[list[float]] | np.ndarray,
) -> dict[str, float]:
    return {
        "point_error": normalized_point_error(manual_quad, predicted_quad),
        "screen_relative_point_error": screen_relative_point_error(manual_quad, predicted_quad),
        "max_corner_error": max_corner_error(manual_quad, predicted_quad),
        "perspective_tilt_error": perspective_tilt_error(manual_quad, predicted_quad),
        "quad_inset_ratio": quad_inset_ratio(manual_quad, predicted_quad),
    }


def summarize_geometry_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "pages": 0,
            "point_error_mean": 0.0,
            "point_error_max": 0.0,
            "point_error_p95": 0.0,
            "point_le_0_05_ratio": 0.0,
            "point_le_0_03_ratio": 0.0,
            "point_le_0_02_ratio": 0.0,
            "point_le_0_01_ratio": 0.0,
            "screen_relative_error_mean": 0.0,
            "screen_relative_error_max": 0.0,
            "screen_relative_error_p95": 0.0,
            "max_corner_error_mean": 0.0,
            "max_corner_error_max": 0.0,
            "max_corner_error_p95": 0.0,
            "max_corner_le_0_01_ratio": 0.0,
            "perspective_tilt_error_mean": 0.0,
            "perspective_tilt_error_p95": 0.0,
            "quad_inset_ratio_mean": 0.0,
            "quad_inset_ratio_abs_mean": 0.0,
        }
    total = float(len(rows))
    point_errors = np.array([float(row["point_error"]) for row in rows], dtype=np.float32)
    screen_errors = np.array([float(row["screen_relative_point_error"]) for row in rows], dtype=np.float32)
    max_corner_errors = np.array([float(row["max_corner_error"]) for row in rows], dtype=np.float32)
    tilt_errors = np.array([float(row["perspective_tilt_error"]) for row in rows], dtype=np.float32)
    inset_ratios = np.array([float(row["quad_inset_ratio"]) for row in rows], dtype=np.float32)
    return {
        "pages": len(rows),
        "point_error_mean": round(float(point_errors.mean()), 4),
        "point_error_max": round(float(point_errors.max()), 4),
        "point_error_p95": round(float(np.percentile(point_errors, 95)), 4),
        "point_le_0_05_ratio": round(float((point_errors <= 0.05).mean()), 4),
        "point_le_0_03_ratio": round(float((point_errors <= 0.03).mean()), 4),
        "point_le_0_02_ratio": round(float((point_errors <= 0.02).mean()), 4),
        "point_le_0_01_ratio": round(float((point_errors <= 0.01).mean()), 4),
        "screen_relative_error_mean": round(float(screen_errors.mean()), 4),
        "screen_relative_error_max": round(float(screen_errors.max()), 4),
        "screen_relative_error_p95": round(float(np.percentile(screen_errors, 95)), 4),
        "max_corner_error_mean": round(float(max_corner_errors.mean()), 4),
        "max_corner_error_max": round(float(max_corner_errors.max()), 4),
        "max_corner_error_p95": round(float(np.percentile(max_corner_errors, 95)), 4),
        "max_corner_le_0_01_ratio": round(float((max_corner_errors <= 0.01).mean()), 4),
        "perspective_tilt_error_mean": round(float(tilt_errors.mean()), 4),
        "perspective_tilt_error_p95": round(float(np.percentile(tilt_errors, 95)), 4),
        "quad_inset_ratio_mean": round(float(inset_ratios.mean()), 4),
        "quad_inset_ratio_abs_mean": round(float(np.abs(inset_ratios).mean()), 4),
    }


def point_success_ratio(errors: list[float], threshold: float) -> float:
    if not errors:
        return 0.0
    return float(sum(error <= threshold for error in errors) / len(errors))


def _resolve_image_path(project_path: Path, page: dict[str, Any]) -> Path | None:
    raw_path = page.get("path")
    if not raw_path:
        return None
    image_path = Path(raw_path)
    candidates = [
        image_path,
        project_path.parent / raw_path,
        project_path.parent / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_manual_pages(dataset_root: Path) -> list[PageRecord]:
    pages: list[PageRecord] = []
    for project_path in sorted(dataset_root.rglob("screen-pdf-project.json")):
        data = json.loads(project_path.read_text(encoding="utf-8"))
        for page in data.get("pages", []):
            manual_quad = page.get("manualQuad")
            if not manual_quad:
                continue
            image_path = _resolve_image_path(project_path, page)
            if image_path is None:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            scene_profile = compute_scene_profile(image, manual_quad)
            pages.append(
                {
                    "project_path": str(project_path),
                    "project_name": project_path.parent.name,
                    "page_id": page.get("id") or image_path.name,
                    "image_path": str(image_path),
                    "manual_quad": manual_quad,
                    "scene_profile": scene_profile,
                    "scene_tags": scene_profile["scene_tags"],
                }
            )
    return pages


def build_split(pages: list[PageRecord], test_ratio: float = 0.25, seed: int = 7) -> dict[str, list[PageRecord]]:
    grouped: dict[str, list[PageRecord]] = defaultdict(list)
    for page in pages:
        grouped[page["project_path"]].append(page)

    rng = random.Random(seed)
    train: list[PageRecord] = []
    test: list[PageRecord] = []
    for entries in grouped.values():
        ordered = sorted(entries, key=lambda item: str(item["page_id"]))
        rng.shuffle(ordered)
        if len(ordered) == 1:
            train.extend(ordered)
            continue
        test_count = max(1, math.ceil(len(ordered) * test_ratio))
        test.extend(ordered[:test_count])
        train.extend(ordered[test_count:])
    return {"train": train, "test": test}


def _matches_focus_project(page: PageRecord, focus_projects: set[str]) -> bool:
    if not focus_projects:
        return False
    project_path = str(page.get("project_path", ""))
    project_name = str(page.get("project_name", ""))
    return project_path in focus_projects or project_name in focus_projects


def build_project_aware_split(
    pages: list[PageRecord],
    *,
    focus_projects: set[str] | list[str] | tuple[str, ...],
    holdout_projects: set[str] | list[str] | tuple[str, ...] = (),
    test_ratio: float = 0.25,
    focus_test_ratio: float = 0.25,
    seed: int = 7,
) -> dict[str, Any]:
    focus_set = {str(item) for item in focus_projects if str(item).strip()}
    holdout_set = {str(item) for item in holdout_projects if str(item).strip()}
    holdout_pages = [page for page in pages if _matches_focus_project(page, holdout_set)]
    remaining_pages = [page for page in pages if not _matches_focus_project(page, holdout_set)]
    focus_pages = [page for page in remaining_pages if _matches_focus_project(page, focus_set)]
    base_pages = [page for page in remaining_pages if not _matches_focus_project(page, focus_set)]

    base_split = build_split(base_pages, test_ratio=test_ratio, seed=seed)
    focus_grouped: dict[str, list[PageRecord]] = defaultdict(list)
    for page in focus_pages:
        focus_grouped[str(page["project_path"])].append(page)

    rng = random.Random(seed)
    focus_train: list[PageRecord] = []
    focus_test: list[PageRecord] = []
    focus_names: set[str] = set()
    for entries in focus_grouped.values():
        ordered = sorted(entries, key=lambda item: str(item["page_id"]))
        rng.shuffle(ordered)
        focus_names.add(str(ordered[0].get("project_name", "")))
        if len(ordered) == 1:
            focus_train.extend(ordered)
            continue
        test_count = max(1, math.ceil(len(ordered) * focus_test_ratio))
        focus_test.extend(ordered[:test_count])
        focus_train.extend(ordered[test_count:])

    train = list(base_split["train"]) + focus_train
    metadata = {
        "focus_project_count": len(focus_grouped),
        "focus_project_names": sorted(name for name in focus_names if name),
        "focus_train_pages": len(focus_train),
        "focus_test_pages": len(focus_test),
        "holdout_pages": len(holdout_pages),
        "base_train_pages": len(base_split["train"]),
        "base_test_pages": len(base_split["test"]),
        "seed": seed,
        "test_ratio": test_ratio,
        "focus_test_ratio": focus_test_ratio,
    }
    return {
        "train": train,
        "test": list(base_split["test"]),
        "focus_train": focus_train,
        "focus_test": focus_test,
        "holdout": holdout_pages,
        "metadata": metadata,
    }


def evaluate_pages(
    pages: list[PageRecord],
    opencv_profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_profile = opencv_profile or load_opencv_profile()
    results = []
    method_counts: dict[str, int] = defaultdict(int)
    point_errors: list[float] = []
    elapsed_ms: list[float] = []
    for page in pages:
        image = cv2.imread(page["image_path"])
        if image is None:
            continue
        t0 = time.perf_counter()
        result = detect_best_candidate_with_profile(image, active_profile)
        elapsed_ms.append((time.perf_counter() - t0) * 1000.0)
        if result is None:
            iou = 0.0
            method = "none"
            confidence = 0.0
            point_error = 1.0
        else:
            iou = quad_iou(page["manual_quad"], result["best"]["quad"])
            method = str(result["best"]["method"])
            confidence = float(result["best"]["confidence"])
            point_error = normalized_point_error(page["manual_quad"], result["best"]["quad"])
        method_counts[method] += 1
        point_errors.append(point_error)
        results.append(
            {
                "project_name": page["project_name"],
                "page_id": page["page_id"],
                "image_path": page["image_path"],
                "iou": round(float(iou), 4),
                "point_error": round(float(point_error), 4),
                "method": method,
                "confidence": round(float(confidence), 4),
            }
        )

    ious = [item["iou"] for item in results]
    return {
        "pages": len(results),
        "top1_mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "top1_ge_0_8": int(sum(value >= 0.8 for value in ious)),
        "top1_ge_0_8_ratio": round(sum(value >= 0.8 for value in ious) / max(len(ious), 1), 4),
        "top1_ge_0_9": int(sum(value >= 0.9 for value in ious)),
        "point_error_mean": round(float(np.mean(point_errors)), 4) if point_errors else 0.0,
        "point_error_p95": round(float(np.percentile(point_errors, 95)), 4) if point_errors else 0.0,
        "point_le_0_01": int(sum(value <= 0.01 for value in point_errors)),
        "point_le_0_01_ratio": round(point_success_ratio(point_errors, 0.01), 4),
        "mean_elapsed_ms": round(float(np.mean(elapsed_ms)), 2) if elapsed_ms else 0.0,
        "p95_elapsed_ms": round(float(np.percentile(elapsed_ms, 95)), 2) if elapsed_ms else 0.0,
        "method_counts": dict(sorted(method_counts.items())),
        "worst_pages": sorted(results, key=lambda item: (item["point_error"], -item["iou"]), reverse=True)[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark screen detection dataset split")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--focus-project", action="append", dest="focus_projects", default=[])
    parser.add_argument("--focus-test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    pages = load_manual_pages(dataset_root)
    if args.focus_projects:
        split = build_project_aware_split(
            pages,
            focus_projects=list(args.focus_projects),
            holdout_projects=[],
            test_ratio=args.test_ratio,
            focus_test_ratio=args.focus_test_ratio,
            seed=args.seed,
        )
    else:
        split = build_split(pages, test_ratio=args.test_ratio, seed=args.seed)
    report = {
        "dataset_root": str(dataset_root),
        "pages": len(pages),
        "train": evaluate_pages(split["train"]),
        "test": evaluate_pages(split["test"]),
    }
    if "focus_train" in split:
        report["focus_train"] = evaluate_pages(split["focus_train"])
        report["focus_test"] = evaluate_pages(split["focus_test"])
        report["metadata"] = split.get("metadata", {})
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
