#!/usr/bin/env python3
"""Generate coarse-tagged screen-pdf-project_v1.json files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


V1_FILE_NAME = "screen-pdf-project_v1.json"
LEGACY_FILE_NAME = "screen-pdf-project.json"
TAG_VERSION = 1
VALID_BUCKETS = {"clean", "hard", "abnormal"}
VALID_FAILURE_TAGS = {
    "corner_out_of_frame",
    "edge_touch_border",
    "heavy_occlusion",
    "edge_only_visible",
    "black_frame",
    "low_contrast",
    "strong_perspective",
    "large_spill",
    "candidate_disagreement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Root directory containing project json files")
    return parser.parse_args()


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^\w\u4e00-\u9fff]+", "-", lowered, flags=re.UNICODE)
    lowered = re.sub(r"-{2,}", "-", lowered).strip("-")
    return lowered or "unknown-event"


def normalize_bucket(value: str) -> str:
    return value if value in VALID_BUCKETS else "clean"


def normalize_failure_tags(tags: list[str]) -> list[str]:
    deduped: list[str] = []
    for tag in tags:
        if tag in VALID_FAILURE_TAGS and tag not in deduped:
            deduped.append(tag)
    return deduped


def event_metadata(project_path: Path) -> tuple[str, str]:
    project_dir = project_path.parent
    event_dir = project_dir.parent if project_dir.parent != project_dir else project_dir
    event_name = event_dir.name
    return slugify(event_name), event_name


def point_outside_image(point: list[float], width: float, height: float, tolerance: float = 12.0) -> bool:
    x, y = point
    return x < -tolerance or y < -tolerance or x > width + tolerance or y > height + tolerance


def point_touches_border(point: list[float], width: float, height: float, margin: float = 8.0) -> bool:
    x, y = point
    return x < margin or y < margin or x > width - margin or y > height - margin


def quad_span(quad: list[list[float]]) -> tuple[float, float]:
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    return max(xs) - min(xs), max(ys) - min(ys)


def candidate_disagreement(page: dict[str, Any]) -> bool:
    candidates = page.get("candidates") or []
    if len(candidates) < 2:
        return False
    quad_a = candidates[0].get("quad") or []
    quad_b = candidates[1].get("quad") or []
    if len(quad_a) != 4 or len(quad_b) != 4:
        return False
    width = float((page.get("details") or {}).get("width") or 0.0)
    height = float((page.get("details") or {}).get("height") or 0.0)
    norm = max(width, height, 1.0)
    total = 0.0
    for left, right in zip(quad_a, quad_b):
        total += abs(float(left[0]) - float(right[0])) + abs(float(left[1]) - float(right[1]))
    return (total / 8.0) / norm >= 0.06


def infer_failure_tags(page: dict[str, Any]) -> tuple[list[str], list[str]]:
    details = page.get("details") or {}
    width = float(details.get("width") or 0.0)
    height = float(details.get("height") or 0.0)
    active_quad = page.get("manualQuad") or page.get("activeQuad") or []
    metrics = {}
    candidates = page.get("candidates") or []
    selected_index = int(page.get("selectedCandidateIndex") or 0)
    if 0 <= selected_index < len(candidates):
        metrics = candidates[selected_index].get("metrics") or {}

    tags: list[str] = []
    reasons: list[str] = []

    if len(active_quad) == 4 and width > 0 and height > 0:
        if any(point_outside_image(point, width, height) for point in active_quad):
            tags.append("corner_out_of_frame")
            reasons.append("角点超出图像边界")
        if any(point_touches_border(point, width, height) for point in active_quad):
            tags.append("edge_touch_border")
            reasons.append("候选框贴近图像边缘")
        span_w, span_h = quad_span(active_quad)
        if span_w < width * 0.42 or span_h < height * 0.42:
            tags.append("heavy_occlusion")
            reasons.append("有效屏幕区域明显偏小")

    spill_penalty = float(metrics.get("spill_penalty") or 0.0)
    blue_penalty = float(metrics.get("blue_penalty") or 0.0)
    parallel_score = float(metrics.get("parallel_score") or 0.0)
    edge_score = float(metrics.get("edge_score") or 0.0)
    coverage_score = float(metrics.get("coverage_score") or 0.0)

    if spill_penalty >= 0.12:
        tags.append("large_spill")
        reasons.append("spill_penalty 偏高")
    if blue_penalty >= 0.55:
        tags.append("black_frame")
        reasons.append("blue_penalty 指示明显黑边")
    if edge_score <= 0.42 or coverage_score <= 0.08:
        tags.append("low_contrast")
        reasons.append("边缘或覆盖度分数偏低")
    if parallel_score <= 0.68:
        tags.append("strong_perspective")
        reasons.append("parallel_score 指示透视较强")
    if edge_score <= 0.28 and coverage_score <= 0.05:
        tags.append("edge_only_visible")
        reasons.append("主要依赖边缘残片")
    if candidate_disagreement(page):
        tags.append("candidate_disagreement")
        reasons.append("前两候选差异较大")

    return normalize_failure_tags(tags), reasons


def infer_bucket(failure_tags: list[str]) -> str:
    severe = {
        "corner_out_of_frame",
        "heavy_occlusion",
        "edge_only_visible",
        "large_spill",
        "candidate_disagreement",
    }
    medium = {"edge_touch_border", "black_frame", "low_contrast", "strong_perspective"}
    if any(tag in severe for tag in failure_tags):
        return "abnormal"
    if sum(tag in medium for tag in failure_tags) >= 1:
        return "hard"
    return "clean"


def summarize_pages(pages: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts = Counter(page.get("difficultyBucket", "clean") for page in pages)
    failure_counts = Counter(tag for page in pages for tag in page.get("failureTags") or [])
    return {
        "pages": len(pages),
        "bucketCounts": dict(sorted(bucket_counts.items())),
        "failureTagCounts": dict(sorted(failure_counts.items())),
    }


def tag_project(project_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    event_slug, event_name = event_metadata(project_path)
    tagged_pages: list[dict[str, Any]] = []
    for page in payload.get("pages") or []:
        tagged_page = dict(page)
        failure_tags, reasons = infer_failure_tags(page)
        tagged_page["eventSlug"] = event_slug
        tagged_page["difficultyBucket"] = normalize_bucket(infer_bucket(failure_tags))
        tagged_page["failureTags"] = failure_tags
        tagged_page["bucketReason"] = reasons
        tagged_page["reviewTags"] = ["auto"]
        tagged_page["tagVersion"] = TAG_VERSION
        tagged_pages.append(tagged_page)

    tagged_payload = dict(payload)
    tagged_payload["eventSlug"] = event_slug
    tagged_payload["eventName"] = event_name
    tagged_payload["tagVersion"] = TAG_VERSION
    tagged_payload["pages"] = tagged_pages
    tagged_payload["tagSummary"] = summarize_pages(tagged_pages)
    return tagged_payload


def write_tagged_project(project_path: Path) -> Path:
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    tagged_payload = tag_project(project_path, payload)
    target_path = project_path.with_name(V1_FILE_NAME)
    tagged_payload["projectPath"] = str(target_path)
    target_path.write_text(json.dumps(tagged_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target_path


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    project_files = sorted(dataset_root.rglob(LEGACY_FILE_NAME))
    if not project_files:
        raise SystemExit(f"no {LEGACY_FILE_NAME} found under {dataset_root}")
    for project_path in project_files:
        write_tagged_project(project_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
