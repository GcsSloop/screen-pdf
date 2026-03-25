#!/usr/bin/env python3
"""Build normalized dataset manifests from existing screen-pdf project roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

SCENE_TAXONOMY = [
    "white_ppt",
    "colorful_ppt",
    "near_color_background",
    "low_contrast_edge",
    "complex_background",
    "floor_reflection",
    "bottom_edge_interference",
    "side_screen_intrusion",
    "black_border",
    "led_screen",
    "corner_occlusion",
    "person_occlusion",
    "ui_overlay",
    "strong_perspective",
    "lens_distortion_sensitive",
    "unknown",
]


@dataclass
class ProjectSummary:
    project_slug: str
    project_name: str
    source_dir: str
    project_file: str
    pages: int
    reviewed_pages: int
    pages_with_manual_quad: int
    pages_with_export_preview: int
    best_method_counts: dict[str, int]
    status_counts: dict[str, int]
    export_dirs: list[str]
    pdf_files: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Original dataset root directory")
    parser.add_argument("--dataset-slug", required=True, help="English slug used inside data/raw")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to the parent of program/engine.",
    )
    return parser.parse_args()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def safe_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_or_refresh_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and Path(os.readlink(link_path)).resolve() == target_path.resolve():
            return
        if link_path.is_symlink():
            link_path.unlink()
        else:
            raise RuntimeError(f"{link_path} exists and is not a symlink")
    link_path.symlink_to(target_path)


def deterministic_bucket(key: str) -> float:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return value / 0xFFFFFFFF


def normalize_candidate_count(page: dict[str, Any]) -> int:
    candidates = page.get("candidates") or []
    return len(candidates)


def discover_export_dirs(project_dir: Path) -> list[str]:
    results: list[str] = []
    for child in project_dir.iterdir():
        if child.is_dir() and child.name.endswith("_export"):
            results.append(str(child))
    return sorted(results)


def discover_pdf_files(project_dir: Path) -> list[str]:
    results: list[str] = []
    for child in project_dir.iterdir():
        if child.is_file() and child.suffix.lower() == ".pdf":
            results.append(str(child))
    return sorted(results)


def slugify_project(index: int) -> str:
    return f"project_{index:03d}"


def build_scene_hints(project_name: str, status_counts: Counter[str]) -> list[str]:
    lowered = project_name.lower()
    tags: list[str] = []
    if "led" in lowered:
        tags.append("led_screen")
    if "ai" in lowered:
        tags.append("complex_background")
    if not tags:
        tags.append("unknown")
    if status_counts.get("reviewed", 0) > 0:
        tags.append("strong_perspective")
    return sorted(set(tags))


def load_manual_scene_overrides(scene_dir: Path, dataset_slug: str) -> dict[str, dict[str, Any]]:
    manual_path = scene_dir / f"{dataset_slug}_project_scene_map.manual.json"
    if not manual_path.exists():
        return {}
    payload = json.loads(manual_path.read_text(encoding="utf-8"))
    projects = payload.get("projects") or []
    overrides: dict[str, dict[str, Any]] = {}
    for item in projects:
        project_slug = item.get("project_slug")
        if project_slug:
            overrides[project_slug] = item
    return overrides


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    source_root = Path(args.source_root).resolve()
    dataset_slug = args.dataset_slug

    if not source_root.exists():
        raise SystemExit(f"source root does not exist: {source_root}")

    data_root = repo_root / "data"
    training_root = repo_root / "training"

    project_files = sorted(source_root.glob("**/screen-pdf-project.json"))
    if not project_files:
        raise SystemExit(f"no screen-pdf-project.json found under {source_root}")

    raw_alias = data_root / "raw" / dataset_slug
    create_or_refresh_symlink(raw_alias, source_root)

    import_rows: list[dict[str, Any]] = []
    page_rows: list[dict[str, Any]] = []
    project_summaries: list[ProjectSummary] = []
    scene_rows: list[dict[str, Any]] = []
    scene_dir = data_root / "curated" / "scenes"
    manual_scene_overrides = load_manual_scene_overrides(scene_dir, dataset_slug)

    holdout_projects: list[str] = []
    train_projects: list[str] = []
    val_projects: list[str] = []

    total_pages = 0
    reviewed_pages = 0
    manual_pages = 0

    for index, project_file in enumerate(project_files, start=1):
        payload = json.loads(project_file.read_text(encoding="utf-8"))
        project_dir = project_file.parent
        project_slug = slugify_project(index)
        project_name = payload.get("name") or project_dir.name
        pages = payload.get("pages") or []

        status_counts: Counter[str] = Counter()
        best_method_counts: Counter[str] = Counter()
        pages_with_export_preview = 0
        project_reviewed_pages = 0
        project_manual_pages = 0

        for page in pages:
            status = page.get("status") or "unknown"
            status_counts[status] += 1
            best_method = page.get("bestMethod") or "unknown"
            best_method_counts[best_method] += 1
            if status == "reviewed":
                project_reviewed_pages += 1
            if page.get("manualQuad"):
                project_manual_pages += 1
            if page.get("previewPath"):
                pages_with_export_preview += 1

            page_rows.append(
                {
                    "dataset_slug": dataset_slug,
                    "project_slug": project_slug,
                    "project_name": project_name,
                    "page_id": page.get("id"),
                    "page_name": page.get("name"),
                    "image_path": page.get("path"),
                    "thumb_path": page.get("thumbPath"),
                    "preview_path": page.get("previewPath"),
                    "status": status,
                    "confidence": page.get("confidence"),
                    "best_method": best_method,
                    "selected_candidate_index": page.get("selectedCandidateIndex"),
                    "candidate_count": normalize_candidate_count(page),
                    "manual_quad": page.get("manualQuad"),
                    "active_quad": page.get("activeQuad"),
                    "has_manual_quad": bool(page.get("manualQuad")),
                    "details": page.get("details") or {},
                }
            )

        project_summary = ProjectSummary(
            project_slug=project_slug,
            project_name=project_name,
            source_dir=str(project_dir),
            project_file=str(project_file),
            pages=len(pages),
            reviewed_pages=project_reviewed_pages,
            pages_with_manual_quad=project_manual_pages,
            pages_with_export_preview=pages_with_export_preview,
            best_method_counts=dict(sorted(best_method_counts.items())),
            status_counts=dict(sorted(status_counts.items())),
            export_dirs=discover_export_dirs(project_dir),
            pdf_files=discover_pdf_files(project_dir),
        )
        project_summaries.append(project_summary)

        import_rows.append(
            {
                "dataset_slug": dataset_slug,
                "project_slug": project_slug,
                "project_name": project_name,
                "source_dir": str(project_dir),
                "project_file": str(project_file),
                "pages": len(pages),
                "reviewed_pages": project_reviewed_pages,
                "pages_with_manual_quad": project_manual_pages,
            }
        )

        manual_override = manual_scene_overrides.get(project_slug, {})
        default_scene_tags = build_scene_hints(project_name, status_counts)
        scene_rows.append(
            {
                "project_slug": project_slug,
                "project_name": project_name,
                "scene_tags": manual_override.get("scene_tags", default_scene_tags),
                "needs_manual_review": manual_override.get("needs_manual_review", not bool(manual_override)),
                "curation_source": manual_override.get(
                    "curation_source",
                    "manual_v1" if manual_override else "heuristic_v1",
                ),
                "notes": manual_override.get("notes", ""),
            }
        )

        bucket = deterministic_bucket(project_slug)
        if bucket < 0.18:
            holdout_projects.append(project_slug)
        elif bucket < 0.33:
            val_projects.append(project_slug)
        else:
            train_projects.append(project_slug)

        total_pages += len(pages)
        reviewed_pages += project_reviewed_pages
        manual_pages += project_manual_pages

    safe_write_json(
        data_root / "staging" / "imports" / f"{dataset_slug}.json",
        {
            "dataset_slug": dataset_slug,
            "source_root": str(source_root),
            "raw_alias": str(raw_alias),
            "project_count": len(project_summaries),
            "page_count": total_pages,
            "reviewed_pages": reviewed_pages,
            "pages_with_manual_quad": manual_pages,
            "projects": import_rows,
        },
    )

    for summary in project_summaries:
        safe_write_json(
            data_root / "curated" / "projects" / f"{summary.project_slug}.json",
            {
                "dataset_slug": dataset_slug,
                "project_slug": summary.project_slug,
                "project_name": summary.project_name,
                "source_dir": summary.source_dir,
                "project_file": summary.project_file,
                "pages": summary.pages,
                "reviewed_pages": summary.reviewed_pages,
                "pages_with_manual_quad": summary.pages_with_manual_quad,
                "pages_with_export_preview": summary.pages_with_export_preview,
                "best_method_counts": summary.best_method_counts,
                "status_counts": summary.status_counts,
                "export_dirs": summary.export_dirs,
                "pdf_files": summary.pdf_files,
            },
        )

    safe_write_jsonl(data_root / "curated" / "annotations" / f"{dataset_slug}_pages.jsonl", page_rows)
    safe_write_json(
        data_root / "curated" / "scenes" / f"{dataset_slug}_scene_taxonomy.json",
        {"scene_tags": SCENE_TAXONOMY},
    )
    safe_write_json(
        data_root / "curated" / "scenes" / f"{dataset_slug}_project_scene_map.json",
        {"projects": scene_rows},
    )

    split_payload = {
        "dataset_slug": dataset_slug,
        "strategy": "deterministic_project_hash_v1",
        "train_projects": train_projects,
        "val_projects": val_projects,
        "holdout_projects": holdout_projects,
    }
    safe_write_json(data_root / "splits" / "cross_project" / f"{dataset_slug}_split_v1.json", split_payload)
    safe_write_json(data_root / "splits" / "holdout" / f"{dataset_slug}_holdout_v1.json", split_payload)

    safe_write_json(
        training_root / "registry" / f"{dataset_slug}.json",
        {
            "dataset_slug": dataset_slug,
            "source_root": str(source_root),
            "raw_alias": str(raw_alias),
            "project_count": len(project_summaries),
            "page_count": total_pages,
            "reviewed_pages": reviewed_pages,
            "pages_with_manual_quad": manual_pages,
            "split_file": str(data_root / "splits" / "cross_project" / f"{dataset_slug}_split_v1.json"),
            "annotation_file": str(data_root / "curated" / "annotations" / f"{dataset_slug}_pages.jsonl"),
            "generated_at_plan": "2026-03-25-dataset-organization",
        },
    )

    print(
        json.dumps(
            {
                "dataset_slug": dataset_slug,
                "project_count": len(project_summaries),
                "page_count": total_pages,
                "reviewed_pages": reviewed_pages,
                "pages_with_manual_quad": manual_pages,
                "train_projects": len(train_projects),
                "val_projects": len(val_projects),
                "holdout_projects": len(holdout_projects),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
