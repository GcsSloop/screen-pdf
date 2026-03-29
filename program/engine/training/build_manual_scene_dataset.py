#!/usr/bin/env python3
"""Build a clean manual-scene dataset from a single screen-pdf project."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from supervision_utils import resolve_manual_quad, resolve_selected_candidate, resolve_supervision_quad
from training.build_page_level_split import build_page_level_split


def _safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _create_or_refresh_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() and Path(os.readlink(link_path)).resolve() == target_path.resolve():
        return
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            link_path.unlink()
        else:
            raise RuntimeError(f"{link_path} exists and is not a symlink")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path)


def _slugify_project(index: int) -> str:
    return f"project_{index:03d}"


def _page_split_label(page: dict[str, Any]) -> str:
    failure_tags = {str(tag) for tag in (page.get("failureTags") or [])}
    difficulty_bucket = str(page.get("difficultyBucket") or "").strip()
    if "corner_out_of_frame" in failure_tags:
        return "corner_out_of_frame"
    if "strong_perspective" in failure_tags or difficulty_bucket == "hard":
        return "strong_perspective"
    return "natural"


def _annotation_row(dataset_slug: str, project_slug: str, project_name: str, page: dict[str, Any]) -> dict[str, Any]:
    supervision_quad, supervision_source = resolve_supervision_quad(page)
    manual_quad, manual_source = resolve_manual_quad(page)
    selected_candidate = resolve_selected_candidate(page) or {}
    return {
        "dataset_slug": dataset_slug,
        "project_slug": project_slug,
        "project_name": project_name,
        "page_id": page.get("id"),
        "page_name": page.get("name"),
        "image_path": page.get("path"),
        "thumb_path": page.get("thumbPath"),
        "preview_path": page.get("previewPath"),
        "status": page.get("status") or "unknown",
        "confidence": page.get("confidence"),
        "best_method": page.get("bestMethod") or "unknown",
        "selected_candidate_index": page.get("selectedCandidateIndex"),
        "candidate_count": len(page.get("candidates") or []),
        "manual_quad": supervision_quad,
        "original_manual_quad": page.get("manualQuad"),
        "resolved_manual_quad": manual_quad,
        "manual_source": manual_source,
        "active_quad": page.get("activeQuad"),
        "supervision_source": supervision_source,
        "has_manual_quad": bool(manual_quad),
        "selected_candidate_source": selected_candidate.get("source"),
        "selected_candidate_method": selected_candidate.get("method"),
        "selected_candidate_model_id": selected_candidate.get("modelId") or selected_candidate.get("model_id"),
        "details": page.get("details") or {},
    }


def build_manual_scene_dataset(
    project_file: Path,
    dataset_slug: str,
    repo_root: Path,
    val_count: int = 4,
    holdout_count: int = 4,
) -> dict[str, Any]:
    project_payload = json.loads(project_file.read_text(encoding="utf-8"))
    project_root = project_file.parent
    project_slug = _slugify_project(1)
    project_name = str(project_payload.get("name") or project_root.name)
    pages = list(project_payload.get("pages") or [])

    raw_alias = repo_root / "data" / "raw" / dataset_slug
    _create_or_refresh_symlink(raw_alias, project_root)

    page_rows = [_annotation_row(dataset_slug, project_slug, project_name, page) for page in pages]
    supervision_pages = [row for row in page_rows if row.get("manual_quad")]
    stratify_labels = {
        f"{dataset_slug}:{project_slug}:{page.get('id')}": _page_split_label(page)
        for page, row in zip(pages, page_rows, strict=True)
        if row.get("manual_quad")
    }
    split_payload = build_page_level_split(
        page_rows,
        dataset_slug=dataset_slug,
        target_counts={
            "train": len(supervision_pages) - val_count - holdout_count,
            "val": val_count,
            "holdout": holdout_count,
        },
        stratify_labels=stratify_labels,
    )

    import_payload = {
        "dataset_slug": dataset_slug,
        "source_root": str(project_root),
        "raw_alias": str(raw_alias),
        "project_count": 1,
        "page_count": len(page_rows),
        "reviewed_pages": sum(1 for row in page_rows if row.get("status") == "reviewed"),
        "pages_with_supervision_quad": len(supervision_pages),
        "pages_with_manual_quad": sum(1 for row in page_rows if row.get("original_manual_quad")),
        "projects": [
            {
                "dataset_slug": dataset_slug,
                "project_slug": project_slug,
                "project_name": project_name,
                "source_dir": str(project_root),
                "project_file": str(project_file),
                "pages": len(page_rows),
                "reviewed_pages": sum(1 for row in page_rows if row.get("status") == "reviewed"),
                "pages_with_supervision_quad": len(supervision_pages),
                "pages_with_manual_quad": sum(1 for row in page_rows if row.get("original_manual_quad")),
            }
        ],
    }
    _safe_write_json(repo_root / "data" / "staging" / "imports" / f"{dataset_slug}.json", import_payload)
    _safe_write_jsonl(repo_root / "data" / "curated" / "annotations" / f"{dataset_slug}_pages.jsonl", page_rows)
    _safe_write_json(
        repo_root / "data" / "curated" / "projects" / f"{project_slug}.json",
        {
            "dataset_slug": dataset_slug,
            "project_slug": project_slug,
            "project_name": project_name,
            "source_dir": str(project_root),
            "project_file": str(project_file),
            "pages": len(page_rows),
            "reviewed_pages": sum(1 for row in page_rows if row.get("status") == "reviewed"),
            "pages_with_supervision_quad": len(supervision_pages),
            "pages_with_manual_quad": sum(1 for row in page_rows if row.get("original_manual_quad")),
        },
    )
    cross_project_split = repo_root / "data" / "splits" / "cross_project" / f"{dataset_slug}_split_manual_only_v1.json"
    holdout_split = repo_root / "data" / "splits" / "holdout" / f"{dataset_slug}_holdout_manual_only_v1.json"
    _safe_write_json(cross_project_split, split_payload)
    _safe_write_json(holdout_split, split_payload)
    _safe_write_json(
        repo_root / "training" / "registry" / f"{dataset_slug}.json",
        {
            "dataset_slug": dataset_slug,
            "source_root": str(project_root),
            "raw_alias": str(raw_alias),
            "project_count": 1,
            "page_count": len(page_rows),
            "reviewed_pages": sum(1 for row in page_rows if row.get("status") == "reviewed"),
            "pages_with_supervision_quad": len(supervision_pages),
            "pages_with_manual_quad": sum(1 for row in page_rows if row.get("original_manual_quad")),
            "split_file": str(cross_project_split),
            "annotation_file": str(repo_root / "data" / "curated" / "annotations" / f"{dataset_slug}_pages.jsonl"),
            "generated_at_plan": "2026-03-26-zhongjiao-generalization",
        },
    )

    return {
        "dataset_slug": dataset_slug,
        "page_count": len(page_rows),
        "manual_pages": len(supervision_pages),
        "train_pages": len(split_payload["train_page_ids"]),
        "val_pages": len(split_payload["val_page_ids"]),
        "holdout_pages": len(split_payload["holdout_page_ids"]),
        "split_file": str(cross_project_split),
        "annotation_file": str(repo_root / "data" / "curated" / "annotations" / f"{dataset_slug}_pages.jsonl"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-file", required=True)
    parser.add_argument("--dataset-slug", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--holdout-count", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_manual_scene_dataset(
        project_file=Path(args.project_file).resolve(),
        dataset_slug=args.dataset_slug,
        repo_root=Path(args.repo_root).resolve(),
        val_count=int(args.val_count),
        holdout_count=int(args.holdout_count),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
