"""Register the user's refined corner labels for the 2026-03-06 white-mix project.

The 2026-03-06 杭州智能交通年会「白屏混合」project was re-corrected by the user.
All 43 project pages exist already; 42 of them carry manual quads in the
current training snapshot and `IMG_7018` is the photo that duplicates
`2026-03-06 中交机电/IMG_7018` and is excluded from `conference_v1` since the
duplicate audit, so this revision changes labels only: no page is added,
removed, or re-assigned a role.

Every moved point stays below 1% of the quad's axis-aligned diagonal, so the
revision cannot flip the frozen 1%-hit metric by construction; it only makes
the training target and the development truth slightly more accurate.

This script writes:

* data/curated/annotations/202603-white-mix_pages_v2.jsonl
* data/splits/cross_project/202603-white-mix_v2.json
* training/registry/202603-white-mix_v2.json
* training/datasets/screen_corner_single_all_manual_3_20fine_whitemix_v1/

The new snapshot is a copy of the latest manual snapshot with only the 42
white-mix quads changed; membership, roles, replay weights and quarantine
sidecars are otherwise byte-for-byte inherited.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from perspective_detect import order_points
from supervision_utils import resolve_manual_quad
from training.screen_corner_ablation import fingerprint, read_rows, write_json, write_rows

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROJECT = Path(
    "/Users/gcssloop/WorkSpace/AIGC/展会/2026/202503 杭州-第十五届智能交通市场年会/"
    "2026-03-06 白屏混合/screen-pdf-project.json"
)
BASE = REPO / "training/datasets/screen_corner_single_all_manual_3_20fine_zhongjiao_v1"
SLUG = "202603-white-mix"
ANNOTATION = REPO / f"data/curated/annotations/{SLUG}_pages_v2.jsonl"
SPLIT = REPO / f"data/splits/cross_project/{SLUG}_v2.json"
REGISTRY = REPO / f"training/registry/{SLUG}_v2.json"
SNAPSHOT = REPO / "training/datasets/screen_corner_single_all_manual_3_20fine_whitemix_v1"
OLD_ANNOTATION = REPO / f"data/curated/annotations/{SLUG}_pages.jsonl"
OLD_SPLIT = REPO / f"data/splits/cross_project/{SLUG}_split_manual_only_v1.json"
FROZEN_PAGES = REPO / "data/curated/collections/conference_v1/pages.jsonl"
# Same photo as 2026-03-06 中交机电/IMG_7018; conference_v1 keeps the 中交机电
# copy out and excludes both, so it is neither train nor evaluation here.
EXCLUDED = "IMG_7018"


def _project_rows(rows: list[dict], project: Path) -> list[dict]:
    target = str(project)
    return [row for row in rows if str(Path(row.get("project_path", ""))) == target]


def _diag(quad: np.ndarray) -> float:
    return float(np.linalg.norm(quad.max(0) - quad.min(0)))


def _content_hashes(image: np.ndarray, quad: list[list[float]]) -> tuple[int, int]:
    destination = np.array([[0, 0], [255, 0], [255, 143], [0, 143]], np.float32)
    crop = cv2.warpPerspective(
        image,
        cv2.getPerspectiveTransform(np.asarray(quad, np.float32), destination),
        (256, 144),
    )
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dh = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), "big")
    freq = cv2.dct(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32))[:8, :8]
    ph = int.from_bytes(np.packbits(freq > np.median(freq.flatten()[1:])).tobytes(), "big")
    return dh, ph


def _validate_quad(page_id: str, quad: np.ndarray, width: int, height: int) -> None:
    assert quad.shape == (4, 2), page_id
    assert np.isfinite(quad).all(), page_id
    assert quad.min() >= -0.01, page_id
    assert (quad <= np.array([width, height]) + 0.01).all(), page_id
    assert cv2.isContourConvex(quad.astype(np.float32)), page_id
    assert abs(cv2.contourArea(quad.astype(np.float32))) >= 100, page_id


def build(project_file: Path = DEFAULT_PROJECT, base: Path = BASE) -> dict:
    project_file = project_file.resolve()
    project_root = project_file.parent
    payload = json.loads(project_file.read_text(encoding="utf-8"))
    pages = {page["id"]: page for page in (payload.get("pages") or [])}
    assert pages, "empty project"

    base_all = _project_rows(read_rows(base / "all_manual.jsonl"), project_file)
    base_train = _project_rows(read_rows(base / "train.jsonl"), project_file)
    base_validation = _project_rows(read_rows(base / "validation.jsonl"), project_file)
    base_holdout = _project_rows(read_rows(base / "holdout.jsonl"), project_file)
    base_by_page = {row["page_id"]: row for row in base_all}
    assert len(base_by_page) == len(base_all) == 42, len(base_all)
    assert set(base_by_page) == set(pages) - {EXCLUDED}, set(pages) - set(base_by_page)
    assert len(base_train) == 33 and len(base_validation) == 9 and not base_holdout
    assert len({row["image_path"] for row in base_all}) == len(base_all)

    excluded_reasons = {EXCLUDED: "exact_duplicate_of_2026-03-06_中交机电/IMG_7018; excluded from conference_v1"}
    frozen_ids = {row["page_id"] for row in _project_rows(read_rows(FROZEN_PAGES), project_file)}
    assert frozen_ids == set(base_by_page), (sorted(frozen_ids ^ set(base_by_page)))

    records: list[dict] = []
    changes: list[dict] = []
    deltas: list[float] = []
    for page_id in sorted(base_by_page):
        page = pages[page_id]
        width = int(page["details"]["width"])
        height = int(page["details"]["height"])
        quad, source = resolve_manual_quad(page)
        assert quad, page_id
        new_quad = order_points(np.asarray(quad, np.float64))
        _validate_quad(page_id, new_quad, width, height)
        old_row = base_by_page[page_id]
        old_quad = order_points(np.asarray(old_row["manual_quad"], np.float64))
        image_path = project_root / page["path"]
        assert image_path.is_file(), image_path
        image_sha = fingerprint(image_path)
        assert image_sha == old_row["image_sha256"], page_id
        image = cv2.imread(str(image_path))
        assert image is not None and image.shape[:2] == (height, width), page_id
        dhash, phash = _content_hashes(image, new_quad.tolist())
        diagonal = _diag(new_quad)
        per_point = np.linalg.norm(new_quad - old_quad, axis=1) / diagonal
        deltas.extend(float(x) for x in per_point)
        changes.append(
            {
                "page_id": page_id,
                "split": old_row["split"],
                "previous_quad": old_quad.tolist(),
                "current_quad": new_quad.tolist(),
                "per_point_fraction_of_diagonal": per_point.tolist(),
                "max_fraction_of_diagonal": float(per_point.max()),
                "max_pixel_offset": float(np.linalg.norm(new_quad - old_quad, axis=1).max()),
            }
        )
        records.append(
            {
                "dataset_slug": SLUG,
                "project_slug": "project_001",
                "project_id": old_row["project_id"],
                "project_name": project_root.name,
                "project_path": str(project_file),
                "project_key": str(project_file),
                "collection_id": old_row["collection_id"],
                "collection_name": old_row["collection_name"],
                "page_id": page_id,
                "page_name": page.get("name"),
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "image_sha256": image_sha,
                "manual_quad": new_quad.tolist(),
                "annotation_source": source,
                "annotation_revision": 2,
                "previous_manual_quad": old_quad.tolist(),
                "split": old_row["split"],
                "current_role": old_row["split"],
                "role_evidence": old_row.get("role_evidence", []),
                "scene_tags": old_row.get("scene_tags", []),
                "historical_scene_dataset": old_row.get("historical_scene_dataset"),
                "valid_manual_quad": True,
                "screen_dhash": f"{dhash:016x}",
                "screen_phash": f"{phash:016x}",
                "content_group": old_row["content_group"],
                "label_quality": "manual_fine_refinement",
                "fine_corner_truth": True,
                "annotation_independence": "not_verified",
                "status": page.get("status", "reviewed"),
            }
        )

    delta_array = np.asarray(deltas, np.float64)
    # The refinement is sub-threshold by construction: no point crosses the 1%
    # hit radius, so this revision cannot manufacture metric flips.
    assert delta_array.max() < 0.01, float(delta_array.max())

    # Cross-check against every other registered label set and the frozen
    # collection. Only the two known intra-project near-duplicate pairs
    # (IMG_7040/IMG_7041, IMG_7043/IMG_7044) may trip the pHash flag.
    others: list[dict] = []
    for path in sorted((REPO / "data/curated/annotations").glob("*.jsonl")):
        if path in {ANNOTATION, OLD_ANNOTATION} or "deep_screen_v1" in path.name:
            continue
        for row in read_rows(path):
            row["_file"] = path.name
            others.append(row)
    for row in read_rows(FROZEN_PAGES):
        row["_file"] = "conference_v1/pages.jsonl"
        others.append(row)
    exact: list[dict] = []
    near: list[dict] = []
    for row in records:
        for other in others:
            if str(Path(other.get("project_path", ""))) == str(project_file):
                continue
            if other.get("image_sha256") and row["image_sha256"] == other["image_sha256"]:
                exact.append({"new": row["page_id"], "old": other.get("page_id"), "old_file": other["_file"]})
                continue
            if not other.get("screen_dhash") or not other.get("screen_phash"):
                continue
            dh = (int(row["screen_dhash"], 16) ^ int(other["screen_dhash"], 16)).bit_count()
            ph = (int(row["screen_phash"], 16) ^ int(other["screen_phash"], 16)).bit_count()
            if dh <= 8 or ph <= 8:
                near.append({"new": row["page_id"], "old": other.get("page_id"), "old_file": other["_file"],
                             "dh": dh, "ph": ph})
    assert not exact, exact
    assert not near, near

    # The v1 annotation file and the base snapshot must describe the same
    # labels (float-32 rounding only), otherwise the delta above would be
    # measured against the wrong previous revision.
    legacy = {row["page_id"]: row for row in read_rows(OLD_ANNOTATION)}
    assert set(legacy) == set(pages), sorted(set(legacy) ^ set(pages))
    legacy_delta = max(
        float(np.abs(order_points(np.asarray(legacy[page_id]["manual_quad"], np.float64))
                     - order_points(np.asarray(base_by_page[page_id]["manual_quad"], np.float64))).max())
        for page_id in base_by_page
    )
    assert legacy_delta < 0.01, legacy_delta

    split_counts = dict(Counter(r["split"] for r in records))
    assert split_counts == {"train": 33, "validation": 9}, split_counts
    metadata = {
        "dataset_slug": SLUG,
        "source_root": str(project_root),
        "source_project": str(project_file),
        "source_project_sha256": fingerprint(project_file),
        "annotation_file": str(ANNOTATION),
        "split_file": str(SPLIT),
        "scene_tags": ["conference_hall_led_screen", "white_slides_mixed", "audience_foreground",
                       "speaker_overlap", "strong_perspective", "mixed_screen_scale"],
        "page_count": len(pages),
        "manual_page_count": len(records),
        "split_counts": split_counts,
        "annotation_revision": 2,
        "previous_annotation_file": str(OLD_ANNOTATION),
        "previous_split_file": str(OLD_SPLIT),
        "previous_split_counts": {"train": 35, "validation": 4, "holdout": 4},
        "role_policy": ("roles are inherited unchanged from the conference_v1 frozen assignment as carried by the "
                        "base manual snapshot (33 train / 9 validation / 0 holdout); the legacy project-level split "
                        "(35/4/4) is historical and is not re-applied"),
        "excluded_pages": excluded_reasons,
        "changed_existing_pages": len(changes),
        "changed_existing_points": len(deltas),
        "moved_points": int((delta_array > 0).sum()),
        "unchanged_points": int((delta_array == 0).sum()),
        "unchanged_pages": [c["page_id"] for c in changes if c["max_fraction_of_diagonal"] == 0],
        "max_fraction_of_diagonal": float(delta_array.max()),
        "mean_fraction_of_diagonal": float(delta_array.mean()),
        "median_fraction_of_diagonal": float(np.median(delta_array)),
        "p90_fraction_of_diagonal": float(np.percentile(delta_array, 90)),
        "points_over_1_percent": int((delta_array > 0.01).sum()),
        "pages_over_1_percent": sum(1 for c in changes if c["max_fraction_of_diagonal"] > 0.01),
        "label_quality": "manual_fine_refinement",
        "fine_corner_truth": True,
        "annotation_independence": "not_verified",
        "label_quality_note": ("user-reported refined manual corners; every point moved by less than 0.87% of the "
                              "quad diagonal, so the change is a fine correction inside the 1% hit radius rather "
                              "than a re-localization"),
        "status": "registered_not_trained",
        "training_status": "not_started",
        "training_run": None,
        "source_untouched": True,
        "objective": "refresh 2026-03-06 white-mix corner labels with the user's refined manual corners",
    }

    outputs = [ANNOTATION, SPLIT, REGISTRY, SNAPSHOT]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"{path} already exists; append-only revision")

    write_rows(ANNOTATION, records)
    write_json(SPLIT, {
        "metadata": metadata,
        "train": [r for r in records if r["split"] == "train"],
        "validation": [r for r in records if r["split"] == "validation"],
        "holdout": [],
    })
    write_json(REGISTRY, metadata)

    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    for name in ["holdout", "new_train", "new_validation", "new_holdout", "quarantined_train"]:
        source = base / f"{name}.jsonl"
        (SNAPSHOT / f"{name}.jsonl").write_bytes(source.read_bytes())

    updates = {row["page_id"]: row for row in records}

    def patch(rows: list[dict]) -> tuple[list[dict], int]:
        output, touched = [], 0
        for row in rows:
            if str(Path(row.get("project_path", ""))) == str(project_file):
                update = updates.get(row["page_id"])
                assert update is not None, row["page_id"]
                row = dict(row, manual_quad=update["manual_quad"], annotation_revision=2,
                           screen_dhash=update["screen_dhash"], screen_phash=update["screen_phash"])
                touched += 1
            output.append(row)
        return output, touched

    train, touched_train = patch(read_rows(base / "train.jsonl"))
    validation, touched_validation = patch(read_rows(base / "validation.jsonl"))
    all_manual, touched_all = patch(read_rows(base / "all_manual.jsonl"))
    assert len(train) == len(read_rows(base / "train.jsonl")), "membership must not change"
    assert len(validation) == len(read_rows(base / "validation.jsonl"))
    assert len(all_manual) == len(read_rows(base / "all_manual.jsonl"))
    assert touched_train == 33 and touched_validation == 9 and touched_all == 42, (
        touched_train, touched_validation, touched_all)
    write_rows(SNAPSHOT / "train.jsonl", train)
    write_rows(SNAPSHOT / "validation.jsonl", validation)
    write_rows(SNAPSHOT / "all_manual.jsonl", all_manual)
    train_paths = {row["image_path"] for row in train}
    for name in ["validation", "holdout", "new_validation", "new_holdout"]:
        assert not train_paths & {row["image_path"] for row in read_rows(SNAPSHOT / f"{name}.jsonl")}, name

    audit = {
        "revision": "white_mix_corner_revision_v2",
        "base_snapshot": str(base),
        "source_project": str(project_file),
        "source_project_sha256": fingerprint(project_file),
        "annotation_file": str(ANNOTATION),
        "split_file": str(SPLIT),
        "page_count": len(pages),
        "manual_page_count": len(records),
        "split_counts": split_counts,
        "excluded_pages": excluded_reasons,
        "label_change": {
            "changed_pages": len(changes),
            "changed_points": len(deltas),
            "unchanged_pages": metadata["unchanged_pages"],
            "max_fraction_of_diagonal": metadata["max_fraction_of_diagonal"],
            "mean_fraction_of_diagonal": metadata["mean_fraction_of_diagonal"],
            "p90_fraction_of_diagonal": metadata["p90_fraction_of_diagonal"],
            "points_over_1_percent": metadata["points_over_1_percent"],
            "per_page": changes,
        },
        "duplicate_audit": {"exact_byte_duplicates": 0, "near_duplicates": 0,
                            "verdict": "no exact or pHash/dHash <=8 duplicate against any other label set or conference_v1"},
        "entry_counts": {name: {"before": len(read_rows(base / f"{name}.jsonl")),
                                "after": len(read_rows(SNAPSHOT / f"{name}.jsonl"))}
                         for name in ["train", "validation", "holdout", "all_manual", "quarantined_train"]},
        "changed_entries": {"train": touched_train, "validation": touched_validation, "all_manual": touched_all},
        "protected_inputs_sha256": {
            "base_train": fingerprint(base / "train.jsonl"),
            "base_validation": fingerprint(base / "validation.jsonl"),
            "base_all_manual": fingerprint(base / "all_manual.jsonl"),
            "conference_pages": fingerprint(FROZEN_PAGES),
            "source_project": fingerprint(project_file),
            "v1_annotation": fingerprint(OLD_ANNOTATION),
        },
        "status": "registered_not_trained",
        "notes": [
            "No page is added or removed: the 42 already-registered pages only receive refined quads.",
            "IMG_7018 stays excluded; it is the same photo as 中交机电 IMG_7018 and conference_v1 keeps neither copy.",
            "Roles are unchanged (33 train / 9 validation / 0 holdout), so the white-mix development pages are reused, not fresh.",
            "All 168 points moved less than 0.87% of the diagonal, i.e. inside the 1% hit radius.",
        ],
    }
    write_json(SNAPSHOT / "audit.json", audit)
    write_json(SNAPSHOT / "summary.json", {
        "revision": "white_mix_corner_revision_v2",
        "base_snapshot": str(base),
        "train_entries": len(train),
        "validation_pages": len(validation),
        "holdout_pages": len(read_rows(SNAPSHOT / "holdout.jsonl")),
        "all_manual_pages": len(all_manual),
        "quarantined_train_entries": len(read_rows(SNAPSHOT / "quarantined_train.jsonl")),
        "split_counts": split_counts,
        "moved_points": metadata["moved_points"],
        "max_fraction_of_diagonal": metadata["max_fraction_of_diagonal"],
        "mean_fraction_of_diagonal": metadata["mean_fraction_of_diagonal"],
        "status": "registered_not_trained",
    })
    (SNAPSHOT / "README.md").write_text(
        "# 白屏混合角点精修训练快照（v2）\n\n"
        f"基线：`{base}`。本快照只替换 2026-03-06 杭州智能交通年会「白屏混合」42 个页面的人工四角，"
        f"`train` 仍为 {len(train)} 条、`validation` {len(validation)} 页、`all_manual` {len(all_manual)} 页，"
        "成员、用途、回放与隔离 sidecar 逐字节继承，不新增也不删除页面。\n\n"
        f"标注变化：42 页 168 个角点重新登记，其中 {int((delta_array > 0).sum())} 个点有位移"
        f"（`IMG_6941` 的 4 个点未变），平均 {100 * metadata['mean_fraction_of_diagonal']:.3f}%、"
        f"P90 {100 * metadata['p90_fraction_of_diagonal']:.3f}%、最大 {100 * metadata['max_fraction_of_diagonal']:.3f}% 对角线，"
        "没有任何角点超过 1% 命中半径；`IMG_6941` 未变。因此这次修订是阈值内的精细修正，"
        "不能靠指标翻转证明收益。\n\n"
        "`IMG_7018` 与 `2026-03-06 中交机电/IMG_7018` 同图，`conference_v1` 两份都排除，本 revision 保持排除。"
        "用途沿用冻结的 33 训练 / 9 开发 / 0 回归，因此 9 页开发不是盲测。"
        "状态 `registered_not_trained`，未训练、未改正式 runtime、未打包。\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "pages": len(records),
        "split_counts": split_counts,
        "changed_pages": len(changes),
        "changed_points": len(deltas),
        "max_pct": round(100 * metadata["max_fraction_of_diagonal"], 3),
        "mean_pct": round(100 * metadata["mean_fraction_of_diagonal"], 3),
        "snapshot": str(SNAPSHOT),
        "train_entries": len(train),
        "all_manual": len(all_manual),
    }, ensure_ascii=False))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--base", type=Path, default=BASE)
    build(parser.parse_args().project, parser.parse_args().base)
