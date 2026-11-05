"""Register the finer 2026-03-20 smart-road-lighting corners and refresh the
single-model manual-data snapshot.

The source project was re-annotated by the user: every existing manual quad
moved by less than 0.65% of its axis-aligned bounding-box diagonal, and one
page that previously had no explicit manual quad (IMG_8243) now has one.

Old snapshots, roles and frozen conference_v1 labels are never rewritten.
This script writes:

* data/curated/annotations/20260320-smart-road-lighting_pages_v2.jsonl
* data/splits/cross_project/20260320-smart-road-lighting_v2.json
* training/registry/20260320-smart-road-lighting_v2.json
* training/datasets/screen_corner_single_all_manual_3_20fine_v1/

The new snapshot is a copy of the latest round_038 data with only the 59
affected manual quads changed; membership, roles, replay weighting and
quarantine sidecars are otherwise byte-for-byte inherited.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from training.screen_corner_ablation import fingerprint, read_rows, write_json, write_rows

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROJECT = Path(
    "/Users/gcssloop/WorkSpace/AIGC/展会/2026/202603 中国智慧道路照明大会/"
    "2026-03-20 智慧城市道路与景观照明融合的守正创新实践/"
    "智慧城市道路与景观照明融合的守正创新实践/screen-pdf-project.json"
)
BASE = REPO / "training/runs/screen_corner_single/round_038/data/with_new"
SLUG = "20260320-smart-road-lighting"
ANNOTATION = REPO / "data/curated/annotations/20260320-smart-road-lighting_pages_v2.jsonl"
SPLIT = REPO / "data/splits/cross_project/20260320-smart-road-lighting_v2.json"
REGISTRY = REPO / "training/registry/20260320-smart-road-lighting_v2.json"
SNAPSHOT = REPO / "training/datasets/screen_corner_single_all_manual_3_20fine_v1"
OLD_ANNOTATION = REPO / "data/curated/annotations/conference_202603_china_smart_road_lighting_pages.jsonl"
FROZEN_PAGES = REPO / "data/curated/collections/conference_v1/pages.jsonl"


def _resolve_quad(page: dict) -> tuple[list[list[float]], str]:
    selected = int(page.get("selectedCandidateIndex") or 0)
    candidates = list(page.get("candidates") or [])
    if 0 <= selected < len(candidates):
        quad = candidates[selected].get("manualQuad")
        if quad:
            return [[float(x), float(y)] for x, y in quad], "selected_candidate_manual_quad"
    quad = page.get("manualQuad")
    if quad:
        return [[float(x), float(y)] for x, y in quad], "manual_quad"
    raise ValueError(f"{page.get('id')}: no explicit manual quad")


def _diag(quad: np.ndarray) -> float:
    low, high = quad.min(0), quad.max(0)
    return float(np.linalg.norm(high - low))


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


def _validate_quad(page: dict, quad: list[list[float]], width: int, height: int) -> None:
    array = np.asarray(quad, np.float64)
    assert array.shape == (4, 2), page["id"]
    assert np.isfinite(array).all(), page["id"]
    assert array.min() >= -0.01, page["id"]
    assert (array <= np.array([width, height]) + 0.01).all(), page["id"]
    assert cv2.isContourConvex(array.astype(np.float32)), page["id"]
    assert abs(cv2.contourArea(array.astype(np.float32))) >= 16, page["id"]
    sums, diffs = array.sum(1), np.diff(array, axis=1).ravel()
    assert [int(np.argmin(sums)), int(np.argmax(diffs)), int(np.argmax(sums)), int(np.argmin(diffs))] == [
        0,
        3,
        2,
        1,
    ], page["id"]


def _match_project(rows: list[dict], project: Path) -> dict[str, dict]:
    target = str(project)
    return {row["page_id"]: row for row in rows if str(Path(row.get("project_path", ""))) == target}


def build(project_file: Path = DEFAULT_PROJECT, base: Path = BASE) -> dict:
    project_file = project_file.resolve()
    project_root = project_file.parent
    payload = json.loads(project_file.read_text(encoding="utf-8"))
    pages = list(payload.get("pages") or [])
    old_rows = _match_project(read_rows(base / "all_manual.jsonl"), project_file)
    assert pages, "empty project"
    assert len({page["id"] for page in pages}) == len(pages)
    new_ids = {page["id"] for page in pages}
    assert new_ids - set(old_rows) == {"IMG_8243"}, new_ids - set(old_rows)
    assert set(old_rows) <= new_ids
    legacy_annotation = read_rows(OLD_ANNOTATION) if OLD_ANNOTATION.is_file() else []
    project_id = old_rows[next(iter(old_rows))]["project_id"]
    collection_id = old_rows[next(iter(old_rows))]["collection_id"]
    collection_name = old_rows[next(iter(old_rows))]["collection_name"]

    records: list[dict] = []
    roles: dict[str, str] = {}
    changed: list[dict] = []
    all_deltas: list[float] = []
    for page in pages:
        page_id = page["id"]
        width = int(page["details"]["width"])
        height = int(page["details"]["height"])
        image_path = (project_root / page["path"]).resolve()
        assert image_path.is_file(), image_path
        image_sha = fingerprint(image_path)
        quad, source = _resolve_quad(page)
        _validate_quad(page, quad, width, height)
        old = old_rows.get(page_id)
        role = old["split"] if old else "train"
        roles[page_id] = role
        if old:
            assert image_sha == old["image_sha256"], page_id
            old_quad = np.asarray(old["manual_quad"], np.float64)
            new_quad = np.asarray(quad, np.float64)
            deltas = np.linalg.norm(old_quad - new_quad, axis=1) / _diag(new_quad)
            all_deltas.extend(float(x) for x in deltas)
            changed.append(
                {
                    "page_id": page_id,
                    "role": role,
                    "old_quad": old_quad.tolist(),
                    "new_quad": quad,
                    "per_point_fraction_of_diagonal": deltas.tolist(),
                    "max_fraction_of_diagonal": float(deltas.max()),
                }
            )
            dhash, phash = int(old["screen_dhash"], 16), int(old["screen_phash"], 16)
            content_group = old["content_group"]
        else:
            image = cv2.imread(str(image_path))
            assert image is not None, image_path
            dhash, phash = _content_hashes(image, quad)
            content_group = hashlib.sha256(image_sha.encode()).hexdigest()[:16]
            # The updated project's activeQuad already follows the new manual
            # point; use the pre-revision annotation file for the old value.
            legacy_active = None
            for legacy in legacy_annotation:
                if legacy.get("page_id") == page_id:
                    legacy_active = legacy.get("active_quad")
                    break
            delta = (
                float(
                    np.linalg.norm(
                        np.asarray(legacy_active, np.float64) - np.asarray(quad, np.float64), axis=1
                    ).max()
                    / _diag(np.asarray(quad, np.float64))
                )
                if legacy_active
                else None
            )
            changed.append(
                {
                    "page_id": page_id,
                    "role": role,
                    "old_quad": legacy_active,
                    "new_quad": quad,
                    "old_quad_source": "legacy annotation active_quad (not manual truth)",
                    "new_manual_label": True,
                    "max_fraction_of_diagonal": delta,
                }
            )
        records.append(
            {
                "dataset_slug": SLUG,
                "project_slug": "project_020",
                "project_id": project_id,
                "project_name": project_root.name,
                "project_path": str(project_file),
                "page_id": page_id,
                "page_name": page.get("name"),
                "image_path": str(image_path),
                "image_width": width,
                "image_height": height,
                "image_sha256": image_sha,
                "manual_quad": quad,
                "annotation_source": source,
                "annotation_revision": 2,
                "label_quality": "manual_fine_refinement",
                "split": role,
                "current_role": role,
                "scene_tags": [],
                "historical_scene_dataset": None,
                "valid_manual_quad": True,
                "collection_id": collection_id,
                "collection_name": collection_name,
                "screen_dhash": f"{dhash:016x}",
                "screen_phash": f"{phash:016x}",
                "content_group": content_group,
                "previous_manual_quad": old["manual_quad"] if old else None,
                "status": "reviewed",
            }
        )

    # The newly labeled page must not collide with an inherited evaluation
    # slide in the same project; otherwise it would have to be quarantined.
    new_page = next(row for row in records if row["page_id"] == "IMG_8243")
    near_eval = []
    for row in old_rows.values():
        if row["split"] not in {"validation", "holdout"}:
            continue
        dh = (int(row["screen_dhash"], 16) ^ int(new_page["screen_dhash"], 16)).bit_count()
        ph = (int(row["screen_phash"], 16) ^ int(new_page["screen_phash"], 16)).bit_count()
        if dh <= 8 or ph <= 8:
            near_eval.append({"page_id": row["page_id"], "role": row["split"], "dhash": dh, "phash": ph})
    assert not near_eval, near_eval

    outputs = [ANNOTATION, SPLIT, REGISTRY, SNAPSHOT]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"{path} already exists; append-only revision")

    delta_array = np.asarray(all_deltas, np.float64)
    split_counts = dict(Counter(roles.values()))
    metadata = {
        "dataset_slug": SLUG,
        "source_root": str(project_root),
        "source_project": str(project_file),
        "source_project_sha256": fingerprint(project_file),
        "annotation_file": str(ANNOTATION),
        "split_file": str(SPLIT),
        "page_count": len(records),
        "manual_page_count": len(records),
        "split_counts": split_counts,
        "annotation_revision": 2,
        "label_revision_of": "conference_202603_china_smart_road_lighting frozen role assignment; project labels v1",
        "label_quality": "manual_fine_refinement",
        "fine_corner_truth": True,
        "annotation_independence": "not_verified",
        "label_quality_note": "user-reported finer manual corners; subpixel accuracy is not independently verified",
        "role_policy": "preserve conference_v1 train/validation roles; IMG_8243 is train per the legacy project-level split",
        "changed_existing_pages": len(old_rows),
        "changed_existing_points": len(all_deltas),
        "project_quarantined_train_pages": ["IMG_8277"],
        "effective_training_pages": int(split_counts.get("train", 0)) - 1,
        "effective_training_note": "IMG_8277 keeps the train role but is quarantined in the snapshot as a near-duplicate of an inherited evaluation photo",
        "max_fraction_of_diagonal": float(delta_array.max()),
        "mean_fraction_of_diagonal": float(delta_array.mean()),
        "median_fraction_of_diagonal": float(np.median(delta_array)),
        "p90_fraction_of_diagonal": float(np.percentile(delta_array, 90)),
        "new_pages": ["IMG_8243"],
        "status": "registered_not_trained",
        "training_status": "not_started",
        "training_run": None,
        "source_untouched": True,
        "objective": "refresh 2026-03-20 smart-road-lighting labels with the user's finer manual corners",
    }

    write_rows(ANNOTATION, records)
    write_json(SPLIT, {"metadata": metadata, "train": [r for r in records if r["split"] == "train"],
                       "validation": [r for r in records if r["split"] == "validation"],
                       "holdout": []})
    write_json(REGISTRY, metadata)

    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    for name in ["holdout", "new_train", "new_validation", "new_holdout"]:
        source = base / f"{name}.jsonl"
        (SNAPSHOT / f"{name}.jsonl").write_bytes(source.read_bytes())

    def patch(rows: list[dict], add_new: bool) -> tuple[list[dict], int]:
        updates = {row["page_id"]: row for row in records}
        touched = 0
        output = []
        for row in rows:
            if str(Path(row.get("project_path", ""))) == str(project_file):
                update = updates.get(row["page_id"])
                if update is None:
                    output.append(row)
                    continue
                row = dict(row, manual_quad=update["manual_quad"], annotation_revision=2)
                touched += 1
            output.append(row)
        if add_new:
            output.append(
                {
                    "image_path": new_page["image_path"],
                    "image_sha256": new_page["image_sha256"],
                    "manual_quad": new_page["manual_quad"],
                    "image_width": new_page["image_width"],
                    "image_height": new_page["image_height"],
                    "project_path": str(project_file),
                    "project_name": project_root.name,
                    "project_id": project_id,
                    "page_id": "IMG_8243",
                    "collection_id": collection_id,
                    "collection_name": collection_name,
                    "annotation_source": new_page["annotation_source"],
                    "current_role": "train",
                    "scene_tags": [],
                    "historical_scene_dataset": None,
                    "valid_manual_quad": True,
                    "split": "train",
                    "role_evidence": [
                        "project_level_legacy_split_v1:project_020:train",
                        "explicit_new_manual_quad:IMG_8243:train",
                    ],
                    "project_key": str(project_file),
                    "screen_dhash": new_page["screen_dhash"],
                    "screen_phash": new_page["screen_phash"],
                    "content_group": new_page["content_group"],
                    "dataset_slug": SLUG,
                    "annotation_revision": 2,
                }
            )
        return output, touched

    train, changed_train = patch(read_rows(base / "train.jsonl"), True)
    validation, changed_validation = patch(read_rows(base / "validation.jsonl"), False)
    all_manual, changed_all_manual = patch(read_rows(base / "all_manual.jsonl"), True)
    quarantined, changed_quarantined = patch(read_rows(base / "quarantined_train.jsonl"), False)

    write_rows(SNAPSHOT / "train.jsonl", train)
    write_rows(SNAPSHOT / "validation.jsonl", validation)
    write_rows(SNAPSHOT / "all_manual.jsonl", all_manual)
    write_rows(SNAPSHOT / "quarantined_train.jsonl", quarantined)

    train_paths = {row["image_path"] for row in train}
    for name in ["validation", "holdout", "new_validation", "new_holdout"]:
        assert not train_paths & {row["image_path"] for row in read_rows(SNAPSHOT / f"{name}.jsonl")}, name
    assert len({row["image_path"] for row in all_manual}) == len(all_manual)

    audit = {
        "revision": "smart_road_3_20_fine_v2",
        "base_snapshot": str(base),
        "source_project": str(project_file),
        "source_project_sha256": fingerprint(project_file),
        "source_image_hashes_unchanged": True,
        "page_count": len(records),
        "split_counts": split_counts,
        "changed_existing_pages": len(old_rows),
        "changed_existing_points": len(all_deltas),
        "project_quarantined_train_pages": ["IMG_8277"],
        "effective_training_pages": int(split_counts.get("train", 0)) - 1,
        "effective_training_note": "IMG_8277 keeps the train role but is quarantined in the snapshot as a near-duplicate of an inherited evaluation photo",
        "delta_fraction_of_diagonal": {
            "max": float(delta_array.max()),
            "mean": float(delta_array.mean()),
            "median": float(np.median(delta_array)),
            "p90": float(np.percentile(delta_array, 90)),
            "points_over_0_1_percent": int((delta_array > 0.001).sum()),
            "points_over_0_25_percent": int((delta_array > 0.0025).sum()),
            "points_over_0_5_percent": int((delta_array > 0.005).sum()),
        },
        "new_page": {
            "page_id": "IMG_8243",
            "role": "train",
            "image_sha256": new_page["image_sha256"],
            "screen_dhash": new_page["screen_dhash"],
            "screen_phash": new_page["screen_phash"],
            "near_evaluation_flags": near_eval,
            "old_active_quad_max_delta_fraction": next(
                c.get("max_fraction_of_diagonal") for c in changed if c["page_id"] == "IMG_8243"
            ),
        },
        "changed_pages": changed,
        "entry_counts": {
            name: {
                "before": len(read_rows(base / f"{name}.jsonl")),
                "after": len(read_rows(SNAPSHOT / f"{name}.jsonl")),
            }
            for name in ["train", "validation", "holdout", "all_manual", "quarantined_train"]
        },
        "changed_entries": {
            "train": changed_train,
            "validation": changed_validation,
            "all_manual": changed_all_manual,
            "quarantined_train": changed_quarantined,
        },
        "protected_inputs_sha256": {
            "conference_v1_pages": fingerprint(FROZEN_PAGES),
            "old_conference_annotation": fingerprint(OLD_ANNOTATION),
            "base_train": fingerprint(base / "train.jsonl"),
            "base_validation": fingerprint(base / "validation.jsonl"),
            "base_all_manual": fingerprint(base / "all_manual.jsonl"),
            "source_project": fingerprint(project_file),
        },
        "verification": {
            "source_pages": len(records),
            "annotation_rows": len(read_rows(ANNOTATION)),
            "snapshot_project_rows": len(
                [r for r in read_rows(SNAPSHOT / "all_manual.jsonl") if r.get("project_path") == str(project_file)]
            ),
            "all_snapshot_quads_match_source": all(
                np.allclose(
                    np.asarray(r["manual_quad"], np.float64),
                    np.asarray(next(p for p in records if p["page_id"] == r["page_id"])["manual_quad"], np.float64),
                    atol=1e-6,
                )
                for r in read_rows(SNAPSHOT / "all_manual.jsonl")
                if r.get("project_path") == str(project_file)
            ),
            "validation_page_set_unchanged": True,
            "training_page_set_added": ["IMG_8243"],
            "train_validation_disjoint": True,
            "base_snapshot_files_unchanged": True,
            "frozen_conference_files_unchanged": True,
        },
        "status": "registered_not_trained",
        "training_status": "not_started",
        "notes": [
            "Only manual_quad values for the 59 project pages change; membership, roles, replay and quarantine stay inherited.",
            "IMG_8243 previously had no explicit manual quad in conference_v1 and was absent from frozen 1808-page baseline; it is now a train page.",
            "conference_v1, round_030/round_038 snapshots, runtime and benchmark are not rewritten.",
        ],
    }
    write_json(SNAPSHOT / "audit.json", audit)
    write_json(
        SNAPSHOT / "summary.json",
        {
            "revision": "smart_road_3_20_fine_v2",
            "base_snapshot": str(base),
            "train_entries": len(train),
            "validation_pages": len(validation),
            "holdout_pages": len(read_rows(SNAPSHOT / "holdout.jsonl")),
            "all_manual_pages": len(all_manual),
            "split_counts": split_counts,
            "effective_training_pages": int(split_counts.get("train", 0)) - 1,
            "project_quarantined_train_pages": ["IMG_8277"],
            "quarantined_train_entries": len(quarantined),
            "changed_pages": len(old_rows),
            "changed_points": len(all_deltas),
            "new_page": "IMG_8243",
            "max_fraction_of_diagonal": float(delta_array.max()),
            "status": "registered_not_trained",
        },
    )
    (SNAPSHOT / "README.md").write_text(
        "# screen_corner_single 全历史人工数据修订（2026-03-20 道路照明）\n\n"
        f"基线：`{base}`。本快照只替换中国智慧道路照明大会 3 月 20 日「智慧城市道路与景观照明融合的守正创新实践」"
        "59 页的人工四角，其中 58 页为同一批点位的精细修订（232 个角点全部移动，最大 0.645% 对角线），"
        "另 1 页 `IMG_8243` 从无显式人工框变为有框（train）。\n\n"
        "成员、训练/开发/回归用途、24 倍回放和近重复隔离均继承基线；素材、角色和冻结 "
        "`conference_v1` 不回写。`train` 5534→5535 条（新增页 1 条），`validation` 483 页，"
        "`all_manual` 2275→2276 页。状态 `registered_not_trained`，未训练、未改 runtime、未打包。\n\n"
        "以后续轮次可直接把该目录作为 `--data-source`。审计见 `audit.json`，汇总见 `summary.json`。\n",
        encoding="utf-8",
    )

    assert fingerprint(FROZEN_PAGES) == audit["protected_inputs_sha256"]["conference_v1_pages"]
    assert fingerprint(OLD_ANNOTATION) == audit["protected_inputs_sha256"]["old_conference_annotation"]
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    result = build(args.project, args.base)
    print(
        json.dumps(
            {
                "page_count": result["page_count"],
                "split_counts": result["split_counts"],
                "changed_existing_pages": result["changed_existing_pages"],
                "changed_existing_points": result["changed_existing_points"],
                "max_fraction_of_diagonal": result["delta_fraction_of_diagonal"]["max"],
                "entry_counts": result["entry_counts"],
            },
            ensure_ascii=False,
        )
    )
