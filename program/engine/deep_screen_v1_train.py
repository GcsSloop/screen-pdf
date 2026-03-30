from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from corner_train import build_corner_heatmaps, select_torch_device
from dataset_benchmark import compute_scene_profile, quad_geometry_metrics, summarize_geometry_metric_rows
from deep_screen_v1_model import DeepScreenV1Net, load_compatible_state_dict
from local_corner_heatmap import build_corner_direction_target, build_corner_visibility_target
from local_corner_refine import build_local_corner_patch_sample
from perspective_detect import detect_best_candidate, order_points
from two_stage_corner_pipeline import GlobalCornerPredictor, LocalCornerMoEPredictor, RoiCornerPredictor, predict_two_stage


DEFAULT_CURATED_ROWS = "data/curated/annotations/conference_202603_china_smart_road_lighting_pages.jsonl"
DEFAULT_SPLIT_FILE = "data/splits/cross_project/conference_202603_china_smart_road_lighting_split_v1.json"
DEFAULT_TEACHER_EXPORT_SUBDIR = "teacher_exports"
DEFAULT_DATASET_SUBDIR = "dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
STRICT_POINT_THRESHOLD = 0.01
STRICT_POINT_TARGET_RATIO = 0.70
MAX_ALLOWED_INFER_MS = 500.0
SCENE_TAGS = (
    "near_color_background",
    "low_contrast_scene",
    "black_frame_scene",
    "bright_screen",
    "border_contact_scene",
)


@dataclass(frozen=True)
class RoundPaths:
    round_root: Path
    data_root: Path
    teacher_export_root: Path
    dataset_root: Path
    checkpoints_root: Path
    reports_root: Path
    artifacts_root: Path
    logs_root: Path


def build_round_paths(round_root: Path) -> RoundPaths:
    round_root = Path(round_root)
    data_root = round_root / "data"
    return RoundPaths(
        round_root=round_root,
        data_root=data_root,
        teacher_export_root=data_root / DEFAULT_TEACHER_EXPORT_SUBDIR,
        dataset_root=data_root / DEFAULT_DATASET_SUBDIR,
        checkpoints_root=round_root / "checkpoints",
        reports_root=round_root / "reports",
        artifacts_root=round_root / "artifacts",
        logs_root=round_root / "logs",
    )


def _artifact_prefix(config: Mapping[str, Any]) -> str:
    return f"{config['public_name']}_{config['round']}"


def _artifact_name(config: Mapping[str, Any], suffix: str, extension: str) -> str:
    return f"{_artifact_prefix(config)}_{suffix}{extension}"


def _student_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("student") or {}


def _loss_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("loss") or {}


def _augmentation_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("augmentation") or {}


def _checkpoint_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("checkpointing") or {}


def _sampling_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("sampling") or {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _snapshot_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _update_ema_state(
    ema_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
    decay: float,
) -> dict[str, torch.Tensor]:
    updated: dict[str, torch.Tensor] = {}
    keep = float(decay)
    blend = 1.0 - keep
    for key, value in model_state.items():
        current = value.detach()
        previous = ema_state.get(key)
        if previous is None:
            updated[key] = current.clone()
            continue
        previous = previous.detach().to(device=current.device)
        if torch.is_floating_point(current) and previous.shape == current.shape:
            updated[key] = previous * keep + current * blend
        else:
            updated[key] = current.clone()
    return updated


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path)


def resolve_teacher_model_paths(repo_root: Path, teacher_runtime_files: Mapping[str, str]) -> dict[str, Path]:
    models_root = repo_root / "models" / "runtime"
    return {alias: models_root / runtime_file for alias, runtime_file in teacher_runtime_files.items()}


@lru_cache(maxsize=4)
def _workspace_image_index(workspace_root: str) -> dict[str, list[Path]]:
    root = Path(workspace_root)
    index: dict[str, list[Path]] = {}
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(candidate.name, []).append(candidate)
    return index


def _resolve_image_path(repo_root: Path, row: Mapping[str, Any]) -> Path:
    raw_path = row.get("image_path")
    page_name = str(row.get("page_name") or "")
    page_id = str(row.get("page_id") or "")
    candidates: list[Path] = []
    if raw_path:
        path = Path(str(raw_path))
        candidates.extend([path, path.parent / page_name, path.parent / path.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    workspace_root = repo_root.parent
    index = _workspace_image_index(str(workspace_root))
    candidates.extend(index.get(page_name, []))
    if page_id:
        for suffix in (".jpeg", ".jpg", ".png"):
            candidates.extend(index.get(f"{page_id}{suffix}", []))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(raw_path or page_name or page_id)


def _load_curated_rows(repo_root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _read_jsonl(_resolve_repo_path(repo_root, str(config.get("curated_rows", DEFAULT_CURATED_ROWS))))


def _load_split_payload(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return _read_json(_resolve_repo_path(repo_root, str(config.get("split_file", DEFAULT_SPLIT_FILE))))


def partition_rows_by_split(rows: list[dict[str, Any]], split_payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    train_projects = set(split_payload.get("train_projects") or [])
    val_projects = set(split_payload.get("val_projects") or [])
    holdout_projects = set(split_payload.get("holdout_projects") or [])
    train_page_ids = set(split_payload.get("train_page_ids") or [])
    val_page_ids = set(split_payload.get("val_page_ids") or [])
    holdout_page_ids = set(split_payload.get("holdout_page_ids") or [])
    partitioned: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "holdout": [], "unassigned": []}
    for row in rows:
        dataset_slug = str(row.get("dataset_slug") or "").strip()
        project_slug = str(row.get("project_slug") or "").strip()
        page_id = str(row.get("page_id") or "").strip()
        page_keys = {f"{project_slug}:{page_id}"}
        if dataset_slug:
            page_keys.add(f"{dataset_slug}:{project_slug}:{page_id}")
        if page_keys & train_page_ids:
            partitioned["train"].append(row)
        elif page_keys & val_page_ids:
            partitioned["val"].append(row)
        elif page_keys & holdout_page_ids:
            partitioned["holdout"].append(row)
        elif project_slug in train_projects:
            partitioned["train"].append(row)
        elif project_slug in val_projects:
            partitioned["val"].append(row)
        elif project_slug in holdout_projects:
            partitioned["holdout"].append(row)
        else:
            partitioned["unassigned"].append(row)
    return partitioned


def _normalize_quad(quad: list[list[float]] | np.ndarray, width: int, height: int) -> np.ndarray:
    points = order_points(np.array(quad, dtype=np.float32))
    normalized = points.copy()
    normalized[:, 0] /= max(float(width), 1.0)
    normalized[:, 1] /= max(float(height), 1.0)
    return np.clip(normalized, 0.0, 1.0)


def _flip_quad_horizontal(quad: np.ndarray) -> np.ndarray:
    flipped = np.array(quad, dtype=np.float32).copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    return order_points(flipped)


def _maybe_augment(image: np.ndarray, quads: list[np.ndarray], enable_flip: bool) -> tuple[np.ndarray, list[np.ndarray]]:
    if not enable_flip or random.random() >= 0.5:
        return image, quads
    image = cv2.flip(image, 1)
    return image, [_flip_quad_horizontal(quad) for quad in quads]


def _apply_perspective_augmentation(
    image: np.ndarray,
    quads: list[np.ndarray],
    settings: Mapping[str, Any],
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = rng or np.random.default_rng()
    prob = float(settings.get("perspective_prob", 0.0))
    if prob <= 0.0 or float(rng.random()) >= prob:
        return image, quads
    height, width = image.shape[:2]
    jitter_ratio = float(settings.get("perspective_jitter_ratio", 0.06))
    max_jitter_x = width * jitter_ratio
    max_jitter_y = height * jitter_ratio
    src = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    jitter = np.stack(
        [
            rng.uniform(-max_jitter_x, max_jitter_x, size=4),
            rng.uniform(-max_jitter_y, max_jitter_y, size=4),
        ],
        axis=-1,
    ).astype(np.float32)
    dst = src + jitter
    dst[:, 0] = np.clip(dst[:, 0], 0.0, width - 1.0)
    dst[:, 1] = np.clip(dst[:, 1], 0.0, height - 1.0)
    transform = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    warped_quads: list[np.ndarray] = []
    for quad in quads:
        quad_px = np.array(quad, dtype=np.float32).copy()
        quad_px[:, 0] *= max(width - 1.0, 1.0)
        quad_px[:, 1] *= max(height - 1.0, 1.0)
        quad_px = cv2.perspectiveTransform(quad_px[None, :, :], transform)[0]
        quad_px[:, 0] /= max(width - 1.0, 1.0)
        quad_px[:, 1] /= max(height - 1.0, 1.0)
        warped_quads.append(np.clip(order_points(quad_px), 0.0, 1.0))
    return warped_image, warped_quads


def _apply_train_augmentation(
    image: np.ndarray,
    settings: Mapping[str, Any],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    augmented = image.copy()
    if float(settings.get("brightness_contrast_prob", 0.0)) > 0.0 and float(rng.random()) < float(
        settings.get("brightness_contrast_prob", 0.0)
    ):
        brightness_delta = float(settings.get("brightness_delta", 18.0))
        contrast_scale = float(settings.get("contrast_scale", 1.12))
        augmented = np.clip((augmented.astype(np.float32) - 127.5) * contrast_scale + 127.5 + brightness_delta, 0.0, 255.0).astype(
            np.uint8
        )
    if float(settings.get("jpeg_prob", 0.0)) > 0.0 and float(rng.random()) < float(settings.get("jpeg_prob", 0.0)):
        quality_min = int(settings.get("jpeg_quality_min", 60))
        quality_max = int(settings.get("jpeg_quality_max", 85))
        quality = quality_min if quality_min >= quality_max else int(rng.integers(quality_min, quality_max + 1))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                augmented = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if float(settings.get("gaussian_noise_prob", 0.0)) > 0.0 and float(rng.random()) < float(settings.get("gaussian_noise_prob", 0.0)):
        noise_std = float(settings.get("gaussian_noise_std", 4.0))
        noise = rng.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
    if float(settings.get("occlusion_prob", 0.0)) > 0.0 and float(rng.random()) < float(settings.get("occlusion_prob", 0.0)):
        height, width = augmented.shape[:2]
        size_min = float(settings.get("occlusion_size_min", 0.12))
        size_max = float(settings.get("occlusion_size_max", 0.28))
        count_min = max(int(settings.get("occlusion_count_min", 1)), 1)
        count_max = max(int(settings.get("occlusion_count_max", count_min)), count_min)
        occlusion_count = count_min if count_min >= count_max else int(rng.integers(count_min, count_max + 1))
        for _ in range(occlusion_count):
            ratio = size_min if size_min >= size_max else float(rng.uniform(size_min, size_max))
            occ_w = max(1, min(width, int(round(width * ratio))))
            occ_h = max(1, min(height, int(round(height * ratio))))
            x_max = max(width - occ_w, 0)
            y_max = max(height - occ_h, 0)
            x0 = int(rng.integers(0, x_max + 1)) if x_max > 0 else 0
            y0 = int(rng.integers(0, y_max + 1)) if y_max > 0 else 0
            augmented[y0 : y0 + occ_h, x0 : x0 + occ_w] = 0
    if float(settings.get("edge_occlusion_prob", 0.0)) > 0.0 and float(rng.random()) < float(settings.get("edge_occlusion_prob", 0.0)):
        height, width = augmented.shape[:2]
        size_min = float(settings.get("edge_occlusion_size_min", 0.06))
        size_max = float(settings.get("edge_occlusion_size_max", 0.18))
        sides_min = max(int(settings.get("edge_occlusion_sides_min", 1)), 1)
        sides_max = max(int(settings.get("edge_occlusion_sides_max", sides_min)), sides_min)
        side_count = sides_min if sides_min >= sides_max else int(rng.integers(sides_min, sides_max + 1))
        selected_sides = list(rng.choice(np.array(["top", "right", "bottom", "left"]), size=min(side_count, 4), replace=False))
        ratio = size_min if size_min >= size_max else float(rng.uniform(size_min, size_max))
        edge_w = max(1, min(width, int(round(width * ratio))))
        edge_h = max(1, min(height, int(round(height * ratio))))
        for side in selected_sides:
            if side == "top":
                augmented[:edge_h, :, :] = 0
            elif side == "bottom":
                augmented[height - edge_h :, :, :] = 0
            elif side == "left":
                augmented[:, :edge_w, :] = 0
            elif side == "right":
                augmented[:, width - edge_w :, :] = 0
    return augmented


def _scene_target_from_tags(scene_tags: list[str] | tuple[str, ...] | None) -> np.ndarray:
    values = np.zeros((len(SCENE_TAGS),), dtype=np.float32)
    active = {str(tag) for tag in (scene_tags or [])}
    for index, tag in enumerate(SCENE_TAGS):
        if tag in active:
            values[index] = 1.0
    return values


def _build_border_contact_target(manual_quad: np.ndarray, threshold: float = 0.035) -> np.ndarray:
    quad = np.array(manual_quad, dtype=np.float32)
    distance_to_border = np.minimum.reduce(
        [
            quad[:, 0],
            quad[:, 1],
            1.0 - quad[:, 0],
            1.0 - quad[:, 1],
        ]
    )
    return (distance_to_border <= float(threshold)).astype(np.float32)


def _compute_border_distance_min(row: Mapping[str, Any], image_width: int, image_height: int) -> float:
    quad = np.array(row.get("manual_quad") or [], dtype=np.float32)
    if quad.shape != (4, 2):
        return 0.0
    normalized = quad.copy()
    normalized[:, 0] /= max(float(image_width), 1.0)
    normalized[:, 1] /= max(float(image_height), 1.0)
    distance_to_border = np.minimum.reduce(
        [
            normalized[:, 0],
            normalized[:, 1],
            1.0 - normalized[:, 0],
            1.0 - normalized[:, 1],
        ]
    )
    return float(np.min(distance_to_border))


def _compute_process_targets(
    image: np.ndarray,
    teacher_roi_quad: list[list[float]] | np.ndarray,
    teacher_final_quad: list[list[float]] | np.ndarray,
    page_id: str = "teacher_process",
    fallback_visibility_threshold: float = 0.35,
) -> dict[str, np.ndarray]:
    roi_quad = order_points(np.array(teacher_roi_quad, dtype=np.float32))
    final_quad = order_points(np.array(teacher_final_quad, dtype=np.float32))
    roi_min = np.amin(roi_quad, axis=0)
    roi_max = np.amax(roi_quad, axis=0)
    roi_span = np.clip(roi_max - roi_min, 1.0, None)
    refine_delta_norm = np.clip((final_quad - roi_quad) / roi_span.reshape(1, 2), -1.0, 1.0).astype(np.float32)
    visibility_targets: list[np.ndarray] = []
    edge_targets: list[np.ndarray] = []
    fallback_values: list[float] = []
    for corner_index in range(4):
        patch_sample = build_local_corner_patch_sample(
            image_path=Path(page_id),
            image=image,
            page_id=page_id,
            corner_index=corner_index,
            predicted_quad=roi_quad,
            manual_quad=final_quad,
            patch_size=None,
        )
        patch_row = {
            "corner_index": corner_index,
            "manual_quad": final_quad.tolist(),
            "predicted_quad": roi_quad.tolist(),
            "patch": patch_sample["patch"],
            "predicted_point": patch_sample["predicted_point"],
        }
        visibility = build_corner_visibility_target(patch_row, patch_sample["patch_image"]).astype(np.float32)
        edge = build_corner_direction_target({"manual_quad": final_quad.tolist(), "corner_index": corner_index}).astype(np.float32)
        visibility_targets.append(visibility)
        edge_targets.append(edge)
        fallback_values.append(float(np.mean(visibility) < fallback_visibility_threshold))
    return {
        "teacher_refine_delta_norm": refine_delta_norm,
        "teacher_corner_visibility": np.stack(visibility_targets, axis=0),
        "teacher_corner_edge_direction": np.stack(edge_targets, axis=0),
        "teacher_corner_fallback_mask": np.array(fallback_values, dtype=np.float32),
        "teacher_roi_box": np.array([roi_min[0], roi_min[1], roi_max[0], roi_max[1]], dtype=np.float32),
    }


def _recent_rounds_hit_plateau(
    rounds: list[Mapping[str, Any]],
    improvement_threshold: float = 0.01,
    patience: int = 3,
) -> bool:
    if len(rounds) < patience + 1:
        return False
    recent = rounds[-(patience + 1) :]
    for previous, current in zip(recent, recent[1:]):
        prev_value = float(previous.get("point_le_0_01_ratio", 0.0))
        current_value = float(current.get("point_le_0_01_ratio", 0.0))
        if current_value - prev_value >= improvement_threshold:
            return False
    return True


def _decide_round_status(student_metrics: Mapping[str, Any], teacher_metrics: Mapping[str, Any]) -> tuple[str, str]:
    student_point = float(student_metrics.get("point_error_mean", math.inf))
    student_strict_point = float(student_metrics.get("point_le_0_01_ratio", 0.0))
    student_infer = float(student_metrics.get("avg_page_infer_ms", math.inf))
    if student_infer > MAX_ALLOWED_INFER_MS:
        return "continue", "latency gate missed"
    if student_strict_point >= STRICT_POINT_TARGET_RATIO:
        return "stop", "meets strict-point threshold"
    teacher_point = float(teacher_metrics.get("point_error_mean", math.inf))
    if student_point <= teacher_point * 3.0:
        return "continue", "still above teacher quality target"
    return "continue", "still far from target quality"


def _build_round_comparison(
    student_metrics: Mapping[str, Any],
    teacher_metrics: Mapping[str, Any],
    r3_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_keys = [
        "point_error_mean",
        "point_le_0_05_ratio",
        "point_le_0_03_ratio",
        "point_le_0_02_ratio",
        "point_le_0_01_ratio",
        "screen_relative_error_mean",
        "max_corner_error_mean",
        "max_corner_le_0_03_ratio",
        "perspective_tilt_error_mean",
        "quad_inset_ratio_mean",
        "quad_inset_ratio_abs_mean",
    ]
    student = {key: student_metrics.get(key) for key in comparison_keys}
    teacher = {key: teacher_metrics.get(key) for key in comparison_keys}
    r3 = {key: r3_metrics.get(key) for key in comparison_keys}
    deltas = {
        key: round(float(student_metrics.get(key, 0.0)) - float(teacher_metrics.get(key, 0.0)), 4)
        for key in comparison_keys
        if isinstance(student_metrics.get(key), (int, float)) and isinstance(teacher_metrics.get(key), (int, float))
    }
    decision, reason = _decide_round_status(student_metrics, teacher_metrics)
    return {
        "student": student,
        "teacher": teacher,
        "r3": r3,
        "delta_to_teacher": deltas,
        "decision": decision,
        "decision_reason": reason,
    }


class DeepScreenV1Dataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        repo_root: Path,
        input_size: int = 256,
        output_size: int = 64,
        augment: bool = False,
        augmentation_settings: Mapping[str, Any] | None = None,
        filter_settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.rows = _read_jsonl(manifest_path)
        if filter_settings:
            self.rows = self._filter_rows(self.rows, filter_settings)
        self.repo_root = repo_root
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment
        self.augmentation_settings = dict(augmentation_settings or {})

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _filter_rows(rows: list[dict[str, Any]], filter_settings: Mapping[str, Any]) -> list[dict[str, Any]]:
        teacher_key = str(filter_settings.get("teacher_key", "teacher_r3_quad"))
        mean_point_error_max = filter_settings.get("mean_point_error_max")
        all_corner_error_max = filter_settings.get("all_corner_error_max")
        min_border_distance = filter_settings.get("min_border_distance")
        if mean_point_error_max is None and all_corner_error_max is None and min_border_distance is None:
            return rows
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if min_border_distance is not None and float(row.get("hard_case_border_distance_min", 0.0)) < float(min_border_distance):
                continue
            if bool(row.get("manual_only", False)):
                filtered.append(row)
                continue
            manual_quad = np.array(row["manual_quad"], dtype=np.float32)
            teacher_quad = np.array(row.get(teacher_key) or row["teacher_v28_quad"], dtype=np.float32)
            teacher_metrics = quad_geometry_metrics(manual_quad, teacher_quad)
            if mean_point_error_max is not None and float(teacher_metrics["point_error"]) > float(mean_point_error_max):
                continue
            if all_corner_error_max is not None:
                corner_errors = [float(teacher_metrics[f"corner_error_{index:02d}"]) for index in range(4)]
                if any(error > float(all_corner_error_max) for error in corner_errors):
                    continue
            filtered.append(row)
        return filtered

    def build_sample_weights(
        self,
        power: float = 0.0,
        difficulty_metric: str = "point_error",
        scene_balance_power: float = 0.0,
        teacher_key: str = "teacher_v28_quad",
        scene_tag_boosts: Mapping[str, float] | None = None,
        dataset_slug_boosts: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        if not self.rows:
            return np.ones((0,), dtype=np.float32)
        weights = np.ones((len(self.rows),), dtype=np.float32)
        has_explicit_multiplier = any(float(row.get("sample_weight_multiplier", 1.0)) != 1.0 for row in self.rows)
        if power <= 0.0 and scene_balance_power <= 0.0 and not scene_tag_boosts and not dataset_slug_boosts and not has_explicit_multiplier:
            return weights
        scene_counts: dict[str, int] = {tag: 0 for tag in SCENE_TAGS}
        no_scene_rows = 0
        for row in self.rows:
            tags = [str(tag) for tag in (row.get("scene_tags") or []) if str(tag) in scene_counts]
            if tags:
                for tag in tags:
                    scene_counts[tag] += 1
            else:
                no_scene_rows += 1
        weights: list[float] = []
        for row in self.rows:
            if bool(row.get("manual_only", False)):
                difficulty = 0.0
            else:
                manual_quad = np.array(row["manual_quad"], dtype=np.float32)
                teacher_quad = np.array(row.get(teacher_key) or row["teacher_v28_quad"], dtype=np.float32)
                teacher_metrics = quad_geometry_metrics(manual_quad, teacher_quad)
                if difficulty_metric == "strict_point_gap":
                    corner_errors = np.array(
                        [float(teacher_metrics[f"corner_error_{index:02d}"]) for index in range(4)],
                        dtype=np.float32,
                    )
                    misses = float((corner_errors > STRICT_POINT_THRESHOLD).mean())
                    excess = np.maximum(corner_errors - STRICT_POINT_THRESHOLD, 0.0)
                    difficulty = misses + float(excess.mean())
                else:
                    difficulty = float(teacher_metrics.get(difficulty_metric, teacher_metrics.get("point_error", 0.0)))
            weight = 1.0 + max(difficulty, 0.0) ** power if power > 0.0 else 1.0
            if scene_balance_power > 0.0:
                tags = [str(tag) for tag in (row.get("scene_tags") or []) if str(tag) in scene_counts]
                if tags:
                    balance_values = [len(self.rows) / max(scene_counts[tag], 1) for tag in tags]
                    scene_balance = float(np.mean(balance_values))
                else:
                    scene_balance = float(len(self.rows) / max(no_scene_rows, 1))
                weight *= max(scene_balance, 1.0) ** scene_balance_power
            if scene_tag_boosts:
                tags = [str(tag) for tag in (row.get("scene_tags") or [])]
                tag_boost = max((float(scene_tag_boosts.get(tag, 1.0)) for tag in tags), default=1.0)
                weight *= max(tag_boost, 1.0)
            if dataset_slug_boosts:
                dataset_slug = str(row.get("dataset_slug") or "")
                weight *= max(float(dataset_slug_boosts.get(dataset_slug, 1.0)), 1.0)
            weight *= max(float(row.get("sample_weight_multiplier", 1.0)), 0.0)
            weights.append(weight)
        return np.array(weights, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image_path = _resolve_image_path(self.repo_root, row)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        manual_only = bool(row.get("manual_only", False))
        manual_process_targets = manual_only and bool(row.get("manual_process_targets", False))
        manual_quad = _normalize_quad(row["manual_quad"], width, height)
        teacher_quad = _normalize_quad(row.get("teacher_v28_quad") or row["manual_quad"], width, height)
        r3_quad = _normalize_quad(row.get("teacher_r3_quad") or row["manual_quad"], width, height)
        teacher_roi_quad = _normalize_quad(row.get("teacher_roi_quad") or row["manual_quad"], width, height)
        has_external_candidate = bool(row.get("opencv_best_quad"))
        external_candidate_quad = (
            _normalize_quad(row["opencv_best_quad"], width, height) if has_external_candidate else np.zeros((4, 2), dtype=np.float32)
        )
        image, [manual_quad, teacher_quad, r3_quad, teacher_roi_quad, external_candidate_quad] = _maybe_augment(
            image,
            [manual_quad, teacher_quad, r3_quad, teacher_roi_quad, external_candidate_quad],
            self.augment and bool(self.augmentation_settings.get("enable_flip", True)),
        )
        if self.augment:
            image, [manual_quad, teacher_quad, r3_quad, teacher_roi_quad, external_candidate_quad] = _apply_perspective_augmentation(
                image,
                [manual_quad, teacher_quad, r3_quad, teacher_roi_quad, external_candidate_quad],
                self.augmentation_settings,
            )
            image = _apply_train_augmentation(image, self.augmentation_settings)
        aug_height, aug_width = image.shape[:2]
        process_roi_quad = manual_quad if manual_process_targets else teacher_roi_quad
        process_final_quad = manual_quad if manual_process_targets else teacher_quad
        teacher_roi_quad_px = process_roi_quad.copy()
        teacher_roi_quad_px[:, 0] *= max(float(aug_width), 1.0)
        teacher_roi_quad_px[:, 1] *= max(float(aug_height), 1.0)
        teacher_quad_px = process_final_quad.copy()
        teacher_quad_px[:, 0] *= max(float(aug_width), 1.0)
        teacher_quad_px[:, 1] *= max(float(aug_height), 1.0)
        process_targets = _compute_process_targets(
            image,
            teacher_roi_quad_px,
            teacher_quad_px,
            page_id=str(row.get("page_id") or image_path.stem),
        )
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        image_f = image.astype(np.float32) / 255.0
        return {
            "image": torch.from_numpy(np.transpose(image_f, (2, 0, 1))),
            "manual_quad": torch.from_numpy(manual_quad),
            "teacher_quad": torch.from_numpy(teacher_quad),
            "r3_quad": torch.from_numpy(r3_quad),
            "teacher_roi_quad": torch.from_numpy(teacher_roi_quad),
            "manual_heatmaps": torch.from_numpy(build_corner_heatmaps(manual_quad.tolist(), output_size=self.output_size)),
            "r3_heatmaps": torch.from_numpy(build_corner_heatmaps(r3_quad.tolist(), output_size=self.output_size)),
            "scene_target": torch.from_numpy(_scene_target_from_tags(row.get("scene_tags"))),
            "teacher_refine_delta": torch.from_numpy(process_targets["teacher_refine_delta_norm"]),
            "teacher_corner_visibility": torch.from_numpy(process_targets["teacher_corner_visibility"]),
            "teacher_corner_edge": torch.from_numpy(process_targets["teacher_corner_edge_direction"]),
            "teacher_corner_fallback": torch.from_numpy(process_targets["teacher_corner_fallback_mask"]),
            "border_contact_target": torch.from_numpy(_build_border_contact_target(manual_quad)),
            "external_candidate_quads": torch.from_numpy(external_candidate_quad[None, ...]),
            "external_candidate_mask": torch.tensor([has_external_candidate], dtype=torch.bool),
            "manual_only_mask": torch.tensor(manual_only, dtype=torch.bool),
            "process_structure_mask": torch.tensor((not manual_only) or manual_process_targets, dtype=torch.bool),
        }


def _build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _build_train_dataloader(
    dataset: DeepScreenV1Dataset,
    batch_size: int,
    sampling_settings: Mapping[str, Any] | None = None,
    sample_weights: np.ndarray | None = None,
) -> DataLoader:
    sampling_settings = dict(sampling_settings or {})
    sample_weight_power = float(sampling_settings.get("sample_weight_power", 0.0))
    difficulty_metric = str(sampling_settings.get("difficulty_metric", "point_error"))
    scene_balance_power = float(sampling_settings.get("scene_balance_power", 0.0))
    teacher_key = str(sampling_settings.get("teacher_key", "teacher_v28_quad"))
    scene_tag_boosts = {str(key): float(value) for key, value in dict(sampling_settings.get("scene_tag_boosts") or {}).items()}
    dataset_slug_boosts = {str(key): float(value) for key, value in dict(sampling_settings.get("dataset_slug_boosts") or {}).items()}
    if sample_weight_power <= 0.0 and scene_balance_power <= 0.0:
        if not scene_tag_boosts and not dataset_slug_boosts:
            return _build_dataloader(dataset, batch_size=batch_size, shuffle=True)
    weights = sample_weights
    if weights is None:
        weights = dataset.build_sample_weights(
            power=sample_weight_power,
            difficulty_metric=difficulty_metric,
            scene_balance_power=scene_balance_power,
            teacher_key=teacher_key,
            scene_tag_boosts=scene_tag_boosts,
            dataset_slug_boosts=dataset_slug_boosts,
        )
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(), num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


def _checkpoint_metric_value(metrics: Mapping[str, Any], key: str, default: float) -> float:
    value = metrics.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def _checkpoint_rank(metrics: Mapping[str, Any], selection_metric: str) -> tuple[float, ...]:
    point_error = _checkpoint_metric_value(metrics, "point_error_mean", math.inf)
    point_le_003 = _checkpoint_metric_value(metrics, "point_le_0_03_ratio", -math.inf)
    point_le_002 = _checkpoint_metric_value(metrics, "point_le_0_02_ratio", -math.inf)
    point_le_001 = _checkpoint_metric_value(metrics, "point_le_0_01_ratio", -math.inf)
    max_corner_le_003 = _checkpoint_metric_value(metrics, "max_corner_le_0_03_ratio", -math.inf)
    screen_error = _checkpoint_metric_value(metrics, "screen_relative_error_mean", math.inf)
    loss_mean = _checkpoint_metric_value(metrics, "loss_mean", math.inf)
    infer_ms = _checkpoint_metric_value(metrics, "avg_page_infer_ms", math.inf)
    latency_invalid = infer_ms > MAX_ALLOWED_INFER_MS
    if selection_metric == "point_error_mean":
        return (latency_invalid, point_error, infer_ms, -max_corner_le_003, -point_le_002, -point_le_003, -point_le_001, screen_error, loss_mean)
    if selection_metric == "point_le_0_02_ratio":
        return (latency_invalid, -point_le_002, infer_ms, -max_corner_le_003, -point_le_001, point_error, -point_le_003, screen_error, loss_mean)
    if selection_metric == "point_le_0_01_ratio":
        return (latency_invalid, -point_le_001, point_error, -point_le_002, -point_le_003, -max_corner_le_003, screen_error, infer_ms, loss_mean)
    if selection_metric == "max_corner_le_0_03_ratio":
        return (latency_invalid, -max_corner_le_003, infer_ms, -point_le_001, point_error, -point_le_002, -point_le_003, screen_error, loss_mean)
    if selection_metric == "point_le_0_03_ratio":
        return (latency_invalid, -point_le_003, infer_ms, -max_corner_le_003, point_error, -point_le_002, -point_le_001, screen_error, loss_mean)
    return (latency_invalid, loss_mean, infer_ms, point_error, -max_corner_le_003, -point_le_002, -point_le_003, -point_le_001, screen_error)


def _is_better_checkpoint(
    candidate_metrics: Mapping[str, Any],
    current_best_metrics: Mapping[str, Any] | None,
    selection_metric: str = "loss_mean",
) -> bool:
    if current_best_metrics is None:
        return True
    return _checkpoint_rank(candidate_metrics, selection_metric) < _checkpoint_rank(current_best_metrics, selection_metric)


def _select_training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _max_corner_constraint_loss(predicted_quad: torch.Tensor, target_quad: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    target_bbox = torch.amax(target_quad, dim=1) - torch.amin(target_quad, dim=1)
    target_diag = torch.linalg.norm(torch.clamp(target_bbox, min=1e-6), dim=-1)
    corner_distances = torch.linalg.norm(predicted_quad - target_quad, dim=-1)
    normalized_corner_distances = corner_distances / torch.clamp(target_diag.unsqueeze(-1), min=1e-6)
    worst_corner_error = torch.amax(normalized_corner_distances, dim=-1)
    return torch.relu(worst_corner_error - float(threshold)).mean()


def _strict_point_constraint_loss(predicted_quad: torch.Tensor, target_quad: torch.Tensor, threshold: float = STRICT_POINT_THRESHOLD) -> torch.Tensor:
    target_bbox = torch.amax(target_quad, dim=1) - torch.amin(target_quad, dim=1)
    target_diag = torch.linalg.norm(torch.clamp(target_bbox, min=1e-6), dim=-1)
    corner_distances = torch.linalg.norm(predicted_quad - target_quad, dim=-1)
    normalized_corner_distances = corner_distances / torch.clamp(target_diag.unsqueeze(-1), min=1e-6)
    return torch.relu(normalized_corner_distances - float(threshold)).mean()


def _strict_point_soft_target_loss(
    predicted_quad: torch.Tensor,
    target_quad: torch.Tensor,
    threshold: float = STRICT_POINT_THRESHOLD,
    temperature: float = 0.002,
    border_contact_target: torch.Tensor | None = None,
    border_contact_boost: float = 1.0,
) -> torch.Tensor:
    target_bbox = torch.amax(target_quad, dim=1) - torch.amin(target_quad, dim=1)
    target_diag = torch.linalg.norm(torch.clamp(target_bbox, min=1e-6), dim=-1, keepdim=True)
    corner_distances = torch.linalg.norm(predicted_quad - target_quad, dim=-1)
    normalized_corner_distances = corner_distances / torch.clamp(target_diag, min=1e-6)
    logits = (float(threshold) - normalized_corner_distances) / max(float(temperature), 1e-6)
    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
    if border_contact_target is None or float(border_contact_boost) <= 1.0:
        return loss.mean()
    weights = 1.0 + border_contact_target.to(device=loss.device, dtype=loss.dtype) * (float(border_contact_boost) - 1.0)
    return torch.sum(loss * weights) / torch.clamp(weights.sum(), min=1.0)


def _quad_area(quad: torch.Tensor) -> torch.Tensor:
    shifted = torch.roll(quad, shifts=-1, dims=1)
    cross = quad[..., 0] * shifted[..., 1] - quad[..., 1] * shifted[..., 0]
    return torch.abs(torch.sum(cross, dim=1)) * 0.5


def _quad_inset_abs_loss(predicted_quad: torch.Tensor, target_quad: torch.Tensor) -> torch.Tensor:
    target_area = _quad_area(target_quad)
    predicted_area = _quad_area(predicted_quad)
    inset_abs = torch.abs(target_area - predicted_area) / torch.clamp(target_area, min=1e-6)
    return inset_abs.mean()


def _quad_inset_inward_loss(predicted_quad: torch.Tensor, target_quad: torch.Tensor) -> torch.Tensor:
    target_area = _quad_area(target_quad)
    predicted_area = _quad_area(predicted_quad)
    inward_inset = torch.relu(target_area - predicted_area) / torch.clamp(target_area, min=1e-6)
    return inward_inset.mean()


def _normalize_quad_to_roi_boxes(quad: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    top_left = boxes[:, None, 0:2]
    span = (boxes[:, None, 2:4] - boxes[:, None, 0:2]).clamp(min=1e-6)
    return torch.clamp((quad - top_left) / span, 0.0, 1.0)


def _samplewise_mean(loss_tensor: torch.Tensor) -> torch.Tensor:
    if loss_tensor.dim() <= 1:
        return loss_tensor
    return loss_tensor.view(loss_tensor.shape[0], -1).mean(dim=1)


def _masked_sample_loss(loss_tensor: torch.Tensor, sample_mask: torch.Tensor | None) -> torch.Tensor:
    sample_losses = _samplewise_mean(loss_tensor)
    if sample_mask is None:
        return sample_losses.mean()
    weights = sample_mask.to(device=sample_losses.device, dtype=sample_losses.dtype).view(-1)
    denom = torch.clamp(weights.sum(), min=1.0)
    return torch.sum(sample_losses * weights) / denom


def _build_adaptive_teacher_target(
    manual_quad: torch.Tensor,
    teacher_quad: torch.Tensor,
    blend_ratio: float,
    corner_error_max: float | None = None,
    sample_error_max: float | None = None,
) -> dict[str, torch.Tensor]:
    target_bbox = torch.amax(manual_quad, dim=1) - torch.amin(manual_quad, dim=1)
    target_diag = torch.linalg.norm(torch.clamp(target_bbox, min=1e-6), dim=-1, keepdim=True)
    corner_errors = torch.linalg.norm(teacher_quad - manual_quad, dim=-1) / torch.clamp(target_diag, min=1e-6)
    corner_mask = torch.ones_like(corner_errors, dtype=torch.bool)
    if corner_error_max is not None and float(corner_error_max) > 0.0:
        corner_mask = corner_mask & (corner_errors <= float(corner_error_max))
    sample_mask = torch.ones((manual_quad.shape[0],), dtype=torch.bool, device=manual_quad.device)
    if sample_error_max is not None and float(sample_error_max) > 0.0:
        sample_mask = corner_errors.mean(dim=-1) <= float(sample_error_max)
        corner_mask = corner_mask & sample_mask.view(-1, 1)
    blend = torch.as_tensor(float(blend_ratio), device=manual_quad.device, dtype=manual_quad.dtype)
    blended_quad = manual_quad + blend * (teacher_quad - manual_quad)
    target_quad = torch.where(corner_mask.unsqueeze(-1), blended_quad, manual_quad)
    return {
        "target_quad": target_quad,
        "corner_mask": corner_mask,
        "sample_mask": sample_mask,
        "corner_errors": corner_errors,
    }


def _build_oracle_teacher_target(
    manual_quad: torch.Tensor,
    primary_teacher_quad: torch.Tensor,
    secondary_teacher_quad: torch.Tensor,
    corner_error_max: float | None = None,
    sample_error_max: float | None = None,
) -> dict[str, torch.Tensor]:
    target_bbox = torch.amax(manual_quad, dim=1) - torch.amin(manual_quad, dim=1)
    target_diag = torch.linalg.norm(torch.clamp(target_bbox, min=1e-6), dim=-1, keepdim=True)
    primary_corner_errors = torch.linalg.norm(primary_teacher_quad - manual_quad, dim=-1) / torch.clamp(target_diag, min=1e-6)
    secondary_corner_errors = torch.linalg.norm(secondary_teacher_quad - manual_quad, dim=-1) / torch.clamp(target_diag, min=1e-6)
    use_primary = primary_corner_errors <= secondary_corner_errors
    best_corner_errors = torch.minimum(primary_corner_errors, secondary_corner_errors)
    best_teacher_quad = torch.where(use_primary.unsqueeze(-1), primary_teacher_quad, secondary_teacher_quad)
    corner_mask = torch.ones_like(best_corner_errors, dtype=torch.bool)
    if corner_error_max is not None and float(corner_error_max) > 0.0:
        corner_mask = corner_mask & (best_corner_errors <= float(corner_error_max))
    sample_mask = torch.ones((manual_quad.shape[0],), dtype=torch.bool, device=manual_quad.device)
    if sample_error_max is not None and float(sample_error_max) > 0.0:
        sample_mask = best_corner_errors.mean(dim=-1) <= float(sample_error_max)
        corner_mask = corner_mask & sample_mask.view(-1, 1)
    target_quad = torch.where(corner_mask.unsqueeze(-1), best_teacher_quad, manual_quad)
    return {
        "target_quad": target_quad,
        "corner_mask": corner_mask,
        "sample_mask": sample_mask,
        "corner_errors": best_corner_errors,
        "primary_selected_mask": use_primary,
    }


def _teacher_agreement_mask(
    teacher_quad: torch.Tensor,
    manual_quad: torch.Tensor,
    max_point_error: float | None = None,
) -> torch.Tensor | None:
    if max_point_error is None or float(max_point_error) <= 0.0:
        return None
    point_errors = _samplewise_mean(torch.linalg.norm(teacher_quad - manual_quad, dim=-1))
    return point_errors <= float(max_point_error)


def _build_corner_heatmaps_torch(corners: torch.Tensor, output_size: int, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.clamp(corners, 0.0, 1.0) * float(output_size - 1)
    ys = torch.arange(output_size, device=corners.device, dtype=corners.dtype).view(1, 1, output_size, 1)
    xs = torch.arange(output_size, device=corners.device, dtype=corners.dtype).view(1, 1, 1, output_size)
    x = coords[..., 0].unsqueeze(-1).unsqueeze(-1)
    y = coords[..., 1].unsqueeze(-1).unsqueeze(-1)
    dist = (xs - x) ** 2 + (ys - y) ** 2
    return torch.exp(-dist / max(2.0 * sigma * sigma, 1e-6))


def _build_candidate_pool_tensors(
    output: Mapping[str, torch.Tensor],
    external_candidate_quads: torch.Tensor | None = None,
    external_candidate_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    internal_candidate_list = [output["coarse_quad"], output["roi_stage_quad"]]
    if "state_aware_quad" in output:
        internal_candidate_list.append(output["state_aware_quad"])
    internal_candidate_list.append(output["base_final_quad"])
    internal_candidates = torch.stack(internal_candidate_list, dim=1)
    batch_size = internal_candidates.shape[0]
    internal_mask = torch.ones((batch_size, internal_candidates.shape[1]), dtype=torch.bool, device=internal_candidates.device)
    if external_candidate_quads is None:
        return {"candidate_quads": internal_candidates, "candidate_mask": internal_mask}
    if external_candidate_quads.dim() == 3:
        external_candidate_quads = external_candidate_quads.unsqueeze(1)
    external_candidate_quads = external_candidate_quads.to(device=internal_candidates.device, dtype=internal_candidates.dtype)
    if external_candidate_mask is None:
        external_candidate_mask = torch.ones(
            (batch_size, external_candidate_quads.shape[1]),
            dtype=torch.bool,
            device=internal_candidates.device,
        )
    else:
        external_candidate_mask = external_candidate_mask.to(device=internal_candidates.device, dtype=torch.bool)
        if external_candidate_mask.dim() == 1:
            external_candidate_mask = external_candidate_mask.unsqueeze(1)
    return {
        "candidate_quads": torch.cat([external_candidate_quads, internal_candidates], dim=1),
        "candidate_mask": torch.cat([external_candidate_mask, internal_mask], dim=1),
    }


def _candidate_rank_loss(
    candidate_quads: torch.Tensor,
    candidate_scores: torch.Tensor,
    manual_quad: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    rank_metric: str = "mean_error",
) -> torch.Tensor:
    target_bbox = torch.amax(manual_quad, dim=1) - torch.amin(manual_quad, dim=1)
    target_diag = torch.linalg.norm(torch.clamp(target_bbox, min=1e-6), dim=-1).view(-1, 1, 1)
    candidate_errors = torch.linalg.norm(candidate_quads - manual_quad[:, None, :, :], dim=-1) / torch.clamp(target_diag, min=1e-6)
    if candidate_mask is not None:
        candidate_errors = candidate_errors.masked_fill(~candidate_mask[:, :, None], 1e6)
        candidate_scores = candidate_scores.masked_fill(~candidate_mask, -1e9)
    candidate_mean_errors = candidate_errors.mean(dim=-1)
    if rank_metric == "strict_point_then_mean_error":
        strict_hits = (candidate_errors <= STRICT_POINT_THRESHOLD).to(dtype=candidate_scores.dtype).sum(dim=-1)
        best_index = torch.argmax(strict_hits * 1_000_000.0 - candidate_mean_errors, dim=-1)
    else:
        best_index = torch.argmin(candidate_mean_errors, dim=-1)
    return F.cross_entropy(candidate_scores, best_index)


def _apply_candidate_pool_selection(
    model: DeepScreenV1Net,
    output: dict[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    enable_external_candidate_selection: bool = False,
) -> dict[str, torch.Tensor]:
    if not enable_external_candidate_selection or model.candidate_selection_head is None:
        return output
    external_candidate_quads = batch.get("external_candidate_quads")
    external_candidate_mask = batch.get("external_candidate_mask")
    if external_candidate_quads is None:
        return output
    selection = model.select_candidate_pool(
        output,
        external_candidate_quads=external_candidate_quads.to(device=output["final_quad"].device, dtype=output["final_quad"].dtype),
        external_candidate_mask=external_candidate_mask.to(device=output["final_quad"].device, dtype=torch.bool)
        if external_candidate_mask is not None
        else None,
    )
    merged = dict(output)
    merged["candidate_quads"] = selection["candidate_quads"]
    merged["candidate_mask"] = selection["candidate_mask"]
    merged["candidate_scores"] = selection["candidate_scores"]
    merged["candidate_selected_index"] = selection["candidate_selected_index"]
    merged["final_quad"] = selection["selected_quad"]
    return merged


def _configure_trainable_parameters(model: nn.Module, student_cfg: Mapping[str, Any]) -> list[str]:
    trainable_modules = [str(item) for item in (student_cfg.get("trainable_modules") or []) if str(item).strip()]
    if not trainable_modules:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_modules):
            parameter.requires_grad = True
            trainable_names.append(name)
    return trainable_names


def _compute_total_loss(
    output: Mapping[str, torch.Tensor],
    manual_quad: torch.Tensor,
    teacher_quad: torch.Tensor,
    r3_quad: torch.Tensor,
    teacher_roi_quad: torch.Tensor,
    manual_heatmaps: torch.Tensor,
    r3_heatmaps: torch.Tensor,
    scene_target: torch.Tensor,
    teacher_refine_delta: torch.Tensor,
    teacher_corner_visibility: torch.Tensor,
    teacher_corner_edge: torch.Tensor,
    teacher_corner_fallback: torch.Tensor,
    loss_settings: Mapping[str, Any],
    device: torch.device,
    manual_only_mask: torch.Tensor | None = None,
    process_structure_mask: torch.Tensor | None = None,
    border_contact_target: torch.Tensor | None = None,
) -> torch.Tensor:
    gates = torch.softmax(output["router_logits"], dim=-1)
    entropy = torch.sum(gates * torch.log(gates + 1e-8), dim=-1).mean()
    teacher_sample_mask = None if manual_only_mask is None else ~manual_only_mask.to(device=device, dtype=torch.bool).view(-1)
    process_structure_sample_mask = teacher_sample_mask
    if process_structure_mask is not None:
        process_structure_sample_mask = process_structure_mask.to(device=device, dtype=torch.bool).view(-1)
    r3_agreement_mask = _teacher_agreement_mask(
        r3_quad,
        manual_quad,
        max_point_error=loss_settings.get("r3_agreement_point_error_max"),
    )
    if teacher_sample_mask is not None:
        r3_agreement_mask = teacher_sample_mask if r3_agreement_mask is None else (r3_agreement_mask & teacher_sample_mask)
    adaptive_target = _build_adaptive_teacher_target(
        manual_quad,
        r3_quad,
        blend_ratio=float(loss_settings.get("adaptive_teacher_blend_ratio", 0.0)),
        corner_error_max=loss_settings.get("adaptive_teacher_corner_error_max"),
        sample_error_max=loss_settings.get("adaptive_teacher_sample_error_max"),
    )
    adaptive_target_quad = adaptive_target["target_quad"]
    adaptive_target_heatmaps = _build_corner_heatmaps_torch(
        adaptive_target_quad,
        output_size=int(output["coarse_heatmaps"].shape[-1]),
    )
    oracle_target = _build_oracle_teacher_target(
        manual_quad,
        r3_quad,
        teacher_quad,
        corner_error_max=loss_settings.get("oracle_teacher_corner_error_max"),
        sample_error_max=loss_settings.get("oracle_teacher_sample_error_max"),
    )
    oracle_target_quad = oracle_target["target_quad"]
    oracle_target_heatmaps = _build_corner_heatmaps_torch(
        oracle_target_quad,
        output_size=int(output["coarse_heatmaps"].shape[-1]),
    )
    spatial_manual_loss = torch.tensor(0.0, device=device)
    spatial_teacher_loss = torch.tensor(0.0, device=device)
    strict_spatial_manual_loss = torch.tensor(0.0, device=device)
    strict_spatial_teacher_loss = torch.tensor(0.0, device=device)
    strict_spatial_heatmap_manual_loss = torch.tensor(0.0, device=device)
    strict_spatial_heatmap_teacher_loss = torch.tensor(0.0, device=device)
    candidate_rank_selection_loss = torch.tensor(0.0, device=device)
    state_aware_manual_loss = torch.tensor(0.0, device=device)
    border_contact_loss = torch.tensor(0.0, device=device)
    strict_point_soft_manual_loss = torch.tensor(0.0, device=device)
    strict_point_soft_teacher_loss = torch.tensor(0.0, device=device)
    if "spatial_quad" in output:
        spatial_manual_loss = F.smooth_l1_loss(output["spatial_quad"], manual_quad)
        spatial_teacher_loss = _masked_sample_loss(F.smooth_l1_loss(output["spatial_quad"], teacher_quad, reduction="none"), teacher_sample_mask)
    if "strict_point_quad" in output and "strict_point_heatmaps" in output and "roi_boxes" in output:
        strict_spatial_manual_loss = F.smooth_l1_loss(output["strict_point_quad"], manual_quad)
        strict_spatial_teacher_loss = _masked_sample_loss(
            F.smooth_l1_loss(output["strict_point_quad"], teacher_quad, reduction="none"),
            teacher_sample_mask,
        )
        heatmap_size = int(output["strict_point_heatmaps"].shape[-1])
        manual_roi_corners = _normalize_quad_to_roi_boxes(manual_quad, output["roi_boxes"])
        teacher_roi_corners = _normalize_quad_to_roi_boxes(teacher_quad, output["roi_boxes"])
        manual_roi_heatmaps = _build_corner_heatmaps_torch(manual_roi_corners, output_size=heatmap_size)
        teacher_roi_heatmaps = _build_corner_heatmaps_torch(teacher_roi_corners, output_size=heatmap_size)
        strict_spatial_heatmap_manual_loss = F.mse_loss(output["strict_point_heatmaps"], manual_roi_heatmaps)
        strict_spatial_heatmap_teacher_loss = _masked_sample_loss(
            F.mse_loss(output["strict_point_heatmaps"], teacher_roi_heatmaps, reduction="none"),
            teacher_sample_mask,
        )
    if "candidate_quads" in output and "candidate_scores" in output:
        candidate_rank_selection_loss = _candidate_rank_loss(
            output["candidate_quads"],
            output["candidate_scores"],
            manual_quad,
            candidate_mask=output.get("candidate_mask"),
            rank_metric=str(loss_settings.get("candidate_rank_metric", "mean_error")),
        )
    if "state_aware_quad" in output:
        state_aware_manual_loss = F.smooth_l1_loss(output["state_aware_quad"], manual_quad)
    if border_contact_target is not None and "corner_state_logits" in output:
        border_contact_loss = F.binary_cross_entropy_with_logits(output["corner_state_logits"], border_contact_target)
    max_corner_threshold = float(loss_settings.get("max_corner_threshold", 0.03))
    strict_point_threshold = float(loss_settings.get("strict_point_threshold", STRICT_POINT_THRESHOLD))
    max_corner_manual_loss = _max_corner_constraint_loss(output["final_quad"], manual_quad, threshold=max_corner_threshold)
    max_corner_teacher_loss = _masked_sample_loss(
        torch.relu(
            torch.amax(
                torch.linalg.norm(output["final_quad"] - teacher_quad, dim=-1)
                / torch.clamp(
                    torch.linalg.norm(torch.clamp(torch.amax(teacher_quad, dim=1) - torch.amin(teacher_quad, dim=1), min=1e-6), dim=-1).unsqueeze(-1),
                    min=1e-6,
                ),
                dim=-1,
            )
            - max_corner_threshold
        ),
        teacher_sample_mask,
    )
    strict_point_manual_loss = _strict_point_constraint_loss(output["final_quad"], manual_quad, threshold=strict_point_threshold)
    strict_point_teacher_loss = _masked_sample_loss(
        torch.relu(
            torch.linalg.norm(output["final_quad"] - teacher_quad, dim=-1)
            / torch.clamp(
                torch.linalg.norm(torch.clamp(torch.amax(teacher_quad, dim=1) - torch.amin(teacher_quad, dim=1), min=1e-6), dim=-1).unsqueeze(-1),
                min=1e-6,
            )
            - strict_point_threshold
        ),
        teacher_sample_mask,
    )
    strict_point_soft_manual_loss = _strict_point_soft_target_loss(
        output["final_quad"],
        manual_quad,
        threshold=strict_point_threshold,
        temperature=float(loss_settings.get("strict_point_soft_temperature", 0.002)),
        border_contact_target=border_contact_target,
        border_contact_boost=float(loss_settings.get("strict_point_soft_border_boost", 1.0)),
    )
    strict_point_teacher_loss_raw = _strict_point_soft_target_loss(
        output["final_quad"],
        teacher_quad,
        threshold=strict_point_threshold,
        temperature=float(loss_settings.get("strict_point_soft_temperature", 0.002)),
    )
    strict_point_soft_teacher_loss = strict_point_teacher_loss_raw if teacher_sample_mask is None else strict_point_teacher_loss_raw * teacher_sample_mask.to(
        device=device,
        dtype=output["final_quad"].dtype,
    ).mean()
    quad_inset_abs_manual_loss = _quad_inset_abs_loss(output["final_quad"], manual_quad)
    quad_inset_inward_manual_loss = _quad_inset_inward_loss(output["final_quad"], manual_quad)
    roi_stage_teacher_loss = _masked_sample_loss(F.smooth_l1_loss(output["roi_stage_quad"], teacher_roi_quad, reduction="none"), teacher_sample_mask)
    process_delta_loss = _masked_sample_loss(F.smooth_l1_loss(output["process_delta"], teacher_refine_delta, reduction="none"), teacher_sample_mask)
    process_visibility_loss = _masked_sample_loss(
        F.smooth_l1_loss(output["process_visibility"], teacher_corner_visibility, reduction="none"),
        process_structure_sample_mask,
    )
    process_edge_loss = _masked_sample_loss(
        F.smooth_l1_loss(output["process_edge"], teacher_corner_edge, reduction="none"),
        process_structure_sample_mask,
    )
    process_fallback_loss = _masked_sample_loss(
        F.binary_cross_entropy_with_logits(output["process_fallback_logits"], teacher_corner_fallback, reduction="none"),
        process_structure_sample_mask,
    )
    scene_loss = torch.tensor(0.0, device=device)
    if "scene_logits" in output:
        scene_loss = F.binary_cross_entropy_with_logits(output["scene_logits"], scene_target)
    return (
        float(loss_settings.get("coarse_heatmap_weight", 1.0)) * F.mse_loss(output["coarse_heatmaps"], manual_heatmaps)
        + float(loss_settings.get("adaptive_coarse_heatmap_weight", 0.0))
        * _masked_sample_loss(F.mse_loss(output["coarse_heatmaps"], adaptive_target_heatmaps, reduction="none"), teacher_sample_mask)
        + float(loss_settings.get("oracle_coarse_heatmap_weight", 0.0))
        * _masked_sample_loss(F.mse_loss(output["coarse_heatmaps"], oracle_target_heatmaps, reduction="none"), teacher_sample_mask)
        + float(loss_settings.get("r3_heatmap_weight", 0.5))
        * _masked_sample_loss(F.mse_loss(output["coarse_heatmaps"], r3_heatmaps, reduction="none"), r3_agreement_mask)
        + float(loss_settings.get("coarse_manual_weight", 2.0)) * F.smooth_l1_loss(output["coarse_quad"], manual_quad)
        + float(loss_settings.get("adaptive_coarse_quad_weight", 0.0))
        * _masked_sample_loss(F.smooth_l1_loss(output["coarse_quad"], adaptive_target_quad, reduction="none"), teacher_sample_mask)
        + float(loss_settings.get("oracle_coarse_quad_weight", 0.0))
        * _masked_sample_loss(F.smooth_l1_loss(output["coarse_quad"], oracle_target_quad, reduction="none"), teacher_sample_mask)
        + float(loss_settings.get("coarse_r3_weight", 1.0))
        * _masked_sample_loss(F.smooth_l1_loss(output["coarse_quad"], r3_quad, reduction="none"), r3_agreement_mask)
        + float(loss_settings.get("roi_stage_teacher_weight", 0.0)) * roi_stage_teacher_loss
        + float(loss_settings.get("final_manual_weight", 3.0)) * F.smooth_l1_loss(output["final_quad"], manual_quad)
        + float(loss_settings.get("final_teacher_weight", 2.0))
        * _masked_sample_loss(F.smooth_l1_loss(output["final_quad"], teacher_quad, reduction="none"), teacher_sample_mask)
        + float(loss_settings.get("spatial_manual_weight", 0.0)) * spatial_manual_loss
        + float(loss_settings.get("spatial_teacher_weight", 0.0)) * spatial_teacher_loss
        + float(loss_settings.get("strict_spatial_manual_weight", 0.0)) * strict_spatial_manual_loss
        + float(loss_settings.get("strict_spatial_teacher_weight", 0.0)) * strict_spatial_teacher_loss
        + float(loss_settings.get("strict_spatial_heatmap_manual_weight", 0.0)) * strict_spatial_heatmap_manual_loss
        + float(loss_settings.get("strict_spatial_heatmap_teacher_weight", 0.0)) * strict_spatial_heatmap_teacher_loss
        + float(loss_settings.get("candidate_rank_weight", 0.0)) * candidate_rank_selection_loss
        + float(loss_settings.get("state_aware_manual_weight", 0.0)) * state_aware_manual_loss
        + float(loss_settings.get("border_contact_weight", 0.0)) * border_contact_loss
        + float(loss_settings.get("max_corner_manual_weight", 0.0)) * max_corner_manual_loss
        + float(loss_settings.get("max_corner_teacher_weight", 0.0)) * max_corner_teacher_loss
        + float(loss_settings.get("strict_point_manual_weight", 0.0)) * strict_point_manual_loss
        + float(loss_settings.get("strict_point_teacher_weight", 0.0)) * strict_point_teacher_loss
        + float(loss_settings.get("strict_point_soft_manual_weight", 0.0)) * strict_point_soft_manual_loss
        + float(loss_settings.get("strict_point_soft_teacher_weight", 0.0)) * strict_point_soft_teacher_loss
        + float(loss_settings.get("process_delta_weight", 0.0)) * process_delta_loss
        + float(loss_settings.get("process_visibility_weight", 0.0)) * process_visibility_loss
        + float(loss_settings.get("process_edge_weight", 0.0)) * process_edge_loss
        + float(loss_settings.get("process_fallback_weight", 0.0)) * process_fallback_loss
        + float(loss_settings.get("quad_inset_abs_weight", 0.0)) * quad_inset_abs_manual_loss
        + float(loss_settings.get("quad_inset_inward_weight", 0.0)) * quad_inset_inward_manual_loss
        + float(loss_settings.get("scene_loss_weight", 0.0)) * scene_loss
        + float(loss_settings.get("router_reg_weight", 0.0025)) * entropy
    )


def _evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, loss_settings: Mapping[str, Any]) -> dict[str, float]:
    model.eval()
    metric_rows: list[dict[str, float]] = []
    teacher_rows: list[dict[str, float]] = []
    r3_rows: list[dict[str, float]] = []
    losses: list[float] = []
    infer_elapsed = 0.0
    infer_pages = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            manual_quad = batch["manual_quad"].to(device=device, dtype=torch.float32)
            teacher_quad = batch["teacher_quad"].to(device=device, dtype=torch.float32)
            r3_quad = batch["r3_quad"].to(device=device, dtype=torch.float32)
            teacher_roi_quad = batch["teacher_roi_quad"].to(device=device, dtype=torch.float32)
            manual_heatmaps = batch["manual_heatmaps"].to(device=device, dtype=torch.float32)
            r3_heatmaps = batch["r3_heatmaps"].to(device=device, dtype=torch.float32)
            scene_target = batch["scene_target"].to(device=device, dtype=torch.float32)
            teacher_refine_delta = batch["teacher_refine_delta"].to(device=device, dtype=torch.float32)
            teacher_corner_visibility = batch["teacher_corner_visibility"].to(device=device, dtype=torch.float32)
            teacher_corner_edge = batch["teacher_corner_edge"].to(device=device, dtype=torch.float32)
            teacher_corner_fallback = batch["teacher_corner_fallback"].to(device=device, dtype=torch.float32)
            border_contact_target = batch["border_contact_target"].to(device=device, dtype=torch.float32)
            manual_only_mask = batch.get("manual_only_mask")
            if manual_only_mask is not None:
                manual_only_mask = manual_only_mask.to(device=device, dtype=torch.bool)
            process_structure_mask = batch.get("process_structure_mask")
            if process_structure_mask is not None:
                process_structure_mask = process_structure_mask.to(device=device, dtype=torch.bool)
            infer_start = perf_counter()
            output = model(images)
            output = _apply_candidate_pool_selection(
                model,
                output,
                batch,
                enable_external_candidate_selection=bool(getattr(model, "opencv_candidate_selection_enabled", False)),
            )
            infer_elapsed += perf_counter() - infer_start
            infer_pages += images.shape[0]
            loss = _compute_total_loss(
                output,
                manual_quad,
                teacher_quad,
                r3_quad,
                teacher_roi_quad,
                manual_heatmaps,
                r3_heatmaps,
                scene_target,
                teacher_refine_delta,
                teacher_corner_visibility,
                teacher_corner_edge,
                teacher_corner_fallback,
                loss_settings,
                device,
                manual_only_mask=manual_only_mask,
                process_structure_mask=process_structure_mask,
                border_contact_target=border_contact_target,
            )
            losses.append(float(loss.detach().cpu().item()))
            pred_quad = output["final_quad"].cpu().numpy()
            manual_np = manual_quad.cpu().numpy()
            teacher_np = teacher_quad.cpu().numpy()
            r3_np = r3_quad.cpu().numpy()
            for idx in range(len(pred_quad)):
                metric_rows.append(quad_geometry_metrics(manual_np[idx], pred_quad[idx]))
                teacher_rows.append(quad_geometry_metrics(manual_np[idx], teacher_np[idx]))
                r3_rows.append(quad_geometry_metrics(manual_np[idx], r3_np[idx]))
    summary = summarize_geometry_metric_rows(metric_rows)
    teacher_summary = summarize_geometry_metric_rows(teacher_rows)
    r3_summary = summarize_geometry_metric_rows(r3_rows)
    summary.update(
        {
            "loss_mean": round(float(np.mean(losses)), 4) if losses else 0.0,
            "avg_page_infer_ms": round((infer_elapsed / max(infer_pages, 1)) * 1000.0, 2),
            "teacher_point_error_mean": teacher_summary["point_error_mean"],
            "teacher_point_le_0_01_ratio": teacher_summary["point_le_0_01_ratio"],
            "r3_point_error_mean": r3_summary["point_error_mean"],
            "r3_point_le_0_01_ratio": r3_summary["point_le_0_01_ratio"],
        }
    )
    return summary


def _train_student(
    repo_root: Path,
    config: Mapping[str, Any],
    train_manifest: Path,
    val_manifest: Path,
    holdout_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    student_cfg = _student_settings(config)
    loss_cfg = _loss_settings(config)
    augmentation_cfg = _augmentation_settings(config)
    checkpoint_cfg = _checkpoint_settings(config)
    sampling_cfg = _sampling_settings(config)
    random.seed(int(student_cfg.get("seed", 7)))
    np.random.seed(int(student_cfg.get("seed", 7)))
    torch.manual_seed(int(student_cfg.get("seed", 7)))
    input_size = int(student_cfg.get("input_size", 256))
    output_size = int(student_cfg.get("output_size", 64))
    batch_size = int(student_cfg.get("batch_size", 8))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_filter_cfg = dict((sampling_cfg.get("train_filter") or {}))
    train_dataset = DeepScreenV1Dataset(
        train_manifest,
        repo_root,
        input_size=input_size,
        output_size=output_size,
        augment=True,
        augmentation_settings=augmentation_cfg,
        filter_settings=train_filter_cfg,
    )
    val_dataset = DeepScreenV1Dataset(val_manifest, repo_root, input_size=input_size, output_size=output_size, augment=False)
    holdout_dataset = DeepScreenV1Dataset(holdout_manifest, repo_root, input_size=input_size, output_size=output_size, augment=False)
    train_loader = _build_train_dataloader(train_dataset, batch_size=batch_size, sampling_settings=sampling_cfg)
    val_loader = _build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    holdout_loader = _build_dataloader(holdout_dataset, batch_size=batch_size, shuffle=False)
    device = _select_training_device()
    candidate_selection_enabled = bool(student_cfg.get("candidate_selection_enabled", False))
    final_output_mode = str(student_cfg.get("final_output_mode", "candidate_selection" if candidate_selection_enabled else "base_final"))
    state_aware_candidate_enabled = bool(student_cfg.get("state_aware_candidate_enabled", False))
    internal_candidate_names = tuple(str(item) for item in (student_cfg.get("internal_candidate_names") or []) if str(item).strip())
    model = DeepScreenV1Net(
        base_channels=int(student_cfg.get("base_channels", 32)),
        roi_size=int(student_cfg.get("roi_size", 16)),
        experts=int(student_cfg.get("experts", 3)),
        expand_ratio=float(student_cfg.get("roi_expand_ratio", 0.08)),
        roi_adapter_layers=int(student_cfg.get("roi_adapter_layers", 0)),
        spatial_refine_layers=int(student_cfg.get("spatial_refine_layers", 0)),
        residual_quad_head_layers=int(student_cfg.get("residual_quad_head_layers", 0)),
        coarse_residual_head_layers=int(student_cfg.get("coarse_residual_head_layers", 0)),
        strict_spatial_refine_layers=int(student_cfg.get("strict_spatial_refine_layers", 0)),
        candidate_selection_enabled=candidate_selection_enabled,
        state_aware_candidate_enabled=state_aware_candidate_enabled,
        internal_candidate_names=internal_candidate_names or None,
        final_output_mode=final_output_mode,
        scene_classes=int(student_cfg.get("scene_classes", len(SCENE_TAGS))),
        scene_embedding_dim=int(student_cfg.get("scene_embedding_dim", 8)),
        coarse_visibility_refine_enabled=bool(student_cfg.get("coarse_visibility_refine_enabled", False)),
    ).to(device)
    model.opencv_candidate_selection_enabled = bool(student_cfg.get("opencv_candidate_selection_enabled", False))
    init_checkpoint_path = student_cfg.get("init_checkpoint_path")
    if init_checkpoint_path:
        checkpoint = torch.load(_resolve_repo_path(repo_root, str(init_checkpoint_path)), map_location="cpu")
        load_compatible_state_dict(model, checkpoint["state_dict"])
    trainable_names = _configure_trainable_parameters(model, student_cfg)
    if not trainable_names:
        raise RuntimeError("no trainable parameters selected")
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=float(student_cfg.get("learning_rate", 2e-4)))
    ema_decay = float(student_cfg.get("ema_decay", 0.0))
    ema_enabled = 0.0 < ema_decay < 1.0
    ema_state = {key: value.detach().clone() for key, value in model.state_dict().items()} if ema_enabled else None
    history: list[dict[str, Any]] = []
    selection_metric = str(checkpoint_cfg.get("selection_metric", "loss_mean"))
    save_each_epoch = bool(checkpoint_cfg.get("save_each_epoch", False))
    epoch_checkpoint_dir = output_dir / "epochs"
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, float] | None = None
    best_holdout_metrics: dict[str, float] | None = None
    for epoch in range(1, int(student_cfg.get("epochs", 2)) + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            manual_quad = batch["manual_quad"].to(device=device, dtype=torch.float32)
            teacher_quad = batch["teacher_quad"].to(device=device, dtype=torch.float32)
            r3_quad = batch["r3_quad"].to(device=device, dtype=torch.float32)
            teacher_roi_quad = batch["teacher_roi_quad"].to(device=device, dtype=torch.float32)
            manual_heatmaps = batch["manual_heatmaps"].to(device=device, dtype=torch.float32)
            r3_heatmaps = batch["r3_heatmaps"].to(device=device, dtype=torch.float32)
            scene_target = batch["scene_target"].to(device=device, dtype=torch.float32)
            teacher_refine_delta = batch["teacher_refine_delta"].to(device=device, dtype=torch.float32)
            teacher_corner_visibility = batch["teacher_corner_visibility"].to(device=device, dtype=torch.float32)
            teacher_corner_edge = batch["teacher_corner_edge"].to(device=device, dtype=torch.float32)
            teacher_corner_fallback = batch["teacher_corner_fallback"].to(device=device, dtype=torch.float32)
            border_contact_target = batch["border_contact_target"].to(device=device, dtype=torch.float32)
            manual_only_mask = batch.get("manual_only_mask")
            if manual_only_mask is not None:
                manual_only_mask = manual_only_mask.to(device=device, dtype=torch.bool)
            process_structure_mask = batch.get("process_structure_mask")
            if process_structure_mask is not None:
                process_structure_mask = process_structure_mask.to(device=device, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            output = _apply_candidate_pool_selection(
                model,
                output,
                batch,
                enable_external_candidate_selection=bool(getattr(model, "opencv_candidate_selection_enabled", False)),
            )
            loss = _compute_total_loss(
                output,
                manual_quad,
                teacher_quad,
                r3_quad,
                teacher_roi_quad,
                manual_heatmaps,
                r3_heatmaps,
                scene_target,
                teacher_refine_delta,
                teacher_corner_visibility,
                teacher_corner_edge,
                teacher_corner_fallback,
                loss_cfg,
                device,
                manual_only_mask=manual_only_mask,
                process_structure_mask=process_structure_mask,
                border_contact_target=border_contact_target,
            )
            loss.backward()
            optimizer.step()
            if ema_state is not None:
                ema_state = _update_ema_state(ema_state, model.state_dict(), decay=ema_decay)
            train_losses.append(float(loss.detach().cpu().item()))
        eval_model = model
        raw_state = None
        if ema_state is not None:
            raw_state = _snapshot_state_dict(model.state_dict())
            model.load_state_dict({key: value.detach().to(device=device) for key, value in ema_state.items()}, strict=False)
        val_metrics = _evaluate_model(eval_model, val_loader, device, loss_cfg)
        holdout_metrics = _evaluate_model(eval_model, holdout_loader, device, loss_cfg)
        if raw_state is not None:
            model.load_state_dict({key: value.detach().to(device=device) for key, value in raw_state.items()}, strict=False)
        history.append({"epoch": epoch, "train_loss_mean": round(float(np.mean(train_losses)), 4), "val": val_metrics, "holdout": holdout_metrics})
        if save_each_epoch:
            epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": _snapshot_state_dict(ema_state if ema_state is not None else model.state_dict()),
                    "val_metrics": val_metrics,
                    "holdout_metrics": holdout_metrics,
                    "selection_metric": selection_metric,
                },
                epoch_checkpoint_dir / f"epoch_{epoch:03d}.pt",
            )
        if _is_better_checkpoint(val_metrics, best_val_metrics, selection_metric=selection_metric):
            best_epoch = epoch
            best_state = _snapshot_state_dict(ema_state if ema_state is not None else model.state_dict())
            best_val_metrics = val_metrics
            best_holdout_metrics = holdout_metrics
    if best_state is None or best_val_metrics is None or best_holdout_metrics is None:
        raise RuntimeError("training produced no checkpoint")
    checkpoint_path = output_dir / _artifact_name(config, "student", ".pt")
    torch.save(
        {
            "state_dict": best_state,
            "input_size": input_size,
            "output_size": output_size,
            "base_channels": int(student_cfg.get("base_channels", 32)),
            "roi_size": int(student_cfg.get("roi_size", 16)),
            "experts": int(student_cfg.get("experts", 3)),
            "roi_expand_ratio": float(student_cfg.get("roi_expand_ratio", 0.08)),
            "roi_adapter_layers": int(student_cfg.get("roi_adapter_layers", 0)),
            "spatial_refine_layers": int(student_cfg.get("spatial_refine_layers", 0)),
            "residual_quad_head_layers": int(student_cfg.get("residual_quad_head_layers", 0)),
            "strict_spatial_refine_layers": int(student_cfg.get("strict_spatial_refine_layers", 0)),
            "candidate_selection_enabled": candidate_selection_enabled,
            "state_aware_candidate_enabled": state_aware_candidate_enabled,
            "internal_candidate_names": list(internal_candidate_names),
            "opencv_candidate_selection_enabled": bool(student_cfg.get("opencv_candidate_selection_enabled", False)),
            "final_output_mode": final_output_mode,
            "coarse_visibility_refine_enabled": bool(student_cfg.get("coarse_visibility_refine_enabled", False)),
            "ema_decay": ema_decay,
            "scene_classes": int(student_cfg.get("scene_classes", len(SCENE_TAGS))),
            "scene_embedding_dim": int(student_cfg.get("scene_embedding_dim", 8)),
            "device": device.type,
        },
        checkpoint_path,
    )
    history_path = output_dir / _artifact_name(config, "history", ".json")
    _write_json(history_path, history)
    return {
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "best_val_metrics": best_val_metrics,
        "best_holdout_metrics": best_holdout_metrics,
        "history": history,
        "trainable_parameters": trainable_names,
    }


def _export_teacher_snapshot(repo_root: Path, round_paths: RoundPaths, config: Mapping[str, Any]) -> dict[str, Any]:
    teacher_paths = resolve_teacher_model_paths(repo_root, {alias: str(item["runtime_file"]) for alias, item in config["teachers"].items()})
    split_payload = _load_split_payload(repo_root, config)
    rows = _load_curated_rows(repo_root, config)
    partitioned = partition_rows_by_split(rows, split_payload)
    round_paths.teacher_export_root.mkdir(parents=True, exist_ok=True)
    global_predictor = GlobalCornerPredictor(teacher_paths["r3"])
    roi_predictor = RoiCornerPredictor(_resolve_repo_path(repo_root, str(config["runtime_models"]["roi"])))
    local_predictor = LocalCornerMoEPredictor(teacher_paths["v28"])
    export_summaries: dict[str, Any] = {}
    for split_name in ("train", "val", "holdout"):
        output_rows: list[dict[str, Any]] = []
        teacher_rows: list[dict[str, float]] = []
        r3_rows: list[dict[str, float]] = []
        for row in partitioned[split_name]:
            if not row.get("manual_quad"):
                continue
            image_path = _resolve_image_path(repo_root, row)
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            scene_profile = compute_scene_profile(image, row["manual_quad"])
            start = perf_counter()
            global_quad = global_predictor.predict_image(image)
            opencv_result = detect_best_candidate(image)
            stage = predict_two_stage(
                image_path=image_path,
                global_predictor=global_predictor,
                roi_predictor=roi_predictor,
                local_predictor=local_predictor,
                page_id=str(row.get("page_id") or image_path.stem),
                image=image,
            )
            export_row = {
                "dataset_slug": row.get("dataset_slug"),
                "project_slug": row.get("project_slug"),
                "page_id": row.get("page_id"),
                "image_path": row.get("image_path"),
                "split": split_name,
                "manual_only": bool(row.get("manual_only", False)),
                "manual_process_targets": bool(row.get("manual_only", False))
                and bool((config.get("student") or {}).get("manual_process_targets_for_manual_only", False)),
                "manual_quad": row.get("manual_quad"),
                "active_quad": row.get("active_quad"),
                "scene_profile": scene_profile,
                "scene_tags": scene_profile.get("scene_tags", []),
                "teacher_r3_quad": [[float(x), float(y)] for x, y in order_points(np.array(global_quad, dtype=np.float32))],
                "teacher_v28_quad": stage["final_quad"],
                "teacher_roi_quad": stage["roi_quad"],
                "teacher_runtime_models": {"r3": str(teacher_paths["r3"]), "v28": str(teacher_paths["v28"])},
                "teacher_infer_ms": round((perf_counter() - start) * 1000.0, 2),
            }
            if opencv_result is not None and opencv_result.get("best") is not None:
                export_row["opencv_best_quad"] = [
                    [float(x), float(y)]
                    for x, y in order_points(np.array(opencv_result["best"]["quad"], dtype=np.float32))
                ]
                export_row["opencv_best_method"] = str(opencv_result["best"].get("method", "opencv_best"))
                export_row["opencv_best_score"] = float(opencv_result["best"].get("score", 0.0))
            process_targets = _compute_process_targets(image, export_row["teacher_roi_quad"], export_row["teacher_v28_quad"], page_id=str(row.get("page_id") or image_path.stem))
            export_row.update(
                {
                    "teacher_roi_box": [float(value) for value in process_targets["teacher_roi_box"]],
                    "teacher_refine_delta_norm": process_targets["teacher_refine_delta_norm"].round(6).tolist(),
                    "teacher_corner_visibility": process_targets["teacher_corner_visibility"].round(6).tolist(),
                    "teacher_corner_edge_direction": process_targets["teacher_corner_edge_direction"].round(6).tolist(),
                    "teacher_corner_fallback_mask": process_targets["teacher_corner_fallback_mask"].round(6).tolist(),
                }
            )
            teacher_rows.append(quad_geometry_metrics(row["manual_quad"], export_row["teacher_v28_quad"]))
            r3_rows.append(quad_geometry_metrics(row["manual_quad"], export_row["teacher_r3_quad"]))
            output_rows.append(export_row)
        output_path = round_paths.teacher_export_root / f"{split_name}.jsonl"
        _write_jsonl(output_path, output_rows)
        export_summaries[split_name] = {
            "rows": len(output_rows),
            "path": str(output_path),
            "teacher_summary": summarize_geometry_metric_rows(teacher_rows),
            "r3_summary": summarize_geometry_metric_rows(r3_rows),
        }
    summary_path = round_paths.teacher_export_root / "summary.json"
    _write_json(summary_path, export_summaries)
    return {"teacher_paths": {key: str(value) for key, value in teacher_paths.items()}, "summary_path": str(summary_path), "splits": export_summaries}


def _reuse_teacher_snapshot(repo_root: Path, round_paths: RoundPaths, config: Mapping[str, Any]) -> dict[str, Any]:
    source_value = config.get("reuse_teacher_export_from")
    if not source_value:
        raise ValueError("reuse_teacher_export_from is required")
    source_root = _resolve_repo_path(repo_root, str(source_value))
    source_export_root = source_root / "data" / "teacher_exports" if (source_root / "data" / "teacher_exports").exists() else source_root
    summary_path = source_export_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    round_paths.teacher_export_root.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "val.jsonl", "holdout.jsonl", "summary.json"):
        src = source_export_root / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, round_paths.teacher_export_root / name)
    splits = _read_json(round_paths.teacher_export_root / "summary.json")
    teacher_paths = resolve_teacher_model_paths(repo_root, {alias: str(item["runtime_file"]) for alias, item in config["teachers"].items()})
    return {
        "reused_from": str(source_export_root),
        "teacher_paths": {key: str(value) for key, value in teacher_paths.items()},
        "summary_path": str(round_paths.teacher_export_root / "summary.json"),
        "splits": splits,
    }


def _materialize_student_dataset(repo_root: Path, round_paths: RoundPaths) -> dict[str, str]:
    round_paths.dataset_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, str] = {}
    for split_name in ("train", "val", "holdout"):
        src = round_paths.teacher_export_root / f"{split_name}.jsonl"
        dst = round_paths.dataset_root / f"{split_name}.jsonl"
        rows = _read_jsonl(src)
        for row in rows:
            if "hard_case_border_distance_min" in row:
                continue
            image_path = _resolve_image_path(repo_root, row)
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            height, width = image.shape[:2]
            row["hard_case_border_distance_min"] = round(_compute_border_distance_min(row, image_width=width, image_height=height), 6)
        _write_jsonl(dst, rows)
        manifests[split_name] = str(dst)
    return manifests


def _build_round_manifest(repo_root: Path, round_paths: RoundPaths, config: Mapping[str, Any], export_result: Mapping[str, Any] | None = None, train_result: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = {
        "public_name": config["public_name"],
        "internal_name": config["internal_name"],
        "round": config["round"],
        "dataset_slug": config["dataset_slug"],
        "teacher_models": list(config["teachers"].keys()),
        "teacher_runtime_files": {alias: str((repo_root / "models" / "runtime" / str(item["runtime_file"])).resolve()) for alias, item in config["teachers"].items()},
        "curated_rows": str(_resolve_repo_path(repo_root, str(config["curated_rows"]))),
        "split_file": str(_resolve_repo_path(repo_root, str(config["split_file"]))),
        "status": "locked" if train_result is None else "candidate",
        "data_root": str(round_paths.data_root),
        "teacher_export_root": str(round_paths.teacher_export_root),
        "dataset_root": str(round_paths.dataset_root),
        "checkpoints_root": str(round_paths.checkpoints_root),
        "reports_root": str(round_paths.reports_root),
        "artifacts_root": str(round_paths.artifacts_root),
        "logs_root": str(round_paths.logs_root),
        "student": dict(_student_settings(config)),
        "loss": dict(_loss_settings(config)),
        "architecture": "shared_backbone_fpn_coarse_to_fine_local_moe",
    }
    if export_result is not None:
        manifest["teacher_export_summary"] = export_result
    if train_result is not None:
        manifest["training_summary"] = train_result
    if export_result is not None and train_result is not None:
        holdout_export = export_result["splits"]["holdout"]
        manifest["round_comparison"] = _build_round_comparison(train_result["best_holdout_metrics"], holdout_export["teacher_summary"], holdout_export["r3_summary"])
    return manifest


def _write_round_report(round_paths: RoundPaths, config: Mapping[str, Any], export_result: Mapping[str, Any], train_result: Mapping[str, Any]) -> Path:
    round_paths.reports_root.mkdir(parents=True, exist_ok=True)
    holdout_export = export_result["splits"]["holdout"]
    comparison = _build_round_comparison(train_result["best_holdout_metrics"], holdout_export["teacher_summary"], holdout_export["r3_summary"])
    report_path = round_paths.reports_root / f"{config['round']}_baseline_report.md"
    lines = [
        f"# {config['public_name']} {config['round']} baseline",
        "",
        f"- public name: `{config['public_name']}`",
        f"- internal name: `{config['internal_name']}`",
        f"- round: `{config['round']}`",
        "- architecture: `shared_backbone_fpn_coarse_to_fine_local_moe`",
        "",
        "## Student training",
        "",
        f"- checkpoint: `{train_result['checkpoint_path']}`",
        f"- history: `{train_result['history_path']}`",
        f"- best epoch: `{train_result['best_epoch']}`",
        f"- holdout decision: `{comparison['decision']}` ({comparison['decision_reason']})",
        "",
        "## Validation summary",
        "",
        json.dumps(train_result["best_val_metrics"], ensure_ascii=False, indent=2),
        "",
        "## Holdout comparison",
        "",
        json.dumps(comparison, ensure_ascii=False, indent=2),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_round(config_path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root.resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    config = _read_json(config_path)
    round_paths = build_round_paths(_resolve_repo_path(repo_root, str(config["round_root"])))
    for path in asdict(round_paths).values():
        Path(path).mkdir(parents=True, exist_ok=True)
    export_result = (
        _reuse_teacher_snapshot(repo_root, round_paths, config)
        if config.get("reuse_teacher_export_from")
        else _export_teacher_snapshot(repo_root, round_paths, config)
    )
    dataset_manifests = _materialize_student_dataset(repo_root, round_paths)
    train_result = _train_student(repo_root, config, Path(dataset_manifests["train"]), Path(dataset_manifests["val"]), Path(dataset_manifests["holdout"]), round_paths.checkpoints_root)
    report_path = _write_round_report(round_paths, config, export_result, train_result)
    manifest = _build_round_manifest(repo_root, round_paths, config, export_result, train_result)
    manifest["report_path"] = str(report_path)
    manifest_path = round_paths.round_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), "report_path": str(report_path), "export_result": export_result, "train_result": train_result}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deep_screen_v1 distillation round")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root")
    args = parser.parse_args()
    result = run_round(Path(args.config), repo_root=Path(args.repo_root).resolve() if args.repo_root else None)
    print(json.dumps({"manifest_path": result["manifest_path"], "report_path": result["report_path"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
