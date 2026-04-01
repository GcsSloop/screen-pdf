from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from corner_train import CornerHeatmapNet, build_corner_heatmaps, decode_heatmaps, select_torch_device
from dataset_benchmark import (
    build_project_aware_split,
    build_split,
    load_manual_pages,
    normalized_point_error,
    quad_geometry_metrics,
    summarize_geometry_metric_rows,
)

GLOBAL_BORDER_WIDTH_RATIO = 0.26
GLOBAL_BORDER_HEIGHT_RATIO = 0.18
GLOBAL_FEATURE_MODE_CHANNELS = {
    "rgb": 3,
    "rgb_gray_border": 5,
}


def denormalize_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    restored = np.array(corners, dtype=np.float32).copy()
    restored[:, 0] *= float(width)
    restored[:, 1] *= float(height)
    return restored


def global_feature_channels(feature_mode: str) -> int:
    channels = GLOBAL_FEATURE_MODE_CHANNELS.get(str(feature_mode))
    if channels is None:
        raise ValueError(f"unsupported global feature_mode: {feature_mode}")
    return int(channels)


def build_global_feature_tensor(image_rgb: np.ndarray, feature_mode: str = "rgb") -> np.ndarray:
    image_f = image_rgb.astype(np.float32) / 255.0
    if feature_mode == "rgb":
        return np.transpose(image_f, (2, 0, 1))
    if feature_mode != "rgb_gray_border":
        raise ValueError(f"unsupported global feature_mode: {feature_mode}")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    height, width = gray.shape[:2]
    border = np.zeros((height, width), dtype=np.float32)
    border_width = max(1, int(round(width * GLOBAL_BORDER_WIDTH_RATIO)))
    border_height = max(1, int(round(height * GLOBAL_BORDER_HEIGHT_RATIO)))
    border[:, :border_width] = 1.0
    border[:, max(width - border_width, 0) :] = 1.0
    border[:border_height, :] = 1.0
    border[max(height - border_height, 0) :, :] = 1.0
    border_gray = gray * border
    return np.concatenate(
        [
            np.transpose(image_f, (2, 0, 1)),
            gray[None, ...],
            border_gray[None, ...],
        ],
        axis=0,
    )


def initialize_global_model_from_checkpoint(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    source_state = checkpoint["state_dict"]
    target_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if key not in target_state:
            continue
        target = target_state[key]
        if tuple(target.shape) == tuple(value.shape):
            compatible[key] = value
            continue
        if (
            key == "stem.block.0.weight"
            and value.ndim == 4
            and target.ndim == 4
            and value.shape[0] == target.shape[0]
            and value.shape[2:] == target.shape[2:]
            and value.shape[1] <= target.shape[1]
        ):
            expanded = target.clone()
            expanded[:, : value.shape[1], :, :] = value
            filler = value.mean(dim=1, keepdim=True)
            for channel_index in range(value.shape[1], target.shape[1]):
                expanded[:, channel_index : channel_index + 1, :, :] = filler
            compatible[key] = expanded
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected init keys: {unexpected}")
    allowed_missing = {"stem.block.0.weight"}
    disallowed = [key for key in missing if key not in allowed_missing]
    if disallowed:
        raise RuntimeError(f"incompatible init checkpoint, missing keys: {disallowed}")


def _normalize_manual_quad(quad: list[list[float]] | np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.array(quad, dtype=np.float32)
    arr[:, 0] /= max(float(width), 1.0)
    arr[:, 1] /= max(float(height), 1.0)
    return arr


def _resolve_training_profile_weights(
    training_profile: str,
    geometry_loss_weight: float | None,
    edge_supervision_weight: float | None,
) -> tuple[float, float]:
    if training_profile == "default":
        return (
            1.0 if geometry_loss_weight is None else float(geometry_loss_weight),
            0.2 if edge_supervision_weight is None else float(edge_supervision_weight),
        )
    if training_profile == "legacy_r3":
        return (
            1.0 if geometry_loss_weight is None else float(geometry_loss_weight),
            0.0 if edge_supervision_weight is None else float(edge_supervision_weight),
        )
    raise ValueError(f"unsupported training_profile: {training_profile}")


def reframe_image_and_corners(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    scale: float,
    shift_x: float,
    shift_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center_x = 0.5 * float(width)
    center_y = 0.5 * float(height)
    tx = (1.0 - float(scale)) * center_x + float(shift_x) * float(width)
    ty = (1.0 - float(scale)) * center_y + float(shift_y) * float(height)
    transform = np.array(
        [[float(scale), 0.0, tx], [0.0, float(scale), ty]],
        dtype=np.float32,
    )
    reframed = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed_corners = transform_corners_with_reframe(
        corners,
        scale=scale,
        shift_x=shift_x,
        shift_y=shift_y,
    )
    return reframed, transformed_corners


def transform_corners_with_reframe(
    corners: np.ndarray,
    *,
    scale: float,
    shift_x: float,
    shift_y: float,
) -> np.ndarray:
    transformed_corners = np.array(corners, dtype=np.float32).copy()
    transformed_corners[:, 0] = (transformed_corners[:, 0] - 0.5) * float(scale) + 0.5 + float(shift_x)
    transformed_corners[:, 1] = (transformed_corners[:, 1] - 0.5) * float(scale) + 0.5 + float(shift_y)
    return np.clip(transformed_corners, 0.0, 1.0)


def apply_global_perspective_augmentation(
    image: np.ndarray,
    corners: np.ndarray,
    *,
    jitter: np.ndarray | None = None,
    jitter_ratio: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    src = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    if jitter is None:
        max_jitter_x = width * float(jitter_ratio)
        max_jitter_y = height * float(jitter_ratio)
        jitter = np.array(
            [
                [random.uniform(-max_jitter_x, max_jitter_x), random.uniform(-max_jitter_y, max_jitter_y)],
                [random.uniform(-max_jitter_x, max_jitter_x), random.uniform(-max_jitter_y, max_jitter_y)],
                [random.uniform(-max_jitter_x, max_jitter_x), random.uniform(-max_jitter_y, max_jitter_y)],
                [random.uniform(-max_jitter_x, max_jitter_x), random.uniform(-max_jitter_y, max_jitter_y)],
            ],
            dtype=np.float32,
        )
    dst = src + np.array(jitter, dtype=np.float32)
    dst[:, 0] = np.clip(dst[:, 0], 0.0, width - 1.0)
    dst[:, 1] = np.clip(dst[:, 1], 0.0, height - 1.0)
    transform = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    quad_px = transform_corners_with_perspective(corners, transform=transform, width=width, height=height)
    return warped_image, quad_px


def transform_corners_with_perspective(
    corners: np.ndarray,
    *,
    transform: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    quad_px = np.array(corners, dtype=np.float32).copy()
    quad_px[:, 0] *= max(width - 1.0, 1.0)
    quad_px[:, 1] *= max(height - 1.0, 1.0)
    quad_px = cv2.perspectiveTransform(quad_px[None, :, :], transform)[0]
    quad_px[:, 0] /= max(width - 1.0, 1.0)
    quad_px[:, 1] /= max(height - 1.0, 1.0)
    return np.clip(quad_px, 0.0, 1.0)


def _allowed_shift_range(corners: np.ndarray, scale: float) -> tuple[float, float]:
    scaled = corners.astype(np.float32).copy()
    scaled[:, 0] = (scaled[:, 0] - 0.5) * float(scale) + 0.5
    min_x = float(np.min(scaled[:, 0]))
    max_x = float(np.max(scaled[:, 0]))
    return -min_x, 1.0 - max_x


def is_legacy_dark_muted_scene(image: np.ndarray) -> bool:
    if image.size == 0:
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_saturation = float(np.mean(hsv[..., 1]))
    mean_value = float(np.mean(hsv[..., 2]))
    return 20.0 <= mean_saturation <= 95.0 and 72.0 <= mean_value <= 110.0


def is_geometry_priority_low_contrast_scene(row: dict[str, Any]) -> bool:
    tags = set(row.get("scene_tags") or [])
    if "near_color_background" in tags or "low_contrast_scene" in tags:
        return True
    profile = row.get("scene_profile") or {}
    if not isinstance(profile, dict):
        return False
    lab_distance = float(profile.get("lab_distance", 1e9))
    luma_delta = abs(float(profile.get("luma_delta", 1.0)))
    edge_strength = float(profile.get("edge_strength", 0.0))
    inner_border_contrast = abs(float(profile.get("inner_border_contrast", 1.0)))
    return (
        lab_distance <= 28.0
        and luma_delta <= 0.12
        and edge_strength >= 0.25
        and inner_border_contrast <= 0.07
    )


def bootstrap_inward_geometry_scale(row: dict[str, Any]) -> float:
    metrics = row.get("bootstrap_metrics") or {}
    if not isinstance(metrics, dict):
        return 1.0
    inset = float(metrics.get("quad_inset_ratio", 0.0))
    screen_error = float(metrics.get("screen_relative_point_error", 0.0))
    max_corner = float(metrics.get("max_corner_error", 0.0))
    if inset <= 0.03:
        return 1.0
    inset_boost = min(0.7, max(0.0, (inset - 0.03) * 2.5))
    screen_boost = min(0.35, max(0.0, screen_error - 0.01) * 6.0)
    corner_boost = min(0.25, max(0.0, max_corner - 0.03) * 2.0)
    return float(1.0 + inset_boost + screen_boost + corner_boost)


def is_bootstrap_inward_hard_page(row: dict[str, Any]) -> bool:
    metrics = row.get("bootstrap_metrics") or {}
    if not isinstance(metrics, dict):
        return False
    inset = float(metrics.get("quad_inset_ratio", 0.0))
    screen_error = float(metrics.get("screen_relative_point_error", 0.0))
    max_corner = float(metrics.get("max_corner_error", 0.0))
    return inset >= 0.08 and (screen_error >= 0.045 or max_corner >= 0.10)


def build_adaptive_teacher_target(
    manual_corners: np.ndarray,
    teacher_corners: np.ndarray | None,
    *,
    blend_ratio: float,
    corner_error_max: float | None = None,
    sample_error_max: float | None = None,
) -> tuple[np.ndarray, float]:
    manual = np.array(manual_corners, dtype=np.float32)
    if teacher_corners is None or float(blend_ratio) <= 0.0:
        return manual.copy(), 0.0
    teacher = np.array(teacher_corners, dtype=np.float32)
    bbox = np.amax(manual, axis=0) - np.amin(manual, axis=0)
    diag = float(np.linalg.norm(np.clip(bbox, 1e-6, None)))
    if diag <= 1e-6:
        return manual.copy(), 0.0
    corner_errors = np.linalg.norm(teacher - manual, axis=1) / diag
    if sample_error_max is not None and float(sample_error_max) > 0.0 and float(corner_errors.mean()) > float(sample_error_max):
        return manual.copy(), 0.0
    corner_mask = np.ones((4,), dtype=bool)
    if corner_error_max is not None and float(corner_error_max) > 0.0:
        corner_mask &= corner_errors <= float(corner_error_max)
    guidance_scale = float(corner_mask.mean())
    if guidance_scale <= 0.0:
        return manual.copy(), 0.0
    target = manual.copy()
    target[corner_mask] = manual[corner_mask] + float(blend_ratio) * (teacher[corner_mask] - manual[corner_mask])
    return np.clip(target, 0.0, 1.0), guidance_scale


def build_multi_teacher_target(
    manual_corners: np.ndarray,
    candidate_corners: np.ndarray | None,
    *,
    blend_ratio: float,
    candidate_mask: np.ndarray | None = None,
    corner_error_max: float | None = None,
    sample_error_max: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    manual = np.array(manual_corners, dtype=np.float32)
    selected_index = np.full((4,), -1, dtype=np.int64)
    if candidate_corners is None or float(blend_ratio) <= 0.0:
        return manual.copy(), 0.0, selected_index
    candidates = np.array(candidate_corners, dtype=np.float32)
    if candidates.ndim == 2:
        candidates = candidates[None, ...]
    if candidates.size == 0:
        return manual.copy(), 0.0, selected_index
    bbox = np.amax(manual, axis=0) - np.amin(manual, axis=0)
    diag = float(np.linalg.norm(np.clip(bbox, 1e-6, None)))
    if diag <= 1e-6:
        return manual.copy(), 0.0, selected_index
    if candidate_mask is None:
        valid_candidates = np.ones((candidates.shape[0],), dtype=bool)
    else:
        valid_candidates = np.array(candidate_mask, dtype=bool).reshape(-1)
        if valid_candidates.shape[0] != candidates.shape[0]:
            raise ValueError("candidate_mask length must match candidate count")
    if not np.any(valid_candidates):
        return manual.copy(), 0.0, selected_index

    corner_errors = np.linalg.norm(candidates - manual[None, :, :], axis=2) / diag
    corner_errors[~valid_candidates, :] = np.inf
    best_errors = np.min(corner_errors, axis=0)
    finite_mask = np.isfinite(best_errors)
    if np.any(finite_mask):
        selected_index[finite_mask] = np.argmin(corner_errors[:, finite_mask], axis=0).astype(np.int64)
    if sample_error_max is not None and float(sample_error_max) > 0.0:
        sample_error = float(np.mean(np.where(finite_mask, best_errors, 1e6)))
        if sample_error > float(sample_error_max):
            return manual.copy(), 0.0, selected_index
    corner_mask = finite_mask.copy()
    if corner_error_max is not None and float(corner_error_max) > 0.0:
        corner_mask &= best_errors <= float(corner_error_max)
    guidance_scale = float(corner_mask.mean())
    if guidance_scale <= 0.0:
        return manual.copy(), 0.0, selected_index
    best_points = manual.copy()
    for corner_index in range(4):
        candidate_index = int(selected_index[corner_index])
        if candidate_index >= 0:
            best_points[corner_index] = candidates[candidate_index, corner_index]
    target = manual.copy()
    target[corner_mask] = manual[corner_mask] + float(blend_ratio) * (best_points[corner_mask] - manual[corner_mask])
    return np.clip(target, 0.0, 1.0), guidance_scale, selected_index


def _normalize_optional_quad(
    quad: list[list[float]] | np.ndarray | None,
    width: int,
    height: int,
) -> np.ndarray | None:
    if quad is None:
        return None
    return _normalize_manual_quad(quad, width, height)


def _resolve_teacher_candidate_quads(
    row: dict[str, Any],
    width: int,
    height: int,
    *,
    sources: tuple[str, ...],
    opencv_score_min: float | None = None,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for source in sources:
        source_key = str(source).strip().lower()
        quad: np.ndarray | None = None
        if source_key == "teacher":
            quad = _normalize_optional_quad(row.get("teacher_quad"), width, height)
        elif source_key == "r3":
            quad = _normalize_optional_quad(row.get("teacher_r3_quad"), width, height)
        elif source_key == "v28":
            quad = _normalize_optional_quad(row.get("teacher_v28_quad"), width, height)
        elif source_key == "roi":
            quad = _normalize_optional_quad(row.get("teacher_roi_quad"), width, height)
        elif source_key == "opencv":
            score = row.get("opencv_best_score")
            if opencv_score_min is not None and score is not None and float(score) < float(opencv_score_min):
                quad = None
            else:
                quad = _normalize_optional_quad(row.get("opencv_best_quad"), width, height)
        else:
            raise ValueError(f"unsupported teacher candidate source: {source}")
        if quad is not None:
            candidates.append(quad)
    return candidates


def _apply_foreground_occluder(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    overlay = image.astype(np.float32).copy()
    mask = np.zeros((height, width), dtype=np.uint8)
    anchor_x = int(random.uniform(0.18, 0.52) * width)
    anchor_y = int(random.uniform(0.72, 0.90) * height)
    occluder_width = int(random.uniform(0.16, 0.30) * width)
    occluder_height = int(random.uniform(0.14, 0.24) * height)
    points = np.array(
        [
            [anchor_x - occluder_width, height - 1],
            [anchor_x + occluder_width, height - 1],
            [anchor_x + int(0.35 * occluder_width), anchor_y],
            [anchor_x - int(0.55 * occluder_width), anchor_y + int(0.08 * occluder_height)],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, points, 255)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=7.0)
    darkness = random.uniform(0.28, 0.48)
    overlay[mask > 0] *= 1.0 - darkness
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _quad_area_torch(corners: torch.Tensor) -> torch.Tensor:
    x = corners[..., 0]
    y = corners[..., 1]
    x_next = torch.cat([x[:, 1:], x[:, :1]], dim=1)
    y_next = torch.cat([y[:, 1:], y[:, :1]], dim=1)
    return 0.5 * torch.sum(x * y_next - y * x_next, dim=1)


def _smooth_abs_torch(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(value * value + eps) - math.sqrt(eps)


def _edge_vectors_torch(corners: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_corners = torch.cat([corners[:, 1:, :], corners[:, :1, :]], dim=1)
    edges = next_corners - corners
    norms = torch.sqrt(torch.sum(edges * edges, dim=-1, keepdim=True) + 1e-12)
    return next_corners, edges, norms


def _point_line_distance_torch(points: torch.Tensor, line_start: torch.Tensor, line_end: torch.Tensor) -> torch.Tensor:
    line = line_end - line_start
    rel = points - line_start
    cross = rel[..., 0] * line[..., 1] - rel[..., 1] * line[..., 0]
    denom = torch.sqrt(torch.sum(line * line, dim=-1) + 1e-12)
    return _smooth_abs_torch(cross) / torch.clamp(denom, min=1e-6)


def _point_segment_distance_map_torch(
    points: torch.Tensor,
    segment_start: torch.Tensor,
    segment_end: torch.Tensor,
) -> torch.Tensor:
    segment = segment_end - segment_start
    rel = points - segment_start.unsqueeze(-2).unsqueeze(-2)
    denom = torch.sum(segment * segment, dim=-1, keepdim=True).unsqueeze(-2).unsqueeze(-2)
    proj = torch.sum(rel * segment.unsqueeze(-2).unsqueeze(-2), dim=-1, keepdim=True) / torch.clamp(denom, min=1e-6)
    proj = torch.clamp(proj, 0.0, 1.0)
    closest = segment_start.unsqueeze(-2).unsqueeze(-2) + proj * segment.unsqueeze(-2).unsqueeze(-2)
    delta = points - closest
    return torch.sqrt(torch.sum(delta * delta, dim=-1) + 1e-12)


def build_quad_edge_soft_map(
    corners: torch.Tensor,
    *,
    output_size: int = 64,
    sigma: float = 0.03,
) -> torch.Tensor:
    xs = torch.linspace(0.0, 1.0, output_size, device=corners.device, dtype=corners.dtype)
    ys = torch.linspace(0.0, 1.0, output_size, device=corners.device, dtype=corners.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    next_corners = torch.cat([corners[:, 1:, :], corners[:, :1, :]], dim=1)
    edge_maps: list[torch.Tensor] = []
    for edge_index in range(4):
        distance = _point_segment_distance_map_torch(points, corners[:, edge_index, :], next_corners[:, edge_index, :])
        edge_maps.append(torch.exp(-(distance * distance) / max(2.0 * sigma * sigma, 1e-6)))
    return torch.stack(edge_maps, dim=1)


def build_quad_soft_mask(
    corners: torch.Tensor,
    *,
    output_size: int = 64,
    sharpness: float = 96.0,
) -> torch.Tensor:
    xs = torch.linspace(0.0, 1.0, output_size, device=corners.device, dtype=corners.dtype)
    ys = torch.linspace(0.0, 1.0, output_size, device=corners.device, dtype=corners.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)

    next_corners = torch.cat([corners[:, 1:, :], corners[:, :1, :]], dim=1)
    edge_start = corners.unsqueeze(-2).unsqueeze(-2)
    edge_end = next_corners.unsqueeze(-2).unsqueeze(-2)
    edge = edge_end - edge_start
    rel = points - edge_start
    cross = rel[..., 0] * edge[..., 1] - rel[..., 1] * edge[..., 0]

    centers = corners.mean(dim=1, keepdim=True)
    center_rel = centers - corners
    center_cross = center_rel[..., 0] * (next_corners - corners)[..., 1] - center_rel[..., 1] * (next_corners - corners)[..., 0]
    side_sign = torch.where(center_cross >= 0.0, 1.0, -1.0).unsqueeze(-1).unsqueeze(-1)
    signed_inside = cross * side_sign
    return torch.prod(torch.sigmoid(signed_inside * float(sharpness)), dim=1)


def build_inner_boundary_band_soft_map(
    corners: torch.Tensor,
    *,
    output_size: int = 64,
    shrink_ratio: float = 0.06,
    sharpness: float = 96.0,
) -> torch.Tensor:
    center = corners.mean(dim=1, keepdim=True)
    shrunk = center + (corners - center) * max(1.0 - float(shrink_ratio), 1e-3)
    outer_mask = build_quad_soft_mask(corners, output_size=output_size, sharpness=sharpness)
    inner_mask = build_quad_soft_mask(shrunk, output_size=output_size, sharpness=sharpness)
    return torch.clamp(outer_mask - inner_mask, min=0.0, max=1.0)


def compute_quad_mask_supervision_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    output_size: int = 64,
    sharpness: float = 96.0,
    sample_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    per_sample = compute_quad_mask_supervision_loss_per_sample(
        predicted,
        target,
        output_size=output_size,
        sharpness=sharpness,
    )
    if sample_scales is not None:
        scales = sample_scales.to(device=predicted.device, dtype=predicted.dtype).view(-1)
        return torch.sum(per_sample * scales) / torch.clamp(scales.sum(), min=1e-6)
    return per_sample.mean()


def compute_quad_mask_supervision_loss_per_sample(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    output_size: int = 64,
    sharpness: float = 96.0,
) -> torch.Tensor:
    pred_mask = build_quad_soft_mask(predicted, output_size=output_size, sharpness=sharpness)
    target_mask = build_quad_soft_mask(target, output_size=output_size, sharpness=sharpness)
    return torch.mean((pred_mask - target_mask) ** 2, dim=(1, 2))


def compute_inner_boundary_band_loss_per_sample(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    output_size: int = 64,
    shrink_ratio: float = 0.06,
    sharpness: float = 96.0,
) -> torch.Tensor:
    pred_band = build_inner_boundary_band_soft_map(
        predicted,
        output_size=output_size,
        shrink_ratio=shrink_ratio,
        sharpness=sharpness,
    )
    target_band = build_inner_boundary_band_soft_map(
        target,
        output_size=output_size,
        shrink_ratio=shrink_ratio,
        sharpness=sharpness,
    )
    return torch.mean((pred_band - target_band) ** 2, dim=(1, 2))


def compute_edge_supervision_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    output_size: int = 64,
    sigma: float = 0.03,
    sample_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_edges = build_quad_edge_soft_map(predicted, output_size=output_size, sigma=sigma)
    target_edges = build_quad_edge_soft_map(target, output_size=output_size, sigma=sigma)
    per_sample = torch.mean((pred_edges - target_edges) ** 2, dim=(1, 2, 3))
    if sample_scales is not None:
        scales = sample_scales.to(device=predicted.device, dtype=predicted.dtype).view(-1)
        return torch.sum(per_sample * scales) / torch.clamp(scales.sum(), min=1e-6)
    return per_sample.mean()


def compute_global_geometry_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    max_corner_weight: float = 1.15,
    edge_weight: float = 0.35,
    edge_line_weight: float = 0.9,
    edge_length_weight: float = 0.4,
    edge_collapse_weight: float = 0.6,
    corner_line_weight: float = 0.7,
    corner_angle_weight: float = 0.28,
    inset_weight: float = 0.25,
    inward_boundary_weight: float = 0.0,
    inward_boundary_margin: float = 0.0,
    quad_mask_weight: float = 0.0,
    inner_boundary_band_weight: float = 0.0,
    sample_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    deltas = predicted - target
    distances = torch.sqrt(torch.sum(deltas * deltas, dim=-1))
    max_corner = distances.max(dim=1).values

    pred_next, pred_edges, pred_edge_norm = _edge_vectors_torch(predicted)
    target_next, target_edges, target_edge_norm = _edge_vectors_torch(target)
    pred_edges = pred_edges / torch.clamp(pred_edge_norm, min=1e-6)
    target_edges = target_edges / torch.clamp(target_edge_norm, min=1e-6)
    edge_direction = (1.0 - torch.sum(pred_edges * target_edges, dim=-1)).mean(dim=1)

    target_line = target_next - target
    target_line_norm = torch.sqrt(torch.sum(target_line * target_line, dim=-1, keepdim=True) + 1e-12)
    target_normal = torch.stack([target_line[..., 1], -target_line[..., 0]], dim=-1) / torch.clamp(target_line_norm, min=1e-6)
    pred_mid = 0.5 * (predicted + pred_next)
    target_mid = 0.5 * (target + target_next)
    edge_signed_offset = torch.sum((pred_mid - target_mid) * target_normal, dim=-1)
    edge_line_offset = _smooth_abs_torch(edge_signed_offset).mean(dim=1)
    inward_edge_offset = torch.relu((-edge_signed_offset) - inward_boundary_margin).mean(dim=1)

    pred_lengths = torch.sqrt(torch.sum((pred_next - predicted) * (pred_next - predicted), dim=-1) + 1e-12)
    target_lengths = torch.sqrt(torch.sum((target_next - target) * (target_next - target), dim=-1) + 1e-12)
    edge_length_ratio = _smooth_abs_torch(pred_lengths - target_lengths) / torch.clamp(target_lengths, min=1e-6)
    edge_length_ratio = edge_length_ratio.mean(dim=1)
    collapse_ratio = pred_lengths / torch.clamp(target_lengths, min=1e-6)
    edge_collapse = torch.relu(0.7 - collapse_ratio) / 0.7
    edge_collapse = torch.max(edge_collapse * edge_collapse, dim=1).values

    target_prev = torch.cat([target[:, -1:, :], target[:, :-1, :]], dim=1)
    corner_prev_line = _point_line_distance_torch(predicted, target_prev, target)
    corner_next_line = _point_line_distance_torch(predicted, target, target_next)
    corner_line = 0.5 * (corner_prev_line + corner_next_line)
    corner_line = corner_line.mean(dim=1)
    target_prev_edge = target - target_prev
    target_prev_edge_norm = torch.sqrt(torch.sum(target_prev_edge * target_prev_edge, dim=-1, keepdim=True) + 1e-12)
    target_prev_normal = torch.stack([target_prev_edge[..., 1], -target_prev_edge[..., 0]], dim=-1) / torch.clamp(
        target_prev_edge_norm,
        min=1e-6,
    )
    corner_prev_signed = torch.sum((predicted - target) * target_prev_normal, dim=-1)
    corner_next_signed = torch.sum((predicted - target) * target_normal, dim=-1)
    inward_corner_offset = 0.5 * (
        torch.relu((-corner_prev_signed) - inward_boundary_margin)
        + torch.relu((-corner_next_signed) - inward_boundary_margin)
    )
    inward_boundary = 0.5 * (inward_edge_offset + inward_corner_offset.mean(dim=1))

    pred_prev = torch.cat([predicted[:, -1:, :], predicted[:, :-1, :]], dim=1)
    pred_in = predicted - pred_prev
    pred_out = pred_next - predicted
    target_in = target - target_prev
    target_out = target_next - target
    pred_in = pred_in / torch.clamp(torch.sqrt(torch.sum(pred_in * pred_in, dim=-1, keepdim=True) + 1e-12), min=1e-6)
    pred_out = pred_out / torch.clamp(torch.sqrt(torch.sum(pred_out * pred_out, dim=-1, keepdim=True) + 1e-12), min=1e-6)
    target_in = target_in / torch.clamp(torch.sqrt(torch.sum(target_in * target_in, dim=-1, keepdim=True) + 1e-12), min=1e-6)
    target_out = target_out / torch.clamp(torch.sqrt(torch.sum(target_out * target_out, dim=-1, keepdim=True) + 1e-12), min=1e-6)
    pred_corner_cos = torch.sum((-pred_in) * pred_out, dim=-1)
    target_corner_cos = torch.sum((-target_in) * target_out, dim=-1)
    corner_angle = _smooth_abs_torch(pred_corner_cos - target_corner_cos).mean(dim=1)

    pred_area = _quad_area_torch(predicted)
    target_area = torch.clamp(_quad_area_torch(target), min=1e-6)
    inset = torch.relu((target_area - pred_area) / torch.clamp(target_area, min=1e-6))
    quad_mask = compute_quad_mask_supervision_loss_per_sample(
        predicted,
        target,
        output_size=64,
        sharpness=96.0,
    )
    inner_boundary_band = compute_inner_boundary_band_loss_per_sample(
        predicted,
        target,
        output_size=64,
        shrink_ratio=0.06,
        sharpness=96.0,
    )

    total_per_sample = (
        max_corner * max_corner_weight
        + edge_direction * edge_weight
        + edge_line_offset * edge_line_weight
        + edge_length_ratio * edge_length_weight
        + edge_collapse * edge_collapse_weight
        + corner_line * corner_line_weight
        + corner_angle * corner_angle_weight
        + inset * inset_weight
        + inward_boundary * inward_boundary_weight
        + quad_mask * quad_mask_weight
        + inner_boundary_band * inner_boundary_band_weight
    )
    if sample_scales is not None:
        scales = sample_scales.to(device=predicted.device, dtype=predicted.dtype).view(-1)
        total = torch.sum(total_per_sample * scales) / torch.clamp(scales.sum(), min=1e-6)
    else:
        total = total_per_sample.mean()
    return total, {
        "max_corner": max_corner.mean().detach(),
        "edge_direction": edge_direction.mean().detach(),
        "edge_line_offset": edge_line_offset.mean().detach(),
        "edge_length_ratio": edge_length_ratio.mean().detach(),
        "edge_collapse": edge_collapse.mean().detach(),
        "corner_line": corner_line.mean().detach(),
        "corner_angle": corner_angle.mean().detach(),
        "inset": inset.mean().detach(),
        "inward_boundary": inward_boundary.mean().detach(),
        "quad_mask": quad_mask.mean().detach(),
        "inner_boundary_band": inner_boundary_band.mean().detach(),
    }


class GlobalCornerDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        input_size: int = 256,
        output_size: int = 64,
        augment: bool = False,
        training_profile: str = "default",
        feature_mode: str = "rgb",
        teacher_blend_ratio: float = 0.0,
        teacher_corner_error_max: float | None = None,
        teacher_sample_error_max: float | None = None,
        teacher_target_mode: str = "adaptive",
        teacher_candidate_sources: tuple[str, ...] = ("teacher",),
        teacher_opencv_score_min: float | None = None,
        teacher_activation_min_disagreement: float | None = None,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment
        self.training_profile = training_profile
        self.feature_mode = feature_mode
        self.teacher_blend_ratio = float(teacher_blend_ratio)
        self.teacher_corner_error_max = None if teacher_corner_error_max is None else float(teacher_corner_error_max)
        self.teacher_sample_error_max = None if teacher_sample_error_max is None else float(teacher_sample_error_max)
        self.teacher_target_mode = str(teacher_target_mode)
        self.teacher_candidate_sources = tuple(str(item) for item in teacher_candidate_sources if str(item).strip())
        self.teacher_opencv_score_min = None if teacher_opencv_score_min is None else float(teacher_opencv_score_min)
        self.teacher_activation_min_disagreement = (
            None if teacher_activation_min_disagreement is None else float(teacher_activation_min_disagreement)
        )
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        self.rows = [json.loads(line) for line in lines if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def _row_teacher_disagreement(self, row: dict[str, Any]) -> float:
        teacher_quad = np.array(row.get("teacher_quad") or [], dtype=np.float32)
        coarse_quad = np.array(row.get("teacher_r3_quad") or [], dtype=np.float32)
        if teacher_quad.shape != (4, 2) or coarse_quad.shape != (4, 2):
            return 0.0
        stacked = np.concatenate([teacher_quad, coarse_quad], axis=0)
        min_xy = np.min(stacked, axis=0)
        max_xy = np.max(stacked, axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        teacher_norm = (teacher_quad - min_xy) / span
        coarse_norm = (coarse_quad - min_xy) / span
        return float(normalized_point_error(teacher_norm, coarse_norm))

    def build_sample_weights(
        self,
        power: float = 1.0,
        hardness_score_power: float = 0.0,
        disagreement_floor: float | None = None,
        disagreement_boost: float = 1.0,
        project_balance_power: float = 0.0,
        geometry_priority_boost: float = 1.0,
        failure_layer_runtime_boost: float = 1.0,
        failure_layer_opencv_boost: float = 1.0,
        failure_layer_hard_boost: float = 1.0,
        failure_layer_gain_power: float = 0.0,
        failure_layer_project_balance: bool = False,
    ) -> np.ndarray:
        project_gain_means: dict[str, float] = {}
        project_counts: dict[str, int] = {}
        if failure_layer_project_balance and failure_layer_gain_power > 0.0:
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in self.rows:
                project_key = str(row.get("project_name") or row.get("project_slug") or "__unknown__")
                grouped[project_key].append(max(float(row.get("failure_layer_union_gain", 0.0) or 0.0), 0.0))
            project_gain_means = {
                key: (sum(values) / max(len(values), 1))
                for key, values in grouped.items()
            }
        if project_balance_power > 0.0:
            grouped_counts: dict[str, int] = defaultdict(int)
            for row in self.rows:
                project_key = str(row.get("project_name") or row.get("project_slug") or "__unknown__")
                grouped_counts[project_key] += 1
            project_counts = dict(grouped_counts)
            mean_project_count = sum(project_counts.values()) / max(len(project_counts), 1)
        else:
            mean_project_count = 0.0
        weights = []
        for row in self.rows:
            weight = float(max(row.get("adaptive_weight", 1.0), 0.1)) ** float(power)
            if hardness_score_power > 0.0:
                hardness_score = max(float(row.get("hardness_score", 0.0) or 0.0), 0.0)
                weight *= float(1.0 + math.log1p(hardness_score)) ** float(hardness_score_power)
            if project_balance_power > 0.0 and project_counts:
                project_key = str(row.get("project_name") or row.get("project_slug") or "__unknown__")
                project_count = max(float(project_counts.get(project_key, 1)), 1.0)
                weight *= float(mean_project_count / project_count) ** float(project_balance_power)
            if (
                disagreement_floor is not None
                and disagreement_floor > 0.0
                and disagreement_boost > 1.0
                and self._row_teacher_disagreement(row) >= float(disagreement_floor)
            ):
                weight *= float(disagreement_boost)
            if geometry_priority_boost > 1.0 and (
                is_geometry_priority_low_contrast_scene(row) or is_bootstrap_inward_hard_page(row)
            ):
                weight *= float(geometry_priority_boost)
            category = str(row.get("failure_layer_category") or "").strip()
            if category == "runtime_candidate_recoverable":
                weight *= float(max(failure_layer_runtime_boost, 1.0))
            elif category == "opencv_recoverable":
                weight *= float(max(failure_layer_opencv_boost, 1.0))
            elif category == "hard_both_fail":
                weight *= float(max(failure_layer_hard_boost, 1.0))
            if failure_layer_gain_power > 0.0:
                union_gain = max(float(row.get("failure_layer_union_gain", 0.0) or 0.0), 0.0)
                if failure_layer_project_balance:
                    project_key = str(row.get("project_name") or row.get("project_slug") or "__unknown__")
                    project_mean = float(project_gain_means.get(project_key, 0.0) or 0.0)
                    if project_mean > 0.0:
                        union_gain = union_gain / project_mean
                weight *= float(1.0 + union_gain) ** float(failure_layer_gain_power)
            weights.append(weight)
        if not weights:
            return np.ones((0,), dtype=np.float32)
        arr = np.array(weights, dtype=np.float32)
        return arr / max(float(arr.mean()), 1e-6)

    def _geometry_scale(self, row: dict[str, Any]) -> float:
        if self.training_profile == "legacy_r3":
            return 1.0
        tags = set(row.get("scene_tags") or [])
        scale = 0.85
        if "bright_screen" in tags:
            scale = min(scale, 0.8)
        if "low_contrast_scene" in tags:
            scale = max(scale, 1.1)
        if "near_color_background" in tags:
            scale = max(scale, 1.2)
        if "black_frame_scene" in tags:
            scale = max(scale, 1.25)
        if is_geometry_priority_low_contrast_scene(row):
            scale = max(scale, 1.2)
        scale *= bootstrap_inward_geometry_scale(row)
        return float(scale)

    def _edge_scale(self, corners: np.ndarray) -> float:
        if self.training_profile == "legacy_r3":
            return 1.0
        width_fraction = float(np.max(corners[:, 0]) - np.min(corners[:, 0]))
        if width_fraction >= 0.84:
            return 0.85
        if width_fraction <= 0.60:
            return 1.45
        ratio = (0.84 - width_fraction) / 0.24
        return float(0.85 + np.clip(ratio, 0.0, 1.0) * 0.6)

    def _teacher_activation_scale(self, row: dict[str, Any], width: int, height: int) -> float:
        threshold = self.teacher_activation_min_disagreement
        if threshold is None or threshold <= 0.0:
            return 1.0
        teacher_quad = _normalize_optional_quad(row.get("teacher_quad"), width, height)
        coarse_quad = _normalize_optional_quad(row.get("teacher_r3_quad"), width, height)
        if teacher_quad is None or coarse_quad is None:
            return 1.0
        disagreement = normalized_point_error(teacher_quad, coarse_quad)
        return 1.0 if disagreement >= float(threshold) else 0.0

    def _augment(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        row: dict[str, Any],
        extra_quads: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        extra_quads = [np.array(quad, dtype=np.float32).copy() for quad in (extra_quads or [])]

        def _return(image_value: np.ndarray, primary: np.ndarray):
            if extra_quads:
                return image_value, primary, extra_quads
            return image_value, primary

        def _flip_all() -> None:
            nonlocal corners
            corners[:, 0] = 1.0 - corners[:, 0]
            corners = corners[[1, 0, 3, 2]]
            for index, quad in enumerate(extra_quads):
                quad[:, 0] = 1.0 - quad[:, 0]
                extra_quads[index] = quad[[1, 0, 3, 2]]

        def _reframe_all(scale: float, shift_x: float, shift_y: float) -> None:
            nonlocal image, corners
            image, corners = reframe_image_and_corners(image, corners, scale=scale, shift_x=shift_x, shift_y=shift_y)
            for index, quad in enumerate(extra_quads):
                extra_quads[index] = transform_corners_with_reframe(
                    quad,
                    scale=scale,
                    shift_x=shift_x,
                    shift_y=shift_y,
                )

        def _perspective_all(jitter_ratio: float) -> None:
            nonlocal image, corners
            height, width = image.shape[:2]
            max_jitter_x = width * float(jitter_ratio)
            max_jitter_y = height * float(jitter_ratio)
            jitter = np.array(
                [
                    [random.uniform(-max_jitter_x, max_jitter_x), random.uniform(-max_jitter_y, max_jitter_y)],
                    [random.uniform(-max_jitter_x, max_jitter_x), random.uniform(-max_jitter_y, max_jitter_y)],
                    [random.uniform(-max_jitter_x, max_jitter_x), random.uniform(-max_jitter_y, max_jitter_y)],
                    [random.uniform(-max_jitter_x, max_jitter_x), random.uniform(-max_jitter_y, max_jitter_y)],
                ],
                dtype=np.float32,
            )
            src = np.array(
                [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
                dtype=np.float32,
            )
            dst = src + jitter
            dst[:, 0] = np.clip(dst[:, 0], 0.0, width - 1.0)
            dst[:, 1] = np.clip(dst[:, 1], 0.0, height - 1.0)
            transform = cv2.getPerspectiveTransform(src, dst)
            image = cv2.warpPerspective(
                image,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            corners = transform_corners_with_perspective(corners, transform=transform, width=width, height=height)
            for index, quad in enumerate(extra_quads):
                extra_quads[index] = transform_corners_with_perspective(quad, transform=transform, width=width, height=height)

        if not self.augment:
            return _return(image, corners)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            _flip_all()
        width_fraction = float(np.max(corners[:, 0]) - np.min(corners[:, 0]))
        center_x = float(np.mean(corners[:, 0]))
        if (
            self.training_profile != "legacy_r3"
            and is_bootstrap_inward_hard_page(row)
            and width_fraction >= 0.70
            and 0.34 <= center_x <= 0.66
            and random.random() < 0.28
        ):
            scale = random.uniform(0.78, 0.88)
            shift_left, shift_right = _allowed_shift_range(corners, scale)
            shift_floor = max(-0.06, shift_left)
            shift_cap = min(0.06, shift_right)
            if shift_floor < shift_cap:
                shift_x = random.uniform(shift_floor, shift_cap)
                shift_y = random.uniform(-0.07, -0.02)
                _reframe_all(
                    scale=scale,
                    shift_x=shift_x,
                    shift_y=shift_y,
                )
                if random.random() < 0.35:
                    _perspective_all(jitter_ratio=0.05)
        if (
            self.training_profile == "legacy_r3"
            and width_fraction >= 0.72
            and 0.38 <= center_x <= 0.62
            and is_legacy_dark_muted_scene(image)
            and random.random() < 0.12
        ):
            scale = random.uniform(0.48, 0.62)
            shift_left, shift_right = _allowed_shift_range(corners, scale)
            shift_floor = max(-0.04, shift_left)
            shift_cap = min(0.04, shift_right)
            if shift_floor < shift_cap:
                shift_x = random.uniform(shift_floor, shift_cap)
                shift_y = random.uniform(-0.10, -0.03)
                _reframe_all(
                    scale=scale,
                    shift_x=shift_x,
                    shift_y=shift_y,
                )
                if random.random() < 0.45:
                    image = _apply_foreground_occluder(image)
        elif (
            self.training_profile == "legacy_r3"
            and 0.58 <= width_fraction <= 0.82
            and 0.34 <= center_x <= 0.66
            and random.random() < 0.22
        ):
            scale = random.uniform(0.72, 0.84)
            shift_left, shift_right = _allowed_shift_range(corners, scale)
            shift_floor = max(-0.05, shift_left)
            shift_cap = min(0.05, shift_right)
            if shift_floor < shift_cap:
                shift_x = random.uniform(shift_floor, shift_cap)
                shift_y = random.uniform(-0.06, 0.03)
                _reframe_all(
                    scale=scale,
                    shift_x=shift_x,
                    shift_y=shift_y,
                )
                if random.random() < 0.30:
                    image = _apply_foreground_occluder(image)
        elif width_fraction > 0.82 and random.random() < 0.55:
            scale = random.uniform(0.84, 0.94)
            shift_span = min(0.10, max((1.0 - scale) * 0.5 - 0.01, 0.02))
            shift_x = random.uniform(-shift_span, shift_span)
            shift_y = random.uniform(-0.04, 0.04)
            _reframe_all(
                scale=scale,
                shift_x=shift_x,
                shift_y=shift_y,
            )
        elif (
            self.training_profile != "legacy_r3"
            and 0.66 <= width_fraction <= 0.82
            and 0.38 <= center_x <= 0.62
            and random.random() < 0.22
        ):
            scale = random.uniform(0.92, 0.98)
            shift_left, shift_right = _allowed_shift_range(corners, scale)
            shift_cap = min(0.14, shift_right)
            shift_floor = max(-0.14, shift_left)
            if shift_floor < shift_cap:
                shift_x = random.uniform(shift_floor, shift_cap)
                shift_y = random.uniform(-0.02, 0.02)
                _reframe_all(
                    scale=scale,
                    shift_x=shift_x,
                    shift_y=shift_y,
                )
        scene_tags = set(row.get("scene_tags") or [])
        profile_low_contrast = is_geometry_priority_low_contrast_scene(row)
        if self.training_profile != "legacy_r3" and ("near_color_background" in scene_tags or "low_contrast_scene" in scene_tags or profile_low_contrast):
            gamma = random.uniform(0.92, 1.08)
            lut = np.array([pow(i / 255.0, gamma) * 255.0 for i in range(256)], dtype=np.uint8)
            image = cv2.LUT(image, lut)
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.round(corners * np.array([width - 1, height - 1], dtype=np.float32)).astype(np.int32)
            cv2.fillConvexPoly(mask, points, 255)
            ring = cv2.dilate(mask, np.ones((17, 17), dtype=np.uint8), iterations=1)
            ring = cv2.subtract(ring, mask)
            if np.any(mask > 0) and np.any(ring > 0) and random.random() < 0.65:
                image_f = image.astype(np.float32)
                inner_mean = image_f[mask > 0].mean(axis=0)
                ring_mean = image_f[ring > 0].mean(axis=0)
                if "near_color_background" in scene_tags or profile_low_contrast:
                    pull = random.uniform(0.16, 0.30)
                    image_f[mask > 0] = image_f[mask > 0] * (1.0 - pull) + ring_mean * pull
                    image_f[ring > 0] = image_f[ring > 0] * (1.0 - pull * 0.5) + inner_mean * (pull * 0.5)
                else:
                    pull = random.uniform(0.08, 0.16)
                    image_f[mask > 0] = image_f[mask > 0] * (1.0 - pull) + ring_mean * pull
                image = np.clip(image_f, 0, 255).astype(np.uint8)
            if random.random() < 0.4:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[..., 1] *= random.uniform(0.72, 0.95)
                image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
            if random.random() < 0.35:
                image = cv2.GaussianBlur(image, (5, 5), sigmaX=0.0)
        if self.training_profile != "legacy_r3" and "black_frame_scene" in scene_tags and random.random() < 0.45:
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.round(corners * np.array([width - 1, height - 1], dtype=np.float32)).astype(np.int32)
            cv2.fillConvexPoly(mask, points, 255)
            border = cv2.subtract(mask, cv2.erode(mask, np.ones((13, 13), dtype=np.uint8), iterations=1))
            image_f = image.astype(np.float32)
            image_f[border > 0] *= random.uniform(0.7, 0.88)
            image = np.clip(image_f, 0, 255).astype(np.uint8)
        if self.training_profile == "legacy_r3" and is_legacy_dark_muted_scene(image) and random.random() < 0.35:
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.round(corners * np.array([width - 1, height - 1], dtype=np.float32)).astype(np.int32)
            cv2.fillConvexPoly(mask, points, 255)
            ring = cv2.dilate(mask, np.ones((15, 15), dtype=np.uint8), iterations=1)
            ring = cv2.subtract(ring, mask)
            image_f = image.astype(np.float32)
            if np.any(mask > 0) and np.any(ring > 0):
                ring_mean = image_f[ring > 0].mean(axis=0)
                image_f[mask > 0] = image_f[mask > 0] * 0.82 + ring_mean * 0.18
            image = np.clip(image_f, 0, 255).astype(np.uint8)
            if random.random() < 0.5:
                image = cv2.GaussianBlur(image, (5, 5), sigmaX=0.0)
        alpha = 1.0 + random.uniform(-0.15, 0.15)
        beta = random.uniform(-18.0, 18.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return _return(image, corners)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        corners = _normalize_manual_quad(row["manual_quad"], width, height)
        teacher_candidates = _resolve_teacher_candidate_quads(
            row,
            width,
            height,
            sources=self.teacher_candidate_sources,
            opencv_score_min=self.teacher_opencv_score_min,
        )
        if teacher_candidates:
            image, corners, extra_quads = self._augment(image, corners, row, extra_quads=teacher_candidates)
            teacher_candidates = extra_quads
        else:
            image, corners = self._augment(image, corners, row)
        if self.teacher_target_mode == "oracle":
            candidate_array = np.stack(teacher_candidates, axis=0) if teacher_candidates else None
            teacher_target, teacher_guidance_scale, _ = build_multi_teacher_target(
                corners,
                candidate_array,
                blend_ratio=self.teacher_blend_ratio,
                corner_error_max=self.teacher_corner_error_max,
                sample_error_max=self.teacher_sample_error_max,
            )
        elif self.teacher_target_mode == "adaptive":
            teacher_quad = teacher_candidates[0] if teacher_candidates else None
            teacher_target, teacher_guidance_scale = build_adaptive_teacher_target(
                corners,
                teacher_quad,
                blend_ratio=self.teacher_blend_ratio,
                corner_error_max=self.teacher_corner_error_max,
                sample_error_max=self.teacher_sample_error_max,
            )
        else:
            raise ValueError(f"unsupported teacher_target_mode: {self.teacher_target_mode}")
        teacher_guidance_scale *= self._teacher_activation_scale(row, width, height)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        heatmaps = build_corner_heatmaps(corners, output_size=self.output_size)
        return {
            "image": torch.from_numpy(build_global_feature_tensor(image, feature_mode=self.feature_mode)),
            "heatmaps": torch.from_numpy(heatmaps),
            "corners": torch.from_numpy(corners.astype(np.float32)),
            "teacher_target": torch.from_numpy(teacher_target.astype(np.float32)),
            "teacher_guidance_scale": torch.tensor(float(teacher_guidance_scale), dtype=torch.float32),
            "geometry_scale": torch.tensor(self._geometry_scale(row), dtype=torch.float32),
            "edge_scale": torch.tensor(self._edge_scale(corners), dtype=torch.float32),
        }


def compute_corner_weighted_coord_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    corner_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_corner = torch.nn.functional.smooth_l1_loss(predicted, target, reduction="none").mean(dim=-1)
    if corner_weights is None:
        return per_corner.mean()
    weights = corner_weights.to(device=predicted.device, dtype=predicted.dtype).view(1, -1)
    return torch.sum(per_corner * weights) / torch.clamp(weights.sum() * predicted.shape[0], min=1e-6)


def compute_corner_weighted_coord_loss_with_sample_weights(
    predicted: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
    corner_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_corner = torch.nn.functional.smooth_l1_loss(predicted, target, reduction="none").mean(dim=-1)
    sample_weights = sample_weights.to(device=predicted.device, dtype=predicted.dtype).view(-1, 1)
    if corner_weights is None:
        return torch.sum(per_corner * sample_weights) / torch.clamp(sample_weights.sum() * predicted.shape[1], min=1e-6)
    weights = corner_weights.to(device=predicted.device, dtype=predicted.dtype).view(1, -1)
    combined = sample_weights * weights
    return torch.sum(per_corner * combined) / torch.clamp(combined.sum(), min=1e-6)


def export_global_corner_split(
    dataset_root: Path,
    output_dir: Path,
    seed: int = 7,
    test_ratio: float = 0.25,
    focus_projects: list[str] | None = None,
    focus_test_ratio: float = 0.25,
    holdout_projects: list[str] | None = None,
) -> dict[str, Any]:
    pages = load_manual_pages(dataset_root)
    if focus_projects:
        split = build_project_aware_split(
            pages,
            focus_projects=focus_projects,
            holdout_projects=holdout_projects or [],
            test_ratio=test_ratio,
            focus_test_ratio=focus_test_ratio,
            seed=seed,
        )
    else:
        if holdout_projects:
            split = build_project_aware_split(
                pages,
                focus_projects=[],
                holdout_projects=holdout_projects,
                test_ratio=test_ratio,
                focus_test_ratio=focus_test_ratio,
                seed=seed,
            )
        else:
            split = build_split(pages, test_ratio=test_ratio, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_root": str(dataset_root),
        "pages": len(pages),
        "train_pages": len(split["train"]),
        "test_pages": len(split["test"]),
        "seed": seed,
        "test_ratio": test_ratio,
    }
    if "metadata" in split:
        summary.update(split["metadata"])
    split_names = ["train", "test"]
    if "focus_train" in split:
        split_names.extend(["focus_train", "focus_test"])
    if "holdout" in split:
        split_names.append("holdout")
    for split_name in split_names:
        rows = [
            {
                "page_id": item["page_id"],
                "project_name": item["project_name"],
                "image_path": item["image_path"],
                "manual_quad": item["manual_quad"],
                "scene_profile": item.get("scene_profile", {}),
                "scene_tags": item.get("scene_tags", []),
            }
            for item in split[split_name]
        ]
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else "")
        (output_dir / f"{split_name}.jsonl").write_text(text, encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def build_effective_split_dir(
    split_dir: Path,
    *,
    merge_focus_train: bool = False,
    focus_train_repeat: int = 1,
    merge_focus_test: bool = False,
) -> Path:
    if not merge_focus_train and not merge_focus_test:
        return split_dir

    focus_train_repeat = max(int(focus_train_repeat), 1)
    train_rows = _read_jsonl_rows(split_dir / "train.jsonl")
    test_rows = _read_jsonl_rows(split_dir / "test.jsonl")
    focus_train_rows = _read_jsonl_rows(split_dir / "focus_train.jsonl")
    focus_test_rows = _read_jsonl_rows(split_dir / "focus_test.jsonl")

    merged_train = list(train_rows)
    if merge_focus_train and focus_train_rows:
        for _ in range(focus_train_repeat):
            merged_train.extend(focus_train_rows)

    merged_test = list(test_rows)
    if merge_focus_test and focus_test_rows:
        merged_test.extend(focus_test_rows)

    effective_dir = Path(tempfile.mkdtemp(prefix="global-corner-split-", dir=str(split_dir.parent)))
    for name in ("focus_train.jsonl", "focus_test.jsonl", "holdout.jsonl", "summary.json"):
        source = split_dir / name
        if source.exists():
            shutil.copy2(source, effective_dir / name)
    _write_jsonl_rows(effective_dir / "train.jsonl", merged_train)
    _write_jsonl_rows(effective_dir / "test.jsonl", merged_test)
    return effective_dir


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metric_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = batch["heatmaps"].to(device=device, dtype=torch.float32)
            corners = batch["corners"].to(device=device, dtype=torch.float32)
            logits = model(images)
            loss = torch.mean((logits - targets) ** 2)
            pred = decode_heatmaps(logits)
            losses.extend(loss.detach().cpu().repeat(images.shape[0]).tolist())
            pred_np = pred.detach().cpu().numpy()
            corners_np = corners.detach().cpu().numpy()
            for pred_row, target_row in zip(pred_np, corners_np, strict=True):
                metric_rows.append(quad_geometry_metrics(target_row, pred_row))
    summary = summarize_geometry_metric_rows(metric_rows)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "point_error_mean": float(summary["point_error_mean"]),
        "point_le_0_05": float(summary["point_le_0_05_ratio"]),
        "point_le_0_03": float(summary["point_le_0_03_ratio"]),
        "point_le_0_02": float(summary["point_le_0_02_ratio"]),
        "point_le_0_01": float(summary["point_le_0_01_ratio"]),
        "screen_relative_error_mean": float(summary["screen_relative_error_mean"]),
        "max_corner_error_mean": float(summary["max_corner_error_mean"]),
        "perspective_tilt_error_mean": float(summary["perspective_tilt_error_mean"]),
        "quad_inset_ratio_mean": float(summary["quad_inset_ratio_mean"]),
    }


@dataclass
class GlobalTrainResult:
    split_dir: str
    model_path: str
    history_path: str
    best_epoch: int
    best_val_loss: float
    best_val_point_error: float
    best_val_point_le_0_05: float
    best_val_point_le_0_03: float
    best_val_point_le_0_02: float
    best_val_point_le_0_01: float


def _save_checkpoint(
    checkpoint_path: Path,
    state_dict: dict[str, torch.Tensor],
    *,
    input_size: int,
    output_size: int,
    channels: int,
    device: torch.device,
    corner_weights: tuple[float, float, float, float],
    sample_weight_power: float,
    hardness_sample_weight_power: float,
    disagreement_sample_weight_floor: float | None,
    disagreement_sample_weight_boost: float,
    project_balance_power: float,
    geometry_priority_sample_weight_boost: float,
    failure_layer_runtime_boost: float,
    failure_layer_opencv_boost: float,
    failure_layer_hard_boost: float,
    failure_layer_gain_power: float,
    failure_layer_project_balance: bool,
    geometry_loss_weight: float,
    edge_supervision_weight: float,
    inset_weight: float,
    inward_boundary_weight: float,
    inward_boundary_margin: float,
    quad_mask_weight: float,
    inner_boundary_band_weight: float,
    teacher_guidance_weight: float,
    teacher_blend_ratio: float,
    teacher_corner_error_max: float | None,
    teacher_sample_error_max: float | None,
    teacher_target_mode: str,
    teacher_candidate_sources: tuple[str, ...],
    teacher_opencv_score_min: float | None,
    teacher_activation_min_disagreement: float | None,
    max_corner_weight: float,
    edge_weight: float,
    edge_line_weight: float,
    edge_length_weight: float,
    edge_collapse_weight: float,
    corner_line_weight: float,
    corner_angle_weight: float,
    training_profile: str,
    input_channels: int,
    feature_mode: str,
) -> None:
    torch.save(
        {
            "state_dict": state_dict,
            "input_size": input_size,
            "output_size": output_size,
            "channels": channels,
            "input_channels": int(input_channels),
            "feature_mode": str(feature_mode),
            "decode_mode": "soft_argmax",
            "device": device.type,
            "corner_weights": [float(value) for value in corner_weights],
            "sample_weight_power": float(sample_weight_power),
            "hardness_sample_weight_power": float(hardness_sample_weight_power),
            "disagreement_sample_weight_floor": (
                None
                if disagreement_sample_weight_floor is None
                else float(disagreement_sample_weight_floor)
            ),
            "disagreement_sample_weight_boost": float(disagreement_sample_weight_boost),
            "project_balance_power": float(project_balance_power),
            "geometry_priority_sample_weight_boost": float(geometry_priority_sample_weight_boost),
            "failure_layer_runtime_boost": float(failure_layer_runtime_boost),
            "failure_layer_opencv_boost": float(failure_layer_opencv_boost),
            "failure_layer_hard_boost": float(failure_layer_hard_boost),
            "failure_layer_gain_power": float(failure_layer_gain_power),
            "failure_layer_project_balance": bool(failure_layer_project_balance),
            "geometry_loss_weight": float(geometry_loss_weight),
            "edge_supervision_weight": float(edge_supervision_weight),
            "inset_weight": float(inset_weight),
            "inward_boundary_weight": float(inward_boundary_weight),
            "inward_boundary_margin": float(inward_boundary_margin),
            "quad_mask_weight": float(quad_mask_weight),
            "inner_boundary_band_weight": float(inner_boundary_band_weight),
            "teacher_guidance_weight": float(teacher_guidance_weight),
            "teacher_blend_ratio": float(teacher_blend_ratio),
            "teacher_corner_error_max": None if teacher_corner_error_max is None else float(teacher_corner_error_max),
            "teacher_sample_error_max": None if teacher_sample_error_max is None else float(teacher_sample_error_max),
            "teacher_target_mode": str(teacher_target_mode),
            "teacher_candidate_sources": [str(item) for item in teacher_candidate_sources],
            "teacher_opencv_score_min": None if teacher_opencv_score_min is None else float(teacher_opencv_score_min),
            "teacher_activation_min_disagreement": (
                None
                if teacher_activation_min_disagreement is None
                else float(teacher_activation_min_disagreement)
            ),
            "max_corner_weight": float(max_corner_weight),
            "edge_weight": float(edge_weight),
            "edge_line_weight": float(edge_line_weight),
            "edge_length_weight": float(edge_length_weight),
            "edge_collapse_weight": float(edge_collapse_weight),
            "corner_line_weight": float(corner_line_weight),
            "corner_angle_weight": float(corner_angle_weight),
            "training_profile": training_profile,
        },
        checkpoint_path,
    )


def train_global_corner_model(
    dataset_root: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    input_size: int = 256,
    output_size: int = 64,
    channels: int = 24,
    seed: int = 7,
    init_model_path: Path | None = None,
    geometry_loss_weight: float | None = None,
    edge_supervision_weight: float | None = None,
    split_dir: Path | None = None,
    sample_weight_power: float = 0.0,
    hardness_sample_weight_power: float = 0.0,
    disagreement_sample_weight_floor: float | None = None,
    disagreement_sample_weight_boost: float = 1.0,
    project_balance_power: float = 0.0,
    geometry_priority_sample_weight_boost: float = 1.0,
    failure_layer_runtime_boost: float = 1.0,
    failure_layer_opencv_boost: float = 1.0,
    failure_layer_hard_boost: float = 1.0,
    failure_layer_gain_power: float = 0.0,
    failure_layer_project_balance: bool = False,
    corner_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    inset_weight: float = 0.25,
    inward_boundary_weight: float = 0.0,
    inward_boundary_margin: float = 0.0,
    quad_mask_weight: float = 0.0,
    inner_boundary_band_weight: float = 0.0,
    teacher_guidance_weight: float = 0.0,
    teacher_blend_ratio: float = 0.0,
    teacher_corner_error_max: float | None = None,
    teacher_sample_error_max: float | None = None,
    teacher_target_mode: str = "adaptive",
    teacher_candidate_sources: tuple[str, ...] = ("teacher",),
    teacher_opencv_score_min: float | None = None,
    teacher_activation_min_disagreement: float | None = None,
    max_corner_weight: float = 1.15,
    edge_weight: float = 0.35,
    edge_line_weight: float = 0.9,
    edge_length_weight: float = 0.4,
    edge_collapse_weight: float = 0.6,
    corner_line_weight: float = 0.7,
    corner_angle_weight: float = 0.28,
    training_profile: str = "default",
    save_epoch_checkpoints: bool = False,
    merge_focus_train: bool = False,
    focus_train_repeat: int = 1,
    merge_focus_test: bool = False,
    feature_mode: str = "rgb",
) -> GlobalTrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_split_dir = split_dir or (output_dir / "split")
    if split_dir is None:
        export_global_corner_split(dataset_root, active_split_dir, seed=seed, test_ratio=0.25)
    active_split_dir = build_effective_split_dir(
        active_split_dir,
        merge_focus_train=merge_focus_train,
        focus_train_repeat=focus_train_repeat,
        merge_focus_test=merge_focus_test,
    )
    geometry_loss_weight, edge_supervision_weight = _resolve_training_profile_weights(
        training_profile,
        geometry_loss_weight,
        edge_supervision_weight,
    )

    train_dataset = GlobalCornerDataset(
        active_split_dir / "train.jsonl",
        input_size=input_size,
        output_size=output_size,
        augment=True,
        training_profile=training_profile,
        feature_mode=feature_mode,
        teacher_blend_ratio=teacher_blend_ratio,
        teacher_corner_error_max=teacher_corner_error_max,
        teacher_sample_error_max=teacher_sample_error_max,
        teacher_target_mode=teacher_target_mode,
        teacher_candidate_sources=teacher_candidate_sources,
        teacher_opencv_score_min=teacher_opencv_score_min,
        teacher_activation_min_disagreement=teacher_activation_min_disagreement,
    )
    test_dataset = GlobalCornerDataset(
        active_split_dir / "test.jsonl",
        input_size=input_size,
        output_size=output_size,
        augment=False,
        training_profile=training_profile,
        feature_mode=feature_mode,
        teacher_blend_ratio=teacher_blend_ratio,
        teacher_corner_error_max=teacher_corner_error_max,
        teacher_sample_error_max=teacher_sample_error_max,
        teacher_target_mode=teacher_target_mode,
        teacher_candidate_sources=teacher_candidate_sources,
        teacher_opencv_score_min=teacher_opencv_score_min,
        teacher_activation_min_disagreement=teacher_activation_min_disagreement,
    )
    train_loader_kwargs: dict[str, Any] = {"batch_size": batch_size, "num_workers": 0}
    use_weighted_sampler = (
        sample_weight_power > 0.0
        or hardness_sample_weight_power > 0.0
        or (
            disagreement_sample_weight_floor is not None
            and disagreement_sample_weight_floor > 0.0
            and disagreement_sample_weight_boost > 1.0
        )
        or project_balance_power > 0.0
        or geometry_priority_sample_weight_boost > 1.0
        or failure_layer_runtime_boost > 1.0
        or failure_layer_opencv_boost > 1.0
        or failure_layer_hard_boost > 1.0
        or failure_layer_gain_power > 0.0
        or failure_layer_project_balance
    )
    if use_weighted_sampler:
        weights = torch.from_numpy(
            train_dataset.build_sample_weights(
                power=sample_weight_power,
                hardness_score_power=hardness_sample_weight_power,
                disagreement_floor=disagreement_sample_weight_floor,
                disagreement_boost=disagreement_sample_weight_boost,
                project_balance_power=project_balance_power,
                geometry_priority_boost=geometry_priority_sample_weight_boost,
                failure_layer_runtime_boost=failure_layer_runtime_boost,
                failure_layer_opencv_boost=failure_layer_opencv_boost,
                failure_layer_hard_boost=failure_layer_hard_boost,
                failure_layer_gain_power=failure_layer_gain_power,
                failure_layer_project_balance=failure_layer_project_balance,
            )
        ).double()
        train_loader_kwargs["sampler"] = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    else:
        train_loader_kwargs["shuffle"] = True
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = select_torch_device()
    input_channels = global_feature_channels(feature_mode)
    model = CornerHeatmapNet(in_channels=input_channels, channels=channels, output_channels=4).to(device)
    if init_model_path is not None:
        checkpoint = torch.load(init_model_path, map_location="cpu")
        initialize_global_model_from_checkpoint(model, checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    corner_weight_tensor = torch.tensor(corner_weights, dtype=torch.float32, device=device)
    best_score = (math.inf, math.inf, math.inf, math.inf)
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    checkpoint_dir = output_dir / "checkpoints"
    if save_epoch_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = batch["heatmaps"].to(device=device, dtype=torch.float32)
            corners = batch["corners"].to(device=device, dtype=torch.float32)
            teacher_target = batch["teacher_target"].to(device=device, dtype=torch.float32)
            teacher_guidance_scales = batch["teacher_guidance_scale"].to(device=device, dtype=torch.float32)
            geometry_scales = batch["geometry_scale"].to(device=device, dtype=torch.float32)
            edge_scales = batch["edge_scale"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            pred = decode_heatmaps(logits)
            heatmap_loss = torch.mean((logits - targets) ** 2)
            coord_loss = compute_corner_weighted_coord_loss(pred, corners, corner_weight_tensor)
            geometry_loss, _ = compute_global_geometry_loss(
                pred,
                corners,
                max_corner_weight=max_corner_weight,
                edge_weight=edge_weight,
                edge_line_weight=edge_line_weight,
                edge_length_weight=edge_length_weight,
                edge_collapse_weight=edge_collapse_weight,
                corner_line_weight=corner_line_weight,
                corner_angle_weight=corner_angle_weight,
                inset_weight=inset_weight,
                inward_boundary_weight=inward_boundary_weight,
                inward_boundary_margin=inward_boundary_margin,
                quad_mask_weight=quad_mask_weight,
                inner_boundary_band_weight=inner_boundary_band_weight,
                sample_scales=geometry_scales,
            )
            edge_loss = compute_edge_supervision_loss(pred, corners, output_size=output_size, sample_scales=edge_scales)
            loss = heatmap_loss + coord_loss * 2.5 + geometry_loss * geometry_loss_weight + edge_loss * edge_supervision_weight
            if float(teacher_guidance_weight) > 0.0 and torch.any(teacher_guidance_scales > 0):
                teacher_geometry_scales = geometry_scales * teacher_guidance_scales
                teacher_edge_scales = edge_scales * teacher_guidance_scales
                teacher_coord_loss = compute_corner_weighted_coord_loss_with_sample_weights(
                    pred,
                    teacher_target,
                    teacher_guidance_scales,
                    corner_weight_tensor,
                )
                teacher_geometry_loss, _ = compute_global_geometry_loss(
                    pred,
                    teacher_target,
                    max_corner_weight=max_corner_weight,
                    edge_weight=edge_weight,
                    edge_line_weight=edge_line_weight,
                    edge_length_weight=edge_length_weight,
                    edge_collapse_weight=edge_collapse_weight,
                    corner_line_weight=corner_line_weight,
                    corner_angle_weight=corner_angle_weight,
                    inset_weight=inset_weight,
                    inward_boundary_weight=inward_boundary_weight,
                    inward_boundary_margin=inward_boundary_margin,
                    quad_mask_weight=quad_mask_weight,
                    inner_boundary_band_weight=inner_boundary_band_weight,
                    sample_scales=teacher_geometry_scales,
                )
                teacher_edge_loss = compute_edge_supervision_loss(
                    pred,
                    teacher_target,
                    output_size=output_size,
                    sample_scales=teacher_edge_scales,
                )
                loss = loss + float(teacher_guidance_weight) * (
                    teacher_coord_loss * 2.5
                    + teacher_geometry_loss * geometry_loss_weight
                    + teacher_edge_loss * edge_supervision_weight
                )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val = _evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)) if train_losses else 0.0, **val}
        history.append(row)
        epoch_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if save_epoch_checkpoints:
            _save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                epoch_state,
                input_size=input_size,
                output_size=output_size,
                channels=channels,
                device=device,
                corner_weights=corner_weights,
                sample_weight_power=sample_weight_power,
                hardness_sample_weight_power=hardness_sample_weight_power,
                disagreement_sample_weight_floor=disagreement_sample_weight_floor,
                disagreement_sample_weight_boost=disagreement_sample_weight_boost,
                project_balance_power=project_balance_power,
                geometry_priority_sample_weight_boost=geometry_priority_sample_weight_boost,
                failure_layer_runtime_boost=failure_layer_runtime_boost,
                failure_layer_opencv_boost=failure_layer_opencv_boost,
                failure_layer_hard_boost=failure_layer_hard_boost,
                failure_layer_gain_power=failure_layer_gain_power,
                failure_layer_project_balance=failure_layer_project_balance,
                geometry_loss_weight=geometry_loss_weight,
                edge_supervision_weight=edge_supervision_weight,
                inset_weight=inset_weight,
                inward_boundary_weight=inward_boundary_weight,
                inward_boundary_margin=inward_boundary_margin,
                quad_mask_weight=quad_mask_weight,
                inner_boundary_band_weight=inner_boundary_band_weight,
                teacher_guidance_weight=teacher_guidance_weight,
                teacher_blend_ratio=teacher_blend_ratio,
                teacher_corner_error_max=teacher_corner_error_max,
                teacher_sample_error_max=teacher_sample_error_max,
                teacher_target_mode=teacher_target_mode,
                teacher_candidate_sources=teacher_candidate_sources,
                teacher_opencv_score_min=teacher_opencv_score_min,
                teacher_activation_min_disagreement=teacher_activation_min_disagreement,
                max_corner_weight=max_corner_weight,
                edge_weight=edge_weight,
                edge_line_weight=edge_line_weight,
                edge_length_weight=edge_length_weight,
                edge_collapse_weight=edge_collapse_weight,
                corner_line_weight=corner_line_weight,
                corner_angle_weight=corner_angle_weight,
                training_profile=training_profile,
                input_channels=input_channels,
                feature_mode=feature_mode,
            )
        val_score = (
            float(val["screen_relative_error_mean"]),
            float(val["max_corner_error_mean"]),
            float(val["perspective_tilt_error_mean"]),
            float(val["point_error_mean"]),
        )
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            best_metrics = dict(val)
            best_state = epoch_state

    if best_state is None or best_metrics is None:
        raise RuntimeError("no checkpoint saved")

    model_path = output_dir / "global_corner_model.pt"
    _save_checkpoint(
        model_path,
        best_state,
        input_size=input_size,
        output_size=output_size,
        channels=channels,
        device=device,
        corner_weights=corner_weights,
        sample_weight_power=sample_weight_power,
        hardness_sample_weight_power=hardness_sample_weight_power,
        disagreement_sample_weight_floor=disagreement_sample_weight_floor,
        disagreement_sample_weight_boost=disagreement_sample_weight_boost,
        project_balance_power=project_balance_power,
        geometry_priority_sample_weight_boost=geometry_priority_sample_weight_boost,
        failure_layer_runtime_boost=failure_layer_runtime_boost,
        failure_layer_opencv_boost=failure_layer_opencv_boost,
        failure_layer_hard_boost=failure_layer_hard_boost,
        failure_layer_gain_power=failure_layer_gain_power,
        failure_layer_project_balance=failure_layer_project_balance,
        geometry_loss_weight=geometry_loss_weight,
        edge_supervision_weight=edge_supervision_weight,
        inset_weight=inset_weight,
        inward_boundary_weight=inward_boundary_weight,
        inward_boundary_margin=inward_boundary_margin,
        quad_mask_weight=quad_mask_weight,
        inner_boundary_band_weight=inner_boundary_band_weight,
        teacher_guidance_weight=teacher_guidance_weight,
        teacher_blend_ratio=teacher_blend_ratio,
        teacher_corner_error_max=teacher_corner_error_max,
        teacher_sample_error_max=teacher_sample_error_max,
        teacher_target_mode=teacher_target_mode,
        teacher_candidate_sources=teacher_candidate_sources,
        teacher_opencv_score_min=teacher_opencv_score_min,
        teacher_activation_min_disagreement=teacher_activation_min_disagreement,
        max_corner_weight=max_corner_weight,
        edge_weight=edge_weight,
        edge_line_weight=edge_line_weight,
        edge_length_weight=edge_length_weight,
        edge_collapse_weight=edge_collapse_weight,
        corner_line_weight=corner_line_weight,
        corner_angle_weight=corner_angle_weight,
        training_profile=training_profile,
        input_channels=input_channels,
        feature_mode=feature_mode,
    )
    history_path = output_dir / "global_corner_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return GlobalTrainResult(
        split_dir=str(active_split_dir),
        model_path=str(model_path),
        history_path=str(history_path),
        best_epoch=best_epoch,
        best_val_loss=float(best_metrics["loss"]),
        best_val_point_error=float(best_metrics["point_error_mean"]),
        best_val_point_le_0_05=float(best_metrics["point_le_0_05"]),
        best_val_point_le_0_03=float(best_metrics["point_le_0_03"]),
        best_val_point_le_0_02=float(best_metrics["point_le_0_02"]),
        best_val_point_le_0_01=float(best_metrics["point_le_0_01"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train model-dominant global corner detector")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--output-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-model")
    parser.add_argument("--geometry-loss-weight", type=float)
    parser.add_argument("--edge-supervision-weight", type=float)
    parser.add_argument("--split-dir")
    parser.add_argument("--sample-weight-power", type=float, default=0.0)
    parser.add_argument("--hardness-sample-weight-power", type=float, default=0.0)
    parser.add_argument("--disagreement-sample-weight-floor", type=float)
    parser.add_argument("--disagreement-sample-weight-boost", type=float, default=1.0)
    parser.add_argument("--project-balance-power", type=float, default=0.0)
    parser.add_argument("--geometry-priority-sample-weight-boost", type=float, default=1.0)
    parser.add_argument("--failure-layer-runtime-boost", type=float, default=1.0)
    parser.add_argument("--failure-layer-opencv-boost", type=float, default=1.0)
    parser.add_argument("--failure-layer-hard-boost", type=float, default=1.0)
    parser.add_argument("--failure-layer-gain-power", type=float, default=0.0)
    parser.add_argument("--failure-layer-project-balance", action="store_true")
    parser.add_argument("--inset-weight", type=float, default=0.25)
    parser.add_argument("--inward-boundary-weight", type=float, default=0.0)
    parser.add_argument("--inward-boundary-margin", type=float, default=0.0)
    parser.add_argument("--quad-mask-weight", type=float, default=0.0)
    parser.add_argument("--inner-boundary-band-weight", type=float, default=0.0)
    parser.add_argument("--teacher-guidance-weight", type=float, default=0.0)
    parser.add_argument("--teacher-blend-ratio", type=float, default=0.0)
    parser.add_argument("--teacher-corner-error-max", type=float)
    parser.add_argument("--teacher-sample-error-max", type=float)
    parser.add_argument("--teacher-target-mode", default="adaptive", choices=["adaptive", "oracle"])
    parser.add_argument(
        "--teacher-candidate-sources",
        default="teacher",
        help="Comma-separated candidate sources for teacher target: teacher,r3,v28,roi,opencv",
    )
    parser.add_argument("--teacher-opencv-score-min", type=float)
    parser.add_argument("--teacher-activation-min-disagreement", type=float)
    parser.add_argument("--max-corner-weight", type=float, default=1.15)
    parser.add_argument("--edge-weight", type=float, default=0.35)
    parser.add_argument("--edge-line-weight", type=float, default=0.9)
    parser.add_argument("--edge-length-weight", type=float, default=0.4)
    parser.add_argument("--edge-collapse-weight", type=float, default=0.6)
    parser.add_argument("--corner-line-weight", type=float, default=0.7)
    parser.add_argument("--corner-angle-weight", type=float, default=0.28)
    parser.add_argument("--training-profile", default="default", choices=["default", "legacy_r3"])
    parser.add_argument("--feature-mode", default="rgb", choices=sorted(GLOBAL_FEATURE_MODE_CHANNELS.keys()))
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--merge-focus-train", action="store_true")
    parser.add_argument("--focus-train-repeat", type=int, default=1)
    parser.add_argument("--merge-focus-test", action="store_true")
    parser.add_argument(
        "--corner-weights",
        default="1,1,1,1",
        help="Comma-separated per-corner weights in TL,TR,BR,BL order.",
    )
    args = parser.parse_args()
    corner_weights = tuple(float(part.strip()) for part in str(args.corner_weights).split(","))
    if len(corner_weights) != 4:
        raise ValueError("--corner-weights requires four comma-separated values")
    teacher_candidate_sources = tuple(
        part.strip() for part in str(args.teacher_candidate_sources).split(",") if part.strip()
    ) or ("teacher",)

    result = train_global_corner_model(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        output_size=args.output_size,
        channels=args.channels,
        seed=args.seed,
        init_model_path=Path(args.init_model) if args.init_model else None,
        geometry_loss_weight=float(args.geometry_loss_weight) if args.geometry_loss_weight is not None else None,
        edge_supervision_weight=float(args.edge_supervision_weight) if args.edge_supervision_weight is not None else None,
        split_dir=Path(args.split_dir) if args.split_dir else None,
        sample_weight_power=float(args.sample_weight_power),
        hardness_sample_weight_power=float(args.hardness_sample_weight_power),
        disagreement_sample_weight_floor=(
            float(args.disagreement_sample_weight_floor)
            if args.disagreement_sample_weight_floor is not None
            else None
        ),
        disagreement_sample_weight_boost=float(args.disagreement_sample_weight_boost),
        project_balance_power=float(args.project_balance_power),
        geometry_priority_sample_weight_boost=float(args.geometry_priority_sample_weight_boost),
        failure_layer_runtime_boost=float(args.failure_layer_runtime_boost),
        failure_layer_opencv_boost=float(args.failure_layer_opencv_boost),
        failure_layer_hard_boost=float(args.failure_layer_hard_boost),
        failure_layer_gain_power=float(args.failure_layer_gain_power),
        failure_layer_project_balance=bool(args.failure_layer_project_balance),
        corner_weights=corner_weights,
        inset_weight=float(args.inset_weight),
        inward_boundary_weight=float(args.inward_boundary_weight),
        inward_boundary_margin=float(args.inward_boundary_margin),
        quad_mask_weight=float(args.quad_mask_weight),
        inner_boundary_band_weight=float(args.inner_boundary_band_weight),
        teacher_guidance_weight=float(args.teacher_guidance_weight),
        teacher_blend_ratio=float(args.teacher_blend_ratio),
        teacher_corner_error_max=float(args.teacher_corner_error_max) if args.teacher_corner_error_max is not None else None,
        teacher_sample_error_max=float(args.teacher_sample_error_max) if args.teacher_sample_error_max is not None else None,
        teacher_target_mode=str(args.teacher_target_mode),
        teacher_candidate_sources=teacher_candidate_sources,
        teacher_opencv_score_min=float(args.teacher_opencv_score_min) if args.teacher_opencv_score_min is not None else None,
        teacher_activation_min_disagreement=(
            float(args.teacher_activation_min_disagreement)
            if args.teacher_activation_min_disagreement is not None
            else None
        ),
        max_corner_weight=float(args.max_corner_weight),
        edge_weight=float(args.edge_weight),
        edge_line_weight=float(args.edge_line_weight),
        edge_length_weight=float(args.edge_length_weight),
        edge_collapse_weight=float(args.edge_collapse_weight),
        corner_line_weight=float(args.corner_line_weight),
        corner_angle_weight=float(args.corner_angle_weight),
        training_profile=str(args.training_profile),
        save_epoch_checkpoints=bool(args.save_epoch_checkpoints),
        merge_focus_train=bool(args.merge_focus_train),
        focus_train_repeat=int(args.focus_train_repeat),
        merge_focus_test=bool(args.merge_focus_test),
        feature_mode=str(args.feature_mode),
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
