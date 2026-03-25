from __future__ import annotations

import argparse
import json
import math
import random
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
    quad_geometry_metrics,
    summarize_geometry_metric_rows,
)


def denormalize_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    restored = np.array(corners, dtype=np.float32).copy()
    restored[:, 0] *= float(width)
    restored[:, 1] *= float(height)
    return restored


def _normalize_manual_quad(quad: list[list[float]] | np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.array(quad, dtype=np.float32)
    arr[:, 0] /= max(float(width), 1.0)
    arr[:, 1] /= max(float(height), 1.0)
    return arr


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
    transformed_corners = corners.astype(np.float32).copy()
    transformed_corners[:, 0] = (transformed_corners[:, 0] - 0.5) * float(scale) + 0.5 + float(shift_x)
    transformed_corners[:, 1] = (transformed_corners[:, 1] - 0.5) * float(scale) + 0.5 + float(shift_y)
    transformed_corners = np.clip(transformed_corners, 0.0, 1.0)
    return reframed, transformed_corners


def _allowed_shift_range(corners: np.ndarray, scale: float) -> tuple[float, float]:
    scaled = corners.astype(np.float32).copy()
    scaled[:, 0] = (scaled[:, 0] - 0.5) * float(scale) + 0.5
    min_x = float(np.min(scaled[:, 0]))
    max_x = float(np.max(scaled[:, 0]))
    return -min_x, 1.0 - max_x


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
    corner_line_weight: float = 0.7,
    corner_angle_weight: float = 0.28,
    inset_weight: float = 0.25,
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
    edge_line_offset = _smooth_abs_torch(torch.sum((pred_mid - target_mid) * target_normal, dim=-1)).mean(dim=1)

    pred_lengths = torch.sqrt(torch.sum((pred_next - predicted) * (pred_next - predicted), dim=-1) + 1e-12)
    target_lengths = torch.sqrt(torch.sum((target_next - target) * (target_next - target), dim=-1) + 1e-12)
    edge_length_ratio = _smooth_abs_torch(pred_lengths - target_lengths) / torch.clamp(target_lengths, min=1e-6)
    edge_length_ratio = edge_length_ratio.mean(dim=1)

    target_prev = torch.cat([target[:, -1:, :], target[:, :-1, :]], dim=1)
    corner_prev_line = _point_line_distance_torch(predicted, target_prev, target)
    corner_next_line = _point_line_distance_torch(predicted, target, target_next)
    corner_line = 0.5 * (corner_prev_line + corner_next_line)
    corner_line = corner_line.mean(dim=1)

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

    total_per_sample = (
        max_corner * max_corner_weight
        + edge_direction * edge_weight
        + edge_line_offset * edge_line_weight
        + edge_length_ratio * edge_length_weight
        + corner_line * corner_line_weight
        + corner_angle * corner_angle_weight
        + inset * inset_weight
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
        "corner_line": corner_line.mean().detach(),
        "corner_angle": corner_angle.mean().detach(),
        "inset": inset.mean().detach(),
    }


class GlobalCornerDataset(Dataset):
    def __init__(self, manifest_path: Path, input_size: int = 256, output_size: int = 64, augment: bool = False) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        self.rows = [json.loads(line) for line in lines if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def build_sample_weights(self, power: float = 1.0) -> np.ndarray:
        weights = []
        for row in self.rows:
            adaptive = float(max(row.get("adaptive_weight", 1.0), 0.1))
            weights.append(adaptive**power)
        if not weights:
            return np.ones((0,), dtype=np.float32)
        arr = np.array(weights, dtype=np.float32)
        return arr / max(float(arr.mean()), 1e-6)

    def _geometry_scale(self, row: dict[str, Any]) -> float:
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
        return float(scale)

    def _edge_scale(self, corners: np.ndarray) -> float:
        width_fraction = float(np.max(corners[:, 0]) - np.min(corners[:, 0]))
        if width_fraction >= 0.84:
            return 0.85
        if width_fraction <= 0.60:
            return 1.45
        ratio = (0.84 - width_fraction) / 0.24
        return float(0.85 + np.clip(ratio, 0.0, 1.0) * 0.6)

    def _augment(self, image: np.ndarray, corners: np.ndarray, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return image, corners
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            corners[:, 0] = 1.0 - corners[:, 0]
            corners = corners[[1, 0, 3, 2]]
        width_fraction = float(np.max(corners[:, 0]) - np.min(corners[:, 0]))
        center_x = float(np.mean(corners[:, 0]))
        if width_fraction > 0.82 and random.random() < 0.55:
            scale = random.uniform(0.84, 0.94)
            shift_span = min(0.10, max((1.0 - scale) * 0.5 - 0.01, 0.02))
            shift_x = random.uniform(-shift_span, shift_span)
            shift_y = random.uniform(-0.04, 0.04)
            image, corners = reframe_image_and_corners(
                image,
                corners,
                scale=scale,
                shift_x=shift_x,
                shift_y=shift_y,
            )
        elif 0.66 <= width_fraction <= 0.82 and 0.38 <= center_x <= 0.62 and random.random() < 0.22:
            scale = random.uniform(0.92, 0.98)
            shift_left, shift_right = _allowed_shift_range(corners, scale)
            shift_cap = min(0.14, shift_right)
            shift_floor = max(-0.14, shift_left)
            if shift_floor < shift_cap:
                shift_x = random.uniform(shift_floor, shift_cap)
                shift_y = random.uniform(-0.02, 0.02)
                image, corners = reframe_image_and_corners(
                    image,
                    corners,
                    scale=scale,
                    shift_x=shift_x,
                    shift_y=shift_y,
                )
        scene_tags = set(row.get("scene_tags") or [])
        if "near_color_background" in scene_tags or "low_contrast_scene" in scene_tags:
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
                if "near_color_background" in scene_tags:
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
        if "black_frame_scene" in scene_tags and random.random() < 0.45:
            height, width = image.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.round(corners * np.array([width - 1, height - 1], dtype=np.float32)).astype(np.int32)
            cv2.fillConvexPoly(mask, points, 255)
            border = cv2.subtract(mask, cv2.erode(mask, np.ones((13, 13), dtype=np.uint8), iterations=1))
            image_f = image.astype(np.float32)
            image_f[border > 0] *= random.uniform(0.7, 0.88)
            image = np.clip(image_f, 0, 255).astype(np.uint8)
        alpha = 1.0 + random.uniform(-0.15, 0.15)
        beta = random.uniform(-18.0, 18.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return image, corners

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(row["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        corners = _normalize_manual_quad(row["manual_quad"], width, height)
        image, corners = self._augment(image, corners, row)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        image_f = image.astype(np.float32) / 255.0
        heatmaps = build_corner_heatmaps(corners, output_size=self.output_size)
        return {
            "image": torch.from_numpy(np.transpose(image_f, (2, 0, 1))),
            "heatmaps": torch.from_numpy(heatmaps),
            "corners": torch.from_numpy(corners.astype(np.float32)),
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
    geometry_loss_weight: float = 1.0,
    edge_supervision_weight: float = 0.2,
    split_dir: Path | None = None,
    sample_weight_power: float = 0.0,
    corner_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> GlobalTrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_split_dir = split_dir or (output_dir / "split")
    if split_dir is None:
        export_global_corner_split(dataset_root, active_split_dir, seed=seed, test_ratio=0.25)

    train_dataset = GlobalCornerDataset(active_split_dir / "train.jsonl", input_size=input_size, output_size=output_size, augment=True)
    test_dataset = GlobalCornerDataset(active_split_dir / "test.jsonl", input_size=input_size, output_size=output_size, augment=False)
    train_loader_kwargs: dict[str, Any] = {"batch_size": batch_size, "num_workers": 0}
    if sample_weight_power > 0.0:
        weights = torch.from_numpy(train_dataset.build_sample_weights(power=sample_weight_power)).double()
        train_loader_kwargs["sampler"] = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    else:
        train_loader_kwargs["shuffle"] = True
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = select_torch_device()
    model = CornerHeatmapNet(in_channels=3, channels=channels, output_channels=4).to(device)
    if init_model_path is not None:
        checkpoint = torch.load(init_model_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    corner_weight_tensor = torch.tensor(corner_weights, dtype=torch.float32, device=device)
    best_score = (math.inf, math.inf, math.inf, math.inf)
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = batch["heatmaps"].to(device=device, dtype=torch.float32)
            corners = batch["corners"].to(device=device, dtype=torch.float32)
            geometry_scales = batch["geometry_scale"].to(device=device, dtype=torch.float32)
            edge_scales = batch["edge_scale"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            pred = decode_heatmaps(logits)
            heatmap_loss = torch.mean((logits - targets) ** 2)
            coord_loss = compute_corner_weighted_coord_loss(pred, corners, corner_weight_tensor)
            geometry_loss, _ = compute_global_geometry_loss(pred, corners, sample_scales=geometry_scales)
            edge_loss = compute_edge_supervision_loss(pred, corners, output_size=output_size, sample_scales=edge_scales)
            loss = heatmap_loss + coord_loss * 2.5 + geometry_loss * geometry_loss_weight + edge_loss * edge_supervision_weight
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val = _evaluate(model, test_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)) if train_losses else 0.0, **val}
        history.append(row)
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
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("no checkpoint saved")

    model_path = output_dir / "global_corner_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_size": input_size,
            "output_size": output_size,
            "channels": channels,
            "decode_mode": "soft_argmax",
            "device": device.type,
            "corner_weights": [float(value) for value in corner_weights],
            "sample_weight_power": float(sample_weight_power),
        },
        model_path,
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
    parser.add_argument("--geometry-loss-weight", type=float, default=1.0)
    parser.add_argument("--edge-supervision-weight", type=float, default=0.2)
    parser.add_argument("--split-dir")
    parser.add_argument("--sample-weight-power", type=float, default=0.0)
    parser.add_argument(
        "--corner-weights",
        default="1,1,1,1",
        help="Comma-separated per-corner weights in TL,TR,BR,BL order.",
    )
    args = parser.parse_args()
    corner_weights = tuple(float(part.strip()) for part in str(args.corner_weights).split(","))
    if len(corner_weights) != 4:
        raise ValueError("--corner-weights requires four comma-separated values")

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
        geometry_loss_weight=float(args.geometry_loss_weight),
        edge_supervision_weight=float(args.edge_supervision_weight),
        split_dir=Path(args.split_dir) if args.split_dir else None,
        sample_weight_power=float(args.sample_weight_power),
        corner_weights=corner_weights,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
