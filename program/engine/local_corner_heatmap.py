from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from corner_train import select_torch_device, soft_argmax_2d
from perspective_detect import order_points
from local_corner_refine import build_patch_features


SPECIAL_OCCLUDED_PROJECT = "全球化重构下道路照明的出海之路"


def build_local_corner_heatmaps(
    points: list[list[float]] | np.ndarray,
    output_size: int = 16,
    sigma: float = 1.5,
) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    heatmaps = np.zeros((len(pts), output_size, output_size), dtype=np.float32)
    grid_y, grid_x = np.mgrid[0:output_size, 0:output_size].astype(np.float32)
    for index, (x_norm, y_norm) in enumerate(pts):
        x = float(np.clip(x_norm, 0.0, 1.0)) * (output_size - 1)
        y = float(np.clip(y_norm, 0.0, 1.0)) * (output_size - 1)
        dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
        heatmaps[index] = np.exp(-dist / max(2.0 * sigma * sigma, 1e-6))
    return heatmaps


def decode_local_corner_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    return soft_argmax_2d(heatmaps)


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def build_patch_metadata(row: dict[str, Any]) -> np.ndarray:
    patch = row.get("patch", {}) or {}
    patch_x = float(patch.get("x", 0.0))
    patch_y = float(patch.get("y", 0.0))
    patch_size = float(max(float(patch.get("size", 96.0)), 1.0))
    predicted_point = np.array(
        row.get("predicted_point", [patch_x + patch_size / 2.0, patch_y + patch_size / 2.0]),
        dtype=np.float32,
    )
    point_norm = np.array(
        [
            np.clip((predicted_point[0] - patch_x) / patch_size, 0.0, 1.0),
            np.clip((predicted_point[1] - patch_y) / patch_size, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    ordered = None
    if row.get("predicted_quad") is not None:
        ordered = order_points(np.array(row["predicted_quad"], dtype=np.float32))
    corner_index = int(row.get("corner_index", 0))
    corner_one_hot = np.zeros((4,), dtype=np.float32)
    corner_one_hot[max(0, min(3, corner_index))] = 1.0
    if ordered is None:
        width_feat = 0.5
        height_feat = 0.5
        angle_feat = 0.5
        prev_dir = np.zeros((2,), dtype=np.float32)
        next_dir = np.zeros((2,), dtype=np.float32)
    else:
        top_w = float(np.linalg.norm(ordered[1] - ordered[0]))
        bottom_w = float(np.linalg.norm(ordered[2] - ordered[3]))
        left_h = float(np.linalg.norm(ordered[3] - ordered[0]))
        right_h = float(np.linalg.norm(ordered[2] - ordered[1]))
        width_feat = float(np.tanh(((top_w + bottom_w) * 0.5) / max(patch_size * 4.0, 1.0)))
        height_feat = float(np.tanh(((left_h + right_h) * 0.5) / max(patch_size * 4.0, 1.0)))
        current = ordered[corner_index]
        prev_pt = ordered[(corner_index - 1) % 4]
        next_pt = ordered[(corner_index + 1) % 4]
        prev_dir = _normalize_vec(prev_pt - current)
        next_dir = _normalize_vec(next_pt - current)
        cosine = float(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0))
        angle_feat = float(np.arccos(cosine) / np.pi)
    metadata = np.concatenate(
        [
            point_norm,
            np.array([np.tanh(patch_size / 256.0), width_feat, height_feat, angle_feat], dtype=np.float32),
            prev_dir.astype(np.float32),
            next_dir.astype(np.float32),
            corner_one_hot,
        ],
        axis=0,
    )
    return metadata.astype(np.float32)


def build_corner_direction_target(row: dict[str, Any]) -> np.ndarray:
    manual_quad = row.get("manual_quad")
    corner_index = int(row.get("corner_index", 0))
    if manual_quad is None:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    ordered = order_points(np.array(manual_quad, dtype=np.float32))
    current = ordered[corner_index]
    prev_pt = ordered[(corner_index - 1) % 4]
    next_pt = ordered[(corner_index + 1) % 4]
    prev_dir = _normalize_vec(prev_pt - current)
    next_dir = _normalize_vec(next_pt - current)
    cosine = float(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0))
    angle_feat = float(np.arccos(cosine) / np.pi)
    return np.concatenate([prev_dir, next_dir, np.array([angle_feat], dtype=np.float32)], axis=0).astype(np.float32)


def build_corner_edge_maps(row: dict[str, Any], output_size: int = 16) -> np.ndarray:
    manual_quad = row.get("manual_quad")
    patch = row.get("patch", {}) or {}
    if manual_quad is None:
        return np.zeros((2, output_size, output_size), dtype=np.float32)
    patch_x = float(patch.get("x", 0.0))
    patch_y = float(patch.get("y", 0.0))
    patch_size = float(max(float(patch.get("size", 96.0)), 1.0))
    ordered = order_points(np.array(manual_quad, dtype=np.float32))
    corner_index = int(row.get("corner_index", 0))
    current = ordered[corner_index]
    prev_pt = ordered[(corner_index - 1) % 4]
    next_pt = ordered[(corner_index + 1) % 4]

    def to_patch_coord(point: np.ndarray) -> np.ndarray:
        x = np.clip((float(point[0]) - patch_x) / patch_size, 0.0, 1.0) * (output_size - 1)
        y = np.clip((float(point[1]) - patch_y) / patch_size, 0.0, 1.0) * (output_size - 1)
        return np.array([x, y], dtype=np.float32)

    def build_segment_band(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        grid_y, grid_x = np.mgrid[0:output_size, 0:output_size].astype(np.float32)
        points = np.stack([grid_x, grid_y], axis=-1)
        seg = end - start
        seg_norm_sq = float(np.dot(seg, seg))
        if seg_norm_sq <= 1e-6:
            dist = np.sqrt(np.sum((points - start.reshape(1, 1, 2)) ** 2, axis=-1))
        else:
            t = np.sum((points - start.reshape(1, 1, 2)) * seg.reshape(1, 1, 2), axis=-1) / seg_norm_sq
            t = np.clip(t, 0.0, 1.0)
            proj = start.reshape(1, 1, 2) + t[..., None] * seg.reshape(1, 1, 2)
            dist = np.sqrt(np.sum((points - proj) ** 2, axis=-1))
        sigma = max(output_size * 0.075, 1.0)
        return np.exp(-(dist**2) / max(2.0 * sigma * sigma, 1e-6)).astype(np.float32)

    current_xy = to_patch_coord(current)
    prev_xy = to_patch_coord(prev_pt)
    next_xy = to_patch_coord(next_pt)
    return np.stack([build_segment_band(current_xy, prev_xy), build_segment_band(current_xy, next_xy)], axis=0)


def build_corner_visibility_target(row: dict[str, Any], patch_image: np.ndarray) -> np.ndarray:
    manual_quad = row.get("manual_quad")
    patch = row.get("patch", {}) or {}
    if manual_quad is None:
        return np.zeros((2,), dtype=np.float32)
    height, width = patch_image.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((2,), dtype=np.float32)
    patch_x = float(patch.get("x", 0.0))
    patch_y = float(patch.get("y", 0.0))
    patch_size = float(max(float(patch.get("size", max(height, width))), 1.0))
    ordered = order_points(np.array(manual_quad, dtype=np.float32))
    corner_index = int(row.get("corner_index", 0))
    current = ordered[corner_index]
    prev_pt = ordered[(corner_index - 1) % 4]
    next_pt = ordered[(corner_index + 1) % 4]

    def to_patch_coord(point: np.ndarray) -> np.ndarray:
        x = np.clip((float(point[0]) - patch_x) / patch_size, 0.0, 1.0) * (width - 1)
        y = np.clip((float(point[1]) - patch_y) / patch_size, 0.0, 1.0) * (height - 1)
        return np.array([x, y], dtype=np.float32)

    gray = cv2.cvtColor(patch_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    local_contrast = np.abs(gray - blur)
    response = np.maximum(grad_mag, local_contrast)
    response = response / max(float(np.percentile(response, 99.0)), 1e-4)
    response = np.clip(response, 0.0, 1.0)

    def sample_branch_visibility(start: np.ndarray, end: np.ndarray) -> float:
        branch = end - start
        branch_len = float(np.linalg.norm(branch))
        if branch_len <= 2.0:
            return 0.0
        radius = max(1, int(round(min(height, width) * 0.035)))
        samples = np.linspace(0.18, 0.88, num=6, dtype=np.float32)
        values: list[float] = []
        for t in samples:
            point = start + branch * float(t)
            cx = int(np.clip(round(float(point[0])), 0, width - 1))
            cy = int(np.clip(round(float(point[1])), 0, height - 1))
            x0 = max(0, cx - radius)
            x1 = min(width, cx + radius + 1)
            y0 = max(0, cy - radius)
            y1 = min(height, cy + radius + 1)
            window = response[y0:y1, x0:x1]
            if window.size == 0:
                values.append(0.0)
                continue
            values.append(float(np.max(window)))
        if not values:
            return 0.0
        return float(np.clip(np.mean(values), 0.0, 1.0))

    current_xy = to_patch_coord(current)
    prev_xy = to_patch_coord(prev_pt)
    next_xy = to_patch_coord(next_pt)
    return np.array(
        [
            sample_branch_visibility(current_xy, prev_xy),
            sample_branch_visibility(current_xy, next_xy),
        ],
        dtype=np.float32,
    )


def compute_local_corner_sample_weight(row: dict[str, Any]) -> float:
    target = np.array(row.get("target_point_norm", [0.5, 0.5]), dtype=np.float32)
    predicted = build_patch_metadata(row)[:2]
    residual = float(np.linalg.norm(target - predicted))
    if residual <= 0.03:
        weight = 1.2
    elif residual <= 0.12:
        weight = 2.4
    elif residual <= 0.22:
        weight = 1.7
    else:
        weight = 1.1
    corner_index = int(row.get("corner_index", 0))
    if corner_index in {2, 3}:
        weight *= 1.2
    residual_norm = np.array(row.get("target_residual_norm", [0.0, 0.0]), dtype=np.float32)
    if corner_index in {2, 3} and abs(float(residual_norm[1])) > 0.12:
        weight *= 1.15
    if corner_index == 3:
        if abs(float(residual_norm[1])) > 0.12:
            weight *= 1.06
        if float(np.linalg.norm(residual_norm)) > 0.18:
            weight *= 1.08
    weight *= float(max(row.get("adaptive_weight", 1.0), 0.1))
    return float(weight)


def augment_patch_for_hard_cases(patch: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    image = np.array(patch, copy=True)
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image
    corner_index = int(row.get("corner_index", 0))
    residual_norm = np.array(row.get("target_residual_norm", [0.0, 0.0]), dtype=np.float32)
    project_name = str(row.get("project_name", ""))
    hard_case = (
        corner_index in {2, 3}
        or float(np.linalg.norm(residual_norm)) > 0.18
        or abs(float(residual_norm[1])) > 0.12
        or SPECIAL_OCCLUDED_PROJECT in project_name
    )
    if not hard_case:
        return image
    if random.random() < 0.18:
        alpha = random.uniform(0.18, 0.42)
        base = np.full_like(image, int(np.mean(image)))
        image = cv2.addWeighted(image, 1.0 - alpha, base, alpha, 0.0)
    target = np.array(row.get("target_point_norm", [0.5, 0.5]), dtype=np.float32)
    center = (
        int(np.clip(round(float(target[0]) * (width - 1)), 0, width - 1)),
        int(np.clip(round(float(target[1]) * (height - 1)), 0, height - 1)),
    )
    edge_target = build_corner_direction_target(row)
    for direction in (edge_target[:2], edge_target[2:4]):
        if random.random() >= 0.18:
            continue
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-5:
            continue
        direction = direction / norm
        length = int(round(random.uniform(0.25, 0.48) * min(width, height)))
        dx = int(round(float(direction[0]) * length))
        dy = int(round(float(direction[1]) * length))
        color_base = int(np.clip(np.mean(image) + random.uniform(-18.0, 18.0), 0.0, 255.0))
        thickness = max(1, int(round(min(width, height) * 0.025)))
        cv2.line(
            image,
            (center[0] - dx, center[1] - dy),
            (center[0] + dx, center[1] + dy),
            (color_base, color_base, color_base),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    if corner_index in {2, 3} and random.random() < 0.38:
        band_top = int(round(height * random.uniform(0.72, 0.84)))
        band_bottom = min(height, band_top + max(4, int(round(height * random.uniform(0.08, 0.16)))))
        band = image[band_top:band_bottom, :, :].copy()
        if band.size > 0:
            offset = random.uniform(-16.0, 22.0)
            band = np.clip(band.astype(np.float32) + offset, 0.0, 255.0).astype(np.uint8)
            image[band_top:band_bottom, :, :] = cv2.GaussianBlur(band, (5, 5), 0)
    return image


def cleanup_patch_for_special_cases(patch: np.ndarray, row: dict[str, Any], augment: bool) -> np.ndarray:
    image = np.array(patch, copy=True)
    project_name = str(row.get("project_name", ""))
    corner_index = int(row.get("corner_index", 0))
    if SPECIAL_OCCLUDED_PROJECT in project_name and corner_index == 1:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        x0 = max(0, int(round(width * 0.76)))
        x1 = min(width, int(round(width * 0.96)))
        y0 = 0
        y1 = min(height, int(round(height * 0.14)))
        mask[y0:y1, x0:x1] = 255
        image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    if augment and corner_index == 1 and random.random() < 0.18:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        x0 = max(0, int(round(width * random.uniform(0.78, 0.86))))
        x1 = min(width, int(round(width * random.uniform(0.92, 0.98))))
        y0 = 0
        y1 = min(height, int(round(height * random.uniform(0.08, 0.16))))
        mask[y0:y1, x0:x1] = 255
        image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return image


class LocalCornerHeatmapDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        input_size: int = 64,
        output_size: int = 16,
        augment: bool = False,
        allow_flips: bool = True,
        flip_prob: float | None = None,
        input_channels: int = 10,
    ) -> None:
        self.dataset_root = dataset_root
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment
        self.allow_flips = allow_flips
        if not allow_flips:
            self.flip_prob = 0.0
        elif flip_prob is None:
            self.flip_prob = 0.5
        else:
            self.flip_prob = float(np.clip(flip_prob, 0.0, 1.0))
        self.input_channels = input_channels
        self.rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        patch = cv2.imread(str(self.dataset_root / row["patch_path"]))
        if patch is None:
            raise FileNotFoundError(row["patch_path"])
        patch = cleanup_patch_for_special_cases(patch, row, augment=self.augment)
        if self.augment:
            patch = augment_patch_for_hard_cases(patch, row)
        target = np.array(row["target_point_norm"], dtype=np.float32)
        metadata = build_patch_metadata(row)
        edge_target = build_corner_direction_target(row)
        edge_maps = build_corner_edge_maps(row, output_size=self.output_size)
        visibility_target = build_corner_visibility_target(row, patch)
        if self.augment and self.flip_prob > 0.0 and random.random() < self.flip_prob:
            patch = cv2.flip(patch, 1)
            target[0] = 1.0 - target[0]
            metadata[0] = 1.0 - metadata[0]
            metadata[6] *= -1.0
            metadata[8] *= -1.0
            edge_target[0] *= -1.0
            edge_target[2] *= -1.0
            edge_maps = edge_maps[:, :, ::-1].copy()
        if self.augment and self.flip_prob > 0.0 and random.random() < self.flip_prob:
            patch = cv2.flip(patch, 0)
            target[1] = 1.0 - target[1]
            metadata[1] = 1.0 - metadata[1]
            metadata[7] *= -1.0
            metadata[9] *= -1.0
            edge_target[1] *= -1.0
            edge_target[3] *= -1.0
            edge_maps = edge_maps[:, ::-1, :].copy()
        features = build_patch_features(
            patch,
            int(row["corner_index"]),
            input_size=self.input_size,
            input_channels=self.input_channels,
        )
        heatmaps = build_local_corner_heatmaps([target.tolist()], output_size=self.output_size)
        return {
            "image": torch.from_numpy(features),
            "heatmaps": torch.from_numpy(heatmaps),
            "target": torch.from_numpy(target),
            "metadata": torch.from_numpy(metadata),
            "edge_target": torch.from_numpy(edge_target),
            "edge_maps": torch.from_numpy(edge_maps),
            "visibility_target": torch.from_numpy(visibility_target),
            "sample_weight": torch.tensor(compute_local_corner_sample_weight(row), dtype=torch.float32),
        }


class LocalCornerHeatmapNet(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(10, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stem(x))


@dataclass
class LocalCornerHeatmapTrainResult:
    model_path: str
    history_path: str
    best_epoch: int
    best_val_loss: float
    best_val_point_error: float


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = batch["heatmaps"].to(device=device, dtype=torch.float32)
            points = batch["target"].to(device=device, dtype=torch.float32)
            logits = model(images)
            pred = decode_local_corner_heatmaps(logits).squeeze(1)
            heatmap_loss = torch.mean((logits - targets) ** 2)
            coord_loss = F.smooth_l1_loss(pred, points)
            loss = heatmap_loss + coord_loss * 3.0
            err = torch.sqrt(torch.sum((pred - points) ** 2, dim=-1) + 1e-12)
            losses.extend(loss.detach().cpu().repeat(images.shape[0]).tolist())
            errors.extend(err.detach().cpu().tolist())
    arr = np.array(losses, dtype=np.float32) if losses else np.array([0.0], dtype=np.float32)
    err_arr = np.array(errors, dtype=np.float32) if errors else np.array([0.0], dtype=np.float32)
    return {"loss": float(arr.mean()), "point_error": float(err_arr.mean())}


def train_local_corner_heatmap_model(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 8e-4,
    input_size: int = 64,
    output_size: int = 16,
    channels: int = 16,
    seed: int = 7,
) -> LocalCornerHeatmapTrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = LocalCornerHeatmapDataset(dataset_dir / "train.jsonl", dataset_dir, input_size=input_size, output_size=output_size, augment=True)
    test_dataset = LocalCornerHeatmapDataset(dataset_dir / "test.jsonl", dataset_dir, input_size=input_size, output_size=output_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = select_torch_device()
    model = LocalCornerHeatmapNet(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = batch["heatmaps"].to(device=device, dtype=torch.float32)
            points = batch["target"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            pred = decode_local_corner_heatmaps(logits).squeeze(1)
            heatmap_loss = torch.mean((logits - targets) ** 2)
            coord_loss = F.smooth_l1_loss(pred, points)
            loss = heatmap_loss + coord_loss * 3.0
            loss.backward()
            optimizer.step()
        metrics = _evaluate(model, test_loader, device)
        history.append({"epoch": epoch, **metrics})
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_metrics = metrics
    if best_state is None or best_metrics is None:
        raise RuntimeError("no checkpoint")
    model_path = output_dir / "local_corner_heatmap_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_size": input_size,
            "output_size": output_size,
            "channels": channels,
            "device": device.type,
        },
        model_path,
    )
    history_path = output_dir / "local_corner_heatmap_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return LocalCornerHeatmapTrainResult(
        str(model_path),
        str(history_path),
        best_epoch,
        float(best_metrics["loss"]),
        float(best_metrics["point_error"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train local corner heatmap refiner")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--output-size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = train_local_corner_heatmap_model(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        output_size=args.output_size,
        channels=args.channels,
        seed=args.seed,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
