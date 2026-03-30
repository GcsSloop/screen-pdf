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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataset_benchmark import quad_geometry_metrics, summarize_geometry_metric_rows


def build_corner_heatmaps(
    corners: list[list[float]] | np.ndarray,
    output_size: int = 64,
    sigma: float = 2.0,
) -> np.ndarray:
    points = np.array(corners, dtype=np.float32)
    heatmaps = np.zeros((4, output_size, output_size), dtype=np.float32)
    grid_y, grid_x = np.mgrid[0:output_size, 0:output_size].astype(np.float32)
    for index, (x_norm, y_norm) in enumerate(points):
        x = float(np.clip(x_norm, 0.0, 1.0)) * (output_size - 1)
        y = float(np.clip(y_norm, 0.0, 1.0)) * (output_size - 1)
        dist = (grid_x - x) ** 2 + (grid_y - y) ** 2
        heatmaps[index] = np.exp(-dist / max(2.0 * sigma * sigma, 1e-6))
    return heatmaps


def _draw_polygon_mask(points: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    pts = np.round(points * (size - 1)).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 1.0)
    return mask


class CornerSampleDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        input_size: int = 256,
        output_size: int = 64,
        augment: bool = False,
        cache_images: bool = False,
    ) -> None:
        self.dataset_root = dataset_root
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment
        self.cache_images = cache_images
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        self.rows = [json.loads(line) for line in lines if line.strip()]
        self._cache: list[dict[str, np.ndarray] | None] | None = None
        if self.cache_images:
            image_cache: dict[str, np.ndarray] = {}
            self._cache = []
            for row in self.rows:
                roi_path = str(row["roi_path"])
                cached_image = image_cache.get(roi_path)
                if cached_image is None:
                    base = self._load_base_item(row)
                    cached_image = base["image"]
                    image_cache[roi_path] = cached_image
                self._cache.append(
                    {
                        "image": cached_image,
                        "corners": np.array(row["corner_norm"], dtype=np.float32),
                        "coarse": np.array(row["coarse_quad_norm"], dtype=np.float32),
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def build_sample_weights(self, power: float = 0.0) -> np.ndarray:
        weights = np.ones((len(self.rows),), dtype=np.float32)
        if power <= 0.0 or not self.rows:
            return weights
        errors: list[float] = []
        for row in self.rows:
            corners = np.array(row["corner_norm"], dtype=np.float32)
            coarse = np.array(row["coarse_quad_norm"], dtype=np.float32)
            errors.append(float(np.linalg.norm(coarse - corners, axis=1).mean()))
        err_arr = np.array(errors, dtype=np.float32)
        scale = max(float(err_arr.mean()), 1e-6)
        return 1.0 + np.power(err_arr / scale, power).astype(np.float32)

    def _maybe_augment(self, image: np.ndarray, corners: np.ndarray, coarse: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.augment:
            return image, corners, coarse
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            corners[:, 0] = 1.0 - corners[:, 0]
            coarse[:, 0] = 1.0 - coarse[:, 0]
            corners = corners[[1, 0, 3, 2]]
            coarse = coarse[[1, 0, 3, 2]]
        alpha = 1.0 + random.uniform(-0.12, 0.12)
        beta = random.uniform(-10.0, 10.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return image, corners, coarse

    def _load_base_item(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        image_path = self.dataset_root / row["roi_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        corners = np.array(row["corner_norm"], dtype=np.float32)
        coarse = np.array(row["coarse_quad_norm"], dtype=np.float32)
        return {
            "image": image,
            "corners": corners,
            "coarse": coarse,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        if self._cache is not None:
            cached = self._cache[index]
            assert cached is not None
            image = np.array(cached["image"], copy=True)
            corners = np.array(cached["corners"], copy=True)
            coarse = np.array(cached["coarse"], copy=True)
        else:
            base = self._load_base_item(row)
            image = base["image"]
            corners = base["corners"]
            coarse = base["coarse"]
        image, corners, coarse = self._maybe_augment(image, corners, coarse)
        image_f = image.astype(np.float32) / 255.0
        coarse_mask = _draw_polygon_mask(coarse, self.input_size)
        features = np.concatenate([np.transpose(image_f, (2, 0, 1)), coarse_mask[None, ...]], axis=0)
        heatmaps = build_corner_heatmaps(corners, output_size=self.output_size)
        return {
            "image": torch.from_numpy(features),
            "heatmaps": torch.from_numpy(heatmaps),
            "corners": torch.from_numpy(corners),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CornerHeatmapNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        channels: int = 32,
        output_channels: int = 4,
        head_mode: str = "heatmap",
    ) -> None:
        super().__init__()
        if head_mode not in {"heatmap", "heatmap_offset"}:
            raise ValueError(f"unsupported head_mode: {head_mode}")
        self.head_mode = head_mode
        self.output_channels = output_channels
        self.stem = ConvBlock(in_channels, channels)
        self.down1 = ConvBlock(channels, channels * 2, stride=2)
        self.down2 = ConvBlock(channels * 2, channels * 4, stride=2)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, output_channels, kernel_size=1),
        )
        self.offset_head = None
        if self.head_mode == "heatmap_offset":
            self.offset_head = nn.Sequential(
                nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels * 2, output_channels * 2, kernel_size=1),
                nn.Tanh(),
            )
            nn.init.zeros_(self.offset_head[2].weight)
            nn.init.zeros_(self.offset_head[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        heatmaps = self.heatmap_head(x)
        if self.offset_head is None:
            return heatmaps
        offsets = self.offset_head(x)
        batch, _, height, width = offsets.shape
        offsets = offsets.view(batch, self.output_channels, 2, height, width)
        return heatmaps, offsets


def remap_legacy_head_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("heatmap_head.") for key in state_dict.keys()):
        return state_dict
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("head."):
            remapped[f"heatmap_{key}"] = value
        else:
            remapped[key] = value
    return remapped


def initialize_model_from_checkpoint(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    source_state = remap_legacy_head_state_dict(checkpoint["state_dict"])
    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected init keys: {unexpected}")
    allowed_missing = [key for key in missing if key.startswith("offset_head.")]
    if len(allowed_missing) != len(missing):
        raise RuntimeError(f"incompatible init checkpoint, missing keys: {missing}")


def freeze_model_backbone_for_offset_tuning(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("offset_head.")


def select_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def soft_argmax_2d(heatmaps: torch.Tensor, temperature: float = 12.0) -> torch.Tensor:
    batch, channels, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, channels, height * width)
    weights = F.softmax(flat * temperature, dim=-1)
    xs = torch.linspace(0.0, 1.0, width, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0.0, 1.0, height, device=heatmaps.device, dtype=heatmaps.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    coords_x = torch.sum(weights * grid_x.reshape(1, 1, -1), dim=-1)
    coords_y = torch.sum(weights * grid_y.reshape(1, 1, -1), dim=-1)
    return torch.stack([coords_x, coords_y], dim=-1)


def decode_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    return soft_argmax_2d(heatmaps)


def decode_heatmaps_with_offsets(heatmaps: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    base = soft_argmax_2d(heatmaps)
    batch, channels, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, channels, height * width)
    weights = F.softmax(flat * 12.0, dim=-1)
    offsets_flat = offsets.reshape(batch, channels, 2, height * width)
    residual = torch.sum(offsets_flat * weights.unsqueeze(2), dim=-1)
    scale = torch.tensor(
        [max(width - 1, 1), max(height - 1, 1)],
        dtype=heatmaps.dtype,
        device=heatmaps.device,
    ).view(1, 1, 2)
    return torch.clamp(base + residual / scale, 0.0, 1.0)


def decode_heatmaps_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = heatmaps.shape
    flat = heatmaps.view(batch, channels, height * width)
    indices = flat.argmax(dim=-1)
    ys = torch.div(indices, width, rounding_mode="floor").float()
    xs = (indices % width).float()
    coords = torch.stack(
        [
            xs / max(width - 1, 1),
            ys / max(height - 1, 1),
        ],
        dim=-1,
    )
    return coords


def split_model_output(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    head_mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if head_mode == "heatmap_offset":
        heatmaps, offsets = output
        return heatmaps, offsets
    return output, None


def decode_model_output(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    decode_mode: str = "soft_argmax",
    head_mode: str = "heatmap",
) -> torch.Tensor:
    heatmaps, offsets = split_model_output(output, head_mode=head_mode)
    if head_mode == "heatmap_offset" and decode_mode == "soft_argmax_offset":
        if offsets is None:
            raise ValueError("offset head requested but model returned no offsets")
        return decode_heatmaps_with_offsets(heatmaps, offsets)
    if decode_mode == "soft_argmax":
        return decode_heatmaps(heatmaps)
    return decode_heatmaps_argmax(heatmaps)


def normalized_point_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = predicted - target
    distances = torch.sqrt(torch.sum(delta * delta, dim=-1) + 1e-12)
    return distances.mean(dim=-1)


@dataclass
class TrainResult:
    model_path: str
    history_path: str
    best_epoch: int
    best_val_loss: float
    best_val_point_error: float
    best_val_point_le_0_05: float
    best_val_point_le_0_03: float
    best_val_point_le_0_02: float
    best_val_point_le_0_01: float


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    head_mode: str = "heatmap",
    decode_mode: str = "soft_argmax",
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metric_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = batch["heatmaps"].to(device=device, dtype=torch.float32)
            corners = batch["corners"].to(device=device, dtype=torch.float32)
            output = model(images)
            logits, _ = split_model_output(output, head_mode=head_mode)
            pred = decode_model_output(output, decode_mode=decode_mode, head_mode=head_mode)
            heatmap_loss = torch.mean((logits - targets) ** 2)
            coord_loss = F.smooth_l1_loss(pred, corners)
            loss = heatmap_loss + coord_loss * 3.0
            losses.extend(loss.detach().cpu().repeat(images.shape[0]).tolist())
            pred_np = pred.detach().cpu().numpy()
            corners_np = corners.detach().cpu().numpy()
            for pred_row, target_row in zip(pred_np, corners_np, strict=True):
                metric_rows.append(quad_geometry_metrics(target_row, pred_row))
    if not metric_rows:
        return {
            "loss": 0.0,
            "point_error_mean": 0.0,
            "point_le_0_05": 0.0,
            "point_le_0_03": 0.0,
            "point_le_0_02": 0.0,
            "point_le_0_01": 0.0,
            "screen_relative_error_mean": 0.0,
            "max_corner_error_mean": 0.0,
            "perspective_tilt_error_mean": 0.0,
            "quad_inset_ratio_mean": 0.0,
        }
    summary = summarize_geometry_metric_rows(metric_rows)
    return {
        "loss": float(np.mean(losses)),
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


def train_corner_model(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int = 18,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    input_size: int = 128,
    output_size: int = 32,
    channels: int = 24,
    seed: int = 7,
    head_mode: str = "heatmap",
    init_model_path: Path | None = None,
    sample_weight_power: float = 0.0,
    freeze_base: bool = False,
    cache_images: bool = False,
) -> TrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CornerSampleDataset(
        dataset_dir / "train.jsonl",
        dataset_dir,
        input_size=input_size,
        output_size=output_size,
        augment=True,
        cache_images=cache_images,
    )
    test_dataset = CornerSampleDataset(
        dataset_dir / "test.jsonl",
        dataset_dir,
        input_size=input_size,
        output_size=output_size,
        augment=False,
        cache_images=cache_images,
    )
    train_loader_kwargs: dict[str, Any] = {"batch_size": batch_size, "num_workers": 0}
    if sample_weight_power > 0.0:
        weights = torch.from_numpy(train_dataset.build_sample_weights(power=sample_weight_power)).double()
        train_loader_kwargs["sampler"] = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
    else:
        train_loader_kwargs["shuffle"] = True
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = select_torch_device()
    decode_mode = "soft_argmax_offset" if head_mode == "heatmap_offset" else "soft_argmax"
    model = CornerHeatmapNet(in_channels=4, channels=channels, output_channels=4, head_mode=head_mode).to(device)
    if init_model_path is not None:
        checkpoint = torch.load(init_model_path, map_location="cpu")
        initialize_model_from_checkpoint(model, checkpoint)
    if freeze_base:
        freeze_model_backbone_for_offset_tuning(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters configured")
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    best_score = (math.inf, math.inf, math.inf, math.inf)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = batch["heatmaps"].to(device=device, dtype=torch.float32)
            corners = batch["corners"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            logits, _ = split_model_output(output, head_mode=head_mode)
            pred = decode_model_output(output, decode_mode=decode_mode, head_mode=head_mode)
            heatmap_loss = torch.mean((logits - targets) ** 2)
            coord_loss = F.smooth_l1_loss(pred, corners)
            loss = heatmap_loss + coord_loss * 3.0
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_metrics = evaluate_model(model, test_loader, device, head_mode=head_mode, decode_mode=decode_mode)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        score = (
            float(val_metrics["screen_relative_error_mean"]),
            float(val_metrics["max_corner_error_mean"]),
            float(val_metrics["perspective_tilt_error_mean"]),
            float(val_metrics["point_error_mean"]),
        )
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None or best_metrics is None:
        raise RuntimeError("training produced no checkpoint")

    model_path = output_dir / "corner_heatmap_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_size": input_size,
            "output_size": output_size,
            "channels": channels,
            "decode_mode": decode_mode,
            "head_mode": head_mode,
            "device": device.type,
        },
        model_path,
    )
    history_path = output_dir / "corner_heatmap_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return TrainResult(
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
    parser = argparse.ArgumentParser(description="Train a lightweight corner heatmap model")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--head-mode", choices=["heatmap", "heatmap_offset"], default="heatmap")
    parser.add_argument("--init-model")
    parser.add_argument("--sample-weight-power", type=float, default=0.0)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    args = parser.parse_args()

    result = train_corner_model(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        output_size=args.output_size,
        channels=args.channels,
        seed=args.seed,
        head_mode=args.head_mode,
        init_model_path=Path(args.init_model) if args.init_model else None,
        sample_weight_power=args.sample_weight_power,
        freeze_base=bool(args.freeze_base),
        cache_images=bool(args.cache_images),
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
