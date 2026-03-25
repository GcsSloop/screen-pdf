from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from corner_train import select_torch_device
from local_corner_heatmap import LocalCornerHeatmapDataset


def build_simcc_target(point: np.ndarray, bins: int, sigma: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    point = np.array(point, dtype=np.float32)
    xs = np.arange(bins, dtype=np.float32)
    ys = np.arange(bins, dtype=np.float32)
    x_center = float(np.clip(point[0], 0.0, 1.0)) * max(bins - 1, 1)
    y_center = float(np.clip(point[1], 0.0, 1.0)) * max(bins - 1, 1)
    x_target = np.exp(-((xs - x_center) ** 2) / max(2.0 * sigma * sigma, 1e-6))
    y_target = np.exp(-((ys - y_center) ** 2) / max(2.0 * sigma * sigma, 1e-6))
    x_target /= max(float(x_target.sum()), 1e-6)
    y_target /= max(float(y_target.sum()), 1e-6)
    return x_target.astype(np.float32), y_target.astype(np.float32)


def decode_simcc_logits(x_logits: torch.Tensor, y_logits: torch.Tensor) -> torch.Tensor:
    x_weights = F.softmax(x_logits, dim=-1)
    y_weights = F.softmax(y_logits, dim=-1)
    x_axis = torch.linspace(0.0, 1.0, x_logits.shape[-1], device=x_logits.device, dtype=x_logits.dtype)
    y_axis = torch.linspace(0.0, 1.0, y_logits.shape[-1], device=y_logits.device, dtype=y_logits.dtype)
    x = torch.sum(x_weights * x_axis.view(1, -1), dim=-1)
    y = torch.sum(y_weights * y_axis.view(1, -1), dim=-1)
    return torch.stack([x, y], dim=-1)


class LocalCornerSimCCDataset(LocalCornerHeatmapDataset):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        input_size: int = 96,
        coord_bins: int = 192,
        augment: bool = False,
    ) -> None:
        super().__init__(manifest_path, dataset_root, input_size=input_size, output_size=16, augment=augment)
        self.coord_bins = coord_bins

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        target = item["target"].numpy()
        x_target, y_target = build_simcc_target(target, bins=self.coord_bins)
        item.pop("heatmaps")
        item["target_x"] = torch.from_numpy(x_target)
        item["target_y"] = torch.from_numpy(y_target)
        return item


class LocalCornerSimCCNet(nn.Module):
    def __init__(self, channels: int = 24, coord_bins: int = 192, metadata_dim: int = 0) -> None:
        super().__init__()
        self.coord_bins = coord_bins
        self.metadata_dim = metadata_dim
        self.stem = nn.Sequential(
            nn.Conv2d(10, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        embed_dim = channels * 4
        self.metadata_head = None
        if metadata_dim > 0:
            self.metadata_head = nn.Sequential(
                nn.Linear(metadata_dim, channels * 2),
                nn.ReLU(inplace=True),
                nn.Linear(channels * 2, channels * 2),
                nn.ReLU(inplace=True),
            )
            embed_dim += channels * 2
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, channels * 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 8, channels * 4),
            nn.ReLU(inplace=True),
        )
        self.x_head = nn.Linear(channels * 4, coord_bins)
        self.y_head = nn.Linear(channels * 4, coord_bins)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.pool(self.stem(x)).flatten(1)
        if self.metadata_head is not None:
            if metadata is None:
                metadata = torch.zeros((feat.shape[0], self.metadata_dim), dtype=feat.dtype, device=feat.device)
            feat = torch.cat([feat, self.metadata_head(metadata)], dim=-1)
        hidden = self.shared(feat)
        return self.x_head(hidden), self.y_head(hidden)


@dataclass
class LocalCornerSimCCTrainResult:
    model_path: str
    history_path: str
    best_epoch: int
    best_val_loss: float
    best_val_point_error: float


def _soft_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            metadata = batch["metadata"].to(device=device, dtype=torch.float32)
            points = batch["target"].to(device=device, dtype=torch.float32)
            target_x = batch["target_x"].to(device=device, dtype=torch.float32)
            target_y = batch["target_y"].to(device=device, dtype=torch.float32)
            sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
            x_logits, y_logits = model(images, metadata)
            pred = decode_simcc_logits(x_logits, y_logits)
            cls_loss = _soft_ce_loss(x_logits, target_x) + _soft_ce_loss(y_logits, target_y)
            coord_loss = F.smooth_l1_loss(pred, points, reduction="none").mean(dim=-1)
            loss = (cls_loss + coord_loss * 4.0) * sample_weight
            err = torch.sqrt(torch.sum((pred - points) ** 2, dim=-1) + 1e-12)
            losses.extend(loss.detach().cpu().tolist())
            errors.extend(err.detach().cpu().tolist())
    arr = np.array(losses, dtype=np.float32) if losses else np.array([0.0], dtype=np.float32)
    err_arr = np.array(errors, dtype=np.float32) if errors else np.array([0.0], dtype=np.float32)
    return {"loss": float(arr.mean()), "point_error": float(err_arr.mean())}


def train_local_corner_simcc_model(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 4e-4,
    input_size: int = 96,
    coord_bins: int = 192,
    channels: int = 24,
    metadata_dim: int = 14,
    seed: int = 7,
) -> LocalCornerSimCCTrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = LocalCornerSimCCDataset(dataset_dir / "train.jsonl", dataset_dir, input_size=input_size, coord_bins=coord_bins, augment=True)
    test_dataset = LocalCornerSimCCDataset(dataset_dir / "test.jsonl", dataset_dir, input_size=input_size, coord_bins=coord_bins, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = select_torch_device()
    model = LocalCornerSimCCNet(channels=channels, coord_bins=coord_bins, metadata_dim=metadata_dim).to(device)
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
            metadata = batch["metadata"].to(device=device, dtype=torch.float32)
            points = batch["target"].to(device=device, dtype=torch.float32)
            target_x = batch["target_x"].to(device=device, dtype=torch.float32)
            target_y = batch["target_y"].to(device=device, dtype=torch.float32)
            sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            x_logits, y_logits = model(images, metadata)
            pred = decode_simcc_logits(x_logits, y_logits)
            cls_loss = _soft_ce_loss(x_logits, target_x) + _soft_ce_loss(y_logits, target_y)
            coord_loss = F.smooth_l1_loss(pred, points, reduction="none").mean(dim=-1)
            loss = ((cls_loss + coord_loss * 4.0) * sample_weight).mean()
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
    model_path = output_dir / "local_corner_simcc_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_size": input_size,
            "coord_bins": coord_bins,
            "channels": channels,
            "metadata_dim": metadata_dim,
            "device": device.type,
        },
        model_path,
    )
    history_path = output_dir / "local_corner_simcc_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return LocalCornerSimCCTrainResult(
        str(model_path),
        str(history_path),
        best_epoch,
        float(best_metrics["loss"]),
        float(best_metrics["point_error"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train local corner SimCC refiner")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--coord-bins", type=int, default=192)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--metadata-dim", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = train_local_corner_simcc_model(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        coord_bins=args.coord_bins,
        channels=args.channels,
        metadata_dim=args.metadata_dim,
        seed=args.seed,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
