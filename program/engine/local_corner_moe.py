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


def decode_moe_output(heatmaps: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = heatmaps.shape
    flat = heatmaps.reshape(batch, 1, height * width)
    weights = F.softmax(flat * 12.0, dim=-1)
    xs = torch.linspace(0.0, 1.0, width, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0.0, 1.0, height, device=heatmaps.device, dtype=heatmaps.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base_x = torch.sum(weights * grid_x.reshape(1, 1, -1), dim=-1)
    base_y = torch.sum(weights * grid_y.reshape(1, 1, -1), dim=-1)
    offsets_flat = offsets.reshape(batch, 2, height * width)
    residual = torch.sum(offsets_flat * weights.squeeze(1).unsqueeze(1), dim=-1)
    scale = torch.tensor([max(width - 1, 1), max(height - 1, 1)], dtype=heatmaps.dtype, device=heatmaps.device)
    coords = torch.stack([base_x.squeeze(1), base_y.squeeze(1)], dim=-1)
    return torch.clamp(coords + residual / scale, 0.0, 1.0)


def remap_legacy_moe_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "gate_head.0.weight" in state_dict or "gate_head.1.weight" not in state_dict:
        return state_dict
    remapped = dict(state_dict)
    if "gate_head.1.weight" in remapped:
        remapped["gate_head.0.weight"] = remapped.pop("gate_head.1.weight")
    if "gate_head.1.bias" in remapped:
        remapped["gate_head.0.bias"] = remapped.pop("gate_head.1.bias")
    if "gate_head.3.weight" in remapped:
        remapped["gate_head.2.weight"] = remapped.pop("gate_head.3.weight")
    if "gate_head.3.bias" in remapped:
        remapped["gate_head.2.bias"] = remapped.pop("gate_head.3.bias")
    return remapped


class LocalCornerMoENet(nn.Module):
    def __init__(self, channels: int = 16, experts: int = 3, metadata_dim: int = 0) -> None:
        super().__init__()
        self.experts = experts
        self.metadata_dim = metadata_dim
        self.stem = nn.Sequential(
            nn.Conv2d(10, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gate_pool = nn.AdaptiveAvgPool2d((1, 1))
        gate_in = channels * 4
        self.metadata_head = None
        if metadata_dim > 0:
            self.metadata_head = nn.Sequential(
                nn.Linear(metadata_dim, channels * 2),
                nn.ReLU(inplace=True),
                nn.Linear(channels * 2, channels * 2),
                nn.ReLU(inplace=True),
            )
            gate_in += channels * 2
        self.gate_head = nn.Sequential(
            nn.Linear(gate_in, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, experts),
        )
        self.heatmap_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels * 4, 1, kernel_size=1),
                )
                for _ in range(experts)
            ]
        )
        self.offset_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels * 4, 2, kernel_size=1),
                    nn.Tanh(),
                )
                for _ in range(experts)
            ]
        )

    def forward(self, x: torch.Tensor, metadata: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        pooled = self.gate_pool(feat).flatten(1)
        if self.metadata_head is not None:
            if metadata is None:
                metadata = torch.zeros((pooled.shape[0], self.metadata_dim), dtype=pooled.dtype, device=pooled.device)
            pooled = torch.cat([pooled, self.metadata_head(metadata)], dim=-1)
        gates = F.softmax(self.gate_head(pooled), dim=-1)
        heatmaps = torch.stack([head(feat) for head in self.heatmap_heads], dim=1)
        offsets = torch.stack([head(feat) for head in self.offset_heads], dim=1)
        gate_map = gates.view(gates.shape[0], self.experts, 1, 1, 1)
        heatmap = torch.sum(heatmaps * gate_map, dim=1)
        offset = torch.sum(offsets * gate_map, dim=1)
        return heatmap, offset, gates


@dataclass
class LocalCornerMoETrainResult:
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
            metadata = batch["metadata"].to(device=device, dtype=torch.float32)
            sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
            heatmaps, offsets, gates = model(images, metadata)
            pred = decode_moe_output(heatmaps, offsets)
            heatmap_loss = ((heatmaps - targets) ** 2).flatten(1).mean(dim=-1)
            coord_loss = F.smooth_l1_loss(pred, points, reduction="none").mean(dim=-1)
            gate_reg = torch.mean(torch.sum(gates * torch.log(gates + 1e-8), dim=-1))
            weighted_loss = (heatmap_loss + coord_loss * 3.5) * sample_weight
            loss = weighted_loss.mean() + gate_reg * 0.005
            err = torch.sqrt(torch.sum((pred - points) ** 2, dim=-1) + 1e-12)
            losses.extend(loss.detach().cpu().repeat(images.shape[0]).tolist())
            errors.extend(err.detach().cpu().tolist())
    arr = np.array(losses, dtype=np.float32) if losses else np.array([0.0], dtype=np.float32)
    err_arr = np.array(errors, dtype=np.float32) if errors else np.array([0.0], dtype=np.float32)
    return {"loss": float(arr.mean()), "point_error": float(err_arr.mean())}


def train_local_corner_moe_model(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 6e-4,
    input_size: int = 96,
    output_size: int = 24,
    channels: int = 24,
    experts: int = 3,
    metadata_dim: int = 14,
    seed: int = 7,
) -> LocalCornerMoETrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = LocalCornerHeatmapDataset(dataset_dir / "train.jsonl", dataset_dir, input_size=input_size, output_size=output_size, augment=True)
    test_dataset = LocalCornerHeatmapDataset(dataset_dir / "test.jsonl", dataset_dir, input_size=input_size, output_size=output_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = select_torch_device()
    model = LocalCornerMoENet(channels=channels, experts=experts, metadata_dim=metadata_dim).to(device)
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
            metadata = batch["metadata"].to(device=device, dtype=torch.float32)
            sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            heatmaps, offsets, gates = model(images, metadata)
            pred = decode_moe_output(heatmaps, offsets)
            heatmap_loss = ((heatmaps - targets) ** 2).flatten(1).mean(dim=-1)
            coord_loss = F.smooth_l1_loss(pred, points, reduction="none").mean(dim=-1)
            gate_reg = torch.mean(torch.sum(gates * torch.log(gates + 1e-8), dim=-1))
            weighted_loss = (heatmap_loss + coord_loss * 3.5) * sample_weight
            loss = weighted_loss.mean() + gate_reg * 0.005
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
    model_path = output_dir / "local_corner_moe_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_size": input_size,
            "output_size": output_size,
            "channels": channels,
            "experts": experts,
            "metadata_dim": metadata_dim,
            "device": device.type,
        },
        model_path,
    )
    history_path = output_dir / "local_corner_moe_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return LocalCornerMoETrainResult(
        str(model_path),
        str(history_path),
        best_epoch,
        float(best_metrics["loss"]),
        float(best_metrics["point_error"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train local corner MoE refiner")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--output-size", type=int, default=24)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--metadata-dim", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = train_local_corner_moe_model(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        output_size=args.output_size,
        channels=args.channels,
        experts=args.experts,
        metadata_dim=args.metadata_dim,
        seed=args.seed,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
