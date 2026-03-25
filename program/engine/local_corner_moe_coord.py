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
from local_corner_moe import decode_moe_output


def decode_moe_coord_output(
    heatmaps: torch.Tensor,
    offsets: torch.Tensor,
    coord_head: torch.Tensor,
    coord_mix: float = 0.25,
) -> torch.Tensor:
    base = decode_moe_output(heatmaps, offsets)
    return torch.clamp(base * (1.0 - coord_mix) + coord_head * coord_mix, 0.0, 1.0)


class LocalCornerMoECoordNet(nn.Module):
    def __init__(self, channels: int = 24, experts: int = 4, metadata_dim: int = 14, input_channels: int = 10) -> None:
        super().__init__()
        self.experts = experts
        self.metadata_dim = metadata_dim
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        gate_in = channels * 4
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
        self.coord_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_in, channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels * 2, 2),
                    nn.Sigmoid(),
                )
                for _ in range(experts)
            ]
        )
        self.edge_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_in, channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels * 2, 5),
                )
                for _ in range(experts)
            ]
        )
        self.edgemap_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels * 4, channels * 4, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels * 4, 2, kernel_size=1),
                )
                for _ in range(experts)
            ]
        )
        self.visibility_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_in, channels * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels * 2, 2),
                    nn.Sigmoid(),
                )
                for _ in range(experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        metadata: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        pooled = self.pool(feat).flatten(1)
        meta = self.metadata_head(metadata)
        gate_feat = torch.cat([pooled, meta], dim=-1)
        gates = F.softmax(self.gate_head(gate_feat), dim=-1)
        heatmaps = torch.stack([head(feat) for head in self.heatmap_heads], dim=1)
        offsets = torch.stack([head(feat) for head in self.offset_heads], dim=1)
        coords = torch.stack([head(gate_feat) for head in self.coord_heads], dim=1)
        edge_raw = torch.stack([head(gate_feat) for head in self.edge_heads], dim=1)
        edge_maps = torch.stack([head(feat) for head in self.edgemap_heads], dim=1)
        visibility_raw = torch.stack([head(gate_feat) for head in self.visibility_heads], dim=1)
        gate_map = gates.view(gates.shape[0], self.experts, 1, 1, 1)
        heatmap = torch.sum(heatmaps * gate_map, dim=1)
        offset = torch.sum(offsets * gate_map, dim=1)
        edge_map = torch.sum(edge_maps * gate_map, dim=1)
        coord = torch.sum(coords * gates.unsqueeze(-1), dim=1)
        edge = torch.sum(edge_raw * gates.unsqueeze(-1), dim=1)
        visibility = torch.sum(visibility_raw * gates.unsqueeze(-1), dim=1)
        prev = F.normalize(torch.tanh(edge[:, 0:2]), dim=-1, eps=1e-6)
        nxt = F.normalize(torch.tanh(edge[:, 2:4]), dim=-1, eps=1e-6)
        angle = torch.sigmoid(edge[:, 4:5])
        edge_target = torch.cat([prev, nxt, angle], dim=-1)
        return heatmap, offset, coord, edge_target, edge_map, visibility, gates


@dataclass
class LocalCornerMoECoordTrainResult:
    model_path: str
    history_path: str
    best_epoch: int
    best_val_loss: float
    best_val_point_error: float


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    coord_mix: float,
    edge_loss_weight: float,
    edge_map_loss_weight: float,
    visibility_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    errors: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            metadata = batch["metadata"].to(device=device, dtype=torch.float32)
            targets = batch["heatmaps"].to(device=device, dtype=torch.float32)
            points = batch["target"].to(device=device, dtype=torch.float32)
            edge_target = batch["edge_target"].to(device=device, dtype=torch.float32)
            edge_maps_target = batch["edge_maps"].to(device=device, dtype=torch.float32)
            visibility_target = batch["visibility_target"].to(device=device, dtype=torch.float32)
            sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
            heatmaps, offsets, coord_head, edge_head, edge_map_logits, visibility_pred, gates = model(images, metadata)
            pred = decode_moe_coord_output(heatmaps, offsets, coord_head, coord_mix=coord_mix)
            heatmap_loss = ((heatmaps - targets) ** 2).flatten(1).mean(dim=-1)
            decode_loss = F.smooth_l1_loss(pred, points, reduction="none").mean(dim=-1)
            direct_loss = F.smooth_l1_loss(coord_head, points, reduction="none").mean(dim=-1)
            edge_loss = F.smooth_l1_loss(edge_head, edge_target, reduction="none").mean(dim=-1)
            edge_map_pred = torch.sigmoid(edge_map_logits)
            edge_map_loss = ((edge_map_pred - edge_maps_target) ** 2).flatten(1).mean(dim=-1)
            visibility_loss = F.smooth_l1_loss(visibility_pred, visibility_target, reduction="none").mean(dim=-1)
            gate_reg = torch.mean(torch.sum(gates * torch.log(gates + 1e-8), dim=-1))
            weighted = (
                heatmap_loss
                + decode_loss * 4.0
                + direct_loss * 2.0
                + edge_loss * edge_loss_weight
                + edge_map_loss * edge_map_loss_weight
                + visibility_loss * visibility_loss_weight
            ) * sample_weight
            loss = weighted.mean() + gate_reg * 0.005
            err = torch.sqrt(torch.sum((pred - points) ** 2, dim=-1) + 1e-12)
            losses.extend(loss.detach().cpu().repeat(images.shape[0]).tolist())
            errors.extend(err.detach().cpu().tolist())
    arr = np.array(losses, dtype=np.float32) if losses else np.array([0.0], dtype=np.float32)
    err_arr = np.array(errors, dtype=np.float32) if errors else np.array([0.0], dtype=np.float32)
    return {"loss": float(arr.mean()), "point_error": float(err_arr.mean())}


def train_local_corner_moe_coord_model(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int = 35,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    input_size: int = 96,
    output_size: int = 24,
    channels: int = 24,
    experts: int = 4,
    metadata_dim: int = 14,
    input_channels: int = 10,
    coord_mix: float = 0.25,
    edge_loss_weight: float = 0.25,
    edge_map_loss_weight: float = 0.0,
    visibility_loss_weight: float = 0.15,
    allow_flip_augment: bool = True,
    flip_prob: float | None = None,
    seed: int = 7,
) -> LocalCornerMoECoordTrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = LocalCornerHeatmapDataset(
        dataset_dir / "train.jsonl",
        dataset_dir,
        input_size=input_size,
        output_size=output_size,
        augment=True,
        allow_flips=allow_flip_augment,
        flip_prob=flip_prob,
        input_channels=input_channels,
    )
    test_dataset = LocalCornerHeatmapDataset(
        dataset_dir / "test.jsonl",
        dataset_dir,
        input_size=input_size,
        output_size=output_size,
        augment=False,
        input_channels=input_channels,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = select_torch_device()
    model = LocalCornerMoECoordNet(
        channels=channels,
        experts=experts,
        metadata_dim=metadata_dim,
        input_channels=input_channels,
    ).to(device)
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
            targets = batch["heatmaps"].to(device=device, dtype=torch.float32)
            points = batch["target"].to(device=device, dtype=torch.float32)
            edge_target = batch["edge_target"].to(device=device, dtype=torch.float32)
            edge_maps_target = batch["edge_maps"].to(device=device, dtype=torch.float32)
            visibility_target = batch["visibility_target"].to(device=device, dtype=torch.float32)
            sample_weight = batch["sample_weight"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            heatmaps, offsets, coord_head, edge_head, edge_map_logits, visibility_pred, gates = model(images, metadata)
            pred = decode_moe_coord_output(heatmaps, offsets, coord_head, coord_mix=coord_mix)
            heatmap_loss = ((heatmaps - targets) ** 2).flatten(1).mean(dim=-1)
            decode_loss = F.smooth_l1_loss(pred, points, reduction="none").mean(dim=-1)
            direct_loss = F.smooth_l1_loss(coord_head, points, reduction="none").mean(dim=-1)
            edge_loss = F.smooth_l1_loss(edge_head, edge_target, reduction="none").mean(dim=-1)
            edge_map_pred = torch.sigmoid(edge_map_logits)
            edge_map_loss = ((edge_map_pred - edge_maps_target) ** 2).flatten(1).mean(dim=-1)
            visibility_loss = F.smooth_l1_loss(visibility_pred, visibility_target, reduction="none").mean(dim=-1)
            gate_reg = torch.mean(torch.sum(gates * torch.log(gates + 1e-8), dim=-1))
            loss = (
                (
                    heatmap_loss
                    + decode_loss * 4.0
                    + direct_loss * 2.0
                    + edge_loss * edge_loss_weight
                    + edge_map_loss * edge_map_loss_weight
                    + visibility_loss * visibility_loss_weight
                )
                * sample_weight
            ).mean() + gate_reg * 0.005
            loss.backward()
            optimizer.step()
        metrics = _evaluate(
            model,
            test_loader,
            device,
            coord_mix=coord_mix,
            edge_loss_weight=edge_loss_weight,
            edge_map_loss_weight=edge_map_loss_weight,
            visibility_loss_weight=visibility_loss_weight,
        )
        history.append({"epoch": epoch, **metrics})
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_metrics = metrics
    if best_state is None or best_metrics is None:
        raise RuntimeError("no checkpoint")
    model_path = output_dir / "local_corner_moe_coord_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_size": input_size,
            "output_size": output_size,
            "channels": channels,
            "experts": experts,
            "metadata_dim": metadata_dim,
            "input_channels": input_channels,
            "coord_mix": coord_mix,
            "edge_map_loss_weight": edge_map_loss_weight,
            "visibility_loss_weight": visibility_loss_weight,
            "allow_flip_augment": allow_flip_augment,
            "flip_prob": 0.0 if not allow_flip_augment else (0.5 if flip_prob is None else float(flip_prob)),
            "device": device.type,
        },
        model_path,
    )
    history_path = output_dir / "local_corner_moe_coord_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return LocalCornerMoECoordTrainResult(
        str(model_path),
        str(history_path),
        best_epoch,
        float(best_metrics["loss"]),
        float(best_metrics["point_error"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train local corner MoE coord refiner")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--output-size", type=int, default=24)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--metadata-dim", type=int, default=14)
    parser.add_argument("--input-channels", type=int, default=10)
    parser.add_argument("--coord-mix", type=float, default=0.25)
    parser.add_argument("--edge-loss-weight", type=float, default=0.25)
    parser.add_argument("--edge-map-loss-weight", type=float, default=0.0)
    parser.add_argument("--visibility-loss-weight", type=float, default=0.15)
    parser.add_argument("--disable-flip-augment", action="store_true")
    parser.add_argument("--flip-prob", type=float)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = train_local_corner_moe_coord_model(
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
        input_channels=args.input_channels,
        coord_mix=args.coord_mix,
        edge_loss_weight=args.edge_loss_weight,
        edge_map_loss_weight=args.edge_map_loss_weight,
        visibility_loss_weight=args.visibility_loss_weight,
        allow_flip_augment=not args.disable_flip_augment,
        flip_prob=args.flip_prob,
        seed=args.seed,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
