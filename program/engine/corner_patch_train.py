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

from corner_train import select_torch_device


class CornerPatchDataset(Dataset):
    def __init__(self, manifest_path: Path, dataset_root: Path, input_size: int = 64, augment: bool = False) -> None:
        self.dataset_root = dataset_root
        self.input_size = input_size
        self.augment = augment
        self.rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image = cv2.imread(str(self.dataset_root / row["patch_path"]))
        if image is None:
            raise FileNotFoundError(row["patch_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = np.array(row["target_norm"], dtype=np.float32)
        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)
            target[0] = 1.0 - target[0]
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        image_f = image.astype(np.float32) / 255.0
        corner_planes = np.zeros((4, self.input_size, self.input_size), dtype=np.float32)
        corner_planes[int(row["corner_index"]), :, :] = 1.0
        features = np.concatenate([np.transpose(image_f, (2, 0, 1)), corner_planes], axis=0)
        return {
            "image": torch.from_numpy(features),
            "target": torch.from_numpy(target),
        }


class CornerPatchNet(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(7, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4, channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 4, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@dataclass
class PatchTrainResult:
    model_path: str
    history_path: str
    best_epoch: int
    best_val_loss: float
    best_val_l1: float


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            target = batch["target"].to(device=device, dtype=torch.float32)
            pred = model(images)
            loss = F.smooth_l1_loss(pred, target)
            losses.extend(loss.detach().cpu().repeat(images.shape[0]).tolist())
    arr = np.array(losses, dtype=np.float32) if losses else np.array([0.0], dtype=np.float32)
    return {"loss": float(arr.mean()), "l1": float(arr.mean())}


def train_corner_patch_model(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    input_size: int = 64,
    channels: int = 16,
    seed: int = 7,
) -> PatchTrainResult:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CornerPatchDataset(dataset_dir / "train.jsonl", dataset_dir, input_size=input_size, augment=True)
    test_dataset = CornerPatchDataset(dataset_dir / "test.jsonl", dataset_dir, input_size=input_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = select_torch_device()
    model = CornerPatchNet(channels=channels).to(device)
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
            target = batch["target"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(images)
            loss = F.smooth_l1_loss(pred, target)
            loss.backward()
            optimizer.step()
        metrics = _evaluate(model, test_loader, device)
        history.append({"epoch": epoch, **metrics})
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = metrics
    if best_state is None or best_metrics is None:
        raise RuntimeError("no checkpoint")
    model_path = output_dir / "corner_patch_model.pt"
    torch.save({"state_dict": best_state, "input_size": input_size, "channels": channels, "device": device.type}, model_path)
    history_path = output_dir / "corner_patch_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return PatchTrainResult(str(model_path), str(history_path), best_epoch, float(best_metrics["loss"]), float(best_metrics["l1"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train corner patch refiner")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = train_corner_patch_model(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        channels=args.channels,
        seed=args.seed,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
