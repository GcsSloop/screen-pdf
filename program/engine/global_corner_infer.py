from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from corner_train import (
    CornerHeatmapNet,
    decode_model_output,
    remap_legacy_head_state_dict,
    select_torch_device,
)
from dataset_benchmark import quad_geometry_metrics, summarize_geometry_metric_rows
from global_corner_train import build_global_feature_tensor, denormalize_corners


def _load_model(model_path: Path, device: torch.device) -> tuple[CornerHeatmapNet, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    input_channels = int(checkpoint.get("input_channels", 3) or 3)
    head_mode = str(checkpoint.get("head_mode", "heatmap"))
    model = CornerHeatmapNet(
        in_channels=input_channels,
        channels=int(checkpoint["channels"]),
        output_channels=4,
        head_mode=head_mode,
    )
    model.load_state_dict(remap_legacy_head_state_dict(checkpoint["state_dict"]))
    model.to(device)
    model.eval()
    return model, checkpoint


def _decode_from_checkpoint(heatmaps: torch.Tensor, checkpoint: dict[str, Any]) -> torch.Tensor:
    return decode_model_output(
        heatmaps,
        decode_mode=str(checkpoint.get("decode_mode", "argmax")),
        head_mode=str(checkpoint.get("head_mode", "heatmap")),
    )


def _load_image_tensor(
    image_path: Path,
    input_size: int,
    *,
    feature_mode: str = "rgb",
) -> tuple[torch.Tensor, tuple[int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(build_global_feature_tensor(image, feature_mode=feature_mode)[None, ...])
    return tensor, (width, height)


def evaluate_global_split(model_path: Path, split_dir: Path, split: str = "test") -> dict[str, Any]:
    device = select_torch_device()
    model, checkpoint = _load_model(model_path, device)
    feature_mode = str(checkpoint.get("feature_mode", "rgb"))
    rows = [json.loads(line) for line in (split_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metric_rows: list[dict[str, float]] = []
    times: list[float] = []
    for row in rows:
        tensor, (width, height) = _load_image_tensor(
            Path(row["image_path"]),
            int(checkpoint["input_size"]),
            feature_mode=feature_mode,
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            heatmaps = model(tensor.to(device=device, dtype=torch.float32))
            pred_norm = _decode_from_checkpoint(heatmaps, checkpoint).cpu().numpy()[0]
        times.append((time.perf_counter() - t0) * 1000.0)
        pred = denormalize_corners(pred_norm, width, height)
        metric_rows.append(quad_geometry_metrics(row["manual_quad"], pred))
    summary = summarize_geometry_metric_rows(metric_rows)
    return {
        "pages": len(rows),
        "device": device.type,
        "decode_mode": str(checkpoint.get("decode_mode", "argmax")),
        **summary,
        "mean_infer_ms": round(float(np.mean(times)), 2) if len(times) else 0.0,
        "p95_infer_ms": round(float(np.percentile(times, 95)), 2) if len(times) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate model-dominant global corner detector")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    result = evaluate_global_split(Path(args.model), Path(args.split_dir), split=args.split)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
