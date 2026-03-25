from __future__ import annotations

import argparse
import json
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


def denormalize_corners_to_image(corners: np.ndarray, roi: dict[str, int]) -> np.ndarray:
    points = np.array(corners, dtype=np.float32)
    restored = np.empty_like(points)
    restored[:, 0] = roi["x"] + points[:, 0] * roi["width"]
    restored[:, 1] = roi["y"] + points[:, 1] * roi["height"]
    return restored.astype(np.float32)


def _load_model(model_path: Path, device: torch.device) -> tuple[CornerHeatmapNet, dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)
    model = CornerHeatmapNet(
        in_channels=4,
        channels=int(checkpoint["channels"]),
        output_channels=4,
        head_mode=str(checkpoint.get("head_mode", "heatmap")),
    )
    model.load_state_dict(remap_legacy_head_state_dict(checkpoint["state_dict"]))
    model.to(device)
    model.eval()
    return model, checkpoint


def _decode_from_checkpoint(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor], checkpoint: dict[str, Any]) -> torch.Tensor:
    return decode_model_output(
        output,
        decode_mode=str(checkpoint.get("decode_mode", "argmax")),
        head_mode=str(checkpoint.get("head_mode", "heatmap")),
    )


def _load_input_tensor(image_path: Path, coarse_quad_norm: list[list[float]], input_size: int) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image_f = image.astype(np.float32) / 255.0
    mask = np.zeros((input_size, input_size), dtype=np.float32)
    pts = np.round(np.array(coarse_quad_norm, dtype=np.float32) * (input_size - 1)).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 1.0)
    features = np.concatenate([np.transpose(image_f, (2, 0, 1)), mask[None, ...]], axis=0)
    return torch.from_numpy(features[None, ...])


def predict_corners(model_path: Path, sample: dict[str, Any]) -> dict[str, Any]:
    device = select_torch_device()
    model, checkpoint = _load_model(model_path, device)
    tensor = _load_input_tensor(Path(sample["roi_path_abs"]), sample["coarse_quad_norm"], int(checkpoint["input_size"]))
    with torch.no_grad():
        output = model(tensor.to(device=device, dtype=torch.float32))
        pred_norm = _decode_from_checkpoint(output, checkpoint).cpu().numpy()[0]
    pred_image = denormalize_corners_to_image(pred_norm, sample["roi"])
    return {
        "pred_norm": pred_norm.tolist(),
        "pred_image": pred_image.tolist(),
    }


def evaluate_manifest(model_path: Path, dataset_dir: Path, split: str = "test") -> dict[str, Any]:
    device = select_torch_device()
    model, checkpoint = _load_model(model_path, device)
    manifest_path = dataset_dir / f"{split}.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metric_rows: list[dict[str, float]] = []
    roi_errors: list[float] = []
    for row in rows:
        tensor = _load_input_tensor(dataset_dir / row["roi_path"], row["coarse_quad_norm"], int(checkpoint["input_size"]))
        with torch.no_grad():
            output = model(tensor.to(device=device, dtype=torch.float32))
            pred_norm = _decode_from_checkpoint(output, checkpoint).cpu().numpy()[0]
        pred_image = denormalize_corners_to_image(pred_norm, row["roi"])
        target_norm = np.array(row["corner_norm"], dtype=np.float32)
        metric_rows.append(quad_geometry_metrics(row["manual_quad"], pred_image))
        roi_errors.append(float(np.linalg.norm(pred_norm - target_norm, axis=1).mean()))
    roi_arr = np.array(roi_errors, dtype=np.float32)
    summary = summarize_geometry_metric_rows(metric_rows)
    return {
        "pages": len(rows),
        "device": device.type,
        "decode_mode": str(checkpoint.get("decode_mode", "argmax")),
        "head_mode": str(checkpoint.get("head_mode", "heatmap")),
        **summary,
        "roi_error_mean": round(float(roi_arr.mean()), 4) if len(roi_arr) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer/evaluate corner heatmap model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--model", required=True)
    eval_parser.add_argument("--dataset-dir", required=True)
    eval_parser.add_argument("--split", default="test")

    args = parser.parse_args()
    result = evaluate_manifest(Path(args.model), Path(args.dataset_dir), split=args.split)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
