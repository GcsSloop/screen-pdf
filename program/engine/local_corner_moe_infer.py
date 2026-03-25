from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from corner_train import select_torch_device
from dataset_benchmark import normalized_point_error
from local_corner_heatmap import build_patch_metadata
from local_corner_moe import LocalCornerMoENet, decode_moe_output, remap_legacy_moe_state_dict
from local_corner_refine import build_local_corner_patch_sample, build_patch_features
from perspective_detect import order_points


def apply_moe_patch_points_to_quad(patch_samples: list[dict[str, Any]], point_norms: np.ndarray) -> list[list[float]]:
    points: list[list[float]] = []
    for sample, point_norm in zip(patch_samples, point_norms, strict=True):
        patch = sample["patch"]
        points.append(
            [
                float(patch["x"] + point_norm[0] * patch["size"]),
                float(patch["y"] + point_norm[1] * patch["size"]),
            ]
        )
    return [[float(x), float(y)] for x, y in order_points(np.array(points, dtype=np.float32))]


class LocalCornerMoEPredictor:
    def __init__(self, model_path: Path) -> None:
        self.device = select_torch_device()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.metadata_dim = int(checkpoint.get("metadata_dim", 0))
        self.model = LocalCornerMoENet(
            channels=int(checkpoint["channels"]),
            experts=int(checkpoint["experts"]),
            metadata_dim=self.metadata_dim,
        )
        self.model.load_state_dict(remap_legacy_moe_state_dict(checkpoint["state_dict"]), strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.input_size = int(checkpoint["input_size"])

    def __call__(self, sample: dict[str, Any], predicted_quad: np.ndarray) -> np.ndarray:
        features = build_patch_features(np.array(sample["patch_image"], copy=False), int(sample["corner_index"]), input_size=self.input_size)
        tensor = torch.from_numpy(features[None, ...]).to(device=self.device, dtype=torch.float32)
        metadata = None
        if self.metadata_dim > 0:
            metadata_row = {
                "corner_index": int(sample["corner_index"]),
                "patch": sample["patch"],
                "predicted_point": sample["predicted_point"],
                "predicted_quad": predicted_quad.tolist(),
            }
            metadata = torch.from_numpy(build_patch_metadata(metadata_row)[None, ...]).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            heatmaps, offsets, _ = self.model(tensor, metadata)
            point = decode_moe_output(heatmaps, offsets).cpu().numpy()[0]
        return point.astype(np.float32)


def evaluate_local_corner_moe(model_path: Path, dataset_dir: Path, split: str = "test") -> dict[str, Any]:
    predictor = LocalCornerMoEPredictor(model_path)
    rows = [json.loads(line) for line in (dataset_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['project_path']}::{row['page_id']}"].append(row)

    errors: list[float] = []
    page_times_ms: list[float] = []
    for entries in grouped.values():
        page_start = perf_counter()
        ordered = sorted(entries, key=lambda item: int(item["corner_index"]))
        predicted_quad = np.array(ordered[0]["predicted_quad"], dtype=np.float32)
        patch_samples: list[dict[str, Any]] = []
        point_norms: list[np.ndarray] = []
        for row in ordered:
            sample = build_local_corner_patch_sample(
                image_path=Path(row["image_path"]),
                page_id=str(row["page_id"]),
                corner_index=int(row["corner_index"]),
                predicted_quad=predicted_quad,
                manual_quad=predicted_quad,
                patch_size=int(row["patch"]["size"]),
                bottom_vertical_bias=float(row.get("patch", {}).get("bottom_vertical_bias", 0.0)),
            )
            patch_samples.append(sample)
            point_norms.append(predictor(sample, predicted_quad))
        refined = apply_moe_patch_points_to_quad(patch_samples, np.stack(point_norms, axis=0))
        errors.append(float(normalized_point_error(ordered[0]["manual_quad"], refined)))
        page_times_ms.append((perf_counter() - page_start) * 1000.0)
    arr = np.array(errors, dtype=np.float32)
    time_arr = np.array(page_times_ms, dtype=np.float32) if page_times_ms else np.array([0.0], dtype=np.float32)
    return {
        "pages": len(grouped),
        "device": predictor.device.type,
        "point_error_mean": round(float(arr.mean()), 4) if len(arr) else 0.0,
        "point_error_p95": round(float(np.percentile(arr, 95)), 4) if len(arr) else 0.0,
        "point_le_0_05_ratio": round(float((arr <= 0.05).mean()), 4) if len(arr) else 0.0,
        "point_le_0_03_ratio": round(float((arr <= 0.03).mean()), 4) if len(arr) else 0.0,
        "point_le_0_02_ratio": round(float((arr <= 0.02).mean()), 4) if len(arr) else 0.0,
        "point_le_0_01_ratio": round(float((arr <= 0.01).mean()), 4) if len(arr) else 0.0,
        "avg_page_infer_ms": round(float(time_arr.mean()), 2),
        "avg_corner_infer_ms": round(float(time_arr.mean() / 4.0), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local corner MoE refiner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    result = evaluate_local_corner_moe(Path(args.model), Path(args.dataset_dir), split=args.split)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
