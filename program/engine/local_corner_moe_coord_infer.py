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
from dataset_benchmark import quad_geometry_metrics
from local_corner_refine import build_local_corner_patch_sample, build_patch_features
from local_corner_moe_coord import LocalCornerMoECoordNet, decode_moe_coord_output
from local_corner_heatmap import build_patch_metadata
from perspective_detect import order_points


def apply_moe_coord_patch_points_to_quad(patch_samples: list[dict[str, Any]], point_norms: np.ndarray) -> list[list[float]]:
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


def blend_structure_aware_point(
    decoded_point: torch.Tensor,
    coord_head: torch.Tensor,
    visibility: torch.Tensor,
    base_coord_mix: float = 0.25,
    use_visibility: bool = True,
) -> torch.Tensor:
    if not use_visibility:
        return torch.clamp(decoded_point * (1.0 - base_coord_mix) + coord_head * base_coord_mix, 0.0, 1.0)
    visibility_score = torch.clamp(visibility.mean(dim=-1, keepdim=True), 0.0, 1.0)
    adaptive_mix = torch.clamp(base_coord_mix * 0.35 + visibility_score * 0.65, 0.05, 0.9)
    return torch.clamp(decoded_point * (1.0 - adaptive_mix) + coord_head * adaptive_mix, 0.0, 1.0)


class LocalCornerMoECoordPredictor:
    def __init__(self, model_path: Path) -> None:
        self.device = select_torch_device()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.input_size = int(checkpoint["input_size"])
        self.coord_mix = float(checkpoint.get("coord_mix", 0.25))
        self.metadata_dim = int(checkpoint.get("metadata_dim", 14))
        self.input_channels = int(checkpoint.get("input_channels", 10))
        self.use_visibility = any(str(key).startswith("visibility_heads.") for key in checkpoint["state_dict"].keys())
        self.model = LocalCornerMoECoordNet(
            channels=int(checkpoint["channels"]),
            experts=int(checkpoint["experts"]),
            metadata_dim=self.metadata_dim,
            input_channels=self.input_channels,
        )
        missing, unexpected = self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        allowed_missing = [
            key
            for key in missing
            if key.startswith("edge_heads.") or key.startswith("edgemap_heads.") or key.startswith("visibility_heads.")
        ]
        if unexpected or len(allowed_missing) != len(missing):
            raise RuntimeError(
                f"incompatible local_corner_moe_coord checkpoint: missing={missing}, unexpected={unexpected}"
            )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, sample: dict[str, Any], predicted_quad: np.ndarray) -> np.ndarray:
        features = build_patch_features(
            np.array(sample["patch_image"], copy=False),
            int(sample["corner_index"]),
            input_size=self.input_size,
            input_channels=self.input_channels,
        )
        tensor = torch.from_numpy(features[None, ...]).to(device=self.device, dtype=torch.float32)
        metadata_row = {
            "corner_index": int(sample["corner_index"]),
            "patch": sample["patch"],
            "predicted_point": sample["predicted_point"],
            "predicted_quad": predicted_quad.tolist(),
        }
        metadata = torch.from_numpy(build_patch_metadata(metadata_row)[None, ...]).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            heatmaps, offsets, coord_head, _, _, visibility, _ = self.model(tensor, metadata)
            decoded = decode_moe_coord_output(heatmaps, offsets, coord_head, coord_mix=0.0)
            point = blend_structure_aware_point(
                decoded,
                coord_head,
                visibility,
                base_coord_mix=self.coord_mix,
                use_visibility=self.use_visibility,
            ).cpu().numpy()[0]
        return point.astype(np.float32)


def evaluate_local_corner_moe_coord(model_path: Path, dataset_dir: Path, split: str = "test") -> dict[str, Any]:
    predictor = LocalCornerMoECoordPredictor(model_path)
    rows = [json.loads(line) for line in (dataset_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['project_path']}::{row['page_id']}"].append(row)
    errors: list[float] = []
    screen_relative_errors: list[float] = []
    max_corner_errors: list[float] = []
    bl_corner_errors: list[float] = []
    all_corners_le_0_01_hits: list[float] = []
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
        refined = apply_moe_coord_patch_points_to_quad(patch_samples, np.stack(point_norms, axis=0))
        metrics = quad_geometry_metrics(ordered[0]["manual_quad"], refined)
        errors.append(float(metrics["point_error"]))
        screen_relative_errors.append(float(metrics["screen_relative_point_error"]))
        max_corner_errors.append(float(metrics["max_corner_error"]))
        manual = order_points(np.array(ordered[0]["manual_quad"], dtype=np.float32))
        pred = order_points(np.array(refined, dtype=np.float32))
        bbox = np.ptp(manual, axis=0)
        diag = float(np.hypot(max(float(bbox[0]), 1.0), max(float(bbox[1]), 1.0)))
        distances = np.linalg.norm(pred - manual, axis=1) / max(diag, 1e-6)
        bl_corner_errors.append(float(distances[3]))
        all_corners_le_0_01_hits.append(float(np.all(distances <= 0.01)))
        page_times_ms.append((perf_counter() - page_start) * 1000.0)
    arr = np.array(errors, dtype=np.float32)
    screen_arr = np.array(screen_relative_errors, dtype=np.float32) if screen_relative_errors else np.array([0.0], dtype=np.float32)
    max_corner_arr = np.array(max_corner_errors, dtype=np.float32) if max_corner_errors else np.array([0.0], dtype=np.float32)
    bl_arr = np.array(bl_corner_errors, dtype=np.float32) if bl_corner_errors else np.array([0.0], dtype=np.float32)
    hit_arr = np.array(all_corners_le_0_01_hits, dtype=np.float32) if all_corners_le_0_01_hits else np.array([0.0], dtype=np.float32)
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
        "screen_relative_point_error_mean": round(float(screen_arr.mean()), 4),
        "max_corner_error_mean": round(float(max_corner_arr.mean()), 4),
        "bl_corner_error_mean": round(float(bl_arr.mean()), 4),
        "all_corners_le_0_01_ratio": round(float(hit_arr.mean()), 4),
        "avg_page_infer_ms": round(float(time_arr.mean()), 2),
        "avg_corner_infer_ms": round(float(time_arr.mean() / 4.0), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local corner MoE coord refiner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    result = evaluate_local_corner_moe_coord(Path(args.model), Path(args.dataset_dir), split=args.split)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
