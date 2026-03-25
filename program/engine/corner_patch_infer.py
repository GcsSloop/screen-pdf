from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from corner_patch_pipeline import build_corner_patch_sample, denormalize_point
from corner_patch_train import CornerPatchNet
from corner_train import select_torch_device
from two_stage_corner_pipeline import GlobalCornerPredictor, RoiCornerPredictor, normalized_point_error, predict_two_stage


PatchPredictor = Callable[[dict[str, Any]], np.ndarray]


def refine_quad_with_patch_predictor(
    image_path: Path,
    predicted_quad: np.ndarray,
    patch_predictor: PatchPredictor,
    patch_scale: float = 0.2,
) -> list[list[float]]:
    manual_proxy = np.array(predicted_quad, dtype=np.float32)
    refined: list[list[float]] = []
    for corner_index in range(4):
        sample = build_corner_patch_sample(
            image_path=image_path,
            page_id=image_path.stem,
            corner_index=corner_index,
            predicted_quad=predicted_quad,
            manual_quad=manual_proxy,
            patch_scale=patch_scale,
        )
        point_norm = patch_predictor(sample)
        refined.append(denormalize_point(point_norm.tolist(), sample["patch"]))
    return refined


class CornerPatchPredictor:
    def __init__(self, model_path: Path) -> None:
        self.device = select_torch_device()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CornerPatchNet(channels=int(checkpoint["channels"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.input_size = int(checkpoint["input_size"])

    def __call__(self, sample: dict[str, Any]) -> np.ndarray:
        image = cv2.cvtColor(np.array(sample["patch_image"], copy=False), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        image_f = image.astype(np.float32) / 255.0
        corner_planes = np.zeros((4, self.input_size, self.input_size), dtype=np.float32)
        corner_planes[int(sample["corner_index"]), :, :] = 1.0
        features = np.concatenate([np.transpose(image_f, (2, 0, 1)), corner_planes], axis=0)
        tensor = torch.from_numpy(features[None, ...]).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(tensor).cpu().numpy()[0]
        return pred.astype(np.float32)


def evaluate_three_stage(
    global_model_path: Path,
    roi_model_path: Path,
    patch_model_path: Path,
    split_dir: Path,
    split: str = "test",
    patch_scale: float = 0.2,
) -> dict[str, Any]:
    global_predictor = GlobalCornerPredictor(global_model_path)
    roi_predictor = RoiCornerPredictor(roi_model_path)
    patch_predictor = CornerPatchPredictor(patch_model_path)
    rows = [json.loads(line) for line in (split_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[float] = []
    times: list[float] = []
    for row in rows:
        image_path = Path(row["image_path"])
        t0 = time.perf_counter()
        two_stage = predict_two_stage(image_path, global_predictor, roi_predictor, page_id=str(row.get("page_id") or image_path.stem))
        refined = refine_quad_with_patch_predictor(image_path, np.array(two_stage["final_quad"], dtype=np.float32), patch_predictor, patch_scale=patch_scale)
        times.append((time.perf_counter() - t0) * 1000.0)
        errors.append(float(normalized_point_error(row["manual_quad"], refined)))
    arr = np.array(errors, dtype=np.float32)
    return {
        "pages": len(rows),
        "point_error_mean": round(float(arr.mean()), 4) if len(arr) else 0.0,
        "point_error_p95": round(float(np.percentile(arr, 95)), 4) if len(arr) else 0.0,
        "point_le_0_05_ratio": round(float((arr <= 0.05).mean()), 4) if len(arr) else 0.0,
        "point_le_0_03_ratio": round(float((arr <= 0.03).mean()), 4) if len(arr) else 0.0,
        "point_le_0_02_ratio": round(float((arr <= 0.02).mean()), 4) if len(arr) else 0.0,
        "point_le_0_01_ratio": round(float((arr <= 0.01).mean()), 4) if len(arr) else 0.0,
        "mean_infer_ms": round(float(np.mean(times)), 2) if len(times) else 0.0,
        "p95_infer_ms": round(float(np.percentile(times, 95)), 2) if len(times) else 0.0,
        "device": patch_predictor.device.type,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer/evaluate corner patch refiner")
    parser.add_argument("--global-model", required=True)
    parser.add_argument("--roi-model", required=True)
    parser.add_argument("--patch-model", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--patch-scale", type=float, default=0.2)
    args = parser.parse_args()
    result = evaluate_three_stage(
        global_model_path=Path(args.global_model),
        roi_model_path=Path(args.roi_model),
        patch_model_path=Path(args.patch_model),
        split_dir=Path(args.split_dir),
        split=args.split,
        patch_scale=float(args.patch_scale),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
