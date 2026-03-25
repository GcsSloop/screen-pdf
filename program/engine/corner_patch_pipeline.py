from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from perspective_detect import order_points
from two_stage_corner_pipeline import GlobalCornerPredictor, RoiCornerPredictor, predict_two_stage


def denormalize_point(point_norm: list[float], patch: dict[str, int]) -> list[float]:
    return [
        float(patch["x"] + point_norm[0] * patch["size"]),
        float(patch["y"] + point_norm[1] * patch["size"]),
    ]


def _clip_square(x: int, y: int, size: int, width: int, height: int) -> tuple[int, int, int]:
    size = max(16, min(size, width, height))
    x = max(0, min(width - size, x))
    y = max(0, min(height - size, y))
    return x, y, size


def build_corner_patch_sample(
    image_path: Path,
    page_id: str,
    corner_index: int,
    predicted_quad: np.ndarray,
    manual_quad: np.ndarray,
    patch_scale: float = 0.2,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    predicted = order_points(np.array(predicted_quad, dtype=np.float32))
    manual = order_points(np.array(manual_quad, dtype=np.float32))
    bbox = np.ptp(predicted, axis=0)
    base = max(float(min(bbox[0], bbox[1])) * patch_scale, 40.0)
    size = int(round(base))
    cx, cy = predicted[corner_index]
    x = int(round(float(cx) - size / 2))
    y = int(round(float(cy) - size / 2))
    x, y, size = _clip_square(x, y, size, image.shape[1], image.shape[0])
    patch_image = image[y : y + size, x : x + size].copy()
    target = manual[corner_index]
    target_norm = [
        round(float((target[0] - x) / size), 6),
        round(float((target[1] - y) / size), 6),
    ]
    return {
        "page_id": page_id,
        "corner_index": corner_index,
        "patch": {"x": x, "y": y, "size": size},
        "target_norm": target_norm,
        "patch_image": patch_image,
    }


def export_corner_patch_dataset(
    global_model_path: Path,
    roi_model_path: Path,
    split_dir: Path,
    output_dir: Path,
    patch_scale: float = 0.2,
) -> dict[str, Any]:
    global_predictor = GlobalCornerPredictor(global_model_path)
    roi_predictor = RoiCornerPredictor(roi_model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "global_model_path": str(global_model_path),
        "roi_model_path": str(roi_model_path),
        "split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "patch_scale": patch_scale,
        "train_samples": 0,
        "test_samples": 0,
    }
    for split in ("train", "test"):
        rows = [json.loads(line) for line in (split_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        exported: list[dict[str, Any]] = []
        patch_dir = output_dir / "patches" / split
        patch_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            image_path = Path(row["image_path"])
            pred = predict_two_stage(image_path, global_predictor, roi_predictor, page_id=str(row.get("page_id") or image_path.stem))
            predicted_quad = np.array(pred["final_quad"], dtype=np.float32)
            manual_quad = np.array(row["manual_quad"], dtype=np.float32)
            for corner_index in range(4):
                sample = build_corner_patch_sample(
                    image_path=image_path,
                    page_id=str(row.get("page_id") or image_path.stem),
                    corner_index=corner_index,
                    predicted_quad=predicted_quad,
                    manual_quad=manual_quad,
                    patch_scale=patch_scale,
                )
                patch_path = patch_dir / f"{sample['page_id']}-c{corner_index}.png"
                cv2.imwrite(str(patch_path), sample["patch_image"])
                exported.append(
                    {
                        "page_id": sample["page_id"],
                        "corner_index": corner_index,
                        "patch_path": str(Path("patches") / split / patch_path.name),
                        "patch": sample["patch"],
                        "target_norm": sample["target_norm"],
                        "predicted_quad": pred["final_quad"],
                        "manual_quad": row["manual_quad"],
                        "image_path": str(image_path),
                    }
                )
        text = "\n".join(json.dumps(item, ensure_ascii=False) for item in exported)
        (output_dir / f"{split}.jsonl").write_text(text + ("\n" if text else ""), encoding="utf-8")
        summary[f"{split}_samples"] = len(exported)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export corner patch refine dataset")
    parser.add_argument("--global-model", required=True)
    parser.add_argument("--roi-model", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-scale", type=float, default=0.2)
    args = parser.parse_args()
    result = export_corner_patch_dataset(
        global_model_path=Path(args.global_model),
        roi_model_path=Path(args.roi_model),
        split_dir=Path(args.split_dir),
        output_dir=Path(args.output_dir),
        patch_scale=float(args.patch_scale),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
