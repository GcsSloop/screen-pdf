from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dataset_benchmark import build_split, load_manual_pages
from perspective_detect import detect_best_candidate_with_profile, load_opencv_profile, order_points


PageRecord = dict[str, Any]


def _clamp_rect(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(x0 + 1, min(width, x1))
    y1 = max(y0 + 1, min(height, y1))
    return x0, y0, x1, y1


def _build_roi_from_quad(
    image_shape: tuple[int, int, int] | tuple[int, int],
    quad: np.ndarray,
    expand_ratio: float = 0.08,
) -> dict[str, int]:
    height, width = image_shape[:2]
    ordered = order_points(quad)
    min_xy = np.floor(np.min(ordered, axis=0)).astype(np.int32)
    max_xy = np.ceil(np.max(ordered, axis=0)).astype(np.int32)
    span = np.maximum(max_xy - min_xy, 1)
    expand = np.maximum((span.astype(np.float32) * expand_ratio).round().astype(np.int32), 12)
    x0, y0 = min_xy - expand
    x1, y1 = max_xy + expand
    x0, y0, x1, y1 = _clamp_rect(int(x0), int(y0), int(x1), int(y1), width, height)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _normalize_quad_to_roi(quad: np.ndarray, roi: dict[str, int]) -> list[list[float]]:
    width = max(float(roi["width"]), 1.0)
    height = max(float(roi["height"]), 1.0)
    out: list[list[float]] = []
    for x, y in order_points(quad):
        out.append(
            [
                round(float((x - roi["x"]) / width), 6),
                round(float((y - roi["y"]) / height), 6),
            ]
        )
    return out


def _normalize_quad_to_image(quad: np.ndarray, image_shape: tuple[int, int, int] | tuple[int, int]) -> list[list[float]]:
    height, width = image_shape[:2]
    out: list[list[float]] = []
    for x, y in order_points(quad):
        out.append(
            [
                round(float(x / max(width, 1)), 6),
                round(float(y / max(height, 1)), 6),
            ]
        )
    return out


def build_infer_request(
    page_id: str,
    image_path: str,
    image_shape: tuple[int, int, int] | tuple[int, int],
    coarse_quad: np.ndarray,
) -> dict[str, Any]:
    coarse = order_points(np.array(coarse_quad, dtype=np.float32))
    roi = _build_roi_from_quad(image_shape, coarse)
    return {
        "page_id": page_id,
        "image_path": image_path,
        "image_size": {
            "width": int(image_shape[1]),
            "height": int(image_shape[0]),
        },
        "roi": roi,
        "coarse_quad": [[float(x), float(y)] for x, y in coarse],
        "coarse_quad_norm": _normalize_quad_to_image(coarse, image_shape),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _sample_from_page(
    page: PageRecord,
    split_name: str,
    output_dir: Path,
    opencv_profile: dict[str, float] | None,
) -> dict[str, Any] | None:
    image = cv2.imread(page["image_path"])
    if image is None:
        return None
    manual_quad = order_points(np.array(page["manual_quad"], dtype=np.float32))
    result = detect_best_candidate_with_profile(image, opencv_profile)
    coarse_quad = manual_quad if result is None else order_points(np.array(result["best"]["quad"], dtype=np.float32))
    roi = _build_roi_from_quad(image.shape, coarse_quad)
    x0, y0 = roi["x"], roi["y"]
    x1, y1 = x0 + roi["width"], y0 + roi["height"]
    roi_image = image[y0:y1, x0:x1]
    roi_rel_dir = Path("roi") / split_name
    roi_rel_path = roi_rel_dir / f"{page['project_name']}-{page['page_id']}.png"
    roi_abs_path = output_dir / roi_rel_path
    roi_abs_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(roi_abs_path), roi_image)
    return {
        "split": split_name,
        "project_name": page["project_name"],
        "project_path": page["project_path"],
        "page_id": page["page_id"],
        "image_path": page["image_path"],
        "image_size": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "roi": roi,
        "roi_path": roi_rel_path.as_posix(),
        "manual_quad": [[float(x), float(y)] for x, y in manual_quad],
        "coarse_quad": [[float(x), float(y)] for x, y in coarse_quad],
        "corner_norm": _normalize_quad_to_roi(manual_quad, roi),
        "coarse_quad_norm": _normalize_quad_to_roi(coarse_quad, roi),
    }


def export_corner_dataset(
    dataset_root: Path,
    output_dir: Path,
    seed: int = 7,
    test_ratio: float = 0.25,
    opencv_profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    pages = load_manual_pages(dataset_root)
    split = build_split(pages, test_ratio=test_ratio, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_profile = opencv_profile or load_opencv_profile()

    exported: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for split_name in ("train", "test"):
        for page in split[split_name]:
            sample = _sample_from_page(page, split_name, output_dir, active_profile)
            if sample is not None:
                exported[split_name].append(sample)

    _write_jsonl(output_dir / "train.jsonl", exported["train"])
    _write_jsonl(output_dir / "test.jsonl", exported["test"])
    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "seed": seed,
        "test_ratio": test_ratio,
        "pages": len(exported["train"]) + len(exported["test"]),
        "train_pages": len(exported["train"]),
        "test_pages": len(exported["test"]),
        "schema": {
            "roi_path": "cropped ROI image relative path",
            "corner_norm": "manual quad normalized to ROI, TL/TR/BR/BL order",
            "coarse_quad_norm": "detector quad normalized to ROI, TL/TR/BR/BL order",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_training_blueprint(dataset_dir: Path, output_path: Path) -> dict[str, Any]:
    blueprint = {
        "dataset_dir": str(dataset_dir),
        "recommended_model": "lightweight heatmap corner refiner",
        "backbone": "mobilenetv3-small or equivalent",
        "input_size": [512, 512],
        "output_heads": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "loss": {
            "heatmap": "focal or mse",
            "offset": "smooth_l1",
        },
        "export_target": "onnx",
        "runtime_target": "onnxruntime or bundled native runtime",
    }
    output_path.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8")
    return blueprint


def main() -> int:
    parser = argparse.ArgumentParser(description="Corner model dataset and pipeline helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export corner-model training dataset")
    export_parser.add_argument("--dataset-root", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--seed", type=int, default=7)
    export_parser.add_argument("--test-ratio", type=float, default=0.25)

    plan_parser = subparsers.add_parser("train-plan", help="Write model training blueprint")
    plan_parser.add_argument("--dataset-dir", required=True)
    plan_parser.add_argument("--output", required=True)

    infer_parser = subparsers.add_parser("infer-request", help="Build inference request from detector coarse quad")
    infer_parser.add_argument("--image", required=True)
    infer_parser.add_argument("--page-id", required=True)

    args = parser.parse_args()
    if args.command == "export":
        report = export_corner_dataset(Path(args.dataset_root), Path(args.output_dir), seed=args.seed, test_ratio=args.test_ratio)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command == "train-plan":
        blueprint = write_training_blueprint(Path(args.dataset_dir), Path(args.output))
        print(json.dumps(blueprint, indent=2, ensure_ascii=False))
        return 0

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"failed to read image: {args.image}")
    result = detect_best_candidate_with_profile(image, load_opencv_profile())
    if result is None:
        raise SystemExit("failed to build coarse quad from current detector")
    request = build_infer_request(args.page_id, args.image, image.shape, np.array(result["best"]["quad"], dtype=np.float32))
    print(json.dumps(request, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
