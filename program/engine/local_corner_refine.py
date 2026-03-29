from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from perspective_detect import order_points
from supervision_utils import resolve_supervision_quad


PageRecord = dict[str, Any]
ResidualPredictor = Callable[[dict[str, Any]], np.ndarray]


def _resolve_image_path(project_path: Path, page: dict[str, Any]) -> Path | None:
    raw_path = page.get("path")
    if not raw_path:
        return None
    image_path = Path(raw_path)
    candidates = [
        image_path,
        project_path.parent / raw_path,
        project_path.parent / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_active_manual_pages(dataset_root: Path) -> list[PageRecord]:
    pages: list[PageRecord] = []
    for project_path in sorted(dataset_root.rglob("screen-pdf-project.json")):
        data = json.loads(project_path.read_text(encoding="utf-8"))
        for page in data.get("pages", []):
            manual_quad, _ = resolve_supervision_quad(page)
            active_quad = page.get("activeQuad")
            if not manual_quad or not active_quad:
                continue
            image_path = _resolve_image_path(project_path, page)
            if image_path is None:
                continue
            pages.append(
                {
                    "project_path": str(project_path),
                    "project_name": project_path.parent.name,
                    "page_id": page.get("id") or image_path.stem,
                    "image_path": str(image_path),
                    "manual_quad": manual_quad,
                    "active_quad": active_quad,
                }
            )
    return pages


def _clip_square(x: int, y: int, size: int, width: int, height: int) -> tuple[int, int, int]:
    size = max(24, min(size, width, height))
    x = max(0, min(width - size, x))
    y = max(0, min(height - size, y))
    return x, y, size


def build_local_corner_patch_sample(
    image_path: Path,
    page_id: str,
    corner_index: int,
    predicted_quad: np.ndarray,
    manual_quad: np.ndarray,
    patch_size: int | None = 96,
    patch_scale: float = 0.2,
    patch_min: int = 96,
    patch_max: int = 256,
    bottom_vertical_bias: float = 0.0,
    bl_patch_scale_multiplier: float = 1.0,
    bl_bottom_vertical_bias: float = 0.0,
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    if image is None:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
    predicted = order_points(np.array(predicted_quad, dtype=np.float32))
    manual = order_points(np.array(manual_quad, dtype=np.float32))
    resolved_patch_size = patch_size
    if resolved_patch_size is None:
        bbox = np.ptp(predicted, axis=0)
        resolved_patch_size = int(round(float(min(bbox[0], bbox[1])) * patch_scale))
        if corner_index == 3:
            resolved_patch_size = int(round(float(resolved_patch_size) * max(bl_patch_scale_multiplier, 1.0)))
        resolved_patch_size = int(np.clip(resolved_patch_size, patch_min, patch_max))
    else:
        resolved_patch_size = int(resolved_patch_size)
    cx, cy = predicted[corner_index]
    x = int(round(float(cx) - resolved_patch_size / 2))
    y = int(round(float(cy) - resolved_patch_size / 2))
    effective_bottom_vertical_bias = float(bottom_vertical_bias)
    if corner_index == 3 and bl_bottom_vertical_bias > 0.0:
        effective_bottom_vertical_bias += float(bl_bottom_vertical_bias)
    if corner_index in {2, 3} and effective_bottom_vertical_bias > 0.0:
        y -= int(round(resolved_patch_size * effective_bottom_vertical_bias))
    x, y, resolved_patch_size = _clip_square(x, y, resolved_patch_size, image.shape[1], image.shape[0])
    patch_image = image[y : y + resolved_patch_size, x : x + resolved_patch_size].copy()
    residual = (manual[corner_index] - predicted[corner_index]) / float(resolved_patch_size)
    residual = np.clip(residual, -1.0, 1.0)
    return {
        "page_id": page_id,
        "corner_index": corner_index,
        "patch": {"x": x, "y": y, "size": resolved_patch_size, "bottom_vertical_bias": effective_bottom_vertical_bias},
        "predicted_point": [float(predicted[corner_index][0]), float(predicted[corner_index][1])],
        "target_residual_norm": [float(residual[0]), float(residual[1])],
        "patch_image": patch_image,
    }


def build_patch_features(
    patch_image: np.ndarray,
    corner_index: int,
    input_size: int = 96,
    input_channels: int = 10,
) -> np.ndarray:
    image = cv2.cvtColor(patch_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    rgb = image.astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sobel_x, sobel_y)
    grad = np.clip(grad, 0.0, 1.0)
    edges = cv2.Canny(gray, 80, 180).astype(np.float32) / 255.0
    ys = np.linspace(-1.0, 1.0, input_size, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, input_size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    corner_plane = np.zeros((input_size, input_size), dtype=np.float32)
    corner_plane[:, :] = float(corner_index) / 3.0 if corner_index > 0 else 1.0
    features = [
        np.transpose(rgb, (2, 0, 1)),
        grad[None, ...],
        sobel_x[None, ...],
        sobel_y[None, ...],
        edges[None, ...],
        grid_x[None, ...],
        grid_y[None, ...],
        corner_plane[None, ...],
    ]
    if input_channels >= 13:
        edge_mask = (edges > 0.1).astype(np.uint8) * 255
        line_mask = np.zeros_like(gray_f, dtype=np.float32)
        lines = cv2.HoughLinesP(edge_mask, 1, np.pi / 180.0, threshold=14, minLineLength=max(8, input_size // 8), maxLineGap=4)
        if lines is not None:
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = [int(v) for v in line]
                cv2.line(line_mask, (x1, y1), (x2, y2), 1.0, 1, lineType=cv2.LINE_AA)
            line_mask = np.clip(line_mask, 0.0, 1.0)
        inv_edges = np.where(edge_mask > 0, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 3)
        scale = max(float(input_size) * 0.12, 1.0)
        edge_proximity = np.exp(-dist / scale).astype(np.float32)
        harris = cv2.cornerHarris(gray_f, 2, 3, 0.04)
        harris = np.maximum(harris, 0.0)
        if float(harris.max()) > 1e-6:
            harris = harris / float(harris.max())
        features.extend([line_mask[None, ...], edge_proximity[None, ...], harris[None, ...]])
    return np.concatenate(
        [
            *features,
        ],
        axis=0,
    ).astype(np.float32)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def export_local_corner_patch_dataset_from_splits(
    split_pages: dict[str, list[PageRecord]],
    output_dir: Path,
    predicted_quad_getter: Callable[[PageRecord], np.ndarray] | None = None,
    dataset_root: Path | None = None,
    patch_size: int | None = 96,
    patch_scale: float = 0.2,
    patch_min: int = 96,
    patch_max: int = 256,
    bottom_vertical_bias: float = 0.0,
    bl_patch_scale_multiplier: float = 1.0,
    bl_bottom_vertical_bias: float = 0.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, list[dict[str, Any]]] = {name: [] for name in split_pages}
    for split_name, split_rows in split_pages.items():
        patch_dir = output_dir / "patches" / split_name
        patch_dir.mkdir(parents=True, exist_ok=True)
        for page in split_rows:
            predicted_quad = np.array(
                predicted_quad_getter(page) if predicted_quad_getter is not None else page["active_quad"],
                dtype=np.float32,
            )
            manual_quad = np.array(page["manual_quad"], dtype=np.float32)
            for corner_index in range(4):
                sample = build_local_corner_patch_sample(
                    image_path=Path(page["image_path"]),
                    page_id=str(page["page_id"]),
                    corner_index=corner_index,
                    predicted_quad=predicted_quad,
                    manual_quad=manual_quad,
                    patch_size=patch_size,
                    patch_scale=patch_scale,
                    patch_min=patch_min,
                    patch_max=patch_max,
                    bottom_vertical_bias=bottom_vertical_bias,
                    bl_patch_scale_multiplier=bl_patch_scale_multiplier,
                    bl_bottom_vertical_bias=bl_bottom_vertical_bias,
                )
                patch_name = f"{page['project_name']}-{page['page_id']}-c{corner_index}.png"
                patch_path = patch_dir / patch_name
                cv2.imwrite(str(patch_path), sample["patch_image"])
                exported[split_name].append(
                    {
                        "project_name": page["project_name"],
                        "project_path": page["project_path"],
                        "page_id": page["page_id"],
                        "corner_index": corner_index,
                        "patch_path": str(Path("patches") / split_name / patch_name),
                        "patch": sample["patch"],
                        "predicted_point": sample["predicted_point"],
                        "target_residual_norm": sample["target_residual_norm"],
                        "target_point_norm": [
                            float(
                                np.clip(
                                    (manual_quad[corner_index][0] - sample["patch"]["x"]) / sample["patch"]["size"],
                                    0.0,
                                    1.0,
                                )
                            ),
                            float(
                                np.clip(
                                    (manual_quad[corner_index][1] - sample["patch"]["y"]) / sample["patch"]["size"],
                                    0.0,
                                    1.0,
                                )
                            ),
                        ],
                        "predicted_quad": [[float(x), float(y)] for x, y in order_points(predicted_quad)],
                        "manual_quad": [[float(x), float(y)] for x, y in order_points(manual_quad)],
                        "image_path": page["image_path"],
                    }
                )
        _write_jsonl(output_dir / f"{split_name}.jsonl", exported[split_name])
    return {
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "output_dir": str(output_dir),
        "page_count": sum(len(rows) for rows in split_pages.values()),
        "pages": sum(len(rows) for rows in split_pages.values()) * 4,
        "train_pages": len(split_pages.get("train", [])),
        "test_pages": len(split_pages.get("test", [])),
        "focus_test_pages": len(split_pages.get("focus_test", [])),
        "broad_test_pages": len(split_pages.get("broad_test", split_pages.get("test", []))),
        "holdout_pages": len(split_pages.get("holdout", [])),
        "train_samples": len(exported.get("train", [])),
        "test_samples": len(exported.get("test", [])),
        "focus_test_samples": len(exported.get("focus_test", [])),
        "broad_test_samples": len(exported.get("broad_test", [])),
        "holdout_samples": len(exported.get("holdout", [])),
        "patch_size": patch_size,
        "patch_scale": patch_scale,
        "patch_min": patch_min,
        "patch_max": patch_max,
        "bottom_vertical_bias": bottom_vertical_bias,
        "bl_patch_scale_multiplier": bl_patch_scale_multiplier,
        "bl_bottom_vertical_bias": bl_bottom_vertical_bias,
    }


def export_local_corner_patch_dataset(
    dataset_root: Path,
    output_dir: Path,
    seed: int = 7,
    test_ratio: float = 0.25,
    focus_projects: list[str] | tuple[str, ...] | None = None,
    holdout_projects: list[str] | tuple[str, ...] | None = None,
    focus_test_ratio: float = 0.25,
    patch_size: int | None = 96,
    patch_scale: float = 0.2,
    patch_min: int = 96,
    patch_max: int = 256,
    bottom_vertical_bias: float = 0.0,
    bl_patch_scale_multiplier: float = 1.0,
    bl_bottom_vertical_bias: float = 0.0,
) -> dict[str, Any]:
    from dataset_benchmark import build_project_aware_split, build_split

    pages = load_active_manual_pages(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    focus_values = [str(item) for item in (focus_projects or []) if str(item).strip()]
    holdout_values = [str(item) for item in (holdout_projects or []) if str(item).strip()]
    if focus_values or holdout_values:
        split = build_project_aware_split(
            pages,
            focus_projects=focus_values,
            holdout_projects=holdout_values,
            test_ratio=test_ratio,
            focus_test_ratio=focus_test_ratio,
            seed=seed,
        )
        split_pages: dict[str, list[PageRecord]] = {
            "train": list(split["train"]),
            "test": list(split["test"]) + list(split["focus_test"]),
            "focus_test": list(split["focus_test"]),
            "broad_test": list(split["test"]),
            "holdout": list(split["holdout"]),
        }
    else:
        split = build_split(pages, test_ratio=test_ratio, seed=seed)
        split_pages = {
            "train": list(split["train"]),
            "test": list(split["test"]),
        }
    summary = export_local_corner_patch_dataset_from_splits(
        split_pages=split_pages,
        output_dir=output_dir,
        dataset_root=dataset_root,
        patch_size=patch_size,
        patch_scale=patch_scale,
        patch_min=patch_min,
        patch_max=patch_max,
        bottom_vertical_bias=bottom_vertical_bias,
        bl_patch_scale_multiplier=bl_patch_scale_multiplier,
        bl_bottom_vertical_bias=bl_bottom_vertical_bias,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def refine_quad_with_residual_predictor(
    image_path: Path,
    predicted_quad: np.ndarray,
    residual_predictor: ResidualPredictor,
    patch_size: int = 96,
) -> list[list[float]]:
    predicted = order_points(np.array(predicted_quad, dtype=np.float32))
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    refined: list[list[float]] = []
    for corner_index in range(4):
        sample = build_local_corner_patch_sample(
            image_path=image_path,
            image=image,
            page_id=image_path.stem,
            corner_index=corner_index,
            predicted_quad=predicted,
            manual_quad=predicted,
            patch_size=patch_size,
        )
        residual = np.array(residual_predictor(sample), dtype=np.float32)
        point = predicted[corner_index] + residual * float(sample["patch"]["size"])
        refined.append([float(point[0]), float(point[1])])
    return [[float(x), float(y)] for x, y in order_points(np.array(refined, dtype=np.float32))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export local corner residual dataset")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--focus-project", action="append", dest="focus_projects", default=[])
    parser.add_argument("--holdout-project", action="append", dest="holdout_projects", default=[])
    parser.add_argument("--focus-test-ratio", type=float, default=0.25)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--patch-scale", type=float, default=0.2)
    parser.add_argument("--patch-min", type=int, default=96)
    parser.add_argument("--patch-max", type=int, default=256)
    parser.add_argument("--bottom-vertical-bias", type=float, default=0.0)
    parser.add_argument("--bl-patch-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--bl-bottom-vertical-bias", type=float, default=0.0)
    args = parser.parse_args()
    result = export_local_corner_patch_dataset(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        test_ratio=args.test_ratio,
        focus_projects=args.focus_projects,
        holdout_projects=args.holdout_projects,
        focus_test_ratio=args.focus_test_ratio,
        patch_size=args.patch_size,
        patch_scale=args.patch_scale,
        patch_min=args.patch_min,
        patch_max=args.patch_max,
        bottom_vertical_bias=args.bottom_vertical_bias,
        bl_patch_scale_multiplier=args.bl_patch_scale_multiplier,
        bl_bottom_vertical_bias=args.bl_bottom_vertical_bias,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
