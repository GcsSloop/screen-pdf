from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

try:
    from corner_train import CornerHeatmapNet, decode_model_output, remap_legacy_head_state_dict, select_torch_device
    from dataset_benchmark import normalized_point_error, quad_geometry_metrics, summarize_geometry_metric_rows
    from perspective_detect import order_points
except ModuleNotFoundError:
    from engine.corner_train import CornerHeatmapNet, decode_model_output, remap_legacy_head_state_dict, select_torch_device
    from engine.dataset_benchmark import normalized_point_error, quad_geometry_metrics, summarize_geometry_metric_rows
    from engine.perspective_detect import order_points


QuadPredictor = Callable[[Path], np.ndarray]
RoiPredictor = Callable[[dict[str, Any]], np.ndarray]
LocalQuadPredictor = Callable[[Path, np.ndarray], np.ndarray]


def _load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    return image


def _clip_quad_to_image(quad: np.ndarray, image_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    height, width = image_shape[:2]
    clipped = np.array(quad, dtype=np.float32).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, max(float(width - 1), 0.0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, max(float(height - 1), 0.0))
    return clipped


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


def _denormalize_corners(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    restored = np.array(corners, dtype=np.float32).copy()
    restored[:, 0] *= float(width)
    restored[:, 1] *= float(height)
    return restored


def apply_roi_prediction(
    pred_norm: np.ndarray,
    roi: dict[str, int],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    points = np.array(pred_norm, dtype=np.float32)
    restored = np.empty_like(points)
    restored[:, 0] = roi["x"] + points[:, 0] * roi["width"]
    restored[:, 1] = roi["y"] + points[:, 1] * roi["height"]
    return _clip_quad_to_image(order_points(restored), image_shape)


def build_refine_request(
    image_path: Path,
    coarse_quad: np.ndarray,
    page_id: str,
    expand_ratio: float = 0.08,
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    if image is None:
        image = _load_image(image_path)
    coarse = order_points(np.array(coarse_quad, dtype=np.float32))
    roi = _build_roi_from_quad(image.shape, coarse, expand_ratio=expand_ratio)
    x0, y0 = roi["x"], roi["y"]
    x1, y1 = x0 + roi["width"], y0 + roi["height"]
    roi_image = image[y0:y1, x0:x1].copy()
    return {
        "page_id": page_id,
        "image_path": str(image_path),
        "image_shape": [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
        "roi": roi,
        "roi_image": roi_image,
        "coarse_quad": coarse.tolist(),
        "coarse_quad_norm": _normalize_quad_to_roi(coarse, roi),
    }


def predict_two_stage(
    image_path: Path,
    global_predictor: QuadPredictor,
    roi_predictor: RoiPredictor,
    local_predictor: LocalQuadPredictor | None = None,
    page_id: str | None = None,
    expand_ratio: float = 0.08,
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    resolved_page_id = page_id or image_path.stem
    if image is None:
        image = _load_image(image_path)
    predict_image = getattr(global_predictor, "predict_image", None)
    coarse_pred = predict_image(image) if callable(predict_image) else global_predictor(image_path)
    coarse_quad = order_points(np.array(coarse_pred, dtype=np.float32))
    request = build_refine_request(
        image_path=image_path,
        coarse_quad=coarse_quad,
        page_id=resolved_page_id,
        expand_ratio=expand_ratio,
        image=image,
    )
    pred_norm = np.array(roi_predictor(request), dtype=np.float32)
    roi_quad = apply_roi_prediction(pred_norm, request["roi"], image.shape)
    final_quad = roi_quad
    if local_predictor is not None:
        try:
            refined = local_predictor(image_path, roi_quad, image=image)
        except TypeError:
            refined = local_predictor(image_path, roi_quad)
        final_quad = order_points(np.array(refined, dtype=np.float32))
    return {
        "page_id": resolved_page_id,
        "image_path": str(image_path),
        "coarse_quad": coarse_quad.tolist(),
        "roi_quad": [[round(float(x), 4), round(float(y), 4)] for x, y in roi_quad],
        "final_quad": [[round(float(x), 4), round(float(y), 4)] for x, y in final_quad],
        "roi": request["roi"],
    }


class GlobalCornerPredictor:
    def __init__(self, model_path: Path) -> None:
        self.device = select_torch_device()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CornerHeatmapNet(
            in_channels=3,
            channels=int(checkpoint["channels"]),
            output_channels=4,
            head_mode=str(checkpoint.get("head_mode", "heatmap")),
        )
        self.model.load_state_dict(remap_legacy_head_state_dict(checkpoint["state_dict"]))
        self.model.to(self.device)
        self.model.eval()
        self.input_size = int(checkpoint["input_size"])
        self.decode_mode = str(checkpoint.get("decode_mode", "argmax"))
        self.head_mode = str(checkpoint.get("head_mode", "heatmap"))

    def predict_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        image_f = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(image_f, (2, 0, 1))[None, ...]).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(tensor)
            pred_norm = decode_model_output(output, decode_mode=self.decode_mode, head_mode=self.head_mode).cpu().numpy()[0]
        return _denormalize_corners(pred_norm, width, height)

    def __call__(self, image_path: Path) -> np.ndarray:
        return self.predict_image(_load_image(image_path))


class RoiCornerPredictor:
    def __init__(self, model_path: Path) -> None:
        self.device = select_torch_device()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CornerHeatmapNet(
            in_channels=4,
            channels=int(checkpoint["channels"]),
            output_channels=4,
            head_mode=str(checkpoint.get("head_mode", "heatmap")),
        )
        self.model.load_state_dict(remap_legacy_head_state_dict(checkpoint["state_dict"]))
        self.model.to(self.device)
        self.model.eval()
        self.input_size = int(checkpoint["input_size"])
        self.decode_mode = str(checkpoint.get("decode_mode", "argmax"))
        self.head_mode = str(checkpoint.get("head_mode", "heatmap"))

    def __call__(self, request: dict[str, Any]) -> np.ndarray:
        roi_image = np.array(request["roi_image"], copy=False)
        image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        image_f = image.astype(np.float32) / 255.0
        mask = np.zeros((self.input_size, self.input_size), dtype=np.float32)
        pts = np.round(np.array(request["coarse_quad_norm"], dtype=np.float32) * (self.input_size - 1)).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 1.0)
        features = np.concatenate([np.transpose(image_f, (2, 0, 1)), mask[None, ...]], axis=0)
        tensor = torch.from_numpy(features[None, ...]).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(tensor)
            pred_norm = decode_model_output(output, decode_mode=self.decode_mode, head_mode=self.head_mode).cpu().numpy()[0]
        return pred_norm


class LocalCornerMoEPredictor:
    def __init__(self, model_path: Path) -> None:
        try:
            from local_corner_heatmap import build_patch_metadata
            from local_corner_moe import LocalCornerMoENet, decode_moe_output, remap_legacy_moe_state_dict
            from local_corner_moe_coord import LocalCornerMoECoordNet, decode_moe_coord_output
            from local_corner_refine import build_local_corner_patch_sample, build_patch_features
        except ModuleNotFoundError:
            from engine.local_corner_heatmap import build_patch_metadata
            from engine.local_corner_moe import LocalCornerMoENet, decode_moe_output, remap_legacy_moe_state_dict
            from engine.local_corner_moe_coord import LocalCornerMoECoordNet, decode_moe_coord_output
            from engine.local_corner_refine import build_local_corner_patch_sample, build_patch_features

        self.device = select_torch_device()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.coord_mix = float(checkpoint.get("coord_mix", 0.25))
        self.use_visibility = any(str(key).startswith("visibility_heads.") for key in checkpoint["state_dict"].keys())
        self.use_coord_model = "coord_mix" in checkpoint or model_path.name == "local_corner_moe_coord_model.pt"
        if self.use_coord_model:
            self.model = LocalCornerMoECoordNet(
                channels=int(checkpoint["channels"]),
                experts=int(checkpoint["experts"]),
                metadata_dim=int(checkpoint.get("metadata_dim", 0) or 0),
            )
            missing, unexpected = self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            allowed_missing = [
                key
                for key in missing
                if key.startswith("edge_heads.") or key.startswith("edgemap_heads.") or key.startswith("visibility_heads.")
            ]
            if unexpected or len(allowed_missing) != len(missing):
                raise RuntimeError(
                    f"incompatible local corner coord checkpoint: missing={missing}, unexpected={unexpected}"
                )
        else:
            self.model = LocalCornerMoENet(
                channels=int(checkpoint["channels"]),
                experts=int(checkpoint["experts"]),
                metadata_dim=int(checkpoint.get("metadata_dim", 0) or 0),
            )
            self.model.load_state_dict(remap_legacy_moe_state_dict(checkpoint["state_dict"]), strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.input_size = int(checkpoint["input_size"])
        self.metadata_dim = int(checkpoint.get("metadata_dim", 0) or 0)
        self._build_patch_metadata = build_patch_metadata
        self._build_local_corner_patch_sample = build_local_corner_patch_sample
        self._build_patch_features = build_patch_features
        self._decode_moe_output = decode_moe_output
        self._decode_moe_coord_output = decode_moe_coord_output

    def __call__(self, image_path: Path, predicted_quad: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        ordered_quad = order_points(np.array(predicted_quad, dtype=np.float32))
        if image is None:
            image = _load_image(image_path)
        point_norms: list[np.ndarray] = []
        patch_samples: list[dict[str, Any]] = []
        for corner_index in range(4):
            sample = self._build_local_corner_patch_sample(
                image_path=image_path,
                image=image,
                page_id=image_path.stem,
                corner_index=corner_index,
                predicted_quad=ordered_quad,
                manual_quad=ordered_quad,
                patch_size=None,
            )
            patch_samples.append(sample)
            features = self._build_patch_features(np.array(sample["patch_image"], copy=False), corner_index, input_size=self.input_size)
            tensor = torch.from_numpy(features[None, ...]).to(device=self.device, dtype=torch.float32)
            metadata = None
            if self.metadata_dim > 0:
                metadata_row = {
                    "corner_index": corner_index,
                    "patch": sample["patch"],
                    "predicted_point": sample["predicted_point"],
                    "predicted_quad": ordered_quad.tolist(),
                }
                metadata = torch.from_numpy(self._build_patch_metadata(metadata_row)[None, ...]).to(device=self.device, dtype=torch.float32)
            with torch.no_grad():
                if self.use_coord_model:
                    heatmaps, offsets, coord_head, _, _, visibility, _ = self.model(tensor, metadata)
                    decoded = self._decode_moe_coord_output(heatmaps, offsets, coord_head, coord_mix=0.0)
                    if self.use_visibility:
                        visibility_score = torch.clamp(visibility.mean(dim=-1, keepdim=True), 0.0, 1.0)
                        adaptive_mix = torch.clamp(self.coord_mix * 0.35 + visibility_score * 0.65, 0.05, 0.9)
                        point = torch.clamp(decoded * (1.0 - adaptive_mix) + coord_head * adaptive_mix, 0.0, 1.0).cpu().numpy()[0]
                    else:
                        point = self._decode_moe_coord_output(
                            heatmaps,
                            offsets,
                            coord_head,
                            coord_mix=self.coord_mix,
                        ).cpu().numpy()[0]
                else:
                    heatmaps, offsets, _ = self.model(tensor, metadata)
                    point = self._decode_moe_output(heatmaps, offsets).cpu().numpy()[0]
            point_norms.append(point.astype(np.float32))
        points: list[list[float]] = []
        for sample, point_norm in zip(patch_samples, point_norms, strict=True):
            patch = sample["patch"]
            points.append(
                [
                    float(patch["x"] + point_norm[0] * patch["size"]),
                    float(patch["y"] + point_norm[1] * patch["size"]),
                ]
            )
        return order_points(np.array(points, dtype=np.float32))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def export_refine_dataset_from_global_predictions(
    global_model_path: Path,
    split_dir: Path,
    output_dir: Path,
    expand_ratio: float = 0.08,
) -> dict[str, Any]:
    predictor = GlobalCornerPredictor(global_model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for split_name in ("train", "test"):
        rows = [json.loads(line) for line in (split_dir / f"{split_name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            image_path = Path(row["image_path"])
            request = build_refine_request(
                image_path=image_path,
                coarse_quad=predictor(image_path),
                page_id=str(row.get("page_id") or image_path.stem),
                expand_ratio=expand_ratio,
            )
            roi_rel = Path("roi") / split_name / f"{image_path.stem}.png"
            roi_abs = output_dir / roi_rel
            roi_abs.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(roi_abs), request["roi_image"])
            exported[split_name].append(
                {
                    "split": split_name,
                    "page_id": str(row.get("page_id") or image_path.stem),
                    "project_name": row.get("project_name", ""),
                    "image_path": str(image_path),
                    "roi_path": roi_rel.as_posix(),
                    "roi": request["roi"],
                    "manual_quad": row["manual_quad"],
                    "coarse_quad": request["coarse_quad"],
                    "corner_norm": _normalize_quad_to_roi(np.array(row["manual_quad"], dtype=np.float32), request["roi"]),
                    "coarse_quad_norm": request["coarse_quad_norm"],
                }
            )
        _write_jsonl(output_dir / f"{split_name}.jsonl", exported[split_name])

    summary = {
        "global_model_path": str(global_model_path),
        "split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "train_pages": len(exported["train"]),
        "test_pages": len(exported["test"]),
        "expand_ratio": expand_ratio,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def evaluate_two_stage(global_model_path: Path, roi_model_path: Path, split_dir: Path, split: str = "test", expand_ratio: float = 0.08) -> dict[str, Any]:
    global_predictor = GlobalCornerPredictor(global_model_path)
    roi_predictor = RoiCornerPredictor(roi_model_path)
    rows = [json.loads(line) for line in (split_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metric_rows: list[dict[str, float]] = []
    timings: list[float] = []
    coarse_errors: list[float] = []
    coarse_metric_rows: list[dict[str, float]] = []
    for row in rows:
        image_path = Path(row["image_path"])
        t0 = time.perf_counter()
        coarse_quad = global_predictor(image_path)
        request = build_refine_request(
            image_path=image_path,
            coarse_quad=coarse_quad,
            page_id=str(row.get("page_id") or image_path.stem),
            expand_ratio=expand_ratio,
        )
        pred_norm = roi_predictor(request)
        final_quad = apply_roi_prediction(pred_norm, request["roi"], tuple(request["image_shape"]))
        timings.append((time.perf_counter() - t0) * 1000.0)
        coarse_errors.append(float(normalized_point_error(row["manual_quad"], coarse_quad)))
        coarse_metric_rows.append(quad_geometry_metrics(row["manual_quad"], coarse_quad))
        metric_rows.append(quad_geometry_metrics(row["manual_quad"], final_quad))
    coarse_arr = np.array(coarse_errors, dtype=np.float32)
    time_arr = np.array(timings, dtype=np.float32)
    summary = summarize_geometry_metric_rows(metric_rows)
    coarse_summary = summarize_geometry_metric_rows(coarse_metric_rows)
    return {
        "pages": len(rows),
        **summary,
        "coarse_point_error_mean": round(float(coarse_arr.mean()), 4) if len(coarse_arr) else 0.0,
        "coarse_screen_relative_error_mean": coarse_summary["screen_relative_error_mean"],
        "coarse_max_corner_error_mean": coarse_summary["max_corner_error_mean"],
        "coarse_perspective_tilt_error_mean": coarse_summary["perspective_tilt_error_mean"],
        "coarse_quad_inset_ratio_mean": coarse_summary["quad_inset_ratio_mean"],
        "mean_infer_ms": round(float(time_arr.mean()), 2) if len(time_arr) else 0.0,
        "p95_infer_ms": round(float(np.percentile(time_arr, 95)), 2) if len(time_arr) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-stage global + ROI corner pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-refine-dataset")
    export_parser.add_argument("--global-model", required=True)
    export_parser.add_argument("--split-dir", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--expand-ratio", type=float, default=0.08)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--global-model", required=True)
    eval_parser.add_argument("--roi-model", required=True)
    eval_parser.add_argument("--split-dir", required=True)
    eval_parser.add_argument("--split", default="test")
    eval_parser.add_argument("--expand-ratio", type=float, default=0.08)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--global-model", required=True)
    predict_parser.add_argument("--roi-model", required=True)
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--page-id")
    predict_parser.add_argument("--expand-ratio", type=float, default=0.08)

    args = parser.parse_args()
    if args.command == "export-refine-dataset":
        result = export_refine_dataset_from_global_predictions(
            global_model_path=Path(args.global_model),
            split_dir=Path(args.split_dir),
            output_dir=Path(args.output_dir),
            expand_ratio=float(args.expand_ratio),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "eval":
        result = evaluate_two_stage(
            global_model_path=Path(args.global_model),
            roi_model_path=Path(args.roi_model),
            split_dir=Path(args.split_dir),
            split=args.split,
            expand_ratio=float(args.expand_ratio),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    result = predict_two_stage(
        image_path=Path(args.image),
        global_predictor=GlobalCornerPredictor(Path(args.global_model)),
        roi_predictor=RoiCornerPredictor(Path(args.roi_model)),
        page_id=args.page_id,
        expand_ratio=float(args.expand_ratio),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
