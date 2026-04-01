from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import torch

try:
    from corner_train import CornerHeatmapNet, decode_model_output, remap_legacy_head_state_dict, select_torch_device
    from dataset_benchmark import normalized_point_error, quad_geometry_metrics, summarize_geometry_metric_rows
    from global_corner_train import build_global_feature_tensor
    from perspective_detect import order_points
except ModuleNotFoundError:
    from engine.corner_train import CornerHeatmapNet, decode_model_output, remap_legacy_head_state_dict, select_torch_device
    from engine.dataset_benchmark import normalized_point_error, quad_geometry_metrics, summarize_geometry_metric_rows
    from engine.global_corner_train import build_global_feature_tensor
    from engine.perspective_detect import order_points


QuadPredictor = Callable[[Path], np.ndarray]
RoiPredictor = Callable[[dict[str, Any]], np.ndarray]
LocalQuadPredictor = Callable[[Path, np.ndarray], np.ndarray]

CANDIDATE_SELECTOR_FEATURE_NAMES = [
    "expand_ratio",
    "candidate_score",
    "mean_trust",
    "min_trust",
    "mean_displacement_norm",
    "max_displacement_norm",
    "roi_width_frac",
    "roi_height_frac",
    "roi_area_frac",
    "coarse_drift",
    "is_baseline",
    "candidate_score_delta_vs_baseline",
    "mean_trust_delta_vs_baseline",
    "min_trust_delta_vs_baseline",
    "mean_displacement_delta_vs_baseline",
    "max_displacement_delta_vs_baseline",
    "roi_area_delta_vs_baseline",
    "coarse_drift_delta_vs_baseline",
]


class LinearCandidateExpandSelector:
    def __init__(
        self,
        weights: Sequence[float],
        bias: float = 0.0,
        *,
        feature_mean: Sequence[float] | None = None,
        feature_std: Sequence[float] | None = None,
        feature_names: Sequence[str] | None = None,
        switch_margin: float = 0.0,
    ) -> None:
        self.weights = np.array(list(weights), dtype=np.float32)
        self.bias = float(bias)
        self.feature_mean = (
            np.array(list(feature_mean), dtype=np.float32)
            if feature_mean is not None
            else np.zeros_like(self.weights, dtype=np.float32)
        )
        self.feature_std = (
            np.array(list(feature_std), dtype=np.float32)
            if feature_std is not None
            else np.ones_like(self.weights, dtype=np.float32)
        )
        self.feature_std = np.where(np.abs(self.feature_std) < 1e-6, 1.0, self.feature_std)
        self.feature_names = list(feature_names or [])
        self.switch_margin = float(switch_margin)
        if self.feature_mean.shape != self.weights.shape or self.feature_std.shape != self.weights.shape:
            raise ValueError("selector normalization shape mismatch")

    @classmethod
    def from_json(cls, path: Path) -> "LinearCandidateExpandSelector":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            payload["weights"],
            bias=float(payload.get("bias", 0.0)),
            feature_mean=payload.get("feature_mean"),
            feature_std=payload.get("feature_std"),
            feature_names=payload.get("feature_names"),
            switch_margin=float(payload.get("switch_margin", 0.0) or 0.0),
        )

    def score_features(self, features: Sequence[float]) -> float:
        vector = np.array(list(features), dtype=np.float32)
        if vector.shape != self.weights.shape:
            raise ValueError(f"feature shape mismatch: expected {self.weights.shape}, got {vector.shape}")
        normalized = (vector - self.feature_mean) / self.feature_std
        return float(np.dot(normalized, self.weights) + self.bias)

    def select_candidate(
        self,
        candidates: list[dict[str, Any]],
        *,
        baseline_expand_ratio: float,
    ) -> dict[str, Any] | None:
        scored: list[tuple[float, dict[str, Any]]] = []
        baseline_item: dict[str, Any] | None = None
        baseline_score: float | None = None
        for item in candidates:
            features = item.get("selector_features")
            if not isinstance(features, list) or not features:
                continue
            score = self.score_features(features)
            scored.append((score, item))
            if abs(float(item.get("expand_ratio", -1.0) or -1.0) - float(baseline_expand_ratio)) <= 1e-9:
                baseline_item = item
                baseline_score = score
        if not scored:
            return None
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_item = scored[0]
        if baseline_item is not None and best_item is not baseline_item and baseline_score is not None:
            if float(best_score) < float(baseline_score) + float(self.switch_margin):
                return baseline_item
        return best_item


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


def _compute_local_candidate_score(details: dict[str, Any] | None) -> float | None:
    if not details:
        return None
    corner_details = details.get("corner_details")
    if not isinstance(corner_details, list) or not corner_details:
        return None
    trust_values: list[float] = []
    displacement_norms: list[float] = []
    for item in corner_details:
        if not isinstance(item, dict):
            continue
        trust = float(item.get("trust", 0.0) or 0.0)
        displacement = float(item.get("displacement", 0.0) or 0.0)
        patch_size = max(float(item.get("patch_size", 1.0) or 1.0), 1.0)
        trust_values.append(trust)
        displacement_norms.append(displacement / patch_size)
    if not trust_values or not displacement_norms:
        return None
    return float(np.mean(trust_values) - 0.5 * np.mean(displacement_norms))


def _build_candidate_selector_features(candidate: dict[str, Any], coarse_quad: np.ndarray, image_shape: Sequence[int]) -> list[float]:
    height, width = image_shape[:2]
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    roi = candidate.get("roi") or {}
    roi_width = max(float(roi.get("width", 0.0) or 0.0), 1.0)
    roi_height = max(float(roi.get("height", 0.0) or 0.0), 1.0)
    corner_details = (candidate.get("local_details") or {}).get("corner_details") or []
    trust_values: list[float] = []
    displacement_norms: list[float] = []
    for item in corner_details:
        if not isinstance(item, dict):
            continue
        trust_values.append(float(item.get("trust", 0.0) or 0.0))
        patch_size = max(float(item.get("patch_size", 1.0) or 1.0), 1.0)
        displacement_norms.append(float(item.get("displacement", 0.0) or 0.0) / patch_size)
    final_source = candidate.get("final_quad")
    if final_source is None:
        final_source = candidate.get("roi_quad")
    if final_source is None:
        final_source = coarse_quad
    final_quad = np.array(final_source, dtype=np.float32)
    final_quad = order_points(final_quad)
    coarse_quad = order_points(np.array(coarse_quad, dtype=np.float32))
    drift = float(normalized_point_error(final_quad / np.array([width, height], dtype=np.float32), coarse_quad / np.array([width, height], dtype=np.float32)))
    return [
        float(candidate.get("expand_ratio", 0.0) or 0.0),
        float(candidate.get("candidate_score", 0.0) or 0.0),
        float(np.mean(trust_values)) if trust_values else 0.0,
        float(np.min(trust_values)) if trust_values else 0.0,
        float(np.mean(displacement_norms)) if displacement_norms else 0.0,
        float(np.max(displacement_norms)) if displacement_norms else 0.0,
        roi_width / width,
        roi_height / height,
        (roi_width * roi_height) / max(width * height, 1.0),
        drift,
    ]


def _attach_candidate_selector_features(
    candidates: list[dict[str, Any]],
    coarse_quad: np.ndarray,
    image_shape: Sequence[int],
    *,
    baseline_expand_ratio: float,
) -> None:
    raw_features = [
        _build_candidate_selector_features(item, coarse_quad, image_shape)
        for item in candidates
    ]
    baseline_index = next(
        (
            index
            for index, item in enumerate(candidates)
            if abs(float(item.get("expand_ratio", -1.0) or -1.0) - float(baseline_expand_ratio)) <= 1e-9
        ),
        0,
    )
    baseline_features = raw_features[baseline_index]
    for index, item in enumerate(candidates):
        raw = raw_features[index]
        item["selector_features"] = [
            *raw,
            1.0 if index == baseline_index else 0.0,
            raw[1] - baseline_features[1],
            raw[2] - baseline_features[2],
            raw[3] - baseline_features[3],
            raw[4] - baseline_features[4],
            raw[5] - baseline_features[5],
            raw[8] - baseline_features[8],
            raw[9] - baseline_features[9],
        ]


def _should_use_candidate_expand_selection(
    baseline_score: float | None,
    best_score: float | None,
    *,
    baseline_gate: float,
    min_score_gain: float,
) -> bool:
    if baseline_score is None or best_score is None:
        return False
    return float(baseline_score) < float(baseline_gate) and float(best_score) > float(baseline_score) + float(min_score_gain)


def _run_two_stage_candidate(
    *,
    image_path: Path,
    page_id: str,
    image: np.ndarray,
    coarse_quad: np.ndarray,
    roi_predictor: RoiPredictor,
    local_predictor: LocalQuadPredictor | None,
    expand_ratio: float,
) -> dict[str, Any]:
    request = build_refine_request(
        image_path=image_path,
        coarse_quad=coarse_quad,
        page_id=page_id,
        expand_ratio=expand_ratio,
        image=image,
    )
    pred_norm = np.array(roi_predictor(request), dtype=np.float32)
    roi_quad = apply_roi_prediction(pred_norm, request["roi"], image.shape)
    final_quad = roi_quad
    local_details: dict[str, Any] | None = None
    if local_predictor is not None:
        predict_with_details = getattr(local_predictor, "predict_with_details", None)
        if callable(predict_with_details):
            try:
                local_details = predict_with_details(image_path=image_path, predicted_quad=roi_quad, image=image)
            except TypeError:
                local_details = predict_with_details(image_path, roi_quad, image=image)
            final_quad = order_points(np.array(local_details.get("quad", roi_quad), dtype=np.float32))
        else:
            try:
                refined = local_predictor(image_path, roi_quad, image=image)
            except TypeError:
                refined = local_predictor(image_path, roi_quad)
            final_quad = order_points(np.array(refined, dtype=np.float32))
    return {
        "roi_quad": roi_quad,
        "final_quad": final_quad,
        "roi": request["roi"],
        "expand_ratio": float(expand_ratio),
        "local_details": local_details,
        "candidate_score": _compute_local_candidate_score(local_details),
    }


def predict_two_stage(
    image_path: Path,
    global_predictor: QuadPredictor,
    roi_predictor: RoiPredictor,
    local_predictor: LocalQuadPredictor | None = None,
    page_id: str | None = None,
    expand_ratio: float = 0.08,
    candidate_expand_ratios: Sequence[float] | None = None,
    candidate_baseline_gate: float = 0.45,
    candidate_min_score_gain: float = 0.03,
    candidate_selector: Any | None = None,
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    resolved_page_id = page_id or image_path.stem
    if image is None:
        image = _load_image(image_path)
    predict_image = getattr(global_predictor, "predict_image", None)
    coarse_pred = predict_image(image) if callable(predict_image) else global_predictor(image_path)
    coarse_quad = order_points(np.array(coarse_pred, dtype=np.float32))
    expand_candidates: list[float] = [float(expand_ratio)]
    if candidate_expand_ratios:
        seen: set[float] = set()
        expand_candidates = []
        for value in [*candidate_expand_ratios, float(expand_ratio)]:
            ratio = float(value)
            if ratio <= 0.0 or ratio in seen:
                continue
            seen.add(ratio)
            expand_candidates.append(ratio)
        expand_candidates.sort()
    selected = _run_two_stage_candidate(
        image_path=image_path,
        page_id=resolved_page_id,
        image=image,
        coarse_quad=coarse_quad,
        roi_predictor=roi_predictor,
        local_predictor=local_predictor,
        expand_ratio=float(expand_ratio),
    )
    if len(expand_candidates) > 1 and local_predictor is not None and callable(getattr(local_predictor, "predict_with_details", None)):
        candidates = [
            _run_two_stage_candidate(
                image_path=image_path,
                page_id=resolved_page_id,
                image=image,
                coarse_quad=coarse_quad,
                roi_predictor=roi_predictor,
                local_predictor=local_predictor,
                expand_ratio=ratio,
            )
            for ratio in expand_candidates
        ]
        _attach_candidate_selector_features(
            candidates,
            coarse_quad,
            image.shape,
            baseline_expand_ratio=float(expand_ratio),
        )
        baseline = next(
            (item for item in candidates if abs(float(item["expand_ratio"]) - float(expand_ratio)) <= 1e-9),
            candidates[0],
        )
        if candidate_selector is not None and callable(getattr(candidate_selector, "select_candidate", None)):
            chosen = candidate_selector.select_candidate(candidates, baseline_expand_ratio=float(expand_ratio))
            if isinstance(chosen, dict):
                selected = chosen
            else:
                selected = baseline
        else:
            best = max(
                candidates,
                key=lambda item: float(item["candidate_score"]) if item["candidate_score"] is not None else float("-inf"),
            )
            if _should_use_candidate_expand_selection(
                baseline["candidate_score"],
                best["candidate_score"],
                baseline_gate=candidate_baseline_gate,
                min_score_gain=candidate_min_score_gain,
            ):
                selected = best
            else:
                selected = baseline
    return {
        "page_id": resolved_page_id,
        "image_path": str(image_path),
        "coarse_quad": coarse_quad.tolist(),
        "roi_quad": [[round(float(x), 4), round(float(y), 4)] for x, y in selected["roi_quad"]],
        "final_quad": [[round(float(x), 4), round(float(y), 4)] for x, y in selected["final_quad"]],
        "roi": selected["roi"],
        "selected_expand_ratio": round(float(selected["expand_ratio"]), 4),
        "candidate_count": len(expand_candidates),
    }


class GlobalCornerPredictor:
    def __init__(self, model_path: Path) -> None:
        self.device = select_torch_device()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.input_channels = int(checkpoint.get("input_channels", 3) or 3)
        self.feature_mode = str(checkpoint.get("feature_mode", "rgb"))
        self.input_size = int(checkpoint["input_size"])
        self.decode_mode = str(checkpoint.get("decode_mode", "argmax"))
        self.head_mode = str(checkpoint.get("head_mode", "heatmap"))
        self.model = CornerHeatmapNet(
            in_channels=self.input_channels,
            channels=int(checkpoint["channels"]),
            output_channels=4,
            head_mode=self.head_mode,
        )
        self.model.load_state_dict(remap_legacy_head_state_dict(checkpoint["state_dict"]))
        self.model.to(self.device)
        self.model.eval()

    def predict_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        features = build_global_feature_tensor(image, feature_mode=self.feature_mode)
        tensor = torch.from_numpy(features[None, ...]).to(device=self.device, dtype=torch.float32)
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
        self.input_channels = int(checkpoint.get("input_channels", 10) or 10)
        if self.use_coord_model:
            self.model = LocalCornerMoECoordNet(
                channels=int(checkpoint["channels"]),
                experts=int(checkpoint["experts"]),
                metadata_dim=int(checkpoint.get("metadata_dim", 0) or 0),
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
        patch_config = checkpoint.get("local_patch_config", {})
        self.local_patch_config = {
            "patch_scale": float(patch_config.get("patch_scale", checkpoint.get("patch_scale", 0.2))),
            "patch_min": int(patch_config.get("patch_min", checkpoint.get("patch_min", 96))),
            "patch_max": int(patch_config.get("patch_max", checkpoint.get("patch_max", 256))),
            "bottom_vertical_bias": float(
                patch_config.get("bottom_vertical_bias", checkpoint.get("bottom_vertical_bias", 0.0))
            ),
            "bl_patch_scale_multiplier": float(
                patch_config.get("bl_patch_scale_multiplier", checkpoint.get("bl_patch_scale_multiplier", 1.0))
            ),
            "bl_bottom_vertical_bias": float(
                patch_config.get("bl_bottom_vertical_bias", checkpoint.get("bl_bottom_vertical_bias", 0.0))
            ),
        }
        blend_config = checkpoint.get("local_blend_config", {})
        default_blend_enabled = self.use_coord_model and self.use_visibility
        self.local_blend_config = {
            "enabled": bool(blend_config.get("enabled", default_blend_enabled)),
            "visibility_scale": float(blend_config.get("visibility_scale", 1.5)),
            "visibility_pow": float(blend_config.get("visibility_pow", 1.0)),
            "gate_pow": float(blend_config.get("gate_pow", 0.0)),
            "displacement_weight": float(blend_config.get("displacement_weight", 4.0)),
            "max_trust": float(blend_config.get("max_trust", 0.7)),
            "min_trust": float(blend_config.get("min_trust", 0.0)),
            "page_fallback_visibility_mean": float(blend_config.get("page_fallback_visibility_mean", 0.0)),
            "page_fallback_visibility_min": float(blend_config.get("page_fallback_visibility_min", 0.0)),
        }

    def _compute_corner_trust(
        self,
        visibility_mean: float,
        gate_max: float,
        displacement: float,
        patch_size: float,
    ) -> float:
        config = self.local_blend_config
        if not config["enabled"]:
            return 1.0
        visibility_term = max(float(visibility_mean), 0.0) ** max(float(config["visibility_pow"]), 0.0)
        gate_term = max(float(gate_max), 0.0) ** max(float(config["gate_pow"]), 0.0)
        displacement_norm = float(displacement) / max(float(patch_size), 1.0)
        raw_trust = float(config["visibility_scale"]) * visibility_term * gate_term
        raw_trust /= 1.0 + float(config["displacement_weight"]) * max(displacement_norm, 0.0)
        return float(np.clip(raw_trust, float(config["min_trust"]), float(config["max_trust"])))

    def _should_fallback_page_to_roi(self, visibility_values: list[float]) -> bool:
        if not visibility_values:
            return False
        config = self.local_blend_config
        visibility_mean = float(np.mean(visibility_values))
        visibility_min = float(np.min(visibility_values))
        mean_threshold = float(config.get("page_fallback_visibility_mean", 0.0))
        min_threshold = float(config.get("page_fallback_visibility_min", 0.0))
        return (mean_threshold > 0.0 and visibility_mean < mean_threshold) or (
            min_threshold > 0.0 and visibility_min < min_threshold
        )

    def predict_with_details(
        self,
        image_path: Path,
        predicted_quad: np.ndarray,
        image: np.ndarray | None = None,
    ) -> dict[str, Any]:
        ordered_quad = order_points(np.array(predicted_quad, dtype=np.float32))
        if image is None:
            image = _load_image(image_path)
        raw_points: list[np.ndarray] = []
        blended_points: list[np.ndarray] = []
        corner_details: list[dict[str, Any]] = []
        patch_samples: list[dict[str, Any]] = []
        point_norms: list[np.ndarray] = []
        confidence_rows: list[dict[str, float]] = []
        for corner_index in range(4):
            sample = self._build_local_corner_patch_sample(
                image_path=image_path,
                image=image,
                page_id=image_path.stem,
                corner_index=corner_index,
                predicted_quad=ordered_quad,
                manual_quad=ordered_quad,
                patch_size=None,
                patch_scale=self.local_patch_config["patch_scale"],
                patch_min=self.local_patch_config["patch_min"],
                patch_max=self.local_patch_config["patch_max"],
                bottom_vertical_bias=self.local_patch_config["bottom_vertical_bias"],
                bl_patch_scale_multiplier=self.local_patch_config["bl_patch_scale_multiplier"],
                bl_bottom_vertical_bias=self.local_patch_config["bl_bottom_vertical_bias"],
            )
            patch_samples.append(sample)
            if self.use_coord_model:
                features = self._build_patch_features(
                    np.array(sample["patch_image"], copy=False),
                    corner_index,
                    input_size=self.input_size,
                    input_channels=self.input_channels,
                )
            else:
                features = self._build_patch_features(
                    np.array(sample["patch_image"], copy=False),
                    corner_index,
                    input_size=self.input_size,
                )
            tensor = torch.from_numpy(features[None, ...]).to(device=self.device, dtype=torch.float32)
            metadata = None
            if self.metadata_dim > 0:
                metadata_row = {
                    "corner_index": corner_index,
                    "patch": sample["patch"],
                    "predicted_point": sample["predicted_point"],
                    "predicted_quad": ordered_quad.tolist(),
                }
                metadata = torch.from_numpy(self._build_patch_metadata(metadata_row)[None, ...]).to(
                    device=self.device, dtype=torch.float32
                )
            visibility_mean = 1.0
            gate_max = 1.0
            with torch.no_grad():
                if self.use_coord_model:
                    heatmaps, offsets, coord_head, _, _, visibility, gates = self.model(tensor, metadata)
                    decoded = self._decode_moe_coord_output(heatmaps, offsets, coord_head, coord_mix=0.0)
                    if self.use_visibility:
                        visibility_score = torch.clamp(visibility.mean(dim=-1, keepdim=True), 0.0, 1.0)
                        adaptive_mix = torch.clamp(self.coord_mix * 0.35 + visibility_score * 0.65, 0.05, 0.9)
                        point = torch.clamp(decoded * (1.0 - adaptive_mix) + coord_head * adaptive_mix, 0.0, 1.0).cpu().numpy()[0]
                        visibility_mean = float(torch.clamp(visibility.mean(), 0.0, 1.0).cpu().item())
                    else:
                        point = self._decode_moe_coord_output(
                            heatmaps,
                            offsets,
                            coord_head,
                            coord_mix=self.coord_mix,
                        ).cpu().numpy()[0]
                    gate_max = float(torch.clamp(gates.max(), 0.0, 1.0).cpu().item())
                else:
                    heatmaps, offsets, _ = self.model(tensor, metadata)
                    point = self._decode_moe_output(heatmaps, offsets).cpu().numpy()[0]
            point_norms.append(point.astype(np.float32))
            confidence_rows.append({"visibility_mean": visibility_mean, "gate_max": gate_max})
        for corner_index, (sample, point_norm, confidence_row) in enumerate(
            zip(patch_samples, point_norms, confidence_rows, strict=True)
        ):
            patch = sample["patch"]
            raw_point = np.array(
                [
                    float(patch["x"] + point_norm[0] * patch["size"]),
                    float(patch["y"] + point_norm[1] * patch["size"]),
                ],
                dtype=np.float32,
            )
            roi_point = ordered_quad[corner_index].astype(np.float32)
            displacement = float(np.linalg.norm(raw_point - roi_point))
            trust = self._compute_corner_trust(
                visibility_mean=confidence_row["visibility_mean"],
                gate_max=confidence_row["gate_max"],
                displacement=displacement,
                patch_size=float(patch["size"]),
            )
            blended_point = roi_point * (1.0 - trust) + raw_point * trust
            raw_points.append(raw_point)
            blended_points.append(blended_point.astype(np.float32))
            corner_details.append(
                {
                    "corner_index": corner_index,
                    "patch_size": float(patch["size"]),
                    "visibility_mean": float(confidence_row["visibility_mean"]),
                    "gate_max": float(confidence_row["gate_max"]),
                    "displacement": displacement,
                    "trust": trust,
                    "roi_point": [float(roi_point[0]), float(roi_point[1])],
                    "raw_point": [float(raw_point[0]), float(raw_point[1])],
                    "blended_point": [float(blended_point[0]), float(blended_point[1])],
                }
            )
        if self._should_fallback_page_to_roi([item["visibility_mean"] for item in corner_details]):
            fallback_quad = order_points(ordered_quad.astype(np.float32))
            return {
                "quad": fallback_quad,
                "raw_quad": order_points(np.array(raw_points, dtype=np.float32)),
                "corner_details": corner_details,
            }
        return {
            "quad": order_points(np.array(blended_points, dtype=np.float32)),
            "raw_quad": order_points(np.array(raw_points, dtype=np.float32)),
            "corner_details": corner_details,
        }

    def __call__(self, image_path: Path, predicted_quad: np.ndarray, image: np.ndarray | None = None) -> np.ndarray:
        return self.predict_with_details(image_path=image_path, predicted_quad=predicted_quad, image=image)["quad"]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _write_roi_png(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError(f"failed to write roi image: {path}")


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
        cached_requests: dict[str, dict[str, Any]] = {}
        rows = [json.loads(line) for line in (split_dir / f"{split_name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            image_path = Path(row["image_path"])
            roi_rel = Path("roi") / split_name / f"{image_path.stem}.png"
            roi_abs = output_dir / roi_rel
            request = cached_requests.get(str(image_path))
            if request is None:
                request = build_refine_request(
                    image_path=image_path,
                    coarse_quad=predictor(image_path),
                    page_id=str(row.get("page_id") or image_path.stem),
                    expand_ratio=expand_ratio,
                )
                roi_abs.parent.mkdir(parents=True, exist_ok=True)
                _write_roi_png(roi_abs, request["roi_image"])
                cached_requests[str(image_path)] = request
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
