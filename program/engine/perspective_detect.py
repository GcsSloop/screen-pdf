from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Candidate:
    method: str
    quad: np.ndarray
    metrics: dict[str, float]
    score: float


DEFAULT_SCORING_PROFILE = {
    "weights": {
        "aspect_score": 0.18,
        "symmetry_score": 0.12,
        "parallel_score": 0.12,
        "rectangularity_score": 0.08,
        "center_score": 0.08,
        "edge_score": 0.18,
        "coverage_score": 0.1,
        "blue_penalty": -0.14,
        "top_dark_penalty": -0.04,
        "floor_penalty": -0.18,
        "spill_penalty": -0.14,
    },
    "method_bias": {},
    "base_method_bias": {},
    "selector_mode": "legacy_rescue",
    "enable_lsd_v2": True,
}

SCORING_PROFILE_PATH = Path(__file__).resolve().parent / "scoring_profile.json"
SCORING_PROFILE_CACHE: dict[str, dict[str, dict[str, float]]] = {}
DEFAULT_OPENCV_PROFILE = {
    "clahe_clip_limit": 2.4,
    "clahe_grid_size": 8,
    "lsd_scale": 0.8,
    "lsd_sigma_scale": 0.6,
    "lsd_quant": 2.0,
    "lsd_ang_th": 22.5,
    "roi_expand_ratio": 0.12,
    "grabcut_iters": 3,
    "mask_close_kernel": 11,
    "mask_open_kernel": 5,
}
OPENCV_PROFILE_PATH = Path(__file__).resolve().parent / "opencv_profile.json"
OPENCV_PROFILE_CACHE: dict[str, dict[str, float]] = {}
ACTIVE_OPENCV_METHODS = {
    "document_quad",
    "contour_quad",
    "contour_quad_edge",
}


def _is_allowed_opencv_method(method: str) -> bool:
    return method in ACTIVE_OPENCV_METHODS


def _filter_candidate_pairs(
    candidates: list[tuple[str, np.ndarray]],
) -> list[tuple[str, np.ndarray]]:
    return [(method, quad) for method, quad in candidates if _is_allowed_opencv_method(method)]


def _filter_variant_candidates(
    candidates: list[tuple[str, np.ndarray, str]],
) -> list[tuple[str, np.ndarray, str]]:
    return [
        (method, quad, base_method)
        for method, quad, base_method in candidates
        if _is_allowed_opencv_method(method)
    ]


def _merge_scoring_profile(loaded: dict | None) -> dict[str, dict[str, float]]:
    profile = {
        "weights": dict(DEFAULT_SCORING_PROFILE["weights"]),
        "method_bias": dict(DEFAULT_SCORING_PROFILE["method_bias"]),
        "base_method_bias": dict(DEFAULT_SCORING_PROFILE["base_method_bias"]),
        "selector_mode": str(DEFAULT_SCORING_PROFILE["selector_mode"]),
        "enable_lsd_v2": bool(DEFAULT_SCORING_PROFILE["enable_lsd_v2"]),
    }
    if not loaded:
        return profile

    weights = loaded.get("weights")
    if isinstance(weights, dict):
        for key, value in weights.items():
            if key in profile["weights"] and isinstance(value, (int, float)):
                profile["weights"][key] = float(value)

    method_bias = loaded.get("method_bias")
    if isinstance(method_bias, dict):
        for key, value in method_bias.items():
            if isinstance(key, str) and isinstance(value, (int, float)):
                profile["method_bias"][key] = float(value)

    base_method_bias = loaded.get("base_method_bias")
    if isinstance(base_method_bias, dict):
        for key, value in base_method_bias.items():
            if isinstance(key, str) and isinstance(value, (int, float)):
                profile["base_method_bias"][key] = float(value)
    selector_mode = loaded.get("selector_mode")
    if isinstance(selector_mode, str) and selector_mode:
        profile["selector_mode"] = selector_mode
    enable_lsd_v2 = loaded.get("enable_lsd_v2")
    if isinstance(enable_lsd_v2, bool):
        profile["enable_lsd_v2"] = enable_lsd_v2
    return profile


def load_scoring_profile(path: Path | None = None) -> dict[str, dict[str, float]]:
    profile_path = str((path or SCORING_PROFILE_PATH).resolve())
    cached = SCORING_PROFILE_CACHE.get(profile_path)
    if cached is not None:
        return {
            "weights": dict(cached["weights"]),
            "method_bias": dict(cached["method_bias"]),
            "base_method_bias": dict(cached["base_method_bias"]),
            "selector_mode": str(cached["selector_mode"]),
            "enable_lsd_v2": bool(cached["enable_lsd_v2"]),
        }

    loaded: dict | None = None
    try:
        loaded = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        loaded = None

    profile = _merge_scoring_profile(loaded)
    SCORING_PROFILE_CACHE[profile_path] = {
        "weights": dict(profile["weights"]),
        "method_bias": dict(profile["method_bias"]),
        "base_method_bias": dict(profile["base_method_bias"]),
        "selector_mode": str(profile["selector_mode"]),
        "enable_lsd_v2": bool(profile["enable_lsd_v2"]),
    }
    return profile


def load_opencv_profile(path: Path | None = None) -> dict[str, float]:
    profile_path = str((path or OPENCV_PROFILE_PATH).resolve())
    cached = OPENCV_PROFILE_CACHE.get(profile_path)
    if cached is not None:
        return dict(cached)

    loaded: dict | None = None
    try:
        loaded = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        loaded = None

    profile = dict(DEFAULT_OPENCV_PROFILE)
    if isinstance(loaded, dict):
        for key, value in loaded.items():
            if key in profile and isinstance(value, (int, float)):
                profile[key] = float(value) if isinstance(profile[key], float) else int(value)
    OPENCV_PROFILE_CACHE[profile_path] = dict(profile)
    return profile


def combine_score_from_metrics(
    metrics: dict[str, float],
    profile: dict[str, dict[str, float]] | None = None,
    method: str | None = None,
    base_method: str | None = None,
) -> float:
    active_profile = profile or load_scoring_profile()
    score = 0.0
    for key, weight in active_profile["weights"].items():
        score += float(metrics.get(key, 0.0)) * float(weight)
    if method:
        score += float(active_profile["method_bias"].get(method, 0.0))
    if base_method:
        score += float(active_profile.get("base_method_bias", {}).get(base_method, 0.0))
    return float(round(score, 4))


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _resize_for_detection(image: np.ndarray, max_side: int = 1200) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale == 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))))
    return resized, scale


def _restore_scale(quad: np.ndarray, scale: float) -> np.ndarray:
    if scale != 1.0:
        quad = quad / scale
    return order_points(quad)


def _candidate_geometry_score(quad: np.ndarray, image_shape: tuple[int, int, int]) -> float:
    quad = order_points(quad)
    widths = [
        np.linalg.norm(quad[1] - quad[0]),
        np.linalg.norm(quad[2] - quad[3]),
    ]
    heights = [
        np.linalg.norm(quad[3] - quad[0]),
        np.linalg.norm(quad[2] - quad[1]),
    ]
    height = float(np.mean(heights))
    width = float(np.mean(widths))
    if min(width, height) <= 1.0:
        return -1.0
    aspect = width / height
    if not 1.0 <= aspect <= 2.8:
        return -1.0
    area = float(cv2.contourArea(quad.astype(np.float32)))
    image_area = float(image_shape[0] * image_shape[1])
    coverage = area / max(image_area, 1.0)
    center = quad.mean(axis=0)
    image_center = np.array([image_shape[1] / 2.0, image_shape[0] / 2.0], dtype=np.float32)
    center_distance = float(np.linalg.norm(center - image_center))
    center_score = 1.0 - min(1.0, center_distance / max(float(np.linalg.norm(image_center)), 1.0))
    border = cv2.boundingRect(quad.astype(np.int32))
    border_touch = (
        border[0] <= 3
        or border[1] <= 3
        or border[0] + border[2] >= image_shape[1] - 3
        or border[1] + border[3] >= image_shape[0] - 3
    )
    return coverage * 1.6 + center_score * 0.5 + min(aspect, 2.0) * 0.15 - (0.25 if border_touch else 0.0)


def _extract_quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        return order_points(approx.reshape(4, 2))
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_points(box)


def _clamp_quad_bottom(quad: np.ndarray, max_bottom_y: float) -> np.ndarray:
    ordered = order_points(quad)
    if ordered[2][1] <= max_bottom_y and ordered[3][1] <= max_bottom_y:
        return ordered
    ordered[2][1] = min(float(ordered[2][1]), float(max_bottom_y))
    ordered[3][1] = min(float(ordered[3][1]), float(max_bottom_y))
    return order_points(ordered)


def _edge_maps(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    return dx, dy, mag


def _line_from_points(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    direction = p1.astype(np.float32) - p0.astype(np.float32)
    length = float(np.linalg.norm(direction))
    if length <= 1e-6:
        return None
    tangent = direction / length
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    return tangent, normal


def _sample_edge_response(
    dx: np.ndarray,
    dy: np.ndarray,
    mag: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    search_px: float,
    samples: int,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    vectors = _line_from_points(start, end)
    if vectors is None:
        return None
    tangent, normal = vectors
    edge_length = float(np.linalg.norm(end - start))
    min_margin = min(0.18, max(0.06, 16.0 / max(edge_length, 1.0)))
    ts = np.linspace(min_margin, 1.0 - min_margin, max(12, samples), dtype=np.float32)
    h, w = mag.shape[:2]
    offset_scores = np.zeros(len(offsets), dtype=np.float32)
    offset_hits = np.zeros(len(offsets), dtype=np.float32)
    for t in ts:
        base = start * (1.0 - float(t)) + end * float(t)
        for idx, offset in enumerate(offsets):
            point = base + normal * float(offset)
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if x < 1 or y < 1 or x >= w - 1 or y >= h - 1:
                continue
            gx = float(dx[y, x])
            gy = float(dy[y, x])
            grad = np.array([gx, gy], dtype=np.float32)
            grad_mag = float(mag[y, x])
            if grad_mag <= 1e-3:
                continue
            alignment = abs(float(np.dot(grad / grad_mag, normal)))
            offset_scores[idx] += grad_mag * (0.35 + 0.65 * alignment)
            offset_hits[idx] += 1.0
    valid = offset_hits > 0
    if not np.any(valid):
        return None
    scores = np.full(len(offsets), -1e9, dtype=np.float32)
    scores[valid] = offset_scores[valid] / offset_hits[valid]
    distance_penalty = np.abs(offsets.astype(np.float32)) / max(float(search_px), 1.0)
    scores[valid] -= distance_penalty[valid] * 4.0
    best_idx = int(np.argmax(scores))
    if offset_hits[best_idx] <= 0:
        return None
    return offsets, scores


def _line_intersection(
    p0: np.ndarray,
    d0: np.ndarray,
    p1: np.ndarray,
    d1: np.ndarray,
) -> np.ndarray | None:
    denom = float(d0[0] * d1[1] - d0[1] * d1[0])
    if abs(denom) < 1e-6:
        return None
    delta = p1 - p0
    t = float((delta[0] * d1[1] - delta[1] * d1[0]) / denom)
    return (p0 + d0 * t).astype(np.float32)


def _is_reasonable_refined_quad(original: np.ndarray, refined: np.ndarray, image_shape: tuple[int, int, int]) -> bool:
    ordered_original = order_points(original)
    ordered_refined = order_points(refined)
    if np.any(~np.isfinite(ordered_refined)):
        return False
    h, w = image_shape[:2]
    if np.min(ordered_refined[:, 0]) < -2.0 or np.min(ordered_refined[:, 1]) < -2.0:
        return False
    if np.max(ordered_refined[:, 0]) > w + 2.0 or np.max(ordered_refined[:, 1]) > h + 2.0:
        return False
    original_area = float(abs(cv2.contourArea(ordered_original.astype(np.float32))))
    refined_area = float(abs(cv2.contourArea(ordered_refined.astype(np.float32))))
    if original_area <= 1.0 or refined_area <= 1.0:
        return False
    area_ratio = refined_area / original_area
    if area_ratio < 0.6 or area_ratio > 1.45:
        return False
    point_shift = float(np.mean(np.linalg.norm(ordered_refined - ordered_original, axis=1)))
    max_shift = max(float(max(h, w)) * 0.08, 18.0)
    return point_shift <= max_shift


def _refine_quad_by_edge_alignment(
    image: np.ndarray,
    quad: np.ndarray,
    search_px: int = 0,
    samples: int = 28,
) -> np.ndarray:
    ordered = order_points(quad)
    dx, dy, mag = _edge_maps(image)
    active_search = int(search_px)
    if active_search <= 0:
        edge_lengths = [
            float(np.linalg.norm(ordered[1] - ordered[0])),
            float(np.linalg.norm(ordered[2] - ordered[1])),
            float(np.linalg.norm(ordered[2] - ordered[3])),
            float(np.linalg.norm(ordered[3] - ordered[0])),
        ]
        active_search = int(
            round(
                min(
                    40.0,
                    max(
                        16.0,
                        max(image.shape[:2]) * 0.035,
                        np.percentile(edge_lengths, 25) * 0.12,
                    ),
                )
            )
        )
    offsets = np.arange(-active_search, active_search + 1, dtype=np.float32)
    shifted_lines: list[tuple[np.ndarray, np.ndarray]] = []
    for start, end in (
        (ordered[0], ordered[1]),
        (ordered[1], ordered[2]),
        (ordered[3], ordered[2]),
        (ordered[0], ordered[3]),
    ):
        vectors = _line_from_points(start, end)
        if vectors is None:
            return ordered
        tangent, normal = vectors
        sampled = _sample_edge_response(dx, dy, mag, start, end, float(active_search), samples, offsets)
        if sampled is None:
            return ordered
        sampled_offsets, scores = sampled
        best_offset = float(sampled_offsets[int(np.argmax(scores))])
        shifted_lines.append((start + normal * best_offset, tangent))

    intersections = [
        _line_intersection(*shifted_lines[0], *shifted_lines[3]),
        _line_intersection(*shifted_lines[0], *shifted_lines[1]),
        _line_intersection(*shifted_lines[2], *shifted_lines[1]),
        _line_intersection(*shifted_lines[2], *shifted_lines[3]),
    ]
    if any(point is None for point in intersections):
        return ordered
    refined = order_points(np.array(intersections, dtype=np.float32))
    if not _is_reasonable_refined_quad(ordered, refined, image.shape):
        return ordered
    return refined


def _dominant_hue_mask(image: np.ndarray) -> np.ndarray | None:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    h, w = hue.shape
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.1), int(w * 0.9)
    region_hue = hue[y0:y1, x0:x1]
    region_sat = sat[y0:y1, x0:x1]
    region_val = val[y0:y1, x0:x1]

    keep = (region_sat > 50) & (region_val > 40)
    if int(keep.sum()) < 500:
        return None

    hist = np.bincount(region_hue[keep].ravel(), minlength=180)
    dominant = int(np.argmax(hist))
    hue_dist = np.minimum(
        np.abs(hue.astype(np.int16) - dominant),
        180 - np.abs(hue.astype(np.int16) - dominant),
    )
    mask = ((hue_dist <= 12) & (sat > 50) & (val > 40)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    return mask


def _fit_horizontal_line(
    edge_map: np.ndarray,
    left: int,
    right: int,
    row_min: int,
    row_max: int,
) -> tuple[float, float] | None:
    if right - left < 40 or row_max - row_min < 20:
        return None
    points: list[tuple[float, float]] = []
    xs = np.linspace(left, right, 10).astype(int)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        seg = edge_map[row_min:row_max, x0:x1]
        if seg.size == 0:
            continue
        profile = seg.mean(axis=1)
        idx = int(np.argmax(profile))
        points.append((((x0 + x1) / 2.0), float(row_min + idx)))
    if len(points) < 3:
        return None
    px = np.array([p[0] for p in points], dtype=np.float32)
    py = np.array([p[1] for p in points], dtype=np.float32)
    coeff = np.polyfit(px, py, 1)
    return float(coeff[0]), float(coeff[1])


def _fit_vertical_line(
    edge_map: np.ndarray,
    col_min: int,
    col_max: int,
    top: int,
    bottom: int,
) -> tuple[float, float] | None:
    if col_max - col_min < 20 or bottom - top < 40:
        return None
    points: list[tuple[float, float]] = []
    ys = np.linspace(top, bottom, 10).astype(int)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        seg = edge_map[y0:y1, col_min:col_max]
        if seg.size == 0:
            continue
        profile = seg.mean(axis=0)
        idx = int(np.argmax(profile))
        points.append((float(col_min + idx), ((y0 + y1) / 2.0)))
    if len(points) < 3:
        return None
    py = np.array([p[1] for p in points], dtype=np.float32)
    px = np.array([p[0] for p in points], dtype=np.float32)
    coeff = np.polyfit(py, px, 1)
    return float(coeff[0]), float(coeff[1])


def _intersect_hv(hline: tuple[float, float], vline: tuple[float, float]) -> tuple[float, float]:
    hm, hb = hline
    vm, vb = vline
    # y = hm*x + hb ; x = vm*y + vb
    denom = 1.0 - (vm * hm)
    if abs(denom) < 1e-6:
        x = vb
        y = hm * x + hb
        return x, y
    x = (vm * hb + vb) / denom
    y = hm * x + hb
    return x, y


def _quad_from_bounds(
    h_top: tuple[float, float],
    h_bottom: tuple[float, float],
    v_left: tuple[float, float],
    v_right: tuple[float, float],
) -> np.ndarray:
    quad = np.array(
        [
            _intersect_hv(h_top, v_left),
            _intersect_hv(h_top, v_right),
            _intersect_hv(h_bottom, v_right),
            _intersect_hv(h_bottom, v_left),
        ],
        dtype=np.float32,
    )
    return order_points(quad)


def _detect_quad_by_contours(image: np.ndarray) -> np.ndarray | None:
    small, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates: list[tuple[float, np.ndarray]] = []
    for t1, t2 in [(50, 150), (30, 120), (75, 200)]:
        edges = cv2.Canny(gray, t1, t2)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.erode(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < small.shape[0] * small.shape[1] * 0.08:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            pts = order_points(approx.reshape(4, 2))
            widths = [
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[3]),
            ]
            heights = [
                np.linalg.norm(pts[3] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
            ]
            height = float(np.mean(heights))
            if height <= 1:
                continue
            aspect = float(np.mean(widths)) / height
            if not 1.15 <= aspect <= 2.5:
                continue
            x, y, wc, hc = cv2.boundingRect(approx)
            border_touch = (
                x <= 5
                or y <= 5
                or x + wc >= small.shape[1] - 5
                or y + hc >= small.shape[0] - 5
            )
            score = area * (1.25 if not border_touch else 0.9) * min(aspect, 2.0)
            candidates.append((score, pts))
    if not candidates:
        return None
    quad = max(candidates, key=lambda item: item[0])[1]
    if scale != 1.0:
        quad = quad / scale
    return order_points(quad)


def _detect_document_quad(image: np.ndarray) -> np.ndarray | None:
    small, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        -7,
    )
    bright = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_and(adaptive, bright)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_quad: np.ndarray | None = None
    best_score = -1.0
    image_area = float(small.shape[0] * small.shape[1])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            continue
        quad = _extract_quad_from_contour(contour)
        if quad is None:
            continue
        local_score = _candidate_geometry_score(quad, small.shape)
        if local_score > best_score:
            best_score = local_score
            best_quad = quad

    if best_quad is None:
        return None
    return _restore_scale(best_quad, scale)


def _segment_grabcut_mask(image: np.ndarray, rect: tuple[int, int, int, int], iterations: int) -> np.ndarray:
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _detect_lsd_grabcut_quad(
    image: np.ndarray,
    profile: dict[str, float] | None = None,
) -> np.ndarray | None:
    active = profile or load_opencv_profile()
    small, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=float(active["clahe_clip_limit"]),
        tileGridSize=(int(active["clahe_grid_size"]), int(active["clahe_grid_size"])),
    )
    enhanced = clahe.apply(gray)
    lsd = cv2.createLineSegmentDetector(
        scale=float(active["lsd_scale"]),
        sigma_scale=float(active["lsd_sigma_scale"]),
        quant=float(active["lsd_quant"]),
        ang_th=float(active["lsd_ang_th"]),
    )
    lines = lsd.detect(enhanced)[0]
    if lines is None:
        return None

    segments: list[tuple[float, np.ndarray]] = []
    for entry in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in entry]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min(small.shape[:2]) * 0.14:
            continue
        segments.append((length, np.array([x1, y1, x2, y2], dtype=np.float32)))
    if len(segments) < 2:
        return None

    segments.sort(key=lambda item: item[0], reverse=True)
    xs: list[float] = []
    ys: list[float] = []
    for _, seg in segments[:18]:
        xs.extend([float(seg[0]), float(seg[2])])
        ys.extend([float(seg[1]), float(seg[3])])
    x0, x1 = max(0, int(min(xs))), min(small.shape[1] - 1, int(max(xs)))
    y0, y1 = max(0, int(min(ys))), min(small.shape[0] - 1, int(max(ys)))
    if x1 - x0 < small.shape[1] * 0.18 or y1 - y0 < small.shape[0] * 0.16:
        return None

    expand = float(active["roi_expand_ratio"])
    pad_x = int((x1 - x0) * expand)
    pad_y = int((y1 - y0) * expand)
    rx0 = max(0, x0 - pad_x)
    ry0 = max(0, y0 - pad_y)
    rx1 = min(small.shape[1] - 1, x1 + pad_x)
    ry1 = min(small.shape[0] - 1, y1 + pad_y)
    rect = (rx0, ry0, max(1, rx1 - rx0), max(1, ry1 - ry0))

    try:
        mask = _segment_grabcut_mask(small, rect, int(active["grabcut_iters"]))
    except cv2.error:
        return None

    close_kernel = np.ones((int(active["mask_close_kernel"]), int(active["mask_close_kernel"])), np.uint8)
    open_kernel = np.ones((int(active["mask_open_kernel"]), int(active["mask_open_kernel"])), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_quad: np.ndarray | None = None
    best_score = -1.0
    image_area = float(small.shape[0] * small.shape[1])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.05:
            continue
        quad = _extract_quad_from_contour(contour)
        if quad is None:
            continue
        score = _candidate_geometry_score(quad, small.shape)
        if score > best_score:
            best_score = score
            best_quad = quad

    if best_quad is None:
        return None
    floor_limit = ry0 + (ry1 - ry0) * 0.88
    floor_overflow = max(float(best_quad[2][1]), float(best_quad[3][1])) - floor_limit
    if floor_overflow > max(18.0, (ry1 - ry0) * 0.05):
        best_quad = _clamp_quad_bottom(best_quad, floor_limit)
    return _restore_scale(best_quad, scale)


def _detect_colored_screen(image: np.ndarray) -> np.ndarray | None:
    small, scale = _resize_for_detection(image)
    mask = _dominant_hue_mask(small)
    if mask is None:
        return None
    h, w = mask.shape
    band = mask[int(h * 0.25):int(h * 0.75), :]
    col_frac = band.mean(axis=0)
    cols = np.where(col_frac > 0.45)[0]
    if len(cols) < w * 0.35:
        return None
    left, right = int(cols[0]), int(cols[-1])
    row_frac = mask[:, left:right + 1].mean(axis=1)
    rows = np.where(row_frac > 0.35)[0]
    if len(rows) < h * 0.2:
        return None
    top, bottom = int(rows[0]), int(rows[-1])
    quad = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float32,
    )
    if scale != 1.0:
        quad[:, 0] /= scale
        quad[:, 1] /= scale
    return order_points(quad)


def _detect_refined_edges(image: np.ndarray) -> np.ndarray | None:
    small, scale = _resize_for_detection(image)
    mask = _dominant_hue_mask(small)
    if mask is None:
        return None

    h, w = mask.shape
    band = mask[int(h * 0.2):int(h * 0.8), :]
    cols = np.where(band.mean(axis=0) > 0.45)[0]
    if len(cols) < w * 0.35:
        return None
    left0, right0 = int(cols[0]), int(cols[-1])

    row_frac = mask[:, left0:right0 + 1].mean(axis=1)
    rows = np.where(row_frac > 0.35)[0]
    if len(rows) < h * 0.2:
        return None
    top0, bottom0 = int(rows[0]), int(rows[-1])

    dx, dy, _ = _edge_maps(small)
    dx = np.abs(dx)
    dy = np.abs(dy)
    height = bottom0 - top0

    left_line = _fit_vertical_line(
        dx,
        max(0, left0 - 80),
        min(w, left0 + 100),
        max(0, top0 - 30),
        min(h, bottom0 + 20),
    )
    right_line = _fit_vertical_line(
        dx,
        max(0, right0 - 100),
        min(w, right0 + 80),
        max(0, top0 - 30),
        min(h, bottom0 + 20),
    )
    top_line = _fit_horizontal_line(
        dy,
        max(0, left0 - 10),
        min(w, right0 + 10),
        max(0, top0 - 60),
        min(h, top0 + 60),
    )
    bottom_line = _fit_horizontal_line(
        dy,
        max(0, left0 - 10),
        min(w, right0 + 10),
        max(top0 + int(height * 0.45), bottom0 - 120),
        min(h, bottom0 + 20),
    )
    if not all([left_line, right_line, top_line, bottom_line]):
        return None

    quad = _quad_from_bounds(top_line, bottom_line, left_line, right_line)
    if scale != 1.0:
        quad[:, 0] /= scale
        quad[:, 1] /= scale
    return order_points(quad)


def _detect_bright_screen(image: np.ndarray) -> np.ndarray | None:
    small, scale = _resize_for_detection(image)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    bright_mask = (
        (hsv[:, :, 2] > 155)
        & ((hsv[:, :, 1] < 95) | (hsv[:, :, 2] > 220))
    ).astype(np.uint8)
    bright_mask = cv2.morphologyEx(
        bright_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2
    )
    bright_mask = cv2.morphologyEx(
        bright_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1
    )

    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_quad: np.ndarray | None = None
    best_score = 0.0
    image_area = float(small.shape[0] * small.shape[1])
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.1:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.float32)
        ordered = order_points(box)
        width = float((np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])) / 2.0)
        height = float((np.linalg.norm(ordered[3] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[1])) / 2.0)
        if min(width, height) < 80:
            continue
        aspect = width / max(height, 1.0)
        if not 1.1 <= aspect <= 2.6:
            continue
        score = area / image_area
        if score > best_score:
            best_score = score
            best_quad = ordered

    if best_quad is None:
        return None
    if scale != 1.0:
        best_quad[:, 0] /= scale
        best_quad[:, 1] /= scale
    return order_points(best_quad)


def _detect_hough_screen(image: np.ndarray) -> np.ndarray | None:
    small, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=120,
        minLineLength=int(min(small.shape[:2]) * 0.25),
        maxLineGap=30,
    )
    if lines is None:
        return None

    horizontals: list[np.ndarray] = []
    verticals: list[np.ndarray] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        length = float(np.hypot(dx, dy))
        if length < min(small.shape[:2]) * 0.2:
            continue
        if angle < 20 or angle > 160:
            horizontals.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        elif 70 < angle < 110:
            verticals.append(np.array([x1, y1, x2, y2], dtype=np.float32))

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    top_line = min(horizontals, key=lambda line: (line[1] + line[3]) / 2.0)
    bottom_line = max(horizontals, key=lambda line: (line[1] + line[3]) / 2.0)
    left_line = min(verticals, key=lambda line: (line[0] + line[2]) / 2.0)
    right_line = max(verticals, key=lambda line: (line[0] + line[2]) / 2.0)

    def fit_h(line: np.ndarray) -> tuple[float, float]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        if abs(dx) < 1e-6:
            return 0.0, y1
        m = (y2 - y1) / dx
        return float(m), float(y1 - m * x1)

    def fit_v(line: np.ndarray) -> tuple[float, float]:
        x1, y1, x2, y2 = line
        dy = y2 - y1
        if abs(dy) < 1e-6:
            return 0.0, x1
        m = (x2 - x1) / dy
        return float(m), float(x1 - m * y1)

    quad = _quad_from_bounds(
        fit_h(top_line),
        fit_h(bottom_line),
        fit_v(left_line),
        fit_v(right_line),
    )
    if scale != 1.0:
        quad[:, 0] /= scale
        quad[:, 1] /= scale
    return order_points(quad)


def _fit_line_from_segment(line: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in line]
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return 0.0, y1
    m = (y2 - y1) / dx
    return float(m), float(y1 - m * x1)


def _fit_vertical_from_segment(line: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in line]
    dy = y2 - y1
    if abs(dy) < 1e-6:
        return 0.0, x1
    m = (x2 - x1) / dy
    return float(m), float(x1 - m * y1)


def _detect_line_fusion_quad(image: np.ndarray) -> np.ndarray | None:
    small, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=90,
        minLineLength=int(min(small.shape[:2]) * 0.18),
        maxLineGap=45,
    )
    if lines is None:
        return None

    horizontals: list[np.ndarray] = []
    verticals: list[np.ndarray] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min(small.shape[:2]) * 0.18:
            continue
        if angle < 28 or angle > 152:
            horizontals.append(np.array(line, dtype=np.float32))
        elif 62 < angle < 118:
            verticals.append(np.array(line, dtype=np.float32))

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    horizontals.sort(key=lambda line: float(np.hypot(line[2] - line[0], line[3] - line[1])), reverse=True)
    verticals.sort(key=lambda line: float(np.hypot(line[2] - line[0], line[3] - line[1])), reverse=True)
    top_candidates = sorted(horizontals[:8], key=lambda line: (line[1] + line[3]) / 2.0)[:3]
    bottom_candidates = sorted(horizontals[:8], key=lambda line: (line[1] + line[3]) / 2.0)[-3:]
    left_candidates = sorted(verticals[:8], key=lambda line: (line[0] + line[2]) / 2.0)[:3]
    right_candidates = sorted(verticals[:8], key=lambda line: (line[0] + line[2]) / 2.0)[-3:]

    best_quad: np.ndarray | None = None
    best_score = -1.0
    for top_line in top_candidates:
        for bottom_line in bottom_candidates:
            if (bottom_line[1] + bottom_line[3]) <= (top_line[1] + top_line[3]):
                continue
            for left_line in left_candidates:
                for right_line in right_candidates:
                    if (right_line[0] + right_line[2]) <= (left_line[0] + left_line[2]):
                        continue
                    quad = _quad_from_bounds(
                        _fit_line_from_segment(top_line),
                        _fit_line_from_segment(bottom_line),
                        _fit_vertical_from_segment(left_line),
                        _fit_vertical_from_segment(right_line),
                    )
                    score = _candidate_geometry_score(quad, small.shape)
                    if score > best_score:
                        best_score = score
                        best_quad = quad

    if best_quad is None:
        return None
    return _restore_scale(best_quad, scale)


def _detect_roi_guided_quad(image: np.ndarray) -> np.ndarray | None:
    small, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    detail = cv2.subtract(gray, cv2.GaussianBlur(gray, (81, 81), 0))
    roi_signal = cv2.normalize(gray.astype(np.float32) * 0.7 + detail.astype(np.float32) * 1.6, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    center_x = (xx - w / 2.0) / max(w / 2.0, 1.0)
    center_y = (yy - h / 2.0) / max(h / 2.0, 1.0)
    center_prior = np.clip(1.2 - (center_x ** 2 * 0.9 + center_y ** 2 * 1.4), 0.0, 1.2)
    guided = np.clip(roi_signal.astype(np.float32) * center_prior, 0.0, 255.0).astype(np.uint8)
    _, mask = cv2.threshold(guided, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_quad: np.ndarray | None = None
    best_score = -1.0
    for contour in contours:
        if cv2.contourArea(contour) < h * w * 0.06:
            continue
        quad = _extract_quad_from_contour(contour)
        if quad is None:
            continue
        score = _candidate_geometry_score(quad, small.shape)
        if score > best_score:
            best_score = score
            best_quad = quad

    if best_quad is None:
        return None
    return _restore_scale(best_quad, scale)


def _target_size(quad: np.ndarray) -> tuple[int, int, float]:
    width = float((np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2.0)
    height = float((np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2.0)
    aspect = width / max(height, 1.0)
    target_ratio = 16 / 9
    if abs(aspect - (4 / 3)) < abs(aspect - target_ratio):
        target_ratio = 4 / 3
    target_width = 1600
    target_height = int(round(target_width / target_ratio))
    return target_width, target_height, target_ratio


def _warp_image(image: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, float]:
    target_width, target_height, target_ratio = _target_size(quad)
    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (target_width, target_height))
    return warped, target_ratio


def _dominant_fraction(arr: np.ndarray) -> float:
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    keep = (sat > 50) & (val > 40)
    if int(keep.sum()) < 20:
        return 0.0
    hist = np.bincount(hue[keep].ravel(), minlength=180)
    dominant = int(np.argmax(hist))
    hue_dist = np.minimum(
        np.abs(hue.astype(np.int16) - dominant),
        180 - np.abs(hue.astype(np.int16) - dominant),
    )
    return float(((hue_dist <= 12) & keep).mean())


def _border_diagnostics(warped: np.ndarray) -> dict[str, float]:
    h, w = warped.shape[:2]
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    border_h = max(8, h // 14)
    border_w = max(8, w // 18)

    top = warped[:border_h, :, :]
    bottom = warped[h - border_h :, :, :]
    left = warped[:, :border_w, :]
    right = warped[:, w - border_w :, :]

    def edge_mean(img: np.ndarray) -> float:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return float(cv2.magnitude(sx, sy).mean())

    def blue_frac(arr: np.ndarray) -> float:
        ahsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        return float(
            (
                (ahsv[:, :, 0] >= 95)
                & (ahsv[:, :, 0] <= 135)
                & (ahsv[:, :, 1] > 80)
                & (ahsv[:, :, 2] > 60)
            ).mean()
        )

    def floor_frac(arr: np.ndarray) -> float:
        ahsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        gray_local = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        return float(
            (
                (ahsv[:, :, 1] > 45)
                & (ahsv[:, :, 2] > 70)
                & (gray_local > 70)
            ).mean()
        )

    return {
        "top_dom": round(_dominant_fraction(top), 4),
        "left_dom": round(_dominant_fraction(left), 4),
        "right_dom": round(_dominant_fraction(right), 4),
        "bottom_dom": round(_dominant_fraction(bottom), 4),
        "top_edge": round(edge_mean(top), 4),
        "left_edge": round(edge_mean(left), 4),
        "right_edge": round(edge_mean(right), 4),
        "bottom_edge": round(edge_mean(bottom), 4),
        "blue_left": round(blue_frac(left), 4),
        "blue_right": round(blue_frac(right), 4),
        "floor_bottom": round(floor_frac(bottom), 4),
    }


def _adjust_quad(
    quad: np.ndarray,
    *,
    left: float = 0.0,
    right: float = 0.0,
    top: float = 0.0,
    bottom: float = 0.0,
) -> np.ndarray:
    tl, tr, br, bl = [pt.astype(np.float32) for pt in order_points(quad)]
    if left:
        tl = tl + (tr - tl) * left
        bl = bl + (br - bl) * left
    if right:
        tr = tr + (tl - tr) * right
        br = br + (bl - br) * right
    if top:
        tl = tl + (bl - tl) * top
        tr = tr + (br - tr) * top
    if bottom:
        bl = bl + (tl - bl) * bottom
        br = br + (tr - br) * bottom
    return order_points(np.array([tl, tr, br, bl], dtype=np.float32))


def _cleanup_candidate(image: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    warped, _ = _warp_image(image, quad)
    diag = _border_diagnostics(warped)

    mean_dom = (diag["top_dom"] + diag["left_dom"] + diag["right_dom"] + diag["bottom_dom"]) / 4.0
    if mean_dom < 0.68:
        return None

    left_expand = 0.0
    top_expand = 0.0
    right_shrink = 0.0
    bottom_shrink = 0.0

    if diag["left_dom"] > 0.72:
        left_expand = -min(0.03, 0.015 + (diag["left_dom"] - 0.72) * 0.06)
    if diag["top_dom"] > 0.66:
        top_expand = -min(0.045, 0.025 + (diag["top_dom"] - 0.66) * 0.08)
    if diag["blue_right"] > 0.08:
        right_shrink = min(0.08, 0.02 + diag["blue_right"] * 0.15)
    if diag["blue_left"] > 0.08:
        left_expand += min(0.04, diag["blue_left"] * 0.1)
    if diag["floor_bottom"] > 0.55:
        bottom_shrink = min(0.12, 0.04 + (diag["floor_bottom"] - 0.55) * 0.16)

    if max(abs(left_expand), abs(top_expand), abs(right_shrink), abs(bottom_shrink)) < 0.015:
        return None

    return _adjust_quad(
        quad,
        left=left_expand,
        right=right_shrink,
        top=top_expand,
        bottom=bottom_shrink,
    )


def _score_candidate(
    image: np.ndarray,
    quad: np.ndarray,
    method: str,
    profile: dict[str, dict[str, float]] | None = None,
    base_method: str | None = None,
) -> tuple[float, dict[str, float]]:
    quad = order_points(quad)
    warped, target_ratio = _warp_image(image, quad)
    h, w = warped.shape[:2]
    diagnostics = _border_diagnostics(warped)

    width = float((np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2.0)
    height = float((np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2.0)
    aspect = width / max(height, 1.0)
    aspect_score = max(0.0, 1.0 - (abs(aspect - target_ratio) / target_ratio) * 3.0)
    width_balance = 1.0 - min(
        1.0,
        abs(np.linalg.norm(quad[1] - quad[0]) - np.linalg.norm(quad[2] - quad[3]))
        / max(width, 1.0),
    )
    height_balance = 1.0 - min(
        1.0,
        abs(np.linalg.norm(quad[3] - quad[0]) - np.linalg.norm(quad[2] - quad[1]))
        / max(height, 1.0),
    )
    symmetry_score = max(0.0, 0.5 * width_balance + 0.5 * height_balance)

    top_angle = abs(np.arctan2(quad[1][1] - quad[0][1], quad[1][0] - quad[0][0]))
    bottom_angle = abs(np.arctan2(quad[2][1] - quad[3][1], quad[2][0] - quad[3][0]))
    left_angle = abs(np.arctan2(quad[3][0] - quad[0][0], quad[3][1] - quad[0][1]))
    right_angle = abs(np.arctan2(quad[2][0] - quad[1][0], quad[2][1] - quad[1][1]))
    parallel_score = max(
        0.0,
        1.0
        - min(1.0, abs(top_angle - bottom_angle) / 0.35) * 0.5
        - min(1.0, abs(left_angle - right_angle) / 0.35) * 0.5,
    )

    contour_area = cv2.contourArea(quad.astype(np.float32))
    min_rect = cv2.minAreaRect(quad.astype(np.float32))
    min_rect_area = max(float(min_rect[1][0] * min_rect[1][1]), 1.0)
    rectangularity_score = max(0.0, min(1.0, contour_area / min_rect_area))

    quad_center = quad.mean(axis=0)
    image_center = np.array([image.shape[1] / 2.0, image.shape[0] / 2.0], dtype=np.float32)
    center_distance = float(np.linalg.norm(quad_center - image_center))
    max_center_distance = float(np.linalg.norm(image_center)) or 1.0
    center_score = max(0.0, 1.0 - center_distance / max_center_distance)

    edge_score = min(
        1.0,
        (
            diagnostics["top_edge"]
            + diagnostics["bottom_edge"]
            + diagnostics["left_edge"]
            + diagnostics["right_edge"]
        )
        / (4.0 * 42.0),
    )

    blue_penalty = float(0.5 * diagnostics["blue_left"] + 0.5 * diagnostics["blue_right"])
    top_dark_penalty = max(0.0, 0.45 - diagnostics["top_dom"])
    floor_penalty = float(diagnostics["floor_bottom"])
    spill_penalty = float(
        max(0.0, diagnostics["top_dom"] - 0.78) * 0.8
        + max(0.0, diagnostics["left_dom"] - 0.78) * 0.7
    )

    coverage = cv2.contourArea(quad.astype(np.float32)) / float(image.shape[0] * image.shape[1])
    coverage_score = max(0.0, 1.0 - abs(coverage - 0.52) / 0.18)

    metrics = {
        "aspect_score": round(aspect_score, 4),
        "symmetry_score": round(float(symmetry_score), 4),
        "parallel_score": round(float(parallel_score), 4),
        "rectangularity_score": round(float(rectangularity_score), 4),
        "center_score": round(float(center_score), 4),
        "edge_score": round(float(edge_score), 4),
        "coverage_score": round(float(coverage_score), 4),
        "blue_penalty": round(float(blue_penalty), 4),
        "top_dark_penalty": round(float(top_dark_penalty), 4),
        "floor_penalty": round(float(floor_penalty), 4),
        "spill_penalty": round(float(spill_penalty), 4),
        "coverage": round(float(coverage), 4),
        "aspect": round(float(aspect), 4),
        **diagnostics,
    }
    score = combine_score_from_metrics(metrics, profile, method, base_method=base_method)
    return float(round(score, 4)), metrics


def _dedupe_candidates(candidates: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    unique: list[tuple[str, np.ndarray]] = []
    for method, quad in candidates:
        q = order_points(quad)
        keep = True
        for _, existing in unique:
            if float(np.mean(np.linalg.norm(q - existing, axis=1))) < 12.0:
                keep = False
                break
        if keep:
            unique.append((method, q))
    return unique


def _build_lsd_v2_profile(profile: dict[str, float] | None) -> dict[str, float]:
    active = dict(profile or load_opencv_profile())
    active["roi_expand_ratio"] = 0.1
    active["grabcut_iters"] = max(int(active.get("grabcut_iters", 3)), 4)
    active["mask_close_kernel"] = 7
    active["mask_open_kernel"] = 3
    return active


def _collect_base_candidates(
    image: np.ndarray,
    opencv_profile: dict[str, float] | None,
) -> list[tuple[str, np.ndarray]]:
    del opencv_profile
    raw_candidates: list[tuple[str, np.ndarray]] = []
    for method, fn in [
        ("document_quad", _detect_document_quad),
        ("contour_quad", _detect_quad_by_contours),
    ]:
        quad = fn(image)
        if quad is not None:
            raw_candidates.append((method, quad))
    return _dedupe_candidates(_filter_candidate_pairs(raw_candidates))


def _build_candidate_variants(
    image: np.ndarray,
    raw_candidates: list[tuple[str, np.ndarray]],
) -> list[tuple[str, np.ndarray, str]]:
    variants: list[tuple[str, np.ndarray, str]] = []
    for method, quad in raw_candidates:
        if not _is_allowed_opencv_method(method):
            continue
        ordered = order_points(quad)
        variants.append((method, ordered, method))
        refined = _refine_quad_by_edge_alignment(image, ordered)
        if float(np.mean(np.linalg.norm(refined - ordered, axis=1))) >= 1.5:
            refined_method = f"{method}_edge"
            if _is_allowed_opencv_method(refined_method):
                variants.append((refined_method, refined, method))
    return _filter_variant_candidates(variants)


def _should_enable_lsd_v2(candidates: list[Candidate], confidence: float) -> bool:
    if not candidates:
        return False
    best = candidates[0]
    floor_penalty = float(best.metrics.get("floor_penalty", 0.0))
    spill_penalty = float(best.metrics.get("spill_penalty", 0.0))
    low_confidence = float(confidence) < 0.06
    bad_bottom = floor_penalty > 0.45
    bad_spill = spill_penalty > 0.12
    return low_confidence and (bad_bottom or bad_spill)


def _select_problem_scene_candidate(candidates: list[Candidate], confidence: float) -> Candidate:
    best = candidates[0]
    best_aspect = float(best.metrics.get("aspect_score", 0.0))
    best_floor = float(best.metrics.get("floor_penalty", 0.0))
    best_spill = float(best.metrics.get("spill_penalty", 0.0))
    best_coverage = float(best.metrics.get("coverage_score", 0.0))
    should_rescue = _should_enable_lsd_v2(candidates, confidence)
    should_rescue = should_rescue or (
        best.method == "refined_edges" and confidence < 0.09 and best_aspect < 0.72
    )
    should_rescue = should_rescue or (
        best.method == "bright_screen" and confidence < 0.02 and best_floor > 0.8
    )
    if not should_rescue:
        return best

    if best.method == "refined_edges" and best_aspect < 0.72:
        pool: list[Candidate] = []
        for candidate in candidates[1:]:
            if candidate.method not in {
                "bright_screen",
                "roi_guided_quad",
                "lsd_grabcut_quad",
                "lsd_grabcut_quad_v2",
            }:
                continue
            if candidate.score < best.score - 0.14:
                continue
            coverage = float(candidate.metrics.get("coverage_score", 0.0))
            aspect = float(candidate.metrics.get("aspect_score", 0.0))
            floor = float(candidate.metrics.get("floor_penalty", 0.0))
            spill = float(candidate.metrics.get("spill_penalty", 0.0))
            if candidate.method.startswith("lsd_grabcut_quad"):
                if aspect < best_aspect + 0.18:
                    continue
                if floor > best_floor - 0.08:
                    continue
                if spill > best_spill + 0.04:
                    continue
                pool.append(candidate)
                continue
            if coverage < 0.28:
                continue
            if aspect < best_aspect + 0.08:
                continue
            if floor > best_floor + 0.02:
                continue
            if spill > best_spill + 0.03:
                continue
            pool.append(candidate)
        if pool:
            pool.sort(
                key=lambda c: (
                    c.score
                    + (float(c.metrics.get("aspect_score", 0.0)) - best_aspect) * 0.3
                    + (best_floor - float(c.metrics.get("floor_penalty", 0.0))) * 0.1
                ),
                reverse=True,
            )
            return pool[0]

    if best.method == "bright_screen" and best_aspect < 0.75 and best_coverage > 0.9:
        pool = []
        for candidate in candidates[1:]:
            if candidate.method not in {"roi_guided_quad", "lsd_grabcut_quad_v2", "document_quad"}:
                continue
            if candidate.score < best.score - 0.09:
                continue
            coverage = float(candidate.metrics.get("coverage_score", 0.0))
            aspect = float(candidate.metrics.get("aspect_score", 0.0))
            floor = float(candidate.metrics.get("floor_penalty", 0.0))
            spill = float(candidate.metrics.get("spill_penalty", 0.0))
            if candidate.method == "lsd_grabcut_quad_v2":
                if candidate.score < best.score - 0.02:
                    continue
                if aspect < best_aspect + 0.05:
                    continue
                if floor > best_floor - 0.2:
                    continue
                if spill > best_spill + 0.12:
                    continue
                pool.append(candidate)
                continue
            if coverage < 0.28:
                continue
            if aspect < 0.9:
                continue
            if floor > best_floor + 0.03:
                continue
            pool.append(candidate)
        if pool:
            pool.sort(
                key=lambda c: (c.score, float(c.metrics.get("aspect_score", 0.0))),
                reverse=True,
            )
            return pool[0]

    if best.method == "bright_screen" and confidence < 0.02 and best_floor > 0.8:
        pool = []
        for candidate in candidates[1:]:
            if candidate.method != "lsd_grabcut_quad_v2":
                continue
            if candidate.score < best.score - 0.02:
                continue
            aspect = float(candidate.metrics.get("aspect_score", 0.0))
            floor = float(candidate.metrics.get("floor_penalty", 0.0))
            spill = float(candidate.metrics.get("spill_penalty", 0.0))
            if aspect < best_aspect + 0.04:
                continue
            if floor > best_floor - 0.2:
                continue
            if spill > best_spill + 0.12:
                continue
            pool.append(candidate)
        if pool:
            pool.sort(
                key=lambda c: (
                    c.score,
                    float(c.metrics.get("aspect_score", 0.0)),
                    -float(c.metrics.get("floor_penalty", 0.0)),
                ),
                reverse=True,
            )
            return pool[0]

    return best


def detect_best_candidate(image: np.ndarray) -> dict[str, object] | None:
    return detect_best_candidate_with_profile(image, None)


def detect_best_candidate_with_profile(
    image: np.ndarray,
    opencv_profile: dict[str, float] | None,
) -> dict[str, object] | None:
    if image is None:
        return None

    raw_candidates = _filter_candidate_pairs(_collect_base_candidates(image, opencv_profile))
    if not raw_candidates:
        return None
    scored_candidates = _filter_variant_candidates(_build_candidate_variants(image, raw_candidates))

    profile = load_scoring_profile()
    candidates: list[Candidate] = []
    for method, quad, base_method in scored_candidates:
        score, metrics = _score_candidate(image, quad, method, profile, base_method=base_method)
        candidates.append(Candidate(method=method, quad=quad, metrics=metrics, score=score))

    candidates.sort(key=lambda item: item.score, reverse=True)
    confidence = candidates[0].score - candidates[1].score if len(candidates) > 1 else candidates[0].score

    if str(profile.get("selector_mode", "legacy_rescue")) == "score_only":
        best = candidates[0]
    else:
        best = _select_problem_scene_candidate(candidates, confidence)

    return {
        "best": {
            "method": best.method,
            "quad": best.quad,
            "metrics": best.metrics,
            "score": best.score,
            "confidence": round(float(confidence), 4),
            "refined": best.method.endswith("_edge"),
        },
        "candidates": [
            {
                "method": item.method,
                "quad": item.quad,
                "metrics": item.metrics,
                "score": item.score,
            }
            for item in candidates
        ],
    }


def detect_screen_quad(image: np.ndarray) -> np.ndarray | None:
    result = detect_best_candidate(image)
    if result is None:
        return None
    return result["best"]["quad"]
