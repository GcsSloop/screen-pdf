from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from dataset_benchmark import quad_geometry_metrics
from perspective_detect import order_points
from two_stage_corner_pipeline import (
    CANDIDATE_SELECTOR_FEATURE_NAMES,
    GlobalCornerPredictor,
    LinearCandidateExpandSelector,
    LocalCornerMoEPredictor,
    RoiCornerPredictor,
    _attach_candidate_selector_features,
    _run_two_stage_candidate,
)


@dataclass
class SelectorTrainingRow:
    features: np.ndarray
    label_index: int
    candidate_metrics: list[dict[str, float]]
    candidate_errors: list[float]
    page_id: str
    project_name: str
    baseline_index: int


def _load_rows(split_dir: Path, split: str) -> list[dict[str, Any]]:
    manifest_path = split_dir / f"{split}.jsonl"
    return [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_priority(metrics: dict[str, float]) -> tuple[int, int, int, float, float]:
    return (
        int(float(metrics.get("max_corner_error", 1.0) or 1.0) <= 0.03),
        int(float(metrics.get("point_error", 1.0) or 1.0) <= 0.01),
        int(float(metrics.get("max_corner_error", 1.0) or 1.0) <= 0.01),
        -float(metrics.get("point_error", 1.0) or 1.0),
        -float(metrics.get("max_corner_error", 1.0) or 1.0),
    )


def _select_training_label(
    candidate_metrics: list[dict[str, float]],
    *,
    baseline_index: int,
    preserve_point_margin: float,
    preserve_max_margin: float,
) -> int:
    if not candidate_metrics:
        raise ValueError("candidate_metrics is empty")
    best_index = max(range(len(candidate_metrics)), key=lambda index: _candidate_priority(candidate_metrics[index]))
    if best_index == baseline_index:
        return baseline_index
    baseline_metrics = candidate_metrics[baseline_index]
    best_metrics = candidate_metrics[best_index]
    if _candidate_priority(best_metrics)[:3] > _candidate_priority(baseline_metrics)[:3]:
        return best_index
    point_gain = float(baseline_metrics.get("point_error", 1.0) or 1.0) - float(best_metrics.get("point_error", 1.0) or 1.0)
    max_gain = float(baseline_metrics.get("max_corner_error", 1.0) or 1.0) - float(best_metrics.get("max_corner_error", 1.0) or 1.0)
    if point_gain >= float(preserve_point_margin) and max_gain >= float(preserve_max_margin):
        return best_index
    return baseline_index


def _candidate_training_utility(
    metrics: dict[str, float],
    *,
    baseline_metrics: dict[str, float] | None = None,
) -> float:
    point_error = float(metrics.get("point_error", 1.0) or 1.0)
    max_corner_error = float(metrics.get("max_corner_error", 1.0) or 1.0)
    utility = 0.0
    utility += 6.0 if max_corner_error <= 0.03 else 0.0
    utility += 3.0 if point_error <= 0.01 else 0.0
    utility += 1.0 if max_corner_error <= 0.01 else 0.0
    utility -= point_error * 120.0
    utility -= max_corner_error * 40.0
    if baseline_metrics is not None:
        baseline_point = float(baseline_metrics.get("point_error", 1.0) or 1.0)
        baseline_max = float(baseline_metrics.get("max_corner_error", 1.0) or 1.0)
        utility += (baseline_point - point_error) * 160.0
        utility += (baseline_max - max_corner_error) * 60.0
    return float(utility)


def _row_pairwise_preferences(row: SelectorTrainingRow) -> list[tuple[int, int, float]]:
    if not row.candidate_metrics:
        return []
    baseline_metrics = row.candidate_metrics[row.baseline_index]
    utilities = [
        _candidate_training_utility(metrics, baseline_metrics=baseline_metrics)
        for metrics in row.candidate_metrics
    ]
    pairs: list[tuple[int, int, float]] = []
    winner = int(row.label_index)
    for loser in range(len(utilities)):
        if winner == loser:
            continue
        margin = float(utilities[winner] - utilities[loser])
        if margin <= 1e-6:
            continue
        pairs.append((winner, loser, margin))
    return pairs


def _infer_project_name(row: dict[str, Any], image_path: Path) -> str:
    for key in ("project_name", "project", "dataset_project", "dataset_name", "scene_name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return image_path.parent.name or image_path.parent.as_posix()


def _build_project_sample_weights(
    rows: list[SelectorTrainingRow],
    *,
    balance_power: float,
) -> dict[str, float]:
    counts = Counter(row.project_name for row in rows)
    if not counts:
        return {}
    raw_weights = {
        project_name: 1.0 / max(float(count), 1.0) ** max(float(balance_power), 0.0)
        for project_name, count in counts.items()
    }
    row_mean = float(np.mean([raw_weights[row.project_name] for row in rows]))
    if row_mean <= 1e-9:
        return {project_name: 1.0 for project_name in counts}
    return {project_name: float(weight / row_mean) for project_name, weight in raw_weights.items()}


def _normalize_quad(quad: list[list[float]] | np.ndarray, width: int, height: int) -> np.ndarray:
    arr = order_points(np.array(quad, dtype=np.float32))
    out = arr.copy()
    out[:, 0] /= max(float(width), 1.0)
    out[:, 1] /= max(float(height), 1.0)
    return out


def build_selector_training_rows(
    *,
    split_dir: Path,
    split: str,
    global_model_path: Path,
    roi_model_path: Path,
    local_model_path: Path,
    candidate_expand_ratios: list[float],
    baseline_expand_ratio: float,
    preserve_point_margin: float = 0.0015,
    preserve_max_margin: float = 0.0025,
) -> list[SelectorTrainingRow]:
    rows = _load_rows(split_dir, split)
    global_predictor = GlobalCornerPredictor(global_model_path)
    roi_predictor = RoiCornerPredictor(roi_model_path)
    local_predictor = LocalCornerMoEPredictor(local_model_path)
    training_rows: list[SelectorTrainingRow] = []
    for row in rows:
        image_path = Path(row["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        coarse_quad = order_points(np.array(global_predictor.predict_image(image), dtype=np.float32))
        manual_quad = _normalize_quad(row["manual_quad"], image.shape[1], image.shape[0])
        candidate_items: list[dict[str, Any]] = []
        candidate_errors: list[float] = []
        candidate_metrics: list[dict[str, float]] = []
        for ratio in candidate_expand_ratios:
            candidate = _run_two_stage_candidate(
                image_path=image_path,
                page_id=str(row.get("page_id") or image_path.stem),
                image=image,
                coarse_quad=coarse_quad,
                roi_predictor=roi_predictor,
                local_predictor=local_predictor,
                expand_ratio=float(ratio),
            )
            final_quad_norm = _normalize_quad(candidate["final_quad"], image.shape[1], image.shape[0])
            metrics = quad_geometry_metrics(manual_quad, final_quad_norm)
            candidate_errors.append(float(metrics["point_error"]))
            candidate_metrics.append({key: float(value) for key, value in metrics.items()})
            candidate_items.append(candidate)
        if not candidate_items:
            continue
        _attach_candidate_selector_features(
            candidate_items,
            coarse_quad,
            image.shape,
            baseline_expand_ratio=float(baseline_expand_ratio),
        )
        baseline_index = next(
            (
                index
                for index, item in enumerate(candidate_items)
                if abs(float(item["expand_ratio"]) - float(baseline_expand_ratio)) <= 1e-9
            ),
            0,
        )
        label_index = _select_training_label(
            candidate_metrics,
            baseline_index=baseline_index,
            preserve_point_margin=float(preserve_point_margin),
            preserve_max_margin=float(preserve_max_margin),
        )
        feature_matrix = np.array([item["selector_features"] for item in candidate_items], dtype=np.float32)
        training_rows.append(
            SelectorTrainingRow(
                features=feature_matrix,
                label_index=label_index,
                candidate_metrics=candidate_metrics,
                candidate_errors=candidate_errors,
                page_id=str(row.get("page_id") or image_path.stem),
                project_name=_infer_project_name(row, image_path),
                baseline_index=baseline_index,
            )
        )
    return training_rows


def train_linear_selector(
    rows: list[SelectorTrainingRow],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    balance_power: float = 1.0,
    switch_margin: float = 0.0,
    objective_mode: str = "cross_entropy",
    pairwise_weight: float = 0.5,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no selector training rows")
    feature_dim = int(rows[0].features.shape[1])
    stacked = np.concatenate([row.features for row in rows], axis=0)
    feature_mean = stacked.mean(axis=0).astype(np.float32)
    feature_std = stacked.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    project_weights = _build_project_sample_weights(rows, balance_power=float(balance_power))
    project_counts = Counter(row.project_name for row in rows)

    weight = torch.zeros((feature_dim,), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=learning_rate, weight_decay=weight_decay)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for row in rows:
            features = torch.from_numpy(((row.features - feature_mean) / feature_std).astype(np.float32))
            logits = features @ weight + bias
            sample_weight = float(project_weights.get(row.project_name, 1.0))
            target = torch.tensor([row.label_index], dtype=torch.long)
            ce_loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
            pairwise_losses: list[torch.Tensor] = []
            for winner, loser, margin in _row_pairwise_preferences(row):
                pairwise_losses.append(
                    torch.nn.functional.softplus(-(logits[winner] - logits[loser])) * float(margin)
                )
            pairwise_loss = torch.stack(pairwise_losses).mean() if pairwise_losses else ce_loss * 0.0
            if objective_mode == "pairwise":
                row_loss = pairwise_loss * sample_weight if pairwise_losses else ce_loss * sample_weight
            elif objective_mode == "hybrid":
                row_loss = (ce_loss + float(pairwise_weight) * pairwise_loss) * sample_weight
            else:
                row_loss = ce_loss * sample_weight
            losses.append(row_loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

    selector = LinearCandidateExpandSelector(
        weight.detach().cpu().numpy().tolist(),
        bias=float(bias.detach().cpu().numpy()[0]),
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        feature_names=CANDIDATE_SELECTOR_FEATURE_NAMES,
        switch_margin=float(switch_margin),
    )
    train_correct = 0
    pairwise_correct = 0
    pairwise_total = 0
    oracle_error = 0.0
    selected_error = 0.0
    for row in rows:
        row_candidates = [
            {
                "selector_features": row.features[index].tolist(),
                "expand_ratio": float(index),
            }
            for index in range(row.features.shape[0])
        ]
        chosen = selector.select_candidate(
            row_candidates,
            baseline_expand_ratio=0.0,
        )
        chosen_index = int(chosen["expand_ratio"]) if chosen is not None else 0
        train_correct += int(chosen_index == row.label_index)
        row_scores = [selector.score_features(candidate["selector_features"]) for candidate in row_candidates]
        for winner, loser, _margin in _row_pairwise_preferences(row):
            pairwise_total += 1
            pairwise_correct += int(float(row_scores[winner]) > float(row_scores[loser]))
        oracle_error += float(min(row.candidate_errors))
        selected_error += float(row.candidate_errors[chosen_index])
    return {
        "weights": weight.detach().cpu().numpy().tolist(),
        "bias": float(bias.detach().cpu().numpy()[0]),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "feature_names": list(CANDIDATE_SELECTOR_FEATURE_NAMES),
        "train_rows": len(rows),
        "train_candidate_accuracy": float(train_correct / max(len(rows), 1)),
        "train_selected_error_mean": float(selected_error / max(len(rows), 1)),
        "train_oracle_error_mean": float(oracle_error / max(len(rows), 1)),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "balance_power": float(balance_power),
        "switch_margin": float(switch_margin),
        "objective_mode": str(objective_mode),
        "pairwise_weight": float(pairwise_weight),
        "train_pairwise_preference_accuracy": float(pairwise_correct / max(pairwise_total, 1)),
        "project_counts": {project_name: int(count) for project_name, count in sorted(project_counts.items())},
        "project_weights": {project_name: round(float(weight), 6) for project_name, weight in sorted(project_weights.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train linear selector for runtime multi-expand candidates")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--global-model", required=True)
    parser.add_argument("--roi-model", required=True)
    parser.add_argument("--local-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-expand-ratios", default="0.02,0.04,0.06,0.08,0.10,0.12")
    parser.add_argument("--baseline-expand-ratio", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--balance-power", type=float, default=1.0)
    parser.add_argument("--switch-margin", type=float, default=0.1)
    parser.add_argument("--preserve-point-margin", type=float, default=0.0015)
    parser.add_argument("--preserve-max-margin", type=float, default=0.0025)
    parser.add_argument("--objective-mode", choices=("cross_entropy", "pairwise", "hybrid"), default="cross_entropy")
    parser.add_argument("--pairwise-weight", type=float, default=0.5)
    args = parser.parse_args()

    candidate_expand_ratios = [float(part.strip()) for part in str(args.candidate_expand_ratios).split(",") if part.strip()]
    rows = build_selector_training_rows(
        split_dir=Path(args.split_dir),
        split=str(args.split),
        global_model_path=Path(args.global_model),
        roi_model_path=Path(args.roi_model),
        local_model_path=Path(args.local_model),
        candidate_expand_ratios=candidate_expand_ratios,
        baseline_expand_ratio=float(args.baseline_expand_ratio),
        preserve_point_margin=float(args.preserve_point_margin),
        preserve_max_margin=float(args.preserve_max_margin),
    )
    result = train_linear_selector(
        rows,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        balance_power=float(args.balance_power),
        switch_margin=float(args.switch_margin),
        objective_mode=str(args.objective_mode),
        pairwise_weight=float(args.pairwise_weight),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
