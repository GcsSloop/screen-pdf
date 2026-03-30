from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from corner_train import (
    CornerHeatmapNet,
    build_corner_heatmaps,
    decode_model_output,
    freeze_model_backbone_for_offset_tuning,
    initialize_model_from_checkpoint,
    select_torch_device,
    split_model_output,
)
from dataset_benchmark import quad_geometry_metrics, summarize_geometry_metric_rows
from perspective_detect import order_points
from two_stage_corner_pipeline import GlobalCornerPredictor, LocalCornerMoEPredictor, RoiCornerPredictor, predict_two_stage


DEFAULT_CURATED_ROWS = "data/curated/annotations/conference_202603_china_smart_road_lighting_pages.jsonl"
DEFAULT_SPLIT_FILE = "data/splits/cross_project/conference_202603_china_smart_road_lighting_split_v1.json"
DEFAULT_TEACHER_EXPORT_SUBDIR = "teacher_exports"
DEFAULT_DATASET_SUBDIR = "dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


@dataclass(frozen=True)
class RoundPaths:
    round_root: Path
    data_root: Path
    teacher_export_root: Path
    dataset_root: Path
    checkpoints_root: Path
    reports_root: Path
    artifacts_root: Path
    logs_root: Path


def build_round_paths(round_root: Path) -> RoundPaths:
    round_root = Path(round_root)
    data_root = round_root / "data"
    return RoundPaths(
        round_root=round_root,
        data_root=data_root,
        teacher_export_root=data_root / DEFAULT_TEACHER_EXPORT_SUBDIR,
        dataset_root=data_root / DEFAULT_DATASET_SUBDIR,
        checkpoints_root=round_root / "checkpoints",
        reports_root=round_root / "reports",
        artifacts_root=round_root / "artifacts",
        logs_root=round_root / "logs",
    )


def resolve_teacher_model_paths(repo_root: Path, teacher_runtime_files: Mapping[str, str]) -> dict[str, Path]:
    models_root = repo_root / "models" / "runtime"
    return {alias: models_root / runtime_file for alias, runtime_file in teacher_runtime_files.items()}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _snapshot_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path)


@lru_cache(maxsize=4)
def _workspace_image_index(workspace_root: str) -> dict[str, list[Path]]:
    root = Path(workspace_root)
    index: dict[str, list[Path]] = {}
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(candidate.name, []).append(candidate)
    return index


def _resolve_image_path(repo_root: Path, row: Mapping[str, Any]) -> Path:
    raw_path = row.get("image_path")
    page_name = str(row.get("page_name") or "")
    page_id = str(row.get("page_id") or "")
    project_name = str(row.get("project_name") or "")
    candidates: list[Path] = []
    if raw_path:
        path = Path(str(raw_path))
        candidates.extend([path, path.parent / page_name, path.parent / path.name])
    workspace_root = repo_root.parent
    index = _workspace_image_index(str(workspace_root))
    candidates.extend(index.get(page_name, []))
    if page_id:
        candidates.extend([candidate for candidate in index.get(f"{page_id}.jpeg", [])])
        candidates.extend([candidate for candidate in index.get(f"{page_id}.jpg", [])])
        candidates.extend([candidate for candidate in index.get(f"{page_id}.png", [])])
    candidates.extend(
        candidate
        for candidate in index.get(page_name, [])
        if project_name and project_name in str(candidate)
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(raw_path or page_name or page_id)


def partition_rows_by_split(rows: list[dict[str, Any]], split_payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    train_projects = set(split_payload.get("train_projects") or [])
    val_projects = set(split_payload.get("val_projects") or [])
    holdout_projects = set(split_payload.get("holdout_projects") or [])
    partitioned: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "holdout": [], "unassigned": []}
    for row in rows:
        project_slug = str(row.get("project_slug") or "").strip()
        if project_slug in train_projects:
            partitioned["train"].append(row)
        elif project_slug in val_projects:
            partitioned["val"].append(row)
        elif project_slug in holdout_projects:
            partitioned["holdout"].append(row)
        else:
            partitioned["unassigned"].append(row)
    return partitioned


def _normalize_quad(quad: list[list[float]] | np.ndarray, width: int, height: int) -> np.ndarray:
    points = order_points(np.array(quad, dtype=np.float32))
    normalized = points.copy()
    normalized[:, 0] /= max(float(width), 1.0)
    normalized[:, 1] /= max(float(height), 1.0)
    return np.clip(normalized, 0.0, 1.0)


def _flip_quad_horizontal(quad: np.ndarray) -> np.ndarray:
    flipped = np.array(quad, dtype=np.float32).copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    return order_points(flipped)


def _maybe_augment(image: np.ndarray, quads: list[np.ndarray], enable_flip: bool) -> tuple[np.ndarray, list[np.ndarray]]:
    if not enable_flip:
        return image, quads
    if random.random() >= 0.5:
        return image, quads
    image = cv2.flip(image, 1)
    return image, [_flip_quad_horizontal(quad) for quad in quads]


def _quad_to_heatmap_targets(quad: np.ndarray, output_size: int) -> np.ndarray:
    return build_corner_heatmaps(quad.tolist(), output_size=output_size)


def _artifact_prefix(config: Mapping[str, Any]) -> str:
    return f"{config['public_name']}_{config['round']}"


def _artifact_name(config: Mapping[str, Any], suffix: str, extension: str) -> str:
    return f"{_artifact_prefix(config)}_{suffix}{extension}"


def _student_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("student") or config.get("training") or {}


def _loss_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("loss") or {}


def _decide_round_status(student_metrics: Mapping[str, Any], teacher_metrics: Mapping[str, Any]) -> tuple[str, str]:
    student_point = float(student_metrics.get("point_error_mean", math.inf))
    student_hit = float(student_metrics.get("point_le_0_01_ratio", 0.0))
    student_infer = float(student_metrics.get("avg_page_infer_ms", math.inf))
    target_point = 0.005
    target_hit = 0.80
    target_infer = 500.0
    if student_point <= target_point and student_hit >= target_hit and student_infer <= target_infer:
        return "stop", "meets promotion thresholds"
    teacher_point = float(teacher_metrics.get("point_error_mean", math.inf))
    if student_point <= teacher_point * 3.0:
        return "continue", "still above teacher quality target"
    return "continue", "still far from target quality"


def _build_round_comparison(
    student_metrics: Mapping[str, Any],
    teacher_metrics: Mapping[str, Any],
    r3_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    comparison_keys = [
        "point_error_mean",
        "point_le_0_05_ratio",
        "point_le_0_03_ratio",
        "point_le_0_02_ratio",
        "point_le_0_01_ratio",
        "screen_relative_error_mean",
        "max_corner_error_mean",
        "perspective_tilt_error_mean",
        "quad_inset_ratio_mean",
        "quad_inset_ratio_abs_mean",
    ]
    student = {key: student_metrics.get(key) for key in comparison_keys}
    teacher = {key: teacher_metrics.get(key) for key in comparison_keys}
    r3 = {key: r3_metrics.get(key) for key in comparison_keys}
    deltas = {
        key: round(float(student_metrics.get(key, 0.0)) - float(teacher_metrics.get(key, 0.0)), 4)
        for key in comparison_keys
        if isinstance(student_metrics.get(key), (int, float)) and isinstance(teacher_metrics.get(key), (int, float))
    }
    decision, reason = _decide_round_status(student_metrics, teacher_metrics)
    return {
        "student": student,
        "teacher": teacher,
        "r3": r3,
        "delta_to_teacher": deltas,
        "decision": decision,
        "decision_reason": reason,
    }


class RoundDistillationDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        repo_root: Path,
        input_size: int = 256,
        output_size: int = 64,
        augment: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        self.repo_root = repo_root
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment
        self.rows = _read_jsonl(manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image_path = _resolve_image_path(self.repo_root, row)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        manual_quad = _normalize_quad(row["manual_quad"], width, height)
        teacher_quad = _normalize_quad(row["teacher_v28_quad"], width, height)
        r3_quad = _normalize_quad(row["teacher_r3_quad"], width, height)
        image, [manual_quad, teacher_quad, r3_quad] = _maybe_augment(image, [manual_quad, teacher_quad, r3_quad], self.augment)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        image_f = image.astype(np.float32) / 255.0
        return {
            "image": torch.from_numpy(np.transpose(image_f, (2, 0, 1))),
            "manual_heatmaps": torch.from_numpy(_quad_to_heatmap_targets(manual_quad, self.output_size)),
            "teacher_heatmaps": torch.from_numpy(_quad_to_heatmap_targets(teacher_quad, self.output_size)),
            "r3_heatmaps": torch.from_numpy(_quad_to_heatmap_targets(r3_quad, self.output_size)),
            "manual_quad": torch.from_numpy(manual_quad),
            "teacher_quad": torch.from_numpy(teacher_quad),
            "r3_quad": torch.from_numpy(r3_quad),
        }


def _build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _heatmap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=-1)


def _evaluate_student(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    head_mode: str,
    loss_settings: Mapping[str, Any],
) -> dict[str, float]:
    model.eval()
    metric_rows: list[dict[str, float]] = []
    teacher_rows: list[dict[str, float]] = []
    r3_rows: list[dict[str, float]] = []
    losses: list[float] = []
    infer_elapsed = 0.0
    infer_pages = 0
    decode_mode = "soft_argmax_offset" if head_mode == "heatmap_offset" else "soft_argmax"
    manual_heatmap_weight = float(loss_settings.get("manual_heatmap_weight", 1.0))
    teacher_heatmap_weight = float(loss_settings.get("teacher_heatmap_weight", 0.5))
    r3_heatmap_weight = float(loss_settings.get("r3_heatmap_weight", 0.15))
    manual_coord_weight = float(loss_settings.get("manual_coord_weight", 3.0))
    teacher_coord_weight = float(loss_settings.get("teacher_coord_weight", 0.0))
    r3_coord_weight = float(loss_settings.get("r3_coord_weight", 0.0))
    offset_smooth_weight = float(loss_settings.get("offset_smooth_weight", 0.0))
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            manual_heatmaps = batch["manual_heatmaps"].to(device=device, dtype=torch.float32)
            teacher_heatmaps = batch["teacher_heatmaps"].to(device=device, dtype=torch.float32)
            r3_heatmaps = batch["r3_heatmaps"].to(device=device, dtype=torch.float32)
            manual_quad = batch["manual_quad"].cpu().numpy()
            teacher_quad = batch["teacher_quad"].cpu().numpy()
            r3_quad = batch["r3_quad"].cpu().numpy()
            infer_start = perf_counter()
            output = model(images)
            pred_heatmaps, pred_offsets = split_model_output(output, head_mode=head_mode)
            pred_coords = decode_model_output(output, decode_mode=decode_mode, head_mode=head_mode)
            infer_elapsed += perf_counter() - infer_start
            infer_pages += len(manual_quad)
            pred_quad = pred_coords.cpu().numpy()
            loss = (
                manual_heatmap_weight * _heatmap_loss(pred_heatmaps, manual_heatmaps)
                + teacher_heatmap_weight * _heatmap_loss(pred_heatmaps, teacher_heatmaps)
                + r3_heatmap_weight * _heatmap_loss(pred_heatmaps, r3_heatmaps)
                + manual_coord_weight
                * F.smooth_l1_loss(
                    pred_coords,
                    torch.from_numpy(manual_quad).to(device=device, dtype=torch.float32),
                    reduction="none",
                ).mean(dim=(1, 2))
                + teacher_coord_weight
                * F.smooth_l1_loss(
                    pred_coords,
                    torch.from_numpy(teacher_quad).to(device=device, dtype=torch.float32),
                    reduction="none",
                ).mean(dim=(1, 2))
                + r3_coord_weight
                * F.smooth_l1_loss(
                    pred_coords,
                    torch.from_numpy(r3_quad).to(device=device, dtype=torch.float32),
                    reduction="none",
                ).mean(dim=(1, 2))
            )
            if pred_offsets is not None and offset_smooth_weight > 0.0:
                loss = loss + offset_smooth_weight * pred_offsets.abs().mean(dim=(1, 2, 3, 4))
            losses.extend(loss.detach().cpu().tolist())
            for idx in range(len(manual_quad)):
                metric_rows.append(quad_geometry_metrics(manual_quad[idx], pred_quad[idx]))
                teacher_rows.append(quad_geometry_metrics(manual_quad[idx], teacher_quad[idx]))
                r3_rows.append(quad_geometry_metrics(manual_quad[idx], r3_quad[idx]))
    summary = summarize_geometry_metric_rows(metric_rows)
    teacher_summary = summarize_geometry_metric_rows(teacher_rows)
    r3_summary = summarize_geometry_metric_rows(r3_rows)
    summary.update(
        {
            "loss_mean": round(float(np.mean(losses)), 4) if losses else 0.0,
            "avg_page_infer_ms": round((infer_elapsed / infer_pages) * 1000.0, 2) if infer_pages else 0.0,
            "teacher_point_error_mean": teacher_summary["point_error_mean"],
            "teacher_point_le_0_01_ratio": teacher_summary["point_le_0_01_ratio"],
            "r3_point_error_mean": r3_summary["point_error_mean"],
            "r3_point_le_0_01_ratio": r3_summary["point_le_0_01_ratio"],
        }
    )
    return summary


def _train_student(
    repo_root: Path,
    config: Mapping[str, Any],
    train_manifest: Path,
    val_manifest: Path,
    holdout_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    student_cfg = _student_settings(config)
    loss_cfg = _loss_settings(config)
    round_name = str(config["round"])
    head_mode = str(student_cfg.get("head_mode", "heatmap"))
    decode_mode = "soft_argmax_offset" if head_mode == "heatmap_offset" else "soft_argmax"
    epochs = int(student_cfg.get("epochs", 2))
    batch_size = int(student_cfg.get("batch_size", 8))
    learning_rate = float(student_cfg.get("learning_rate", 1e-3))
    input_size = int(student_cfg.get("input_size", 256))
    output_size = int(student_cfg.get("output_size", 64))
    channels = int(student_cfg.get("channels", 32))
    seed = int(student_cfg.get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = RoundDistillationDataset(train_manifest, repo_root, input_size=input_size, output_size=output_size, augment=True)
    val_dataset = RoundDistillationDataset(val_manifest, repo_root, input_size=input_size, output_size=output_size, augment=False)
    holdout_dataset = RoundDistillationDataset(holdout_manifest, repo_root, input_size=input_size, output_size=output_size, augment=False)
    train_loader = _build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = _build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    holdout_loader = _build_dataloader(holdout_dataset, batch_size=batch_size, shuffle=False)

    device = select_torch_device()
    model = CornerHeatmapNet(in_channels=3, channels=channels, output_channels=4, head_mode=head_mode).to(device)
    init_checkpoint_path = student_cfg.get("init_checkpoint_path")
    if init_checkpoint_path:
        checkpoint = torch.load(_resolve_repo_path(repo_root, str(init_checkpoint_path)), map_location="cpu")
        initialize_model_from_checkpoint(model, checkpoint)
    if bool(student_cfg.get("freeze_backbone", False)):
        freeze_model_backbone_for_offset_tuning(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_val_metrics: dict[str, float] | None = None
    best_holdout_metrics: dict[str, float] | None = None

    manual_heatmap_weight = float(loss_cfg.get("manual_heatmap_weight", 1.0))
    teacher_heatmap_weight = float(loss_cfg.get("teacher_heatmap_weight", 0.5))
    r3_heatmap_weight = float(loss_cfg.get("r3_heatmap_weight", 0.15))
    manual_coord_weight = float(loss_cfg.get("manual_coord_weight", 3.0))
    teacher_coord_weight = float(loss_cfg.get("teacher_coord_weight", 0.0))
    r3_coord_weight = float(loss_cfg.get("r3_coord_weight", 0.0))
    offset_smooth_weight = float(loss_cfg.get("offset_smooth_weight", 0.0))

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device=device, dtype=torch.float32)
            manual_heatmaps = batch["manual_heatmaps"].to(device=device, dtype=torch.float32)
            teacher_heatmaps = batch["teacher_heatmaps"].to(device=device, dtype=torch.float32)
            r3_heatmaps = batch["r3_heatmaps"].to(device=device, dtype=torch.float32)
            manual_quad = batch["manual_quad"].to(device=device, dtype=torch.float32)
            teacher_quad = batch["teacher_quad"].to(device=device, dtype=torch.float32)
            r3_quad = batch["r3_quad"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            pred_heatmaps, pred_offsets = split_model_output(output, head_mode=head_mode)
            pred_coords = decode_model_output(output, decode_mode=decode_mode, head_mode=head_mode)
            loss = (
                manual_heatmap_weight * _heatmap_loss(pred_heatmaps, manual_heatmaps)
                + teacher_heatmap_weight * _heatmap_loss(pred_heatmaps, teacher_heatmaps)
                + r3_heatmap_weight * _heatmap_loss(pred_heatmaps, r3_heatmaps)
                + manual_coord_weight
                * F.smooth_l1_loss(pred_coords, manual_quad, reduction="none").mean(dim=(1, 2))
                + teacher_coord_weight
                * F.smooth_l1_loss(pred_coords, teacher_quad, reduction="none").mean(dim=(1, 2))
                + r3_coord_weight * F.smooth_l1_loss(pred_coords, r3_quad, reduction="none").mean(dim=(1, 2))
            ).mean()
            if pred_offsets is not None and offset_smooth_weight > 0.0:
                loss = loss + offset_smooth_weight * pred_offsets.abs().mean()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        val_metrics = _evaluate_student(model, val_loader, device, head_mode=head_mode, loss_settings=loss_cfg)
        holdout_metrics = _evaluate_student(model, holdout_loader, device, head_mode=head_mode, loss_settings=loss_cfg)
        epoch_row = {
            "epoch": epoch,
            "train_loss_mean": round(float(np.mean(train_losses)), 4) if train_losses else 0.0,
            "val": val_metrics,
            "holdout": holdout_metrics,
        }
        history.append(epoch_row)
        if float(val_metrics["loss_mean"]) < best_loss:
            best_loss = float(val_metrics["loss_mean"])
            best_epoch = epoch
            best_state = _snapshot_state_dict(model.state_dict())
            best_val_metrics = val_metrics
            best_holdout_metrics = holdout_metrics

    if best_state is None or best_val_metrics is None or best_holdout_metrics is None:
        raise RuntimeError("training produced no checkpoint")

    checkpoint_path = output_dir / _artifact_name(config, "student", ".pt")
    torch.save(
        {
            "state_dict": best_state,
            "input_size": input_size,
            "output_size": output_size,
            "channels": channels,
            "device": device.type,
            "head_mode": head_mode,
            "decode_mode": decode_mode,
            "round": round_name,
        },
        checkpoint_path,
    )
    history_path = output_dir / _artifact_name(config, "history", ".json")
    _write_json(history_path, history)
    return {
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_holdout_metrics": best_holdout_metrics,
        "history": history,
    }


def _load_round_config(config_path: Path) -> dict[str, Any]:
    return _read_json(config_path)


def _load_curated_rows(repo_root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _read_jsonl(_resolve_repo_path(repo_root, str(config["curated_rows"])))


def _load_split_payload(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return _read_json(_resolve_repo_path(repo_root, str(config["split_file"])))


def _export_teacher_snapshot(
    repo_root: Path,
    round_paths: RoundPaths,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    teacher_paths = resolve_teacher_model_paths(repo_root, {alias: str(item["runtime_file"]) for alias, item in config["teachers"].items()})
    split_payload = _load_split_payload(repo_root, config)
    rows = _load_curated_rows(repo_root, config)
    partitioned = partition_rows_by_split(rows, split_payload)
    round_paths.teacher_export_root.mkdir(parents=True, exist_ok=True)

    global_predictor = GlobalCornerPredictor(teacher_paths["r3"])
    roi_predictor = RoiCornerPredictor(_resolve_repo_path(repo_root, str(config["runtime_models"]["roi"])))
    local_predictor = LocalCornerMoEPredictor(teacher_paths["v28"])

    export_summaries: dict[str, Any] = {}
    for split_name in ("train", "val", "holdout"):
        split_rows = partitioned[split_name]
        output_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, float]] = []
        r3_rows: list[dict[str, float]] = []
        teacher_rows: list[dict[str, float]] = []
        for row in split_rows:
            if not row.get("manual_quad"):
                continue
            image_path = _resolve_image_path(repo_root, row)
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(image_path)
            start = perf_counter()
            global_quad = global_predictor.predict_image(image)
            stage = predict_two_stage(
                image_path=image_path,
                global_predictor=global_predictor,
                roi_predictor=roi_predictor,
                local_predictor=local_predictor,
                page_id=str(row.get("page_id") or image_path.stem),
                image=image,
            )
            export_row = {
                "dataset_slug": row.get("dataset_slug"),
                "project_slug": row.get("project_slug"),
                "page_id": row.get("page_id"),
                "image_path": row.get("image_path"),
                "split": split_name,
                "manual_quad": row.get("manual_quad"),
                "active_quad": row.get("active_quad"),
                "teacher_r3_quad": [[float(x), float(y)] for x, y in order_points(np.array(global_quad, dtype=np.float32))],
                "teacher_v28_quad": stage["final_quad"],
                "teacher_roi_quad": stage["roi_quad"],
                "teacher_runtime_models": {
                    "r3": str(teacher_paths["r3"]),
                    "v28": str(teacher_paths["v28"]),
                },
                "teacher_infer_ms": round((perf_counter() - start) * 1000.0, 2),
            }
            if row.get("manual_quad"):
                metric_rows.append(quad_geometry_metrics(row["manual_quad"], export_row["teacher_v28_quad"]))
                r3_rows.append(quad_geometry_metrics(row["manual_quad"], export_row["teacher_r3_quad"]))
                teacher_rows.append(quad_geometry_metrics(row["manual_quad"], export_row["teacher_v28_quad"]))
            output_rows.append(export_row)
        output_path = round_paths.teacher_export_root / f"{split_name}.jsonl"
        _write_jsonl(output_path, output_rows)
        export_summaries[split_name] = {
            "rows": len(output_rows),
            "path": str(output_path),
            "teacher_summary": summarize_geometry_metric_rows(teacher_rows),
            "r3_summary": summarize_geometry_metric_rows(r3_rows),
        }
    summary_path = round_paths.teacher_export_root / "summary.json"
    _write_json(summary_path, export_summaries)
    return {
        "teacher_paths": {key: str(value) for key, value in teacher_paths.items()},
        "summary_path": str(summary_path),
        "splits": export_summaries,
    }


def _materialize_student_dataset(round_paths: RoundPaths) -> dict[str, str]:
    round_paths.dataset_root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split_name in ("train", "val", "holdout"):
        source = round_paths.teacher_export_root / f"{split_name}.jsonl"
        destination = round_paths.dataset_root / f"{split_name}.jsonl"
        rows = _read_jsonl(source)
        _write_jsonl(destination, rows)
        manifests[split_name] = str(destination)
    return manifests


def _build_round_manifest(
    repo_root: Path,
    round_paths: RoundPaths,
    config: Mapping[str, Any],
    export_result: Mapping[str, Any] | None = None,
    train_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    student_cfg = dict(_student_settings(config))
    loss_cfg = dict(_loss_settings(config))
    manifest = {
        "public_name": config["public_name"],
        "internal_name": config["internal_name"],
        "round": config["round"],
        "dataset_slug": config["dataset_slug"],
        "teacher_models": list(config["teachers"].keys()),
        "teacher_runtime_files": {alias: str((repo_root / "models" / "runtime" / str(item["runtime_file"])).resolve()) for alias, item in config["teachers"].items()},
        "curated_rows": str(_resolve_repo_path(repo_root, str(config["curated_rows"]))),
        "split_file": str(_resolve_repo_path(repo_root, str(config["split_file"]))),
        "status": "locked" if train_result is None else "candidate",
        "data_root": str(round_paths.data_root),
        "teacher_export_root": str(round_paths.teacher_export_root),
        "dataset_root": str(round_paths.dataset_root),
        "checkpoints_root": str(round_paths.checkpoints_root),
        "reports_root": str(round_paths.reports_root),
        "artifacts_root": str(round_paths.artifacts_root),
        "logs_root": str(round_paths.logs_root),
        "student": student_cfg,
        "loss": loss_cfg,
    }
    if export_result is not None:
        manifest["teacher_export_summary"] = export_result
    if train_result is not None:
        manifest["training_summary"] = train_result
    if export_result is not None and train_result is not None:
        holdout_export = export_result["splits"]["holdout"]
        manifest["round_comparison"] = _build_round_comparison(
            train_result["best_holdout_metrics"],
            holdout_export["teacher_summary"],
            holdout_export["r3_summary"],
        )
    return manifest


def _write_round_report(round_paths: RoundPaths, config: Mapping[str, Any], export_result: Mapping[str, Any], train_result: Mapping[str, Any]) -> Path:
    round_paths.reports_root.mkdir(parents=True, exist_ok=True)
    report_path = round_paths.reports_root / f"{config['round']}_baseline_report.md"
    holdout_export = export_result["splits"]["holdout"]
    comparison = _build_round_comparison(
        train_result["best_holdout_metrics"],
        holdout_export["teacher_summary"],
        holdout_export["r3_summary"],
    )
    lines = [
        f"# {config['public_name']} {config['round']} baseline",
        "",
        f"- public name: `{config['public_name']}`",
        f"- internal name: `{config['internal_name']}`",
        f"- round: `{config['round']}`",
        f"- dataset: `{config['dataset_slug']}`",
        f"- student head mode: `{_student_settings(config).get('head_mode', 'heatmap')}`",
        f"- teacher aliases: `r3`, `v28`",
        "",
        "## Teacher export",
        "",
        f"- export summary: `{export_result['summary_path']}`",
        "",
        "## Student training",
        "",
        f"- checkpoint: `{train_result['checkpoint_path']}`",
        f"- history: `{train_result['history_path']}`",
        f"- best epoch: `{train_result['best_epoch']}`",
        f"- holdout decision: `{comparison['decision']}` ({comparison['decision_reason']})",
        "",
        "## Validation summary",
        "",
        json.dumps(train_result["best_val_metrics"], ensure_ascii=False, indent=2),
        "",
        "## Holdout comparison",
        "",
        json.dumps(comparison, ensure_ascii=False, indent=2),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_round(config_path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root.resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    config = _load_round_config(config_path)
    round_paths = build_round_paths(_resolve_repo_path(repo_root, str(config["round_root"])))
    for path in asdict(round_paths).values():
        Path(path).mkdir(parents=True, exist_ok=True)
    export_result = _export_teacher_snapshot(repo_root, round_paths, config)
    dataset_manifests = _materialize_student_dataset(round_paths)
    train_result = _train_student(
        repo_root=repo_root,
        config=config,
        train_manifest=Path(dataset_manifests["train"]),
        val_manifest=Path(dataset_manifests["val"]),
        holdout_manifest=Path(dataset_manifests["holdout"]),
        output_dir=round_paths.checkpoints_root,
    )
    report_path = _write_round_report(round_paths, config, export_result, train_result)
    manifest = _build_round_manifest(repo_root, round_paths, config, export_result, train_result)
    manifest["report_path"] = str(report_path)
    manifest_path = round_paths.round_root / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "export_result": export_result,
        "train_result": train_result,
    }


def run_round_001(config_path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    return run_round(config_path, repo_root=repo_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deep_screen_v1 distillation round")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root")
    parser.add_argument("--mode", choices=["export", "train", "run"], default="run")
    args = parser.parse_args()

    config_path = Path(args.config)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    config = _load_round_config(config_path)
    resolved_repo_root = repo_root or Path(__file__).resolve().parents[2]
    round_paths = build_round_paths(_resolve_repo_path(resolved_repo_root, str(config["round_root"])))
    for path in asdict(round_paths).values():
        Path(path).mkdir(parents=True, exist_ok=True)

    if args.mode in {"export", "run"}:
        export_result = _export_teacher_snapshot(resolved_repo_root, round_paths, config)
        _materialize_student_dataset(round_paths)
        if args.mode == "export":
            print(json.dumps(export_result, ensure_ascii=False, indent=2))
            return 0
    else:
        export_result = None

    if args.mode in {"train", "run"}:
        dataset_manifests = {
            split: round_paths.dataset_root / f"{split}.jsonl"
            for split in ("train", "val", "holdout")
        }
        if not all(path.exists() for path in dataset_manifests.values()):
            if export_result is None:
                export_result = _export_teacher_snapshot(resolved_repo_root, round_paths, config)
            _materialize_student_dataset(round_paths)
        train_result = _train_student(
            repo_root=resolved_repo_root,
            config=config,
            train_manifest=dataset_manifests["train"],
            val_manifest=dataset_manifests["val"],
            holdout_manifest=dataset_manifests["holdout"],
            output_dir=round_paths.checkpoints_root,
        )
        if args.mode == "train":
            print(json.dumps(train_result, ensure_ascii=False, indent=2))
            return 0
    else:
        train_result = None

    if export_result is None:
        export_result = _export_teacher_snapshot(resolved_repo_root, round_paths, config)
    if train_result is None:
        raise RuntimeError("training result missing")
    report_path = _write_round_report(round_paths, config, export_result, train_result)
    manifest = _build_round_manifest(resolved_repo_root, round_paths, config, export_result, train_result)
    manifest["report_path"] = str(report_path)
    _write_json(round_paths.round_root / "manifest.json", manifest)
    print(json.dumps({"manifest_path": str(round_paths.round_root / "manifest.json"), "report_path": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
