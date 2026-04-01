from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import cv2
import numpy as np
from PIL import Image
from pypdf import PdfWriter

from perspective_detect import detect_best_candidate

_MODEL_RUNTIME: dict[str, object] | None = None
_DEEP_SCREEN_V1_RUNTIME: dict[str, object] | None = None
_RUNTIME_RELEASE_MODEL_ID: str | None = None

TEACHER_MULTI_EXPAND_RATIOS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
TEACHER_CANDIDATE_BASELINE_GATE = 0.45
TEACHER_CANDIDATE_MIN_SCORE_GAIN = 0.03


def to_plain_candidate(candidate: dict) -> dict:
    return {
        "method": candidate["method"],
        "score": float(candidate["score"]),
        "metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in candidate["metrics"].items()
        },
        "quad": [[float(x), float(y)] for x, y in candidate["quad"]],
        "source": str(candidate.get("source", "opencv")),
        "modelId": str(candidate.get("modelId", candidate.get("model_id", candidate["method"]))),
        "debugOnly": bool(candidate.get("debugOnly", candidate.get("debug_only", False))),
    }


def choose_target_ratio(width: float, height: float) -> float:
    aspect = width / max(height, 1.0)
    widescreen = 16 / 9
    standard = 4 / 3
    if abs(aspect - standard) < abs(aspect - widescreen):
        return standard
    return widescreen


def warp(image: np.ndarray, quad: np.ndarray, max_dimension: int = 1600) -> np.ndarray:
    width = float(
        (
            np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])
        )
        / 2.0
    )
    height = float(
        (
            np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])
        )
        / 2.0
    )
    target_ratio = choose_target_ratio(width, height)
    if target_ratio >= 1.0:
        target_width = max_dimension
        target_height = int(round(target_width / target_ratio))
    else:
        target_height = max_dimension
        target_width = int(round(target_height * target_ratio))
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
    return cv2.warpPerspective(image, matrix, (target_width, target_height))


def platform_key() -> str:
    if sys.platform.startswith("darwin"):
        return "macos"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"


def resolve_binary(env_key: str, names: list[str]) -> str | None:
    override = os.environ.get(env_key)
    if override:
        return override

    root = Path(__file__).resolve().parent
    search_dirs = [
        root / "vendor" / platform_key() / "bin",
        root / "vendor" / "bin",
    ]
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate)

    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    return None


def list_tesseract_languages(tesseract_bin: str) -> list[str]:
    result = subprocess.run(
        [tesseract_bin, "--list-langs"],
        check=True,
        capture_output=True,
        text=True,
    )
    languages = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("List of available languages"):
            continue
        languages.append(line)
    return languages


def resolve_requested_languages(requested: str, available: list[str]) -> str:
    parts = [item.strip() for item in requested.split("+") if item.strip()]
    resolved = [item for item in parts if item in available]
    if not resolved and "eng" in available:
        resolved = ["eng"]
    if not resolved and available:
        resolved = [available[0]]
    return "+".join(resolved)


def save_compressed_jpeg(image: np.ndarray, output_path: Path, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb_image).save(
        output_path,
        format="JPEG",
        quality=int(quality),
        optimize=True,
        progressive=True,
        subsampling=0,
    )


def create_image_pdf(image_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image.save(output_path, "PDF", resolution=144.0)


def run_tesseract_pdf(
    tesseract_bin: str, image_path: Path, output_base: Path, languages: str
) -> Path:
    command = [tesseract_bin, str(image_path), str(output_base)]
    if languages:
        command.extend(["-l", languages])
    command.append("pdf")
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output_base.with_suffix(".pdf")


def run_tesseract_text(
    tesseract_bin: str, image_path: Path, output_base: Path, languages: str
) -> Path | None:
    command = [tesseract_bin, str(image_path), str(output_base)]
    if languages:
        command.extend(["-l", languages])
    command.append("txt")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    return output_base.with_suffix(".txt")


def optimize_pdf_with_ghostscript(
    gs_bin: str, input_path: Path, optimized_path: Path
) -> bool:
    command = [
        gs_bin,
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.6",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dPDFSETTINGS=/ebook",
        f"-sOutputFile={optimized_path}",
        str(input_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return False
    return optimized_path.exists() and optimized_path.stat().st_size < input_path.stat().st_size


def safe_slug(value: str) -> str:
    output = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            output.append(char)
        else:
            output.append("-")
    slug = "".join(output).strip("-")
    return slug or "page"


def _engine_root() -> Path:
    return Path(__file__).resolve().parent


def _model_root() -> Path:
    override = os.environ.get("SCREEN_PDF_MODEL_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    engine_root = _engine_root()
    candidates = []
    if len(engine_root.parents) >= 2:
        candidates.append(engine_root.parents[1] / "models" / "runtime")
    candidates.append(engine_root / "models")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _model_path(name: str) -> Path:
    return _model_root() / name


def runtime_release_model_id() -> str:
    global _RUNTIME_RELEASE_MODEL_ID
    if _RUNTIME_RELEASE_MODEL_ID is not None:
        return _RUNTIME_RELEASE_MODEL_ID

    model_root = _model_root()
    for manifest_path in sorted(model_root.glob("*.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        release_id = str(payload.get("model_release_id", "")).strip()
        public_name = str(payload.get("public_name", "")).strip()
        if not release_id and not public_name:
            continue
        if str(payload.get("status", "")).strip().lower() == "promoted":
            _RUNTIME_RELEASE_MODEL_ID = release_id or public_name
            return _RUNTIME_RELEASE_MODEL_ID

    _RUNTIME_RELEASE_MODEL_ID = "teacher_current"
    return _RUNTIME_RELEASE_MODEL_ID


def local_model_path() -> Path | None:
    preferred_names = [
        "local_corner_moe_coord_model.pt",
        "local_corner_moe_model.pt",
    ]
    for name in preferred_names:
        path = _model_path(name)
        if path.exists():
            return path
    return None


def candidate_selector_path() -> Path | None:
    path = _model_path("candidate_expand_selector.json")
    return path if path.exists() else None


def model_detection_enabled() -> bool:
    return os.environ.get("SCREEN_PDF_DISABLE_MODEL", "").strip().lower() not in {"1", "true", "yes"}


def dual_model_debug_enabled() -> bool:
    return os.environ.get("SCREEN_PDF_DEBUG_DUAL_MODEL", "").strip().lower() in {"1", "true", "yes"}


def _model_files_exist() -> bool:
    return _model_path("global_corner_model.pt").exists() and _model_path("corner_heatmap_model.pt").exists()


def _local_model_exists() -> bool:
    return local_model_path() is not None


def deep_screen_v1_model_path() -> Path | None:
    override = os.environ.get("SCREEN_PDF_DEEP_SCREEN_V1_MODEL", "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        return path if path.exists() else None
    for name in ("deep_screen_v1_debug.pt", "deep_screen_v1.pt", "deep_screen_v1_round_022_student.pt"):
        path = _model_path(name)
        if path.exists():
            return path
    return None


def _get_model_runtime() -> dict[str, object] | None:
    global _MODEL_RUNTIME
    if _MODEL_RUNTIME is not None:
        return _MODEL_RUNTIME
    if not model_detection_enabled() or not _model_files_exist():
        return None
    try:
        from two_stage_corner_pipeline import (
            GlobalCornerPredictor,
            LinearCandidateExpandSelector,
            LocalCornerMoEPredictor,
            RoiCornerPredictor,
        )
    except Exception:
        return None
    try:
        local_model = local_model_path()
        selector_model = candidate_selector_path()
        _MODEL_RUNTIME = {
            "global_predictor": GlobalCornerPredictor(_model_path("global_corner_model.pt")),
            "roi_predictor": RoiCornerPredictor(_model_path("corner_heatmap_model.pt")),
            "local_predictor": LocalCornerMoEPredictor(local_model) if local_model is not None else None,
            "candidate_selector": LinearCandidateExpandSelector.from_json(selector_model) if selector_model is not None else None,
            "candidate_expand_ratios": list(TEACHER_MULTI_EXPAND_RATIOS),
            "candidate_baseline_gate": float(TEACHER_CANDIDATE_BASELINE_GATE),
            "candidate_min_score_gain": float(TEACHER_CANDIDATE_MIN_SCORE_GAIN),
        }
    except Exception:
        _MODEL_RUNTIME = None
    return _MODEL_RUNTIME


def _get_deep_screen_v1_runtime() -> dict[str, object] | None:
    global _DEEP_SCREEN_V1_RUNTIME
    if _DEEP_SCREEN_V1_RUNTIME is not None:
        return _DEEP_SCREEN_V1_RUNTIME
    model_path = deep_screen_v1_model_path()
    if model_path is None:
        return None
    try:
        import torch

        from deep_screen_v1_model import DeepScreenV1Net, load_compatible_state_dict
    except Exception:
        return None
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        model = DeepScreenV1Net(
            base_channels=int(checkpoint.get("base_channels", 32)),
            roi_size=int(checkpoint.get("roi_size", 16)),
            experts=int(checkpoint.get("experts", 3)),
            expand_ratio=float(checkpoint.get("roi_expand_ratio", 0.08)),
            roi_adapter_layers=int(checkpoint.get("roi_adapter_layers", 0)),
            spatial_refine_layers=int(checkpoint.get("spatial_refine_layers", 0)),
            residual_quad_head_layers=int(checkpoint.get("residual_quad_head_layers", 0)),
            strict_spatial_refine_layers=int(checkpoint.get("strict_spatial_refine_layers", 0)),
            candidate_selection_enabled=bool(checkpoint.get("candidate_selection_enabled", False)),
            state_aware_candidate_enabled=bool(checkpoint.get("state_aware_candidate_enabled", False)),
            internal_candidate_names=checkpoint.get("internal_candidate_names"),
            final_output_mode=str(
                checkpoint.get(
                    "final_output_mode",
                    "candidate_selection" if checkpoint.get("candidate_selection_enabled", False) else "base_final",
                )
            ),
            scene_classes=int(checkpoint.get("scene_classes", 4)),
            scene_embedding_dim=int(checkpoint.get("scene_embedding_dim", 8)),
            coarse_visibility_refine_enabled=bool(checkpoint.get("coarse_visibility_refine_enabled", False)),
        )
        load_compatible_state_dict(model, checkpoint["state_dict"])
        model.eval()
        _DEEP_SCREEN_V1_RUNTIME = {
            "model": model,
            "input_size": int(checkpoint.get("input_size", 256)),
            "model_path": model_path,
            "torch": torch,
            "opencv_candidate_selection_enabled": bool(checkpoint.get("opencv_candidate_selection_enabled", False)),
        }
    except Exception:
        _DEEP_SCREEN_V1_RUNTIME = None
    return _DEEP_SCREEN_V1_RUNTIME


def run_teacher_detection(image_path: str, image: np.ndarray | None = None) -> dict[str, object] | None:
    if not model_detection_enabled() or not _model_files_exist():
        return None
    try:
        from two_stage_corner_pipeline import predict_two_stage
    except Exception:
        return None

    runtime = _get_model_runtime()
    if runtime is None:
        return None

    if image is None:
        image = cv2.imread(image_path)
    if image is None:
        return None

    try:
        result = predict_two_stage(
            image_path=Path(image_path),
            image=image,
            global_predictor=runtime["global_predictor"],
            roi_predictor=runtime["roi_predictor"],
            local_predictor=runtime["local_predictor"],
            candidate_selector=runtime.get("candidate_selector"),
            page_id=Path(image_path).stem,
            candidate_expand_ratios=runtime.get("candidate_expand_ratios"),
            candidate_baseline_gate=float(runtime.get("candidate_baseline_gate", TEACHER_CANDIDATE_BASELINE_GATE)),
            candidate_min_score_gain=float(runtime.get("candidate_min_score_gain", TEACHER_CANDIDATE_MIN_SCORE_GAIN)),
        )
    except Exception:
        return None

    metrics: dict[str, float] = {
        "model": 1.0,
        "stage_count": 3.0 if runtime["local_predictor"] is not None else 2.0,
    }
    confidence = 0.08

    return {
        "method": "teacher_current",
        "score": 0.96,
        "confidence": confidence,
        "metrics": metrics,
        "quad": result["final_quad"],
        "source": "runtime_teacher",
        "model_id": runtime_release_model_id(),
        "debug_only": False,
    }


def run_model_detection(image_path: str, image: np.ndarray | None = None) -> dict[str, object] | None:
    return run_teacher_detection(image_path, image=image)


def run_deep_screen_v1_detection(image_path: str, image: np.ndarray | None = None) -> dict[str, object] | None:
    runtime = _get_deep_screen_v1_runtime()
    if runtime is None:
        return None
    if image is None:
        image = cv2.imread(image_path)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_size = int(runtime["input_size"])
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image_f = resized.astype(np.float32) / 255.0
    tensor = runtime["torch"].from_numpy(np.transpose(image_f, (2, 0, 1))).unsqueeze(0)
    with runtime["torch"].no_grad():
        output = runtime["model"](tensor)
        if runtime.get("opencv_candidate_selection_enabled"):
            opencv_result = detect_best_candidate(image)
            if opencv_result is not None and opencv_result.get("best") is not None:
                opencv_quad = np.array(opencv_result["best"]["quad"], dtype=np.float32).copy()
                opencv_quad[:, 0] /= max(float(image.shape[1]), 1.0)
                opencv_quad[:, 1] /= max(float(image.shape[0]), 1.0)
                external_candidate_quads = runtime["torch"].from_numpy(opencv_quad).to(dtype=tensor.dtype).unsqueeze(0).unsqueeze(0)
                selection = runtime["model"].select_candidate_pool(
                    output,
                    external_candidate_quads=external_candidate_quads,
                )
                output["final_quad"] = selection["selected_quad"]
    quad_norm = output["final_quad"][0].detach().cpu().numpy()
    height, width = image.shape[:2]
    quad = [[float(x * width), float(y * height)] for x, y in quad_norm]
    return {
        "method": "deep_screen_v1_best",
        "score": 0.95,
        "confidence": 0.06,
        "metrics": {
            "model": 1.0,
            "student": 1.0,
        },
        "quad": quad,
        "source": "runtime_student",
        "model_id": f"deep_screen_v1_{runtime['model_path'].stem}",
        "debug_only": True,
    }


def _build_best_payload(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "method": str(candidate["method"]),
        "score": float(candidate["score"]),
        "confidence": float(candidate.get("confidence", 0.0)),
        "quad": [[float(x), float(y)] for x, y in candidate["quad"]],
        "source": str(candidate.get("source", "runtime")),
        "modelId": str(candidate.get("modelId", candidate.get("model_id", candidate["method"]))),
        "debugOnly": bool(candidate.get("debugOnly", candidate.get("debug_only", False))),
    }


def _build_runtime_candidate(candidate: dict[str, object]) -> dict[str, object]:
    return {
        "method": str(candidate["method"]),
        "score": float(candidate["score"]),
        "metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in dict(candidate.get("metrics", {})).items()
        },
        "quad": [[float(x), float(y)] for x, y in candidate["quad"]],
        "source": str(candidate.get("source", "runtime")),
        "modelId": str(candidate.get("modelId", candidate.get("model_id", candidate["method"]))),
        "debugOnly": bool(candidate.get("debugOnly", candidate.get("debug_only", False))),
    }


def build_detect_payload(image_path: str, image: np.ndarray) -> dict[str, object]:
    result = detect_best_candidate(image)
    if result is None:
        base_payload: dict[str, object] = {"best": None, "candidates": []}
    else:
        base_payload = {
            "best": {
                "method": result["best"]["method"],
                "score": float(result["best"]["score"]),
                "confidence": float(result["best"]["confidence"]),
                "quad": [[float(x), float(y)] for x, y in result["best"]["quad"]],
            },
            "candidates": [to_plain_candidate(item) for item in result["candidates"]],
        }

    teacher_result = run_teacher_detection(image_path, image=image)
    if teacher_result is None:
        return base_payload

    candidates = [_build_runtime_candidate(teacher_result)]
    if dual_model_debug_enabled():
        student_result = run_deep_screen_v1_detection(image_path, image=image)
        if student_result is not None:
            candidates.append(_build_runtime_candidate(student_result))
    candidates.extend(list(base_payload["candidates"]))
    return {
        "best": _build_best_payload(teacher_result),
        "candidates": candidates,
    }


def command_detect(image_path: str) -> int:
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"failed to read image: {image_path}")
    output = build_detect_payload(image_path, image)
    print(json.dumps(output, ensure_ascii=False))
    return 0


def command_detect_batch(manifest_path: str) -> int:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    images = [str(item) for item in manifest.get("images", [])]
    for image_path in images:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"failed to read image: {image_path}")
            payload = build_detect_payload(image_path, image)
            line = {
                "imagePath": image_path,
                "best": payload.get("best"),
                "candidates": payload.get("candidates", []),
                "error": None,
            }
        except Exception as err:
            line = {
                "imagePath": image_path,
                "best": None,
                "candidates": [],
                "error": str(err),
            }
        print(json.dumps(line, ensure_ascii=False), flush=True)
    return 0


def command_preview(image_path: str, quad_json: str, output_path: str) -> int:
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"failed to read image: {image_path}")
    quad = np.array(json.loads(quad_json), dtype=np.float32)
    warped = warp(image, quad)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output), warped)
    if not ok:
        raise SystemExit(f"failed to write preview: {output}")
    print(str(output))
    return 0


def command_export(manifest_path: str) -> int:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output_path = Path(manifest["output_path"])
    work_dir = Path(manifest["work_dir"])
    images_dir = work_dir / "images"
    page_pdf_dir = work_dir / "page-pdfs"
    ocr_dir = work_dir / "ocr"
    report_path = work_dir / "export-report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    page_pdf_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    tesseract_bin = resolve_binary(
        "SCREEN_PDF_TESSERACT",
        ["tesseract.exe", "tesseract"] if platform_key() == "windows" else ["tesseract"],
    )
    gs_bin = resolve_binary(
        "SCREEN_PDF_GS",
        ["gswin64c.exe", "gswin32c.exe", "gs.exe", "gs"]
        if platform_key() == "windows"
        else ["gs"],
    )

    warnings: list[str] = []
    requested_languages = manifest["options"]["ocr_languages"]
    effective_languages: str | None = None
    ocr_enabled = bool(manifest["options"]["ocr_enabled"])
    if ocr_enabled and not tesseract_bin:
        warnings.append("tesseract not found; exported image-only PDF without hidden text layer")
        ocr_enabled = False
    if ocr_enabled and tesseract_bin:
        available_languages = list_tesseract_languages(tesseract_bin)
        effective_languages = resolve_requested_languages(
            requested_languages, available_languages
        )
        if requested_languages and effective_languages != requested_languages:
            warnings.append(
                f"OCR languages fallback: requested {requested_languages}, using {effective_languages}"
            )

    writer = PdfWriter()
    exported_pages = []

    for index, page in enumerate(manifest["pages"], start=1):
        image = cv2.imread(page["path"])
        if image is None:
            raise SystemExit(f"failed to read image: {page['path']}")

        quad = np.array(page["quad"], dtype=np.float32)
        warped = warp(image, quad, max_dimension=int(manifest["options"]["max_dimension"]))
        slug = safe_slug(f"{index:03d}-{page['id']}")
        image_path = images_dir / f"{slug}.jpg"
        pdf_path = page_pdf_dir / f"{slug}.pdf"
        ocr_base = ocr_dir / slug
        ocr_text_path: Path | None = None
        page_warning: str | None = None

        save_compressed_jpeg(
            warped,
            image_path,
            quality=int(manifest["options"]["jpeg_quality"]),
        )

        if ocr_enabled and tesseract_bin and effective_languages:
            pdf_path = run_tesseract_pdf(
                tesseract_bin, image_path, page_pdf_dir / slug, effective_languages
            )
            ocr_text_path = run_tesseract_text(
                tesseract_bin, image_path, ocr_base, effective_languages
            )
        else:
            create_image_pdf(image_path, pdf_path)
            page_warning = "ocr-disabled"

        writer.append(str(pdf_path))
        exported_pages.append(
            {
                "id": page["id"],
                "name": page["name"],
                "imagePath": str(image_path),
                "pdfPath": str(pdf_path),
                "ocrTextPath": str(ocr_text_path) if ocr_text_path else None,
                "warning": page_warning,
            }
        )

    merged_path = work_dir / "merged-raw.pdf"
    with merged_path.open("wb") as handle:
        writer.write(handle)
    writer.close()

    final_path = output_path
    if gs_bin:
        optimized_path = work_dir / "merged-optimized.pdf"
        if optimize_pdf_with_ghostscript(gs_bin, merged_path, optimized_path):
            shutil.copyfile(optimized_path, final_path)
        else:
            shutil.copyfile(merged_path, final_path)
    else:
        shutil.copyfile(merged_path, final_path)
        warnings.append("ghostscript not found; skipped final PDF optimization pass")

    report = {
        "projectName": manifest["project_name"],
        "sourceDir": manifest["source_dir"],
        "outputPath": str(final_path),
        "pageCount": len(exported_pages),
        "effectiveOcrLanguages": effective_languages,
        "warnings": warnings,
        "pages": exported_pages,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = {
        "outputPath": str(final_path),
        "reportPath": str(report_path),
        "pageCount": len(exported_pages),
        "effectiveOcrLanguages": effective_languages,
        "warnings": warnings,
        "pages": exported_pages,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect")
    detect_parser.add_argument("--image", required=True)

    detect_batch_parser = subparsers.add_parser("detect-batch")
    detect_batch_parser.add_argument("--manifest", required=True)

    preview_parser = subparsers.add_parser("preview")
    preview_parser.add_argument("--image", required=True)
    preview_parser.add_argument("--quad", required=True)
    preview_parser.add_argument("--output", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--manifest", required=True)

    args = parser.parse_args()
    if args.command == "detect":
        return command_detect(args.image)
    if args.command == "detect-batch":
        return command_detect_batch(args.manifest)
    if args.command == "preview":
        return command_preview(args.image, args.quad, args.output)
    if args.command == "export":
        return command_export(args.manifest)
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
