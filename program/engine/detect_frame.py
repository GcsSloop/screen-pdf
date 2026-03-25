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


def to_plain_candidate(candidate: dict) -> dict:
    return {
        "method": candidate["method"],
        "score": float(candidate["score"]),
        "metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in candidate["metrics"].items()
        },
        "quad": [[float(x), float(y)] for x, y in candidate["quad"]],
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


def model_detection_enabled() -> bool:
    return os.environ.get("SCREEN_PDF_DISABLE_MODEL", "").strip().lower() not in {"1", "true", "yes"}


def _model_files_exist() -> bool:
    return _model_path("global_corner_model.pt").exists() and _model_path("corner_heatmap_model.pt").exists()


def _local_model_exists() -> bool:
    return local_model_path() is not None


def _get_model_runtime() -> dict[str, object] | None:
    global _MODEL_RUNTIME
    if _MODEL_RUNTIME is not None:
        return _MODEL_RUNTIME
    if not model_detection_enabled() or not _model_files_exist():
        return None
    try:
        from two_stage_corner_pipeline import GlobalCornerPredictor, LocalCornerMoEPredictor, RoiCornerPredictor
    except Exception:
        return None
    try:
        local_model = local_model_path()
        _MODEL_RUNTIME = {
            "global_predictor": GlobalCornerPredictor(_model_path("global_corner_model.pt")),
            "roi_predictor": RoiCornerPredictor(_model_path("corner_heatmap_model.pt")),
            "local_predictor": LocalCornerMoEPredictor(local_model) if local_model is not None else None,
        }
    except Exception:
        _MODEL_RUNTIME = None
    return _MODEL_RUNTIME


def run_model_detection(image_path: str, image: np.ndarray | None = None) -> dict[str, object] | None:
    if not model_detection_enabled() or not _model_files_exist():
        return None
    try:
        from two_stage_corner_pipeline import predict_two_stage
    except Exception:
        return None

    runtime = _get_model_runtime()
    if runtime is None:
        return None

    try:
        result = predict_two_stage(
            image_path=Path(image_path),
            image=image,
            global_predictor=runtime["global_predictor"],
            roi_predictor=runtime["roi_predictor"],
            local_predictor=runtime["local_predictor"],
            page_id=Path(image_path).stem,
        )
    except Exception:
        return None

    return {
        "method": "model_three_stage_local_moe" if _local_model_exists() else "model_two_stage",
        "score": 0.96,
        "confidence": 0.08,
        "metrics": {
            "model": 1.0,
            "stage_count": 3.0 if runtime["local_predictor"] is not None else 2.0,
        },
        "quad": result["final_quad"],
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

    model_result = run_model_detection(image_path, image=image)
    if model_result is None:
        return base_payload

    model_candidate = {
        "method": str(model_result["method"]),
        "score": float(model_result["score"]),
        "metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in dict(model_result.get("metrics", {})).items()
        },
        "quad": [[float(x), float(y)] for x, y in model_result["quad"]],
    }
    candidates = [model_candidate, *list(base_payload["candidates"])]
    return {
        "best": {
            "method": str(model_result["method"]),
            "score": float(model_result["score"]),
            "confidence": float(model_result["confidence"]),
            "quad": [[float(x), float(y)] for x, y in model_result["quad"]],
        },
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
