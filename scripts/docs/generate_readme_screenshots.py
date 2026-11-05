#!/usr/bin/env python3
"""Generate the README screenshots from the current runtime model.

Every image under ``docs/assets/screenshots/`` is produced by this script, so the
gallery can be regenerated whenever the promoted runtime changes:

    make readme-screenshots
    # or, with any interpreter that has torch + opencv + Pillow:
    program/engine/vendor/macos/bin/python3 scripts/docs/generate_readme_screenshots.py

The samples are fixed pages from the ``holdout`` (already-viewed regression)
split of curated scenes. Captions report the measured corner error, so the
images never claim more than the numbers do.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "program/engine"))

OUTPUT_DIR = REPO / "docs/assets/screenshots"
DEFAULT_MODEL = REPO / "models/runtime/screen_corner_single.pt"

# Canvas palette (GitHub dark friendly).
BG = (13, 17, 23)
PANEL = (22, 27, 34)
BORDER = (48, 54, 61)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)
GREEN = (34, 197, 94)
BLUE = (96, 165, 250)

CJK_FONTS = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

SAMPLES = [
    {
        "key": "wide-led",
        "scene": "202610-nanjing-edge-computing-forum_v1",
        "page": "149_20261015_081037606_iOS",
        "label": "宽幅 LED 论坛大屏",
    },
    {
        "key": "multi-screen",
        "scene": "202610-tunnel-lighting-duchao_v1",
        "page": "IMG_2477",
        "label": "大会场多屏 · 只取右侧 PPT",
    },
    {
        "key": "stage-led",
        "scene": "202610-embodied-intelligence-yaozhiqiang_v1",
        "page": "IMG_2118",
        "label": "舞台 LED 屏 · 强色偏",
    },
    {
        "key": "booth",
        "scene": "202606-guangya-sanxiong_v1",
        "page": "IMG_0348",
        "label": "展台灯箱 · 斜角拍摄",
    },
    {
        "key": "occluded",
        "scene": "202511-openatom-open-source-fund_v1",
        "page": "IMG_5136",
        "label": "观众遮挡 · 底边被挡住",
    },
    {
        "key": "projector",
        "scene": "202610-open-source-bigdata-qujingjing_v1",
        "page": "IMG_2801",
        "label": "会议室投影 · 常规白底幻灯片",
    },
]

HERO = {
    "key": "hero",
    "scene": "202610-open-source-bigdata-zhangliwen_v1",
    "page": "IMG_2775",
    "label": "会议室投影",
}

# Optional end-to-end panel: a real page of an exported PDF.
EXPORT_PDF = Path(
    "/Users/gcssloop/WorkSpace/AIGC/展会/2026/202610 上海开源大数据创新论坛/"
    "2026-10-29 AsterHarness：事实即本体-瞿晶晶/"
    "2026-10-29 AsterHarness：事实即本体-瞿晶晶_export/merged-optimized.pdf"
)
EXPORT_PAGE = 11


def bgr(color) -> tuple[int, int, int]:
    """OpenCV draws in BGR order; the palette above is written as RGB."""
    r, g, b = color
    return (int(b), int(g), int(r))


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in CJK_FONTS:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def ordered_quad(points) -> np.ndarray:
    """Return points ordered top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array(
        [pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]],
        dtype=np.float32,
    )


def find_record(scene: str, page: str) -> dict:
    base = REPO / "data/curated/scenes" / scene
    for split in ("holdout.jsonl", "validation.jsonl", "train.jsonl"):
        path = base / split
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["page_id"] == page:
                record["split_file"] = split
                return record
    raise SystemExit(f"page {page} not found in {base}")


def detect(predictor, record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from dataset_benchmark import normalized_corner_errors

    photo = cv2.imread(record["image_path"])
    if photo is None:
        raise SystemExit(f"cannot read {record['image_path']}")
    out = predictor.predict(photo)
    quad = ordered_quad(out["final_quad"])
    manual = ordered_quad(record["manual_quad"])
    return photo, quad, normalized_corner_errors(manual, quad)


def draw_quad(image: np.ndarray, quad: np.ndarray, color=None, thickness=0,
              dots: bool = True) -> np.ndarray:
    canvas = image.copy()
    color = bgr(GREEN) if color is None else color
    h, w = canvas.shape[:2]
    if thickness <= 0:
        thickness = max(3, round(min(w, h) / 260))
    pts = np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)
    if dots:
        radius = max(5, round(min(w, h) / 150))
        for x, y in pts.reshape(-1, 2):
            cv2.circle(canvas, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), radius, (255, 255, 255), max(1, radius // 3),
                       cv2.LINE_AA)
    return canvas


def rectify(photo: np.ndarray, quad: np.ndarray) -> np.ndarray:
    pts = ordered_quad(quad)
    width = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
    height = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))
    width, height = max(width, 8), max(height, 8)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(photo, matrix, (width, height), flags=cv2.INTER_CUBIC)


def to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def fit(image: np.ndarray, height: int) -> Image.Image:
    pil = to_pil(image)
    scale = height / pil.height
    return pil.resize((max(1, round(pil.width * scale)), height), Image.LANCZOS)


def rounded_paste(canvas: Image.Image, panel: Image.Image, x: int, y: int, radius: int = 14) -> None:
    mask = Image.new("L", panel.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, panel.width - 1, panel.height - 1],
                                           radius=radius, fill=255)
    canvas.paste(panel, (x, y), mask)


def chip(draw: ImageDraw.ImageDraw, xy, text: str, font, fg=TEXT, bg=(0, 0, 0, 170)) -> None:
    x, y = xy
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    pad = max(6, h // 3)
    draw.rounded_rectangle([x, y, x + w + 2 * pad, y + h + 2 * pad], radius=h // 2 + pad // 2,
                           fill=bg)
    draw.text((x + pad, y + pad - box[1]), text, font=font, fill=fg)


def caption_text(errors: np.ndarray) -> str:
    hit = int((errors <= 0.01).sum())
    return f"{hit}/4 角点在 1% 容差内 · 平均误差 {errors.mean() * 100:.2f}%"


def compose_pair(photo: np.ndarray, quad: np.ndarray, errors: np.ndarray, record: dict,
                 label: str, width: int, panel_h: int, title_size: int) -> Image.Image:
    left = draw_quad(photo, quad)
    right = rectify(photo, quad)

    margin, gap = 28, 46
    left_panel = fit(left, panel_h)
    right_panel = fit(right, panel_h)
    total_w = left_panel.width + gap + right_panel.width + 2 * margin
    header_h = title_size + 44
    total_h = header_h + panel_h + 46

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(title_size)
    small_font = load_font(max(13, title_size - 7))
    tiny_font = load_font(max(12, title_size - 10))
    panel_font = load_font(max(12, round(title_size * 0.62)))

    name = record.get("project_name", "")
    draw.text((margin, 14), f"{label} · {name}", font=title_font, fill=TEXT)
    meta = f"{record['page_id']} · {record.get('split_file', 'holdout.jsonl').replace('.jsonl', '')}"
    meta_w = draw.textbbox((0, 0), meta, font=tiny_font)[2]
    draw.text((total_w - margin - meta_w, 20), meta, font=tiny_font, fill=MUTED)

    top = header_h
    right_x = margin + left_panel.width + gap
    draw.text((margin, top - 24), "现场照片 · 绿框为检测到的屏幕四角", font=panel_font, fill=MUTED)
    draw.text((right_x, top - 24), "矫正后画面 · 透视还原", font=panel_font, fill=MUTED)
    rounded_paste(canvas, left_panel, margin, top)
    rounded_paste(canvas, right_panel, right_x, top)

    arrow = "→"
    arrow_box = draw.textbbox((0, 0), arrow, font=title_font)
    draw.text((margin + left_panel.width + gap / 2 - (arrow_box[2] - arrow_box[0]) / 2,
               top + panel_h / 2 - (arrow_box[3] - arrow_box[1]) / 2 - arrow_box[1]),
              arrow, font=title_font, fill=GREEN)

    footer_y = top + panel_h + 14
    draw.text((margin, footer_y), caption_text(errors), font=small_font, fill=GREEN)
    size_note = f"原图 {photo.shape[1]}×{photo.shape[0]} → 矫正 {right.shape[1]}×{right.shape[0]}"
    note_w = draw.textbbox((0, 0), size_note, font=tiny_font)[2]
    draw.text((total_w - margin - note_w, footer_y + 4), size_note, font=tiny_font, fill=MUTED)

    return canvas.resize((width, round(total_h * width / total_w)), Image.LANCZOS)


def compose_accuracy(photo: np.ndarray, quad: np.ndarray, manual: np.ndarray,
                     errors: np.ndarray, record: dict) -> Image.Image:
    """Show the 1% tolerance circle around each manual corner and the prediction."""
    h, w = photo.shape[:2]
    span = np.ptp(manual, axis=0)
    diagonal = float(np.hypot(max(float(span[0]), 1.0), max(float(span[1]), 1.0)))
    radius_tol = max(6, round(diagonal * 0.01))

    overlay = photo.copy()
    line_t = max(2, round(min(w, h) / 420))
    cv2.polylines(overlay, [np.asarray(quad, np.int32).reshape(-1, 1, 2)], True, bgr(GREEN),
                  line_t, cv2.LINE_AA)
    for index in range(4):
        mx, my = int(manual[index][0]), int(manual[index][1])
        px, py = int(quad[index][0]), int(quad[index][1])
        cv2.circle(overlay, (mx, my), radius_tol, bgr(BLUE), line_t, cv2.LINE_AA)
        cv2.drawMarker(overlay, (mx, my), (255, 255, 255), cv2.MARKER_CROSS,
                       max(8, radius_tol), max(1, line_t - 1), cv2.LINE_AA)
        cv2.circle(overlay, (px, py), max(3, line_t + 1), bgr(GREEN), -1, cv2.LINE_AA)

    width, margin, gap, header_h, zoom = 1600, 28, 40, 76, 250
    grid_h = 2 * zoom + 42 + 18
    panel_h = max(520, grid_h)
    left_panel = fit(overlay, panel_h)
    canvas = Image.new("RGB", (width, header_h + panel_h + 68), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22)
    tiny_font = load_font(13)

    draw.text((margin, 14), "四角偏差：绿点是模型预测，蓝圈是 1% 容差范围", font=title_font, fill=TEXT)
    draw.text((margin, header_h - 24), "现场照片", font=tiny_font, fill=MUTED)
    rounded_paste(canvas, left_panel, margin, header_h)

    zoom_x0 = margin + left_panel.width + gap
    draw.text((zoom_x0, header_h - 24), "四角放大（蓝圈半径 = 人工框对角线 1%）",
              font=tiny_font, fill=MUTED)
    names = ["左上", "右上", "右下", "左下"]
    crop_r = max(radius_tol * 3, int(min(w, h) * 0.045))
    for index in range(4):
        mx, my = int(manual[index][0]), int(manual[index][1])
        x0 = int(np.clip(mx - crop_r, 0, w - 2 * crop_r))
        y0 = int(np.clip(my - crop_r, 0, h - 2 * crop_r))
        crop = overlay[y0:y0 + 2 * crop_r, x0:x0 + 2 * crop_r]
        crop = cv2.resize(crop, (zoom, zoom), interpolation=cv2.INTER_LINEAR)
        cx_panel = zoom_x0 + (index % 2) * (zoom + gap) - (1 if index % 2 else 0) * 0
        cy_panel = header_h + (index // 2) * (zoom + 42)
        rounded_paste(canvas, to_pil(crop), cx_panel, cy_panel, radius=10)
        hit = errors[index] <= 0.01
        label = f"{names[index]} {errors[index] * 100:.2f}% · {'命中' if hit else '超差'}"
        draw.text((cx_panel + 2, cy_panel + zoom + 3), label, font=tiny_font,
                  fill=GREEN if hit else (248, 113, 113))

    footer = (f"{caption_text(errors)} · 误差 = 预测点到人工点的距离 ÷ 人工框轴对齐包围盒对角线"
              f" · 原图 {w}×{h}")
    draw.text((margin, header_h + panel_h + 34), footer, font=load_font(15), fill=MUTED)
    return canvas


def compose_export_pdf(pb: Path, page: int) -> Image.Image | None:
    gs = shutil.which("gs") or shutil.which("gswin64c")
    if gs is None:
        for candidate in (REPO / "program/engine/vendor/macos/bin/gs",
                          REPO / "program/engine/vendor/linux/bin/gs"):
            if candidate.is_file():
                gs = str(candidate)
                break
    if gs is None or not pb.is_file():
        print(f"[skip] export PDF panel: pdf={pb.is_file()} gs={bool(gs)}", file=sys.stderr)
        return None
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "page.jpg"
        subprocess.run(
            [gs, "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=jpeg", "-dJPEGQ=88", "-r140",
             f"-dFirstPage={page}", f"-dLastPage={page}", f"-sOutputFile={target}", str(pb)],
            check=True,
        )
        page_img = cv2.imread(str(target))
    if page_img is None:
        return None

    width, panel_h = 1600, 720
    margin = 28
    panel = to_pil(page_img)
    scale = panel_h / panel.height
    panel = panel.resize((round(panel.width * scale), panel_h), Image.LANCZOS)
    total_w = max(width, panel.width + 2 * margin)
    canvas = Image.new("RGB", (total_w, 58 + panel_h + 54), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), f"导出结果 · 合并 PDF 第 {page} 页（含 OCR 文本层）", font=load_font(22),
              fill=TEXT)
    rounded_paste(canvas, panel, (total_w - panel.width) // 2, 58)
    draw.text((margin, 58 + panel_h + 16), "照片 → 矫正 → 压缩 → OCR 叠层 → 合并 PDF，全部在桌面端完成",
              font=load_font(15), fill=MUTED)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path(os.environ.get("SCREEN_PDF_SINGLE_MODEL", DEFAULT_MODEL)))
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--quality", type=int, default=84)
    args = parser.parse_args()

    try:
        from screen_corner_single import SingleModelPredictor
    except ImportError as exc:  # pragma: no cover - environment hint
        print(f"missing dependency: {exc}\n"
              "run with an interpreter that has torch + opencv, e.g.\n"
              "  program/engine/vendor/macos/bin/python3 "
              "scripts/docs/generate_readme_screenshots.py", file=sys.stderr)
        return 2

    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2

    predictor = SingleModelPredictor(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"model: {predictor.model_id}")
    print(f"output: {args.out.relative_to(REPO)}")

    # Hero.
    hero_record = find_record(HERO["scene"], HERO["page"])
    photo, quad, errors = detect(predictor, hero_record)
    hero = compose_pair(photo, quad, errors, hero_record, HERO["label"],
                        width=1600, panel_h=470, title_size=21)
    hero.save(args.out / "hero.jpg", quality=args.quality, optimize=True, progressive=True)
    print(f"  hero.jpg · {HERO['page']} · {caption_text(errors)}")

    # Example gallery.
    for sample in SAMPLES:
        record = find_record(sample["scene"], sample["page"])
        photo, quad, errors = detect(predictor, record)
        image = compose_pair(photo, quad, errors, record, sample["label"],
                             width=1180, panel_h=330, title_size=17)
        name = f"example-{sample['key']}.jpg"
        image.save(args.out / name, quality=args.quality, optimize=True, progressive=True)
        print(f"  {name} · {record['page_id']} · {caption_text(errors)}")

    # Accuracy figure.
    record = find_record(HERO["scene"], HERO["page"])
    photo, quad, errors = detect(predictor, record)
    manual = ordered_quad(record["manual_quad"])
    accuracy = compose_accuracy(photo, quad, manual, errors, record)
    accuracy.save(args.out / "accuracy.jpg", quality=args.quality, optimize=True, progressive=True)
    print(f"  accuracy.jpg · {caption_text(errors)}")

    # Exported PDF page.
    export = compose_export_pdf(EXPORT_PDF, EXPORT_PAGE)
    if export is not None:
        export.save(args.out / "export-pdf.jpg", quality=args.quality, optimize=True,
                    progressive=True)
        print("  export-pdf.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
