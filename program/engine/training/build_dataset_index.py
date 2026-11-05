"""Build the deduplicated dataset index at docs/status/dataset-index.md.

A dataset unit is one canonical dataset_slug. Revisions of the same slug
(v1/v2, old/new annotation files) are collapsed; derived deep_screen_v1
training subsets are not separate datasets. The conference collection is one
dataset unit that contains 13 source collections and many per-talk scenes, so
its pages are counted once in the unit table and removed once in the unique
image total.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ANNOTATIONS = REPO / "data/curated/annotations"
SPLITS = REPO / "data/splits/cross_project"
REGISTRY = REPO / "training/registry"
CONFERENCE = REPO / "data/curated/collections/conference_v1"
OUTPUT = REPO / "docs/status/dataset-index.md"
CONFERENCE_SLUG = "conference_202603_china_smart_road_lighting"
REVISION_SLUG = "20260320-smart-road-lighting"
COMBO_SLUG = "20260929-black-background-unified"

SCENE_SLUGS = {
    "202404-pm-management-cube",
    "202511-openatom-open-source-fund",
    "202512-qinghui-alumni",
    "202606-guangya-sanxiong",
    "20260929-citywalk-liyang",
    "20260929-immersive-lilinlin",
    "20260929-metaverse-xingjie",
    "20260929-transport-dongzhiguo",
    "20260929-urban-renewal-luoying",
    "20260929-xr-zhouqunhua",
    "202610-embodied-intelligence-ai-chain",
    "202610-embodied-intelligence-gongtao",
    "202610-embodied-intelligence-huawei-ai",
    "202610-embodied-intelligence-industrial_v1",
    "202610-embodied-intelligence-liuyiyang",
    "202610-embodied-intelligence-luchen",
    "202610-embodied-intelligence-mazhankai",
    "202610-embodied-intelligence-qintong",
    "202610-embodied-intelligence-yaozhiqiang",
    "202610-embodied-intelligence-zhangkaiyuan",
    "202610-embodied-intelligence-zhouhaojie",
    "202610-embodied-intelligence-zhufeng",
    "202610-nanjing-computing-network-forum",
    "202610-nanjing-edge-computing-forum",
    "202610-nanjing-future-network-main-am",
    "202610-nanjing-future-network-main-pm",
    "202610-open-source-bigdata-zhangliwen",
    "202610-open-source-bigdata-qujingjing",
    "202610-open-source-bigdata-xuhui",
    "202610-tunnel-lighting-duchao",
    "202610-tunnel-lighting-gengchiyuan",
    "202610-tunnel-lighting-konglingmin",
    "202610-tunnel-lighting-lihuabin",
    "202610-tunnel-lighting-linyandan",
    "202610-tunnel-lighting-liuqi",
    "202610-tunnel-lighting-liusongrong",
    "202610-tunnel-lighting-liwei",
    "202610-tunnel-lighting-mazhangyixiang",
    "202610-tunnel-lighting-sunquanming",
    "202610-tunnel-lighting-tuyun",
    "202610-tunnel-lighting-zhangwenju",
    "202610-zhuiguang-ai",
}
HISTORICAL_SLUGS = {
    "202304-metaverse-mengliang",
    "202304-novastar-lecture",
    "202307-beijing-infocomm-zengqingwen",
    "202312-openatom-developer-conference",
    "202603-awe",
    "202603-guochenghao-yangzhou",
    "202603-led-dengganping-v2",
    "202603-white-mix",
    "202603-zhongjiao-jidian",
    "202603-zhongkong-jiaoan",
    "202603-zhouming-outdoor-lighting",
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rel(path: Path) -> str:
    return os.path.relpath(path, OUTPUT.parent)


def choose_annotation(slug: str, paths: list[Path]) -> Path:
    if slug == "202610-tunnel-lighting-linyandan":
        return next(p for p in paths if p.name.endswith("_pages_v1.jsonl"))
    if len(paths) == 1:
        return paths[0]
    for path in sorted(paths, key=lambda p: p.name, reverse=True):
        if "_v2" in path.name:
            return path
    return paths[0]


def sha(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def role_counts(slug: str, rows: list[dict], meta: dict) -> str:
    if slug == CONFERENCE_SLUG:
        roles = Counter(r.get("current_role") for r in read_rows(CONFERENCE / "pages.jsonl"))
        return f"{roles.get('train', 0)}/{roles.get('validation', 0)}/{roles.get('observed_regression', 0)} (+{roles.get('outside_current_snapshot', 0)})"
    from_rows = Counter(r.get("split") for r in rows if r.get("split"))
    if from_rows:
        return "/".join(str(from_rows.get(k, 0)) for k in ("train", "validation", "holdout"))
    split_file = meta.get("split_file")
    if split_file and Path(split_file).is_file():
        spec = json.loads(Path(split_file).read_text())
        if all(isinstance(spec.get(k), list) and all(isinstance(x, dict) for x in spec[k])
               for k in ("train", "validation", "holdout")):
            return "/".join(str(len(spec[k])) for k in ("train", "validation", "holdout"))
        if all(isinstance(spec.get(k), list) for k in ("train_page_ids", "val_page_ids", "holdout_page_ids")):
            return "/".join(str(len(spec[k])) for k in ("train_page_ids", "val_page_ids", "holdout_page_ids"))
        if spec.get("train_projects") is not None:
            return "项目级"
    return "—"


def title(slug: str, meta: dict) -> str:
    custom = {
        CONFERENCE_SLUG: "中国智慧道路照明大会（3.19 + 3.20 + 专项合集）",
        REVISION_SLUG: "中国智慧道路照明大会 3 月 20 日（精细标注修订）",
        COMBO_SLUG: "跨场次黑底统一集（XR + 李琳琳 + 新增）",
        "202603-awe": "AWE 专项测试",
        "202603-guochenghao-yangzhou": "扬州国城豪专项测试",
    }
    if slug in custom:
        return custom[slug]
    if meta.get("talk_name"):
        return str(meta["talk_name"])
    source = Path(str(meta.get("source_root", "")))
    return source.name if source.name else slug


def kind(slug: str) -> str:
    if slug == CONFERENCE_SLUG:
        return "会议合集"
    if slug == REVISION_SLUG:
        return "修订场景"
    if slug == COMBO_SLUG:
        return "组合场景"
    if slug in SCENE_SLUGS:
        return "独立场景"
    if slug in HISTORICAL_SLUGS:
        return "历史专项"
    return "其他"


def main() -> None:
    by_slug: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(ANNOTATIONS.glob("*.jsonl")):
        if "deep_screen_v1" in path.name:
            continue
        rows = read_rows(path)
        if rows:
            by_slug[rows[0].get("dataset_slug")].append(path)

    conference_rows = read_rows(CONFERENCE / "pages.jsonl")
    conference_map = {(r.get("project_name"), r.get("page_id")): r.get("image_sha256")
                      for r in conference_rows if r.get("manual_quad") and r.get("image_sha256")}
    records = []
    for slug in sorted(by_slug):
        annotation = choose_annotation(slug, by_slug[slug])
        rows = read_rows(annotation)
        manual = [r for r in rows if r.get("manual_quad")]
        meta_path = REGISTRY / f"{slug}.json"
        if not meta_path.is_file():
            candidates = sorted(REGISTRY.glob(f"{slug}*.json"), reverse=True)
            meta_path = candidates[0] if candidates else meta_path
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        if slug == CONFERENCE_SLUG:
            source_pages, manual_pages = 1946, len(conference_rows)
            split_path = CONFERENCE / "README.md"
            split_label = "集合"
            annotation = CONFERENCE / "pages.jsonl"
        else:
            source_pages, manual_pages = len(rows), len(manual)
            split_path = Path(meta["split_file"]) if meta.get("split_file") and Path(meta["split_file"]).is_file() else None
            if split_path is None:
                candidates = sorted(SPLITS.glob(f"{slug}*.json"), reverse=True)
                if candidates:
                    split_path = candidates[0]
            split_label = "拆分"
        keys = set()
        for row in (conference_rows if slug == CONFERENCE_SLUG else manual):
            digest = row.get("image_sha256") or conference_map.get((row.get("project_name"), row.get("page_id"))) or sha(row.get("image_path"))
            keys.add("sha:" + digest if digest else "path:" + str(row.get("image_path")))
        scene_dir = REPO / f"data/curated/scenes/{slug}_v1"
        records.append({
            "slug": slug,
            "title": title(slug, meta),
            "source_pages": source_pages,
            "manual_pages": manual_pages,
            "keys": keys,
            "roles": role_counts(slug, rows, meta),
            "kind": kind(slug),
            "annotation": annotation,
            "split": split_path,
            "split_label": split_label,
            "scene": (scene_dir / ("README.md" if (scene_dir / "README.md").is_file() else "manifest.json"))
            if scene_dir.is_dir() else None,
        })

    annotation_files = len(list(ANNOTATIONS.glob("*.jsonl")))
    scene_dirs = len(list((REPO / "data/curated/scenes").glob("*_v1")))
    union = set().union(*(r["keys"] for r in records))
    duplicate_memberships = sum(len(r["keys"]) for r in records) - len(union)
    pair_overlaps = Counter()
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            n = len(a["keys"] & b["keys"])
            if n:
                pair_overlaps[(a["slug"], b["slug"])] = n

    order = {"会议合集": 0, "修订场景": 1, "组合场景": 2, "独立场景": 3, "历史专项": 4, "其他": 5}
    records.sort(key=lambda r: (order[r["kind"]], r["slug"]))
    kind_counts = Counter(r["kind"] for r in records)

    lines = [
        "# 数据集索引（去重）",
        "",
        f"**结论：{len(records)} 个不重复数据集单元；去重后 {len(union):,} 张唯一人工图像。**",
        "",
        f"生成日期：{date.today().isoformat()}。按 `dataset_slug` 去重：同一数据集的 v1/v2、旧/新标注合并为一个单元；"
        "派生训练快照、回放数据、`deep_screen_v1_*` 实验拆分不单独计为数据集。",
        "",
        "## 总览",
        "",
        "| 指标 | 数量 | 说明 |",
        "| --- | ---: | --- |",
        f"| 数据集单元 | **{len(records)}** | 会议合集 {kind_counts.get('会议合集', 0)}、独立场景 {kind_counts.get('独立场景', 0)}、历史专项 {kind_counts.get('历史专项', 0)}、组合场景 {kind_counts.get('组合场景', 0)}、修订场景 {kind_counts.get('修订场景', 0)} |",
        f"| 单元内人工标注页 | **{sum(r['manual_pages'] for r in records):,}** | 各单元页数相加；同一张图可在会议合集与场景子集同时出现 |",
        f"| 唯一人工图像 | **{len(union):,}** | 按图像 SHA / 会议页映射去重；跨单元重叠 {duplicate_memberships} 页 |",
        f"| 标注文件 | {annotation_files} | 含 6 个 `deep_screen_v1_*` 派生文件和 3 组同 slug revision；实际 {len(records)} 个单元 |",
        f"| 会议子集合 | 13 | 1 个会议合集单元内含 13 个场次集合、48 个源项目 |",
        f"| 场景目录 | {scene_dirs} | 含 3 个场景族汇总目录；不等于 {len(records)} 个数据集单元 |",
        "",
        "**重叠说明**：会议合集覆盖 48 个源项目，因此与多数场次场景存在同图；"
        f"当前 {len(pair_overlaps)} 对单元有重叠，最大的是会议合集 ↔ 开放原子 277 页、会议合集 ↔ 扬州 200 页、"
        "会议合集 ↔ 邢杰 98 页、3 月 20 日修订 ↔ 会议合集 58 页；"
        "`20260929-black-background-unified` 另与 XR 重叠 14 页、与李琳琳重叠 5 页。"
        "`202603-awe`、`202603-guochenghao-yangzhou` 和多数 202603 历史专项的页面已经包含在会议合集中，"
        "单独保留的是旧实验拆分，不是新的独立照片。",
        "",
        "## 数据集表",
        "",
        "| # | 数据集 slug | 名称 / 场次 | 源页 / 人工 | 训练/开发/回归 | 类型 | 标注 / 拆分 |",
        "| ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for index, r in enumerate(records, 1):
        title_cell = r["title"].replace("|", "\\|")
        if r["scene"]:
            title_cell = f"[{title_cell}]({rel(r['scene'])})"
        annotation_link = f"[标注]({rel(r['annotation'])})"
        split_link = f"[{r.get('split_label', '拆分')}]({rel(r['split'])})" if r["split"] else ""
        entry = " · ".join(x for x in (annotation_link, split_link) if x)
        lines.append(
            f"| {index} | `{r['slug']}` | {title_cell} | {r['source_pages']} / {r['manual_pages']} | "
            f"{r['roles']} | {r['kind']} | {entry} |"
        )

    lines += [
        "",
        "## 口径与排除",
        "",
        "- 只统计至少有一条显式人工框的数据集单元；未标注源目录、只有图片/PDF 的资料、纯导出目录不计。",
        "- 同一 `dataset_slug` 的 v2 标注覆盖 v1；`20260929-metaverse-xingjie`、`202610-zhuiguang-ai`、`202610-tunnel-lighting-linyandan` 的旧文件只为 revision 溯源，不重复计数。",
        "- `deep_screen_v1_manual_only_*` 6 个文件是旧训练子集/场景加权拆分，属于派生数据，不进入数据集数量。",
        "- `training/datasets/*`、`training/runs/*/data` 的训练快照、24 倍回放和合并数据是训练派生物，不进入本索引；它们会重复引用这里的数据集。",
        "- 会议合集页数按 `conference_v1` 的 1946 页 / 1808 人工页计；旧 `conference_202603..._pages.jsonl`（720 行 / 591 人工）只作为历史标注来源，不另算一个单元。",
        "- 源页 / 人工页不同的单元：会议合集 1946 / 1808、扬州 205 / 200、中交机电 20 / 12、中控交安 34 / 33、洲明 40 / 39；3 月 20 日修订为 59 / 59。",
        f"- 每个单元的唯一图数等于其人工页数（revision 已合并）；全局唯一数 {len(union):,} 已扣除跨单元同图。",
        "",
        "## 重算",
        "",
        "```bash",
        "PYTHONPATH=program/engine program/engine/vendor/macos/bin/python3 program/engine/training/build_dataset_index.py",
        "```",
        "",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "dataset_units": len(records),
        "manual_pages": sum(r["manual_pages"] for r in records),
        "unique_images": len(union),
        "duplicate_memberships": duplicate_memberships,
        "kind_counts": kind_counts,
        "pairs": len(pair_overlaps),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
