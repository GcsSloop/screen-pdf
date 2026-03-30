from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_model_release_tag(tag: str) -> str:
    value = tag.strip()
    if not re.fullmatch(r"model-\d{8}-\d{6}-[0-9a-f]{8}", value):
        raise ValueError(f"invalid model release tag: {tag}")
    return value


def build_model_release_id(timestamp_utc: str, digest: str) -> str:
    dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    return f"model-{dt.strftime('%Y%m%d-%H%M%S')}-{digest[:8].lower()}"


def _sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_runtime_digest(manifest: dict[str, Any]) -> str:
    model_entries = manifest.get("models", {})
    parts: list[str] = []
    for key in sorted(model_entries.keys()):
        entry = model_entries.get(key) or {}
        parts.append(f"{key}:{entry.get('runtime_sha1','')}:{entry.get('model_id','')}")
    joined = "|".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def promote_runtime_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timestamp = str(
        manifest.get("released_at")
        or manifest.get("promoted_at")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", timestamp):
        timestamp = f"{timestamp}T00:00:00Z"
    digest = build_runtime_digest(manifest)
    release_id = build_model_release_id(timestamp, digest)
    manifest["model_release_id"] = release_id
    manifest["runtime_digest"] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model release helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote = subparsers.add_parser("promote-runtime-manifest")
    promote.add_argument("--manifest", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "promote-runtime-manifest":
        payload = promote_runtime_manifest(Path(args.manifest))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
