from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def repo_root_from(path: Path | None = None) -> Path:
    current = (path or Path(__file__)).resolve()
    if current.is_dir():
        return current
    return current.parents[2]


def resolve_bundle_root(bundle_path: Path) -> Path:
    if bundle_path.suffix == ".app":
        return bundle_path / "Contents"
    return bundle_path


def resolve_resource_path_from_base(resource_dir: Path, relative: Path) -> Path | None:
    base = resource_dir
    for _ in range(0, 9):
        candidate = base / relative
        if candidate.exists():
            return candidate
        base = base / "_up_"
    return None


def bundled_python_candidates(engine_dir: Path, platform_name: str) -> list[Path]:
    if platform_name == "windows":
        return [
            engine_dir / "vendor" / "windows" / "bin" / "python.exe",
            engine_dir / "vendor" / "python.exe",
        ]
    return [
        engine_dir / "vendor" / "macos" / "bin" / "python3",
        engine_dir / "vendor" / "linux" / "bin" / "python3",
        engine_dir / "vendor" / "bin" / "python3",
    ]


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_promoted_manifest(model_dir: Path) -> Path:
    for path in sorted(model_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("status", "")).strip().lower() == "promoted":
            return path
    raise FileNotFoundError(f"no promoted runtime manifest found in {model_dir}")


def expected_model_sha_map(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("artifacts", {})
    result: dict[str, str] = {}
    for value in entries.values():
        if not isinstance(value, dict):
            continue
        rel_path = str(value.get("path", "")).strip()
        hash_value = str(value.get("sha1", "")).strip()
        if rel_path and hash_value:
            result[Path(rel_path).name] = hash_value
    return result


def verify_bundle(
    bundle_path: Path,
    platform_name: str,
    smoke_image: Path | None = None,
) -> dict[str, Any]:
    contents_dir = resolve_bundle_root(bundle_path)
    resource_dir = contents_dir / "Resources"
    engine_dir = resolve_resource_path_from_base(resource_dir, Path("engine"))
    model_dir = resolve_resource_path_from_base(resource_dir, Path("models") / "runtime")
    if engine_dir is None:
        raise FileNotFoundError(f"engine resources not found inside {bundle_path}")
    if model_dir is None:
        raise FileNotFoundError(f"runtime model resources not found inside {bundle_path}")

    python_candidates = bundled_python_candidates(engine_dir, platform_name)
    bundled_python = next((path for path in python_candidates if path.exists()), None)
    if bundled_python is None:
        names = ", ".join(str(path) for path in python_candidates)
        raise FileNotFoundError(f"bundled python not found; checked: {names}")

    required_bins = {
        "macos": [
            engine_dir / "vendor" / "macos" / "bin" / "tesseract",
            engine_dir / "vendor" / "macos" / "bin" / "gs",
        ],
        "windows": [
            engine_dir / "vendor" / "windows" / "bin" / "tesseract.exe",
            engine_dir / "vendor" / "windows" / "bin" / "gswin64c.exe",
        ],
        "linux": [
            engine_dir / "vendor" / "linux" / "bin" / "tesseract",
            engine_dir / "vendor" / "linux" / "bin" / "gs",
        ],
    }
    missing_bins = [str(path) for path in required_bins.get(platform_name, []) if not path.exists()]
    if missing_bins:
        raise FileNotFoundError(f"missing bundled OCR/runtime binaries: {', '.join(missing_bins)}")

    manifest_path = find_promoted_manifest(model_dir)
    expected = expected_model_sha_map(manifest_path)
    actual = {}
    for name in [
        "global_corner_model.pt",
        "corner_heatmap_model.pt",
        "local_corner_moe_coord_model.pt",
    ]:
        path = model_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing runtime model file: {path}")
        actual[name] = sha1(path)
        expected_hash = expected.get(name)
        if expected_hash and expected_hash != actual[name]:
            raise ValueError(f"runtime model sha1 mismatch for {name}: expected={expected_hash} actual={actual[name]}")

    smoke_result: dict[str, Any] | None = None
    if smoke_image is not None:
        env = os.environ.copy()
        env["SCREEN_PDF_MODEL_DIR"] = str(model_dir)
        cmd = [
            str(bundled_python),
            str(engine_dir / "detect_frame.py"),
            "detect-json",
            "--image",
            str(smoke_image),
        ]
        output = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=engine_dir)
        smoke_result = {
            "command": cmd,
            "returncode": output.returncode,
            "stdout_tail": output.stdout[-1000:],
            "stderr_tail": output.stderr[-1000:],
        }
        if output.returncode != 0:
            raise RuntimeError(f"smoke detection failed with exit code {output.returncode}")

    return {
        "bundle_path": str(bundle_path),
        "platform": platform_name,
        "engine_dir": str(engine_dir),
        "model_dir": str(model_dir),
        "bundled_python": str(bundled_python),
        "manifest_path": str(manifest_path),
        "actual_model_sha1": actual,
        "smoke_result": smoke_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify packaged desktop bundle")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--platform", required=True, choices=["macos", "windows", "linux"])
    parser.add_argument("--smoke-image")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = verify_bundle(
        bundle_path=Path(args.bundle),
        platform_name=args.platform,
        smoke_image=Path(args.smoke_image) if args.smoke_image else None,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
