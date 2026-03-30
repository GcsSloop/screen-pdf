from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


def repo_root_from(path: Path | None = None) -> Path:
    current = (path or Path(__file__)).resolve()
    if current.is_dir():
        return current
    return current.parents[2]


def desktop_dir(repo_root: Path) -> Path:
    return repo_root / "program" / "desktop"


def tauri_dir(repo_root: Path) -> Path:
    return desktop_dir(repo_root) / "src-tauri"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_desktop_version(repo_root: Path) -> str:
    package_version, tauri_version = verify_desktop_versions_match(repo_root)
    if package_version != tauri_version:
        raise ValueError(f"version mismatch: package.json={package_version} tauri.conf.json={tauri_version}")
    return package_version


def parse_release_tag(tag: str) -> str:
    value = tag.strip()
    match = re.fullmatch(r"v(\d+\.\d+\.\d+)", value)
    if not match:
        raise ValueError(f"invalid app release tag: {tag}")
    return match.group(1)


def validate_release_version(version: str) -> str:
    value = version.strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", value):
        raise ValueError(f"invalid app version: {version}")
    return value


def resolve_requested_release_version(tag: str | None, version: str | None) -> str:
    resolved_tag = parse_release_tag(tag) if tag else None
    resolved_version = validate_release_version(version) if version else None
    if resolved_tag and resolved_version and resolved_tag != resolved_version:
        raise ValueError(
            f"requested app version conflicts with tag: tag={resolved_tag} version={resolved_version}"
        )
    if resolved_tag:
        return resolved_tag
    if resolved_version:
        return resolved_version
    raise ValueError("either tag or version is required")


def verify_desktop_versions_match(repo_root: Path) -> tuple[str, str]:
    package = load_json(desktop_dir(repo_root) / "package.json")
    tauri = load_json(tauri_dir(repo_root) / "tauri.conf.json")
    package_version = str(package["version"]).strip()
    tauri_version = str(tauri["version"]).strip()
    if package_version != tauri_version:
        raise ValueError(f"version mismatch: package.json={package_version} tauri.conf.json={tauri_version}")
    return package_version, tauri_version


def _replace_cargo_version(contents: str, version: str) -> str:
    pattern = r'(?ms)(^\[package\]\s+.*?^version\s*=\s*")([^"]+)(")'
    replaced, count = re.subn(pattern, rf'\g<1>{version}\g<3>', contents, count=1)
    if count != 1:
        raise ValueError("failed to update Cargo.toml package version")
    return replaced


def apply_release_version(repo_root: Path, version: str) -> None:
    desktop_root = desktop_dir(repo_root)
    tauri_root = tauri_dir(repo_root)
    package_path = desktop_root / "package.json"
    tauri_conf_path = tauri_root / "tauri.conf.json"
    cargo_toml_path = tauri_root / "Cargo.toml"

    package = load_json(package_path)
    package["version"] = version
    package_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")

    tauri = load_json(tauri_conf_path)
    tauri["version"] = version
    tauri_conf_path.write_text(json.dumps(tauri, indent=2) + "\n", encoding="utf-8")

    cargo_contents = cargo_toml_path.read_text(encoding="utf-8")
    cargo_toml_path.write_text(_replace_cargo_version(cargo_contents, version), encoding="utf-8")


def normalize_release_platform(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in {"darwin", "mac", "macos", "osx"}:
        return "macos"
    if value in {"win", "win32", "windows"}:
        return "windows"
    if value in {"linux", "linux-gnu"}:
        return "linux"
    if value in {"", "auto"}:
        if sys.platform.startswith("darwin"):
            return "macos"
        if sys.platform.startswith("win"):
            return "windows"
        return "linux"
    raise ValueError(f"unsupported platform: {raw}")


def default_bundle_dir(repo_root: Path) -> Path:
    return tauri_dir(repo_root) / "target" / "release" / "bundle"


def default_release_asset_root(repo_root: Path) -> Path:
    return repo_root / "release-assets"


def _matching_paths(bundle_dir: Path, platform_name: str) -> list[Path]:
    patterns: dict[str, list[str]] = {
        "macos": [
            "macos/*.app",
            "dmg/*.dmg",
            "macos/*.tar.gz",
            "macos/*.sig",
            "updater/*.json",
            "updater/*.sig",
            "updater/*.tar.gz",
            "updater/*.zip",
        ],
        "windows": [
            "msi/*.msi",
            "msi/*.sig",
            "nsis/*.exe",
            "nsis/*.sig",
            "updater/*.zip",
            "updater/*.sig",
            "updater/*.json",
        ],
        "linux": [
            "appimage/*.AppImage",
            "deb/*.deb",
            "rpm/*.rpm",
            "updater/*.tar.gz",
            "updater/*.sig",
            "updater/*.json",
        ],
    }
    matches: list[Path] = []
    for pattern in patterns.get(platform_name, []):
        matches.extend(sorted(bundle_dir.glob(pattern)))
    deduped: list[Path] = []
    for path in matches:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _copy_entry(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return
    shutil.copy2(source, target)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_release_assets(
    version: str,
    platform_name: str,
    bundle_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    platform_name = normalize_release_platform(platform_name)
    output_dir = output_root / version / platform_name
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for source in _matching_paths(bundle_dir, platform_name):
        target = output_dir / source.name
        _copy_entry(source, target)
        digest = _hash_file(target) if target.is_file() else None
        artifacts.append(
            {
                "source_path": str(source),
                "target_path": str(target),
                "sha256": digest,
                "is_dir": target.is_dir(),
            }
        )

    checksum_path = output_dir / "SHA256SUMS.txt"
    lines = [
        f"{item['sha256']}  {Path(item['target_path']).name}"
        for item in artifacts
        if item["sha256"]
    ]
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    summary = {
        "version": version,
        "platform": platform_name,
        "bundle_dir": str(bundle_dir),
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "checksum_file": str(checksum_path),
    }
    (output_dir / "release-assets.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def cli_collect(args: argparse.Namespace) -> int:
    repo_root = repo_root_from(Path(args.repo_root) if args.repo_root else None)
    version = args.version or resolve_desktop_version(repo_root)
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else default_bundle_dir(repo_root)
    output_root = Path(args.output_dir) if args.output_dir else default_release_asset_root(repo_root)
    summary = collect_release_assets(version, args.platform, bundle_dir, output_root)
    print(json.dumps(summary, indent=2))
    return 0


def cli_apply_version(args: argparse.Namespace) -> int:
    repo_root = repo_root_from(Path(args.repo_root) if args.repo_root else None)
    version = resolve_requested_release_version(args.tag, args.version)
    apply_release_version(repo_root, version)
    print(version)
    return 0


def cli_version(args: argparse.Namespace) -> int:
    repo_root = repo_root_from(Path(args.repo_root) if args.repo_root else None)
    version = resolve_desktop_version(repo_root)
    print(version)
    return 0


def cli_platform(args: argparse.Namespace) -> int:
    print(normalize_release_platform(args.platform or os.environ.get("RELEASE_PLATFORM")))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Desktop release metadata helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="Print synchronized desktop version")
    version_parser.add_argument("--repo-root")
    version_parser.set_defaults(func=cli_version)

    platform_parser = subparsers.add_parser("platform", help="Normalize release platform")
    platform_parser.add_argument("--platform")
    platform_parser.set_defaults(func=cli_platform)

    collect_parser = subparsers.add_parser("collect", help="Collect packaged desktop artifacts")
    collect_parser.add_argument("--repo-root")
    collect_parser.add_argument("--version")
    collect_parser.add_argument("--platform", required=True)
    collect_parser.add_argument("--bundle-dir")
    collect_parser.add_argument("--output-dir")
    collect_parser.set_defaults(func=cli_collect)

    apply_parser = subparsers.add_parser("apply-version", help="Apply app version to desktop metadata")
    apply_parser.add_argument("--repo-root")
    apply_parser.add_argument("--tag")
    apply_parser.add_argument("--version")
    apply_parser.set_defaults(func=cli_apply_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
