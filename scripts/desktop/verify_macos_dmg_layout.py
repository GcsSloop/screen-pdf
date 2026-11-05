#!/usr/bin/env python3
"""Verify that a macOS DMG is a drag-to-Applications installer."""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import subprocess
import tempfile
from pathlib import Path


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dmg", required=True, type=Path)
    parser.add_argument("--reference-dmg", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    dmg = args.dmg.resolve()
    if not dmg.is_file():
        raise SystemExit(f"DMG not found: {dmg}")

    mount = Path(tempfile.mkdtemp(prefix="screenpdf-dmg-"))
    attached = False
    reference_mount: Path | None = None
    reference_attached = False
    try:
        run(["hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", str(mount), str(dmg)])
        attached = True
        app = mount / "ScreenPDF.app"
        applications = mount / "Applications"
        if not app.is_dir():
            raise SystemExit("DMG root is missing ScreenPDF.app")
        if not applications.is_symlink() or applications.resolve() != Path("/Applications"):
            raise SystemExit("DMG root must contain Applications -> /Applications")
        if not (mount / ".DS_Store").is_file():
            raise SystemExit("DMG is missing Finder layout metadata (.DS_Store)")
        visible_entries = sorted(path.name for path in mount.iterdir() if not path.name.startswith("."))
        if visible_entries != ["Applications", "ScreenPDF.app"]:
            raise SystemExit(
                "DMG root must show only ScreenPDF.app and Applications; "
                f"found {visible_entries}"
            )

        run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)])
        info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
        layout_match = None
        if args.reference_dmg:
            reference_dmg = args.reference_dmg.resolve()
            if not reference_dmg.is_file():
                raise SystemExit(f"layout reference DMG not found: {reference_dmg}")
            reference_mount = Path(tempfile.mkdtemp(prefix="screenpdf-dmg-reference-"))
            run(["hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", str(reference_mount), str(reference_dmg)])
            reference_ds_store = reference_mount / ".DS_Store"
            reference_volume_icon = reference_mount / ".VolumeIcon.icns"
            if not reference_ds_store.is_file() or not reference_volume_icon.is_file():
                raise SystemExit("layout reference DMG is missing Finder metadata")
            reference_attached = True
            layout_match = {
                "reference_dmg": str(reference_dmg),
                "ds_store_sha256": sha256(mount / ".DS_Store"),
                "reference_ds_store_sha256": sha256(reference_ds_store),
                "volume_icon_sha256": sha256(mount / ".VolumeIcon.icns"),
                "reference_volume_icon_sha256": sha256(reference_volume_icon),
            }
            if (
                layout_match["ds_store_sha256"] != layout_match["reference_ds_store_sha256"]
                or layout_match["volume_icon_sha256"] != layout_match["reference_volume_icon_sha256"]
            ):
                raise SystemExit("DMG Finder layout metadata differs from the reference DMG")
            layout_match["matched"] = True
        report = {
            "dmg_path": str(dmg),
            "dmg_sha256": sha256(dmg),
            "dmg_size_bytes": dmg.stat().st_size,
            "root_entries": sorted(path.name for path in mount.iterdir() if path.name != ".fseventsd"),
            "visible_entries": visible_entries,
            "applications_link": "/Applications",
            "finder_layout": "passed",
            "reference_layout": layout_match,
            "app_path": str(app),
            "bundle_version": info.get("CFBundleShortVersionString"),
            "signature_verification": "passed",
            "drag_to_applications": "passed",
        }
    finally:
        if attached:
            subprocess.run(["hdiutil", "detach", str(mount)], check=True, capture_output=True, text=True)
        mount.rmdir()
        if reference_attached and reference_mount is not None:
            subprocess.run(["hdiutil", "detach", str(reference_mount)], check=True, capture_output=True, text=True)
        if reference_mount is not None:
            reference_mount.rmdir()

    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
