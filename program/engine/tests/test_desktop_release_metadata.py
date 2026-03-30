from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.desktop.release_metadata import (  # noqa: E402
    collect_release_assets,
    normalize_release_platform,
    resolve_desktop_version,
    verify_desktop_versions_match,
)


class ReleaseMetadataTests(unittest.TestCase):
    def test_resolve_desktop_version_reads_matching_desktop_versions(self) -> None:
        version = resolve_desktop_version(REPO_ROOT)
        self.assertEqual(version, "0.2.1")

    def test_verify_desktop_versions_match_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            desktop_dir = root / "program" / "desktop"
            tauri_dir = desktop_dir / "src-tauri"
            tauri_dir.mkdir(parents=True)
            (desktop_dir / "package.json").write_text(json.dumps({"version": "1.2.3"}), encoding="utf-8")
            (tauri_dir / "tauri.conf.json").write_text(json.dumps({"version": "9.9.9"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                verify_desktop_versions_match(root)

    def test_normalize_release_platform_maps_common_aliases(self) -> None:
        self.assertEqual(normalize_release_platform("darwin"), "macos")
        self.assertEqual(normalize_release_platform("macos"), "macos")
        self.assertEqual(normalize_release_platform("win32"), "windows")
        self.assertEqual(normalize_release_platform("linux"), "linux")

    def test_collect_release_assets_copies_matching_artifacts_and_writes_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = root / "program" / "desktop" / "src-tauri" / "target" / "release" / "bundle"
            dmg_dir = bundle_dir / "dmg"
            macos_dir = bundle_dir / "macos"
            dmg_dir.mkdir(parents=True)
            macos_dir.mkdir(parents=True)
            (dmg_dir / "ScreenPDF_0.2.0_aarch64.dmg").write_bytes(b"dmg")
            (macos_dir / "ScreenPDF.app").mkdir()
            output_dir = root / "release-assets"

            collected = collect_release_assets(
                version="0.2.0",
                platform_name="macos",
                bundle_dir=bundle_dir,
                output_root=output_dir,
            )

            paths = [Path(item["target_path"]) for item in collected["artifacts"]]
            self.assertEqual(len(paths), 2)
            self.assertTrue(any(path.name == "ScreenPDF_0.2.0_aarch64.dmg" for path in paths))
            self.assertTrue(any(path.name == "ScreenPDF.app" for path in paths))
            checksum_file = output_dir / "0.2.0" / "macos" / "SHA256SUMS.txt"
            self.assertTrue(checksum_file.exists())

    def test_collect_release_assets_includes_windows_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle_dir = root / "program" / "desktop" / "src-tauri" / "target" / "release" / "bundle"
            msi_dir = bundle_dir / "msi"
            nsis_dir = bundle_dir / "nsis"
            msi_dir.mkdir(parents=True)
            nsis_dir.mkdir(parents=True)
            (msi_dir / "ScreenPDF_0.2.1_x64_en-US.msi").write_bytes(b"msi")
            (msi_dir / "ScreenPDF_0.2.1_x64_en-US.msi.sig").write_text("msi-signature", encoding="utf-8")
            (nsis_dir / "ScreenPDF_0.2.1_x64-setup.exe").write_bytes(b"exe")
            (nsis_dir / "ScreenPDF_0.2.1_x64-setup.exe.sig").write_text("exe-signature", encoding="utf-8")
            output_dir = root / "release-assets"

            collected = collect_release_assets(
                version="0.2.1",
                platform_name="windows",
                bundle_dir=bundle_dir,
                output_root=output_dir,
            )

            paths = {Path(item["target_path"]).name for item in collected["artifacts"]}
            self.assertIn("ScreenPDF_0.2.1_x64_en-US.msi", paths)
            self.assertIn("ScreenPDF_0.2.1_x64_en-US.msi.sig", paths)
            self.assertIn("ScreenPDF_0.2.1_x64-setup.exe", paths)
            self.assertIn("ScreenPDF_0.2.1_x64-setup.exe.sig", paths)


if __name__ == "__main__":
    unittest.main()
