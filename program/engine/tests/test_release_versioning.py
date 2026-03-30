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
    apply_release_version,
    parse_release_tag,
    resolve_requested_release_version,
)
from scripts.model_release import (  # noqa: E402
    build_model_release_id,
    parse_model_release_tag,
    promote_runtime_manifest,
)


class ReleaseVersioningTests(unittest.TestCase):
    def test_parse_release_tag_accepts_semver_tag(self) -> None:
        self.assertEqual(parse_release_tag("v0.2.1"), "0.2.1")

    def test_parse_release_tag_rejects_non_app_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid app release tag"):
            parse_release_tag("model-20260330-153045-ab12cd34")

    def test_apply_release_version_updates_desktop_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            desktop = root / "program" / "desktop"
            tauri = desktop / "src-tauri"
            tauri.mkdir(parents=True)
            (desktop / "package.json").write_text(json.dumps({"version": "0.2.0"}), encoding="utf-8")
            (tauri / "tauri.conf.json").write_text(json.dumps({"version": "0.2.0"}), encoding="utf-8")
            (tauri / "Cargo.toml").write_text('[package]\nname = "screen-pdf"\nversion = "0.2.0"\n', encoding="utf-8")

            apply_release_version(root, "0.2.1")

            self.assertEqual(json.loads((desktop / "package.json").read_text(encoding="utf-8"))["version"], "0.2.1")
            self.assertEqual(json.loads((tauri / "tauri.conf.json").read_text(encoding="utf-8"))["version"], "0.2.1")
            self.assertIn('version = "0.2.1"', (tauri / "Cargo.toml").read_text(encoding="utf-8"))

    def test_resolve_requested_release_version_prefers_tag(self) -> None:
        self.assertEqual(resolve_requested_release_version("v0.2.1", None), "0.2.1")

    def test_resolve_requested_release_version_accepts_explicit_version(self) -> None:
        self.assertEqual(resolve_requested_release_version(None, "0.2.1"), "0.2.1")

    def test_resolve_requested_release_version_rejects_conflicting_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_requested_release_version("v0.2.1", "0.2.0")

    def test_parse_model_release_tag_accepts_model_tag(self) -> None:
        self.assertEqual(
            parse_model_release_tag("model-20260330-153045-ab12cd34"),
            "model-20260330-153045-ab12cd34",
        )

    def test_build_model_release_id_uses_timestamp_and_digest_prefix(self) -> None:
        release_id = build_model_release_id("2026-03-30T15:30:45Z", "ab12cd34ef567890")
        self.assertEqual(release_id, "model-20260330-153045-ab12cd34")

    def test_promote_runtime_manifest_populates_release_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "runtime.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "public_name": "deep_screen_r1_2026_03_28",
                        "released_at": "2026-03-30T15:30:45Z",
                        "models": {
                            "global": {
                                "runtime_sha1": "1111111111111111111111111111111111111111",
                                "model_id": "r85",
                            },
                            "roi": {
                                "runtime_sha1": "2222222222222222222222222222222222222222",
                                "model_id": "c21",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            promoted = promote_runtime_manifest(manifest_path)

            self.assertEqual(promoted["model_release_id"], "model-20260330-153045-af85df26")
            self.assertEqual(promoted["runtime_digest"], "af85df265db9431fcdcf811f6cdca80059b7658d")


if __name__ == "__main__":
    unittest.main()
