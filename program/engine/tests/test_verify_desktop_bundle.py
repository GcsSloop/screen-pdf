from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.desktop.verify_desktop_bundle import (  # noqa: E402
    expected_model_sha_map,
    resolve_resource_path_from_base,
    sha1,
    verify_bundle,
)


class VerifyDesktopBundleTests(unittest.TestCase):
    def _single_bundle(self, root: Path):
        bundle = root / "ScreenPDF.app"
        resources = bundle / "Contents" / "Resources"
        engine = resources / "engine"
        binaries = engine / "vendor" / "macos" / "bin"
        binaries.mkdir(parents=True)
        for name in ("python3", "tesseract", "gs"):
            (binaries / name).write_bytes(b"binary")
        (engine / "screen_corner_single.py").write_text("# loader")
        models = resources / "models" / "runtime"
        models.mkdir(parents=True)
        model = models / "screen_corner_single.pt"
        model.write_bytes(b"single-graph")
        manifest = {"architecture": "screen_corner_single_v1", "status": "packaged_candidate",
                    "artifacts": {"single": {"path": model.name, "sha1": sha1(model),
                    "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}}}
        (models / "screen_corner_single.json").write_text(json.dumps(manifest))
        return bundle, models, engine

    def test_single_graph_bundle_needs_only_one_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _, _ = self._single_bundle(Path(tmp))
            result = verify_bundle(bundle, "macos")
            self.assertEqual(set(result["actual_model_sha1"]), {"screen_corner_single.pt"})

    def test_single_graph_bundle_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, models, _ = self._single_bundle(Path(tmp))
            (models / "screen_corner_single.pt").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "sha1 mismatch"):
                verify_bundle(bundle, "macos")

    def test_single_graph_bundle_rejects_missing_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _, engine = self._single_bundle(Path(tmp))
            (engine / "screen_corner_single.py").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "loader"):
                verify_bundle(bundle, "macos")

    def test_single_graph_bundle_rejects_extra_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, models, _ = self._single_bundle(Path(tmp))
            (models / "global_corner_model.pt").write_bytes(b"old")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                verify_bundle(bundle, "macos")

    def test_resolve_resource_path_from_base_walks_up_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resource_dir = root / "Resources"
            nested = resource_dir / "_up_" / "_up_" / "models" / "runtime"
            nested.mkdir(parents=True)
            resolved = resolve_resource_path_from_base(resource_dir, Path("models") / "runtime")
            self.assertEqual(resolved, nested)

    def test_expected_model_sha_map_reads_manifest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "global_corner_model.pt"
            model_path.write_bytes(b"abc")
            manifest = root / "deep_screen_r1.json"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "promoted",
                        "artifacts": {
                            "global": {
                                "path": "global_corner_model.pt",
                                "sha1": sha1(model_path),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            values = expected_model_sha_map(manifest)
            self.assertEqual(values["global_corner_model.pt"], sha1(model_path))


if __name__ == "__main__":
    unittest.main()
