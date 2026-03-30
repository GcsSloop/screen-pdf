from __future__ import annotations

import json
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
)


class VerifyDesktopBundleTests(unittest.TestCase):
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
