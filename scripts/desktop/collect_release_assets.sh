#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/desktop/release_version_helpers.sh
source "$ROOT_DIR/scripts/desktop/release_version_helpers.sh"

VERSION="${RELEASE_VERSION:-$(resolve_release_version)}"
PLATFORM="$(resolve_release_platform "${RELEASE_PLATFORM:-auto}")"
BUNDLE_DIR="${RELEASE_BUNDLE_DIR:-$ROOT_DIR/program/desktop/src-tauri/target/release/bundle}"
OUTPUT_DIR="${RELEASE_ASSET_DIR:-$ROOT_DIR/release-assets}"

python "$ROOT_DIR/scripts/desktop/release_metadata.py" collect \
  --repo-root "$ROOT_DIR" \
  --version "$VERSION" \
  --platform "$PLATFORM" \
  --bundle-dir "$BUNDLE_DIR" \
  --output-dir "$OUTPUT_DIR"
