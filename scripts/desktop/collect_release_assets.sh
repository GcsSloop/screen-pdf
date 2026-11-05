#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/desktop/release_version_helpers.sh
source "$ROOT_DIR/scripts/desktop/release_version_helpers.sh"

VERSION="${RELEASE_VERSION:-$(resolve_release_version)}"
PLATFORM="$(resolve_release_platform "${RELEASE_PLATFORM:-auto}")"
BUNDLE_DIR="${RELEASE_BUNDLE_DIR:-$ROOT_DIR/program/desktop/src-tauri/target/release/bundle}"
OUTPUT_DIR="${RELEASE_ASSET_DIR:-$ROOT_DIR/release-assets}"
DMG_LAYOUT_REFERENCE="${SCREEN_PDF_DMG_LAYOUT_REFERENCE:-$ROOT_DIR/release-assets/local-single-20261017-v0216/0.2.16/macos/ScreenPDF_0.2.16_aarch64.dmg}"

if [[ "$PLATFORM" == "macos" ]]; then
  BUNDLE_APP="$(find "$BUNDLE_DIR/macos" -maxdepth 1 -name '*.app' -type d -print0 | xargs -0 ls -td 2>/dev/null | head -n1 || true)"
  DMG_PATH="$(find "$BUNDLE_DIR/dmg" -maxdepth 1 -name '*.dmg' -type f -print0 | xargs -0 ls -t 2>/dev/null | head -n1 || true)"
  if [[ -z "$BUNDLE_APP" || -z "$DMG_PATH" ]]; then
    echo "macOS release collection requires both an app bundle and a DMG" >&2
    exit 1
  fi
  python "$ROOT_DIR/scripts/desktop/verify_macos_dmg_layout.py" \
    --dmg "$DMG_PATH" \
    --reference-dmg "$DMG_LAYOUT_REFERENCE"
fi

python "$ROOT_DIR/scripts/desktop/release_metadata.py" collect \
  --repo-root "$ROOT_DIR" \
  --version "$VERSION" \
  --platform "$PLATFORM" \
  --bundle-dir "$BUNDLE_DIR" \
  --output-dir "$OUTPUT_DIR"

if [[ "$PLATFORM" == "macos" ]]; then
  COLLECTED_DIR="$OUTPUT_DIR/$VERSION/$PLATFORM"
  COPIED_DMG="$(find "$COLLECTED_DIR" -maxdepth 1 -name '*.dmg' -type f -print0 | xargs -0 ls -t 2>/dev/null | head -n1 || true)"
  if [[ -z "$COPIED_DMG" ]]; then
    echo "macOS release collection did not produce a DMG" >&2
    exit 1
  fi
  python "$ROOT_DIR/scripts/desktop/verify_macos_dmg_layout.py" \
    --dmg "$COPIED_DMG" \
    --reference-dmg "$DMG_LAYOUT_REFERENCE" \
    --report "$COLLECTED_DIR/dmg-verification.json"
fi
