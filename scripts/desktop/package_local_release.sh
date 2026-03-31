#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/desktop/release_version_helpers.sh
source "$ROOT_DIR/scripts/desktop/release_version_helpers.sh"

VERSION="$(resolve_release_version)"
PLATFORM="$(resolve_release_platform "${RELEASE_PLATFORM:-auto}")"
ASSET_DIR="${RELEASE_ASSET_DIR:-$ROOT_DIR/release-assets}"
SMOKE_IMAGE="${SCREEN_PDF_SMOKE_IMAGE:-}"

case "$PLATFORM" in
  macos)
    if [[ "${SKIP_RUNTIME_PREPARE:-0}" != "1" ]]; then
      bash "$ROOT_DIR/scripts/desktop/prepare_runtime_macos.sh"
    fi
    ;;
  linux)
    if [[ "${SKIP_RUNTIME_PREPARE:-0}" != "1" ]]; then
      bash "$ROOT_DIR/scripts/desktop/prepare_runtime_linux.sh"
    fi
    ;;
  windows)
    if [[ "${SKIP_RUNTIME_PREPARE:-0}" != "1" ]]; then
      pwsh -File "$ROOT_DIR/scripts/desktop/prepare_runtime_windows.ps1"
    fi
    ;;
  *)
    echo "unsupported platform: $PLATFORM" >&2
    exit 1
    ;;
esac

bash "$ROOT_DIR/scripts/release/sync_release_metadata.sh"
pnpm --dir "$ROOT_DIR/program/desktop" tauri build

if [[ "$PLATFORM" == "macos" ]]; then
  bash "$ROOT_DIR/scripts/desktop/notarize_macos.sh"
fi

RELEASE_VERSION="$VERSION" RELEASE_PLATFORM="$PLATFORM" RELEASE_ASSET_DIR="$ASSET_DIR" \
  bash "$ROOT_DIR/scripts/desktop/collect_release_assets.sh"

if [[ "$PLATFORM" == "macos" ]]; then
  BUNDLE_PATH="$(find "$ROOT_DIR/program/desktop/src-tauri/target/release/bundle/macos" -maxdepth 1 -name '*.app' -type d -print0 | xargs -0 ls -td 2>/dev/null | head -n1 || true)"
  if [[ -z "$BUNDLE_PATH" ]]; then
    echo "macOS app bundle not found after tauri build" >&2
    exit 1
  fi
  VERIFY_ARGS=(
    --bundle "$BUNDLE_PATH"
    --platform "$PLATFORM"
  )
  if [[ -n "$SMOKE_IMAGE" ]]; then
    VERIFY_ARGS+=(--smoke-image "$SMOKE_IMAGE")
  fi
  python "$ROOT_DIR/scripts/desktop/verify_desktop_bundle.py" "${VERIFY_ARGS[@]}"
else
  echo "Bundle verification is currently implemented for macOS app bundles. Skipping packaged artifact verification for $PLATFORM." >&2
fi

printf 'Packaged ScreenPDF %s for %s into %s\n' "$VERSION" "$PLATFORM" "$ASSET_DIR"
