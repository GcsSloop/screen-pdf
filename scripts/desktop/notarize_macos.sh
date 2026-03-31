#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_ROOT="${RELEASE_BUNDLE_DIR:-$ROOT_DIR/program/desktop/src-tauri/target/release/bundle}"
APP_PATH="$(find "$BUNDLE_ROOT/macos" -maxdepth 1 -name "*.app" -type d -print0 | xargs -0 ls -td 2>/dev/null | head -n1 || true)"
DMG_PATH="$(find "$BUNDLE_ROOT/dmg" -maxdepth 1 -name "*.dmg" -type f -print0 | xargs -0 ls -t 2>/dev/null | head -n1 || true)"
REQUIRE_NOTARIZATION="${REQUIRE_MACOS_NOTARIZATION:-0}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 1
  fi
}

if [[ -z "$APP_PATH" ]]; then
  echo "No macOS app bundle found under $BUNDLE_ROOT, skip notarization"
  exit 0
fi

if [[ "$REQUIRE_NOTARIZATION" == "1" ]]; then
  require_env APPLE_SIGNING_IDENTITY
  require_env APPLE_API_KEY_PATH
  require_env APPLE_API_KEY_ID
  require_env APPLE_API_ISSUER
fi

if [[ "${APPLE_SIGNING_IDENTITY:-}" == "-" ]]; then
  echo "Using ad-hoc codesign for macOS bundle"
  codesign --force --deep --sign - "$APP_PATH"
elif [[ -n "${APPLE_SIGNING_IDENTITY:-}" ]]; then
  echo "Code signing app with Developer ID identity"
  codesign --force --deep --options runtime --timestamp \
    --sign "$APPLE_SIGNING_IDENTITY" \
    "$APP_PATH"
else
  echo "APPLE_SIGNING_IDENTITY not set, falling back to ad-hoc codesign"
  codesign --force --deep --sign - "$APP_PATH"
fi

codesign --verify --deep --strict --verbose=2 "$APP_PATH"

if [[ "${APPLE_SIGNING_IDENTITY:-}" == "-" ]]; then
  echo "Developer ID identity missing, skip notarization for ad-hoc signed bundle"
  exit 0
fi

if [[ -z "${APPLE_API_KEY_PATH:-}" || -z "${APPLE_API_KEY_ID:-}" || -z "${APPLE_API_ISSUER:-}" ]]; then
  echo "Apple notarization credentials are incomplete, skip notarization"
  exit 0
fi

if [[ -n "$DMG_PATH" ]]; then
  NOTARY_INPUT="$DMG_PATH"
else
  NOTARY_INPUT="$ROOT_DIR/program/desktop/src-tauri/target/notary/ScreenPDF.zip"
  mkdir -p "$(dirname "$NOTARY_INPUT")"
  ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$NOTARY_INPUT"
fi

xcrun notarytool submit "$NOTARY_INPUT" \
  --key "$APPLE_API_KEY_PATH" \
  --key-id "$APPLE_API_KEY_ID" \
  --issuer "$APPLE_API_ISSUER" \
  --wait

xcrun stapler staple "$APP_PATH"
if [[ -n "$DMG_PATH" ]]; then
  xcrun stapler staple "$DMG_PATH"
fi

spctl -a -vv -t exec "$APP_PATH"
