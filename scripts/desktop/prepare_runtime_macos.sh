#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_DIR="$ROOT_DIR/program/engine/vendor/macos/bin"
mkdir -p "$TARGET_DIR"

copy_required_file() {
  local source_path="$1"
  local target_path="$2"
  if [[ -z "$source_path" || ! -e "$source_path" ]]; then
    echo "missing required runtime input: $source_path" >&2
    exit 1
  fi
  cp -R "$source_path" "$target_path"
}

if [[ -n "${SCREEN_PDF_RUNTIME_SOURCE_DIR:-}" ]]; then
  SRC="$SCREEN_PDF_RUNTIME_SOURCE_DIR/macos/bin"
  copy_required_file "$SRC/python3" "$TARGET_DIR/python3"
  copy_required_file "$SRC/tesseract" "$TARGET_DIR/tesseract"
  copy_required_file "$SRC/gs" "$TARGET_DIR/gs"
else
  echo "SCREEN_PDF_RUNTIME_SOURCE_DIR is required for macOS runtime preparation" >&2
  exit 1
fi

chmod +x "$TARGET_DIR/python3" "$TARGET_DIR/tesseract" "$TARGET_DIR/gs"
printf 'Prepared macOS runtime under %s\n' "$TARGET_DIR"
