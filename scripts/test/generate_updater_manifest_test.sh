#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

assert_contains() {
  local file="$1"
  local pattern="$2"
  if ! grep -Fq "$pattern" "$file"; then
    echo "FAIL: expected $file to contain: $pattern" >&2
    exit 1
  fi
}

run_case() {
  local case_name="$1"
  local mac_asset="$2"
  local windows_asset="$3"
  local case_dir="$TMP_DIR/$case_name"
  local assets_root="$case_dir/release-assets"
  local output="$case_dir/latest.json"

  mkdir -p "$assets_root/0.2.1/macos" "$assets_root/0.2.1/windows"
  printf 'mac-signature\n' >"$assets_root/0.2.1/macos/${mac_asset}.sig"
  printf 'windows-signature\n' >"$assets_root/0.2.1/windows/${windows_asset}.sig"
  : >"$assets_root/0.2.1/macos/${mac_asset}"
  : >"$assets_root/0.2.1/windows/${windows_asset}"

  bash "$ROOT_DIR/scripts/release/generate_updater_manifest.sh" \
    --tag v0.2.1 \
    --assets-root "$assets_root" \
    --output "$output"

  assert_contains "$output" "\"version\": \"0.2.1\""
  assert_contains "$output" "\"url\": \"https://github.com/GcsSloop/screen-pdf/releases/download/v0.2.1/${mac_asset}\""
  assert_contains "$output" "\"url\": \"https://github.com/GcsSloop/screen-pdf/releases/download/v0.2.1/${windows_asset}\""
  assert_contains "$output" "\"signature\": \"mac-signature\""
  assert_contains "$output" "\"signature\": \"windows-signature\""
}

run_case "current-tauri-output" "ScreenPDF.app.tar.gz" "ScreenPDF_0.2.1_x64_en-US.msi"
run_case "legacy-versioned-output" "ScreenPDF_0.2.1_aarch64.app.tar.gz" "ScreenPDF_0.2.1_x64_en-US.msi.zip"

echo "PASS: generate_updater_manifest_test"
