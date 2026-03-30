#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local file="$1"
  local pattern="$2"
  if ! grep -Fq "$pattern" "$file"; then
    fail "expected $file to contain: $pattern"
  fi
}

assert_contains "$ROOT_DIR/program/desktop/src-tauri/Cargo.toml" 'tauri-plugin-updater = "2"'
assert_contains "$ROOT_DIR/program/desktop/src-tauri/src/main.rs" '.plugin(tauri_plugin_updater::Builder::new().build())'
assert_contains "$ROOT_DIR/program/desktop/src-tauri/src/main.rs" 'check_for_app_update'
assert_contains "$ROOT_DIR/program/desktop/src-tauri/src/main.rs" 'install_app_update'
assert_contains "$ROOT_DIR/program/desktop/src-tauri/capabilities/default.json" '"updater:default"'
assert_contains "$ROOT_DIR/program/desktop/src-tauri/tauri.conf.json" '"createUpdaterArtifacts": true'
assert_contains "$ROOT_DIR/program/desktop/src-tauri/tauri.conf.json" '"https://github.com/GcsSloop/screen-pdf/releases/latest/download/latest.json"'
assert_contains "$ROOT_DIR/program/desktop/src-tauri/tauri.conf.json" '"pubkey"'

echo "PASS: desktop_updater_config_test"
