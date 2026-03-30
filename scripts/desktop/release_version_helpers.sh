#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

resolve_release_version() {
  python "$ROOT_DIR/scripts/desktop/release_metadata.py" version --repo-root "$ROOT_DIR"
}

resolve_release_platform() {
  local raw="${1:-${RELEASE_PLATFORM:-auto}}"
  python "$ROOT_DIR/scripts/desktop/release_metadata.py" platform --platform "$raw"
}
