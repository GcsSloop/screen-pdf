#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_SCRIPT_PATH="$ROOT_DIR/scripts/desktop/package_local_release.sh"
HELPER_PATH="$ROOT_DIR/scripts/desktop/release_version_helpers.sh"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "expected file: $path"
}

assert_contains() {
  local file="$1"
  local pattern="$2"
  local content
  content="$(cat "$file" 2>/dev/null || true)"
  if [[ "$content" != *"$pattern"* ]]; then
    echo "=== $file ===" >&2
    cat "$file" >&2 || true
    fail "expected $file to contain $pattern"
  fi
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

repo_dir="$tmp_dir/repo"
mkdir -p "$repo_dir/scripts/release" "$repo_dir/scripts/desktop" "$repo_dir/bin" "$repo_dir/program/desktop"
cp "$SOURCE_SCRIPT_PATH" "$repo_dir/scripts/desktop/package_local_release.sh"
cp "$HELPER_PATH" "$repo_dir/scripts/desktop/release_version_helpers.sh"
mkdir -p "$repo_dir/program/desktop/src-tauri/target/release/bundle/macos/ScreenPDF.app"

cat >"$repo_dir/scripts/release/sync_release_metadata.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'sync:%s\n' "$*" >>"$CALL_LOG"
EOF
chmod +x "$repo_dir/scripts/release/sync_release_metadata.sh"

cat >"$repo_dir/scripts/desktop/collect_release_assets.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'collect:%s:%s\n' "${RELEASE_VERSION:-missing}" "${RELEASE_PLATFORM:-missing}" >>"$CALL_LOG"
mkdir -p "${RELEASE_ASSET_DIR:-$PWD/release-assets}"
printf 'artifact\n' >"${RELEASE_ASSET_DIR:-$PWD/release-assets}/artifact.txt"
EOF
chmod +x "$repo_dir/scripts/desktop/collect_release_assets.sh"

cat >"$repo_dir/scripts/desktop/prepare_runtime_macos.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'prepare-runtime:macos\n' >>"$CALL_LOG"
EOF
chmod +x "$repo_dir/scripts/desktop/prepare_runtime_macos.sh"

cat >"$repo_dir/scripts/desktop/notarize_macos.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'notarize:macos\n' >>"$CALL_LOG"
EOF
chmod +x "$repo_dir/scripts/desktop/notarize_macos.sh"

cat >"$repo_dir/bin/pnpm" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'pnpm:%s\n' "$*" >>"$CALL_LOG"
EOF
chmod +x "$repo_dir/bin/pnpm"

cat >"$repo_dir/bin/python" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'python:%s\n' "$*" >>"$CALL_LOG"
if [[ "${1:-}" == *"release_metadata.py" && "${2:-}" == "version" ]]; then
  printf '2.3.4\n'
elif [[ "${1:-}" == *"release_metadata.py" && "${2:-}" == "platform" ]]; then
  printf '%s\n' "${@: -1}"
fi
EOF
chmod +x "$repo_dir/bin/python"

(
  cd "$repo_dir"
  git init >/dev/null
  git config user.name "Codex"
  git config user.email "codex@example.com"
  printf 'seed\n' >README.md
  git add README.md
  git commit -m "init" >/dev/null
  git tag v2.3.4
  printf 'next\n' >>README.md
  git add README.md
  git commit -m "next" >/dev/null

  CALL_LOG="$tmp_dir/calls.log" \
  PATH="$repo_dir/bin:$PATH" \
  RELEASE_PLATFORM="macos" \
  RELEASE_ASSET_DIR="$tmp_dir/release-assets" \
  bash "$repo_dir/scripts/desktop/package_local_release.sh" >/dev/null
)

assert_file "$tmp_dir/release-assets/artifact.txt"
assert_contains "$tmp_dir/calls.log" "prepare-runtime:macos"
assert_contains "$tmp_dir/calls.log" "sync:"
assert_contains "$tmp_dir/calls.log" "pnpm:--dir"
assert_contains "$tmp_dir/calls.log" "notarize:macos"

echo "PASS: package_local_release_test"
