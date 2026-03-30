#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKFLOW_PATH="$ROOT_DIR/.github/workflows/desktop-release.yml"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

assert_contains() {
  local pattern="$1"
  if ! grep -Fq -- "$pattern" "$WORKFLOW_PATH"; then
    fail "expected release workflow to contain: $pattern"
  fi
}

assert_not_contains() {
  local pattern="$1"
  if grep -Fq -- "$pattern" "$WORKFLOW_PATH"; then
    fail "expected release workflow to not contain: $pattern"
  fi
}

assert_contains 'bash scripts/test/desktop_release_workflow_publish_test.sh'
assert_contains '--repo "${GITHUB_REPOSITORY}"'
assert_contains 'choco install tesseract -y'
assert_contains 'ghostscriptInstallerUrl = "https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10070/gs10070w64.exe"'
assert_contains 'Get-Command 7z.exe'
assert_contains 'gswin64c.exe not found in extracted Ghostscript installer'
assert_contains 'release-assets/**/*.dmg'
assert_contains 'release-assets/**/*.msi'
assert_contains 'release-assets/**/*.AppImage'
assert_contains 'release-assets/**/*.deb'
assert_contains 'release-assets/**/*.rpm'
assert_contains 'release-assets/**/*.zip'
assert_contains 'release-assets/**/*.tar.gz'
assert_contains 'release-assets/latest.json'
assert_contains 'fail_on_unmatched_files: true'
assert_not_contains '.sig'
assert_not_contains 'choco install tesseract ghostscript -y'
assert_not_contains 'release-assets.json'
assert_not_contains 'release-metadata.json'
assert_not_contains 'SHA256SUMS.txt'

echo "PASS: desktop_release_workflow_publish_test"
