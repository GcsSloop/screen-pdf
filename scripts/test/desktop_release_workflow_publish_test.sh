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
assert_contains 'Prepare notarization key'
assert_contains 'APPLE_SIGNING_IDENTITY'
assert_contains 'APPLE_API_KEY'
assert_contains 'APPLE_API_KEY_ID'
assert_contains 'APPLE_API_ISSUER'
assert_contains 'APPLE_API_KEY_PATH'
assert_contains 'APPLE_API_KEY not set, skip notarization key preparation'
assert_contains 'Package desktop app'
assert_contains 'APPLE_SIGNING_IDENTITY: ${{ secrets.APPLE_SIGNING_IDENTITY }}'
assert_contains 'name: Collect publish files'
assert_contains '**/*.dmg'
assert_contains '**/*.msi'
assert_contains '**/*.AppImage'
assert_contains '**/*.deb'
assert_contains '**/*.rpm'
assert_contains '**/*.zip'
assert_contains '**/*.tar.gz'
assert_contains 'latest = root / "latest.json"'
assert_contains 'Missing required release assets:'
assert_contains 'files: ${{ steps.publish_files.outputs.files }}'
assert_not_contains '.sig'
assert_not_contains 'choco install tesseract ghostscript -y'
assert_not_contains 'release-assets.json'
assert_not_contains 'release-metadata.json'
assert_not_contains 'SHA256SUMS.txt'

echo "PASS: desktop_release_workflow_publish_test"
