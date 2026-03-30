#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

tag=""
repo="GcsSloop/screen-pdf"
assets_root=""
output=""
notes=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/release/generate_updater_manifest.sh \
    --tag v0.2.1 \
    --assets-root release-assets \
    --output release-assets/latest.json \
    [--repo GcsSloop/screen-pdf] \
    [--notes "Release notes"]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      tag="$2"
      shift 2
      ;;
    --repo)
      repo="$2"
      shift 2
      ;;
    --assets-root)
      assets_root="$2"
      shift 2
      ;;
    --output)
      output="$2"
      shift 2
      ;;
    --notes)
      notes="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

[[ -n "$tag" ]] || {
  echo "--tag is required" >&2
  exit 1
}
[[ -n "$assets_root" ]] || {
  echo "--assets-root is required" >&2
  exit 1
}
[[ -n "$output" ]] || {
  echo "--output is required" >&2
  exit 1
}

version="${tag#v}"

find_asset_by_glob() {
  local pattern="$1"
  find "$assets_root" -type f -name "$pattern" | sort | head -n1 || true
}

find_first_matching_asset() {
  local path=""
  local pattern=""
  for pattern in "$@"; do
    path="$(find_asset_by_glob "$pattern")"
    if [[ -n "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 0
}

read_trimmed_file() {
  local path="$1"
  tr -d '\r\n' <"$path"
}

mac_path="$(find_first_matching_asset \
  "ScreenPDF_${version}_*.app.tar.gz" \
  "ScreenPDF.app.tar.gz")"
windows_path="$(find_first_matching_asset \
  "ScreenPDF_${version}_*_en-US.msi.zip" \
  "ScreenPDF_${version}_*_en-US.msi")"
mac_asset="$(basename "$mac_path")"
windows_asset="$(basename "$windows_path")"
mac_sig_path="${mac_path}.sig"
windows_sig_path="${windows_path}.sig"
mac_sig_asset="$(basename "$mac_sig_path")"
windows_sig_asset="$(basename "$windows_sig_path")"

[[ -f "$mac_path" ]] || {
  echo "Missing macOS updater asset: $mac_asset" >&2
  exit 1
}
[[ -f "$mac_sig_path" ]] || {
  echo "Missing macOS updater signature: $mac_sig_asset" >&2
  exit 1
}
[[ -f "$windows_path" ]] || {
  echo "Missing Windows updater asset: $windows_asset" >&2
  exit 1
}
[[ -f "$windows_sig_path" ]] || {
  echo "Missing Windows updater signature: $windows_sig_asset" >&2
  exit 1
}

mac_signature="$(read_trimmed_file "$mac_sig_path")"
windows_signature="$(read_trimmed_file "$windows_sig_path")"
pub_date="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
mkdir -p "$(dirname "$output")"

VERSION="$version" \
NOTES="$notes" \
PUB_DATE="$pub_date" \
REPO="$repo" \
TAG="$tag" \
MAC_ASSET="$mac_asset" \
MAC_SIGNATURE="$mac_signature" \
WINDOWS_ASSET="$windows_asset" \
WINDOWS_SIGNATURE="$windows_signature" \
OUTPUT_FILE="$output" \
node <<'EOF'
const fs = require("fs");

const output = process.env.OUTPUT_FILE;
const repo = process.env.REPO;
const tag = process.env.TAG;
const baseUrl = `https://github.com/${repo}/releases/download/${tag}`;

const manifest = {
  version: process.env.VERSION,
  notes: process.env.NOTES || "",
  pub_date: process.env.PUB_DATE,
  platforms: {
    "darwin-aarch64": {
      signature: process.env.MAC_SIGNATURE,
      url: `${baseUrl}/${process.env.MAC_ASSET}`,
    },
    "darwin-x86_64": {
      signature: process.env.MAC_SIGNATURE,
      url: `${baseUrl}/${process.env.MAC_ASSET}`,
    },
    "windows-x86_64": {
      signature: process.env.WINDOWS_SIGNATURE,
      url: `${baseUrl}/${process.env.WINDOWS_ASSET}`,
    },
  },
};

fs.writeFileSync(output, JSON.stringify(manifest, null, 2) + "\n");
EOF

printf 'Generated updater manifest at %s\n' "$output"
