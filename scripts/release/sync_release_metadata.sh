#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tag=""
version=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      tag="$2"
      shift 2
      ;;
    --version)
      version="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/release/sync_release_metadata.sh [--tag v0.2.1 | --version 0.2.1]
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "$tag" || -n "$version" ]]; then
  VERSION="$(python "$ROOT_DIR/scripts/desktop/release_metadata.py" apply-version --repo-root "$ROOT_DIR" --tag "$tag" --version "$version")"
else
  VERSION="$(python "$ROOT_DIR/scripts/desktop/release_metadata.py" version --repo-root "$ROOT_DIR")"
fi
OUT_DIR="${RELEASE_ASSET_DIR:-$ROOT_DIR/release-assets/$VERSION/meta}"

mkdir -p "$OUT_DIR"
cat >"$OUT_DIR/release-metadata.json" <<EOF
{
  "version": "$VERSION",
  "tag": "v$VERSION",
  "generatedAt": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

printf 'Synchronized desktop release metadata for %s\n' "$VERSION"
