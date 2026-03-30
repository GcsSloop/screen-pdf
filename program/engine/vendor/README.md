Place packaged runtime files here for self-contained desktop builds.

Expected layout:

- `macos/bin/python3`
- `windows/bin/python.exe`
- `linux/bin/python3`
- `macos/bin/tesseract`
- `windows/bin/tesseract.exe`
- `linux/bin/tesseract`
- `macos/bin/gs`
- `windows/bin/gswin64c.exe`
- `linux/bin/gs`

Optional additional payloads may be staged alongside the binaries when a platform runtime needs them, for example:

- `macos/lib/...`
- `windows/lib/...`
- `linux/lib/...`
- OCR language data directories such as `tessdata/`

This directory is owned by the packaging scripts under `scripts/desktop/`.

- `scripts/desktop/prepare_runtime_macos.sh`
- `scripts/desktop/prepare_runtime_windows.ps1`
- `scripts/desktop/prepare_runtime_linux.sh`

Default contract:

- packaging scripts populate `program/engine/vendor/<platform>/`
- desktop bundle includes this directory as a Tauri resource
- runtime verification rejects a release bundle when the expected files are missing

The Python engine checks these bundled paths before falling back to system binaries during development. Release verification is stricter and requires the bundled runtime to be present.
