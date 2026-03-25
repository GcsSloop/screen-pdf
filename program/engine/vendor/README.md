Place bundled runtime files here for green builds.

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

The Python engine already checks these paths before falling back to system binaries.
