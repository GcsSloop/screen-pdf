.PHONY: package-desktop package-desktop-macos package-desktop-linux readme-screenshots

package-desktop:
	bash scripts/desktop/package_local_release.sh

package-desktop-macos:
	RELEASE_PLATFORM=macos bash scripts/desktop/package_local_release.sh

package-desktop-linux:
	RELEASE_PLATFORM=linux bash scripts/desktop/package_local_release.sh

# Regenerate docs/assets/screenshots/ with the promoted runtime model.
# Uses the vendored runtime Python (torch + opencv); override with SCREENSHOT_PYTHON=... .
readme-screenshots:
	@PY="$(SCREENSHOT_PYTHON)"; \
	if [ -z "$$PY" ]; then \
	  for candidate in program/engine/vendor/macos/bin/python3 program/engine/vendor/linux/bin/python3; do \
	    if [ -x "$$candidate" ]; then PY="$$candidate"; break; fi; \
	  done; \
	fi; \
	PY="$${PY:-python3}"; \
	echo "python: $$PY"; \
	"$$PY" scripts/docs/generate_readme_screenshots.py
