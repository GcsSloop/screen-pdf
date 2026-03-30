.PHONY: package-desktop package-desktop-macos package-desktop-linux

package-desktop:
	bash scripts/desktop/package_local_release.sh

package-desktop-macos:
	RELEASE_PLATFORM=macos bash scripts/desktop/package_local_release.sh

package-desktop-linux:
	RELEASE_PLATFORM=linux bash scripts/desktop/package_local_release.sh
