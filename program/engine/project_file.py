"""Root-anchored project file paths, with relocation support for legacy JSON."""
from __future__ import annotations
from pathlib import Path, PureWindowsPath
from typing import Any

PROJECT_FILE_NAME = 'screen-pdf-project.json'
PROJECT_FILE_VERSION = 2


def project_files(root: Path) -> list[Path]:
    """One project per root; ignore copies nested in its old export directories."""
    selected=[]
    for path in sorted(root.rglob(PROJECT_FILE_NAME),key=lambda p:(len(p.parts),str(p))):
        if not any(parent.parent in path.parents for parent in selected):
            selected.append(path)
    return sorted(selected)


def resolve_image_path(project_path: Path, page: dict[str, Any], project: dict[str, Any] | None = None) -> Path | None:
    raw=str(page.get('path') or '')
    if not raw:return None
    root=project_path.parent.resolve()
    windows=PureWindowsPath(raw)
    absolute=Path(raw).is_absolute() or windows.is_absolute()
    text=raw.replace('\\','/')
    choices=[]
    if not absolute:
        local=(root/text).resolve()
        if not local.is_relative_to(root):return None
        choices.append(local)
    else:
        old_source=str((project or {}).get('sourceDir') or '').replace('\\','/').rstrip('/')
        for prefix in [root.as_posix(),old_source]:
            if prefix and text.startswith(prefix+'/'):
                local=(root/text[len(prefix)+1:]).resolve()
                if local.is_relative_to(root):choices.append(local)
        choices.append(root/text.rsplit('/',1)[-1])
        choices.append(Path(raw))
    return next((p for p in choices if p.is_file()),None)
