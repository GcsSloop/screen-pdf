"""Create editable, portable desktop projects without writing through to sources."""
from __future__ import annotations
import argparse
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import sys

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from project_file import resolve_image_path
from training.build_conference_collections import digest, write_json


def independent_copy(source, target):
    """APFS copy-on-write clone; never a hard link or a symlink to original photos."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == 'darwin':
        libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
        clone = libc.clonefile
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        if clone(bytes(source), bytes(target), 0) == 0:
            return 'apfs_clone'
    shutil.copy2(source, target)
    return 'copy'


def materialize(collections):
    index = json.loads((collections/'index.json').read_text())
    excluded = json.loads((collections/'excluded_pages.json').read_text())
    blocked = {(r['project_path'], r['page_id']) for r in excluded
               if r['reason'] in {'invalid_manual_quad', 'duplicate_label_or_role_conflict'}}
    output = collections/'desktop-projects'
    if output.exists():
        raise FileExistsError('Desktop project copies already exist; preserve user corrections')
    output.mkdir()
    audit = []
    for group in index['collections']:
        folder = output/group['name']
        folder.mkdir()
        pages = []; provenance = []; images = {}
        for member in group['member_projects']:
            source = Path(member['project_path'])
            assert digest(source) == index['source_project_sha256'][str(source)]
            payload = json.loads(source.read_text())
            for i, page in enumerate(payload.get('pages', [])):
                if (str(source), page.get('id')) in blocked:
                    continue
                image = resolve_image_path(source, page, payload)
                if image is None:
                    raise FileNotFoundError(f'Missing source image: {source}, {page.get("id")}')
                sha = digest(image)
                if sha not in images:
                    relative = Path('images')/(sha+image.suffix.lower())
                    target = folder/relative
                    mode = independent_copy(image, target)
                    assert digest(target) == sha
                    assert target.stat().st_ino != image.stat().st_ino
                    images[sha] = relative.as_posix()
                else:
                    mode = 'shared_within_collection_copy'
                item = copy.deepcopy(page)
                item['id'] = f'{member["project_id"]}_{i}_{page.get("id", "page")}'
                item['name'] = f'{member["project_name"]} / {page.get("name", image.name)}'
                item['path'] = images[sha]
                item['thumbPath'] = item['previewPath'] = None
                pages.append(item)
                provenance.append(dict(collection_page_id=item['id'], source_project=str(source),
                                       source_page_id=page.get('id'), source_image=str(image), image_sha256=sha, copy_mode=mode))
        project = dict(version=2, dataStructureVersion=2, sourceDir='.', projectPath='screen-pdf-project.json',
                       selectedPageId=pages[0]['id'] if pages else None, pages=pages)
        write_json(folder/'screen-pdf-project.json', project)
        write_json(folder/'source-provenance.json', provenance)
        collection_readme=collections/group['id']/'README.md'
        with collection_readme.open('a') as handle:
            handle.write(f'\n[打开本场次全部 {len(pages)} 页的 ScreenPDF 合并项目](<../desktop-projects/{group["name"]}/screen-pdf-project.json>)\n')
        (folder/'README.md').write_text('# '+group['name']+'\n\n打开本目录的 screen-pdf-project.json，可集中查看所有成员演讲。图片为独立副本，编辑此项目不会改动原始项目。\n\nsource-provenance.json 记录每页原项目与图片；本副本的后续人工校正不会自动回写原项目或已冻结基准。\n')
        audit.append(dict(collection_id=group['id'], project_path=str(folder/'screen-pdf-project.json'),
                          pages=len(pages), distinct_images=len(images)))
        print(group['name'], len(pages), flush=True)
    assert all(digest(p) == sha for p,sha in index['source_project_sha256'].items())
    write_json(collections/'desktop-projects.json', dict(projects=audit, source_projects_unchanged=True,
               excluded_anomaly_pages=len(blocked),
               image_copy_policy='independent APFS clone or byte copy; no original-image hardlinks or symlinks'))
    return audit


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collections', required=True, type=Path)
    materialize(p.parse_args().collections.resolve())
