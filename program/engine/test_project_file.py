from pathlib import Path
import json

from project_file import project_files,resolve_image_path


def test_relative_paths_ignore_working_directory(tmp_path,monkeypatch):
    root=tmp_path/'renamed';root.mkdir();(root/'same.jpg').write_bytes(b'project')
    other=tmp_path/'cwd';other.mkdir();(other/'same.jpg').write_bytes(b'wrong')
    monkeypatch.chdir(other)
    assert resolve_image_path(root/'screen-pdf-project.json',{'path':'same.jpg'})==root/'same.jpg'


def test_legacy_copy_rebases_subdirectories_before_existing_absolute_source(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new'
    for root in [old,new]:
        (root/'photos').mkdir(parents=True);(root/'photos/p.jpg').write_bytes(b'photo')
    assert resolve_image_path(new/'screen-pdf-project.json',{'path':str(old/'photos/p.jpg')},{'sourceDir':str(old)})==new/'photos/p.jpg'


def test_windows_paths_are_portable_on_another_system(tmp_path):
    (tmp_path/'photos').mkdir();(tmp_path/'photos/p.jpg').write_bytes(b'photo')
    assert resolve_image_path(tmp_path/'screen-pdf-project.json',{'path':'C:\\old\\photos\\p.jpg'},{'sourceDir':'C:\\old'})==tmp_path/'photos/p.jpg'


def test_discovery_ignores_nested_export_copies(tmp_path):
    for root in [tmp_path/'one',tmp_path/'one/one_export',tmp_path/'two']:
        root.mkdir(parents=True,exist_ok=True);(root/'screen-pdf-project.json').write_text('{}')
    assert project_files(tmp_path)==[tmp_path/'one/screen-pdf-project.json',tmp_path/'two/screen-pdf-project.json']


def test_relative_parent_escape_is_rejected(tmp_path):
    root=tmp_path/'project';root.mkdir();(tmp_path/'outside.jpg').write_bytes(b'photo')
    assert resolve_image_path(root/'screen-pdf-project.json',{'path':'../outside.jpg'}) is None
