import copy
import hashlib
import json
from pathlib import Path

import pytest

from training.small_scene_round import audit_splits


def fixture(tmp_path):
    result={}
    for i,name in enumerate(['train','validation','holdout']):
        image=tmp_path/f'{name}.jpg'
        image.write_bytes(f'unique photo {i}'.encode())
        result[name]=[{'image_path':str(image),'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                       'capture_group':f'group-{i}'}]
    return result


@pytest.mark.parametrize('key',['image_path','image_sha256','capture_group'])
def test_scene_split_rejects_leakage_for_each_identity(tmp_path,key):
    splits=fixture(tmp_path)
    splits['validation'][0][key]=splits['train'][0][key]
    with pytest.raises(ValueError,match='New scene overlap'):
        audit_splits(splits,[])


def test_audit_detects_renamed_duplicate_of_any_old_split(tmp_path):
    splits=fixture(tmp_path)
    old=tmp_path/'copied.jpg'
    old.write_bytes((tmp_path/'holdout.jpg').read_bytes())
    with pytest.raises(ValueError,match='duplicates previous image bytes'):
        audit_splits(splits,[{'image_path':str(old)}])


def test_audit_accepts_disjoint_rows_without_mutating_them(tmp_path):
    splits=fixture(tmp_path)
    before=copy.deepcopy(splits)
    assert audit_splits(splits,[])=={'path_overlap':0,'exact_image_overlap':0,'new_group_overlap':0}
    assert splits==before


def prepare_scene(tmp_path,monkeypatch,**options):
    from training import small_scene_round as experiment
    from training.screen_corner_ablation import write_rows,write_json
    repo=tmp_path/'repo'
    for folder in ['data/splits/cross_project','data/curated/annotations','training/registry']:
        (repo/folder).mkdir(parents=True)
    base=repo/'previous';(base/'checkpoints').mkdir(parents=True)
    (base/'checkpoints/screen_corner_single.pt').write_bytes(b'initial checkpoint')
    data=base/'data/with_xr'
    (tmp_path/'old').mkdir();(tmp_path/'new').mkdir()
    old=fixture(tmp_path/'old');new=fixture(tmp_path/'new')
    # Distinct contents as well as paths across old/new projects.
    for name,rows in new.items():
        path=Path(rows[0]['image_path']);path.write_bytes(('new '+name).encode())
        rows[0]['image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    for splits in [old,new]:
        for name,rows in splits.items():
            for row in rows:
                row.update(manual_quad=[[1,1],[90,1],[90,50],[1,50]],split=name)
    for name,rows in old.items():
        write_rows(data/f'{name}.jsonl',rows*4 if name=='train' else rows)
        write_rows(data/f'xr_{name}.jsonl',rows)
    annotation=repo/'data/curated/annotations/new_pages_v1.jsonl'
    write_json(repo/'data/splits/cross_project/new_v1.json',dict(new,metadata={
        'annotation_file':str(annotation),'page_count':3,'split_counts':{k:1 for k in new},
        'content_groups':3,'source_project_sha256':'fixture'}))
    contact=tmp_path/'contact.jpg';contact.write_bytes(b'contact')
    monkeypatch.setattr(experiment,'REPO',repo)
    run=repo/'round'
    experiment.prepare(run,slug='new',base=base,base_data=data,prefix='immersive',replay=8,contact_sheet=contact,**options)
    return experiment,run,base


def test_prepare_keeps_previous_replay_and_uses_requested_initial_model(tmp_path,monkeypatch):
    from training.screen_corner_ablation import read_rows
    _,run,base=prepare_scene(tmp_path,monkeypatch)
    settings=json.loads((run/'manifest.json').read_text())
    assert settings['initial_model']==str(base/'checkpoints/screen_corner_single.pt')
    assert settings['train_entries']=={'control':4,'with_immersive':12}
    assert settings['unique_new_train_pages']==1
    assert settings['conditions']==['control','with_immersive']
    assert settings['validation_pages']==2
    assert read_rows(run/'data/control/xr_train.jsonl')==read_rows(run/'data/with_immersive/xr_train.jsonl')
    assert len(read_rows(run/'data/with_immersive/immersive_train.jsonl'))==1


def test_new_physical_scene_does_not_inherit_blue_side_panel_tags(tmp_path,monkeypatch):
    from training.screen_corner_ablation import read_rows
    tags=['black_background','low_contrast_screen_border','dark_physical_frame']
    _,run,_=prepare_scene(tmp_path,monkeypatch,scene_tags=tags,scene_description='dark display against dark frame')
    settings=json.loads((run/'manifest.json').read_text())
    assert settings['dataset']['scene_tags']==tags
    assert settings['dataset']['scene_description']=='dark display against dark frame'
    assert 'blue_side_panels' not in settings['dataset']['scene_tags']
    rows=read_rows(run/'data/with_immersive/immersive_train.jsonl')
    assert all(row['scene_tags']==tags for row in rows)


def test_generalized_finish_keeps_initial_on_dev_tie_despite_better_regression(tmp_path,monkeypatch):
    from dataset_benchmark import summarize_point_hits
    experiment,run,base=prepare_scene(tmp_path,monkeypatch)
    for label in ['control','with_immersive']:
        folder=run/f'cells/{label}/checkpoints';folder.mkdir(parents=True)
        (folder/'screen_corner_single.pt').write_bytes(label.encode())
    def evaluate(path,rows):
        is_new='with_immersive' in path.parts
        records=[]
        for row in rows:
            predicted=row['manual_quad'] if row['split']=='validation' or is_new else [[0,0]]*4
            records.append(dict(image_path=row['image_path'],manual_quad=row['manual_quad'],
                                final_quad=predicted,coarse_quad=predicted))
        metric=summarize_point_hits([r['manual_quad'] for r in rows],[r['final_quad'] for r in records])
        return {'final':metric,'coarse':metric},records
    monkeypatch.setattr(experiment,'evaluate',evaluate)
    experiment.finish(run)
    result=json.loads((run/'reports/evaluation.json').read_text())
    assert result['selection']['selected']=='initial'
    assert (run/'checkpoints/screen_corner_single.pt').read_bytes()==(base/'checkpoints/screen_corner_single.pt').read_bytes()
    assert result['results']['initial']['immersive_holdout']['final']['hit_corners']==0
    assert result['results']['with_immersive']['immersive_holdout']['final']['hit_corners']==4
    assert set(result['results']['initial']) >= {'xr_validation','immersive_validation','xr_holdout','immersive_holdout'}
