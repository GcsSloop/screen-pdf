import copy
import torch
from training.import_screen_project import content_groups, split_groups
from training.train_screen_corner_single import training_order


def row(i, dhash, phash=None):
    r={'page_id':str(i),'image_sha256':f'image{i}', 'screen_dhash':f'{dhash:016x}'}
    if phash is not None:r['screen_phash']=f'{phash:016x}'
    return r


def test_duplicate_groups_are_transitive_and_keep_exact_files_together():
    rows=[row(0,0),row(1,255),row(2,65535),row(3,2**64-1)]
    rows[3]['image_sha256']=rows[0]['image_sha256']
    assert len(content_groups(rows))==1


def test_perceptual_duplicates_stay_together_despite_dhash_difference():
    rows=[row(0,0,10),row(1,2**64-1,10)]
    assert len(content_groups(rows))==1


def test_splitting_is_repeatable_and_never_breaks_a_content_group():
    groups=[[row(i*2,1<<i),row(i*2+1,1<<i)] for i in range(10)]
    first=split_groups(copy.deepcopy(groups))
    second=split_groups(copy.deepcopy(groups))
    assert first==second
    assert all(first.values())
    for group in groups:
        assigned=[name for name,items in first.items() if any(r['page_id']==group[0]['page_id'] for r in items)]
        assert len(assigned)==1
        assert all(r['page_id'] in {v['page_id'] for v in first[assigned[0]]} for r in group)


def test_step_budget_is_equal_for_different_training_sizes():
    for count in [1097,1155]:
        first=training_order(count,torch.Generator().manual_seed(7),37)
        second=training_order(count,torch.Generator().manual_seed(7),37)
        assert first==second
        assert len(first)==37*32
        assert max(first)<count
        assert min(first)>=0
        assert len(set(first))==count


def test_import_requires_explicit_manual_labels_and_canonicalizes_point_order(tmp_path,monkeypatch):
    import json
    import cv2
    import numpy as np
    from training import import_screen_project as importer
    repo=tmp_path/'repo'
    for folder in ['data/raw','data/curated/annotations','data/splits/cross_project','training/registry']:
        (repo/folder).mkdir(parents=True)
    source=tmp_path/'source';source.mkdir()
    rng=np.random.default_rng(7)
    pages=[]
    for i in range(4):
        image=source/f'{i}.jpg'
        cv2.imwrite(str(image),rng.integers(0,256,(60,90,3),dtype=np.uint8))
        candidate={'method':'screen_corner_single','source':'runtime_single','modelId':'test',
                   'quad':[[10,10],[80,10],[80,50],[10,50]]}
        if i<3:candidate['manualQuad']=[[80,10],[80,50],[10,50],[10,10]]
        pages.append({'id':str(i),'name':image.name,'path':str(image),'status':'reviewed',
                      'selectedCandidateIndex':0,'candidates':[candidate],
                      'activeQuad':candidate['quad']})
    project=source/'screen-pdf-project.json';project.write_text(json.dumps({'pages':pages}))
    before=project.read_bytes()
    monkeypatch.setattr(importer,'REPO',repo)
    result=importer.import_project(source,'fixture')
    rows=[r for split in result.values() for r in split]
    assert len(rows)==3
    assert all(r['manual_quad']==[[10.,10.],[80.,10.],[80.,50.],[10.,50.]] for r in rows)
    metadata=json.loads((repo/'training/registry/fixture.json').read_text())
    assert len(metadata['excluded'])==1
    assert project.read_bytes()==before
