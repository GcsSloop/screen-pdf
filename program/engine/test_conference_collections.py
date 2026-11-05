import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.build_conference_collections import build, classify
from eval.benchmark_conference_collections import predict_one, summarize
from training.materialize_conference_projects import materialize
from training.build_conflict_review import prepare


def test_day_sessions_follow_explicit_conference_mapping(tmp_path):
    config={'collections':[
        {'id':'day19','directory':'Lighting','project_prefix':'2026-03-19'},
        {'id':'day20','directory':'Lighting','project_prefix':'2026-03-20'}]}
    assert classify(tmp_path/'Lighting/2026-03-19 talk/nested/screen-pdf-project.json',tmp_path,config)['id']=='day19'
    with pytest.raises(ValueError,match='exactly one'):
        classify(tmp_path/'Lighting/unknown/screen-pdf-project.json',tmp_path,config)


def test_conflicting_duplicate_human_labels_are_quarantined_without_editing_source(tmp_path):
    source=tmp_path/'sources';project=source/'Event';project.mkdir(parents=True)
    im=np.zeros((80,100,3),np.uint8)
    cv2.imwrite(str(project/'a.png'),im);(project/'b.png').write_bytes((project/'a.png').read_bytes())
    q=[[1,1],[90,1],[90,70],[1,70]];other=[[10,1],[90,1],[90,70],[10,70]]
    payload={'pages':[{'id':'a','path':'a.png','manualQuad':q},
                      {'id':'b','path':'b.png','manualQuad':other},
                      {'id':'auto','path':'a.png','activeQuad':q,'status':'reviewed'}]}
    p=project/'screen-pdf-project.json';p.write_text(json.dumps(payload));before=p.read_bytes()
    config=tmp_path/'config.json';config.write_text(json.dumps({'source_root':str(source),'collections':[{'id':'event','directory':'Event','name':'Event'}]}))
    out=tmp_path/'collections';result=build(out,config_path=config)
    assert result['usable_manual_pages']==0
    assert result['conflicting_image_groups']==1
    assert result['exclusion_reasons']=={'missing_explicit_manual_quad':1,'duplicate_label_or_role_conflict':2}
    assert p.read_bytes()==before
    materialize(out)
    combined=out/'desktop-projects/Event/screen-pdf-project.json'
    data=json.loads(combined.read_text())
    assert len(data['pages'])==1 and data['pages'][0]['id'].endswith('_auto')
    assert data['pages'][0]['activeQuad']==q
    photo=combined.parent/data['pages'][0]['path']
    assert photo.stat().st_ino!=(project/'a.png').stat().st_ino
    photo.write_bytes(b'edited independent photo')
    assert cv2.imread(str(project/'a.png')) is not None and p.read_bytes()==before
    prepare(out)
    review=json.loads((out/'conflict-review/review.json').read_text())
    assert review['conflicting_photos']==1 and review['conflicting_versions']==2
    assert p.read_bytes()==before
    with pytest.raises(FileExistsError):prepare(out)
    with pytest.raises(FileExistsError):build(out,config_path=config)


def test_invalid_duplicate_label_still_participates_in_conflict_audit(tmp_path):
    source=tmp_path/'sources';project=source/'Event';project.mkdir(parents=True)
    cv2.imwrite(str(project/'a.png'),np.zeros((80,100,3),np.uint8))
    q=[[1,1],[90,1],[90,70],[1,70]]
    payload={'pages':[{'id':'valid','path':'a.png','manualQuad':q},
                      {'id':'invalid','path':'a.png','manualQuad':[[1,1],[90,1],[90,100],[1,70]]}]}
    (project/'screen-pdf-project.json').write_text(json.dumps(payload))
    config=tmp_path/'config.json';config.write_text(json.dumps({'source_root':str(source),'collections':[{'id':'event','directory':'Event','name':'Event'}]}))
    result=build(tmp_path/'collection',config_path=config)
    assert result['conflicting_image_groups']==1 and result['usable_manual_pages']==0


def test_benchmark_failure_remains_four_misses_in_denominator(tmp_path):
    im=tmp_path/'image.png';cv2.imwrite(str(im),np.zeros((80,100,3),np.uint8))
    row=dict(image_path=str(im),manual_quad=[[1,1],[90,1],[90,70],[1,70]],collection_id='event',project_id='p',current_role='validation')
    class Broken:
        def predict(self,pixels):raise RuntimeError('inference failed')
    result=predict_one(Broken(),row)
    assert result['final_quad'] is None and result['corner_hits']==[False]*4
    assert summarize([row],{str(im):result})=={'pages':1,'corners':4,'hit_corners':0,'point_le_0_01_ratio':0.0}
