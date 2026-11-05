import copy
from collections import Counter

import pytest

from training.build_black_scene_dataset import assign_groups,deduplicate
from training.unified_black_round import compose_training


def row(name,role=None,new=True):
    return dict(image_sha256=name,image_path='/photos/'+name+'.jpg',page_id=name,
                historical_split=role,new_source=new,split=role,
                source_dataset_slug='source',source_page_id=name,project_path='/source/project.json',
                manual_quad=[[1,1],[100,1],[100,60],[1,60]])


def test_fresh_near_duplicate_inherits_old_validation_exposure():
    old=row('old','validation',False);fresh=row('fresh')
    result=assign_groups([[old,fresh],[row('trained','train',False)]])
    assert {r['page_id'] for r in result['validation']}=={'old','fresh'}
    assert [r['page_id'] for r in result['train']]==['trained']
    assert old['capture_group']==fresh['capture_group']


def test_near_duplicate_conflicting_old_roles_cannot_be_reshuffled():
    with pytest.raises(ValueError,match='historical split'):
        assign_groups([[row('trained','train',False),row('tested','holdout',False)]])


def test_five_fresh_groups_are_deterministic_and_do_not_change_old_roles():
    groups=[[row(str(i))] for i in range(5)]+[[row('old','holdout',False)]]
    first=assign_groups(copy.deepcopy(groups));second=assign_groups(copy.deepcopy(groups))
    assert first==second
    assert {k:sum(r['new_source'] for r in v) for k,v in first.items()}=={'train':3,'validation':1,'holdout':1}
    assert next(r for r in first['holdout'] if r['page_id']=='old')['split']=='holdout'


@pytest.mark.parametrize('reverse',[False,True])
def test_exact_copies_keep_provenance_and_old_role_regardless_of_order(reverse):
    old=row('same','holdout',False);fresh=row('same')
    values=[old,fresh]
    if reverse:values.reverse()
    result=deduplicate(values)
    assert len(result)==1
    assert len(result[0]['provenance'])==2
    assert result[0]['historical_split']=='holdout'
    assert not result[0]['new_source']


def test_conflicting_duplicate_labels_are_not_silently_replaced():
    old=row('same','train',False);new=row('same')
    new['manual_quad'][0]=[20,20]
    with pytest.raises(ValueError,match='Conflicting human labels'):
        deduplicate([old,new])


def test_conflicting_duplicate_roles_are_not_silently_replaced():
    with pytest.raises(ValueError,match='conflicting historical splits'):
        deduplicate([row('same','train',False),row('same','validation',False)])


def test_scene_sampling_replaces_separate_replay_and_preserves_ordinary_weights():
    old_a=row('a','train',False);old_b=row('b','train',False);new=row('fresh','train',True)
    ordinary=row('ordinary','train',False)
    previous=[old_a]*4+[old_b]*8+[ordinary]*3
    control=compose_training(previous,[old_a,old_b,new],6,False)
    full=compose_training(previous,[old_a,old_b,new],6,True)
    assert Counter(r['page_id'] for r in control)=={'ordinary':3,'a':6,'b':6}
    assert Counter(r['page_id'] for r in full)=={'ordinary':3,'a':6,'b':6,'fresh':6}
