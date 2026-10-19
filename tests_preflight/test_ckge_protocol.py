from pathlib import Path
import pytest
from safecadence_preflight.ckge_protocol import load_ckge,audit_ckge


def make_fixture(root, count=5):
    relation='related'
    names=['e0','e1','e2']
    for s in range(count):
        f=root/str(s);f.mkdir(parents=True)
        if s:names.append('new'+str(s))
        (f/'entity2id.txt').write_text('\n'.join(f'{n}\t{j}' for j,n in enumerate(names))+'\n',encoding='utf-8')
        (f/'relation2id.txt').write_text('related\t0\n',encoding='utf-8')
        train=[(0,0,1)] if s==0 else [(len(names)-1,0,0)]
        valid=[(1,0,2)]
        test=[(2,0,1)]
        enc=lambda row:f'{names[row[0]]} related {names[row[2]]}'
        for fn,lines in [('train_id.txt',[f'{h}\t{r}\t{t}' for h,r,t in train]),
                         ('train.txt',[enc(t) for t in train]),
                         ('valid.txt',[enc(t) for t in valid]),
                         ('test.txt',[enc(t) for t in test])]:
            (f/fn).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def test_load_incremental_and_cumulative(tmp_path):
    make_fixture(tmp_path)
    s=load_ckge(tmp_path)
    assert [len(t.incoming) for t in s]==[1]*5
    assert [len(t.cumulative) for t in s]==[1,2,3,4,5]
    assert [t.n_entities for t in s]==[3,4,5,6,7]
    report=audit_ckge(tmp_path)
    assert len(report['file_sha256'])==30
    assert report['stage_statistics'][4]['cumulative_train']==5


def test_entity_remapping_detected(tmp_path):
    make_fixture(tmp_path)
    (tmp_path/'2'/'entity2id.txt').write_text('e0\t1\ne1\t0\ne2\t2\nnew1\t3\nnew2\t4\n',encoding='utf-8')
    with pytest.raises(ValueError,match='append-only'):load_ckge(tmp_path)


def test_numeric_named_mismatch_detected(tmp_path):
    make_fixture(tmp_path)
    (tmp_path/'1'/'train.txt').write_text('e0 related e2\n',encoding='utf-8')
    with pytest.raises(ValueError,match='disagree'):load_ckge(tmp_path)


def test_within_stage_overlap_detected(tmp_path):
    make_fixture(tmp_path)
    (tmp_path/'0'/'test.txt').write_text('e0 related e1\n',encoding='utf-8')
    with pytest.raises(ValueError,match='overlapping'):load_ckge(tmp_path)


def test_historical_train_overlap_detected(tmp_path):
    make_fixture(tmp_path)
    (tmp_path/'1'/'train_id.txt').write_text('0\t0\t1\n',encoding='utf-8')
    (tmp_path/'1'/'train.txt').write_text('e0 related e1\n',encoding='utf-8')
    with pytest.raises(ValueError,match='historical train'):load_ckge(tmp_path)


def test_legacy_cumulative_loader_is_not_used_for_fastkge():
    from benchmarks.ckge_snapshot_protocol import inspect as original
    # Intentionally distinct; new loader supports official per-stage incoming data.
    assert original is not load_ckge
