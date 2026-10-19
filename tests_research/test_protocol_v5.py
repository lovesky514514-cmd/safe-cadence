from pathlib import Path
import pytest
from benchmarks.ckge_snapshot_protocol import inspect

def test_cumulative_snapshot_requires_no_leakage(tmp_path):
    rows=[['a r b'],['c r d'],['e r f'],['g r h'],['i r j']]
    for i in range(5):
        p=tmp_path/str(i);p.mkdir()
        (p/'train.txt').write_text('\n'.join(x for block in rows[:i+1] for x in block))
        (p/'valid.txt').write_text('other r tail')
        (p/'test.txt').write_text('test r tail')
    assert len(inspect(tmp_path))==5
    (tmp_path/'2'/'train.txt').write_text('e r f')
    with pytest.raises(ValueError,match='non-cumulative'):inspect(tmp_path)
