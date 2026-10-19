"""Strict loading/audit for externally supplied continual KGE snapshot exports.

Source data must be placed locally by the researcher. No network in this runner.
Expected layout: root/0/{train,valid,test}.txt ... root/4/*.txt;
each text line is a (head,relation,tail) string triple, WN18RR-style.
Each snapshot train must be cumulative (may repeat previous triples).
The observed incoming facts are strictly train(t) - train(t-1).
DO NOT use test/validation facts to optimize or construct negative samples.

This is a loader, NOT a reproduction of any canonical benchmark until the
specific producer's snapshot and filtering conventions have been validated.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np


def read_rows(path):
    result=[]
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        parts=line.split()
        if len(parts)!=3:raise ValueError(f'{path}:{i}: need three columns h r t')
        result.append(tuple(parts))
    if len(set(result))!=len(result):raise ValueError(f'duplicate triples in {path}')
    return result


def inspect(root):
    root=Path(root)
    items=[];prev=set()
    for i in range(5):
        folder=root/str(i)
        tr=set(read_rows(folder/'train.txt'))
        vl=set(read_rows(folder/'valid.txt'))
        te=set(read_rows(folder/'test.txt'))
        if not tr:raise ValueError('empty snapshot training set')
        if not prev.issubset(tr):raise ValueError(f'snapshot {i}: non-cumulative train; must convert explicitly')
        if tr&vl or tr&te or vl&te:raise ValueError(f'snapshot {i}: overlapping splits')
        items.append({'snapshot':i,'cumulative_train':len(tr),'new_train':len(tr-prev),
                      'valid':len(vl),'test':len(te)})
        prev=tr
    return items


def test_protocol(root):
    return inspect(root)

if __name__=='__main__':
    import sys,json
    print(json.dumps(inspect(sys.argv[1]),indent=2))
