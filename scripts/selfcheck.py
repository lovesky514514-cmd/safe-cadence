"""Offline integration test, including Chinese paths and a rejected update."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory


def run_selfcheck() -> int:
    import numpy as np
    from safecadence_preflight.ckge_protocol import audit_ckge, save_report
    from safecadence_preflight.frozen_append_guard import FrozenAppendGuard
    from safecadence_preflight.pilot import run_pilot
    from scripts.create_tiny_fixture import make

    with TemporaryDirectory(prefix='safecadence_') as temp:
        folder = Path(temp) / '中文路径_离线检查'
        data = folder / '合成快照'
        make(data)
        audit = audit_ckge(data, snapshot_count=5)
        if len(audit['stage_statistics']) != 5:
            raise RuntimeError('synthetic stage count did not match')
        save_report(audit, folder / '审计.json')
        decoded = json.loads((folder / '审计.json').read_text(encoding='utf-8'))
        if decoded['stage_statistics'][-1]['total_entities'] != 7:
            raise RuntimeError('Unicode roundtrip or entity mapping failed')
        result = run_pilot(data, folder / '试跑.json', max_snapshots=2,
                           steps=3, seed=123, max_eval=4, dim=8)
        if result['status'] != 'SMOKE_ONLY_NOT_CANONICAL_BENCHMARK':
            raise RuntimeError('smoke run was not labeled correctly')

        # Verify one update that preserves Top-k and one that violates it.
        entities = np.array([[0.], [2.], [4.]], dtype=np.float64)
        triple = np.array([[0, 0, 1]], dtype=np.int64)
        guard = FrozenAppendGuard(entities, np.array([[0.]]), triple, triple, k=2)
        if not guard.verify_appended(np.array([[20.]])).certified:
            raise RuntimeError('safe append unexpectedly failed')
        if guard.verify_appended(np.array([[1.5]])).certified:
            raise RuntimeError('unsafe append unexpectedly passed')
    print('自检 PASS：中文路径、快照协议、训练与安全门控。')
    return 0
