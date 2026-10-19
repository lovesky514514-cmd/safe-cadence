"""Check the external operator contract, not implementation internals."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quick_doctor_creates_machine_readable_report(tmp_path):
    report = tmp_path / '中文诊断.json'
    p = subprocess.run(
        [sys.executable, str(ROOT / 'launcher.py'), 'doctor', '--quick',
         '--report', str(report)], cwd=ROOT, capture_output=True,
        text=True, encoding='utf-8', timeout=50,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    info = json.loads(report.read_text(encoding='utf-8'))
    assert info['ok']
    assert {x['code'] for x in info['checks']} >= {'PYTHON', 'DEPENDENCIES', 'ENCODING', 'INTEGRITY'}
    assert next(x for x in info['checks'] if x['code'] == 'DATA')['status'] == 'SKIP'


def test_doctor_reports_bad_data_as_failure(tmp_path):
    report = tmp_path / 'bad.json'
    p = subprocess.run(
        [sys.executable, str(ROOT / 'launcher.py'), 'doctor', '--quick',
         '--data', str(tmp_path / 'missing'), '--report', str(report)],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=50,
    )
    assert p.returncode != 0
    info = json.loads(report.read_text(encoding='utf-8'))
    assert not info['ok']
    assert next(x for x in info['checks'] if x['code'] == 'DATA')['status'] == 'FAIL'
