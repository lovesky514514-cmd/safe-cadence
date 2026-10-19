"""Subprocess tests ensure startup subcommands really run in a clean shell."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run via absolute launcher path without depending on the current folder."""
    return subprocess.run(
        [sys.executable, '-X', 'utf8', str(ROOT / 'launcher.py'), *args],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=60,
    )


def test_cli_synthetic_chinese_roundtrip_and_training():
    p = run_cli('selfcheck')
    assert p.returncode == 0, p.stdout + p.stderr
    assert '自检 PASS' in p.stdout
    assert 'SMOKE_ONLY' not in p.stderr


def test_cli_encoding_check():
    p = run_cli('encoding')
    assert p.returncode == 0, p.stdout + p.stderr
    assert 'ENCODING PASS' in p.stdout


def test_batch_captures_exit_code_outside_parenthesized_blocks():
    # cmd.exe expands %ERRORLEVEL% before executing a parenthesized IF block.
    # A previous version used that pattern and could falsely report success.
    bat = (ROOT / 'start.bat').read_text(encoding='ascii')
    assert ':capture' in bat
    assert 'set "EXITCODE=%ERRORLEVEL%"' in bat
    assert 'goto capture' in bat
