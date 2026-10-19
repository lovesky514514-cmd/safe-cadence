"""Check source files for UTF-8, and Windows launcher for ASCII/CRLF."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {'.venv', '.git', '.pytest_cache', '__pycache__',
                'data', 'outputs', 'logs'}
TEXT_SUFFIXES = {'.md', '.txt', '.json', '.py', '.sh', '.toml', '.yml', '.yaml'}


def check() -> int:
    problems = []
    inspected = 0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        inspected += 1
        try:
            content = path.read_text(encoding='utf-8-sig')
            if '\ufffd' in content:
                problems.append(f'{path.relative_to(ROOT)}: Unicode replacement character')
        except UnicodeError as exc:
            problems.append(f'{path.relative_to(ROOT)}: {exc}')
    bat = (ROOT / 'start.bat').read_bytes()
    if any(byte >= 128 for byte in bat) or b'\r\n' not in bat:
        problems.append('start.bat must use ASCII and CRLF line endings')
    if problems:
        print('ENCODING CHECK FAIL:\n' + '\n'.join(problems))
        return 1
    print(f'ENCODING PASS: {inspected} UTF-8 files; BAT ASCII/CRLF.')
    return 0


if __name__ == '__main__':
    raise SystemExit(check())
