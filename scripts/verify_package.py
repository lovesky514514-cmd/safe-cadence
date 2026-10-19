"""Verify every packaged file against SHA-256 manifest (stdlib, offline).

Run from repository root: `python scripts/verify_package.py`.
Do not put credentials or server secrets inside the manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'PACKAGE_SHA256.json'


def sha256(path: Path) -> str:
    """Hash a file without loading the whole file into memory."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify() -> int:
    """Check file digests, missing files, path traversal and manifest schema."""
    if not MANIFEST.is_file():
        print('ERROR: PACKAGE_SHA256.json missing; use a complete release ZIP.')
        return 2
    try:
        info = json.loads(MANIFEST.read_text(encoding='utf-8'))
    except (ValueError, UnicodeError) as exc:
        print(f'ERROR: invalid manifest: {exc}')
        return 2
    if not isinstance(info, dict) or info.get('schema') != 'safe-cadence-sha256/1.0' or not isinstance(info.get('files'), dict):
        print('ERROR: unsupported package manifest')
        return 2
    failures = []
    for raw, expected in info['files'].items():
        relative = Path(raw)
        # Reject zip-slip/manifest injection and absolute paths.
        if relative.is_absolute() or '..' in relative.parts:
            failures.append(f'unsafe path: {raw}')
            continue
        file = (ROOT / relative).resolve()
        if not file.is_relative_to(ROOT.resolve()) or not file.is_file():
            failures.append(f'missing/escaped: {raw}')
            continue
        if sha256(file) != expected:
            failures.append(f'checksum mismatch: {raw}')
    if failures:
        print('PACKAGE VERIFY FAIL:')
        for err in failures[:30]:
            print('  ' + err)
        return 1
    print(f'PACKAGE VERIFY PASS: {len(info["files"])} files have expected SHA-256.')
    return 0


if __name__ == '__main__':
    sys.exit(verify())
