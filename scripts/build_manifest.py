"""Regenerate the release manifest after intentional source changes."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_package import MANIFEST, ROOT, sha256

EXCLUDE_DIRS = {'.github', '.git', '.venv', '.pytest_cache', '__pycache__',
                'data', 'outputs', 'logs'}


def build() -> int:
    files = {}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS for part in relative.parts):
            continue
        files[relative.as_posix()] = sha256(path)
    MANIFEST.write_text(json.dumps({
        'schema': 'safe-cadence-sha256/1.0', 'files': files,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Updated SHA-256 manifest for {len(files)} source files.')
    return 0


if __name__ == '__main__':
    raise SystemExit(build())
