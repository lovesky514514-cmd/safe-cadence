"""The integrity checker must reject tampering and untrusted manifest paths."""
import json
from pathlib import Path

from scripts import verify_package


def test_manifest_detects_tampering(tmp_path, monkeypatch):
    data = tmp_path / 'safe.py'
    data.write_text('print(1)\n', encoding='utf-8')
    manifest = tmp_path / 'PACKAGE_SHA256.json'
    info = {'schema': 'safe-cadence-sha256/1.0',
            'files': {'safe.py': verify_package.sha256(data)}}
    manifest.write_text(json.dumps(info), encoding='utf-8')
    monkeypatch.setattr(verify_package, 'ROOT', tmp_path)
    monkeypatch.setattr(verify_package, 'MANIFEST', manifest)
    assert verify_package.verify() == 0
    data.write_text('print(2)\n', encoding='utf-8')
    assert verify_package.verify() != 0


def test_manifest_rejects_parent_directory(tmp_path, monkeypatch):
    manifest = tmp_path / 'PACKAGE_SHA256.json'
    manifest.write_text(json.dumps({
        'schema': 'safe-cadence-sha256/1.0',
        'files': {'../outside': '0' * 64},
    }), encoding='utf-8')
    monkeypatch.setattr(verify_package, 'ROOT', tmp_path)
    monkeypatch.setattr(verify_package, 'MANIFEST', manifest)
    assert verify_package.verify() != 0
