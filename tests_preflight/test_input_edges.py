"""Regression tests for invalid input that must fail, not silently pass."""
import pytest

from safecadence_preflight.ckge_protocol import load_ckge
from safecadence_preflight.download_ckge import download


@pytest.mark.parametrize('count', [0, -1, 6])
def test_snapshot_loader_rejects_invalid_count(tmp_path, count):
    with pytest.raises(ValueError, match='between 1 and 5'):
        load_ckge(tmp_path, snapshot_count=count)


@pytest.mark.parametrize('count', [0, -1, 6])
def test_downloader_rejects_invalid_count_before_network(tmp_path, count):
    with pytest.raises(ValueError, match='between 1 and 5'):
        download('WN_CKGE', tmp_path, count=count)
