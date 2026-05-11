import numpy as np
import pytest

from src.compression import (
    compress_tt, decompress_tt,
    compress_qtt, decompress_qtt,
    storage_bytes,
)
from src.imaging import metrics


def smooth_snapshot(nx=64, ny=48):
    """Гладкий сигнал — хорошо сжимается TT/QTT."""
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 2*np.pi, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.sin(2*X) * np.cos(3*Y) + 0.5*np.sin(5*X + Y)


@pytest.mark.parametrize('eps', [0.1, 0.01])
def test_tt_roundtrip(eps):
    snap = smooth_snapshot()
    rec  = decompress_tt(compress_tt(snap, eps))
    assert metrics(rec, snap)['rel_error'] < eps * 10


@pytest.mark.parametrize('eps', [0.1, 0.01])
def test_qtt_roundtrip(eps):
    snap = smooth_snapshot()
    rec  = decompress_qtt(compress_qtt(snap, eps))
    assert metrics(rec, snap)['rel_error'] < eps * 10


def test_tt_output_shape():
    snap = smooth_snapshot(nx=100, ny=80)
    assert decompress_tt(compress_tt(snap, 0.01)).shape == snap.shape


def test_qtt_output_shape():
    snap = smooth_snapshot(nx=100, ny=80)
    assert decompress_qtt(compress_qtt(snap, 0.01)).shape == snap.shape


def test_storage_bytes_is_positive():
    snap = smooth_snapshot()
    assert storage_bytes(compress_tt(snap, 0.1)) > 0
    assert storage_bytes(compress_qtt(snap, 0.1)) > 0


def test_smaller_eps_means_more_storage():
    snap    = smooth_snapshot()
    s_loose = storage_bytes(compress_tt(snap, 0.5))
    s_tight = storage_bytes(compress_tt(snap, 0.01))
    assert s_tight >= s_loose
