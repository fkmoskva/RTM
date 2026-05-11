import numpy as np
import pytest

from src.models import setup_twolayer
from src.solver import run_forward, run_adjoint
from src.imaging import cross_correlate


def small_cfg():
    return setup_twolayer(nx=80, ny=60, dx=10., freq=30., t_max=0.15)


def test_forward_returns_snapshots():
    cfg = small_cfg()
    snaps, seis = run_forward(cfg, save_every=10, verbose=False)
    assert len(snaps) > 0
    assert seis.shape == (cfg['n_receivers'], cfg['nodes_time'])


def test_forward_seismogram_nonzero():
    cfg = small_cfg()
    _, seis = run_forward(cfg, save_every=10, verbose=False)
    assert np.abs(seis).max() > 0


def test_adjoint_matches_forward_steps():
    cfg = small_cfg()
    snaps, seis = run_forward(cfg, save_every=10, verbose=False)
    adj, _ = run_adjoint(cfg, seis, set(snaps.keys()), verbose=False)
    assert set(adj.keys()) == set(snaps.keys())


def test_rtm_image_correct_shape():
    cfg = small_cfg()
    snaps, seis = run_forward(cfg, save_every=10, verbose=False)
    adj, _      = run_adjoint(cfg, seis, set(snaps.keys()), verbose=False)
    img = cross_correlate(snaps, adj)
    assert img.shape == (cfg['nx'], cfg['ny'])
