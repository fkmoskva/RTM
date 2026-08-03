import numpy as np
import pytest

segyio = pytest.importorskip('segyio')

from src.models import setup_from_segy


def make_segy(path, nx, ny, dx_us, vp_values):
    '''Создаёт синтетический SEG-Y файл с заданными трассами.'''
    spec = segyio.spec()
    spec.sorting  = None
    spec.format   = 1          # IBM float32
    spec.samples  = np.arange(ny, dtype=np.float32)
    spec.tracecount = nx

    with segyio.create(path, spec) as f:
        f.bin[segyio.BinField.Interval] = dx_us
        for i in range(nx):
            f.trace[i] = vp_values[i].astype(np.float32)


def test_setup_from_segy_shape(tmp_path):
    nx, ny = 20, 15
    vp = np.full((nx, ny), 2000., dtype=np.float32)
    sgy = str(tmp_path / 'model.sgy')
    make_segy(sgy, nx, ny, 10000, vp)

    cfg = setup_from_segy(sgy, freq=10., dx=10.)
    assert cfg['Vp'].shape == (nx, ny)
    assert cfg['nx'] == nx
    assert cfg['ny'] == ny


def test_setup_from_segy_dx_explicit(tmp_path):
    nx, ny = 10, 12
    vp = np.full((nx, ny), 1500., dtype=np.float32)
    sgy = str(tmp_path / 'model.sgy')
    make_segy(sgy, nx, ny, 10000, vp)

    cfg = setup_from_segy(sgy, freq=10., dx=5.)
    assert cfg['dx'] == 5.


def test_setup_from_segy_dx_from_header(tmp_path):
    nx, ny = 10, 12
    vp = np.full((nx, ny), 1500., dtype=np.float32)
    sgy = str(tmp_path / 'model.sgy')
    # dto=4000 мкс → dx = 4.0 м
    make_segy(sgy, nx, ny, 4000, vp)

    cfg = setup_from_segy(sgy, freq=10.)
    assert cfg['dx'] == pytest.approx(4.0)


def test_setup_from_segy_downsample(tmp_path):
    nx, ny = 20, 16
    vp = np.random.uniform(1500., 3000., (nx, ny)).astype(np.float32)
    sgy = str(tmp_path / 'model.sgy')
    make_segy(sgy, nx, ny, 5000, vp)

    cfg = setup_from_segy(sgy, freq=10., dx=5., downsample=2)
    assert cfg['Vp'].shape == (nx // 2, ny // 2)
    assert cfg['dx'] == pytest.approx(10.)


def test_setup_from_segy_vp_values(tmp_path):
    nx, ny = 8, 10
    vp = np.arange(nx * ny, dtype=np.float32).reshape(nx, ny) + 1000.
    sgy = str(tmp_path / 'model.sgy')
    make_segy(sgy, nx, ny, 10000, vp)

    cfg = setup_from_segy(sgy, freq=10., dx=10.)
    assert np.allclose(cfg['Vp'], vp.astype(np.float64))


def test_setup_from_segy_file_not_found():
    with pytest.raises(FileNotFoundError):
        setup_from_segy('/nonexistent/path/model.sgy', freq=10., dx=10.)


def test_setup_from_segy_cfg_keys(tmp_path):
    nx, ny = 12, 10
    vp = np.full((nx, ny), 2500., dtype=np.float32)
    sgy = str(tmp_path / 'model.sgy')
    make_segy(sgy, nx, ny, 10000, vp)

    cfg = setup_from_segy(sgy, freq=10., dx=10.)
    for key in ('Vp', 'dx', 'dt', 'nx', 'ny', 'nodes_time', 'freq',
                'j_src', 'rec_y', 'absorb_cells'):
        assert key in cfg, f'Ключ {key!r} отсутствует в cfg'


def test_setup_from_segy_surface_placement(tmp_path):
    nx, ny = 20, 15
    vp = np.full((nx, ny), 2000., dtype=np.float32)
    sgy = str(tmp_path / 'model.sgy')
    make_segy(sgy, nx, ny, 10000, vp)

    cfg = setup_from_segy(sgy, freq=10., dx=10.)
    expected_z = cfg['absorb_cells'] + 2
    assert cfg['j_src'] == expected_z
    assert np.all(cfg['rec_y'] == expected_z)
