import os
import numpy as np

try:
    import segyio
except ImportError:
    segyio = None


'''Строит словарь параметров модели по сетке скоростей Vp.'''
def build_params(Vp, dx, freq, nodes_time):
    nx, ny         = Vp.shape
    vp_max, vp_min = Vp.max(), Vp.min()
    dt             = 0.5 * dx / (vp_max * np.sqrt(2))

    absorb_len   = 6.5 * (vp_min / freq)
    absorb_cells = int(absorb_len / dx)

    grid_x, grid_z = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    dist_to_boundary = np.minimum.reduce([grid_x, nx-1-grid_x, grid_z, ny-1-grid_z])
    damping = np.where(
        dist_to_boundary < absorb_cells,
        (42.5 * vp_min / absorb_len) * ((absorb_cells - dist_to_boundary) / absorb_cells)**2,
        0.
    )

    n_receivers = max(20, min(nx // 10, nx - 2*absorb_cells))
    recv_x = np.linspace(min(absorb_cells, nx//4), max(nx-1-absorb_cells, nx*3//4),
                         n_receivers, dtype=int)
    recv_z = np.full(n_receivers, min(max(absorb_cells+2, 5), ny-2), dtype=int)

    return dict(
        Vp=Vp, dx=dx, dt=dt, nx=nx, ny=ny,
        nodes_time=nodes_time, freq=freq,
        amplitude=1., t0=1.2/freq,
        i_src=nx//2, j_src=min(max(absorb_cells+5, 10), ny-2),
        absorb_cells=absorb_cells,
        C1=1./(1 + damping*dt),
        C2=1. - damping*dt,
        vp2=(Vp*dt/dx)**2,
        src_scale=dt**2 / (dx**2 * 0.01),
        rec_x=recv_x, rec_y=recv_z, n_receivers=n_receivers,
        extent_km=[0, nx*dx/1e3, ny*dx/1e3, 0],
    )


'''Двухслойная модель: vp_top сверху, vp_bot снизу, граница на ny//2.'''
def setup_twolayer(nx=2000, ny=500, dx=10., vp_top=2000., vp_bot=3500.,
                   freq=10., t_max=None):
    Vp = np.full((nx, ny), vp_top)
    Vp[:, ny//2:] = vp_bot

    absorb_cells_est = int(6.5 * vp_top / freq / dx)
    dt_est = 0.5 * dx / (vp_bot * np.sqrt(2))

    if t_max is None:
        # NMO: время до рефлектора и обратно для самого дальнего приёмника
        depth  = max(ny//2 - absorb_cells_est, 1) * dx
        offset = max(nx//2 - absorb_cells_est, 1) * dx
        t_max  = np.sqrt(offset**2 + 4 * depth**2) / vp_top

    nodes_time = max(200, int(t_max / dt_est) + 1)
    cfg = build_params(Vp, dx, freq, nodes_time)
    surface_z = min(cfg['absorb_cells'] + 2, ny - 2)
    cfg['j_src'] = surface_z
    cfg['rec_y'][:] = surface_z
    return cfg


'''Модель Marmousi II (2301×751, dx=4 м). downsample прореживает сетку в N раз.'''
def setup_marmousi(freq=30., t_max=None, downsample=1):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'marmousi', 'marmousi_vp.bin')
    Vp = np.fromfile(path, dtype=np.float32).reshape(2301, 751).astype(np.float64)
    if downsample > 1:
        Vp = Vp[::downsample, ::downsample]
    nx, ny = Vp.shape
    dx     = 4. * downsample
    vp_min = float(Vp.min())

    absorb_cells_est = int(6.5 * vp_min / freq / dx)
    dt_est = 0.5 * dx / (float(Vp.max()) * np.sqrt(2))

    if t_max is None:
        depth  = max(ny - 2 * absorb_cells_est, 1) * dx
        offset = max(nx//2 - absorb_cells_est, 1) * dx
        t_max  = np.sqrt(offset**2 + 4 * depth**2) / vp_min

    nodes_time = max(200, int(t_max / dt_est) + 1)
    cfg = build_params(Vp, dx, freq, nodes_time)
    surface_z = cfg['absorb_cells'] + 2
    cfg['j_src'] = surface_z
    cfg['rec_y'][:] = surface_z
    return cfg


'''Загружает 2D модель Vp из SEG-Y файла; трассы → ось X, отсчёты → ось Z.'''
def setup_from_segy(path, freq=10., t_max=None, downsample=1, dx=None):
    if segyio is None:
        raise ImportError('Пакет segyio не установлен. Выполните: pip install segyio')

    if not os.path.isfile(path):
        raise FileNotFoundError(f'SEG-Y файл не найден: {path}')

    try:
        with segyio.open(path, ignore_geometry=True) as f:
            Vp = np.stack([tr.copy() for tr in f.trace], axis=0).astype(np.float64)
            if dx is None:
                try:
                    interval = f.bin[segyio.BinField.Interval]
                    if interval and interval > 0:
                        dx = float(interval) / 1000.0
                    else:
                        raise ValueError('нулевой интервал')
                except Exception:
                    print('  Предупреждение: не удалось прочитать шаг сетки из бинарного заголовка SEG-Y. '
                          'Используется dx=10.0 м по умолчанию.')
                    dx = 10.0
    except segyio.exceptions.InvalidFile as exc:
        raise ValueError(f'Не удалось прочитать SEG-Y файл ({path}): {exc}') from exc

    if downsample > 1:
        Vp = Vp[::downsample, ::downsample]
        dx *= downsample

    nx, ny = Vp.shape
    vp_min = float(Vp.min())

    absorb_cells_est = int(6.5 * vp_min / freq / dx)
    dt_est = 0.5 * dx / (float(Vp.max()) * np.sqrt(2))

    if t_max is None:
        depth  = max(ny - 2 * absorb_cells_est, 1) * dx
        offset = max(nx//2 - absorb_cells_est, 1) * dx
        t_max  = np.sqrt(offset**2 + 4 * depth**2) / vp_min

    nodes_time = max(200, int(t_max / dt_est) + 1)
    cfg = build_params(Vp, dx, freq, nodes_time)
    surface_z = cfg['absorb_cells'] + 2
    cfg['j_src'] = surface_z
    cfg['rec_y'][:] = surface_z
    return cfg
