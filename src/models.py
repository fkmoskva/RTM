import os
import numpy as np


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
