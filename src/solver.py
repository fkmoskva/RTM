import numpy as np
import time
from numba import njit, prange


'''Один шаг волнового уравнения, 4-й порядок по пространству (Numba parallel).'''
@njit(parallel=True, fastmath=True, cache=True)
def calc_step(pres, pres_prev, pres_next, C1, C2, vp2, nx, ny):
    for i in prange(2, nx - 2):
        for j in range(2, ny - 2):
            laplacian = (  16.*(pres[i+1,j] + pres[i-1,j] + pres[i,j+1] + pres[i,j-1])
                         -      pres[i+2,j] - pres[i-2,j] - pres[i,j+2] - pres[i,j-2]
                         - 60. * pres[i,j] ) / 12.
            pres_next[i,j] = C1[i,j] * (
                2.*pres[i,j] - C2[i,j]*pres_prev[i,j] + vp2[i,j]*laplacian
            )
    return pres_next


'''Форвард-моделирование: сохраняет снэпшоты каждые save_every шагов и сейсмограмму.'''
def run_forward(cfg, save_every=20, verbose=True):
    nx, ny, nt   = cfg['nx'], cfg['ny'], cfg['nodes_time']
    pres         = np.zeros((nx, ny))
    pres_prev    = np.zeros((nx, ny))
    pres_next    = np.zeros((nx, ny))
    seismogram   = np.zeros((cfg['n_receivers'], nt))
    forward_snaps = {}

    src_x, src_z    = cfg['i_src'], cfg['j_src']
    C1_at_src       = cfg['C1'][src_x, src_z] * cfg['src_scale']
    recv_x, recv_z  = cfg['rec_x'], cfg['rec_y']
    C1, C2, vp2    = cfg['C1'], cfg['C2'], cfg['vp2']
    freq, t0_wavelet, amplitude = cfg['freq'], cfg['t0'], cfg['amplitude']
    dt             = cfg['dt']
    pi_f           = np.pi * freq
    t_start        = time.time()

    for step in range(1, nt):
        pres_next = calc_step(pres, pres_prev, pres_next, C1, C2, vp2, nx, ny)

        t       = step * dt
        arg     = (pi_f * (t - t0_wavelet))**2
        ricker  = amplitude * (1 - 2*arg) * np.exp(-arg)
        pres_next[src_x, src_z] += C1_at_src * ricker

        pres_prev, pres, pres_next = pres, pres_next, pres_prev

        seismogram[:, step] = pres[recv_x, recv_z]

        if step % save_every == 0:
            forward_snaps[step] = pres.copy()
        if verbose and step % 500 == 0:
            print(f"  fwd {step}/{nt-1}  ({time.time()-t_start:.1f}s)")

    return forward_snaps, seismogram


'''Обратное распространение адджоинта, сохраняет снэпшоты на шагах saved_steps.'''
def run_adjoint(cfg, seismogram, saved_steps, verbose=True):
    nx, ny, nt  = cfg['nx'], cfg['ny'], cfg['nodes_time']
    adj         = np.zeros((nx, ny))
    adj_prev    = np.zeros((nx, ny))
    adj_next    = np.zeros((nx, ny))
    adj_snaps   = {}
    adj_vis     = []
    vis_steps   = {nt//4, nt//2, 3*nt//4}

    recv_x, recv_z = cfg['rec_x'], cfg['rec_y']
    C1_at_recv     = cfg['C1'][recv_x, recv_z]
    C1, C2, vp2   = cfg['C1'], cfg['C2'], cfg['vp2']
    t_start        = time.time()

    for step in range(nt-1, 0, -1):
        adj_next = calc_step(adj, adj_prev, adj_next, C1, C2, vp2, nx, ny)
        adj_next[recv_x, recv_z] += C1_at_recv * seismogram[:, step]
        adj_prev, adj, adj_next = adj, adj_next, adj_prev
        if step in saved_steps:
            adj_snaps[step] = adj.copy()
        if step in vis_steps:
            adj_vis.append((step, adj.copy()))
        if verbose and step % 500 == 0:
            print(f"  adj {step}/{nt-1}  ({time.time()-t_start:.1f}s)")

    return adj_snaps, adj_vis


'''Обратное распространение + кросс-корреляция on-the-fly. Возвращает (image, illumination).'''
def run_adjoint_imaging(cfg, seismogram, forward_snaps, verbose=True):
    nx, ny, nt  = cfg['nx'], cfg['ny'], cfg['nodes_time']
    adj         = np.zeros((nx, ny))
    adj_prev    = np.zeros((nx, ny))
    adj_next    = np.zeros((nx, ny))
    image       = np.zeros((nx, ny))
    illumination = np.zeros((nx, ny))

    recv_x, recv_z = cfg['rec_x'], cfg['rec_y']
    C1_at_recv     = cfg['C1'][recv_x, recv_z]
    C1, C2, vp2   = cfg['C1'], cfg['C2'], cfg['vp2']
    t_start        = time.time()

    for step in range(nt-1, 0, -1):
        adj_next = calc_step(adj, adj_prev, adj_next, C1, C2, vp2, nx, ny)
        adj_next[recv_x, recv_z] += C1_at_recv * seismogram[:, step]
        adj_prev, adj, adj_next = adj, adj_next, adj_prev

        if step in forward_snaps:
            snap_fwd      = forward_snaps[step]
            image        += snap_fwd * adj
            illumination += snap_fwd * snap_fwd

        if verbose and step % 500 == 0:
            print(f"  adj {step}/{nt-1}  ({time.time()-t_start:.1f}s)")

    return image, illumination
