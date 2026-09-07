import numpy as np
import time
from numba import njit, prange

from .models import leapfrog_coeffs


'''Один шаг волнового уравнения, 4-й порядок по пространству (Numba parallel).
coef_n заменяет константу "2" равномерной схемы: для укороченного шага у узла
квадратуры (шаг h1, перед которым был шаг h0 != h1) coef_n = 1 + h1/h0, см.
src.models.leapfrog_coeffs.'''
@njit(parallel=True, fastmath=True, cache=True)
def calc_step(pres, pres_prev, pres_next, C1, C2, vp2, coef_n, nx, ny):
    for i in prange(2, nx - 2):
        for j in range(2, ny - 2):
            laplacian = (  16.*(pres[i+1,j] + pres[i-1,j] + pres[i,j+1] + pres[i,j-1])
                         -      pres[i+2,j] - pres[i-2,j] - pres[i,j+2] - pres[i,j-2]
                         - 60. * pres[i,j] ) / 12.
            pres_next[i,j] = C1[i,j] * (
                coef_n*pres[i,j] - C2[i,j]*pres_prev[i,j] + vp2[i,j]*laplacian
            )
    return pres_next


'''Коэффициенты для шага с индексом i (1-based) при обходе step_sizes в
порядке order ("forward" — от начала к концу, "adjoint" — в обратном
порядке, см. src/time_sampling.py:build_step_schedule). Использует
предвычисленные cfg['C1']/C2/vp2 (coef_n=2), если оба соседних шага —
полные dt; иначе пересчитывает коэффициенты для этой конкретной пары
шагов через leapfrog_coeffs (h0 != h1 только у двух шагов на каждый узел
квадратуры, поэтому пересчёт — редкий случай).'''
def _step_coeffs(cfg, step_sizes, is_nominal, i, order):
    K = len(step_sizes)
    if order == 'forward':
        h1 = step_sizes[i - 1]
        h0 = step_sizes[i - 2] if i > 1 else step_sizes[0]
        nominal = is_nominal[i - 1] and (i == 1 or is_nominal[i - 2])
    else:
        h1 = step_sizes[i - 1]
        h0 = step_sizes[i] if i < K else step_sizes[i - 1]
        nominal = is_nominal[i - 1] and (i == K or is_nominal[i])

    if nominal:
        return 2., cfg['C1'], cfg['C2'], cfg['vp2']
    return leapfrog_coeffs(cfg['damping'], cfg['Vp'], cfg['dx'], h0, h1)


'''Форвард-моделирование: сохраняет снэпшоты на шагах из weight_at и сейсмограмму.
step_sizes/is_nominal — сетка шагов по времени (src.time_sampling.build_step_schedule
для составной квадратуры Гаусса-Лежандра, uniform_step_schedule для равномерной);
weight_at{i: вес} задаёт и то, какие шаги сохранять, и вес снимка для imaging condition.
snapshot_store — опциональный dict-подобный приёмник снэпшотов (по умолч. обычный
dict в памяти); позволяет подставить диск-бэкенд для плотных наборов, не влезающих в ОЗУ.'''
def run_forward(cfg, step_sizes, weight_at=None, is_nominal=None, verbose=True, snapshot_store=None):
    nx, ny        = cfg['nx'], cfg['ny']
    K             = len(step_sizes)
    pres          = np.zeros((nx, ny))
    pres_prev     = np.zeros((nx, ny))
    pres_next     = np.zeros((nx, ny))
    seismogram    = np.zeros((cfg['n_receivers'], K + 1))
    forward_snaps = {} if snapshot_store is None else snapshot_store

    weight_at  = weight_at or {}
    is_nominal = is_nominal if is_nominal is not None else [s == cfg['dt'] for s in step_sizes]

    src_x, src_z    = cfg['i_src'], cfg['j_src']
    C1_at_src       = cfg['C1'][src_x, src_z] * cfg['src_scale']
    recv_x, recv_z  = cfg['rec_x'], cfg['rec_y']
    freq, t0_wavelet, amplitude = cfg['freq'], cfg['t0'], cfg['amplitude']
    pi_f           = np.pi * freq
    t_start        = time.time()
    t_cur          = 0.0

    for i in range(1, K + 1):
        coef_n, C1, C2, vp2 = _step_coeffs(cfg, step_sizes, is_nominal, i, 'forward')
        pres_next = calc_step(pres, pres_prev, pres_next, C1, C2, vp2, coef_n, nx, ny)

        t_cur   += step_sizes[i - 1]
        arg      = (pi_f * (t_cur - t0_wavelet))**2
        ricker   = amplitude * (1 - 2*arg) * np.exp(-arg)
        pres_next[src_x, src_z] += C1_at_src * ricker

        pres_prev, pres, pres_next = pres, pres_next, pres_prev

        seismogram[:, i] = pres[recv_x, recv_z]

        if i in weight_at:
            forward_snaps[i] = pres.copy()
        if verbose and i % 500 == 0:
            print(f"  fwd {i}/{K}  ({time.time()-t_start:.1f}s)")

    return forward_snaps, seismogram


'''Обратное распространение адджоинта, сохраняет снэпшоты на шагах saved_at.
Использует тот же step_sizes, что и run_forward, но в обратном порядке — так
момент каждого шага адджоинта совпадает с моментом соответствующего форвард-шага.'''
def run_adjoint(cfg, seismogram, step_sizes, saved_at, is_nominal=None, verbose=True):
    nx, ny    = cfg['nx'], cfg['ny']
    K         = len(step_sizes)
    adj       = np.zeros((nx, ny))
    adj_prev  = np.zeros((nx, ny))
    adj_next  = np.zeros((nx, ny))
    adj_snaps = {}
    adj_vis   = []
    vis_steps = {K//4, K//2, 3*K//4}

    is_nominal = is_nominal if is_nominal is not None else [s == cfg['dt'] for s in step_sizes]
    saved_at   = set(saved_at)

    recv_x, recv_z = cfg['rec_x'], cfg['rec_y']
    C1_at_recv     = cfg['C1'][recv_x, recv_z]
    t_start        = time.time()

    for i in range(K, 0, -1):
        coef_n, C1, C2, vp2 = _step_coeffs(cfg, step_sizes, is_nominal, i, 'adjoint')
        adj_next = calc_step(adj, adj_prev, adj_next, C1, C2, vp2, coef_n, nx, ny)
        adj_next[recv_x, recv_z] += C1_at_recv * seismogram[:, i]
        adj_prev, adj, adj_next = adj, adj_next, adj_prev
        if i in saved_at:
            adj_snaps[i] = adj.copy()
        if i in vis_steps:
            adj_vis.append((i, adj.copy()))
        if verbose and i % 500 == 0:
            print(f"  adj {i}/{K}  ({time.time()-t_start:.1f}s)")

    return adj_snaps, adj_vis


'''Обратное распространение + кросс-корреляция on-the-fly. step_sizes/is_nominal —
та же сетка шагов, что использовал run_forward (форвард- и адджоинт-проход обязаны
использовать одну и ту же сетку, иначе снимки полей окажутся в разные моменты времени).
Возвращает (image, illum_source): энергия форвард-поля (source-side)
для водоуровневой компенсации освещённости, см. src.imaging.illumination_compensate.'''
def run_adjoint_imaging(cfg, seismogram, forward_snaps, step_sizes,
                        weight_at=None, is_nominal=None, verbose=True):
    nx, ny       = cfg['nx'], cfg['ny']
    K            = len(step_sizes)
    adj          = np.zeros((nx, ny))
    adj_prev     = np.zeros((nx, ny))
    adj_next     = np.zeros((nx, ny))
    image        = np.zeros((nx, ny))
    illum_source = np.zeros((nx, ny))

    is_nominal = is_nominal if is_nominal is not None else [s == cfg['dt'] for s in step_sizes]
    weight_at  = weight_at or {}

    recv_x, recv_z = cfg['rec_x'], cfg['rec_y']
    C1_at_recv     = cfg['C1'][recv_x, recv_z]
    t_start        = time.time()

    for i in range(K, 0, -1):
        coef_n, C1, C2, vp2 = _step_coeffs(cfg, step_sizes, is_nominal, i, 'adjoint')
        adj_next = calc_step(adj, adj_prev, adj_next, C1, C2, vp2, coef_n, nx, ny)
        adj_next[recv_x, recv_z] += C1_at_recv * seismogram[:, i]
        adj_prev, adj, adj_next = adj, adj_next, adj_prev

        if i in forward_snaps:
            snap_fwd      = forward_snaps[i]
            weight        = weight_at.get(i, 1.0)
            image        += weight * snap_fwd * adj
            illum_source += weight * snap_fwd * snap_fwd

        if verbose and i % 500 == 0:
            print(f"  adj {i}/{K}  ({time.time()-t_start:.1f}s)")

    return image, illum_source
