import os
import numpy as np
from scipy.interpolate import interp1d

try:
    import segyio
except ImportError:
    segyio = None


'''X-индекс приёмной линии (VSP-скважина, фиксированный X, приёмники разнесены по Z).
Совпадает с формулой в build_params — единая точка правды для plot_rtm.py (receiver mute).'''
def receiver_x_index(nx, absorb_cells):
    return min(max(nx // 2, absorb_cells + 2), nx - 3)


'''Коэффициенты leapfrog-обновления волнового уравнения для шага h1, перед
которым был сделан шаг h0 (h0 != h1 для неравномерного шага возле узла
квадратуры). При h0 == h1 эта формула в точности сводится к равномерной
схеме (coef_n=2, C1=1/(1+damping*dt), C2=1-damping*dt, vp2=(Vp*dt/dx)**2) —
она получена обобщением того же дискретного уравнения на разные шаги
до/после текущего момента, а не выведена из PDE заново.

coef_n и C2 пропорциональны h1/h0 — это не опечатка, а прямое следствие
разложения нецентрального второго дифференциала по неравномерной сетке:
чем короче предыдущий шаг h0 относительно текущего h1, тем сильнее вклад
p_prev должен быть скомпенсирован. Если build_step_schedule (см.
src/time_sampling.py) когда-нибудь отдаст h0, на порядки меньший h1 —
например, узел квадратуры почти совпал по времени с соседней точкой, —
буквальное h1/h0 взрывается (h0 ~ 1e-19 давало coef_n ~ 1e15) и разносит
схему за один шаг. Поэтому для расчёта coef_n/C2 используем h0, снизу
ограниченный долей h1: h0_ratio_floor не меняет физику при нормальном
соотношении шагов (h0 обычно ~ h1) и лишь не даёт знаменателю схлопнуться
в вырожденном случае — vp2 такого ограничения не требует, там h0 входит
только слагаемым, без деления.'''
def leapfrog_coeffs(damping, Vp, dx, h0, h1, h0_ratio_floor=0.05):
    h0_safe = max(h0, h0_ratio_floor * h1)
    coef_n = 1. + h1 / h0_safe
    C1     = 1. / (1. + damping * h1)
    C2     = h1 / h0_safe - damping * h1
    vp2    = 0.5 * (h0 + h1) * h1 * (Vp / dx)**2
    return coef_n, C1, C2, vp2


'''Строит словарь параметров модели по сетке скоростей Vp.'''
def build_params(Vp, dx, freq, nodes_time):
    nx, ny         = Vp.shape
    vp_max, vp_min = Vp.max(), Vp.min()
    dt             = 0.5 * dx / (vp_max * np.sqrt(2))

    absorb_len   = 6.5 * (vp_min / freq)
    absorb_cells = int(absorb_len / dx)
    absorb_cells = max(1, min(absorb_cells, nx - 4, ny - 4))

    grid_x, grid_z = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    dist_to_boundary = np.minimum.reduce([grid_x, nx-1-grid_x, grid_z, ny-1-grid_z])
    damping = np.where(
        dist_to_boundary < absorb_cells,
        (42.5 * vp_min / absorb_len) * ((absorb_cells - dist_to_boundary) / absorb_cells)**2,
        0.
    )

    n_receivers = max(20, min(ny // 10, max(1, ny - 2 * absorb_cells)))
    recv_x = np.full(n_receivers, receiver_x_index(nx, absorb_cells), dtype=int)
    # На маленьких сетках с толстой поглощающей границей absorb_cells+1 может
    # оказаться ниже ny-2-absorb_cells — сортируем границы явно, иначе
    # linspace даёт recv_z в убывающем порядке.
    z_lo = min(absorb_cells + 1, ny - 2 - absorb_cells)
    z_hi = max(absorb_cells + 1, ny - 2 - absorb_cells)
    recv_z = np.linspace(
        max(z_lo, 1),
        min(z_hi, ny - 2),
        n_receivers,
        dtype=int,
    )

    _, C1, C2, vp2 = leapfrog_coeffs(damping, Vp, dx, dt, dt)

    return dict(
        Vp=Vp, dx=dx, dt=dt, nx=nx, ny=ny,
        nodes_time=nodes_time, freq=freq,
        amplitude=1., t0=1.2 / freq,
        i_src=nx // 2, j_src=min(max(absorb_cells + 5, 10), ny - 2),
        absorb_cells=absorb_cells,
        damping=damping,
        C1=C1,
        C2=C2,
        vp2=vp2,
        src_scale=dt**2 / (dx**2 * 0.01),
        rec_x=recv_x, rec_y=recv_z, n_receivers=n_receivers,
        extent_km=[0, nx * dx / 1e3, ny * dx / 1e3, 0],
    )


'''Интерполирует 2D сетку скорости на более мелкий равномерный шаг.'''
def interpolate_vp_grid(Vp, dx_in, dx_out):
    if dx_in <= 0:
        raise ValueError('dx_in должен быть положительным')
    if dx_out <= 0:
        raise ValueError('dx_out должен быть положительным')

    Vp = np.asarray(Vp, dtype=np.float64)
    if dx_out >= dx_in:
        return Vp, dx_in

    nx, ny = Vp.shape
    x_old = np.arange(nx, dtype=np.float64) * dx_in
    y_old = np.arange(ny, dtype=np.float64) * dx_in

    x_max = x_old[-1]
    y_max = y_old[-1]
    x_new = np.arange(int(np.ceil(x_max / dx_out)) + 1, dtype=np.float64) * dx_out
    y_new = np.arange(int(np.ceil(y_max / dx_out)) + 1, dtype=np.float64) * dx_out

    interp_x = interp1d(x_old, Vp, axis=0, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    Vp_x = interp_x(x_new)
    interp_y = interp1d(y_old, Vp_x, axis=1, kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    Vp_new = interp_y(y_new)
    return np.asarray(Vp_new, dtype=np.float64), dx_out


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
    surface_z = min(cfg['absorb_cells'] + 2, ny - 2)
    cfg['j_src'] = surface_z
    return cfg


'''Загружает 2D модель Vp из SEG-Y файла; трассы → ось X, отсчёты → ось Z.'''
def setup_from_segy(path, freq=10., t_max=None, downsample=1, dx=None, lambda_points=5.0):
    if segyio is None:
        raise ImportError('Пакет segyio не установлен. Выполните: pip install segyio')

    if not os.path.isfile(path):
        raise FileNotFoundError(f'SEG-Y файл не найден: {path}')

    if freq <= 0:
        raise ValueError('freq должен быть положительным')
    if lambda_points <= 0:
        raise ValueError('lambda_points должен быть положительным')

    try:
        with segyio.open(path, ignore_geometry=True) as f:
            Vp = np.stack([tr.copy() for tr in f.trace], axis=0).astype(np.float64)
            if dx is None:
                # Заголовок трассы (byte 117-118) надёжнее общего бинарного
                # заголовка: последний нередко содержит устаревшее/нулевое
                # значение, тогда как поле трассы одинаково для всех трасс.
                try:
                    trace_interval = f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
                except Exception:
                    trace_interval = None
                try:
                    bin_interval = f.bin[segyio.BinField.Interval]
                except Exception:
                    bin_interval = None

                if trace_interval and trace_interval > 0:
                    dx = float(trace_interval) / 1000.0
                elif bin_interval and bin_interval > 0:
                    dx = float(bin_interval) / 1000.0
                else:
                    print('  Предупреждение: не удалось прочитать шаг сетки из заголовков SEG-Y. '
                          'Используется dx=10.0 м по умолчанию.')
                    dx = 10.0
    except segyio.exceptions.InvalidFile as exc:
        raise ValueError(f'Не удалось прочитать SEG-Y файл ({path}): {exc}') from exc

    if downsample > 1:
        Vp = Vp[::downsample, ::downsample]
        dx *= downsample

    vp_min = float(Vp.min())
    lambda_min = vp_min / freq
    dx_limit = lambda_min / lambda_points

    dx_before_interp = dx
    if dx > dx_limit:
        Vp, dx = interpolate_vp_grid(Vp, dx, dx_limit)
        print(f'  SEG-Y grid refined: dx {dx_before_interp:.3f} -> {dx:.3f} м '
              f'(lambda_min={lambda_min:.3f} м, limit=lambda/{lambda_points:g}={dx_limit:.3f} м)')
    else:
        print(f'  SEG-Y grid kept: dx={dx:.3f} м '
              f'(lambda_min={lambda_min:.3f} м, limit=lambda/{lambda_points:g}={dx_limit:.3f} м)')

    nx, ny = Vp.shape

    absorb_cells_est = int(6.5 * vp_min / freq / dx)
    dt_est = 0.5 * dx / (float(Vp.max()) * np.sqrt(2))

    if t_max is None:
        depth  = max(ny - 2 * absorb_cells_est, 1) * dx
        offset = max(nx//2 - absorb_cells_est, 1) * dx
        t_max  = np.sqrt(offset**2 + 4 * depth**2) / vp_min

    nodes_time = max(200, int(t_max / dt_est) + 1)
    cfg = build_params(Vp, dx, freq, nodes_time)
    surface_z = min(cfg['absorb_cells'] + 2, ny - 2)
    cfg['j_src'] = surface_z
    return cfg
