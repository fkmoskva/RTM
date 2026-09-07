'''
Эталонный multi-shot RTM без сжатия.

Источник равномерно перемещается по физической области (n_shots позиций),
RTM-изображения суммируются. Для каждого шота хранятся только его
форвард-снимки; обратное поле считается on-the-fly. После шота память
освобождается.

Usage:
  python reference_rtm.py                                        # twolayer, 500x500, 5 шотов
  python reference_rtm.py --nx 1000 --ny 500 --freq 30          # крупнее, выше частота
  python reference_rtm.py --model marmousi --factor 2 --freq 15 --n-shots 64 --yes
  python reference_rtm.py --model marmousi --n-shots 32 --save-every 50 --yes
'''

import argparse
import os
import time
import numpy as np

from src.models import setup_twolayer, setup_marmousi, setup_from_segy
from src.solver import run_forward, run_adjoint_imaging
from src.imaging import illumination_compensate
from src.time_sampling import composite_gauss_schedule, build_step_schedule, uniform_step_schedule


OUT = 'results/reference'


'''Объём RAM на один шот в ГБ.'''
def estimate_memory_gb(cfg, n_snaps):
    return n_snaps * cfg['nx'] * cfg['ny'] * 8 / 1e9


'''Равномерно расставляет n_shots источников в физической X-области.'''
def shot_positions(cfg, n_shots):
    n_abs = cfg['absorb_cells']
    return np.linspace(n_abs + 1, cfg['nx'] - n_abs - 1, n_shots, dtype=int)


'''Печатает параметры запуска в консоль.'''
def print_header(cfg, n_snaps, n_shots, model):
    mem_gb  = estimate_memory_gb(cfg, n_snaps)
    shots   = shot_positions(cfg, n_shots)
    xs_km   = ', '.join(f'{x*cfg["dx"]/1e3:.1f}' for x in shots)
    phys_x_cells = cfg['nx'] - 2 * cfg['absorb_cells']
    phys_z_cells = cfg['ny'] - 2 * cfg['absorb_cells']
    phys_note = ''
    if phys_x_cells <= 0 or phys_z_cells <= 0:
        phys_x_km = cfg['nx'] * cfg['dx'] / 1e3
        phys_z_km = cfg['ny'] * cfg['dx'] / 1e3
        phys_note = ' (crop disabled; showing full extent)'
    else:
        phys_x_km = phys_x_cells * cfg['dx'] / 1e3
        phys_z_km = phys_z_cells * cfg['dx'] / 1e3
    print('=' * 64)
    print(f'  Multi-shot RTM — {model}')
    print(f'  Сетка         : {cfg["nx"]} x {cfg["ny"]}   dx = {cfg["dx"]:.0f} м')
    print(f'  Физич. область: {phys_x_km:.1f} x {phys_z_km:.1f} км{phys_note}')
    print(f'  Частота       : {cfg["freq"]} Гц')
    print(f'  t_max         : {cfg["nodes_time"]*cfg["dt"]:.2f} с  ({cfg["nodes_time"]} шагов)')
    print(f'  Шотов         : {n_shots}  x_src = [{xs_km}] км')
    print(f'  Снимков/шот   : {n_snaps}')
    print(f'  RAM/шот       : ~{mem_gb:.1f} ГБ')
    print('=' * 64)
    if mem_gb > 6:
        print(f'  ВНИМАНИЕ: {mem_gb:.1f} ГБ/шот — увеличьте --save-every / --gll-segments')


'''Безопасно обрезает физическую область и возвращает срезы по осям.'''
def physical_crop(image, n_abs, crop_x=True, crop_z=True):
    nx, ny = image.shape
    n_abs = max(0, int(n_abs))
    crop_x_ok = crop_x and 2 * n_abs < nx
    crop_z_ok = crop_z and 2 * n_abs < ny
    x0 = n_abs if crop_x_ok else 0
    z0 = n_abs if crop_z_ok else 0
    x1 = nx - n_abs if crop_x_ok else nx
    z1 = ny - n_abs if crop_z_ok else ny
    return image[x0:x1, z0:z1], x0, x1, z0, z1


def save_rtm_image(image, cfg, out, filename, title_extra='', clip_val=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_abs  = cfg['absorb_cells']
    dx_km  = cfg['dx'] / 1e3

    phys, x0_i, x1_i, z0_i, z1_i = physical_crop(image, n_abs, crop_x=True, crop_z=True)
    x0, x1 = x0_i * dx_km, x1_i * dx_km
    z0, z1 = z0_i * dx_km, z1_i * dx_km
    extent = [x0, x1, z1, z0]

    phys_abs = np.abs(phys)
    clip = clip_val if clip_val is not None \
           else max(np.percentile(phys_abs, 99.9), 1e-30)

    # Держим фигуру читаемой, даже если физическая область сильно вытянута.
    fig_w = 14.0
    fig_h = 6.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.imshow(phys.T, cmap='gray', aspect='auto',
              extent=extent, vmin=-clip, vmax=clip)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Z, км')
    ax.set_title(f'RTM {cfg["nx"]}x{cfg["ny"]} {cfg["freq"]} Гц{title_extra}  '
                 f'clip={clip:.2e}', fontsize=10)
    path = f'{out}/{filename}'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


'''Форвард + адджоинт для одного шота. Возвращает (image, illum_source).
step_sizes/weight_at/is_nominal — сетка шагов по времени (src.time_sampling),
общая для forward- и adjoint-прохода.'''
def run_one_shot(cfg, i_src, step_sizes, weight_at, is_nominal, out, shot_idx, n_shots):
    cfg = dict(cfg)
    cfg['i_src'] = i_src
    x_km = i_src * cfg['dx'] / 1e3
    print(f'\n  [Шот {shot_idx+1}/{n_shots}]  i_src={i_src}  x={x_km:.2f} км')

    t_start = time.time()
    forward_snaps, seismogram = run_forward(
        cfg, step_sizes, weight_at, is_nominal, verbose=False,
    )
    print(f'    forward: {len(forward_snaps)} снимков  ({time.time()-t_start:.1f}с)')

    t_start = time.time()
    shot_image, shot_illum_source = run_adjoint_imaging(
        cfg, seismogram, forward_snaps, step_sizes, weight_at, is_nominal, verbose=False,
    )
    print(f'    adjoint: max={np.abs(shot_image).max():.3e}  ({time.time()-t_start:.1f}с)')

    del forward_snaps, seismogram
    return shot_image, shot_illum_source


'''Multi-shot RTM: суммирует шоты, сохраняет .npy и PNG.'''
def run_reference(cfg, step_sizes, weight_at, is_nominal, out, n_shots=5, clip_val=None,
                  illum_comp=False, illum_damping=1e-2):
    os.makedirs(out, exist_ok=True)
    t_total = time.time()

    shot_xs              = shot_positions(cfg, n_shots)
    total_image          = np.zeros((cfg['nx'], cfg['ny']))
    total_illum_source   = np.zeros((cfg['nx'], cfg['ny']))

    for idx, i_src in enumerate(shot_xs):
        shot_img, shot_illum_source = run_one_shot(
            cfg, i_src, step_sizes, weight_at, is_nominal, out, idx, n_shots,
        )
        total_image          += shot_img
        total_illum_source   += shot_illum_source

    elapsed = time.time() - t_total
    print(f'\n  Все шоты готовы за {elapsed/60:.1f} мин')
    print(f'  Image max = {np.abs(total_image).max():.4e}')

    if illum_comp:
        out_image = illumination_compensate(total_image, total_illum_source, damping=illum_damping)
        print(f'  Illumination comp. (water-level, source-only): damping={illum_damping:.3g}')
    else:
        out_image = total_image

    npy_path = f'{out}/rtm_reference.npy'
    np.save(npy_path, out_image)
    np.save(f'{out}/rtm_illum_source.npy', total_illum_source)
    np.savez(f'{out}/rtm_shots.npz', i_src=shot_xs, j_src=cfg['j_src'])
    np.savez(
        f'{out}/rtm_meta.npz',
        dx=float(cfg['dx']),
        freq=float(cfg['freq']),
        vp_min=float(cfg['Vp'].min()),
        vp_max=float(cfg['Vp'].max()),
        absorb_cells=int(cfg['absorb_cells']),
        nx=int(cfg['nx']),
        ny=int(cfg['ny']),
        i_src=int(cfg['i_src']),
        j_src=int(cfg['j_src']),
        n_shots=int(n_shots),
        n_snapshots=int(len(weight_at)),
    )

    n_abs = cfg['absorb_cells']
    phys, *_ = physical_crop(out_image, n_abs, crop_x=True, crop_z=True)
    label = f' | {n_shots} шотов' + (' | illum' if illum_comp else '')

    final_path = save_rtm_image(out_image, cfg, out, 'rtm_reference.png',
                                title_extra=label, clip_val=clip_val)
    tight_path = save_rtm_image(out_image, cfg, out, 'rtm_reference_tight.png',
                                title_extra=label,
                                clip_val=max(np.percentile(np.abs(phys), 99.5), 1e-30))
    print(f'  {final_path}')
    print(f'  {tight_path}')
    print(f'  {npy_path}')
    return out_image


'''Точка входа: разбор аргументов и запуск.'''
def main():
    ap = argparse.ArgumentParser(description='Multi-shot эталонный RTM')
    ap.add_argument('--model',          choices=['twolayer', 'marmousi', 'segy'], default='twolayer')
    ap.add_argument('--nx',             type=int,   default=500)
    ap.add_argument('--ny',             type=int,   default=500)
    ap.add_argument('--dx',             type=float, default=None,
                    help='Шаг сетки в м (по умолч. из --pts-per-lambda или 20 м)')
    ap.add_argument('--freq',           type=float, default=10.)
    ap.add_argument('--pts-per-lambda', type=float, default=None,
                    help='Точек на длину волны; авто-вычисляет dx = Vp_min / (freq * N)')
    ap.add_argument('--n-shots',        type=int,   default=None)
    ap.add_argument('--save-every',     type=int,   default=None)
    ap.add_argument('--time-sampling',  choices=['uniform', 'gauss'], default='uniform',
                    help='Выбор временных снимков: равномерная сетка или составная квадратура Гаусса-Лежандра')
    ap.add_argument('--gauss-segments', type=int, default=5,
                    help='Число временных сегментов составной квадратуры Гаусса-Лежандра')
    ap.add_argument('--gauss-points',   type=int, default=5,
                    help='Число узлов Гаусса-Лежандра в одном временном сегменте')
    ap.add_argument('--src-z',          type=int,   default=None,
                    help='Глубина источника в ячейках (только twolayer)')
    ap.add_argument('--factor',         type=int,   default=1,
                    help='Прореживание Marmousi/SEG-Y: 2=dx8м (~8x быстрее), 3=dx12м (~27x быстрее)')
    ap.add_argument('--vp-file',        default=None,
                    help='Путь к SEG-Y файлу с моделью Vp (обязателен при --model segy)')
    ap.add_argument('--segy-dx',        type=float, default=None,
                    help='Шаг сетки в м для SEG-Y модели (авто-определение из заголовка, если не задан; '
                        'при необходимости модель будет интерполирована до dx <= lambda_min/5)')
    ap.add_argument('--clip',           type=float, default=None)
    ap.add_argument('--illum-comp',     action='store_true',
                    help='Illumination compensation (выкл по умолч.)')
    ap.add_argument('--illum-damping',  type=float, default=1e-2,
                    help='Water-level damping для --illum-comp: eps = damping * max(illum) '
                         '(меньше=резче/шумнее, больше=мягче)')
    ap.add_argument('--out',            default=OUT)
    ap.add_argument('--yes',            action='store_true')
    args = ap.parse_args()

    if args.model == 'marmousi':
        freq    = args.freq if args.freq != 10. else 30.
        cfg     = setup_marmousi(freq=freq, downsample=args.factor)
        n_shots = args.n_shots or 20
        nt             = cfg['nodes_time']
        bytes_per_snap = cfg['nx'] * cfg['ny'] * 8
        auto_every     = max(1, nt // int(2e9 / bytes_per_snap))
        save_every     = args.save_every or auto_every
        if not args.save_every:
            print(f'  Авто save_every={save_every} '
                  f'(RAM/шот ~{nt//save_every*bytes_per_snap/1e9:.1f} ГБ)')
        if args.pts_per_lambda is not None:
            dx_actual  = 4. * args.factor
            pts_actual = 1500. / (freq * dx_actual)
            print(f'  Marmousi: dx фиксирован={dx_actual:.0f} м ({pts_actual:.1f} пт/λ), '
                  f'--pts-per-lambda игнорируется')
    elif args.model == 'segy':
        if not args.vp_file:
            raise SystemExit('Ошибка: для --model segy необходимо указать --vp-file <путь к .sgy файлу>')
        freq = args.freq
        cfg  = setup_from_segy(args.vp_file, freq=freq, downsample=args.factor, dx=args.segy_dx)
        n_shots = args.n_shots or 20
        nt             = cfg['nodes_time']
        bytes_per_snap = cfg['nx'] * cfg['ny'] * 8
        auto_every     = max(1, nt // int(2e9 / bytes_per_snap))
        save_every     = args.save_every or auto_every
        if not args.save_every:
            print(f'  Авто save_every={save_every} '
                  f'(RAM/шот ~{nt//save_every*bytes_per_snap/1e9:.1f} ГБ)')
    else:
        freq   = args.freq
        vp_min = 2000.
        if args.pts_per_lambda is not None:
            dx = vp_min / (freq * args.pts_per_lambda)
            print(f'  dx = {dx:.2f} м  ({args.pts_per_lambda:.0f} пт/λ)')
        else:
            dx = args.dx if args.dx is not None else 20.
        cfg     = setup_twolayer(nx=args.nx, ny=args.ny, dx=dx, freq=freq)
        n_shots = args.n_shots or 5
        save_every = args.save_every or 10
        if args.src_z is not None:
            cfg['j_src'] = max(cfg['absorb_cells'] + 2, min(args.src_z, cfg['ny'] - 2))

    dt = cfg['dt']
    if args.time_sampling == 'gauss':
        times, weights = composite_gauss_schedule(
            cfg['nodes_time'], dt, args.gauss_segments, args.gauss_points,
        )
        step_sizes, weight_at, is_nominal = build_step_schedule(times, weights, cfg['nodes_time'], dt)
        print(f'  Составная квадратура Гаусса-Лежандра: {args.gauss_segments} сегм. × '
              f'{args.gauss_points} узл. = {len(weight_at)} снимков '
              f'({len(step_sizes)} шагов решателя, из них {len(step_sizes)-cfg["nodes_time"]+1} доп. к базовой сетке)')
    else:
        step_sizes, weight_at, is_nominal = uniform_step_schedule(cfg['nodes_time'], dt, save_every)

    print_header(cfg, len(weight_at), n_shots, args.model)

    ram_gb = estimate_memory_gb(cfg, len(weight_at))
    if not args.yes and ram_gb > 10:
        ans = input('  Продолжить? [y/N]: ').strip().lower()
        if ans != 'y':
            raise SystemExit('Прервано.')

    out = args.out if args.out != OUT else f'{OUT}/{args.model}'
    run_reference(cfg, step_sizes, weight_at, is_nominal, out,
                  n_shots=n_shots, clip_val=args.clip,
                  illum_comp=args.illum_comp, illum_damping=args.illum_damping)


if __name__ == '__main__':
    main()
