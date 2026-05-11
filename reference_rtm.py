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

from src.models import setup_twolayer, setup_marmousi
from src.solver import run_forward, run_adjoint_imaging
from src.plotting import plot_seismogram


OUT = 'results/reference'


'''Объём RAM на один шот в ГБ.'''
def estimate_memory_gb(cfg, save_every):
    n_snaps = cfg['nodes_time'] // save_every
    return n_snaps * cfg['nx'] * cfg['ny'] * 8 / 1e9


'''Равномерно расставляет n_shots источников в физической X-области.'''
def shot_positions(cfg, n_shots):
    n_abs = cfg['absorb_cells']
    return np.linspace(n_abs + 1, cfg['nx'] - n_abs - 1, n_shots, dtype=int)


'''Печатает параметры запуска в консоль.'''
def print_header(cfg, save_every, n_shots, model):
    mem_gb  = estimate_memory_gb(cfg, save_every)
    n_snaps = cfg['nodes_time'] // save_every
    shots   = shot_positions(cfg, n_shots)
    xs_km   = ', '.join(f'{x*cfg["dx"]/1e3:.1f}' for x in shots)
    print('=' * 64)
    print(f'  Multi-shot RTM — {model}')
    print(f'  Сетка         : {cfg["nx"]} x {cfg["ny"]}   dx = {cfg["dx"]:.0f} м')
    print(f'  Физич. область: {(cfg["nx"]-2*cfg["absorb_cells"])*cfg["dx"]/1e3:.1f} x '
          f'{(cfg["ny"]-2*cfg["absorb_cells"])*cfg["dx"]/1e3:.1f} км')
    print(f'  Частота       : {cfg["freq"]} Гц')
    print(f'  t_max         : {cfg["nodes_time"]*cfg["dt"]:.2f} с  ({cfg["nodes_time"]} шагов)')
    print(f'  Шотов         : {n_shots}  x_src = [{xs_km}] км')
    print(f'  Снимков/шот   : ~{n_snaps}  (save_every={save_every})')
    print(f'  RAM/шот       : ~{mem_gb:.1f} ГБ')
    print('=' * 64)
    if mem_gb > 6:
        print(f'  ВНИМАНИЕ: {mem_gb:.1f} ГБ/шот — увеличьте --save-every')


'''Сохраняет RTM-изображение в PNG, обрезая спанж-границы.'''
def save_rtm_image(image, cfg, out, filename, title_extra='', clip_val=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_abs  = cfg['absorb_cells']
    dx_km  = cfg['dx'] / 1e3

    phys   = image[n_abs:-n_abs, 0:-n_abs]
    x0, x1 = n_abs * dx_km, (cfg['nx'] - n_abs) * dx_km
    z0, z1 = 0., (cfg['ny'] - n_abs) * dx_km
    extent = [x0, x1, z1, z0]

    phys_abs = np.abs(phys)
    clip = clip_val if clip_val is not None \
           else max(np.percentile(phys_abs, 99.9), 1e-30)

    pw, ph = x1 - x0, z1 - z0
    scale  = 10.0 / max(pw, ph)

    fig, ax = plt.subplots(figsize=(pw * scale, ph * scale))
    ax.imshow(phys.T, cmap='gray', aspect='auto',
              extent=extent, vmin=-clip, vmax=clip)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Z, км')
    ax.set_title(f'RTM {cfg["nx"]}x{cfg["ny"]} {cfg["freq"]} Гц{title_extra}  '
                 f'clip={clip:.2e}', fontsize=10)
    plt.tight_layout()
    path = f'{out}/{filename}'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


'''Форвард + адджоинт для одного шота. Возвращает (image, illumination).'''
def run_one_shot(cfg, i_src, save_every, out, shot_idx, n_shots):
    cfg = dict(cfg)
    cfg['i_src'] = i_src
    x_km = i_src * cfg['dx'] / 1e3
    print(f'\n  [Шот {shot_idx+1}/{n_shots}]  i_src={i_src}  x={x_km:.2f} км')

    t_start = time.time()
    forward_snaps, seismogram = run_forward(cfg, save_every=save_every, verbose=False)
    print(f'    forward: {len(forward_snaps)} снимков  ({time.time()-t_start:.1f}с)')

    t_start = time.time()
    shot_image, shot_illum = run_adjoint_imaging(cfg, seismogram, forward_snaps, verbose=False)
    print(f'    adjoint: max={np.abs(shot_image).max():.3e}  ({time.time()-t_start:.1f}с)')

    if shot_idx == 0:
        plot_seismogram(seismogram, cfg, f'{out}/seismogram_shot0.png')

    del forward_snaps, seismogram
    return shot_image, shot_illum


'''Multi-shot RTM: суммирует шоты, сохраняет .npy и PNG.'''
def run_reference(cfg, save_every, out, n_shots=5, clip_val=None, illum_comp=False):
    os.makedirs(out, exist_ok=True)
    t_total = time.time()

    shot_xs     = shot_positions(cfg, n_shots)
    total_image = np.zeros((cfg['nx'], cfg['ny']))
    total_illum = np.zeros((cfg['nx'], cfg['ny']))

    for idx, i_src in enumerate(shot_xs):
        shot_img, shot_illum = run_one_shot(cfg, i_src, save_every, out, idx, n_shots)
        total_image += shot_img
        total_illum += shot_illum

    elapsed = time.time() - t_total
    print(f'\n  Все шоты готовы за {elapsed/60:.1f} мин')
    print(f'  Image max = {np.abs(total_image).max():.4e}')

    if illum_comp:
        illum_max = total_illum.max()
        illum_eps = 0.05 * illum_max if illum_max > 0 else 1.
        out_image = total_image / (total_illum + illum_eps)
        print(f'  Illumination comp.: eps={illum_eps:.3e}')
    else:
        out_image = total_image

    npy_path = f'{out}/rtm_reference.npy'
    np.save(npy_path, out_image)
    np.save(f'{out}/rtm_illum.npy', total_illum)

    n_abs = cfg['absorb_cells']
    phys  = out_image[n_abs:-n_abs, n_abs:-n_abs]
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
    ap.add_argument('--model',          choices=['twolayer', 'marmousi'], default='twolayer')
    ap.add_argument('--nx',             type=int,   default=500)
    ap.add_argument('--ny',             type=int,   default=500)
    ap.add_argument('--dx',             type=float, default=None,
                    help='Шаг сетки в м (по умолч. из --pts-per-lambda или 20 м)')
    ap.add_argument('--freq',           type=float, default=10.)
    ap.add_argument('--pts-per-lambda', type=float, default=None,
                    help='Точек на длину волны; авто-вычисляет dx = Vp_min / (freq * N)')
    ap.add_argument('--n-shots',        type=int,   default=None)
    ap.add_argument('--save-every',     type=int,   default=None)
    ap.add_argument('--src-z',          type=int,   default=None,
                    help='Глубина источника в ячейках (только twolayer)')
    ap.add_argument('--factor',         type=int,   default=1,
                    help='Прореживание Marmousi: 2=dx8м (~8x быстрее), 3=dx12м (~27x быстрее)')
    ap.add_argument('--clip',           type=float, default=None)
    ap.add_argument('--illum-comp',     action='store_true',
                    help='Illumination compensation (выкл по умолч.)')
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

    print_header(cfg, save_every, n_shots, args.model)

    ram_gb = estimate_memory_gb(cfg, save_every)
    if not args.yes and ram_gb > 6:
        ans = input('  Продолжить? [y/N]: ').strip().lower()
        if ans != 'y':
            raise SystemExit('Прервано.')

    out = args.out if args.out != OUT else f'{OUT}/{args.model}'
    run_reference(cfg, save_every, out,
                  n_shots=n_shots, clip_val=args.clip,
                  illum_comp=args.illum_comp)


if __name__ == '__main__':
    main()
