'''
Визуализация сохранённого RTM-изображения из .npy файла. Симуляцию не запускает.

Usage:
  python plot_rtm.py results/reference/twolayer/rtm_reference.npy --dx 20 --freq 10
  python plot_rtm.py ...npy --dx 20 --freq 10 --illum
  python plot_rtm.py ...npy --dx 20 --freq 10 --illum --agc 30 --clip-pct 97
'''

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import laplace as ndimage_laplace, uniform_filter


'''Ширина спанж-границы в ячейках.'''
def absorb_cells(dx, vp_min=1500., freq=30.):
    return int(6.5 * vp_min / freq / dx)


'''Нормировка на локальный RMS в окне — выравнивает динамический диапазон.'''
def apply_agc(image, window):
    rms = np.sqrt(uniform_filter(image ** 2, size=window) + 1e-30)
    return image / rms


'''Делит изображение на освещённость из файла rtm_illum.npy рядом с npy.'''
def apply_illum_comp(image, illum_path, eps_frac=0.05):
    if not os.path.exists(illum_path):
        print(f'  [!] Файл освещённости не найден: {illum_path}')
        return image
    illum     = np.load(illum_path)
    illum_max = illum.max()
    eps       = eps_frac * illum_max if illum_max > 0 else 1.
    print(f'  illum comp: max={illum_max:.3e}, eps={eps:.3e}')
    return image / (illum + eps)


'''Отрисовывает физическую область RTM-изображения и сохраняет в PNG.'''
def plot(image, dx, clip_pct, clip_val, cmap, out_path,
         apply_lap=False, agc_window=0, freq=30., vp_min=1500., title='RTM'):
    nx, ny = image.shape
    n_abs  = absorb_cells(dx, vp_min=vp_min, freq=freq)
    dx_km  = dx / 1e3

    phys   = image[n_abs:-n_abs, 0:-n_abs].copy()
    x0     = n_abs * dx_km
    x1     = (nx - n_abs) * dx_km
    z1     = (ny - n_abs) * dx_km
    extent = [x0, x1, z1, 0.]

    if agc_window > 0:
        phys = apply_agc(phys, agc_window)
        print(f'  AGC window={agc_window} ячеек')

    if apply_lap:
        phys = ndimage_laplace(phys)

    clip = clip_val if clip_val is not None \
           else float(np.percentile(np.abs(phys), clip_pct))
    print(f'  clip = {clip:.3e}  (p{clip_pct})')

    pw, ph = x1 - x0, z1
    scale  = 10.0 / max(pw, ph)
    fig, ax = plt.subplots(figsize=(pw * scale, ph * scale))
    ax.imshow(phys.T, cmap=cmap, aspect='auto',
              extent=extent, vmin=-clip, vmax=clip)
    ax.set_xlabel('X, км')
    ax.set_ylabel('Z, км')
    parts  = []
    if agc_window > 0: parts.append(f'AGC={agc_window}')
    if apply_lap:      parts.append('Lap')
    suffix = (' + ' + ' + '.join(parts)) if parts else ''
    ax.set_title(f'{title}  clip={clip:.2e}{suffix}', fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {out_path}')


'''Точка входа: разбор аргументов и вызов plot.'''
def main():
    ap = argparse.ArgumentParser(description='Визуализация RTM без симуляции')
    ap.add_argument('npy',               help='Путь к .npy файлу')
    ap.add_argument('--dx',     type=float, default=20.,   help='Шаг сетки в м')
    ap.add_argument('--freq',   type=float, default=10.,   help='Частота симуляции в Гц')
    ap.add_argument('--vp-min', type=float, default=1500., help='Мин. скорость в м/с')
    ap.add_argument('--illum',  action='store_true', help='Компенсация освещённости')
    ap.add_argument('--lap',    action='store_true', help='Лапласиан-фильтр')
    ap.add_argument('--agc',    type=int,   default=0,     help='Окно AGC в ячейках (0=выкл)')
    ap.add_argument('--clip',   type=float, default=None,  help='Явный clip-уровень')
    ap.add_argument('--clip-pct', type=float, default=99.5, help='Перцентиль авто-clip')
    ap.add_argument('--cmap',   default='gray')
    ap.add_argument('--out',    default=None)
    args = ap.parse_args()

    image = np.load(args.npy)
    print(f'Загружено: {args.npy}  shape={image.shape}  max={np.abs(image).max():.3e}')

    tags = []
    if args.illum:
        illum_path = os.path.join(os.path.dirname(args.npy), 'rtm_illum.npy')
        image = apply_illum_comp(image, illum_path)
        tags.append('illum')
    if args.agc:
        tags.append(f'agc{args.agc}')
    if args.lap:
        tags.append('lap')
    tags.append(f'clip{args.clip:.0e}' if args.clip else f'p{args.clip_pct:.0f}')

    out   = args.out or args.npy.replace('.npy', f'_{"_".join(tags)}.png')
    title = 'RTM' + (' | illum-comp' if args.illum else '')

    plot(image, args.dx, args.clip_pct, args.clip, args.cmap, out,
         apply_lap=args.lap, agc_window=args.agc,
         freq=args.freq, vp_min=args.vp_min, title=title)


if __name__ == '__main__':
    main()
