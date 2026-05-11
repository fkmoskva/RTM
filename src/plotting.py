import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


'''Уровень clip: frac * max(abs(img)).'''
def clip_level(img, frac=0.02):
    return max(frac * np.abs(img).max(), 1e-30)


'''Сохраняет фигуру в файл и закрывает.'''
def save_fig(fig, path):
    plt.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


'''Отрисовывает несколько снэпшотов волнового поля в ряд.'''
def plot_snapshots(snaps, titles, extent, suptitle, path=None):
    fig, axes = plt.subplots(1, len(snaps), figsize=(5*len(snaps), 4))
    if len(snaps) == 1:
        axes = [axes]
    for ax, snap, title in zip(axes, snaps, titles):
        clip_val = clip_level(snap, 0.1)
        ax.imshow(snap.T, cmap='RdBu', extent=extent, aspect='auto',
                  vmin=-clip_val, vmax=clip_val)
        ax.set(xlabel='X (км)', ylabel='Z (км)', title=title)
    fig.suptitle(suptitle)
    save_fig(fig, path)


'''Отрисовывает сейсмограмму (приёмники × время).'''
def plot_seismogram(seismogram, cfg, path=None):
    time_axis = np.arange(seismogram.shape[1]) * cfg['dt']
    clip_val  = clip_level(seismogram)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(seismogram.T, cmap='gray', aspect='auto',
              extent=[cfg['rec_x'][0]*cfg['dx']/1e3,
                      cfg['rec_x'][-1]*cfg['dx']/1e3,
                      time_axis[-1], 0],
              vmin=-clip_val, vmax=clip_val)
    ax.set(xlabel='Приёмник X (км)', ylabel='Время (с)', title='Сейсмограмма')
    save_fig(fig, path)


'''Отрисовывает RTM-изображение.'''
def plot_image(img, extent, title='RTM Image', path=None):
    clip_val = clip_level(img)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(img.T, cmap='gray', extent=extent, aspect='auto',
              vmin=-clip_val, vmax=clip_val)
    ax.set(xlabel='X (км)', ylabel='Z (км)', title=title)
    save_fig(fig, path)


'''Сравнение нескольких RTM-изображений в ряд (одинаковый clip).'''
def plot_comparison(images, titles, extent, suptitle, path=None):
    clip_val = clip_level(images[0])
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 3.5))
    if len(images) == 1:
        axes = [axes]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image.T, cmap='gray', extent=extent, aspect='auto',
                  vmin=-clip_val, vmax=clip_val)
        ax.set(xlabel='X (км)', ylabel='Z (км)', title=title)
    fig.suptitle(suptitle)
    save_fig(fig, path)


'''График rel_error и PSNR от коэффициента сжатия.'''
def plot_metrics_curve(results, psnr_threshold=40.0, rel_threshold=0.1, path=None):
    fig, (ax_err, ax_psnr) = plt.subplots(1, 2, figsize=(12, 5))
    for method, points in results.items():
        ratios     = [pt['cr'] for pt in points]
        rel_errors = [pt['rel_error'] for pt in points]
        ax_err.semilogy(ratios, rel_errors, 'o-', label=method)
        finite_psnr = [(r, pt['psnr']) for r, pt in zip(ratios, points)
                       if pt['psnr'] != float('inf')]
        if finite_psnr:
            ax_psnr.plot(*zip(*finite_psnr), 'o-', label=method)
    ax_err.axhline(rel_threshold, color='red', linestyle='--',
                   label=f'rel_err = {rel_threshold*100:.0f}%')
    ax_psnr.axhline(psnr_threshold, color='red', linestyle='--',
                    label=f'PSNR = {psnr_threshold:.0f} дБ')
    for ax, ylabel in [(ax_err, 'Relative error'), (ax_psnr, 'PSNR (дБ)')]:
        ax.set(xlabel='Коэффициент сжатия', ylabel=ylabel)
        ax.legend()
        ax.grid(True, alpha=.3)
    fig.suptitle('Качество RTM vs сжатие')
    save_fig(fig, path)
