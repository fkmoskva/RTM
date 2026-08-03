'''
Эксперимент по сжатию волнового поля RTM.

Для каждой модели:
  1. RTM без сжатия → эталонное изображение
  2. TT и QTT при разных ε → сравнение с эталоном
  3. Критерий качества: rel_error <= 10%
  4. Сводная таблица: коэффициент сжатия, ошибка, экономия памяти

Использование:
  python main.py                        # двухслойная модель
  python main.py --model marmousi
  python main.py --model both
  python main.py --nx 2000 --ny 500 --freq 15
'''

import argparse
import os

from src.compression import compress_tt, decompress_tt, compress_qtt, decompress_qtt, storage_bytes
from src.solver import run_forward, run_adjoint
from src.models import setup_twolayer, setup_marmousi, setup_from_segy
from src.imaging import cross_correlate, metrics, analyse_snapshot
from src.plotting import plot_snapshots, plot_seismogram, plot_image, plot_comparison, plot_metrics_curve


METHODS = [
    ('TT',  compress_tt,  decompress_tt),
    ('QTT', compress_qtt, decompress_qtt),
]

EPSILONS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

REL_THRESHOLD  = 0.1   # 10% — основной критерий качества
PSNR_THRESHOLD = 40.0  # дБ  — справочный порог


'''Форвард + адджоинт, сохраняет графики, возвращает снэпшоты.'''
def propagate(cfg, save_every, out, name):
    forward_snaps, seismogram = run_forward(cfg, save_every=save_every)

    snap_steps    = sorted(forward_snaps)
    display_steps = [snap_steps[len(snap_steps)//4],
                     snap_steps[len(snap_steps)//2],
                     snap_steps[3*len(snap_steps)//4]]
    plot_snapshots(
        [forward_snaps[step] for step in display_steps],
        [f't={step*cfg["dt"]:.3f}с' for step in display_steps],
        cfg['extent_km'], 'Форвард-поле', f'{out}/{name}_fwd.png',
    )
    plot_seismogram(seismogram, cfg, f'{out}/{name}_seis.png')

    adj_snaps, adj_vis = run_adjoint(cfg, seismogram, set(forward_snaps.keys()))
    if adj_vis:
        plot_snapshots(
            [snap for _, snap in adj_vis],
            [f't={step*cfg["dt"]:.3f}с' for step, _ in adj_vis],
            cfg['extent_km'], 'Адджоинт-поле', f'{out}/{name}_adj.png',
        )
    return forward_snaps, adj_snaps


'''[OK] если rel_error <= порог, иначе [!!].'''
def quality_label(result):
    return '[OK]' if result['rel_error'] <= REL_THRESHOLD else '[!!]'


'''Перебор epsilon: сжатие снэпшотов, кросс-корреляция, метрики.'''
def sweep(method, compress_fn, decompress_fn,
          forward_snaps, adj_snaps, ref_image,
          epsilons, ref_bytes, out, name, extent):
    points, images, labels = [], [], []

    for eps in sorted(epsilons, reverse=True):
        compressed      = {step: compress_fn(forward_snaps[step], eps) for step in forward_snaps}
        comp_bytes      = sum(storage_bytes(comp) for comp in compressed.values())
        ratio           = ref_bytes / comp_bytes
        compressed_image = cross_correlate(compressed, adj_snaps, decompress_fn=decompress_fn)
        result          = metrics(compressed_image, ref_image)
        saved_pct       = (1 - comp_bytes / ref_bytes) * 100
        result.update(cr=ratio, epsilon=eps,
                      saved_gb=(ref_bytes - comp_bytes) / 1e9,
                      saved_pct=saved_pct)

        psnr_str = f'{result["psnr"]:.1f}' if result['psnr'] != float('inf') else '∞'
        print(f'    ε={eps:<5}  CR={ratio:>7.1f}x  '
              f'rel_err={result["rel_error"]:.4f}  PSNR={psnr_str:>6} dБ  '
              f'saved={saved_pct:.0f}%  {quality_label(result)}')

        points.append(result)
        images.append(compressed_image)
        labels.append(f'{method} ε={eps}\nCR={ratio:.1f}x')

    plot_comparison([ref_image] + images, ['Эталон'] + labels, extent,
                    f'{method}: качество RTM vs сжатие',
                    f'{out}/{name}_{method.lower()}_cmp.png')
    return points


'''Сводная таблица результатов в консоль.'''
def print_table(results, ref_bytes, model_name, cfg):
    line_w = 82
    print('\n' + '=' * line_w)
    print(f'  {model_name.upper()}  |  {cfg["nx"]}x{cfg["ny"]}  '
          f'|  {cfg["freq"]} Гц  |  ref = {ref_bytes/1e9:.2f} ГБ')
    print(f'  Критерий: rel_error = ||img - ref|| / ||ref|| <= {REL_THRESHOLD*100:.0f}%')
    print('-' * line_w)
    print(f'  {"Метод":<12} {"eps":>6}  {"CR":>9}   {"rel_err":>8}  '
          f'{"PSNR(дБ)":>9}  {"Экономия":>9}  Качество')
    print('-' * line_w)
    print(f'  {"Без сжатия":<12} {"—":>6}  {"1.0x":>9}   {"0.0000":>8}  '
          f'{"inf":>9}  {"0.0%":>9}  [OK]')

    for method, points in results.items():
        print()
        for point in points:
            psnr_str = f'{point["psnr"]:.1f}' if point['psnr'] != float('inf') else 'inf'
            print(f'  {method:<12} {point["epsilon"]:>6}  {point["cr"]:>8.1f}x   '
                  f'{point["rel_error"]:>8.4f}  {psnr_str:>9}  '
                  f'{point["saved_pct"]:>8.1f}%  {quality_label(point)}')

    print('\n' + '-' * line_w)
    print('  Итог (критерий: rel_error <= 10%):')
    for method, points in results.items():
        good_results = [pt for pt in points if pt['rel_error'] <= REL_THRESHOLD]
        if good_results:
            best_result  = max(good_results, key=lambda pt: pt['cr'])
            compressed_gb = ref_bytes / 1e9 / best_result['cr']
            print(f'    {method}: eps={best_result["epsilon"]}  CR={best_result["cr"]:.1f}x  '
                  f'экономия {best_result["saved_pct"]:.1f}%  '
                  f'({ref_bytes/1e9:.2f} ГБ → {compressed_gb:.2f} ГБ)')
        else:
            best_result = min(points, key=lambda pt: pt['rel_error'])
            print(f'    {method}: порог не достигнут. '
                  f'Мин. rel_err={best_result["rel_error"]:.3f} при CR={best_result["cr"]:.1f}x')
    print('=' * line_w)


'''Полный эксперимент: RTM + сжатие + таблица.'''
def run_experiment(name, cfg, save_every, epsilons, out):
    os.makedirs(out, exist_ok=True)

    n_snaps = cfg['nodes_time'] // save_every
    ram_est = n_snaps * cfg['nx'] * cfg['ny'] * 8 / 1e9
    print(f'\n{"="*62}')
    print(f'  Модель: {name}  |  {cfg["nx"]}×{cfg["ny"]}  |  {cfg["freq"]} Гц')
    print(f'  Шагов: {cfg["nodes_time"]}  dt={cfg["dt"]*1e3:.3f} мс  '
          f'~{n_snaps} снэпшотов  ~{ram_est:.1f} ГБ')
    print('=' * 62)

    print('\n[1/3] Распространение волн...')
    forward_snaps, adj_snaps = propagate(cfg, save_every, out, name)
    ref_bytes = sum(snap.size for snap in forward_snaps.values()) * 8
    ref_image = cross_correlate(forward_snaps, adj_snaps)
    plot_image(ref_image, cfg['extent_km'], 'Эталонный RTM', f'{out}/{name}_rtm_ref.png')
    print(f'  {len(forward_snaps)} снэпшотов  {ref_bytes/1e9:.2f} ГБ  '
          f'max={abs(ref_image).max():.3e}')

    print('\n[2/3] Анализ сжатия одного снэпшота...')
    mid_snapshot = forward_snaps[sorted(forward_snaps)[len(forward_snaps)//2]]
    for method in ('tt', 'qtt'):
        print(f'\n  {method.upper()}')
        analyse_snapshot(mid_snapshot, [0.5, 0.1, 0.01, 0.001], method=method)

    print('\n[3/3] Перебор ε по всем снэпшотам...')
    results = {}
    for method, compress_fn, decompress_fn in METHODS:
        print(f'\n  [{method}]')
        results[method] = sweep(
            method, compress_fn, decompress_fn,
            forward_snaps, adj_snaps, ref_image,
            epsilons, ref_bytes, out, name, cfg['extent_km'],
        )

    plot_metrics_curve(results, psnr_threshold=PSNR_THRESHOLD,
                       rel_threshold=REL_THRESHOLD,
                       path=f'{out}/{name}_metrics.png')
    print_table(results, ref_bytes, name, cfg)
    return results


'''Точка входа: разбор аргументов и запуск.'''
def main():
    ap = argparse.ArgumentParser(description='Эксперимент по сжатию волнового поля RTM')
    ap.add_argument('--model',      choices=['twolayer', 'marmousi', 'both', 'segy'], default='twolayer')
    ap.add_argument('--save-every', type=int,   default=20)
    ap.add_argument('--out',        default='results')
    ap.add_argument('--nx',         type=int,   default=2000)
    ap.add_argument('--ny',         type=int,   default=500)
    ap.add_argument('--dx',         type=float, default=10.)
    ap.add_argument('--freq',       type=float, default=10.)
    ap.add_argument('--vp-file',    default=None,
                    help='Путь к SEG-Y файлу с моделью Vp (обязателен при --model segy)')
    ap.add_argument('--segy-dx',    type=float, default=None,
                    help='Шаг сетки в м для SEG-Y модели (авто-определение из заголовка, если не задан)')
    ap.add_argument('--factor',     type=int,   default=1,
                    help='Прореживание SEG-Y модели: 2 → в 2 раза реже по каждой оси')
    args = ap.parse_args()

    if args.model in ('twolayer', 'both'):
        cfg = setup_twolayer(nx=args.nx, ny=args.ny, dx=args.dx, freq=args.freq)
        run_experiment('twolayer', cfg, args.save_every, EPSILONS, args.out)

    if args.model in ('marmousi', 'both'):
        cfg = setup_marmousi(freq=args.freq)
        run_experiment('marmousi', cfg, max(args.save_every, 20), EPSILONS, args.out)

    if args.model == 'segy':
        if not args.vp_file:
            raise SystemExit('Ошибка: для --model segy необходимо указать --vp-file <путь к .sgy файлу>')
        cfg = setup_from_segy(args.vp_file, freq=args.freq, downsample=args.factor, dx=args.segy_dx)
        run_experiment('segy', cfg, max(args.save_every, 20), EPSILONS, args.out)


if __name__ == '__main__':
    main()
