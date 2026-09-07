import numpy as np


'''Кросс-корреляция форвард- и адджоинт-снэпшотов → RTM-изображение.'''
def cross_correlate(forward_snaps, adjoint_snaps, decompress_fn=None, snapshot_weights=None):
    common_steps = sorted(set(forward_snaps) & set(adjoint_snaps))
    if not common_steps:
        return np.zeros_like(next(iter(adjoint_snaps.values())))
    image = np.zeros_like(adjoint_snaps[common_steps[0]])
    for step in common_steps:
        snap_fwd = decompress_fn(forward_snaps[step]) if decompress_fn else forward_snaps[step]
        weight   = snapshot_weights.get(step, 1.0) if snapshot_weights else 1.0
        image   += weight * snap_fwd * adjoint_snaps[step]
    return image


'''
Водоуровневая (water-level) деконволюция RTM-образа по энергии форвард-поля.

I_comp = I / (L_S + eps)

eps = damping * max(L), НЕ абсолютная константа: L_S может
различаться на десятки порядков между зоной у источника и тенью,
поэтому демпфер обязан масштабироваться относительно самой карты
иллюминации (Clayton & Wiggins, 1976, water-level regularization).
'''
def illumination_compensate(image, illum_source, damping=1e-2):
    illum_source = illum_source.astype(np.float64, copy=False)
    eps_s = damping * illum_source.max()

    denom = illum_source + eps_s

    return image / denom


'''Метрики качества: rel_error (L2), rel_error_c (C-норма, max|diff|/max|ref|), RMSE, PSNR.'''
def metrics(image, reference):
    diff      = image - reference
    ref_max   = np.abs(reference).max()
    rel_err   = np.linalg.norm(diff) / max(np.linalg.norm(reference), 1e-30)
    rel_err_c = np.abs(diff).max() / max(ref_max, 1e-30)
    rmse      = np.sqrt(np.mean(diff**2))
    psnr      = 20 * np.log10(ref_max / rmse) if ref_max > 0 and rmse > 0 else np.inf
    return dict(rel_error=rel_err, rel_error_c=rel_err_c, rmse=rmse, psnr=psnr)