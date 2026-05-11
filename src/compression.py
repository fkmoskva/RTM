import numpy as np
import teneva


'''Наименьшее целое >= n, разложимое только на простые из primes.'''
def smooth_ceil(n, primes=(2, 3, 5, 7)):
    for candidate in range(n, n + 2000):
        remainder = candidate
        for prime in primes:
            while remainder % prime == 0:
                remainder //= prime
        if remainder == 1:
            return candidate
    raise RuntimeError(f"нет 7-гладкого числа в [{n}, {n+2000})")


'''Разложение n на моды размером <= max_mode.'''
def factorize(n, primes=(2, 3, 5, 7), max_mode=16):
    raw_factors = []
    for prime in primes:
        while n % prime == 0:
            raw_factors.append(prime)
            n //= prime
    modes, current = [], 1
    for prime in sorted(raw_factors):
        if current * prime <= max_mode:
            current *= prime
        else:
            modes.append(current)
            current = prime
    return modes + ([current] if current > 1 else [])


'''Перемежающая перестановка битов x и z: (x_MSB, z_MSB, ...). Снижает TT-ранг для 2D данных.'''
def interleave_perm(bits_x, bits_z):
    bit_perm = []
    for i in range(bits_z):
        bit_perm.append(i)
        bit_perm.append(bits_x + i)
    bit_perm.extend(range(bits_z, bits_x))
    return bit_perm


# ── TT-сжатие ─────────────────────────────────────────────────────────────────

'''Сжатие снэпшота в TT-формат с точностью eps.'''
def compress_tt(snapshot, eps, max_mode=16):
    nx, nz = snapshot.shape
    nx_pad, nz_pad = smooth_ceil(nx), smooth_ceil(nz)
    padded = np.zeros((nx_pad, nz_pad))
    padded[:nx, :nz] = snapshot
    shape = tuple(factorize(nx_pad, max_mode=max_mode) + factorize(nz_pad, max_mode=max_mode))
    cores = teneva.svd(padded.reshape(shape), e=eps)
    return {'cores': cores, 'shape': (nx, nz), 'padded_shape': (nx_pad, nz_pad)}


'''Восстановление снэпшота из TT-формата.'''
def decompress_tt(compressed):
    nx, nz = compressed['shape']
    nx_pad, nz_pad = compressed['padded_shape']
    return teneva.full(compressed['cores']).reshape(nx_pad, nz_pad)[:nx, :nz]


# ── QTT-сжатие с перемежением битов x/z ──────────────────────────────────────

'''Сжатие снэпшота в QTT-формат с перемежением битов x/z.'''
def compress_qtt(snapshot, eps):
    nx, nz = snapshot.shape
    nx_pad = 1 << int(np.ceil(np.log2(max(nx, 2))))
    nz_pad = 1 << int(np.ceil(np.log2(max(nz, 2))))
    bits_x = int(np.log2(nx_pad))
    bits_z = int(np.log2(nz_pad))

    padded = np.zeros((nx_pad, nz_pad))
    padded[:nx, :nz] = snapshot

    bit_perm = interleave_perm(bits_x, bits_z)
    tensor   = np.transpose(padded.reshape([2]*bits_x + [2]*bits_z), bit_perm).copy()
    cores    = teneva.svd(tensor, e=eps)
    return {'cores': cores, 'shape': (nx, nz), 'padded_shape': (nx_pad, nz_pad),
            'bits_x': bits_x, 'bits_z': bits_z, 'bit_perm': bit_perm}


'''Восстановление снэпшота из QTT-формата.'''
def decompress_qtt(compressed):
    nx, nz   = compressed['shape']
    nx_pad, nz_pad = compressed['padded_shape']
    bit_perm = compressed['bit_perm']
    inv_bit_perm = [0] * len(bit_perm)
    for i, pos in enumerate(bit_perm):
        inv_bit_perm[pos] = i
    tensor = np.transpose(teneva.full(compressed['cores']), inv_bit_perm)
    return tensor.reshape(nx_pad, nz_pad)[:nx, :nz]


'''Суммарный объём TT-ядер в байтах.'''
def storage_bytes(compressed):
    return sum(core.size for core in compressed['cores']) * 8
