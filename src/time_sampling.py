import numpy as np
from numpy.polynomial.legendre import leggauss

DT_DELTA = 1e-10

def gauss_legendre_nodes_weights(n_points):
    """Узлы и веса квадратуры Гаусса-Лежандра на стандартном интервале [-1, 1]."""
    if n_points < 1:
        raise ValueError('квадратура Гаусса-Лежандра требует хотя бы одну точку')
    return leggauss(n_points)


def composite_gauss_schedule(nodes_time, dt, n_segments=5, points_per_segment=5):
    '''Building segments and weights for gauss'''
    if nodes_time < 3:
        raise ValueError('nodes_time должно быть не меньше 3')
    if dt <= 0:
        raise ValueError('dt должно быть положительным')
    if n_segments < 1:
        raise ValueError('n_segments должно быть положительным')
    if points_per_segment < 1:
        raise ValueError('points_per_segment должно быть не меньше 1')

    t_first, t_last = dt, (nodes_time - 1) * dt
    boundaries = np.linspace(t_first, t_last, n_segments + 1)

    base_nodes, base_weights = gauss_legendre_nodes_weights(points_per_segment)

    times_by_seg, weights_by_seg = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]): #loaded segments
        half_len = 0.5 * (right - left)
        times_by_seg.append(half_len * base_nodes + 0.5 * (right + left))
        weights_by_seg.append(half_len * base_weights / dt)

    times = np.concatenate(times_by_seg)
    weights = np.concatenate(weights_by_seg)

    return times, weights


def build_step_schedule(times, weights, nodes_time, dt, min_frac=0.05):
    '''Building gauss steps'''
    if nodes_time < 3:
        raise ValueError('nodes_time should be >= 3')
    if dt <= 0:
        raise ValueError('dt must be positive')

    target_t = (nodes_time - 1) * dt

    # order = np.argsort(times)
    times = np.asarray(times)
    weights = np.asarray(weights)

    step_sizes = []
    is_nominal = [] 
    weight_at = {}
    t = 0.0
    node_idx = 0
    n_nodes = len(times)

    def add_weight(node_i):
        idx = len(step_sizes)
        weight_at[idx] = weight_at.get(idx, 0.0) + weights[node_i]

    while t < target_t - DT_DELTA: # while we are not in the end of all segs
        node_t = times[node_idx] if node_idx < n_nodes else target_t
        next_step = min(node_t, t + dt)
        if next_step == node_t and node_idx < n_nodes:
            is_nominal.append(False)
            add_weight(node_idx)
            node_idx += 1
        else:
            is_nominal.append(True)
        step_sizes.append(next_step - t)
        t = next_step

    assert node_idx == n_nodes, (
        "Node not in the interval"
    )
    print(is_nominal == False)
    return step_sizes, weight_at, is_nominal


def uniform_step_schedule(nodes_time, dt, save_every=1):
    if save_every < 1:
        raise ValueError('save_every должно быть положительным')
    step_sizes = [dt] * (nodes_time - 1)
    is_nominal = [True] * (nodes_time - 1)
    weight_at = {i: float(save_every) for i in range(save_every, nodes_time, save_every)}
    return step_sizes, weight_at, is_nominal
