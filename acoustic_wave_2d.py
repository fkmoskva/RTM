"""2D acoustic wave modeling on Marmousi model (finite differences + sponge ABC)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import os

# ------------------------
# GRID & SOURCE PARAMETERS
# ------------------------
nx, ny = 2301, 751
dx = 4.0  # m (Marmousi: 9.2 km x 3 km)
nodes_time = 3000
freq = 15.0
amplitude = 1.0
t0 = 1.2 / freq
i_src, j_src = nx // 2, 200

# ------------------------
# VELOCITY MODEL
# ------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "marmousi", "marmousi_vp.bin")

try:
    Vp = np.fromfile(file_path, dtype=np.float32).reshape((nx, ny))
except FileNotFoundError:
    print(f"{file_path} not found, using homogeneous model 2000 m/s")
    Vp = np.full((nx, ny), 2000.0)

Vp_max, Vp_min = Vp.max(), Vp.min()
dt = 0.5 * dx / (Vp_max * np.sqrt(2))  # CFL = 0.5

# ------------------------
# ABSORBING LAYER (SPONGE)
# ------------------------
L_abs = 6.5 * (Vp_min / freq)
n_abs = int(L_abs / dx)
X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
dist = np.minimum.reduce([X, nx - 1 - X, Y, ny - 1 - Y])
beta = np.where(dist < n_abs, (42.5 * Vp_min / L_abs) * ((n_abs - dist) / n_abs) ** 2, 0)

# ------------------------
# INITIALIZATION
# ------------------------
p, p_prev, p_next = np.zeros((3, nx, ny))
C1 = 1.0 / (1 + beta * dt)
C2 = 1.0 - beta * dt
vp2_dt2_dx2 = (Vp * dt / dx) ** 2


# ------------------------
# NUMBA-OPTIMIZED SOLVER
# ------------------------
@njit
def calc_step(p, p_prev, p_next, C1, C2, vp2_dt2_dx2, nx, ny):
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            lap = p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1] - 4.0 * p[i, j]
            p_next[i, j] = C1[i, j] * (2.0 * p[i, j] - C2[i, j] * p_prev[i, j] + vp2_dt2_dx2[i, j] * lap)
    return p_next


# ------------------------
# TIME-STEPPING LOOP
# ------------------------
p_frames = []
for it in range(1, nodes_time):
    t = it * dt
    p_next = calc_step(p, p_prev, p_next, C1, C2, vp2_dt2_dx2, nx, ny)

    # Ricker wavelet source
    arg = (np.pi * freq * (t - t0)) ** 2
    ricker = amplitude * (1 - 2 * arg) * np.exp(-arg)
    p_next[i_src, j_src] += C1[i_src, j_src] * (dt ** 2 * ricker / (dx ** 2 * 0.01))

    p_prev[:] = p
    p[:] = p_next

    if it % 40 == 0:
        p_frames.append(p.copy())
    if it % 500 == 0:
        print(f"Step {it//500}% completed")

# ------------------------
# VISUALIZATION
# ------------------------
global_max = max(np.abs(f).max() for f in p_frames)
threshold = global_max * 0.02
extent_km = [0, nx * dx / 1000, ny * dx / 1000, 0]

fig, ax = plt.subplots(figsize=(12, 4))
ax.imshow(Vp.T, cmap='gray', extent=extent_km, aspect='auto')

p_masked = np.ma.masked_where(np.abs(p_frames[0].T) < threshold, p_frames[0].T)
im_wave = ax.imshow(p_masked, cmap='RdBu', extent=extent_km, aspect='auto',
                    vmin=-global_max / 20, vmax=global_max / 20, alpha=0.8)
ax.set_xlabel("X (km)")
ax.set_ylabel("Z (km)")
ax.set_title("Marmousi Wave Propagation")


def update(frame):
    data = p_frames[frame].T
    im_wave.set_array(np.ma.masked_where(np.abs(data) < threshold, data))
    return [im_wave]


anim = FuncAnimation(fig, update, frames=len(p_frames), interval=40, blit=True)
plt.tight_layout()
plt.show()