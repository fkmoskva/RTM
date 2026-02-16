import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------
# ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
# ------------------------

length = 200          # размер области (м)
c = 340               # скорость звука (м/с)
rho = 1.2             # плотность воздуха (кг/м^3)
nu = 25               # частота источника (Гц)
amplitude = 1

n = 200               # узлы по пространству
n_t = 800             # шаги по времени

dx = length / n

# CFL условие
dt = dx / (c * np.sqrt(2)) * 0.9

print(f"CFL число = {c * dt / dx:.3f}")

t0 = 1.2 / nu

# ------------------------
# МАССИВЫ
# ------------------------

p_prev = np.zeros((n, n))
p = np.zeros((n, n))
p_next = np.zeros((n, n))

i_src = n // 2
j_src = n // 2

# ------------------------
# Ricker wavelet
# ------------------------

def ricker(t):
    arg = (np.pi * nu * (t - t0))**2
    return amplitude * (1 - 2*arg) * np.exp(-arg)

# ------------------------
# СЕТКА
# ------------------------

x = np.linspace(0, length, n)
y = np.linspace(0, length, n)

p_frames = []

print("Симуляция запущена...")

# ------------------------
# ОСНОВНОЙ ЦИКЛ
# ------------------------

for it in range(1, n_t):

    t = it * dt

    source = np.zeros((n, n))
    source[i_src, j_src] = ricker(t) / (dx * dx)

    # ------------------------
    # ЛАПЛАСИАН через np.diff
    # ------------------------

    d2x = np.diff(p, n=2, axis=0) / dx**2
    d2y = np.diff(p, n=2, axis=1) / dx**2

    laplacian = np.zeros_like(p)
    laplacian[1:-1, 1:-1] = d2x[:,1:-1] + d2y[1:-1,:]

    # ------------------------
    # Обновление
    # ------------------------

    p_next = (2*p - p_prev +
              (c**2 * dt**2) * laplacian +
              dt**2 * source / rho)

    # Граничные условия (жёсткие стенки)
    p_next[0, :] = 0
    p_next[-1, :] = 0
    p_next[:, 0] = 0
    p_next[:, -1] = 0

    p_prev, p = p, p_next

    if it % 5 == 0:
        p_frames.append(p.copy())

print(f"Сохранено кадров: {len(p_frames)}")

# ------------------------
# ВИЗУАЛИЗАЦИЯ
# ------------------------

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(p_frames[0],
               extent=[0, length, 0, length],
               origin='lower',
               cmap='viridis',
               animated=True)

plt.colorbar(im, ax=ax, label="Давление")
ax.set_xlabel("X (м)")
ax.set_ylabel("Y (м)")

def update(frame):
    im.set_array(p_frames[frame])
    ax.set_title(f"2D Волна, t = {frame*5*dt:.3f} c")
    return [im]

anim = FuncAnimation(fig,
                     update,
                     frames=len(p_frames),
                     interval=30,
                     blit=True)

plt.tight_layout()
plt.show()
