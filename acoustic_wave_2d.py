import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------
# ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
# ------------------------

length = 200              # размер области (м)
speed = 340               # скорость звука (м/с)
rho = 1.2                 # плотность воздуха (кг/м^3)
freq = 25                 # частота источника (Гц)
amplitude = 1
nodes_space = 200         # узлы по пространству
nodes_time = 800          # шаги по времени
t0 = 1.2 / freq
dx = length / nodes_space

# CFL условие
dt = dx / (speed * np.sqrt(2)) * 0.33
print(f"CFL  = {speed * dt / dx:.3f}")

# ---- Поглощающий слой 6.25λ ----
lambda_wave = speed / freq
L_abs = 6.25 * lambda_wave
n_abs = int(L_abs / dx)


beta = np.zeros((nodes_space, nodes_space))
x = np.arange(nodes_space)
y = np.arange(nodes_space)
X, Y = np.meshgrid(x, y, indexing='ij')

dist_left   = X
dist_right  = nodes_space - 1 - X
dist_bottom = Y
dist_top    = nodes_space - 1 - Y

dist = np.minimum.reduce([dist_left, dist_right, dist_bottom, dist_top])

mask = dist < n_abs

beta_max = 42.5 * speed / L_abs

beta[mask] = beta_max * ((n_abs - dist[mask]) / n_abs)**2
#----------------------------------


# ------------------------
# СРЕДА: rho(x,y), K(x,y)
# ------------------------

rho = np.ones((nodes_space, nodes_space))
rho[(Y <= 100)] = 1200
rho[(Y > 100)] = 800


K = np.ones((nodes_space, nodes_space)) * (340**2 * 1.2)
K[(X <= 100)] = 300**2 * 1200
K[(X >  100)] = 380**2 * 800


# ------------------------
# МАССИВЫ
# ------------------------

p_prev = np.zeros((nodes_space, nodes_space))
p = np.zeros((nodes_space, nodes_space))
p_next = np.zeros((nodes_space, nodes_space))
i_src = nodes_space // 2
j_src = nodes_space // 2


# ------------------------
# Ricker wavelet
# ------------------------

def ricker(t):
    arg = (np.pi * freq * (t - t0))**2
    return amplitude * (1 - 2*arg) * np.exp(-arg)


# ------------------------
# СЕТКА
# ------------------------

x = np.linspace(0, length, nodes_space)
y = np.linspace(0, length, nodes_space)

p_frames = []
print("Starting simulation...")


# ------------------------
# ОСНОВНОЙ ЦИКЛ
# ------------------------

for it in range(1, nodes_time):

    t = it * dt
    source = np.zeros((nodes_space, nodes_space))
    source[i_src, j_src] = ricker(t) / (dx * dx) * 0.1


    # ------------------------
    # ЛАПЛАСИАН через np.diff
    # ------------------------
    #
    # ------------------------
    # ОПЕРАТОР ∇·(1/ρ ∇p)
    # ------------------------

    inv_rho = 1.0 / rho

    # --- по x ---
    dpdx_plus = p[1:, :] - p[:-1, :]                       # градиент на полуцелых
    inv_rho_x = 0.5 * (inv_rho[1:, :] + inv_rho[:-1, :])   # 1/rho на полуцелых
    flux_x = inv_rho_x * dpdx_plus / dx                    # поток (1/rho * dp/dx)

    div_x = np.zeros_like(p)
    div_x[1:-1, :] = (flux_x[1:, :] - flux_x[:-1, :]) / dx

    # --- по y ---
    dpdy_plus = p[:, 1:] - p[:, :-1]
    inv_rho_y = 0.5 * (inv_rho[:, 1:] + inv_rho[:, :-1])
    flux_y = inv_rho_y * dpdy_plus / dx

    div_y = np.zeros_like(p)
    div_y[:, 1:-1] = (flux_y[:, 1:] - flux_y[:, :-1]) / dx

    # итоговый оператор
    div_term = div_x + div_y


    # ------------------------
    # Обновление (неоднородная среда)
    # ------------------------

    p_next = (2*p - (1 - beta*dt) * p_prev
              + K * dt**2 * div_term        # K(x,y) * ∇·(1/ρ ∇p)
              + dt**2 * source) / (1 + beta*dt)

    if np.any(np.isnan(p_next)) or np.any(np.isinf(p_next)):
        print(f"NaN/Inf на шаге {it}")
        break





    p_prev, p = p, p_next
    if it % 5 == 0:
        p_frames.append(p.copy())

print(f"Saving frames: {len(p_frames)}")


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
