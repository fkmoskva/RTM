import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------
# ФИЗИЧЕСКИЕ ПАРАМЕТРЫ
# ------------------------

length = 200          # размер области (м)
c = 340               # скорость звука (м/с)
nu = 25               # частота источника (Гц)
amplitude = 10        # амплитуда источника (увеличена для лучшей визуализации)
n = 200               # узлы по пространству
n_t = 800             # шаги по времени

dx = length / n

# CFL условие
dt = dx / (c * np.sqrt(2)) * 0.9

print(f"CFL число = {c * dt / dx:.3f}")

t0 = 1.2 / nu
# ---- Поглощающий слой 2λ ----
lambda_wave = c / nu
L_abs = 6.25 * lambda_wave
n_abs = int(L_abs / dx)

print("Толщина слоя (узлы):", n_abs)

# ------------------------
# СОЗДАНИЕ ПРОСТРАНСТВЕННЫХ ПОЛЕЙ
# ------------------------

# Создание координатной сетки
x = np.arange(n)
y = np.arange(n)
X, Y = np.meshgrid(x, y, indexing='ij')

# Расстояние от каждого узла до ближайшей границы
dist_left   = X
dist_right  = n - 1 - X
dist_bottom = Y
dist_top    = n - 1 - Y

dist = np.minimum.reduce([dist_left, dist_right, dist_bottom, dist_top])

# ------------------------
# НЕОДНОРОДНАЯ ПЛОТНОСТЬ ρ(x,y)
# ------------------------

# Создаём массив плотности для всей области
rho = np.ones((n, n)) * 1200  # По умолчанию высокая плотность

# В области вблизи границы (поглощающий слой) устанавливаем низкую плотность
mask_absorb = dist < L_abs / 3
rho[mask_absorb] = 800

print(f"Плотность: мин={rho.min()}, макс={rho.max()}")

# ------------------------
# BULK MODULUS K(x,y)
# ------------------------

# Bulk modulus связан со скоростью звука: c² = K/ρ
# Можем задать различный K(x,y) для создания неоднородностей

K = np.ones((n, n)) * (c**2 * 1200)  # По умолчанию K = c² * ρ₀

# ВНИМАНИЕ: Неоднородность создаёт отражения!
# Закомментируйте следующий блок для устранения отражений от неоднородности
# 
# # Создадим неоднородность в центре области (например, область с другими свойствами)
# # Круглая область с другим bulk modulus
# center_x, center_y = n // 2, n // 2
# radius = n // 8
# 
# dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
# mask_inhomo = dist_from_center < radius
# K[mask_inhomo] = (c * 1.2)**2 * 1000  # Область с увеличенной скоростью

print(f"Bulk modulus: мин={K.min():.2e}, макс={K.max():.2e}")

# ------------------------
# КОЭФФИЦИЕНТ ПОГЛОЩЕНИЯ β
# ------------------------

beta = np.zeros((n, n))
mask = dist < n_abs
# Увеличиваем коэффициент поглощения для устранения отражений от границ
beta_max = 150 * c / L_abs  # Увеличено с 42.5 до 150
beta[mask] = beta_max * ((n_abs - dist[mask]) / n_abs)**2

# ------------------------
# МАССИВЫ ДЛЯ ДАВЛЕНИЯ
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
# ПРОСТРАНСТВЕННАЯ СЕТКА
# ------------------------

x_coords = np.linspace(0, length, n)
y_coords = np.linspace(0, length, n)

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
    # Вычисление ∇·(K∇p) для неоднородной среды
    # ------------------------
    
    # Градиент давления
    # Используем центральные разности
    grad_p_x = np.zeros_like(p)
    grad_p_y = np.zeros_like(p)
    
    grad_p_x[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * dx)
    grad_p_y[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dx)
    
    # K∇p
    K_grad_p_x = K * grad_p_x
    K_grad_p_y = K * grad_p_y
    
    # Дивергенция ∇·(K∇p)
    div_K_grad_p = np.zeros_like(p)
    div_K_grad_p[1:-1, :] += (K_grad_p_x[2:, :] - K_grad_p_x[:-2, :]) / (2 * dx)
    div_K_grad_p[:, 1:-1] += (K_grad_p_y[:, 2:] - K_grad_p_y[:, :-2]) / (2 * dx)
    
    # ------------------------
    # Обновление волнового уравнения
    # ρ * ∂²p/∂t² = ∇·(K∇p) - β*ρ*∂p/∂t + source
    # ------------------------
    
    # Скорость изменения давления (для демпфирования)
    dp_dt = (p - p_prev) / dt
    
    # Временная производная второго порядка
    d2p_dt2 = div_K_grad_p / rho - beta * dp_dt + source / rho
    
    # Схема leap-frog
    p_next = 2*p - p_prev + dt**2 * d2p_dt2
    
    # Граничные условия (жёсткие стенки)
    # p_next[0, :] = 0
    # p_next[-1, :] = 0
    # p_next[:, 0] = 0
    # p_next[:, -1] = 0

    p_prev, p = p, p_next

    if it % 5 == 0:
        p_frames.append(p.copy())

print(f"Сохранено кадров: {len(p_frames)}")

# ------------------------
# ВИЗУАЛИЗАЦИЯ
# ------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Левый график - волновое поле
# Фиксированные пределы для лучшей визуализации волн
im = ax1.imshow(p_frames[0],
               extent=[0, length, 0, length],
               origin='lower',
               cmap='RdBu_r',  # Красно-сине-белая палитра для волн
               vmin=-1e-7,     # Минимальное значение
               vmax=1e-7,      # Максимальное значение
               animated=True)

plt.colorbar(im, ax=ax1, label="Давление (Па)")
ax1.set_xlabel("X (м)")
ax1.set_ylabel("Y (м)")

# Правый график - структура среды (плотность)
im_rho = ax2.imshow(rho,
                    extent=[0, length, 0, length],
                    origin='lower',
                    cmap='coolwarm')
plt.colorbar(im_rho, ax=ax2, label="Плотность ρ (кг/м³)")
ax2.set_xlabel("X (м)")
ax2.set_ylabel("Y (м)")
ax2.set_title("Распределение плотности")

def update(frame):
    im.set_array(p_frames[frame])
    ax1.set_title(f"2D Волна, t = {frame*5*dt:.3f} с")
    return [im]

anim = FuncAnimation(fig,
                     update,
                     frames=len(p_frames),
                     interval=30,
                     blit=True)

plt.tight_layout()
plt.show()
