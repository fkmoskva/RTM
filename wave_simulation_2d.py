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

# Вертикальный раздел сред во внутреннем квадрате (где beta = 0)
# Определяем внутренний квадрат (область без поглощения)
inner_region_mask = dist >= n_abs

# Вертикальная линия посередине: левая половина имеет одну плотность, правая - другую
middle_x = n // 2

# В правой половине внутреннего квадрата устанавливаем другую плотность
right_half_mask = inner_region_mask & (X >= middle_x)
rho[right_half_mask] = 1000  # Меньшая плотность справа

print(f"Плотность: мин={rho.min()}, макс={rho.max()}")
print(f"Внутренний квадрат: dist >= {n_abs} узлов")
print(f"Вертикальный раздел на X = {middle_x} ({middle_x * dx:.1f} м)")

# ------------------------
# BULK MODULUS K(x,y)
# ------------------------

# Bulk modulus связан со скоростью звука: c² = K/ρ
# Можем задать различный K(x,y) для создания неоднородностей

# Базовая скорость звука для левой половины
c_left = c  # 340 м/с

# Создаём поле K для всей области
K = np.ones((n, n)) * (c_left**2 * 1200)

# Для правой половины внутреннего квадрата устанавливаем другую скорость звука
c_right = c * 1.3  # Увеличенная скорость справа (442 м/с)

# Обновляем K в правой половине
# K = c² * ρ, где ρ уже установлена как 1000 для правой половины
K[right_half_mask] = c_right**2 * 1000

print(f"Bulk modulus: мин={K.min():.2e}, макс={K.max():.2e}")
print(f"Скорость звука слева: {c_left} м/с, справа: {c_right} м/с")

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Левый график - волновое поле
# Фиксированные пределы для лучшей визуализации волн
# Примечание: При amplitude=10, максимальное давление ~1e-7
# Если изменяете amplitude, измените vmin/vmax пропорционально
im = ax1.imshow(p_frames[0],
               extent=[0, length, 0, length],
               origin='lower',
               cmap='RdBu_r',  # Красно-сине-белая палитра для волн
               vmin=-1e-7,     # Минимальное значение
               vmax=1e-7,      # Максимальное значение
               animated=True)

plt.colorbar(im, ax=ax1, label="Давление (Па)", fraction=0.046)
ax1.set_xlabel("X (м)", fontsize=11)
ax1.set_ylabel("Y (м)", fontsize=11)

# Добавляем вертикальную линию показывающую раздел сред
ax1.axvline(x=length/2, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label='Раздел сред')
ax1.legend(loc='upper right', fontsize=9)

# Правый график - структура среды (скорость звука)
# Вычисляем локальную скорость звука: c(x,y) = sqrt(K/ρ)
c_field = np.sqrt(K / rho)

im_c = ax2.imshow(c_field,
                  extent=[0, length, 0, length],
                  origin='lower',
                  cmap='coolwarm')
plt.colorbar(im_c, ax=ax2, label="Скорость звука c (м/с)", fraction=0.046)
ax2.set_xlabel("X (м)", fontsize=11)
ax2.set_ylabel("Y (м)", fontsize=11)
ax2.set_title("Структура среды (раздел посередине)", fontsize=12)

# Добавляем вертикальную линию на правом графике
ax2.axvline(x=length/2, color='black', linestyle='-', linewidth=2, label='Раздел сред')
ax2.legend(loc='upper right', fontsize=9)

# Добавляем текстовые аннотации
ax2.text(length/4, length*0.95, f'c={c_left} м/с\nρ=1200 кг/м³', 
         ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.text(3*length/4, length*0.95, f'c={c_right:.0f} м/с\nρ=1000 кг/м³', 
         ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
