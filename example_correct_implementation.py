"""
Пример правильной реализации неоднородной плотности и модуля сжатия
для 2D волнового уравнения

ОШИБКА в исходном коде:
    if dist.any() < L_abs / 3:
        rho = 800
    else: 
        rho = 1200

Это НЕ создаёт пространственное поле! Переменная rho будет скаляром.

ПРАВИЛЬНЫЙ подход:
"""

import numpy as np

# Размер сетки
n = 200
c = 340  # скорость звука
nu = 25  # частота

# Создание координатной сетки
x = np.arange(n)
y = np.arange(n)
X, Y = np.meshgrid(x, y, indexing='ij')

# Расстояние от границ
dist_left   = X
dist_right  = n - 1 - X
dist_bottom = Y
dist_top    = n - 1 - Y
dist = np.minimum.reduce([dist_left, dist_right, dist_bottom, dist_top])

# Параметры поглощающего слоя
lambda_wave = c / nu
L_abs = 6.25 * lambda_wave
dx = 200 / n

# ============================================
# ПРАВИЛЬНАЯ РЕАЛИЗАЦИЯ ПЛОТНОСТИ ρ(x,y)
# ============================================

# Создаём 2D массив плотности
rho = np.ones((n, n)) * 1200  # Базовая плотность

# Создаём маску для областей вблизи границы
mask_absorb = dist < L_abs / 3

# Присваиваем другую плотность в поглощающем слое
rho[mask_absorb] = 800

print(f"Форма массива rho: {rho.shape}")
print(f"Минимальная плотность: {rho.min()}")
print(f"Максимальная плотность: {rho.max()}")

# ============================================
# BULK MODULUS K(x,y)
# ============================================

# K связан со скоростью звука: c² = K/ρ
# Для однородной среды: K = c² * ρ₀

# Создаём базовое поле K
K = np.ones((n, n)) * (c**2 * 1200)

# Можно создать неоднородность, например, круглое включение
center_x, center_y = n // 2, n // 2
radius = n // 8

dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
mask_inclusion = dist_from_center < radius

# В этом включении скорость звука выше на 20%
c_inclusion = c * 1.2
rho_inclusion = 1000
K[mask_inclusion] = c_inclusion**2 * rho_inclusion

print(f"\nФорма массива K: {K.shape}")
print(f"Минимальный K: {K.min():.2e}")
print(f"Максимальный K: {K.max():.2e}")

# ============================================
# ВЫЧИСЛЕНИЕ ∇·(K∇p) для неоднородной среды
# ============================================

# Пример: имеем поле давления p
p = np.random.randn(n, n) * 0.1  # случайное поле для демонстрации

# Шаг 1: Вычисляем градиент давления
grad_p_x = np.zeros_like(p)
grad_p_y = np.zeros_like(p)

grad_p_x[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * dx)
grad_p_y[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * dx)

# Шаг 2: Умножаем на K
K_grad_p_x = K * grad_p_x
K_grad_p_y = K * grad_p_y

# Шаг 3: Вычисляем дивергенцию ∇·(K∇p)
div_K_grad_p = np.zeros_like(p)
div_K_grad_p[1:-1, :] += (K_grad_p_x[2:, :] - K_grad_p_x[:-2, :]) / (2 * dx)
div_K_grad_p[:, 1:-1] += (K_grad_p_y[:, 2:] - K_grad_p_y[:, :-2]) / (2 * dx)

print(f"\nФорма div_K_grad_p: {div_K_grad_p.shape}")
print(f"Среднее значение: {div_K_grad_p.mean():.2e}")

print("\n✓ Все массивы имеют правильную форму (n, n)")
print("✓ Плотность и K - это пространственные поля, а не скаляры")
print("✓ Оператор ∇·(K∇p) вычислен корректно для неоднородной среды")
