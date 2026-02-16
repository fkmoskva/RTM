import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#параметры
length = 1000
nu = 50          
amplituda = 1
ro = int(input("RHO is "))
speed = 1000
timing = 1
n = 128        
n_t = 500     

t_0 = 1.2/nu
k_tight = ro * speed**2
dx = length/n
dt = timing/n_t

#давление
p_prev = np.zeros((n, n))
p = np.zeros((n, n))
p_next = np.zeros((n, n))

i_src = n // 2
j_src = n // 2

def riker(t):
    arg = (np.pi * nu * (t - t_0))**2
    return amplituda * (1 - 2*arg) * np.exp(-arg)

#сетка
x = np.linspace(0, length, n)
y = np.linspace(0, length, n)
X, Y = np.meshgrid(x, y)
p_frames = []

print("Симуляция запущена...")

for it in range(1, n_t+1):
    t = it * dt
    
    func = np.zeros((n, n))
    func[i_src, j_src] = riker(t) / (dx * dx)
    
    for i in range(1, n-1):
        for j in range(1, n-1):
            d2p_dx2 = (p[i+1,j] - 2*p[i,j] + p[i-1,j]) / (dx**2)
            d2p_dy2 = (p[i,j+1] - 2*p[i,j] + p[i,j-1]) / (dx**2)
            
            p_next[i,j] = (2*p[i,j] - p_prev[i,j] + 
                          (nu**2 * dt**2) * (d2p_dx2 + d2p_dy2) + 
                          dt**2 * func[i,j] / ro)

    p_next[0, :] = 0
    p_next[-1, :] = 0
    p_next[:, 0] = 0
    p_next[:, -1] = 0

    p_prev, p, p_next = p.copy(), p_next.copy(), p_prev.copy()
    
    if it % 5 == 0:
        p_frames.append(p.copy())

print(f"Сохранено кадров: {len(p_frames)}")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(p_frames[0], extent=[0, length, 0, length], 
               origin='lower', cmap='viridis', animated=True)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("2D Волновое уравнение (Ricker source)")
plt.colorbar(im, ax=ax, label='Давление P')

def update(frame_idx):
    im.set_array(p_frames[frame_idx])
    t_current = frame_idx * 5 * dt
    ax.set_title(f"2D Волна t = {t_current:.3f} с")
    return [im]

anim = FuncAnimation(fig, update, frames=len(p_frames), interval=25, blit=True, repeat=True)
plt.tight_layout()
plt.show()

