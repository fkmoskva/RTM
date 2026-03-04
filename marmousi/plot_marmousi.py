import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def _roughness_score(arr: np.ndarray) -> float:
    dx = np.mean(np.abs(np.diff(arr, axis=1)))
    dz = np.mean(np.abs(np.diff(arr, axis=0)))
    p1 = float(np.percentile(arr, 1))
    p99 = float(np.percentile(arr, 99))
    penalty = 0.0
    if p1 < 1000:
        penalty += (1000 - p1) * 5.0
    if p99 > 7000:
        penalty += (p99 - 7000) * 5.0
    return dx + dz + penalty


def read_marmousi_binary(path: Path, nx: int = 2301, nz: int = 751) -> tuple[np.ndarray, str]:
    """Загружает бинарный файл Marmousi, автоопределяя byte-order и раскладку."""
    raw = np.fromfile(path, dtype=np.uint8)
    expected = nx * nz
    if raw.size != expected * 4:
        raise ValueError(
            f"Размер файла {raw.size} байт не совпадает с ожиданием {expected * 4} байт "
            f"(nx={nx}, nz={nz}, float32)"
        )

    candidates: list[tuple[np.ndarray, str]] = []
    for dtype_name, dtype in [("little-endian", np.dtype("<f4")), ("big-endian", np.dtype(">f4"))]:
        data = raw.view(dtype)
        candidates.append((data.reshape((nz, nx)), f"{dtype_name}, reshape(nz, nx)"))
        candidates.append((data.reshape((nx, nz)).T, f"{dtype_name}, reshape(nx, nz).T"))

    scored = [(_roughness_score(arr), arr, desc) for arr, desc in candidates]
    scored.sort(key=lambda item: item[0])
    _, best_arr, best_desc = scored[0]
    return best_arr, best_desc


def main():
    """Загружает и отрисовывает модель."""
    parser = argparse.ArgumentParser(description="Отрисовка Marmousi модели")
    parser.add_argument("--bin", type=str, default="marmousi_vp.bin", help="Файл с моделью")
    parser.add_argument("--nx", type=int, default=2301, help="Узлов по X")
    parser.add_argument("--nz", type=int, default=751, help="Узлов по Z (глубина)")
    parser.add_argument("--dx", type=float, default=4.0, help="Шаг сетки X, м")
    parser.add_argument("--dz", type=float, default=4.0, help="Шаг сетки Z, м")
    args = parser.parse_args()

    # Путь к файлу
    bin_path = Path(args.bin)
    if not bin_path.exists():
        bin_path = Path(__file__).parent / args.bin
    
    if not bin_path.exists():
        print(f"ОШИБКА: файл не найден {bin_path}")
        return

    print(f"Загружаю {bin_path}...")
    vp, decode_info = read_marmousi_binary(bin_path, nx=args.nx, nz=args.nz)
    print(f"Форма данных: {vp.shape}")
    print(f"Vp диапазон: [{vp.min():.0f}, {vp.max():.0f}] м/с")
    print(f"Vp среднее: {vp.mean():.0f} м/с")
    print(f"Выбранный вариант чтения: {decode_info}")

    # Координаты в км
    x_km = np.arange(args.nx + 1) * args.dx / 1000.0
    z_km = np.arange(args.nz + 1) * args.dz / 1000.0

    # Отрисовка
    fig, ax = plt.subplots(figsize=(14, 5))
    
    vmin = float(np.percentile(vp, 1))
    vmax = float(np.percentile(vp, 99))

    im = ax.imshow(
        vp,
        cmap="jet",
        aspect="auto",
        extent=[0, x_km[-1], z_km[-1], 0],  # left, right, bottom, top
        vmin=vmin,
        vmax=vmax,
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Vp, м/с", fontsize=11)

    ax.set_xlabel("Distance X, км", fontsize=11)
    ax.set_ylabel("Depth Z, км", fontsize=11)
    ax.set_title("Marmousi Velocity Model", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
