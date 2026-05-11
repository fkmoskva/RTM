# RTM: сжатие волнового поля

Эксперимент по обратной миграции во времени (RTM): многошотовое акустическое
моделирование, сравнение TT (Tensor Train) и QTT (Quantized Tensor Train)
сжатия форвард-снэпшотов относительно эталонного изображения без сжатия.

## Структура проекта

```
src/
  solver.py       — форвард и адджоинт распространение
  models.py       — двухслойная и Marmousi модели скорости
  compression.py  — TT/QTT сжатие/распаковка, подсчёт памяти
  imaging.py      — условие визуализации кросс-корреляции, метрики качества
  plotting.py     — вспомогательные функции matplotlib
test/
  test_compression.py  — тесты TT/QTT
  test_solver.py       — тесты форвард/адджоинт распространения
marmousi/
  marmousi_vp.bin — модель скорости Marmousi II (2301×751, dx=4 м, не в git — скачать отдельно)
main.py           — эксперимент по сжатию: перебор ε, таблица качества
reference_rtm.py  — эталонный multi-shot RTM без сжатия
plot_rtm.py       — постобработка и визуализация сохранённых .npy файлов
```

## Установка

```bash
pip install -r requirements.txt
```

## Эталонный RTM

```bash
# Двухслойная модель, быстрая проверка (~30 с)
python reference_rtm.py --nx 600 --ny 300 --freq 10 --n-shots 15 --save-every 5 --yes

# Marmousi, factor=2 (dx=8 м, ~15 мин)
python reference_rtm.py --model marmousi --factor 2 --freq 15 --n-shots 64 --save-every 15 --yes

# Marmousi, полное разрешение (dx=4 м, ~3 ч)
python reference_rtm.py --model marmousi --n-shots 64 --save-every 30 --yes
```

Результаты сохраняются в `results/reference/<model>/`:
- `rtm_reference.npy` — массив изображения
- `rtm_illum.npy` — освещённость (sum P_fwd²) для постобработки
- `rtm_reference.png` / `rtm_reference_tight.png` — PNG с авто-clip

## Визуализация

Перерисовать сохранённый `.npy` без пересчёта:

```bash
python plot_rtm.py results/reference/marmousi/rtm_reference.npy --dx 8 --freq 15
python plot_rtm.py ...npy --dx 8 --freq 15 --illum              # компенсация освещённости
python plot_rtm.py ...npy --dx 8 --freq 15 --agc 30             # AGC (выравнивание амплитуды)
python plot_rtm.py ...npy --dx 8 --freq 15 --illum --agc 30 --clip-pct 97
```

## Эксперимент по сжатию

```bash
python main.py                      # двухслойная модель
python main.py --model marmousi     # Marmousi
python main.py --model both         # обе модели
```

## Тесты

```bash
python -m pytest test/ -v
```

## Параметры

### `reference_rtm.py`

| Флаг | По умолч. | Описание |
|------|-----------|----------|
| `--model` | `twolayer` | `twolayer` или `marmousi` |
| `--factor` | `1` | Прореживание Marmousi: 2 = dx 8 м (~8× быстрее), 3 = dx 12 м (~27× быстрее) |
| `--freq` | `10` | Доминирующая частота вейвлета Рикера, Гц |
| `--n-shots` | 5 / 20 | Количество источников |
| `--save-every` | авто | Шаг сохранения снэпшотов (авто ≤ 2 ГБ/шот для Marmousi) |
| `--illum-comp` | выкл | Деление на освещённость для выравнивания амплитуды |
| `--nx`, `--ny` | 500×500 | Размер сетки (только twolayer) |
| `--pts-per-lambda` | — | Авто dx = Vp_min / (freq × N) (только twolayer) |

### `plot_rtm.py`

| Флаг | По умолч. | Описание |
|------|-----------|----------|
| `--dx` | `20` | Шаг сетки в м (должен совпадать с симуляцией) |
| `--freq` | `10` | Частота симуляции (для расчёта absorb_cells) |
| `--illum` | выкл | Загрузить `rtm_illum.npy` и применить компенсацию освещённости |
| `--agc` | `0` | Окно AGC в ячейках (0 = выкл). Рекомендуется 20–50 |
| `--lap` | выкл | Лапласиан-фильтр (подчёркивает рефлекторы) |
| `--clip-pct` | `99.5` | Перцентиль для авто-clip физической области |
| `--clip` | — | Явный clip-уровень |
