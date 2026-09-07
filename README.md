# RTM compression research

## Структура проекта

```
src/
  solver.py       — форвард и адджоинт распространение
  models.py       — двухслойная, Marmousi и SEG-Y модели скорости
  imaging.py      — условие визуализации кросс-корреляции, метрики качества
reference_rtm.py  — файл, который прогоняет ртм
```

## Установка

```bash
source venv/bin/activate # venv activation
pip install -r requirements.txt
```

## Запуск

```bash
# SEG-Y модель с явным шагом сетки и прореживанием c квадратурой Гаусса
python reference_rtm.py --model segy --vp-file DYBN_VP_1D.sgy --time-sampling gauss --gauss-segments 5  --gauss-points 100  --freq 60 --segy-dx 1 --factor 3 --n-shots 1 --save-every 1

# SEG-Y модель с явным шагом сетки и прореживанием c дефолтным schedule
 python reference_rtm.py --model segy --vp-file DYBN_VP_1D.sgy     --time-sampling uniform  --freq 60 --segy-dx 1 --factor 3--n-shots 1 --save-every 10
```

Для SEG-Y модели шаг сетки автоматически уточняется по критерию
$dx \le \lambda_{\min} / 5$, где $\lambda_{\min} = V_{p,\min} / f$. Если
исходный `--segy-dx` крупнее этого порога, модель интерполируется на более
мелкую равномерную сетку перед запуском RTM.

Результаты сохраняются в `results/reference/<model>/`:
- `rtm_reference.npy` — массив изображения
- `rtm_illum.npy` — освещённость (sum P_fwd²) для постобработки
- `rtm_shots.npz` — позиции источников (i_src, j_src в ячейках) для мьютинга засветки
- `rtm_reference.png` / `rtm_reference_tight.png` — PNG с авто-clip

## Визуализация

Временно удалена. Для простоты структуры проекта новой ветки.


## Параметры

### `reference_rtm.py`

| Флаг | По умолч. | Описание |
|------|-----------|----------|
| `--model` | `twolayer` | `twolayer`, `marmousi` или `segy` |
| `--factor` | `1` | Прореживание Marmousi/SEG-Y: 2 = dx×2 (~8× быстрее), 3 = dx×3 (~27× быстрее) |
| `--freq` | `10` | Доминирующая частота вейвлета Рикера, Гц |
| `--n-shots` | 5 / 20 | Количество источников |
| `--save-every` | авто | Шаг сохранения снэпшотов (авто ≤ 2 ГБ/шот для Marmousi/SEG-Y) |
| `--illum-comp` | выкл | Деление на освещённость для выравнивания амплитуды |
| `--nx`, `--ny` | 500×500 | Размер сетки (только twolayer) |
| `--pts-per-lambda` | — | Авто dx = Vp_min / (freq × N) (только twolayer) |
| `--vp-file` | — | Путь к SEG-Y файлу с моделью Vp (обязателен при `--model segy`) |
| `--segy-dx` | авто | Шаг сетки в м для SEG-Y модели (читается из заголовка файла, если не задан) |
