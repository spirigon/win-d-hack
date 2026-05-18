# Wind Power Forecasting — АРВЭ 2026 Hackathon

Прогнозирование почасовой выработки электроэнергии ВЭС мощностью 90.09 МВт (26 турбин Vestas V126-3.465) на побережье Азовского моря.

**Пояснительная записка**: [`report/jury_note.md`](report/jury_note.md) | [`report/jury_note.docx`](report/jury_note.docx)

## Результаты

| Задача | Метрика nMAE | Файл |
|--------|-------------|------|
| Q1 2026 (2126 часов) | **7.315%** | `submissions/final/q1_forecast.csv` |
| Day-i 18.05.2026 (24 часа) | — | `submissions/final/dayi_forecast.csv` |

## Установка зависимостей

```bash
# Conda (рекомендуется)
conda env create -f environment.yml
conda activate wind-d

# Или pip
pip install -r requirements.txt
```

Требуется Python 3.11+. GPU не требуется.

## Воспроизведение результатов

### Быстрый инференс (из предобученных весов, ~1 сек)

```bash
python submissions/final/predict_final.py --mode all
```

### Полное обучение с нуля (~20 мин)

```bash
# Поместите исходные данные в data/raw/:
#   - train_dataset.csv
#   - valid_features.csv  
#   - 18.05_test_dataset.csv

# Запуск полного пайплайна (обучение + инференс):
python submissions/final/predict_final.py --mode all --from-scratch
```

Результат будет **идентичен** при каждом запуске (детерминированное обучение).

## Выходные файлы

| Файл | Описание |
|------|----------|
| `submissions/final/q1_forecast.csv` | Прогноз Q1 2026 (2126 строк, 1 столбец) |
| `submissions/final/dayi_forecast.csv` | Прогноз 18.05.2026 (24 строки, 1 столбец) |
| `submissions/archive/v97b.0_cfonly.csv` | Кэш предсказаний базовой модели LightGBM (для быстрого инференса без переобучения) |
| `submissions/18_05_2026_forecast.csv` | Базовый прогноз Day-i (зависимость predict_final.py) |

## Параметры модели

### Гиперпараметры LightGBM (Optuna TPE, 200 trials)

```
num_leaves=86, min_data_in_leaf=14, learning_rate=0.00844
feature_fraction=0.430, bagging_fraction=0.564, bagging_freq=3
lambda_l1=0.253, lambda_l2=0.00971
num_boost_round=5000, early_stopping_rounds=250
objective=regression_l1, deterministic=True
```

Расположение: `src/training/train_v32_era5v2.py` → `LGBM_PARAMS`

### Коэффициенты постобработки

```
# Условная коррекция смещения (из OOF анализа):
hw_bias = -2.056 MW   (12-18 м/с)  × 0.7
co_bias = -12.976 MW  (>18 м/с)    × 0.5
q1_bias = -0.828 MW   (сезонный)   × 0.7

# KNN аналоговый ансамбль:
K=100, percentile=55, blend_weight=0.07

# Временное сглаживание:
Savitzky-Golay window=5, polyorder=2
```

Расположение: `submissions/final/predict_final.py`

## Архитектура решения

### Q1 2026 (основная задача, вес 0.9)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. LightGBM v97b (базовая модель)                                  │
│     - Целевая переменная: capacity factor (CF)                      │
│     - 3 режимных специалиста (0-7, 4-12, 8-25 м/с)                │
│     - 5 seeds × 3 CV-bag folds                                      │
│     - K=90 отобранных признаков (из 342 кандидатов)                 │
│     - Дедупликация байт-идентичных столбцов (ERA5 ≡ ECMWF IFS)     │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. Условная коррекция смещения (из OOF анализа)                    │
│     - Высокий ветер 12-18 м/с: hw_bias × 0.7                       │
│     - Cutout >18 м/с: co_bias × 0.5                                │
│     - Сезонная Q1 коррекция: q1_bias × 0.7                         │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. KNN аналоговый ансамбль (7% вес)                                │
│     - K=100 ближайших соседей по погоде (BallTree, Euclidean)       │
│     - 55-й перцентиль выработки соседей                             │
│     - Корректировка на число активных турбин                         │
│     - Корреляция с LGBM = 0.96 (реальная диверсификация)           │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. Временное сглаживание Савицкого-Голея (w=5, p=2)                │
│     - Удаляет шум независимых почасовых предсказаний                │
│     - Квадратичная аппроксимация через 5-часовое окно               │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
                    q1_forecast.csv
```

### Day-i 18.05.2026 (вес 0.1)

```
Ансамбль: 40% MLP + 20% LGBM_CF + 20% LGBM_MW + 20% TFT
  - MLP: ResNet + MAE loss, обучен на полных данных (32k строк)
  - LGBM: spring-retrained, 3 режимных специалиста × 5 seeds
  - TFT: Temporal Fusion Transformer, прямой 24h прогноз
  - Persistence correction для часов 0-3 (ферма отключена в 23:00)
```

## Feature Engineering (89 из 342 отобранных)

| Группа | Примеры | Кол-во |
|--------|---------|--------|
| Физика ветра | REWS, v_eff, WPD, сдвиг, индекс Хеллмана | ~20 |
| ERA5 100м | Скорость, направление, rolling mean/std, тенденции давления | ~25 |
| NWP ансамбль | GFS, ICON-EU/Global, GEM — спред, консенсус, разногласие | ~20 |
| Кривая мощности | Изотоническая PC по 8 секторам, wake-коррекция | ~8 |
| NASA MERRA2 | Дополнительный реанализ, приземный слой | ~10 |
| Календарные/сезонные | hour/doy sin/cos, is_winter × v_eff | ~6 |

## Детерминированность

- Global seed: `set_global_seed(42)` (`src/utils/seeding.py`)
- LightGBM: `deterministic=True, force_col_wise=True`
- PyTorch: `torch.manual_seed()` + `torch.use_deterministic_algorithms(True)`
- Seed bags: `[42, 123, 456, 789, 2026]`
- Результат **бит-в-бит идентичен** при каждом запуске

## Структура проекта

```
├── data/raw/                  # Исходные данные (НЕ включены)
├── data/external/             # ERA5, NWP (parquet)
├── src/
│   ├── data/                  # Загрузка, схемы, сплиты, outliers
│   ├── features/              # Feature engineering pipeline
│   ├── models/                # LightGBM config
│   ├── training/              # Эксперименты (v32-v133)
│   ├── eval/                  # Метрики (nMAE)
│   ├── inference/             # Submission writer + sanity gates
│   └── utils/                 # Seeding, I/O
├── scripts/                   # Day-i prediction, diagnostics
├── submissions/
│   ├── final/                 # Финальные предсказания + скрипт инференса
│   └── archive/               # Все промежуточные версии (~200 файлов)
├── configs/                   # Hydra YAML (model params)
├── report/                    # Пояснительная записка
├── ARCHITECTURE.md            # Детальная архитектура системы
├── environment.yml            # Conda окружение
└── requirements.txt           # pip зависимости
```

## Прогресс (Q1, LB nMAE)

| Версия | Подход | nMAE |
|--------|--------|------|
| v27.1 | Базовый LGBM + CF/MW blend | 7.605% |
| v94 | +ERA5v2, K=90, clean outliers | 7.432% |
| v97b | +Dedup, GEM/ICON-G features | 7.415% |
| v123.B | +Bias corrections (hw+co+Q1) | 7.352% |
| v128.A | +KNN-q55 analog ensemble 7% | 7.340% |
| **v131** | **+Savitzky-Golay(5,2) smoothing** | **7.315%** |

## Ограничения

- Данные хакатона (`train_dataset.csv`, `valid_features.csv`, `18.05_test_dataset.csv`) в репозиторий **НЕ включены**
- Для воспроизведения необходимо поместить их в `data/raw/`
- ERA5 и NWP данные (`data/external/*.parquet`) необходимы для обучения

## Литература

1. Wang D. et al. (2026) Physics-Constrained Transformer for Wind Power Forecasting. Scientific Reports.
2. Ally S. et al. (2025) Modular deep learning for wind farm power forecasting. Wind Energy Science.
3. Jachuła W. & Wydra M. (2025) Wind power prediction using TFT and NWP. BPASTS.
