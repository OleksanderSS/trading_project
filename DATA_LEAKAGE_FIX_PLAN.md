# План рефакторингу витоків даних (Data Leakage Fix Plan)

## Стратегія
Централізація логіки `rolling/expanding` для автоматичного додавання `shift(1)`.

## Класифікація виправлень
1. **Feature Engineering (Критично):** `src/pipeline/stages/stage_0_data_generation.py`, `src/features/enrichers/*.py`.
2. **Targets/Guards (Критично):** `src/pipeline/guards/temporal_target_guard.py`.
3. **Analytics/Indicators (Важливо):** `src/analytics/calculators/*.py`, `src/features/utils/technical_indicators_lib.py`.
4. **Research/Non-ML Code (Нижчий пріоритет):** Код, який не бере участі в навчанні (наприклад, візуалізація), потребує менш суворих змін.

## Утилітарний підхід
Створити обгортку в `src/utils/data_safety.py`:

```python
def safe_rolling(series, window, ...):
    return series.shift(1).rolling(window=window, ...)
```

## Кроки
1. Створення `src/utils/data_safety.py`.
2. Поступовий перехід на `safe_rolling` у вищезазначених модулях.
3. Валідація за допомогою `audit_script.py` або аналогічних інструментів.
4. Повторний запуск тестів.
