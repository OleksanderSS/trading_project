# 🔴 ПЛАН ВИПРАВЛЕННЯ КРИТИЧНИХ ПРОБЛЕМ

**Дата**: 16 квітня 2026  
**Статус**: Готово до реалізації  
**Пріоритет**: КРИТИЧНИЙ  
**Час**: 1-2 дні

---

## 📋 СТАТУС КРИТИЧНИХ ПРОБЛЕМ

### ✅ ВЖЕ ВИПРАВЛЕНО (3/10)

1. ✅ **Batch Name Generation** - Подвійні префікси
   - Файл: `run_hybrid_pipeline.py`, лінія ~200
   - Виправлення: `.replace('target_target_', 'target_')`
   - Статус: ГОТОВО

2. ✅ **Y-Scaling** - Неправильне масштабування
   - Файл: `colab_clean_cell.py`, лінія 555-646
   - Виправлення: Time-based split + Y scaler
   - Статус: ГОТОВО

3. ✅ **Confidence Score** - Часто = 0
   - Файл: `src/pipeline/stages/stage_5_prediction.py`
   - Виправлення: 4 компоненти з вагами
   - Статус: ГОТОВО

---

### ⏳ ПОТРЕБУЮТЬ ВИПРАВЛЕННЯ (7/10)

#### 4. ❌ Картезіанський добуток при Merge

**Файл**: `src/pipeline/stages/stage_3_feature_engineering.py`  
**Проблема**: Merge без явного ключа

```python
# ❌ НЕПРАВИЛЬНО
merged = features_df.merge(targets_df)  # Без ключа!
# Результат: 2520 * 2520 = 6,350,400 рядків!
```

**Рішення**:
```python
# ✅ ПРАВИЛЬНО
merged = features_df.merge(
    targets_df, 
    on=['ticker', 'datetime'], 
    how='inner'
)
# Результат: 2520 рядків (правильно)
```

**Час**: 30 хвилин  
**Складність**: Низька

---

#### 5. ❌ Жорстко закодовані шляхи

**Файл**: `src/config/paths.yaml`  
**Проблема**: Шляхи не адаптуються до ОС

```yaml
# ❌ НЕПРАВИЛЬНО
paths:
  root: /home/user/1/data  # Linux тільки!
```

**Рішення**:
```python
# ✅ ПРАВИЛЬНО
from pathlib import Path
root = Path.home() / "trading_data"  # Адаптується до ОС
```

**Час**: 1 година  
**Складність**: Середня

---

#### 6. ❌ Залежність від Google Drive

**Файл**: `src/pipeline/hybrid_orchestrator.py`, лінія ~1200  
**Проблема**: Без Google Drive Colab не працює

**Рішення**: Додати fallback на S3 або GCS

```python
# ✅ ПРАВИЛЬНО
def _init_storage():
    try:
        self._init_gdrive()
    except Exception as e:
        logger.warning(f"Google Drive failed: {e}, trying S3...")
        self._init_s3()
```

**Час**: 2 години  
**Складність**: Висока

---

#### 7. ❌ Відсутність перевірки цілісності даних

**Файл**: `src/data/management/data_manager.py`  
**Проблема**: Немає перевірки на дублікати після upsert

**Рішення**: Додати перевірку на дублікати

```python
# ✅ ПРАВИЛЬНО
def upsert(self, table_name: str, df: pd.DataFrame):
    # Перевіряємо дублікати
    duplicates = df.duplicated(subset=['ticker', 'datetime'])
    if duplicates.any():
        logger.warning(f"Found {duplicates.sum()} duplicates, removing...")
        df = df.drop_duplicates(subset=['ticker', 'datetime'], keep='last')
    
    # Вставляємо дані
    self.con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
```

**Час**: 1 година  
**Складність**: Середня

---

#### 8. ❌ Ненадійна хеш-порівняння

**Файл**: `src/data/management/data_manager.py`  
**Проблема**: Хеш-порівняння case-sensitive

**Рішення**: Нормалізувати дані перед хешуванням

```python
# ✅ ПРАВИЛЬНО
def _normalize_for_hash(row):
    # Нормалізуємо регістр та пробіли
    normalized = str(row).lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()
```

**Час**: 30 хвилин  
**Складність**: Низька

---

#### 9. ❌ Глобальна змінна в ConfigManager

**Файл**: `src/config/unified_config_manager.py`  
**Проблема**: Не thread-safe

**Рішення**: Використовувати thread-safe singleton

```python
# ✅ ПРАВИЛЬНО
import threading

class UnifiedConfigManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

**Час**: 1 година  
**Складність**: Середня

---

#### 10. ❌ Відсутність валідації аргументів

**Файл**: `run_hybrid_pipeline.py`, лінія ~70  
**Проблема**: Немає перевірки на коректність аргументів

**Рішення**: Додати валідацію аргументів

```python
# ✅ ПРАВИЛЬНО
def validate_args(args):
    # Перевіряємо тікери
    if args.test_ticker:
        valid_tickers = config_manager.get_config('assets.presets.*.tickers')
        if args.test_ticker not in valid_tickers:
            raise ValueError(f"Invalid ticker: {args.test_ticker}")
    
    # Перевіряємо таргети
    if args.test_target:
        valid_targets = config_manager.get_config('targets')
        if args.test_target not in valid_targets:
            raise ValueError(f"Invalid target: {args.test_target}")
    
    # Перевіряємо моделі
    if args.test_model:
        valid_models = config_manager.get_config('models.available')
        if args.test_model not in valid_models:
            raise ValueError(f"Invalid model: {args.test_model}")
```

**Час**: 1 година  
**Складність**: Середня

---

## 📊 РЕЗЮМЕ

| # | Проблема | Статус | Час | Складність |
|---|----------|--------|-----|-----------|
| 1 | Batch Name | ✅ | - | - |
| 2 | Y-Scaling | ✅ | - | - |
| 3 | Confidence | ✅ | - | - |
| 4 | Merge | ⏳ | 30 хв | Низька |
| 5 | Шляхи | ⏳ | 1 год | Середня |
| 6 | Google Drive | ⏳ | 2 год | Висока |
| 7 | Цілісність | ⏳ | 1 год | Середня |
| 8 | Хеш | ⏳ | 30 хв | Низька |
| 9 | ConfigManager | ⏳ | 1 год | Середня |
| 10 | Валідація | ⏳ | 1 год | Середня |

**Всього часу**: ~8.5 годин  
**Всього складності**: Середня

---

## 🚀 ПОРЯДОК ВИПРАВЛЕННЯ

### День 1 (4 години):
1. Merge (30 хв)
2. Хеш (30 хв)
3. Цілісність (1 год)
4. Валідація (1 год)
5. ConfigManager (1 год)

### День 2 (4.5 години):
1. Шляхи (1 год)
2. Google Drive (2 год)
3. Тестування (1.5 год)

---

**Статус**: Готово до реалізації  
**Складність**: Середня  
**Час**: 1-2 дні

