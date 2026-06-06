# Stage 2: Processing - Детальний аналіз

## 📋 Огляд етапу

**Файл:** `src/pipeline/stages/stage_2_processing.py` (фасад)  
**Оркестратор:** `src/pipeline/stages/processing/orchestrator.py`  
**Конфігурація:** `src/config/processing.yaml`  
**Призначення:** Обробка, очищення та нормалізація даних

---

## 🔧 Архітектура

### Ключові компоненти:

1. **ProcessingStage (Orchestrator)** - Оркестратор обробки даних
2. **ProcessingDataHandler** - Обробка та нормалізація даних
3. **ProcessingStorage** - Збереження оброблених даних
4. **ProcessingValidator** - Валідація якості даних
5. **IntelligentDataFilter** - Інтелектуальна фільтрація
6. **NormalizationManager** - Нормалізація даних
7. **PricePreprocessor** - Попередня обробка цін
8. **DataCleaner** - Очищення даних

---

## 🔄 Процес обробки даних

### Крок 1: Екстракція сирих даних
```python
raw_data = self._extract_raw_data(kwargs)
```
- Витягує raw_data з kwargs
- Підтримує вкладену структуру

### Крок 2: Обробка різних типів даних
```python
self._process_all_data_types(raw_data, cleaned_data_map)
```

**Market Data:**
```python
df_m = self.data_handler.clean_and_normalize_market_data(raw_data['market_data'])
cleaned_data_map['prices'] = self.data_handler.group_by_timeframes(df_m)
```
- Очищення та нормалізація цін
- Групування по таймфреймах

**Macro Data:**
```python
if 'macro_data' in raw_data:
    cleaned_data_map['macro_data'] = raw_data['macro_data']
```
- Пряме передання макро даних з Stage 1

**News Data:**
```python
if 'news' in raw_data:
    cleaned_data_map['news'] = raw_data['news']
```
- Пряме передання новин з Stage 1

### Крок 3: Інтелектуальна фільтрація
```python
filtered_results = self.data_handler.apply_intelligent_filtering(cleaned_data_map)
```
- Застосовує IntelligentDataFilter
- Фільтрує шум в даних
- Зберігає quality_report

### Крок 4: Нормалізація
```python
self.data_handler.apply_normalization(
    filtered_results,
    features_to_normalize=features_to_normalize,
    fit_scalers=run_mode != 'predict',
)
```
- Нормалізація фіч
- Fit scalers в train mode
- Load scalers в predict mode

### Крок 5: Валідація
```python
self.modular_validator.run_system_validation(filtered_results)
```
- Системна валідація
- Створення quality metrics

### Крок 6: Збереження
```python
storage_paths = self.storage_manager.save_cleaned_data_to_files(filtered_results)
```
- Збереження оброблених даних
- Cloud offloading (GCS)

### Крок 7: Фіналізація
```python
result = self._finalize_results(filtered_results, storage_paths)
```
- Створення результату
- Додавання timestamp
- Моніторинг системних ресурсів

---

## 🧹 Очищення даних

### PricePreprocessor.normalize_price_df()
```python
df_m = PricePreprocessor().normalize_price_df(df_m)
```
- Нормалізація колонок цін
- Стандартизація назв колонок

### DataCleaner.remove_outliers_zscore()
```python
df_m = DataCleaner.remove_outliers_zscore(df_m, columns=['close'], threshold=3.0)
```
- Видалення викидів за Z-score
- Threshold: 3.0 (3 стандартні відхилення)

### DataCleaner.handle_missing_values()
```python
df_m = DataCleaner.handle_missing_values(df_m, method='ffill')
```
- Заповнення пропущених значень
- Метод: forward fill (ffill)

---

## 📊 Групування по таймфреймах

### group_by_timeframes()
```python
def group_by_timeframes(self, df_m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if 'interval' not in df_m.columns:
        return {'daily': df_m}  # Default
    
    groups = {}
    for interval, group in df_m.groupby('interval'):
        groups[str(interval)] = group
    return groups
```

**Результат:**
- Якщо є колонка 'interval' - групує по таймфреймах
- Якщо немає - повертає {'daily': df_m}
- Підтримує: 15m, 60m, 1d (з конфігурації)

---

## 🔍 Інтелектуальна фільтрація

### IntelligentDataFilter.filter_quality_data()
```python
filter_result = self.data_filter.filter_quality_data(cleaned_data_map)
```

**Компоненти фільтрації:**
- **PriceFilter** - Фільтрація цін
- **NewsFilter** - Фільтрація новин
- **SocialFilter** - Фільтрація соціальних даних
- **PatternExtractor** - Витягування патернів

**Quality Report:**
- Кількість відфільтрованих записів
- Метрики якості
- Знайдені патерни

---

## 📏 Нормалізація

### NormalizationManager

**Fit Scalers (Train Mode):**
```python
combined = pd.concat(frames, ignore_index=True, sort=False)
self.normalization_manager.fit_scalers(combined, features_to_normalize)
```

**Load Scalers (Predict Mode):**
```python
self.normalization_manager.load_scalers(
    [cfg["feature"] for cfg in features_to_normalize if "feature" in cfg]
)
```

**Transform:**
```python
self._transform_nested_dataframes(filtered_results)
```
- Застосовує scalers до всіх DataFrame
- Підтримує вкладені структури

### Конфігурація нормалізації
```yaml
normalization:
  features:
    - feature: "close"
      method: "standard"
    - feature: "volume"
      method: "minmax"
```

---

## ✅ Валідація

### ProcessingValidator.run_system_validation()
```python
self.modular_validator.run_system_validation(filtered_results)
```

**Перевірки:**
- Наявність обов'язкових колонок
- Типи даних
- Відсутність null значень
- Діапазони значень

**Quality Metrics:**
- Completeness
- Consistency
- Validity
- Uniqueness

---

## 💾 Збереження

### ProcessingStorage.save_cleaned_data_to_files()
```python
storage_paths = self.storage_manager.save_cleaned_data_to_files(filtered_results)
```

**Локації:**
- Local files (`data/processed/`)
- Cloud storage (GCS) - якщо налаштовано
- Scalers (`data/scalers/`)

**Формати:**
- Parquet для даних
- Pickle для scalers

---

## 📈 Конфігурація

### safe_fill
```yaml
safe_fill:
  fill_with_zero:
    - "news_score"
    - "impact_score"
    - "reverse_impact"
    - "daily_sentiment"
    - "match_count"
    - "news_count"
```
- Заповнює нулями відсутні значення

### data_preparation
```yaml
data_preparation:
  test_size: 0.2
  seq_len: 10
  feature_columns:
    - "feature_rsi"
    - "feature_ema_cross"
    - "feature_volatility"
    - "feature_sentiment_score"
    - "feature_news_volume"
```

### targets_config
```yaml
targets_config:
  - name: "target_regression_1d"
    type: "regression"
    params:
      base_col: "close"
      shift: -1
  - name: "target_binary_1d_0_0"
    type: "classification_binary"
    params:
      base_col: "close"
      shift: -1
      threshold: 0.0
  - name: "target_multiclass_1d"
    type: "classification_multiclass"
    params:
      base_col: "close"
      shift: -1
      thresholds: [-0.01, 0.01]
  - name: "target_rsi_prediction_3d"
    type: "indicator_prediction"
    params:
      indicator_col: "RSI_14"
      shift: -3
```

---

## 🎯 Результати роботи

### Очікувані дані на виході:
1. **prices** - Оброблені цінові дані (по таймфреймах)
2. **macro_data** - Макроекономічні дані (прямо з Stage 1)
3. **news** - Новини (прямо з Stage 1)
4. **quality_report** - Звіт про якість даних
5. **patterns** - Знайдені патерни
6. **filtering_summary** - Підсумок фільтрації

### Storage Paths:
- `data/processed/prices/` - Цінові дані
- `data/processed/news/` - Новини
- `data/processed/macro/` - Макро дані
- `data/scalers/` - Scalers

---

## ⚠️ Потенціальні проблеми

### 1. **Пряме передання macro_data та news**
```python
if 'macro_data' in raw_data:
    cleaned_data_map['macro_data'] = raw_data['macro_data']
```
- Не проходить через очищення
- Може містити сирі дані

### 2. **GCS Manager initialization**
```python
try:
    self.gcs_manager = GCSManager()
except Exception as e:
    self.logger.warning(f'GCS Manager initialization failed: {e}')
    self.gcs_manager = None
```
- GCS може бути недоступний
- Продовжує без cloud storage

### 3. **Нормалізація тільки для конфігураційних фіч**
```python
if not features_to_normalize:
    self.logger.info("No normalization features configured; skipping normalization.")
    return
```
- Якщо не налаштовано - пропускається
- Може призвести до проблем з моделями

---

## ✅ Статус Stage 2

**Загальний статус:** ✅ Працює коректно

**Компоненти:**
- ✅ ProcessingStage - оркеструє обробку
- ✅ ProcessingDataHandler - обробляє дані
- ✅ IntelligentDataFilter - фільтрує дані
- ✅ NormalizationManager - нормалізує дані
- ✅ ProcessingValidator - валідує дані
- ✅ ProcessingStorage - зберігає дані

**Очищення:** ✅ Працює
- PricePreprocessor - нормалізація цін
- DataCleaner - видалення викидів
- DataCleaner - заповнення пропущених значень

**Фільтрація:** ✅ Працює
- PriceFilter, NewsFilter, SocialFilter
- PatternExtractor
- Quality Report

**Нормалізація:** ✅ Працює (якщо налаштовано)
- Fit scalers в train mode
- Load scalers in predict mode
- Transform nested dataframes

**Валідація:** ✅ Працює
- Системна валідація
- Quality metrics

**Збереження:** ✅ Працює
- Local files
- Cloud storage (опціонально)

**Рекомендації:**
1. Додати очищення для macro_data та news
2. Налаштувати normalization features
3. Перевірити GCS налаштування
4. Додати більше метрик якості
