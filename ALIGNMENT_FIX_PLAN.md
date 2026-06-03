# Alignment Validation Fix Plan

## 🎯 Проблема

Alignment validation не працює через відсутність datetime колонки:
```
⚠️ Missing datetime column for 15m. Cannot validate alignment.
⚠️ Missing datetime column for 1d. Cannot validate alignment.
⚠️ Missing datetime column for 60m. Cannot validate alignment.
```

## 🔍 Діагностика

### Порядок викликів (поточний):
1. `_process_single_timeframe()` викликається
2. `_enrich_with_cache()` - enrichment + datetime conversion
3. `_ensure_enriched_datetime()` - додає datetime якщо відсутній
4. Повертається `df_enriched` з datetime
5. **В основному циклі**: `validate_and_align_features_targets()` викликається
6. ❌ Alignment validation каже що datetime відсутній

### Гіпотези:
1. ✅ Datetime конвертується після enrichment (бачимо в логах)
2. ❌ Але alignment validation не бачить datetime
3. 🤔 Можливо datetime втрачається між кроками 4 і 5?

## 🔧 Виправлення застосовані

### Fix 1: Додано перевірку datetime перед alignment
```python
# В основному циклі, перед alignment validation:
if 'datetime' not in df_enriched_tf.columns:
    logger.error(f"❌ Enriched DataFrame for {tf} missing datetime column after processing!")
    continue
```

### Fix 2: Додано debug logging
```python
# В _process_single_timeframe:
logger.debug(f"🔍 Before _ensure_enriched_datetime: datetime={'✅' if has_datetime else '❌'}")
logger.debug(f"🔍 After _ensure_enriched_datetime: datetime={'✅' if has_datetime else '❌'}")
```

## 🧪 Тестування

### Команда:
```bash
python run_hybrid_pipeline.py --mode prepare --tickers AMD
```

### Що перевірити в логах:
1. `🔍 Before _ensure_enriched_datetime` - чи є datetime після enrichment?
2. `🔍 After _ensure_enriched_datetime` - чи є datetime після ensure?
3. `❌ Enriched DataFrame missing datetime` - чи спрацьовує перевірка?
4. `✅ Alignment validated` - чи працює alignment validation?

## 📊 Очікувані результати

### Якщо datetime є після enrichment:
```
✅ Converted DatetimeIndex to datetime column after enrichment for 15m
🔍 Before _ensure_enriched_datetime for 15m: datetime=✅
🔍 After _ensure_enriched_datetime for 15m: datetime=✅
✅ Alignment validated for 15m: N rows
```

### Якщо datetime втрачається:
```
✅ Converted DatetimeIndex to datetime column after enrichment for 15m
🔍 Before _ensure_enriched_datetime for 15m: datetime=❌
⚠️ Enriched DataFrame for 15m missing datetime column. Adding from original.
✅ Copied datetime from original data for 15m
🔍 After _ensure_enriched_datetime for 15m: datetime=✅
✅ Alignment validated for 15m: N rows
```

### Якщо datetime втрачається після ensure:
```
✅ Converted DatetimeIndex to datetime column after enrichment for 15m
🔍 Before _ensure_enriched_datetime for 15m: datetime=✅
🔍 After _ensure_enriched_datetime for 15m: datetime=❌
❌ CRITICAL: datetime column lost for 15m!
❌ Enriched DataFrame for 15m missing datetime column after processing!
```

## 🎯 Можливі причини проблеми

### 1. Feature Cache видаляє datetime
**Перевірка**: Чи datetime в metadata_cols?
```python
metadata_cols = ['_cache_ticker', '_cache_date', '_cache_config_hash']
```
**Результат**: ❌ Datetime НЕ в metadata_cols

### 2. _validate_context_fingerprint змінює DataFrame
**Перевірка**: Чи повертає новий DataFrame?
```python
return df_enriched_tf  # Повертає той самий
```
**Результат**: ✅ Не змінює

### 3. _ensure_enriched_datetime не зберігає datetime
**Перевірка**: Чи повертає DataFrame з datetime?
```python
if 'datetime' not in df_enriched.columns:
    df_enriched['datetime'] = ...  # Додає
return df_enriched  # Повертає
```
**Результат**: ✅ Повинно працювати

### 4. Datetime є, але як індекс, а не колонка
**Перевірка**: Чи перевіряємо індекс?
```python
if 'datetime' not in df.columns:  # Перевіряємо тільки колонки
```
**Результат**: 🤔 Можлива причина!

## 🔧 Додаткове виправлення (якщо потрібно)

### Якщо datetime в індексі:
```python
# В validate_and_align_features_targets:
# Check if datetime is in index
if isinstance(features_df.index, pd.DatetimeIndex):
    features_df = features_df.reset_index()
    if 'index' in features_df.columns:
        features_df = features_df.rename(columns={'index': 'datetime'})

if isinstance(targets_df.index, pd.DatetimeIndex):
    targets_df = targets_df.reset_index()
    if 'index' in targets_df.columns:
        targets_df = targets_df.rename(columns={'index': 'datetime'})
```

---

**Статус**: Виправлення застосовані, очікується тестування
**Дата**: 2026-05-07 11:15
