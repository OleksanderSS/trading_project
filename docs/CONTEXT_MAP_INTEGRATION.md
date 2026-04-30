# 📊 Context Map Integration Guide

## Overview

Context Map - це система кодування ринкового стану в компактний "відбиток" (fingerprint), який допомагає моделям розуміти різні ринкові режими.

## 🏗️ Architecture

### 1. Context Map Generation (Local - Stage 3)

```python
# Для кожного рядка даних:
for row in data:
    # Порівнюємо кожен індикатор з попереднім значенням
    for indicator in indicators:
        change = (current - previous) / previous
        
        if change > threshold:    → state = 1  (зростання)
        elif change < -threshold: → state = -1 (падіння)
        else:                     → state = 0  (без змін)
    
    # Об'єднуємо всі стани в fingerprint
    context_fingerprint = "|".join(all_states)
    # Приклад: "-1|-1|1|0|0|-1|1|..."
```

**Результат:**
- 147 `state_*` колонок (розкладений fingerprint)
- 1 `context_fingerprint` колонка (об'єднаний string)
- 8 temporal features (hour, day_of_week, etc.)

### 2. Feature Selection (Colab - Stage 4)

```python
# Context-Aware Feature Selection
selector = ContextAwareFeatureSelector(method='mutual_info', top_k=50)

for ticker in tickers:
    for target in targets:
        # Вибираємо features (включаючи state_*)
        selected_features, analysis = selector.select_features(
            X=features_df[all_features],  # 253 base + 147 state = 400
            y=targets_df[target]
        )
        
        # Analysis містить:
        # - base_count: кількість базових features
        # - context_count: кількість state_* features
        # - top_context_features: найважливіші context features
        # - context_ratio: % context features серед відібраних
```

### 3. Model Training (Colab - Stage 4)

```python
# Модель тренується на відібраних features
model = train_model(
    X=data[selected_features],  # Включає state_* features
    y=targets_df[target]
)

# Модель САМА вчиться розуміти контекст через state_* features!
# Не потрібно окремих моделей для кожного контексту
```

### 4. Inference (Prediction)

```python
# Для нових даних:
new_data_with_states = calculate_states(new_data)

# Prediction
prediction = model.predict(
    new_data_with_states[selected_features]
)

# Модель автоматично враховує поточний ринковий контекст
```

---

## 📋 Integration Steps

### Step 1: Enable MarketContextEnricher

**File:** `src/config/enrichment.yaml`

```yaml
enrichment:
  market_context:
    module: "src.features.enrichers.market_context_enricher"
    class: "MarketContextEnricher"
    params:
      context_features:
        - "volatility_5d"
        - "volatility_20d"
        - "volume_ratio"
        - "rsi_current"
        # ... etc
```

### Step 2: Fix Target Column Names

**File:** `src/config/targets.yaml`

```yaml
targets:
  target_volume_ratio_f1:
    type: "indicator_prediction"
    params:
      indicator_col: "market_context_volume_ratio"  # ← Виправлено!
      shift: -1
```

### Step 3: Use Context-Aware Feature Selector

**File:** `colab_clean_cell.py`

```python
# Replace ColabFeatureSelector with ContextAwareColabFeatureSelector
from src.features.colab_context_integration import ContextAwareColabFeatureSelector

# In ColabTrainingPipeline.__init__:
self.feature_selector = ContextAwareColabFeatureSelector(self.env.PROJECT_PATH)
```

### Step 4: Save Feature Analysis

```python
# After feature selection:
from src.features.colab_context_integration import save_feature_analysis

save_feature_analysis(
    analysis=analysis,
    ticker=ticker,
    target=target,
    model_type=model_type,
    output_dir=batch_dir / "feature_analysis"
)
```

### Step 5: Visualize Context Importance

```python
# Optional: Create visualization
from src.features.colab_context_integration import visualize_context_importance

visualize_context_importance(
    analysis=analysis,
    output_path=batch_dir / f"context_importance_{ticker}_{target}.png"
)
```

---

## 📊 Expected Results

### Feature Selection Output

```
✅ Selected 50 features for AMD:
   Base: 35, Context: 12, Temporal: 3
   
   Top context features:
      - state_RSI: 0.0234
      - state_MACD: 0.0189
      - state_volume: 0.0156
      - state_ATR: 0.0142
      - state_BB_width: 0.0128
```

### Analysis Metadata

```json
{
  "ticker": "AMD",
  "target": "target_return_1d",
  "model_type": "mlp",
  "base_count": 35,
  "context_count": 12,
  "temporal_count": 3,
  "uses_context": true,
  "context_ratio": 0.24,
  "base_avg_importance": 0.0198,
  "context_avg_importance": 0.0167,
  "top_context_features": [
    {"name": "state_RSI", "importance": 0.0234},
    {"name": "state_MACD", "importance": 0.0189},
    ...
  ]
}
```

---

## 🎯 Benefits

### 1. Simplicity
- ✅ Одна модель на (ticker, target) замість тисяч
- ✅ Простий inference (немає fallback для нових контекстів)
- ✅ Легко підтримувати та масштабувати

### 2. Effectiveness
- ✅ Більше даних для тренування
- ✅ Модель сама вчиться важливості контексту
- ✅ Feature selection автоматично відбирає найважливіші state_* features

### 3. Interpretability
- ✅ Можна побачити, які context features важливі
- ✅ Візуалізація важливості features
- ✅ Аналіз використання контексту

---

## 🔧 Troubleshooting

### Problem: No context features selected

**Причина:** Context features не створені або мають низьку важливість

**Рішення:**
1. Перевірте, чи `ContextMapEnricher` запущений в Stage 3
2. Перевірте, чи є `state_*` колонки в `features.parquet`
3. Збільште `top_k` в feature selector

### Problem: Too many context features

**Причина:** Context features домінують над базовими

**Рішення:**
1. Зменшіть `top_k` або додайте обмеження на context_ratio
2. Використайте `random_forest` method для кращого балансу
3. Додайте регуляризацію в модель

### Problem: Context features not improving performance

**Причина:** Модель не вчиться використовувати контекст

**Рішення:**
1. Збільште складність моделі (більше layers/neurons)
2. Додайте dropout для регуляризації
3. Спробуйте інший тип моделі (LSTM, Transformer)

---

## 📚 References

- **Context Map Enricher:** `src/features/enrichers/context_map_enricher.py`
- **Context-Aware Selector:** `src/features/context_aware_feature_selector.py`
- **Colab Integration:** `src/features/colab_context_integration.py`
- **Configuration:** `src/config/enrichment.yaml`
- **Targets Config:** `src/config/targets.yaml`

---

## 🚀 Next Steps

1. ✅ Run Stage 0-3 to generate context features
2. ✅ Verify `state_*` columns in `features.parquet`
3. ✅ Update Colab cell to use Context-Aware selector
4. ✅ Train models and analyze context feature importance
5. ✅ Visualize and interpret results
