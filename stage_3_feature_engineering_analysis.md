# Stage 3: Feature Engineering - Детальний аналіз

## 📋 Огляд етапу

**Файл:** `src/pipeline/stages/stage_3_feature_engineering.py` (фасад)  
**Оркестратор:** `src/pipeline/stages/feature_engineering/orchestrator.py`  
**Конфігурація:** `src/config/features.yaml`  
**Призначення:** Створення фіч та таргетів для моделей

---

## 🔧 Архітектура

### Ключові компоненти:

1. **FeatureEngineeringStage (Orchestrator)** - Оркестратор feature engineering
2. **FeatureEnricher** - Збагачення даних фічами
3. **FeatureOrchestrator** - Оркестратор збагачувачів
4. **FeatureGuards** - Безпека фіч
5. **TargetGenerator** - Генерація таргетів
6. **EnhancedSmartSelector** - Вибір фіч
7. **FeatureCache** - Кешування фіч

---

## 🔄 Процес feature engineering

### Крок 1: Валідація та підготовка даних
```python
cleaned_data, market_data_dict = self._validate_and_prepare_market_data(**kwargs)
```
- Витягує cleaned_data з kwargs
- Готує market_data_dict (підтримує DataFrame та dict)
- Перевіряє наявність даних

### Крок 2: Збагачення фічами для кожного таймфрейму
```python
for tf, df in market_data_dict.items():
    enrich_kwargs = {}
    if 'macro_data' in cleaned_data:
        enrich_kwargs['macro_data'] = cleaned_data['macro_data']
    if 'news' in cleaned_data:
        enrich_kwargs['news'] = cleaned_data['news']
    
    enriched_df = self.enricher.enrich_features(df, timeframe=tf, **enrich_kwargs)
```

**Передача даних:**
- ✅ macro_data передається з Stage 1
- ✅ news передається з Stage 1
- ✅ Використовується в збагачувачах

### Крок 3: Генерація таргетів (для 1d)
```python
if tf == '1d':
    targets_df = self.target_gen.generate_targets(enriched_df)
    all_targets[tf] = targets_df
    target_cols = [col for col in targets_df.columns if col.startswith('target_')]
    for col in target_cols:
        enriched_df[col] = targets_df[col].reindex(enriched_df.index)
```

**Таргети з processing.yaml:**
- target_regression_1d (regression)
- target_binary_1d_0_0 (classification_binary)
- target_multiclass_1d (classification_multiclass)
- target_rsi_prediction_3d (indicator_prediction)

### Крок 4: Застосування safety guards
```python
enriched_df = self.guards.apply_guards(enriched_df)
```
- Перевірка на NaN/Inf
- Перевірка діапазонів
- Перевірка типів даних

### Крок 5: Вибір фіч (для 1d)
```python
final_features = enriched_data.get('1d', pd.DataFrame())
selected_features = list(final_features.columns) if not final_features.empty else []
feature_importance: dict[str, float] = {}
if not final_features.empty:
    target_col = kwargs.get('target_column', 'target_up_1d')
    if target_col in final_features.columns:
        selected_features, feature_importance = await self._select_features(
            final_features,
            target_col,
            kwargs,
        )
```

**Вибір фіч:**
- EnhancedSmartSelector
- Аналіз важливості фіч
- Вибір найкращих фіч
- Fallback на всі фіч якщо помилка

### Крок 6: Фіналізація
```python
return {
    'status': 'success',
    'features': final_features,
    'enriched_data': final_features,
    'all_timeframes': enriched_data,
    'enriched_prices': enriched_data,
    'all_targets': all_targets,
    'combined_features': final_features,
    'selected_features': selected_features,
    'feature_importance': feature_importance,
    'timestamp': datetime.now().isoformat()
}
```

---

## 🎯 Збагачувачі (Enrichers)

### Включені збагачувачі (17/17):

1. **time_features** - Часові фічі (day of week, month, etc.)
2. **technical_analysis** - Технічні індикатори (SMA, EMA, RSI, MACD, etc.)
3. **derived_features** - Похідні фічі (price ratios, etc.)
4. **macro_features** - Макроекономічні фічі
5. **keyword_entity** - Ключові слова та сутності
6. **news_quality** - Якість новин
7. **sentiment_features** - Сентимент новин
8. **nlp_features** - NLP фічі
9. **news_impact** - Вплив новин (time-decaying)
10. **hype_features** - Hype/attention
11. **significance_features** - Статистична значущість
12. **decay_features** - Decay функції
13. **advanced_analytics** - Advanced analytics
14. **context_map** - Context fingerprint
15. **market_context** - Macro/market indicators
16. **volatility** ✅ (новий) - Volatility indicators
17. **volume** ✅ (новий) - Volume indicators

### Нові збагачувачі (додані):

#### Volatility Enricher
**Індикатори:**
- volatility_5, volatility_10, volatility_20
- atr_14
- gk_volatility (Garman-Klass)
- volatility_regime (low, normal, high, extreme)

#### Volume Enricher
**Індикатори:**
- volume_sma_5, volume_sma_10
- volume_roc (Rate of Change)
- price_volume_trend
- obv (On-Balance Volume)
- volume_rs (Volume Relative Strength)

---

## 📊 Technical Analysis Enricher

### Конфігурація:
```yaml
technical_analysis:
  sma:
    enabled: true
    windows: [5, 10, 20, 50, 100, 200]
  ema:
    enabled: true
    windows: [5, 10, 20, 50, 100, 200]
  rsi:
    enabled: true
    period: 14
  macd:
    enabled: true
    fast: 12
    slow: 26
    signal: 9
  bollinger_bands:
    enabled: true
    period: 20
    std: 2
  atr:
    enabled: true
    period: 14
  stochastic:
    enabled: true
    k_period: 14
    d_period: 3
  williams_r:
    enabled: true
    period: 14
  cci:
    enabled: true
    period: 20
  market_regime:
    enabled: true
    window: 10
```

### Всього індикаторів: ~50+

---

## 🎯 Target Generator

### Таргети з processing.yaml:

1. **target_regression_1d**
   - Тип: regression
   - Base: close
   - Shift: -1 (на 1 день вперед)

2. **target_binary_1d_0_0**
   - Тип: classification_binary
   - Base: close
   - Shift: -1
   - Threshold: 0.0 (up/down)

3. **target_multiclass_1d**
   - Тип: classification_multiclass
   - Base: close
   - Shift: -1
   - Thresholds: [-0.01, 0.01] (down/sideways/up)

4. **target_rsi_prediction_3d**
   - Тип: indicator_prediction
   - Indicator: RSI_14
   - Shift: -3 (на 3 дні вперед)

---

## 🔒 Feature Guards

### Перевірки:
- **NaN/Inf detection** - Виявлення NaN та Inf значень
- **Range validation** - Перевірка діапазонів
- **Type validation** - Перевірка типів даних
- **Outlier detection** - Виявлення викидів

### Результати:
- Фільтрація некоректних даних
- Логування проблем
- Quality metrics

---

## 🎯 Feature Selection

### EnhancedSmartSelector:
- Аналіз важливості фіч
- Кореляційний аналіз
- Mutual information
- Recursive Feature Elimination
- Вибір найкращих фіч

### Параметри:
- target_column - цільова колонка
- max_features - макс. кількість фіч
- context_id - ідентифікатор контексту
- market_data - ринкові дані

### Fallback:
- Якщо помилка - всі фічі
- Importance = 1.0 / (rank + 1)

---

## 💾 Feature Cache

### Кешування:
- cache_dir: `data/cache/features`
- Зберігає обчислені фічі
- Прискорює повторні обчислення
- Підтримує інвалідацію

---

## 📈 Результати роботи

### Очікувані дані на виході:
1. **features** - Фінальні фічі (1d)
2. **enriched_data** - Збагачені дані (1d)
3. **all_timeframes** - Всі таймфрейми
4. **enriched_prices** - Збагачені ціни
5. **all_targets** - Всі таргети
6. **selected_features** - Вибрані фічі
7. **feature_importance** - Важливість фіч

### Кількість фіч:
- Technical analysis: ~50 індикаторів
- Volatility: 6 індикаторів
- Volume: 6 індикаторів
- Time features: ~10 індикаторів
- NLP features: ~20 індикаторів
- News impact: ~5 індикаторів
- **Всього:** ~100+ фіч

---

## ⚠️ Потенціальні проблеми

### 1. **Тільки 1d для таргетів**
```python
if tf == '1d':
    targets_df = self.target_gen.generate_targets(enriched_df)
```
- Таргети генеруються тільки для 1d
- Інші таймфрейми без таргетів

### 2. **Feature selection тільки для 1d**
```python
final_features = enriched_data.get('1d', pd.DataFrame())
```
- Вибір фіч тільки для 1d
- Інші таймфрейми без вибору

### 3. **Fallback на всі фічі**
```python
if not selected:
    selected = list(candidate_features.columns)
```
- Якщо помилка - всі фічі
- Може включати погані фічі

### 4. **Пряме передання macro_data та news**
```python
if 'macro_data' in cleaned_data:
    enrich_kwargs['macro_data'] = cleaned_data['macro_data']
```
- Не проходить через очищення в Stage 2
- Може містити сирі дані

---

## ✅ Статус Stage 3

**Загальний статус:** ✅ Працює коректно

**Компоненти:**
- ✅ FeatureEngineeringStage - оркеструє feature engineering
- ✅ FeatureEnricher - збагачує фічами
- ✅ FeatureOrchestrator - координує збагачувачі
- ✅ FeatureGuards - перевіряє якість
- ✅ TargetGenerator - генерує таргети
- ✅ EnhancedSmartSelector - вибирає фічі
- ✅ FeatureCache - кешує фічі

**Збагачувачі:** ✅ 17/17 працюють
- ✅ Включені volatility та volume (нові)
- ✅ Всі збагачувачі з features.yaml

**Таргети:** ✅ Працюють
- 4 таргети з processing.yaml
- Генеруються тільки для 1d

**Feature Selection:** ✅ Працює
- EnhancedSmartSelector
- Fallback на всі фічі

**Кешування:** ✅ Працює
- FeatureCache
- Прискорює обчислення

**Рекомендації:**
1. Додати таргети для інших таймфреймів
2. Додати feature selection для інших таймфреймів
3. Покращити fallback логіку
4. Додати очищення для macro_data та news
