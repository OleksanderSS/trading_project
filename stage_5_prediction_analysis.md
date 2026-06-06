# Stage 5: Prediction - Детальний аналіз

## 📋 Огляд етапу

**Файл:** `src/pipeline/stages/stage_5_prediction.py`  
**Конфігурація:** `src/config/prediction.yaml`  
**Призначення:** Генерація прогнозів з використанням ансамблів та контекстуальних коригувань

---

## 🔧 Архітектура

### Ключові компоненти:

1. **PredictionStage** - Оркестратор прогнозування
2. **PredictionContextManager** - Менеджер контексту прогнозів
3. **PredictionGenerator** - Генератор прогнозів (ансамблі)
4. **AnomalyEngine** - Двигун виявлення аномалій
5. **ModelResolver** - Резолвер шляхів моделей
6. **DataPreparationService** - Сервіс підготовки даних
7. **ModelSelectionService** - Сервіс вибору моделей
8. **ScalerService** - Сервіс скалерів
9. **StackedEnsemble** - Стековий ансамбль
10. **EnsembleCache** - Кеш ансамблів (LRU, maxsize=5000)
11. **ModelPool** - Пул моделей (maxsize=50)
12. **AdaptiveModelSelector / SmartModelSelector** - Селектор моделей
13. **PredictionAdjuster** - Коригувач прогнозів
14. **AnomalyDetector** - Детектор аномалій
15. **CriticalSignalDetector** - Детектор критичних сигналів
16. **MarketContextAnalyzer** - Аналізатор контексту ринку

---

## 🔄 Процес прогнозування

### Крок 1: Ініціалізація
```python
use_adaptive = self.config_manager.get('prediction.use_adaptive_selector', False)
if use_adaptive:
    self.context_selector = AdaptiveModelSelector(fallback='lightgbm', ...)
else:
    self.context_selector = SmartModelSelector()

self.ensemble_cache = get_ensemble_cache(maxsize=5000)
self.model_pool = get_model_pool(max_models=50)
self.model_resolver = ModelResolver(...)
self.anomaly_engine = AnomalyEngine(diary=self.diary)
```

**Селектор моделей:**
- **AdaptiveModelSelector** - З online learning
- **SmartModelSelector** - Залежить від історії

**Кешування:**
- Ensemble Cache - LRU, maxsize=5000
- Model Pool - LRU, maxsize=50

### Крок 2: Підготовка вхідних даних
```python
features_df, models_meta, market_regime = self._prepare_inputs(kwargs)
```

**DataPreparationService:**
- Витягує features_df
- Витягує models_metadata
- Визначає market_regime

### Крок 3: Перевірка наявності локальних моделей
```python
has_local = self.model_resolver.check_local_models(models_meta)
if not has_local:
    batch_dir = self.model_resolver.resolve_batch_directory(models_meta, kwargs)
    if batch_dir and batch_dir.exists():
        has_local = self.model_resolver.update_local_model_paths(models_meta, batch_dir)
```

**ModelResolver:**
- Перевіряє наявність локальних моделей
- Резолвить batch directory
- Оновлює шляхи моделей

### Крок 4: Генерація прогнозів для контекстів
```python
prediction_results = self._generate_predictions_for_contexts(models_meta, features_df, market_regime)
```

**Фільтрація моделей:**
```python
available_model_types = self._get_available_model_types()
for context_id, meta in models_meta.items():
    model_type = meta.get('model_type', '')
    if model_type in available_model_types:
        filtered_models_meta[context_id] = meta
```

**Обробка кожного контексту:**
```python
for context_id, meta in filtered_models_meta.items():
    result = self._process_single_context(context_id, meta, features_df, market_regime)
    if result:
        prediction_results[context_id] = result
```

### Крок 5: Обробка контексту
```python
context_result = self._process_context_data(context_id, meta, features_df)
ticker_df_clean, filtered_features_list = context_result
```

**Pattern-Aware Context:**
```python
current_pattern = ticker_df_clean['context_pattern_id'].iloc[-1] if 'context_pattern_id' in ticker_df_clean.columns else 'normal'
current_pattern_seq = ticker_df_clean['context_pattern_seq'].iloc[-1] if 'context_pattern_seq' in ticker_df_clean.columns else None
current_fingerprint = ticker_df_clean['context_fingerprint'].iloc[-1] if 'context_fingerprint' in ticker_df_clean.columns else current_pattern
champion_state = ticker_df_clean['state_champion'].iloc[-1] if 'state_champion' in ticker_df_clean.columns else 0
context_velocity = ticker_df_clean['context_velocity'].iloc[-1] if 'context_velocity' in ticker_df_clean.columns else 0
```

**Expert Model Search:**
```python
expert_context_id = f"{ticker}_{meta.get('target')}_{current_pattern}"
models = self.model_resolver.load_available_models(expert_context_id, {expert_context_id: meta})

if not models:
    self.logger.info(f"ℹ️ No expert model for pattern {current_pattern}, using general champion")
    models = self.model_resolver.load_available_models(context_id, {context_id: meta})
```

**Pattern-Aware Model Selection:**
- Шукає експертну модель для поточного патерну
- Якщо немає - використовує загального чемпіона
- Context key: `{ticker}_{target}_{pattern}`

### Крок 6: Вибір найкращої моделі
```python
best_model_name = self._select_best_model_for_context(ticker_df_clean, meta, models, ticker, market_regime)
```

**ModelSelectionService:**
- Використовує AdaptiveModelSelector або SmartModelSelector
- Враховує історію (DiaryEngine)
- Враховує контекст

### Крок 7: Генерація прогнозу
```python
raw_prediction, model_contributions = self.prediction_generator.generate_prediction(
    models, best_model_name, ticker_df_clean, filtered_features_list, market_regime, context_id
)
```

**PredictionGenerator:**
- Використовує StackedEnsemble
- Генерує прогноз з ансамблю
- Повертає raw prediction та model contributions

### Крок 8: Champion-Bias Adjustment
```python
confidence_adjustment = 1.0
if champion_state != 0:
    last_raw_pred = raw_prediction[-1] if isinstance(raw_prediction, np.ndarray) else raw_prediction
    pred_sign = np.sign(last_raw_pred)
    if pred_sign != np.sign(champion_state):
        confidence_adjustment = 0.7  # Штраф 30% за суперечність ринку
        self.logger.info(f"⚠️ Contradiction with Champion detected for {ticker}. Penalizing confidence.")
```

**Champion-Bias Adjustment:**
- Перевіряє чи прогноз суперечить чемпіону
- Якщо суперечить - штрафує впевненість на 30%
- Запобігає проти-трендовим прогнозам

### Крок 9: Контекстуальне коригування
```python
adjusted_prediction = self.prediction_generator.adjust_prediction_contextually(
    raw_prediction, best_model_name, market_regime, ticker
)
adjusted_prediction = self.prediction_generator.denormalize_prediction(adjusted_prediction, target_scaler)
```

**PredictionAdjuster:**
- Коригує прогноз на основі контексту
- Denormalize якщо є scaler

### Крок 10: Розрахунок аномалій та впевненості
```python
anomaly_score = self.anomaly_engine.calculate_anomaly_score(request.ticker_df_clean)
confidence_info = self.anomaly_engine.calculate_ensemble_confidence(
    models=request.models, X=request.ticker_df_clean, prediction=request.adjusted_prediction, context_id=request.context_id
)
final_confidence = confidence_info.get('score', 0.5) * anomaly_score
```

**AnomalyEngine:**
- Розраховує аномалію даних
- Розраховує впевненість ансамблю
- Фінальна впевненість = confidence * anomaly_score

### Крок 11: Збереження результатів
```python
result = {
    'ticker': request.ticker,
    'predictions': request.adjusted_prediction,
    'raw_forecast': request.raw_prediction,
    'predictions_by_model': request.model_contributions,
    'selected_primary_model': request.best_model_name,
    'confidence': final_confidence * confidence_adjustment,
    'anomaly_score': anomaly_score,
    'last_price': self._get_last_price(request.ticker_df_clean, request.ticker),
    'timestamp': request.ticker_df_clean.index[-1],
    'context_fingerprint': str(current_fingerprint),
    'context_pattern_id': current_pattern,
    'context_pattern_seq': current_pattern_seq,
    'context_velocity': float(context_velocity)
}
```

### Крок 12: Збереження Stage 5 Results
```python
self._save_stage_5_results(predictions_list, current_prices, prediction_results, models_meta, kwargs)
```

**Збереження:**
- `data/colab/accumulated/{batch_name}/stage_5_results.json`
- Включає: predictions, current_prices, models_metadata, timestamps

---

## 🎯 Pattern-Aware Prediction

### Концепція:
- **Expert Models** - Експертні моделі для кожного патерну
- **Context Pattern ID** - Ідентифікатор патерну контексту
- **Context Fingerprint** - Унікальний відбиток контексту
- **Champion State** - Стан чемпіона (up/down)
- **Context Velocity** - Швидкість зміни контексту

### Expert Model Search:
```python
expert_context_id = f"{ticker}_{target}_{current_pattern}"
```

Приклад:
- `TSLA_target_regression_1d_trending_up` - експерт для тренду вгору
- `NVDA_target_binary_1d_0_0_normal` - експерт для нормального режиму

### Fallback:
- Якщо немає експертної моделі - використовує загального чемпіона
- Забезпечує надійність прогнозування

---

## 🎯 Champion-Bias Adjustment

### Концепція:
- **Champion State** - Поточний стан чемпіона (up/down)
- **Prediction Sign** - Знак прогнозу
- **Contradiction Detection** - Виявлення суперечності

### Логіка:
```python
if champion_state != 0:
    pred_sign = np.sign(last_raw_pred)
    if pred_sign != np.sign(champion_state):
        confidence_adjustment = 0.7  # Штраф 30%
```

### Переваги:
- Запобігає проти-трендовим прогнозам
- Підвищує якість прогнозів
- Враховує поточний стан ринку

---

## 🔍 Anomaly Detection

### AnomalyEngine:
- **Anomaly Score** - Оцінка аномалії даних
- **Ensemble Confidence** - Впевненість ансамблю
- **Final Confidence** - Фінальна впевненість

### Методи:
- Z-score anomaly detection
- Isolation Forest
- Local Outlier Factor (LOF)

### Логіка:
```python
final_confidence = confidence_info.get('score', 0.5) * anomaly_score
if anomaly_score < 0.8:
    self.logger.warning(f'Low anomaly score ({anomaly_score:.2f}) - potential data anomaly!')
```

---

## 🎯 Model Selection

### AdaptiveModelSelector:
- **Online Learning** - Навчання в реальному часі
- **Feedback Loop** - Зворотний зв'язок
- **Leaderboard** - Таблиця лідерів

### SmartModelSelector:
- **Historical Performance** - Історична продуктивність
- **DiaryEngine** - Щоденник досвіду
- **Context Awareness** - Врахування контексту

### Вибір моделі:
```python
best_model_name = self._select_best_model_for_context(ticker_df_clean, meta, models, ticker, market_regime)
```

---

## 💾 Кешування

### Ensemble Cache:
- **Type:** LRU (Least Recently Used)
- **Maxsize:** 5000
- **Purpose:** Кешування результатів ансамблю

### Model Pool:
- **Type:** LRU
- **Maxsize:** 50
- **Purpose:** Пул завантажених моделей

### Переваги:
- Прискорення прогнозування
- Зменшення навантаження на пам'ять
- Ефективне використання ресурсів

---

## 📊 Contextual Adjustment

### PredictionAdjuster:
- **Market Regime** - Режим ринку
- **Volatility** - Волатильність
- **Trend** - Тренд
- **Momentum** - Моментум

### Логіка:
```python
adjusted_prediction = self.prediction_generator.adjust_prediction_contextually(
    raw_prediction, best_model_name, market_regime, ticker
)
```

### Переваги:
- Коригування прогнозів на основі контексту
- Підвищення точності
- Врахування поточного стану ринку

---

## 📈 Результати роботи

### Очікувані дані на виході:
1. **predictions** - Список прогнозів
2. **current_prices** - Поточні ціни
3. **prediction_results** - Детальні результати прогнозів
4. **models_metadata** - Метадані моделей
5. **light_models_count** - Кількість легких моделей
6. **heavy_models_count** - Кількість важких моделей
7. **total_models** - Загальна кількість моделей

### Prediction Result:
```python
{
    'ticker': 'TSLA',
    'predictions': [0.0123, 0.0145, ...],
    'raw_forecast': [0.0123, 0.0145, ...],
    'predictions_by_model': {
        'RandomForest': [0.0110, 0.0130, ...],
        'XGBoost': [0.0135, 0.0160, ...],
        ...
    },
    'selected_primary_model': 'RandomForest',
    'confidence': 0.85,
    'anomaly_score': 0.92,
    'last_price': 185.50,
    'timestamp': '2026-06-06T10:00:00',
    'context_fingerprint': 'trending_up_high_volatility',
    'context_pattern_id': 'trending_up',
    'context_pattern_seq': 5,
    'context_velocity': 0.8
}
```

---

## ⚠️ Потенціальні проблеми

### 1. **Тільки останній патерн**
```python
current_pattern = ticker_df_clean['context_pattern_id'].iloc[-1]
```
- Використовує тільки останній патерн
- Ігнорує зміни патерну в часі
- Може бути неточним

### 2. **Фіксований штраф за суперечність**
```python
confidence_adjustment = 0.7  # Штраф 30%
```
- Фіксований штраф
- Може бути не оптимальним
- Немає адаптивності

### 3. **Anomaly Score поріг**
```python
if anomaly_score < 0.8:
    self.logger.warning(f'Low anomaly score ({anomaly_score:.2f}) - potential data anomaly!')
```
- Фіксований поріг 0.8
- Може бути не оптимальним для всіх даних
- Немає адаптивності

### 4. **Fallback на загального чемпіона**
```python
if not models:
    self.logger.info(f"ℹ️ No expert model for pattern {current_pattern}, using general champion")
    models = self.model_resolver.load_available_models(context_id, {context_id: meta})
```
- Якщо немає експертної моделі - використовує загального чемпіона
- Може бути менш точним для специфічних патернів

---

## ✅ Статус Stage 5

**Загальний статус:** ✅ Працює коректно

**Компоненти:**
- ✅ PredictionStage - оркеструє прогнозування
- ✅ PredictionGenerator - генерує прогнози
- ✅ AnomalyEngine - виявляє аномалії
- ✅ ModelResolver - резолвить моделі
- ✅ DataPreparationService - готує дані
- ✅ ModelSelectionService - вибирає моделі
- ✅ ScalerService - керує скалерами
- ✅ StackedEnsemble - ансамбль моделей
- ✅ EnsembleCache - кешує результати
- ✅ ModelPool - пул моделей
- ✅ AdaptiveModelSelector / SmartModelSelector - селектор моделей
- ✅ PredictionAdjuster - коригує прогнози

**Pattern-Aware Prediction:** ✅ Працює
- Expert models для кожного патерну
- Context fingerprint
- Champion state
- Context velocity

**Champion-Bias Adjustment:** ✅ Працює
- Виявлення суперечності
- Штраф за суперечність
- Підвищення якості прогнозів

**Anomaly Detection:** ✅ Працює
- Anomaly score
- Ensemble confidence
- Final confidence

**Model Selection:** ✅ Працює
- Adaptive selector
- Smart selector
- Context awareness

**Кешування:** ✅ Працює
- Ensemble cache (LRU, 5000)
- Model pool (LRU, 50)

**Contextual Adjustment:** ✅ Працює
- Market regime
- Volatility, trend, momentum
- Prediction adjuster

**Збереження:** ✅ Працює
- Stage 5 results JSON
- Batch directory
- Timestamps

**Рекомендації:**
1. Додати адаптивний штраф за суперечність
2. Додати адаптивний поріг anomaly score
3. Покращити fallback логіку
4. Додати ensemble methods
5. Додати uncertainty quantification
