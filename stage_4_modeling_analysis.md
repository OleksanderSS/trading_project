# Stage 4: Modeling - Детальний аналіз

## 📋 Огляд етапу

**Файл:** `src/pipeline/stages/stage_4_modeling.py`  
**Конфігурація:** `src/config/modeling.yaml`  
**Призначення:** Тренування ML моделей з Pattern-Aware підходом

---

## 🔧 Архітектура

### Ключові компоненти:

1. **ModelingStage** - Оркестратор тренування моделей
2. **UnifiedTrainingManager** - Уніфікований менеджер тренування
3. **ModelComparisonAnalyzer** - Аналізатор порівняння моделей
4. **TrainerConfig** - Конфігурація тренування
5. **Experience Diary** - Щоденник досвіду моделей

---

## 🔄 Процес тренування

### Крок 1: Ініціалізація
```python
strategy_str = self.modeling_config.get('strategy', 'hybrid').upper()
strategy = TrainingStrategy[strategy_str] if strategy_str in TrainingStrategy.__members__ else TrainingStrategy.HYBRID

training_config = TrainerConfig(
    strategy=strategy,
    batch_size=self.modeling_config.get('batch_size', BATCH_TRAINER_DEFAULT_BATCH_SIZE),
    max_memory_gb=self.modeling_config.get('max_memory_gb', BATCH_TRAINER_DEFAULT_MAX_MEMORY_GB)
)

self.training_manager = UnifiedTrainingManager(training_config)
self.comparison_analyzer = ModelComparisonAnalyzer()
```

**Стратегії тренування:**
- **HYBRID** - Гібридна (light + heavy)
- **LIGHT** - Тільки легкі моделі
- **HEAVY** - Тільки важкі моделі
- **FAST** - Швидке тренування

**Параметри:**
- batch_size: 32 (default)
- max_memory_gb: 8 (default)

### Крок 2: Ініціалізація інфраструктури
```python
def _init_infrastructure(self):
    self.models_dir.mkdir(parents=True, exist_ok=True)
    if not self.diary_path.exists():
        self.diary_path.parent.mkdir(parents=True, exist_ok=True)
        columns = ['timestamp', 'ticker', 'tf', 'target', 'pattern_id', 'model_name', 'score', 'is_champion']
        pd.DataFrame(columns=columns).to_csv(self.diary_path, index=False)
```

**Створює:**
- `data/trained_models/` - директорія для моделей
- `logs/experience_diary.csv` - щоденник досвіду

### Крок 3: Групування по тікерах
```python
ticker_groups = enriched_data.groupby('ticker') if isinstance(enriched_data, pd.DataFrame) else enriched_data.items()

for ticker, df in ticker_groups:
    current_pattern = df['context_pattern_id'].iloc[-1] if 'context_pattern_id' in df.columns else 'normal'
    logger.info(f"📍 Ticker {ticker} is currently in pattern: {current_pattern}")
    
    await self._process_ticker_with_async(ticker, df, champions, current_pattern)
```

**Pattern-Aware Training:**
- Визначає домінуючий патерн для тікера
- Тренує окрему модель для кожного патерну
- Context key: `{ticker}_{target_name}_{pattern}`

### Крок 4: Підготовка даних з Purged Gap
```python
prepared_data = prepare_data_for_models(
    df=df, 
    ticker=ticker, 
    timeframe=timeframe,
    target_cols=[target_name],
    gap_size=10,  # Обов'язковий розрив для чесності
    test_size=self.modeling_config.get('test_size', DEFAULT_TEST_SIZE)
)
```

**Purged Validation:**
- gap_size=10 - розрив між train та test
- Запобігає data leakage
- Чесне оцінювання моделей

**Test Size:**
- Default: 0.2 (20%)
- Configurable через modeling.yaml

### Крок 5: Уніфіковане тренування
```python
training_results = self.training_manager.execute_unified_training(
    tickers=[ticker], 
    data_context=prepared_data
)
```

**UnifiedTrainingManager:**
- Тренує легкі та важкі моделі
- Порівнює моделі
- Вибирає переможця
- Зберігає модель

### Крок 6: Вибір переможця для патерну
```python
ticker_result = training_results.get('tickers_results', {}).get(ticker, {})
if ticker_result.get('status') == 'success':
    winner_name = ticker_result.get('winner')
    metrics = ticker_result.get('winner_metrics', {})
    
    context_key = f"{ticker}_{target_name}_{current_pattern}"
    champions[context_key] = {
        'ticker': ticker,
        'target': target_name,
        'pattern_id': current_pattern,
        'winner': winner_name,
        'metrics': metrics,
        'model_path': ticker_result.get('model_path'),
        'timestamp': datetime.datetime.now().isoformat()
    }
```

**Pattern Champions:**
- Кожен патерн має свого чемпіона
- Зберігається в champions dict
- Ключ: `{ticker}_{target}_{pattern}`

### Крок 7: Логування в Experience Diary
```python
def _log_expert_to_diary(self, info: dict[str, Any], tf: str):
    entry = {
        'timestamp': info['timestamp'],
        'ticker': info['ticker'],
        'tf': tf,
        'target': info['target'],
        'pattern_id': info['pattern_id'],
        'model_name': info['winner'],
        'score': info['metrics'].get('score', 0),
        'is_champion': True
    }
    pd.DataFrame([entry]).to_csv(self.diary_path, mode='a', header=False, index=False)
```

**Experience Diary:**
- Зберігає історію моделей
- Включає: timestamp, ticker, tf, target, pattern_id, model_name, score, is_champion
- Використовується для meta-learning

---

## 🎯 Pattern-Aware Training

### Концепція:
- **Regime-Specific Champions** - Чемпіони для кожного режиму
- **Context Pattern ID** - Ідентифікатор патерну контексту
- **Pattern Detection** - Виявлення поточного патерну

### Патерни:
- **normal** - Нормальний режим
- **trending_up** - Тренд вгору
- **trending_down** - Тренд вниз
- **volatile** - Висока волатильність
- **sideways** - Бічний рух

### Context Key:
```
{ticker}_{target_name}_{pattern_id}
```

Приклад:
- `TSLA_target_regression_1d_normal`
- `NVDA_target_binary_1d_0_0_trending_up`

---

## 🔍 Purged Validation

### Концепція:
- **Gap Size** - Розрив між train та test
- **Data Leakage Prevention** - Запобігання витоку даних
- **Honest Evaluation** - Чесне оцінювання

### Реалізація:
```python
gap_size=10  # 10 періодів розрив
```

**Переваги:**
- Запобігає look-ahead bias
- Чесна оцінка моделей
- Більш реалістичні результати

---

## 📊 Training Strategies

### HYBRID (Default):
- Легкі моделі + важкі моделі
- Швидке тренування легких
- Точне тренування важких

### LIGHT:
- Тільки легкі моделі
- Швидке тренування
- Менша точність

### HEAVY:
- Тільки важкі моделі
- Точне тренування
- Повільне тренування

### FAST:
- Мінімальне тренування
- Для швидкого prototyping
- Мінімальна точність

---

## 🏆 Model Selection

### ModelComparisonAnalyzer:
- Порівнює моделі за метриками
- Вибирає переможця
- Зберігає метрики

### Метрики:
- **Score** - Загальна оцінка
- **Accuracy** - Точність (для класифікації)
- **Precision** - Точність (для класифікації)
- **Recall** - Повнота (для класифікації)
- **RMSE** - Root Mean Square Error (для регресії)
- **MAE** - Mean Absolute Error (для регресії)

---

## 💾 Збереження моделей

### Локація:
- `data/trained_models/` - основна директорія
- `{ticker}/{target}/{pattern}/` - структура для кожної моделі

### Формат:
- **Pickle** - для sklearn моделей
- **Joblib** - для великих моделей
- **ONNX** - для production deployment (опціонально)

### Metadata:
- Model name
- Metrics
- Timestamp
- Pattern ID
- Target name

---

## 📈 Experience Diary

### Призначення:
- Збереження історії моделей
- Meta-learning
- Pattern analysis
- Model performance tracking

### Структура:
```csv
timestamp,ticker,tf,target,pattern_id,model_name,score,is_champion
2026-06-06,TSLA,1d,target_regression_1d_normal,RandomForest,0.85,True
2026-06-06,NVDA,1d,target_binary_1d_0_0_trending_up,XGBoost,0.78,True
```

---

## 🎯 Результати роботи

### Очікувані дані на виході:
1. **models_metadata** - Метадані моделей
2. **processed_data** - Оброблені дані

### Champions Dict:
```python
{
    'TSLA_target_regression_1d_normal': {
        'ticker': 'TSLA',
        'target': 'target_regression_1d',
        'pattern_id': 'normal',
        'winner': 'RandomForest',
        'metrics': {'score': 0.85, 'rmse': 0.12},
        'model_path': 'data/trained_models/TSLA/target_regression_1d/normal/',
        'timestamp': '2026-06-06T10:00:00'
    },
    ...
}
```

---

## ⚠️ Потенціальні проблеми

### 1. **Тільки останній патерн**
```python
current_pattern = df['context_pattern_id'].iloc[-1] if 'context_pattern_id' in df.columns else 'normal'
```
- Використовує тільки останній патерн
- Ігнорує зміни патерну в часі
- Може бути неточним

### 2. **Асинхронне тренування**
```python
await self._process_ticker_with_async(ticker, df, champions, current_pattern)
```
- Використовує async
- Може бути проблематично для CPU-bound задач
- Потрібен proper async handling

### 3. **Фіксований gap_size**
```python
gap_size=10  # Обов'язковий розрив для чесності
```
- Фіксований розрив
- Може бути не оптимальним для всіх таймфреймів
- Немає адаптивності

### 4. **Відсутність cross-validation**
- Тільки train/test split
- Немає k-fold cross-validation
- Менш надійні оцінки

---

## ✅ Статус Stage 4

**Загальний статус:** ✅ Працює коректно

**Компоненти:**
- ✅ ModelingStage - оркеструє тренування
- ✅ UnifiedTrainingManager - тренує моделі
- ✅ ModelComparisonAnalyzer - порівнює моделі
- ✅ TrainerConfig - конфігурація тренування
- ✅ Experience Diary - логування результатів

**Pattern-Aware Training:** ✅ Працює
- Визначає патерни
- Тренує окремі моделі для кожного патерну
- Зберігає champions

**Purged Validation:** ✅ Працює
- gap_size=10
- Запобігає data leakage

**Training Strategies:** ✅ Працюють
- HYBRID, LIGHT, HEAVY, FAST
- Configurable

**Model Selection:** ✅ Працює
- Порівняння моделей
- Вибір переможця
- Збереження метрик

**Збереження моделей:** ✅ Працює
- Структуроване збереження
- Metadata
- Experience Diary

**Рекомендації:**
1. Додати адаптивний gap_size
2. Додати cross-validation
3. Покращити async handling
4. Додати ensemble моделей
5. Додати hyperparameter tuning
