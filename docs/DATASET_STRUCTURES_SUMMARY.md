# 📊 Dataset Structures Summary

## Overview

У проекті є **3 основні датасети**, кожен з різною структурою та призначенням:

---

## 1️⃣ Enriched Dataset (Збагачений датасет)

**Файли:**
- `features.parquet` - Фічі
- `targets.parquet` - Таргети

**Структура:** Event-series (один рядок = одна свічка)

**Призначення:** Основний датасет для тренування моделей

### Features.parquet

**Розмір:** ~45,375 рядків × ~206 колонок

**Структура рядка:**
```
[Metadata: 3] + [Price: 5] + [Technical: 50] + [Derived: 20] + 
[Time: 10] + [Macro: 25] + [Market Context: 18] + [News: 15] + 
[Sentiment: 10] + [Advanced: 15] + [Context Map: 1] + [Other: ~44]
```

**Приклад:**
```python
datetime: 2026-04-26 15:30:00
ticker: AMD
interval: 15m
open: 145.20
close: 145.50
volume: 125000
rsi_14: 58.3
sma_20: 144.80
volatility_5d: 0.025
sentiment_score: 0.65
context_fingerprint: "abc123..."
... (~200 фічей)
```

**Таймфрейми:** 15m, 60m, 1d (всі разом в одному файлі)

**Тікери:** 18 (AMD, NVDA, TSLA, AAPL, GOOGL, MSFT, AMZN, META, NFLX, INTC, QCOM, AVGO, TXN, AMAT, LRCX, KLAC, ASML, MU)

---

### Targets.parquet

**Розмір:** ~45,375 рядків × ~16 колонок

**Структура рядка:**
```
[Metadata: 3] + [Targets: 13]
```

**Таргети (13 шт):**
```python
# Binary Classification (2)
target_direction_1d          # Up/Down
target_significant_move_1d   # Significant/Not

# Multiclass Classification (2)
target_trend_strength_1d     # Strong Up/Weak Up/Neutral/Weak Down/Strong Down
target_volatility_regime_1d  # Low/Medium/High

# Regression (2)
target_return_1d             # Return %
target_volatility_1d         # Volatility

# Indicator Prediction (7)
target_rsi_1d               # RSI через 1 день
target_macd_1d              # MACD через 1 день
target_bb_position_1d       # Bollinger Bands position
target_volume_ratio_1d      # Volume ratio
target_atr_1d               # ATR через 1 день
target_trend_1d             # Trend через 1 день
target_momentum_1d          # Momentum через 1 день
```

**Таймфрейми:** 15m, 60m, 1d (окремі таргети для кожного)

---

## 2️⃣ News Dataset (Новинний датасет)

**Файл:** `news_features.parquet`

**Структура:** Event-series (один рядок = одна новина)

**Призначення:** Передбачення реакції ринку на новини

**Розмір:** ~100-1000 рядків × ~43,236 колонок

**Структура рядка:**
```
[Новина: 6] + [Макро: 30] + [Контекст ДО: 21,600] + [Реакція ПІСЛЯ: 21,600]
```

### Блоки:

**1. Новина (6 колонок):**
```python
news_id: "google_news_20260426_153000"
news_timestamp: 2026-04-26 15:30:00
news_title: "Tesla announces record Q1 deliveries"
news_sentiment: 0.85
news_type: "TSLA"
news_source: "google_news"
```

**2. Макро контекст (30 колонок):**
```python
fed_funds_rate: 5.25
treasury_10y: 4.15
vix: 18.5
fear_greed_index: 65
hour_of_day: 15
...
```

**3. Контекст ДО (21,600 колонок):**
- 18 тікерів × 3 таймфрейми × 2 свічки × ~200 фічей
```python
AMD_15m_before_1_datetime: 2026-04-26 15:15:00
AMD_15m_before_1_close: 145.50
AMD_15m_before_1_rsi_14: 58.3
AMD_15m_before_2_datetime: 2026-04-26 15:00:00
AMD_15m_before_2_close: 145.20
...
NVDA_15m_before_1_datetime: ...
...
```

**4. Реакція ПІСЛЯ (21,600 колонок):**
- 18 тікерів × 3 таймфрейми × 2 свічки × ~200 фічей
```python
AMD_15m_after_1_datetime: 2026-04-26 15:30:00
AMD_15m_after_1_close: 146.20  # +0.70 (+0.48%)
AMD_15m_after_1_volume: 250000  # Обсяг подвоївся
AMD_15m_after_2_datetime: 2026-04-26 15:45:00
AMD_15m_after_2_close: 146.80
...
```

**Фільтрація:**
- Тільки новини з 2+ свічками після публікації
- Кластеризація схожих новин (similarity > 0.85)
- Sentiment confidence > 0.75

---

## 3️⃣ Synthetic Dataset (Синтетичний датасет)

**Файли:**
- `synthetic_typical_scenarios_*.json`
- `synthetic_shock_scenarios_*.json`
- `synthetic_context_scenarios_*.json`

**Структура:** JSON (сценарії)

**Призначення:** Тестування стратегій на різних ринкових умовах

### Типи сценаріїв:

**1. Typical (Monte Carlo):**
- 100+ симуляцій типових ринкових умов
- Історичні розподіли та кореляції
- Метрики: Sharpe, Max Drawdown, VaR

**2. Shock (MonsterTest):**
- 5 типів шоків:
  - Flash crash (-10%)
  - Volatility spike (+50%)
  - Liquidity crisis (-5%)
  - Black swan (-20%)
  - Circuit breaker (-7%)

**3. Context (DEAN Bootstrap):**
- 5 ринкових режимів:
  - Trending up (trend=+0.1%, vol=1.5%)
  - Trending down (trend=-0.1%, vol=2.0%)
  - Ranging (trend=0%, vol=1.0%)
  - Volatile (trend=0%, vol=4.0%)
  - Crisis (trend=-0.2%, vol=6.0%)

---

## 📊 Comparison Table

| Dataset | Rows | Columns | Row = | Purpose |
|---------|------|---------|-------|---------|
| **Enriched Features** | ~45,375 | ~206 | 1 свічка | Основне тренування |
| **Enriched Targets** | ~45,375 | ~16 | 1 свічка | Таргети для моделей |
| **News Dataset** | ~100-1000 | ~43,236 | 1 новина | Передбачення реакції на новини |
| **Synthetic** | ~110+ | varies | 1 сценарій | Тестування стратегій |

---

## 🔄 Data Flow

```
Stage 0: Setup & Validation
    ↓
Stage 1: Collection (Market + Macro + News)
    ↓
Stage 2: Processing & Cleaning
    ↓
Stage 3: Feature Engineering
    ├─→ Enriched Dataset (features.parquet + targets.parquet)
    └─→ News Dataset (news_features.parquet)
    ↓
Stage 4: Feature Selection
    ↓
Stage 5: Model Training
    ├─→ Light Models (XGBoost, LightGBM, CatBoost)
    └─→ Heavy Models (Neural Networks, Transformers)
    ↓
Stage 6: Evaluation & Backtesting
    ├─→ Real Data (Enriched + News)
    └─→ Synthetic Data (Typical + Shock + Context)
```

---

## 🎯 Usage Examples

### 1. Train on Enriched Dataset
```python
# Load features and targets
features_df = pd.read_parquet('data/batches/main_database/features.parquet')
targets_df = pd.read_parquet('data/batches/main_database/targets.parquet')

# Filter by timeframe
features_15m = features_df[features_df['interval'] == '15m']
targets_15m = targets_df[targets_df['interval'] == '15m']

# Train model
model.fit(features_15m, targets_15m['target_return_1d'])
```

### 2. Train on News Dataset
```python
# Load news dataset
news_df = pd.read_parquet('data/batches/main_database/news_features.parquet')

# Extract features (БЛОК 1-3) and targets (БЛОК 4)
X = news_df[[col for col in news_df.columns if 'before' in col or 'news_' in col]]
y = news_df[[col for col in news_df.columns if 'after' in col]]

# Train model
model.fit(X, y)
```

### 3. Test on Synthetic Data
```python
# Load synthetic scenarios
with open('synthetic_shock_scenarios_20260426.json') as f:
    shock_scenarios = json.load(f)

# Run backtest on each scenario
for scenario in shock_scenarios:
    results = backtest_strategy(strategy, scenario)
    print(f"Scenario: {scenario['type']}, Sharpe: {results['sharpe']}")
```

---

## 📚 References

- **Enriched Dataset:** `docs/ENRICHED_DATASET_STRUCTURE.md` (потрібно створити)
- **News Dataset:** `docs/NEWS_DATASET_STRUCTURE.md`
- **Synthetic Data:** `scripts/data_accumulation_strategy.md`
- **Pipeline:** `src/pipeline/stages/stage_3_feature_engineering.py`
- **Code:** `src/features/feature_orchestrator.py`, `src/features/news_dataset_builder.py`
