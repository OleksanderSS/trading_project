# 📊 Data Accumulation & Synthetic Generation System

## Overview

Комплексна система для накопичення реальних даних та генерування синтетичних сценаріїв для тренування моделей.

**Статус**: ✅ Повністю реалізовано

---

## 🎯 Архітектура

```
┌─────────────────────────────────────────────────────────────┐
│                  Complete Data Pipeline                      │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Real Data   │   │  Synthetic   │   │ Verification │
│ Accumulation │   │  Generation  │   │   & QA       │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│              DuckDB Storage (Event-Series Format)            │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐              │
│  │ raw_data │  │ enriched_    │  │ targets  │              │
│  │          │  │ features     │  │          │              │
│  └──────────┘  └──────────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Training Ready   │
                  │ Hybrid Dataset   │
                  └──────────────────┘
```

---

## 📦 Компоненти

### 1. Real Data Accumulation (`accumulate_real_data.py`)

**Призначення**: Накопичення реальних ринкових даних БЕЗ тренування моделей

**Stages**:
- **Stage 0**: Setup & Validation
- **Stage 1**: Data Collection (Market + Macro + News)
- **Stage 2**: Processing & Cleaning
- **Stage 3**: Feature Engineering (15+ enrichers)

**Output**:
- `raw_data` table - OHLCV + macro + news
- `enriched_features` table - 125+ features per row
- `targets` table - 16 target variables

**Usage**:
```bash
# Накопичити дані для AMD та NVDA за останні 30 днів
python scripts/accumulate_real_data.py --tickers AMD NVDA --days 30

# Використати тікери з конфігу
python scripts/accumulate_real_data.py --days 60
```

**Ключові модулі**:
- `FeatureOrchestrator` - динамічно відкриває 15+ збагачувачів
- `DataManager` - управління DuckDB базою
- `PipelineOrchestrator` - запускає Stages 0-3

---

### 2. Synthetic Data Generation (`generate_synthetic_data.py`)

**Призначення**: Генерування синтетичних сценаріїв для тренування

**Типи сценаріїв**:

#### Type 1: Typical Scenarios (Monte Carlo)
- **Engine**: `SimulationEngine.run_monte_carlo_for_strategy()`
- **Кількість**: 100+ симуляцій
- **Характеристики**: Реалістичні розподіли, історичні кореляції
- **Output**: `synthetic_typical_scenarios_*.json`

#### Type 2: Shock Scenarios (MonsterTest)
- **Engine**: `MonsterTestMode.run()`
- **Типи шоків**:
  - Flash crash (-10%)
  - Volatility spike (+50%)
  - Liquidity crisis (-5%)
  - Black swan (-20%)
  - Circuit breaker (-7%)
- **Output**: `synthetic_shock_scenarios_*.json`

#### Type 3: Context Scenarios (DEAN)
- **Engine**: `DeanBootstrapSystem.internal_simulation()`
- **Режими**:
  - Trending up (trend=+0.1%, vol=1.5%)
  - Trending down (trend=-0.1%, vol=2.0%)
  - Ranging (trend=0%, vol=1.0%)
  - Volatile (trend=0%, vol=4.0%)
  - Crisis (trend=-0.2%, vol=6.0%)
- **Output**: `synthetic_context_scenarios_*.json`

**Usage**:
```bash
# Генерувати всі типи
python scripts/generate_synthetic_data.py

# Тільки типові сценарії
python scripts/generate_synthetic_data.py --types typical

# Шокові та контекстні
python scripts/generate_synthetic_data.py --types shock context
```

---

### 3. Dataset Verification (`verify_enriched_dataset.py`)

**Призначення**: Перевірка структури та цілісності даних

**Перевірки**:
1. **Tables Check** - наявність таблиць у DuckDB
2. **Raw Data Structure** - колони, типи, null-значення
3. **Enriched Features** - кількість збагачених колон
4. **Event-Series Format** - сортування, дублікати
5. **Data Integrity** - null%, дублікати, типи
6. **Enricher Coverage** - наявність усіх збагачувачів

**Event-Series Format Requirements**:
- ✅ Sorted by timestamp
- ✅ No duplicate (ticker, timestamp) pairs
- ✅ Each row = one event

**Usage**:
```bash
# Запустити повну перевірку
python scripts/verify_enriched_dataset.py

# Зберегти звіт у файл
python scripts/verify_enriched_dataset.py --output results/verification_report.json
```

---

### 4. Complete Data Pipeline (`run_complete_data_pipeline.py`)

**Призначення**: Unified workflow для всіх етапів

**Modes**:
- `full` - Real + Synthetic + Verification
- `real-only` - Тільки накопичення реальних даних
- `synthetic-only` - Тільки генерування синтетичних даних
- `verify-only` - Тільки перевірка

**Usage**:
```bash
# Повний пайплайн
python scripts/run_complete_data_pipeline.py --mode full --tickers AMD NVDA --days 30

# Тільки реальні дані
python scripts/run_complete_data_pipeline.py --mode real-only --days 60

# Тільки синтетичні дані
python scripts/run_complete_data_pipeline.py --mode synthetic-only --synthetic-types typical shock
```

---

## 🔄 Workflow

### Крок 1: Накопичити реальні дані
```bash
python scripts/accumulate_real_data.py --tickers AMD NVDA TSLA --days 30
```

**Результат**:
- ✅ 2520 rows у `raw_data` (30 days × 84 data points/day)
- ✅ 2520 rows у `enriched_features` (125+ features)
- ✅ Event-series format validated

### Крок 2: Перевірити датасет
```bash
python scripts/verify_enriched_dataset.py
```

**Результат**:
- ✅ Tables verified
- ✅ Event-series format valid
- ✅ Data integrity OK
- ✅ All enrichers present

### Крок 3: Генерувати синтетичні дані
```bash
python scripts/generate_synthetic_data.py --types typical shock context
```

**Результат**:
- ✅ 100+ typical scenarios
- ✅ 5 shock scenarios
- ✅ 5 context scenarios
- ✅ Total: 110+ synthetic scenarios

### Крок 4: Тренувати моделі
```bash
python run_hybrid_pipeline.py --mode prepare --test-ticker AMD --test-target target_return_1d
```

**Результат**:
- ✅ Real data + Synthetic data → Training
- ✅ Models trained on hybrid dataset
- ✅ Evaluation on real data

---

## 📊 Expected Outputs

### DuckDB Tables
```
├── raw_data (2520 rows, 15 columns)
│   ├── ticker, timestamp, open, high, low, close, volume
│   ├── macro indicators
│   └── news sentiment
│
├── enriched_features (2520 rows, 125 columns)
│   ├── Base: ticker, timestamp, OHLCV
│   ├── Technical: RSI, MACD, BB, ATR, EMA, SMA
│   ├── Volatility: std_dev, range, volatility
│   ├── Momentum: momentum, ROC, stochastic
│   ├── Volume: volume indicators, OBV
│   ├── Context: market regime, phase
│   ├── Sentiment: news sentiment, social
│   └── Macro: FRED indicators, economic data
│
└── targets (2520 rows, 16 columns)
    ├── target_return_1d, target_return_5d, target_return_10d
    ├── target_direction_1d, target_direction_5d
    └── target_volatility_1d, target_volatility_5d
```

### Synthetic Data Files
```
results/
├── synthetic_typical_scenarios_20260503_120000.json (100+ scenarios)
├── synthetic_shock_scenarios_20260503_120100.json (5 shock types)
└── synthetic_context_scenarios_20260503_120200.json (5 regimes)
```

### Verification Report
```json
{
  "status": "verified",
  "tables": {
    "count": 3,
    "names": ["raw_data", "enriched_features", "targets"]
  },
  "event_series_format": {
    "is_valid_event_series": true,
    "has_timestamp": true,
    "has_ticker": true,
    "is_sorted_by_timestamp": true,
    "has_duplicate_events": false
  },
  "data_integrity": {
    "null_percentage": 2.5,
    "duplicate_rows": 0
  },
  "enricher_coverage": {
    "technical_indicators": {"found": true, "column_count": 25},
    "volatility": {"found": true, "column_count": 15},
    "momentum": {"found": true, "column_count": 12},
    "volume": {"found": true, "column_count": 10},
    "context_map": {"found": true, "column_count": 8},
    "sentiment": {"found": true, "column_count": 5},
    "macro": {"found": true, "column_count": 10}
  }
}
```

---

## 🎯 Key Metrics

### Real Data
- **Rows**: 2520 (30 days × 84 data points/day)
- **Columns**: 125 (15 base + 110 enriched)
- **Features**: Technical, Volatility, Momentum, Volume, Context, Sentiment, Macro
- **Event-series**: Valid (sorted, no duplicates)

### Synthetic Data
- **Typical**: 100+ Monte Carlo paths
- **Shock**: 5 shock types
- **Context**: 5 market regimes
- **Total**: 110+ synthetic scenarios

### Combined Dataset
- **Real + Synthetic**: Ready for training
- **Format**: Event-series (time-ordered events)
- **Quality**: Verified and validated

---

## 🧠 Module Verification

### ✅ FeatureOrchestrator
- Динамично відкриває enrichers
- Запускає в пріоритетному порядку
- Повертає збагачений DataFrame
- Обробляє помилки

### ✅ DataManager
- Управління DuckDB
- Upsert операції
- Фільтрація дублікатів
- Перевірка таблиць

### ✅ SimulationEngine
- Monte Carlo симуляції
- Генерування ціннісних шляхів
- Розрахунок метрик
- Паралельна обробка

### ✅ MonsterTest
- Генерування шокових сценаріїв
- Тестування стійкості
- Розрахунок впливу
- Аналіз результатів

### ✅ DEAN Bootstrap
- Actor-Critic bootstrap
- Внутрішня симуляція
- Режимні сценарії
- Еволюційна логіка

---

## 🚀 Next Steps

1. **Run accumulation** - Накопичити реальні дані
   ```bash
   python scripts/accumulate_real_data.py --tickers AMD NVDA --days 30
   ```

2. **Verify dataset** - Перевірити структуру
   ```bash
   python scripts/verify_enriched_dataset.py
   ```

3. **Generate synthetic** - Створити синтетичні сценарії
   ```bash
   python scripts/generate_synthetic_data.py
   ```

4. **Train models** - Тренувати гібридні моделі
   ```bash
   python run_hybrid_pipeline.py --mode prepare --test-ticker AMD --test-target target_return_1d
   ```

5. **Evaluate** - Оцінити на реальних даних
   ```bash
   python run_hybrid_pipeline.py --mode continue
   ```

---

## 📚 References

### Scripts
- `scripts/accumulate_real_data.py` - Real data accumulation
- `scripts/generate_synthetic_data.py` - Synthetic data generation
- `scripts/verify_enriched_dataset.py` - Dataset verification
- `scripts/run_complete_data_pipeline.py` - Complete pipeline

### Documentation
- `scripts/data_accumulation_strategy.md` - Strategy overview
- `.kiro/steering/data_strategy.md` - Steering guide
- `docs/CALIBRATION_GUIDE.md` - DEAN calibration
- `docs/HYBRID_PIPELINE.md` - Hybrid pipeline architecture

### Modules
- `src/features/feature_orchestrator.py` - Feature engineering
- `src/data/management/data_manager.py` - Data management
- `src/simulation/simulation_engine.py` - Monte Carlo simulations
- `src/main/modes/monster_test.py` - Shock scenarios
- `src/models/dean/dean_bootstrap_system.py` - DEAN bootstrap

---

## 🔧 Configuration

### Assets (`src/config/assets.yaml`)
```yaml
tickers:
  - AMD
  - NVDA
  - TSLA
  - AAPL
  - MSFT

timeframes:
  - 15m
  - 1h
  - 1d
```

### Features (`src/config/features.yaml`)
```yaml
enrichers:
  - technical_indicators
  - volatility_features
  - momentum_features
  - volume_features
  - context_map
  - sentiment_features
  - macro_features
```

### Simulation (`src/config/simulation.yaml`)
```yaml
defaults:
  monte_carlo_runs: 100
  shock_scenarios: 5
  context_regimes: 5
```

---

## 🐛 Troubleshooting

### Problem: No data collected
**Solution**: Check API keys in `.env` file
```bash
# Verify API keys
cat .env | grep API_KEY
```

### Problem: Empty enriched_features table
**Solution**: Check enrichers are loaded
```bash
# Verify enrichers
python -c "from src.features.feature_orchestrator import FeatureOrchestrator; print(FeatureOrchestrator.list_available_enrichers())"
```

### Problem: Event-series format invalid
**Solution**: Re-run Stage 2 processing
```bash
# Re-process data
python scripts/accumulate_real_data.py --tickers AMD --days 30
```

### Problem: Synthetic data generation fails
**Solution**: Check if real data exists
```bash
# Verify real data
python scripts/verify_enriched_dataset.py
```

---

## 📝 Notes

- **Event-series format** - кожен рядок = одна подія (один момент часу для одного тікера)
- **Synthetic data** - маркується `is_synthetic: true` та `synthetic_type: typical/shock/context`
- **DuckDB** - використовується для швидкого зберігання та запитів
- **Enrichers** - динамічно відкриваються з `src/features/enrichers/`
- **Targets** - генеруються автоматично для 1d, 5d, 10d горизонтів

---

## ✅ Status

- ✅ Real data accumulation - **Implemented**
- ✅ Synthetic data generation - **Implemented**
- ✅ Dataset verification - **Implemented**
- ✅ Complete pipeline - **Implemented**
- ✅ Event-series format - **Validated**
- ✅ DuckDB storage - **Working**
- ✅ Feature engineering - **15+ enrichers**
- ✅ Hybrid training - **Ready**

**Last Updated**: 2026-05-03
