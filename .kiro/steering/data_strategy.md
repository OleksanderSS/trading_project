# 📊 Data Accumulation & Synthetic Generation Strategy

## Overview

Комплексна стратегія для накопичення реальних даних та генерування синтетичних сценаріїв без очікування великих обсягів реальних даних.

---

## 🎯 Goals

1. **Накопичувати реальні дані** - Stages 0-3 (без тренування)
2. **Генерувати синтетичні дані** - 3 типи сценаріїв
3. **Перевіряти структуру** - Event-series формат
4. **Комбінувати** - Реальні + синтетичні для тренування

---

## 🔄 Phase 1: Real Data Accumulation

### Process
```
Stage 0: Setup & Validation
  ↓
Stage 1: Collection (Market + Macro + News)
  ↓
Stage 2: Processing & Cleaning
  ↓
Stage 3: Feature Engineering (15+ enrichers)
  ↓
DuckDB Storage
```

### Key Components

**FeatureOrchestrator** (`src/features/feature_orchestrator.py`)
- Динамично відкриває 15+ збагачувачів
- Запускає їх в пріоритетному порядку
- Повертає збагачений DataFrame

**DataManager** (`src/data/management/data_manager.py`)
- Управління DuckDB базою
- Upsert операції
- Фільтрація дублікатів

**PipelineOrchestrator** (`src/pipeline/pipeline_orchestrator.py`)
- Запускає Stages 0-3
- Управління даними між стадіями

### Output

**DuckDB Tables:**
- `raw_data` - OHLCV + macro + news
- `enriched_features` - 125+ features per row
- `targets` - 16 target variables

### Usage

```bash
python scripts/accumulate_real_data.py \
  --tickers AMD NVDA TSLA \
  --days 60
```

---

## 🎲 Phase 2: Synthetic Data Generation

### Type 1: Typical Scenarios (Monte Carlo)

**Purpose**: Типові ринкові умови

**Engine**: `SimulationEngine.run_monte_carlo_for_strategy()`

**Process**:
1. Завантажити історичні повернення
2. Генерувати 100+ синтетичних ціннісних шляхів
3. Розрахувати метрики (Sharpe, Max Drawdown, VaR)

**Output**: `synthetic_typical_scenarios_*.json`

**Characteristics**:
- Реалістичні розподіли
- Історичні кореляції
- Метрики для оцінки

### Type 2: Shock Scenarios (MonsterTest)

**Purpose**: Стійкість до ринкових шоків

**Engine**: `MonsterTestMode.run()`

**Shock Types**:
- Flash crash (-10%)
- Volatility spike (+50%)
- Liquidity crisis (-5%)
- Black swan (-20%)
- Circuit breaker (-7%)

**Output**: `synthetic_shock_scenarios_*.json`

**Characteristics**:
- Миттєві шоки
- Відновлення
- Вплив на портфель

### Type 3: Context Scenarios (DEAN)

**Purpose**: Різні ринкові режими

**Engine**: `DeanBootstrapSystem.internal_simulation()`

**Market Regimes**:
- Trending up (trend=+0.1%, vol=1.5%)
- Trending down (trend=-0.1%, vol=2.0%)
- Ranging (trend=0%, vol=1.0%)
- Volatile (trend=0%, vol=4.0%)
- Crisis (trend=-0.2%, vol=6.0%)

**Output**: `synthetic_context_scenarios_*.json`

**Characteristics**:
- Actor-Critic bootstrap
- Режимні характеристики
- Симуляційні кроки

### Usage

```bash
# Всі типи
python scripts/generate_synthetic_data.py

# Тільки типові
python scripts/generate_synthetic_data.py --types typical

# Шокові та контекстні
python scripts/generate_synthetic_data.py --types shock context
```

---

## 🔍 Phase 3: Dataset Verification

### Verification Steps

1. **Tables Check** - Наявність таблиць
2. **Raw Data Structure** - Колони, типи, null-значення
3. **Enriched Features** - Кількість збагачених колон
4. **Event-Series Format** - Сортування, дублікати
5. **Data Integrity** - Null%, дублікати, типи
6. **Enricher Coverage** - Наявність усіх збагачувачів

### Event-Series Format

**Requirements**:
- Sorted by timestamp
- No duplicate (ticker, timestamp) pairs
- Each row = one event

**Structure**:
```
ticker | timestamp | open | high | low | close | volume | feature_1 | feature_2 | ...
```

### Usage

```bash
python scripts/verify_enriched_dataset.py \
  --output results/verification_report.json
```

---

## 📋 Workflow

### Step 1: Accumulate Real Data
```bash
python scripts/accumulate_real_data.py --tickers AMD NVDA --days 30
```

### Step 2: Verify Dataset
```bash
python scripts/verify_enriched_dataset.py
```

### Step 3: Generate Synthetic Data
```bash
python scripts/generate_synthetic_data.py --types typical shock context
```

### Step 4: Combine & Train
```bash
# Real data + Synthetic data → Training
python run_hybrid_pipeline.py --mode prepare --test-ticker AMD --test-target target_return_1d
```

---

## 🧠 Module Verification

### FeatureOrchestrator
- ✅ Динамично відкриває enrichers
- ✅ Запускає в пріоритетному порядку
- ✅ Повертає збагачений DataFrame
- ✅ Обробляє помилки

### DataManager
- ✅ Управління DuckDB
- ✅ Upsert операції
- ✅ Фільтрація дублікатів
- ✅ Перевірка таблиць

### SimulationEngine
- ✅ Monte Carlo симуляції
- ✅ Генерування ціннісних шляхів
- ✅ Розрахунок метрик
- ✅ Паралельна обробка

### MonsterTest
- ✅ Генерування шокових сценаріїв
- ✅ Тестування стійкості
- ✅ Розрахунок впливу
- ✅ Аналіз результатів

### DEAN Bootstrap
- ✅ Actor-Critic bootstrap
- ✅ Внутрішня симуляція
- ✅ Режимні сценарії
- ✅ Еволюційна логіка

---

## 📊 Expected Outputs

### Real Data
```
DuckDB Tables:
├── raw_data (2520 rows, 15 columns)
├── enriched_features (2520 rows, 125 columns)
└── targets (2520 rows, 16 columns)
```

### Synthetic Data
```
JSON Files:
├── synthetic_typical_scenarios_*.json (100+ scenarios)
├── synthetic_shock_scenarios_*.json (5 shock types)
└── synthetic_context_scenarios_*.json (5 regimes)
```

### Verification Report
```
JSON File:
└── verification_report.json
    ├── tables
    ├── raw_data
    ├── enriched_features
    ├── event_series_format
    ├── data_integrity
    └── enricher_coverage
```

---

## 🎯 Key Metrics

### Real Data
- Rows: 2520 (30 days × 84 data points/day)
- Columns: 125 (15 base + 110 enriched)
- Features: Technical, Volatility, Momentum, Volume, Context, Sentiment, Macro
- Event-series: Valid (sorted, no duplicates)

### Synthetic Data
- Typical: 100+ Monte Carlo paths
- Shock: 5 shock types
- Context: 5 market regimes
- Total: 110+ synthetic scenarios

---

## 🚀 Next Steps

1. **Run accumulation** - Накопичити реальні дані
2. **Verify dataset** - Перевірити структуру
3. **Generate synthetic** - Створити синтетичні сценарії
4. **Train models** - Тренувати гібридні моделі
5. **Evaluate** - Оцінити на реальних даних

---

## 📚 References

- **Scripts**: `scripts/accumulate_real_data.py`, `scripts/generate_synthetic_data.py`, `scripts/verify_enriched_dataset.py`
- **Guide**: `scripts/DATA_ACCUMULATION_README.md`
- **Strategy**: `scripts/data_accumulation_strategy.md`
- **Modules**: `src/features/`, `src/data/`, `src/simulation/`, `src/models/dean/`
