# 📊 Data Accumulation & Synthetic Generation Guide

## Overview

Три скрипти для управління даними в проекті:

1. **`accumulate_real_data.py`** - Накопичення реальних даних (без тренування)
2. **`generate_synthetic_data.py`** - Генерування синтетичних сценаріїв
3. **`verify_enriched_dataset.py`** - Перевірка структури збагаченого датасету

---

## 🔄 Phase 1: Real Data Accumulation

### Purpose
Накопичити реальні дані з ринку, збагатити їх та зберегти в DuckDB **БЕЗ тренування моделей**.

### Usage

```bash
# Базовий запуск (AMD, NVDA за останні 30 днів)
python scripts/accumulate_real_data.py

# З параметрами
python scripts/accumulate_real_data.py \
  --tickers AMD NVDA TSLA \
  --days 60 \
  --config-path src/config
```

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
  ↓
Verification
```

### Output

**DuckDB Tables:**
```
raw_data
├── ticker, timestamp, open, high, low, close, volume
├── macro indicators (FRED)
└── news sentiment

enriched_features
├── All raw_data columns
├── Technical indicators (RSI, MACD, Bollinger Bands, ATR, EMA, SMA)
├── Volatility metrics (std_dev, range, volatility)
├── Momentum indicators (momentum, ROC, Stochastic)
├── Volume indicators (OBV, A/D)
├── Context map (regime, market_phase)
├── Sentiment features (news, social)
└── Macro features (FRED indicators)

targets (optional)
├── target_return_15m, target_return_1h, target_return_1d
├── target_direction_15m, target_direction_1h, target_direction_1d
└── ... (16 targets total)
```

### Expected Output

```json
{
  "status": "success",
  "tickers": ["AMD", "NVDA"],
  "days_back": 30,
  "collected_rows": 2520,
  "enriched_rows": 2520,
  "enriched_features": 125,
  "verification": {
    "tables": ["raw_data", "enriched_features"],
    "raw_data_rows": 2520,
    "enriched_features_rows": 2520,
    "has_event_series_format": true
  }
}
```

---

## 🎲 Phase 2: Synthetic Data Generation

### Purpose
Генерувати синтетичні дані для тренування моделей **без очікування накопичення реальних даних**.

### Usage

```bash
# Генерувати всі типи сценаріїв
python scripts/generate_synthetic_data.py

# Генерувати тільки типові сценарії
python scripts/generate_synthetic_data.py --types typical

# Генерувати шокові та контекстні сценарії
python scripts/generate_synthetic_data.py --types shock context
```

### Type 1: Typical Scenarios (Monte Carlo)

**Purpose:** Типові ринкові умови для тренування

**Engine:** `SimulationEngine.run_monte_carlo_for_strategy()`

**Output:** `results/synthetic_typical_scenarios_YYYYMMDD_HHMMSS.json`

```json
{
  "scenario_0000": {
    "scenario_id": "typical_0000",
    "type": "monte_carlo",
    "price_path": [100.0, 100.5, 101.2, ...],
    "returns": [0.005, 0.007, ...],
    "metrics": {
      "total_return": 0.15,
      "sharpe_ratio": 1.2,
      "max_drawdown": -0.08,
      "volatility": 0.18,
      "final_price": 115.0
    }
  },
  ...
}
```

**Characteristics:**
- 100+ симуляцій
- Історичні повернення як основа
- Реалістичні розподіли
- Метрики: Sharpe, Max Drawdown, Volatility

### Type 2: Shock Scenarios (MonsterTest)

**Purpose:** Стійкість до ринкових шоків

**Engine:** `MonsterTestMode.run()`

**Output:** `results/synthetic_shock_scenarios_YYYYMMDD_HHMMSS.json`

```json
{
  "shock_flash_crash": {
    "scenario_id": "shock_flash_crash",
    "type": "shock",
    "shock_type": "flash_crash",
    "shock_magnitude": -0.10,
    "shock_duration": 1,
    "price_path": [100.0, 90.0, 92.5, ...],
    "metrics": {
      "shock_impact": -0.10,
      "recovery_time": 15,
      "total_return": -0.05,
      "max_drawdown": -0.15
    }
  },
  ...
}
```

**Shock Types:**
- `flash_crash` - Миттєве падіння (-10%)
- `volatility_spike` - Скачок волатильності (+50%)
- `liquidity_crisis` - Криза ліквідності (-5%)
- `black_swan` - Чорний лебідь (-20%)
- `circuit_breaker` - Автоматична зупинка (-7%)

### Type 3: Context Scenarios (DEAN)

**Purpose:** Різні ринкові режими

**Engine:** `DeanBootstrapSystem.internal_simulation()`

**Output:** `results/synthetic_context_scenarios_YYYYMMDD_HHMMSS.json`

```json
{
  "context_trending_up": {
    "scenario_id": "context_trending_up",
    "type": "context",
    "regime": "trending_up",
    "regime_description": "Восходящий тренд з низькою волатильністю",
    "regime_characteristics": {
      "trend": 0.001,
      "volatility": 0.015
    },
    "price_path": [100.0, 100.1, 100.2, ...],
    "dean_simulation": {
      "actor_actions": [
        {"action_type": "buy", "confidence": 0.7},
        {"action_type": "hold", "confidence": 0.8}
      ],
      "critic_feedback": [
        {"critique_score": 0.5, "points": ["Good entry", "Adequate risk"]}
      ],
      "simulation_steps": 5
    }
  },
  ...
}
```

**Market Regimes:**
- `trending_up` - Восходящий тренд (trend=+0.1%, vol=1.5%)
- `trending_down` - Нисходящий тренд (trend=-0.1%, vol=2.0%)
- `ranging` - Бічний рух (trend=0%, vol=1.0%)
- `volatile` - Висока волатильність (trend=0%, vol=4.0%)
- `crisis` - Паніка (trend=-0.2%, vol=6.0%)

---

## 🔍 Phase 3: Dataset Verification

### Purpose
Перевірити структуру збагаченого датасету та event-series формат.

### Usage

```bash
# Базовий запуск
python scripts/verify_enriched_dataset.py

# З параметрами
python scripts/verify_enriched_dataset.py \
  --config-path src/config \
  --output results/verification_report.json
```

### Verification Steps

1. **Tables Check** - Наявність таблиць в DuckDB
2. **Raw Data Structure** - Колони, типи, null-значення
3. **Enriched Features Structure** - Кількість збагачених колон
4. **Event-Series Format** - Сортування, дублікати, унікальність
5. **Data Integrity** - Null%, дублікати, типи даних
6. **Enricher Coverage** - Наявність усіх збагачувачів

### Output

**File:** `results/verification_report.json`

```json
{
  "timestamp": "2026-04-10T13:40:48.123456",
  "status": "verified",
  "verification_report": {
    "tables": {
      "count": 2,
      "names": ["raw_data", "enriched_features"],
      "has_raw_data": true,
      "has_enriched_features": true
    },
    "raw_data": {
      "rows": 2520,
      "columns": 15,
      "has_expected_columns": true,
      "null_percentage": 0.5
    },
    "enriched_features": {
      "rows": 2520,
      "columns": 125,
      "enriched_columns": 110,
      "memory_usage_mb": 45.2
    },
    "event_series_format": {
      "has_timestamp": true,
      "has_ticker": true,
      "is_sorted_by_timestamp": true,
      "has_duplicate_events": false,
      "is_valid_event_series": true
    },
    "enricher_coverage": {
      "technical_indicators": {"found": true, "column_count": 25},
      "volatility": {"found": true, "column_count": 8},
      "momentum": {"found": true, "column_count": 6},
      "volume": {"found": true, "column_count": 5},
      "context_map": {"found": true, "column_count": 12},
      "sentiment": {"found": true, "column_count": 8},
      "macro": {"found": true, "column_count": 15}
    }
  },
  "recommendations": []
}
```

---

## 📋 Workflow Example

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

## 🎯 Key Concepts

### Event-Series Format
```
ticker | timestamp           | open  | high  | low   | close | volume | feature_1 | feature_2 | ...
-------|---------------------|-------|-------|-------|-------|--------|-----------|-----------|-----
AMD    | 2026-04-10 09:30:00 | 100.0 | 101.5 | 99.5  | 101.0 | 1000000| 0.75      | 0.45      | ...
AMD    | 2026-04-10 10:00:00 | 101.0 | 102.0 | 100.8 | 101.5 | 950000 | 0.78      | 0.48      | ...
NVDA   | 2026-04-10 09:30:00 | 200.0 | 202.0 | 199.5 | 201.5 | 2000000| 0.82      | 0.52      | ...
```

**Requirements:**
- Sorted by timestamp
- No duplicate (ticker, timestamp) pairs
- Each row = one event (one time point for one ticker)

### Enrichers (15+)
- Technical: RSI, MACD, Bollinger Bands, ATR, EMA, SMA
- Volatility: Std Dev, Range, Volatility
- Momentum: Momentum, ROC, Stochastic
- Volume: OBV, A/D
- Context: Regime, Market Phase
- Sentiment: News, Social
- Macro: FRED indicators

---

## 🚀 Next Steps

1. **Run accumulation** - Накопичити реальні дані
2. **Verify dataset** - Перевірити структуру
3. **Generate synthetic** - Створити синтетичні сценарії
4. **Train models** - Тренувати гібридні моделі
5. **Evaluate** - Оцінити на реальних даних

---

## 📞 Troubleshooting

### No data collected
- Check tickers in config
- Verify API credentials
- Check date range

### Event-series format invalid
- Ensure timestamp column exists
- Sort by timestamp
- Remove duplicate (ticker, timestamp) pairs

### Missing enrichers
- Check enricher config in `src/config/features.yaml`
- Verify enricher is enabled
- Check for errors in enricher logs

### DuckDB connection issues
- Check database path in config
- Verify write permissions
- Check disk space

---

## 📚 References

- **FeatureOrchestrator**: `src/features/feature_orchestrator.py`
- **DataManager**: `src/data/management/data_manager.py`
- **SimulationEngine**: `src/simulation/simulation_engine.py`
- **MonsterTest**: `src/main/modes/monster_test.py`
- **DEAN**: `src/models/dean/dean_bootstrap_system.py`
- **Config**: `src/config/features.yaml`, `src/config/simulation.yaml`
