# 📊 Data Accumulation & Synthetic Generation Strategy

## Overview
Стратегія накопичення реальних даних та генерування синтетичних сценаріїв для тренування моделей.

---

## 🔄 Phase 1: Real Data Accumulation (WITHOUT Training)

### Goal
Накопичити реальні дані з ринку, збагатити їх та зберегти в DuckDB.

### Process
```
Stage 1: Collection → Stage 2: Processing → Stage 3: Feature Engineering → DuckDB Storage
```

### Implementation
1. **Collect Real Data** (Stage 1)
   - Завантажити OHLCV дані для тікерів (AMD, NVDA, TSLA, тощо)
   - Завантажити макроекономічні дані (FRED)
   - Завантажити новини та sentiment

2. **Process & Clean** (Stage 2)
   - Нормалізація
   - Видалення аномалій
   - Синхронізація часових меток

3. **Enrich Features** (Stage 3)
   - Запустити `FeatureOrchestrator.run()` з усіма 15+ збагачувачами
   - Генерувати технічні індикатори
   - Генерувати контекстну карту (Context Map)

4. **Store in DuckDB**
   - Використати `DataManager.upsert()` для зберігання
   - Таблиці: `raw_data`, `enriched_features`, `targets`

### Expected Output
```
DuckDB Tables:
├── raw_data (OHLCV + macro + news)
├── enriched_features (125+ features per row)
├── targets (4 timeframes × 4 target types = 16 targets)
└── event_series (event-based format for analysis)
```

---

## 🎲 Phase 2: Synthetic Data Generation

### Type 1: Typical Scenarios (Monte Carlo)
**Purpose**: Генерувати типові ринкові умови для тренування

**Engine**: `SimulationEngine.run_monte_carlo_for_strategy()`

**Process**:
1. Завантажити історичні повернення з реальних даних
2. Генерувати 1000+ синтетичних ціннісних шляхів
3. Для кожного шляху:
   - Застосувати стратегію
   - Розрахувати метрики (Sharpe, Max Drawdown, VaR)
   - Зберегти результати

**Output**: `synthetic_typical_scenarios.json`
```json
{
  "scenario_0": {
    "price_path": [...],
    "returns": [...],
    "signals": [...],
    "metrics": {"sharpe": 1.2, "max_dd": -0.15}
  },
  ...
}
```

### Type 2: Shock Scenarios (MonsterTest)
**Purpose**: Тестування стійкості до ринкових шоків

**Engine**: `MonsterTestMode.run()`

**Process**:
1. Генерувати синтетичні шокові сценарії:
   - Flash crash (-10% за 1 хвилину)
   - Volatility spike (VIX +50%)
   - Liquidity crisis (bid-ask spread +500%)
   - Black swan event (-20% за день)

2. Для кожного шоку:
   - Запустити модель на шокованих даних
   - Виміряти, як модель реагує
   - Розрахувати loss/profit

**Output**: `synthetic_shock_scenarios.json`
```json
{
  "flash_crash": {
    "shock_magnitude": -0.10,
    "model_response": [...],
    "portfolio_impact": -0.05,
    "recovery_time": 15
  },
  ...
}
```

### Type 3: Context Scenarios (DEAN)
**Purpose**: Генерувати ситуації для різних ринкових режимів

**Engine**: `DeanBootstrapSystem.internal_simulation()`

**Process**:
1. Визначити ринкові режими:
   - Trending (восходящий/нисходящий)
   - Ranging (бічний рух)
   - Volatile (висока волатильність)
   - Crisis (паніка)

2. Для кожного режиму:
   - Генерувати синтетичні дані з характеристиками режиму
   - Запустити Actor-Critic bootstrap
   - Зберегти дії та критики

**Output**: `synthetic_context_scenarios.json`
```json
{
  "trending_up": {
    "regime_characteristics": {...},
    "actor_actions": [...],
    "critic_feedback": [...],
    "simulation_steps": [...]
  },
  ...
}
```

---

## 📋 Implementation Checklist

### Real Data Accumulation
- [ ] Create `scripts/accumulate_real_data.py`
  - Stages 0-3 (no training)
  - Save to DuckDB
  - Verify event-series format

### Synthetic Data Generation
- [ ] Create `scripts/generate_typical_scenarios.py`
  - Use SimulationEngine
  - 1000+ Monte Carlo runs
  - Save results

- [ ] Create `scripts/generate_shock_scenarios.py`
  - Use MonsterTestMode
  - 5+ shock types
  - Save results

- [ ] Create `scripts/generate_context_scenarios.py`
  - Use DeanBootstrapSystem
  - 4+ market regimes
  - Save results

### Verification
- [ ] Verify enriched dataset structure
- [ ] Check event-series format
- [ ] Validate DuckDB tables
- [ ] Compare real vs synthetic distributions

---

## 🎯 Next Steps

1. **Verify Modules** (THIS TASK)
   - Check FeatureOrchestrator output
   - Check DataManager storage
   - Check event-series format

2. **Create Accumulation Script**
   - Real data only (no training)
   - Stages 0-3
   - DuckDB storage

3. **Create Synthetic Generators**
   - Typical scenarios (Monte Carlo)
   - Shock scenarios (MonsterTest)
   - Context scenarios (DEAN)

4. **Combine & Train**
   - Mix real + synthetic data
   - Train hybrid models
   - Evaluate performance
