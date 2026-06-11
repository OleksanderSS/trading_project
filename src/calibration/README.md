# 🎯 Calibration Module

## Overview

Модуль для **налаштування гіперпараметрів** DEAN моделі через Optuna optimization.

## 📁 Structure

```
src/calibration/
├── __init__.py                 # Module exports
├── calibration_engine.py       # Main calibration engine
└── README.md                   # This file
```

## 🔧 Components

### CalibrationEngine

Основний клас для калібрування гіперпараметрів.

**Methods:**
- `load_real_data()` - Завантажує реальні дані з DuckDB
- `load_synthetic_scenarios()` - Завантажує синтетичні сценарії
- `define_hyperparameter_space()` - Визначає простір пошуку
- `evaluate_hyperparameters()` - Оцінює конфігурацію
- `run_calibration()` - Запускає процес калібрування

## 🚀 Usage

### Via Hybrid Pipeline

```bash
python run_hybrid_pipeline.py --mode calibrate --test-ticker AMD --n-trials 50
```

### Via Standalone Script

```bash
python scripts/calibrate_dean.py --ticker AMD --trials 50
```

### Programmatic Usage

```python
from src.calibration import CalibrationEngine

engine = CalibrationEngine(
    real_data_path="data/duckdb/trading.db",
    synthetic_data_path="data/synthetic/",
    n_trials=50,
    metric="sharpe_ratio",
    batch_name="my_calibration"
)

results = engine.run_calibration(
    test_ticker="AMD",
    test_target="target_return_1d"
)

print(f"Best Sharpe Ratio: {results['best_value']:.4f}")
print(f"Best hyperparameters: {results['best_params']}")
```

## 📊 Hyperparameters

### Learning Rates
- `actor_lr`: 1e-5 to 1e-3 (log scale)
- `critic_lr`: 1e-5 to 1e-3 (log scale)

### Network Architecture
- `hidden_dim`: [128, 256, 512]
- `num_layers`: 2 to 4

### Training Parameters
- `batch_size`: [32, 64, 128, 256]
- `replay_buffer_size`: [10000, 50000, 100000]

### RL Parameters
- `gamma`: 0.95 to 0.999
- `tau`: 0.001 to 0.01
- `exploration_noise`: 0.01 to 0.3

### Regularization
- `dropout`: 0.0 to 0.3
- `weight_decay`: 1e-6 to 1e-3 (log scale)

## 🎯 Optimization Metrics

### Supported Metrics
- `sharpe_ratio` (default) - Risk-adjusted returns
- `max_drawdown` - Maximum portfolio decline
- `win_rate` - Percentage of profitable trades
- `profit_factor` - Gross profit / Gross loss
- `calmar_ratio` - Return / Max Drawdown

## 📁 Output

### calibration_results.json

```json
{
  "status": "success",
  "best_params": {
    "actor_lr": 0.0001,
    "critic_lr": 0.0002,
    "hidden_dim": 256,
    "num_layers": 3,
    "batch_size": 128,
    "replay_buffer_size": 50000,
    "gamma": 0.99,
    "tau": 0.005,
    "exploration_noise": 0.1,
    "dropout": 0.1,
    "weight_decay": 0.0001
  },
  "best_value": 1.85,
  "metric": "sharpe_ratio",
  "n_trials": 50,
  "test_ticker": "AMD",
  "test_target": "target_return_1d",
  "study_name": "dean_calibration_my_calibration"
}
```

## 🔄 Integration with Pipeline

### Step 1: Accumulate Real Data
```bash
python scripts/accumulate_real_data.py --tickers AMD NVDA --days 30
```

### Step 2: Generate Synthetic Data
```bash
python scripts/generate_synthetic_data.py
```

### Step 3: Run Calibration
```bash
python run_hybrid_pipeline.py --mode calibrate --test-ticker AMD
```

### Step 4: Use Best Hyperparameters
```bash
# Copy best_params to DEAN config
# Then train with optimized hyperparameters
python run_hybrid_pipeline.py --mode prepare --test-ticker AMD
```

## 🧠 How It Works

1. **Load Data** - Real data from DuckDB + Synthetic scenarios
2. **Define Space** - Hyperparameter search space
3. **Optimize** - Optuna samples and evaluates configurations
4. **Evaluate** - Train DEAN and calculate metrics
5. **Select Best** - Return best hyperparameters

## 📈 Expected Results

### Typical Run (50 trials)
- Duration: 30-60 minutes
- Best Sharpe Ratio: 1.5-2.5 (good), 2.5+ (excellent)
- Convergence: After 20-30 trials

## 🚨 Troubleshooting

### No Real Data
```bash
python scripts/accumulate_real_data.py --tickers AMD --days 30
```

### No Synthetic Scenarios
```bash
python scripts/generate_synthetic_data.py
```

### Slow Calibration
```bash
# Reduce trials or use single ticker
python run_hybrid_pipeline.py --mode calibrate --test-ticker AMD --n-trials 10
```

## 🔗 Related

- **Guide**: `docs/CALIBRATION_GUIDE.md`
- **Data Strategy**: `scripts/data_accumulation_strategy.md`
- **DEAN**: `src/models/dean/`
- **Pipeline**: `src/pipeline/hybrid_orchestrator.py`

## 📚 Dependencies

- `optuna` - Hyperparameter optimization
- `duckdb` - Real data loading
- `pandas` - Data manipulation
- `json` - Scenario loading

## 🎯 Future Improvements

- [ ] Multi-objective optimization (Sharpe + Max DD)
- [ ] Parallel trial execution
- [ ] Optuna study persistence (SQLite)
- [ ] Visualization (Optuna plots)
- [ ] Early stopping
- [ ] Cross-validation
- [ ] Ensemble calibration
