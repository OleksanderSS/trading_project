# 🎯 DEAN Calibration Guide

## Overview

Calibration - це процес **налаштування гіперпараметрів** DEAN моделі для оптимальної продуктивності.

## 🔧 Що таке Calibration?

Calibration використовує:
- **Реальні дані** з DuckDB (enriched_features, targets)
- **Синтетичні сценарії** (typical, shock, context)
- **Optuna** для оптимізації гіперпараметрів
- **Метрики** (Sharpe Ratio, Max Drawdown, Win Rate)

## 🚀 Quick Start

### Basic Usage

```bash
# Calibrate with default settings (50 trials)
python run_hybrid_pipeline.py --mode calibrate

# Calibrate with specific ticker
python run_hybrid_pipeline.py --mode calibrate --test-ticker AMD

# Calibrate with specific target
python run_hybrid_pipeline.py --mode calibrate --test-target target_return_1d

# Calibrate with custom number of trials
python run_hybrid_pipeline.py --mode calibrate --n-trials 100

# Full calibration with all parameters
python run_hybrid_pipeline.py --mode calibrate \
  --test-ticker AMD \
  --test-target target_return_1d \
  --n-trials 100 \
  --batch-name my_calibration
```

## 📊 Hyperparameters

### Learning Rates
- `actor_lr`: Actor learning rate (1e-5 to 1e-3)
- `critic_lr`: Critic learning rate (1e-5 to 1e-3)

### Network Architecture
- `hidden_dim`: Hidden layer dimension (128, 256, 512)
- `num_layers`: Number of layers (2-4)

### Training Parameters
- `batch_size`: Batch size (32, 64, 128, 256)
- `replay_buffer_size`: Replay buffer size (10k, 50k, 100k)

### RL Parameters
- `gamma`: Discount factor (0.95-0.999)
- `tau`: Target network update rate (0.001-0.01)
- `exploration_noise`: Exploration noise (0.01-0.3)

### Regularization
- `dropout`: Dropout rate (0.0-0.3)
- `weight_decay`: Weight decay (1e-6 to 1e-3)

## 🎯 Optimization Metrics

### Primary Metric
- **Sharpe Ratio** (default) - Risk-adjusted returns

### Additional Metrics
- **Max Drawdown** - Maximum portfolio decline
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross profit / Gross loss
- **Calmar Ratio** - Return / Max Drawdown

## 📁 Output Structure

```
results/calibration/{batch_name}/
├── calibration_results.json    # Best hyperparameters and metrics
└── optuna_study.db             # Optuna study database (future)
```

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

## 🔄 Workflow

### Step 1: Accumulate Real Data
```bash
python scripts/accumulate_real_data.py --tickers AMD NVDA --days 30
```

### Step 2: Generate Synthetic Data
```bash
python scripts/generate_synthetic_data.py --types typical shock context
```

### Step 3: Run Calibration
```bash
python run_hybrid_pipeline.py --mode calibrate --test-ticker AMD --n-trials 50
```

### Step 4: Use Best Hyperparameters
```bash
# Copy best_params from calibration_results.json to DEAN config
# Then train with optimized hyperparameters
python run_hybrid_pipeline.py --mode prepare --test-ticker AMD
```

## 🧠 How It Works

### 1. Data Loading
- Load enriched features from DuckDB
- Load targets from DuckDB
- Load synthetic scenarios from JSON files

### 2. Hyperparameter Search
- Define search space (ranges for each hyperparameter)
- Use Optuna to sample hyperparameters
- Evaluate each configuration

### 3. Evaluation
- Train DEAN with sampled hyperparameters
- Evaluate on validation set
- Calculate primary metric (Sharpe Ratio)

### 4. Optimization
- Optuna selects next hyperparameters based on previous results
- Repeat for n_trials iterations
- Return best configuration

## 📈 Expected Results

### Typical Calibration Run (50 trials)
- **Duration**: 30-60 minutes (depends on data size)
- **Best Sharpe Ratio**: 1.5-2.5 (good), 2.5+ (excellent)
- **Convergence**: Usually after 20-30 trials

### Factors Affecting Results
- **Data Quality**: More data → better calibration
- **Synthetic Scenarios**: More scenarios → better generalization
- **Number of Trials**: More trials → better optimization

## 🎯 Best Practices

### 1. Start with Small Dataset
```bash
# Test calibration with single ticker first
python run_hybrid_pipeline.py --mode calibrate --test-ticker AMD --n-trials 20
```

### 2. Increase Trials Gradually
```bash
# Start with 20 trials, then 50, then 100
python run_hybrid_pipeline.py --mode calibrate --n-trials 20
python run_hybrid_pipeline.py --mode calibrate --n-trials 50
python run_hybrid_pipeline.py --mode calibrate --n-trials 100
```

### 3. Use Multiple Targets
```bash
# Calibrate for different targets
python run_hybrid_pipeline.py --mode calibrate --test-target target_return_1d
python run_hybrid_pipeline.py --mode calibrate --test-target target_return_1w
```

### 4. Monitor Progress
- Check `results/calibration/{batch_name}/calibration_results.json`
- Look for convergence in metric values
- Verify best hyperparameters make sense

## 🚨 Troubleshooting

### No Real Data Available
```bash
# Run data accumulation first
python scripts/accumulate_real_data.py --tickers AMD --days 30
```

### No Synthetic Scenarios
```bash
# Generate synthetic data first
python scripts/generate_synthetic_data.py
```

### Calibration Takes Too Long
```bash
# Reduce number of trials
python run_hybrid_pipeline.py --mode calibrate --n-trials 10

# Or use single ticker
python run_hybrid_pipeline.py --mode calibrate --test-ticker AMD
```

### Poor Results
- Check data quality (verify_enriched_dataset.py)
- Increase number of trials
- Try different primary metric
- Verify synthetic scenarios are realistic

## 🔗 Related Documentation

- **Data Accumulation**: `scripts/DATA_ACCUMULATION_README.md`
- **Synthetic Generation**: `scripts/data_accumulation_strategy.md`
- **DEAN Architecture**: `docs/DEAN_ARCHITECTURE.md`
- **Hybrid Pipeline**: `docs/HYBRID_PIPELINE.md`

## 📚 References

- **Optuna**: https://optuna.org/
- **Hyperparameter Tuning**: https://en.wikipedia.org/wiki/Hyperparameter_optimization
- **Sharpe Ratio**: https://en.wikipedia.org/wiki/Sharpe_ratio
