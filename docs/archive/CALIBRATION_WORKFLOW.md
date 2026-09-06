# 🔄 Calibration Workflow

## Overview

Візуальна схема процесу калібрування DEAN гіперпараметрів.

## 📊 Complete Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CALIBRATION WORKFLOW                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Step 1: Data   │
│  Accumulation   │
└────────┬────────┘
         │
         │  python scripts/accumulate_real_data.py
         │  --tickers AMD NVDA --days 30
         │
         ▼
┌─────────────────┐
│   DuckDB        │
│   Database      │
│                 │
│ • raw_data      │
│ • enriched_     │
│   features      │
│ • targets       │
└────────┬────────┘
         │
         │
         ▼
┌─────────────────┐
│  Step 2:        │
│  Synthetic      │
│  Generation     │
└────────┬────────┘
         │
         │  python scripts/generate_synthetic_data.py
         │  --types typical shock context
         │
         ▼
┌─────────────────┐
│   Synthetic     │
│   Scenarios     │
│                 │
│ • typical       │
│ • shock         │
│ • context       │
└────────┬────────┘
         │
         │
         ▼
┌─────────────────┐
│  Step 3:        │
│  Calibration    │
└────────┬────────┘
         │
         │  python run_hybrid_pipeline.py --mode calibrate
         │  --test-ticker AMD --n-trials 50
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CALIBRATION ENGINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load Real Data (DuckDB)                                     │
│     ├─ enriched_features (125+ features)                        │
│     └─ targets (16 targets)                                     │
│                                                                 │
│  2. Load Synthetic Scenarios (JSON)                             │
│     ├─ typical (100+ scenarios)                                 │
│     ├─ shock (5 shock types)                                    │
│     └─ context (5 market regimes)                               │
│                                                                 │
│  3. Define Hyperparameter Space                                 │
│     ├─ Learning rates (actor_lr, critic_lr)                     │
│     ├─ Architecture (hidden_dim, num_layers)                    │
│     ├─ Training (batch_size, replay_buffer_size)                │
│     ├─ RL (gamma, tau, exploration_noise)                       │
│     └─ Regularization (dropout, weight_decay)                   │
│                                                                 │
│  4. Optuna Optimization Loop                                    │
│     ┌──────────────────────────────────────┐                   │
│     │  For trial in range(n_trials):       │                   │
│     │    1. Sample hyperparameters         │                   │
│     │    2. Train DEAN model               │                   │
│     │    3. Evaluate on validation set     │                   │
│     │    4. Calculate Sharpe Ratio         │                   │
│     │    5. Update Optuna study            │                   │
│     └──────────────────────────────────────┘                   │
│                                                                 │
│  5. Select Best Configuration                                   │
│     └─ Highest Sharpe Ratio                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         │
         │
         ▼
┌─────────────────┐
│  Step 4:        │
│  Save Results   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  results/calibration/{batch_name}/calibration_results.json      │
├─────────────────────────────────────────────────────────────────┤
│  {                                                              │
│    "status": "success",                                         │
│    "best_params": {                                             │
│      "actor_lr": 0.0001,                                        │
│      "critic_lr": 0.0002,                                       │
│      "hidden_dim": 256,                                         │
│      "num_layers": 3,                                           │
│      "batch_size": 128,                                         │
│      "replay_buffer_size": 50000,                               │
│      "gamma": 0.99,                                             │
│      "tau": 0.005,                                              │
│      "exploration_noise": 0.1,                                  │
│      "dropout": 0.1,                                            │
│      "weight_decay": 0.0001                                     │
│    },                                                           │
│    "best_value": 1.85,                                          │
│    "metric": "sharpe_ratio",                                    │
│    "n_trials": 50                                               │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
         │
         │
         ▼
┌─────────────────┐
│  Step 5:        │
│  Use Best       │
│  Hyperparams    │
└────────┬────────┘
         │
         │  Copy best_params to DEAN config
         │  python run_hybrid_pipeline.py --mode prepare
         │
         ▼
┌─────────────────┐
│  DEAN Training  │
│  with Optimized │
│  Hyperparams    │
└─────────────────┘
```

## 🎯 Detailed Calibration Process

```
┌─────────────────────────────────────────────────────────────────┐
│              OPTUNA OPTIMIZATION LOOP                           │
└─────────────────────────────────────────────────────────────────┘

Trial 1:
  Sample: actor_lr=0.001, critic_lr=0.001, hidden_dim=128, ...
  Train DEAN → Evaluate → Sharpe Ratio = 1.2
  
Trial 2:
  Sample: actor_lr=0.0001, critic_lr=0.0002, hidden_dim=256, ...
  Train DEAN → Evaluate → Sharpe Ratio = 1.5
  
Trial 3:
  Sample: actor_lr=0.0005, critic_lr=0.0001, hidden_dim=512, ...
  Train DEAN → Evaluate → Sharpe Ratio = 1.3
  
...

Trial 50:
  Sample: actor_lr=0.0001, critic_lr=0.0002, hidden_dim=256, ...
  Train DEAN → Evaluate → Sharpe Ratio = 1.85 ← BEST!

┌─────────────────────────────────────────────────────────────────┐
│  Best Configuration Found:                                      │
│  • actor_lr: 0.0001                                             │
│  • critic_lr: 0.0002                                            │
│  • hidden_dim: 256                                              │
│  • num_layers: 3                                                │
│  • batch_size: 128                                              │
│  • replay_buffer_size: 50000                                    │
│  • gamma: 0.99                                                  │
│  • tau: 0.005                                                   │
│  • exploration_noise: 0.1                                       │
│  • dropout: 0.1                                                 │
│  • weight_decay: 0.0001                                         │
│                                                                 │
│  Best Sharpe Ratio: 1.85                                        │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Integration with Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  HYBRID PIPELINE MODES                          │
└─────────────────────────────────────────────────────────────────┘

Mode: local
  └─ Stages 0-3 (Data accumulation)

Mode: prepare
  └─ Prepare data for Colab training

Mode: light
  └─ Train light models

Mode: full
  └─ Full pipeline execution

Mode: continue
  └─ Continue after Colab results

Mode: calibrate  ← NEW!
  └─ DEAN hyperparameter calibration
     ├─ Load real data (DuckDB)
     ├─ Load synthetic scenarios (JSON)
     ├─ Run Optuna optimization
     └─ Save best hyperparameters
```

## 📊 Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA FLOW                                  │
└─────────────────────────────────────────────────────────────────┘

Real Data (DuckDB)
  ├─ enriched_features (2520 rows × 125 columns)
  │  ├─ Technical indicators
  │  ├─ Volatility features
  │  ├─ Momentum features
  │  ├─ Volume features
  │  ├─ Context features
  │  ├─ Sentiment features
  │  └─ Macro features
  │
  └─ targets (2520 rows × 16 columns)
     ├─ target_return_1d
     ├─ target_return_1w
     ├─ target_return_1m
     └─ ...

Synthetic Scenarios (JSON)
  ├─ typical (100+ scenarios)
  │  └─ Monte Carlo simulations
  │
  ├─ shock (5 scenarios)
  │  ├─ Flash crash
  │  ├─ Volatility spike
  │  ├─ Liquidity crisis
  │  ├─ Black swan
  │  └─ Circuit breaker
  │
  └─ context (5 scenarios)
     ├─ Trending up
     ├─ Trending down
     ├─ Ranging
     ├─ Volatile
     └─ Crisis

         ↓
         
CalibrationEngine
  ├─ Combines real + synthetic data
  ├─ Trains DEAN with different hyperparameters
  ├─ Evaluates on validation set
  └─ Selects best configuration

         ↓
         
Best Hyperparameters (JSON)
  └─ Used for DEAN training
```

## 🎯 Usage Examples

### Example 1: Quick Test
```bash
# Test with single ticker, 10 trials
python run_hybrid_pipeline.py --mode calibrate \
  --test-ticker AMD \
  --n-trials 10
```

### Example 2: Full Calibration
```bash
# Full calibration with 50 trials
python run_hybrid_pipeline.py --mode calibrate \
  --test-ticker AMD \
  --test-target target_return_1d \
  --n-trials 50 \
  --batch-name amd_calibration
```

### Example 3: Standalone Script
```bash
# Using standalone script
python scripts/calibrate_dean.py \
  --ticker AMD \
  --target target_return_1d \
  --trials 50 \
  --metric sharpe_ratio
```

## 📈 Expected Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIMELINE                                     │
└─────────────────────────────────────────────────────────────────┘

Step 1: Data Accumulation (30 days)
  Duration: 5-10 minutes
  Output: DuckDB database (2520 rows)

Step 2: Synthetic Generation
  Duration: 2-5 minutes
  Output: 110+ synthetic scenarios

Step 3: Calibration (50 trials)
  Duration: 30-60 minutes
  Output: Best hyperparameters

Step 4: DEAN Training
  Duration: 1-2 hours
  Output: Trained model

Total: ~2-3 hours
```

## 🔗 Related Documentation

- **User Guide**: `docs/CALIBRATION_GUIDE.md`
- **Module README**: `src/calibration/README.md`
- **Summary**: `CALIBRATION_SUMMARY.md`
- **Checklist**: `CALIBRATION_CHECKLIST.md`
- **Data Strategy**: `docs/archive/data_strategy.md`

---

**Created**: 2026-05-02  
**Purpose**: Visual workflow documentation for calibration process
