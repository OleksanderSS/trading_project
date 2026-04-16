# 📚 SRC Folder Structure Guide

**Quick Reference for Developers**

---

## 🎯 Core Pipeline (7 Stages)

```
src/pipeline/stages/
├─ stage_0_setup.py          → Environment & config initialization
├─ stage_1_collection.py     → Data collection from sources
├─ stage_2_processing.py     → Data cleaning & preprocessing
├─ stage_3_feature_engineering.py → Feature creation
├─ stage_4_modeling.py       → Model training
├─ stage_5_prediction.py     → Inference & predictions
├─ stage_6_trading_execution.py → Live trading/backtesting
├─ stage_7_evaluation.py     → Performance metrics
└─ base_stage.py             → Base class for all stages
```

**Entry Points**:
- Sequential: `src/pipeline/pipeline_orchestrator.py`
- Hybrid (adaptive): `src/pipeline/hybrid_orchestrator.py`

---

## 🤖 Models (9 Types)

```
src/models/
├─ tree/                    → XGBoost, LightGBM, CatBoost, RandomForest
├─ neural/                  → LSTM, GRU, CNN, Transformer, AutoEncoder, MLP
├─ linear/                  → Linear, SVM, KNN
├─ ensemble/                → Ensemble combinations
├─ adapters/                → Model adapters/wrappers
├─ dean/                    → DEAN-specific models
├─ model_selector/          → Automated model selection
├─ factory.py               → Model factory (creates any model)
└─ interfaces.py            → Base interfaces (BaseModel, etc.)
```

**Usage**: `factory.create_model('xgboost', **config)`

---

## 📊 Features (Multi-level)

```
src/features/
├─ builders/                → Dataset builders
├─ enrichers/               → Data enrichment (5+ enrichers)
│  ├─ technical_analysis_enricher.py
│  ├─ time_features_enricher.py
│  ├─ sentiment_enricher.py
│  ├─ context_enricher.py
│  ├─ market_regime_enricher.py
├─ selection/               → Feature selection (3+ methods)
│  ├─ smart_selector.py     → Smart selection algorithm
│  ├─ volatility_driver_selector.py
│  ├─ pca_selector.py
├─ nlp/                     → NLP-specific features
└─ utils/                   → Utilities (time, indicators, etc.)
```

**Entry Point**: `FeatureOrchestrator` (manages all enrichers & selectors)

---

## 🎓 Training (Multiple Strategies)

```
src/training/
├─ unified_training_manager.py → Orchestrates BATCH/PROGRESSIVE/HYBRID
├─ adaptive_training_manager.py → Adds adaptive targets layer
├─ batch_trainer.py         → Batch training strategy
├─ progressive_trainer.py   → Progressive training (adaptive sizing)
├─ light_model_trainer.py   → Lightweight/fast training
├─ pattern_aware_training.py → Pattern-based training
└─ run_training.py          → CLI entry point
```

**Usage**:
```python
manager = UnifiedTrainingManager()
results = manager.execute_unified_training(tickers, data)
```

---

## 🔮 Predictions

```
src/predictions/
├─ models_predict.py        → Batch predictions on multiple models
├─ deep_predict.py          → Deep learning predictions
└─ prediction_utils.py      → Utilities (caching, etc.)
```

---

## 💰 Trading

```
src/trading/
├─ trader.py                → Main trader logic
├─ trading_orchestrator.py  → Coordinates trading workflow
├─ consensus_engine.py      → Multi-model consensus
├─ portfolio_manager.py     → Portfolio management
├─ post_inference_filter.py → Signal filtering
├─ virtual_portfolio.py     → Backtesting portfolio
└─ archive/                 → Old/archive trading code
```

---

## 🔧 Core Infrastructure (12 Subfolders)

```
src/core/
├─ logging/                 → Logging system (ProjectLogger)
├─ caching/                 → Caching mechanisms (Redis, file)
├─ error_handling/          → Exception classes, retry logic
├─ file_management/         → File operations, paths
├─ clients/                 → External service clients
├─ cloud/                   → Cloud storage (GCS, S3)
├─ quality/                 → Data/code quality checks
├─ security/                → Security utilities
├─ validation/              → Data validation
├─ retry/                   → Retry mechanisms
├─ system/                  → System utilities
└─ version_checker.py       → Version checking
```

---

## 📈 Analytics & Monitoring

```
src/analytics/
├─ analyzers/               → 6+ specialized analyzers
├─ arena/                   → Arena battle system
├─ backtesting/             → Backtesting analytics
├─ calculators/             → Performance calculations
├─ context/                 → Market context analysis
├─ data_managers/           → Data aggregation
├─ detectors/               → Pattern detectors
├─ reporting/               → Automated reports
├─ signals/                 → Signal generation
└─ unified_analytics_engine.py → Main analytics coordinator

src/monitoring/
├─ monitoring_system.py     → Core monitoring
├─ health_hub.py            → System health checks
├─ infrastructure/          → Resource monitoring
├─ dashboard/               → Monitoring dashboard
└─ config/                  → Monitoring config
```

---

## ⚙️ Configuration (30+ YAML Files)

```
src/config/
├─ YAML Configs:
│  ├─ system.yaml           → Paths, caching, logging
│  ├─ models.yaml           → Model defaults
│  ├─ features.yaml         → Feature engineering
│  ├─ data_sources.yaml     → Data source definitions
│  ├─ processing.yaml       → Data processing config
│  ├─ collectors.yaml       → Data collectors config
│  ├─ enrichment.yaml       → Enricher configuration
│  ├─ risk_management.yaml  → Risk parameters
│  ├─ strategy.yaml         → Trading strategy config
│  ├─ monitoring.yaml       → Monitoring configuration
│  └─ (17 more YAML files...)
├─ Python Managers:
│  ├─ unified_config_manager.py → Central config access
│  ├─ tickers.py            → Ticker configuration
│  └─ __init__.py
└─ Generated Files (Ignored):
   ├─ runtime_params.json    ⚠️ Ignored by .gitignore
   └─ selected_features_cache.json ⚠️ Ignored by .gitignore
```

---

## 🔗 Relationships Diagram

```
┌─────────────────────────────────────────────────┐
│         PIPELINE ORCHESTRATOR                   │
│ (Manages stages 0-7 sequentially or adaptive)   │
└──────────┬──────────────────────────────────────┘
           │
     ┌─────┴─────────────────────────────────────┐
     ▼                                            ▼
┌──────────────┐                          ┌────────────────┐
│ FEATURES     │                          │ MODELS         │
│              │                          │                │
│ Enrichers ▼  │                          │ Factory ▼      │
│ - Technical  │                          │ - Create any   │
│ - Sentiment  │                          │   model        │
│ - Context    │                          │ - Routes to    │
│ - NLP        │                          │   tree/neural  │
│              │                          │                │
│ Selection ▼  │                          │ Consensus ▼    │
│ - Smart      │                          │ - Ensemble     │
│ - Volatility │                          │   predictions  │
│ - PCA        │                          │                │
└──────────────┘                          └────────────────┘
      ▲                                            ▲
      └─────────────────┬──────────────────────────┘
                        │
                   ┌────────────┐
                   │ TRAINING   │
                   │            │
                   │ Unified ▼  │
                   │ - Batch    │
                   │ - Prog     │
                   │ - Hybrid   │
                   │            │
                   │ Adaptive ▼ │
                   │ - Smart    │
                   │   targets  │
                   │            │
                   │ Predict ▼  │
                   │ - Inference│
                   └────┬───────┘
                        │
                   ┌────▼───────┐
                   │ TRADING    │
                   │            │
                   │ Consensus  │
                   │ Filter     │
                   │ Portfolio  │
                   │ Execution  │
                   └────────────┘
```

---

## 🚀 Common Commands

### Train Models
```python
from src.training import UnifiedTrainingManager
manager = UnifiedTrainingManager()
results = manager.execute_unified_training(tickers, data)
```

### Get Predictions
```python
from src.predictions import models_predict
predictions = models_predict.predict(data, model_ids)
```

### Enrich Features
```python
from src.features import FeatureOrchestrator
orchest = FeatureOrchestrator()
enriched_data = orchest.enrich(data, with_sentiment=True)
```

### Analyze Backtests
```python
from src.analytics import UnifiedAnalyticsEngine
analytics = UnifiedAnalyticsEngine()
report = analytics.generate_detailed_report(results)
```

---

## ⚠️ Known Limitations

1. **Sentiment Analysis**: Disabled if PyTorch/Transformers not available (returns neutral)
2. **Database Validation**: Lock conflicts on startup (disabled by default)
3. **Analysis Folder**: Currently empty (planned for future use)
4. **Config Sprawl**: 30+ YAML files can be overwhelming
5. **Circular Dependencies**: Not fully verified (potential import cycles)

---

## 📞 Questions?

- **Architecture**: See `ARCHITECTURE.md` + `ARCHITECTURE_ANALYSIS_SESSION_4.md`
- **Configuration**: See `CONFIG_INDEX.md` (to-do: create)
- **Training Flow**: See `src/pipeline/README.md`
- **Model Selection**: See `src/models/factory.py` docstrings

---

**Last Updated**: April 16, 2026 (Session 4)  
**Maintainer**: Architecture Review Team

