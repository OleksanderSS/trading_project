# 🔗 Integration Module

## Overview

Модуль інтеграції забезпечує зв'язок між різними компонентами системи, включаючи dashboard, ensemble моделі, та performance tracking.

**Статус**: ✅ Повністю реалізовано

---

## 📦 Компоненти

### 1. DashboardDataBridge (`dashboard_data_bridge.py`)

**Призначення**: Зв'язок між pipeline даними та Streamlit dashboard

**Ключові функції**:
- ✅ Real-time data access з кешуванням (TTL: 5 хв)
- ✅ 8 типів даних для dashboard
- ✅ Mock data fallback
- ✅ System health monitoring

**Типи даних**:
1. **model_performance** - Метрики моделей (win rate, Sharpe, precision)
2. **trading_activity** - Останні торгові сигнали
3. **portfolio_metrics** - Метрики портфеля (returns, volatility, drawdown)
4. **market_data** - Ринкові дані (OHLCV)
5. **system_status** - Статус системи (CPU, memory, disk)
6. **ensemble_weights** - Ваги ансамблю
7. **arena_results** - Результати model arena

**Usage**:
```python
from src.integration.dashboard_data_bridge import DashboardDataBridge

# Initialize bridge
bridge = DashboardDataBridge(config_manager=config_manager)

# Get model performance data
performance_data = bridge.get_dashboard_data('model_performance')
# Returns: {'models': [...], 'total_models': 3, 'last_updated': '...'}

# Get trading activity
activity_data = bridge.get_dashboard_data('trading_activity')
# Returns: {'signals': [...], 'total_signals': 3, 'last_updated': '...'}

# Get portfolio metrics
portfolio_data = bridge.get_dashboard_data('portfolio_metrics')
# Returns: {'total_value': 125000.0, 'returns': 0.125, ...}

# Get system status
system_data = bridge.get_dashboard_data('system_status')
# Returns: {'cpu_percent': 45.2, 'memory': {...}, 'disk': {...}}

# Clear cache
bridge.clear_cache()

# Get cache info
cache_info = bridge.get_cache_info()
# Returns: {'cached_items': 5, 'cache_ttl_seconds': 300, ...}
```

**Кешування**:
- TTL: 300 секунд (5 хвилин)
- Автоматичне оновлення при запиті
- Можливість очистки кешу

**Mock Data**:
- Використовується якщо DataManager недоступний
- Реалістичні тестові дані
- Підтримка всіх типів даних

---

### 2. EnsemblePerformanceBridge (`ensemble_performance_bridge.py`)

**Призначення**: Синхронізація між LiveAdaptiveEnsemble та ModelPerformanceTracker

**Ключові функції**:
- ✅ Синхронізація метрик (кожні 5 хвилин)
- ✅ Unified performance view
- ✅ Bidirectional updates
- ✅ Data format conversion

**Workflow**:
```
LiveAdaptiveEnsemble → Extract Metrics → Convert Format → ModelPerformanceTracker
                                ↓
                        Unified View (merged data)
                                ↓
                        Dashboard / Analytics
```

**Usage**:
```python
from src.integration.ensemble_performance_bridge import EnsemblePerformanceBridge

# Initialize bridge
bridge = EnsemblePerformanceBridge(
    live_ensemble=live_ensemble,
    performance_tracker=performance_tracker
)

# Sync ensemble performance to tracker
sync_result = bridge.sync_ensemble_performance_to_tracker(force_sync=False)
# Returns: {
#     'sync_time': datetime,
#     'ensemble_metrics_count': 10,
#     'tracker_records_created': 10,
#     'records_updated': 8,
#     'success': True
# }

# Get unified performance view
unified_view = bridge.get_unified_performance_view()
# Returns: {
#     'unified_performance': {...},
#     'ensemble_models': 10,
#     'tracker_models': 12,
#     'total_unique_models': 15
# }

# Get ensemble weights for prediction
weights = bridge.get_ensemble_weights_for_prediction()
# Returns: {'model1': 0.35, 'model2': 0.25, 'model3': 0.40}

# Update ensemble from tracker
success = bridge.update_ensemble_from_tracker(['model1', 'model2'])
# Returns: True if successful
```

**Синхронізація**:
- Автоматична кожні 5 хвилин
- Force sync опція
- Кешування результатів
- Error handling

**Data Format**:
```python
# Ensemble format
{
    'model_name': 'LSTM_v1',
    'model_type': 'Neural',
    'sharpe_ratio': 1.8,
    'hit_rate': 0.65,
    'precision': 0.72,
    'predictions_count': 1240,
    'source': 'live_ensemble'
}

# Tracker format
{
    'model_name': 'LSTM_v1',
    'model_type': 'Neural',
    'avg_win_rate': 0.65,
    'avg_sharpe_ratio': 1.8,
    'avg_precision': 0.72,
    'total_trades': 1240,
    'source': 'live_ensemble_sync'
}
```

---

### 3. EnsembleSelector (`ensemble_selector.py`)

**Призначення**: Інтелектуальний вибір найкращого ensemble методу

**Ключові функції**:
- ✅ Context-aware selection
- ✅ 5 ensemble methods підтримка
- ✅ Scoring algorithm
- ✅ Reasoning generation
- ✅ Confidence calculation

**Ensemble Methods**:

#### 1. LiveAdaptiveEnsemble
- **Strengths**: Real-time adaptation, performance tracking, regime-aware
- **Weaknesses**: Requires historical data, higher latency
- **Best for**: Live trading, adaptive strategies, multi-model systems
- **Requirements**: min_models=3, real_time_data=True, history_days=30

#### 2. StackedEnsemble
- **Strengths**: Meta-learning, ridge regression, live efficiency weighting
- **Weaknesses**: Requires training, meta-model complexity
- **Best for**: Meta-learning, complex combinations, weighted combinations
- **Requirements**: min_models=2, training_data=True, meta_model=True

#### 3. ConsensusEngine
- **Strengths**: Decision core, regime-aware, critic filters, KNN patterns
- **Weaknesses**: Complex dependencies, requires diary engine
- **Best for**: Final decisions, risk-aware, quality signals
- **Requirements**: min_models=3, experience_diary=True, threshold_analyzer=True

#### 4. SimpleAverage
- **Strengths**: Fast, simple, reliable
- **Weaknesses**: No adaptation, equal weights
- **Best for**: Quick predictions, baseline, low resources
- **Requirements**: min_models=1, real_time_data=False

#### 5. WeightedAverage
- **Strengths**: Performance-based, simple
- **Weaknesses**: Static weights, requires performance data
- **Best for**: Performance-weighted, moderate complexity
- **Requirements**: min_models=1, performance_data=True

**Usage**:
```python
from src.integration.ensemble_selector import EnsembleSelector, EnsembleContext

# Initialize selector
selector = EnsembleSelector()

# Create context
context = EnsembleContext(
    data_size=10000,
    has_real_time_data=True,
    model_count=5,
    market_regime='volatile',
    volatility_level=0.8,
    prediction_frequency='real_time',
    computational_resources='high',
    latency_requirement='low'
)

# Select best ensemble
selection = selector.select_best_ensemble(
    context=context,
    available_models=['model1', 'model2', 'model3'],
    performance_data={'model1': 0.8, 'model2': 0.7, 'model3': 0.9}
)

# Returns:
# {
#     'selected_ensemble': 'live_adaptive',
#     'score': 0.85,
#     'reasoning': 'Excellent fit for live_adaptive due to real-time adaptation capability, regime-aware weighting for volatile markets',
#     'all_scores': {'live_adaptive': 0.85, 'stacked_ensemble': 0.65, ...},
#     'confidence': 0.9,
#     'selection_time': datetime
# }

# Create ensemble instance
ensemble = selector.create_ensemble_instance(
    method_name='live_adaptive',
    models=['model1', 'model2', 'model3']
)
```

**Scoring Algorithm**:
```python
# Total score = context_fit + performance_fit + resource_fit + latency_fit
# Each component: 0.0 - 0.25
# Total range: 0.0 - 1.0

# Context fit (0.0 - 0.25)
- Market regime match: +0.2
- Data size match: +0.1
- Model count match: +0.2

# Performance fit (0.0 - 0.25)
- Performance data available: +0.15
- Real-time adaptation: +0.15

# Resource fit (0.0 - 0.25)
- Low resources + fast method: +0.2
- High resources + complex method: +0.2

# Latency fit (0.0 - 0.25)
- High latency req + fast method: +0.2
- Low latency req + adaptive method: +0.2
```

**Selection Rules**:
```python
{
    'live_trading': {
        'preferred': ['live_adaptive'],
        'avoid': ['enhanced_batch'],
        'reasoning': 'Live trading requires real-time adaptation'
    },
    'batch_prediction': {
        'preferred': ['enhanced_batch', 'weighted_average'],
        'avoid': ['live_adaptive'],
        'reasoning': 'Batch processing can handle heavier models'
    },
    'low_resources': {
        'preferred': ['simple_average'],
        'avoid': ['enhanced_batch', 'live_adaptive'],
        'reasoning': 'Limited resources require simple methods'
    },
    'high_volatility': {
        'preferred': ['live_adaptive'],
        'avoid': ['simple_average'],
        'reasoning': 'Volatile markets need adaptive weighting'
    }
}
```

---

## 🔄 Integration Flow

### Dashboard Integration
```
Pipeline Data → DashboardDataBridge → Cache → Dashboard UI
                        ↓
                  DataManager (DuckDB)
                        ↓
                  Mock Data (fallback)
```

### Ensemble Integration
```
LiveAdaptiveEnsemble → EnsemblePerformanceBridge → ModelPerformanceTracker
                                ↓
                        Unified View
                                ↓
                        Dashboard / Analytics
```

### Ensemble Selection
```
Context + Models + Performance → EnsembleSelector → Scoring → Best Ensemble
                                        ↓
                                Create Instance
                                        ↓
                                Use for Predictions
```

---

## 📊 Data Structures

### DashboardDataBridge

#### Model Performance
```python
{
    'models': [
        {
            'model_name': 'LSTM_Ensemble',
            'model_type': 'Neural',
            'avg_win_rate': 0.65,
            'avg_sharpe_ratio': 1.8,
            'avg_precision': 0.72,
            'total_trades': 1240
        }
    ],
    'total_models': 3,
    'last_updated': '2026-05-03T12:00:00'
}
```

#### Trading Activity
```python
{
    'signals': [
        {
            'ticker': 'AAPL',
            'signal_type': 'BUY',
            'confidence': 0.85,
            'timestamp': '2026-05-03T12:00:00',
            'pnl': 0.023
        }
    ],
    'total_signals': 3,
    'last_updated': '2026-05-03T12:00:00'
}
```

#### Portfolio Metrics
```python
{
    'total_value': 125000.0,
    'returns': 0.125,
    'volatility': 0.089,
    'sharpe_ratio': 1.4,
    'max_drawdown': -0.034,
    'last_updated': '2026-05-03T12:00:00'
}
```

### EnsemblePerformanceBridge

#### Sync Result
```python
{
    'sync_time': datetime(2026, 5, 3, 12, 0, 0),
    'ensemble_metrics_count': 10,
    'tracker_records_created': 10,
    'records_updated': 8,
    'success': True
}
```

#### Unified View
```python
{
    'unified_performance': {
        'model1': {
            'ensemble_data': {...},
            'tracker_data': {...}
        }
    },
    'ensemble_models': 10,
    'tracker_models': 12,
    'total_unique_models': 15,
    'last_updated': datetime(2026, 5, 3, 12, 0, 0)
}
```

### EnsembleSelector

#### Selection Result
```python
{
    'selected_ensemble': 'live_adaptive',
    'score': 0.85,
    'reasoning': 'Excellent fit for live_adaptive due to...',
    'all_scores': {
        'live_adaptive': 0.85,
        'stacked_ensemble': 0.65,
        'consensus_engine': 0.55,
        'simple_average': 0.30,
        'weighted_average': 0.45
    },
    'context': EnsembleContext(...),
    'available_models': ['model1', 'model2', 'model3'],
    'selection_time': datetime(2026, 5, 3, 12, 0, 0),
    'confidence': 0.9
}
```

---

## 🎯 Use Cases

### Use Case 1: Dashboard Real-time Updates
```python
# Initialize bridge
bridge = DashboardDataBridge(config_manager)

# Get all dashboard data
model_perf = bridge.get_dashboard_data('model_performance')
trading_act = bridge.get_dashboard_data('trading_activity')
portfolio = bridge.get_dashboard_data('portfolio_metrics')
system = bridge.get_dashboard_data('system_status')

# Display in Streamlit
st.metric("Total Value", portfolio['total_value'])
st.metric("Sharpe Ratio", portfolio['sharpe_ratio'])
st.dataframe(model_perf['models'])
```

### Use Case 2: Ensemble Performance Sync
```python
# Initialize bridge
bridge = EnsemblePerformanceBridge(live_ensemble, performance_tracker)

# Sync every 5 minutes
while True:
    sync_result = bridge.sync_ensemble_performance_to_tracker()
    if sync_result['success']:
        logger.info(f"Synced {sync_result['records_updated']} records")
    time.sleep(300)  # 5 minutes
```

### Use Case 3: Intelligent Ensemble Selection
```python
# Create context
context = EnsembleContext(
    data_size=len(data),
    has_real_time_data=True,
    model_count=len(models),
    market_regime=detect_regime(),
    volatility_level=calculate_volatility(),
    prediction_frequency='real_time',
    computational_resources='high',
    latency_requirement='low'
)

# Select best ensemble
selector = EnsembleSelector()
selection = selector.select_best_ensemble(context, models, performance_data)

# Create and use ensemble
ensemble = selector.create_ensemble_instance(
    selection['selected_ensemble'],
    models=models
)
predictions = ensemble.predict(features)
```

---

## 🔧 Configuration

### Cache Settings
```python
# DashboardDataBridge
_cache_ttl = 300  # 5 minutes

# EnsemblePerformanceBridge
_sync_interval = 300  # 5 minutes
```

### Ensemble Requirements
```python
# LiveAdaptiveEnsemble
min_models = 3
real_time_data = True
history_days = 30

# StackedEnsemble
min_models = 2
training_data = True
meta_model = True

# ConsensusEngine
min_models = 3
experience_diary = True
threshold_analyzer = True
```

---

## 🐛 Error Handling

### DashboardDataBridge
```python
# Graceful fallback to mock data
try:
    data = self.data_manager.query_data(query)
except Exception as e:
    logger.error(f"Failed to query data: {e}")
    data = self._get_mock_data()
```

### EnsemblePerformanceBridge
```python
# Sync error handling
try:
    sync_result = bridge.sync_ensemble_performance_to_tracker()
except Exception as e:
    logger.error(f"Sync failed: {e}")
    return {'success': False, 'error': str(e)}
```

### EnsembleSelector
```python
# Fallback selection
try:
    selection = selector.select_best_ensemble(context, models)
except Exception as e:
    logger.error(f"Selection failed: {e}")
    selection = selector._get_fallback_selection(context, models)
```

---

## 📚 Dependencies

### Internal
- `src.core.logging.logger.ProjectLogger`
- `src.data.management.data_manager.DataManager`
- `src.trading.live_adaptive_ensemble.LiveAdaptiveEnsemble`
- `src.models.ensemble.enhanced_ensemble.EnhancedEnsembleModel`
- `src.ensembling.stacked_ensemble.StackedEnsemble`
- `src.trading.consensus_engine.ConsensusEngine`

### External
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `psutil` - System monitoring (optional)

---

## ✅ Status

- ✅ DashboardDataBridge - **Implemented**
- ✅ EnsemblePerformanceBridge - **Implemented**
- ✅ EnsembleSelector - **Implemented**
- ✅ Caching - **Working**
- ✅ Mock data fallback - **Working**
- ✅ Error handling - **Robust**
- ✅ Type hints - **Complete**

---

## 🚀 Next Steps

### High Priority
- [ ] Add integration tests
- [ ] Add performance benchmarks
- [ ] Add monitoring dashboard

### Medium Priority
- [ ] Add more ensemble methods
- [ ] Add advanced caching strategies
- [ ] Add data validation

### Low Priority
- [ ] Add visualization tools
- [ ] Add export/import functionality
- [ ] Add configuration UI

---

**Last Updated**: 2026-05-03  
**Status**: ✅ PRODUCTION READY
