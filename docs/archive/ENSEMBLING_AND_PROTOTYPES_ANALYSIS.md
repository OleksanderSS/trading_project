# 🎯 Ensembling & Prototypes Analysis

## Date: 2026-05-03

---

## 📊 Overview

Комплексний аналіз **ensembling** (ансамблювання моделей) та **prototypes** (прототипів) у системі, їх ролі, інтеграції та використання в гібридному пайплайні.

---

## 🔄 Part 1: Ensembling Architecture

### 1.1 Core Components

#### **StackedEnsemble** (`src/ensembling/stacked_ensemble.py`)

**Purpose**: Meta-модель для оптимального комбінування прогнозів базових моделей

**Key Features**:
- ✅ Ridge regression як meta-learner (запобігає overfitting)
- ✅ Live Efficiency Weighting через Meta-Learning
- ✅ Динамічна адаптація ваг на основі контексту
- ✅ Інтеграція з Experience Diary для історичної ефективності
- ✅ Divergence detection (виявлення розбіжностей між моделями)
- ✅ Confidence adjustment (зниження впевненості при високій розбіжності)

**Architecture**:
```python
StackedEnsemble
├── meta_model: Ridge(alpha=1.0)  # Meta-learner
├── diary_engine: DiaryEngine     # Historical performance
├── feature_names: List[str]      # Base model names
└── is_trained: bool              # Training status
```

**Methods**:
- `train(X, y)` - Тренує meta-модель на прогнозах базових моделей
- `predict(X, context_params)` - Генерує ensemble прогноз з динамічними вагами
- `save(path)` / `load(path)` - Збереження/завантаження стану

**Dynamic Weighting Logic**:
```python
# 1. Retrieve recent performance from Experience Diary
live_stats = diary_engine.get_recent_performance(models, context, window=20)

# 2. Adjust weights based on accuracy
if accuracy < 0.5:
    weight *= 0.5  # Penalty for poor performance
if is_champion:
    weight *= 1.5  # Bonus for champion models

# 3. Normalize weights
weights /= sum(weights)

# 4. Calculate divergence
divergence = std(predictions)

# 5. Adjust confidence
if divergence > 0.7:
    confidence *= 0.3  # Reduce confidence on high disagreement
```

**Output**: `EnsembleResult`
```python
EnsembleResult(
    final_signal: np.ndarray,      # Weighted prediction
    confidence: np.ndarray,         # Adjusted confidence
    divergence: np.ndarray,         # Model disagreement
    active_weights: Dict[str, float],  # Applied weights
    stats: Dict[str, Any]           # Metadata
)
```

---

#### **ensemble_forecast()** Function

**Purpose**: Advanced ensemble forecast з regime-based weights

**Parameters**:
- `model_predictions` - Dict[model_name, predictions]
- `model_confidences` - Optional confidence scores
- `weights` - Optional base weights
- `market_regime` - Current market regime ('bull', 'bear', 'sideways')
- `regime_configs` - Regime-specific weight configurations
- `method` - 'weighted', 'mean', 'median'
- `divergence_shrinkage` - Apply divergence penalty
- `rolling_window` - Smoothing window

**Process Flow**:
```
1. Determine Regime Weights
   ↓
2. Apply Weight Constraints (min/max)
   ↓
3. Align Predictions & Confidences
   ↓
4. Calculate Effective Weights (base * confidence)
   ↓
5. Generate Ensemble Signal (weighted/mean/median)
   ↓
6. Apply Divergence Penalty (if enabled)
   ↓
7. Apply Smoothing (rolling window)
   ↓
8. Calculate Final Confidence
   ↓
9. Return EnsembleResult
```

**Regime-Based Weighting**:
```python
regime_configs = {
    'bull': {'lgbm': 0.4, 'lstm': 0.3, 'xgboost': 0.3},
    'bear': {'lgbm': 0.3, 'lstm': 0.4, 'xgboost': 0.3},
    'sideways': {'lgbm': 0.35, 'lstm': 0.35, 'xgboost': 0.3}
}
```

---

### 1.2 Integration with Pipeline

#### **Stage 5: Prediction** → **Ensembling** → **Stage 6: Signal Generation**

```
Stage 5 (Prediction)
├── LGBM predictions
├── LSTM predictions
├── XGBoost predictions
└── DEAN predictions
    ↓
StackedEnsemble.predict()
├── Load context from Experience Diary
├── Adjust weights dynamically
├── Calculate divergence
└── Generate consensus signal
    ↓
Stage 6 (Signal Generation)
└── Use ensemble signal for trading decisions
```

**Code Example**:
```python
# Stage 5: Collect predictions
model_predictions = {
    'lgbm': lgbm_model.predict(X),
    'lstm': lstm_model.predict(X),
    'xgboost': xgb_model.predict(X),
    'dean': dean_model.predict(X)
}

# Ensembling
ensemble = StackedEnsemble(config_manager=config)
result = ensemble.predict(
    X=pd.DataFrame(model_predictions),
    context_params={
        'ticker': 'AMD',
        'tf': '15m',
        'regime': 'bull_market'
    }
)

# Stage 6: Use result
final_signal = result.final_signal
confidence = result.confidence
divergence = result.divergence
```

---

### 1.3 Meta-Learning Integration

#### **DiaryEngine** (`src/meta_learning/memory/diary_engine.py`)

**Purpose**: System's memory for tracking trade performance and context

**Key Features**:
- ✅ DuckDB-based persistent storage
- ✅ Context Map 2.0 support (30+ drivers)
- ✅ Decision recording (BUY/SELL/HOLD)
- ✅ Outcome tracking (PROFITABLE/UNPROFITABLE/PENDING)
- ✅ Context vulnerability analysis
- ✅ Agent comparison and promotion recommendations
- ✅ Contextual model weights

**Schema**:
```sql
CREATE TABLE experience_diary (
    id INTEGER PRIMARY KEY,
    agent_id VARCHAR NOT NULL,
    decision_timestamp BIGINT NOT NULL,
    ticker VARCHAR NOT NULL,
    decision_type VARCHAR NOT NULL,
    reasoning VARCHAR,
    market_context VARCHAR,        -- JSON string
    context_fingerprint VARCHAR,   -- Tri-state drivers map
    model_prediction DOUBLE,
    model_confidence DOUBLE,
    entry_price DOUBLE,
    exit_price DOUBLE,
    outcome VARCHAR NOT NULL,
    profit_loss DOUBLE
)
```

**Key Methods**:
- `record_decision(decision)` - Record trading decision
- `get_recent_performance(models, context, window)` - Get model performance
- `get_context_vulnerability(agent_id)` - Find failure patterns
- `get_context_success_analysis(agent_id)` - Find success patterns
- `get_contextual_model_weights(context_fingerprint)` - Get context-specific weights
- `compare_agents(agent_ids)` - Compare agent performance

**Integration with StackedEnsemble**:
```python
# StackedEnsemble uses DiaryEngine for dynamic weighting
live_stats = self.diary_engine.get_recent_performance(
    models=self.feature_names,
    context=context_fingerprint,
    window=20
)

# Adjust weights based on historical performance
for model_name in models:
    accuracy = live_stats.get(model_name, {}).get('accuracy', 0.5)
    if accuracy < 0.5:
        weight *= 0.5  # Penalty
    if is_champion:
        weight *= 1.5  # Bonus
```

---

### 1.4 Experiment: CompareLayersExperiment

**Purpose**: Порівняння комбінацій feature layers та ensemble methods

**Location**: `src/experiments/compare_layers.py`

**Dimensions**:
- **Tickers**: All from assets.yaml
- **Timeframes**: 15m, 1h, 1d
- **Market Regimes**: Bull, Bear, Sideways, All
- **Ensemble Methods**: weighted, mean, median
- **Feature Layers**: All combinations (1-3 layers)

**Process**:
```
1. Generate Test Cases
   ↓
2. Run Pipeline with Specific Layers
   ↓
3. Apply Ensemble Method
   ↓
4. Calculate Performance Metrics
   ↓
5. Save Best Configurations to Experience Diary
   ↓
6. Generate Visualizations
```

**Metrics**:
- Total Return
- Sharpe Ratio
- Max Drawdown
- Win Rate

**Output**:
- Best layer combinations per ticker/timeframe/regime
- Performance heatmaps
- Ensemble method comparison plots

**Usage**:
```bash
python src/experiments/compare_layers.py
```

---

## 🧪 Part 2: Prototypes

### 2.1 Current Prototypes

#### **LiveTradingTickerManager** (`src/devtools/prototypes/live_trading_ticker_manager.py`)

**Status**: ⚠️ NON-FUNCTIONAL PROTOTYPE

**Purpose**: Intelligent ticker selection for live trading

**Planned Features**:
- Market condition analysis
- Base strategy ticker selection
- Trending ticker detection from news
- Ticker scoring (volatility, momentum, news, sector, liquidity)
- Resource optimization
- Dynamic ticker updates during session

**Architecture**:
```python
LiveTradingTickerManager
├── analyze_market_conditions() → MarketCondition
├── get_base_strategy_tickers() → List[str]
├── get_trending_tickers() → List[str]
├── score_tickers() → List[TickerScore]
├── optimize_for_resources() → List[TickerScore]
└── get_optimal_tickers_for_live_trading() → (List[str], Dict)
```

**MarketCondition**:
```python
@dataclass
class MarketCondition:
    volatility_level: float      # 0-1
    trend_direction: str          # 'bull', 'bear', 'sideways'
    volume_level: float           # 0-1
    news_intensity: float         # 0-1
    sector_rotation: str          # 'tech', 'finance', 'energy', 'balanced'
    market_phase: str             # 'pre_market', 'regular', 'after_hours'
```

**TickerScore**:
```python
@dataclass
class TickerScore:
    ticker: str
    volatility_score: float
    momentum_score: float
    news_score: float
    sector_score: float
    liquidity_score: float
    total_score: float
    recommended_position_size: float
    optimal_timeframes: List[str]
```

**Current Status**:
- ❌ All methods raise `NotImplementedError`
- ❌ Not integrated into pipeline
- ⚠️ Logs critical warnings when used

**Recommendation**: 
- Either implement or remove
- If implementing, integrate with:
  - `src/features/nlp/extractors/news_ticker_detector.py`
  - `config/enhanced_sector_tickers.py`
  - `src/analytics/context/market_phase_analyzer.py`

---

#### **PatternAwareModelTrainer** (`src/training/pattern_aware_training.py`)

**Status**: ⚠️ NON-FUNCTIONAL PROTOTYPE

**Purpose**: Intelligent model training incorporating patterns from Stages 1-3

**Planned Features**:
- Market condition analysis
- Pattern-aware training
- Historical pattern integration
- Real-time pattern adaptation

**Current Status**:
- ❌ `train_pattern_aware_models()` raises `NotImplementedError`
- ⚠️ Logs warnings on initialization

**Recommendation**: 
- Either implement or remove
- If implementing, integrate with:
  - `src/patterns/` module
  - `src/analytics/context/market_phase_analyzer.py`

---

### 2.2 Prototype Management Strategy

#### **Option 1: Implement Prototypes**

**LiveTradingTickerManager Implementation Plan**:
```
1. Implement analyze_market_conditions()
   - Use MarketPhaseAnalyzer
   - Calculate volatility from recent data
   - Detect trend direction
   
2. Implement get_base_strategy_tickers()
   - Load from enhanced_sector_tickers
   - Filter by market conditions
   
3. Implement get_trending_tickers()
   - Use NewsTickerDetector
   - Analyze news sentiment
   
4. Implement score_tickers()
   - Calculate composite scores
   - Weight by market conditions
   
5. Implement optimize_for_resources()
   - Limit to max_tickers
   - Balance sectors
   
6. Integrate into pipeline
   - Add to Stage 0 (Setup)
   - Use in live trading mode
```

**Estimated Effort**: 2-3 days

---

#### **Option 2: Remove Prototypes**

**Cleanup Plan**:
```bash
# Delete prototype files
rm src/devtools/prototypes/live_trading_ticker_manager.py
rm src/training/pattern_aware_training.py

# Remove empty prototypes directory
rmdir src/devtools/prototypes

# Update imports (if any)
grep -r "live_trading_ticker_manager" src/
grep -r "pattern_aware_training" src/
```

**Benefits**:
- ✅ Clean codebase
- ✅ No confusion about functionality
- ✅ Easier navigation

**Drawbacks**:
- ❌ Lose design ideas
- ❌ Need to recreate if needed later

---

#### **Option 3: Document and Archive**

**Archive Plan**:
```
1. Move to docs/prototypes/
   - live_trading_ticker_manager.md
   - pattern_aware_training.md
   
2. Document design ideas
   - Architecture
   - Integration points
   - Implementation plan
   
3. Remove code files
   
4. Add to MIGRATION_NOTES.md
```

**Benefits**:
- ✅ Preserve design ideas
- ✅ Clean codebase
- ✅ Easy to reference later

---

## 🎯 Part 3: Integration with Hybrid Pipeline

### 3.1 Current Integration

#### **Ensembling in Pipeline**:

```python
# run_hybrid_pipeline.py
orchestrator = HybridOrchestrator(config_manager, batch_name)

# Stage 5: Prediction
predictions = orchestrator.run_stage_5_prediction(tickers, timeframes)

# Ensembling (implicit in Stage 5)
ensemble_results = {}
for ticker in tickers:
    model_preds = {
        'lgbm': predictions[ticker]['lgbm'],
        'lstm': predictions[ticker]['lstm'],
        'xgboost': predictions[ticker]['xgboost']
    }
    
    ensemble = StackedEnsemble(config_manager)
    result = ensemble.predict(
        X=pd.DataFrame(model_preds),
        context_params={
            'ticker': ticker,
            'tf': timeframe,
            'regime': market_regime
        }
    )
    
    ensemble_results[ticker] = result

# Stage 6: Signal Generation
signals = orchestrator.run_stage_6_signals(ensemble_results)
```

---

### 3.2 Proposed Enhancements

#### **Enhancement 1: Explicit Ensemble Stage**

**Add Stage 5.5: Ensembling**:
```python
# src/pipeline/stages/stage_5_5_ensembling.py

class Stage5_5Ensembling:
    """Explicit ensembling stage between prediction and signal generation"""
    
    def run(self, predictions: Dict, context: Dict) -> Dict:
        """
        Args:
            predictions: {ticker: {model: predictions}}
            context: {ticker: {regime, volatility, etc.}}
        
        Returns:
            {ticker: EnsembleResult}
        """
        ensemble_results = {}
        
        for ticker, model_preds in predictions.items():
            # Get context
            ticker_context = context.get(ticker, {})
            regime = ticker_context.get('regime', 'unknown')
            
            # Create ensemble
            ensemble = StackedEnsemble(self.config_manager)
            
            # Predict with context
            result = ensemble.predict(
                X=pd.DataFrame(model_preds),
                context_params={
                    'ticker': ticker,
                    'tf': ticker_context.get('timeframe', '15m'),
                    'regime': regime
                }
            )
            
            ensemble_results[ticker] = result
            
            # Log to Experience Diary
            self.diary_engine.record_decision(
                DecisionRecord(
                    agent_id='ensemble',
                    ticker=ticker,
                    decision_type=DecisionType.METADATA,
                    reasoning=f"Ensemble prediction (regime: {regime})",
                    market_context=ticker_context,
                    context_fingerprint=self._create_fingerprint(ticker_context),
                    model_prediction=float(result.final_signal.mean()),
                    model_confidence=float(result.confidence.mean()),
                    outcome=DecisionOutcome.PENDING
                )
            )
        
        return ensemble_results
```

**Integration**:
```python
# src/pipeline/hybrid_orchestrator.py

async def run_full_pipeline(self, tickers, timeframes):
    # ... Stages 0-4 ...
    
    # Stage 5: Prediction
    predictions = await self.run_stage_5_prediction(tickers, timeframes)
    
    # Stage 5.5: Ensembling (NEW)
    ensemble_results = await self.run_stage_5_5_ensembling(predictions, context)
    
    # Stage 6: Signal Generation
    signals = await self.run_stage_6_signals(ensemble_results)
    
    # ... Stages 7-8 ...
```

---

#### **Enhancement 2: Regime-Aware Ensemble Configs**

**Add to config/ensembling.yaml**:
```yaml
ensembling:
  meta_model:
    type: "ridge"
    alpha: 1.0
  
  regime_weights:
    bull_market:
      lgbm: 0.35
      lstm: 0.30
      xgboost: 0.25
      dean: 0.10
    
    bear_market:
      lgbm: 0.25
      lstm: 0.35
      xgboost: 0.30
      dean: 0.10
    
    sideways:
      lgbm: 0.30
      lstm: 0.30
      xgboost: 0.30
      dean: 0.10
    
    high_volatility:
      lgbm: 0.20
      lstm: 0.40
      xgboost: 0.25
      dean: 0.15
  
  divergence:
    threshold: 0.7
    confidence_penalty: 0.3
    shrinkage_enabled: true
  
  smoothing:
    rolling_window: 5
    fill_na: 0.0
  
  weight_constraints:
    min_weight: 0.0
    max_weight: 0.8
```

---

#### **Enhancement 3: Ensemble Performance Tracking**

**Add to Experience Diary**:
```python
# src/meta_learning/memory/diary_engine.py

def get_ensemble_performance(self, window: int = 100) -> Dict[str, Any]:
    """Get ensemble performance metrics"""
    query = f"""
    SELECT 
        AVG(CASE WHEN outcome = 'profitable' THEN 1.0 ELSE 0.0 END) as win_rate,
        AVG(profit_loss) as avg_pnl,
        AVG(model_confidence) as avg_confidence,
        COUNT(*) as total_decisions
    FROM {self.table_name}
    WHERE agent_id = 'ensemble'
    ORDER BY decision_timestamp DESC
    LIMIT {window}
    """
    
    result = pd.DataFrame(self.data_manager.fetch_all(query))
    
    return {
        'win_rate': float(result['win_rate'].iloc[0]),
        'avg_pnl': float(result['avg_pnl'].iloc[0]),
        'avg_confidence': float(result['avg_confidence'].iloc[0]),
        'total_decisions': int(result['total_decisions'].iloc[0])
    }

def compare_ensemble_vs_individual(self) -> Dict[str, Any]:
    """Compare ensemble performance vs individual models"""
    # Get ensemble performance
    ensemble_perf = self.get_ensemble_performance()
    
    # Get individual model performance
    models = ['lgbm', 'lstm', 'xgboost', 'dean']
    individual_perf = {}
    
    for model in models:
        query = f"""
        SELECT 
            AVG(CASE WHEN outcome = 'profitable' THEN 1.0 ELSE 0.0 END) as win_rate,
            AVG(profit_loss) as avg_pnl
        FROM {self.table_name}
        WHERE agent_id = '{model}'
        ORDER BY decision_timestamp DESC
        LIMIT 100
        """
        result = pd.DataFrame(self.data_manager.fetch_all(query))
        individual_perf[model] = {
            'win_rate': float(result['win_rate'].iloc[0]),
            'avg_pnl': float(result['avg_pnl'].iloc[0])
        }
    
    return {
        'ensemble': ensemble_perf,
        'individual': individual_perf,
        'ensemble_advantage': {
            'win_rate_improvement': ensemble_perf['win_rate'] - max(m['win_rate'] for m in individual_perf.values()),
            'pnl_improvement': ensemble_perf['avg_pnl'] - max(m['avg_pnl'] for m in individual_perf.values())
        }
    }
```

---

## 📋 Part 4: Action Items

### 4.1 Immediate Actions

#### **1. Decide on Prototypes**
- [ ] Review LiveTradingTickerManager design
- [ ] Decide: Implement / Remove / Archive
- [ ] Review PatternAwareModelTrainer design
- [ ] Decide: Implement / Remove / Archive

#### **2. Enhance Ensembling**
- [ ] Add explicit Stage 5.5 (Ensembling)
- [ ] Create config/ensembling.yaml
- [ ] Add regime-aware ensemble configs
- [ ] Implement ensemble performance tracking

#### **3. Improve Meta-Learning Integration**
- [ ] Add `get_ensemble_performance()` to DiaryEngine
- [ ] Add `compare_ensemble_vs_individual()` to DiaryEngine
- [ ] Implement contextual weight caching
- [ ] Add ensemble decision logging

---

### 4.2 Testing Plan

#### **Test 1: Ensemble Performance**
```bash
# Run with ensemble tracking
python run_hybrid_pipeline.py --mode prepare --test-ticker AMD --test-target target_return_1d

# Check ensemble results
python scripts/analyze_ensemble_performance.py --batch-name test_batch
```

#### **Test 2: Regime-Aware Weighting**
```bash
# Test different regimes
python scripts/test_regime_weighting.py --ticker AMD --regime bull_market
python scripts/test_regime_weighting.py --ticker AMD --regime bear_market
python scripts/test_regime_weighting.py --ticker AMD --regime sideways
```

#### **Test 3: CompareLayersExperiment**
```bash
# Run layer comparison experiment
python src/experiments/compare_layers.py

# Check results
ls results/experiments/CompareLayers/
```

---

### 4.3 Documentation Updates

#### **1. Update README.md**
- [ ] Add Ensembling section
- [ ] Document Stage 5.5
- [ ] Add ensemble configuration guide

#### **2. Create Ensemble Guide**
- [ ] `docs/ENSEMBLING_GUIDE.md`
- [ ] Architecture overview
- [ ] Configuration examples
- [ ] Performance analysis

#### **3. Update MIGRATION_NOTES.md**
- [ ] Document prototype decisions
- [ ] Document ensembling enhancements
- [ ] Add testing results

---

## 🎯 Part 5: Summary

### 5.1 Current State

#### **Ensembling**:
- ✅ StackedEnsemble implemented with Ridge meta-learner
- ✅ Dynamic weighting via Experience Diary
- ✅ Divergence detection and confidence adjustment
- ✅ Regime-aware ensemble_forecast() function
- ✅ Integration with Meta-Learning (DiaryEngine)
- ⚠️ No explicit Stage 5.5 (implicit in Stage 5)
- ⚠️ No ensemble-specific config file
- ⚠️ Limited ensemble performance tracking

#### **Prototypes**:
- ⚠️ LiveTradingTickerManager - Non-functional, needs decision
- ⚠️ PatternAwareModelTrainer - Non-functional, needs decision
- ❌ No clear prototype management strategy

#### **Meta-Learning**:
- ✅ DiaryEngine with DuckDB persistence
- ✅ Context Map 2.0 support (30+ drivers)
- ✅ Contextual model weights
- ✅ Agent comparison and promotion
- ✅ Context vulnerability analysis
- ⚠️ No ensemble-specific performance tracking

---

### 5.2 Recommendations

#### **Priority 1: Prototypes Decision**
1. **Review** both prototypes
2. **Decide** for each: Implement / Remove / Archive
3. **Execute** decision within 1 week

#### **Priority 2: Ensembling Enhancement**
1. **Add** explicit Stage 5.5 (Ensembling)
2. **Create** config/ensembling.yaml
3. **Implement** ensemble performance tracking
4. **Test** with real data

#### **Priority 3: Documentation**
1. **Create** docs/ENSEMBLING_GUIDE.md
2. **Update** README.md with ensembling section
3. **Document** prototype decisions in MIGRATION_NOTES.md

---

### 5.3 Expected Benefits

#### **After Ensembling Enhancement**:
- ✅ Explicit ensemble stage for better visibility
- ✅ Regime-aware weighting for better adaptation
- ✅ Performance tracking for continuous improvement
- ✅ Better integration with Meta-Learning

#### **After Prototype Decision**:
- ✅ Clean codebase (if removed)
- ✅ New functionality (if implemented)
- ✅ Clear roadmap (if archived)

#### **After Documentation**:
- ✅ Clear understanding of ensembling architecture
- ✅ Easy configuration and tuning
- ✅ Better onboarding for new developers

---

## 📚 References

- **Ensembling**: `src/ensembling/`
- **Meta-Learning**: `src/meta_learning/`
- **Experiments**: `src/experiments/compare_layers.py`
- **Prototypes**: `src/devtools/prototypes/`
- **Pipeline**: `src/pipeline/hybrid_orchestrator.py`
- **Config**: `src/config/`

---

## 🔗 Related Documents

- `docs/DATA_ACCUMULATION_STRATEGY.md` - Data strategy
- `docs/CALIBRATION_GUIDE.md` - DEAN calibration
- `MIGRATION_NOTES.md` - Migration history
- `src/ensembling/README.md` - Ensembling overview
- `src/meta_learning/README.md` - Meta-learning overview

