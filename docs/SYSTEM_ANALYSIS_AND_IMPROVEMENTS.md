# 🔍 System Analysis & Improvement Recommendations

**Date:** 2026-04-25  
**Analyst:** Kiro AI  
**Project:** Trading ML Pipeline

---

## 📊 Current State Analysis

### Data Structure

#### Features Dataset
```
Total rows: 792
Total columns: 291
├─ Base features: 107 (OHLCV, indicators, macro, news)
├─ State features: 179 (context states: state_*)
├─ Context metadata: 2 (context_fingerprint, context_stability)
└─ Metadata: 3 (ticker, datetime, interval)
```

#### Targets Dataset
```
Total rows: 514
Total columns: 176
├─ Target columns: ~16 (target_return_1d, target_rsi_f1, etc.)
└─ Metadata: 160 (includes duplicated features + state_*)
```

### ⚠️ Critical Issues Identified

#### 1. **Data Leakage in Targets DataFrame**
**Severity:** 🔴 CRITICAL

**Problem:**
```python
# targets.parquet містить:
- Всі base features (SMA, EMA, RSI, MACD, etc.)
- Всі state features (state_SMA, state_RSI, etc.)
- Всі FRED макро дані
```

**Impact:**
- Targets DataFrame має 160 колонок, які НЕ є таргетами
- Це дублювання даних (features вже є в features.parquet)
- Ризик data leakage при merge

**Root Cause:**
Target generation додає features до targets DataFrame замість створення чистого targets-only DataFrame.

**Solution:**
```python
# В TargetOrchestrator або TargetCalculator:
def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
    """Generate ONLY target columns."""
    targets = {}
    
    for target_name, target_config in self.targets_config.items():
        targets[target_name] = self._calculate_target(df, target_config)
    
    # Return ONLY targets + minimal metadata
    return pd.DataFrame({
        'datetime': df['datetime'],
        'ticker': df['ticker'],
        'interval': df['interval'],
        **targets  # Only target columns!
    })
```

---

#### 2. **Missing Market Context Features**
**Severity:** 🟡 MEDIUM

**Problem:**
```python
# MarketContextEnricher не створює колонки:
market_context_volume_ratio  # ← Потрібно для target_volume_ratio_f1
market_context_rsi_current
market_context_volatility_5d
# ... etc
```

**Impact:**
- Target `target_volume_ratio_f1` не може бути розрахований
- Втрата 18 важливих context features
- Модель не має доступу до агрегованих ринкових метрик

**Root Cause:**
MarketContextEnricher не в списку активних enrichers або не запускається через помилку.

**Solution:**
```python
# 1. Перевірити, чи enricher в конфігурації
# src/config/enrichment.yaml
enrichment:
  market_context:
    module: "src.features.enrichers.market_context_enricher"
    class: "MarketContextEnricher"

# 2. Додати в список enrichers для автоматичного виявлення
# src/features/feature_orchestrator.py
ENRICHER_MODULES = [
    "src.features.enrichers.technical_enricher",
    "src.features.enrichers.market_context_enricher",  # ← Додати!
    # ...
]
```

---

#### 3. **Inefficient Context Map Storage**
**Severity:** 🟢 LOW

**Problem:**
```python
# context_fingerprint зберігається як string:
"-1|-1|1|0|0|-1|1|0|0|0|..."  # 179 states × 2 chars = ~358 bytes per row

# Це неефективно для:
- Пошуку схожих контекстів
- Кластеризації
- Аналізу
```

**Impact:**
- Повільний пошук схожих контекстів
- Складно робити аналітику
- Неможливо використати для similarity search

**Solution:**
```python
# Додати векторне представлення:
class ContextMapEnricher:
    def _enrich_impl(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... existing code ...
        
        # Додати векторне представлення
        context_vector = self._fingerprint_to_vector(
            df['context_fingerprint']
        )
        df['context_vector'] = context_vector.tolist()
        
        return df
    
    def _fingerprint_to_vector(self, fingerprints: pd.Series) -> np.ndarray:
        """Convert fingerprint string to numeric vector."""
        vectors = []
        for fp in fingerprints:
            vector = [int(x) for x in fp.split('|')]
            vectors.append(vector)
        return np.array(vectors)
```

---

#### 4. **State Features Duplication**
**Severity:** 🟡 MEDIUM

**Problem:**
```python
# State features дублюються в обох DataFrame:
features.parquet:  179 state_* columns
targets.parquet:   179 state_* columns (дублікат!)
```

**Impact:**
- Подвійне зберігання (2× storage)
- Ризик inconsistency при merge
- Плутанина при feature selection

**Solution:**
```python
# Зберігати state_* тільки в features.parquet
# targets.parquet має містити ТІЛЬКИ таргети + мінімальні metadata

# При merge в Colab:
merged = features_df.merge(
    targets_df[['datetime', 'ticker', 'interval'] + target_columns],
    on=['datetime', 'ticker', 'interval'],
    how='inner'
)
```

---

## 🎯 Architecture Improvements

### 1. **Clean Separation of Concerns**

#### Current (Problematic):
```
features.parquet: 291 columns
├─ Base features (107)
├─ State features (179)
├─ Context metadata (2)
└─ Metadata (3)

targets.parquet: 176 columns
├─ Targets (~16)
└─ EVERYTHING ELSE (160) ← PROBLEM!
```

#### Recommended:
```
features.parquet: 291 columns
├─ Base features (107)
├─ State features (179)
├─ Context metadata (2)
└─ Metadata (3)

targets.parquet: 19 columns
├─ Targets (16)
└─ Metadata (3: datetime, ticker, interval)
```

---

### 2. **Context-Aware Feature Selection**

#### Implementation:
```python
# src/features/context_aware_feature_selector.py (вже створено!)

selector = ContextAwareFeatureSelector(
    method='mutual_info',  # або 'random_forest'
    top_k=50
)

selected_features, analysis = selector.select_features(
    X=features_df[all_features],  # 107 base + 179 state = 286
    y=targets_df[target]
)

# analysis містить:
{
    'base_count': 35,           # Базових features
    'context_count': 12,        # State features
    'temporal_count': 3,        # Часових features
    'uses_context': True,
    'context_ratio': 0.24,      # 24% context features
    'top_context_features': [
        {'name': 'state_RSI', 'importance': 0.0234},
        {'name': 'state_MACD', 'importance': 0.0189},
        ...
    ]
}
```

---

### 3. **Model Training with Context Metadata**

#### Implementation:
```python
# src/pipeline/hybrid/model_training_orchestrator.py (вже оновлено!)

metadata = {
    'ticker': ticker,
    'timeframe': timeframe,
    'target': target,
    'model_type': model_type,
    'selected_features': selected_features,
    
    # Context information:
    'uses_context_states': True,
    'context_features_count': 12,
    'context_features': ['state_RSI', 'state_MACD', ...],
    'context_ratio': 0.24,
    
    # Performance metrics:
    'train_mae': 0.0123,
    'val_mae': 0.0145,
    ...
}
```

---

## 🔧 Immediate Action Items

### Priority 1: Fix Data Leakage (CRITICAL)

**File:** `src/targets/target_orchestrator.py` or `src/targets/target_calculator.py`

**Change:**
```python
def generate_targets(self, enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Generate ONLY target columns."""
    
    # Calculate all targets
    targets_dict = {}
    for target_name, target_config in self.targets_config.items():
        targets_dict[target_name] = self._calculate_target(
            enriched_df, target_config
        )
    
    # Return ONLY targets + minimal metadata
    return pd.DataFrame({
        'datetime': enriched_df['datetime'],
        'ticker': enriched_df['ticker'],
        'interval': enriched_df['interval'],
        **targets_dict
    })
```

**Verification:**
```python
# After fix:
targets_df = pd.read_parquet('targets.parquet')
assert len(targets_df.columns) < 20, "Too many columns in targets!"
assert all(c.startswith('target_') or c in ['datetime', 'ticker', 'interval'] 
           for c in targets_df.columns), "Non-target columns found!"
```

---

### Priority 2: Enable MarketContextEnricher

**File:** `src/features/feature_orchestrator.py`

**Change:**
```python
# Add to ENRICHER_MODULES list
ENRICHER_MODULES = [
    "src.features.enrichers.technical_enricher",
    "src.features.enrichers.volatility_enricher",
    "src.features.enrichers.momentum_enricher",
    "src.features.enrichers.volume_enricher",
    "src.features.enrichers.market_context_enricher",  # ← ADD THIS!
    "src.features.enrichers.context_map_enricher",
    # ...
]
```

**Verification:**
```python
# After fix:
features_df = pd.read_parquet('features.parquet')
market_cols = [c for c in features_df.columns if 'market_context' in c]
assert len(market_cols) >= 18, f"Expected 18+ market_context columns, got {len(market_cols)}"
```

---

### Priority 3: Update Colab Cell

**File:** `colab_clean_cell.py`

**Change:**
```python
# Replace ColabFeatureSelector import
from src.features.colab_context_integration import (
    ContextAwareColabFeatureSelector,
    save_feature_analysis,
    visualize_context_importance
)

# In ColabTrainingPipeline.__init__:
self.feature_selector = ContextAwareColabFeatureSelector(
    self.env.PROJECT_PATH
)

# After feature selection:
selected_features, analysis = self.feature_selector.select_features(
    features_df, targets_df, ticker, target_col, model_type
)

# Save analysis
save_feature_analysis(
    analysis, ticker, target_col, model_type,
    output_dir=self.batch_dir / "feature_analysis"
)

# Optional: visualize
visualize_context_importance(
    analysis,
    output_path=self.batch_dir / f"context_importance_{ticker}_{target_col}.png"
)
```

---

## 📈 Performance Optimizations

### 1. **Parallel Feature Selection**

```python
from concurrent.futures import ProcessPoolExecutor

def select_features_parallel(
    features_df, targets_df, 
    ticker_target_pairs, 
    max_workers=4
):
    """Parallel feature selection for multiple (ticker, target) pairs."""
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for ticker, target in ticker_target_pairs:
            future = executor.submit(
                select_features_for_pair,
                features_df, targets_df, ticker, target
            )
            futures.append((ticker, target, future))
        
        results = {}
        for ticker, target, future in futures:
            results[(ticker, target)] = future.result()
    
    return results
```

---

### 2. **Feature Selection Caching**

```python
class CachedFeatureSelector:
    """Feature selector with intelligent caching."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def select_features(self, X, y, cache_key: str):
        """Select features with caching."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Check cache
        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            
            # Verify data signature
            current_sig = self._compute_signature(X, y)
            if cached['signature'] == current_sig:
                return cached['selected_features'], cached['analysis']
        
        # Perform selection
        selected, analysis = self._select(X, y)
        
        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump({
                'signature': self._compute_signature(X, y),
                'selected_features': selected,
                'analysis': analysis
            }, f)
        
        return selected, analysis
```

---

### 3. **Memory-Efficient Data Loading**

```python
def load_data_chunked(
    features_path: Path,
    targets_path: Path,
    chunk_size: int = 10000
):
    """Load data in chunks to reduce memory usage."""
    
    # Load features in chunks
    features_chunks = pd.read_parquet(
        features_path,
        engine='pyarrow',
        use_threads=True
    )
    
    # Load only necessary target columns
    targets_df = pd.read_parquet(
        targets_path,
        columns=['datetime', 'ticker', 'interval'] + target_columns
    )
    
    return features_chunks, targets_df
```

---

## 🎯 Expected Results After Improvements

### Data Structure
```
features.parquet: 291 columns, 792 rows
├─ Base: 107
├─ State: 179
├─ Market Context: 18 (NEW!)
├─ Context metadata: 2
└─ Metadata: 3

targets.parquet: 19 columns, 514 rows (CLEAN!)
├─ Targets: 16
└─ Metadata: 3
```

### Feature Selection
```
✅ Selected 50 features for AMD (target_return_1d):
   Base: 32 (64%)
   Context: 15 (30%)
   Temporal: 3 (6%)
   
   Top context features:
   1. state_RSI: 0.0234
   2. state_MACD: 0.0189
   3. state_volume: 0.0156
   4. state_ATR: 0.0142
   5. state_BB_width: 0.0128
```

### Model Metadata
```json
{
  "ticker": "AMD",
  "target": "target_return_1d",
  "model_type": "mlp",
  "uses_context_states": true,
  "context_features_count": 15,
  "context_ratio": 0.30,
  "train_mae": 0.0123,
  "val_mae": 0.0145
}
```

---

## 📋 Implementation Checklist

### Phase 1: Critical Fixes (Day 1)
- [ ] Fix data leakage in targets DataFrame
- [ ] Enable MarketContextEnricher
- [ ] Verify market_context_* columns created
- [ ] Re-run Stage 3 to regenerate clean data

### Phase 2: Integration (Day 2)
- [ ] Update Colab cell with Context-Aware selector
- [ ] Test feature selection with context analysis
- [ ] Verify metadata includes context information
- [ ] Save feature analysis to JSON

### Phase 3: Optimization (Day 3)
- [ ] Implement parallel feature selection
- [ ] Add feature selection caching
- [ ] Optimize data loading
- [ ] Add visualization of context importance

### Phase 4: Validation (Day 4)
- [ ] Run full pipeline end-to-end
- [ ] Verify no data leakage
- [ ] Check model performance with context features
- [ ] Generate analysis reports

---

## 🚀 Next Steps

1. **Immediate:** Fix data leakage (Priority 1)
2. **Today:** Enable MarketContextEnricher (Priority 2)
3. **Tomorrow:** Update Colab cell (Priority 3)
4. **This Week:** Implement optimizations

---

## 📚 References

- **Context-Aware Selector:** `src/features/context_aware_feature_selector.py`
- **Colab Integration:** `src/features/colab_context_integration.py`
- **Integration Guide:** `docs/CONTEXT_MAP_INTEGRATION.md`
- **Model Training:** `src/pipeline/hybrid/model_training_orchestrator.py`
