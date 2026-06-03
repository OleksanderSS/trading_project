# 📋 Week 1, Day 3-5 Implementation Plan

## 🎯 Goal
Complete Week 1 foundation by implementing:
1. PersistentModelPool (extends ModelPool)
2. ModelQualityController (validation + drift detection)
3. Integration with run_hybrid_pipeline.py

---

## Day 3: PersistentModelPool - Part 1

### File: `src/models/persistent_pool.py`

**Tasks:**
- [ ] Extend ModelPool with persistence layer
- [ ] Add cache index (metadata + quality scores)
- [ ] Implement save/load cache index
- [ ] Add metadata tracking per model

**Implementation:**
```python
class PersistentModelPool(ModelPool):
    def __init__(self, max_models: int = 50, cache_dir: str = ".model_cache"):
        super().__init__(max_models)
        self.cache_dir = Path(cache_dir)
        self.model_metadata: Dict[str, Dict] = {}
        self.quality_scores: Dict[str, float] = {}
        self._load_cache_index()
    
    def add_model_with_metadata(self, model_id, model, metadata, quality_score):
        """Add model with metadata and quality score"""
        pass
    
    def _load_cache_index(self):
        """Load cache index from disk"""
        pass
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        pass
```

**Tests:**
```python
# tests/models/test_persistent_pool.py
def test_persistent_pool_creation():
    pool = PersistentModelPool(max_models=5)
    assert pool.cache_dir.exists()

def test_add_model_with_metadata():
    pool = PersistentModelPool()
    model = MockModel()
    pool.add_model_with_metadata(
        "test_model", model,
        metadata={"version": "1.0"},
        quality_score=0.85
    )
    assert "test_model" in pool.model_metadata
```

**Acceptance Criteria:**
- ✅ Cache index persists to disk
- ✅ Metadata tracked per model
- ✅ Quality scores stored
- ✅ 100% test coverage

---

## Day 4: PersistentModelPool - Part 2

### File: `src/models/persistent_pool.py` (continue)

**Tasks:**
- [ ] Implement warm-up mechanism
- [ ] Add quality check on retrieval
- [ ] Add enhanced statistics
- [ ] Integration tests

**Implementation:**
```python
class PersistentModelPool(ModelPool):
    def warm_up(self, model_ids: List[str], loader_fns: Dict[str, Callable]):
        """Pre-load models into pool"""
        logger.info(f"Warming up {len(model_ids)} models...")
        for model_id in model_ids:
            if model_id in loader_fns:
                self.get_model(model_id, loader_fns[model_id])
    
    def get_model_with_quality_check(self, model_id, loader_fn, min_quality=0.5):
        """Get model with quality validation"""
        model = self.get_model(model_id, loader_fn)
        if not model:
            return None
        
        quality = self.quality_scores.get(model_id, 0.0)
        if quality < min_quality:
            logger.warning(f"Model {model_id} quality {quality} below threshold")
            return None
        
        return model
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics"""
        base_stats = self.get_stats()
        avg_quality = np.mean(list(self.quality_scores.values()))
        return {**base_stats, 'avg_quality': avg_quality}
```

**Tests:**
```python
def test_warm_up():
    pool = PersistentModelPool()
    loader_fns = {"model1": lambda: MockModel(), "model2": lambda: MockModel()}
    pool.warm_up(["model1", "model2"], loader_fns)
    assert len(pool.models) == 2

def test_quality_check():
    pool = PersistentModelPool()
    pool.add_model_with_metadata("low_quality", MockModel(), {}, quality_score=0.3)
    model = pool.get_model_with_quality_check("low_quality", lambda: MockModel(), min_quality=0.5)
    assert model is None  # Should reject low quality
```

**Acceptance Criteria:**
- ✅ Warm-up loads models efficiently
- ✅ Quality check filters low-quality models
- ✅ Enhanced stats include quality metrics
- ✅ Integration tests pass

---

## Day 5: ModelQualityController

### File: `src/models/quality/controller.py`

**Tasks:**
- [ ] Create ModelQualityController class
- [ ] Implement prediction validation
- [ ] Implement drift detection
- [ ] Add quality scoring
- [ ] Unit tests

**Implementation:**
```python
class ModelQualityController:
    """Quality control for model predictions and drift detection"""
    
    def __init__(self, drift_threshold: float = 0.3):
        self.drift_threshold = drift_threshold
        self.baseline_stats: Dict[str, Dict] = {}
        self.logger = ProjectLogger.get_logger(__name__)
    
    def validate_predictions(self, predictions: np.ndarray) -> bool:
        """Validate predictions for NaN/Inf and reasonable values"""
        # Check for NaN/Inf
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            self.logger.warning("Predictions contain NaN or Inf")
            return False
        
        # Check for unrealistic values
        if np.any(np.abs(predictions) > 10):  # >1000% return
            self.logger.warning("Predictions contain unrealistic values")
            return False
        
        return True
    
    def check_drift(self, current: np.ndarray, baseline: np.ndarray) -> float:
        """Check for distribution drift using KL divergence approximation"""
        current_mean = np.mean(current)
        current_std = np.std(current)
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)
        
        # KL divergence approximation
        drift = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)
        
        if drift > self.drift_threshold:
            self.logger.warning(f"Drift detected: {drift:.3f} > {self.drift_threshold}")
        
        return drift
    
    def get_quality_score(self, ensemble_pred: float, 
                         predictions: Dict[str, float],
                         weights: Dict[str, float]) -> float:
        """Calculate quality score based on agreement and weight distribution"""
        # Variance of predictions (lower = better agreement)
        pred_values = list(predictions.values())
        variance = np.var(pred_values)
        agreement = 1.0 / (1.0 + variance)
        
        # Weight entropy (more balanced = better)
        weight_values = list(weights.values())
        weight_entropy = -sum(w * np.log(w + 1e-6) for w in weight_values)
        max_entropy = np.log(len(weights))
        balance = weight_entropy / max_entropy if max_entropy > 0 else 0
        
        # Combined score
        quality = 0.6 * agreement + 0.4 * balance
        return quality
    
    def update_baseline(self, model_id: str, predictions: np.ndarray):
        """Update baseline statistics for drift detection"""
        self.baseline_stats[model_id] = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'updated_at': datetime.now().isoformat()
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate quality control report"""
        return {
            'drift_threshold': self.drift_threshold,
            'baseline_models': list(self.baseline_stats.keys()),
            'total_baselines': len(self.baseline_stats),
            'timestamp': datetime.now().isoformat()
        }
```

**Tests:**
```python
# tests/models/quality/test_controller.py
def test_validate_predictions_valid():
    controller = ModelQualityController()
    predictions = np.array([0.01, 0.02, -0.01, 0.03])
    assert controller.validate_predictions(predictions) == True

def test_validate_predictions_nan():
    controller = ModelQualityController()
    predictions = np.array([0.01, np.nan, 0.02])
    assert controller.validate_predictions(predictions) == False

def test_check_drift_no_drift():
    controller = ModelQualityController(drift_threshold=0.5)
    baseline = np.random.normal(0, 1, 100)
    current = np.random.normal(0.1, 1, 100)
    drift = controller.check_drift(current, baseline)
    assert drift < 0.5

def test_check_drift_with_drift():
    controller = ModelQualityController(drift_threshold=0.5)
    baseline = np.random.normal(0, 1, 100)
    current = np.random.normal(2, 1, 100)  # Significant shift
    drift = controller.check_drift(current, baseline)
    assert drift > 0.5

def test_quality_score():
    controller = ModelQualityController()
    predictions = {"model1": 0.05, "model2": 0.06, "model3": 0.05}
    weights = {"model1": 0.33, "model2": 0.33, "model3": 0.34}
    score = controller.get_quality_score(0.053, predictions, weights)
    assert 0 <= score <= 1
```

**Acceptance Criteria:**
- ✅ Validates predictions correctly
- ✅ Detects drift accurately
- ✅ Calculates quality scores
- ✅ 100% test coverage
- ✅ Report generation works

---

## Integration with Pipeline

### File: `run_hybrid_pipeline.py` (update)

**Tasks:**
- [ ] Import new components
- [ ] Initialize PersistentModelPool
- [ ] Initialize ModelQualityController
- [ ] Update orchestrator to use new components
- [ ] Add quality checks in prediction flow

**Implementation:**
```python
# run_hybrid_pipeline.py (additions)
from src.models.persistent_pool import PersistentModelPool
from src.models.quality.controller import ModelQualityController

async def main():
    # ... existing code ...
    
    # Initialize enhanced components
    model_pool = PersistentModelPool(
        max_models=50,
        cache_dir=".model_cache"
    )
    
    quality_controller = ModelQualityController(
        drift_threshold=0.3
    )
    
    # Warm-up critical models
    critical_models = ["catboost_v1", "lightgbm_v1", "xgboost_v1"]
    loader_fns = {
        model: lambda m=model: registry.clone(m)
        for model in critical_models
    }
    model_pool.warm_up(critical_models, loader_fns)
    
    # Pass to orchestrator
    orchestrator = HybridOrchestrator(
        config_manager,
        batch_name=args.batch_name,
        model_pool=model_pool,
        quality_controller=quality_controller
    )
    
    # ... rest of pipeline ...
    
    # Post-execution stats
    pool_stats = model_pool.get_enhanced_stats()
    logger.info(f"📊 Model Pool Stats: {pool_stats}")
    
    quality_report = quality_controller.generate_report()
    logger.info(f"✅ Quality Report: {quality_report}")
```

**Tests:**
```python
# tests/integration/test_enhanced_pipeline.py
async def test_pipeline_with_persistent_pool():
    config_manager = UnifiedConfigManager()
    pool = PersistentModelPool()
    controller = ModelQualityController()
    
    orchestrator = HybridOrchestrator(
        config_manager,
        model_pool=pool,
        quality_controller=controller
    )
    
    results = await orchestrator.run_stages([0, 1, 2, 3])
    
    assert results is not None
    assert pool.get_stats()['hits'] > 0
    assert len(controller.baseline_stats) > 0
```

**Acceptance Criteria:**
- ✅ Pipeline uses PersistentModelPool
- ✅ Quality checks integrated
- ✅ Stats logged correctly
- ✅ Integration tests pass

---

## Testing Strategy

### Unit Tests
```bash
# Run all Week 1 Day 3-5 tests
python -m pytest tests/models/test_persistent_pool.py -v
python -m pytest tests/models/quality/test_controller.py -v
```

### Integration Tests
```bash
# Run integration tests
python -m pytest tests/integration/test_enhanced_pipeline.py -v
```

### Performance Tests
```bash
# Verify 30-40% speedup maintained
python -m pytest tests/performance/test_pool_performance.py -v
```

---

## Success Metrics

### Day 3-4: PersistentModelPool
- ✅ Cache index persists across runs
- ✅ Warm-up loads 10 models in <5 seconds
- ✅ Quality filtering works correctly
- ✅ 30-40% speedup maintained
- ✅ 100% test coverage

### Day 5: ModelQualityController
- ✅ Validates predictions (95%+ accuracy)
- ✅ Detects drift (threshold configurable)
- ✅ Quality scores calculated correctly
- ✅ 100% test coverage

### Integration
- ✅ Pipeline runs end-to-end
- ✅ All components work together
- ✅ No breaking changes
- ✅ Performance maintained

---

## Timeline

**Day 3 (Today):**
- Morning: Implement PersistentModelPool Part 1
- Afternoon: Write tests, verify persistence

**Day 4 (Tomorrow):**
- Morning: Implement PersistentModelPool Part 2
- Afternoon: Integration tests, warm-up mechanism

**Day 5 (Day After):**
- Morning: Implement ModelQualityController
- Afternoon: Integration with pipeline, full testing

---

## Next Steps After Completion

Once Week 1 Day 3-5 is complete:
1. ✅ Week 1 fully complete
2. 🚀 Start Week 2 (Adaptive Systems)
3. 📊 Performance benchmarks
4. 📚 Update documentation

---

**Status:** Ready to implement  
**Priority:** HIGH  
**Estimated Effort:** 3 days  
**Dependencies:** Week 1 Day 1-2 (COMPLETE)
