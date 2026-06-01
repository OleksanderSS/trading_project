# Audit Report Update (2026-06-01)

## Виконані роботи:
1. **Реструктуризація:** Перенесено всі аудиторські інструменти та legacy-скрипти до папки `/audit` (engine/reports/legacy).
2. **Фікс P0 (Data Leakage & Data Integrity):**
   - У `src/training/portfolio_optimizer.py` замінено `train_test_split` з випадковим перемішуванням на `shuffle=False` для збереження часової послідовності.
   - У `src/calibration/calibration_engine.py` замінено `train_test_split` на версію з `shuffle=False`.
   - У `src/monitoring/ml_analytics.py` видалено зайвий `random_state` для коректного часового спліту.
   - У `src/features/enrichers/macro_features_enricher.py` замінено `bfill()` на `ffill()` для запобігання lookahead leakage при злитті макро-даних.
   - У `src/data/collectors/cftc_collector.py` та `src/data/collectors/put_call_ratio_collector.py` замінено мовчазний перехід на синтетичні дані при помилках на явне виключення (`RuntimeError`) з можливістю `opt-in` через конфігурацію.
3. **Верифікація:**
   - Створено та успішно пройдено регресійні тести (`tests/p0_regression/`):
     - `test_temporal_split.py`
     - `test_leakage_macro.py`
   - Повторний запуск сканера підтвердив зменшення кількості критичних проблем.

## Статус:
- P0: 14 (було 22)
- P1: 375

---

# Codex Audit Report - Trading/ML Codebase

**Date:** 2025-01-XX  
**Auditor:** Cascade (Static Analysis)  
**Scope:** Complete trading/ML pipeline codebase  
**Categories Audited:** 16  

---

## Executive Summary

| Severity | Count | Categories |
|----------|-------|------------|
| P0 (Critical) | 5 | Temporal Leakage, NaN Policy, Security, Synthetic Gates |
| P1 (High) | 8 | Financial Math, Error Policy, Config Consistency, Artifact Safety |
| P2 (Medium) | 12 | Model Routing, Performance, Data Lineage, Evaluation |
| P3 (Low) | 6 | Dead Code, Long Modules, Tests |
| **Total** | **31** | |

---

## 1. Temporal Correctness / Leakage

### P0-1: pct_change with fillna(0) in synthetic data generator
- **Severity:** P0 (Critical)
- **File/Function:** `src/data/synthetic/data_generator.py:99`, `src/pipeline/stages/stage_0_data_generation.py:100`
- **Pattern:** `pct_change(fill_method=None).fillna(0)` and `pct_change().fillna(0)`
- **Danger:** Filling NaN returns with 0 creates artificial zero-return periods that don't exist in reality. This masks missing data and creates false signal patterns, leading to unrealistic backtest results and poor live performance.
- **Suggested Fix:** 
  ```python
  # Instead of:
  features['volatility'] = features['close'].pct_change(fill_method=None).fillna(0).rolling(window=20, min_periods=2).std().shift(1)
  
  # Use:
  features['volatility'] = features['close'].pct_change(fill_method=None).rolling(window=20, min_periods=2).std().shift(1)
  # Then explicitly drop rows with NaN after feature engineering
  features = features.dropna()
  ```
- **Test to Add:** Test that synthetic data generation preserves NaN values from pct_change and does not silently fill with 0. Verify that volatility calculations fail gracefully with insufficient data.

### P0-2: pct_change with fillna(0) in adaptive technical indicators
- **Severity:** P0 (Critical)
- **File/Function:** `src/features/utils/modular_adaptive_technical_indicators.py:27`, `:81`
- **Pattern:** `pct_change(fill_method=None).fillna(0)`
- **Danger:** Same as P0-1 - artificial zero returns mask missing data and create false patterns in feature engineering.
- **Suggested Fix:** Remove `.fillna(0)` and handle NaN explicitly through drop or forward-fill with clear policy documentation.
- **Test to Add:** Unit test for RSI/MACD calculators with input series containing NaN values to verify proper handling without silent zero-filling.

### P1-3: bfill() usage in smart missing data handler for indicators
- **Severity:** P1 (High)
- **File/Function:** `src/utils/smart_missing_data_handler.py:243`
- **Pattern:** `filled = series.bfill(limit=5)` in `_fill_indicator_data()`
- **Danger:** Backfill uses future data to fill past values, introducing lookahead bias in causal time-series data. This is particularly dangerous for indicators used in training models.
- **Suggested Fix:** Remove bfill() for indicator data. Use only ffill() with explicit availability rules, or drop rows with missing indicator values.
- **Test to Add:** Test that indicator filling never uses future data. Create time-series with known missing pattern and verify filled values don't incorporate future information.

### P1-4: bfill() allowed in cleaners with only warning
- **Severity:** P1 (High)
- **File/Function:** `src/processing/cleaners.py:86-94`
- **Pattern:** `df_out.bfill().ffill()` with warning log but no enforcement
- **Danger:** The warning is easily ignored, and bfill is still executed for causal time-series data, introducing lookahead bias.
- **Suggested Fix:** Make bfill() a fatal error for time-series data by default, or require explicit opt-in flag with clear documentation of data availability assumptions.
- **Test to Add:** Test that handle_missing_values with method='bfill' raises an exception for time-series data unless explicitly enabled with a flag.

### P2-5: Missing tail row drop after shift(-horizon) for targets
- **Severity:** P2 (Medium)
- **File/Function:** `src/pipeline/guards/temporal_target_guard.py:38-39`, `src/targets/target_orchestrator.py:195-202`
- **Pattern:** `future_price = df_enriched.groupby('ticker')['close'].shift(-shift)` without explicit tail row drop
- **Danger:** The last `horizon` rows have NaN targets (no future data). If these are filled with 0 or ffill, it creates artificial target values that don't exist.
- **Suggested Fix:** Explicitly drop tail rows after shift(-horizon) operations:
  ```python
  future_price = df_enriched.groupby('ticker')['close'].shift(-shift)
  results[name] = (future_price - df_enriched['close']) / df_enriched['close']
  # Drop tail rows where target is NaN
  results[name] = results[name].dropna()
  ```
- **Test to Add:** Test that target generation with shift(-horizon) results in exactly `horizon` fewer rows than input, and that no tail rows are silently filled.

### P3-6: pct_change without fill_method parameter
- **Severity:** P3 (Low)
- **File/Function:** `src/pipeline/stages/stage_0_data_generation.py:100`
- **Pattern:** `pct_change()` without explicit `fill_method=None`
- **Danger:** Relies on pandas default behavior which may change in future versions. Current pandas 2.0+ defaults to no fill, but older versions had different behavior.
- **Suggested Fix:** Always use `pct_change(fill_method=None)` for explicit behavior.
- **Test to Add:** Test verifies that pct_change behavior is consistent across pandas versions by checking for unexpected forward-fill behavior.

---

## 2. NaN / Missing Policy

### P0-7: fillna(0) for returns and volatility in synthetic data
- **Severity:** P0 (Critical)
- **File/Function:** `src/data/synthetic/data_generator.py:103-105`, `:114`
- **Pattern:** `fillna(0)` after pct_change for returns and volatility
- **Danger:** Zero-filling returns creates artificial zero-return periods, distorting volatility calculations and risk metrics. This is particularly dangerous for financial ML where return distribution shape matters.
- **Suggested Fix:** 
  ```python
  # Instead of fillna(0), use:
  features['returns_1h'] = features['close'].pct_change(fill_method=None)
  features = features.dropna()  # Drop rows with missing returns
  ```
- **Test to Add:** Test that synthetic data generation produces realistic return distributions without artificial zero-return spikes.

### P1-8: Global ffill/bfill across tickers in cleaners
- **Severity:** P1 (High)
- **File/Function:** `src/processing/cleaners.py:82-94`
- **Pattern:** `df_out.groupby('ticker')[data_cols].ffill()` - correct, but fallback to `df_out.ffill()` without groupby
- **Danger:** If ticker column is missing or groupby fails, global ffill/bfill could cross-contaminate data between different tickers, creating artificial price relationships.
- **Suggested Fix:** Make ticker-based groupby mandatory for time-series data, raise error if ticker column is missing.
- **Test to Add:** Test that handle_missing_values raises error when ticker column is missing for time-series data.

### P2-9: No explicit missing policy per column type
- **Severity:** P2 (Medium)
- **File/Function:** `src/utils/smart_missing_data_handler.py:178-269`
- **Pattern:** Different fill strategies for price/volume/indicator/macro but no explicit documentation of which columns use which policy
- **Danger:** Inconsistent missing data handling across the codebase makes it difficult to reason about data quality and potential leakage.
- **Suggested Fix:** Add explicit column type mapping in configuration, and validate that all columns have a declared missing data policy.
- **Test to Add:** Test that all columns in DataFrame have a declared missing data policy before processing.

---

## 3. Synthetic / Sample / Demo Gates

### P0-10: No opt-in gate for synthetic data usage
- **Severity:** P0 (Critical)
- **File/Function:** `src/data/synthetic/data_generator.py:29-49`, `src/pipeline/stages/stage_0_data_generation.py:30-50`
- **Pattern:** Synthetic data generation happens automatically without opt-in flag
- **Danger:** If real data collection fails, the pipeline may silently fall back to synthetic data, leading to models trained on unrealistic data being deployed to production.
- **Suggested Fix:** Add explicit opt-in flag for synthetic data:
  ```python
  def generate_synthetic_data(self, use_synthetic: bool = False) -> dict[str, Any]:
      if not use_synthetic:
          raise ValueError("Synthetic data generation requires explicit opt-in")
      # ... rest of code
  ```
- **Test to Add:** Test that synthetic data generation raises error when not explicitly enabled via configuration flag.

### P1-11: No marking of synthetic data in output
- **Severity:** P1 (High)
- **File/Function:** `src/data/synthetic/data_generator.py:41-48`
- **Pattern:** Returns dictionary with features/targets but no metadata indicating data is synthetic
- **Danger:** Downstream stages cannot distinguish between real and synthetic data, potentially mixing them in training/validation.
- **Suggested Fix:** Add metadata fields:
  ```python
  return {
      'status': 'success',
      'features_df': features_df,
      'targets_df': targets_df,
      'metadata': {
          'is_synthetic': True,
          'data_kind': 'synthetic',
          'eligible_for_training': False,  # Default to false for synthetic
          'source_type': 'generated'
      }
  }
  ```
- **Test to Add:** Test that synthetic data output includes metadata fields and that downstream stages check these fields before using data.

### P2-12: Synthetic data used in default pipeline without warning
- **Severity:** P2 (Medium)
- **File/Function:** `src/pipeline/stages/stage_0_data_generation.py`
- **Pattern:** Stage 0 defaults to synthetic data generation when real data is unavailable
- **Danger:** Creates a path where production pipeline could run on synthetic data without clear indication.
- **Suggested Fix:** Add explicit warning log when synthetic data is used, and require configuration override to enable.
- **Test to Add:** Test that pipeline logs warning when synthetic data is used and that this is captured in monitoring.

---

## 4. Model Selection / Routing

### P1-13: Hardcoded model lists in factory
- **Severity:** P1 (High)
- **File/Function:** `src/factories/model_factory.py:28-42`
- **Pattern:** `_model_map` and `_model_aliases` are hardcoded dictionaries
- **Danger:** Model lists are duplicated across factory, config, and selectors, leading to inconsistency. Adding a new model requires updating multiple places.
- **Suggested Fix:** Move model registry to central configuration file:
  ```yaml
  # config/models.yaml
  models:
    lightgbm:
      class_path: src.models.tree.lightgbm_model:LightGBMModel
      aliases: [lgbm, lightgbm]
      role: predictor
      can_be_primary: true
  ```
- **Test to Add:** Test that model factory loads model definitions from central config and that adding a model to config automatically makes it available.

### P2-14: Duplicated aliases across factory and config
- **Severity:** P2 (Medium)
- **File/Function:** `src/factories/model_factory.py:34-42`
- **Pattern:** Aliases like 'xgboost', 'lightgbm', 'catboost' exist in factory but may also exist in config
- **Danger:** Inconsistent alias resolution could lead to wrong model being instantiated.
- **Suggested Fix:** Single source of truth for aliases in central config, factory validates against it.
- **Test to Add:** Test that alias resolution is consistent across factory, config, and selectors.

### P2-15: Smart selector not actually called in prediction stage
- **Severity:** P2 (Medium)
- **File/Function:** `src/pipeline/stages/stage_5_prediction.py:77-88`
- **Pattern:** Adaptive selector is initialized but may not be used for actual model selection
- **Danger:** Selector logic exists but may be bypassed, making the sophisticated selection logic ineffective.
- **Suggested Fix:** Verify that selector is actually called for model selection in prediction generation, or remove if unused.
- **Test to Add:** Test that model selection in prediction stage uses the configured selector and logs which model was selected.

---

## 5. Financial Math

### P1-16: Silent 0.0 return for std=0 in Sharpe calculation
- **Severity:** P1 (High)
- **File/Function:** `src/metrics/financial/financial_metrics_library.py:55-56`
- **Pattern:** `if returns.empty or returns.std() == 0: return 0.0`
- **Danger:** Silent 0.0 Sharpe ratio for zero-volatility periods masks edge cases and could indicate data issues. Should be explicit about why Sharpe is undefined.
- **Suggested Fix:** 
  ```python
  if returns.empty:
      return float('nan')  # Or raise exception
  if returns.std() == 0:
      logger.warning("Sharpe ratio undefined: zero volatility")
      return float('nan')  # Or return 0.0 with explicit comment
  ```
- **Test to Add:** Test that Sharpe calculation handles zero-volatility edge cases explicitly and logs warnings.

### P1-17: Silent 0.0 return for downside_std=0 in Sortino
- **Severity:** P1 (High)
- **File/Function:** `src/metrics/financial/financial_metrics_library.py:69-70`
- **Pattern:** `if downside_std == 0: return 0.0`
- **Danger:** Similar to P1-16 - silent handling of edge case where all returns are positive.
- **Suggested Fix:** Return NaN with warning log for undefined Sortino ratio.
- **Test to Add:** Test Sortino calculation with all-positive returns to verify proper handling.

### P2-18: No explicit annualization parameter validation
- **Severity:** P2 (Medium)
- **File/Function:** `src/metrics/financial/financial_metrics_library.py:44-48`, `:52-58`
- **Pattern:** `trading_days_per_year` parameter used without validation
- **Danger:** Incorrect annualization factor (e.g., using 365 for daily data instead of 252) leads to wrong risk-adjusted metrics.
- **Suggested Fix:** Validate trading_days_per_year against expected range (252 for daily, 252*24 for hourly, etc.) based on data frequency.
- **Test to Add:** Test that annualization parameter is validated and raises error for inconsistent values.

### P3-19: Drawdown sign convention not documented
- **Severity:** P3 (Low)
- **File/Function:** `src/metrics/financial/financial_metrics_library.py:75-86`
- **Pattern:** Drawdown calculated as negative values but not explicitly documented
- **Danger:** Inconsistent sign convention across codebase could lead to misinterpretation (e.g., max drawdown of -0.25 vs 0.25).
- **Suggested Fix:** Document sign convention in docstring and ensure consistency across all drawdown calculations.
- **Test to Add:** Test that drawdown values are always negative and that max_drawdown returns the most negative value.

---

## 6. Pipeline Error Policy

### P1-20: Silent None return in model loader
- **Severity:** P1 (High)
- **File/Function:** `src/models/loader.py:94-96`
- **Pattern:** Returns None when all loaders fail, caller may not check
- **Danger:** If caller doesn't check for None, subsequent operations will fail with unclear error message.
- **Suggested Fix:** Raise explicit exception when all loaders fail, or require caller to handle None explicitly.
- **Test to Add:** Test that model loader raises exception when all loading strategies fail.

### P2-21: Broad exception catching in error handler
- **Severity:** P2 (Medium)
- **File/Function:** `src/core/error_handling/error_handler.py:85-93`
- **Pattern:** `except Exception as e:` catches all exceptions
- **Danger:** Catches system exceptions (KeyboardInterrupt, SystemExit) that should not be caught.
- **Suggested Fix:** Use more specific exception types, or re-raise critical exceptions.
- **Test to Add:** Test that critical exceptions (KeyboardInterrupt, SystemExit) are not caught by error handler.

### P2-22: No distinction between fatal and non-fatal errors
- **Severity:** P2 (Medium)
- **File/Function:** `src/core/error_handling/error_handler.py:57-67`
- **Pattern:** `log_and_raise` treats all errors as fatal
- **Danger:** Non-fatal errors (e.g., optional collector failure) should not halt the entire pipeline.
- **Suggested Fix:** Add severity parameter to distinguish fatal vs non-fatal errors, and handle accordingly.
- **Test to Add:** Test that non-fatal errors are logged but don't halt pipeline, while fatal errors do.

### P3-23: Duplicate logging in error handler
- **Severity:** P3 (Low)
- **File/Function:** `src/core/error_handling/error_handler.py:66`
- **Pattern:** Logs error then raises, caller may also log
- **Danger:** Duplicate log entries make debugging difficult.
- **Suggested Fix:** Use single logging point, or add flag to prevent duplicate logging.
- **Test to Add:** Test that error is logged exactly once even when re-raised through multiple layers.

---

## 7. Security / Secrets / Paths

### P0-24: Joblib loading without path validation
- **Severity:** P0 (Critical)
- **File/Function:** `src/models/loader.py:16`, `:98-100`
- **Pattern:** `import joblib` and loading from arbitrary paths
- **Danger:** Loading models from untrusted paths could execute malicious code through pickle/joblib deserialization.
- **Suggested Fix:** Validate model paths against allowed base directory before loading:
  ```python
  def load_path(self, model_path: str, meta: Dict[str, Any]) -> Optional[Any]:
      allowed_base = self.config_manager.get_models_path()
      resolved_path = Path(model_path).resolve()
      if not str(resolved_path).startswith(str(allowed_base.resolve())):
          raise SecurityError(f"Model path outside allowed directory: {model_path}")
      # ... proceed with loading
  ```
- **Test to Add:** Test that model loading raises security error when path is outside allowed directory.

### P1-25: No path traversal protection in config manager
- **Severity:** P1 (High)
- **File/Function:** `src/config/unified_config_manager.py:92-95`
- **Pattern:** `Path(config_dir).resolve()` without validation against project root
- **Danger:** Malicious config_dir parameter could access files outside project directory.
- **Suggested Fix:** Validate config_dir is within project root or allowlist of directories.
- **Test to Add:** Test that config manager raises error when config_dir attempts path traversal.

### P2-26: Secrets manager usage not verified
- **Severity:** P2 (Medium)
- **File/Function:** `src/config/unified_config_manager.py:14`
- **Pattern:** Imports SecretsManager but usage not verified in code read
- **Danger:** If secrets are not properly loaded from secure source, they may fall back to insecure defaults.
- **Suggested Fix:** Verify that SecretsManager is actually used for sensitive config values and has fallback protection.
- **Test to Add:** Test that missing secrets raise error rather than using insecure defaults.

---

## 8. Heavy Imports / Performance

### P2-27: Top-level imports of heavy libraries not verified
- **Severity:** P2 (Medium)
- **File/Function:** Various files (need grep for tensorflow, torch, transformers, spacy)
- **Pattern:** Heavy libraries imported at module level
- **Danger:** Slow import time even when not using those features, unnecessary memory usage.
- **Suggested Fix:** Lazy import heavy libraries inside functions that actually use them.
- **Test to Add:** Test that importing lightweight modules (config, logging) is fast and doesn't load heavy dependencies.

### P3-28: No lazy loading in model factory
- **Severity:** P3 (Low)
- **File/Function:** `src/factories/model_factory.py:9-19`
- **Pattern:** All model classes imported at module level
- **Danger:** Loading all model classes even when only using one.
- **Suggested Fix:** Lazy import model classes when actually needed.
- **Test to Add:** Test that importing model factory doesn't load all model implementations.

---

## 9. Dead Code Usefulness

### P3-29: Old stage files may contain deprecated logic
- **Severity:** P3 (Low)
- **File/Function:** `src/pipeline/stages/stage_2_processing.py`, `stage_3_feature_engineering.py`
- **Pattern:** Very short files (412 bytes, 491 bytes) that may be stubs
- **Danger:** Unclear if these are deprecated, in progress, or actively used.
- **Suggested Fix:** Either remove if deprecated, or add TODO comments with status.
- **Test to Add:** Verify which stages are actually called in pipeline and remove unused ones.

### P3-30: Multiple model analyzers with unclear usage
- **Severity:** P3 (Low)
- **File/Function:** `src/models/analysis/` directory with many analyzer files
- **Pattern:** Multiple analyzer classes, unclear which are actively used
- **Danger:** Dead code accumulates, making maintenance difficult.
- **Suggested Fix:** Audit analyzer usage and remove or mark as deprecated if unused.
- **Test to Add:** Test that all registered analyzers are actually called in pipeline.

---

## 10. Config / Factory Consistency

### P1-31: Model lists duplicated across factory and config
- **Severity:** P1 (High)
- **File/Function:** `src/factories/model_factory.py:28-42` vs config files
- **Pattern:** Hardcoded model map in factory duplicates config
- **Danger:** Adding model requires updating both places, risk of inconsistency.
- **Suggested Fix:** Single source of truth in config, factory loads from config.
- **Test to Add:** Test that factory model list matches config model list exactly.

### P2-32: Default model hardcoded in multiple places
- **Severity:** P2 (Medium)
- **File/Function:** `src/factories/model_factory.py:89`, `src/models/model_selector/adaptive_selector.py:82`
- **Pattern:** Default model 'lightgbm' hardcoded
- **Danger:** Changing default requires updating multiple files.
- **Suggested Fix:** Centralize default model in config.
- **Test to Add:** Test that default model is loaded from config, not hardcoded.

---

## 11. Long Modules / God Objects

### P2-33: Stage 5 Prediction very long (669 lines)
- **Severity:** P2 (Medium)
- **File/Function:** `src/pipeline/stages/stage_5_prediction.py`
- **Pattern:** Single file with many responsibilities (model loading, prediction, anomaly detection, context management)
- **Danger:** Difficult to maintain, test, and understand. High cognitive complexity.
- **Suggested Fix:** Already partially refactored into sub-package `prediction/`. Continue refactoring remaining logic.
- **Test to Add:** Test that refactored components maintain same behavior as original monolithic stage.

### P2-34: Smart Missing Data Handler large (444 lines)
- **Severity:** P2 (Medium)
- **File/Function:** `src/utils/smart_missing_data_handler.py`
- **Pattern:** Single class with many fill strategies
- **Danger:** Complex logic for different data types mixed in one class.
- **Suggested Fix:** Split into separate strategy classes for each data type (price, volume, indicator, macro).
- **Test to Add:** Test that refactored strategy classes produce same fill results as original.

---

## 12. Offline / Deterministic Tests

### P2-35: Test uses hardcoded random seed but not verified
- **Severity:** P2 (Medium)
- **File/Function:** `src/data/synthetic/data_generator.py:27`, `tests/test_algorithms_integrity.py`
- **Pattern:** `np.random.default_rng(42)` but not all tests use it
- **Danger:** Some tests may be non-deterministic due to missing seed.
- **Suggested Fix:** Centralize RNG seed configuration in test fixtures.
- **Test to Add:** Test that all tests produce identical results when run with same seed.

### P3-36: Test may use datetime.now() not verified
- **Severity:** P3 (Low)
- **File/Function:** Various test files (need grep for datetime.now)
- **Pattern:** datetime.now() in tests could cause non-determinism
- **Danger:** Tests may fail depending on when they run.
- **Suggested Fix:** Use fixed datetime fixtures in tests.
- **Test to Add:** Test that all datetime-dependent tests use fixed fixtures.

---

## 13. Data Lineage

### P2-37: No feature metadata tracking
- **Severity:** P2 (Medium)
- **File/Function:** `src/features/feature_orchestrator.py` (inferred)
- **Pattern:** Features generated without tracking source, calculation window, availability time
- **Danger:** Difficult to debug feature issues, unclear when features are actually available for use.
- **Suggested Fix:** Add feature metadata tracking:
  ```python
  feature_metadata = {
      'feature_name': 'rsi_14',
      'source': 'close',
      'calculation_window': 14,
      'availability_lag': 1,  # periods
      'causal': True
  }
  ```
- **Test to Add:** Test that feature generation includes metadata and that metadata is validated.

### P2-38: No macro/news publication time tracking
- **Severity:** P2 (Medium)
- **File/Function:** `src/features/enrichers/macro_features_enricher.py`, `sentiment_features_enricher.py`
- **Pattern:** Macro/sentiment features don't track publication/ingestion time
- **Danger:** May use macro data before it was actually published, introducing lookahead bias.
- **Suggested Fix:** Add publication_time field to macro/sentiment data and validate feature availability.
- **Test to Add:** Test that macro features are only used after their publication time.

---

## 14. Artifact / Model Persistence Safety

### P1-39: Model loading without version/schema check
- **Severity:** P1 (High)
- **File/Function:** `src/models/loader.py:98-100`
- **Pattern:** Loads model without checking version or schema compatibility
- **Danger:** Loading old model with incompatible schema could cause runtime errors or incorrect predictions.
- **Suggested Fix:** Add model metadata validation:
  ```python
  model_metadata = joblib.load(model_path + '.meta')
  if model_metadata['schema_version'] != CURRENT_SCHEMA:
      raise IncompatibleModelError(...)
  ```
- **Test to Add:** Test that loading model with incompatible schema raises error.

### P1-40: No training metadata saved with model
- **Severity:** P1 (High)
- **File/Function:** Model saving logic (inferred from training code)
- **Pattern:** Models saved without training metadata (data range, features, config hash)
- **Danger:** Cannot reproduce model training or understand what data it was trained on.
- **Suggested Fix:** Save comprehensive metadata with model:
  ```python
  metadata = {
      'data_range': (start_date, end_date),
      'features': feature_list,
      'target_horizon': horizon,
      'config_hash': config_hash,
      'git_commit': git_commit
  }
  joblib.dump(metadata, model_path + '.meta')
  ```
- **Test to Add:** Test that model saving includes all required metadata fields.

### P2-41: Model path not versioned
- **Severity:** P2 (Medium)
- **File/Function:** `src/models/loader.py:59`
- **Pattern:** Model path from metadata, no version/run ID
- **Danger:** Cannot distinguish between different model versions, may load wrong model.
- **Suggested Fix:** Include version/run ID in model path and metadata.
- **Test to Add:** Test that model paths include version identifiers and that loading validates version.

---

## 15. Evaluation Contamination

### P2-42: Validation/test separation not verified
- **Severity:** P2 (Medium)
- **File/Function:** `src/pipeline/stages/stage_4_modeling.py:100` (inferred)
- **Pattern:** Purged gap mentioned but not verified in code read
- **Danger:** If validation/test sets are not properly separated, hyperparameter tuning could see test data.
- **Suggested Fix:** Verify that train/val/test split is chronological with purged gaps, and add validation.
- **Test to Add:** Test that validation and test sets have no temporal overlap with proper purged gaps.

### P2-43: Leaderboard may be updated on test data
- **Severity:** P2 (Medium)
- **File/Function:** `src/models/model_selector/adaptive_selector.py:82`
- **Pattern:** Leaderboard path referenced, unclear if test data is excluded
- **Danger:** Repeated leaderboard optimization on test data leads to overfitting.
- **Suggested Fix:** Ensure leaderboard only uses validation data, never test data.
- **Test to Add:** Test that leaderboard updates are rejected if they include test data metrics.

---

## 16. Trading Realism

### P2-44: Transaction cost modeling exists but not verified in backtest
- **Severity:** P2 (Medium)
- **File/Function:** `src/backtesting/advanced/advanced_engine.py:21-57`
- **Pattern:** TransactionCostModel class exists but usage in actual backtest not verified
- **Danger:** Backtest may not include realistic costs, leading to overoptimistic results.
- **Suggested Fix:** Verify that TransactionCostModel is actually used in backtest execution.
- **Test to Add:** Test that backtest includes transaction costs and that costs are reasonable.

### P2-45: No execution delay modeling
- **Severity:** P2 (Medium)
- **File/Function:** Backtest engine (inferred)
- **Pattern:** No explicit execution delay between signal generation and execution
- **Danger:** Assumes instant execution, unrealistic in live trading.
- **Suggested Fix:** Add execution delay parameter (e.g., signal at close T, execute at open T+1).
- **Test to Add:** Test that backtest includes execution delay and results differ from instant execution.

### P3-46: Position sizing not verified
- **Severity:** P3 (Low)
- **File/Function:** Position sizer exists but usage not verified
- **Pattern:** Adaptive position sizer exists but integration not verified
- **Danger:** May use fixed position sizing instead of risk-based sizing.
- **Suggested Fix:** Verify that position sizer is used in trading execution.
- **Test to Add:** Test that trading execution uses position sizer and respects risk limits.

---

## Summary by Severity

### P0 (Critical) - 5 issues
1. pct_change with fillna(0) in synthetic data generator
2. pct_change with fillna(0) in adaptive technical indicators
3. No opt-in gate for synthetic data usage
4. Joblib loading without path validation
5. fillna(0) for returns and volatility in synthetic data

### P1 (High) - 8 issues
6. bfill() usage in smart missing data handler for indicators
7. bfill() allowed in cleaners with only warning
8. Missing tail row drop after shift(-horizon) for targets
9. Global ffill/bfill across tickers in cleaners
10. No marking of synthetic data in output
11. Hardcoded model lists in factory
12. Silent 0.0 return for std=0 in Sharpe calculation
13. Silent 0.0 return for downside_std=0 in Sortino
14. Silent None return in model loader

### P2 (Medium) - 12 issues
15. pct_change without fill_method parameter
16. No explicit missing policy per column type
17. Synthetic data used in default pipeline without warning
18. Duplicated aliases across factory and config
19. Smart selector not actually called in prediction stage
20. No explicit annualization parameter validation
21. Drawdown sign convention not documented
22. Broad exception catching in error handler
23. No distinction between fatal and non-fatal errors
24. Secrets manager usage not verified
25. Top-level imports of heavy libraries not verified
26. Model lists duplicated across factory and config
27. Default model hardcoded in multiple places
28. Stage 5 Prediction very long (669 lines)
29. Smart Missing Data Handler large (444 lines)
30. Test uses hardcoded random seed but not verified
31. No feature metadata tracking
32. No macro/news publication time tracking
33. Model loading without version/schema check
34. No training metadata saved with model
35. Model path not versioned
36. Validation/test separation not verified
37. Leaderboard may be updated on test data
38. Transaction cost modeling exists but not verified in backtest
39. No execution delay modeling

### P3 (Low) - 6 issues
40. Old stage files may contain deprecated logic
41. Multiple model analyzers with unclear usage
42. No lazy loading in model factory
43. Test may use datetime.now() not verified
44. Position sizing not verified

---

## Recommended Action Plan

### Immediate (P0)
1. Remove all `fillna(0)` after `pct_change()` operations
2. Add opt-in gate for synthetic data with metadata marking
3. Add path validation for model loading
4. Remove or make fatal `bfill()` for causal time-series data

### Short-term (P1)
5. Add explicit tail row drop after `shift(-horizon)` for targets
6. Add NaN handling with explicit logging for edge cases in financial metrics
7. Raise exception instead of returning None in model loader
8. Add security validation for config paths
9. Centralize model registry in configuration

### Medium-term (P2)
10. Add feature metadata tracking system
11. Add model versioning and schema validation
12. Verify transaction costs and execution delays in backtest
13. Refactor long modules (Stage 5, SmartMissingDataHandler)
14. Add validation/test separation verification

### Long-term (P3)
15. Clean up dead/unused code
16. Implement lazy loading for heavy libraries
17. Centralize RNG seeding for tests
18. Document all sign conventions and assumptions

---

## Test Coverage Recommendations

Add the following test categories:
1. Temporal leakage tests (shift, pct_change, rolling operations)
2. Missing data policy tests (fillna, ffill, bfill behavior)
3. Synthetic data gate tests (opt-in verification)
4. Model loading security tests (path validation)
5. Financial metrics edge case tests (zero volatility, all-positive returns)
6. Feature metadata tests (lineage tracking)
7. Model versioning tests (schema compatibility)
8. Backtest realism tests (costs, delays, position sizing)

---

## Conclusion

The codebase shows good architectural patterns with guards for temporal leakage and data safety, but has several critical issues around:
- Silent NaN handling that masks missing data
- Lack of opt-in gates for synthetic data
- Insufficient security validation for model loading
- Missing metadata tracking for features and models

Addressing the P0 and P1 issues should be the immediate priority, as they directly impact data quality and security. The P2 and P3 issues can be addressed incrementally as part of ongoing maintenance.

**Overall Assessment:** The codebase is well-structured but requires hardening around data handling, security, and metadata tracking before production deployment.
