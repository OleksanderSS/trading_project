# Audit Triage Report

Selected severities: P0, P1
Selected findings: **301**

## Counts

### By severity
- P1: 301

### By rule
- P1 error_policy/BROAD_EXCEPTION_SILENT_RETURN: 103
- P1 missing_policy/FILLNA_ZERO_SUSPICIOUS: 72
- P1 security/UNSAFE_MODEL_OR_PICKLE_LOAD: 31
- P1 financial_math/SHARPE_SORTINO_STD_ZERO_REVIEW: 27
- P1 security/PLACEHOLDER_SECRET_REVIEW: 23
- P1 model_routing/AUTOENCODER_ROUTING_REVIEW: 16
- P1 financial_math/VAR_SIGN_OR_EMPTY_DATA_REVIEW: 13
- P1 temporal/NEGATIVE_SHIFT_LOOKAHEAD: 9
- P1 heavy_imports/HEAVY_TOP_LEVEL_IMPORT: 7

---

## P1 error_policy / BROAD_EXCEPTION_SILENT_RETURN — 103 finding(s)

**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.

### Examples
- `analytics/reporting/automated_reports.py:70` fingerprint `14dc8b2f8fd8ae3f` confidence `high`
- `analytics/reporting/automated_reports.py:99` fingerprint `702c31acf76c4d79` confidence `high`
- `analytics/reporting/results_manager.py:88` fingerprint `b519abd6f7419ef9` confidence `high`
- `analytics/reporting/results_manager.py:108` fingerprint `da18e9d44bda6439` confidence `high`
- `analytics/reporting/results_manager.py:121` fingerprint `a6ad92c34b6bf262` confidence `high`
- `analytics/reporting/results_manager.py:217` fingerprint `6f9de539fecdec26` confidence `high`
- `analytics/reporting/visualization.py:39` fingerprint `676205973af2fc60` confidence `high`
- `core/cloud/gcs_manager.py:106` fingerprint `a647b5479758f02f` confidence `high`
- ... 95 more

## P1 missing_policy / FILLNA_ZERO_SUSPICIOUS — 72 finding(s)

**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.

### Examples
- `algorithms/advanced_backtest_engine.py:38` fingerprint `8e9408713a149fea` confidence `high`
- `algorithms/advanced_backtest_engine.py:60` fingerprint `0df04dfdcb581e91` confidence `high`
- `analytics/calculators/macro_score_calculator.py:129` fingerprint `953e099cecbc3d9c` confidence `high`
- `analytics/context/market_context_analyzer.py:63` fingerprint `f3cf62beabc0e8d6` confidence `high`
- `analytics/context/market_context_analyzer.py:101` fingerprint `f3f2a70be1ff8c78` confidence `high`
- `analytics/context/market_context_analyzer.py:106` fingerprint `b03ffa40b437c4d7` confidence `high`
- `analytics/context/market_regime_analyzer.py:55` fingerprint `e5ad41fdfe18ac00` confidence `high`
- `analytics/detectors/anomaly_detector.py:71` fingerprint `fd6172ab4feafd2d` confidence `high`
- ... 64 more

## P1 security / UNSAFE_MODEL_OR_PICKLE_LOAD — 31 finding(s)

**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.

### Examples
- `analytics/arena/arena_battle.py:183` fingerprint `c87f35519e190caa` confidence `medium`
- `calibration/adaptive_confidence_calibrator.py:290` fingerprint `0184797265d097f4` confidence `medium`
- `colab/utils/utils.py:66` fingerprint `882c742ba4634b38` confidence `medium`
- `core/cache/cache_manager.py:146` fingerprint `547c7d2975667c7c` confidence `medium`
- `ensembling/stacked_ensemble.py:318` fingerprint `4972ce6ccde3f732` confidence `medium`
- `models/ensemble/confidence_calibrator.py:128` fingerprint `968e1dca0df230e8` confidence `medium`
- `models/ensemble/enhanced_ensemble.py:130` fingerprint `8bb68b2096b340b7` confidence `medium`
- `models/ensemble/enhanced_ensemble.py:165` fingerprint `2b61dc125ff055cf` confidence `medium`
- ... 23 more

## P1 financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — 27 finding(s)

**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.

### Examples
- `algorithms/advanced_backtest_engine.py:125` fingerprint `a586eaefc2ffa8d5` confidence `low`
- `algorithms/advanced_backtest_engine.py:129` fingerprint `4b75807bd0b19587` confidence `low`
- `algorithms/advanced_backtest_engine.py:130` fingerprint `4f210d71cc87623a` confidence `low`
- `algorithms/advanced_backtest_engine.py:132` fingerprint `e984cb9f2cd21fc1` confidence `low`
- `algorithms/advanced_backtest_engine.py:133` fingerprint `aad0f0a35db7e52e` confidence `low`
- `algorithms/advanced_backtest_engine.py:134` fingerprint `6a3f2e1b721ff2bd` confidence `low`
- `algorithms/advanced_backtest_engine.py:135` fingerprint `1270ce7af8578de4` confidence `low`
- `algorithms/advanced_backtest_engine.py:148` fingerprint `59037b5c85bc18cf` confidence `low`
- ... 19 more

## P1 security / PLACEHOLDER_SECRET_REVIEW — 23 finding(s)

**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.

### Examples
- `config/unified_config_manager.py:221` fingerprint `38772162bb1edd8b` confidence `medium`
- `config/unified_config_manager.py:222` fingerprint `986062d2ba0009d1` confidence `medium`
- `config/unified_config_manager.py:230` fingerprint `099f3c6490d790f6` confidence `medium`
- `config/unified_config_manager.py:244` fingerprint `3df3a9ad367705bb` confidence `medium`
- `config/unified_config_manager.py:246` fingerprint `b617fc91c8211af7` confidence `medium`
- `config/unified_config_manager.py:249` fingerprint `c65b5960e30323a9` confidence `medium`
- `config/unified_config_manager.py:250` fingerprint `f6a573a03d00bee6` confidence `medium`
- `config/unified_config_manager.py:251` fingerprint `901cc5a7b8f80d6f` confidence `medium`
- ... 15 more

## P1 model_routing / AUTOENCODER_ROUTING_REVIEW — 16 finding(s)

**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.

### Examples
- `colab/models/model_factory.py:93` fingerprint `bda761d868be0ff6` confidence `medium`
- `colab/models/model_factory.py:220` fingerprint `ae470adc604aa1e2` confidence `medium`
- `colab/models/torch_models.py:9` fingerprint `1bfc9b66e06e4e3e` confidence `medium`
- `colab/models/torch_models.py:126` fingerprint `d39710bb94c692ac` confidence `medium`
- `factories/model_factory.py:18` fingerprint `3724778cc0f3f8f8` confidence `medium`
- `models/loader.py:438` fingerprint `344d81598e217a5b` confidence `medium`
- `models/neural/autoencoder_model.py:1` fingerprint `5cd70b4e0934aecd` confidence `medium`
- `models/neural/autoencoder_model.py:30` fingerprint `912fde47afe4e813` confidence `medium`
- ... 8 more

## P1 financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — 13 finding(s)

**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.

### Examples
- `analytics/analyzers/risk_decomposition_analyzer.py:119` fingerprint `1474ac820c2cd449` confidence `medium`
- `analytics/calculators/risk_reward_calculator.py:153` fingerprint `6f149b3f6c84b1fb` confidence `medium`
- `metrics/financial/financial_metrics_library.py:182` fingerprint `f873d7f066535916` confidence `medium`
- `metrics/financial/financial_metrics_library.py:184` fingerprint `d2978c50798cec24` confidence `medium`
- `risk/elite_risk_metrics.py:94` fingerprint `2ec5d68b7d98d645` confidence `medium`
- `risk/elite_risk_metrics.py:95` fingerprint `861fd6fec99ab398` confidence `medium`
- `risk/elite_risk_metrics.py:249` fingerprint `d4a6167599d10970` confidence `medium`
- `risk/elite_risk_metrics.py:252` fingerprint `c0c5e97fa6c13137` confidence `medium`
- ... 5 more

## P1 temporal / NEGATIVE_SHIFT_LOOKAHEAD — 9 finding(s)

**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.

### Examples
- `pipeline/guards/temporal_target_guard.py:38` fingerprint `3d9f060bd8365a27` confidence `high`
- `pipeline/guards/temporal_target_guard.py:39` fingerprint `6a7758432ac64758` confidence `high`
- `pipeline/guards/temporal_target_guard.py:41` fingerprint `1da8f44f5165b280` confidence `high`
- `pipeline/guards/temporal_target_guard.py:42` fingerprint `1ad3e021e8895528` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:134` fingerprint `d1fb6ab4a00ece4d` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:135` fingerprint `b7cbe1fa08bf06c4` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:136` fingerprint `384cc0fd4ed81c3b` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:138` fingerprint `d0e33c64e5a6da25` confidence `high`
- ... 1 more

## P1 heavy_imports / HEAVY_TOP_LEVEL_IMPORT — 7 finding(s)

**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.

### Examples
- `colab/models/model_factory.py:8` fingerprint `964b926535b40f69` confidence `high`
- `colab/models/model_factory.py:9` fingerprint `11f3e21f1f1843bb` confidence `high`
- `colab/models/model_factory.py:89` fingerprint `f79515824ad9fafb` confidence `high`
- `colab/models/model_factory.py:90` fingerprint `46882002b08f23af` confidence `high`
- `colab/models/model_factory.py:115` fingerprint `008ed79bfd4f8e08` confidence `high`
- `features/nlp/models/finbert_pipeline.py:3` fingerprint `5fadebe514a66ce7` confidence `high`
- `features/nlp/models/finbert_pipeline.py:4` fingerprint `b845b85f447cbff1` confidence `high`
