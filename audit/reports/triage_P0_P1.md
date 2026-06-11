# Audit Triage Report

Selected severities: P0, P1
Selected findings: **390**

## Counts

### By severity
- P0: 22
- P1: 368

### By rule
- P0 temporal/NEGATIVE_SHIFT_LOOKAHEAD: 7
- P0 missing_policy/FILLNA_ZERO_SUSPICIOUS: 5
- P0 splits/RANDOM_TRAIN_TEST_SPLIT: 5
- P0 synthetic_gates/EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA: 3
- P0 missing_policy/BFILL_IN_CAUSAL_DATA: 2
- P1 error_policy/BROAD_EXCEPTION_SILENT_RETURN: 116
- P1 missing_policy/FILLNA_ZERO_SUSPICIOUS: 80
- P1 financial_math/SHARPE_SORTINO_STD_ZERO_REVIEW: 49
- P1 security/UNSAFE_MODEL_OR_PICKLE_LOAD: 31
- P1 security/PLACEHOLDER_SECRET_REVIEW: 29
- P1 missing_policy/PCT_CHANGE_IMPLICIT_FILL_METHOD: 20
- P1 model_routing/AUTOENCODER_ROUTING_REVIEW: 17
- P1 financial_math/VAR_SIGN_OR_EMPTY_DATA_REVIEW: 13
- P1 temporal/NEGATIVE_SHIFT_LOOKAHEAD: 5
- P1 heavy_imports/HEAVY_TOP_LEVEL_IMPORT: 4
- P1 missing_policy/BFILL_IN_CAUSAL_DATA: 4

---

## P0 temporal / NEGATIVE_SHIFT_LOOKAHEAD — 7 finding(s)

**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.

### Examples
- `pipeline/guards/temporal_target_guard.py:49` fingerprint `dbcce2f2afacb6b9` confidence `medium`
- `pipeline/guards/temporal_target_guard.py:50` fingerprint `c6511cee15bb7938` confidence `medium`
- `pipeline/stages/stage_0_data_generation.py:133` fingerprint `bd7e9be06e1fd965` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:134` fingerprint `eae698a69326b3a6` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:135` fingerprint `db397a00b56e5700` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:143` fingerprint `366feef5e1ece30f` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:144` fingerprint `44b9f107b7ea17d3` confidence `high`

## P0 missing_policy / FILLNA_ZERO_SUSPICIOUS — 5 finding(s)

**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.

### Examples
- `risk/analyzers/correlation_analyzer.py:29` fingerprint `da1eeea31bea9d50` confidence `high`
- `risk/analyzers/correlation_analyzer.py:31` fingerprint `314ee39358ec9201` confidence `high`
- `risk/metrics.py:36` fingerprint `736c347b5238c446` confidence `high`
- `risk/metrics.py:110` fingerprint `ada5bed299cd5d1d` confidence `high`
- `risk/metrics.py:153` fingerprint `2cede644d3a38e94` confidence `high`

## P0 splits / RANDOM_TRAIN_TEST_SPLIT — 5 finding(s)

**Problem:** train_test_split detected in time-series/trading code path.
**Why:** Random splits leak future regimes into train/validation and invalidate backtest-like evaluation.
**Fix:** Use chronological or purged time split with gap >= max target horizon.
**Test:** Assert train max timestamp < validation min timestamp and purge gap >= target horizon.

### Examples
- `calibration/calibration_engine.py:207` fingerprint `16a93f542bf8270f` confidence `high`
- `monitoring/ml_analytics.py:254` fingerprint `91c7e9a8d53eaeec` confidence `high`
- `pipeline/hybrid/model_training_orchestrator.py:96` fingerprint `814982e27edd8208` confidence `high`
- `training/portfolio_optimizer.py:201` fingerprint `9216bf2b32203d65` confidence `high`
- `training/portfolio_optimizer.py:238` fingerprint `d4c8b51628bb7fa2` confidence `high`

## P0 synthetic_gates / EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA — 3 finding(s)

**Problem:** Exception handler appears to return sample/synthetic/demo data.
**Why:** A failed real collector can silently inject fake data into train/eval.
**Fix:** Make sample fallback opt-in and mark data_kind/is_synthetic/eligible_for_training=False.
**Test:** Simulate collector failure and assert it raises or returns failed status unless allow_sample_fallback=True.

### Examples
- `analytics/unified_analytics_engine.py:109` fingerprint `45b1787e5c82db06` confidence `medium`
- `data/collectors/cftc_collector.py:219` fingerprint `76881225db52eec6` confidence `medium`
- `data/collectors/put_call_ratio_collector.py:161` fingerprint `064d9abe5ea2ba5f` confidence `medium`

## P0 missing_policy / BFILL_IN_CAUSAL_DATA — 2 finding(s)

**Problem:** bfill() detected in likely causal time-series path.
**Why:** Backward fill moves future-known values into earlier timestamps.
**Fix:** For train/eval causal data, remove bfill; use forward-fill only when availability policy permits, or leave NaN with indicator columns.
**Test:** Add a fixture where first known future value must not appear in earlier rows.

### Examples
- `features/enrichers/macro_features_enricher.py:277` fingerprint `de282fca71b68110` confidence `high`
- `features/enrichers/sentiment_features_enricher.py:296` fingerprint `fe5f0b363bb072d1` confidence `high`

## P1 error_policy / BROAD_EXCEPTION_SILENT_RETURN — 116 finding(s)

**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.

### Examples
- `analytics/arena/arena_battle.py:182` fingerprint `76cc7791c1b89308` confidence `high`
- `analytics/arena/performance_tracker.py:345` fingerprint `3aa4e08d28a9c6a8` confidence `high`
- `analytics/calculators/explainability_calculator.py:63` fingerprint `fe2ce98450e20a91` confidence `high`
- `analytics/calculators/explainability_calculator.py:88` fingerprint `c3d41d0d16c30156` confidence `high`
- `analytics/calculators/fama_french_factors.py:92` fingerprint `29ef19ce64834831` confidence `high`
- `analytics/context/ensemble_selector.py:125` fingerprint `918e9087efb89879` confidence `high`
- `analytics/reporting/automated_reports.py:67` fingerprint `d640bcf590ae94b0` confidence `high`
- `analytics/reporting/automated_reports.py:96` fingerprint `7ed2e3ae19569c27` confidence `high`
- ... 108 more

## P1 missing_policy / FILLNA_ZERO_SUSPICIOUS — 80 finding(s)

**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.

### Examples
- `algorithms/advanced_backtest_engine.py:38` fingerprint `8e9408713a149fea` confidence `high`
- `algorithms/advanced_backtest_engine.py:60` fingerprint `0df04dfdcb581e91` confidence `high`
- `analytics/analyzers/knn_similarity_finder.py:57` fingerprint `3ec4e82aca4aeccd` confidence `high`
- `analytics/analyzers/knn_similarity_finder.py:58` fingerprint `34d1b59fe7f370b3` confidence `high`
- `analytics/arena/arena_battle.py:125` fingerprint `abf5dcd5190e94c5` confidence `high`
- `analytics/calculators/drawdown_calculator.py:69` fingerprint `282fd8af0949d589` confidence `high`
- `analytics/calculators/macro_score_calculator.py:86` fingerprint `7df27fad00bae926` confidence `high`
- `analytics/calculators/macro_score_calculator.py:90` fingerprint `411301753a28063a` confidence `high`
- ... 72 more

## P1 financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — 49 finding(s)

**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.

### Examples
- `algorithms/advanced_backtest_engine.py:65` fingerprint `168b8999b1c40915` confidence `low`
- `algorithms/advanced_backtest_engine.py:125` fingerprint `a586eaefc2ffa8d5` confidence `low`
- `algorithms/advanced_backtest_engine.py:129` fingerprint `4b75807bd0b19587` confidence `low`
- `algorithms/advanced_backtest_engine.py:130` fingerprint `4f210d71cc87623a` confidence `low`
- `algorithms/advanced_backtest_engine.py:132` fingerprint `e984cb9f2cd21fc1` confidence `low`
- `algorithms/advanced_backtest_engine.py:133` fingerprint `aad0f0a35db7e52e` confidence `low`
- `algorithms/advanced_backtest_engine.py:134` fingerprint `6a3f2e1b721ff2bd` confidence `low`
- `algorithms/advanced_backtest_engine.py:135` fingerprint `1270ce7af8578de4` confidence `low`
- ... 41 more

## P1 security / UNSAFE_MODEL_OR_PICKLE_LOAD — 31 finding(s)

**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.

### Examples
- `analytics/arena/arena_battle.py:180` fingerprint `74973ce280df7539` confidence `medium`
- `calibration/adaptive_confidence_calibrator.py:290` fingerprint `0184797265d097f4` confidence `medium`
- `colab/utils/utils.py:66` fingerprint `882c742ba4634b38` confidence `medium`
- `core/cache/cache_manager.py:146` fingerprint `547c7d2975667c7c` confidence `medium`
- `ensembling/stacked_ensemble.py:307` fingerprint `b55dfb59dd0fc6a1` confidence `medium`
- `models/ensemble/confidence_calibrator.py:117` fingerprint `ea04b4aa9d463716` confidence `medium`
- `models/ensemble/enhanced_ensemble.py:115` fingerprint `c422faaa56b27a7e` confidence `medium`
- `models/ensemble/enhanced_ensemble.py:142` fingerprint `4dd700ce6d14089b` confidence `medium`
- ... 23 more

## P1 security / PLACEHOLDER_SECRET_REVIEW — 29 finding(s)

**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.

### Examples
- `config/unified_config_manager.py:193` fingerprint `407b8bb9bc7513a2` confidence `medium`
- `config/unified_config_manager.py:199` fingerprint `20bf97463b26dfdb` confidence `medium`
- `config/unified_config_manager.py:221` fingerprint `38772162bb1edd8b` confidence `medium`
- `config/unified_config_manager.py:222` fingerprint `986062d2ba0009d1` confidence `medium`
- `config/unified_config_manager.py:230` fingerprint `099f3c6490d790f6` confidence `medium`
- `config/unified_config_manager.py:231` fingerprint `f2f3ea8576ae45d1` confidence `medium`
- `config/unified_config_manager.py:244` fingerprint `3df3a9ad367705bb` confidence `medium`
- `config/unified_config_manager.py:245` fingerprint `ec5afbb13fc0bf60` confidence `medium`
- ... 21 more

## P1 missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — 20 finding(s)

**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.

### Examples
- `pipeline/stages/evaluation/metrics_calculator.py:77` fingerprint `a59dc42ef6c28ccf` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:100` fingerprint `296b666b6332d52d` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:103` fingerprint `ada8e8a6616c387f` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:104` fingerprint `e73c344b38e6b078` confidence `high`
- `pipeline/stages/stage_0_data_generation.py:105` fingerprint `68641064fbc48d3d` confidence `high`
- `pipeline/stages/trading/recommendation_engine.py:167` fingerprint `18d41c666c68a8e2` confidence `high`
- `pipeline/stages/trading/recommendation_engine.py:199` fingerprint `c5ced334a3333a2e` confidence `high`
- `risk/analyzers/correlation_analyzer.py:29` fingerprint `b325489402126384` confidence `high`
- ... 12 more

## P1 model_routing / AUTOENCODER_ROUTING_REVIEW — 17 finding(s)

**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.

### Examples
- `analytics/analyzers/model_comparison_analyzer.py:30` fingerprint `4b092c94076c5158` confidence `medium`
- `colab/models/model_factory.py:123` fingerprint `f86fc1a4f647f4ff` confidence `medium`
- `colab/models/model_factory.py:215` fingerprint `8b1ab9d6a656f355` confidence `medium`
- `colab/models/torch_models.py:5` fingerprint `aab1fe1eadf38829` confidence `medium`
- `colab/models/torch_models.py:117` fingerprint `68e9824f085fd4ae` confidence `medium`
- `factories/model_factory.py:18` fingerprint `3724778cc0f3f8f8` confidence `medium`
- `models/loader.py:438` fingerprint `344d81598e217a5b` confidence `medium`
- `models/neural/autoencoder_model.py:1` fingerprint `5cd70b4e0934aecd` confidence `medium`
- ... 9 more

## P1 financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — 13 finding(s)

**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.

### Examples
- `analytics/analyzers/risk_decomposition_analyzer.py:119` fingerprint `1474ac820c2cd449` confidence `medium`
- `analytics/calculators/risk_reward_calculator.py:152` fingerprint `18d0321c58a8c20d` confidence `medium`
- `metrics/financial/financial_metrics_library.py:182` fingerprint `f873d7f066535916` confidence `medium`
- `metrics/financial/financial_metrics_library.py:184` fingerprint `d2978c50798cec24` confidence `medium`
- `risk/elite_risk_metrics.py:94` fingerprint `2ec5d68b7d98d645` confidence `medium`
- `risk/elite_risk_metrics.py:95` fingerprint `861fd6fec99ab398` confidence `medium`
- `risk/elite_risk_metrics.py:249` fingerprint `d4a6167599d10970` confidence `medium`
- `risk/elite_risk_metrics.py:252` fingerprint `c0c5e97fa6c13137` confidence `medium`
- ... 5 more

## P1 temporal / NEGATIVE_SHIFT_LOOKAHEAD — 5 finding(s)

**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.

### Examples
- `data/synthetic/data_generator.py:135` fingerprint `7454eeb284065cf6` confidence `high`
- `data/synthetic/data_generator.py:136` fingerprint `4fb9371ab7294041` confidence `high`
- `data/synthetic/data_generator.py:137` fingerprint `22600faab7382a4b` confidence `high`
- `data/synthetic/data_generator.py:149` fingerprint `52fd0e96545d8a1b` confidence `high`
- `validation/data_leakage_detector.py:126` fingerprint `406478a9797916bd` confidence `high`

## P1 heavy_imports / HEAVY_TOP_LEVEL_IMPORT — 4 finding(s)

**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.

### Examples
- `colab/models/model_factory.py:8` fingerprint `964b926535b40f69` confidence `high`
- `colab/models/model_factory.py:9` fingerprint `11f3e21f1f1843bb` confidence `high`
- `features/nlp/models/finbert_pipeline.py:3` fingerprint `5fadebe514a66ce7` confidence `high`
- `features/nlp/models/finbert_pipeline.py:4` fingerprint `b845b85f447cbff1` confidence `high`

## P1 missing_policy / BFILL_IN_CAUSAL_DATA — 4 finding(s)

**Problem:** bfill() detected in likely causal time-series path.
**Why:** Backward fill moves future-known values into earlier timestamps.
**Fix:** For train/eval causal data, remove bfill; use forward-fill only when availability policy permits, or leave NaN with indicator columns.
**Test:** Add a fixture where first known future value must not appear in earlier rows.

### Examples
- `data/management/data_manager.py:464` fingerprint `37835177ea6a5b75` confidence `high`
- `processing/cleaners.py:91` fingerprint `96b413f9a9768aad` confidence `high`
- `processing/cleaners.py:94` fingerprint `7fd9c88c1a8b092b` confidence `high`
- `utils/smart_missing_data_handler.py:243` fingerprint `cbe0acd2fa6543a1` confidence `high`
