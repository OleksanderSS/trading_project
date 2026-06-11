# Deep Static Audit Findings

## Summary

Total findings: **1234**

### By severity

- P0: 22
- P1: 368
- P2: 433
- P3: 411

### By category

- error_policy: 383
- determinism: 337
- security: 118
- missing_policy: 111
- financial_math: 94
- data_lineage: 69
- heavy_imports: 48
- config_factory: 31
- model_routing: 17
- temporal: 12
- architecture: 6
- splits: 5
- synthetic_gates: 3

---


## P0

### missing_policy / BFILL_IN_CAUSAL_DATA — `features/enrichers/macro_features_enricher.py:277`
**Problem:** bfill() detected in likely causal time-series path.
**Why:** Backward fill moves future-known values into earlier timestamps.
**Fix:** For train/eval causal data, remove bfill; use forward-fill only when availability policy permits, or leave NaN with indicator columns.
**Test:** Add a fixture where first known future value must not appear in earlier rows.
**Confidence:** high  
```python
df = df.bfill()
```

### missing_policy / BFILL_IN_CAUSAL_DATA — `features/enrichers/sentiment_features_enricher.py:296`
**Problem:** bfill() detected in likely causal time-series path.
**Why:** Backward fill moves future-known values into earlier timestamps.
**Fix:** For train/eval causal data, remove bfill; use forward-fill only when availability policy permits, or leave NaN with indicator columns.
**Test:** Add a fixture where first known future value must not appear in earlier rows.
**Confidence:** high  
```python
sentiment_values.groupby(df_enriched['ticker']).ffill()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `risk/analyzers/correlation_analyzer.py:29`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns_df[symbol] = market_data[symbol].pct_change().fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `risk/analyzers/correlation_analyzer.py:31`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns_df[symbol] = market_data["close"][symbol].pct_change().fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `risk/metrics.py:36`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
symbol_returns = close_df[symbol].pct_change().fillna(0).dropna()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `risk/metrics.py:110`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
symbol_returns = symbol_prices.pct_change().fillna(0).dropna()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `risk/metrics.py:153`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = close_df.pct_change().fillna(0).dropna()
```

### splits / RANDOM_TRAIN_TEST_SPLIT — `calibration/calibration_engine.py:207`
**Problem:** train_test_split detected in time-series/trading code path.
**Why:** Random splits leak future regimes into train/validation and invalidate backtest-like evaluation.
**Fix:** Use chronological or purged time split with gap >= max target horizon.
**Test:** Assert train max timestamp < validation min timestamp and purge gap >= target horizon.
**Confidence:** high  
```python
X_train, X_val, y_train, y_val = train_test_split(X, y,
```

### splits / RANDOM_TRAIN_TEST_SPLIT — `monitoring/ml_analytics.py:254`
**Problem:** train_test_split detected in time-series/trading code path.
**Why:** Random splits leak future regimes into train/validation and invalidate backtest-like evaluation.
**Fix:** Use chronological or purged time split with gap >= max target horizon.
**Test:** Assert train max timestamp < validation min timestamp and purge gap >= target horizon.
**Confidence:** high  
```python
X_train, X_test, y_train, y_test = train_test_split(features_df,
```

### splits / RANDOM_TRAIN_TEST_SPLIT — `pipeline/hybrid/model_training_orchestrator.py:96`
**Problem:** train_test_split detected in time-series/trading code path.
**Why:** Random splits leak future regimes into train/validation and invalidate backtest-like evaluation.
**Fix:** Use chronological or purged time split with gap >= max target horizon.
**Test:** Assert train max timestamp < validation min timestamp and purge gap >= target horizon.
**Confidence:** high  
```python
split_data = self._prepare_train_test_split(c_features_df, c_targets_df, available_features, target_col)
```

### splits / RANDOM_TRAIN_TEST_SPLIT — `training/portfolio_optimizer.py:201`
**Problem:** train_test_split detected in time-series/trading code path.
**Why:** Random splits leak future regimes into train/validation and invalidate backtest-like evaluation.
**Fix:** Use chronological or purged time split with gap >= max target horizon.
**Test:** Assert train max timestamp < validation min timestamp and purge gap >= target horizon.
**Confidence:** high  
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### splits / RANDOM_TRAIN_TEST_SPLIT — `training/portfolio_optimizer.py:238`
**Problem:** train_test_split detected in time-series/trading code path.
**Why:** Random splits leak future regimes into train/validation and invalidate backtest-like evaluation.
**Fix:** Use chronological or purged time split with gap >= max target horizon.
**Test:** Assert train max timestamp < validation min timestamp and purge gap >= target horizon.
**Confidence:** high  
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### synthetic_gates / EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA — `analytics/unified_analytics_engine.py:109`
**Problem:** Exception handler appears to return sample/synthetic/demo data.
**Why:** A failed real collector can silently inject fake data into train/eval.
**Fix:** Make sample fallback opt-in and mark data_kind/is_synthetic/eligible_for_training=False.
**Test:** Simulate collector failure and assert it raises or returns failed status unless allow_sample_fallback=True.
**Confidence:** medium  
```python
except Exception as e:
```

### synthetic_gates / EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA — `data/collectors/cftc_collector.py:219`
**Problem:** Exception handler appears to return sample/synthetic/demo data.
**Why:** A failed real collector can silently inject fake data into train/eval.
**Fix:** Make sample fallback opt-in and mark data_kind/is_synthetic/eligible_for_training=False.
**Test:** Simulate collector failure and assert it raises or returns failed status unless allow_sample_fallback=True.
**Confidence:** medium  
```python
except Exception as e:
```

### synthetic_gates / EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA — `data/collectors/put_call_ratio_collector.py:161`
**Problem:** Exception handler appears to return sample/synthetic/demo data.
**Why:** A failed real collector can silently inject fake data into train/eval.
**Fix:** Make sample fallback opt-in and mark data_kind/is_synthetic/eligible_for_training=False.
**Test:** Simulate collector failure and assert it raises or returns failed status unless allow_sample_fallback=True.
**Confidence:** medium  
```python
except Exception as e:
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `pipeline/guards/temporal_target_guard.py:49`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** medium  
```python
results['target_volatility_1d'] = ret1_series.groupby(df_enriched['ticker']).transform(lambda s: s.rolling(window=5, min_periods=1).std().shift(-1))  # audit-ignore: target label
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `pipeline/guards/temporal_target_guard.py:50`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** medium  
```python
results['target_volatility_5d'] = ret1_series.groupby(df_enriched['ticker']).transform(lambda s: s.rolling(window=20, min_periods=1).std().shift(-5))  # audit-ignore: target label
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `pipeline/stages/stage_0_data_generation.py:133`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** high  
```python
targets['return_1h'] = features_df['close'].pct_change(1, fill_method=None).shift(-1)  # audit-ignore: target label
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `pipeline/stages/stage_0_data_generation.py:134`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** high  
```python
targets['return_4h'] = features_df['close'].pct_change(4, fill_method=None).shift(-4)  # audit-ignore: target label
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `pipeline/stages/stage_0_data_generation.py:135`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** high  
```python
targets['return_24h'] = features_df['close'].pct_change(24, fill_method=None).shift(-24)  # audit-ignore: target label
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `pipeline/stages/stage_0_data_generation.py:143`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** high  
```python
targets['volatility_1h'] = features_df['close'].pct_change(fill_method=None).rolling(window=5, min_periods=1).std().shift(-1)  # audit-ignore: target label
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `pipeline/stages/stage_0_data_generation.py:144`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** high  
```python
targets['volatility_4h'] = features_df['close'].pct_change(fill_method=None).rolling(window=10, min_periods=1).std().shift(-4)  # audit-ignore: target label
```


## P1

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/arena/arena_battle.py:182`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/arena/performance_tracker.py:345`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/calculators/explainability_calculator.py:63`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/calculators/explainability_calculator.py:88`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/calculators/fama_french_factors.py:92`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/context/ensemble_selector.py:125`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/reporting/automated_reports.py:67`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/reporting/automated_reports.py:96`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/reporting/results_manager.py:88`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/reporting/results_manager.py:108`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/reporting/results_manager.py:121`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/reporting/results_manager.py:217`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `analytics/reporting/visualization.py:39`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `core/cloud/gcs_manager.py:106`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `core/file_management/file_manager.py:95`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `core/file_management/file_manager.py:135`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `core/file_management/file_manager.py:212`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `core/security/secure_secrets_manager.py:45`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `core/security/secure_secrets_manager.py:117`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `core/system/archive_manager.py:59`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/aaii_sentiment_collector.py:61`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/aaii_sentiment_collector.py:93`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/aaii_sentiment_collector.py:107`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/alternative_me_collector.py:67`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/alternative_me_collector.py:112`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/alternative_me_collector.py:138`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/bigquery_collector.py:44`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/cftc_collector.py:80`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/cftc_collector.py:142`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/collector_factory.py:87`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/custom_csv_collector.py:44`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/economic_calendar_collector.py:101`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/economic_calendar_collector.py:155`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/fear_greed_collector.py:61`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/fear_greed_collector.py:108`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/fred_collector.py:112`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/google_news_collector.py:201`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/huggingface_collector.py:42`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/huggingface_collector.py:107`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/insider_collector.py:68`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/insider_collector.py:185`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/local_file_collector.py:56`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/put_call_ratio_collector.py:72`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/reddit_sentiment_collector.py:75`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/reddit_sentiment_collector.py:160`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/rss_collector.py:169`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as exc:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/rss_collector.py:185`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as exc:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/sec_filings_collector.py:202`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/vix_collector.py:92`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/collectors/vix_collector.py:161`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/management/connectors/bigquery_connector.py:51`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `data/management/data_versioning.py:35`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `devtools/rule_generator.py:53`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/enrichers/news_impact_enricher.py:323`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/feature_orchestrator.py:87`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/feature_selection_cache.py:42`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/news_impact_classifier.py:51`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/news_impact_classifier.py:68`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/nlp/extractors/entity_extractor.py:73`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/nlp/processors/news_harmonizer.py:13`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/selection/smart_selector.py:219`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/selection/smart_selector.py:264`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/selection/smart_selector.py:275`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/selection/volatility_driver_selector.py:60`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/utils/modular_adaptive_technical_indicators.py:227`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `features/utils/modular_adaptive_technical_indicators.py:248`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `integration/ensemble_performance_bridge.py:154`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `integration/ensemble_performance_bridge.py:185`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `integration/ensemble_performance_bridge.py:217`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `integration/ensemble_performance_bridge.py:238`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `meta_learning/memory/diary_engine.py:539`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `metrics/model/ml_evaluator.py:101`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/analysis/overfitting_detection/metrics.py:22`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/ensemble/correlation/correlation_engine.py:168`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/ensemble/correlation/correlation_engine.py:212`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/ensemble/correlation/correlation_engine.py:318`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/ensemble/correlation/correlation_engine.py:338`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/ensemble/correlation/correlation_engine.py:395`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/ensemble/correlation/correlation_engine.py:499`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/ensemble/enhanced_ensemble.py:152`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/loader.py:221`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/loader.py:232`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as fallback_error:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/model_pool.py:137`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/model_selector/adaptive_selector.py:131`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/model_selector/adaptive_selector.py:281`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/model_selector/smart_selector.py:33`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/monitoring/drift/drift_calculator.py:199`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/monitoring/drift/drift_calculator.py:286`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/neural/transformer_model.py:340`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/prototypes/prototype.py:98`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/statistics/model_statistics.py:84`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `models/tree/catboost_model.py:144`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `monitoring/monitoring_system.py:129`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `monitoring/monitoring_system.py:187`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `monitoring/monitoring_system.py:243`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `monitoring/monitoring_system.py:427`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `optimization/portfolio/optimizer.py:432`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `pipeline/hybrid/model_training_orchestrator.py:120`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `pipeline/hybrid/selected_features_processor.py:105`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `pipeline/stages/evaluation/backtest_analyzer.py:237`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `pipeline/stages/evaluation/metrics_calculator.py:69`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `pipeline/stages/evaluation/metrics_calculator.py:107`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `pipeline/stages/evaluation/metrics_calculator.py:141`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `pipeline/stages/evaluation/metrics_calculator.py:202`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `pipeline/stages/modeling/orchestration.py:55`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `pipeline/stages/prediction/data_preparer.py:301`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `pipeline/stages/prediction/scaler_service.py:89`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `predictions/caching.py:89`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `risk/metrics.py:41`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `risk/metrics.py:82`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `risk/metrics.py:134`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `risk/metrics.py:229`
**Problem:** Broad exception returns silent fallback: Dict.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `simulation/simulation_engine.py:115`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `trading/portfolio_manager.py:272`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `trading/signal_processor.py:88`
**Problem:** Broad exception returns silent fallback: None.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### error_policy / BROAD_EXCEPTION_SILENT_RETURN — `utils/trading_calendar.py:79`
**Problem:** Broad exception returns silent fallback: List.
**Why:** Pipeline may continue with None/{}/[] as if the stage succeeded.
**Fix:** Return a typed StageResult with status failed/degraded, or re-raise for fatal stages.
**Test:** Test that fatal failures in target generation/split/training do not continue silently.
**Confidence:** high  
```python
except Exception as e:
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:65`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
def _calculate_sharpe(self, equity: pd.Series, risk_free_rate: float=0.02
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:125`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe_values = []
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:129`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe_values.append(perf.get('sharpe', 0))
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:130`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
if len(sharpe_values) < 2:
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:132`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
std_sharpe = np.std(sharpe_values)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:133`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
mean_sharpe = np.mean(sharpe_values)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:134`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
if mean_sharpe != 0:
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:135`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
cv = abs(std_sharpe / mean_sharpe)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:148`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return {'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:154`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return {'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/advanced_backtest_engine.py:157`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe = returns.mean() / std_val * np.sqrt(252
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/walk_forward_optimizer.py:126`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/walk_forward_optimizer.py:131`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/walk_forward_optimizer.py:134`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe = float(returns.mean() / std_val * np.sqrt(252)) if std_val > 0 else 0.0
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `algorithms/walk_forward_optimizer.py:138`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return {"return": total_return, "sharpe": sharpe, "max_drawdown": max_drawdown}
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/analyzers/hedge_fund_analyzer.py:84`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
metrics['sharpe_ratio'
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/analyzers/hedge_fund_analyzer.py:85`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
] = RiskRewardCalculator.calculate_sharpe_ratio(returns,
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/analyzers/performance_attribution_analyzer.py:108`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
port_sharpe = ((port_returns.mean() - const_rf_daily) / port_std * np.sqrt(252)) if port_std > 0 else 0.0
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `analytics/analyzers/risk_decomposition_analyzer.py:119`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
var_05_threshold = np.percentile(weighted_returns, 5)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/analyzers/risk_decomposition_analyzer.py:127`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
realized_sharpe = excess_mean / np.std(weighted_returns) * np.sqrt(
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/arena/arena_battle.py:390`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe_ratio = np.mean(predictions) / (np.std(predictions) + 1e-08
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/arena/arena_battle.py:393`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
recall=accuracy, f1_score=accuracy, sharpe_ratio=
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/arena/arena_battle.py:394`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe_ratio, max_drawdown=0.0, win_rate=accuracy,
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/calculators/fama_french_factors.py:241`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
'annualized_sharpe': float((f_series.mean() / f_series.std()) * np.sqrt(252)) if f_series.std() != 0 else 0.0,
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/calculators/risk_reward_calculator.py:76`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
def calculate_sharpe_ratio(returns: pd.Series, config: Optional[TradeConfig] = None) -> float:
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/calculators/risk_reward_calculator.py:77`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
"""Calculates the annualized Sharpe Ratio."""
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/calculators/risk_reward_calculator.py:85`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
annualized_sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(config.periods_per_year)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/calculators/risk_reward_calculator.py:86`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return float(annualized_sharpe)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/calculators/risk_reward_calculator.py:106`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sortino_ratio = (expected_return - target_return) / downside_std
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/calculators/risk_reward_calculator.py:107`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
annualized_sortino = sortino_ratio * np.sqrt(config.periods_per_year)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `analytics/calculators/risk_reward_calculator.py:108`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return float(annualized_sortino)
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `analytics/calculators/risk_reward_calculator.py:152`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
var = returns.quantile(quantile)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `backtesting/advanced/advanced_engine.py:198`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return {'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `backtesting/advanced/advanced_engine.py:204`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return {'return': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0}
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `backtesting/advanced/advanced_engine.py:207`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe = returns.mean() / std_val * np.sqrt(252) if std_val > 0 else 0.0
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `backtesting/advanced/advanced_engine.py:364`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
def _calculate_sharpe(self, equity: pd.Series, risk_free_rate: float=0.02
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `backtesting/advanced/advanced_engine.py:366`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
"""Розрахунок Sharpe Ratio"""
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `calibration/calibration_engine.py:267`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return float(avg_sharpe)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `calibration/calibration_engine.py:269`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
def _calculate_sharpe_ratio(self, y_true: np.ndarray, y_pred: np.ndarray
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `calibration/calibration_engine.py:271`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
"""Calculate Sharpe ratio from predictions."""
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `calibration/calibration_engine.py:278`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe = mean_return / std_return * np.sqrt(252)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `calibration/calibration_engine.py:279`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe = np.clip(sharpe, -5.0, 5.0)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `calibration/calibration_engine.py:280`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
return float(sharpe)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `meta_learning/memory/diary_engine.py:402`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) != 0 else 0
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `meta_learning/real_time_learning.py:272`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
"""Розраховує Sharpe ratio"""
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `metrics/financial/financial_metrics_library.py:51`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
def calculate_sharpe_ratio(
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `metrics/financial/financial_metrics_library.py:54`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
"""Calculates annualized Sharpe Ratio."""
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `metrics/financial/financial_metrics_library.py:182`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
return {"var": 0.0, "cvar": 0.0}
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `metrics/financial/financial_metrics_library.py:184`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
var = returns.quantile(quantile)
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `metrics/financial/portfolio_metrics.py:84`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(self._trading_days_per_year) if excess_returns.std() > 0 else 0.0
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `metrics/financial/portfolio_metrics.py:88`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `monitoring/health_hub.py:286`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
for metric in ['win_rate', 'sharpe_ratio']:
```

### financial_math / SHARPE_SORTINO_STD_ZERO_REVIEW — `pipeline/stages/evaluation/metrics_calculator.py:85`
**Problem:** Sharpe/Sortino calculation near std usage without obvious zero-std guard.
**Why:** Constant returns can produce inf/nan Sharpe and break model ranking/risk reports.
**Fix:** Guard zero/near-zero denominator explicitly and return status/NaN instead of silent 0 unless policy says otherwise.
**Test:** Test constant returns and single-observation returns.
**Confidence:** low  
```python
sharpe_ratio = returns.mean() / (volatility + 1e-9) if volatility > 0 else 0
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `risk/elite_risk_metrics.py:94`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
var_percentile = (1 - confidence_level) * 100
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `risk/elite_risk_metrics.py:95`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
var_value: float = float(-np.percentile(recent_returns, var_percentile)
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `risk/elite_risk_metrics.py:249`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
return {'var': 0.05, 'cvar': 0.08, 'method': 'parametric_fallback'}
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `risk/elite_risk_metrics.py:252`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
return {'var': 0.05, 'cvar': 0.08, 'method': 'parametric_fallback'}
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `risk/elite_risk_metrics.py:290`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
return {'var': 0.05, 'cvar': 0.08, 'method': 'monte_carlo_fallback'
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `risk/elite_risk_metrics.py:294`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
return {'var': 0.05, 'cvar': 0.08, 'method': 'monte_carlo_fallback'
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `risk/elite_risk_metrics.py:304`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `risk_management/var_calculator.py:13`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
return {'var': 0.0}
```

### financial_math / VAR_SIGN_OR_EMPTY_DATA_REVIEW — `risk_management/var_calculator.py:17`
**Problem:** VaR percentile/zero-return pattern found.
**Why:** VaR sign and empty-data behavior are often inconsistent; returning 0 for no data means false no-risk.
**Fix:** Use var_loss_positive naming, return insufficient_data for empty returns, and apply/document horizon scaling.
**Test:** Test empty returns, positive-only returns, negative tail returns, and time_horizon > 1.
**Confidence:** medium  
```python
var_val = -np.percentile(returns, percentile)
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `colab/models/model_factory.py:8`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `colab/models/model_factory.py:9`
**Problem:** Top-level import of heavy optional dependency 'torch.nn'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch.nn as nn
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `features/nlp/models/finbert_pipeline.py:3`
**Problem:** Top-level import of heavy optional dependency 'transformers'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `features/nlp/models/finbert_pipeline.py:4`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `algorithms/advanced_backtest_engine.py:38`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
net_equity = raw_equity - costs.cumsum().reindex(raw_equity.index,
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `algorithms/advanced_backtest_engine.py:60`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = prices.pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/analyzers/knn_similarity_finder.py:57`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
X_hist = X_hist[common_cols].fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/analyzers/knn_similarity_finder.py:58`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
X_target = X_target[common_cols].fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/arena/arena_battle.py:125`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
baseline_preds = actual_targets.shift(1).rolling(window=5).mean(
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/calculators/drawdown_calculator.py:69`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
drawdown_blocks = (is_underwater.astype(int).diff().fillna(0) != 0).cumsum()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/calculators/macro_score_calculator.py:86`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
return aligned_series.fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/calculators/macro_score_calculator.py:90`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
return series.pct_change(periods=int(rolling_window/12), fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/calculators/macro_score_calculator.py:129`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
scaled_values = minmax_scale(composite_score.fillna(0), feature_range=(0, 100))
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/context/market_context_analyzer.py:63`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
final_vector = context_vector.fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/context/market_context_analyzer.py:101`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
return df[self.close_col].pct_change(fill_method=None).fillna(0).tail(5).std()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/context/market_context_analyzer.py:106`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
return df[self.close_col].pct_change(fill_method=None).fillna(0).tail(20).std()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/context/market_regime_analyzer.py:55`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = df["close"].pct_change(fill_method=None).fillna(0).fillna(0).values
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/detectors/anomaly_detector.py:71`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
anomaly_labels = self.isolation_forest.predict(numeric_features.fillna(0))
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/utils/analytics_math.py:17`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = prices.pct_change(fill_method=None).fillna(0).dropna()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `analytics/utils/analytics_math.py:54`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns_clean = returns.fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `backtesting/advanced/advanced_engine.py:329`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `backtesting/advanced/advanced_engine.py:334`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
lagged_weights = positions.shift(1).fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `backtesting/advanced/advanced_engine.py:337`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
turnover = lagged_weights.diff().abs().sum(axis=1).fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `backtesting/advanced/advanced_engine.py:353`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
aligned = aligned.apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `backtesting/advanced/advanced_engine.py:356`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
return aligned.div(exposure, axis=0).fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `data/management/data_cleaner.py:13`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df[numeric_cols] = df[numeric_cols].fillna(0)
```

### missing_policy / BFILL_IN_CAUSAL_DATA — `data/management/data_manager.py:464`
**Problem:** bfill() detected in likely causal time-series path.
**Why:** Backward fill moves future-known values into earlier timestamps.
**Fix:** For train/eval causal data, remove bfill; use forward-fill only when availability policy permits, or leave NaN with indicator columns.
**Test:** Add a fixture where first known future value must not appear in earlier rows.
**Confidence:** high  
```python
df[numeric_cols] = df[numeric_cols].bfill()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `data/management/data_manager.py:465`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df[numeric_cols] = df[numeric_cols].fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `data/synthetic/data_generator.py:99`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
features['volatility'] = features['close'].pct_change(fill_method=None).fillna(0).rolling(window=20, min_periods=2).std().shift(1)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `data/synthetic/data_generator.py:103`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
features['returns_1h'] = features['close'].pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `data/synthetic/data_generator.py:104`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
features['returns_4h'] = features['close'].pct_change(4, fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `data/synthetic/data_generator.py:105`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
features['returns_24h'] = features['close'].pct_change(24, fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/context_map_enricher.py:120`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
return pd.Series(state, index=champ_data.index).reindex(df.index).ffill().fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/context_map_enricher.py:139`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df[state_name] = df[feat].fillna(0).astype(int)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/context_map_enricher.py:161`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = res_df[col].pct_change(fill_method=None).replace([np.inf, -np.inf], 0).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/context_map_enricher.py:162`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
rolling_std = returns.rolling(window=20, min_periods=1).std().fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/derived_features_enricher.py:89`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched[self.returns_column] = df_enriched[price_target_col].pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/hype_enricher.py:161`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched['hype_score'] = df_enriched['news_count'].fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/keyword_entity_enricher.py:241`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_merged['keyword_count'] = df_merged['keyword_count'].fillna(0
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/keyword_entity_enricher.py:243`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_merged['entity_count'] = df_merged['entity_count'].fillna(0).astype(
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/news_impact_enricher.py:227`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched['news_impact_score'] = impact_scores_aligned.fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/news_impact_enricher.py:233`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched['news_significance_level'] = significance_aligned.map(significance_map).fillna(0).astype(int)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/news_impact_enricher.py:388`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched['news_impact_score'] = impact_scores_aligned.fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/news_impact_enricher.py:394`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched['news_significance_level'] = significance_aligned.map(significance_map).fillna(0).astype(int)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/news_quality_enricher.py:162`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_merged['news_quality_score'] = df_merged['news_quality_score'].fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/news_quality_enricher.py:163`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_merged['news_source_count'] = df_merged['news_source_count'].fillna(0).astype(int)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/sentiment_features_enricher.py:296`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
sentiment_values.groupby(df_enriched['ticker']).ffill()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/sentiment_features_enricher.py:330`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
ticker_group[f'sentiment_std_{window}'] = (ticker_group[sentiment_col]
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/sentiment_features_enricher.py:344`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
ticker_group['sentiment_velocity'] = (ticker_group[sentiment_col]
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/significance_features_enricher.py:147`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_out['_temp_returns'] = df_out['close'].pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/technical_analysis_enricher.py:190`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = df_enriched['close'].pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/technical_analysis_enricher.py:193`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched['VOLATILITY_5'] = returns.rolling(5, min_periods=1).std().fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/technical_analysis_enricher.py:203`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched['RSI_VELOCITY'] = df_enriched['RSI_14'].diff(3).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/technical_analysis_enricher.py:219`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = df_enriched['close'].pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/technical_analysis_enricher.py:248`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = df_enriched['close'].pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/volatility_enricher.py:38`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched["returns"] = df_enriched["close"].pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/volume_enricher.py:47`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched["volume_roc"] = df_enriched["volume"].pct_change(periods=5, fill_method=None).fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/volume_enricher.py:51`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched["volume"] * df_enriched["close"].pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/enrichers/volume_enricher.py:62`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df_enriched["volume_rs"] = (
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/selection/volatility_driver_selector.py:37`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
y_vol = df[target_col].pct_change(fill_method=None).abs().fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/selection/volatility_driver_selector.py:41`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
x_sub = df[valid_aux].ffill().fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/utils/modular_adaptive_technical_indicators.py:27`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = prices.pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/utils/modular_adaptive_technical_indicators.py:41`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
delta_clean = delta.fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/utils/modular_adaptive_technical_indicators.py:81`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = prices.pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/utils/modular_adaptive_technical_indicators.py:111`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
return macd_composite.fillna(0), signal_composite.fillna(0), histogram_composite.fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `features/utils/modular_adaptive_technical_indicators.py:235`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = prices.pct_change(fill_method=None).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `models/neural/transformer_model.py:313`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
X = df[feature_cols].fillna(0).values
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `pipeline/hybrid/data_manager.py:82`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df[numeric_cols] = df[numeric_cols].fillna(0).replace([np.inf, -np.
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `pipeline/hybrid/data_utils.py:37`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df[numeric_cols] = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `pipeline/stages/evaluation/analytics.py:54`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
predictions = signals_df["signal"].map({"BUY": 1, "SELL": -1, "HOLD": 0}).fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `pipeline/stages/evaluation/analytics.py:62`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
"benchmark_returns": pd.DataFrame({"Benchmark": price_data["close"].pct_change(fill_method=None).fillna(0).fillna(0)})
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `pipeline/stages/evaluation/metrics_calculator.py:77`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns = values.pct_change().dropna()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `pipeline/stages/prediction/data_preparation_service.py:135`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
ticker_df_clean = ticker_df_clean.fillna(0).replace([np.inf, -np.inf], 0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `pipeline/stages/prediction/data_preparer.py:213`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
ticker_df_clean = ticker_df_clean.fillna(0).replace([np.inf, -np.
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `pipeline/stages/prediction/prediction_context_manager.py:42`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
ticker_df_clean = ticker_df_clean.fillna(0).replace([np.inf, -np.inf], 0)
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `pipeline/stages/stage_0_data_generation.py:100`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
features['volatility'] = features['close'].pct_change().rolling(window=20, min_periods=1).std()
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `pipeline/stages/stage_0_data_generation.py:103`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
features['returns_1h'] = features['close'].pct_change()
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `pipeline/stages/stage_0_data_generation.py:104`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
features['returns_4h'] = features['close'].pct_change(4)
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `pipeline/stages/stage_0_data_generation.py:105`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
features['returns_24h'] = features['close'].pct_change(24)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `pipeline/stages/stage_0_data_generation.py:165`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
targets = targets.ffill().fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `pipeline/stages/trading/recommendation_engine.py:167`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = ticker_df['close'].pct_change().fillna(0).dropna(
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `pipeline/stages/trading/recommendation_engine.py:167`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns = ticker_df['close'].pct_change().fillna(0).dropna(
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `pipeline/stages/trading/recommendation_engine.py:199`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
returns = ticker_df['close'].pct_change().fillna(0).dropna(
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `pipeline/stages/trading/recommendation_engine.py:199`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns = ticker_df['close'].pct_change().fillna(0).dropna(
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `processing/cleaners.py:55`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
col_mask = (z_scores.abs() > threshold).fillna(False)
```

### missing_policy / BFILL_IN_CAUSAL_DATA — `processing/cleaners.py:91`
**Problem:** bfill() detected in likely causal time-series path.
**Why:** Backward fill moves future-known values into earlier timestamps.
**Fix:** For train/eval causal data, remove bfill; use forward-fill only when availability policy permits, or leave NaN with indicator columns.
**Test:** Add a fixture where first known future value must not appear in earlier rows.
**Confidence:** high  
```python
df_out[data_cols] = (df_out.groupby('ticker')[data_cols].
```

### missing_policy / BFILL_IN_CAUSAL_DATA — `processing/cleaners.py:94`
**Problem:** bfill() detected in likely causal time-series path.
**Why:** Backward fill moves future-known values into earlier timestamps.
**Fix:** For train/eval causal data, remove bfill; use forward-fill only when availability policy permits, or leave NaN with indicator columns.
**Test:** Add a fixture where first known future value must not appear in earlier rows.
**Confidence:** high  
```python
df_out = df_out.bfill().ffill()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `processing/cleaners.py:148`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df[col] = df[col].fillna(0)
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `risk/analyzers/correlation_analyzer.py:29`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns_df[symbol] = market_data[symbol].pct_change().fillna(0)
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `risk/analyzers/correlation_analyzer.py:31`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns_df[symbol] = market_data["close"][symbol].pct_change().fillna(0)
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `risk/elite_risk_metrics.py:386`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns = price_data.pct_change()
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `risk/kill_switch/calculator.py:116`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
symbol_returns = close_prices[symbol].pct_change().dropna()
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `risk/kill_switch/calculator.py:143`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
symbol_returns = symbol_prices.pct_change().dropna()
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `risk/kill_switch/calculator.py:174`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns = close_prices.pct_change().dropna()
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `risk/max_exposure_monitor.py:55`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
self.elite_metrics.update_returns(symbol, market_data[symbol].pct_change().dropna())
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `risk/metrics.py:36`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
symbol_returns = close_df[symbol].pct_change().fillna(0).dropna()
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `risk/metrics.py:110`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
symbol_returns = symbol_prices.pct_change().fillna(0).dropna()
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `risk/metrics.py:153`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns = close_df.pct_change().fillna(0).dropna()
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `training/pattern_aware_training.py:108`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns = frame['close'].astype(float).pct_change().dropna()
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `training/pattern_aware_training.py:110`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns = frame.select_dtypes(include=[np.number]).pct_change().stack().replace([np.inf, -np.inf], np.nan).dropna()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `training/pattern_aware_training.py:138`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `training/pattern_aware_training.py:139`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

### missing_policy / PCT_CHANGE_IMPLICIT_FILL_METHOD — `training/portfolio_optimizer.py:76`
**Problem:** pct_change() called without explicit fill_method.
**Why:** Depending on pandas version/defaults, missing values may be forward-filled before returns are computed.
**Fix:** Use pct_change(fill_method=None), then explicitly handle NaN/inf according to data kind.
**Test:** Add a test where a missing price gap does not become a zero or forward-filled return.
**Confidence:** high  
```python
returns_data[ticker] = df['close'].pct_change().dropna()
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `training/portfolio_optimizer.py:199`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
y = market_features.groupby('ticker')['returns'].first().fillna(0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `utils/feature_preparation.py:95`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
df = df.fillna(0.0)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `utils/missing_data_anomaly_detector.py:144`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
fill_transitions = fill_mask & fill_mask.shift(1).fillna(False)
```

### missing_policy / FILLNA_ZERO_SUSPICIOUS — `utils/smart_missing_data_handler.py:218`
**Problem:** fillna(0) detected.
**Why:** Zero-filling targets, returns, risk metrics, or financial features can fabricate labels/returns and suppress risk.
**Fix:** Replace with explicit per-column policy: drop targets, leave unavailable features as NaN plus availability indicator, or domain-specific fill.
**Test:** Add tests that target tails and missing price gaps are not converted to zeros.
**Confidence:** high  
```python
filled = series.fillna(0.0)
```

### missing_policy / BFILL_IN_CAUSAL_DATA — `utils/smart_missing_data_handler.py:243`
**Problem:** bfill() detected in likely causal time-series path.
**Why:** Backward fill moves future-known values into earlier timestamps.
**Fix:** For train/eval causal data, remove bfill; use forward-fill only when availability policy permits, or leave NaN with indicator columns.
**Test:** Add a fixture where first known future value must not appear in earlier rows.
**Confidence:** high  
```python
filled = series.bfill(limit=5)
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `analytics/analyzers/model_comparison_analyzer.py:30`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
self.HEAVY_MODELS = ["gru", "tabnet", "transformer", "cnn", "lstm", "autoencoder"]
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `colab/models/model_factory.py:123`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
from src.colab.models.architectures import LSTMModel, GRUModel, CNNModel, TransformerModel, AutoencoderModel
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `colab/models/model_factory.py:215`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
return AutoencoderModel(input_size)
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `colab/models/torch_models.py:5`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
from src.colab.models.architectures import LSTMModel, GRUModel, CNNModel, TransformerModel, AutoencoderModel
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `colab/models/torch_models.py:117`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
return AutoencoderModel(input_size)
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `factories/model_factory.py:18`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
from src.models.neural.autoencoder_model import AutoencoderModel
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `models/loader.py:438`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
return AutoencoderModel(input_size)
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `models/neural/autoencoder_model.py:1`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
# src/models/neural/autoencoder_model.py
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `models/neural/autoencoder_model.py:30`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
return "autoencoder_conv1d"
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `pipeline/hybrid/final_stages_executor.py:71`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
heavy_models = [m for m in all_models if m.lower() in ['cnn', 'lstm', 'gru', 'transformer', 'tabnet', 'autoencoder']]
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `pipeline/stages/prediction/model_selection_service.py:173`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
prediction_models = [name for name in models if 'autoencoder' not in name.lower()]
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `pipeline/stages/prediction/prediction_generator.py:62`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
f'⚠️ No models for prediction (only autoencoder), skipping {context_id}'
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `pipeline/stages/prediction/prediction_generator.py:79`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
'⚠️ Autoencoder not suitable for regression prediction')
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `predictions/deep_predict.py:71`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
def predict_autoencoder(model, X):
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `predictions/deep_predict.py:78`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
logger.info(f"[OK] Autoencoder prediction complete ({preds.shape[0]} points).")
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `predictions/models_predict.py:10`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
from .deep_predict import predict_lstm, predict_cnn, predict_transformer, predict_autoencoder
```

### model_routing / AUTOENCODER_ROUTING_REVIEW — `predictions/models_predict.py:46`
**Problem:** Autoencoder appears in model routing/prediction context.
**Why:** Autoencoders should be representation/anomaly models, not primary target predictors unless explicitly designed so.
**Fix:** Use model capability metadata: role, can_predict_target, can_be_primary; never fallback to autoencoder as primary.
**Test:** Test that a model set containing only autoencoder raises NoPrimaryPredictorAvailable.
**Confidence:** medium  
```python
return predict_autoencoder(model, X)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `analytics/arena/arena_battle.py:180`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
model = joblib.load(latest_champ)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `calibration/adaptive_confidence_calibrator.py:290`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
data = joblib.load(path_obj)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `colab/utils/utils.py:66`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
checkpoint = torch.load(checkpoint_path, weights_only=True)
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:193`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
"""Resolve secrets and placeholders in configuration."""
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:199`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
"""Recursively parses configuration for environment markers and path placeholders."""
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:221`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
elif self._has_placeholders(value):
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:222`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
return self._resolve_placeholders(value, secrets)
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:230`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
def _has_placeholders(self, value: Any) -> bool:
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:231`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
"""Check if value contains placeholders."""
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:244`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
def _resolve_placeholders(self, value: str, secrets: dict[str, str]) -> Any:
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:245`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
"""Resolve placeholders in string value."""
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:246`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
placeholders = re.findall(r'\$\{([^}]+)\}', value)
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:249`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
for placeholder in placeholders:
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:250`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
resolved_placeholder = self.get(placeholder, "")
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:251`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
if resolved_placeholder:
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:252`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
resolved_value = resolved_value.replace(f'${{{placeholder}}}', str(resolved_placeholder))
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:421`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
value = self._resolve_placeholders_in_value(value)
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:445`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
def _resolve_placeholders_in_value(self, value: Any) -> Any:
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:446`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
"""Resolve placeholders in string value."""
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:450`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
placeholders = re.findall(r'\$\{([^}]+)\}', value)
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:451`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
for placeholder in placeholders:
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:452`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
resolved_placeholder = self.get(placeholder, "")
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:453`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
if not isinstance(resolved_placeholder, (str, int, float, bool)):
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:454`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
resolved_placeholder = str(resolved_placeholder)
```

### security / PLACEHOLDER_SECRET_REVIEW — `config/unified_config_manager.py:455`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
value = value.replace(f'${{{placeholder}}}', str(resolved_placeholder))
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `core/cache/cache_manager.py:146`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
value = pickle.load(f)
```

### security / PLACEHOLDER_SECRET_REVIEW — `core/security/secure_secrets_manager.py:248`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
# Block placeholder values from development templates
```

### security / PLACEHOLDER_SECRET_REVIEW — `core/security/secure_secrets_manager.py:252`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
f"SECURITY PROTOCOL BREACH: Key '{key_name}' contains a template placeholder."
```

### security / PLACEHOLDER_SECRET_REVIEW — `core/security/secure_secrets_manager.py:254`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
raise SecurityError(f"Secret '{key_name}' contains a placeholder value.")
```

### security / PLACEHOLDER_SECRET_REVIEW — `data/management/data_manager.py:219`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
placeholders = ', '.join(['?' for _ in tickers])
```

### security / PLACEHOLDER_SECRET_REVIEW — `data/management/data_manager.py:223`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
WHERE ticker IN ({placeholders}) AND interval = ?
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `ensembling/stacked_ensemble.py:307`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
state = joblib.load(f)
```

### security / PLACEHOLDER_SECRET_REVIEW — `meta_learning/evolution/dual_loops.py:203`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
placeholders = ', '.join(['?'] * len(status_list))
```

### security / PLACEHOLDER_SECRET_REVIEW — `meta_learning/evolution/dual_loops.py:204`
**Problem:** Placeholder-looking secret/default detected.
**Why:** Placeholders can be mistaken for valid credentials or leak into production config.
**Fix:** Validate secrets at startup and reject known placeholder patterns.
**Test:** Test that placeholder secrets fail validation.
**Confidence:** medium  
```python
query = f"SELECT * FROM rules WHERE status IN ({placeholders})"
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/ensemble/confidence_calibrator.py:117`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
payload = joblib.load(filepath)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/ensemble/enhanced_ensemble.py:115`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
model_obj = torch.load(path_candidate, map_location='cpu')
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/ensemble/enhanced_ensemble.py:142`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
model = joblib.load(model_file)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/ensemble/ensemble_model.py:82`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
loaded_model = joblib.load(path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/linear/knn_model.py:84`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
loaded_model = joblib.load(path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/linear/linear_model.py:74`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
loaded_model = joblib.load(path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/linear/svm_model.py:90`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
loaded_model = joblib.load(path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/loader.py:168`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
return joblib.load(str(consensus_path))
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/loader.py:191`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
return joblib.load(str(path))
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/loader.py:200`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
return joblib.load(str(path))
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/loader.py:205`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
return pickle.load(f)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/loader.py:305`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
loaded_obj = torch.load(path, map_location='cpu', weights_only=
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/neural/tabnet_model.py:122`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
metadata = joblib.load(metadata_path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/tree/lightgbm_model.py:95`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
loaded_model = joblib.load(path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/tree/random_forest_model.py:90`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
loaded_model = joblib.load(path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `models/tree/xgboost_model.py:99`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
loaded_model = joblib.load(path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `monitoring/health_hub.py:94`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
self.models[model_name] = joblib.load(model_path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `monitoring/health_hub.py:100`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
self.scalers['resource_scaler'] = joblib.load(scaler_path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `monitoring/ml_analytics.py:48`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
self.models[model_name] = joblib.load(model_path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `monitoring/ml_analytics.py:53`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
self.scalers['resource_scaler'] = joblib.load(scaler_path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `pipeline/stages/prediction/data_preparer.py:267`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
target_scaler = joblib.load(scaler_path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `pipeline/stages/prediction/scaler_service.py:74`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
target_scaler = joblib.load(scaler_path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `predictions/models_predict.py:99`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
model = joblib.load(os.path.join(models_path, f))
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `processing/normalization_manager.py:172`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
scaler = joblib.load(scaler_path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `training/light_model_trainer.py:170`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
self.models_in_memory[model_key] = joblib.load(path)
```

### security / UNSAFE_MODEL_OR_PICKLE_LOAD — `utils/checkpoint_manager.py:53`
**Problem:** Pickle/joblib/torch model load detected.
**Why:** These formats can execute code or load unsafe artifacts if the path is not trusted and validated.
**Fix:** Only load from validated artifact directories; store metadata/hash and reject untrusted paths.
**Test:** Add tests that traversal/untrusted artifact paths are rejected before load.
**Confidence:** medium  
```python
checkpoint = torch.load(checkpoint_path, weights_only=True)
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `data/synthetic/data_generator.py:135`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** high  
```python
targets['return_1h'] = (close_prices.shift(-1) / close_prices) - 1  # audit-ignore: target label
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `data/synthetic/data_generator.py:136`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** high  
```python
targets['return_4h'] = (close_prices.shift(-4) / close_prices) - 1  # audit-ignore: target label
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `data/synthetic/data_generator.py:137`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** high  
```python
targets['return_24h'] = (close_prices.shift(-24) / close_prices) - 1  # audit-ignore: target label
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `data/synthetic/data_generator.py:149`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** high  
```python
future_1h_returns = close_prices.pct_change(fill_method=None).shift(-1)  # audit-ignore: target label
```

### temporal / NEGATIVE_SHIFT_LOOKAHEAD — `validation/data_leakage_detector.py:126`
**Problem:** Negative shift detected. It may be valid for target generation, but dangerous elsewhere.
**Why:** shift(-h) reads future rows. Without ticker grouping and tail-row dropping, it creates lookahead/cross-ticker leakage.
**Fix:** Use it only in target modules, group by ticker, and drop/mark the last horizon rows instead of filling them.
**Test:** Create a multi-ticker fixture and assert last row of each ticker has NaN target and never uses the next ticker.
**Confidence:** high  
```python
future_target = df[target_col].shift(-1)  # audit-ignore: detector intentionally checks future target correlation
```


## P2

### config_factory / HARDCODED_MODEL_LIST — `analytics/analyzers/model_comparison_analyzer.py:30`
**Problem:** Hardcoded model list detected.
**Why:** Duplicated model lists across factory/arena/pipeline drift over time.
**Fix:** Move models/aliases/capabilities into one registry/config and reference it everywhere.
**Test:** Test that factory, CLI, arena, and prediction stage resolve the same registry entries.
**Confidence:** medium  
```python
self.HEAVY_MODELS = ["gru", "tabnet", "transformer", "cnn", "lstm", "autoencoder"]
```

### config_factory / HARDCODED_MODEL_LIST — `analytics/arena/performance_tracker.py:64`
**Problem:** Hardcoded model list detected.
**Why:** Duplicated model lists across factory/arena/pipeline drift over time.
**Fix:** Move models/aliases/capabilities into one registry/config and reference it everywhere.
**Test:** Test that factory, CLI, arena, and prediction stage resolve the same registry entries.
**Confidence:** medium  
```python
all_models = ['lgbm', 'rf', 'xgboost', 'catboost', 'linear', 'mlp',
```

### config_factory / HARDCODED_MODEL_LIST — `analytics/arena/performance_tracker.py:77`
**Problem:** Hardcoded model list detected.
**Why:** Duplicated model lists across factory/arena/pipeline drift over time.
**Fix:** Move models/aliases/capabilities into one registry/config and reference it everywhere.
**Test:** Test that factory, CLI, arena, and prediction stage resolve the same registry entries.
**Confidence:** medium  
```python
enhanced_models = ['dean_ensemble', 'sentiment', 'lgbm_bayesian']
```

### config_factory / HARDCODED_MODEL_LIST — `analytics/arena/performance_tracker.py:78`
**Problem:** Hardcoded model list detected.
**Why:** Duplicated model lists across factory/arena/pipeline drift over time.
**Fix:** Move models/aliases/capabilities into one registry/config and reference it everywhere.
**Test:** Test that factory, CLI, arena, and prediction stage resolve the same registry entries.
**Confidence:** medium  
```python
heavy_models = ['lstm', 'gru', 'transformer', 'cnn', 'tabnet',
```

### config_factory / HARDCODED_MODEL_LIST — `analytics/data_managers/model_results_manager.py:25`
**Problem:** Hardcoded model list detected.
**Why:** Duplicated model lists across factory/arena/pipeline drift over time.
**Fix:** Move models/aliases/capabilities into one registry/config and reference it everywhere.
**Test:** Test that factory, CLI, arena, and prediction stage resolve the same registry entries.
**Confidence:** medium  
```python
self.LIGHT_MODEL_TYPES = ['lgbm', 'rf', 'linear', 'mlp', 'ensemble']
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `analytics/signals/signal_analytics.py:67`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
model_performance[model_name] = {
```

### config_factory / DUPLICATED_MODEL_REGISTRY_ENTRIES — `factories/model_factory.py:34`
**Problem:** Model/alias registry entries overlap with other files.
**Why:** Duplicated registries drift and cause selector/factory/prediction inconsistencies.
**Fix:** Move all model names, aliases, class paths, role, heavy flag, and can_be_primary to one registry.
**Test:** Snapshot-test that factory, CLI, arena, and prediction load the same registry.
**Confidence:** medium  
```python
_model_aliases = ['linear', 'svm', 'knn', 'mlp', 'cnn', 'lstm', 'gru', 'transformer', 'tabnet', 'autoencoder']...
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `factories/model_factory.py:34`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
_model_aliases: Dict[str, str] = {
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `features/colab_context_integration.py:132`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
max_features_map = {'mlp': 256, 'lstm': 128, 'gru': 128, 'cnn': 64,
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `features/enrichers/news_impact_enricher.py:232`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
significance_map = {'low': 0, 'medium': 1, 'high': 2}
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `features/enrichers/news_impact_enricher.py:393`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
significance_map = {'low': 0, 'medium': 1, 'high': 2}
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `features/feature_selector.py:121`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
max_features_map = {'mlp': 256, 'lstm': 128, 'gru': 128, 'cnn': 64,
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `features/feature_selector.py:146`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
max_features_map = {'mlp': 256, 'lstm': 128, 'gru': 128, 'cnn': 64,
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `features/news_impact_classifier.py:246`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
mapping = {
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `features/nlp/utils/mention_utils.py:9`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
TICKER_ALIASES = {
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `models/analysis/baseline_dominance_detector.py:32`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
self.BASELINE_MODELS = {
```

### config_factory / HARDCODED_MODEL_LIST — `models/ensemble/enhanced_ensemble.py:164`
**Problem:** Hardcoded model list detected.
**Why:** Duplicated model lists across factory/arena/pipeline drift over time.
**Fix:** Move models/aliases/capabilities into one registry/config and reference it everywhere.
**Test:** Test that factory, CLI, arena, and prediction stage resolve the same registry entries.
**Confidence:** medium  
```python
light_model_types = ['catboost', 'lightgbm', 'xgboost',
```

### config_factory / HARDCODED_MODEL_LIST — `models/loader.py:349`
**Problem:** Hardcoded model list detected.
**Why:** Duplicated model lists across factory/arena/pipeline drift over time.
**Fix:** Move models/aliases/capabilities into one registry/config and reference it everywhere.
**Test:** Test that factory, CLI, arena, and prediction stage resolve the same registry entries.
**Confidence:** medium  
```python
light_models = ['catboost', 'lightgbm', 'xgboost', 'random_forest',
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `monitoring/config.py:58`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
env_mappings = {'MONITORING_CPU_THRESHOLD': ('system_health',
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `monitoring/dashboard.py:384`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
color_map = {
```

### config_factory / HARDCODED_MODEL_LIST — `monitoring/example_usage.py:198`
**Problem:** Hardcoded model list detected.
**Why:** Duplicated model lists across factory/arena/pipeline drift over time.
**Fix:** Move models/aliases/capabilities into one registry/config and reference it everywhere.
**Test:** Test that factory, CLI, arena, and prediction stage resolve the same registry entries.
**Confidence:** medium  
```python
model_names = ['price_predictor', 'trend_analyzer', 'risk_model', 'portfolio_optimizer']
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `monitoring/ml_analytics.py:39`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
model_files = {'performance_predictor':
```

### config_factory / HARDCODED_MODEL_LIST — `monitoring/ml_analytics.py:68`
**Problem:** Hardcoded model list detected.
**Why:** Duplicated model lists across factory/arena/pipeline drift over time.
**Fix:** Move models/aliases/capabilities into one registry/config and reference it everywhere.
**Test:** Test that factory, CLI, arena, and prediction stage resolve the same registry entries.
**Confidence:** medium  
```python
problem_models = ['performance', 'memory', 'disk', 'network']
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `pipeline/guards/safe_feature_combiner.py:180`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
prefix_map = {
```

### config_factory / HARDCODED_MODEL_LIST — `pipeline/hybrid/final_stages_executor.py:84`
**Problem:** Hardcoded model list detected.
**Why:** Duplicated model lists across factory/arena/pipeline drift over time.
**Fix:** Move models/aliases/capabilities into one registry/config and reference it everywhere.
**Test:** Test that factory, CLI, arena, and prediction stage resolve the same registry entries.
**Confidence:** medium  
```python
heavy_models = ['cnn', 'lstm', 'gru', 'transformer', 'tabnet',
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `pipeline/stages/prediction/data_preparation_service.py:221`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
regime_map = {'bull': 1, 'bear': -1, 'sideways': 0, 'volatile': 2}
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `pipeline/stages/prediction/model_selection_service.py:197`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
known_aliases = {
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `pipeline/stages/prediction/model_selection_service.py:226`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
regime_map = {'bull': 1, 'bear': -1, 'sideways': 0, 'volatile': 2}
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `pipeline/stages/prediction/prediction_context_manager.py:57`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
regime_map = {'bull': 1, 'bear': -1, 'sideways': 0, 'volatile': 2}
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `predictions/models_predict.py:133`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
signal_map = {'buy': 0.015, 'sell': -0.015, 'hold': 0.0, 'strong_buy':
```

### config_factory / HARDCODED_MODEL_MAP_OR_ALIASES — `sentiment/sentiment_models.py:86`
**Problem:** Hardcoded model map/alias registry detected.
**Why:** Multiple registries make routing inconsistent; e.g. training knows a model but prediction/arena does not.
**Fix:** Use a single model registry with class_path, aliases, role, heavy flag, and can_be_primary.
**Test:** Snapshot-test that all model names and aliases resolve from one source of truth.
**Confidence:** medium  
```python
label_map = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
```

### determinism / NON_INJECTED_CLOCK — `main/modes/monster_test.py:64`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp=datetime.now(),
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `algorithms/adaptive_position_sizer.py:216`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `algorithms/advanced_backtest_engine.py:115`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `algorithms/regime/clustering.py:40`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `algorithms/regime/rules.py:41`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `algorithms/regime_detector.py:77`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/analyzers/adaptive_confidence_analyzer.py:46`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/analyzers/hedge_fund_analyzer.py:72`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/analyzers/knn_similarity_finder.py:71`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/analyzers/model_comparison_analyzer.py:108`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/analyzers/news_impact_analyzer.py:98`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/analyzers/shap_analyzer.py:37`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/arena/arena_battle.py:101`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (AttributeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/arena/arena_battle.py:295`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/arena/arena_battle.py:315`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/arena/arena_battle.py:407`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/arena/ensemble_performance_bridge.py:114`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/context/contextual_model_selector.py:79`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/context/counterfactual_generator.py:91`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/context/counterfactual_generator.py:153`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/context/counterfactual_generator.py:205`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/context/counterfactual_generator.py:257`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/context/counterfactual_generator.py:712`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as exc:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/context/market_context_analyzer.py:55`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/context/prediction_adjuster.py:92`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/data_managers/model_results_manager.py:67`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/data_managers/model_results_manager.py:119`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/data_managers/model_results_manager.py:152`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/data_managers/model_results_manager.py:208`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/engines/causal_engine.py:47`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/engines/causal_engine.py:85`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/engines/causal_engine.py:106`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/engines/causal_engine.py:115`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `analytics/reporting/results_manager.py:56`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `backtesting/advanced/advanced_engine.py:180`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `calibration/adaptive_confidence_calibrator.py:128`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `calibration/adaptive_confidence_calibrator.py:216`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `calibration/adaptive_confidence_calibrator.py:231`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `calibration/adaptive_confidence_calibrator.py:262`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `calibration/adaptive_confidence_calibrator.py:331`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `cli/pipeline_data_loader.py:88`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as ex:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `cli/pipeline_executor.py:223`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as ex:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `colab/config/config_loader.py:122`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `config/unified_config_manager.py:150`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `core/base_integration.py:36`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `core/cache/cache_manager.py:152`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `core/cache/cache_manager.py:294`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `core/error_handling/error_handler.py:89`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `core/error_handling/error_handler.py:254`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `core/file_management/file_manager.py:53`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `core/file_management/file_manager.py:235`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `core/logging/exception_decorator.py:17`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `core/security/secure_secrets_manager.py:178`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `core/security/secure_secrets_manager.py:211`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/collectors/bigquery_collector.py:62`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/collectors/collector_factory.py:31`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/collectors/free_google_trends_collector.py:68`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/collectors/free_google_trends_collector.py:101`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/collectors/insider_collector.py:90`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/collectors/insider_collector.py:110`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/collectors/newsapi_collector.py:193`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/collectors/yf_collector.py:226`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/connectors/bigquery_connector.py:30`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/connectors/bigquery_connector.py:100`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:130`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:146`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:163`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:171`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:180`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:194`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:202`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:236`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:262`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:283`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:325`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:356`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as insert_error:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:384`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as idx_e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:395`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:425`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:433`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_manager.py:443`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (duckdb.Error, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/data_versioning.py:215`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/handlers/connection_handler.py:27`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/management/handlers/connection_handler.py:44`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `data/validation/event_dataset_validator.py:63`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `devtools/rule_generator.py:151`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `factories/model_factory.py:198`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `factories/tree_model_factory.py:37`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/analysis/regime_importance_tracker.py:149`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/analysis/regime_importance_tracker.py:175`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/builders/news_event_dataset_builder.py:78`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/colab_context_integration.py:63`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/colab_context_integration.py:160`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/colab_context_integration.py:197`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/enrichers/advanced_analytics_enricher.py:36`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/enrichers/advanced_analytics_enricher.py:52`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/enrichers/base.py:89`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/enrichers/context_map_enricher.py:61`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/enrichers/keyword_entity_enricher.py:33`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/enrichers/macro_features_enricher.py:79`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/enrichers/technical_analysis_enricher.py:255`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/enrichers/technical_analysis_enricher.py:271`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/enrichers/technical_analysis_enricher.py:288`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/enrichers/technical_analysis_enricher.py:304`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/feature_cache.py:99`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (IOError, ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/feature_cache.py:146`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (IOError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/feature_cache.py:208`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/feature_cache.py:271`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/feature_orchestrator.py:229`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/feature_selector.py:101`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/news_dataset_builder.py:108`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/news_dataset_builder.py:135`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/news_dataset_builder.py:193`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/nlp/models/finbert_pipeline.py:51`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/validation/feature_leakage_guard.py:172`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `features/validation/feature_leakage_guard.py:190`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `integrations/base.py:41`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `main/modes/backtest.py:96`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `main/modes/web_ui.py:121`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `main/modes/web_ui.py:203`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `main/modes/web_ui.py:227`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `main/system_orchestrator.py:224`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `meta_learning/security/constraint_engine.py:439`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `meta_learning/security/constraint_engine.py:465`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `meta_learning/security/constraint_engine.py:499`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `meta_learning/security/constraint_engine.py:520`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `meta_learning/security/constraint_engine.py:534`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `meta_learning/security/constraint_engine.py:550`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `meta_learning/security/constraint_engine.py:570`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `metrics/model/ml_evaluator.py:87`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/actions/action_trigger.py:62`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/actions/action_trigger.py:92`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/adapters/data_preparation.py:124`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/adapters/sentiment_integration.py:27`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/adapters/sentiment_integration.py:47`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/adapters/sentiment_integration.py:80`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/adapters/sentiment_integration.py:198`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/adapters/sentiment_integration.py:251`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/adapters/unified_model_adapter.py:135`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/baseline/comparison.py:49`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/baseline/recommendations.py:39`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/baseline/recommendations.py:70`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/baseline_dominance_detector.py:105`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/baselines/models.py:41`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/baselines/models.py:75`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/baselines/strategies.py:33`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/baselines/strategies.py:73`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/baselines/strategies.py:104`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/model_analyzer.py:76`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/model_analyzer.py:103`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/model_analyzer.py:121`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/model_analyzer.py:137`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/model_analyzer.py:157`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/overfitting_detection/analyzer.py:42`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/overfitting_detection/analyzer.py:61`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/overfitting_detection/analyzer.py:87`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/overfitting_detection/manager.py:71`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/overfitting_detection/manager.py:95`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/regime/detector.py:26`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/regime/detector.py:42`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/regime/detector.py:57`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/regime/metrics.py:31`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/regime/patterns.py:37`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/regime/patterns.py:68`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/regime/patterns.py:109`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/regime/stability.py:28`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/analysis/regime/stability.py:47`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/ensemble/confidence_calibrator.py:59`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/ensemble/confidence_calibrator.py:71`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/ensemble/correlation/correlation_engine.py:131`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, KeyError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/ensemble/correlation/correlation_engine.py:193`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/ensemble/enhanced_ensemble.py:119`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/ensemble/ensemble_model.py:45`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/linear/knn_model.py:47`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/linear/linear_model.py:37`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/linear/svm_model.py:53`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/loader.py:74`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/loader.py:169`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/loader.py:183`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, ImportError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/model_selector/adaptive_selector.py:209`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/model_selector/adaptive_selector.py:318`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/model_selector/smart_selector.py:42`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/monitoring/drift/analyzer.py:42`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/monitoring/drift/analyzer.py:66`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/monitoring/drift/analyzer.py:86`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/monitoring/drift/drift_calculator.py:76`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/monitoring/drift/drift_calculator.py:131`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/monitoring/drift/drift_calculator.py:170`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/monitoring/drift/history.py:39`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/monitoring/prediction_drift_monitor.py:112`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/neural/base_neural.py:106`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/neural/cnn_model.py:105`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/neural/cnn_model.py:134`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/neural/transformer_model.py:115`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/neural/transformer_model.py:122`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e2:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/neural/transformer_model.py:239`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/neural/transformer_model.py:244`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e2:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/registry/model_registry.py:62`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/registry/model_registry.py:138`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/registry/model_registry.py:159`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/tree/catboost_model.py:76`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/tree/catboost_model.py:88`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/tree/catboost_model.py:103`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/tree/lightgbm_model.py:58`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/tree/random_forest_model.py:53`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `models/tree/xgboost_model.py:62`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `monitoring/config.py:36`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `monitoring/feature_drift_monitor.py:175`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `monitoring/ml_analytics.py:221`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `optimization/hyperparameter_searcher.py:201`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `optimization/hyperparameters/bayesian.py:136`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `optimization/portfolio/optimizer.py:62`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `optimization/portfolio/optimizer.py:198`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `optimization/portfolio/optimizer.py:241`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `patterns/pattern_analyzer.py:250`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/component_factory.py:101`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/data_batch_manager.py:59`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/feature_loader.py:34`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/feature_loader.py:49`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/final_stages_executor.py:182`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/light_models_trainer.py:104`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/light_models_trainer.py:144`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/pipeline_metadata_manager.py:75`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/results_processor.py:67`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/results_processor.py:87`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/hybrid/test_mode_manager.py:43`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/pipeline_orchestrator.py:257`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/pipeline_orchestrator.py:284`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/evaluation/analytics.py:19`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/evaluation/analytics.py:81`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/evaluation/backtest_adapter.py:70`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/evaluation/data_recovery.py:26`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/evaluation/io.py:25`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/evaluation/reporting.py:38`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/modeling/orchestration.py:73`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/prediction/anomaly_engine.py:169`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/prediction/data_preparer.py:78`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/prediction/data_preparer.py:258`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/prediction/model_selection_service.py:80`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/prediction/result_builder.py:59`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as fe:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/prediction/result_builder.py:82`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/prediction/result_builder.py:204`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/stage_0_setup.py:76`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/trading/data_io.py:103`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `pipeline/stages/trading/recommendation_engine.py:146`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `predictions/models_predict.py:49`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `predictions/models_predict.py:126`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `processing/cleaners.py:166`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `processing/cleaners.py:181`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `risk/kill_switch/calculator.py:51`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `risk/kill_switch/calculator.py:91`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `risk/kill_switch/calculator.py:119`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `risk/kill_switch/calculator.py:159`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `risk/kill_switch/calculator.py:196`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `targets/target_orchestrator.py:55`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `trading/consensus_engine.py:219`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `trading/consensus_engine.py:294`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `trading/signal_processor.py:71`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `trading/trading_orchestrator.py:148`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `trading/trading_orchestrator.py:171`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `trading/virtual_portfolio.py:67`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `trading/virtual_portfolio.py:102`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `training/base_trainer.py:132`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `training/base_trainer.py:196`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `training/base_trainer.py:268`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (ValueError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `training/base_trainer.py:351`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except (IOError, TypeError, Exception) as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `training/progressive_trainer.py:198`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### error_policy / LOGGER_ERROR_THEN_RAISE — `validation/time_series_validator.py:225`
**Problem:** Exception is logged and re-raised in the same handler.
**Why:** If upper layers also log, this creates duplicate error reports and noisy traces.
**Fix:** Log only at boundary layers, or add context and re-raise without error-level logging.
**Test:** Add a test/logger capture for one error event per failing operation.
**Confidence:** medium  
```python
except Exception as e:
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `algorithms/advanced_backtest_engine.py:79`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
return float(drawdown.min())
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `algorithms/advanced_backtest_engine.py:162`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
max_dd = drawdown.min()
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `algorithms/walk_forward_optimizer.py:137`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
max_drawdown = float(((cumulative - running_max) / running_max).min())
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `analytics/analyzers/hedge_fund_analyzer.py:92`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
metrics['max_drawdown'] = float(drawdown_series.min()
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `analytics/analyzers/wrappers.py:32`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
"max_drawdown": float(clean_drawdown.min()) if not clean_drawdown.empty else 0.0,
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `analytics/arena/arena_battle.py:406`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
return float(np.min(drawdown))
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `backtesting/advanced/advanced_engine.py:211`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
max_dd = drawdown.min()
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `backtesting/advanced/advanced_engine.py:377`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
return float(drawdown.min())
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `features/enrichers/technical_analysis_enricher.py:251`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
df_enriched['CURRENT_DRAWDOWN'
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `meta_learning/real_time_learning.py:269`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
return float(drawdown.min())
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `metrics/financial/financial_metrics_library.py:86`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
return float(drawdowns.min()) if not drawdowns.empty else 0.0
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `metrics/financial/portfolio_metrics.py:100`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
max_drawdown = drawdowns.min()
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `pipeline/stages/evaluation/metrics_calculator.py:91`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
max_drawdown = drawdown.min()
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/elite_risk_metrics.py:429`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
Dict[str, Any]], daily_pnl: float, current_drawdown: float) ->Dict[
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/elite_risk_metrics.py:438`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
current_drawdown: Current drawdown
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/elite_risk_metrics.py:466`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
if current_drawdown > self.limits['max_drawdown']:
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/elite_risk_metrics.py:468`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
current_drawdown, 'limit': self.limits['max_drawdown'],
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/elite_risk_metrics.py:469`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
'message': f'Drawdown {current_drawdown:.1%} exceeds limit'})
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/kill_switch/calculator.py:80`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
current_drawdown = drawdowns.iloc[-1] if hasattr(drawdowns, 'iloc') else (drawdowns[-1] if len(drawdowns) > 0 else 0)
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/kill_switch/calculator.py:88`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
'current_drawdown': current_drawdown,
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/kill_switch/calculator.py:150`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
current_drawdown = drawdowns.iloc[-1] if len(drawdowns) > 0 else 0
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/kill_switch/calculator.py:156`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
'current_drawdown': current_drawdown
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/kill_switch/calculator.py:219`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
if portfolio_metrics.get('current_drawdown', 0) > thresholds.get('max_drawdown_threshold', 1.0):
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/kill_switch/calculator.py:236`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
if portfolio_metrics.get('current_drawdown', 0) > 0.10: # 10% drawdown
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/metrics.py:65`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/metrics.py:67`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
current_drawdown = 0.0
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/metrics.py:71`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
current_drawdown = (peak - current) / peak if peak > 0 else 0.0
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/metrics.py:79`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
"current_drawdown": current_drawdown,
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/metrics.py:117`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/metrics.py:119`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
current_drawdown = 0.0
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/metrics.py:123`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
current_drawdown = (peak - current) / peak if peak > 0 else 0.0
```

### financial_math / DRAWDOWN_SIGN_CONVENTION_REVIEW — `risk/metrics.py:129`
**Problem:** Drawdown calculation found; sign convention needs review.
**Why:** Mixing signed max_drawdown (-0.25) and positive current_drawdown (0.25) breaks risk thresholds.
**Fix:** Expose both max_drawdown_signed and max_drawdown_pct, and use pct in risk limits.
**Test:** Test monotonic loss series and assert documented sign convention.
**Confidence:** low  
```python
"current_drawdown": current_drawdown,
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `analytics/calculators/fama_french_factors.py:17`
**Problem:** Top-level import of heavy optional dependency 'yfinance'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import yfinance as yf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `colab/models/architectures.py:1`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `colab/models/architectures.py:2`
**Problem:** Top-level import of heavy optional dependency 'torch.nn'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch.nn as nn
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `colab/models/torch_models.py:3`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `colab/models/torch_models.py:4`
**Problem:** Top-level import of heavy optional dependency 'torch.nn'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch.nn as nn
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `colab/utils/utils.py:14`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `data/collectors/vix_collector.py:22`
**Problem:** Top-level import of heavy optional dependency 'yfinance'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import yfinance as yf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `data/collectors/vix_collector.py:100`
**Problem:** Top-level import of heavy optional dependency 'yfinance'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import yfinance as yf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `data/collectors/yf_collector.py:9`
**Problem:** Top-level import of heavy optional dependency 'yfinance'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import yfinance as yf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `features/nlp/extractors/entity_extractor.py:4`
**Problem:** Top-level import of heavy optional dependency 'spacy'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import spacy
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `features/nlp/models/roberta_sentiment.py:2`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `features/nlp/models/roberta_sentiment.py:3`
**Problem:** Top-level import of heavy optional dependency 'transformers'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `features/nlp/models/sentiment_core.py:6`
**Problem:** Top-level import of heavy optional dependency 'transformers'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from transformers import AutoModelForSequenceClassification
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `features/nlp/scoring/summarizer.py:5`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `features/nlp/scoring/summarizer.py:6`
**Problem:** Top-level import of heavy optional dependency 'transformers'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from transformers import pipeline, Pipeline
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/ensemble/enhanced_ensemble.py:9`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/loader.py:215`
**Problem:** Top-level import of heavy optional dependency 'tensorflow.keras.models'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from tensorflow.keras.models import load_model
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/loader.py:240`
**Problem:** Top-level import of heavy optional dependency 'tensorflow.keras.models'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from tensorflow.keras.models import load_model
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/loader.py:246`
**Problem:** Top-level import of heavy optional dependency 'tensorflow'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import tensorflow as tf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/loader.py:252`
**Problem:** Top-level import of heavy optional dependency 'tensorflow.keras.models'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from tensorflow.keras.models import load_model
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/loader.py:257`
**Problem:** Top-level import of heavy optional dependency 'tensorflow'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import tensorflow as tf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/loader.py:304`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/loader.py:347`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/loader.py:348`
**Problem:** Top-level import of heavy optional dependency 'torch.nn'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch.nn as nn
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/loader.py:445`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/autoencoder_model.py:4`
**Problem:** Top-level import of heavy optional dependency 'tensorflow'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import tensorflow as tf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/autoencoder_model.py:5`
**Problem:** Top-level import of heavy optional dependency 'tensorflow.keras'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from tensorflow.keras import layers, models
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/base_neural.py:7`
**Problem:** Top-level import of heavy optional dependency 'tensorflow'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import tensorflow as tf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/cnn_model.py:5`
**Problem:** Top-level import of heavy optional dependency 'tensorflow'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import tensorflow as tf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/cnn_model.py:6`
**Problem:** Top-level import of heavy optional dependency 'tensorflow.keras'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from tensorflow.keras import Sequential
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/cnn_model.py:7`
**Problem:** Top-level import of heavy optional dependency 'tensorflow.keras.layers'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/gru_model.py:4`
**Problem:** Top-level import of heavy optional dependency 'tensorflow'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import tensorflow as tf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/gru_model.py:5`
**Problem:** Top-level import of heavy optional dependency 'tensorflow.keras'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from tensorflow.keras import layers, models
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/lstm_model.py:4`
**Problem:** Top-level import of heavy optional dependency 'tensorflow'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import tensorflow as tf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/lstm_model.py:5`
**Problem:** Top-level import of heavy optional dependency 'tensorflow.keras'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from tensorflow.keras import layers, models
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/mlp_model.py:4`
**Problem:** Top-level import of heavy optional dependency 'tensorflow'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import tensorflow as tf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/mlp_model.py:5`
**Problem:** Top-level import of heavy optional dependency 'tensorflow.keras'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from tensorflow.keras import layers, models
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/transformer_model.py:128`
**Problem:** Top-level import of heavy optional dependency 'tensorflow.keras.optimizers'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from tensorflow.keras.optimizers import Adam
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `models/neural/transformer_model.py:168`
**Problem:** Top-level import of heavy optional dependency 'tensorflow'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import tensorflow as tf
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `predictions/deep_predict.py:4`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `sentiment/sentiment_models.py:26`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `sentiment/sentiment_models.py:27`
**Problem:** Top-level import of heavy optional dependency 'transformers'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `utils/checkpoint_manager.py:50`
**Problem:** Top-level import of heavy optional dependency 'torch'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import torch
```

### heavy_imports / HEAVY_TOP_LEVEL_IMPORT — `utils/trading_calendar.py:5`
**Problem:** Top-level import of heavy optional dependency 'yfinance'.
**Why:** Lightweight CLI/tests/factories may import TensorFlow/PyTorch/HF/spaCy/yfinance even when not needed.
**Fix:** Move optional heavy imports inside the function/class that needs them, or use a lazy class-path registry.
**Test:** Add a test importing factory/config/CLI and assert heavy modules are not present in sys.modules.
**Confidence:** high  
```python
import yfinance as yf
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `cli/pipeline_data_loader.py:19`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
df = pd.read_parquet(path)
```

### security / ENV_LOADING_REVIEW — `colab/__init__.py:16`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
from .environment import ColabEnvironment
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `colab/config/config_loader.py:136`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(config_path, 'r') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `config/tickers.py:382`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(filepath, 'w') as f:
```

### security / ENV_LOADING_REVIEW — `config/unified_config_manager.py:90`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
self.env = env
```

### security / ENV_LOADING_REVIEW — `config/unified_config_manager.py:118`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
logger.info(f"UnifiedConfigManager initialized for '{self.env.value}' environment.")
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:67`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(p, 'w', encoding='utf-8') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:72`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(p, encoding='utf-8') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:91`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(path, encoding='utf-8') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:107`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(p, 'w', encoding='utf-8') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:112`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(p, encoding='utf-8') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:131`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(path, encoding='utf-8') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:177`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
pd.read_parquet(p, columns=[df_to_save.columns[0]])
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:179`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
pd.read_csv(p, nrows=1)
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:181`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
pd.read_json(p, nrows=1)
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:203`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
df = pd.read_parquet(path, **kwargs)
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:205`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
df = pd.read_csv(path, **kwargs)
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/file_management/file_manager.py:207`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
df = pd.read_json(path, **kwargs)
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:43`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
configured_paths = config.get('security.env_search_paths', [])
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:49`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
def load_dotenv(dotenv_path: str = '.env'):
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:51`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
Manually parses a .env file and injects keys into os.environ.
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:55`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
1. Specified parameter path (default: .env)
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:67`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
# Hierarchical list of potential .env locations
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:78`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
'/content/drive/MyDrive/trading_project/.env',
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:79`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
'/content/drive/MyDrive/.env',
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:80`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
'/content/.env',
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:81`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
'../.env',
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:82`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
Path.home() / '.env',
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:94`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
f"No .env configuration file found across search vectors: {search_paths}. Utilizing existing environment variables."
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/security/secure_secrets_manager.py:100`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(found_path, encoding="utf-8") as f:
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:112`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
os.environ[key] = value
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:133`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
def __init__(self, dotenv_path: str = ".env", encrypted_path: str = ".env.enc"):
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:137`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
self.dotenv_keys = load_dotenv(dotenv_path)
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/security/secure_secrets_manager.py:162`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(path, 'rb') as f:
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:173`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
os.environ[key] = value
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:182`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
def encrypt_secrets(self, secrets: dict[str, str], output_path: str = ".env.enc"):
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `core/security/secure_secrets_manager.py:206`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(output_path, 'wb') as f:
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:236`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
Hierarchy: os.environ -> Local Cache.
```

### security / ENV_LOADING_REVIEW — `core/security/secure_secrets_manager.py:270`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
for key, value in os.environ.items():
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `data/collectors/custom_csv_collector.py:55`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(file_path, mode='r', encoding=encoding) as infile:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `data/data_loader.py:40`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
self.features_df = pd.read_parquet(features_file)
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `data/data_loader.py:41`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
self.targets_df = pd.read_parquet(targets_file)
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `data/data_loader.py:60`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(cache_file, 'r') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `data/data_loader.py:74`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(cache_file, 'w') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `data_sources/local_file_data_source.py:35`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
df = pd.read_csv(file_path)
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `data_sources/local_file_data_source.py:37`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
df = pd.read_parquet(file_path)
```

### security / ENV_LOADING_REVIEW — `integrations/data/bigquery_client.py:37`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
self.use_simulator = os.environ.get('BIGQUERY_SIMULATOR_MODE', 'false'
```

### security / ENV_LOADING_REVIEW — `integrations/data/bigquery_client.py:205`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
gcp_project_id = os.environ.get('GCP_PROJECT_ID')
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `meta_learning/security/agent_permissions.py:526`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(audit_file, 'w') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `models/loader.py:204`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(path, 'rb') as f:
```

### security / ENV_LOADING_REVIEW — `models/neural/base_neural.py:34`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
os.environ['PYTHONHASHSEED'] = str(self.random_state)
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `monitoring/config.py:31`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(self.config_file, 'r', encoding='utf-8') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `monitoring/config.py:50`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(self.config_file, 'w', encoding='utf-8') as f:
```

### security / ENV_LOADING_REVIEW — `monitoring/config.py:175`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
create_config_file(args.create_config, args.environment)
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `optimization/dynamic_config_updater.py:166`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(filepath, 'w') as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `pipeline/hybrid/feature_loader.py:30`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
async with aiofiles.open(candidate, encoding="utf-8") as f:
```

### security / FILE_READ_NEEDS_PATH_VALIDATION — `pipeline/hybrid/feature_loader.py:46`
**Problem:** File read detected in config/data loading path.
**Why:** User/config-controlled paths need resolve()+allowed-base validation to prevent traversal or wrong-file reads.
**Fix:** Route all config paths through a single PathSecurityValidator before reading.
**Test:** Test that '../secret.env' and absolute paths outside allowed base are rejected.
**Confidence:** medium  
```python
with open(candidate, encoding="utf-8") as f:
```

### security / ENV_LOADING_REVIEW — `sentiment/sentiment_models.py:33`
**Problem:** .env loading/search path detected.
**Why:** Loose .env search paths can load the wrong secrets; file values may override real environment unexpectedly.
**Fix:** Make search paths explicit per environment and keep os.environ priority unless override=True.
**Test:** Test that parent/home .env is not loaded in production/test mode.
**Confidence:** medium  
```python
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')  # 5 minutes
```


## P3

### architecture / GOD_CLASS_REVIEW — `algorithms/risk_parity_allocator.py:32`
**Problem:** Class 'RiskParityAllocator' has 35 methods.
**Why:** God classes hide responsibilities and make fatal/non-fatal error policy hard to enforce.
**Fix:** Before splitting, add characterization tests; then extract cohesive services by responsibility.
**Test:** Test public behavior of the class before extraction.
**Confidence:** high  
```python
class RiskParityAllocator:
```

### architecture / GOD_CLASS_REVIEW — `analytics/context/market_context_analyzer.py:10`
**Problem:** Class 'MarketContextAnalyzer' has 40 methods.
**Why:** God classes hide responsibilities and make fatal/non-fatal error policy hard to enforce.
**Fix:** Before splitting, add characterization tests; then extract cohesive services by responsibility.
**Test:** Test public behavior of the class before extraction.
**Confidence:** high  
```python
class MarketContextAnalyzer(IAnalyzer):
```

### architecture / GOD_CLASS_REVIEW — `config/unified_config_manager.py:76`
**Problem:** Class 'UnifiedConfigManager' has 37 methods.
**Why:** God classes hide responsibilities and make fatal/non-fatal error policy hard to enforce.
**Fix:** Before splitting, add characterization tests; then extract cohesive services by responsibility.
**Test:** Test public behavior of the class before extraction.
**Confidence:** high  
```python
class UnifiedConfigManager:
```

### architecture / GOD_CLASS_REVIEW — `data/management/data_manager.py:53`
**Problem:** Class 'DataManager' has 30 methods.
**Why:** God classes hide responsibilities and make fatal/non-fatal error policy hard to enforce.
**Fix:** Before splitting, add characterization tests; then extract cohesive services by responsibility.
**Test:** Test public behavior of the class before extraction.
**Confidence:** high  
```python
class DataManager(IDatabaseManager):
```

### architecture / GOD_CLASS_REVIEW — `optimization/portfolio/optimizer.py:29`
**Problem:** Class 'PortfolioOptimizer' has 27 methods.
**Why:** God classes hide responsibilities and make fatal/non-fatal error policy hard to enforce.
**Fix:** Before splitting, add characterization tests; then extract cohesive services by responsibility.
**Test:** Test public behavior of the class before extraction.
**Confidence:** high  
```python
class PortfolioOptimizer(BaseOptimizer):
```

### architecture / GOD_CLASS_REVIEW — `pipeline/hybrid/orchestrator_context.py:6`
**Problem:** Class 'OrchestratorContext' has 28 methods.
**Why:** God classes hide responsibilities and make fatal/non-fatal error policy hard to enforce.
**Fix:** Before splitting, add characterization tests; then extract cohesive services by responsibility.
**Test:** Test public behavior of the class before extraction.
**Confidence:** high  
```python
class OrchestratorContext:
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/advanced_analytics_enricher.py:84`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/advanced_analytics_enricher.py:141`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched['macro_composite_score'] = scores_df[
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/base.py:76`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
self.logger.debug(f"✅ {self.__class__.__name__} completed: {result.shape[1] - df.shape[1]} features added")
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/decay_features_enricher.py:63`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
enriched_df = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/decay_features_enricher.py:113`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
enriched_df[f"{col}_decayed"] = decayed_values
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/derived_features_enricher.py:61`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/derived_features_enricher.py:73`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
logger.info(f"Derived features enrichment complete. Added {len(df_enriched.columns) - len(df.columns)} features.")
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/hype_enricher.py:98`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping hype enrichment.")
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/hype_enricher.py:103`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/keyword_entity_enricher.py:115`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
f'No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping keyword/entity enrichment.'
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/keyword_entity_enricher.py:180`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/macro_features_enricher.py:280`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
f'Macro features successfully added. Final shape: {df.shape}')
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/market_context_enricher.py:105`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
result_df[col_name] = feature_value
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/market_context_enricher.py:131`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
result_df[col_name] = feature_value
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/news_impact_enricher.py:59`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
logger.debug(f"📊 NewsImpactEnricher.enrich() called. DataFrame shape: {df.shape}")
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/news_impact_enricher.py:96`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/news_impact_enricher.py:208`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/news_impact_enricher.py:247`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/news_impact_enricher.py:301`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping news impact enrichment.")
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/news_impact_enricher.py:358`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/news_quality_enricher.py:76`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
logger.error(f"No time column found in news data. Available columns: {news_df.columns.tolist()[:10]}. Skipping news quality enrichment.")
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/news_quality_enricher.py:131`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/nlp_features_enricher.py:178`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/sentiment_features_enricher.py:76`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
logger.info(f"Sentiment enrichment complete. Added {len(final_df.columns) - len(df.columns)} features.")
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/sentiment_features_enricher.py:287`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/technical_analysis_enricher.py:72`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/time_features_enricher.py:49`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/volatility_enricher.py:34`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/enrichers/volume_enricher.py:34`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/feature_orchestrator.py:182`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
f'🔄 Starting enrichment: {df.shape[0]} rows, {df.shape[1]} columns'
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/feature_orchestrator.py:184`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
df_enriched = df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/feature_orchestrator.py:265`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
feature_cols = [col for col in df.columns if col not in exclude_cols]
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/feature_selector.py:66`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
return features_df.xs(ticker, level='ticker')
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/nlp/deduplication_service.py:22`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
max_features (int): The maximum number of features to use for TF-IDF.
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/enhanced_smart_selector.py:77`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
if 'context_pattern_id' in features_df.columns:
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/enhanced_smart_selector.py:78`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
current_pattern = features_df['context_pattern_id'].iloc[-1]
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/enhanced_smart_selector.py:85`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
'original_feature_count': len(features_df.columns),
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/enhanced_smart_selector.py:101`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
clean_features = features_df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/smart_selector.py:89`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
if set(cached_data.get("input_features", [])) == set(features_df.columns):
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/smart_selector.py:145`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
"input_features": features_df.columns.tolist(),
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/smart_selector.py:159`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
features_clean = features_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/smart_selector.py:195`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
return features_df.apply(lambda x: x.corr(target_series, method=self.correlation_method)).abs().sort_values(ascending=False)
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/smart_selector.py:201`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
return pd.Series(mi, index=features_df.columns).sort_values(ascending=False)
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/smart_selector.py:218`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
index=features_df.columns).sort_values(ascending=False)
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/smart_selector.py:263`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
return pd.Series(model.feature_importances_, index=features_df.columns).sort_values(ascending=False)
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/selection/smart_selector.py:271`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
variances = features_df.var()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/utils/datetime_utils.py:161`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
result = features_df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/feature_leakage_guard.py:94`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
feature_cols = [c for c in df.columns if c not in meta_cols]
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/feature_leakage_guard.py:145`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
numeric_features = [c for c in feature_cols if c in df.columns and
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/feature_leakage_guard.py:151`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
sample_df = df[numeric_features + numeric_targets].dropna()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/feature_leakage_guard.py:157`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
corr_matrix = sample_df[numeric_features].corrwith(sample_df[
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:75`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
self.logger.info(f"🔍 Analyzing {len(features_df.columns)} features for redundancy")
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:78`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
'original_features': list(features_df.columns),
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:79`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
'original_count': len(features_df.columns),
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:92`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
numeric_features = features_df.select_dtypes(include=[np.number])
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:93`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
non_numeric_features = features_df.select_dtypes(exclude=[np.number])
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:143`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
'reduction_ratio': (len(redundant_features) / len(features_df.columns)) * 100,
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:160`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
'remaining_features': features_df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:165`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
for feature_name in features_df.columns:
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:166`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
feature_variance = features_df[feature_name].var()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:191`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
correlation_matrix = features_df.corr().abs()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:210`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
for feature_name, cluster_id in zip(features_df.columns, cluster_labels):
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:259`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
X = features_df.copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:311`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
'selected_features': features_df.copy(),
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:317`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
features_to_keep = set(features_df.columns)
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:326`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
features_df[group_features], group_features
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `features/validation/redundancy_detector.py:370`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
final_features = features_df[list(features_to_keep)].copy()
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `targets/calculators/indicator_prediction_calculator.py:23`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
if indicator_col not in df.columns:
```

### data_lineage / FEATURE_WITHOUT_LOCAL_LINEAGE_HINT — `targets/calculators/indicator_prediction_calculator.py:27`
**Problem:** Feature/enricher code updates data without nearby lineage/availability metadata.
**Why:** Trading features need source, ticker/timeframe granularity, calculation window, and availability time.
**Fix:** Add feature manifest entries or emit lineage metadata from each enricher.
**Test:** Test that every emitted feature has source, window, granularity, availability_time, causal flag.
**Confidence:** low  
```python
target_series = df[indicator_col].shift(shift)
```

### determinism / NON_INJECTED_CLOCK — `analytics/analyzers/hedge_fund_analyzer.py:71`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `analytics/analyzers/hedge_fund_analyzer.py:119`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
min_idx = pd.Timestamp.now() - pd.Timedelta(days=365)
```

### determinism / NON_INJECTED_CLOCK — `analytics/analyzers/hedge_fund_analyzer.py:124`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
max_idx = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `analytics/analyzers/performance_attribution_analyzer.py:82`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'analysis_timestamp': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `analytics/analyzers/performance_attribution_analyzer.py:262`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
f'{datetime.now().year}-01-01')
```

### determinism / NON_INJECTED_CLOCK — `analytics/analyzers/risk_decomposition_analyzer.py:99`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'analysis_timestamp': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/arena_battle.py:94`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
model_type, 'registered_at': datetime.now(), 'activations':
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/arena_battle.py:343`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
battle.end_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/arena_battle.py:357`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/arena_battle.py:443`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return {'leaderboard': leaderboard, 'last_updated': datetime.now().
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/ensemble_performance_bridge.py:41`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
elapsed = (datetime.now() - self._last_sync).total_seconds()
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/ensemble_performance_bridge.py:53`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"sync_time": datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/ensemble_performance_bridge.py:57`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self._last_sync = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/ensemble_performance_bridge.py:89`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"last_updated": datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/ensemble_performance_bridge.py:105`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": getattr(m, "timestamp", datetime.now()),
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/ensemble_performance_bridge.py:133`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"last_updated": m.get("timestamp", datetime.now()),
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/performance_tracker.py:110`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()})
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/performance_tracker.py:126`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp=datetime.now(), accuracy=metrics.get('accuracy',
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/performance_tracker.py:143`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
stats['last_battle'] = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/performance_tracker.py:222`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
=float(avg_win_rate), last_updated=datetime.now())
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/performance_tracker.py:244`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_date = datetime.now() - timedelta(days=days)
```

### determinism / NON_INJECTED_CLOCK — `analytics/arena/performance_tracker.py:356`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `analytics/calculators/fama_french_factors.py:99`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
if self.last_cache_time and (datetime.now() - self.last_cache_time) < self.cache_expiry:
```

### determinism / NON_INJECTED_CLOCK — `analytics/calculators/fama_french_factors.py:143`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.last_cache_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `analytics/calculators/fama_french_factors.py:188`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
if self.last_cache_time and (datetime.now() - self.last_cache_time) < self.cache_expiry:
```

### determinism / NON_INJECTED_CLOCK — `analytics/calculators/fama_french_factors.py:221`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.last_cache_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `analytics/context/ensemble_selector.py:99`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"selection_time": datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `analytics/context/market_context_analyzer.py:163`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
) else datetime.now().hour
```

### determinism / NON_INJECTED_CLOCK — `analytics/context/market_context_analyzer.py:167`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
) else datetime.now().weekday()
```

### determinism / NON_INJECTED_CLOCK — `analytics/data_managers/model_results_manager.py:52`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
df_to_save['ingestion_timestamp'] = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `analytics/reporting/automated_reports.py:30`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
filename = f"daily_{datetime.now().strftime('%Y%m%d')}.json"
```

### determinism / NON_INJECTED_CLOCK — `analytics/reporting/automated_reports.py:41`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"date": datetime.now().strftime("%Y-%m-%d"),
```

### determinism / NON_INJECTED_CLOCK — `analytics/reporting/automated_reports.py:76`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `analytics/reporting/automated_reports.py:84`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
filename = f"trends_{datetime.now().strftime('%Y%m%d')}.json"
```

### determinism / NON_INJECTED_CLOCK — `analytics/reporting/model_analyzer.py:63`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `analytics/reporting/model_analyzer.py:149`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
file_path = self.report_dir / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
```

### determinism / NON_INJECTED_CLOCK — `analytics/reporting/results_manager.py:43`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### determinism / NON_INJECTED_CLOCK — `analytics/signals/signal_analytics.py:225`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'analysis_timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `backtesting/advanced/advanced_engine.py:264`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
report: dict[str, Any] = {'timestamp': datetime.now().isoformat
```

### determinism / NON_INJECTED_CLOCK — `calibration/adaptive_confidence_calibrator.py:116`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.calibration_history.append({'timestamp': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `calibration/adaptive_confidence_calibrator.py:140`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
now = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `calibration/adaptive_confidence_calibrator.py:189`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.last_retrain_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `colab/memory/memory_monitor.py:17`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.start_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `colab/memory/memory_monitor.py:38`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `core/base_integration.py:44`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"last_check": datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `core/error_handling/error_handler.py:134`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
error_info = {'timestamp': datetime.now().isoformat(), 'error_type':
```

### determinism / NON_INJECTED_CLOCK — `core/error_handling/error_handler.py:150`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
now = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `core/error_handling/error_handler.py:248`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
start_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `core/error_handling/error_handler.py:251`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
duration = datetime.now() - start_time
```

### determinism / NON_INJECTED_CLOCK — `core/error_handling/error_handler.py:255`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
duration = datetime.now() - start_time
```

### determinism / NON_INJECTED_CLOCK — `core/error_handling/error_handler.py:326`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.start_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `core/error_handling/error_handler.py:333`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
duration = datetime.now() - self.start_time
```

### determinism / NON_INJECTED_CLOCK — `core/logging/logger.py:156`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `core/logging/logger.py:187`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
structured_data = {'message': message, 'timestamp': datetime.now().
```

### determinism / NON_INJECTED_CLOCK — `core/system/archive_manager.py:32`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### determinism / NON_INJECTED_CLOCK — `core/system/batch_processor.py:232`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### determinism / NON_INJECTED_CLOCK — `core/system/version_manager.py:116`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
```

### determinism / NON_INJECTED_CLOCK — `core/system/version_manager.py:124`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

### determinism / NON_INJECTED_CLOCK — `core/system/version_manager.py:181`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
date=datetime.now().strftime("%Y-%m-%d"),
```

### determinism / NON_INJECTED_CLOCK — `core/validation/validators.py:109`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp: datetime = Field(default_factory=datetime.now)
```

### determinism / NON_INJECTED_CLOCK — `core/validation/validators.py:127`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp: datetime = Field(default_factory=datetime.now)
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/aaii_sentiment_collector.py:56`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
df['collected_at'] = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/alternative_me_collector.py:129`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
date = datetime.now().strftime('%Y-%m-%d')
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/alternative_me_collector.py:134`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.fromtimestamp(timestamp).isoformat() if timestamp > 0 else datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/cftc_collector.py:72`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
df['collected_at'] = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/cftc_collector.py:226`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
base_date = datetime.now() - timedelta(days=140)
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/economic_calendar_collector.py:109`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
start = datetime.now() - timedelta(days=days_back)
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/economic_calendar_collector.py:110`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
end = datetime.now() + timedelta(days=days_ahead)
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/fear_greed_collector.py:56`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
df['collected_at'] = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/fred_collector.py:28`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/fred_collector.py:31`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/fred_collector.py:34`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/put_call_ratio_collector.py:64`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
df['collected_at'] = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/put_call_ratio_collector.py:111`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
base_date = datetime.now() - timedelta(days=60)
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/put_call_ratio_collector.py:150`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'date': datetime.now().strftime('%Y-%m-%d'),
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/put_call_ratio_collector.py:156`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/put_call_ratio_collector.py:168`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
base_date = datetime.now() - timedelta(days=60)
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/reddit_sentiment_collector.py:67`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
df['collected_at'] = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/reddit_sentiment_collector.py:101`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
base_date = datetime.now() - timedelta(days=60)
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/rss_collector.py:190`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff = datetime.now(timezone.utc) - timedelta(days=self.
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/sec_filings_collector.py:76`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
run_date = kwargs.get("run_date", datetime.now())
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/vix_collector.py:84`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
df['collected_at'] = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/yf_collector.py:52`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
end_date = end_date or datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/yf_collector.py:77`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
# Use reference_now from kwargs if provided for stable testing, otherwise datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `data/collectors/yf_collector.py:78`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
reference_now = kwargs.get('reference_now', datetime.now())
```

### determinism / NON_INJECTED_CLOCK — `data/data_loader.py:73`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `data/management/data_versioning.py:90`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'description': description, 'created_at': datetime.now().
```

### determinism / NON_INJECTED_CLOCK — `data/management/data_versioning.py:122`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
file_age = datetime.now() - current_mtime
```

### determinism / NON_INJECTED_CLOCK — `data/management/data_versioning.py:134`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
metadata_age = datetime.now() - stored_mtime
```

### determinism / NON_INJECTED_CLOCK — `data/management/data_versioning.py:265`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
report = {'generated_at': datetime.now().isoformat(), 'data_type':
```

### determinism / NON_INJECTED_CLOCK — `data/quality/data_freshness_checker.py:62`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.metrics['last_check_time'] = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `data/quality/data_freshness_checker.py:96`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
now = pd.Timestamp.now()  # This is tz-naive by default
```

### determinism / NON_INJECTED_CLOCK — `features/analysis/news_decay_modeler.py:268`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### determinism / NON_INJECTED_CLOCK — `features/analysis/news_decay_modeler.py:305`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = datetime.now() - timedelta(days=days)
```

### determinism / NON_INJECTED_CLOCK — `features/analysis/regime_importance_tracker.py:79`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
current_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `features/analysis/regime_importance_tracker.py:181`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = datetime.now() - timedelta(days=days)
```

### determinism / NON_INJECTED_CLOCK — `features/enrichers/macro_features_enricher.py:109`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
end_date = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `features/feature_orchestrator.py:194`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
start_time = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `features/feature_orchestrator.py:205`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
end_time = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `features/feature_selection_cache.py:125`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
datetime.now().isoformat(), 'n_available_features': len(
```

### determinism / NON_INJECTED_CLOCK — `features/monitoring/feature_drift_detector.py:77`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.metrics['last_check_time'] = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `features/monitoring/feature_drift_detector.py:137`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': pd.Timestamp.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `features/monitoring/feature_drift_detector.py:212`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
```

### determinism / NON_INJECTED_CLOCK — `features/news_dataset_builder.py:257`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
dataset_df['generated_at'] = datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `features/news_dataset_builder.py:274`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'generated_at': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `features/nlp/processors/news_harmonizer.py:48`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
pub_dt = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `features/nlp/processors/news_processing.py:36`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
fname = f"clustered_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
```

### determinism / NON_INJECTED_CLOCK — `features/selection/enhanced_smart_selector.py:84`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `features/selection/smart_selector.py:148`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `features/utils/datetime_utils.py:81`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
df['datetime'] = pd.Timestamp.now().tz_localize(None)
```

### determinism / NON_INJECTED_CLOCK — `features/validation/feature_leakage_guard.py:32`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.timestamp = datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `features/validation/feature_leakage_guard.py:183`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:93`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self._cache_timestamps[cache_key] = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:145`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
age = (datetime.now() - cache_time).total_seconds()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:165`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:178`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:200`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:206`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(), 'pnl': 0.023},
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:208`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(), 'pnl': -0.015},
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:210`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(), 'pnl': 0.008}
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:213`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:239`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:248`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:272`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:276`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:291`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:321`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:331`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:347`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_rebalanced': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:370`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/dashboard_data_bridge.py:383`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `integration/ensemble_performance_bridge.py:50`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
time_since_sync = datetime.now() - self._last_sync_time
```

### determinism / NON_INJECTED_CLOCK — `integration/ensemble_performance_bridge.py:71`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'sync_time': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `integration/ensemble_performance_bridge.py:80`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self._last_sync_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `integration/ensemble_performance_bridge.py:111`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `integration/ensemble_performance_bridge.py:148`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `integrations/base.py:47`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
datetime.now().isoformat(), 'error': error}
```

### determinism / NON_INJECTED_CLOCK — `integrations/data/bigquery_client.py:185`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
dates = [(datetime.now() - timedelta(days=i)).strftime('%Y%m%d') for
```

### determinism / NON_INJECTED_CLOCK — `integrations/data/bigquery_client.py:199`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
datetime.now(), periods=5, freq='D')), 'value': np.random.randn
```

### determinism / NON_INJECTED_CLOCK — `main/modes/web_ui.py:173`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return {'status': 'idle', 'last_update': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `main/modes/web_ui.py:201`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
10000000), 'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `main/modes/web_ui.py:213`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return {'recent_activity': [{'time': datetime.now().strftime(
```

### determinism / NON_INJECTED_CLOCK — `main/modes/web_ui.py:215`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'price': 245.5}, {'time': datetime.now().strftime('%H:%M'),
```

### determinism / NON_INJECTED_CLOCK — `main/modes/web_ui.py:217`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
178.25}, {'time': datetime.now().strftime('%H:%M'), 'action':
```

### determinism / NON_INJECTED_CLOCK — `main/modes/web_ui.py:233`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'disk_usage': 23.1, 'last_update': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/awareness/context/manager.py:59`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp=datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/awareness/context/storage.py:115`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/evolution/dual_loops.py:130`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
rule_id = f"RULE_{agent_id.upper()}_{datetime.now().strftime('%Y%m%d%H%M')}_{i}"
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/memory/diary_engine.py:71`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
decision_timestamp: int = field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/memory/diary_engine.py:160`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
decision_timestamp=int(datetime.now(timezone.utc).timestamp()),
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/memory/diary_engine.py:207`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"decision_timestamp": int(pd.Timestamp.now().timestamp() * 1000),
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/memory/diary_engine.py:371`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/real_time_learning.py:100`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': trade.get('timestamp', datetime.now()),
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/real_time_learning.py:165`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_update': datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/real_time_learning.py:256`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
next_adaptation = datetime.now() + timedelta(hours=trades_until_adaptation * 0.5)
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/real_time_learning.py:311`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/agent_permissions.py:125`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
created_at=datetime.now(), last_active=datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/agent_permissions.py:437`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
now = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/agent_permissions.py:465`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
now = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/agent_permissions.py:503`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.action_counts[agent_id][action_type].append(datetime.now())
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/agent_permissions.py:505`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self._registered_agents[agent_id].last_active = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/agent_permissions.py:512`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
audit_entry = {'timestamp': datetime.now().isoformat(), 'agent_id':
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/agent_permissions.py:525`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
f"agent_audit_{datetime.now().strftime('%Y%m%d')}.json")
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/constraint_engine.py:139`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp=datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/constraint_engine.py:303`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
if v.timestamp > datetime.now() - timedelta(hours=1)
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/constraint_engine.py:316`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = datetime.now() - timedelta(hours=hours)
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/constraint_engine.py:474`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
current_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/constraint_engine.py:579`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp=datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `meta_learning/security/constraint_engine.py:598`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = datetime.now() - timedelta(hours=24)
```

### determinism / NON_INJECTED_CLOCK — `models/analysis/baseline_dominance_detector.py:74`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `models/analysis/model_health_analyzer.py:41`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `models/analysis/overfitting_detection/manager.py:38`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `models/analysis/regime_winner_analyzer.py:101`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
current_time = current_time or datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/analysis/regime_winner_analyzer.py:273`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff = datetime.now() - timedelta(days=days)
```

### determinism / NON_INJECTED_CLOCK — `models/dean/dean_bootstrap_system.py:161`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/dean/dean_bootstrap_system.py:216`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

### determinism / NON_INJECTED_CLOCK — `models/dean/dean_bootstrap_system.py:261`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp=datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `models/ensemble/correlation/correlation_engine.py:81`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `models/ensemble/correlation/correlation_engine.py:460`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = datetime.now() - timedelta(days=days)
```

### determinism / NON_INJECTED_CLOCK — `models/ensemble/dynamic_weights.py:245`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'exported_at': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `models/ensemble/weight_stability/manager.py:42`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
if timestamp is None: timestamp = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/ensemble/weight_stability/manager.py:113`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff = datetime.now() - timedelta(days=days)
```

### determinism / NON_INJECTED_CLOCK — `models/ensemble/weight_stability_monitor.py:43`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return monitor.update_weights(new_weights, datetime.now())
```

### determinism / NON_INJECTED_CLOCK — `models/integrated_model_manager.py:100`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `models/model_selector/adaptive_selector.py:161`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.selection_history.append({'timestamp': datetime.now().
```

### determinism / NON_INJECTED_CLOCK — `models/model_selector/adaptive_selector.py:310`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'last_updated': datetime.now().isoformat(), 'arena_integrated':
```

### determinism / NON_INJECTED_CLOCK — `models/model_selector/adaptive_selector.py:360`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
performance_tracker.items()}, 'exported_at': datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/model_selector/heavy_light_comparator.py:84`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp=datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/model_selector/smart_selector.py:289`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
run_data = {'timestamp': pd.Timestamp.now().isoformat(), 'metrics':
```

### determinism / NON_INJECTED_CLOCK — `models/monitoring/drift/alert_system.py:163`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/monitoring/drift/alert_system.py:223`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/monitoring/drift/alert_system.py:294`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/monitoring/drift/alert_system.py:330`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
```

### determinism / NON_INJECTED_CLOCK — `models/monitoring/drift/alert_system.py:390`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/monitoring/prediction_drift_monitor.py:63`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/monitoring/prediction_drift_monitor.py:313`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = datetime.now() - timedelta(days=days)
```

### determinism / NON_INJECTED_CLOCK — `models/persistent_pool.py:110`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'added_at': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `models/persistent_pool.py:246`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'updated_at': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `models/prototypes/prototype.py:64`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.created_at = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `models/quality/controller.py:111`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'updated_at': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `models/quality/controller.py:183`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `models/registry/model_registry.py:56`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'registered_at': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `models/registry/model_registry.py:132`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### determinism / NON_INJECTED_CLOCK — `models/statistics/model_statistics.py:106`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = datetime.now() - timedelta(days=30)
```

### determinism / NON_INJECTED_CLOCK — `monitoring/dashboard.py:450`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
```

### determinism / NON_INJECTED_CLOCK — `monitoring/dashboard.py:559`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### determinism / NON_INJECTED_CLOCK — `monitoring/data_freshness_monitor.py:129`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
current_time = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/data_freshness_monitor.py:165`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': pd.Timestamp.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/data_freshness_monitor.py:460`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = pd.Timestamp.now() - pd.Timedelta(hours=hours)
```

### determinism / NON_INJECTED_CLOCK — `monitoring/example_usage.py:87`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/example_usage.py:108`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `monitoring/example_usage.py:145`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/example_usage.py:244`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### determinism / NON_INJECTED_CLOCK — `monitoring/example_usage.py:278`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/example_usage.py:285`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/feature_drift_monitor.py:92`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.metrics['last_check_time'] = datetime.now()  # type: ignore
```

### determinism / NON_INJECTED_CLOCK — `monitoring/feature_drift_monitor.py:149`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### determinism / NON_INJECTED_CLOCK — `monitoring/health_hub.py:164`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return {'timestamp': datetime.now().isoformat(), 'metrics':
```

### determinism / NON_INJECTED_CLOCK — `monitoring/health_hub.py:244`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
drift_detected, 'timestamp': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `monitoring/health_hub.py:274`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff = datetime.now() - timedelta(days=window_days)
```

### determinism / NON_INJECTED_CLOCK — `monitoring/infrastructure/resource_monitor.py:81`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return {'timestamp': datetime.now().isoformat(), 'system': futures[
```

### determinism / NON_INJECTED_CLOCK — `monitoring/ml_analytics.py:69`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
results = {'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `monitoring/ml_analytics.py:107`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return {'timestamp': datetime.now().isoformat(), 'metrics':
```

### determinism / NON_INJECTED_CLOCK — `monitoring/ml_analytics.py:137`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff = datetime.now() - timedelta(days=window_days)
```

### determinism / NON_INJECTED_CLOCK — `monitoring/ml_analytics.py:168`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
datetime.now().hour), float(datetime.now().dayofweek)]
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:82`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return {'monitor_name': self.name, 'timestamp': datetime.now().
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:123`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
process_count, 'timestamp': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:158`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
alertstatus.ACTIVE.value, 'timestamp': datetime.now().isoformat
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:184`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
0.0, 'timestamp': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:210`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
alertstatus.ACTIVE.value, 'timestamp': datetime.now().isoformat
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:239`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
s in self.data_sources.values()), 'timestamp': datetime.now
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:263`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
alertstatus.ACTIVE.value, 'timestamp': datetime.now().isoformat
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:297`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
alert['resolved_at'] = datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:312`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = datetime.now() - timedelta(hours=self.auto_resolve_hours)
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:370`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
dashboard_data: dict[str, Any] = {'timestamp': datetime.now().
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:400`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = datetime.now() - timedelta(hours=hours)
```

### determinism / NON_INJECTED_CLOCK — `monitoring/monitoring_system.py:503`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
get_active_alerts()), 'last_collection': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/reporting/performance_reports.py:47`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': (timestamp or datetime.now()).isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/reporting/performance_reports.py:104`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `monitoring/tests.py:110`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/tests.py:172`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `monitoring/tests.py:191`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/tests.py:212`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/tests.py:219`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `monitoring/tests.py:325`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `optimization/dynamic_config_updater.py:144`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `optimization/hyperparameter_searcher.py:305`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
f"hyperparameter_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
```

### determinism / NON_INJECTED_CLOCK — `optimization/hyperparameter_searcher.py:309`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'trial_history': self.trial_history, 'timestamp': datetime.now(
```

### determinism / NON_INJECTED_CLOCK — `patterns/pattern_analyzer.py:36`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
f"pattern_{datetime.now().strftime('%H%M%S')}")
```

### determinism / NON_INJECTED_CLOCK — `patterns/pattern_analyzer.py:38`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.log('INFO', f'Start time: {datetime.now().isoformat()}')
```

### determinism / NON_INJECTED_CLOCK — `patterns/pattern_analyzer.py:44`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `patterns/pattern_analyzer.py:144`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### determinism / NON_INJECTED_CLOCK — `patterns/pattern_analyzer.py:235`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
results['analysis_timestamp'] = datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/guards/macro_release_timing_guard.py:497`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
current_time = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/guards/safe_feature_combiner.py:359`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': pd.Timestamp.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/guards/safe_feature_combiner.py:442`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
current_time = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/guards/temporal_leakage_guard.py:533`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
current_time = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/guards/temporal_target_guard.py:120`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return guard.generate_targets_safe(df, timeframe, pd.Timestamp.now(), configs)
```

### determinism / NON_INJECTED_CLOCK — `pipeline/guards/timeframe_alignment_guard.py:392`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
current_time = pd.Timestamp.now()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/cache_manager.py:58`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
days = (datetime.now() - datetime.fromisoformat(last_ts)).days
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/colab_manager.py:59`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/feature_selection_manager.py:85`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': pd.Timestamp.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/feature_selection_validator.py:78`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
days = (datetime.now() - datetime.fromisoformat(last_ts)).days
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/feature_selection_validator.py:154`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/final_stages_executor.py:192`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return {'timestamp': datetime.now().isoformat(), 'batch_name': self
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/final_stages_executor.py:202`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
) / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/final_stages_orchestrator.py:86`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/final_stages_orchestrator.py:95`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
output_path = self.output_dir / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/light_models_trainer.py:51`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/metadata_manager.py:94`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/metadata_manager.py:100`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/model_training_orchestrator.py:221`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/pipeline_executor.py:34`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/pipeline_metadata_manager.py:62`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
accumulated_results = {'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/hybrid/pipeline_runner.py:40`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/analytics.py:93`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": pd.Timestamp.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/backtest_adapter.py:24`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
price_pivot.index = [pd.Timestamp.now()]
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/backtest_adapter.py:33`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
price_pivot.index = [pd.Timestamp.now()]
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/backtest_analyzer.py:67`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
price_pivot.index = [pd.Timestamp.now()]
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/backtest_analyzer.py:77`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
price_pivot.index = [pd.Timestamp.now()]
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/backtest_analyzer.py:107`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
end_date = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/backtest_analyzer.py:144`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
dates = pd.date_range(end=datetime.now(), periods=2, freq='D')
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/backtest_analyzer.py:267`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
dates = pd.date_range(end=pd.Timestamp.now(), periods=2, freq='D')
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/report_generator.py:96`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': pd.Timestamp.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/report_generator.py:116`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': pd.Timestamp.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/evaluation/report_generator.py:160`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
file_path = save_dir / f"summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/feature_engineering/orchestrator.py:86`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/modeling/metrics.py:23`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/modeling/metrics.py:35`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/modeling/utils.py:31`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/modeling/utils.py:42`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/prediction/result_builder.py:116`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
ts_val = datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/prediction/result_builder.py:188`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
stage_5_results = {'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/processing/orchestrator.py:115`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/processing/storage.py:20`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/stage_4_modeling.py:129`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/stage_5_prediction.py:439`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
stage_5_results = {'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/stage_7_evaluation.py:174`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': pd.Timestamp.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/trading/data_io.py:94`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
stage_6_results = {'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `pipeline/stages/trading/recommendation_engine.py:447`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `predictions/caching.py:115`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
if ttl_seconds and (datetime.now() - timestamp).total_seconds(
```

### determinism / NON_INJECTED_CLOCK — `predictions/caching.py:130`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.cache[cache_key] = result, datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `risk/elite_risk_metrics.py:222`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
position_risks, 'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `risk/elite_risk_metrics.py:476`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
violations, 'warnings': warnings, 'checked_at': datetime.now().
```

### determinism / NON_INJECTED_CLOCK — `risk/kill_switch/alerts.py:16`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
ts = timestamp or datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `risk/kill_switch/manager.py:41`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `risk/kill_switch/manager.py:87`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
save_path = self.config_manager.storage_path / f"risk_event_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
```

### determinism / NON_INJECTED_CLOCK — `risk/max_exposure_monitor.py:38`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `trading/adaptive_parameter_manager.py:277`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': pd.Timestamp.now(),
```

### determinism / NON_INJECTED_CLOCK — `trading/consensus_engine.py:31`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp: datetime = field(default_factory=datetime.now)
```

### determinism / NON_INJECTED_CLOCK — `trading/live_adaptive_ensemble.py:99`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp=datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `trading/live_adaptive_ensemble.py:124`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
(datetime.now() - self.last_reweight_time).days >= self.reweight_interval_days
```

### determinism / NON_INJECTED_CLOCK — `trading/live_adaptive_ensemble.py:129`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
cutoff_time = datetime.now() - timedelta(days=lookback_days)
```

### determinism / NON_INJECTED_CLOCK — `trading/live_adaptive_ensemble.py:207`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp=datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `trading/live_adaptive_ensemble.py:253`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
self.last_reweight_time = datetime.now()
```

### determinism / NON_INJECTED_CLOCK — `trading/virtual_portfolio.py:113`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
performance_history, 'last_updated': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `trading/virtual_portfolio.py:183`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
return {'timestamp': datetime.now(), 'type': 'BUY', 'ticker':
```

### determinism / NON_INJECTED_CLOCK — `trading/virtual_portfolio.py:203`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
total_cost / quantity, 'entry_time': datetime.now(),
```

### determinism / NON_INJECTED_CLOCK — `trading/virtual_portfolio.py:225`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
transaction = {'timestamp': datetime.now(), 'type': 'SELL',
```

### determinism / NON_INJECTED_CLOCK — `trading/virtual_portfolio.py:259`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
equity_curve[datetime.now()] = total_value
```

### determinism / NON_INJECTED_CLOCK — `trading/virtual_portfolio.py:272`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
positions_report, 'metrics': metrics, 'timestamp': datetime.now
```

### determinism / NON_INJECTED_CLOCK — `trading/virtual_portfolio.py:278`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
record = {'timestamp': datetime.now().isoformat(), 'total_value':
```

### determinism / NON_INJECTED_CLOCK — `training/adaptive_training_manager.py:350`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'execution_phases']), 'timestamp': datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `training/adaptive_training_manager.py:357`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
```

### determinism / NON_INJECTED_CLOCK — `training/base_trainer.py:385`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `training/pattern_aware_training.py:75`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(),
```

### determinism / NON_INJECTED_CLOCK — `training/progressive_trainer.py:326`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `training/progressive_trainer.py:331`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### determinism / NON_INJECTED_CLOCK — `training/state/training_state_manager.py:100`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
"timestamp": datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `training/unified_training_manager.py:98`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
datetime.now().isoformat()}
```

### determinism / NON_INJECTED_CLOCK — `training/unified_training_manager.py:168`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'timestamp': datetime.now().isoformat(), 'ticker_plans': {}})
```

### determinism / NON_INJECTED_CLOCK — `training/unified_training_manager.py:203`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
f"unified_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
```

### determinism / NON_INJECTED_CLOCK — `training/unified_training_manager.py:210`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
f"unified_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
```

### determinism / NON_INJECTED_CLOCK — `utils/trading_calendar.py:17`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
def __init__(self, start_year: int=2020, end_year: int=datetime.now().
```

### determinism / NON_INJECTED_CLOCK — `validation/temporal_feature_separator.py:93`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'analysis_timestamp': datetime.now().isoformat()
```

### determinism / NON_INJECTED_CLOCK — `validation/temporal_feature_separator.py:287`
**Problem:** Direct current-time call detected.
**Why:** Runtime and tests become nondeterministic; relative dates can drift.
**Fix:** Inject a clock/reference_now parameter or central time provider.
**Test:** Freeze clock in tests and assert stable outputs.
**Confidence:** medium  
```python
'generated_at': datetime.now().isoformat()
```
