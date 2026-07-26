# src/archive/ — Manifest

Everything under `src/archive/` is **confirmed dead code**: retired,
superseded, or never-finished implementations that are **not imported by
any live code path** (only by their own tests, if that). Kept for
reference instead of deleted, so history and any reusable logic aren't
lost. Do not import from `src.archive.*` in new production code — if you
need something from here, either promote it back to `src/` deliberately
(with a real caller and a live test) or leave it archived.

This file exists because three separate archival passes happened across
different sessions/agents without a shared record of what was already
done, which cost real time re-discovering "is this actually dead?" more
than once. Update this file whenever you add to or promote something out
of `src/archive/`.

## Wave 1 — commit `16b207494` (2026-07-22)

*"Superseded/retired implementations kept for reference... Same-session
leakage audit confirmed none of this is imported anywhere in the active
src/ tree — it's inert."*

- `backtesting/` — old backtesting engine, superseded.
- `data/` (`clean_price_anomalies.py`, `data_loader.py`), `data_sources/`
  (`local_file_data_source.py`) — superseded data loading.
- `features/` (`colab_context_integration.py`,
  `context_aware_feature_selector.py`, `feature_selection_cache.py`) —
  superseded feature-selection experiments.
- `meta_learning/` (`dean_trading_models.py`, `experience_diary.py`) —
  superseded meta-learning experiments.
- `models/` (`factory.py`, `feature_selector.py`) — an earlier model
  factory/selector, superseded by the current `src/factories/model_factory.py`.
- `monitoring/` (`example_usage.py`, `tests.py`) — example/demo scripts.
- `patterns/` (`pattern_analyzer.py`, `pattern_tuning.py`) — superseded
  pattern-detection experiments.
- `processing/sampling.py`, `reporting/` (whole subpackage), `risk/`
  (`exposure_calculator.py`, `kill_switch/` subpackage,
  `kill_switch_manager.py`, `metrics.py`, `risk_manager.py`) — an earlier,
  standalone risk-management/kill-switch implementation. **Superseded by
  the current live risk stack**: `src/risk/analyzers/` (concentration,
  correlation, VaR), `src/risk/elite_risk_metrics.py`,
  `src/risk/max_exposure_monitor.py`, and dean_os's own
  `anxiety_kill_switch.py`/`risk_engine.py` for governance-layer checks.
- `utils/` (`collector_fixes.py`, `data_safety.py`,
  `feature_preparation.py`, `missing_data_anomaly_detector.py`,
  `resource_limiter.py`, `smart_missing_data_handler.py`,
  `type_conversion.py`) — superseded utility modules.
- `validation/` (`temporal_feature_separator.py`, `validation_protocols.py`)
  — superseded validation helpers.
- Root-level singles: `dean_integration.py`, `financial_metrics.py`,
  `legacy_smart_trader.py`, `online_model_integration.py`,
  `portfolio_optimizer.py`.

## Wave 2 — commit `7f8f1cd7` ("archive confirmed-dead code, not delete;
fix stale imports uncovered along the way")

An earlier, separate archival pass, unrelated to waves 1 and 3.

- `model_selector_dead/` — an entire retired model-selection subsystem
  (`competence_analyzer.py`, `competence_map.py`,
  `context_prediction_mapper.py`, `dynamic_weight_selector.py`,
  `enhanced_context_analyzer.py`, `heavy_light_comparator.py`,
  `selector.py`, `temporal_feature_analyzer.py`). Superseded by the
  current live model-selection stack under `src/models/model_selector/`.
- `models_dead/` — retired model implementations
  (`enhanced_ensemble.py`, `integrated_model_manager.py`,
  `knowledge_ingestor.py`, `neural_network_model.py`,
  `universal_registry.py`).

## Wave 3 — this session, 2026-07-25 (commits `573ad98b`, `7bf7d53f`,
plus the `PredictionResultRequest` archival in `155c85ca`)

Commit `e34650e0` (2026-07-22, "repo root cleanup") deleted these files
in the same commit as genuine junk (`scratch/`, `mlruns/` binaries,
one-off audit reports) — but unlike the files in Wave 1, these specific
ones were **never actually archived anywhere**, leaving 14 test files
broken (`ModuleNotFoundError`/`ImportError` at collection time) for
~3 days before this audit found and fixed it. Confirmed zero live callers
for every one of these before restoring (same standard as waves 1-2),
then restored each from `e34650e0^` (the commit right before the
deletion) into its original relative path under `src/archive/`.

- `algorithms/` — `advanced_backtest_engine.py`, `bias_detector.py` (a
  dependency of the former — **do not confuse with the LIVE, different**
  `src.backtesting.advanced.advanced_engine.BiasDetector`, which is a
  re-export of `src.analytics.detectors.bias_detector.BiasDetector`, a
  completely separate, currently-used class), `walk_forward_optimizer.py`.
- `calibration/` — `calibration_engine.py`, `adaptive_confidence_calibrator.py`.
  Both were meant to be archived alongside their sibling risk files in
  Wave 1 but were missed.
- `features/utils/` — `hybrid_adaptive_technical_indicators.py`,
  `modular_adaptive_technical_indicators.py`,
  `simple_adaptive_technical_indicators.py`. Note:
  `hybrid_adaptive_technical_indicators.py` has a known, pre-existing
  math bug (Bollinger Bands produce NaNs) — left as-is since it's
  confirmed dead code, not worth fixing.
- `meta_learning/real_time_learning.py`, `trading/signal_processor.py`,
  `training/pattern_aware_training.py` — restored at their original
  nested paths. (A flattened, top-level copy of each of these three
  already existed from Wave 1 with byte-identical content — those
  duplicates were removed 2026-07-26 once discovered, keeping only the
  nested versions all current tests reference.)
- `integration/ensemble_selector.py` — do not confuse with the live,
  different `src.analytics.context.ensemble_selector`, which this
  archived file's own (fixed) import points at.
- `pipeline/stages/stage_3_improvements.py`.
- `pipeline/stages/prediction/result_request.py` — a different kind of
  entry than the rest of this wave: **not from the `e34650e0` deletion**.
  This was an orphaned, diverged duplicate `PredictionResultRequest`
  dataclass (missing the `models` field the real one has) that existed
  alongside the actual live class defined inline in
  `src/pipeline/stages/prediction/orchestrator.py`. Archived after fixing
  the one test that had been mistakenly pointed at it. If you're looking
  for the real `PredictionResultRequest`, it's in `orchestrator.py`, not
  here.
- `pipeline/stages/prediction/result_builder.py` (archived 2026-07-26) —
  the class `PredictionResultBuilder` this file defines is a complete,
  never-wired-in parallel implementation of the same "build Stage 5
  result" job `orchestrator.py`'s own `_create_prediction_result`/
  `_prepare_final_results`/`_save_stage_5_results` already do live. Its
  own `from .result_request import PredictionResultRequest` broke the
  moment `result_request.py` (above) was archived, and it had zero real
  callers anywhere (grep found only comments referencing the filename) —
  confirmed via direct import before archiving. The live orchestrator
  path is more advanced in one way (confidence calibration +
  `prediction_ledger` recording, absent here) but this file has one
  capability the live path lacks: `_integrate_autoencoder_anomaly()`
  blends an autoencoder reconstruction-error "normalcy" score into the
  anomaly score, using a model keyed `{ticker}_{target}_autoencoder` in
  the batch dir. If that signal is ever wanted in production, port
  `_load_autoencoder_model`/`_calculate_autoencoder_normalcy`/
  `_integrate_autoencoder_anomaly` into `orchestrator.py`'s
  `_create_prediction_result` — don't resurrect this file as-is, since
  it duplicates the surrounding logic that already diverged.
- `data_sources/local_file_data_source.py` was already archived in
  Wave 1; this session only fixed its own internal cross-import
  (`__init__.py` was still importing from the pre-archival `src.data_sources.*`
  path) and updated `src/config/data_sources.yaml`'s module path to match
  (confirmed nothing in production dynamically reads that config entry
  today — inert drift, not a live crash, but fixed to keep the config
  honest).

## Wave 4 — commit `0f0aa460`, 2026-07-26 (`src/targets/` audit pass)

- `targets/calculators/base_news_target_calculator.py`,
  `post_news_target_calculator.py`, `pre_news_target_calculator.py` —
  never wired into `TargetOrchestrator.CALCULATOR_MAPPING` (which only has
  `regression`/`classification_binary`/`classification_multiclass`/
  `indicator_prediction`) and no config anywhere names a `post_news`/
  `pre_news` target type. Confirmed zero callers outside their own three
  files (repo-wide grep; the only other hits are `dean_os/draft/` and
  `audit/legacy/quarantine/`, both already-known non-live). Archived
  instead of deleted because they contain a real logic bug worth
  remembering if anyone ever re-wires them: `_get_upcoming_news()` /
  the inline equivalent in `post_news_target_calculator.py` build the
  ticker filter as
  `news_df[(news_df['ticker'] == ticker) | (news_df.get('news_type', 'general') == 'general')]`.
  `DataFrame.get(key, default)` returns the scalar `default` (not a
  per-row fallback) when the column is missing — if `news_type` isn't
  present on `news_df`, that OR clause becomes `True` for every row,
  silently matching ALL tickers' news instead of just the target
  ticker's, the same cross-ticker-contamination failure class as the
  `shift()`-without-`groupby` bug fixed earlier in this project's
  `regression_calculator.py`/`classification_calculator.py`/
  `indicator_prediction_calculator.py`. Also: neither subclass actually
  uses `BaseNewsTargetCalculator` (defined in the same directory) despite
  it existing specifically to share this logic — they reimplement a
  diverged copy inline instead. Fix the `.get()` bug and route both
  through the base class before ever wiring these back into
  `CALCULATOR_MAPPING`.

## Wave 5 — commit TBD, 2026-07-26 (`src/algorithms/` audit pass)

- `algorithms/transaction_cost_model.py` — a second, diverged
  `TransactionCostModel` class. The LIVE one is a different class in
  `src/backtesting/advanced/advanced_engine.py`, imported by
  `src/trading/virtual_portfolio.py`. The two have identical `__init__`
  config keys but diverged `calculate_execution_costs()` signatures: this
  archived version takes `(trade_value, daily_volume=1000000.0)` and
  returns a `float`; the live one requires an extra positional
  `volatility` arg and returns a `dict[str, float]`. Confirmed zero real
  callers outside this file and `src/algorithms/__init__.py`'s own
  re-export (removed) before archiving — only reference elsewhere was
  `archive/algorithms/advanced_backtest_engine.py`, already-archived
  Wave-3 dead code (its own import fixed to
  `from src.archive.algorithms.transaction_cost_model import ...` per the
  cross-import gotcha below). Same duplicate-class-divergence pattern as
  `PredictionResultRequest`/`result_builder.py` and the `backtest.py`-vs-
  Stage-7 case — if you're looking for the real `TransactionCostModel`,
  it's in `src/backtesting/advanced/advanced_engine.py`, not here.

## Wave 6 — commit TBD, 2026-07-26 (`src/analytics/` audit pass)

- `analytics/analyzer_registry.py` — a static `ANALYZER_REGISTRY` dict
  registering 8 analyzers by hand, stale against the real analyzer set
  (missing 6 of the 11 real classes under `analytics/analyzers/`) and
  unrelated to the actual live registration mechanism,
  `UnifiedAnalyticsEngine._register_analyzers_from_config()` (dynamically
  imports classes per `src/config/analysis.yaml`). Confirmed zero
  production callers (only `tests/smoke_test_system.py`, a standalone
  diagnostic script, not a pytest test) before archiving.
  `smoke_test_system.py` updated to check the real
  `UnifiedAnalyticsEngine.analyzers` dict instead — running it now
  correctly reports only the 2 analyzers actually enabled in
  `analysis.yaml` (`market_regime`, `critical_signals`), not the stale
  registry's fake 8. Note: `ANALYZER_REGISTRY["ensemble_selector"] =
  EnsembleSelector  # type: ignore` registered a class that doesn't
  implement `IAnalyzer.analyze()` at all (has
  `select_best_ensemble`/`create_ensemble_instance` instead) — a real
  contract mismatch, moot now that the registry itself is archived, but
  don't resurrect this entry as-is if this file is ever un-archived.
  `analytics/analyzers/wrappers.py` (the 3 classes this registry also
  registered) was deliberately **kept live, not archived**, despite also
  having zero production callers — `tests/unit/analytics/analyzers/test_wrappers.py`
  exercises it directly and found no bugs in it, only in the registry
  that pointed at it. Same reasoning kept `analytics/detectors/anomaly_detector.py`
  and `analytics/utils/analytics_math.py` live too (both zero production
  callers but real regression tests in `tests/unit/test_p1_missing_policy_math.py`
  locking in a specific historical missing-data-imputation fix) — orphaned
  code with real, passing test coverage stays, only orphaned code with
  zero test coverage gets archived.
- `analytics/detectors/critical_signal_detector.py`,
  `analytics/signals/signal_analytics.py`,
  `analytics/signals/significance_detector.py` — all three were
  initialized in `src/pipeline/stages/prediction/orchestrator.py.__init__`
  (`self.critical_signal_detector`/`self.signal_analytics`/
  `self.significance_detector`) but never called anywhere else in that
  file or the rest of `src/pipeline/stages/prediction/` — confirmed via
  grep, and confirmed zero test coverage for all three (unlike
  `anomaly_detector.py`/`wrappers.py`/`analytics_math.py` above). The
  file's own docstring documents the real refactor
  ("AnomalyEngine → anomaly detection & confidence scoring" - a
  *different*, actually-used class) that made these three obsolete
  leftovers. Removed the dead init block + the misleading `'✅ ...
  initialized'` log line from `orchestrator.py` and archived all three.
  `critical_signal_detector.py`'s `from ..interfaces import IAnalyzer`
  relative import was fixed to the absolute
  `from src.analytics.interfaces import IAnalyzer` (interfaces.py itself
  stayed live, only this file moved). Worth noting for whoever
  reconsiders re-enabling `significance_detector.py`:
  `_create_significance_column` correctly does
  `df.groupby('ticker')[col].shift(1)` but the neighboring
  `_create_macro_significance_column` does a plain `df[col].shift(1)`
  with no groupby — a cross-ticker leak for macro columns specifically,
  inconsistent with the sibling function 3 lines away. Not fixed since
  the whole file is dead; fix before ever re-wiring it in.

## Wave 7 — commit TBD, 2026-07-26 (`src/data/` audit pass)

- `data/collectors/alternative_me_collector.py`,
  `data/collectors/market_data_collector.py` — neither `collector_type`
  (`alternative_me`, `market_data`) is a key under `collectors:` in
  `src/config/collectors.yaml`, so `CollectorFactory.get_all_collectors()`
  (which iterates `collectors_config.keys()`) can never produce either,
  even though `CollectorFactory._discover_collector_classes()` still
  auto-discovers both classes via `pkgutil.walk_packages`. Confirmed zero
  callers anywhere in `src/`/`tests/` before archiving.
  `alternative_me_collector.py` is a near-duplicate of the live
  `fear_greed_collector.py` (both target the Fear & Greed Index, different
  endpoints). Both files' relative `from .base_collector import
  BaseCollector` fixed to the absolute `from src.data.collectors.base_collector
  import BaseCollector` (that module stayed live, only these two moved).
- `data/quality/temporal_alignment_checker.py`,
  `data/quality/news_price_availability_filter.py`,
  `data/quality/data_freshness_checker.py`,
  `data/validation/event_dataset_validator.py`,
  `data/management/data_versioning.py`,
  `data/management/handlers/connection_handler.py` — **an entire
  point-in-time-leakage-prevention layer that was never wired into the
  live pipeline.** Every one of these modules is built specifically to
  prevent exactly the leak bugs this project has repeatedly had, and
  every one has zero callers anywhere in `src/` or `tests/` — confirmed
  via repo-wide grep before archiving each. The real, live equivalent of
  `temporal_alignment_checker.py`'s intent is a completely separate
  implementation, `src/pipeline/guards/timeframe_alignment_guard.py::TimeframeAlignmentGuard`
  (see the `src/pipeline/guards/` note below for whether even *that* one
  is fully wired in). `event_dataset_validator.py` also had a live
  `self.logger`-doesn't-exist bug (same pattern fixed multiple times
  elsewhere this session) inside `_check_datetime_columns` — fixed before
  archiving, so it's at least correct if anyone resurrects it.
  `data_versioning.py`'s `cleanup_stale_files` has a partial-completion
  bug (a deletion failure appends to `failed_deletions` then re-raises,
  aborting the loop and skipping any remaining stale files in the batch)
  — not fixed, since the whole file is dead; fix before re-wiring it in.
  `connection_handler.py` is a near-verbatim duplicate of the connection-
  pooling logic already in the live `DataManager`
  (`src/data/management/data_manager.py` lines 57-145) — if this predates
  `DataManager`'s pooling, it's a superseded draft; if it postdates it,
  someone started migrating and never finished. Either way, don't treat
  it as the "real" connection handler.
- `data/management/data_cleaner.py`'s `DataCleaner` was deliberately
  **kept live, not archived**, despite zero production callers — same
  "orphaned but has real regression test coverage" rule as Wave 6:
  `tests/unit/test_p1_missing_policy_math.py::test_management_data_cleaner_preserves_missing_numeric_values`
  exercises `clean_numeric_data()` directly, locking in the same
  historical "don't zero-fill missing values" fix as `anomaly_detector.py`
  and `analytics_math.py` (Wave 6). **Real landmine, addressed via a
  docstring warning rather than a rename**: this class shares its exact
  name with the actually-live `DataCleaner` in
  `src/processing/cleaners.py` (imported by
  `src/pipeline/stages/processing/data_handler.py` for
  `remove_outliers_zscore`/`handle_missing_values`) — a future
  contributor or AI edit could easily patch this dead one believing
  they've fixed the pipeline. Added a docstring pointing to the real
  file.

**Also found, not archived (in `src/pipeline/guards/`, discovered while
tracing `temporal_alignment_checker.py`'s live equivalent)**:
`FeatureGuards._initialize_guards()` (`src/pipeline/stages/feature_engineering/guards.py`,
confirmed live via `FeatureEngineeringStage.__init__`) constructs 5
guards but `apply_guards()` only ever invokes one of them
(`temporal_leakage_guard.validate_rolling_windows`) — `timeframe_guard`,
`safe_combiner` (wrapping `TimeframeAlignmentGuard`), `macro_guard`
(`MacroReleaseTimingGuard` — checks that macro data wasn't used before
its real official release time, exactly the point-in-time bug class this
project has been bitten by before), and `temporal_target_guard` are all
constructed and then never invoked anywhere. Confirmed via grep that
none of their real validation methods
(`validate_macro_data_timing`/`combine_features_safe`/
`generate_targets_safe`) are called from anywhere except their own
defining files. This is the same "safety net that looks wired but
isn't" shape as `FeatureLeakageGuard` (fixed this session, commit
`919cda10`, with explicit user sign-off after a dry-run risk check) —
**not fixed yet, needs the same treatment**: present findings, dry-run
against real data, get explicit approval before wiring any of these in,
since enabling a never-battle-tested check in production carries real
risk of blocking legitimate data. `macro_guard` (`MacroReleaseTimingGuard`)
is the highest-priority one to investigate first.

## Known cross-import gotcha

Files moved into `src/archive/` sometimes still import sibling
now-archived modules by their **old**, pre-archival path (e.g.
`from src.utils.data_safety import ...` instead of
`from src.archive.utils.data_safety import ...`). If you restore or touch
anything here, grep the file's own `from src.` imports and check whether
each target still exists at the live path or needs an `src.archive.`
prefix.
