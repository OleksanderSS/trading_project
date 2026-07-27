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

## Wave 8 — `src/models/` monitoring/health/actions cleanup (2026-07-26)

`src/models/actions/action_trigger.py` (`ActionTrigger`),
`src/models/health/model_health_evaluator.py` (`ModelHealthEvaluator`),
`src/models/statistics/model_statistics.py` (`ModelStatistics`) → moved to
`src/archive/models_dead/actions/`, `.../health/`, `.../statistics/`.
Confirmed via grep: each class's only callers were the already-archived
`src/archive/models_dead/integrated_model_manager.py` and its
never-promoted `dean_os/draft/dean_os_agent_system_v7/` duplicate — zero
live callers, zero test coverage for any of the three. Each was the sole
file in its directory; the now-empty `actions/`, `health/`, `statistics/`
directories were removed.

Also fixed (not archived — these are live, just broken):
- `src/models/analysis/model_health_analyzer.py`'s `ModelHealthAnalyzer`
  called method names that don't exist on any of its sub-components
  (`.analyze()`/`.monitor()` vs. the real `analyze_baseline_dominance()`/
  `analyze_regime_consistency()`/`detect_overfitting()`/
  `monitor_predictions()`), two of which also needed a `model_results`
  dict neither ModelHealthAnalyzer nor its own test ever built. Found
  `src/models/analysis/model_analyzer.py`'s `ModelAnalyzer` already
  implements this correctly (was itself dead — only caller was the
  archived `IntegratedModelManager`). Rewired `ModelHealthAnalyzer` to
  delegate to `ModelAnalyzer` instead of duplicating the logic, which
  fixes both classes' orphan status at once.
- `src/models/analysis/regime_winner_analyzer.py` imported a class named
  `MarketRegimeDetector` from `.regime` that has never existed there
  (real name: `RegimeDetector`) — constructor crashed unconditionally,
  on every instantiation. Zero test coverage existed for this class.
- `src/models/monitoring/prediction_drift_monitor.py`'s
  `PredictionDriftMonitor` read `self.reference_predictions`/
  `self.performance_history`/`self.drift_history`/`self.retraining_history`
  directly, but `__init__` only ever sets `self.history_manager` (a
  `HistoryManager` that actually holds all four) — guaranteed
  `AttributeError` the first time enough samples accumulated for real
  drift detection to run. `self.drift_analyzer` (`DriftAnalyzer`, already
  constructed in `__init__`) already implements the correct modular
  equivalents; rewired the three broken private methods to pull data from
  `self.history_manager` and delegate computation to `self.drift_analyzer`.
  Zero test file existed for this class at all.
- `src/models/__init__.py` still lazy-exported `IntegratedModelManager`/
  `get_integrated_model_manager` from `.integrated_model_manager` — a
  module archived out of `src/models/` earlier this session. Accessing it
  raised `ModuleNotFoundError` instead of a clean `AttributeError` for an
  unknown name. Removed the stale entries.

**Second recon batch (2026-07-26), 22 remaining `src/models/` files
(calibration, dean, adapters, quality, registry, linear, tree, prototypes,
model_selector) — 3 more archived, 1 more live bug fixed, 2 real findings
documented but deliberately not touched:**

- `src/models/adapters/adapters.py` (`LightModelInterface`,
  `HeavyModelInterface`) → `src/archive/models_dead/adapters/adapters.py`.
  Both confirmed broken even on their own terms:
  `LightModelInterface.train()` called
  `self.trainer.train_light_model(features_df=..., model_type=..., ticker=...,
  timeframe=..., target_col=..., task_type=...)`, but the real
  `LightModelTrainer.train_light_model(self, features_df, config)`
  (`src/training/light_model_trainer.py`) only takes a single `config`
  dict — guaranteed `TypeError` on every call.
  `HeavyModelInterface._initialize_manager()` imported
  `from utils.colab_manager import ColabManager`, a path that has never
  existed (no top-level `utils` package; the real class is
  `src/pipeline/hybrid/colab_manager.ColabManager`) — guaranteed
  `ImportError` on every call. Zero external callers, zero test coverage
  for either class.
- `src/models/adapters/unified_model_adapter.py`
  (`UnifiedModelAdapter`) and `src/models/adapters/categorical_helper.py`
  (`handle_categorical_features_split`) → also archived to
  `src/archive/models_dead/adapters/`. Both confirmed zero callers, zero
  tests. `data_preparation.py` and `sentiment_integration.py` in the same
  directory are real, live (imported by
  `src/pipeline/stages/modeling/orchestrator.py` and
  `src/scripts/predictions/models_predict.py` respectively, both with real
  test coverage) — left untouched.
- **Fixed**: `src/models/model_selector/smart_selector.py`'s
  `PerformanceHistorySelector.critique_action()` called
  `_get_historical_reliability(model_name, ticker, target_type)` against a
  signature of `_get_historical_reliability(self, model_name, target_type,
  context)` — `ticker` bound to the `target_type` parameter, so the
  key-matching (`key.endswith(target_type)`, keys are
  `f'{model_name}_{ticker}_{target_type}'`) essentially never matched a
  real key, silently defaulting historical reliability to a neutral `0.5`
  every time regardless of real performance history. Fixed the call to
  pass `(model_name, target_type, context)`, matching both the real
  signature and the ticker-agnostic matching its sibling
  `_get_historical_score(model_name, target_type)` already uses. Only
  reachable via `bootstrap_action_critique()` (see the DEAN Critic finding
  below), so no live blast radius today — fixed anyway since it's cheap
  and a landmine for whenever that path gets wired up.
- **Found, documented only — real but low-priority, self-acknowledged
  incomplete feature**: `src/models/prototypes/registry.py`'s
  `PrototypeRegistry._load_registry()` reconstructs every prototype
  persisted to disk with `model_class=Any` (a literal `typing.Any`, not a
  real class) — the code's own comment admits this ("we need a way to
  resolve model_class from name... In production, we'd use a registry or
  importlib"). Any prototype reloaded from JSON (rather than freshly
  `register()`-ed in the same process) is unclonable — `clone()` raises
  `RuntimeError` at first real use. Not fixed: the whole
  `EnhancedModelFactory`/`PrototypeRegistry` prototype-pattern subsystem
  has zero live callers anywhere in `src/`/`dean_os/` (confirmed via
  grep — only its own test file, `tests/models/prototypes/test_registry.py`,
  which passes today because it never exercises reload-then-clone
  together), and fixing it properly needs a real name→class resolution
  design, not a one-line patch.
- **Found, NOT fixed — needs your input, this is a policy/design
  question, not a quick bug fix**: `src/models/dean/dean_bootstrap_system.py`'s
  `DeanBootstrapSystem.bootstrap_action_critique()` requires at least one
  registered `ACTOR` and one registered `CRITIC` model
  (`self.models`, populated only via `register_model()`); grepping the
  entire live tree, **nothing ever calls `register_model()`** on the real
  singleton returned by `get_dean_system()` (only an unrelated
  `arena.register_model` elsewhere). This is called from a genuinely live
  path — `src/trading/consensus_engine.py`'s `ConsensusEngine._apply_critic_filter()`,
  used by every real `ConsensusEngine.decide()`/`evaluate()` call — but
  the call is wrapped in a broad `except (...) as e:
  logger.warning('DEAN Critic unavailable, skipping filter: {e}');
  critic_score = 0.0`, so this permanently-unconfigured subsystem has
  silently no-op'd on every single trade decision since it was written,
  with only a warning log (not an alert) marking it. `ConsensusReport`'s
  `blocked_by_critic`/`critic_score` fields exist and look load-bearing
  but can never actually block a trade today. Same class of finding as
  the already-noted "safety net that looks wired but isn't" pattern this
  whole audit keeps surfacing, but this one is a genuine risk-management
  gap on a live trading-decision path, not dormant code — worth a
  deliberate conversation about whether real actor/critic models should
  be registered (and if so, which ones), or whether this feature should
  be removed/disabled explicitly rather than silently inert. Zero test
  coverage for `dean_bootstrap_system.py`.

**Still not yet resolved, deferred to a future pass** (documented in
`memory/project_colab_pipeline_audit.md`, not repeated here): the
model-health/drift/overfitting stack's remaining orphaned peripherals
(`PersistentModelPool` disconnected duplicate cache, a dormant second
`ConfidenceCalibrator` concept, orphaned ensembling infra —
`EnsembleComposer`/`DynamicWeightCalculator`/`WeightStabilityMonitor` + 6
files, `ModelCorrelationAnalyzer` facade).

## Wave 9 — `src/ensembling/` cleanup + a real trainer/consumer format mismatch (2026-07-26)

`src/ensembling/ensemble/ensemble_model.py` (`EnsembleModel`) →
`src/archive/models_dead/ensembling/ensemble_model.py`. Broken import
(`src.models.model_interface` doesn't exist; real path is
`src.models.interfaces`), not even exported by its own package's
`__init__.py`, zero callers. The duplicate `EnsembleModel` flagged
earlier this session as "needs triage" — a separate, correctly-tested,
LIVE `EnsembleModel` exists at `src/models/ensemble/ensemble_model.py`
(the one `test_ensemble_correlation_weighting.py` exercises, and whose
`correlation_engine.py` sibling was fixed earlier this session) — this
one was just an orphaned, broken copy under a different package.

`src/ensembling/base_ensemble.py` (`StackedEnsemble`/`EnsembleResult`/
`ensemble_forecast`) → `src/archive/models_dead/ensembling/base_ensemble.py`.
Confirmed superseded duplicate of `src/ensembling/stacked_ensemble.py`'s
same-named classes (which the package's `__init__.py` actually exports) —
older, simpler, `pickle`-based save/load with no path-security check,
vs. the live version's `joblib` + `resolve_trusted_artifact_path`. Also
had a broken import of its own
(`ExperienceDiaryEngine` — renamed to `DiaryEngine` long ago, no longer
exists under the old name).

**Real bug found and fixed, not just archived**: `src/scripts/modeling/train_consensus_model.py`
— whose own docstring says it trains "the meta-model... used by the
real-time ConsensusEngine" — imported `StackedEnsemble` from the stale
`base_ensemble.py` above. `ConsensusEngine` (`src/trading/consensus_engine.py`,
confirmed live) loads its meta-model via
`stacked_ensemble.StackedEnsemble.load()`, which expects a very different
on-disk format (joblib state dict) than `base_ensemble.py`'s
`pickle.dump(self, f)`. So even setting aside the broken
`ExperienceDiaryEngine` import (which made the trainer crash before it
could run at all), whatever this script produced would never have loaded
correctly in the real consumer — a genuine trainer/consumer format
mismatch. No live blast radius today only because
`data/trained_models/consensus_meta_model.pkl` doesn't exist yet
(`ConsensusEngine` gracefully falls back to live-adaptive ensembling when
the file is absent), but the training path itself had never worked.
Fixed by redirecting the import to `src.ensembling.stacked_ensemble`,
matching the real consumer.

Same stale-import pattern (`ExperienceDiaryEngine`/`base_ensemble`) also
fixed in `src/scripts/experiments/compare_layers.py` for consistency, but
**that script remains non-functional for unrelated reasons, documented
here rather than chased further** (out of scope for an ensembling pass):
it also imports a `devtools.experimentation.base` module that doesn't
exist, its own `ensemble_forecast(...)` call unpacks the 5-field
`EnsembleResult` NamedTuple into 2 variables (`ensemble_result, stats =
ensemble_forecast(...)`, which would raise `ValueError` even with a
correct import), and it calls a `DiaryEngine.add_entry()` method that
doesn't exist on the real class. This script needs a proper rewrite, not
a spot-fix — worth a dedicated look in a future `scripts/`/`devtools`
audit pass.

`src/ensembling/ensemble/archive/adaptive_ensemble.py` → moved to
`src/archive/models_dead/ensembling/adaptive_ensemble.py` (was already
informally set aside in a folder literally named `archive/` inside live
`src/`, with a stale header comment from a pre-restructure path
`core/pipeline/adaptive_ensemble.py` and a broken `from utils.logger`
import — moved to the project's actual archival convention for
consistency). Zero live callers; the real, live equivalent is
`src/trading/live_adaptive_ensemble.py`'s `LiveAdaptiveEnsemble`.

`src/ensembling/caching.py` (`PredictionCache`/`EnsembleResultCache`) —
read in full, confirmed live (`src/pipeline/stages/prediction/orchestrator.py`),
no bugs found.

Verified: `tests/ -k "ensemble or consensus"` → 27 passed, zero
regressions.

## Wave 10 — `src/pipeline/` core sweep begins: spine + `stages/modeling/` (2026-07-26)

User explicitly asked to finish `src/pipeline/` (the ~117-file execution
core) before moving to peripheral directories (scripts/devtools/cli/etc.)
— those call into the pipeline, they aren't part of its execution.

**Spine pass** (top-level orchestration + stage 0-7 entry points + the 5
guard classes in `guards/`) — 3 real bugs found and fixed in the guard
classes (all currently dormant/orphaned, but would fire the moment any
gets wired in): `TemporalTargetGuard` called a `.calculate()` method that
doesn't exist on `ClassificationCalculator`; `TimeframeAlignmentGuard`
re-read a `'datetime'` column from the wrong (untransformed) DataFrame,
`KeyError` on any DatetimeIndex-only frame; `MacroReleaseTimingGuard`
mixed `.iterrows()` index labels with `.iloc` positional indexing. No
bugs found in the 8 top-level orchestration files or the 8 stage entry
points — all correctly delegate to their real implementations.

**`stages/modeling/` pass** — archived `orchestration.py`, `training.py`,
`io.py`, `metrics.py`, `dataclasses.py` (all to
`src/archive/dead_pipeline_code/modeling/`): a fully self-contained dead
island. `orchestration.py`/`training.py` call `stage._get_context_fingerprint()`/
`_log_training_debug_info()`/`_create_champion_info()`/`_log_to_diary()`/
`_get_light_model_training_data()`/`_determine_task_type()`/
`_get_light_model_types()` on `ModelingStage`
(`src/pipeline/stages/modeling/orchestrator.py`) — none of which exist
there (`ModelingStage` was rewritten around `_process_ticker_with_async`
and never adopted them). `io.py`/`metrics.py`/`dataclasses.py` are
imported only by this dead chain. Zero callers, zero tests for any of
the 5 files. The working, tested implementations of the same operations
already exist as free functions in `utils.py` — kept live (real test
coverage in `tests/unit/test_modeling_utils.py`) but its docstring
falsely claimed `ModelingStage` calls them; corrected to say it doesn't.

Also fixed a live bug: `walk_forward_validation.py`'s
`_get_target_horizon_rows()` (computes the purge gap that prevents label
leakage at the train/validation boundary) swallowed any lookup exception
via a bare `except Exception: pass`, silently falling back to
`purge=1` with zero logging — for any real target with `shift` < -1 (e.g.
`target_up_5d`, `shift: -5`), a silent failure here means an
under-purged, leaky fold boundary with no trace. Narrowed the exception
types and added a warning log. Verified real targets with `shift=-5`
correctly resolve to `horizon=5`.

`pipeline_control_artifacts.py` (modeling) was double-checked and
confirmed genuinely live (used directly by `ModelingStage.orchestrator.py`,
`walk_forward_validation.py`, `evaluation/orchestrator.py`, `dean_os`
pipeline-control chain, multiple tests) — NOT part of the dead
`training.py` island despite `training.py` importing it too.

Verified: `tests/ -k "guard"` → 12 passed; `tests/unit/test_stage3_data_contracts.py`
→ 10 passed; `tests/ -k "walk_forward"` → 14 passed. Zero regressions
across all of Wave 10 so far.

**`stages/evaluation/` pass (2026-07-26, commits `d1344a2a`, `68312ed2`,
`3587af62`, `c343acae`, `dbba3cee`, `7b5ba095`) — 5 real bugs fixed, 1
file archived.** This is Stage 7, the final stage that decides whether a
model/strategy actually worked — extra scrutiny paid off:
- **Most severe**: `AdvancedBacktestEngine.run_comprehensive_backtest()`
  (`src/backtesting/advanced/advanced_engine.py`) computes a genuine
  simulated equity curve internally but never returned it — the only
  consumer, `BacktestAnalyzer._normalize_backtest_results()`, detected
  the "missing" curve and fabricated a fake straight-line substitute via
  `np.linspace(initial_capital, final_value, n)`, discarding the real
  daily-return path entirely. **Every evaluation report, notification,
  equity plot, and dean_os pipeline-control artifact was computed from
  this fabricated line** — any profitable run reported `max_drawdown ≈ 0`
  regardless of real volatility. Fixed by exposing the real equity curve
  in the engine's own return; verified a synthetic up-crash-recover path
  (real max_drawdown ~47%) now correctly propagates instead of showing
  ~0.
- Backtests silently fall back to `np.random.normal`-generated fake data
  whenever real pivoted input has fewer than 2 rows (reachable — the
  entry gate doesn't check row count), with only a log line marking it —
  nothing in the results/summary/artifacts recorded the metrics came from
  fabricated data. Added an `is_simulated_data` flag threaded through to
  `final_summary` with a loud warning.
- `_run_stress_testing()`'s `market_crash` scenario checked
  `'max_drawdown_pct'`, but the real metrics calculator only ever
  produces `'max_drawdown'` (a fraction) — this scenario has silently
  never run in production (gated behind a config flag that's `False`
  everywhere today, so no current blast radius, but a real bug for
  whenever it's enabled). Fixed the key + unit conversion.
- `EvaluationMetricsCalculator`'s optional `PortfolioMetricsCalculator`
  branch passed a one-column DataFrame where a Series is required —
  `validate_input()` silently returns `{}` instead of raising. Dormant
  today (that branch is never constructed in production) but would
  silently zero out all financial metrics the moment anyone wires in a
  real calculator — which is the entire reason that constructor param
  exists.
- `analytics.py`'s benchmark-return calculation (`_build_data_map`) had
  no per-ticker grouping before `.pct_change()` — a bogus spike at every
  ticker boundary in a multi-ticker frame. Dormant-but-tested
  (`orchestrator.py` has its own separate, already-correct partitioned
  implementation and doesn't call this module) — fixed anyway, cheap and
  matches this audit's most-repeated bug class.
- Archived `data_recovery.py` (confirmed zero callers/tests, correct but
  never wired implementation).
- Also confirmed (no bug, no action): `pipeline_control_artifacts.py`
  (evaluation) is genuinely live, no new issues beyond the
  already-documented deliberate candidate/locked provenance gap.
  `backtest_adapter.py`/`reporting.py` are dormant-but-tested (real test
  coverage, just not called by the live `orchestrator.py`, which uses
  `BacktestAnalyzer`/`ReportGenerator` instead) — left alone per the
  standing "orphaned but tested" rule.
- Verified: `tests/ -k "stage7 or evaluation or backtest_analyzer or
  metrics_calculator or portfolio_metrics"` → 48 passed, zero
  regressions.

**`stages/prediction/` pass (2026-07-26, commits `7da457eb`,
`651c5f0c`) — 2 real bugs fixed (silent wrong-prediction class, the most
dangerous kind for a live prediction stage), 1 dead file archived.**
- **`ModelResolver`'s fallback model-loading path
  (`_try_load_model_from_path`, used whenever Colab-sourced model
  metadata has no `model_path`, a normal case) computed its `ModelPool`
  cache key by stripping `context_id` from the filename stem — for any
  file matching the standard `model_{ticker}_{target}_{model_type}`
  naming convention, this collapses to the literal string `"model"` for
  every ticker/target/model_type.** `ModelPool` is a single
  process-wide cache; a cache hit never re-invokes the loader. The first
  ticker resolved through this path caches its model under `"model"`;
  every subsequent ticker hits that same cache entry and **silently
  receives the first ticker's model instance instead of its own** — a
  silent wrong-model, wrong-prediction bug, not a crash. The sibling
  direct-load path already keyed correctly (full `path.stem`), confirming
  this was never intentional. Fixed to match.
- `PredictionGenerator.generate_ensemble_prediction()`'s `context_params`
  built `'ticker'` from a DataFrame column that had already been dropped
  as metadata upstream — always resolved to the `'unknown'` default (and
  a `.get()` on a DataFrame returns a Series, not a scalar, a second
  latent defect on the same line) — and never included a `'tf'` key at
  all. `StackedEnsemble._predict_stacked` (the default ensemble method)
  builds its live-performance-weighting context fingerprint from exactly
  these two fields — with both broken, **live per-ticker performance
  weights and dynamic-router adjustments collapsed onto one shared bucket
  and leaked across every ticker being predicted**. Fixed by threading
  the real ticker/timeframe (already available in the caller's `meta`
  dict) down as explicit parameters instead of trying to recover them
  from data that had already been stripped.
- Archived `data_preparer.py` (`PredictionDataPreparer`) — a duplicate of
  the live, already-fixed `DataPreparationService`, but still containing
  the OLD, unfixed bug the sibling class's own docstring narrates
  (zero-filling missing technical-indicator values for a live prediction
  row instead of dropping it — feeding the model a fabricated,
  indistinguishable-from-real value). Confirmed zero callers anywhere,
  not even re-exported by the package's `__init__.py`, zero test
  coverage.
- Also noted, not archived (instantiated but unused, not itself a bug):
  `PredictionContextManager` is constructed in `orchestrator.py` but none
  of its methods are ever called again anywhere.
- Verified: `tests/ -k "stage5 or prediction_generator or prediction_stage
  or model_pool or model_resolver or select_champions"` → 31 passed,
  zero regressions.

**`stages/processing/` (Stage 2) + `stages/trading/` (Stage 6) pass
(2026-07-27, commits `532bead9`, `5dcd106b`, `62db25ca`, `9ec3fb73`) — 4
real bugs fixed, including one genuine data-corruption bug.**
- **Data corruption**: `ProcessingStorage._save_persistent_macro_snapshot()`
  wrote each run's incremental FRED delta to a fixed "persistent" parquet
  path via a pure overwrite, with no read-merge-write step — every Stage
  2 cycle with new FRED data silently destroyed all previously
  accumulated macro history, leaving only that cycle's few new rows.
  Downstream consumers (`cli/pipeline_data_loader.py`,
  `cli/pipeline_executor.py`) read this file as the full historical
  fallback. Fixed with a proper read-merge-dedupe-write (newest row wins
  per `series_id`+`datetime`, so revisions still apply). Verified two
  sequential incremental writes now correctly accumulate instead of the
  second wiping the first.
- `ProcessingStage._process_all_data_types()` never copied
  `raw_data['reddit_sentiment']` into `cleaned_data_map` — real data
  collected by `RedditSentimentCollector` in Stage 1 vanished every
  cycle with zero logging; the downstream filter already has a dedicated
  `reddit_sentiment` branch, it just never received anything. Fixed.
- `ProcessingValidator.run_system_validation()`/`create_quality_metrics()`
  were non-functional stubs — the validator's own comment admitted it:
  *"In the original code, this likely calls self.validator.validate(...).
  For now, we provide the structure to hold this logic."* Every run
  logged/reported that validation passed and quality was perfect
  (`data_consistency_score: 1.0`) regardless of real data state. Wired in
  the real (pure, non-blocking) `UnifiedValidator.validate_cleaned_data()`
  call and computed real row/missing-value/consistency numbers (walking
  nested dicts like `{'prices': {'1d': df, '1h': df}}`). No downstream
  code branches on these values today, so this was giving false
  assurance rather than causing a wrong decision — but matches this
  audit's "safety net that looks wired but isn't" pattern.
- `TradingExecutionStage._find_latest_batch_name()` (Stage 6) only
  globbed for `test_ticker_*` dirs, missing the `'main_database'`
  default-batch check that the sibling (otherwise orphaned)
  `TradingDataIO` already has correctly. Any invocation without an
  explicit `batch_name` (e.g. CLI runs) would silently fail to find real
  Stage 5 output even when it exists, returning `'no_predictions'`
  instead. Fixed to match.
- Also confirmed (status, not a new bug): `TradingRecommendationEngine`
  is entirely orphaned from the live Stage 6 flow — matches
  `dean_os/current_architecture_map.py`'s own
  `can_generate_recommendations_now: False` flags, consistent with the
  already-documented DEAN-Critic-inert finding (review-only system state
  today, not an accidental wiring bug).
- Verified: `tests/ -k "stage2 or stage6 or processing or trading_stage or
  macro_point_in_time"` → 39 passed, zero regressions.

**`stages/feature_engineering/` (remaining files, minus `guards.py`
already investigated) + `stages/utils/` pass (2026-07-27, commit
`f257af73`) — notably clean batch, no leakage/lookahead bugs found.**
Reconfirmed this is the pipeline's most point-in-time-sensitive stage and
gave it maximum scrutiny: `TargetOrchestrator` groups by
`['ticker', 'interval']` and sorts chronologically before any calculator
runs; `BackwardTimeframeContextAssembler` does a proper
`merge_asof(direction='backward')` on a bar-close availability timestamp
with an explicit future-violation check that raises; train-only holdout
logic groups by ticker. No cross-ticker leakage, no lookahead bias found
in `enricher.py`/`orchestrator.py`/`targets.py`/`timeframe_context.py`.
- **Found, documented only — real but needs deliberate design work, not
  a quick fix**: `FeatureEnricher.__init__` constructs a real
  `FeatureCache` (`get_feature_cache()`, real disk I/O — creates
  `data/cache/features`, deletes >7-day-old files on every
  instantiation) advertised as giving "60-80% speedup for repeated
  enrichments," but `self.feature_cache` is never read again anywhere —
  `enrich_features()` calls `self.orchestrator.run(...)` directly, and
  `FeatureOrchestrator` has zero references to the cache. Every Stage 3
  run recomputes every feature from scratch regardless of ticker/date
  repetition — the optimization is completely disconnected, while disk
  activity gives the false impression it's working. Not fixed: the
  cache's real API (`get_features(ticker, date, config_hash)`/
  `save_features(...)`) operates per single ticker+date, but
  `enrich_features(df, timeframe, **kwargs)` takes a whole
  (possibly multi-ticker, multi-date) DataFrame — wiring this in
  correctly requires understanding the real batch shape Stage 3 runs
  against and a real cache-key design, not a one-line connection. Doing
  it wrong risks a worse bug (silently serving stale features) than the
  current "just slow" state.
- Archived `src/pipeline/stages/utils/collection_manager.py`
  (`CollectionManager`) and `data_schema_mapper.py`
  (`DataSchemaMapper`) — confirmed zero callers anywhere; the directory
  had no `__init__.py` at all, meaning it was never even a real
  importable package. Superseded by `CollectionStage`'s own independent
  collector-dispatch logic after a prior refactor.
- Verified: `tests/ -k "stage3 or feature_engineering_stage or
  timeframe_context or timeframe_lineage or target_orchestrator"` → 42
  passed, zero regressions.

**Final `src/pipeline/` single-file stage dirs pass (2026-07-27, commits
`2479709a`, `f145a997`) — `stages/collection/` (Stage 1, the very first
stage everything else depends on) had 2 severe, live bugs; 3 more dead
files archived.**
- **Most severe finding of the whole `src/pipeline/` sweep**:
  `CollectionStage._normalize_data()` called
  `collector.generate_hash(row.to_dict())`, but `BaseCollector` defines
  no such method at all, and every subclass is inconsistent — some
  define a public `generate_hash`, most define a private `_generate_hash`
  instead, one defines neither. For every collector matching the
  private/absent case, this raised `AttributeError` on every run,
  silently caught by a broad `except` in `process_and_save_results` — the
  collector's HTTP fetch succeeds and logs "Received N records" (looks
  like success), but the data never reaches `_save_collector_data`.
  **Cross-checked against the live `src/config/collectors.yaml`: `cftc`,
  `fear_greed`, `put_call_ratio`, and `economic_calendar` are all
  `enabled: true` — 4 currently-enabled Stage 1 collectors were silently
  discarding every record on every run, with data never reaching their DB
  tables.** Every collector that does define a hash method uses
  byte-for-byte the same formula
  (`sha256('|'.join(row.get(k,'') for k in hash_keys))`) — fixed by
  computing that directly in `_normalize_data` using
  `collector.get_hash_keys()` (already correctly read from config),
  instead of depending on any particular per-collector method name.
  Verified byte-identical output against `CFTCCollector`'s own
  `_generate_hash` for the same input.
- Same file: `_run_collector()` caught every `TimeoutError`/`Exception`
  from a collector's `run()` and returned `None` — `process_and_save_results`
  treats `None` identically to "ran fine, nothing new," so a full
  collector crash or timeout was indistinguishable from a benign empty
  result. `process_and_save_results` already had a dedicated
  `isinstance(res, Exception)` branch for real failures, but it was dead
  code as long as `_run_collector` never let an exception reach
  `asyncio.gather(..., return_exceptions=True)`. Fixed by re-raising
  instead of swallowing — now correctly flows into the already-existing
  handling. No test coverage existed for `CollectionStage` at all,
  confirming why both bugs went unnoticed. Verified both fixes directly
  (hash formula match, and a simulated crash now correctly propagates as
  an `Exception` through `gather` instead of becoming `None`).
- Archived 3 more confirmed-dead, zero-test single-file stage modules:
  `stages/news/news_manager.py`, `stages/features/orchestrator_manager.py`,
  `stages/cache/feature_cache_manager.py`.
- **Incident during this archival, self-corrected**: `git mv` silently
  failed for `feature_cache_manager.py` because it was never git-tracked
  in the first place — `.gitignore` has a blanket `cache/` pattern (only
  `src/core/cache/` is excepted), which had been silently excluding this
  real source file from version control the whole time. The subsequent
  `rm -rf` on the parent directory deleted it with no git history to
  recover from. Recovered by finding and diff-confirming an identical
  copy under `.archive_docs/draft/dean_os_agent_system_v7/`, then
  force-adding (`git add -f`) the restored file so it's now actually
  tracked. Confirmed no other `.py` source files in `src/` are currently
  hidden by this same gitignore pattern. **The `cache/` gitignore pattern
  itself is still overly broad and untouched — worth a deliberate fix in
  a future session** (either narrow it to actual cache-data directories,
  or add more exceptions like the existing `!src/core/cache/` one).
- Verified: `tests/ -k "stage1 or collection_stage or collector"` → 15
  passed, zero regressions.

**`src/pipeline/` core sweep (Wave 10) is now complete for the "spine +
all stages/" scope** — every `stages/<name>/` subdirectory and the
top-level orchestration files have been read and fixed. Only `hybrid/`
(36 files, partially covered by the original Colab pipeline audit at the
top of this document) remains before `src/pipeline/` can be called fully
closed out.

**`src/pipeline/hybrid/` pass complete — `src/pipeline/` core sweep now
fully closed out.** Recon subagent read all 36 files. Key structural
finding: roughly half this directory (~15 of ~20 components built by
`OrchestratorComponentFactory.initialize_components()`) is constructed and
attached to the orchestrator via `setattr`, but never actually called by
any live code path — `HybridOrchestrator`'s own public API only touches
`pipeline_runner`, `pipeline_manager`, `colab_manager`, and
`light_models_trainer`. Confirmed via grep (`orchestrator.<name>.method(`
across `src/pipeline`, `src/cli`, `src/main`) and zero test coverage for
any of the ~15 (one apparent test hit, `test_pipeline_executor.py`, is a
false positive — it tests the unrelated `src.cli.pipeline_executor`,
which merely shares a class name with `src.pipeline.hybrid.pipeline_executor`).
Real, confirmed bugs fixed:
- `colab_manager.py`'s `_load_single_file`: when a file's JSON wraps its
  payload in a top-level `models_metadata` key (the real shape
  `colab_results.json` always uses — confirmed against
  `scripts/colab/colab_clean_cell.py`'s own writer), the load
  **overwrote** `results['models_metadata']` instead of merging into it,
  unlike the merge path used for unwrapped data. `trained_models_metadata.json`
  isn't written by any real script today so this never fires in practice
  yet, but it's a live method (`load_colab_results` is called from
  `src/cli/pipeline_executor.py`) and a latent trap the moment a second
  wrapped-shape file is ever written alongside `colab_results.json`.
  Fixed by unwrapping first, then always merging through the same path.
  Added a regression test (`test_load_colab_results_merges_wrapped_models_metadata_instead_of_overwriting`
  in `tests/unit/test_hybrid_feature_target_safety.py`) since this method
  had zero prior coverage.
- `selected_features_processor.py`: called
  `self.feature_selection_validator._create_mock_selected_features_for_test(...)`
  (leading underscore) but the real method on `FeatureSelectionValidator`
  has no underscore — guaranteed `AttributeError` the moment this dormant
  path is ever exercised. Fixed the call site.
- `model_training_orchestrator.py`'s `_prepare_training_data` read
  `context_data.get('features', [])`, but the only real producer
  (`context_builder.py`'s `_create_context_data`) writes the key
  `selected_features` — `available_features` was always `[]`, so
  `train_models_for_contexts` silently trained 0 models regardless of
  input. Fixed the key. (The same function's `timeframe` read is a
  separate, deeper gap — `context_builder.py` never captures a timeframe
  at all — left as-is rather than inventing new plumbing for a
  zero-caller method; noted for whoever eventually wires this class in.)
- Archived 4 confirmed fully-dead files (zero references anywhere,
  including instantiation — not just "unused once built" like the ~15
  above): `hybrid_dataclasses.py`, `storage_helpers.py`,
  `data_components_context.py` (its only caller, `DataComponentsContext`,
  is itself never instantiated; also independently broken —
  `HybridDataManager(config_manager)` vs. the real
  `HybridDataManager.__init__(self, output_dir)`), `feature_loader.py`
  (`FeatureLoader.__init__` never sets `self.logger`, so its own except
  blocks raise `AttributeError` masking the real error — moot, since
  nothing instantiates it).
- **Deliberately NOT fixed / archived, deferred for a user decision**:
  the ~15-component dormant cluster itself (`cache_manager.py`,
  `orchestrator_interface.py`, `feature_selection_manager.py`,
  `feature_selection_validator.py` (only reachable via the now-fixed but
  still-dormant chain above), `test_mode_manager.py`, `context_builder.py`,
  `data_manager.py`/`HybridDataManager`, `data_processor.py`,
  `data_utils.py`, `data_batch_manager.py`, `pipeline_metadata_manager.py`,
  `pipeline_executor.py`, `colab_workflow_manager.py`,
  `model_training_orchestrator.py`, `selected_features_processor.py`).
  Each duplicates responsibility already handled by a working live
  component (e.g. `colab_workflow_manager.py` is superseded by inline
  logic already in `pipeline_manager.py`; `pipeline_executor.py`'s own
  stage methods are literally `# Implementation would go here` stubs,
  fully superseded by the real, working `pipeline_runner.py`). Asked the
  user via AskUserQuestion whether to archive the whole cluster, fix
  bugs cheaply and leave wired, or just document — no response given, so
  took the lower-risk, reversible path (fix the 3 confirmed contract bugs
  above in place, leave the files wired but otherwise untouched) rather
  than the bigger, harder-to-reverse factory rewrite. Still an open
  question for a future session.
- Also verified, not a bug: `component_factory.py` builds
  `components['pipeline_runner']` without passing the shared
  `components['db_data_manager']` (unlike `light_models_trainer`, which
  does get it), so `PipelineOrchestrator` falls back to constructing its
  own separate `DataManager` instance for stages 0-3. Checked
  `DataManager.__init__`: its only real shared state
  (`_connections`, the DuckDB connection cache) is a classvar keyed by
  `db_path`, not per-instance, so a second instance still resolves to the
  same underlying connection. Confirmed harmless — redundant object
  construction, not a correctness bug.
- Verified: `tests/ -k "hybrid or colab_manager or selected_features or
  model_training_orchestrator or component_factory or pipeline_runner"`
  → 17 passed (plus the new regression test, 10/10 in its file), zero
  regressions. One unrelated pre-existing failure
  (`test_run_hybrid_pipeline_help_if_available`, a subprocess-timeout
  smoke test) confirmed via `git stash` to fail identically on the
  pre-session baseline.

**Peripheral `src/` sweep begun (2026-07-27, commits `5c137c43`, `ec9c08e2`,
`d27994bd`, `bed1874e`, `63769761`) — first batch: `src/patterns/`,
`src/sentiment/`, `src/factories/`, `src/integrations/`, `src/simulation/`,
`src/dashboard/` (11 files).** Recon subagent read all 11 files in full.
Several real, live bugs found — this batch was unusually severe:
- **CRITICAL, LIVE**: `src/dashboard/main_app.py`'s `get_data_from_db()`
  called `_db_manager.load_data(query)` — `DataManager` has no such
  method, only `fetch_df(query)`. Every query-backed tab (header metrics,
  overview, trading signals, news analysis, risk management) raised
  `AttributeError`; only System Monitoring and World State survived. This
  is the actual live dashboard entry point (`.claude/launch.json` runs
  `streamlit run src/dashboard/main_app.py`), so 4 of 6 tabs were broken
  in the real, current launch path. Fixed the method name.
- **CRITICAL, LIVE**: `src/pipeline/stages/prediction/orchestrator.py`'s
  `_process_single_context` did `if news_data:` where `news_data` is
  `pd.DataFrame | None` throughout its real producer chain
  (`cli/pipeline_executor.py` → `final_stages_orchestrator.py` →
  prediction `orchestrator.py`) — raises `ValueError: truth value of a
  DataFrame is ambiguous` whenever real news data is present (the normal
  case). Caught by a broad `except`, so it silently failed the *entire*
  prediction for that context, not just the optional NLP step. Even past
  that, `adjust_predictions_with_patterns` (`src/patterns/pattern_recognition_adjustment.py`)
  expects `list[dict]` and iterates `news_item.get(...)` — a raw
  DataFrame would iterate column-name strings instead of rows. Fixed both:
  proper `is not None and not .empty` check, and `.to_dict('records')`
  before passing through.
- **HIGH, LIVE**: `src/factories/model_factory.py`'s
  `_extract_model_params` only ever extracted `n_neighbors` for KNN —
  every other model type (LSTM, GRU, CNN, Transformer, TabNet, MLP,
  Autoencoder, SVM, Linear, and recursively every non-tree member of an
  Ensemble) had its whole `per_model` config dict silently discarded
  before construction, training with constructor defaults regardless of
  what's tuned in `models.per_model.<type>`. `TreeModelFactory` already
  does this correctly (`{**(config or {}), **kwargs}`) — matched that
  pattern.
- **MEDIUM, LIVE**: `ModelRegistry.MODELS` listed `'dean_ensemble'`
  (class: `DeanEnsemble`) and `'sentiment'` (class: `SentimentModel`) —
  neither class exists anywhere in the live codebase (confirmed via
  repo-wide grep; the only near-match, `SentimentModelIntegrator`, is a
  different class). `ModelFactory.get_available_models()` returns these
  verbatim, so every "train all available models" fallback run
  (`DEFAULT_ENABLED_MODEL_TYPES`) was guaranteed to attempt and fail on
  both, every time — not fatal (caught, skipped per-type) but permanently
  wasted. Removed both entries rather than speculatively building
  never-implemented classes; no live caller requests either name
  explicitly.
- **Fixed, dormant**: `src/dashboard/dashboard_data_bridge.py` — all 5
  date-range queries used SQLite-dialect `datetime('now', '-N days')`;
  `DataManager`'s real backend is DuckDB, which doesn't implement that
  function (`Catalog Error`, verified directly). This bridge is dormant
  (unit-tested only via a `FakeDataManager` stub, not wired into
  `main_app.py` yet — the project's own prior audit notes recommend
  wiring it in as the actual fix for the `main_app.py` bug above), so
  invisible to its own tests, but every real query would fail the moment
  it's connected. Fixed to `CURRENT_TIMESTAMP - INTERVAL N DAY`.
- **Archived, confirmed dead**: `src/integrations/infra/github_actions.py`
  (`GitHubActionsClient`) — zero real callers anywhere (repo-wide grep),
  already independently flagged as an orphan in this project's own
  `diagnostic_reports/orphan_modules.txt`/`dead_code_classification.csv`.
  Moved to `src/archive/integrations/infra/github_actions.py`.
- Read clean, no bugs found: `src/sentiment/sentiment_models.py`,
  `src/factories/tree_model_factory.py`,
  `src/integrations/data/bigquery_client.py`,
  `src/simulation/simulation_engine.py`, `src/simulation/__init__.py`,
  `src/dashboard/__init__.py`.
- Verified: `tests/ -k "model_factory or model_registry or dashboard or
  pattern_recognition or prediction_orchestrator or stage5 or
  dashboard_data_bridge or tree_model_factory"` → 25 passed, zero
  regressions. One pre-existing unrelated failure
  (`test_model_factory_import_does_not_top_level_import_neural_models`,
  already documented earlier in this file) confirmed unaffected.

**Peripheral `src/` sweep, second batch (2026-07-27, commit `a715b741`
+ this doc) — `src/validation/`, `src/cli/`, `src/metrics/`,
`src/devtools/` (22 files).** Recon subagent read all 22 files. Another
severe batch:
- **CRITICAL, LIVE, two stacked bugs**: `TimeSeriesValidator.validate_time_gaps()`
  called a nonexistent `self.calendar.get_trading_days(start=, end=)` —
  `TradingCalendar`'s real API only has `.trading_days` (a pre-generated
  `DatetimeIndex` attribute, meant to be sliced). Every call raised
  `AttributeError`. Even past that, the caller
  (`UnifiedValidator._check_time_continuity`, run on every pipeline
  execution's Stage 2 processing validation) read
  `gaps.get('has_gaps')`/`gaps.get('gap_count')` — keys the function
  never produces (real keys: `is_valid`/`missing_points_count`) — so once
  the first bug is fixed, the check still silently no-ops regardless of
  real gaps. Zero test coverage existed for either function. Fixed both,
  verified end-to-end with a real dropped-trading-day repro, added 4
  regression tests (`tests/unit/test_time_series_gap_detection.py`).
- **CRITICAL, LIVE, documented not fixed**: `run_hybrid_pipeline.py --mode
  calibrate` calls `PipelineExecutor.execute_calibrate_mode()`, which
  **does not exist anywhere in the codebase** — not defined, not
  aliased. `calibrate` is a fully advertised CLI mode (in
  `argument_parser.py`'s choices, documented in the script's own
  docstring, has its own `--n-trials` flag), but has never worked even
  once. The only related code (`src/devtools/experimentation/run_hyperparameter_tuning.py`)
  is a synthetic-data demo script (`make_regression`), not a real
  calibration pipeline wired to actual tickers/models. Building a correct
  `execute_calibrate_mode` (load real data, run BayesianOptimizer per
  model type, write best params into `models.per_model.<type>` — which
  `ModelFactory` can now actually consume, per this session's earlier
  fix) is a genuine feature build, not a quick contract fix. Asked the
  user via AskUserQuestion (document-only vs. build a real
  implementation vs. remove the mode) — no response given, so documented
  only per the lower-risk default. **Still broken, needs a decision.**
- **Documented, not fixed — needs real design work, not a typo**:
  `src/meta_learning/evolution/dual_loops.py:143`'s
  `run_hypothesis_generation()` (the default/normal meta-learning update
  path) calls `self.rule_generator.generate_rules_from_context(simulated_losing_trades)`
  — `ContextRuleGenerator` (`src/devtools/rule_generator.py`) has no such
  method, only `run_analysis()`/private helpers operating on a
  completely different data shape (config-driven indicator/threshold
  scanning, not raw trade records). The caller's own comments
  self-admit this is a "temporary compatibility layer" and "might need a
  refactor of ContextRuleGenerator to accept vulnerability data
  directly" — i.e. already known incomplete by whoever wrote it. Left
  unfixed rather than guessing at the real business logic (how to derive
  rule `conditions`/`action`/`description` from simulated trade
  records) — this needs a deliberate design session, not a mechanical
  fix.
- **Archived, confirmed dead** (zero test coverage, zero real callers
  anywhere, confirmed via repo-wide grep):
  `src/devtools/task_manager.py` (also independently broken — imports a
  `Logger` class that doesn't exist in `src.core.logging.logger`, only
  `ProjectLogger`/`ContextAdapter` do — moot since nothing ever imported
  it to trigger the `ImportError`); `src/devtools/system_validator.py`
  (`SystemValidator`); `src/cli/pipeline_data_loader.py`
  (`PipelineDataLoader` — its functionality was independently
  re-implemented inline inside `pipeline_executor.py`'s own
  `_safe_load_parquet`/`_try_load_parquet`/etc., leaving this module an
  orphaned duplicate that nobody deleted after the inlining).
- **Noted, not touched** (per standing "clearly-marked prototypes stay as
  they are" convention): `src/devtools/prototypes/live_trading_ticker_manager.py`
  self-labels as a non-functional prototype in its own header comment,
  every method deliberately raises `NotImplementedError`.
  `src/devtools/experimentation/run_hyperparameter_tuning.py` is an
  intentional, documented demo/template script (guarded by
  `if __name__ == "__main__"`), not abandoned code. Two small dead
  helpers left alone inside otherwise-live files (low value, higher risk
  of unrelated diff than benefit): `pipeline_schemas.py::create_validation_middleware`,
  `pipeline_executor.py::_merge_results_data`.
- Read clean, no bugs found: `src/validation/__init__.py`,
  `src/validation/data_leakage_detector.py`,
  `src/validation/pipeline_schemas.py` (aside from the one dead function
  above), `src/cli/argument_parser.py`, `src/cli/batch_manager.py`,
  `src/metrics/base.py`, `src/metrics/calculator.py`,
  `src/metrics/model/ml_evaluator.py`,
  `src/metrics/financial/portfolio_metrics.py`,
  `src/metrics/financial/financial_metrics_library.py`,
  `src/metrics/utils/calculation_tools.py`, `src/devtools/__init__.py`,
  `src/devtools/rule_generator.py` (bug is on the caller side, not here).
- Verified: `tests/ -k "stage2 or processing_stage or time_series or
  walk_forward or cross_val or purged"` → 38 passed, zero regressions.

**Peripheral `src/` sweep, third batch (2026-07-27, commits `9b73bf28`,
`d1561047`, `05d21251` + this doc) — `src/utils/` (12 files),
`src/monitoring/` (13 files, top-level).** Recon subagent read all 25
files, cross-verified via direct execution/introspection, not just
reading. Another severe batch:
- **Fixed, LIVE**: `health_hub.py` (instantiated in the real pipeline,
  `check_system_health()` runs on every use) — `extract_features_from_metrics`
  read `metrics['system']['disk']['percent']`, but the real producer
  (`ResourceMonitor.get_health_status()`) nests disk as a sibling
  top-level key (`metrics['disk']['usage']['percent']`) — the disk
  feature fed into the live ML risk-prediction models was always `0.0`
  regardless of real usage (verified live: real 14.0% → extracted 0.0).
  Also fixed `_load_performance_data` (used by `check_model_drift`),
  which called two nonexistent `DataManager` methods
  (`load_data`/`query_data` — real method: `fetch_df`), so financial
  drift detection always failed. Zero prior test coverage; added 3
  regression tests.
- **Fixed, dormant-but-tested**: `ml_analytics.py`'s `check_model_drift`
  had the identical nonexistent-`DataManager`-method bug — its own unit
  test's hand-rolled stub only implemented the buggy method name
  (`load_data`), matching the bug instead of the real class, which is
  exactly why it went undetected. Updated the stub to `fetch_df`.
  Same file's `extract_features_from_metrics` called
  `datetime.now().dayofweek` — a pandas `Timestamp` attribute, not on
  stdlib `datetime` — `AttributeError` on every call, silently caught,
  always returning `[0.0]*17`. Fixed to `.weekday()`.
- **Fixed, dormant/dead, cheap**: `monitoring/config.py`'s
  `_parse_env_value` referenced a module-level `logger` that only exists
  inside the file's own `__main__` guard — `NameError` on any
  non-numeric env var. `data_freshness_monitor.py` imported
  `UniversalNotifier` from a module that doesn't exist
  (`src.utils.universal_notifier`; real path:
  `src.core.logging.notifier`) — not even caught by the surrounding
  except tuple. `reporting/performance_reports.py`'s
  `ComprehensiveReporter._check_system_status` parsed a flat
  `"45.2%"`-style string format `ResourceMonitor` never produces (real
  shape is nested floats) — CPU/memory alerts could never fire. All
  three fixed.
- **Architectural finding, NOT fixed — same shape as the time-gap-
  detection bug from the last batch**: `DataFreshnessMonitor` and
  `FeatureDriftMonitor` are both constructed live inside the real
  feature-engineering pipeline (`enhanced_smart_selector.py`, wired via
  `pipeline/stages/feature_engineering/orchestrator.py`), but neither
  object's check methods are ever actually called after construction —
  the monitoring scaffolding runs on every real pipeline execution and
  silently does nothing. Fixing this properly means deciding where in
  the pipeline these checks should actually fire, which is a design
  decision, not a mechanical fix — noted alongside the other
  constructed-but-never-invoked findings for the eventual project-level
  review.
- **Noted, not fixed (low priority, currently unreachable)**:
  `path_safety.py`'s hardened `validate_path()`/`safe_join()` functions
  are never actually called by any of their 5 real call sites (all use
  the trivial path getters instead) — and `validate_path()`'s own
  containment check (`str(path).startswith(str(root))`) has no
  path-separator boundary check, so a sibling directory like
  `data_leak/` would incorrectly pass validation against an allowed
  root `data/`. Contrast with `artifact_security.py`'s `_is_within()`,
  which does this correctly via `Path.relative_to()`. Unreachable today
  since nothing calls the buggy function, but worth fixing if this ever
  gets wired in for real path-traversal protection.
- **Archived, confirmed dead** (zero callers anywhere, including tests):
  `src/utils/checkpoint_manager.py` (name collides with an unrelated,
  live `CheckpointParams` in `src/colab/config/training_config.py`),
  `src/utils/json_utils.py`, `src/utils/math_utils.py` (name collides
  with two other, unrelated, actually-used modules of similar name —
  `src/utils/math_safe.py` and `src/core/utils/math_utils.py` — real
  risk of importing the wrong one), `src/monitoring/base.py`
  (`BaseMonitor` — a second, orphaned, incompatible class of the same
  name; `monitoring/__init__.py` actually re-exports the real
  `BaseMonitor` from `monitoring_system.py`), `src/monitoring/drift_detector.py`,
  `src/monitoring/performance_monitor.py`.
- Read clean, no bugs found: `src/utils/__init__.py`,
  `src/utils/artifact_security.py`, `src/utils/dynamic_module_loader.py`,
  `src/utils/math_safe.py`, `src/utils/path_utils.py`,
  `src/utils/rate_limiter.py`, `src/utils/trading_calendar.py`,
  `src/monitoring/__init__.py`, `src/monitoring/dashboard.py`,
  `src/monitoring/monitoring_system.py`,
  `src/monitoring/feature_drift_monitor.py` (class itself is correct —
  the bug is nothing calls it, see architectural finding above),
  `src/monitoring/infrastructure/resource_monitor.py` (this is the
  ground-truth schema both `health_hub.py` and `performance_reports.py`
  got wrong).
- Verified: `tests/ -k "health_hub or ml_analytics or monitoring or
  resource_monitor or performance_reports or data_freshness or utils"`
  → 21 passed, zero regressions.

**Peripheral `src/` sweep, fourth batch (2026-07-27, commits `69c2858c`,
`38a1a8e6`, `c7505a0b`, `195ee17c` + this doc) — `src/main/` (13 files),
`src/processing/` (13 files, top-level).** Recon subagent read all 26
files. Major structural finding, bigger than a routine dead-file
cleanup: **`SystemOrchestrator` (`src/main/system_orchestrator.py`) — the
"Central Control Center" the module's own docs/README call "the primary
hub"/"Production Ready" — has zero live callers anywhere** (confirmed via
repo-wide grep: only its own file, `modes/intelligent.py` — itself
zero-callers — and an already-quarantined verify script reference it).
No root-level script (110 checked) touches it; the real production path
goes through `run_hybrid_pipeline.py` → `HybridOrchestrator` →
`PipelineRunner`/`PipelineManager` → `PipelineOrchestrator` entirely
separately. Zero test coverage for `SystemOrchestrator`, `TrainMode`,
`PredictMode`, `BacktestMode`, `IntelligentMode`, or `WebUIMode`. **Not
archived this pass** (unlike smaller confirmed-dead utilities) — this
is architecturally significant enough, and different enough from the
docs' own claims about it, that it's flagged for the user's holistic
project review rather than silently archived. Two real bugs live
entirely inside this orphaned dispatch path, documented but not fixed
since they're moot if the whole thing gets archived:
`MonsterTestMode.run(ticker=...)` (singular) vs. `SystemOrchestrator`
calling it with `tickers=[...]` (plural) — silently absorbed into
`**kwargs`, always defaults to SPY; and `_run_intelligent_mode`'s DEAN
self-diagnosis check guards on `dean_brain.experience_diary`, an
attribute that's never set anywhere, so the "retraining" feature is a
permanent no-op, plus the `brain` object is never actually wired into
`PipelineOrchestrator.self.brain` despite being passed through several
layers. Separately, 3 Mode classes DO have real, live, standalone
entry points bypassing `SystemOrchestrator` entirely: `MonsterTestMode`
(`run_monster_test.py`, calls `.run()` with no args — the ticker/tickers
bug never manifests via this real path), `ShadowBattleMode`
(`run_shadow_battle.py`), `HistoricalEventReplayMode`
(`run_historical_replay.py`).
- **Fixed, LIVE** (via `run_historical_replay.py`):
  `historical_replay.py` had 2 silent except blocks (a bare
  `except: pass` loading candidate models, and
  `except Exception as e: pass` around the whole per-model prediction
  loop) discarding real errors with zero logging — the only symptom was
  a generic "no successful predictions" warning with no way to diagnose
  why. Added logging to both.
- **Fixed, LIVE** (via `run_shadow_battle.py`): `shadow_battle.py`
  constructed `SimulationEngine`/`SimulationContext` that were never
  used again — the real scenario data comes directly from
  `SyntheticGenerator`, bypassing the simulation framework entirely.
  Removed the dead construction and now-unused imports.
- **Fixed, LIVE** (via `pipeline/stages/processing/data_handler.py`,
  called on every ingested price dataframe): `price_preprocessor.py`'s
  `_finalize_dataframe` unconditionally indexed
  `processed_df['datetime']` after a fallback path that can return a
  DataFrame without that column — an unguarded, confusing `KeyError`
  deep inside `pd.to_datetime`. Added an explicit, diagnosable check.
- **Archived, confirmed dead** (zero callers anywhere, zero test
  coverage): 6 standalone functions from `processing/cleaners.py`
  (`harmonize_dataframe`, `safe_fill`, `sanitize_dataframe_timezone`,
  `normalize_to_unified_schema`, `merge_and_deduplicate`,
  `filter_by_terms`, plus private helpers — leftovers from an abandoned
  "unified schema" effort; moved to
  `src/archive/processing/cleaners_unused_functions.py`, removed 2
  now-unused imports from the live file); `src/processing/parallel_processor.py`
  (`ParallelProcessor`, whole file, zero callers/tests).
- **Documented, not fixed — low priority, currently unreachable**:
  `scripts/verify_backtesting.py` calls `mode._run_portfolio_simulation(...)`,
  a method that no longer exists on `BacktestMode` (removed during a
  refactor that moved simulation to Stage 7 `EvaluationStage`) — a
  broken manual verification script, not production code.
  `main/modes/web_ui.py`'s two catch-then-raise blocks are moot since
  `WebUIMode` is itself dead code (see above).
- Read clean, no bugs found: `src/main/__init__.py`,
  `src/main/modes/__init__.py`, `src/main/modes/base.py`,
  `src/main/modes/train.py`/`predict.py`/`backtest.py` (aside from the
  orphaned-dispatcher context above),
  `src/main/modes/training_data_pipeline.py`, `src/processing/__init__.py`,
  `src/processing/data_filter.py`, `src/processing/deduplication_utils.py`,
  `src/processing/normalization_manager.py`,
  `src/processing/filters/*` (orchestrator, price_filter, news_filter,
  social_filter, pattern_extractor — the last one intentionally stubbed
  and honestly labeled).
- Verified: `tests/ -k "historical_replay or shadow_battle or cleaners
  or data_cleaner or price_preprocessor or hybrid_cleaners or
  parallel_processor"` → 10 passed, zero regressions.

**SystemOrchestrator archival, user-confirmed (2026-07-27, commit
`792f4793`):** following the finding documented in the batch above, the
user confirmed they run the pipeline via `run_hybrid_pipeline.py` — so
`system_orchestrator.py`, `modes/intelligent.py` (a Streamlit dashboard
launcher, zero callers outside the dead dispatcher), and `modes/web_ui.py`
(an alternate HTTP dashboard, likewise zero callers) were archived to
`src/archive/main/`. `TrainMode`/`PredictMode`/`BacktestMode` were left
in place — they share the same live `BaseMode` framework as 3 confirmed-
live standalone modes (`MonsterTestMode`/`ShadowBattleMode`/
`HistoricalEventReplayMode`, each with its own real `run_*.py` entry
point), so archiving them is a separate, not-yet-confirmed decision.
Verified: `tests/ -k "main or orchestrator"` → 247 passed, zero
regressions.

**Peripheral `src/` sweep, fifth batch (2026-07-27, commits `7ced90e8`,
`abead8ce`, `09a174a0` + this doc) — `src/meta_learning/` (19 files),
`src/colab/` (20 files).** Recon subagent read all 39 files.
- **Fixed, LIVE**: `meta_learning/memory/diary_engine.py`'s
  `record_decision()` (runs on every real trading decision, called from
  `pipeline/stages/trading/orchestrator.py`) computed a stable UUID
  `decision_id` on `DecisionRecord` (per its own comment: "Stable UUID
  string instead of random 31-bit int") and even had the DB schema
  migrated from INTEGER to VARCHAR to support it — but the actual insert
  path never used it, still generating a fresh
  `uuid.uuid4().int & 0x7FFFFFFF` truncated int on every call. Net
  effect: `decision_id` was dead weight, and the real primary key used a
  collision-prone value not covered by the upsert's de-dup key. Fixed to
  use `decision.decision_id` directly. Added a regression test (zero
  prior coverage of this method).
- **Fixed, dormant, cheap**: `evolution/dual_loops.py`'s `get_state()`
  crashed (`IndexError`) on a fresh arena with zero battles fought, since
  `TradingModelArena.get_leaderboard()` always returns the `'leaderboard'`
  key even when its value is `[]` — the `.get(..., [{}])` default never
  triggers when the key IS present. Fixed to handle the empty-list case
  explicitly. (Separate from the already-documented
  `generate_rules_from_context` finding in this same file from an
  earlier batch — not re-flagged.)
- **Major structural finding, NOT archived — needs your confirmation,
  different shape from `SystemOrchestrator`**: `src/colab/` (all 20
  files) has zero real callers anywhere via grep, and the package was
  even completely unimportable (`ImportError`) due to
  `utils/__init__.py` importing `retry_on_timeout` from the wrong
  sibling module. However, `src/colab/README.md` explicitly documents
  this as **intentional** — the package is meant to be uploaded and run
  *inside* a Google Colab notebook, not imported by the local pipeline,
  so local-repo grep can't observe whether it's actually used that way.
  What grep *can* confirm: the real Colab-side script that runs today,
  `scripts/colab/colab_clean_cell.py`, does **not** import `src.colab` —
  it reimplements its own `MemoryMonitor` and config-loading logic from
  scratch instead. Also found: `environment/setup.py` is a line-for-line
  duplicate of `environment/colab_environment.py` (never imported by the
  package's own `__init__.py`); `models/sklearn_fallback.py` and
  `models/torch_models.py` duplicate logic already inlined in
  `models/model_factory.py`; `model_factory.py` itself has 4 dead stub
  functions (`pass`-only, superseded by inline lambdas) plus a second,
  unused, duplicate `_create_autoencoder_model`. Fixed the 2 confirmed
  bugs anyway (the `ImportError` and a `self.logger`-doesn't-exist bug in
  `config_loader.py`, matching the earlier `monitoring/config.py`
  pattern) since they're cheap and real regardless of whether the
  package is actively used — but did NOT archive the package, unlike
  every other confirmed-dead find this session, since the manual-Colab-
  upload usage pattern genuinely can't be ruled out from this repo alone.
- **Other confirmed-dead, no bugs in the code itself** (documented, not
  touched — lower priority than the two structural findings above):
  `AgentPermissionSystem` (zero callers anywhere), `SecurityConstraintEngine`/
  `ConstraintValidators` (only caller, `DeanBootstrapSystem`, is itself
  self-documented as "needs further work... not yet integrated into the
  main workflow", and even there `validate_action()` is constructed but
  never invoked — same "orphaned construction" shape as the earlier
  `shadow_battle.py` fix, but fully inert here since the whole chain is
  dead), a second, unrelated, fully-dead `BayesianOptimizer` class in
  `evolution/optimization/` (distinct from the live one used by
  `src/scripts/optimization/`).
- Verified: `tests/ -k "diary or dual_loops or learning_loops or colab
  or meta_learning"` → 51 passed, zero regressions.

**Peripheral `src/` sweep, sixth batch (2026-07-27, commits `0fdfb308`,
`a72a8b53` + this doc) — `src/scripts/` (22 files, part of `src/` —
distinct from the root-level `scripts/` folder, which is already
covered by the original Colab pipeline audit).** Recon subagent read all
22 files. Another batch with severe, currently-broken live tools:
- **Fixed, LIVE**: `monitoring/run_health_check.py` (documented in
  `src/scripts/README.md` as "a key tool for ensuring stability") passed
  a `UnifiedConfigManager` instance into `ModelResultsManager.__init__`,
  which expects a `base_path: str` — crashed with `TypeError` before
  producing any report. Fixed to use the default path (matching the
  other 2 real callers). Verified end-to-end: now runs and produces a
  real health report.
- **Fixed, dormant**: `modeling/train_consensus_model.py` (trains the
  meta-model used by the real-time `ConsensusEngine`, per its own
  docstring — already touched once in the earlier `src/ensembling/`
  pass for a different bug) called 2 nonexistent `DataManager` methods
  (`get_all_tables()`/`load_data()` — real: `get_all_table_names()`/
  `fetch_df()`) plus a `finally: data_manager.close()` with no matching
  instance method, which would have overridden even the graceful
  empty-dataframe fallback with a fresh `AttributeError` on every call.
  Fixed all 3, verified end-to-end against the real database.
- **Archived, confirmed dead+broken, superseded by a working root-level
  equivalent**: `analysis/generate_context_rules.py` — imports 2
  nonexistent module paths (`src.core.analysis.rule_generator`,
  `src.core.data.data_manager`); the real, working version lives at
  root `scripts/core/generate_context_rules.py` with correct imports.
- **Archived, confirmed dead+broken**: `config/ticker_config_updater.py`
  — imports a nonexistent `config.tickers` module, plus an independent
  path-depth bug (`project_root = current_dir.parent` resolves to
  `src/scripts`, not the repo root) and a reference to a nonexistent
  `collectors/collectors_config.json` (real config:
  `src/config/collectors.yaml`, different format entirely).
- **Documented, NOT fixed — needs a genuine rewrite, not a mechanical
  fix, same class of deferral as `compare_layers.py` from the
  `src/ensembling/` pass**: `data/auto_accumulator.py` has multiple
  deeply stacked bugs: (1) `from src.data.collector_factory import
  create_all_collectors` — wrong module path (real:
  `src.data.collectors.collector_factory`, plural) AND the function
  itself doesn't exist there anymore — the real API is a
  `CollectorFactory(configs, http_client_factory).get_all_collectors()`
  class+method, not a standalone function; (2)
  `AssetUniverseManager(config_manager.get_config('asset_universe', {}))`
  double-extracts the config (the class itself already does
  `config.get('asset_universe', {})` internally), and no config
  anywhere defines an `asset_universe` key or a `'day_trading_tech'`
  preset at all — `self.presets` is always `{}`, crashing on
  `.get_preset(...).tickers` since `.get_preset()` returns `None`; (3)
  `db_manager.get_all_tables()` doesn't exist (real:
  `get_all_table_names()`). Its own dedicated test
  (`tests/scripts/data/test_auto_accumulator.py`) can't even collect
  (`ModuleNotFoundError` from the same broken import) and, even setting
  that aside, mocks a class (`AutoAccumulator`) and CLI arguments
  (`--group`, `--hours`) that don't match the real file's actual API
  (`AutoAccumulatorGuard`, `--mode once|cycle`) at all — the test and
  the source have diverged completely. The near-identical root-level
  copy `scripts/core/auto_accumulator.py` has the same broken import.
  Needs real design work (redesign the collector-instantiation call
  site, fix the asset-universe config wiring, reconcile or rewrite the
  test) — left undocumented-but-broken rather than risk an incorrect
  partial patch.
- Read clean, no bugs found: `colab/auto_colab_sync.py`,
  `debug/data_merge_debugger.py`, `fix/data_fixer.py`,
  `monitoring/run_dashboard.py`, `optimization/base.py`,
  `optimization/__init__.py`, `optimization/factory.py`,
  `optimization/dynamic_config_updater.py`,
  `optimization/hyperparameters/bayesian.py`,
  `optimization/hyperparameter_searcher.py`,
  `optimization/portfolio/optimizer.py`, `predictions/deep_predict.py`,
  `predictions/models_predict.py`, `simulation/shadow_arena.py`
  (cosmetic duplicate-import pyflakes hits only),
  `test_modular_pipeline.py`. `predictions/prediction_utils.py` is an
  intentionally-emptied stub with zero real importers, inert.
  `experiments/compare_layers.py` reconfirmed still broken exactly as
  previously documented — not re-investigated in depth, already deferred
  for a full rewrite.

## Known cross-import gotcha

Files moved into `src/archive/` sometimes still import sibling
now-archived modules by their **old**, pre-archival path (e.g.
`from src.utils.data_safety import ...` instead of
`from src.archive.utils.data_safety import ...`). If you restore or touch
anything here, grep the file's own `from src.` imports and check whether
each target still exists at the live path or needs an `src.archive.`
prefix.
