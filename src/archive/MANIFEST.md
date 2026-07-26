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

## Known cross-import gotcha

Files moved into `src/archive/` sometimes still import sibling
now-archived modules by their **old**, pre-archival path (e.g.
`from src.utils.data_safety import ...` instead of
`from src.archive.utils.data_safety import ...`). If you restore or touch
anything here, grep the file's own `from src.` imports and check whether
each target still exists at the live path or needs an `src.archive.`
prefix.
