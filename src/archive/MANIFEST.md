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
- `data_sources/local_file_data_source.py` was already archived in
  Wave 1; this session only fixed its own internal cross-import
  (`__init__.py` was still importing from the pre-archival `src.data_sources.*`
  path) and updated `src/config/data_sources.yaml`'s module path to match
  (confirmed nothing in production dynamically reads that config entry
  today — inert drift, not a live crash, but fixed to keep the config
  honest).

## Known cross-import gotcha

Files moved into `src/archive/` sometimes still import sibling
now-archived modules by their **old**, pre-archival path (e.g.
`from src.utils.data_safety import ...` instead of
`from src.archive.utils.data_safety import ...`). If you restore or touch
anything here, grep the file's own `from src.` imports and check whether
each target still exists at the live path or needs an `src.archive.`
prefix.
