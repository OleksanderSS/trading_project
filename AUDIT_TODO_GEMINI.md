# Audit TODO (For Gemini/Windsurf)

This file is a handoff queue for non-critical refactors, cleanups, and larger changes.

## False Positives (Audit Tool)

- Imports inside functions/conditional blocks: The audit tool may falsely report these as unused. Verify locally before removing.
    - Example: `src/colab/models/model_factory.py` (numpy import inside `forward` method).

## P2 (Refactor / Tech Debt)

- `audit_tool_coverage.py`: script times out during stage enumeration; consider adding CLI flags like `--only-analytics`, `--skip-stages`, `--max-stages N`, and/or lazy stage init to make it usable as a quick report.

## P1/P2 Mechanical Fixes

Generated from `audit_13_after_complex.json` after complex fixes. These are suitable for Gemini/Windsurf: mostly `pct_change(fill_method=None)` and silent exception handling.

- `src/algorithms/risk_parity_allocator.py:356` - exception handler contains only `pass`.
- `src/analytics/calculators/macro_score_calculator.py:90` - `pct_change()` needs explicit `fill_method`.
- `src/colab/config/config_loader.py:50` - exception handler contains only `pass`.
- `src/data/collectors/insider_collector.py:144` - exception handler contains only `pass`.
- `src/data/collectors/vix_collector.py:24` - exception handler contains only `pass`.
- `src/features/colab_context_integration.py:126` - exception handler contains only `pass`.
- `src/features/feature_selector.py:67`, `:77`, `:196` - exception handlers contain only `pass`.
- `src/models/adapters/unified_model_adapter.py:189` - exception handler contains only `pass`.
- `src/monitoring/config.py:78` - exception handler contains only `pass`.
- `src/monitoring/example_usage.py:176` - exception handler contains only `pass`.
- `src/pipeline/stages/evaluation/metrics_calculator.py:77` - `pct_change()` needs explicit `fill_method`.
- `src/pipeline/stages/stage_0_data_generation.py:100`, `:103`, `:104`, `:105` - `pct_change()` needs explicit `fill_method`.
- `src/pipeline/stages/trading/recommendation_engine.py:167`, `:199` - `pct_change()` needs explicit `fill_method`.
- `src/pipeline/stages/trading/recommendation_engine.py:419` - exception handler contains only `pass`.
- `src/risk/analyzers/correlation_analyzer.py:29`, `:31` - `pct_change()` needs explicit `fill_method`.
- `src/risk/elite_risk_metrics.py:386` - `pct_change()` needs explicit `fill_method`.
- `src/risk/kill_switch/calculator.py:116`, `:143`, `:174` - `pct_change()` needs explicit `fill_method`.
- `src/risk/max_exposure_monitor.py:55` - `pct_change()` needs explicit `fill_method`.
- `src/risk/metrics.py:36`, `:110`, `:153` - `pct_change()` needs explicit `fill_method`.
- `src/training/pattern_aware_training.py:108`, `:110` - `pct_change()` needs explicit `fill_method`.
- `src/training/portfolio_optimizer.py:76` - `pct_change()` needs explicit `fill_method`.
- `src/devtools/rule_generator.py:151` - `except Exception:` needs binding/logging or a clearer fallback.
