# Existing Pipeline Automation Map

Based on the provided `src(6).zip` structure.

## What the current pipeline already partially covers

| Automation need | Existing repo area | Current status | Missing governance layer |
|---|---|---|---|
| Daily data/news collection | `src/data/collectors/`, `src/config/collectors.yaml` | Collectors exist for news, macro, sentiment, filings, market data | scheduler/run manifest/source hashes/daily audit |
| News feature generation | `src/pipeline/stages/news/news_manager.py`, `src/features/news_dataset_builder.py` | News clustering and feature dataset generation exist | normalized event packet contract and source lineage |
| News impact tagging | `src/features/news_impact_classifier.py`, `src/config/news_impact_classification.yaml` | Impact type, affected tickers/sectors/timeframes exist | no final probability; add risk archetype/expectation gap/evidence gaps |
| Source quality | `src/features/enrichers/news_quality_enricher.py`, `src/data/quality/` | Quality/freshness/alignment components exist | source-quality ledger, dedupe decisions, provenance hash |
| Causal/event analysis | `src/analytics/context/difference_in_differences_methods.py`, `synthetic_control_methods.py`, `counterfactual_generator.py` | DiD/synthetic control/counterfactual primitives exist | target/secondary/baseline basket policy and outcome review |
| Macro regime | `src/analytics/context/macro_context_analyzer.py`, `src/algorithms/regime/` | Macro scoring/regime components exist | long-run regime snapshots, archetypes, historical priors |
| Leakage protection | `src/pipeline/guards/temporal_leakage_guard.py`, `timeframe_alignment_guard.py`, `features/validation/feature_leakage_guard.py` | Guards exist | daily gate reports and replay-as-of manifests |
| Training | `src/training/`, `src/pipeline/stages/modeling/`, `src/pipeline/hybrid/model_training_orchestrator.py` | Training managers/orchestrators exist | model promotion gate, champion/challenger governance |
| Risk/kill switch | `src/risk/kill_switch/`, `src/risk/risk_manager.py`, `src/risk/max_exposure_monitor.py` | Kill switch/risk stack exists | execution-gateway contract and autonomy escalation policy |
| Trading simulation/execution | `src/pipeline/stages/stage_6_trading_execution.py`, `src/trading/`, `VirtualPortfolio` | Trading stage and virtual portfolio exist | paper/shadow/supervised/constrained autonomous gates |
| Review queue/digest | no clear direct module found by filename | likely not formalized | add daily digest + review queue schemas |
| Unified audit log | logging exists, but no unified daily run audit by filename | partial | add daily run audit and decision lineage |

## Interpretation

The pipeline already does many technical parts. The missing layer is not "more raw pipeline".
The missing layer is:

```text
scheduler + governance + event/hypothesis contracts + review queue + audit + autonomy gates
```

Codex should integrate the kits as wrappers/contracts around existing modules, not replace the pipeline.
