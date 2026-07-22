# Repo-Specific Priority Map After `src(6).zip`

This map is based on the observed `src(6).zip` structure shared by the user.

## Priority A — Existing collectors and data input

Current areas:

- `src/data/collectors/google_news_collector.py`
- `src/data/collectors/newsapi_collector.py`
- `src/data/collectors/rss_collector.py`
- `src/data/collectors/sec_filings_collector.py`
- `src/data/collectors/fred_collector.py`
- `src/data/collectors/fear_greed_collector.py`
- `src/data/collectors/vix_collector.py`

Add/adapt:

- source provenance;
- source hashes;
- collector health manifest;
- daily run manifest;
- freshness/staleness flags.

Related kits:

- Automation Governance Kit v2
- Domain Learning Kit v3
- Data Provenance section in Advanced Governance Kit

## Priority B — News/event processing

Current areas:

- `src/features/news_impact_classifier.py`
- `src/features/enrichers/news_impact_enricher.py`
- `src/features/news_dataset_builder.py`
- `src/pipeline/stages/news/news_manager.py`
- `src/analytics/analyzers/news_impact_analyzer.py`

Add/adapt:

- normalized event packet;
- tag-only analyst output;
- direct/indirect/contextual classification;
- risk archetype tags;
- expectation gap candidate;
- evidence gaps;
- analyst routing.

Related kits:

- Domain Learning Kit v3
- Macro Regime Kit v3
- Operator Review Kit
- Agent Memory Kit

## Priority C — Macro/context/causal layer

Current areas:

- `src/analytics/context/macro_context_analyzer.py`
- `src/analytics/context/market_regime_analyzer.py`
- `src/analytics/context/causal_engine.py`
- `src/analytics/context/difference_in_differences_methods.py`
- `src/analytics/context/synthetic_control_methods.py`
- `src/analytics/context/counterfactual_generator.py`

Add/adapt:

- macro regime snapshots;
- historical indicator crosswalk;
- risk archetype dictionary;
- affected vs baseline sector framework;
- synthetic control for policy impact;
- hypothesis ledger;
- outcome review.

Related kits:

- Macro Regime Kit v3
- Advanced Governance Kit
- Scenario Stress Kit

## Priority D — Guards/eval/audit

Current areas:

- `src/pipeline/guards/temporal_leakage_guard.py`
- `src/pipeline/guards/timeframe_alignment_guard.py`
- `src/features/validation/feature_leakage_guard.py`
- `src/validation/data_leakage_detector.py`
- `src/pipeline/stages/evaluation/`
- `src/monitoring/`

Add/adapt:

- daily audit log;
- source grounding eval;
- unit/period traps;
- decision lineage;
- stress tests;
- incident/postmortem.

Related kits:

- Eval/Audit Kit
- Scenario Stress Kit
- Advanced Governance Kit

## Priority E — Strategy/execution boundary

Current areas:

- `src/trading/`
- `src/pipeline/stages/prediction/`
- `src/pipeline/stages/training/`

Add/adapt:

- strategy playbook registry;
- input contracts;
- promotion gates;
- replay/paper/shadow gates;
- risk engine boundary;
- execution gateway boundary.

Related kits:

- Strategy Library Kit
- Automation Governance Kit v2
- Scenario Stress Kit
- Advanced Governance Kit

## Codex warning

Do not start by modifying live execution.
Start with manifests, event packets, daily governance, review, and eval.
