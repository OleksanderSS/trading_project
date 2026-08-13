# DEAN-OS Command Checklist

Use these commands to gather useful logs and review state without running the heavy trading pipeline.

> **Which file is authoritative.** This checklist explains *why* and *when* to run
> a workflow, and what its boundaries are. It is hand-written, so it can go stale.
> For *what exists and what options it takes*, use `dean_os/COMMAND_INDEX.md` —
> that file is generated from the `run_agent_*.py` wrappers themselves and cannot
> drift. Regenerate it with `python run_agent_command_index.py`.
>
> On 2026-08-13 this checklist advertised 192 commands, 93 of which did not exist.
> Commands still named here but absent from disk are recorded, with reasons, in
> `dean_os/config/retired_commands.yaml`; `tests/dean_os/test_agent_cli_restore.py`
> fails if a command is named here without either existing or being recorded there.

## World Model Event Learning Packet (2026-07-09)

Use this path when news/materials should become review-only hypotheses and
replay tasks. This is not paper trading and not a prediction signal.

Focused check:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/dean_os/test_world_model_event_learning_packet.py -q -p no:cacheprovider --basetemp D:\trading_project\.pytest_tmp_world_model_event
```

Expected: `3 passed`.

Wider lens/evidence check:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/dean_os/test_world_model_event_learning_packet.py tests/dean_os/test_analyst_core_phase2_lenses.py tests/dean_os/test_analyst_core_schemas.py tests/dean_os/test_domain_data_feeder.py tests/dean_os/test_saved_semiconductor_news_evidence_producer.py -q -p no:cacheprovider --basetemp D:\trading_project\.pytest_tmp_world_model_event_wider
```

Expected: `77 passed`.

Build from the current saved semiconductor news artifact:

```powershell
python run_agent_world_model_event_learning_packet.py --news-artifact reports/dean_os/saved_semiconductor_news_evidence_producer/latest.json --domain-id semiconductor_ai_infrastructure --output-dir reports/dean_os/world_model_event_learning_packet_current
```

Boundary: `WorldModelEventLearningPacket` creates event classifications,
historical analog candidates, falsifiable hypotheses, scenario branches,
evidence gaps, and replay tasks. It must not write learning memory, promote
models, write config, register outcomes, or trade without later review gates.

## External Materials Evidence Path (2026-07-09)

Books, notes, templates, JSON stats, PDFs, DOCX, and user-fed ideas should enter
domain analysis as `ResearchDocument` evidence through the shared material
loader. Do not bypass quarantine/provenance with direct file reads.

Run the focused source/evidence checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/dean_os/test_domain_data_feeder.py tests/dean_os/test_context_evidence_point_in_time.py tests/dean_os/test_material_quarantine_and_financial_nlp.py -q -p no:cacheprovider --basetemp D:\trading_project\.pytest_tmp_domain_feeder
```

Expected: `22 passed`.

Run the wider integration smoke:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests/dean_os/test_parallel_scaffold_safety.py tests/dean_os/test_domain_analyst_agent.py tests/dean_os/test_orchestrator_integration.py tests/dean_os/test_current_architecture_map.py tests/dean_os/test_pipeline_stage23_regeneration.py tests/dean_os/test_saved_semiconductor_news_evidence_producer.py tests/dean_os/test_domain_data_feeder.py -q -p no:cacheprovider --basetemp D:\trading_project\.pytest_tmp_domain_feeder_wider
```

Expected: `30 passed`.

Important boundary: `title` is not a stable news locator. News rows need URL,
link, URI, source id, document id, id, or hash style locator to pass the
point-in-time source audit.

## Parallel Scaffold Safety Boundary (2026-07-09)

The July 7-8 "15/19 agents online" orchestrator runs are smoke evidence, not
the canonical default activation policy. Before enabling new agents, verify the
registry state and enable only the bounded path being reviewed.

Current corrected registry state: 37 registered / 16 enabled. Standalone domain
analysts, the composite `pipeline_manager`, model/tuning/paper/stateful agents,
and new experimental analyzers remain default-off. `NewsEventAnalyzerAgent`
must not write `OutcomeTracker` cases unless `register_outcomes: true` is set
for that explicit run.

Check the registry without running the heavy pipeline:

```powershell
python -c "import yaml; d=yaml.safe_load(open('dean_os/config/agent_registry.yaml', encoding='utf-8'))['agents']; print(len(d), sum(1 for c in d.values() if c.get('enabled'))); print([n for n,c in d.items() if c.get('enabled')])"
```

## Orchestrator CLI (2026-07-06)

Run all 9 enabled agents with preloaded prices:

```powershell
python run_agent_orchestrator.py --as-of 2026-06-30T21:00:00+00:00 --soft-mode --preload-prices latest --ticker NVDA --ticker AMD
```

Use latest 1d prices instead of 15m:
```powershell
python run_agent_orchestrator.py --as-of 2026-06-30T21:00:00+00:00 --soft-mode --preload-prices latest-1d --ticker NVDA
```

Run with real HybridPipelineAdapter (requires production pipeline data):
```powershell
python run_agent_orchestrator.py --as-of 2026-06-30T21:00:00+00:00 --pipeline-mode local --batch-name main_database --preload-prices latest --ticker NVDA
```

Expected state: 9 agents load (pipeline_audit, data_quality, risk, regime, context_synthesis, semiconductor_analyst, macro_policy, geopolitical, news_catalyst). semiconductor_analyst produces full thesis with evidence_count=152, 5 lanes, stance=mixed, recommendation=partial_ready_for_review.

`--as-of` must match runtime artifact cutoff (currently `2026-06-30T21:00:00+00:00`). Mismatch causes runtime artifact rejection for semiconductor_analyst.

## Composite Domain Agent And Timeframe Gate

Audit the current feature artifact before any Stage 4/5 reuse:

```powershell
python run_agent_pipeline_feature_timeframe_audit.py data\colab\accumulated\main_database\features.parquet --stage5-json data\colab\accumulated\main_database\stage_5_results.json --ticker AMD --ticker INTC --ticker NVDA --ticker TSM --output-dir reports\dean_os\pipeline_feature_timeframe_audit_current
```

Expected current state: `pipeline_feature_timeframe_audit_blocked_mismatch`,
4/4 tickers declare `1d` but have observed `15m` cadence, 0/4 have
timezone-aware datetimes, and Stage 5 feature parentage is unproven.

Regenerate the bounded `15m` Stage 2/3 batch without collectors or models:

```powershell
python run_agent_pipeline_stage23_regeneration.py data\colab\accumulated\main_database\main_database_stage1_raw_data_20260629_195400.parquet --ticker AMD --ticker INTC --ticker NVDA --ticker TSM --timeframe 15m --max-rows-per-ticker 300 --batch-dir data\colab\regenerated\semiconductor_15m_stage23_current --output-dir reports\dean_os\pipeline_stage23_regeneration_current
```

Expected state: `stage23_regeneration_review_ready`, 1,200 selected source
rows, 1,170 accepted Stage 2/3 rows, UTC `15m`, and no Stage 4/5 action. The
legacy `60m` and `1d` partitions are rejected because their observed cadence is
intraday; they are not relabeled.

The broader pipeline still keeps three separate timeframe lanes:
`15m` is the current exact repaired path, while `60m` and `1d` remain
their own lineage-backed lanes and must not be collapsed into the `15m`
rebuild just to save time.

Audit exact target semantics and feature/target hash bindings:

```powershell
python run_agent_pipeline_target_readiness_audit.py data\colab\regenerated\semiconductor_15m_stage23_current\targets.parquet --features-parquet data\colab\regenerated\semiconductor_15m_stage23_current\features.parquet --batch-metadata-json data\colab\regenerated\semiconductor_15m_stage23_current\batch_metadata.json --ticker AMD --ticker INTC --ticker NVDA --ticker TSM --timeframe 15m --output-dir reports\dean_os\pipeline_target_readiness_audit_current
```

Expected state: `pipeline_target_readiness_ready`, `7/7` applicable targets
ready, exact feature/target SHA matches, and `can_trade=false`. Daily indicator
targets must not appear in the intraday target table.

Build the larger exact NVDA shard only when a three-fold Stage 4 review is
needed. This avoids recomputing all four sector tickers:

```powershell
python run_agent_pipeline_stage23_regeneration.py data\colab\accumulated\main_database\main_database_stage1_raw_data_20260629_195400.parquet --ticker NVDA --timeframe 15m --max-rows-per-ticker 600 --batch-dir data\colab\regenerated\nvda_15m_stage23_review600 --output-dir reports\dean_os\pipeline_stage23_regeneration_nvda_review600
python run_agent_pipeline_target_readiness_audit.py data\colab\regenerated\nvda_15m_stage23_review600\targets.parquet --features-parquet data\colab\regenerated\nvda_15m_stage23_review600\features.parquet --batch-metadata-json data\colab\regenerated\nvda_15m_stage23_review600\batch_metadata.json --ticker NVDA --timeframe 15m --output-dir reports\dean_os\pipeline_target_readiness_audit_nvda_review600
```

Run one development-only exact Stage 4 review:

```powershell
python run_agent_pipeline_stage4_exact_context_review.py --features-parquet data\colab\regenerated\nvda_15m_stage23_review600\features.parquet --targets-parquet data\colab\regenerated\nvda_15m_stage23_review600\targets.parquet --batch-metadata-json data\colab\regenerated\nvda_15m_stage23_review600\batch_metadata.json --feature-audit-json reports\dean_os\pipeline_stage23_regeneration_nvda_review600\feature_timeframe_audit\latest.json --target-audit-json reports\dean_os\pipeline_target_readiness_audit_nvda_review600\latest.json --ticker NVDA --timeframe 15m --target-name target_intraday_up_15m --min-train-rows 160 --validation-rows 50 --step-rows 50 --purge-rows 5 --max-folds 3 --max-features 30 --output-dir reports\dean_os\pipeline_stage4_exact_context_review_nvda_15m_review600
```

Expected current state:
`walk_forward_candidate_blocked_by_validation_contract`, three folds,
balanced accuracy `0.567852`, feature stability `0.706589`, and failed checks
for train-validation gap, positive-rate stability, and majority baseline.
No test partition is read, no model is saved, and Stage 5 must remain blocked.

Run the resource-efficient composite analyst over saved review artifacts:

```powershell
python run_agent_composite_domain_pipeline.py --domain-id semiconductor_ai_infrastructure --as-of 2026-06-30T21:00:00+00:00 --ticker NVDA --ticker AMD --ticker INTC --ticker TSM --timeframe 15m --horizon-days 180 --runtime-json reports\dean_os\semiconductor_analyst_runtime_current\latest.json --feature-timeframe-audit-json reports\dean_os\pipeline_stage23_regeneration_nvda_review600\feature_timeframe_audit\latest.json --target-readiness-json reports\dean_os\pipeline_target_readiness_audit_nvda_review600\latest.json --stage4-review-json reports\dean_os\pipeline_stage4_exact_context_review_nvda_15m_review600\latest.json --prediction-review-json reports\dean_os\pipeline_prediction_source_review_current\latest.json --sector-to-ticker-review-json reports\dean_os\sector_to_ticker_review_packet_current\latest.json --output-dir reports\dean_os\sector_pipeline_manager_semiconductor_current
```

Expected current state: 152 hash-verified sector evidence items, five lens
deltas, `caution`, and `pipeline_readiness_blocked`, including
`stage4_validation_contract_failed`. The report must preserve
`decision_influence=false`, `can_create_ticker_forecast=false`, and
`can_trade=false`.

The composite manager is the canonical full path. The standalone
`DomainAnalystAgent` is only an alternative for an already-populated
`MarketContext`. Never enable both with the same `execution_group` and
overlapping `run_phases`; registry validation must reject that configuration.

The canonical registry intentionally keeps both alternatives disabled. For a
bounded run, enable exactly one path and supply its verified artifacts. Do not
enable model review, tuning, chief review, paper portfolio, diary, source
routing, or operations agents merely to make the registry look complete.
Standalone runtime loading must match the requested timezone-aware `as_of`
exactly; a later wall-clock date is not a valid substitute for the artifact's
point-in-time cutoff.

Refresh the architecture and agent observability artifacts:

```powershell
python run_agent_current_architecture_map.py --output-dir reports\dean_os\current_architecture_map_current
python run_agent_capability_matrix.py --output-dir reports\dean_os\agent_capability_matrix_current
```

Do not proceed from legacy `main_database/features.parquet`, and do not run
Stage 5 from the current exact candidate. The next pipeline build is
source-hash-bound Stage 3 ticker-shard caching plus genuinely new forward
development observations; do not tune variants against the same three folds.

## Working Semiconductor Analyst Runtime

Build the review-only sector market fragment from the existing repaired price
artifact:

```powershell
python run_agent_saved_sector_market_evidence.py reports\dean_os\pipeline_control_saved_price_repair_current\latest.json --as-of 2026-07-01T00:00:00+03:00 --sector-tickers NVDA AMD INTC TSM --benchmark QQQ --output-dir reports\dean_os\saved_sector_market_evidence_producer_current
```

Build strict saved-news candidates:

```powershell
python run_agent_saved_semiconductor_news_evidence.py data\processed\features\news_data.parquet --as-of 2026-07-01T00:00:00+03:00 --output-dir reports\dean_os\saved_semiconductor_news_evidence_producer_current
```

Build reviewed same-period ratios:

```powershell
python run_agent_saved_sec_derived_ratios.py reports\dean_os\saved_sec_fundamental_evidence_merger_current\latest.json --as-of 2026-07-01T00:00:00+03:00 --output-dir reports\dean_os\saved_sec_derived_ratio_producer_current
```

Build official policy evidence after the immutable BIS snapshot exists:

```powershell
python run_agent_saved_official_policy_evidence.py reports\dean_os\bis_policy_snapshot_current\latest.json reports\dean_os\saved_semiconductor_news_evidence_producer_current\latest.json --as-of 2026-07-01T00:00:00+03:00 --output-dir reports\dean_os\saved_official_policy_evidence_producer_current
```

Run the combined sector analyst:

```powershell
python run_agent_semiconductor_analyst.py --fundamental-artifact reports\dean_os\saved_sec_fundamental_evidence_merger_current\latest.json --derived-ratio-artifact reports\dean_os\saved_sec_derived_ratio_producer_current\latest.json --macro-artifact reports\dean_os\saved_macro_evidence_producer_current\latest.json --sector-market-artifact reports\dean_os\saved_sector_market_evidence_producer_current\latest.json --news-artifact reports\dean_os\saved_semiconductor_news_evidence_producer_current\latest.json --official-policy-artifact reports\dean_os\saved_official_policy_evidence_producer_current\latest.json --pipeline-case-artifact reports\dean_os\pipeline_model_case_packet_current\latest.json --as-of 2026-07-01T00:00:00+03:00 --output-dir reports\dean_os\semiconductor_analyst_runtime_current
```

Expected current state is `partial_ready_for_review`, `5/5` required lanes
satisfied, four `basket_candidate` companies, zero direct ticker theses, and
the AMD pipeline case excluded from sector evidence.

Build the runtime-linked human-review packet:

```powershell
python run_agent_domain_analyst_thesis_review_packet.py --domain-intake-json reports\dean_os\semiconductor_analyst_runtime_current\latest.json --domain-instance-contract-json reports\dean_os\domain_analyst_instance_contract_current\latest.json --architecture-map-json reports\dean_os\current_architecture_map_current\latest.json --output-dir reports\dean_os\domain_analyst_thesis_review_packet_current
```

Expected state: `domain_thesis_review_ready_with_cautions`, all linked hashes
verified, `23 pass / 3 warn / 0 fail`. The warnings are deliberate: confidence
is not a calibrated probability, the market window is short, and fundamentals
are not fully comparable.

Register the first prospective sector case without attaching the stale
template-standardization packet:

```powershell
python run_agent_domain_analyst_case_registry_packet.py --domain-thesis-review-json reports\dean_os\domain_analyst_thesis_review_packet_current\latest.json --without-template-standardization --output-dir reports\dean_os\domain_analyst_case_registry_packet_current
```

Expected state: one `pending_domain_outcome` case, exact review SHA preserved,
and 30/90/180-day review checkpoints.

Build the exact ticker-pipeline readiness bridge:

```powershell
python run_agent_saved_ticker_specific_evidence.py reports\dean_os\saved_semiconductor_news_evidence_producer_current\latest.json --as-of 2026-07-01T00:00:00+03:00 --tickers NVDA AMD INTC TSM --registry-path dean_os\config\semiconductor_issuer_identity_registry.yaml --output-dir reports\dean_os\saved_ticker_specific_evidence_producer_current
python run_agent_pipeline_prediction_review_packet.py data\colab\accumulated\main_database\stage_5_results.json --ticker AMD --ticker INTC --ticker NVDA --ticker TSM --filter-to-requested-scope --output-dir reports\dean_os\pipeline_prediction_source_review_current
python run_agent_sector_to_ticker_bridge.py --domain-thesis-review-json reports\dean_os\domain_analyst_thesis_review_packet_current\latest.json --ticker-evidence-json reports\dean_os\saved_ticker_specific_evidence_producer_current\latest.json --prediction-review-json reports\dean_os\pipeline_prediction_source_review_current\latest.json --feature-timeframe-audit-json reports\dean_os\pipeline_feature_timeframe_audit_current\latest.json --pipeline-case-json reports\dean_os\pipeline_model_case_packet_current\latest.json --output-dir reports\dean_os\sector_thesis_to_ticker_basket_current
python run_agent_sector_to_ticker_review_packet.py --bridge-json reports\dean_os\sector_thesis_to_ticker_basket_current\latest.json --output-dir reports\dean_os\sector_to_ticker_review_packet_current
python run_agent_pipeline_prediction_review_packet.py data\colab\accumulated\main_database\stage_5_results.json --ticker AMD --ticker INTC --ticker NVDA --ticker TSM --filter-to-requested-scope --sector-to-ticker-review-json reports\dean_os\sector_to_ticker_review_packet_current\latest.json --output-dir reports\dean_os\pipeline_prediction_review_packet_current
```

Expected ticker-evidence state: 49 company candidates, 6 strong candidates,
and only AMD with one corroborated demand/guidance lane. Expected bridge state:
`ticker_pipeline_inputs_incomplete`; AMD evidence-ready but pipeline-blocked,
INTC/NVDA/TSM missing corroboration, 389 real Stage 5 contexts quarantined,
0 complete prediction contexts, and zero ticker forecasts.
Expected review state: `review_ready_with_limitations`; this means the gap map
is reviewable, not that any ticker forecast is ready. Run compatibility tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_saved_sector_market_evidence_producer.py tests\dean_os\test_saved_semiconductor_news_evidence_producer.py tests\dean_os\test_saved_ticker_specific_evidence_producer.py tests\dean_os\test_semiconductor_analyst_runtime.py tests\dean_os\test_domain_analyst_thesis_review_packet.py tests\dean_os\test_domain_analyst_case_registry_packet.py tests\dean_os\test_sector_thesis_to_ticker_basket_bridge.py tests\dean_os\test_sector_to_ticker_review_packet.py tests\dean_os\test_structured_context_point_in_time.py tests\dean_os\test_context_evidence_point_in_time.py -q -p no:cacheprovider
```

The two prediction-review artifacts have different roles. The base
`pipeline_prediction_source_review_current` must not have a sector overlay; the
bridge consumes it before the readiness review exists. The final
`pipeline_prediction_review_packet_current` may then attach the readiness review
as supporting-only context. Never feed that final overlay back into the bridge.

Do not mutate or backfill the quarantined Stage 5 result. The context overlay
cannot change the prediction, fill lineage, clear evaluation, promote a model,
or create a ticker forecast. A future regenerated artifact must obtain lineage
from the repaired Stage 4/5 runtime.

These commands do not train, tune, run the trading pipeline, write learning
memory, create a ticker forecast, or trade.

## SEC Inline-XBRL And Merged Cohort Fundamentals

Run source, inline-XBRL, merger, structured-context, and gate tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_saved_sec_companyfacts_producer.py tests\dean_os\test_saved_sec_inline_xbrl_producer.py tests\dean_os\test_saved_sec_fundamental_evidence_merger.py tests\dean_os\test_structured_context_point_in_time.py tests\dean_os\test_fundamental_gate_agent_lab_integration.py -q -p no:cacheprovider
```

Current safe build sequence:

```powershell
python run_agent_saved_sec_inline_xbrl.py reports\dean_os\sec_primary_document_snapshot_current\latest.json reports\dean_os\semiconductor_sec_filing_index_current\latest.json --tickers TSM --output-dir reports\dean_os\saved_sec_inline_xbrl_producer_current
python run_agent_saved_sec_submissions_filing_index.py reports\dean_os\sec_submissions_snapshot_current\latest.json --tickers NVDA --forms 10-Q 10-K --as-of 2026-06-30T21:00:00+00:00 --output-dir reports\dean_os\nvda_sec_filing_index_current
python run_agent_saved_sec_companyfacts.py reports\dean_os\nvda_sec_filing_index_current\latest.json --source-dir data\dean_os\sec_companyfacts_raw --output-dir reports\dean_os\nvda_saved_sec_companyfacts_producer_current
python run_agent_saved_sec_fundamental_merger.py reports\dean_os\saved_sec_companyfacts_producer_current\latest.json --additional-companyfacts-artifacts reports\dean_os\nvda_saved_sec_companyfacts_producer_current\latest.json --inline-xbrl-artifacts reports\dean_os\saved_sec_inline_xbrl_producer_current\latest.json --output-dir reports\dean_os\saved_sec_fundamental_evidence_merger_current
python run_agent_fundamental_input_readiness_gate.py --fundamentals-json reports\dean_os\saved_sec_fundamental_evidence_merger_current\latest.json --as-of 2026-06-30T21:00:00+00:00 --output-dir reports\dean_os\merged_fundamental_input_readiness_gate_current
```

Expected state: 29 facts, all four tickers represented, `4/4` source coverage,
and raw comparison still blocked by period/currency mismatch. Do not turn this
into a sector ranking or value score.

## Sector Cohort And Exact Tuning Scope

Run the exact-context tuning and filing-coverage tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_tuning_exact_context_scope.py tests\dean_os\test_tuning_surface_gate.py tests\dean_os\test_saved_sec_filing_index_producer.py -q -p no:cacheprovider
```

Build the active four-ticker semiconductor filing inventory:

```powershell
python run_agent_saved_sec_filing_index.py --tickers NVDA AMD INTC TSM --forms 10-Q 10-K 20-F 40-F --as-of 2026-07-01T00:00:00+03:00 --database-path data\trading_data.duckdb --output-dir reports\dean_os\semiconductor_sec_filing_index_current
```

The local DuckDB-only filing window remains partial, but the immutable official
SEC submissions recovery adds NVDA and the merged fundamental artifact has
`4/4` source coverage. Period/currency comparability is still incomplete. AMD
artifacts remain single-ticker smoke/model evidence; do not use them as
semiconductor-domain truth or as tuning scope for other tickers.

Any tuning preview must carry exact ticker, model, target, timeframe, and
context fingerprint. A missing lineage or broader configured ticker list must
produce validation only.

## Saved SEC Filing Index

Run acceptance-time, collector-hash, artifact-tamper, and real-DuckDB checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_saved_sec_filing_index_producer.py -q -p no:cacheprovider
```

Build the current AMD periodic filing index read-only:

```powershell
python run_agent_saved_sec_filing_index.py --tickers AMD --forms 10-Q 10-K --as-of 2026-07-01T00:00:00+03:00 --database-path data\trading_data.duckdb --output-dir reports\dean_os\saved_sec_filing_index_current
```

Share the filing fingerprint, accession, acceptance/report dates, canonical SEC
locator, collector-hash result, content boundary, and extraction requests.
Current DuckDB rows are metadata only. They do not contain primary-document
HTML or XBRL fact values and cannot feed FundamentalInputReadinessGate,
ValueScreening, ratios, or valuation.

Do not run the disabled SEC collector or fetch filing content as part of this
index command.

## Saved Macro Evidence Producer

Run schema, vintage, registry, tamper, verified-loader, Agent Lab, and real
saved-parquet checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_saved_macro_evidence_producer.py -q -p no:cacheprovider
```

Build a review-only fragment from the current saved snapshot:

```powershell
python run_agent_saved_macro_evidence_producer.py data\processed\macro_data_20260701_133641.parquet --as-of 2026-07-01T00:00:00+03:00 --output-dir reports\dean_os\saved_macro_evidence_producer_current
```

Use only the verified producer artifact in isolated Agent Lab:

```powershell
python run_agent_lab.py --macro-evidence-json reports\dean_os\saved_macro_evidence_producer_current\latest.json --as-of 2026-07-01T00:00:00+03:00 --no-financial-nlp --no-learning-records --no-operation-proposals --output-dir reports\dean_os\agent_lab_macro_review_current
```

Share `source_provenance`, registry status, selected observations, structured
audit, accepted fingerprint, MarketContext fragment, and safety. The registry
still requires operator confirmation. Do not call `realtime_start` the original
release time, infer missing units, fall back to file mtime, or feed raw macro
tables directly to agents.

These commands do not authorize pipeline features, predictions, learning,
paper execution, or trading.

## Structured Context And Fundamental Fingerprint Checks

Run structured semantics, future-data quarantine, raw-macro separation,
fingerprint mismatch, Agent Lab, and lazy-package checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_structured_context_point_in_time.py tests\dean_os\test_fundamental_input_readiness_gate.py tests\dean_os\test_fundamental_gate_agent_lab_integration.py tests\dean_os\test_package_lazy_import.py tests\dean_os\test_pipeline_adapter_review_contract.py -q -p no:cacheprovider
```

Audit a real saved fundamental input without running the trading pipeline:

```powershell
python run_agent_fundamental_input_readiness_gate.py --fundamentals-json <real_fundamentals.json> --as-of 2026-06-30T00:00:00+00:00 --output-dir reports\dean_os\fundamental_input_readiness_gate_current
```

Each metric must provide value, unit, period, availability timestamp, and a
stable source locator. Share `structured_context_audit`,
`structured_accepted_fingerprint`, and decision guidance. The same exact
payload and cutoff must be supplied to Agent Lab; a mismatched fingerprint is
correctly blocked. Do not add inferred timestamps or units to make a report
pass.

Raw macro DataFrames and document-count inventories are not structured
evidence. No command in this section computes ratios, values a company, runs
the pipeline, writes learning/config, or trades.

## Context Evidence Point-In-Time Checks

Run future-row, missing-time, missing-locator, duplicate, direct-ticker,
research-note, pipeline-adapter, and review-packet checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_context_evidence_point_in_time.py tests\dean_os\test_pipeline_adapter_review_contract.py tests\dean_os\test_orchestrator_review_boundary.py -q -p no:cacheprovider
```

Review an already-saved `MarketContext` JSON without running the pipeline:

```powershell
python run_agent_context_evidence_review.py <saved_market_context.json> --domain-id semiconductor_ai_infrastructure --as-of 2026-06-30T00:00:00+00:00 --output-dir reports\dean_os\context_evidence_review_current
```

Share `status`, `summary`, `evidence`, `exclusions`,
`integration_boundary`, and `safety`. Do not create a placeholder context to
make the packet look ready. Raw dataframe rows remain available separately;
only admissible rows enter analyst evidence.

## Isolated Paper Lifecycle Boundary Checks

Run lineage, tamper, missing-external-evidence, and human-review-only checks
together with the active Stage6 boundary tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_paper_lifecycle_boundary.py tests\unit\test_stage6_execution_boundary.py -q -p no:cacheprovider
```

The following commands only move already-existing hash-bound artifacts through
the dormant lifecycle:

```powershell
python run_agent_paper_simulation_plan.py <explicit_unexpired_paper_only_receipt.json> --output-dir reports\dean_os\paper_simulation_plans_current
python run_agent_paper_simulation_result.py <hash_bound_paper_plan.json> <immutable_isolated_executor_output.json> --output-dir reports\dean_os\paper_simulation_results_current
python run_agent_post_paper_simulation_review.py <hash_bound_paper_result.json> --output-dir reports\dean_os\post_paper_simulation_review_current
```

Do not create placeholder receipts or executor outputs to make this path look
ready. The result command records external evidence; it does not execute a
simulation. Ordinary Stage6 paper/live requests must remain blocked.

## Analyst Knowledge Point-In-Time Checks

Run the focused provenance, future-knowledge, collision, analyst propagation,
and readiness tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_analyst_knowledge_point_in_time.py -q -p no:cacheprovider
```

Audit the current local knowledge store for one exact timezone-aware `as_of`:

```powershell
python run_agent_analyst_knowledge_readiness.py --as-of 2026-06-30T00:00:00+00:00 --store-dir data\dean_os\analyst_knowledge --output-dir reports\dean_os\analyst_knowledge_readiness_current
```

Share `status`, `summary`, `reason_counts`, `records`,
`integration_contract`, and `safety`. `knowledge_store_empty_blocked` is the
correct result when no real pack is present. Do not weaken the gate or invent
publication/retrieval timestamps, source hashes, or pack lineage.

These commands do not run collectors, Stage5, Stage7, training, replay,
learning, recommendation, paper execution, or live trading.

## Current Deterministic Shadow Diagnostics

Run aligned-episode, semantic metric-availability, duplicate, and zero-case checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_shadow_calibration_diagnostics.py tests\dean_os\test_shadow_calibration_readiness.py tests\dean_os\test_shadow_component_case_producer.py tests\dean_os\test_shadow_calibration_case_index.py -q -p no:cacheprovider
```

Run diagnostics only on a real chained case index:

```powershell
python run_agent_shadow_calibration_diagnostics.py <shadow_calibration_case_index.json> --output-dir reports\dean_os\shadow_calibration_diagnostics_current
```

Expected current state: `shadow_diagnostics_blocked`. Do not calculate
probability metrics from adjusted classification scores, combine disjoint
episode IDs, fill unavailable metrics with zero, or treat diagnostic output as
consensus-weight authority.

## Current Exact-Context Shadow Component Cases

Run component-specific, time-leakage, chain-preservation, and common-context checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_shadow_component_case_producer.py tests\dean_os\test_shadow_calibration_case_index.py tests\dean_os\test_shadow_calibration_readiness.py -q -p no:cacheprovider
```

After a real prediction case index exists, chain each assessment family separately:

```powershell
python run_agent_shadow_component_case_producer.py regime <base_case_index.json> <stage7_regime_review.json> --output-dir reports\dean_os\shadow_calibration_case_index_current
python run_agent_shadow_component_case_producer.py specialist <base_case_index.json> <specialist_context_review.json> --output-dir reports\dean_os\shadow_calibration_case_index_current
python run_agent_shadow_component_case_producer.py context_synthesis <base_case_index.json> <context_synthesis.json> --output-dir reports\dean_os\shadow_calibration_case_index_current
```

Use each newly written `latest.json` as the next base. Post-prediction regime
evidence, inferred contexts, manual-pending specialist evidence, mismatched
lineage, and cross-ticker aggregate counts must remain blocked.

## Current Transferred Workbench And Outcome Case Index

Run immutable source binding, exact realization, and invalid-case rejection checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_shadow_calibration_case_index.py tests\dean_os\test_shadow_calibration_readiness.py tests\dean_os\test_pipeline_prediction_review_packet.py -q -p no:cacheprovider
```

Build cases only when both inputs are real and already saved:

```powershell
python run_agent_shadow_calibration_case_index.py <saved_prediction_review.json> <immutable_outcome_source.csv_or.parquet_or.json> --output-dir reports\dean_os\shadow_calibration_case_index_current
```

Expected without exact matured inputs: blocked/zero cases. Do not substitute
unit fixtures, nearest timestamps, unverified pipeline outputs, inferred sector
evidence, or later price rows. This command does not calibrate, change weights,
write learning/config, recommend, or trade.

## Current Stage5 Output Semantics Contract

Run the repaired Stage4/5 path, output-contract, target-semantics, and review checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\unit\test_stage5_model_output_contract.py tests\unit\test_stage4_active_training_contract.py tests\unit\test_prediction_stage_model_selection.py tests\dean_os\test_prediction_target_semantics.py tests\dean_os\test_pipeline_prediction_review_packet.py tests\dean_os\test_shadow_calibration_readiness.py -q -p no:cacheprovider
```

Expected contract: Stage5 `.predict()` classification output is a label only
when the runtime class contract verifies that meaning; otherwise scale stays
unknown. An adjusted classification score is never assumed to be a
positive-class probability.
The packet remains review-only and directional inference remains disabled.
Do not run the pipeline merely to manufacture a current artifact.

## Current Template Harvest And Shadow Calibration Readiness

Run target unit/period/class and calibration-policy checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_prediction_target_semantics.py tests\dean_os\test_pipeline_prediction_review_packet.py tests\dean_os\test_shadow_calibration_readiness.py -q -p no:cacheprovider
```

Refresh the review-only readiness artifact:

```powershell
python run_agent_shadow_calibration_readiness.py
```

Expected current result: `shadow_calibration_blocked`, 0/30 cases for
prediction, regime, specialist, and context synthesis. Do not use unit fixtures
as historical cases, infer missing saved output contracts, change consensus weights, or
write production config.

## Current Specialist Context Boundary

Run domain-vs-ticker, point-in-time, and synthesis separation checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_specialist_context_review_packet.py tests\dean_os\test_context_synthesis_agent.py -q -p no:cacheprovider
```

Refresh the AMD/15m review-only context:

```powershell
python run_agent_specialist_context_review.py --ticker AMD --timeframe 15m --as-of 2026-06-24T19:30:00+00:00 --output-dir reports\dean_os\specialist_context_review_amd_15m_current
```

Expected: `direct_ticker_review_candidate`,
`older_than_review_window`, undeclared source timeframe, manual review pending,
and `eligible_for_exact_pipeline_context=false`. This does not turn the
semiconductor domain into an AMD thesis.

## Current Per-Context Shadow Synthesis

Run Stage7 window provenance, exact-context synthesis, and no-consensus-influence checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\unit\test_stage7_review_boundaries.py tests\dean_os\test_context_synthesis_agent.py tests\dean_os\test_regime_agent_stage7_bridge.py tests\dean_os\test_pipeline_prediction_review_packet.py -q -p no:cacheprovider
```

Expected registry state: `context_synthesis` and `regime` are `pre_trade`
shadow agents. Both remain visible for review but have
`decision_influence=false`. Do not interpret a Stage5 scalar directionally
without an explicit target-semantics contract.

## Current Stage5 Prediction Review Contract

Run the packet and adapter boundary checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_pipeline_prediction_review_packet.py tests\dean_os\test_pipeline_adapter_review_contract.py -q -p no:cacheprovider
```

Build the current base packet from the explicit saved real pipeline result:

```powershell
python run_agent_pipeline_prediction_review_packet.py data\colab\accumulated\main_database\stage_5_results.json --ticker AMD --ticker INTC --ticker NVDA --ticker TSM --filter-to-requested-scope --output-dir reports\dean_os\pipeline_prediction_source_review_current
```

Expected current state is `stage5_prediction_review_partial`, 389 selected
contexts, 0 complete, and 389 quarantined. Do not substitute a unit fixture,
mutate the saved source, or label quarantine evidence as a forecast. Packet
authority is always `decision_influence=false`, `is_model_evaluation=false`,
`is_realized_outcome=false`, and `can_trade=false`.

## Current Stage7 Regime Shadow Bridge And Agent Matrix

Run the exact-context, no-fallback, no-consensus-influence, adapter, and matrix checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_regime_agent_stage7_bridge.py tests\dean_os\test_pipeline_adapter_review_contract.py tests\dean_os\test_orchestrator_review_boundary.py tests\dean_os\test_agent_capability_matrix.py -q -p no:cacheprovider
```

Refresh the review-only capability artifact:

```powershell
python run_agent_capability_matrix.py
```

Expected state: `regime` is enabled only for `pre_trade`, consumes
`dean_stage7_regime_review_v1`, and has `decision_influence=false`. AMD is one
ticker/model evaluation case and is not semiconductor-domain evidence. These
commands do not run Stage7, the trading pipeline, training, learning, or
execution.

## Current Shared Feedback Boundary

Run the shared taxonomy, model-feedback, case, and review-routing checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_pipeline_model_feedback_packet.py tests\dean_os\test_pipeline_model_case_packet.py tests\dean_os\test_domain_analyst_feedback_loop_packet.py -q -p no:cacheprovider
```

Refresh the pending feedback artifact without inventing human labels:

```powershell
python run_agent_pipeline_model_feedback_packet.py
python run_agent_review_index.py
python run_agent_chief_review_index.py
```

Expected current feedback status is
`pipeline_model_feedback_ready_pending_manual_feedback`, with zero feedback
records and zero learning candidates. Do not pass a fabricated feedback file,
and do not route model cases into `ReviewApprovedLearningLoop`.

## Current Pipeline Model Case Review

Run the case, agent binding, and review-routing checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_pipeline_model_case_packet.py tests\dean_os\test_pipeline_adapter_review_contract.py -q -p no:cacheprovider
```

Refresh only saved review artifacts:

```powershell
python run_agent_pipeline_model_case_packet.py
python run_agent_model_performance.py reports\dean_os\pipeline_control_metric_artifact_materializer_current\model_evaluation\latest.json --evidence-chain-path reports\dean_os\pipeline_control_real_metric_evidence_run_current\latest.json --model-case-path reports\dean_os\pipeline_model_case_packet_current\latest.json --output-dir reports\dean_os\model_performance
python run_agent_review_index.py
python run_agent_chief_review_index.py
```

Expected current result: the model case is a negative evaluation-block case,
the agent is caution with zero actionable signal, and Chief Review returns the
candidate-scoped `model_candidate_blocked`. Unrelated review/research work is
not globally paused. These commands do not train, tune, learn, recommend, or
trade.

## Current Locked Evidence To Model-Agent Chain

Run the provenance and evidence-chain checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_pipeline_control_evidence_inventory.py tests\dean_os\test_pipeline_control_locked_evaluation_assembler.py tests\dean_os\test_pipeline_control_metric_artifact_materializer.py tests\dean_os\test_pipeline_control_real_metric_evidence_run.py tests\dean_os\test_pipeline_adapter_review_contract.py -q
```

Refresh only saved review artifacts:

```powershell
python run_agent_pipeline_control_evidence_inventory.py --output-dir reports\dean_os\pipeline_control_evidence_inventory_current
python run_agent_pipeline_control_metric_artifact_materializer.py --output-dir reports\dean_os\pipeline_control_metric_artifact_materializer_current
python run_agent_pipeline_control_real_metric_evidence_run.py --model-evaluation-json reports\dean_os\pipeline_control_metric_artifact_materializer_current\model_evaluation\latest.json --feature-stability-report reports\dean_os\pipeline_control_metric_artifact_materializer_current\feature_stability\latest.json --output-dir reports\dean_os\pipeline_control_real_metric_evidence_run_current
```

Current chain is valid evidence but blocked on validation and feature stability.
These commands do not collect data, train, tune, learn, recommend, or trade.

## Current Stage 7 Analyzer Review Contract

Run the analyzer routing, Stage 7 boundary, and final-orchestrator checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\unit\test_unified_analytics_review_routing.py tests\unit\test_stage7_review_boundaries.py tests\unit\test_stage6_execution_boundary.py tests\unit\test_exception_handling.py -q
```

Current result: 16 passed. Expected active analyzer set is exactly
`market_regime` plus `critical_signals`; all other catalog entries must appear
as disabled or be skipped/failed visibly. Analyzer output is supporting review
context only and may not promote, write config, notify, or trade.

Check the DEAN adapter and model-performance boundary:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_pipeline_adapter_review_contract.py tests\dean_os\test_orchestrator_review_boundary.py -q
```

Current result: 9 passed. The adapter must expose
`dean_stage7_analyzer_review_v1`; `ModelPerformanceAgent` may reference it but
must accept pipeline metrics only from canonical `evaluation_summary.metrics`.

Do not run the full pipeline merely to test this contract.

## Current Active Stage 4 -> Stage 7 Contract

Run the focused normal-path checks without training real models:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\unit\test_stage4_active_training_contract.py tests\dean_os\test_pipeline_control_metric_artifact_candidates.py tests\dean_os\test_pipeline_control_evaluation_metric_artifact_candidates.py -q -p no:cacheprovider
```

The active contract is now:

- nested prepared splits are adapted to the unified trainer;
- model selection uses validation, not the reserved holdout;
- candidate model files do not overwrite each other;
- only the actual winner receives the stable champion path;
- Stage 4 artifacts remain partial when train score, importance, or drawdown is unavailable;
- Stage 5 carries model/target/timeframe/context lineage into Stage 7.

Do not run normal Stage 4 training until a new forward artifact clears the
accrual gate.

## Current Causal Multi-Timeframe Pipeline Integration

Run the focused causal context, target-alignment, and model-preparation checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\unit\test_timeframe_context_integration.py tests\unit\test_target_orchestrator_alignment.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_timeframe_context_final
```

Current result: 13 tests passed. The separate Stage 3 contract set has 10
passing tests, and the async feature-selection leakage test passes when
`pytest_asyncio.plugin` is enabled.

The active contract is:

- base row identity and row count must be preserved;
- higher-timeframe joins are backward-only and ticker/partition isolated;
- 60m context is unavailable until its bar completes;
- daily context is unavailable to same-day intraday rows;
- context targets never become model features;
- Stage 4 model contexts are isolated by `(ticker, interval)`;
- frozen corrected test windows remain unavailable for iteration.

Run the already-integrated development-only walk-forward evaluator only when a
new predeclared development context is justified:

```powershell
python run_agent_pipeline_control_walk_forward_validation.py --historical-recovery-json reports\dean_os\pipeline_control_historical_price_recovery_current\latest.json --ticker NVDA --timeframe 15m --target-name target_intraday_up_15m --macro-source-path data\processed\features\macro_data.parquet --acknowledge-development-only --min-train-rows 360 --validation-rows 120 --step-rows 120 --purge-rows 5 --max-folds 4 --max-features 40 --output-dir reports\dean_os\pipeline_control_walk_forward_validation_current
```

Current NVDA result is
`walk_forward_candidate_blocked_by_validation_contract`: four temporal folds
passed, but mean validation balanced accuracy=0.516836, mean train-validation
gap=0.297556, mean feature stability=0.528056, and maximum positive-rate
gap=0.308333 failed the declared contract. Test and past-evaluation rows loaded
were both zero.

Do not rerun model/feature variants on these folds. The artifact is
development-only supporting evidence and cannot satisfy locked test evidence.
Accumulate genuinely new forward observations before defining a virgin holdout.

Register or refresh the prospective development-data boundary without loading
prices or labels:

```powershell
python run_agent_pipeline_control_forward_data_accrual_plan.py --walk-forward-json reports\dean_os\pipeline_control_walk_forward_validation_current\latest.json --acknowledge-development-refresh-only --output-dir reports\dean_os\pipeline_control_forward_data_accrual_plan_current
```

Current result is `forward_development_accrual_plan_ready`. The boundary records
`2026-05-06T17:30:00+00:00` as the last used validation timestamp and requires
at least 120 new 15m rows in a new immutable source artifact acquired after plan
registration. Those rows remain development-refresh data; they are not a virgin
holdout.

Validate a future saved artifact before it reaches Stage 3:

```powershell
python run_agent_pipeline_control_forward_data_accrual_gate.py --accrual-plan-json reports\dean_os\pipeline_control_forward_data_accrual_plan_current\latest.json --source-path PATH_TO_NEW_IMMUTABLE_PRICE_ARTIFACT.parquet --output-dir reports\dean_os\pipeline_control_forward_data_accrual_gate_current
```

The existing June 25 artifact was tested and correctly returns
`blocked_forward_development_artifact`: 1,018 candidate rows after the watermark
but 0 eligible rows. Blockers are pre-registration file time, maximum absolute
return 8.03446, and 1,490 cross-ticker copied-OHLCV groups. Do not copy/rename
the file to bypass first-seen provenance.

Only after that gate reports `forward_development_artifact_ready`, pass it into
the development walk-forward runner:

```powershell
python run_agent_pipeline_control_walk_forward_validation.py --historical-recovery-json reports\dean_os\pipeline_control_historical_price_recovery_current\latest.json --forward-accrual-gate-json reports\dean_os\pipeline_control_forward_data_accrual_gate_current\latest.json --ticker NVDA --timeframe 15m --target-name target_intraday_up_15m --macro-source-path data\processed\features\macro_data.parquet --acknowledge-development-only --output-dir reports\dean_os\pipeline_control_walk_forward_validation_current
```

The runner rechecks gate mode, artifact class, SHA, context, watermark, and row
count. The current blocked gate cannot enter this path.

## Current Causality And Corrected Baseline

Verify that adding a future suffix cannot change earlier Stage 3 features:

```powershell
python run_agent_pipeline_control_feature_causality_audit.py --batch-json reports\dean_os\pipeline_control_bounded_evidence_batch_current\latest.json --ticker NVDA --ticker SPY --max-contexts 2 --output-dir reports\dean_os\pipeline_control_feature_causality_audit_current
```

Current result is `feature_prefix_invariance_passed`: NVDA 0/229 violations, SPY 0/230, and service OHLCV identity is invariant.

The current bounded batch is the one corrected post-causality baseline. It completed four real locked pairs and cleared zero cautions. Do not rerun model or feature variants against its frozen test windows. The development-only walk-forward evaluator now exists and is blocked; use the registered forward-data accrual boundary before any next predeclared development run.

## 0. One-Command Review Refresh

Refresh all safe review artifacts and automatically run real-metric review only when both locked inputs exist:

```powershell
python run_agent_dean_os_review_automation.py --output-dir reports\dean_os\review_only_automation_run_current
```

Force report refresh while always skipping the final real-metric review:

```powershell
python run_agent_dean_os_review_automation.py --no-real-metric-run --output-dir reports\dean_os\review_only_automation_run_current
```

This command does not start collectors, training, Stage 7 evaluation, replay, backtests, tuning, learning/config writes, recommendations, or trading. Review `summary`, `steps`, `next_runner_inputs`, and `operator_next_steps` in `reports\dean_os\review_only_automation_run_current\latest.json`.

Run saved-data coverage and non-destructive repair together:

```powershell
python run_agent_pipeline_control_data_preflight.py --output-dir reports\dean_os\pipeline_control_data_preflight_current
```

This is the normal data-readiness command. It does not call collectors or APIs and does not modify DuckDB/source parquet files.

Inventory all configured assets, saved timeframe quality, and saved macro coverage:

```powershell
python run_agent_pipeline_control_saved_data_coverage.py --output-dir reports\dean_os\pipeline_control_saved_data_coverage_current
```

Build only the non-destructive price repair candidates from the latest coverage:

```powershell
python run_agent_pipeline_control_saved_price_repair.py --coverage-json reports\dean_os\pipeline_control_saved_data_coverage_current\latest.json --output-dir reports\dean_os\pipeline_control_saved_price_repair_current
```

Validate and partition the trusted long historical contexts:

```powershell
python run_agent_pipeline_control_historical_price_recovery.py --historical-15m data\colab\backup_20260510_153551\stage2_prices_15m_20260507_161411.parquet --current-15m reports\dean_os\pipeline_control_saved_price_repair_current\pipeline_control_saved_price_repair_20260627T120633739200+0000\artifacts\prices_15m_clean.parquet --historical-1d data\colab\backup_20260510_153551\stage2_prices_1d_20260426_083142.parquet
```

Current result is `historical_context_partitions_ready`. Development is ready at 15m, 60m, and 1d for all 18 tickers; the separate past-evaluation partition is ready at 15m and 60m. The runner performs no training and does not merge the partitions.

Run the current predeclared semiconductor/control batch while keeping the old AMD test frozen:

```powershell
python run_agent_pipeline_control_bounded_evidence_batch.py --coverage-json reports\dean_os\pipeline_control_saved_data_coverage_current\latest.json --ticker NVDA --ticker INTC --ticker TSM --ticker SPY --frozen-context AMD/15m --rows-per-context 480 --output-dir reports\dean_os\pipeline_control_bounded_evidence_batch_current
```

This batch uses only saved prices and the saved macro artifact chosen by coverage. It creates locked review evidence, not production models or trading actions.

Create the frozen real AMD/15m bounded evidence slice from saved prices:

```powershell
python run_agent_pipeline_control_bounded_evidence.py --source-path data\processed\prices_15m_20260625_125005.parquet --macro-source-path data\processed\features\macro_data.parquet --ticker AMD --timeframe 15m --target-name target_intraday_up_15m --start 2026-05-28T16:45:00+03:00 --max-rows 480 --max-features 40 --gap-size 5 --output-dir reports\dean_os\pipeline_control_bounded_evidence_run_current
```

The current test window is a frozen benchmark. Do not repeatedly tune against it. Review `split_metrics`, `feature_stability_analysis`, `heldout_evaluation`, and `summary` before defining one train/validation-only revision.

## 1. Agent Lab

Real materials:

```powershell
python run_agent_lab.py docs/research --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle --regime-tags rising_market
```

With reviewed caller-supplied fundamentals:

```powershell
python run_agent_fundamental_input_readiness_gate.py --fundamentals-json reports\dean_os\fundamentals_input\latest.json --output-dir reports\dean_os\fundamental_input_readiness_gate_current
python run_agent_lab.py docs/research --fundamentals-json reports\dean_os\fundamentals_input\latest.json --fundamental-gate-json reports\dean_os\fundamental_input_readiness_gate_current\latest.json --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --tickers AMD --sectors semiconductor --tags ai_cycle
```

When `--fundamentals-json` is supplied, Agent Lab runs `value_screening`. If the fundamental gate has warnings or failures, `ValueScreeningAgent` should not score those fundamentals; it returns `needs_more_data` until the gate is clean.

Smoke test only:

```powershell
python run_agent_lab.py --sample --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle --regime-tags rising_market
```

Use `--tags` for event/theme labels such as `ai_cycle`, `ipo`, `energy_shock`.
Use `--regime-tags` for market-state labels such as `calm_market`, `rising_market`, `falling_market`, `crisis`, `volatility_spike`.

## 1a. Regime Context From CSV

Use the latest processed daily prices already present in the project:

```powershell
python run_regime_context.py --latest-processed-prices 1d --ticker AMD --output reports/dean_os/regime_context/amd_latest.json
```

Specific CSV/parquet file:

```powershell
python run_regime_context.py data/processed/prices_1d_20260606_022214.parquet --ticker AMD --output reports/dean_os/regime_context/amd_latest.json
```

Use the existing project market-regime analyzer through the DEAN-OS bridge:

```powershell
python run_regime_context.py --latest-processed-prices 1d --ticker AMD --engine project --output reports/dean_os/regime_context/amd_project.json
```

Manual review context when no price file is available:

```powershell
python run_regime_context.py --manual-regime TRENDING_UP --manual-tags ai_cycle --output reports/dean_os/regime_context/amd_manual.json
```

Synthetic smoke context only:

```powershell
python run_regime_context.py --sample-scenario rising --output reports/dean_os/regime_context/sample_rising.json
```

Feed a saved regime context into Agent Lab:

```powershell
python run_agent_lab.py docs/research --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle --regime-context-json reports/dean_os/regime_context/amd_latest.json
```

Run the same regime context as a pipeline soft-agent report for consensus:

```powershell
python run_agent_regime.py --latest-processed-prices 1d --ticker AMD
```

Share the `report.verdict`, `report.signal_strength`, `regime_context.regime`, `regime_context.context_tags`, and `regime_context.metrics`.

## 2. Review Snapshot

```powershell
python run_agent_review.py --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --review-actions-store data/dean_os/review_actions.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --log-path logs/dean_os/events.jsonl
```

Share the `next_actions`, `review_actions`, `memory`, and `operations` sections.

## 3. Logs

```powershell
python run_agent_logs.py summary
python run_agent_logs.py tail --limit 10
```

## 4. Learning Records

```powershell
python run_agent_learning.py --store data/dean_os/agent_learning.sqlite list
python run_agent_learning.py --store data/dean_os/agent_learning.sqlite score evidence_synthesis
python run_agent_learning.py --store data/dean_os/agent_learning.sqlite score specialist_research
```

Only update an outcome after you have a real `record_id` from `list`.

## 5. Operation Queue

```powershell
python run_agent_ops.py --store data/dean_os/operation_queue.sqlite list
python run_agent_ops.py --store data/dean_os/operation_queue.sqlite dry-run PROPOSAL_ID_FROM_LIST
```

Use `dry-run` before approving anything. Dry-run does not execute the pipeline.

## 6. Review Actions

```powershell
python run_agent_review_actions.py list
python run_agent_review_actions.py mark-reviewed --source-type agent_lab_report --source-id ACTUAL_RUN_ID_FROM_REVIEW --notes "Reviewed"
python run_agent_review_actions.py needs-more-data --source-type agent_lab_report --source-id ACTUAL_RUN_ID_FROM_REVIEW --data-request "Add filings and transcripts"
python run_agent_review_actions.py void-action ACTION_ID_FROM_LIST --reason "Mistyped id or invalid action"
```

Do not type placeholders such as `RUN_ID_HERE`, `RECORD_ID_HERE`, or `PROPOSAL_ID_HERE`.

## 7. Recommendation Memory

Summary:

```powershell
python run_agent_memory.py summary
```

Add a miss case:

```powershell
python run_agent_memory.py add-case --source-id fuel-crisis-case --agent-name macro_policy --topic "fuel crisis" --thesis "Fuel stress would be short-lived" --expected-direction neutral --outcome-label miss --context-tags fuel_crisis energy_shock --lesson "Fuel shock persistence was underestimated"
```

Add a hit case:

```powershell
python run_agent_memory.py add-case --source-id space-ipo-case --agent-name specialist_research --topic "space IPO" --thesis "High-quality space IPO improved sector sentiment" --expected-direction bullish --outcome-label hit --context-tags ipo space quality_growth --lesson "Strong narrative plus scarcity improved follow-through"
```

List by context:

```powershell
python run_agent_memory.py list --context-tag fuel_crisis
python run_agent_memory.py list --context-tag ai_cycle
```

## 8. Context Performance

Overall agent performance by context and regime:

```powershell
python run_agent_context_performance.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite
```

Filter by a specific regime/theme:

```powershell
python run_agent_context_performance.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --context-tag crisis
python run_agent_context_performance.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --context-tag ai_cycle
```

Share the `weak_contexts`, `strengths`, `recent_miss_lessons`, and `recommendations` sections.

## 9. Outcome Evaluation

Dry-run pending learning records against latest processed prices:

```powershell
python run_agent_outcome_evaluation.py --learning-store data/dean_os/agent_learning.sqlite --latest-processed-prices 1d
```

Limit output while debugging:

```powershell
python run_agent_outcome_evaluation.py --learning-store data/dean_os/agent_learning.sqlite --latest-processed-prices 1d --limit 5
```

Apply only after reviewing dry-run output:

```powershell
python run_agent_outcome_evaluation.py --learning-store data/dean_os/agent_learning.sqlite --latest-processed-prices 1d --apply
```

Do not use `--apply` if records are `not_due`, `no_price_after_created_at`, or `missing_price_window`.

## 10. Market Data Freshness

Check local market prices without running the trading pipeline:

```powershell
python run_agent_market_freshness.py --latest-processed-prices 1d --tickers AMD NVDA --max-age-hours 24
```

Also show the operation proposal that would be generated from stale data:

```powershell
python run_agent_market_freshness.py --latest-processed-prices 1d --tickers AMD NVDA --max-age-hours 24 --include-operation-proposal
```

Share the `data_freshness.market_prices`, `report.verdict`, and `action_proposals` sections.

## 10a. Collector Inventory

Map collector configs/classes without running collectors or network calls:

```powershell
python run_agent_collector_inventory.py --output reports/dean_os/collector_inventory/latest.json
```

Share the `collector_inventory.summary`, `collector_inventory.rss_pipeline_status`, `collector_inventory.recommendations`, and any `enabled_missing_classes`.

Interpretation:
- `pipeline_news_feed`: RSS, Google News, and NewsAPI style feeds that belong to the candle/news/sentiment pipeline after isolated health checks.
- `research_specialist_feed`: SEC filings, insider/CFTC-style evidence sources that should first feed ResearchCorpus/Agent Lab instead of blocking daily pipeline runs.

## 10b. Model Performance

Check local evaluation/backtest metrics without training, tuning, or running the pipeline:

```powershell
python run_agent_model_performance.py PATH_TO_EVALUATION_JSON_OR_CSV --include-operation-proposal
```

Share the `model_performance.metrics`, `model_performance.threshold_failures`, `report.verdict`, and any `action_proposals`.

Use this before model promotion or tuning review. A generated proposal is only a review item; it does not tune, train, or change production config.

## 10c. Tuning Proposal

Create a review-only tuning experiment proposal from model metrics, optional regime context, and the pipeline control surface:

```powershell
python run_agent_tuning.py PATH_TO_EVALUATION_JSON_OR_CSV --control-surface-json reports/dean_os/pipeline_control_surface/latest.json --regime-context-json reports/dean_os/regime_context/amd_latest.json --tickers AMD --timeframes 1d --require-control-surface
```

Share the `tuning.status`, `tuning.model_failures`, `tuning.control_surface_gate`, `tuning.allowed_variation`, `tuning.guardrails`, and `action_proposals` sections.

This does not train, tune, run Optuna, or write production config. If the control surface is blocked, it should only propose validation/refresh work.

## 10d. Chief Review

Synthesize saved review/model/regime/tuning context into one supervised-autonomy review:

```powershell
python run_agent_chief_review.py --review-snapshot reports/dean_os/review/REVIEW_FILE.json
```

With separate saved outputs:

```powershell
python run_agent_chief_review.py --model-performance-json PATH_TO_MODEL_PERFORMANCE_JSON --regime-context-json reports/dean_os/regime_context/amd_latest.json --tuning-json PATH_TO_TUNING_JSON
```

Share the `chief_review.decision`, `chief_review.autonomy_recommendation`, `chief_review.reasons`, `chief_review.risks`, and `chief_review.next_actions` sections.

Paper autonomy may continue only as logged simulation. Promotion, production config changes, and live execution stay gated.

## 10e. Paper Trade Log

Record an autonomous paper decision:

```powershell
python run_agent_paper_trades.py record --action candidate_long --tickers AMD --horizon-days 30 --thesis "Paper-only thesis" --context-tags ai_cycle --regime-tags rising_market
```

List and summarize paper decisions:

```powershell
python run_agent_paper_trades.py list
python run_agent_paper_trades.py summary
```

Evaluate pending paper decisions against local prices without updating the store:

```powershell
python run_agent_paper_trades.py evaluate --latest-processed-prices 1d --tickers AMD NVDA
```

Apply outcomes only after reviewing dry-run output:

```powershell
python run_agent_paper_trades.py evaluate --latest-processed-prices 1d --tickers AMD NVDA --apply
```

Share the `status_counts`, `evaluations`, `summary_after`, and `recommendations` sections.

## 10f. Paper Portfolio Simulation

Simulate logged paper decisions as a paper-only portfolio against local prices:

```powershell
python run_agent_paper_portfolio.py --latest-processed-prices 1d --tickers AMD NVDA --output reports/dean_os/paper_portfolio/latest.json
```

Optional diagnostics:

```powershell
python run_agent_paper_portfolio.py --latest-processed-prices 1d --tickers AMD NVDA --include-watchlist --watchlist-position-size-pct 0.02 --slippage-bps 5 --commission-bps 1
```

Share the `report.verdict`, `paper_portfolio.summary`, `paper_portfolio.skipped`, and `paper_portfolio.recommendations` sections.

This does not update `paper_trades.sqlite`, does not place orders, and does not run the heavy trading pipeline.

## 10g. Paper Autonomy Loop

Run the supervised paper-autonomy loop without broker access or heavy pipeline execution:

```powershell
python run_agent_paper_autonomy.py --latest-processed-prices 1d --tickers AMD NVDA --max-age-hours 24
```

Use a saved review snapshot if you want ChiefReviewAgent to include a specific review state:

```powershell
python run_agent_paper_autonomy.py --latest-processed-prices 1d --tickers AMD NVDA --review-snapshot reports/dean_os/review/REVIEW_FILE.json --max-age-hours 24
```

Share the `decision`, `data_freshness.market_prices`, `regime_context`, `chief_review`, `paper_portfolio.summary`, `diary_bridge`, `action_proposals`, `journals`, and `recommendations` sections.

This reads `logs/experience_diary.csv` and DEAN logs as journal evidence, runs the diary bridge inspector, but does not write to the pipeline diary.

## 10h. Diary Bridge

Inspect whether evaluated DEAN paper outcomes can safely influence the existing pipeline diary:

```powershell
python run_agent_diary_bridge.py --experience-diary logs/experience_diary.csv --paper-store data/dean_os/paper_trades.sqlite --output reports/dean_os/diary_bridge/latest.json
```

Share the `diary_bridge.status`, `diary_bridge.pipeline_diary`, `diary_bridge.paper_records`, `diary_bridge.recommendations`, and `action_proposals` sections.

This is review-only. It does not write to `logs/experience_diary.csv`; it only reports whether the schema and evaluated paper records are safe to bridge.

## 10i. Source Routing

Route local materials and collector inventory to pipeline/specialist intake paths:

```powershell
python run_agent_source_routing.py docs/research --collector-inventory reports/dean_os/collector_inventory/latest.json --output reports/dean_os/source_routing/latest.json
```

Share the `source_routing.summary`, `source_routing.analyst_inputs`, `source_routing.recommendations`, and `source_routing.warnings` sections.

Interpretation:
- Research materials go to `ResearchCorpus`, then Agent Lab, specialist notes, citations, synthesis, memory.
- Pipeline feeds go through health/schema/timestamp checks before model inputs.
- Analysts consume `MarketContext` and `ResearchCorpus`; they do not execute trades.

## 10i-a. Analyst Evidence Pack

Build a normalized, citable analyst input pack from local materials, cached news, and macro files:

```powershell
python run_agent_analyst_evidence_pack.py --materials docs/research --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --source-routing-json reports/dean_os/source_routing/latest.json --tickers AMD NVDA --sectors semiconductor --tags ai_cycle
```

Cached news/macro smoke without `docs\research` materials:

```powershell
python run_agent_analyst_evidence_pack.py --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tickers AAPL AMD MSFT NVDA TSM --sectors semiconductor --tags ai_cycle cached_source_smoke --max-rows-per-table 200 --output-dir reports\dean_os\analyst_evidence_pack_cached_source_current
python run_agent_source_evidence_validation_gate.py --source-json reports\dean_os\analyst_evidence_pack_cached_source_current\latest.json --output-dir reports\dean_os\source_evidence_validation_gate_cached_source_current
```

Strict sector-only semiconductor smoke, without forcing a ticker basket:

```powershell
python run_agent_analyst_evidence_pack.py --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --sectors semiconductor --tags ai_cycle sector_only_strict_smoke --sector-keywords semiconductor semiconductors chip chips GPU GPUs accelerator accelerators foundry foundries wafer wafers fab fabs HBM DRAM memory lithography packaging "export control" Taiwan equipment --max-rows-per-table 200 --output-dir reports\dean_os\analyst_evidence_pack_semiconductor_sector_only_strict_current
python run_agent_source_evidence_validation_gate.py --source-json reports\dean_os\analyst_evidence_pack_semiconductor_sector_only_strict_current\latest.json --output-dir reports\dean_os\source_evidence_validation_gate_semiconductor_sector_only_strict_current
```

Feed the saved evidence pack into Agent Lab:

```powershell
python run_agent_lab.py --evidence-pack-json reports/dean_os/analyst_evidence_pack/latest.json --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle
```

Share the `coverage`, `analyst_inputs.base_analyst`, `analyst_inputs.manager_plan`, `warnings`, `dropped`, and `recommendations` sections.

Interpretation:
- This is the preferred bridge from raw sources to analysts.
- Start with one strong base analyst/evidence contract, then specialize by sector or domain later.
- The evidence pack does not run the heavy pipeline and does not create trades.
- Cached news/macro are acceptable local source inputs after timestamp/source-shape checks.
- Live collectors remain separate: inventory/health first, then evidence-pack ingestion from saved artifacts only.
- For sector-only analyst smoke, use strict sector keywords and do not use bare `AI` as a standalone semiconductor filter.
- Sector-only packs may mention companies, but they must not become ticker theses unless the separate bridge finds direct ticker evidence.

## 10i-b. Analyst Profile Manager

Run the centrally managed analyst profile flow from an evidence pack:

```powershell
python run_agent_analyst_profiles.py reports/dean_os/analyst_evidence_pack/latest.json --output-dir reports/dean_os/analyst_profiles
```

Preview requested specialist profiles without allowing them:

```powershell
python run_agent_analyst_profiles.py reports/dean_os/analyst_evidence_pack/latest.json --profiles generalist_base_analyst news_catalyst macro_policy sector_cycle value_screening --no-review-snapshot
```

Allow candidate specialist profiles only when you intentionally want to run them:

```powershell
python run_agent_analyst_profiles.py reports/dean_os/analyst_evidence_pack/latest.json --profiles news_catalyst macro_policy sector_cycle --allow-candidate-profiles --no-review-snapshot
```

Share the `profile_plan`, `profile_runs`, `analytical_reports`, `review_snapshot`, and `recommendations` sections.

Interpretation:
- The manager runs `generalist_base_analyst` first.
- Candidate profiles stay skipped unless `--allow-candidate-profiles` is explicitly passed.
- This does not run the heavy pipeline and does not create trades.
- Learning records and operation proposals stay off unless explicitly enabled with flags.

## 10i-c. Analyst Profile Scorecard

Score saved analyst profile runs before promoting any specialist profile:

```powershell
python run_agent_analyst_scorecard.py --profile-runs-dir reports/dean_os/analyst_profiles --output-dir reports/dean_os/analyst_profile_scorecard
```

For diagnostics only, lower thresholds to see how activation logic behaves:

```powershell
python run_agent_analyst_scorecard.py --profile-runs-dir reports/dean_os/analyst_profiles --min-completed-runs 1 --min-avg-confidence 0.1 --min-avg-citations 1
```

Share the `summary`, `profiles`, `profiles.PROFILE.blockers`, `profiles.PROFILE.activation_status`, and `recommendations` sections.

Interpretation:
- This does not run analysts or the trading pipeline.
- It is a promotion ledger for profile defaults.
- Keep `generalist_base_analyst` as default until scorecard evidence supports specialization.

## 10i-d. Analyst Learning Promotion Bridge

Dry-run promotion from reviewed analyst/profile output into durable learning records:

```powershell
python run_agent_analyst_learning_bridge.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite
```

Only after the source Agent Lab run has a real `mark-reviewed` review action, apply promotion:

```powershell
python run_agent_analyst_learning_bridge.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite --apply
```

Share the `promotion_gate`, `sources`, `sources[].candidates`, `promoted_records`, and `recommendations` sections.

Interpretation:
- Dry-run is the default.
- Without a non-voided `mark_reviewed` action for the source Agent Lab report, promotion is blocked.
- Weak notes and duplicate note IDs are blocked by default.
- Each promoted learning record stores evidence-pack/profile/run metadata for later audit.

## 10i-e. Review-Approved Learning Loop

Preview the full review-approved learning path without recording review actions or learning rows:

```powershell
python run_agent_review_approved_learning.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite
```

Record an explicit review action after manually checking the profile Agent Lab report:

```powershell
python run_agent_review_approved_learning.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite --mark-reviewed --review-notes "Reviewed citations and accepted for pending outcome tracking."
```

Apply promotion only after the dry-run is promotable and the source has a real review action:

```powershell
python run_agent_review_approved_learning.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite --apply
```

Record a data gap instead of approving promotion:

```powershell
python run_agent_review_approved_learning.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite --needs-more-data "Add filings or transcript evidence before learning promotion." --review-notes "Current source is too thin."
```

Share the `loop_gate`, `pre_review_bridge`, `review_actions`, `final_bridge.promotion_gate`, `final_bridge.sources`, `context_performance.overall`, and `recommendations` sections.

Interpretation:
- This wraps `AnalystLearningPromotionBridge` in an auditable ceremony.
- It does not run the heavy pipeline and does not create trades.
- `--mark-reviewed` requires `--review-notes`.
- `--apply` still cannot write learning records unless bridge gates pass.
- `needs_more_data` blocks promotion until the review action is voided or the source is rerun with better evidence.

## 10i-f. Analyst Outcome Evaluation Loop

Dry-run evaluated outcomes for reviewed analyst learning records:

```powershell
python run_agent_analyst_outcome_loop.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --latest-processed-prices 1d
```

Use a specific local price artifact:

```powershell
python run_agent_analyst_outcome_loop.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --market-data-path data/dean_os/replay_prices/replay_prices_1d_normalized_20260612_073159.parquet
```

Apply only after the dry-run has valid ticker windows and mature horizons:

```powershell
python run_agent_analyst_outcome_loop.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --latest-processed-prices 1d --apply
```

Run early as a diagnostic only, not as production learning truth:

```powershell
python run_agent_analyst_outcome_loop.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --market-data-path data/dean_os/replay_prices/replay_prices_1d_normalized_20260612_073159.parquet --historical-diagnostic --as-of 2026-03-01T00:00:00+00:00
```

Share the `evaluation_gate`, `outcome_evaluation.status_counts`, `outcome_evaluation.evaluations`, `profile_outcomes`, `context_performance.overall`, and `recommendations` sections.

Interpretation:
- It evaluates only analyst learning records by default: metadata `analyst_learning_bridge=True`.
- Dry-run is the default.
- `--apply` writes outcome labels only through `OutcomeEvaluationRunner`.
- `--historical-diagnostic` is dry-run by default and should not be treated as production learning truth.
- `blocked_need_newer_prices` means local prices end before the thesis was created; do not force an outcome.

## 10i-g. Analyst Calibration Gate

Build proposal-only calibration guidance from profile scorecards and evaluated outcomes:

```powershell
python run_agent_analyst_calibration_gate.py --profile-scorecard-json reports/dean_os/analyst_profile_scorecard/latest.json --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite
```

Use relaxed thresholds for diagnostics only:

```powershell
python run_agent_analyst_calibration_gate.py --profile-scorecard-json reports/dean_os/analyst_profile_scorecard/latest.json --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --min-profile-runs 1 --min-completed-outcomes 1 --min-hit-rate 0.5 --max-miss-rate 0.5
```

Share the `summary`, `profiles.PROFILE.calibration_status`, `profiles.PROFILE.blockers`, `profiles.PROFILE.outcomes`, `context_performance.weak_contexts`, and `recommendations` sections.

Interpretation:
- This does not run analysts, evaluate prices, or change config.
- `ready_for_review` means "prepare a human-reviewed calibration proposal", not "auto-increase weight".
- `blocked` usually means scorecard/outcome evidence is not enough.
- `keep_candidate` means the profile has some evidence but should not become a default or higher-weight agent yet.

## 10i-h. Calibration Proposals

Create dry-run calibration proposals from a calibration gate report:

```powershell
python run_agent_calibration_proposals.py reports/dean_os/analyst_calibration_gate/latest.json --operations-store data/dean_os/operation_queue.sqlite
```

Enqueue proposed review items only after inspecting the dry-run report:

```powershell
python run_agent_calibration_proposals.py reports/dean_os/analyst_calibration_gate/latest.json --operations-store data/dean_os/operation_queue.sqlite --enqueue
```

Share the `proposal_gate`, `calibration_gate.summary`, `proposals`, `enqueued_proposal_ids`, and `recommendations` sections.

Interpretation:
- This does not change analyst weights, defaults, consensus, or production config.
- Without `--enqueue`, it only writes a report artifact.
- With `--enqueue`, proposals still enter `OperationQueue` as `proposed`, `dry_run=True`, and `requires_human_approval=True`.
- If all profiles are blocked, status should be `no_ready_profiles` and no proposal should be created.

## 10i-i. Calibration Review Lifecycle

Snapshot calibration proposals in the operation queue:

```powershell
python run_agent_calibration_review_lifecycle.py --operations-store data/dean_os/operation_queue.sqlite
```

Preview calibration proposal actions without changing config:

```powershell
python run_agent_calibration_review_lifecycle.py --operations-store data/dean_os/operation_queue.sqlite --dry-run-proposals
```

Mark an explicit calibration proposal approved or rejected in the queue:

```powershell
python run_agent_calibration_review_lifecycle.py --operations-store data/dean_os/operation_queue.sqlite --approve PROPOSAL_ID_HERE --dry-run-proposals
python run_agent_calibration_review_lifecycle.py --operations-store data/dean_os/operation_queue.sqlite --reject PROPOSAL_ID_HERE
```

Share the `lifecycle_gate`, `initial_status_counts`, `final_status_counts`, `action_results`, `dry_run_previews`, `approved_waiting_manual_implementation`, and `recommendations` sections.

Interpretation:
- This never writes production config, analyst defaults, or consensus weights.
- Approval only changes the queue proposal status to `approved`.
- Approved calibration proposals are still `approved_waiting_manual_implementation`.
- Non-calibration proposals are skipped unless `--include-non-calibration` is explicit.

## 10i-j. Manual Implementation Backlog

List approved calibration proposals waiting for manual implementation:

```powershell
python run_agent_manual_implementation_backlog.py --operations-store data/dean_os/operation_queue.sqlite
```

Include proposed/rejected items for visibility:

```powershell
python run_agent_manual_implementation_backlog.py --operations-store data/dean_os/operation_queue.sqlite --include-proposed --include-rejected
```

Share the `backlog_gate`, `status_counts`, `tasks`, `tasks[].implementation_checklist`, and `recommendations` sections.

Interpretation:
- This is read-only and never writes config, code, queue status, analyst defaults, or consensus weights.
- Approved proposals become `waiting_manual_implementation` tasks.
- Each task requires a separate branch/PR/config change and rollback note.
- `operation_queue_empty` or `no_manual_tasks_in_scope` means there is nothing approved to implement manually.

## 10i-k. Agent Learning Loop Runbook

Show the current safe analyst-learning loop position without running any stages:

```powershell
python run_agent_learning_loop_runbook.py
```

Use explicit artifacts when reviewing a smoke or historical run:

```powershell
python run_agent_learning_loop_runbook.py --evidence-pack-json reports/dean_os/analyst_evidence_pack/latest.json --analyst-profiles-json reports/dean_os/analyst_profiles/latest.json --profile-scorecard-json reports/dean_os/analyst_profile_scorecard/latest.json --learning-bridge-json reports/dean_os/analyst_learning_bridge/latest.json --review-approved-learning-json reports/dean_os/review_approved_learning/latest.json --outcome-evaluation-json reports/dean_os/analyst_outcome_evaluation/latest.json --calibration-gate-json reports/dean_os/analyst_calibration_gate/latest.json --calibration-proposals-json reports/dean_os/calibration_proposals/latest.json --calibration-review-json reports/dean_os/calibration_review_lifecycle/latest.json --manual-backlog-json reports/dean_os/manual_implementation_backlog/latest.json
```

Share the `summary.current_stage`, `summary.current_status`, `loop_position.stop_reason`, `loop_position.next_command`, `stages`, and `safety_contract` sections.

Interpretation:
- This is an operator runbook, not a runner for the whole chain.
- It never writes config, never runs the trading pipeline, and never accesses a broker.
- It points to the next safest command and explains why the loop is stopped.
- `manual_implementation_required` means open a separate manual PR/config change; do not auto-apply analyst weights.

## 10i-l. Analyst Loop Daily Check

Build a cheap daily operator check across the analyst loop, local market freshness, evidence coverage, scorecard state, and DEAN logs:

```powershell
python run_agent_analyst_loop_daily_check.py --tickers AMD NVDA --max-age-hours 24
```

Use explicit artifacts when reviewing a smoke run:

```powershell
python run_agent_analyst_loop_daily_check.py --evidence-pack-json reports/dean_os/analyst_evidence_pack/latest.json --analyst-profiles-json reports/dean_os/analyst_profiles/latest.json --profile-scorecard-json reports/dean_os/analyst_profile_scorecard/latest.json --learning-bridge-json reports/dean_os/analyst_learning_bridge/latest.json --review-approved-learning-json reports/dean_os/review_approved_learning/latest.json --outcome-evaluation-json reports/dean_os/analyst_outcome_evaluation/latest.json --calibration-gate-json reports/dean_os/analyst_calibration_gate/latest.json --calibration-proposals-json reports/dean_os/calibration_proposals/latest.json --calibration-review-json reports/dean_os/calibration_review_lifecycle/latest.json --manual-backlog-json reports/dean_os/manual_implementation_backlog/latest.json --tickers AMD NVDA --max-age-hours 24
```

Share the `summary.decision`, `summary.current_stage`, `summary.current_status`, `checks.learning_loop`, `checks.market_freshness`, `blockers`, `warnings`, and `operator_actions` sections.

Interpretation:
- This is the preferred first command before spending resources on new analyst/profile/promotion runs.
- It does not run analysts, evaluate outcomes, enqueue proposals, write config, run the pipeline, or access a broker.
- `blocked` usually means a safety gate did its job.
- `needs_operator_review` means read warnings before continuing.
- `safe_to_continue` still does not authorize apply/config/live actions.

## 10i-m. Analyst Review Inbox

Build a read-only inbox of Agent Lab/profile reports that need human review:

```powershell
python run_agent_analyst_review_inbox.py --learning-bridge-json reports/dean_os/analyst_learning_bridge/latest.json --profile-run-json reports/dean_os/analyst_profiles/latest.json
```

Use isolated smoke artifacts:

```powershell
python run_agent_analyst_review_inbox.py --learning-bridge-json reports/dean_os/analyst_learning_bridge_smoke/latest.json --profile-run-json reports/dean_os/analyst_profiles_real_smoke/latest.json --review-actions-store reports/dean_os/analyst_learning_bridge_smoke/review_actions.sqlite --learning-store reports/dean_os/analyst_learning_bridge_smoke/learning.sqlite --operations-store reports/dean_os/analyst_learning_bridge_smoke/operation_queue.sqlite --output-dir reports/dean_os/analyst_review_inbox_smoke
```

Share the `summary.status`, `groups.ready_for_manual_review`, `groups.needs_more_data_candidate`, `groups.not_reviewable_yet`, `items[].suggested_commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never records review actions.
- `ready_for_manual_review` means inspect citations/thesis manually before any `mark-reviewed`.
- `needs_more_data_candidate` means do not approve; improve evidence or request data.
- `not_reviewable_yet` includes already reviewed, missing report, or structurally incomplete sources.
- Suggested commands are previews, not executed actions.

## 10i-n. Review Decision Packet

Build a compact read-only packet for one inbox source before deciding `mark-reviewed` vs `needs-more-data`:

```powershell
python run_agent_review_decision_packet.py --inbox-json reports/dean_os/analyst_review_inbox/latest.json
```

Use the latest smoke inbox:

```powershell
python run_agent_review_decision_packet.py --inbox-json reports/dean_os/analyst_review_inbox_smoke/latest.json --output-dir reports/dean_os/review_decision_packet_smoke
```

Share the `summary.packet_status`, `summary.recommended_review_action`, `source`, `evidence_pack`, `notes`, `review_checks`, `decision_guidance`, and `source.suggested_commands` sections.

Interpretation:
- This is read-only and never records review actions.
- `reviewable` means a source is a candidate for `mark-reviewed` only after manual citation/thesis inspection.
- `manual_review_with_warnings` means the operator must decide whether warnings are acceptable or should become `needs-more-data`.
- `needs_more_data_recommended` means do not approve yet.
- `mark-reviewed` only allows pending learning promotion; it is not a trade signal and does not change weights.

## 10i-o. Review Action Dry Run

Preview a review action from a decision packet without writing it:

```powershell
python run_agent_review_action_dry_run.py --packet-json reports/dean_os/review_decision_packet/latest.json --intent needs_more_data --review-notes "Evidence coverage is too thin." --data-request "Add missing ticker/source coverage before learning promotion."
```

Preview `mark_reviewed` only after warnings are explicitly accepted:

```powershell
python run_agent_review_action_dry_run.py --packet-json reports/dean_os/review_decision_packet/latest.json --intent mark_reviewed --review-notes "Reviewed citations and accepted warnings for diagnostic pending learning." --acknowledge-warnings
```

Share the `summary.dry_run_status`, `summary.can_record_review_action`, `validation`, `would_record_review_action`, `commands`, `bridge_expectation`, and `recommendations` sections.

Interpretation:
- This is read-only and never records review actions.
- `blocked_warning_ack_required` means do not record `mark_reviewed` unless the operator explicitly accepts packet warnings.
- `needs_more_data` is the safer action when evidence coverage is partial or tickers are missing.
- The `--apply` command shown in the report is only a later gated step after bridge dry-run passes.

## 10i-p. Review Action Apply Ceremony

Validate and optionally record exactly one review action from a dry-run artifact:

```powershell
python run_agent_review_action_apply_ceremony.py --dry-run-json reports/dean_os/review_action_dry_run/latest.json
```

Record the action only when the dry-run is acceptable and the operator explicitly chooses to write it:

```powershell
python run_agent_review_action_apply_ceremony.py --dry-run-json reports/dean_os/review_action_dry_run/latest.json --apply-review-action
```

Use an isolated store for smoke checks:

```powershell
python run_agent_review_action_apply_ceremony.py --dry-run-json reports/dean_os/review_action_dry_run_needs_more_data_smoke/latest.json --review-actions-store reports/dean_os/review_action_apply_ceremony_needs_more_data_smoke/review_actions.sqlite --operations-store reports/dean_os/review_action_apply_ceremony_needs_more_data_smoke/operation_queue.sqlite --log-path reports/dean_os/review_action_apply_ceremony_needs_more_data_smoke/events.jsonl --output-dir reports/dean_os/review_action_apply_ceremony_needs_more_data_smoke --apply-review-action
```

Share the `summary.apply_status`, `summary.review_action_write_performed`, `validation`, `recorded_action`, `commands`, and `recommendations` sections.

Interpretation:
- Without `--apply-review-action`, this command never records the review action.
- With `--apply-review-action`, it records one review action only when the dry-run is recordable and no active duplicate action exists.
- It never writes learning records, enqueues proposals, changes config, runs the pipeline, or accesses a broker.
- After a `needs_more_data` action, the source should remain blocked until the requested data is added or the action is voided.

## 10i-q. Evidence Gap Resolution Plan

Turn an active `needs_more_data` review action into concrete source/data tasks:

```powershell
python run_agent_evidence_gap_plan.py --review-action-json reports/dean_os/review_action_apply_ceremony/latest.json --decision-packet-json reports/dean_os/review_decision_packet/latest.json
```

Use the latest isolated smoke artifacts:

```powershell
python run_agent_evidence_gap_plan.py --review-action-json reports/dean_os/review_action_apply_ceremony_needs_more_data_smoke/latest.json --decision-packet-json reports/dean_os/review_decision_packet_smoke/latest.json --output-dir reports/dean_os/evidence_gap_resolution_plan_smoke
```

If the plan says cached tables were truncated, rerun the evidence pack with a larger row window:

```powershell
python run_agent_analyst_evidence_pack.py --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tickers AAPL AMD MSFT NVDA TSM --sectors semiconductor --tags ai_cycle --max-rows-per-table 200 --output-dir reports/dean_os/analyst_evidence_pack_refreshed_gap_check
```

Share the `summary`, `current_coverage`, `resolution_tasks`, `acceptance_criteria`, `commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never fetches data, records review actions, writes learning records, changes config, runs the pipeline, or accesses a broker.
- `ready_to_collect` means the current `needs_more_data` action has concrete source/data tasks.
- If the refreshed evidence pack removes missing tickers, rebuild profiles, inbox, and decision packet before changing review state.
- A clean refreshed decision packet can then go through `ReviewActionDryRun` and `ReviewActionApplyCeremony`.

## 10i-r. Analyst Learning Apply Ceremony

Apply pending analyst learning records only from a validated learning-bridge dry-run:

```powershell
python run_agent_learning_apply_ceremony.py --bridge-dry-run-json reports/dean_os/analyst_learning_bridge/latest.json
```

Write learning records only with an explicit apply flag:

```powershell
python run_agent_learning_apply_ceremony.py --bridge-dry-run-json reports/dean_os/analyst_learning_bridge_refreshed_mark_reviewed_check/latest.json --learning-store reports/dean_os/analyst_learning_apply_ceremony_apply_smoke/learning.sqlite --review-actions-store reports/dean_os/review_action_apply_ceremony_refreshed_mark_reviewed_check/review_actions.sqlite --operations-store reports/dean_os/analyst_learning_apply_ceremony_apply_smoke/operation_queue.sqlite --output-dir reports/dean_os/analyst_learning_apply_ceremony_apply_smoke --apply-learning
```

Share the `summary`, `validation`, `sources`, `promoted_records`, `commands`, and `recommendations` sections.

Interpretation:
- Without `--apply-learning`, this command never writes learning records.
- With `--apply-learning`, it writes pending learning records only when the saved bridge dry-run is `dry_run_ready`, review actions are still active, and note ids are not duplicates.
- It never records review actions, enqueues proposals, changes config, runs the pipeline, or accesses a broker.
- Pending learning records are not calibration changes; outcomes must mature and pass outcome evaluation first.

## 10i-s. Outcome Readiness Gate

Check whether pending analyst learning records are ready for outcome evaluation:

```powershell
python run_agent_outcome_readiness.py --learning-store reports/dean_os/analyst_learning_apply_ceremony_apply_smoke/learning.sqlite --market-data-path data/colab/backup_20260510_153551/stage2_prices_1d_20260505_151233.parquet --tickers AAPL AMD MSFT NVDA TSM
```

Share the `summary`, `readiness_gate`, `pending_records`, `profile_readiness`, `dry_run_outcome_evaluation`, `commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never updates outcomes.
- `ready_for_outcome_dry_run` means run `run_agent_analyst_outcome_loop.py` in dry-run mode, then inspect before any apply.
- `waiting_for_horizon` means prices exist but the thesis horizon has not elapsed.
- `blocked_need_newer_prices` means market data ends before the learning record was created.
- Pending records must not change calibration/profile weights until outcome evaluation is mature, applied, and reviewed.

## 10i-t. Outcome Price Coverage Plan

Convert an outcome-readiness blocker into concrete local price coverage requirements:

```powershell
python run_agent_outcome_price_coverage.py --readiness-json reports/dean_os/outcome_readiness_gate/latest.json
```

Use the latest isolated smoke readiness artifact:

```powershell
python run_agent_outcome_price_coverage.py --readiness-json reports/dean_os/outcome_readiness_gate_smoke/latest.json --output-dir reports/dean_os/outcome_price_coverage_plan_smoke
```

Share the `summary`, `coverage_targets`, `market_data_snapshot`, `ticker_coverage`, `coverage_tasks`, `commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never fetches prices, updates outcomes, writes learning records, changes config, runs the pipeline, or accesses a broker.
- `needs_price_refresh_after_record_creation` means local prices exist but end before pending learning records were created.
- `waiting_for_outcome_horizon` means prices exist after creation, but production outcome labels still need the configured horizon to elapse.
- `coverage_ready_for_outcome_readiness_rerun` means rerun `OutcomeReadinessGate` before any outcome apply ceremony.
- For the current smoke, AAPL/AMD/MSFT/NVDA/TSM need prices after `2026-06-13T07:06:38.358225+00:00`; production outcomes are due around `2027-06-13`.

## 10i-u. Market Data Refresh Runbook

Build a read-only runbook for clearing outcome price coverage blockers:

```powershell
python run_agent_market_data_refresh_runbook.py --coverage-plan-json reports/dean_os/outcome_price_coverage_plan/latest.json --collector-inventory-json reports/dean_os/collector_inventory/latest.json
```

Use the latest isolated smoke artifacts:

```powershell
python run_agent_collector_inventory.py --output reports/dean_os/collector_inventory/latest.json
python run_agent_market_data_refresh_runbook.py --coverage-plan-json reports/dean_os/outcome_price_coverage_plan_smoke/latest.json --collector-inventory-json reports/dean_os/collector_inventory/latest.json --output-dir reports/dean_os/market_data_refresh_runbook_smoke
```

Share the `summary`, `requirements`, `price_feed_candidates`, `known_price_artifacts`, `refresh_options`, `operator_tasks`, `commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never runs collectors, network calls, config writes, pipeline stages, outcome writes, or broker actions.
- `refresh_runbook_ready` means an enabled local price feed exists, but the operator still must create or provide a refreshed local price artifact.
- The preferred path is a separate refreshed CSV/parquet file, then freshness -> outcome readiness -> price coverage recheck.
- Do not overwrite stale price artifacts until the refreshed artifact passes checks.
- For the current smoke, the primary price feed candidate is `yahoo_finance`, but `can_refresh_automatically=false`.

## 10j. Historical Replay

Run a safe old-data replay. The analyst sees only data at or before `--as-of`; future prices are used only after the thesis is formed, for evaluation:

```powershell
python run_agent_historical_replay.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --as-of 2026-03-01T00:00:00+00:00 --lookback-days 180 --horizon-days 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --normalize-daily-bars
```

Share the `decision`, `report.thesis`, `historical_replay.coverage.price_quality`, `historical_replay.rankings`, `evaluation`, and `recommendations` sections.

Interpretation:
- This is not paper trading and does not write to `paper_trades.sqlite`.
- This does not run the heavy pipeline.
- `--normalize-daily-bars` collapses multiple rows per ticker/day into one OHLCV bar before scoring.
- If `price_quality.warnings` is non-empty, treat hit/miss as diagnostic, not as learning truth.

## 10j-a. Historical Research Replay

Run the combined old-data research exam. It builds a pre-`as-of` evidence pack
from cached news/macro/materials, runs Agent Lab in isolated stores, then attaches
post-hoc price outcome evaluation:

```powershell
python run_agent_historical_research_replay.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --as-of 2026-03-01T00:00:00+00:00 --lookback-days 180 --horizon-days 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tags historical_replay ai_cycle raw_period_check --normalize-daily-bars --output-dir reports\dean_os\historical_research_replay_20260301
```

Share the `research_exam`, `evidence_pack.coverage`, `agent_lab.summary`,
`price_replay.decision`, `price_replay.evaluation`, `price_replay.quality_warnings`,
and `recommendations` sections.

Interpretation:
- This is an agent reasoning exam, not paper trading and not a live recommendation.
- It creates no learning records and no operation proposals.
- It writes only isolated report/corpus files under the chosen output directory.
- If `research_exam.learning_gate.status` is `blocked`, do not promote the miss/hit into learning memory.
- If evidence dates look collapsed or suspicious, audit source timestamp columns before running a large replay batch.

## 10j-b. Evidence Timestamp Audit

Audit cached news/macro/material timestamps before scaling historical research replay:

```powershell
python run_agent_evidence_timestamp_audit.py --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --evidence-pack-json reports\dean_os\historical_research_replay_20260301_filtered\runs\historical_research_replay_20260613T125906_753943+0000\evidence_pack\latest.json --start-at 2025-09-02T00:00:00+00:00 --as-of 2026-03-01T00:00:00+00:00 --output-dir reports\dean_os\evidence_timestamp_audit_20260301_filtered_v2
```

Share the `summary`, `source_audits`, `evidence_pack_audit`, and `recommendations` sections.

Interpretation:
- Raw source files may contain rows after `as-of`; that is acceptable only if a usable timestamp column exists and the evidence pack filters them out.
- `timestamp_ready` means replay can run as a diagnostic exam, not that results may be promoted to learning memory.
- `timestamp_suspicious` or `timestamp_blocked` means do not scale old-data research replay until source dates are reviewed.
- `AnalystEvidencePackRunner` recognizes `published_date` and related publication-date columns; keep this guarded by tests.

## 10j-c. Historical Research Replay Batch

Run multiple old-data research replay slices:

```powershell
python run_agent_historical_research_replay_batch.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --as-of 2026-03-01T00:00:00+00:00 2026-04-01T00:00:00+00:00 --lookback-days 180 --horizon-days 30 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tags historical_replay ai_cycle published_date_fixed mini_batch --normalize-daily-bars --output-dir reports\dean_os\historical_research_replay_batch_202603_202604
```

Share the `summary`, `learning_gate`, `runs`, `summary.by_research_stance`,
`summary.by_price_ticker`, `summary.quality_warnings`, and `recommendations` sections.

Interpretation:
- This is still an exam, not paper trading and not analyst-weight calibration.
- If `quality_blocked_runs` is non-zero, fix price-quality warnings before trusting hit rates.
- If `weak_evidence_runs` is non-zero, backfill sources or narrow the ticker universe before judging analyst skill.
- Neutral/mixed research can be the correct answer when evidence is thin.

## 10j-d. Replay Price Quality Investigation

Investigate replay price-quality blockers without mutating data:

```powershell
python run_agent_replay_price_quality_investigation.py --report-json reports\dean_os\replay_price_normalizer\latest.json --report-json reports\dean_os\historical_replay_batch\latest.json --report-json reports\dean_os\historical_research_replay_batch_202603_202604\latest.json --report-json reports\dean_os\historical_research_replay_20260301_filtered\latest.json --price-data data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --price-data data\dean_os\replay_prices\replay_prices_1d_normalized_20260612_073159.parquet --output-dir reports\dean_os\replay_price_quality_investigation_current
```

Share the `summary`, `warning_summary`, `hypotheses`, `artifact_diagnostics`,
`window_diagnostics`, `operator_tasks`, and `recommendations` sections.

Interpretation:
- This is read-only and does not repair or delete price rows.
- `blocked_price_quality` means replay hit/miss is diagnostic only.
- Large SPY one-step moves usually indicate adjusted/unadjusted mixing, bad cached rows, split/dividend handling problems, or interval aggregation mistakes.
- Do not clear learning gates until the same windows are clean after a refreshed or repaired price artifact.

## 10j-e. Replay Price Artifact Repair

Create a non-destructive candidate repaired artifact after investigation finds mixed daily/intraday-like rows:

```powershell
python run_agent_replay_price_artifact_repair.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --benchmark-ticker SPY --write-artifact --output-dir reports\dean_os\replay_price_artifact_repair_current --artifact-dir data\dean_os\replay_prices
python run_agent_replay_price_quality_investigation.py --artifact-only --price-data data\dean_os\replay_prices\replay_prices_1d_repaired_20260613_135839.parquet --benchmark-ticker SPY --output-dir reports\dean_os\replay_price_quality_investigation_repaired_artifact_only_v2
```

Share the `summary`, `artifact`, `quality`, `quarantine`, `learning_gate`,
`artifact_diagnostics`, and `recommendations` sections.

Interpretation:
- This never mutates the source cache; it writes a new candidate artifact only.
- Use `--artifact-only` when checking the repaired parquet, otherwise older default reports can keep old warnings in the investigation summary.
- A clean repaired artifact permits diagnostic replay expansion, not learning promotion by itself.
- If `learning_gate.status` is `candidate_ready_for_replay_review`, run replay batches next and check sample size/evidence quality.

## 10j-f. Replay Calibration Readiness Gate

Check whether repaired replay evidence is ready for manual analyst-calibration review:

```powershell
python run_agent_replay_calibration_readiness.py --replay-batch-json reports\dean_os\historical_replay_batch_repaired_expanded\latest.json --research-batch-json reports\dean_os\historical_research_replay_batch_repaired_expanded_step14\latest.json --output-dir reports\dean_os\replay_calibration_readiness_gate_after_step14_research
```

If expanded batches do not exist yet, create them first:

```powershell
python run_agent_historical_replay_batch.py data\dean_os\replay_prices\replay_prices_1d_repaired_20260613_135839.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --start-as-of 2025-09-01T00:00:00+00:00 --end-as-of 2026-03-01T00:00:00+00:00 --step-days 14 --lookback-days 180 --horizon-days 30 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --output-dir reports\dean_os\historical_replay_batch_repaired_expanded
python run_agent_historical_research_replay_batch.py data\dean_os\replay_prices\replay_prices_1d_repaired_20260613_135839.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --start-as-of 2025-09-01T00:00:00+00:00 --end-as-of 2026-03-01T00:00:00+00:00 --step-days 14 --lookback-days 180 --horizon-days 30 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tags historical_replay ai_cycle repaired_price_artifact expanded_batch step14 --output-dir reports\dean_os\historical_research_replay_batch_repaired_expanded_step14
```

Share the `summary`, `gate`, `checks`, `commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never writes learning records, calibration proposals, config, pipeline outputs, or broker actions.
- `ready_for_manual_review` means prepare a human review packet, not auto-calibrate.
- `need_evidence_backfill` means replay mechanics are clean enough, but analysts need better pre-`as_of` evidence before skill can be judged.
- Current expanded run status: price quality clean, 26 clean price replay evaluations, 13 clean research replay price evaluations, but all 13 research runs are weak evidence and inconclusive.

## 10j-g. Historical Evidence Backfill Plan

Build a read-only source-task plan when replay calibration readiness is blocked by weak historical evidence:

```powershell
python run_agent_historical_evidence_backfill.py --readiness-json reports\dean_os\replay_calibration_readiness_gate_after_step14_research\latest.json --research-batch-json reports\dean_os\historical_research_replay_batch_repaired_expanded_step14\latest.json --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --output-dir reports\dean_os\historical_evidence_backfill_plan_current
```

Share the `summary`, `coverage_gaps`, `source_audits`, `backfill_tasks`, `commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never runs collectors, network calls, pipeline, learning writes, config writes, or broker actions.
- `backfill_required` means analyst calibration is blocked by source coverage, not by price replay mechanics.
- Current audit: cached news starts `2026-02-25`, cached macro starts `2026-03-01`, while weak replay windows are `2025-09-01` through `2026-02-16`.
- Either provide historical evidence before those dates or select replay windows where evidence exists.

## 10j-h. Replay Evidence Window Selector

Select replay dates where the repaired price artifact, future outcome horizon, and pre-`as_of` evidence coverage overlap:

```powershell
python run_agent_replay_evidence_windows.py --price-data data\dean_os\replay_prices\replay_prices_1d_repaired_20260613_135839.parquet --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --lookback-days 180 --horizon-days 30 --step-days 7 --output-dir reports\dean_os\replay_evidence_window_selector_current
```

Run the selected-window research replay after reviewing the generated command:

```powershell
python run_agent_historical_research_replay_batch.py data\dean_os\replay_prices\replay_prices_1d_repaired_20260613_135839.parquet --tickers AAPL AMD MSFT NVDA QQQ SPY TSM --as-of 2026-03-04T00:00:00+00:00 2026-03-11T00:00:00+00:00 2026-03-18T00:00:00+00:00 2026-03-25T00:00:00+00:00 2026-04-01T00:00:00+00:00 --lookback-days 180 --horizon-days 30 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tags historical_replay ai_cycle repaired_price_artifact evidence_window_selected --output-dir reports\dean_os\historical_research_replay_batch_evidence_window_selected
```

Share the `summary`, `source_coverage`, `eligible_windows`, `rejected_windows_sample`, `commands`, `runs`, `learning_gate`, and `recommendations` sections.

Interpretation:
- This is read-only and never runs collectors, network calls, pipeline, learning writes, config writes, or broker actions.
- Current selector result: 5 eligible dates from `2026-03-04` through `2026-04-01`; `2026-02-25` is rejected because evidence rows are still zero before `as_of`.
- Current selected research replay after the directionality rule fix: 5 evaluated, 0 price-quality blocks, hit rate `0.8`, stance counts `constructive=4` and `mixed=1`, but 2 early windows are still partial evidence.
- Do not calibrate analyst weights from this yet; diagnose evidence coverage and ticker-specific attribution first.

## 10j-i. Research Replay Directionality Diagnostic

Diagnose why selected-window research replay is still blocked after evidence-window selection:

```powershell
python run_agent_research_replay_directionality.py --research-batch-json reports\dean_os\historical_research_replay_batch_evidence_window_selected_after_directionality_fix\latest.json --readiness-json reports\dean_os\replay_calibration_readiness_gate_after_directionality_fix\latest.json --backfill-plan-json reports\dean_os\historical_evidence_backfill_plan_after_directionality_fix\latest.json --output-dir reports\dean_os\research_replay_directionality_diagnostic_after_fix
```

Share the `summary`, `issue_counts`, `run_diagnostics`, `diagnostic_tasks`, `commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never runs collectors, network calls, pipeline, learning writes, config writes, or broker actions.
- Current post-fix diagnostic result: 4 directional runs, 1 inconclusive strong run, missing tickers `AAPL` and `QQQ`.
- Research directionality now passes readiness with ratio `0.8`; the current blockers are evidence coverage and ticker-specific attribution.
- All selected runs are still `basket_or_sector`, so do not change analyst weights until ticker-specific support is audited.

## 10j-j. Ticker-Specific Attribution Audit

Audit whether selected-window directional research is backed by direct evidence for the ticker chosen by price replay:

```powershell
python run_agent_ticker_attribution_audit.py --research-batch-json reports\dean_os\historical_research_replay_batch_evidence_window_selected_after_directionality_fix\latest.json --output-dir reports\dean_os\ticker_specific_attribution_audit_current
```

Share the `summary`, `issue_counts`, `run_audits`, `tasks`, `commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never runs collectors, network calls, pipeline, learning writes, config writes, or broker actions.
- Current result: `blocked_weak_ticker_evidence`, 5 audited runs, 0 ticker-ready, 5 basket-note runs, and 2 weak direct-evidence runs.
- Early `TSM` windows have only 1 direct document; later `TSM`/`AMD` windows have more direct documents but still rely on 7-ticker basket notes.
- Do not calibrate analyst weights until a ticker-focused note-selection/building layer replaces broad basket notes for ticker-level conclusions.

## 10j-k. Ticker-Focused Research Notes

Build ticker-focused note candidates from the same pre-`as_of` evidence packs after the price-selected ticker is known:

```powershell
python run_agent_ticker_focused_notes.py --research-batch-json reports\dean_os\historical_research_replay_batch_evidence_window_selected_after_directionality_fix\latest.json --output-dir reports\dean_os\ticker_focused_research_notes_current
```

Share the `summary`, `focused_notes`, `issue_counts`, `tasks`, `commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never runs collectors, network calls, pipeline, learning writes, config writes, or broker actions.
- Current result: `partial_focused_notes_ready`, 5 runs processed, 3 focused-note-ready, and 2 weak direct-evidence early `TSM` windows.
- Ready focused notes are review artifacts only until the historical replay exam explicitly consumes them.
- Next safe step is a replay-exam bridge/overlay, not analyst calibration.

## 10j-l. Ticker-Focused Replay Exam Bridge

Compare original basket-note replay exams with ticker-focused replay-exam overlays:

```powershell
python run_agent_ticker_focused_replay_bridge.py --research-batch-json reports\dean_os\historical_research_replay_batch_evidence_window_selected_after_directionality_fix\latest.json --focused-notes-json reports\dean_os\ticker_focused_research_notes_current\latest.json --output-dir reports\dean_os\ticker_focused_replay_exam_bridge_current
```

Share the `summary`, `run_overlays`, `issue_counts`, `tasks`, `commands`, and `recommendations` sections.

Interpretation:
- This is read-only and never runs collectors, network calls, pipeline, learning writes, config writes, or broker actions.
- Current result: `partial_focused_overlay_ready`, 5 runs compared, 3 overlay-ready, 2 blocked early `TSM` overlays.
- Focused directional runs are 2, because `AMD` on `2026-04-01` stays mixed/neutral instead of being forced bullish.
- Next safe step is optional runner integration that preserves original basket-note exam fields for audit comparison.

## 10j-m. Historical Research Replay With Focused Overlay

Rerun selected-window historical research replay with reviewed ticker-focused overlay applied:

```powershell
python run_agent_historical_research_replay_batch.py data\dean_os\replay_prices\replay_prices_1d_repaired_20260613_135839.parquet --tickers AAPL AMD MSFT NVDA QQQ SPY TSM --as-of 2026-03-04T00:00:00+00:00 2026-03-11T00:00:00+00:00 2026-03-18T00:00:00+00:00 2026-03-25T00:00:00+00:00 2026-04-01T00:00:00+00:00 --lookback-days 180 --horizon-days 30 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tags historical_replay ai_cycle repaired_price_artifact evidence_window_selected directionality_rule_fix focused_overlay_integration --focused-overlay-json reports\dean_os\ticker_focused_replay_exam_bridge_current\latest.json --apply-focused-overlay --output-dir reports\dean_os\historical_research_replay_batch_focused_overlay_integration_current
```

Then rerun attribution and readiness on the focused-overlay batch:

```powershell
python run_agent_ticker_attribution_audit.py --research-batch-json reports\dean_os\historical_research_replay_batch_focused_overlay_integration_current\latest.json --output-dir reports\dean_os\ticker_specific_attribution_audit_after_focused_overlay_integration
python run_agent_replay_calibration_readiness.py --replay-batch-json reports\dean_os\historical_replay_batch_repaired_expanded\latest.json --research-batch-json reports\dean_os\historical_research_replay_batch_focused_overlay_integration_current\latest.json --output-dir reports\dean_os\replay_calibration_readiness_after_focused_overlay_integration
```

Share the integrated batch `summary`, `runs[].focused_overlay_status`, `runs[].focused_overlay_applied`, attribution `summary`, and readiness `gate/checks`.

Interpretation:
- This does not run collectors, network calls, the heavy pipeline, learning writes, config writes, or broker actions.
- Current integrated batch result: 5 evaluated, hit rate `0.8`, `constructive=2`, `insufficient_data=2`, `mixed=1`.
- Current overlay-aware attribution result: `ticker-ready=3`, `basket-note=0`, `weak direct evidence=2`.
- Current readiness result: `need_more_research_replay_samples`, blocked by `research_sample` and `evidence_coverage`.

## 10j-n. Sector Thesis To Ticker Basket Bridge

Convert a sector/domain thesis into reviewed ticker candidates without pretending that sector evidence is direct ticker evidence:

```powershell
python run_agent_sector_to_ticker_bridge.py --research-batch-json reports\dean_os\historical_research_replay_batch_focused_overlay_integration_current\latest.json --domain-profile semiconductor_ai_infrastructure --sector semiconductor --output-dir reports\dean_os\sector_thesis_to_ticker_basket_current
```

Share the `summary`, `domain_analyst_contract`, `sector_thesis`, `ticker_candidates`, `blocked_or_limited_candidates`, `tasks`, and `recommendations` sections.

Interpretation:
- This is read-only and never runs collectors, network calls, the heavy pipeline, learning writes, config writes, analyst weight changes, recommendations, or broker actions.
- A sector thesis can propose a basket or candidate list, but it is not a direct ticker thesis until the ticker has its own evidence.
- Current result: `partial_basket_ready`, `sector_stance=evidence_limited`, candidates `AMD` and `TSM`.
- `AMD` has 2 overlay-ready windows; one is constructive and one remains neutral/mixed.
- `TSM` has 1 overlay-ready window and 2 blocked early windows, so the bridge stays partial even though `TSM` can appear as a reviewed candidate.
- Do not clone more sector analysts or change analyst weights from this result. Build a review packet or backfill direct evidence first.

Suggested next local command after the review-packet module exists:

```powershell
python run_agent_sector_to_ticker_review_packet.py --bridge-json reports\dean_os\sector_thesis_to_ticker_basket_current\latest.json --output-dir reports\dean_os\sector_to_ticker_review_packet_current
```

If the review-packet module does not exist yet, implement it as JSON/Markdown-only and keep the same no-write/no-trading boundaries.

## 10j-o. Source Evidence Validation Gate

Validate the source artifact before domain-specialist review. This is the local integration of the useful boundary idea from the staged web/draft 245 work:

```powershell
python run_agent_source_evidence_validation_gate.py --source-json reports\dean_os\analyst_evidence_pack_refreshed_gap_check\latest.json --output-dir reports\dean_os\source_evidence_validation_gate_current
```

Share the `summary`, `candidate_routing_indexes`, `decision_guidance`, `validation_checks` counts, `safety_assertions`, and `recommendations` sections.

Interpretation:
- This is read-only and never runs collectors, connector fetches, network calls, learning writes, config writes, recommendations, or broker actions.
- Draft normalized packet fixtures are accepted only for staged contract review and must not be promoted into production evidence.
- Real-source normalized packets from `run_agent_real_source_normalized_packet.py` are accepted as review-only source artifacts via `normalized_packet_rows`.
- Current real result: `source_evidence_ready_with_warnings`, 158 documents, 5 candidate entities, 321 pass, 111 warn, 0 fail.
- The current warning theme is missing per-document `published_at`; domain review can proceed manually, but evidence promotion and extraction stay blocked.
- Run this gate before reviewing the domain-specialist packet.

## 10j-p. Domain Specialist Review Packet

Build the domain-first review packet after the source gate and sector-to-ticker bridge are available:

```powershell
python run_agent_domain_specialist_review_packet.py --bridge-json reports\dean_os\sector_thesis_to_ticker_basket_current\latest.json --source-gate-json reports\dean_os\source_evidence_validation_gate_current\latest.json --output-dir reports\dean_os\domain_specialist_review_packet_current
```

Share the `summary`, `domain_review_boundary`, `domain_thesis`, `source_evidence_context`, `claims_events_entities`, `sector_exposure_map`, `ticker_candidate_bridge`, and `recommendations` sections.

Interpretation:
- This packet is domain-first: AMD/TSM are pilot entities and exposure nodes, not the architecture axis.
- `--source-gate-json` attaches source validation status as explicit `source_evidence_context`.
- Ticker mapping remains a separate bridge and requires direct ticker evidence.
- Current real result: `domain_review_ready_with_limitations`, candidate entities `AMD` and `TSM`, standardization still blocked by source warnings plus bridge limitations.
- Do not clone additional domain profiles until one template is manually accepted.

## 10j-q. Source Extraction Review Packet

Build the review-only extraction contract after the source gate and domain packet are available:

```powershell
python run_agent_source_extraction_review_packet.py --source-json reports\dean_os\analyst_evidence_pack_refreshed_gap_check\latest.json --source-gate-json reports\dean_os\source_evidence_validation_gate_current\latest.json --domain-packet-json reports\dean_os\domain_specialist_review_packet_current\latest.json --output-dir reports\dean_os\source_extraction_review_packet_current
```

Share the `summary`, `extraction_contract`, `candidate_routing_indexes`, `source_anchor_plan`, `extraction_work_queue` samples, `review_checks`, and `recommendations` sections.

Interpretation:
- This is contract definition only; it does not execute extraction or emit claims/events/entities.
- Every future candidate claim, event, entity mention, topic, sector, asset, and financial implication must carry a source anchor.
- Financial implication candidates are not recommendations, ratings, price targets, allocation advice, or trade signals.
- Real-source normalized packets are supported as input; quarantined source units are carried as explicit extraction blockers.
- Current real result: `extraction_contract_ready_with_warnings`, 158 source units, 5 candidate entities, 111 missing timestamps, 10 pass, 3 warn, 0 fail.
- Fixture-only extraction candidate shape now exists in the next checklist step; keep it staged/review-only.

## 10j-r. Source Extraction Fixture Packet

Build a fixture-only extraction candidate packet over a small reviewed subset:

```powershell
python run_agent_source_extraction_fixture_packet.py --contract-json reports\dean_os\source_extraction_review_packet_current\latest.json --max-items 12 --output-dir reports\dean_os\source_extraction_fixture_packet_current
```

Share the `summary`, `fixture_boundary`, `selected_source_anchors`, candidate fixture samples, `review_checks`, and `recommendations` sections.

Interpretation:
- This materializes candidate output shapes only; it is not production extraction.
- Candidate claim/event/entity/financial implication fixtures are not evidence.
- Fixture text may mirror source previews only to test anchoring and required fields.
- Current real result: `extraction_fixture_ready_with_warnings`, 12 selected anchors, 12 claim fixtures, 12 event fixtures, 12 entity fixtures, 12 financial implication fixtures, 11 pass, 2 warn, 0 fail.
- Selected anchors are entity-bearing but timestamp-limited; do not use event chronology without timestamp repair or explicit limitation review.
- No evidence promotion, learning writes, recommendations, allocation, paper trading, or live trading.

## 10j-s. Source Extraction Fixture Review Gate

Review the fixture-only extraction candidate shape before any real extractor implementation:

```powershell
python run_agent_source_extraction_fixture_review_gate.py --fixture-json reports\dean_os\source_extraction_fixture_packet_current\latest.json --output-dir reports\dean_os\source_extraction_fixture_review_gate_current
```

Share the `summary`, `fixture_shape_review`, `timestamp_review`, `review_checks`, and `recommendations` sections.

Interpretation:
- This is a review gate only; it does not execute extraction.
- A reviewable fixture shape is not evidence and does not authorize real extraction, learning promotion, recommendation, allocation, paper trading, or live trading.
- Current real result: `fixture_review_ready_with_warnings`, shape reviewable, candidate anchor links valid, evidence boundary disabled, 12/12 selected anchors missing timestamps, 12 pass, 2 warn, 0 fail.
- `can_standardize_fixture_shape=false` until timestamp limitations are repaired or explicitly accepted.
- Next safe work is timestamp strategy for entity-bearing news rows or manual acceptance of timestamp-limited fixture shape.

## 10j-t. Real Source Dropzone Inventory

Inventory operator-supplied local research files before normalizing them:

```powershell
python run_agent_real_source_dropzone_inventory.py --dropzone docs\research --output-dir reports\dean_os\real_source_dropzone_inventory_current
```

Share the `summary`, `supported_files`, `unsupported_files`, `commands`, and `recommendations` sections.

Interpretation:
- This reads file metadata only; it does not read research content.
- `ready_for_operator_source_review` means at least one supported file is ready for review-only normalization.
- `empty_dropzone` means add one local research file before running real-source normalization.
- No live fetch, extraction, evidence promotion, learning write, recommendation, allocation, paper trading, or live trading is authorized.

## 10j-u. Real Source Normalized Packet

Build a review-only normalized packet from an operator-supplied local source file:

```powershell
python run_agent_real_source_normalized_packet.py docs\research\YOUR_FILE.md --source-type report --ticker AMD --sector semiconductors --tag semiconductor_supply_chain --output-dir reports\dean_os\real_source_normalized_packet_current
```

Share the `summary`, first `normalized_packet_rows` item, `quarantine_partitions`, `routing_prefilter`, and `output_boundary` sections.

Interpretation:
- This connects the useful block-245 normalized packet template to local real-source intake.
- It is not a fetcher and does not call external APIs.
- It creates provenance, hashes, anchors, quarantine partitions, quality precheck, and candidate routing.
- It does not execute claim/event/entity extraction, thesis generation, valuation, recommendation, learning writes, allocation, paper trading, or live trading.
- Validate the output before designing any real extraction step:

```powershell
python run_review_only_real_source_normalized_packet_validation_gate.py --input-json reports\dean_os\real_source_normalized_packet_current\latest.json --output-dir reports\dean_os\real_source_normalized_packet_validation_gate_current
```

Then pass the same real-source packet through the existing source review path:

```powershell
python run_agent_source_evidence_validation_gate.py --source-json reports\dean_os\real_source_normalized_packet_current\latest.json --output-dir reports\dean_os\source_evidence_validation_gate_current
python run_agent_source_extraction_review_packet.py --source-json reports\dean_os\real_source_normalized_packet_current\latest.json --source-gate-json reports\dean_os\source_evidence_validation_gate_current\latest.json --domain-packet-json reports\dean_os\domain_specialist_review_packet_current\latest.json --output-dir reports\dean_os\source_extraction_review_packet_current
```

## 10j-v. Fundamental Input Readiness Gate

Check caller-supplied fundamentals before value-agent review:

```powershell
python run_agent_fundamental_input_readiness_gate.py --fundamentals-json reports\dean_os\fundamentals_input\latest.json --output-dir reports\dean_os\fundamental_input_readiness_gate_current
```

Share the `summary`, `ticker_metric_summary`, `readiness_checks`, `output_boundary`, and `recommendations` sections.

Interpretation:
- This is the safe local integration point for the financial statement / numeric / ratio draft axis.
- It validates supplied metric shape, numeric values, units, periods, and source-citation presence.
- It does not extract numbers from filings, reconcile statements, compute ratios, interpret ratios, create valuation, recommend, allocate, paper trade, or live trade.
- Missing periods or source citations are warnings; nonnumeric values or invalid units block review.
- Pass the gate JSON into Agent Lab with `--fundamental-gate-json`; warning/failing gates block value screening instead of silently scoring raw fundamentals.

## 10j-w. Current Architecture Map

Build the current active architecture map before making system-level decisions:

```powershell
python run_agent_current_architecture_map.py --output-dir reports\dean_os\current_architecture_map_current
```

Share the `summary`, `branch_map`, `pipeline_metric_control_branch`, `domain_analyst_branch`, `corrections_to_user_plan`, `orchestrator_contract`, and `next_safe_steps` sections.

Interpretation:
- This is the active replacement for stale `system_audit_summary.py`.
- The pipeline branch is a metric-plane review surface, not an automatic optimizer.
- PnL is only one plane; split quality, leakage, replay repeatability, drawdown, feature stability, outcome coverage, and freshness are separate gates.
- Domain analysts output sector/domain theses first; ticker theses require a direct-evidence bridge.
- The orchestrator coordinates review gates and human decisions; it never produces trade signals.
- This map does not approve cloning analysts, live collectors, learning promotion, production config writes, recommendations, allocation, paper trading, or live trading.

## 10j-x. Domain Analyst Intake Packet

Normalize a validated source/evidence pack into one full domain analyst intake:

```powershell
python run_agent_domain_analyst_intake_packet.py --evidence-pack-json reports\dean_os\analyst_evidence_pack_cached_source_current\latest.json --source-gate-json reports\dean_os\source_evidence_validation_gate_cached_source_current\latest.json --domain-id semiconductor_ai_infrastructure --tickers AAPL AMD MSFT NVDA TSM --sectors semiconductor --output-dir reports\dean_os\domain_analyst_intake_packet_current
```

Strict sector-only semiconductor intake:

```powershell
python run_agent_domain_analyst_intake_packet.py --evidence-pack-json reports\dean_os\analyst_evidence_pack_semiconductor_sector_only_strict_current\latest.json --source-gate-json reports\dean_os\source_evidence_validation_gate_semiconductor_sector_only_strict_current\latest.json --domain-id semiconductor_ai_infrastructure --sectors semiconductor --max-items 500 --output-dir reports\dean_os\domain_analyst_intake_packet_semiconductor_sector_only_strict_current
```

Share the `summary`, `source_gate_context`, `evidence_type_summary`, `directness_summary`, `analyst_report`, `review_checks`, and `recommendations` sections.

Interpretation:
- This is the first full domain analyst intake contract.
- It accepts news, articles, reports, macro/context rows, and other normalized source documents through the evidence pack.
- It turns documents into `AnalystEvidenceItem` rows with `evidence_type`, `directness`, strength, freshness, reliability, and limitations.
- Domain/sector evidence remains domain thesis context; ticker-direct evidence is partitioned separately.
- The analyst report is review-only and cannot recommend, allocate, write learning memory, paper trade, or live trade.
- Current cached pack is ticker-filtered; for a cleaner sector-only analyst test, build an evidence pack without forcing all rows through requested tickers.
- Current strict sector-only semiconductor smoke produced 144 documents, 0 ticker-direct evidence, full required evidence coverage, and `partial_ready_for_review`.
- `capex_cycle` depends on classifier priority for phrases such as `capital spending`, `AI spending`, and `data center investment/spending`; do not solve that lane by forcing tickers.

## 10j-y. Domain Analyst Instance Contract

Build the review-only passport for one reusable domain analyst instance:

```powershell
python run_agent_domain_analyst_instance_contract.py --evidence-pack-json reports\dean_os\analyst_evidence_pack_semiconductor_sector_only_strict_current\latest.json --source-gate-json reports\dean_os\source_evidence_validation_gate_semiconductor_sector_only_strict_current\latest.json --domain-intake-json reports\dean_os\domain_analyst_intake_packet_semiconductor_sector_only_strict_current\latest.json --architecture-map-json reports\dean_os\current_architecture_map_current\latest.json --output-dir reports\dean_os\domain_analyst_instance_contract_current
```

Share the `summary`, `portable_template_slots`, `fixed_contract_sequence`, `review_checks`, and `operator_next_steps` sections.

Interpretation:
- This is the current passport for the first reusable domain analyst instance.
- Current result: `domain_analyst_instance_review_ready`, domain `semiconductor_ai_infrastructure`, 144 documents, 144 evidence items, `partial_ready_for_review`, 0 ticker-direct evidence.
- `can_reuse_as_template_after_manual_review=true`, but `can_scale_to_other_domains_now=false`.
- Portable slots are the domain ID, sectors, sector keywords, required/useful evidence types, and ticker universe hints.
- Fixed contract is source pack -> source gate -> domain intake -> sector/domain thesis -> separate ticker bridge -> separate learning/trading gates.

## 10j-y1. Domain Analyst Thesis Review Packet

Build the review-only packet for the sector/domain thesis before any ticker bridge:

```powershell
python run_agent_domain_analyst_thesis_review_packet.py --domain-intake-json reports\dean_os\domain_analyst_intake_packet_semiconductor_sector_only_strict_current\latest.json --domain-instance-contract-json reports\dean_os\domain_analyst_instance_contract_current\latest.json --architecture-map-json reports\dean_os\current_architecture_map_current\latest.json --output-dir reports\dean_os\domain_analyst_thesis_review_packet_current
```

Share the `summary`, `thesis_snapshot`, `evidence_lane_coverage`, `ticker_bridge_boundary`, `review_checks`, and `operator_next_steps` sections.

Interpretation:
- This reviews the domain/sector thesis only.
- Current result: `domain_thesis_review_ready`, 144 evidence items, 0 ticker-direct evidence, no required evidence missing, 19 pass, 0 warn, 0 fail.
- `can_standardize_domain_template_after_manual_review=true`, but this still requires manual acceptance.
- `can_create_direct_ticker_thesis_without_bridge=false`, `can_trade=false`.
- Run sector-to-ticker bridge only after this thesis packet is manually accepted.

## 10j-y2. Pipeline Metric Input Readiness Gate

Inventory saved model/replay/feature/data-quality inputs before refreshing the pipeline control surface:

```powershell
python run_agent_pipeline_metric_input_readiness_gate.py --model-performance performance_data.json --replay-batch reports\dean_os\historical_replay_batch_repaired_expanded\latest.json --data-quality diagnostic_reports\feature_lineage_report_current_cache.json --output-dir reports\dean_os\pipeline_metric_input_readiness_gate_current
```

Share the `summary`, `input_inventory`, `metric_plane_readiness`, `commands`, and `operator_next_steps` sections.

Interpretation:
- This gate does not run replay, train, tune, write config, recommend, paper trade, or live trade.
- Current result with the repaired replay batch and current cached feature lineage: `metric_inputs_ready_with_cautions`.
- Blocked planes: none.
- Risk/validation/feature stability remain cautions while their saved metrics are absent.
- The old `diagnostic_reports\feature_lineage_report.json` still points at a stale contaminated processed feature artifact and should not be used as the current data-quality input.

## 10j-z. Pipeline Control Instance Contract

Build the review-only passport for the pipeline-control branch:

```powershell
python run_agent_pipeline_control_instance_contract.py --pipeline-surface-json reports\dean_os\pipeline_control_surface\latest.json --architecture-map-json reports\dean_os\current_architecture_map_current\latest.json --domain-instance-contract-json reports\dean_os\domain_analyst_instance_contract_current\latest.json --output-dir reports\dean_os\pipeline_control_instance_contract_current
```

Share the `summary`, `metric_plane_contract`, `fixed_contract_sequence`, `review_checks`, and `operator_next_steps` sections.

Interpretation:
- This is the current passport for the pipeline-control branch.
- Current result from the available saved surface: `pipeline_control_instance_review_ready_with_cautions`.
- Blocked planes: none.
- Caution planes: `risk`, `validation`, `feature_stability`.
- It still covers all required metric planes and may propose only bounded reviewed experiments after manual review.
- It never runs autonomous tuning, writes production config, writes learning memory, recommends, allocates, paper trades, or live trades.

## 10j-z1. Pipeline Control Caution Review Packet

Review the remaining pipeline-control caution planes without clearing them with weak evidence:

```powershell
python run_agent_pipeline_control_caution_review_packet.py --pipeline-metric-input-readiness-json reports\dean_os\pipeline_metric_input_readiness_gate_current\latest.json --pipeline-control-instance-json reports\dean_os\pipeline_control_instance_contract_current\latest.json --model-performance-report-json reports\dean_os\model_performance\smoke.json --data-quality-json diagnostic_reports\feature_lineage_report_current_cache.json --output-dir reports\dean_os\pipeline_control_caution_review_packet_current
```

Share the `summary`, `artifact_triage`, `caution_plane_reviews`, `review_checks`, and `operator_next_steps` sections.

Interpretation:
- This packet does not run collectors, replay, training, tuning, config writes, recommendations, paper trades, or live trades.
- Current result: `pipeline_cautions_need_reviewed_inputs`.
- Blocked planes: none.
- Caution/missing-evidence planes: `risk`, `validation`, `feature_stability`.
- `reports\dean_os\model_performance\smoke.json` is warning evidence only because it has no recognized metrics.
- Clean data lineage supports `data_quality`; it does not clear drawdown, holdout validation, or feature stability.
- Code-audit reports cannot clear these metric planes.

## 10j-z2. Pipeline Control Metric Fixture Validation

Run a synthetic contract check that proves the pipeline-control chain can clear when complete metric fields are supplied:

```powershell
python run_agent_pipeline_control_metric_fixture_validation.py --output-dir reports\dean_os\pipeline_control_metric_fixture_validation_current
```

Share the `summary`, `chain_results`, `review_checks`, and `operator_next_steps` sections.

Interpretation:
- This is synthetic validation only, not model evidence.
- Current result: `synthetic_fixture_control_flow_passed`.
- It writes only under `reports\dean_os\pipeline_control_metric_fixture_validation_current`.
- Expected chain: readiness `metric_inputs_ready`, surface `clear`, instance `pipeline_control_instance_review_ready`, caution review `pipeline_ready_for_manual_proposal_review`.
- It must not be used to clear current real cautions; real metrics are still required.

## 10j-aa. Current System Alignment Review

Periodically check whether the current DEAN-OS work is still useful and aligned:

```powershell
python run_agent_current_system_alignment_review.py --evidence-pack-json reports\dean_os\analyst_evidence_pack_semiconductor_sector_only_strict_current\latest.json --source-gate-json reports\dean_os\source_evidence_validation_gate_semiconductor_sector_only_strict_current\latest.json --architecture-map-json reports\dean_os\current_architecture_map_current\latest.json --domain-analyst-intake-json reports\dean_os\domain_analyst_intake_packet_semiconductor_sector_only_strict_current\latest.json --domain-analyst-instance-contract-json reports\dean_os\domain_analyst_instance_contract_current\latest.json --domain-analyst-thesis-review-json reports\dean_os\domain_analyst_thesis_review_packet_current\latest.json --domain-analyst-template-standardization-json reports\dean_os\domain_analyst_template_standardization_packet_current\latest.json --domain-analyst-case-registry-json reports\dean_os\domain_analyst_case_registry_packet_current\latest.json --pipeline-metric-input-readiness-json reports\dean_os\pipeline_metric_input_readiness_gate_current\latest.json --pipeline-control-instance-contract-json reports\dean_os\pipeline_control_instance_contract_current\latest.json --pipeline-control-caution-review-json reports\dean_os\pipeline_control_caution_review_packet_current\latest.json --output-dir reports\dean_os\current_system_alignment_review_two_branch_current
```

Share the `summary`, `artifact_statuses`, `boundary_checks`, `usefulness_assessment`, and `recommendations` sections.

Interpretation:
- `aligned_with_cautions` is acceptable for continued staged work when blockers are zero.
- Cached/local news and macro evidence packs are useful source inputs; live collectors remain a separate health/inventory task.
- Empty `docs\research` means the real-source normalized packet path is waiting for one supported operator file.
- A missing fundamental gate artifact is acceptable unless value-screening fundamentals are being supplied.
- A ready domain analyst thesis review packet is still only a manual standardization candidate, not sector scaling.
- A blocked pipeline metric input readiness gate is acceptable as a safety stop when it blocks tuning and keeps config/trading disabled.
- A blocked pipeline-control instance is acceptable as a safety stop when it blocks tuning and keeps config/trading disabled.
- A pipeline-control caution review packet is useful when it separates reviewable cautions from missing empirical evidence.
- Treat `system_audit_summary.py` as historical once `CurrentArchitectureMap` is available.
- This checkpoint never approves sector scaling, learning promotion, recommendations, allocation, paper trading, or live trading.

## 10k. Replay Price Normalizer

Create a reusable normalized daily OHLCV artifact before trusting historical replay hit/miss results:

```powershell
python run_agent_replay_price_normalizer.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --compare-replay --as-of 2026-03-01T00:00:00+00:00 --lookback-days 180 --horizon-days 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet
```

Share the `artifact`, `quality`, `learning_gate`, `replay_comparison`, and `recommendations` sections.

Interpretation:
- This is data normalization only; it does not create paper trades or write learning memory.
- Use the saved artifact path as `price_data_path` for future historical replay batches.
- If `learning_gate.status` is `blocked`, replay outcomes remain diagnostic even if a single run looks correct.

## 10l. Historical Replay Batch

Run many replay slices against the normalized replay artifact:

```powershell
python run_agent_historical_replay_batch.py data\dean_os\replay_prices\replay_prices_1d_normalized_20260612_073159.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --start-as-of 2025-09-01T00:00:00+00:00 --end-as-of 2026-03-01T00:00:00+00:00 --step-days 30 --lookback-days 180 --horizon-days 30 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet
```

Preferred current repaired candidate:

```powershell
python run_agent_historical_replay_batch.py data\dean_os\replay_prices\replay_prices_1d_repaired_20260613_135839.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --start-as-of 2025-09-01T00:00:00+00:00 --end-as-of 2026-03-01T00:00:00+00:00 --step-days 30 --lookback-days 180 --horizon-days 30 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --output-dir reports\dean_os\historical_replay_batch_repaired
```

Share the `summary`, `learning_gate`, `summary.by_ticker`, `summary.by_horizon`, `summary.quality_warnings`, and `recommendations` sections.

Interpretation:
- This is still an evidence exam, not paper trading and not learning-memory promotion.
- Prefer `clear_hit_rate` over raw `hit_rate` when quality warnings exist.
- If `learning_gate.status` is `blocked`, investigate data windows before changing agent weights.

## 10m. Pipeline Control Surface

Build the bounded variation area for proposal-only pipeline tuning:

```powershell
python run_agent_pipeline_control_surface.py --model-performance performance_data.json --replay-batch reports\dean_os\historical_replay_batch_repaired_expanded\latest.json --data-quality diagnostic_reports\feature_lineage_report_current_cache.json
```

Share the `surface.status`, `surface.axes`, `surface.allowed_variation`, `proposal_gate`, and `recommendations` sections.

Interpretation:
- This is not a tuner and does not write production config.
- Run `PipelineMetricInputReadinessGate` first when changing or refreshing saved metric inputs.
- `surface.status=blocked` means TuningAgent should not propose experiments yet.
- `surface.status=caution` means only small reviewed experiments are allowed.
- `surface.status=clear` means reviewed experiments may be proposed inside the listed bounds.

## 10n. Domain Analyst Template Standardization Packet

Build the final manual-review candidate before accepting one domain analyst template:

```powershell
python run_agent_domain_analyst_template_standardization_packet.py --domain-instance-contract-json reports\dean_os\domain_analyst_instance_contract_current\latest.json --domain-thesis-review-json reports\dean_os\domain_analyst_thesis_review_packet_current\latest.json --architecture-map-json reports\dean_os\current_architecture_map_current\latest.json --output-dir reports\dean_os\domain_analyst_template_standardization_packet_current
```

Share the `summary`, `template_scope`, `manual_acceptance_checklist`, `review_checks`, and `operator_next_steps` sections.

Interpretation:
- Current result: `ready_for_manual_template_acceptance`, 23 pass, 0 warn, 0 fail.
- This is not acceptance. `can_mark_template_accepted_now=false`.
- It keeps domain scaling, sector-to-ticker bridge execution, learning, config writes, recommendations, and trading disabled.
- Manual acceptance must be recorded separately before cloning another domain or preparing the sector-to-ticker bridge.

Current full alignment command with this packet attached:

```powershell
python run_agent_current_system_alignment_review.py --evidence-pack-json reports\dean_os\analyst_evidence_pack_semiconductor_sector_only_strict_current\latest.json --source-gate-json reports\dean_os\source_evidence_validation_gate_semiconductor_sector_only_strict_current\latest.json --architecture-map-json reports\dean_os\current_architecture_map_current\latest.json --domain-analyst-intake-json reports\dean_os\domain_analyst_intake_packet_semiconductor_sector_only_strict_current\latest.json --domain-analyst-instance-contract-json reports\dean_os\domain_analyst_instance_contract_current\latest.json --domain-analyst-thesis-review-json reports\dean_os\domain_analyst_thesis_review_packet_current\latest.json --domain-analyst-template-standardization-json reports\dean_os\domain_analyst_template_standardization_packet_current\latest.json --pipeline-metric-input-readiness-json reports\dean_os\pipeline_metric_input_readiness_gate_current\latest.json --pipeline-control-instance-contract-json reports\dean_os\pipeline_control_instance_contract_current\latest.json --output-dir reports\dean_os\current_system_alignment_review_two_branch_current
```

Current result: `aligned_with_cautions`, 60 pass, 4 warn, 0 fail.

## 10o. Domain Analyst Case Registry Packet

Build the neutral casebook before any future learning promotion:

```powershell
python run_agent_domain_analyst_case_registry_packet.py --domain-thesis-review-json reports\dean_os\domain_analyst_thesis_review_packet_current\latest.json --domain-template-standardization-json reports\dean_os\domain_analyst_template_standardization_packet_current\latest.json --output-dir reports\dean_os\domain_analyst_case_registry_packet_current
```

Share the `summary`, `registry_policy`, `case_entries`, `source_observation_entries`, `comparison_axes`, `review_checks`, and `operator_next_steps` sections.

Interpretation:
- Current result: `case_registry_ready_pending_outcomes`, 1 thesis case, 16 source observations, 13 pass, 1 warn, 0 fail.
- The warning is expected until an outcome-evaluation artifact is attached.
- This is a case registry, not learning promotion.
- It explicitly blocks training from hit/correct cases only.
- It keeps miss, inconclusive, pending, invalid/unresolved, seasonal, policy, macro, and source-directness context visible.
- It does not write learning memory, analyst weights, config, recommendations, or trades.

Current full alignment command with case registry attached:

```powershell
python run_agent_current_system_alignment_review.py --evidence-pack-json reports\dean_os\analyst_evidence_pack_semiconductor_sector_only_strict_current\latest.json --source-gate-json reports\dean_os\source_evidence_validation_gate_semiconductor_sector_only_strict_current\latest.json --architecture-map-json reports\dean_os\current_architecture_map_current\latest.json --domain-analyst-intake-json reports\dean_os\domain_analyst_intake_packet_semiconductor_sector_only_strict_current\latest.json --domain-analyst-instance-contract-json reports\dean_os\domain_analyst_instance_contract_current\latest.json --domain-analyst-thesis-review-json reports\dean_os\domain_analyst_thesis_review_packet_current\latest.json --domain-analyst-template-standardization-json reports\dean_os\domain_analyst_template_standardization_packet_current\latest.json --domain-analyst-case-registry-json reports\dean_os\domain_analyst_case_registry_packet_current\latest.json --pipeline-metric-input-readiness-json reports\dean_os\pipeline_metric_input_readiness_gate_current\latest.json --pipeline-control-instance-contract-json reports\dean_os\pipeline_control_instance_contract_current\latest.json --output-dir reports\dean_os\current_system_alignment_review_two_branch_current
```

Current result: `aligned_with_cautions`, 69 pass, 4 warn, 0 fail.

## 10p. Build Focus Review Packet

Run this when the next build step is ambiguous or a branch feels over-deepened:

```powershell
python run_agent_build_focus_review_packet.py --alignment-review-json reports\dean_os\current_system_alignment_review_two_branch_current\latest.json --template-standardization-json reports\dean_os\domain_analyst_template_standardization_packet_current\latest.json --case-registry-json reports\dean_os\domain_analyst_case_registry_packet_current\latest.json --pipeline-control-instance-json reports\dean_os\pipeline_control_instance_contract_current\latest.json --output-dir reports\dean_os\build_focus_review_packet_current
```

Share the `summary`, `decision_rubric`, `branch_assessment`, `review_checks`, and `operator_next_steps` sections.

Interpretation:
- Current result: `focus_review_ready`, 10 pass, 0 warn, 0 fail.
- Recommended next operation: `manual_template_acceptance_or_review_pipeline_cautions`.
- `should_stop_adding_domain_template_gates=true`.
- `should_switch_to_pipeline_control_blockers=false`.
- Domain work is still useful only for attaching real outcome evaluation to the case registry.
- This packet performs no learning, config, recommendation, bridge, scaling, paper-trading, or live-trading action.

## 11. Tests

```powershell
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_manual
```

Use the explicit `--basetemp` on Windows if `.tmp_pytest` is locked or cleanup fails. Use `-p no:cacheprovider` if `.pytest_cache` has permission noise.

## 12. Save Long Sonar/Pytest Logs

Do not run `sonar_test_errors.py` as Python if it contains copied terminal output. Save a fresh run as a log instead:

```powershell
New-Item -ItemType Directory -Force logs\sonar | Out-Null
python YOUR_COMMAND_HERE *>&1 | Tee-Object -FilePath logs\sonar\sonar_run_2026-06-07.txt
```
## Active Final-Stage Execution Boundary (2026-06-28)

- Normal final-stage execution is `Stage 5 -> Stage 7`; Stage 6 is not run by default.
- If Stage 6 is explicitly selected, verify `execution_status=review_only_no_execution` and `execution_boundary.portfolio_mutated=false`.
- Paper and live requests must remain blocked in the active pipeline.
- Do not use a boolean acknowledgement to bypass the DEAN-OS paper receipt/plan/review lane.
- Stage 7 trading activity may create only `learning_review_candidate`; it must not invoke real-time adaptation or write learning/config state.
- Stage 7 external notification must remain disabled unless that individual request explicitly sets `evaluation_notification_authorized=true`.
- Do not run Stage 6, a paper cycle, or the full pipeline merely to test this boundary; use the focused unit contracts.

Verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests\unit\test_stage6_execution_boundary.py tests\unit\test_stage7_review_boundaries.py tests\unit\test_stage6_diary_logging.py tests\unit\test_hybrid_pipeline_manager.py tests\unit\test_pipeline_executor.py
```
## DEAN Orchestrator Two-Phase Review (2026-06-28)

- Hard-veto pipeline agents run before the pipeline and again after pipeline outputs are available.
- Missing hard-agent prerequisites must create an evidence-backed blocked report and prevent the pipeline runner.
- Post-pipeline reports supersede preflight reports from the same agent.
- Active consensus may emit `blocked`, `needs_more_data`, `no_trade`, or `watchlist`; it must not emit an execution candidate.

Verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests\dean_os\test_orchestrator_review_boundary.py tests\dean_os\test_current_architecture_map.py
```
## Pipeline Adapter Review Contract (2026-06-28)

- Read `MarketContext.metadata.pipeline_review_contract` instead of searching arbitrary nested pipeline output for Stage 4/7 artifacts.
- Treat its artifact paths as discovery references only; evidence inventory/locked assemblers still determine evidence class.
- Realized returns must take precedence over `target_return_*`.
- If `returns_offline_only=true`, pre-trade RiskAgent must block.

Verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests\dean_os\test_pipeline_adapter_review_contract.py tests\dean_os\test_orchestrator_review_boundary.py
```

## Verified Analyst Reasoning Chain (2026-07-03)

Run in this order after refreshing the saved semiconductor runtime:

```powershell
python run_agent_analyst_core_reasoning_snapshot.py --runtime-json reports\dean_os\semiconductor_analyst_runtime_current\latest.json --output-dir reports\dean_os\analyst_core_reasoning_snapshot_current
python run_agent_domain_analyst_thesis_review_packet.py --domain-intake-json reports\dean_os\semiconductor_analyst_runtime_current\latest.json --reasoning-snapshot-json reports\dean_os\analyst_core_reasoning_snapshot_current\latest.json --domain-instance-contract-json reports\dean_os\domain_analyst_instance_contract_current\latest.json --architecture-map-json reports\dean_os\current_architecture_map_current\latest.json --output-dir reports\dean_os\domain_analyst_thesis_review_packet_current
python run_agent_domain_analyst_template_standardization_packet.py --domain-instance-contract-json reports\dean_os\domain_analyst_instance_contract_current\latest.json --domain-thesis-review-json reports\dean_os\domain_analyst_thesis_review_packet_current\latest.json --architecture-map-json reports\dean_os\current_architecture_map_current\latest.json --output-dir reports\dean_os\domain_analyst_template_standardization_packet_current
python run_agent_domain_analyst_case_registry_packet.py --domain-thesis-review-json reports\dean_os\domain_analyst_thesis_review_packet_current\latest.json --domain-template-standardization-json reports\dean_os\domain_analyst_template_standardization_packet_current\latest.json --output-dir reports\dean_os\domain_analyst_case_registry_packet_current
python run_agent_sector_to_ticker_bridge.py --domain-thesis-review-json reports\dean_os\domain_analyst_thesis_review_packet_current\latest.json --pipeline-case-json reports\dean_os\pipeline_model_case_packet_current\latest.json --ticker-evidence-json reports\dean_os\saved_ticker_specific_evidence_producer_current\latest.json --output-dir reports\dean_os\sector_thesis_to_ticker_basket_current
python run_agent_sector_to_ticker_review_packet.py --bridge-json reports\dean_os\sector_thesis_to_ticker_basket_current\latest.json --output-dir reports\dean_os\sector_to_ticker_review_packet_current
```

Expected current boundaries:

- reasoning snapshot: `reasoning_snapshot_ready_with_cautions`
- evidence/classification: `152/152`
- directional ticker reasoning: `0`
- scenario graph: `not_generated`
- thesis review: ready with cautions and runtime hash bound
- template: manual-acceptance candidate only
- prospective case: one pending sector case with 30/90/180 checkpoints
- ticker bridge: four blocked basket candidates and zero forecasts
- Stage 5 sector reasoning: supporting annotation only

Verification:

```powershell
python -m pytest tests\dean_os\test_analyst_core_artifact_loader.py tests\dean_os\test_analyst_core_phase2_lenses.py tests\dean_os\test_analyst_core_sector_analyst.py tests\dean_os\test_analyst_core_verified_reasoning.py tests\dean_os\test_domain_analyst_thesis_review_packet.py tests\dean_os\test_domain_analyst_template_standardization_packet.py tests\dean_os\test_domain_analyst_case_registry_packet.py tests\dean_os\test_sector_thesis_to_ticker_basket_bridge.py tests\dean_os\test_sector_to_ticker_review_packet.py tests\dean_os\test_pipeline_prediction_review_packet.py tests\dean_os\test_current_architecture_map.py -q --basetemp D:\trading_project\.pytest_tmp\verified_reasoning
```

## World Model Event Packet + Replay Review Gate (2026-07-09)

Discover exact pipeline/indicator context from existing pipeline review
artifacts:

```powershell
python run_agent_world_model_pipeline_context.py --ticker NVDA --timeframe 15m --timeframe 60m --timeframe 1d --output-dir reports\dean_os\world_model_pipeline_context_current
```

Current interpretation:

- `pipeline_context_bundle_ready_with_gaps` is acceptable as context but not as
  proof of full pipeline readiness.
- Current real discovery has `15m` exact context available and `60m`/`1d`
  missing.
- If `stage3_cache_missing_ready_lane_count>0`, at least one ready Stage23
  artifact was created before `stage3_cache` metadata was materialized.
- Current real `15m` state:
  `stage3_cache_status=stage3_cache_missing_from_ready_stage23_artifact`.
- A bounded NVDA/15m 600-row regeneration attempt exceeded a 3-minute local
  budget; do not blindly retry that exact heavy command.
- This bundle is review-only context; it does not run Stage 4/5, tune, write
  learning memory, or trade.
- Stage5 review is included as compact summary/binding only
  (`contexts_included=false`), not full context duplication.

Build the world-model event packet from verified saved news plus exact
pipeline/indicator/expectation context:

```powershell
python run_agent_world_model_event_learning_packet.py --news-artifact reports\dean_os\saved_semiconductor_news_evidence_producer\latest.json --pipeline-context-json PATH\TO\pipeline_context.json --indicator-context-json PATH\TO\indicator_context.json --expectation-context-json PATH\TO\expectation_context.json --domain-id semiconductor_ai_infrastructure --output-dir reports\dean_os\world_model_event_learning_packet_current
```

Or attach the discovered bundle inline:

```powershell
python run_agent_world_model_event_learning_packet.py --news-artifact reports\dean_os\saved_semiconductor_news_evidence_producer_current\latest.json --discover-pipeline-context --ticker NVDA --timeframe 15m --timeframe 60m --timeframe 1d --domain-id semiconductor_ai_infrastructure --output-dir reports\dean_os\world_model_event_learning_packet_current
```

Do not run the full packet if the saved-news artifact fails verification. As of
the 2026-07-09 check, refreshing
`reports\dean_os\saved_semiconductor_news_evidence_producer_current\latest.json`
against current `data\processed\features\news_data.parquet` produced
`blocked_no_semiconductor_news_evidence`; restore/regenerate verified news
first.

Review-only interpretation:

- `pipeline_indicator_context_status=pipeline_indicator_context_ready` means
  the packet could condition hypotheses on supplied metrics/regime/tags.
- `expectation_context_available=true` means expectation/crowdedness/surprise
  context was supplied.
- `replay_tasks[*].registration_status=candidate_pending_manual_review`.
- The packet still cannot register replay tasks, write learning memory, tune,
  promote, recommend execution, paper trade, or live trade.

Run the manual gate without approval:

```powershell
python run_agent_world_model_replay_review_gate.py --packet-json reports\dean_os\world_model_event_learning_packet_current\latest.json --output-dir reports\dean_os\world_model_replay_review_gate_current
```

Expected: `gate_status=manual_review_required_for_replay_registration`.

Run the manual gate after explicit operator approval:

```powershell
python run_agent_world_model_replay_review_gate.py --packet-json reports\dean_os\world_model_event_learning_packet_current\latest.json --approve --reviewer "operator" --review-notes "manual replay registration approved" --output-dir reports\dean_os\world_model_replay_review_gate_current
```

Expected: `gate_status=replay_tasks_approved_for_registration` and
`registration_bundle_created=true`.

Still not performed:

- replay queue write;
- outcome registration;
- learning-memory write;
- production config write;
- model promotion;
- paper/live execution.

Verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_world_model_event_learning_packet.py tests\dean_os\test_world_model_replay_review_gate.py tests\dean_os\test_analyst_core_phase2_lenses.py tests\dean_os\test_analyst_core_schemas.py tests\dean_os\test_domain_data_feeder.py tests\dean_os\test_saved_semiconductor_news_evidence_producer.py -q -p no:cacheprovider --basetemp .pytest_tmp\world_model_integrated
# 81 passed

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_world_model_pipeline_context.py tests\dean_os\test_world_model_event_learning_packet.py tests\dean_os\test_world_model_replay_review_gate.py tests\dean_os\test_analyst_core_phase2_lenses.py tests\dean_os\test_analyst_core_schemas.py tests\dean_os\test_domain_data_feeder.py tests\dean_os\test_saved_semiconductor_news_evidence_producer.py -q -p no:cacheprovider --basetemp .pytest_tmp\world_model_pipeline_context_integrated
# 83 passed

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_world_model_pipeline_context.py tests\dean_os\test_world_model_event_learning_packet.py tests\dean_os\test_world_model_replay_review_gate.py tests\dean_os\test_analyst_core_phase2_lenses.py tests\dean_os\test_analyst_core_schemas.py tests\dean_os\test_domain_data_feeder.py tests\dean_os\test_saved_semiconductor_news_evidence_producer.py -q -p no:cacheprovider --basetemp .pytest_tmp\world_model_pipeline_context_cache_status_integrated
# 84 passed
```

## World Model Approved Replay Registration Bridge (2026-07-09)

After the manual review gate creates an approved registration bundle, build a
dry-run OutcomeTracker registration plan:

```powershell
python run_agent_world_model_replay_registration.py --gate-json reports\dean_os\world_model_replay_review_gate_current\latest.json --output-dir reports\dean_os\world_model_replay_registration_current
```

Expected:

- `bridge_status=dry_run_ready_for_outcome_tracker_registration`
- `dry_run=true`
- `outcome_tracker_registration_performed=false`
- no outcome scoring, learning-memory write, model promotion, config write,
  paper trade, or live trade.

Only after reviewing the dry-run plan, apply to OutcomeTracker explicitly:

```powershell
python run_agent_world_model_replay_registration.py --gate-json reports\dean_os\world_model_replay_review_gate_current\latest.json --source-packet-json reports\dean_os\world_model_event_learning_packet_current\latest.json --tracker-db data\dean_os\outcome_tracker.sqlite --apply --output-dir reports\dean_os\world_model_replay_registration_current
```

Expected:

- `bridge_status=outcome_tracker_registration_applied`, or
  `outcome_tracker_registration_already_applied` on repeat runs;
- one OutcomeTracker event per approved replay task;
- source traceability in the form
  `world_model_replay|bundle=<bundle_id>|task=<task_id>`;
- dedupe prevents duplicate events for the same approved bundle/task.

Important:

- This bridge consumes only an approved `WorldModelReplayReviewGate` artifact.
  It must block direct use of raw event packets.
- OutcomeTracker currently uses fixed directional horizons `1/5/30/60/120d`.
  World-model hypotheses may be non-directional; the bridge records neutral
  projections unless an explicit direction is present. Do not interpret those
  as ticker/sector forecasts without a later outcome review.
- Applying this bridge is not learning promotion. It creates pending
  tracking events only.

Verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_world_model_replay_registration_bridge.py -q -p no:cacheprovider --basetemp .pytest_tmp\world_model_replay_registration
# 3 passed

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_world_model_pipeline_context.py tests\dean_os\test_world_model_event_learning_packet.py tests\dean_os\test_world_model_replay_review_gate.py tests\dean_os\test_world_model_replay_registration_bridge.py tests\dean_os\test_analyst_core_phase2_lenses.py tests\dean_os\test_analyst_core_schemas.py tests\dean_os\test_domain_data_feeder.py tests\dean_os\test_saved_semiconductor_news_evidence_producer.py -q -p no:cacheprovider --basetemp .pytest_tmp\world_model_replay_registration_integrated
# 87 passed
```

## Saved Semiconductor News Current Cache Schema (2026-07-09)

Build the saved-news artifact from the current cached parquet:

```powershell
python run_agent_saved_semiconductor_news_evidence.py data\processed\features\news_data.parquet --as-of 2026-06-30T21:00:00+00:00 --output-dir reports\dean_os\saved_semiconductor_news_evidence_producer_current
```

Current expected result:

- `status=semiconductor_news_evidence_ready_with_gaps`
- `usable_source_row_count=4482`
- `domain_candidate_count=20`
- `classified_candidate_count=4`
- `accepted_news_record_count=4`
- `ready_required_lanes=[]`
- missing lanes:
  `sector_demand`, `capex_cycle`, `supply_chain`,
  `policy_or_geopolitical`, `market_confirmation`

## Saved-news shard recovery and full-cycle rerun (2026-07-13)

Build a point-in-time local snapshot from allowlisted DuckDB news tables and
the current saved parquet. This is read-only with respect to DuckDB and starts
no collector:

```powershell
python run_agent_saved_news_shard_snapshot.py `
  --database-path data\trading_data.duckdb `
  --output-parquet-path data\dean_os\saved_news_snapshots\latest.parquet `
  --as-of 2026-06-30T21:00:00Z `
  --include-parquet data\processed\features\news_data.parquet `
  --output-dir reports\dean_os\saved_news_shard_snapshot_current
```

Run the existing strict news and policy producers:

```powershell
python run_agent_saved_semiconductor_news_evidence.py `
  data\dean_os\saved_news_snapshots\latest.parquet `
  --as-of 2026-06-30T21:00:00Z `
  --output-dir reports\dean_os\saved_semiconductor_news_evidence_producer_current

python run_agent_saved_official_policy_evidence.py `
  reports\dean_os\bis_policy_snapshot_current\latest.json `
  reports\dean_os\saved_semiconductor_news_evidence_producer_current\latest.json `
  --as-of 2026-06-30T21:00:00Z `
  --output-dir reports\dean_os\saved_official_policy_evidence_producer_current
```

For the world-model bridge, use the clean three-timeframe bundle, not the stale
legacy bundle:

```powershell
python run_agent_full_system_cycle_world_model.py `
  --cycle reports\dean_os\full_system_review_cycle_current\latest.json `
  --pipeline-context-bundle reports\dean_os\world_model_pipeline_context_clean_current\latest.json `
  --domain-id semiconductor_ai_infrastructure `
  --output-dir reports\dean_os\world_model_event_learning_cycle_current

python run_agent_full_system_cycle_closure.py `
  --cycle reports\dean_os\full_system_review_cycle_current\latest.json `
  --world-model reports\dean_os\world_model_event_learning_cycle_current\latest.json `
  --prior-checkpoint-monitor reports\dean_os\replay_checkpoint_monitor_current\latest.json `
  --output-dir reports\dean_os\full_system_cycle_closure_current
```

Expected current closure is `current_cycle_requires_new_replay_review`, not
automatic registration. Ten new tasks are candidates pending manual review;
the authorization ledger remains empty.
- `can_enter_market_context_review=true`
- `can_influence_ticker_prediction=false`
- `can_trade=false`

Interpretation:

- The producer now supports the current cache schema:
  `title`, `summary`, `ticker`, `source`, `timestamp`.
- It can extract an embedded URL from text when `link/url` is absent.
- URLs are stripped before keyword matching, and keyword matching uses word
  boundaries to avoid false `Intel` hits from `Intelsat/intelligence`.
- Current HF/twitter-financial-news rows are weak market/rating context. They
  can enter review context but must not close required evidence lanes.

Run a real world-model packet against current saved news plus discovered
pipeline context:

```powershell
python run_agent_world_model_event_learning_packet.py --news-artifact reports\dean_os\saved_semiconductor_news_evidence_producer_current\latest.json --discover-pipeline-context --ticker NVDA --timeframe 15m --timeframe 60m --timeframe 1d --domain-id semiconductor_ai_infrastructure --output-dir reports\dean_os\world_model_event_learning_packet_current
```

Current expected result:

- `packet_status=world_model_event_learning_ready_with_gaps`
- accepted evidence `4`
- classified events `4`
- hypotheses `0`
- replay tasks `0`
- pipeline context ready with gaps
- no learning/config/trading authority.

Verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_saved_semiconductor_news_evidence_producer.py -q -p no:cacheprovider --basetemp .pytest_tmp\saved_news_market_confirmation
# 6 passed

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_saved_semiconductor_news_evidence_producer.py tests\dean_os\test_world_model_pipeline_context.py tests\dean_os\test_world_model_event_learning_packet.py tests\dean_os\test_world_model_replay_review_gate.py tests\dean_os\test_world_model_replay_registration_bridge.py tests\dean_os\test_analyst_core_phase2_lenses.py tests\dean_os\test_analyst_core_schemas.py tests\dean_os\test_domain_data_feeder.py -q -p no:cacheprovider --basetemp .pytest_tmp\saved_news_real_packet_integrated
# 89 passed
```

## Pipeline Timeframe Lane Readiness Plan (2026-07-09)

Before rerunning expensive Stage23 jobs, inspect source coverage versus current
pipeline-context artifacts:

```powershell
python run_agent_pipeline_timeframe_lane_readiness.py data\colab\accumulated\main_database\main_database_stage1_raw_data_20260629_195400.parquet --ticker NVDA --timeframe 15m --timeframe 60m --timeframe 1d --max-rows-per-ticker 200 --pipeline-context-json reports\dean_os\world_model_pipeline_context_current\latest.json --output-dir reports\dean_os\pipeline_timeframe_lane_readiness_current
```

Current expected result:

- `status=pipeline_timeframe_lanes_ready_with_gaps`
- source-available lanes: `3`
- exact-context lanes: `1`
- artifact-missing lanes: `2`
- ready lanes missing Stage3 cache: `1`
- batch artifact lanes: `1`
- `can_condition_world_model=true`
- `can_write_learning_memory=false`
- `can_trade=false`

Current lane interpretation:

- `15m`: source exists, exact context exists, but true Stage3 shard-cache
  metadata is missing. Verified batch artifacts are present but are not
  reusable Stage3 shard-cache.
- `60m`: source exists, Stage23 artifact missing.
- `1d`: source exists, Stage23 artifact missing.

Do not confuse:

- verified `batch_artifacts` = useful review lineage;
- true `stage3_cache` = reusable Stage3 shard-cache expected by the current
  Stage23 implementation.

Bounded diagnostic note:

- Compact interactive Stage23 attempts for NVDA `60m` and `1d` with
  `max_rows_per_ticker=200` exceeded roughly 60 seconds and wrote no latest
  artifact.
- Do not keep retrying interactively; profile Stage3 runtime or run a scheduled
  bounded job.

Verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_pipeline_timeframe_lane_readiness.py -q -p no:cacheprovider --basetemp .pytest_tmp\pipeline_timeframe_lane_readiness
# 2 passed

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_pipeline_timeframe_lane_readiness.py tests\dean_os\test_world_model_pipeline_context.py tests\dean_os\test_saved_semiconductor_news_evidence_producer.py tests\dean_os\test_world_model_event_learning_packet.py tests\dean_os\test_world_model_replay_review_gate.py tests\dean_os\test_world_model_replay_registration_bridge.py -q -p no:cacheprovider --basetemp .pytest_tmp\p1_lane_readiness_integrated
# 21 passed
```

## Stage23 Runtime Profile + Source Cadence Validation (2026-07-09)

Run a safe source-only runtime profile before any expensive Stage23 work:

```powershell
python run_agent_pipeline_stage23_runtime_profile.py data\colab\accumulated\main_database\main_database_stage1_raw_data_20260629_195400.parquet --ticker NVDA --timeframe 15m --timeframe 60m --timeframe 1d --max-rows-per-ticker 200 --output-dir reports\dean_os\pipeline_stage23_runtime_profile_current
```

Current expected result:

- `status=pipeline_stage23_runtime_profile_ready_with_gaps`
- ready lanes: `1`
- blocked lanes: `2`
- Stage2 included: `false`
- Stage3 included: `false`
- no Stage23 batch/cache/learning/trading writes.

Refresh lane readiness with cadence validation:

```powershell
python run_agent_pipeline_timeframe_lane_readiness.py data\colab\accumulated\main_database\main_database_stage1_raw_data_20260629_195400.parquet --ticker NVDA --timeframe 15m --timeframe 60m --timeframe 1d --max-rows-per-ticker 200 --pipeline-context-json reports\dean_os\world_model_pipeline_context_current\latest.json --output-dir reports\dean_os\pipeline_timeframe_lane_readiness_current
```

Current expected result:

- source-available lanes: `3`
- source-valid lanes: `1`
- source-invalid lanes: `2`
- exact-context lanes: `1`
- artifact-missing lanes: `0`
- ready lanes missing Stage3 cache: `1`

Interpretation:

- `15m`: valid; exact context exists; true Stage3 shard-cache missing.
- `60m`: rows exist but bounded cadence validation fails; do not run Stage23.
- `1d`: rows exist but cadence/OHLCV validation fails; do not run Stage23.

If testing Stage2 interactively, use a very small explicit sample:

```powershell
python run_agent_pipeline_stage23_runtime_profile.py data\colab\accumulated\main_database\main_database_stage1_raw_data_20260629_195400.parquet --ticker NVDA --timeframe 15m --max-rows-per-ticker 40 --include-stage2 --output-dir reports\dean_os\pipeline_stage23_runtime_profile_15m_stage2_sample
```

Materialize the valid `15m` shard-cache only via a scheduled/long-running run:

```powershell
python run_agent_pipeline_stage23_regeneration.py data\colab\accumulated\main_database\main_database_stage1_raw_data_20260629_195400.parquet --ticker NVDA --timeframe 15m --max-rows-per-ticker 200 --batch-dir data\colab\regenerated\lane_15m_stage23_review --output-dir reports\dean_os\pipeline_stage23_regeneration_lane_15m_review --shard-cache-dir data\colab\stage3_shard_cache\dean_review
```

Verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_pipeline_stage23_runtime_profile.py tests\dean_os\test_pipeline_timeframe_lane_readiness.py tests\dean_os\test_pipeline_stage23_regeneration.py tests\dean_os\test_world_model_pipeline_context.py -q -p no:cacheprovider --basetemp .pytest_tmp\pipeline_runtime_readiness_final
# 13 passed

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_package_lazy_import.py tests\dean_os\test_pipeline_stage23_runtime_profile.py tests\dean_os\test_pipeline_timeframe_lane_readiness.py -q -p no:cacheprovider --basetemp .pytest_tmp\pipeline_runtime_lazy_final
# 7 passed
```

## Memory lifecycle and observability verification (2026-07-11)

Use a workspace-local temporary path because the host pytest temporary folder
may be access-restricted:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests/dean_os/test_branch_observability.py tests/dean_os/test_agent_observability.py tests/dean_os/test_orchestrator_integration.py tests/dean_os/test_orchestrator_review_boundary.py tests/dean_os/test_memory_lifecycle_guard.py tests/dean_os/test_analyst_learning_promotion_bridge.py -q --basetemp D:\trading_project\.pytest_tmp\control_plane
```

Expected: 17 passed.

Controller policy checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests/dean_os/test_agent_evaluation_controller.py -q --basetemp D:\trading_project\.pytest_tmp\evaluation_controller
```

Expected: 3 passed. Keep `agent_evaluation_controller.enabled: false` until a
reviewed trace baseline exists; otherwise the controller can only report
telemetry completeness, not trustworthy agent quality.

Runtime enabling is explicit and additive:

```python
from dean_os.agent_observability import AgentRunTraceStore

trace_store = AgentRunTraceStore("logs/dean_os/agent_run_traces.jsonl")
orchestrator = DEANOrchestrator(registry=registry, trace_store=trace_store)
```

Do not interpret missing metrics as zero. Task success requires outcome/review
labels, tool-call accuracy requires correctness review, and cost per success
requires real usage cost plus reviewed task success.

## Candle source safety checkpoint (2026-07-10)

Do not run the previously suggested Stage23 command against
`main_database_stage1_raw_data_20260629_195400.parquet`. That artifact and the
adjacent June 28/29 snapshots failed the global ticker/timeframe identity audit.

Completed source protections:

- serialized `yf.download`;
- yfinance internal threads disabled;
- returned MultiIndex ticker must equal the requested ticker;
- exact OHLCV shared across ticker/timeframe identities is a hard failure;
- cadence mismatch remains a hard failure.

Safe order of operations:

1. create a separate clean Stage1 staging destination;
2. collect native `15m`, `1h`, `1d` data;
3. normalize only the label `1h→60m` after source validation;
4. run global identity/cadence/finite-value audit;
5. materialize shard-cache only after the audit passes;
6. rerun pipeline/world-model context discovery.

Never overwrite or silently repair the legacy snapshots. Preserve them only as
quarantined forensic evidence.

## Clean three-lane operational checkpoint (2026-07-10)

Canonical artifacts:

- `data/dean_os/clean_market_snapshots/latest.parquet`;
- `reports/dean_os/clean_market_snapshot_current/latest.json`;
- `data/colab/stage3_shard_cache/dean_clean`;
- `reports/dean_os/world_model_pipeline_context_clean_current/latest.json`;
- `reports/dean_os/pipeline_timeframe_lane_readiness_clean_current/latest.json`.

Expected state: 3 source-valid lanes, 12 Stage3 shards, 3 hash-compatible
exact-context lanes, zero missing lanes, World Model conditioning true, and all
learning/trading/promotion flags false.

Do not use June 28/29 accumulated Stage1 snapshots. Do not count Stage4 reviews
unless their parent feature/target hashes match the selected Stage23 batch.
For Stage5 use only `stage4_eligible_targets`; the current 60m
`target_hourly_volume_spike_1h` is excluded.

Next command family must exercise the unified analyst evidence merge with the
clean context plus news/macro/filings/knowledge. It must remain review-only.

## Unified domain analyst review (real clean context, 2026-07-11)

```powershell
python run_agent_domain_analyst_review.py --domain-id semiconductor_ai_infrastructure --as-of 2026-07-11T01:44:49.7478763Z --tickers ASML MU NVDA TSM --horizon-days 180 --news-artifact reports/dean_os/saved_semiconductor_news_evidence_producer_current/latest.json --macro-artifact reports/dean_os/saved_macro_evidence_producer_current/latest.json --sector-market-artifact reports/dean_os/saved_sector_market_evidence_producer_current/latest.json --fundamental-artifact reports/dean_os/saved_sec_fundamental_evidence_merger_current/latest.json --pipeline-context-artifact reports/dean_os/world_model_pipeline_context_clean_current/latest.json --output-dir reports/dean_os/domain_analyst_review_clean_current
```

Expected current result: 74 evidence records, five lenses,
`domain_analyst_review_needs_more_data`, `can_trade=false`. The 71 producer
records have producer cutoff `2026-06-30T21:00:00+00:00`; they are valid
historical context, not current July 11 news.

Verification:

```powershell
python -m pytest tests/dean_os/test_pipeline_context_evidence_loader.py tests/dean_os/test_analyst_core_sector_analyst.py tests/dean_os/test_analyst_core_artifact_loader.py tests/dean_os/test_domain_analyst_agent.py -q -p no:cacheprovider --basetemp .pytest_tmp/lane_identity_fix
# 57 passed
```

## Full-text research context and visible hypotheses

Verified knowledge readiness:

```powershell
python run_agent_analyst_knowledge_readiness.py --store-dir data/dean_os/analyst_knowledge_verified --as-of 2026-07-11T01:44:49.7478763Z --intended-use evidence --output-dir reports/dean_os/analyst_knowledge_readiness_verified_current
# knowledge_review_ready; 72/72 eligible; cannot influence prediction/trade
```

Unified run with policy and 20 context-only full-text matches:

```powershell
python run_agent_domain_analyst_review.py --domain-id semiconductor_ai_infrastructure --as-of 2026-07-11T02:17:15.3338241Z --tickers ASML MU NVDA TSM --horizon-days 180 --news-artifact reports/dean_os/saved_semiconductor_news_evidence_producer_current/latest.json --policy-artifact reports/dean_os/saved_official_policy_evidence_producer_current/latest.json --macro-artifact reports/dean_os/saved_macro_evidence_producer_current/latest.json --sector-market-artifact reports/dean_os/saved_sector_market_evidence_producer_current/latest.json --fundamental-artifact reports/dean_os/saved_sec_fundamental_evidence_merger_current/latest.json --pipeline-context-artifact reports/dean_os/world_model_pipeline_context_clean_current/latest.json --research-corpus data/dean_os/research_corpus.sqlite --research-top-k 20 --output-dir reports/dean_os/domain_analyst_review_clean_current
```

Expected: 95 evidence records, 5 lenses, 3 hypotheses, 11 gaps,
`needs_more_data`, `can_trade=false`. Corpus records are always
`required_lane_eligible=false` and cannot close sector evidence gates.

```powershell
python -m pytest tests/dean_os/test_research_corpus_evidence_loader.py tests/dean_os/test_domain_analyst_agent.py tests/dean_os/test_analyst_core_sector_analyst.py -q -p no:cacheprovider --basetemp .pytest_tmp/research_hypothesis_output
# 39 passed
```

## SEC inventory and hypothesis evidence-gap review

Canonical inventory-aware SEC chain is now 33 facts. Real gap review:

```powershell
python run_agent_hypothesis_evidence_gap_review.py reports/dean_os/domain_analyst_review_clean_current/latest.json reports/dean_os/saved_sec_fundamental_evidence_merger_current/latest.json --ratio-artifact reports/dean_os/saved_sec_derived_ratio_producer_current/latest.json --primary-snapshot reports/dean_os/sec_primary_document_snapshot_current/latest.json --as-of CURRENT_TIMEZONE_AWARE_UTC --output-dir reports/dean_os/hypothesis_evidence_gap_review_current
```

Expected:

- `partial_supported`: 4;
- `context_only_not_resolved`: 2;
- `missing`: 5;
- replay candidates: 3;
- registered: false; can trade: false.

Verification:

```powershell
python -m pytest tests/dean_os/test_hypothesis_evidence_gap_review.py tests/dean_os/test_saved_sec_fundamental_evidence_merger.py -q -p no:cacheprovider --basetemp .pytest_tmp/hypothesis_gap_inventory
# 4 passed
```

## Manual replay gate adapter (real non-approving run completed)

```powershell
python run_agent_hypothesis_gap_replay_packet.py reports/dean_os/hypothesis_evidence_gap_review_current/latest.json --output-dir reports/dean_os/hypothesis_gap_replay_packet_current
python run_agent_world_model_replay_review_gate.py --packet-json reports/dean_os/hypothesis_gap_replay_packet_current/latest.json --output-dir reports/dean_os/hypothesis_gap_replay_review_gate_current
```

Run without `--approve`. Expected: 3 hypotheses, 9 tasks,
`manual_review_required_for_replay_registration`, bundle absent, replay
registration false, learning write false and can trade false.

```powershell
python -m pytest tests/dean_os/test_hypothesis_gap_replay_packet.py tests/dean_os/test_world_model_replay_review_gate.py tests/dean_os/test_world_model_replay_registration_bridge.py -q -p no:cacheprovider --basetemp .pytest_tmp/hypothesis_manual_replay_gate
# 8 passed
```

Current verified gate invariants:

- packet hypotheses: 3;
- replay tasks: 9;
- indicator metrics: 13 aggregate pipeline metrics;
- 15m/60m/1d: exact-context available;
- Stage5 complete contexts: 0;
- registration bundle: absent;
- registration, learning write and trading: false.

Do not run with `--approve` and do not run the registration bridge until an
identified human reviewer has made and recorded the approval decision.

## Causal and Bayesian contract verification (2026-07-11)

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests/dean_os/test_causal_epistemic_contract.py tests/dean_os/test_analyst_core_schemas.py tests/dean_os/test_analyst_core_phase2_lenses.py tests/dean_os/test_world_model_event_learning_packet.py tests/dean_os/test_domain_analyst_regime_scenario_packet.py -q --basetemp D:\trading_project\.pytest_tmp\causal_contract
# 77 passed

python -m pytest tests/dean_os/test_bayesian_scenario_update.py tests/dean_os/test_causal_epistemic_contract.py tests/dean_os/test_analyst_core_schemas.py tests/dean_os/test_world_model_event_learning_packet.py tests/dean_os/test_shadow_calibration_diagnostics.py -q --basetemp D:\trading_project\.pytest_tmp\bayesian_causal
# 41 passed
```

Review rules:

- directed edge does not imply causality;
- `assumed_mechanism` cannot set `causal_claim_allowed=true`;
- temporal sequence and statistical association cannot authorize causal text;
- probability, confidence, impact, market reaction and fundamental change are
  separate fields;
- Bayesian posterior remains uncalibrated until replay/outcome evaluation;
- do not call Granger predictability causal identification.

## Event-study, expectation-gap and dynamic-edge verification (2026-07-11)

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests/dean_os/test_analyst_core_phase2_lenses.py tests/dean_os/test_analyst_core_sector_analyst.py tests/dean_os/test_domain_analyst_agent.py tests/dean_os/test_world_model_event_learning_packet.py tests/dean_os/test_event_study_eligibility.py -q --basetemp D:\trading_project\.pytest_tmp\expectation_event_study
# 85 passed

python -m pytest tests/dean_os/test_causal_epistemic_contract.py tests/dean_os/test_analyst_core_schemas.py tests/dean_os/test_analyst_core_phase2_lenses.py tests/dean_os/test_domain_analyst_regime_scenario_packet.py tests/dean_os/test_event_study_eligibility.py -q --basetemp D:\trading_project\.pytest_tmp\dynamic_graph_event
# 78 passed
```

Do not compute or interpret AR/CAR unless the eligibility artifact says
`can_estimate_abnormal_returns=true`. `descriptive_only` explicitly forbids clean
event attribution. Do not populate edge persistence or `last_validated_at` from
templates; those fields require reviewed replay/event-study evidence.

## Replay evaluation routing and evidence-plan verification (2026-07-11)

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests/dean_os/test_replay_evaluation_router.py tests/dean_os/test_event_study_eligibility.py tests/dean_os/test_hypothesis_gap_replay_packet.py tests/dean_os/test_world_model_replay_review_gate.py -q --basetemp D:\trading_project\.pytest_tmp\replay_routing
# 11 passed

python -m pytest tests/dean_os/test_replay_outcome_evidence_plan.py tests/dean_os/test_replay_evaluation_router.py tests/dean_os/test_event_study_eligibility.py -q --basetemp D:\trading_project\.pytest_tmp\replay_evidence_plan
# 7 passed
```

Current real artifacts:

```text
reports/dean_os/replay_evaluation_routing_current/latest.json
reports/dean_os/replay_outcome_evidence_plan_current/latest.json
```

Expected invariants: 9 hypothesis routes, 9 waiting, 0 event-study tasks, 11
unique gaps, collection allowed, outcome evaluation false, and no registration,
learning write or trading. Never reinterpret hypothesis `as_of` as a verified
event release timestamp.

## Industry operational metrics intake (2026-07-11)

```powershell
python run_agent_industry_operational_metrics.py data\dean_os\industry_operational_metrics.json --as-of 2026-07-11T00:00:00+00:00 --domain-id semiconductor --output-dir reports\dean_os\industry_operational_metrics_current

python run_agent_hypothesis_evidence_gap_review.py reports\dean_os\domain_analyst_review_clean_current\latest.json reports\dean_os\saved_sec_fundamental_evidence_current\latest.json --operational-metrics reports\dean_os\industry_operational_metrics_current\latest.json --as-of 2026-07-11T00:00:00+00:00 --output-dir reports\dean_os\hypothesis_evidence_gap_review_current

python -m pytest tests\dean_os\test_industry_operational_metrics.py tests\dean_os\test_hypothesis_evidence_gap_review.py -q --basetemp D:\trading_project\.tmp\pytest_operational_final
# 8 passed
```

Input must contain `records[]`; each row requires `record_id`, `entity`,
`metric_name`, numeric `value`, explicit `unit`, `period`, timezone-aware
`available_at`, `source_locator`, `source_sha256`, and `value_kind`. Do not turn
prose into numbers. Guidance/estimates are context, not observed support. Review
the artifact manually; it cannot close gaps, register replay, write learning or
trade.

## Expectation evidence contract v1 (2026-07-12)

```powershell
python -m pytest tests\dean_os\test_expectation_evidence.py tests\dean_os\test_analyst_core_phase2_lenses.py -q --basetemp D:\trading_project\.tmp\pytest_expectation_v3
# 45 passed

python -m pytest tests\dean_os\test_expectation_evidence.py tests\dean_os\test_analyst_core_phase2_lenses.py tests\dean_os\test_analyst_core_sector_analyst.py tests\dean_os\test_domain_analyst_agent.py tests\dean_os\test_world_model_event_learning_packet.py -q --basetemp D:\trading_project\.tmp\pytest_expectation_integration_v3
# 86 passed
```

Quantify `actual - expected` only from `dean_expectation_evidence_v1` with a
typed expectation source, structured actual/expected values, matching units,
point-in-time timestamps, source locators and SHA-256 hashes. Expectation must
be available no later than the actual observation. Never treat headline words,
flat source labels, IV, spreads or positioning as analyst consensus.

## Unknowns / value-of-information review (2026-07-12)

```powershell
python run_agent_replay_outcome_evidence_plan.py reports\dean_os\hypothesis_gap_replay_packet_current\latest.json reports\dean_os\replay_evaluation_routing_current\latest.json --output-dir reports\dean_os\replay_outcome_evidence_plan_current

python run_agent_unknown_voi_review.py reports\dean_os\replay_outcome_evidence_plan_current\latest.json --output-dir reports\dean_os\unknown_voi_review_current

python -m pytest tests\dean_os\test_unknown_graph_value_of_information.py tests\dean_os\test_unknown_voi_review.py tests\dean_os\test_replay_outcome_evidence_plan.py -q --basetemp D:\trading_project\.tmp\pytest_unknown_voi_review
# 8 passed
```

Current real invariants: 11 unique gaps, all `unassessed`, zero VoI scores, no
collector ordering and no collector execution. A validated assessment requires
all ordinal components plus assessor, timestamp and evidence basis. The score
is review triage only; it is not probability, expected monetary value,
automatic task authorization, learning write or trading authority.

## Unified review decision state (2026-07-12)

```powershell
python run_agent_review_decision_state.py reports\dean_os\replay_outcome_evidence_plan_current\latest.json reports\dean_os\unknown_voi_review_current\latest.json --previous-state blocked --actor dean_os_policy --output-dir reports\dean_os\review_decision_state_current

python -m pytest tests\dean_os\test_review_decision_state.py tests\dean_os\test_unknown_voi_review.py tests\dean_os\test_replay_outcome_evidence_plan.py -q --basetemp D:\trading_project\.tmp\pytest_review_decision_state
# 9 passed
```

Current expected state is `needs_more_data`: outcomes not matured, 42
unresolved evidence-lane references and 11 unscored VoI gaps. Do not interpret
this as architectural failure, and do not promote to `ready_for_review` without
a permitted hash-bound transition. Every state remains review-only and cannot
authorize collector execution, replay registration, learning or trading.

## Bounded VoI candidate proposal (2026-07-12)

```powershell
python run_agent_unknown_voi_candidate_proposal.py reports\dean_os\replay_outcome_evidence_plan_current\latest.json reports\dean_os\unknown_voi_review_current\latest.json --max-candidates 3 --output-dir reports\dean_os\unknown_voi_candidate_proposal_current

python -m pytest tests\dean_os\test_unknown_voi_candidate_proposal.py tests\dean_os\test_unknown_voi_review.py tests\dean_os\test_review_decision_state.py -q --basetemp D:\trading_project\.tmp\pytest_voi_candidate_proposal
# 10 passed
```

Expected candidates: actual backlog versus narrative claims, supplier equipment
orders, and actual production capacity/utilization. Expected inferred VoI
scores: zero. This artifact only narrows manual review; it cannot validate
assessment values, execute collectors, close gaps, learn or trade.

## Filing order/backlog proxy evidence (2026-07-12)

```powershell
python run_agent_filing_order_evidence.py --companyfacts AMD=data\dean_os\sec_companyfacts_raw\CIK0000002488\latest.json --companyfacts INTC=data\dean_os\sec_companyfacts_raw\CIK0000050863\latest.json --companyfacts NVDA=data\dean_os\sec_companyfacts_raw\CIK0001045810\latest.json --companyfacts TSM=data\dean_os\sec_companyfacts_raw\CIK0001046179\latest.json --as-of 2026-07-12T23:59:59+00:00 --max-age-days 730 --output-dir reports\dean_os\filing_order_evidence_current

python -m pytest tests\dean_os\test_filing_order_evidence.py tests\dean_os\test_hypothesis_evidence_gap_review.py -q --basetemp D:\trading_project\.tmp\pytest_filing_order_freshness
# 7 passed
```

Expected real result: 3 RPO observations, 2 current-gap eligible, 1 historical
context only, 0 full-backlog observations. RPO can support backlog only
partially. Purchase obligations, contract liabilities and narrative wording do
not become backlog. After rebuilding gap review, regenerate every downstream
hash-bound replay/routing/evidence/VoI/decision artifact.

## Industry operational source coverage (2026-07-12)

```powershell
python run_agent_industry_operational_source_coverage.py --duckdb data\trading_data.duckdb --research-sqlite data\dean_os\research_corpus.sqlite --knowledge-pack data\dean_os\verified_knowledge_packs\semiconductor_ai_infrastructure_2026-07-11.json --output-dir reports\dean_os\industry_operational_source_coverage_current

python -m pytest tests\dean_os\test_industry_operational_source_coverage.py tests\dean_os\test_replay_outcome_evidence_plan.py tests\dean_os\test_unknown_voi_candidate_proposal.py -q --basetemp D:\trading_project\.tmp\pytest_industry_source_coverage_v2
# 5 passed
```

Expected real gate: `structured_adapter_ready_source_feed_missing`, with zero
structured operational candidates. Narrative matches remain source-discovery
context only. Do not archive the replay-outcome, unknown-VoI, VoI-candidate or
review-decision-state runner scripts; they are active CLI entry points used by
this checklist.

## Synthetic collector production boundary (2026-07-12)

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_collector_synthetic_production_boundary.py tests\dean_os\test_config_yamls.py -q --basetemp D:\trading_project\.tmp\pytest_collector_synthetic_boundary_clean_v2
# 11 passed
```

No enabled collector may set `use_synthetic_data: true`. Development fixtures
must remain disabled and isolated from Stage1 production tables, analyst
evidence, calibration and replay outcomes. Treat newly enabled collectors as
unverified until source lineage, point-in-time timestamps and fallback behavior
are audited.

## Replay checkpoint monitor (2026-07-12)

```powershell
python run_agent_replay_checkpoint_monitor.py reports\dean_os\replay_outcome_evidence_plan_current\latest.json --as-of 2026-07-12T23:59:59+00:00 --output-dir reports\dean_os\replay_checkpoint_monitor_current

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_replay_checkpoint_monitor.py tests\dean_os\test_replay_outcome_evidence_plan.py tests\dean_os\test_replay_evaluation_router.py -q --basetemp D:\trading_project\.tmp\pytest_replay_checkpoint_monitor
# 6 passed
```

Expected current status: 9 collecting and 0 outcome reviews. Never evaluate
outcomes early. Rerun after source refreshes and near pre-due/due checkpoints;
this is a review artifact, not a collector scheduler or learning authorization.

## Prospective accumulation runbook (2026-07-12)

```powershell
$asOf=(Get-Date).ToUniversalTime().ToString('o')
python run_agent_prospective_accumulation_runbook.py --as-of $asOf

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_prospective_accumulation_runbook.py tests\dean_os\test_agent_cli_restore.py -q --basetemp D:\trading_project\.tmp\pytest_accumulation
# 4 passed
```

Expected real result: 9 replay tasks, 7/7 explicit lane runners, 7/7 current
artifacts, nearest pre-due review 2026-08-03, automatic execution false. The
runbook preserves all three market timeframes: 15m, 60m and 1d.

All `run_agent_*.py` wrappers in `.archive_temp` were restored to the project
root after an over-broad parallel cleanup. Do not archive a wrapper merely
because it is not imported as a Python module: these files are operator-facing
entry points and are referenced by architecture/checklist tests.

## Prospective accumulation schedule (2026-07-12)

```powershell
$asOf=(Get-Date).ToUniversalTime().ToString('o')
python run_agent_prospective_accumulation_schedule.py --as-of $asOf

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_prospective_accumulation_schedule.py tests\dean_os\test_prospective_accumulation_runbook.py tests\dean_os\test_agent_cli_restore.py -q --basetemp D:\trading_project\.tmp\pytest_accumulation_schedule
# 6 passed
```

Current expected schedule: 5 lanes due and only one authorization-ready command
(clean 15m/60m/1d). Sector-market is dependency-blocked; macro, news and policy
must resolve real source/as-of parameters first. `--help` is never an
executable refresh command, and `due_for_review` is never execution approval.

## Accumulation authorization ledger (2026-07-12)

```powershell
python run_agent_accumulation_authorization.py --verify-only

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_accumulation_authorization_ledger.py tests\dean_os\test_prospective_accumulation_schedule.py tests\dean_os\test_prospective_accumulation_runbook.py tests\dean_os\test_agent_cli_restore.py -q --basetemp D:\trading_project\.tmp\pytest_accumulation_auth
# 10 passed
```

To approve a lane, first calculate and visibly compare the exact command hash
using `--hash-command`, then provide that hash with `--confirm-command-sha256`,
plus `--approved-by` and a timezone-aware `--expires-at`. Approval appends an
audit record only; it does not execute the command. Never populate the real
ledger with a test or fabricated approval.

## Canonical system topology and current manifest (2026-07-12)

```powershell
$asOf=(Get-Date).ToUniversalTime().ToString('o')
python run_agent_current_system_manifest.py --as-of $asOf

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_system_topology_and_manifest.py tests\dean_os\test_accumulation_authorization_ledger.py -q --basetemp D:\trading_project\.tmp\pytest_system_manifest_v2
# 8 passed
```

Expected current result: `observed_complete`, nine branch records,
`operations_authorization` registered with a valid zero-record ledger, and both
`independent_branch_execution_claimed` and `operational_readiness_claimed`
false. `observed_complete` means artifact coverage is complete, not that the
autonomous system is production-complete.

## Active full-system review slice (2026-07-13)

```powershell
$asOf=(Get-Date).ToUniversalTime().ToString('o')
python run_agent_full_system_review_cycle.py --as-of $asOf `
  --news reports\dean_os\saved_semiconductor_news_evidence_producer_current\latest.json `
  --macro reports\dean_os\saved_macro_evidence_producer_current\latest.json `
  --sector-market reports\dean_os\saved_sector_market_evidence_producer_current\latest.json `
  --policy reports\dean_os\saved_official_policy_evidence_producer_current\latest.json `
  --fundamental reports\dean_os\saved_sec_fundamental_evidence_merger_current\latest.json `
  --timeframe-lane-readiness reports\dean_os\pipeline_timeframe_lane_readiness_clean_current\latest.json

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_full_system_review_cycle.py tests\dean_os\test_system_topology_and_manifest.py tests\dean_os\test_pipeline_readiness_agent.py tests\dean_os\test_pipeline_manager_agent.py -q --basetemp D:\trading_project\.tmp\pytest_full_system_cycle
# 19 passed
```

Expected real result: 76 evidence items, five lenses, analysis recommendation
`needs_more_data`, multitimeframe analysis context ready, and downstream refresh
required for world model, replay and governance. Ledger record count remains
zero. Never treat prior downstream artifacts as hash-bound to this cycle.

## Cycle-bound world model and governance closure (2026-07-13)

```powershell
python run_agent_full_system_cycle_world_model.py
python run_agent_full_system_cycle_closure.py

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\dean_os\test_full_system_cycle_closure.py tests\dean_os\test_full_system_cycle_world_model_bridge.py tests\dean_os\test_full_system_review_cycle.py tests\dean_os\test_system_topology_and_manifest.py -q --basetemp D:\trading_project\.tmp\pytest_cycle_closure
# 9 passed
```

Expected current closure: world model hash-bound to the current cycle, zero new
hypotheses/replay tasks, decision `needs_more_data`, and nine old replay tasks
continuing strictly under prior lineage. Authorization ledger record count must
remain zero unless a real operator approval was explicitly recorded.

## Full-context world-model replay review (2026-07-13, current)

The earlier zero/two-hypothesis expectations above are historical. Current
recovered evidence produces four aligned, trigger-only hypotheses.

```powershell
python run_agent_full_system_cycle_world_model.py `
  --cycle reports\dean_os\full_system_review_cycle_current\latest.json `
  --pipeline-context-bundle reports\dean_os\world_model_pipeline_context_clean_current\latest.json `
  --domain-id semiconductor_ai_infrastructure `
  --max-events 12 `
  --output-dir reports\dean_os\world_model_event_learning_cycle_current

python run_agent_world_model_replay_review_gate.py `
  --packet-json reports\dean_os\world_model_event_learning_cycle_current\latest.json `
  --output-dir reports\dean_os\world_model_replay_review_gate_cycle_current

python run_agent_full_system_cycle_closure.py `
  --cycle reports\dean_os\full_system_review_cycle_current\latest.json `
  --world-model reports\dean_os\world_model_event_learning_cycle_current\latest.json `
  --prior-checkpoint-monitor reports\dean_os\replay_checkpoint_monitor_current\latest.json `
  --replay-review-gate reports\dean_os\world_model_replay_review_gate_cycle_current\latest.json `
  --output-dir reports\dean_os\full_system_cycle_closure_current

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest `
  tests\dean_os\test_analyst_core_schemas.py `
  tests\dean_os\test_analyst_core_phase2_lenses.py `
  tests\dean_os\test_world_model_event_learning_packet.py `
  tests\dean_os\test_full_system_cycle_world_model_bridge.py `
  tests\dean_os\test_world_model_replay_review_gate.py `
  tests\dean_os\test_world_model_replay_registration_bridge.py `
  tests\dean_os\test_full_system_review_cycle.py `
  tests\dean_os\test_full_system_cycle_closure.py `
  -q -p no:cacheprovider `
  --basetemp D:\trading_project\.tmp\pytest_world_trigger_semantics
# 87 passed
```

Expected: 468 evidence items; 12 events from 12 unique sources; all six lanes;
four sector hypotheses at 30/90/180 days; four event-response hypotheses and
20 candidates at 1/5/20/60/120 days. Initial evidence role must be
`trigger_only`, never automatic support.

The unapproved gate must say
`manual_review_required_for_replay_registration`; closure must report manual
review submission true and registration false. Never use `--approve` merely to
advance the pipeline. First disposition each hypothesis/source relation. The
source SHA is part of the gate and a changed packet must fail registration.

For any future cycle-bound approval, prepare a reviewed JSON object mapping
every current hypothesis ID to exactly one of `accept_for_replay`,
`reformulate`, `defer`, or `reject`, then pass it with
`--hypothesis-dispositions-json`. `--review-notes` is mandatory. The command
must remain unapproved while any disposition is pending. Do not create this
file mechanically from the presence of replay tasks.

## Reviewed event-time replay state (2026-07-13, current)

The current disposition file is bound by hypothesis IDs to the current packet.
Do not reuse it after regenerating the world-model packet; a new run creates a
new review surface and must receive a new explicit mapping.

Record the completed content review without requesting approval:

```powershell
python run_agent_world_model_replay_review_gate.py `
  --packet-json reports\dean_os\world_model_event_learning_cycle_current\latest.json `
  --reviewer codex_content_review_recommendation `
  --review-notes "Content review only, not operator approval: demand is coherent for replay observation; capex, policy, and supply claims require narrower trigger-aligned reformulation. Event checkpoints are anchored to trigger publication time; matured checkpoints require historical point-in-time review." `
  --hypothesis-dispositions-json data\dean_os\world_model_hypothesis_dispositions_cycle_current.json `
  --output-dir reports\dean_os\world_model_replay_review_gate_cycle_current

python run_agent_full_system_cycle_closure.py `
  --cycle reports\dean_os\full_system_review_cycle_current\latest.json `
  --world-model reports\dean_os\world_model_event_learning_cycle_current\latest.json `
  --prior-checkpoint-monitor reports\dean_os\replay_checkpoint_monitor_current\latest.json `
  --replay-review-gate reports\dean_os\world_model_replay_review_gate_cycle_current\latest.json `
  --output-dir reports\dean_os\full_system_cycle_closure_current
```

Expected current result: 20/20 tasks have exact event anchors; 11 checkpoints
are matured and 9 scheduled. The four dispositions are 1
`accept_for_replay` and 3 `reformulate`. Gate status is
`hypothesis_review_complete_reformulation_required`; closure status is
`current_cycle_hypothesis_review_complete_reformulation_required`; replay
registration, learning and trading remain false.

Regression set:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest `
  tests\dean_os\test_analyst_core_schemas.py `
  tests\dean_os\test_analyst_core_phase2_lenses.py `
  tests\dean_os\test_world_model_event_learning_packet.py `
  tests\dean_os\test_full_system_cycle_world_model_bridge.py `
  tests\dean_os\test_world_model_replay_review_gate.py `
  tests\dean_os\test_world_model_replay_registration_bridge.py `
  tests\dean_os\test_full_system_review_cycle.py `
  tests\dean_os\test_full_system_cycle_closure.py `
  tests\test_outcome_tracker.py `
  -q -p no:cacheprovider `
  --basetemp D:\trading_project\.tmp\pytest_world_event_anchor_full
# 95 passed
```

Do not call `--approve` on this mixed packet. First create a new hash-bound
packet that applies the three claim reformulations while preserving the one
accepted demand claim and the original review lineage.

## Canonical journal and failure learning (2026-07-13)

Build the root-cause review. This creates proposals only; it cannot update
learning memory or production rules:

```powershell
python run_agent_hypothesis_learning_review.py `
  --packet-json reports\dean_os\world_model_event_learning_cycle_current\latest.json `
  --review-gate-json reports\dean_os\world_model_replay_review_gate_cycle_current\latest.json `
  --journal-path data\dean_os\system_journal.jsonl `
  --output-dir reports\dean_os\hypothesis_learning_review_current
```

Preview the current-cycle journal import:

```powershell
python run_agent_current_cycle_journal.py
```

Append the verified cycle idempotently:

```powershell
python run_agent_current_cycle_journal.py --apply
```

Expected current state: 430 total records and a valid hash chain. The first
import adds 430; an identical rerun adds 0 and reports 430 existing records.
There must be 396 `news_observed`, 4 `hypothesis_created`, 4
`hypothesis_reviewed`, 3 `action_proposed`, 3 `learning_proposal_created` and
no `action_executed` event.

Run the dedicated tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest `
  tests\dean_os\test_system_journal.py `
  tests\dean_os\test_hypothesis_learning_review.py `
  tests\dean_os\test_current_cycle_journal.py `
  -q -p no:cacheprovider `
  --basetemp D:\trading_project\.tmp\pytest_system_journal
```

Full regression result after this slice: 100 passed (the prior 95-test set plus
the five new journal/learning tests).

When a matured outcome is supplied, it must identify `hypothesis_id`, result
label and, when known, an error label. A `falsified`/`miss` result without a
root-cause label must remain `unknown_falsification_cause`; do not infer a
template change from market direction alone. Empirical rule candidates require
3 independent reviewed cases and a separate human promotion ceremony.

## Review resolution packet (2026-07-13)

Create the new versioned packet from the immutable source packet, completed
review gate and manual resolution specs:

```powershell
python run_agent_world_model_review_resolution.py
```

Record the resolved content decisions without requesting registration:

```powershell
python run_agent_world_model_replay_review_gate.py `
  --packet-json reports\dean_os\world_model_review_resolution_current\latest.json `
  --reviewer codex_content_review_recommendation `
  --review-notes "Resolved-claim content review only, not operator registration approval: retain Applied Materials demand for observation; defer capex, BIS and ASML claims until their named point-in-time measurement context is attached." `
  --hypothesis-dispositions-json data\dean_os\world_model_hypothesis_dispositions_resolved_current.json `
  --output-dir reports\dean_os\world_model_replay_review_gate_resolved_current
```

Close the resolved cycle state:

```powershell
python run_agent_full_system_cycle_closure.py `
  --cycle reports\dean_os\full_system_review_cycle_current\latest.json `
  --world-model reports\dean_os\world_model_review_resolution_current\latest.json `
  --prior-checkpoint-monitor reports\dean_os\replay_checkpoint_monitor_current\latest.json `
  --replay-review-gate reports\dean_os\world_model_replay_review_gate_resolved_current\latest.json `
  --output-dir reports\dean_os\full_system_cycle_closure_resolved_current
```

Append the version/review/closure trace to the canonical journal:

```powershell
python run_agent_world_model_resolution_journal.py
```

Expected: gate `hypothesis_review_complete_deferred`, closure
`current_cycle_hypothesis_review_complete_deferred`, decision
`deferred_pending_evidence`, journal 446 records and a valid chain. A repeated
journal command must add zero records. Do not pass `--approve` unless the
operator explicitly authorizes demand-only replay registration; content review
does not confer that authority.

Resolution regression:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest `
  tests\dean_os\test_world_model_review_resolution.py `
  tests\dean_os\test_world_model_replay_review_gate.py `
  tests\dean_os\test_full_system_cycle_closure.py `
  tests\dean_os\test_system_journal.py `
  -q -p no:cacheprovider `
  --basetemp D:\trading_project\.tmp\pytest_review_resolution
```

Focused result: 17 passed. Full world-model/journal regression: 104 passed.

## Capex measurement-ready resolved review (2026-07-13)

Rebuild the hash-bound resolution packet after reviewing the official capex
baselines in
`data/dean_os/world_model_hypothesis_resolution_specs_cycle_current.json`:

```powershell
python run_agent_world_model_review_resolution.py
```

Record the 2 accept / 2 defer content decision without operator approval:

```powershell
python run_agent_world_model_replay_review_gate.py `
  --packet-json reports\dean_os\world_model_review_resolution_current\latest.json `
  --reviewer codex_content_review_recommendation `
  --review-notes "Content review only, not operator registration approval: capex is measurement-ready from official pre-trigger plans and predeclared baskets; retain demand; defer BIS and ASML until their named contexts are attached." `
  --hypothesis-dispositions-json data\dean_os\world_model_hypothesis_dispositions_resolved_current.json `
  --output-dir reports\dean_os\world_model_replay_review_gate_resolved_current
```

Then rerun closure and append the resolution trace with the commands in the
previous section. Expected current state: 2 content-ready hypotheses, 10
content-ready checkpoints, 0 operator-approved/registered checkpoints, 2
deferred hypotheses, journal 472 and a valid chain.

Refresh the actionable learning report and append it idempotently:

```powershell
python run_agent_hypothesis_learning_review.py `
  --packet-json reports\dean_os\world_model_event_learning_cycle_current\latest.json `
  --review-gate-json reports\dean_os\world_model_replay_review_gate_cycle_current\latest.json `
  --journal-path data\dean_os\system_journal.jsonl `
  --output-dir reports\dean_os\hypothesis_learning_review_current

python run_agent_current_cycle_journal.py --apply
```

Do not use `--approve` until the operator explicitly authorizes registration.

Integrated regression result for this slice: 105 passed.

## Hypothesis quality cards and outcome separation (2026-07-13)

The resolved replay review report now scores hypothesis readiness without
claiming forecast accuracy. Rebuild it with the same content-only command in
the previous section; do not add `--approve`.

Expected current state:

- capex and demand: 69/100, `moderate`, replay quality floor met;
- BIS and ASML: 39/100, `weak`, deferred below the quality floor;
- quality bands: 2 moderate / 2 weak;
- truth probabilities: unavailable/uncalibrated for all four;
- content-ready checkpoints: 10; operator-approved checkpoints: 0;
- registration, learning-memory writes and trading: false.

Run the quality and gate checks:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest `
  tests\dean_os\test_hypothesis_quality_assessment.py `
  tests\dean_os\test_world_model_replay_review_gate.py `
  -q -p no:cacheprovider `
  --basetemp D:\trading_project\.tmp\pytest_hypothesis_quality
```

Expected: 11 passed. The wider world-model, evidence, governance and journal
regression passes 126 tests. After rebuilding the resolved gate, rerun closure
and `python run_agent_world_model_resolution_journal.py`; the current journal
has 490 records and a valid SHA-256 chain. Repeating the same resolution import
adds zero records.
## Hypothesis reverse analysis after outcomes mature (2026-07-13)

The normal learning-review command now also writes the dedicated card report to
`reports\dean_os\hypothesis_reverse_analysis_current\latest.md`.

Without matured outcomes it builds pre-outcome diagnostic cards:

```powershell
python run_agent_hypothesis_learning_review.py `
  --packet-json reports\dean_os\world_model_event_learning_cycle_current\latest.json `
  --review-gate-json reports\dean_os\world_model_replay_review_gate_cycle_current\latest.json `
  --journal-path data\dean_os\system_journal.jsonl `
  --output-dir reports\dean_os\hypothesis_learning_review_current
```

After a verified outcome artifact matures, add:

```powershell
  --outcome-json <verified-hypothesis-outcomes.json>
```

Then preview and append the cards to the canonical journal:

```powershell
python run_agent_current_cycle_journal.py
python run_agent_current_cycle_journal.py --apply
```

The apply command records analysis and proposals only. It does not promote rules,
write learning memory, register replay tasks or trade. Current verified journal:
504 records, valid SHA-256 chain.

Focused verification:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest `
  tests\dean_os\test_hypothesis_quality_assessment.py `
  tests\dean_os\test_hypothesis_reverse_analysis.py `
  tests\dean_os\test_hypothesis_learning_review.py `
  tests\dean_os\test_world_model_replay_review_gate.py `
  tests\dean_os\test_current_cycle_journal.py `
  tests\dean_os\test_system_journal.py `
  -q -p no:cacheprovider `
  --basetemp D:\trading_project\.tmp\pytest_hypothesis_reverse
```

Expected: 20 passed.
## Applied observation-only replay state (2026-07-13)

The approved registration sequence has completed. Canonical artifacts:

- approved gate: `reports\dean_os\world_model_replay_review_gate_approved_current\latest.json`;
- registration: `reports\dean_os\world_model_replay_registration_approved_current\latest.json`;
- closure: `reports\dean_os\full_system_cycle_closure_approved_current\latest.json`.

To rebuild the post-registration closure, include both the approved gate and the
applied registration artifact:

```powershell
python run_agent_full_system_cycle_closure.py `
  --cycle reports\dean_os\full_system_review_cycle_current\latest.json `
  --world-model reports\dean_os\world_model_review_resolution_current\latest.json `
  --prior-checkpoint-monitor reports\dean_os\replay_checkpoint_monitor_current\latest.json `
  --replay-review-gate reports\dean_os\world_model_replay_review_gate_approved_current\latest.json `
  --replay-registration reports\dean_os\world_model_replay_registration_approved_current\latest.json `
  --output-dir reports\dean_os\full_system_cycle_closure_approved_current
```

Append the registration audit trace idempotently:

```powershell
python run_agent_world_model_replay_registration_journal.py
```

Expected current state: first append 15, repeated append 0; journal 519 and valid.
Tracker: 5 events / 5 predictions / 0 outcomes. Five historical tasks remain for
point-in-time review. No learning or trading action is authorized.

## Verified historical checkpoint market pass (2026-07-13)

```powershell
python run_agent_clean_yahoo_market_snapshot.py `
  --ticker AMAT --ticker LRCX --ticker KLAC --ticker ASML --ticker SOXX `
  --timeframe 1d `
  --end-date 2026-07-13T00:00:00+00:00 `
  --artifact-dir data\dean_os\historical_outcome_market_snapshots `
  --report-dir reports\dean_os\historical_outcome_market_snapshot_current

python run_agent_historical_replay_outcome_review.py `
  --price-path data\dean_os\historical_outcome_market_snapshots\clean_yahoo_market_2026-07-13T180116.915465Z0000.parquet
```

The audit also binds the accumulated and regenerated Stage 2/3 pipeline feature
artifacts as secondary context. Expected: 5 historical tasks, 1 unobservable
primary outcome, 4 unresolved intermediate checkpoints, no automatic outcome
scoring and no trading.

## Calibrate a direction-neutral relative-return policy for a new hypothesis

Run this before registration and bind the produced
`measurement_spec_patch_for_new_hypotheses` into a v2 resolution spec:

```powershell
python run_agent_relative_return_direction_policy.py `
  --price-path data\dean_os\historical_outcome_market_snapshots\latest.parquet `
  --member AMAT --member LRCX --member KLAC --member ASML `
  --benchmark SOXX `
  --calibration-cutoff-at 2026-06-25T07:53:06+00:00 `
  --horizon-days 20 `
  --expected-direction negative
```

Current expected diagnostic result: 475 strictly pre-cutoff samples and a
4.367% neutral band. Do not retrofit this contract into the already reviewed
current hypothesis; use it in new v2 hypothesis resolution specs.

## Future hypothesis measurement-policy stage

For every new cycle, use the v2 draft template at
`dean_os\config\world_model_resolution_specs_v2.template.json`, fill the
reviewed claim, explicit relative-return direction, universe, benchmark and
source bindings, then run:

```powershell
python run_agent_hypothesis_measurement_policy_preparer.py `
  --resolution-specs-json <new_v2_draft.json> `
  --price-path <verified_market_snapshot.parquet> `
  --output-dir reports\dean_os\hypothesis_measurement_policy_prepared_current

python run_agent_world_model_review_resolution.py `
  --resolution-specs-json reports\dean_os\hypothesis_measurement_policy_prepared_current\latest.json
```

The preparer automatically also checks the saved accumulated and regenerated
pipeline artifacts. Exact verified-price rows take precedence over duplicate
pipeline rows. Missing direction, universe, benchmark, cutoff or sufficient
history becomes a registration blocker; the agent does not guess.

The composed review-only lifecycle can run all three safe stages and stop at the
new manual gate:

```powershell
python run_agent_world_model_hypothesis_lifecycle.py `
  --packet-json <world_model_packet.json> `
  --source-review-gate-json <completed_source_review_gate.json> `
  --resolution-specs-v2-json <new_v2_draft.json> `
  --price-path <verified_market_snapshot.parquet>
```

Expected successful status:
`prepared_resolved_pending_manual_review`. Replay registration, learning writes
and trading remain false.

To surface the lifecycle in the existing Chief Review inbox:

```powershell
python run_agent_chief_review_index.py `
  --hypothesis-lifecycle-path reports\dean_os\world_model_hypothesis_lifecycle_current\latest.json
```

Chief Review shows only lifecycle blockers, proposed contracts and pending
decisions. It does not approve or register them.

## Evidence-aware checkpoint due router

Build the automatic checkpoint inbox from the approved registration, verified
prices, saved pipeline context and prior outcome reviews:

```powershell
python run_agent_replay_checkpoint_due_router.py
python run_agent_chief_review_index.py
```

The default router binds:

- `reports\dean_os\world_model_replay_registration_approved_current\latest.json`;
- `reports\dean_os\world_model_replay_review_gate_approved_current\latest.json`;
- `data\dean_os\historical_outcome_market_snapshots\latest.parquet`;
- the accumulated and regenerated pipeline feature artifacts;
- `reports\dean_os\historical_replay_outcome_review_current\latest.json`.

Expected current state after demand 60d became due: 10 tasks, 5 already
reviewed, 4 future/silent, 1 due-soon/silent, 1 waiting for verified data and 0
matured operator decisions. A due market task with no
verified post-close checkpoint session moves to
`due_waiting_for_verified_checkpoint_data`; it is not judged and does not
become a human decision. Future checkpoints never appear in the action list.

Focused verification:

```powershell
python -m pytest `
  tests\dean_os\test_replay_checkpoint_due_router.py `
  tests\dean_os\test_chief_review_hypothesis_lifecycle_inbox.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_checkpoint_router
```

Expected: 9 passed. Cross-layer router, Chief Review, historical outcome,
hypothesis lifecycle, measurement preparer and direction-policy verification:
17 passed. The router performs no network collection, scoring, learning,
registration or trading.

## Composed replay outcome lifecycle

Run the full read-only outcome path:

```powershell
python run_agent_replay_outcome_lifecycle.py
python run_agent_chief_review_index.py
```

Current expected status is `waiting_for_verified_checkpoint_data` for demand
60d, with one structured refresh recommendation and zero outcome packets,
reverse-analysis runs, learning proposals or human decisions. The lifecycle
does not use the July 10 snapshot as a July 13 outcome.

After a verified snapshot contains the routed post-close session, the same
command:

1. selects only matured task IDs from the approved registration;
2. builds a scoped outcome packet using verified prices plus pipeline context;
3. records intermediate checkpoints without final hypothesis scoring;
4. invokes reverse analysis for primary outcomes only;
5. reruns the SHA-bound router so processed tasks cannot reappear.

Focused verification:

```powershell
python -m pytest `
  tests\dean_os\test_replay_outcome_lifecycle_orchestrator.py `
  tests\dean_os\test_replay_checkpoint_due_router.py `
  tests\dean_os\test_chief_review_hypothesis_lifecycle_inbox.py `
  tests\dean_os\test_historical_replay_outcome_review.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_outcome_lifecycle_chief
```

Expected: 15 passed. Collection, causal approval, scoring, learning-rule
promotion, registration and trading remain disabled.

## Controlled replay evidence refresh and journal

Preview the single-pass refresh plan:

```powershell
python run_agent_replay_evidence_refresh.py
```

Execute one allowlisted refresh and append its audit events:

```powershell
python run_agent_replay_evidence_refresh.py --apply-refresh --apply-journal
python run_agent_chief_review_index.py
```

Current job is AMAT/1d from the hash-bound demand measurement spec. The latest
Yahoo attempt returned no rows and is recorded as
`single_refresh_pass_failed`. Do not automatically repeat it. Expected lifecycle
state remains `waiting_for_verified_checkpoint_data`; missing source data is not
a negative outcome.

The standalone idempotent journal command is:

```powershell
python run_agent_replay_lifecycle_journal.py --apply
```

Current journal result: first bridge append 3, repeated append 0, total 573,
valid SHA-256 chain.

Focused verification:

```powershell
python -m pytest `
  tests\dean_os\test_replay_evidence_refresh_controller.py `
  tests\dean_os\test_replay_lifecycle_journal_bridge.py `
  tests\dean_os\test_replay_outcome_lifecycle_orchestrator.py `
  tests\dean_os\test_replay_checkpoint_due_router.py `
  tests\dean_os\test_chief_review_hypothesis_lifecycle_inbox.py `
  tests\dean_os\test_historical_replay_outcome_review.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_refresh_journal_full
```

Expected: 19 passed. No automatic retry loop, learning promotion or trading is
allowed.

## Ranked verified-market source router

After a provider attempt fails, compute the next bounded route:

```powershell
python run_agent_verified_market_source_router.py
python run_agent_chief_review_index.py
```

Current expected status:
`awaiting_operator_supplied_verified_snapshot`. Yahoo is exhausted for demand
60d; required local coverage is AMAT/1d. No automatic retry or provider loop is
allowed.

Validate a candidate without ingesting it:

```powershell
python run_agent_verified_market_source_router.py `
  --local-snapshot <point_in_time_AMAT_daily.csv_or_parquet>
```

The candidate must contain `datetime`, `ticker`, `close`, timezone-aware values,
finite closes and a complete AMAT session after the task due time whose US close
is not later than `as_of`. A valid candidate produces
`verified_local_snapshot_ready`; invalid files remain rejected and cannot enter
the outcome lifecycle.

Focused verification:

```powershell
python -m pytest `
  tests\dean_os\test_verified_market_source_router.py `
  tests\dean_os\test_chief_review_hypothesis_lifecycle_inbox.py `
  tests\dean_os\test_replay_evidence_refresh_controller.py `
  tests\dean_os\test_replay_lifecycle_journal_bridge.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_source_router_chief
```

Expected: 13 passed. This stage routes and validates only; it does not fetch,
ingest, score, promote or trade.

## Verified local snapshot ingestion ceremony

Current no-candidate state:

```powershell
python run_agent_verified_local_snapshot_ingestion.py `
  --as-of <timezone-aware-as-of>
```

Expected: `awaiting_candidate`, no write and no polling.

Preview a real candidate:

```powershell
python run_agent_verified_local_snapshot_ingestion.py `
  --candidate <validated_AMAT_daily.csv_or.parquet> `
  --as-of <timezone-aware-as-of>
```

Only after preview reports `candidate_valid_ready_for_ingestion`:

```powershell
python run_agent_verified_local_snapshot_ingestion.py `
  --candidate <same_unchanged_file> `
  --as-of <same-as-of> `
  --apply-ingestion `
  --apply-journal
```

Apply atomically writes one immutable canonical parquet, runs the existing
outcome lifecycle once and journals the source separately from any later
outcome. Never create a placeholder candidate to satisfy this gate.

Focused verification:

```powershell
python -m pytest `
  tests\dean_os\test_verified_local_snapshot_ingestion.py `
  tests\dean_os\test_verified_market_source_router.py `
  tests\dean_os\test_replay_outcome_lifecycle_orchestrator.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_local_ingestion
```

Expected: 9 passed. Journal source/outcome separation checks: 2 passed.

## Reusable domain analyst lifecycle profile

```powershell
python run_agent_domain_analyst_lifecycle_profile.py
```

Expected: clone contract materialized `True`, analysis runnable `False`, and
six missing context bindings. This is a portability dry run, not an operational
energy analysis.

Focused verification:

```powershell
python -m pytest `
  tests\dean_os\test_domain_analyst_lifecycle_profile.py `
  tests\dean_os\test_config_yamls.py `
  tests\dean_os\test_domain_analyst_profile_policy_packet.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_domain_profile
```

Do not mark a binding `configured` until its domain-specific news, policy,
macro, fundamentals, sector-market or pipeline-context artifact contract has
been validated.

## Energy profile binding plan

```powershell
python run_agent_domain_analyst_binding_plan.py `
  --domain energy `
  --as-of 2026-07-14T12:00:00Z
```

Expected: 6 unresolved bindings, 6 proposal-only tasks, collection and analyst
invocation both false.

Validate an explicitly supplied candidate without binding it:

```powershell
python run_agent_domain_analyst_binding_plan.py `
  --domain energy `
  --candidate news=<domain-scoped-review-artifact.json> `
  --as-of <timezone-aware-cutoff>
```

The candidate must match the energy domain, allowed producer contract, cutoff
and review-only safety boundary. Passing validation creates a reuse proposal,
not an accepted binding.

Focused verification:

```powershell
python -m pytest `
  tests\dean_os\test_domain_analyst_binding_planner.py `
  tests\dean_os\test_domain_analyst_lifecycle_profile.py `
  tests\dean_os\test_config_yamls.py `
  tests\dean_os\test_domain_analyst_profile_policy_packet.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_binding_plan
```

Expected: 23 passed.

## Bounded energy binding task dispatch

```powershell
python run_agent_domain_binding_task_dispatch.py
```

Expected: `dispatch_plan_ready_no_executable_tasks`, 6 generalization tasks,
execution eligible 0, next priority `macro`.

Focused verification:

```powershell
python -m pytest `
  tests\dean_os\test_domain_binding_task_dispatcher.py `
  tests\dean_os\test_domain_analyst_binding_planner.py `
  tests\dean_os\test_domain_analyst_lifecycle_profile.py `
  tests\dean_os\test_config_yamls.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_dispatch
```

Expected: 26 passed. This dispatcher currently classifies only; do not change
`execution_eligible` manually or run a producer with semiconductor defaults for
energy.

## Energy domain-scoped macro envelope

No-source preview and journal proposal:

```powershell
python run_agent_domain_scoped_macro_envelope.py `
  --as-of <timezone-aware-cutoff> `
  --apply-journal
```

Preview one explicit local source:

```powershell
python run_agent_domain_scoped_macro_envelope.py `
  --source <macro.csv-or-parquet> `
  --as-of <same-cutoff> `
  --apply-journal
```

Required source columns include observation time, series id, value and an
authoritative point-in-time availability field such as `available_at`,
`released_at` or `realtime_start`. Never replace it with file mtime.

Current canonical pipeline file is blocked because it contains only
`datetime/series/value/hash`. Expected current status:
`blocked_macro_core_not_ready`, candidate false, binding false.

Focused verification:

```powershell
python -m pytest tests\dean_os\test_domain_scoped_macro_envelope.py -q `
  --basetemp D:\trading_project\reports\pytest_tmp_macro_envelope
```

Expected: 7 passed. Repeating an identical journaled preview must append 0.

## Upstream macro point-in-time contract

Focused contract tests:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest tests\unit\test_macro_point_in_time_contract.py -q `
  --basetemp D:\trading_project\reports\pytest_tmp_macro_contract
```

Expected: 8 passed. Compatibility regression includes the existing
normalizer checks and macro envelope suite.

The pipeline contract requires `realtime_start`, `released_at` or
`available_at`. A fresh valid Stage 2 run writes the canonical persistent macro
parquet atomically. Never retrofit the old file with observation dates or file
mtime.

Current partial binding candidate:

```powershell
python run_agent_domain_analyst_binding_plan.py `
  --domain energy `
  --candidate macro=reports\dean_os\domain_scoped_macro_envelope_current\latest.json `
  --as-of <candidate-as-of>
```

Current result: macro reuse candidate ready, DGS10 present, six requested
series missing, binding still unresolved. Recommendation: replace or defer.

## Energy macro binding quality review

Score the exact candidate currently SHA-bound in the binding plan:

```powershell
python run_agent_domain_macro_binding_quality_review.py `
  --domain energy `
  --review-as-of <timezone-aware-review-cutoff> `
  --apply-journal
```

Current expected result: `replace_candidate`, score 0.200, required coverage
0%, supporting coverage 20%, total coverage 14.3%, decision false and binding
false. Repeating with the same review cutoff and unchanged candidate appends 0
journal records.

Focused verification:

```powershell
python -m pytest tests\dean_os\test_domain_macro_binding_quality_review.py -q `
  --basetemp D:\trading_project\reports\pytest_tmp_macro_quality
```

Expected: 6 passed. `accept_binding` in this report is still only a machine
recommendation; an explicit SHA-bound gate must record the actual decision.

## Exact energy macro collection request

Prepare and journal one proposal without running the collector:

```powershell
python run_agent_domain_macro_collection_request.py `
  --domain energy `
  --request-as-of <timezone-aware-cutoff> `
  --apply-journal
```

Current expected result: `macro_collection_request_ready`, replacement scope 7,
gap scope 6, missing required 2, execution authorized false and collector run
false. Repeating with the same cutoff and unchanged SHA-bound inputs appends 0.

Focused verification:

```powershell
python -m pytest `
  tests\dean_os\test_domain_macro_collection_request.py `
  tests\unit\test_macro_point_in_time_contract.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_macro_request
```

Expected: 14 passed. This command prepares the request only; it does not need a
FRED API key and must not be confused with collection authorization.

## Bounded macro collection execution gate

Run preflight only; this does not call FRED:

```powershell
python run_agent_domain_macro_collection_execution_gate.py `
  --domain energy `
  --evaluated-at <timezone-aware-cutoff> `
  --apply-journal
```

Current expected result: `macro_collection_execution_ready_single_run`,
credential present true, one exact ticket issued, collector run false and
network false. The project configuration bootstrap may load the key from the
existing `.env`; the value must never appear in output or journal records.

Focused verification:

```powershell
python -m pytest `
  tests\dean_os\test_domain_macro_collection_execution_gate.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_macro_execution_gate
```

Expected: 6 passed. This is authorization/preflight only. Do not manually edit
the ticket or reuse it for a second collection run.

## Completed single-use macro execution

The canonical ticket has already been consumed. Do **not** run the executor
again or create a replacement ticket merely to repeat collection.

Current artifacts:

- Executor: `reports/dean_os/domain_macro_collection_executor_current/latest.md`
- Retrieval receipt: `reports/dean_os/domain_macro_retrieval_receipt_current/latest.md`
- Full envelope: `reports/dean_os/domain_scoped_macro_envelope_current/latest.md`
- Quality review: `reports/dean_os/domain_macro_binding_quality_review_current/latest.md`

Expected final state: 1,596 persisted rows, 7/7 series, quality 1.000 strong,
recommendation `accept_binding`, decision false and binding false.

Security action: rotate `FRED_API_KEY` externally before any future FRED call.
Do not place the replacement value in reports, commands, chat, tests or journal
records. Local logs are scrubbed and executor HTTP URL logging is suppressed.

Full macro vertical verification:

```powershell
python -m pytest `
  tests\dean_os\test_domain_macro_collection_executor.py `
  tests\dean_os\test_domain_macro_collection_execution_gate.py `
  tests\dean_os\test_domain_macro_collection_request.py `
  tests\dean_os\test_domain_macro_binding_quality_review.py `
  tests\dean_os\test_domain_scoped_macro_envelope.py `
  tests\dean_os\test_domain_analyst_binding_planner.py `
  tests\dean_os\test_domain_analyst_lifecycle_profile.py `
  tests\dean_os\test_config_yamls.py `
  tests\unit\test_macro_point_in_time_contract.py `
  -q --basetemp D:\trading_project\reports\pytest_tmp_macro_final
```

Expected: 60 passed.

## Phase 8 maturity and execution boundary

```powershell
python -m pytest `
  tests\dean_os\test_phase8_maturity_execution_gates.py `
  dean_os\stress\test_phase8.py `
  -q --basetemp D:\trading_project\.tmp\pytest_phase8
```

Expected: 14 passed. This verifies simulated paper/shadow boundaries only;
`supervised_live` must remain blocked.

## Universal context-acquisition state machine

```powershell
python -m pytest `
  tests\dean_os\test_context_acquisition_state_machine.py `
  tests\dean_os\test_domain_macro_collection_executor.py `
  tests\dean_os\test_domain_macro_collection_execution_gate.py `
  tests\dean_os\test_domain_macro_collection_request.py `
  tests\dean_os\test_domain_macro_binding_quality_review.py `
  tests\dean_os\test_domain_scoped_macro_envelope.py `
  tests\dean_os\test_domain_analyst_binding_planner.py `
  tests\dean_os\test_domain_analyst_lifecycle_profile.py `
  tests\dean_os\test_config_yamls.py `
  tests\unit\test_macro_point_in_time_contract.py `
  -q --basetemp D:\trading_project\.tmp\pytest_context_orchestrator_regression
```

Expected: 72 passed. This command performs no network collection and records
no binding decision.

## Two-family context orchestration and maturity operations

```powershell
python -m pytest `
  tests\dean_os\test_context_acquisition_state_machine.py `
  tests\dean_os\test_domain_scoped_pipeline_context_envelope.py `
  tests\dean_os\test_pipeline_context_evidence_loader.py `
  tests\dean_os\test_world_model_pipeline_context.py `
  tests\dean_os\test_world_model_event_learning_packet.py `
  tests\dean_os\test_domain_macro_collection_executor.py `
  tests\dean_os\test_domain_macro_collection_execution_gate.py `
  tests\dean_os\test_domain_macro_collection_request.py `
  tests\dean_os\test_domain_macro_binding_quality_review.py `
  tests\dean_os\test_domain_scoped_macro_envelope.py `
  tests\dean_os\test_domain_analyst_binding_planner.py `
  tests\dean_os\test_domain_analyst_lifecycle_profile.py `
  tests\dean_os\test_config_yamls.py `
  tests\unit\test_macro_point_in_time_contract.py `
  tests\dean_os\test_strategy_maturity_operations.py `
  tests\dean_os\test_phase8_maturity_execution_gates.py `
  dean_os\stress\test_phase8.py `
  -q --basetemp D:\trading_project\.tmp\pytest_orchestrator_maturity_regression
```

Expected: 116 passed.

## Canonical reasoning receipt and hypothesis lifecycle

Build a deterministic review-only snapshot using the existing hypothesis
journal (no paid API and no journal write):

```powershell
python run_agent_analyst_core_reasoning_snapshot.py `
  --runtime-json <verified-runtime.json> `
  --hypothesis-journal-path data\dean_os\system_journal.jsonl `
  --no-save
```

Import a reviewed snapshot into the canonical journal only after its inputs are
the intended cycle artifacts:

```powershell
python run_agent_current_cycle_journal.py `
  --cycle-json <cycle.json> `
  --reasoning-snapshot-json <reasoning-snapshot.json> `
  --apply
```

This records machine proposals as `hypothesis_assessed`, not
`hypothesis_reviewed`; it does not approve replay or learning.

## Sector-market adapter

Verify one existing saved sector-market artifact without running a producer,
pipeline, or network call:

```powershell
python run_agent_domain_scoped_sector_market_envelope.py <domain_id> `
  --as-of <timezone-aware-cutoff> `
  --source-path <saved-sector-market-artifact.json> `
  --no-save
```

The artifact must match the domain profile's complete primary universe and
benchmark. The current semiconductor artifact must not be reused as energy
evidence; the adapter will correctly block it.

Focused regression:

```powershell
python -m pytest -q `
  tests\dean_os\test_hypothesis_journal_projection.py `
  tests\dean_os\test_current_cycle_journal.py `
  tests\dean_os\test_world_model_hypothesis_lifecycle_orchestrator.py `
  tests\dean_os\test_domain_scoped_sector_market_envelope.py `
  tests\dean_os\test_context_acquisition_state_machine.py
```

## Fundamentals adapter

Verify the explicit terminal SEC ratio artifact and its recursive saved lineage
without running a producer or writing the journal:

```powershell
python run_agent_domain_scoped_fundamentals_envelope.py `
  semiconductor_ai_infrastructure `
  --as-of 2026-06-30T21:00:00+00:00 `
  --source-path reports\dean_os\saved_sec_derived_ratio_producer_current\latest.json `
  --dispatch-path reports\dean_os\fundamentals_semis_dispatch_current\latest.json `
  --no-save
```

Current expected result: `domain_fundamentals_candidate_ready_with_gaps`,
recursive lineage true, issuer identity true, configured coverage 4/4, profile
coverage 4/12, binding false and trading false. Do not accept it as complete
sector fundamentals.

Canonical review files:

- `reports/dean_os/domain_scoped_fundamentals_envelope_current/latest.md`
- `reports/dean_os/fundamentals_semis_binding_review_current/latest.md`

Focused verification:

```powershell
python -m pytest -q `
  tests\dean_os\test_domain_scoped_fundamentals_envelope.py `
  tests\dean_os\test_domain_analyst_binding_planner.py `
  tests\dean_os\test_domain_binding_task_dispatcher.py `
  tests\dean_os\test_context_acquisition_state_machine.py `
  tests\dean_os\test_saved_sec_fundamental_evidence_merger.py `
  tests\dean_os\test_saved_sec_derived_ratio_producer.py
```

## News adapter and optional LLM boundary

Verify the existing saved semiconductor news artifact without collecting news,
calling an LLM or writing the journal:

```powershell
python run_agent_domain_scoped_news_envelope.py `
  semiconductor_ai_infrastructure `
  --as-of 2026-06-30T21:00:00+00:00 `
  --source-path reports\dean_os\saved_semiconductor_news_evidence_producer_current\latest.json `
  --dispatch-path reports\dean_os\news_semis_dispatch_current\latest.json `
  --no-save
```

Current expected result: `domain_news_candidate_ready_with_gaps`, 396 accepted
records, 4/5 lanes, source lineage true, trigger-only semantics true, binding
false and trading false. Canonical review files:

- `reports/dean_os/domain_scoped_news_envelope_current/latest.md`
- `reports/dean_os/news_semis_binding_review_current/latest.md`
- `reports/dean_os/news_semis_candidate_binding_plan_current/latest.md`

Focused offline verification:

```powershell
python -m pytest -q `
  tests\test_llm_proposal_boundary.py `
  tests\dean_os\test_domain_scoped_news_envelope.py `
  tests\dean_os\test_context_acquisition_state_machine.py
```

The optional LLM package set is in `requirements-llm.txt`. Installing it does
not enable calls: both `OPENAI_API_KEY` and an explicit `OPENAI_MODEL` are still
required, and the default modular orchestrator does not instantiate the client.

## Official-policy adapter

Verify the saved policy packet, raw PDF, registry and exact domain-news lineage
without running a producer, network call or journal write:

```powershell
python run_agent_domain_scoped_official_policy_envelope.py `
  semiconductor_ai_infrastructure `
  --as-of 2026-06-30T21:00:00+00:00 `
  --source-path reports\dean_os\saved_official_policy_evidence_producer_current\latest.json `
  --news-envelope-path reports\dean_os\domain_scoped_news_envelope_current\latest.json `
  --dispatch-path reports\dean_os\official_policy_semis_dispatch_current\latest.json `
  --no-save
```

Current expected result: `domain_official_policy_candidate_ready_with_gaps`,
zero structural blockers, source/PDF/news lineage true, policy fact true,
market direction false, hypothesis confirmation false, binding false and
trading false. The only current quality gap is pending operator acceptance of
the official-source registry.

Canonical review files:

- `reports/dean_os/domain_scoped_official_policy_envelope_current/latest.md`
- `reports/dean_os/official_policy_semis_binding_review_current/latest.md`
- `reports/dean_os/official_policy_semis_candidate_binding_plan_current/latest.md`

Focused and broad offline verification:

```powershell
python -m pytest -q `
  --basetemp D:\trading_project\.pytest_tmp_phase6_broad `
  tests\dean_os\test_domain_scoped_news_envelope.py `
  tests\dean_os\test_domain_scoped_official_policy_envelope.py `
  tests\dean_os\test_context_acquisition_state_machine.py `
  tests\dean_os\test_domain_binding_task_dispatcher.py `
  tests\dean_os\test_domain_analyst_binding_planner.py `
  tests\dean_os\test_domain_analyst_lifecycle_profile.py
```

Expected: 36 passed. Do not use the default user Temp directory in the managed
workspace; pytest cannot write there.

## DomainContextSet (explicit 5/6 review packet)

Rebuild the current semiconductor packet without accepting bindings, writing
the journal, invoking the analyst, or collecting data:

```powershell
python run_agent_domain_context_set.py `
  semiconductor_ai_infrastructure `
  --analysis-cutoff 2026-07-10T19:50:45.683169+00:00 `
  --output-dir reports\dean_os\domain_context_set_semis_current
```

The CLI defaults are six fixed canonical envelope paths; it does not scan the
filesystem. Current expected result is `domain_context_set_incomplete`, 5/6
verified, with `sector_market` blocked by universe and benchmark mismatch.
`can_invoke_domain_analysis`, learning and trading must remain false.

Focused offline verification:

```powershell
$env:TEMP='D:\trading_project\.tmp'
$env:TMP='D:\trading_project\.tmp'
python -m pytest -q `
  --basetemp D:\trading_project\.tmp\pytest_domain_context `
  tests\dean_os\test_domain_context_set.py `
  tests\dean_os\test_domain_scoped_fundamentals_envelope.py
```

Do not use the old clean-Yahoo command documented by Gemini. It bypassed the
domain bridge and treated a clean snapshot as if it were a repair artifact. Use
the bounded domain command below instead.

## Sector-market 5/6 -> 6/6 path

Current offline audit only (no network):

```powershell
python run_agent_domain_sector_market_coverage_bridge.py `
  semiconductor_ai_infrastructure `
  --analysis-cutoff 2026-07-10T19:50:45.683169+00:00 `
  --snapshot-manifest reports\dean_os\clean_market_snapshot_current\latest.json
```

Expected current result: `domain_sector_market_coverage_blocked`, 4/13. Do not
run repair against this artifact; the domain repair CLI must reject it.

Future bounded network command, only after explicit authorization and with the
actual current UTC cutoff substituted:

```powershell
python run_agent_clean_yahoo_market_snapshot.py `
  --domain-id semiconductor_ai_infrastructure `
  --timeframe 15m `
  --end-date <CURRENT_TIMEZONE_AWARE_UTC>
```

Then execute only if the new bridge reports 13/13 ready:

```powershell
python run_agent_pipeline_control_saved_price_repair.py `
  --coverage-json reports\dean_os\domain_sector_market_coverage_bridge_current\latest.json `
  --domain-id semiconductor_ai_infrastructure `
  --required-model-rows 20 `
  --output-dir reports\dean_os\pipeline_control_saved_price_repair_semis_current
```

The subsequent saved evidence, domain envelope, DomainContextSet rebuild and
orchestrator gate remain offline. Never use `effective_start=None`, generic
coverage, QQQ, or a partial ticker list on the domain path.
