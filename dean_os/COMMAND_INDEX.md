<!-- GENERATED FILE -- DO NOT EDIT BY HAND.
     Regenerate with: python run_agent_command_index.py
     Source of truth: the run_agent_*.py wrappers in the project root. -->

# DEAN-OS Command Index

Every `run_agent_*.py` entrypoint in the project root, with the options each one
actually accepts. Generated from the wrappers themselves, so it cannot drift from
the code. For the reasoning and boundaries behind each workflow see
`dean_os/COMMAND_CHECKLIST.md`; for what a command is *for*, that prose is still
the place to look. This file only answers "does it exist, and what does it take".


**154 commands.**


## `run_agent_analyst_calibration_gate.py`

Build proposal-only analyst calibration guidance from scorecards and outcomes.

Options:

- `--profile-scorecard-json`
- `--profile-runs-dir` — default `reports/dean_os/analyst_profiles`
- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--memory-store` — default `data/dean_os/recommendation_memory.sqlite`
- `--min-profile-runs` — int, default `3`
- `--min-completed-outcomes` — int, default `3`
- `--min-hit-rate` — float, default `0.55`
- `--max-miss-rate` — float, default `0.4`
- `--allow-scorecard-candidate` — flag
- `--output-dir` — default `reports/dean_os/analyst_calibration_gate`
- `--print-json` — flag


## `run_agent_analyst_core_reasoning_snapshot.py`

Build a review-only analyst reasoning snapshot.

Options:

- `--runtime-json` — default `reports/dean_os/semiconductor_analyst_runtime_current/latest.json`
- `--hypothesis-journal-path`
- `--output-dir` — default `reports/dean_os/analyst_core_reasoning_snapshot_current`
- `--no-save` — flag


## `run_agent_analyst_evidence_pack.py`

Build a local-only evidence pack for analyst agents from materials, cached news, and macro data.

Options:

- `--materials` — nargs=*. Local files/directories with research materials.
- `--news-data` — nargs=*. Cached news CSV/parquet/json files.
- `--macro-data` — nargs=*. Cached macro CSV/parquet/json files.
- `--source-routing-json`
- `--tickers` — nargs=*
- `--sectors` — nargs=*
- `--tags` — nargs=*
- `--start-at`
- `--end-at`
- `--as-of`
- `--max-rows-per-table` — int, default `200`
- `--max-documents` — int, default `500`
- `--max-text-chars` — int, default `6000`
- `--no-routed-materials` — flag
- `--output-dir` — default `reports/dean_os/analyst_evidence_pack`
- `--print-json` — flag


## `run_agent_analyst_knowledge_readiness.py`

Run AnalystKnowledgeReadiness (analyst_knowledge_readiness).

Options:

- `--store-dir` — default `data/dean_os/analyst_knowledge`
- `--output-dir` — default `reports/dean_os/analyst_knowledge_readiness_current`
- `--as-of`. ISO-8601 timestamp; defaults to now in UTC.
- `--intended-use` — default `evidence`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_analyst_learning_bridge.py`

Promote reviewed analyst notes into learning records; dry-run by default.

Options:

- `--profile-run-json`. AnalystProfileOrchestrator JSON, usually reports/dean_os/analyst_profiles/latest.json.
- `--agent-lab-report-json`. Direct Agent Lab report JSON.
- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--review-actions-store` — default `data/dean_os/review_actions.sqlite`
- `--operations-store` — default `data/dean_os/operation_queue.sqlite`
- `--allow-unreviewed` — flag. Diagnostics only; do not use for durable promotion policy.
- `--allow-weak-notes` — flag
- `--allow-duplicates` — flag
- `--default-horizon-days` — int, default `365`
- `--apply` — flag. Write promotable learning records.
- `--output-dir` — default `reports/dean_os/analyst_learning_bridge`
- `--print-json` — flag


## `run_agent_analyst_loop_daily_check.py`

Build a read-only daily operator check for the analyst learning loop.

Options:

- `--evidence-pack-json`
- `--analyst-profiles-json`
- `--profile-scorecard-json`
- `--learning-bridge-json`
- `--review-approved-learning-json`
- `--outcome-evaluation-json`
- `--calibration-gate-json`
- `--calibration-proposals-json`
- `--calibration-review-json`
- `--manual-backlog-json`
- `--market-data-path`
- `--latest-processed-prices` — default `1d`
- `--tickers` — nargs=*
- `--as-of`
- `--max-age-hours` — float, default `72.0`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--event-log-path` — default `logs/dean_os/events.jsonl`
- `--event-limit` — int, default `10`
- `--output-dir` — default `reports/dean_os/analyst_loop_daily_check`
- `--print-json` — flag


## `run_agent_analyst_outcome_loop.py`

Evaluate reviewed analyst learning records against local prices; dry-run by default.

Options:

- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--memory-store` — default `data/dean_os/recommendation_memory.sqlite`
- `--market-data-path`
- `--latest-processed-prices`
- `--tickers` — nargs=*
- `--as-of`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--apply` — flag
- `--allow-early` — flag
- `--historical-diagnostic` — flag
- `--allow-diagnostic-apply` — flag
- `--neutral-band` — float, default `0.01`
- `--limit` — int
- `--profile`
- `--agent-names` — nargs=*
- `--include-non-analyst-records` — flag
- `--output-dir` — default `reports/dean_os/analyst_outcome_evaluation`
- `--print-json` — flag


## `run_agent_analyst_profiles.py`

Run the managed analyst profile flow from an evidence pack.

Positional:

- `evidence_pack_json`. Evidence pack JSON from run_agent_analyst_evidence_pack.py.

Options:

- `--profiles` — nargs=*
- `--tickers` — nargs=*
- `--sectors` — nargs=*
- `--tags` — nargs=*
- `--allow-candidate-profiles` — flag
- `--create-learning-records` — flag
- `--include-operation-proposals` — flag
- `--no-review-snapshot` — flag
- `--output-dir` — default `reports/dean_os/analyst_profiles`
- `--corpus`
- `--learning-store`
- `--operations-store`
- `--review-actions-store`
- `--memory-store`
- `--log-path`
- `--print-json` — flag


## `run_agent_analyst_review_inbox.py`

Build a read-only analyst report review inbox.

Options:

- `--learning-bridge-json` — default `reports/dean_os/analyst_learning_bridge/latest.json`
- `--profile-run-json` — default `reports/dean_os/analyst_profiles/latest.json`
- `--review-actions-store` — default `data/dean_os/review_actions.sqlite`
- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--operations-store` — default `data/dean_os/operation_queue.sqlite`
- `--output-dir` — default `reports/dean_os/analyst_review_inbox`
- `--print-json` — flag


## `run_agent_analyst_scorecard.py`

Build activation scorecards for analyst profiles from saved profile runs.

Options:

- `--profile-runs-dir` — default `reports/dean_os/analyst_profiles`
- `--min-completed-runs` — int, default `3`
- `--min-avg-confidence` — float, default `0.55`
- `--min-avg-citations` — float, default `1.0`
- `--output-dir` — default `reports/dean_os/analyst_profile_scorecard`
- `--print-json` — flag


## `run_agent_build_focus_review_packet.py`

Build the focus review packet from alignment/template/case-registry/pipeline-control inputs.

Options:

- `--alignment-review-json`
- `--template-standardization-json`
- `--case-registry-json`
- `--pipeline-control-instance-json`
- `--output-dir` — default `reports/dean_os/build_focus_review_packet`
- `--no-save` — flag


## `run_agent_calibration_proposals.py`

Create proposal-only calibration review items from an analyst calibration gate report.

Positional:

- `calibration_gate_json`

Options:

- `--operations-store` — default `data/dean_os/operation_queue.sqlite`
- `--log-path` — default `logs/dean_os/events.jsonl`
- `--include-caution` — flag
- `--enqueue` — flag. Write proposed review items to OperationQueue.
- `--output-dir` — default `reports/dean_os/calibration_proposals`
- `--print-json` — flag


## `run_agent_calibration_review_lifecycle.py`

Review calibration operation proposals without writing config or changing weights.

Options:

- `--operations-store` — default `data/dean_os/operation_queue.sqlite`
- `--log-path` — default `logs/dean_os/events.jsonl`
- `--proposal-ids` — nargs=*
- `--dry-run-proposals` — flag
- `--approve` — nargs=*. Explicit proposal IDs to mark approved in OperationQueue.
- `--reject` — nargs=*. Explicit proposal IDs to mark rejected in OperationQueue.
- `--include-non-calibration` — flag
- `--output-dir` — default `reports/dean_os/calibration_review_lifecycle`
- `--print-json` — flag


## `run_agent_chief_review.py`

Synthesize saved DEAN-OS state into one supervised-autonomy review.

Options:

- `--review-snapshot`
- `--model-performance-json`
- `--regime-context-json`
- `--tuning-json`
- `--context-performance-json`
- `--autonomy-mode` — default `paper_supervised`
- `--output`
- `--output-dir` — default `reports/dean_os/chief_review`
- `--print-json` — flag


## `run_agent_chief_review_index.py`

Run ChiefReviewIndexBuilder (chief_review_index).

Options:

- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/chief_review_index_current`
- `--print-json` — flag


## `run_agent_clean_yahoo_market_snapshot.py`

Run one bounded Yahoo snapshot outside the legacy database. Network access is performed; no learning or trading action runs.

Options:

- `--domain-id`
- `--ticker` — repeatable
- `--timeframe` — repeatable
- `--end-date` — **required**, _aware
- `--max-download-attempts` — int, default `2`
- `--config-path` — default `src/config/collectors.yaml`
- `--artifact-dir` — default `data/dean_os/clean_market_snapshots`
- `--output-dir` — default `reports/dean_os/clean_market_snapshot_current`
- `--no-save` — flag


## `run_agent_command_index.py`

Regenerate dean_os/COMMAND_INDEX.md from the run_agent_*.py wrappers on disk.

Options:

- `--check` — flag. Report drift without writing the index. Exits 1 if the index is stale.
- `--print-markdown` — flag


## `run_agent_context_performance.py`

Summarize agent performance by theme/regime context.

Options:

- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--memory-store` — default `data/dean_os/recommendation_memory.sqlite`
- `--agent-name`
- `--context-tag`
- `--min-completed` — int, default `1`
- `--limit` — int, default `10`
- `--output`
- `--output-dir` — default `reports/dean_os/context_performance`
- `--print-json` — flag


## `run_agent_current_architecture_map.py`

Build the current architecture map report.

Options:

- `--output-dir` — default `reports/dean_os/current_architecture_map`
- `--no-save` — flag


## `run_agent_current_cycle_journal.py`

Import one verified analysis cycle into SystemJournal.

Options:

- `--cycle-json` — default `reports/dean_os/full_system_review_cycle_current/latest.json`
- `--world-model-json` — default `reports/dean_os/world_model_event_learning_cycle_current/latest.json`
- `--review-gate-json` — default `reports/dean_os/world_model_replay_review_gate_cycle_current/latest.json`
- `--closure-json` — default `reports/dean_os/full_system_cycle_closure_current/latest.json`
- `--learning-review-json` — default `reports/dean_os/hypothesis_learning_review_current/latest.json`
- `--reasoning-snapshot-json`
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--output-dir` — default `reports/dean_os/current_cycle_journal_current`
- `--apply` — flag
- `--exclude-full-news` — flag
- `--no-save` — flag


## `run_agent_current_system_alignment_review.py`

Build the current system alignment review report.

Options:

- `--evidence-pack-json`
- `--source-gate-json`
- `--agent-lab-path`
- `--dropzone-inventory-json`
- `--fundamental-gate-json`
- `--architecture-map-json`
- `--domain-analyst-intake-json`
- `--domain-analyst-instance-contract-json`
- `--domain-analyst-thesis-review-json`
- `--domain-analyst-template-standardization-json`
- `--domain-analyst-case-registry-json`
- `--pipeline-metric-input-readiness-json`
- `--pipeline-control-instance-contract-json`
- `--pipeline-control-caution-review-json`
- `--output-dir` — default `reports/dean_os/current_system_alignment_review`
- `--no-save` — flag


## `run_agent_current_system_manifest.py`

Run CurrentSystemManifestBuilder (current_system_manifest).

Options:

- `--output-dir` — default `reports/dean_os/current_system_manifest_current`
- `--topology-path` — default `dean_os/config/system_topology.yaml`
- `--authorization-ledger-path` — default `data/dean_os/accumulation_authorization_ledger.jsonl`
- `--as-of`. ISO-8601 timestamp; defaults to now in UTC.
- `--domain-id` — default `semiconductor_ai_infrastructure`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_dean_os_review_automation.py`

Run the safe DEAN-OS review chain without starting the trading pipeline.

Options:

- `--candidate-paths` — nargs=+
- `--training-candidate-json`
- `--evaluation-candidate-json`
- `--feature-stability-candidate-json`
- `--replay-batch-json`
- `--data-quality-json`
- `--constraints-path`
- `--domain-instance-contract-json`
- `--no-real-metric-run` — flag
- `--output-dir` — default `reports/dean_os/review_only_automation_run_current`
- `--no-save` — flag


## `run_agent_diary_bridge.py`

Inspect whether DEAN paper outcomes can safely bridge into pipeline diary review.

Options:

- `--experience-diary` — default `logs/experience_diary.csv`
- `--paper-store` — default `data/dean_os/paper_trades.sqlite`
- `--output`
- `--output-dir` — default `reports/dean_os/diary_bridge`
- `--print-json` — flag


## `run_agent_domain_analyst_case_registry_packet.py`

Build the domain analyst case registry packet.

Options:

- `--domain-thesis-review-json`
- `--domain-template-standardization-json`
- `--domain-forecast-review-json`
- `--outcome-evaluation-json`
- `--output-dir` — default `reports/dean_os/domain_analyst_case_registry_packet`
- `--no-save` — flag


## `run_agent_domain_analyst_event_interpretation_packet.py`

Build the domain analyst event interpretation packet.

Options:

- `--evidence-pack-json`
- `--pipeline-context-json`
- `--domain-id`
- `--max-events` — int
- `--output-dir` — default `reports/dean_os/domain_analyst_event_interpretation_packet_current`
- `--no-save` — flag


## `run_agent_domain_analyst_feedback_loop_packet.py`

Build the domain analyst feedback loop packet.

Options:

- `--case-registry-json`
- `--forecast-review-json`
- `--profile-policy-json`
- `--template-decision-json`
- `--manual-feedback-json`
- `--output-dir` — default `reports/dean_os/domain_analyst_feedback_loop_packet_current`
- `--no-save` — flag


## `run_agent_domain_analyst_forecast_review_packet.py`

Build the domain analyst forecast review packet.

Options:

- `--domain-thesis-review-json`
- `--vertical-slice-json`
- `--regime-scenario-json`
- `--output-dir` — default `reports/dean_os/domain_analyst_forecast_review_packet`
- `--no-save` — flag


## `run_agent_domain_analyst_instance_contract.py`

Build the domain analyst instance contract.

Options:

- `--evidence-pack-json`
- `--source-gate-json`
- `--domain-intake-json`
- `--architecture-map-json`
- `--output-dir` — default `reports/dean_os/domain_analyst_instance_contract`
- `--no-save` — flag


## `run_agent_domain_analyst_intake_packet.py`

Build the domain analyst intake packet.

Options:

- `--evidence-pack-json`
- `--source-gate-json`
- `--domain-id`
- `--tickers` — nargs=+
- `--sectors` — nargs=+
- `--horizon-days` — int
- `--as-of`
- `--max-items` — int
- `--output-dir` — default `reports/dean_os/domain_analyst_intake_packet`
- `--no-save` — flag


## `run_agent_domain_analyst_lifecycle_profile.py`

Run DomainAnalystLifecycleProfileReport (domain_analyst_lifecycle_profile).

Options:

- `--source-domain-id` — default `semiconductor_ai_infrastructure`
- `--clone-domain-id` — default `energy`
- `--template-path` — default `dean_os/config/domain_analyst_lifecycle.template.json`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/domain_analyst_lifecycle_profile_current`
- `--print-json` — flag


## `run_agent_domain_analyst_portability_review.py`

Build the domain analyst portability review.

Options:

- `--vertical-slice-json`
- `--architecture-map-json`
- `--output-dir` — default `reports/dean_os/domain_analyst_portability_review_current`
- `--no-save` — flag


## `run_agent_domain_analyst_profile_policy_packet.py`

Build the domain analyst profile policy packet.

Options:

- `--output-dir` — default `reports/dean_os/domain_analyst_profile_policy_packet_current`
- `--no-save` — flag


## `run_agent_domain_analyst_regime_scenario_packet.py`

Build the domain analyst regime scenario packet.

Options:

- `--event-interpretation-json`
- `--domain-id`
- `--max-events` — int
- `--horizons` — nargs=+
- `--output-dir` — default `reports/dean_os/domain_analyst_regime_scenario_packet_current`
- `--no-save` — flag


## `run_agent_domain_analyst_template_decision_packet.py`

Build the domain analyst template decision packet.

Options:

- `--vertical-slice-json`
- `--template-standardization-json`
- `--forecast-review-json`
- `--case-registry-json`
- `--portability-review-json`
- `--architecture-map-json`
- `--decision`
- `--reviewer`
- `--rationale`
- `--required-followups` — nargs=+
- `--output-dir` — default `reports/dean_os/domain_analyst_template_decision_packet_current`
- `--no-save` — flag


## `run_agent_domain_analyst_template_standardization_packet.py`

Build the domain analyst template standardization packet.

Options:

- `--domain-instance-contract-json`
- `--domain-thesis-review-json`
- `--regime-scenario-json`
- `--architecture-map-json`
- `--output-dir` — default `reports/dean_os/domain_analyst_template_standardization_packet`
- `--no-save` — flag


## `run_agent_domain_analyst_thesis_review_packet.py`

Build the domain analyst thesis review packet.

Options:

- `--domain-intake-json`
- `--domain-instance-contract-json`
- `--regime-scenario-json`
- `--architecture-map-json`
- `--reasoning-snapshot-json`
- `--output-dir` — default `reports/dean_os/domain_analyst_thesis_review_packet`
- `--no-save` — flag


## `run_agent_domain_analyst_vertical_slice.py`

Run the domain analyst vertical slice.

Options:

- `--domain-id`
- `--evidence-pack-json`
- `--source-gate-json`
- `--pipeline-context-json`
- `--materials` — nargs=+
- `--news-data` — nargs=+
- `--macro-data` — nargs=+
- `--source-routing-path`
- `--tickers` — nargs=+
- `--sectors` — nargs=+
- `--tags` — nargs=+
- `--sector-keywords` — nargs=+
- `--start-at`
- `--end-at`
- `--as-of`
- `--horizon-days` — int
- `--output-dir` — default `reports/dean_os/domain_analyst_vertical_slice_current`
- `--no-save` — flag


## `run_agent_domain_context_set.py`

Recursively verify six explicit domain context envelopes and emit a review-only complete or partial DomainContextSet.

Positional:

- `domain_id`

Options:

- `--analysis-cutoff` — **required**
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--output-dir` — default `reports/dean_os/domain_context_set_current`
- `--apply-journal` — flag
- `--no-save` — flag


## `run_agent_domain_macro_binding_quality_review.py`

Run DomainMacroBindingQualityReview (domain_macro_binding_quality_review).

Options:

- `--domain-id` — default `energy`
- `--candidate-path` — default `reports/dean_os/domain_scoped_macro_envelope_current/latest.json`
- `--binding-plan-path` — default `reports/dean_os/domain_analyst_binding_plan_current/latest.json`
- `--review-as-of`
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--apply-journal` — flag
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/domain_macro_binding_quality_review_current`
- `--print-json` — flag


## `run_agent_domain_macro_collection_execution_gate.py`

Run DomainMacroCollectionExecutionGate (domain_macro_collection_execution_gate).

Options:

- `--domain-id` — default `energy`
- `--request-path` — default `reports/dean_os/domain_macro_collection_request_current/latest.json`
- `--registry-path` — default `dean_os/config/macro_series_registry.yaml`
- `--evaluated-at`
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--apply-journal` — flag
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/domain_macro_collection_execution_gate_current`
- `--print-json` — flag


## `run_agent_domain_macro_collection_request.py`

Run DomainMacroCollectionRequest (domain_macro_collection_request).

Options:

- `--domain-id` — default `energy`
- `--quality-review-path` — default `reports/dean_os/domain_macro_binding_quality_review_current/latest.json`
- `--candidate-path` — default `reports/dean_os/domain_scoped_macro_envelope_current/latest.json`
- `--registry-path` — default `dean_os/config/macro_series_registry.yaml`
- `--request-as-of`
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--apply-journal` — flag
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/domain_macro_collection_request_current`
- `--print-json` — flag


## `run_agent_domain_orchestrator.py`

Run the review-only DEAN-OS domain orchestrator.

Positional:

- `domain_id`

Options:

- `--as-of`
- `--ticker` — repeatable
- `--include-profile-agents` — flag
- `--context-set`
- `--legacy-unbound-diagnostic` — flag. Run the pre-existing unbound diagnostic path explicitly.
- `--no-save` — flag


## `run_agent_domain_scoped_fundamentals_envelope.py`

Verify one saved terminal SEC fundamentals artifact and its recursive lineage as a review-only domain binding candidate.

Positional:

- `domain_id`

Options:

- `--as-of` — **required**
- `--source-path` — default `reports/dean_os/saved_sec_derived_ratio_producer_current/latest.json`
- `--dispatch-path` — default `reports/dean_os/domain_binding_task_dispatch_current/latest.json`
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--output-dir` — default `reports/dean_os/domain_scoped_fundamentals_envelope_current`
- `--apply-journal` — flag
- `--no-save` — flag


## `run_agent_domain_scoped_macro_envelope.py`

Run DomainScopedMacroEnvelopeCeremony (domain_scoped_macro_envelope).

Options:

- `--domain-id` — default `energy`
- `--source-path`
- `--as-of`
- `--registry-path` — default `dean_os/config/macro_series_registry.yaml`
- `--dispatch-path` — default `reports/dean_os/domain_binding_task_dispatch_current/latest.json`
- `--execution-gate-path`
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--apply-journal` — flag
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/domain_scoped_macro_envelope_current`
- `--print-json` — flag


## `run_agent_domain_scoped_news_envelope.py`

Verify one saved news artifact as a review-only, trigger-evidence domain binding candidate.

Positional:

- `domain_id`

Options:

- `--as-of` — **required**
- `--source-path` — default `reports/dean_os/saved_semiconductor_news_evidence_producer_current/latest.json`
- `--dispatch-path` — default `reports/dean_os/domain_binding_task_dispatch_current/latest.json`
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--output-dir` — default `reports/dean_os/domain_scoped_news_envelope_current`
- `--apply-journal` — flag
- `--no-save` — flag


## `run_agent_domain_scoped_official_policy_envelope.py`

Verify saved official-policy, raw-document, registry, and domain-news lineage as a review-only domain binding candidate.

Positional:

- `domain_id`

Options:

- `--as-of` — **required**
- `--source-path` — default `reports/dean_os/saved_official_policy_evidence_producer_current/latest.json`
- `--news-envelope-path` — default `reports/dean_os/domain_scoped_news_envelope_current/latest.json`
- `--dispatch-path` — default `reports/dean_os/domain_binding_task_dispatch_current/latest.json`
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--output-dir` — default `reports/dean_os/domain_scoped_official_policy_envelope_current`
- `--apply-journal` — flag
- `--no-save` — flag


## `run_agent_domain_scoped_sector_market_envelope.py`

Verify one saved sector-market artifact as a review-only domain binding candidate.

Positional:

- `domain_id`

Options:

- `--as-of` — **required**
- `--source-path` — default `reports/dean_os/saved_sector_market_evidence_producer_current/latest.json`
- `--dispatch-path` — default `reports/dean_os/domain_binding_task_dispatch_current/latest.json`
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--output-dir` — default `reports/dean_os/domain_scoped_sector_market_envelope_current`
- `--apply-journal` — flag
- `--no-save` — flag


## `run_agent_domain_sector_market_coverage_bridge.py`

Verify one clean market snapshot against one domain's exact universe and emit a repair-compatible coverage candidate.

Positional:

- `domain_id`

Options:

- `--analysis-cutoff` — **required**
- `--snapshot-manifest` — **required**
- `--min-rows` — int, default `180`
- `--max-rows` — int, default `600`
- `--max-abs-return` — float, default `0.25`
- `--min-cadence-ratio` — float, default `0.75`
- `--output-dir` — default `reports/dean_os/domain_sector_market_coverage_bridge_current`
- `--no-save` — flag


## `run_agent_evidence_gap_plan.py`

Build a read-only plan for resolving analyst evidence gaps.

Options:

- `--review-action-json` — default `reports/dean_os/review_action_apply_ceremony/latest.json`
- `--decision-packet-json` — default `reports/dean_os/review_decision_packet/latest.json`
- `--evidence-pack-json`
- `--source-routing-json`
- `--min-documents-per-missing-ticker` — int, default `2`
- `--min-date-span-days` — int, default `30`
- `--suggested-max-rows-per-table` — int, default `200`
- `--output-dir` — default `reports/dean_os/evidence_gap_resolution_plan`
- `--print-json` — flag


## `run_agent_evidence_timestamp_audit.py`

Audit cached evidence source timestamps before historical research replay.

Options:

- `--source-data` — nargs=*. Generic CSV/parquet/json source tables.
- `--news-data` — nargs=*. Cached news CSV/parquet/json tables.
- `--macro-data` — nargs=*. Cached macro CSV/parquet/json tables.
- `--evidence-pack-json`. Optional AnalystEvidencePack JSON to compare.
- `--as-of`
- `--start-at`
- `--min-parse-rate` — float, default `0.75`
- `--collapse-share-threshold` — float, default `0.95`
- `--collapse-min-rows` — int, default `10`
- `--output-dir` — default `reports/dean_os/evidence_timestamp_audit`
- `--print-json` — flag


## `run_agent_filing_order_evidence.py`

Run FilingOrderEvidenceBuilder (filing_order_evidence).

Options:

- `--output-dir` — default `reports/dean_os/filing_order_evidence_current`
- `--companyfacts-paths` — **required**, repeatable. Repeatable path to a saved artifact.
- `--as-of`. ISO-8601 timestamp; defaults to now in UTC.
- `--max-age-days` — int, default `730`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_full_system_cycle_closure.py`

Run FullSystemCycleClosureBuilder (full_system_cycle_closure).

Options:

- `--cycle-path`
- `--world-model-path`
- `--prior-checkpoint-monitor-path`
- `--replay-review-gate-path`
- `--replay-registration-path`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/full_system_cycle_closure_current`
- `--print-json` — flag


## `run_agent_fundamental_input_readiness_gate.py`

Check caller-supplied fundamental inputs before value-agent review.

Options:

- `--fundamentals-json`
- `--output-dir` — default `reports/dean_os/fundamental_input_readiness_gate_current`
- `--as-of`. Timezone-aware analysis cutoff. Without it the gate may support manual inspection but cannot authorize value-screen input.
- `--print-json` — flag


## `run_agent_historical_evidence_backfill.py`

Build a read-only plan for backfilling weak historical research replay evidence.

Options:

- `--readiness-json` — default `reports/dean_os/replay_calibration_readiness_gate_after_step14_research/latest.json`
- `--research-batch-json` — default `reports/dean_os/historical_research_replay_batch_repaired_expanded_step14/latest.json`
- `--news-data` — nargs=*
- `--macro-data` — nargs=*
- `--materials` — nargs=*
- `--tickers` — nargs=*
- `--lookback-days` — int, default `180`
- `--min-documents-per-run` — int, default `5`
- `--output-dir` — default `reports/dean_os/historical_evidence_backfill_plan`
- `--print-json` — flag


## `run_agent_historical_replay.py`

Run a safe old-data DEAN-OS replay without paper trades, live broker access, or heavy pipeline execution.

Positional:

- `price_data_path`. Historical price CSV/parquet file.

Options:

- `--tickers` — **required**, nargs=+. Tickers visible to the replay analyst.
- `--as-of` — **required**. Cutoff timestamp; thesis sees only data at or before this time.
- `--lookback-days` — int, default `180`
- `--horizon-days` — int, default `60`
- `--news-data`
- `--macro-data`
- `--benchmark-ticker` — default `SPY`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--neutral-band` — float, default `0.01`
- `--max-news-items` — int, default `80`
- `--normalize-daily-bars` — flag
- `--output-dir` — default `reports/dean_os/historical_replay`
- `--print-json` — flag. Print full JSON payload.


## `run_agent_historical_replay_batch.py`

Run multiple safe DEAN-OS historical replay slices without learning writes or pipeline execution.

Positional:

- `price_data_path`. Normalized or raw historical price CSV/parquet file.

Options:

- `--tickers` — **required**, nargs=+
- `--as-of` — nargs=*. Explicit as_of dates.
- `--start-as-of`
- `--end-as-of`
- `--step-days` — int, default `14`
- `--lookback-days` — int, default `180`
- `--horizon-days` — int, nargs=+, default `[60]`
- `--news-data`
- `--macro-data`
- `--benchmark-ticker` — default `SPY`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--neutral-band` — float, default `0.01`
- `--max-runs` — int, default `50`. Maximum generated as_of dates.
- `--stop-on-quality-warning` — flag
- `--output-dir` — default `reports/dean_os/historical_replay_batch`
- `--print-json` — flag


## `run_agent_historical_replay_outcome_review.py`

Run HistoricalReplayOutcomeReview (historical_replay_outcome_review).

Options:

- `--review-gate-json`
- `--registration-json`
- `--price-paths` — repeatable
- `--pipeline-paths` — repeatable
- `--task-ids` — repeatable
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/historical_replay_outcome_review_current`
- `--print-json` — flag


## `run_agent_historical_research_replay.py`

Run a safe historical research replay: evidence pack + Agent Lab + price outcome, without learning writes, broker access, or pipeline execution.

Positional:

- `price_data_path`. Historical price CSV/parquet file.

Options:

- `--tickers` — **required**, nargs=+. Tickers visible to the replay exam.
- `--as-of` — **required**. Cutoff timestamp; agents see only data at or before this time.
- `--lookback-days` — int, default `180`
- `--horizon-days` — int, default `60`
- `--news-data` — nargs=*
- `--macro-data` — nargs=*
- `--materials` — nargs=*
- `--tags` — nargs=*
- `--benchmark-ticker` — default `SPY`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--neutral-band` — float, default `0.01`
- `--max-rows-per-table` — int, default `300`
- `--max-documents` — int, default `600`
- `--normalize-daily-bars` — flag
- `--output-dir` — default `reports/dean_os/historical_research_replay`
- `--print-json` — flag


## `run_agent_historical_research_replay_batch.py`

Run multiple historical research replay exams without learning writes or pipeline execution.

Positional:

- `price_data_path`. Historical price CSV/parquet file.

Options:

- `--tickers` — **required**, nargs=+
- `--as-of` — nargs=*. Explicit as_of dates.
- `--start-as-of`
- `--end-as-of`
- `--step-days` — int, default `30`
- `--lookback-days` — int, default `180`
- `--horizon-days` — int, nargs=+, default `[60]`
- `--news-data` — nargs=*
- `--macro-data` — nargs=*
- `--materials` — nargs=*
- `--tags` — nargs=*
- `--benchmark-ticker` — default `SPY`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--neutral-band` — float, default `0.01`
- `--max-runs` — int, default `20`
- `--normalize-daily-bars` — flag
- `--output-dir` — default `reports/dean_os/historical_research_replay_batch`
- `--print-json` — flag


## `run_agent_hypothesis_evidence_gap_review.py`

Run HypothesisEvidenceGapReview (hypothesis_evidence_gap_review).

Options:

- `--output-dir` — default `reports/dean_os/hypothesis_evidence_gap_review_current`
- `--analyst-review-path` — **required**
- `--fundamental-artifact-path` — **required**
- `--ratio-artifact-path`
- `--primary-snapshot-path`
- `--operational-metrics-path`
- `--filing-order-evidence-path`
- `--as-of`. ISO-8601 timestamp; defaults to now in UTC.
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_hypothesis_gap_replay_packet.py`

Run HypothesisGapReplayPacketBridge (hypothesis_gap_replay_packet).

Options:

- `--gap-review-path`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/hypothesis_gap_replay_packet_current`
- `--print-json` — flag


## `run_agent_hypothesis_learning_review.py`

Run HypothesisLearningReview (hypothesis_learning_review).

Options:

- `--output-dir` — default `reports/dean_os/hypothesis_learning_review_current`
- `--packet-json` — **required**. Path to a saved JSON artifact.
- `--review-gate-json` — **required**. Path to a saved JSON artifact.
- `--outcome-json`. Path to a saved JSON artifact.
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_hypothesis_measurement_policy_preparer.py`

Run HypothesisMeasurementPolicyPreparer (hypothesis_measurement_policy_preparer).

Options:

- `--resolution-specs-json`
- `--price-paths` — repeatable
- `--pipeline-paths` — repeatable
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/hypothesis_measurement_policy_preparer_current`
- `--print-json` — flag


## `run_agent_industry_operational_metrics.py`

Run IndustryOperationalMetricsBuilder (industry_operational_metrics).

Options:

- `--output-dir` — default `reports/dean_os/industry_operational_metrics_current`
- `--records` — **required**, repeatable. Repeatable path to a saved artifact.
- `--as-of`. ISO-8601 timestamp; defaults to now in UTC.
- `--domain-id` — **required**
- `--input-reference` — default `in_memory_operator_packet`
- `--input-sha256`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_industry_operational_source_coverage.py`

Run IndustryOperationalSourceCoverageBuilder (industry_operational_source_coverage).

Options:

- `--duckdb-path`
- `--research-sqlite-path`
- `--knowledge-pack-path`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/industry_operational_source_coverage_current`
- `--print-json` — flag


## `run_agent_lab.py`

Run isolated DEAN-OS Agent Lab without starting the trading pipeline.

Positional:

- `materials_path` — nargs=?

Options:

- `--sample` — flag. Use deterministic sample documents.
- `--evidence-pack-json`. Analyst evidence pack JSON from run_agent_analyst_evidence_pack.py.
- `--corpus` — default `data/dean_os/research_corpus.sqlite`
- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--operations-store`
- `--memory-store` — default `data/dean_os/recommendation_memory.sqlite`
- `--log-path` — default `logs/dean_os/events.jsonl`
- `--output-dir` — default `reports/dean_os/agent_lab`
- `--tickers` — nargs=*
- `--sectors` — nargs=*
- `--tags` — nargs=*
- `--regime-tags` — nargs=*
- `--regime-context-json`
- `--source-type`
- `--chunk-size` — int, default `1200`
- `--no-financial-nlp` — flag
- `--no-synthesis` — flag
- `--no-learning-records` — flag
- `--no-operation-proposals` — flag
- `--print-json` — flag


## `run_agent_learning.py`

Inspect or update DEAN-OS learning records.

Positional:

- `agent_name`
- `record_id`

Options:

- `--store` — default `data/dean_os/agent_learning.sqlite`
- `--print-json` — flag
- `--agent-name`
- `--realized-return` — **required**, float
- `--outcome-at`
- `--neutral-band` — float, default `0.01`


## `run_agent_learning_apply_ceremony.py`

Apply pending analyst learning records from a validated bridge dry-run.

Options:

- `--bridge-dry-run-json` — default `reports/dean_os/analyst_learning_bridge/latest.json`
- `--learning-store`
- `--review-actions-store`
- `--operations-store`
- `--apply-learning` — flag
- `--output-dir` — default `reports/dean_os/analyst_learning_apply_ceremony`
- `--print-json` — flag


## `run_agent_learning_loop_runbook.py`

Build a read-only operator runbook for the safe analyst learning loop.

Options:

- `--evidence-pack-json`
- `--analyst-profiles-json`
- `--profile-scorecard-json`
- `--learning-bridge-json`
- `--review-approved-learning-json`
- `--outcome-evaluation-json`
- `--calibration-gate-json`
- `--calibration-proposals-json`
- `--calibration-review-json`
- `--manual-backlog-json`
- `--output-dir` — default `reports/dean_os/agent_learning_loop_runbook`
- `--print-json` — flag


## `run_agent_logs.py`

Inspect DEAN-OS structured event logs.

Options:

- `--log-path` — default `logs/dean_os/events.jsonl`
- `--print-json` — flag
- `--limit` — int, default `10`
- `--event-type`


## `run_agent_manual_implementation_backlog.py`

Report approved calibration proposals waiting for manual implementation.

Options:

- `--operations-store` — default `data/dean_os/operation_queue.sqlite`
- `--include-proposed` — flag
- `--include-rejected` — flag
- `--include-non-calibration` — flag
- `--output-dir` — default `reports/dean_os/manual_implementation_backlog`
- `--print-json` — flag


## `run_agent_market_data_refresh_runbook.py`

Build a read-only market-data refresh runbook for outcome blockers.

Options:

- `--coverage-plan-json` — default `reports/dean_os/outcome_price_coverage_plan/latest.json`
- `--collector-inventory-json`. Path to a saved collector-inventory JSON, or 'live' (default) to scan src/config/collectors.yaml now. Saved snapshots are reported as 'snapshot_*' status because nothing regenerates them any more.
- `--price-glob` — repeatable
- `--max-price-artifacts` — int, default `25`
- `--refreshed-price-placeholder` — default `PATH_TO_REFRESHED_PRICE_FILE`
- `--output-dir` — default `reports/dean_os/market_data_refresh_runbook`
- `--print-json` — flag


## `run_agent_market_freshness.py`

Check local market data freshness without running the trading pipeline.

Options:

- `--market-data-path`
- `--latest-processed-prices` — default `1d`
- `--tickers` — nargs=*
- `--as-of`
- `--max-age-hours` — float, default `72.0`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--include-operation-proposal` — flag
- `--output`
- `--output-dir` — default `reports/dean_os/market_freshness`
- `--print-json` — flag


## `run_agent_memory.py`

Manage DEAN-OS recommendation memory cases.

Positional:

- `memory_id`

Options:

- `--store` — default `data/dean_os/recommendation_memory.sqlite`
- `--print-json` — flag
- `--agent-name`
- `--context-tag`
- `--outcome-label`
- `--source-type` — default `manual_case`
- `--source-id` — **required**
- `--agent-name` — **required**
- `--topic` — **required**
- `--thesis` — **required**
- `--expected-direction` — **required**
- `--context-tags` — nargs=*
- `--tickers` — nargs=*
- `--sectors` — nargs=*
- `--outcome-label` — default `pending`
- `--realized-return` — float
- `--lesson` — default `""`
- `--confidence-before` — float
- `--confidence-after` — float
- `--outcome-at`
- `--outcome-label` — **required**
- `--realized-return` — float
- `--lesson`
- `--confidence-after` — float
- `--outcome-at`


## `run_agent_model_performance.py`

Inspect model/backtest metrics without training, tuning, or pipeline execution.

Positional:

- `performance_path` — nargs=?

Options:

- `--min-validation-score` — float, default `0.55`
- `--min-sharpe` — float, default `0.0`
- `--max-drawdown` — float, default `0.25`
- `--min-sample-count` — int, default `50`
- `--max-age-hours` — float
- `--include-operation-proposal` — flag
- `--output`
- `--output-dir` — default `reports/dean_os/model_performance`
- `--print-json` — flag


## `run_agent_ops.py`

Review DEAN-OS operation proposals without executing the pipeline.

Positional:

- `report_path`
- `proposal_id`

Options:

- `--store` — default `data/dean_os/operation_queue.sqlite`
- `--log-path` — default `logs/dean_os/events.jsonl`
- `--print-json` — flag
- `--status`
- `--action-type`


## `run_agent_outcome_evaluation.py`

Evaluate pending learning records against local prices; dry-run by default.

Options:

- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--market-data-path`
- `--latest-processed-prices`
- `--tickers` — nargs=*
- `--as-of`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--allow-early` — flag
- `--apply` — flag
- `--neutral-band` — float, default `0.01`
- `--limit` — int
- `--output`
- `--output-dir` — default `reports/dean_os/outcome_evaluation`
- `--print-json` — flag


## `run_agent_outcome_price_coverage.py`

Build a read-only price coverage plan for analyst outcome evaluation.

Options:

- `--readiness-json` — default `reports/dean_os/outcome_readiness_gate/latest.json`
- `--market-data-path`
- `--latest-processed-prices`
- `--tickers` — nargs=*
- `--close-col`
- `--datetime-col`
- `--output-dir` — default `reports/dean_os/outcome_price_coverage_plan`
- `--print-json` — flag


## `run_agent_outcome_readiness.py`

Check whether pending analyst learning records are ready for outcome evaluation.

Options:

- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--memory-store` — default `data/dean_os/recommendation_memory.sqlite`
- `--market-data-path`
- `--latest-processed-prices`
- `--tickers` — nargs=*
- `--as-of`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--neutral-band` — float, default `0.01`
- `--limit` — int
- `--profile`
- `--agent-names` — nargs=*
- `--include-non-analyst-records` — flag
- `--historical-diagnostic` — flag
- `--output-dir` — default `reports/dean_os/outcome_readiness_gate`
- `--print-json` — flag


## `run_agent_paper_autonomy.py`

Run supervised paper-autonomy diagnostics without broker access.

Options:

- `--tickers` — nargs=*
- `--timeframe` — default `1d`
- `--market-data-path`
- `--latest-processed-prices` — default `1d`
- `--as-of`
- `--max-age-hours` — float, default `72.0`
- `--paper-store` — default `data/dean_os/paper_trades.sqlite`
- `--initial-cash` — float, default `100000.0`
- `--position-size-pct` — float, default `0.05`
- `--include-watchlist` — flag
- `--review-snapshot`
- `--max-drawdown-limit` — float, default `0.1`
- `--output-dir` — default `reports/dean_os/paper_autonomy`
- `--event-log-path` — default `logs/dean_os/events.jsonl`
- `--decision-log-path` — default `logs/dean_os/decisions.jsonl`
- `--experience-diary` — default `logs/experience_diary.csv`
- `--print-json` — flag


## `run_agent_paper_portfolio.py`

Simulate logged paper decisions as a paper-only portfolio.

Options:

- `--store` — default `data/dean_os/paper_trades.sqlite`
- `--market-data-path`
- `--latest-processed-prices` — default `1d`
- `--tickers` — nargs=*
- `--as-of`
- `--initial-cash` — float, default `100000.0`
- `--position-size-pct` — float, default `0.05`
- `--include-watchlist` — flag
- `--watchlist-position-size-pct` — float, default `0.0`
- `--confidence-weighting` — flag
- `--slippage-bps` — float, default `5.0`
- `--commission-bps` — float, default `1.0`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--statuses` — nargs=*, default `['pending', 'evaluated']`
- `--limit` — int
- `--max-drawdown-limit` — float, default `0.1`
- `--output`
- `--output-dir` — default `reports/dean_os/paper_portfolio`
- `--print-json` — flag


## `run_agent_paper_simulation_plan.py`

Run PaperSimulationPlanBuilder (paper_simulation_plan).

Options:

- `--receipt-path` — **required**
- `--output-dir` — default `reports/dean_os/paper_simulation_plan_current`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_paper_trades.py`

Record, inspect, and evaluate DEAN-OS paper-only decisions.

Positional:

- `trade_id`

Options:

- `--store` — default `data/dean_os/paper_trades.sqlite`
- `--print-json` — flag
- `--action` — **required**
- `--tickers` — nargs=*
- `--expected-direction`
- `--source-type` — default `manual`
- `--source-id` — default `""`
- `--agent-name` — default `chief_review`
- `--horizon-days` — int, default `30`
- `--thesis` — default `""`
- `--confidence` — float, default `0.0`
- `--context-tags` — nargs=*
- `--regime-tags` — nargs=*
- `--status`
- `--agent-name`
- `--market-data-path`
- `--latest-processed-prices`
- `--tickers` — nargs=*
- `--as-of`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--allow-early` — flag
- `--apply` — flag
- `--neutral-band` — float, default `0.01`
- `--limit` — int
- `--output`
- `--output-dir` — default `reports/dean_os/paper_trades`
- `--reason` — **required**


## `run_agent_pipeline_control_bounded_evidence_batch.py`

Run a predeclared set of offline bounded evidence contexts.

Options:

- `--coverage-json` — **required**
- `--ticker` — **required**, repeatable
- `--frozen-context` — repeatable
- `--macro-source-path`
- `--rows-per-context` — int, default `480`
- `--max-features` — int, default `40`
- `--gap-size` — int, default `5`
- `--min-rows` — int, default `180`
- `--transaction-cost` — float, default `0.0025`
- `--no-real-metric-review` — flag
- `--max-contexts` — int, default `8`
- `--input-is-enriched` — flag
- `--output-dir` — default `reports/dean_os/pipeline_control_bounded_evidence_batch_current`
- `--print-json` — flag


## `run_agent_pipeline_control_caution_review_packet.py`

Build the pipeline control caution review packet.

Options:

- `--pipeline-metric-input-readiness-json`
- `--pipeline-control-instance-json`
- `--model-performance-report-json`
- `--feature-report-json`
- `--data-quality-json`
- `--output-dir` — default `reports/dean_os/pipeline_control_caution_review_packet`
- `--no-save` — flag


## `run_agent_pipeline_control_data_preflight.py`

Run saved-data coverage and non-destructive repair as one offline command.

Options:

- `--assets-yaml` — default `src/config/assets.yaml`
- `--price-path` — repeatable
- `--macro-path` — repeatable
- `--required-model-rows` — int, default `180`
- `--min-daily-source-bars` — int, default `24`
- `--output-dir` — default `reports/dean_os/pipeline_control_data_preflight_current`
- `--print-json` — flag


## `run_agent_pipeline_control_evidence_inventory.py`

Inventory real local pipeline outputs as metric evidence candidates.

Options:

- `--candidate-paths` — nargs=+
- `--output-dir` — default `reports/dean_os/pipeline_control_evidence_inventory_current`
- `--no-save` — flag


## `run_agent_pipeline_control_forward_data_accrual_gate.py`

Validate a saved source against a registered forward-development boundary.

Options:

- `--accrual-plan-json` — **required**
- `--source-path` — **required**
- `--output-dir` — default `reports/dean_os/pipeline_control_forward_data_accrual_gate_current`
- `--no-save` — flag


## `run_agent_pipeline_control_forward_data_accrual_plan.py`

Register a prospective boundary for the next development refresh.

Options:

- `--walk-forward-json` — **required**
- `--acknowledge-development-refresh-only` — flag
- `--output-dir` — default `reports/dean_os/pipeline_control_forward_data_accrual_plan_current`
- `--no-save` — flag


## `run_agent_pipeline_control_historical_price_recovery.py`

Run PipelineControlHistoricalPriceRecovery (pipeline_control_historical_price_recovery).

Options:

- `--historical-15m-path`
- `--current-15m-path`
- `--historical-1d-path`
- `--required-development-rows` — int, default `180`
- `--minimum-past-evaluation-rows` — int, default `60`
- `--min-daily-source-bars` — int, default `24`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/pipeline_control_historical_price_recovery_current`
- `--print-json` — flag


## `run_agent_pipeline_control_instance_contract.py`

Build the pipeline control instance contract.

Options:

- `--pipeline-surface-json`
- `--architecture-map-json`
- `--domain-instance-contract-json`
- `--output-dir` — default `reports/dean_os/pipeline_control_instance_contract`
- `--no-save` — flag


## `run_agent_pipeline_control_locked_evaluation_assembler.py`

Assemble a locked model-evaluation artifact from joined real candidates.

Options:

- `--training-candidate-json`
- `--evaluation-candidate-json`
- `--no-write-artifact` — flag
- `--output-dir` — default `reports/dean_os/pipeline_control_locked_evaluation_assembler_current`
- `--no-save` — flag


## `run_agent_pipeline_control_locked_feature_stability_assembler.py`

Assemble a locked feature-stability report from a measured candidate.

Options:

- `--feature-stability-candidate-json`
- `--no-write-artifact` — flag
- `--output-dir` — default `reports/dean_os/pipeline_control_locked_feature_stability_assembler_current`
- `--no-save` — flag


## `run_agent_pipeline_control_metric_artifact_materializer.py`

Materialize real metric artifacts from saved pipeline inputs.

Options:

- `--candidate-paths` — nargs=+
- `--no-write-artifacts` — flag
- `--output-dir` — default `reports/dean_os/pipeline_control_metric_artifact_materializer_current`
- `--no-save` — flag


## `run_agent_pipeline_control_metric_fixture_validation.py`

Run the synthetic control-flow validation for pipeline metric gates.

Options:

- `--output-dir` — default `reports/dean_os/pipeline_control_metric_fixture_validation`
- `--no-save` — flag


## `run_agent_pipeline_control_real_metric_evidence_run.py`

Run pipeline-control gates from real locked metric artifacts.

Options:

- `--model-evaluation-json`
- `--feature-stability-report`
- `--replay-batch-json`
- `--data-quality-json`
- `--constraints-path`
- `--architecture-map-json`
- `--domain-instance-contract-json`
- `--output-dir` — default `reports/dean_os/pipeline_control_real_metric_evidence_run`
- `--no-save` — flag


## `run_agent_pipeline_control_saved_data_coverage.py`

Inventory saved asset, timeframe, and macro coverage without training.

Options:

- `--assets-yaml` — default `src/config/assets.yaml`
- `--price-path` — repeatable
- `--macro-path` — repeatable
- `--min-rows` — int, default `180`
- `--max-rows` — int, default `600`
- `--max-abs-return` — float, default `0.25`
- `--min-cadence-ratio` — float, default `0.75`
- `--output-dir` — default `reports/dean_os/pipeline_control_saved_data_coverage_current`
- `--print-json` — flag


## `run_agent_pipeline_control_saved_price_repair.py`

Build non-destructive clean and resampled price candidates from coverage.

Options:

- `--coverage-json` — **required**
- `--required-model-rows` — int, default `180`
- `--min-daily-source-bars` — int, default `24`
- `--domain-id`. Require a recursively verified domain sector-market coverage bridge instead of generic pipeline coverage.
- `--output-dir` — default `reports/dean_os/pipeline_control_saved_price_repair_current`
- `--print-json` — flag


## `run_agent_pipeline_control_surface.py`

Build a bounded control surface for proposal-only pipeline tuning.

Options:

- `--model-performance`. JSON/CSV model or backtest performance artifact.
- `--replay-batch`. Historical replay batch JSON artifact.
- `--feature-report`. JSON/CSV feature importance/stability artifact.
- `--data-quality`. JSON/CSV leakage/data-quality artifact.
- `--constraints`. Optional JSON constraints override.
- `--output-dir` — default `reports/dean_os/pipeline_control_surface`
- `--print-json` — flag


## `run_agent_pipeline_feature_timeframe_audit.py`

Run PipelineFeatureTimeframeAudit (pipeline_feature_timeframe_audit).

Options:

- `--features-path`
- `--stage5-path`
- `--tickers` — repeatable
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/pipeline_feature_timeframe_audit_current`
- `--print-json` — flag


## `run_agent_pipeline_metric_input_readiness_gate.py`

Build the pipeline metric input readiness gate.

Options:

- `--model-performance`
- `--replay-batch`
- `--feature-report`
- `--data-quality`
- `--constraints-path`
- `--output-dir` — default `reports/dean_os/pipeline_metric_input_readiness_gate`
- `--no-save` — flag


## `run_agent_pipeline_model_case_packet.py`

Build a review-only case from one locked pipeline evaluation chain.

Options:

- `--real-metric-evidence-json`
- `--model-evaluation-json`
- `--feature-stability-json`
- `--output-dir` — default `reports/dean_os/pipeline_model_case_packet_current`
- `--no-save` — flag


## `run_agent_pipeline_model_feedback_packet.py`

Normalize human feedback for a pipeline model case.

Options:

- `--pipeline-model-case-json`
- `--manual-feedback-json`
- `--output-dir` — default `reports/dean_os/pipeline_model_feedback_packet_current`
- `--no-save` — flag


## `run_agent_pipeline_prediction_review_packet.py`

Run PipelinePredictionReviewPacket (pipeline_prediction_review_packet).

Options:

- `--output-dir` — default `reports/dean_os/pipeline_prediction_review_packet_current`
- `--pipeline-result` — **required**. Path to a saved JSON artifact.
- `--requested-tickers` — repeatable
- `--requested-timeframes` — repeatable
- `--filter-to-requested-scope` — flag
- `--source-artifact-path`
- `--sector-to-ticker-review-path`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_pipeline_stage23_runtime_profile.py`

Run PipelineStage23RuntimeProfile (pipeline_stage23_runtime_profile).

Options:

- `--output-dir` — default `reports/dean_os/pipeline_stage23_runtime_profile_current`
- `--source-path` — **required**
- `--tickers` — **required**, repeatable
- `--timeframes` — **required**, repeatable
- `--max-rows-per-ticker` — int, default `80`
- `--include-stage2` — flag
- `--include-stage3` — flag
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_pipeline_target_readiness_audit.py`

Run PipelineTargetReadinessAudit (pipeline_target_readiness_audit).

Options:

- `--targets-path`
- `--tickers` — repeatable
- `--timeframe`
- `--features-path`
- `--batch-metadata-path`
- `--target-registry-path` — default `src/config/targets.yaml`
- `--minimum-non-null-ratio` — float, default `0.5`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/pipeline_target_readiness_audit_current`
- `--print-json` — flag


## `run_agent_pipeline_timeframe_lane_readiness.py`

Run PipelineTimeframeLaneReadinessPlan (pipeline_timeframe_lane_readiness).

Options:

- `--source-path`
- `--tickers` — repeatable
- `--timeframes` — repeatable
- `--max-rows-per-ticker` — int, default `200`
- `--pipeline-context-base` — default `reports/dean_os`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/pipeline_timeframe_lane_readiness_current`
- `--print-json` — flag


## `run_agent_post_paper_simulation_review.py`

Run PostPaperSimulationReviewBuilder (post_paper_simulation_review).

Options:

- `--paper-simulation-result-path` — **required**
- `--output-dir` — default `reports/dean_os/post_paper_simulation_review_current`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_prospective_accumulation_runbook.py`

Run ProspectiveAccumulationRunbookBuilder (prospective_accumulation_runbook).

Options:

- `--output-dir` — default `reports/dean_os/prospective_accumulation_runbook_current`
- `--evidence-plan-path` — **required**
- `--checkpoint-monitor-path` — **required**
- `--as-of`. ISO-8601 timestamp; defaults to now in UTC.
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_prospective_accumulation_schedule.py`

Run ProspectiveAccumulationScheduleBuilder (prospective_accumulation_schedule).

Options:

- `--output-dir` — default `reports/dean_os/prospective_accumulation_schedule_current`
- `--runbook-path` — **required**
- `--as-of`. ISO-8601 timestamp; defaults to now in UTC.
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_real_source_dropzone_inventory.py`

Inventory operator-supplied research files in a dropzone.

Options:

- `--dropzone`
- `--recursive` — flag
- `--output-dir` — default `reports/dean_os/real_source_dropzone_inventory_current`
- `--no-save` — flag


## `run_agent_regime.py`

Run RegimeAgent as a soft pipeline report without trading execution.

Options:

- `--market-data-path`
- `--latest-processed-prices` — default `1d`
- `--ticker`
- `--engine` — default `fallback`
- `--manual-regime`
- `--manual-tags` — nargs=*
- `--close-col` — default `close`
- `--volume-col` — default `volume`
- `--output`
- `--output-dir` — default `reports/dean_os/regime`
- `--print-json` — flag


## `run_agent_replay_calibration_readiness.py`

Check whether repaired replay evidence is ready for manual analyst calibration review.

Options:

- `--repair-report-json` — default `reports/dean_os/replay_price_artifact_repair_current/latest.json`
- `--price-quality-json` — default `reports/dean_os/replay_price_quality_investigation_repaired_artifact_only_v2/latest.json`
- `--replay-batch-json` — default `reports/dean_os/historical_replay_batch_repaired_202603_202604/latest.json`
- `--research-batch-json` — default `reports/dean_os/historical_research_replay_batch_repaired_202603_202604/latest.json`
- `--min-clean-replay-runs` — int, default `10`
- `--min-clean-research-runs` — int, default `10`
- `--max-quality-blocked-runs` — int, default `0`
- `--max-price-warning-records` — int, default `0`
- `--max-weak-evidence-runs` — int, default `0`
- `--min-directional-research-ratio` — float, default `0.25`
- `--output-dir` — default `reports/dean_os/replay_calibration_readiness_gate`
- `--print-json` — flag


## `run_agent_replay_checkpoint_due_router.py`

Run ReplayCheckpointDueRouter (replay_checkpoint_due_router).

Options:

- `--output-dir` — default `reports/dean_os/replay_checkpoint_due_router_current`
- `--registration-json` — **required**
- `--review-gate-json` — **required**
- `--as-of`. ISO-8601 timestamp; defaults to now in UTC.
- `--verified-price-paths` — repeatable. Repeatable path to a saved artifact.
- `--pipeline-paths` — repeatable. Repeatable path to a saved artifact.
- `--outcome-json-paths` — repeatable. Repeatable path to a saved artifact.
- `--due-soon-days` — int, default `3`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_replay_checkpoint_monitor.py`

Run ReplayCheckpointMonitorBuilder (replay_checkpoint_monitor).

Options:

- `--output-dir` — default `reports/dean_os/replay_checkpoint_monitor_current`
- `--evidence-plan-path` — **required**
- `--as-of`. ISO-8601 timestamp; defaults to now in UTC.
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_replay_outcome_evidence_plan.py`

Run ReplayOutcomeEvidencePlanBuilder (replay_outcome_evidence_plan).

Options:

- `--packet-path`
- `--routing-path`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/replay_outcome_evidence_plan_current`
- `--print-json` — flag


## `run_agent_replay_price_artifact_repair.py`

Create a non-destructive candidate repair for mixed replay price artifacts.

Positional:

- `price_data_path`. Raw cached or normalized price CSV/parquet file.

Options:

- `--tickers` — nargs=*. Optional ticker allow-list.
- `--output-dir` — default `reports/dean_os/replay_price_artifact_repair`
- `--artifact-dir` — default `data/dean_os/replay_prices`
- `--artifact-path`. Optional explicit .csv or .parquet artifact path.
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--benchmark-ticker` — default `SPY`
- `--anomaly-threshold` — float, default `0.3`
- `--anchor-bridge-threshold` — float, default `0.15`
- `--write-artifact` — flag. Write the candidate repaired artifact.
- `--print-json` — flag


## `run_agent_replay_price_normalizer.py`

Create a normalized daily OHLCV artifact for safe DEAN-OS historical replay.

Positional:

- `price_data_path`. Raw cached price CSV/parquet file.

Options:

- `--tickers` — nargs=*. Optional ticker allow-list.
- `--output-dir` — default `reports/dean_os/replay_price_normalizer`
- `--artifact-dir` — default `data/dean_os/replay_prices`
- `--artifact-path`. Optional explicit .csv or .parquet artifact path.
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--benchmark-ticker` — default `SPY`
- `--compare-replay` — flag. Run raw vs normalized historical replay comparison.
- `--as-of`. Required with --compare-replay.
- `--lookback-days` — int, default `180`
- `--horizon-days` — int, default `60`
- `--news-data`
- `--macro-data`
- `--neutral-band` — float, default `0.01`
- `--print-json` — flag. Print full JSON payload.


## `run_agent_replay_price_quality_investigation.py`

Build a read-only investigation plan for replay price-quality blockers.

Options:

- `--report-json` — repeatable. Replay/normalizer/batch report JSON.
- `--artifact-only` — flag. Skip default report JSONs and inspect only price artifacts.
- `--price-data` — repeatable. Additional price CSV/parquet artifacts to inspect.
- `--benchmark-ticker` — default `SPY`
- `--close-col` — default `close`
- `--datetime-col` — default `datetime`
- `--large-step-threshold` — float, default `0.15`
- `--output-dir` — default `reports/dean_os/replay_price_quality_investigation`
- `--print-json` — flag


## `run_agent_review.py`

Build a DEAN-OS human-review snapshot from lab, learning, queue, memory, and logs.

Options:

- `--report-path`
- `--reports-dir` — default `reports/dean_os/agent_lab`
- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--operations-store` — default `data/dean_os/operation_queue.sqlite`
- `--review-actions-store` — default `data/dean_os/review_actions.sqlite`
- `--memory-store` — default `data/dean_os/recommendation_memory.sqlite`
- `--log-path` — default `logs/dean_os/events.jsonl`
- `--output-dir` — default `reports/dean_os/review`
- `--event-limit` — int, default `10`
- `--print-json` — flag


## `run_agent_review_action_apply_ceremony.py`

Record exactly one validated review action from a dry-run artifact.

Options:

- `--dry-run-json` — default `reports/dean_os/review_action_dry_run/latest.json`
- `--review-actions-store` — default `data/dean_os/review_actions.sqlite`
- `--operations-store` — default `data/dean_os/operation_queue.sqlite`
- `--log-path` — default `logs/dean_os/events.jsonl`
- `--apply-review-action` — flag
- `--output-dir` — default `reports/dean_os/review_action_apply_ceremony`
- `--print-json` — flag


## `run_agent_review_action_dry_run.py`

Preview a review action from a decision packet without writing it.

Options:

- `--packet-json` — default `reports/dean_os/review_decision_packet/latest.json`
- `--intent` — default `needs_more_data`
- `--reviewer` — default `human`
- `--review-notes` — default `""`
- `--data-request` — default `Add stronger citations or missing source coverage before learning promotion.`
- `--acknowledge-warnings` — flag
- `--output-dir` — default `reports/dean_os/review_action_dry_run`
- `--print-json` — flag


## `run_agent_review_actions.py`

Record human review lifecycle actions for DEAN-OS.

Positional:

- `action_id`

Options:

- `--store` — default `data/dean_os/review_actions.sqlite`
- `--operations-store` — default `data/dean_os/operation_queue.sqlite`
- `--log-path` — default `logs/dean_os/events.jsonl`
- `--reports-dir` — default `reports/dean_os/agent_lab`
- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--print-json` — flag
- `--source-type`
- `--action-type`
- `--reason` — default `""`
- `--source-type` — **required**
- `--source-id` — **required**
- `--notes` — default `""`
- `--reviewer` — default `human`
- `--data-request` — **required**
- `--tickers` — nargs=*
- `--thesis` — **required**
- `--reason` — **required**


## `run_agent_review_approved_learning.py`

Run the explicit review-approved loop for analyst learning promotion.

Options:

- `--profile-run-json`. AnalystProfileOrchestrator JSON.
- `--agent-lab-report-json`. Direct Agent Lab report JSON.
- `--learning-store` — default `data/dean_os/agent_learning.sqlite`
- `--review-actions-store` — default `data/dean_os/review_actions.sqlite`
- `--operations-store` — default `data/dean_os/operation_queue.sqlite`
- `--memory-store` — default `data/dean_os/recommendation_memory.sqlite`
- `--reviewer` — default `human`
- `--review-notes` — default `""`
- `--mark-reviewed` — flag. Record mark_reviewed for discovered Agent Lab reports.
- `--needs-more-data`. Record an open data request instead of promotion approval.
- `--apply` — flag. Apply learning promotion after review gates pass.
- `--allow-weak-notes` — flag
- `--allow-duplicates` — flag
- `--default-horizon-days` — int, default `365`
- `--no-context-summary` — flag
- `--output-dir` — default `reports/dean_os/review_approved_learning`
- `--print-json` — flag


## `run_agent_review_decision_packet.py`

Build a read-only review decision packet for an analyst inbox source.

Options:

- `--inbox-json` — default `reports/dean_os/analyst_review_inbox/latest.json`
- `--source-id`
- `--max-notes` — int, default `6`
- `--max-citations-per-note` — int, default `3`
- `--max-text-chars` — int, default `500`
- `--output-dir` — default `reports/dean_os/review_decision_packet`
- `--print-json` — flag


## `run_agent_review_decision_state.py`

Run ReviewDecisionStateBuilder (review_decision_state).

Options:

- `--evidence-plan-path`
- `--voi-review-path`
- `--actor` — default `dean_os_policy`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/review_decision_state_current`
- `--print-json` — flag


## `run_agent_review_index.py`

Run ReviewIndexBuilder (review_index).

Options:

- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/review_index_current`
- `--print-json` — flag


## `run_agent_saved_macro_evidence_producer.py`

Build review-only macro evidence from a saved long-form FRED snapshot (series_id/datetime/value/realtime_start rows -- e.g. the FredCollector output at data/processed/features/macro_data.parquet).

Positional:

- `source_path`

Options:

- `--as-of` — **required**
- `--registry-path`. Override dean_os/config/macro_series_registry.yaml.
- `--output-dir` — default `reports/dean_os/saved_macro_evidence_producer_current`
- `--no-save` — flag


## `run_agent_saved_news_shard_snapshot.py`

Run SavedNewsShardSnapshotBuilder (saved_news_shard_snapshot).

Options:

- `--output-dir` — default `reports/dean_os/saved_news_shard_snapshot_current`
- `--database-path` — **required**
- `--output-parquet-path` — **required**
- `--as-of`. ISO-8601 timestamp; defaults to now in UTC.
- `--include-parquet-paths` — repeatable
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_saved_official_policy_evidence.py`

Bind one official policy source to independent news corroboration.

Positional:

- `snapshot_artifact_path`. Path to the official policy snapshot artifact
- `corroborating_news_artifact_path`. Path to the corroborating news artifact

Options:

- `--as-of` — **required**. Time boundary (ISO 8601 timezone-aware)
- `--registry-path`. Path to the official policy evidence registry
- `--output-dir` — default `reports/dean_os/saved_official_policy_evidence_producer`. Output directory for the artifact
- `--no-save` — flag. Run without saving the artifact


## `run_agent_saved_sector_market_evidence.py`

Build review-only sector market evidence from one verified saved price-repair artifact.

Positional:

- `repair_artifact`

Options:

- `--domain-id` — **required**
- `--as-of` — **required**
- `--lookback-sessions` — int, default `20`
- `--min-source-bars-per-day` — int, default `24`
- `--max-staleness-days` — int, default `7`
- `--output-dir` — default `reports/dean_os/saved_sector_market_evidence_producer_current`
- `--no-save` — flag


## `run_agent_sector_to_ticker_review_packet.py`

Run SectorToTickerReviewPacket (sector_to_ticker_review_packet).

Options:

- `--bridge-path` — default `reports/dean_os/sector_thesis_to_ticker_basket_current/latest.json`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/sector_to_ticker_review_packet_current`
- `--print-json` — flag


## `run_agent_shadow_calibration_case_index.py`

Run ShadowCalibrationCaseIndexBuilder (shadow_calibration_case_index).

Options:

- `--prediction-review-path` — **required**
- `--outcome-source-path` — **required**
- `--output-dir` — default `reports/dean_os/shadow_calibration_case_index_current`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_shadow_calibration_diagnostics.py`

Run ShadowCalibrationDiagnostics (shadow_calibration_diagnostics).

Options:

- `--case-index-path` — **required**
- `--policy-path` — default `dean_os/config/shadow_calibration_policy.yaml`
- `--output-dir` — default `reports/dean_os/shadow_calibration_diagnostics_current`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_shadow_calibration_readiness.py`

Run ShadowCalibrationReadinessPacket (shadow_calibration_readiness).

Options:

- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/shadow_calibration_readiness_current`
- `--print-json` — flag


## `run_agent_shadow_component_case_producer.py`

Run ShadowComponentCaseProducer (shadow_component_case_producer).

Options:

- `--base-case-index-path` — **required**
- `--component` — **required**
- `--component-artifact-path` — **required**
- `--output-dir` — default `reports/dean_os/shadow_component_case_producer_current`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_source_evidence_validation_gate.py`

Run SourceEvidenceValidationGate (source_evidence_validation_gate).

Options:

- `--source-json` — default `reports/dean_os/analyst_evidence_pack_refreshed_gap_check/latest.json`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/source_evidence_validation_gate_current`
- `--print-json` — flag


## `run_agent_source_extraction_fixture_packet.py`

Run SourceExtractionFixturePacket (source_extraction_fixture_packet).

Options:

- `--contract-json` — default `reports/dean_os/source_extraction_review_packet_current/latest.json`
- `--max-items` — int, default `12`
- `--no-prefer-timestamped` — flag
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/source_extraction_fixture_packet_current`
- `--print-json` — flag


## `run_agent_source_extraction_fixture_review_gate.py`

Run SourceExtractionFixtureReviewGate (source_extraction_fixture_review_gate).

Options:

- `--fixture-json` — default `reports/dean_os/source_extraction_fixture_packet_current/latest.json`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/source_extraction_fixture_review_gate_current`
- `--print-json` — flag


## `run_agent_source_extraction_review_packet.py`

Run SourceExtractionReviewPacket (source_extraction_review_packet).

Options:

- `--source-json` — default `reports/dean_os/analyst_evidence_pack_refreshed_gap_check/latest.json`
- `--source-gate-json` — default `reports/dean_os/source_evidence_validation_gate_current/latest.json`
- `--domain-packet-json` — default `reports/dean_os/domain_specialist_review_packet_current/latest.json`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/source_extraction_review_packet_current`
- `--print-json` — flag


## `run_agent_source_routing.py`

Route local materials and collector inventory to specialist/pipeline intake paths.

Positional:

- `materials_path` — nargs=?

Options:

- `--collector-inventory`
- `--output`
- `--output-dir` — default `reports/dean_os/source_routing`
- `--print-json` — flag


## `run_agent_staged_workbench_integration_review.py`

Review-only audit of staged web-bot workbench material.

Options:

- `--draft-bundle`
- `--dropzone`
- `--output-dir` — default `reports/dean_os/staged_workbench_integration_review_current`
- `--no-save` — flag


## `run_agent_strategy_maturity_daily_reconciliation.py`

Reconcile a candidate playbook with the verified maturity-decision ledger.

Options:

- `--assessment` — **required**. Path to the strategy replay candidate assessment JSON
- `--risk-snapshot`. Optional path to the strategy risk snapshot JSON
- `--no-save` — flag. Run in dry-run mode without saving outputs


## `run_agent_strategy_replay_candidate_assessment.py`

Evaluate one real reviewed hypothesis as a research-only strategy candidate.

Options:

- `--review-gate` — **required**. Path to the world model replay review gate JSON
- `--hypothesis-id`. Optional specific hypothesis ID to evaluate
- `--no-save` — flag. Run in dry-run mode without saving outputs


## `run_agent_tuning.py`

Run TuningAgent as a proposal-only planner with optional PipelineControlSurface gating.

Positional:

- `performance_path` — nargs=?. Model performance JSON/CSV artifact.

Options:

- `--regime-context-json`
- `--control-surface-json`
- `--tickers` — nargs=*
- `--timeframes` — nargs=*
- `--require-control-surface` — flag
- `--output-dir` — default `reports/dean_os/tuning`
- `--print-json` — flag


## `run_agent_unknown_voi_candidate_proposal.py`

Run UnknownValueOfInformationCandidateProposalBuilder (unknown_voi_candidate_proposal).

Options:

- `--evidence-plan-path`
- `--voi-review-path`
- `--max-candidates` — int, default `3`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/unknown_voi_candidate_proposal_current`
- `--print-json` — flag


## `run_agent_unknown_voi_review.py`

Run UnknownValueOfInformationReviewBuilder (unknown_voi_review).

Options:

- `--evidence-plan-path`
- `--assessments-path`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/unknown_voi_review_current`
- `--print-json` — flag


## `run_agent_verified_local_snapshot_ingestion.py`

Run VerifiedLocalSnapshotIngestion (verified_local_snapshot_ingestion).

Options:

- `--source-router-json`
- `--candidate-path`
- `--registration-json`
- `--review-gate-json`
- `--as-of`
- `--pipeline-paths` — repeatable
- `--prior-outcome-json-paths` — repeatable
- `--packet-json`
- `--journal-path` — default `data/dean_os/system_journal.jsonl`
- `--apply-ingestion` — flag
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/verified_local_snapshot_ingestion_current`
- `--print-json` — flag


## `run_agent_verified_market_source_router.py`

Run VerifiedMarketSourceRouter (verified_market_source_router).

Options:

- `--lifecycle-json`
- `--registration-json`
- `--review-gate-json`
- `--source-policy-json` — default `dean_os/config/replay_verified_market_sources.template.json`
- `--previous-refresh-json-paths` — repeatable
- `--local-snapshot-paths` — repeatable
- `--as-of`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/verified_market_source_router_current`
- `--print-json` — flag


## `run_agent_world_model_pipeline_context.py`

Run WorldModelPipelineContextDiscovery (world_model_pipeline_context).

Options:

- `--tickers` — repeatable
- `--timeframes` — repeatable
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/world_model_pipeline_context_current`
- `--print-json` — flag


## `run_agent_world_model_replay_registration.py`

Run WorldModelReplayRegistrationBridge (world_model_replay_registration).

Options:

- `--output-dir` — default `reports/dean_os/world_model_replay_registration_current`
- `--gate-json` — **required**. Path to a saved JSON artifact.
- `--source-packet-json`. Path to a saved JSON artifact.
- `--tracker-db-path` — default `data/dean_os/outcome_tracker.sqlite`
- `--apply` — flag
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_world_model_replay_review_gate.py`

Run WorldModelReplayReviewGate (world_model_replay_review_gate).

Options:

- `--output-dir` — default `reports/dean_os/world_model_replay_review_gate_current`
- `--packet-json` — **required**. Path to a saved JSON artifact.
- `--approve` — flag
- `--reviewer`
- `--review-notes`
- `--no-save` — flag. Build the payload without writing report files.
- `--print-json` — flag


## `run_agent_world_model_review_resolution.py`

Run WorldModelReviewResolutionBuilder (world_model_review_resolution).

Options:

- `--packet-json`
- `--review-gate-json`
- `--resolution-specs-json`
- `--no-save` — flag. Build the payload without writing report files.
- `--output-dir` — default `reports/dean_os/world_model_review_resolution_current`
- `--print-json` — flag


## Retired / never implemented

Commands referenced by the prose docs that have no wrapper on disk. Listed here so a reference to a missing command is a recorded decision rather than silent drift.

- `run_agent_accumulation_authorization.py` — Missing wrapper. Candidate: dean_os/accumulation_authorization_ledger.py (unverified).
- `run_agent_capability_matrix.py` — Missing wrapper. Candidate: dean_os/agent_capability_matrix.py (unverified).
- `run_agent_collector_inventory.py` — Retired. CollectorInventoryAgent was archived 2026-07-24 for never being registered or instantiated. The scan it wrapped now lives in dean_os/collector_inventory_scan.py and runs in-process during MarketDataRefreshRunbook and SourceRoutingAgent, so no wrapper is needed.
- `run_agent_composite_domain_pipeline.py` — Never implemented. No backing module found under dean_os/. The composite pipeline_manager agent it implies is default-off in agent_registry.yaml.
- `run_agent_context_evidence_review.py` — Missing wrapper. Candidate: dean_os/packets/context_evidence_review_packet.py (unverified).
- `run_agent_domain_analyst_binding_plan.py` — Missing wrapper. Candidate: dean_os/analyst_core/domain_analyst_binding_planner.py (unverified).
- `run_agent_domain_analyst_review.py` — Missing wrapper. Candidates: dean_os/analyst_core/domain_analyst_review_run.py or domain_analyst_thesis_review_packet.py (unverified, and note that run_agent_domain_analyst_thesis_review_packet.py already exists).
- `run_agent_domain_binding_task_dispatch.py` — Missing wrapper. Candidate: dean_os/domain_binding_task_dispatcher.py (unverified).
- `run_agent_domain_specialist_review_packet.py` — Missing wrapper. Candidates: dean_os/packets/specialist_context_review_packet.py or dean_os/analyst_core/domain_analyst_review_run.py (unverified).
- `run_agent_full_system_cycle_world_model.py` — Missing wrapper. Candidate: dean_os/full_system_cycle_world_model_bridge.py (unverified).
- `run_agent_full_system_review_cycle.py` — Missing wrapper. Candidate: dean_os/full_system_review_cycle.py (unverified).
- `run_agent_orchestrator.py` — Missing wrapper. Candidate: dean_os/orchestrator.py (unverified). Note run_agent_domain_orchestrator.py exists and may already cover this.
- `run_agent_paper_simulation_result.py` — Missing wrapper. Candidate: dean_os/paper_simulation_result.py (unverified). Its builder requires a paper_simulation_result_path at construction time.
- `run_agent_pipeline_control_bounded_evidence.py` — Missing wrapper. Candidate: dean_os/pipeline_control/pipeline_control_bounded_evidence_run.py (unverified). Note run_agent_pipeline_control_bounded_evidence_batch.py exists.
- `run_agent_pipeline_control_feature_causality_audit.py` — Missing wrapper. Candidate: dean_os/pipeline_control/pipeline_control_feature_causality_audit.py (unverified).
- `run_agent_pipeline_control_walk_forward_validation.py` — Missing wrapper. Candidate: dean_os/pipeline_control/pipeline_control_walk_forward_validation_run.py (unverified).
- `run_agent_pipeline_stage23_regeneration.py` — Missing wrapper. Candidate: dean_os/pipeline_stage23_regeneration.py (unverified). Its PipelineStage23Regeneration class has no plain build() signature that a flat CLI can drive, which is why generation was skipped.
- `run_agent_pipeline_stage4_exact_context_review.py` — Missing wrapper. Candidate: dean_os/pipeline_stage4_exact_context_review.py (unverified).
- `run_agent_real_source_normalized_packet.py` — Missing wrapper. Candidate: dean_os/packets/real_source_normalized_packet.py (unverified).
- `run_agent_relative_return_direction_policy.py` — Missing wrapper. Candidate: dean_os/relative_return_direction_policy.py (unverified).
- `run_agent_replay_evidence_refresh.py` — Missing wrapper. Candidate: dean_os/replays/replay_evidence_refresh_controller.py (unverified).
- `run_agent_replay_evidence_windows.py` — Never implemented. No backing module found under dean_os/.
- `run_agent_replay_lifecycle_journal.py` — Missing wrapper. Candidate: dean_os/replays/replay_lifecycle_journal_bridge.py (unverified).
- `run_agent_replay_outcome_lifecycle.py` — Missing wrapper. Candidate: dean_os/replays/replay_outcome_lifecycle_orchestrator.py (unverified).
- `run_agent_research_replay_directionality.py` — Missing wrapper. Candidate: dean_os/research_replay_directionality_diagnostic.py (unverified).
- `run_agent_saved_sec_companyfacts.py` — Missing wrapper. Candidate: dean_os/analysts/_producers/sec/companyfacts.py (unverified).
- `run_agent_saved_sec_derived_ratios.py` — Missing wrapper. Candidate: dean_os/analysts/_producers/sec/ratios.py (unverified).
- `run_agent_saved_sec_filing_index.py` — Missing wrapper. Candidate: dean_os/analysts/_producers/sec/filing_index.py (unverified).
- `run_agent_saved_sec_fundamental_merger.py` — Missing wrapper. Candidate: dean_os/analysts/_producers/sec/merger.py (unverified).
- `run_agent_saved_sec_inline_xbrl.py` — Missing wrapper. Candidate: dean_os/analysts/_producers/sec/inline_xbrl.py (unverified).
- `run_agent_saved_sec_submissions_filing_index.py` — Missing wrapper. Candidate: dean_os/analysts/_producers/sec/submissions_index.py (unverified).
- `run_agent_saved_semiconductor_news_evidence.py` — Missing wrapper. Candidate: dean_os/analysts/_producers/news.py (unverified). SavedSemiconductorNewsEvidenceProducer is referenced by tests, so the producer itself is live; only the CLI is absent.
- `run_agent_saved_ticker_specific_evidence.py` — Missing wrapper. Candidate: dean_os/analysts/_producers/ticker.py (unverified).
- `run_agent_sector_to_ticker_bridge.py` — Missing wrapper. Candidates: dean_os/sector_thesis_to_ticker_basket_bridge.py (the bridge) or dean_os/packets/sector_to_ticker_review_packet.py (the review packet). Unverified, and these are two different things.
- `run_agent_semiconductor_analyst.py` — Missing wrapper. The semiconductor_analyst agent is registered and enabled in agent_registry.yaml, so it is reachable through the orchestrator rather than a dedicated wrapper. Confirm before adding one.
- `run_agent_specialist_context_review.py` — Missing wrapper. Candidate: dean_os/packets/specialist_context_review_packet.py (unverified).
- `run_agent_ticker_attribution_audit.py` — Missing wrapper. Candidate: dean_os/ticker_specific_attribution_audit.py (unverified).
- `run_agent_ticker_focused_notes.py` — Missing wrapper. Candidate: dean_os/analysts/_producers/ticker.py (unverified).
- `run_agent_ticker_focused_replay_bridge.py` — Missing wrapper. Candidate: dean_os/ticker_focused_replay_exam_bridge.py (unverified).
- `run_agent_world_model_event_learning_packet.py` — Missing wrapper. Candidate: dean_os/world_model/world_model_event_learning.py (unverified). IMPLEMENTATION_STATUS.md claims this wrapper was added on 2026-07-09; it was never committed on any branch.
- `run_agent_world_model_hypothesis_lifecycle.py` — Missing wrapper. Candidate: dean_os/world_model/world_model_hypothesis_lifecycle_orchestrator.py (unverified).
- `run_agent_world_model_replay_registration_journal.py` — Missing wrapper. Candidate: dean_os/world_model/world_model_replay_registration_journal.py (unverified). Note run_agent_world_model_replay_registration.py exists and is a different thing (the bridge, not the journal).
- `run_agent_world_model_resolution_journal.py` — Missing wrapper. Candidate: dean_os/world_model/world_model_resolution_journal.py (unverified). Note run_agent_world_model_review_resolution.py exists and is a different thing.
