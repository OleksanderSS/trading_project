# DEAN-OS Command Checklist

Use these commands to gather useful logs and review state without running the heavy trading pipeline.

## 1. Agent Lab

Real materials:

```powershell
python run_agent_lab.py docs/research --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle --regime-tags rising_market
```

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

Feed the saved evidence pack into Agent Lab:

```powershell
python run_agent_lab.py --evidence-pack-json reports/dean_os/analyst_evidence_pack/latest.json --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle
```

Share the `coverage`, `analyst_inputs.base_analyst`, `analyst_inputs.manager_plan`, `warnings`, `dropped`, and `recommendations` sections.

Interpretation:
- This is the preferred bridge from raw sources to analysts.
- Start with one strong base analyst/evidence contract, then specialize by sector or domain later.
- The evidence pack does not run the heavy pipeline and does not create trades.

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
python run_agent_pipeline_control_surface.py --model-performance performance_data.json --replay-batch reports\dean_os\historical_replay_batch\latest.json --data-quality diagnostic_reports\feature_lineage_report.json
```

Share the `surface.status`, `surface.axes`, `surface.allowed_variation`, `proposal_gate`, and `recommendations` sections.

Interpretation:
- This is not a tuner and does not write production config.
- `surface.status=blocked` means TuningAgent should not propose experiments yet.
- `surface.status=caution` means only small reviewed experiments are allowed.
- `surface.status=clear` means reviewed experiments may be proposed inside the listed bounds.

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
