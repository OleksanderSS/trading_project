# DEAN-OS Next Chat Handoff

Last updated: 2026-06-13

This file preserves project context for a new Codex thread. Read this first, then
`dean_os/IMPLEMENTATION_STATUS.md`, `dean_os/COMMAND_CHECKLIST.md`, and the latest
reports under `reports/dean_os/`.

## Current Mission

Build a controlled multi-agent research and governance layer over the existing
trading project. The goal is not to run every collector or pipeline stage as soon
as possible. The goal is to create a system that can:

- inspect pipeline/data health;
- ingest research materials such as news, articles, filings, reports, transcripts,
  and books;
- produce cited specialist theses;
- remember hits, misses, lessons, regimes, and context tags;
- propose pipeline/research operations without executing them automatically;
- support human review before any promotion, training, tuning, or trading action.

## Operating Rules

- Do not run the heavy trading pipeline unless the user explicitly asks.
- Prefer isolated tests for every new layer.
- Keep live execution disabled.
- Agents may propose actions, but execution must go through review/approval.
- Do not treat sample Agent Lab theses as investment evidence.
- Do not print secrets or API keys.
- Avoid modifying unrelated files because the worktree is shared with other agents.

## Strategic Split

Pipeline feeds:

- `pipeline_price_feed`: market prices such as Yahoo Finance / market data.
- `pipeline_news_feed`: RSS, Google News, NewsAPI. These belong to the candle,
  macro-before/after, sentiment, and event-study pipeline after health checks.
- `pipeline_macro_feed`: FRED and economic-calendar style data.
- `pipeline_context_feed`: VIX, fear-greed, put/call, broad sentiment and context
  sources after schema/timestamp checks.

Research-specialist feeds:

- SEC filings, insider/CFTC-style feeds, transcripts, investor letters, books,
  sector reports, and long-form articles.
- These should first flow into `ResearchCorpus` as `ResearchDocument` records,
  then chunks/citations, then specialist patterns.
- They should not block daily pipeline runs until schema, retry, rate-limit,
  timestamp, and storage contracts are stable.

## Implemented Agent-System Pieces

- `AgentLabRunner`: isolated research-material run without the trading pipeline.
- `ResearchCorpus`: SQLite storage for research documents, chunks, notes.
- `FinancialNLPAgent`: rule-based fallback, FinBERT-ready interface.
- `SpecialistResearchAgent`: extracts early specialist patterns from materials.
- `EvidenceSynthesisAgent`: evidence-bound thesis synthesis.
- `LearningStore`: pending/completed thesis outcome tracking.
- `RecommendationMemoryStore`: manual and automatic hit/miss case memory.
- `RegimeContextBuilder`: converts OHLCV/regime output into stable context tags.
- `RegimeAgent`: pipeline soft-agent that turns regime context into a consensus-ready `PipelineReport`.
- `AgentPerformanceByContext`: weak/strong context reports by agent and regime.
- `OutcomeEvaluationRunner`: dry-run/apply evaluation of learning records.
- `MarketDataFreshnessAgent`: local price freshness preflight.
- `CollectorInventoryAgent`: local-only collector config/class inventory.
- `CollectorHealthAgent`: isolated local collector-output shape/timestamp/duplicate check.
- `ModelPerformanceAgent`: local evaluation/backtest metric preflight before model promotion or tuning.
- `TuningAgent`: proposal-only walk-forward tuning experiment planner.
- `ChiefReviewAgent`: top-level review synthesizer for supervised paper autonomy.
- `PaperTradeStore` and `PaperTradeEvaluationRunner`: durable autonomous paper decision log and outcome evaluator.
- `PaperPortfolioSimulator` and `PaperPortfolioAgent`: deterministic paper-only portfolio simulation from logged decisions, local prices, sizing, slippage, and commission assumptions.
- `PaperAutonomyRunner`: supervised daily paper-autonomy loop that combines freshness, regime, chief review, paper portfolio, DEAN logs, and pipeline experience diary summaries without executing trades.
- `DiaryBridgeAgent`: review-only bridge inspector between DEAN paper outcomes and the pipeline experience diary; creates schema/candidate proposals but never writes automatically.
- `HistoricalReplayRunner` and `HistoricalReplayAnalyst`: safe old-data replay that cuts data at `as_of`, removes future/target leakage columns, optionally normalizes daily bars, forms a thesis, and evaluates only afterward.
- `ReplayPriceNormalizer`: data-only runner that creates a reusable normalized daily OHLCV replay artifact, records price-quality warnings, and keeps learning-memory writes blocked when warnings remain.
- `HistoricalReplayBatchRunner`: repeated old-data replay exam across dates/horizons; summarizes hit/miss by ticker, horizon, and quality state without learning writes.
- `HistoricalResearchReplayRunner`: combined old-data research exam that builds a pre-`as_of` evidence pack, runs Agent Lab in isolated stores, and attaches post-hoc price outcome evaluation without learning writes.
- `EvidenceTimestampAudit`: read-only timestamp gate for cached news/macro/material tables and evidence packs before scaling historical research replay.
- `HistoricalResearchReplayBatchRunner`: repeated old-data research replay exam across dates/horizons; summarizes research stance, evidence coverage, price outcome, and quality gates.
- `ReplayPriceQualityInvestigationPlan`: read-only forensic plan for repeated replay price-quality blockers; inspects benchmark windows, artifacts, interval-mixing warnings, and large one-step moves.
- `ReplayPriceArtifactRepairPlan`: non-destructive candidate artifact repair that prefers valid midnight daily bars, quarantines mixed/intraday-like daily rows, writes audit metadata, and never mutates source caches.
- `ReplayCalibrationReadinessGate`: read-only gate that decides whether repaired replay evidence is ready for manual analyst-calibration review or still blocked by sample size, price quality, evidence coverage, or neutral/inconclusive research.
- `PipelineControlSurface`: bounded tuning surface that intersects profitability, risk, validation, feature stability, data quality, and replay repeatability before tuning proposals are allowed.
- `AnalystEvidencePackRunner`: local-only source normalizer that turns materials, cached news, and macro tables into citable `ResearchDocument` payloads for Agent Lab.
- `AnalystProfileOrchestrator`: central manager that runs the base analyst first and gates candidate specialist profiles behind explicit approval.
- `AnalystProfileScorecard`: promotion ledger for analyst profiles; aggregates saved profile runs and decides whether candidates stay blocked/candidate/ready.
- `AnalystLearningPromotionBridge`: dry-run/apply bridge from reviewed analyst notes/profile runs into durable learning records with evidence-pack/profile metadata.
- `ReviewApprovedLearningLoop`: explicit preview -> review action -> apply ceremony around analyst learning promotion.
- `AnalystOutcomeEvaluationLoop`: reviewed analyst thesis outcome evaluator with analyst-only filtering, dry-run/apply gates, profile outcomes, and audit metadata.
- `AnalystCalibrationGate`: proposal-only gate that combines profile scorecards, outcomes, and context performance before any profile/weight recommendation.
- `CalibrationProposalAgent`: turns ready calibration gate outputs into dry-run or enqueued `OperationQueue` review proposals.
- `CalibrationReviewLifecycle`: review-only lifecycle manager for calibration proposals, including operation dry-run and approve/reject queue statuses without config writes.
- `ManualImplementationBacklog`: read-only backlog for approved calibration proposals that still require separate manual implementation.
- `AgentLearningLoopRunbook`: read-only operator runbook that shows the current safe analyst-learning loop position, stop reason, next command, and safety contract.
- `AnalystLoopDailyCheck`: read-only daily operator check that combines the runbook, market freshness, evidence coverage, profile scorecard state, and DEAN logs into blockers/warnings.
- `AnalystReviewInbox`: read-only inbox for Agent Lab/profile reports that need human review before learning promotion.
- `ReviewDecisionPacket`: read-only per-source packet that summarizes notes, citations, evidence coverage, warnings, and command previews before a human review decision.
- `ReviewActionDryRun`: read-only validator for the selected review intent before any review action is recorded.
- `ReviewActionApplyCeremony`: explicit one-action review write gate after a successful review action dry-run.
- `EvidenceGapResolutionPlan`: read-only needs-more-data planner that turns evidence gaps into source/data tasks and rebuild commands.
- `AnalystLearningApplyCeremony`: explicit learning-record write gate after a validated analyst learning bridge dry-run.
- `OutcomeReadinessGate`: read-only outcome maturity and price-coverage gate for pending analyst learning records.
- `OutcomePriceCoveragePlan`: read-only price coverage planner after an outcome-readiness blocker.
- `MarketDataRefreshRunbook`: read-only operator runbook for clearing market-price coverage blockers without running collectors.
- `SourceRoutingAgent`: local source/material routing map for pipeline feeds and specialist-agent intake.
- `OperationQueue` and `OperationsProposalAgent`: proposal-only automation loop.
- `ReviewActionStore` and `AgentReviewBuilder`: durable human review actions.
- `EventLog`: JSONL observability for agent runs and operation actions.

## Latest Verified State

Latest full DEAN-OS test command:

```powershell
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_replay_calibration_readiness_full
```

Latest result:

```text
105 passed
```

Latest collector inventory command:

```powershell
python run_agent_collector_inventory.py --output reports/dean_os/collector_inventory/latest.json
```

Important collector inventory result:

- report verdict: `clear`
- configured collectors: 19
- discovered collector classes: 21
- enabled collectors: 8
- enabled missing classes: none
- RSS: enabled, class found, `pipeline_news_feed`, table `rss_news`
- Google News: enabled, class found, `pipeline_news_feed`
- NewsAPI: enabled, class found, `pipeline_news_feed`, requires `NEWS_API_KEY`
- SEC filings: disabled, class found, `research_specialist_feed`
- research-specialist feeds listed: `bigquery`, `cftc`, `custom_csv`,
  `hugging_face`, `insider`, `sec_filings`

The copied log in `sonar_test_errors.py` is a saved command output draft, not a
Python script to run.

## Current Recommendation

Continue system-first, not source-first:

1. Treat collector work as input triage, not the main mission.
2. Keep building the agent governance loop: model performance, regime, tuning proposals, review actions, and memory.
3. Use `ModelPerformanceAgent` to read local evaluation/backtest metrics before any model promotion or tuning discussion.
4. Use `RegimeAgent` to convert local regime context into a soft pipeline report for consensus.
5. Use `TuningAgent` only as a proposal-only experiment planner; it must never change production config directly.
6. Use `ChiefReviewAgent` as the supervised-autonomy review layer over pipeline state, specialist theses, memory, and operation proposals.
7. Paper autonomy is acceptable only as logged simulation; promotion/config/live execution must stay gated.
8. Use `PaperTradeStore` to record autonomous paper decisions and later grade outcomes by regime/context.
9. Use `PaperPortfolioAgent` to convert logged decisions into paper-only position/exposure/PnL/drawdown history with explicit slippage/commission assumptions.
10. Use `PaperAutonomyRunner` for the daily safe loop; it should report status, not create new decisions by itself.
11. Use `DiaryBridgeAgent` to inspect whether evaluated DEAN paper outcomes can safely influence the pipeline diary.
12. Use `SourceRoutingAgent` to decide whether inputs go to ResearchCorpus, pipeline feeds, macro/context feeds, or source health checks.
13. Use `HistoricalReplayRunner` to test agent reasoning on old local data before trusting new paper-autonomy decisions.
14. Use `ReplayPriceNormalizer` before treating replay hit/miss as evidence; raw cached daily-like rows are not trusted learning truth.
15. Use `HistoricalReplayBatchRunner` to measure whether replay reasoning is repeatable across dates/horizons.
16. Use `PipelineControlSurface` to define the allowed variation area before TuningAgent proposes experiments.
17. Use `HistoricalResearchReplayRunner` when the user asks whether raw news/macro data can produce an analyst view for an old period and be checked after the fact.
18. Use `EvidenceTimestampAudit` before scaling old-period research replay across many dates.
19. Use `HistoricalResearchReplayBatchRunner` to get first calibration statistics across several dates/horizons before changing analyst weights.
20. Use `ReplayPriceQualityInvestigationPlan` when replay batches remain blocked by benchmark or interval warnings.
21. Use `ReplayPriceArtifactRepairPlan` only as a non-destructive candidate artifact builder; never overwrite raw caches.
22. After a repaired/refreshed artifact is clean, scale historical research replay across more dates before calibration.
23. Use `ReplayCalibrationReadinessGate` to decide whether repaired replay evidence can move to manual calibration review.
24. Let source-specific collector fixes happen separately from the DEAN-OS agent-system layer.

## Next Useful Commands For User Logs

Collector inventory:

```powershell
python run_agent_collector_inventory.py --output reports/dean_os/collector_inventory/latest.json
```

Market data freshness:

```powershell
python run_agent_market_freshness.py --latest-processed-prices 1d --tickers AMD NVDA --max-age-hours 24 --include-operation-proposal
```

Model performance:

```powershell
python run_agent_model_performance.py PATH_TO_EVALUATION_JSON_OR_CSV --include-operation-proposal
```

Regime agent:

```powershell
python run_agent_regime.py --latest-processed-prices 1d --ticker AMD
```

Tuning proposal:

```powershell
python run_agent_tuning.py PATH_TO_EVALUATION_JSON_OR_CSV --regime-context-json reports/dean_os/regime_context/amd_latest.json --tickers AMD --timeframes 1d
```

Chief review:

```powershell
python run_agent_chief_review.py --review-snapshot reports/dean_os/review/REVIEW_FILE.json
```

Paper trading:

```powershell
python run_agent_paper_trades.py summary
python run_agent_paper_trades.py evaluate --latest-processed-prices 1d --tickers AMD NVDA
```

Paper portfolio simulation:

```powershell
python run_agent_paper_portfolio.py --latest-processed-prices 1d --tickers AMD NVDA --output reports/dean_os/paper_portfolio/latest.json
```

Paper autonomy loop:

```powershell
python run_agent_paper_autonomy.py --latest-processed-prices 1d --tickers AMD NVDA --max-age-hours 24
```

Share the `decision`, `data_freshness.market_prices`, `regime_context`, `chief_review`, `paper_portfolio.summary`, `diary_bridge`, `journals`, and `recommendations` sections.

Diary bridge:

```powershell
python run_agent_diary_bridge.py --experience-diary logs/experience_diary.csv --paper-store data/dean_os/paper_trades.sqlite --output reports/dean_os/diary_bridge/latest.json
```

Share the `diary_bridge.status`, `diary_bridge.pipeline_diary`, `diary_bridge.paper_records`, `diary_bridge.recommendations`, and `action_proposals` sections.

Historical replay:

```powershell
python run_agent_historical_replay.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --as-of 2026-03-01T00:00:00+00:00 --lookback-days 180 --horizon-days 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --normalize-daily-bars
```

Share the `decision`, `report.thesis`, `historical_replay.coverage.price_quality`, `historical_replay.rankings`, `evaluation`, and `recommendations` sections.

Historical research replay:

```powershell
python run_agent_historical_research_replay.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --as-of 2026-03-01T00:00:00+00:00 --lookback-days 180 --horizon-days 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tags historical_replay ai_cycle raw_period_check published_date_fixed --normalize-daily-bars --output-dir reports\dean_os\historical_research_replay_20260301_filtered
```

Share the `research_exam`, `evidence_pack.coverage`, `agent_lab.summary`,
`price_replay.decision`, `price_replay.evaluation`, `price_replay.quality_warnings`,
and `recommendations` sections.

Evidence timestamp audit:

```powershell
python run_agent_evidence_timestamp_audit.py --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --evidence-pack-json reports\dean_os\historical_research_replay_20260301_filtered\runs\historical_research_replay_20260613T125906_753943+0000\evidence_pack\latest.json --start-at 2025-09-02T00:00:00+00:00 --as-of 2026-03-01T00:00:00+00:00 --output-dir reports\dean_os\evidence_timestamp_audit_20260301_filtered_v2
```

Share the `summary`, `source_audits`, `evidence_pack_audit`, and `recommendations` sections.

Historical research replay batch:

```powershell
python run_agent_historical_research_replay_batch.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --as-of 2026-03-01T00:00:00+00:00 2026-04-01T00:00:00+00:00 --lookback-days 180 --horizon-days 30 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --tags historical_replay ai_cycle published_date_fixed mini_batch --normalize-daily-bars --output-dir reports\dean_os\historical_research_replay_batch_202603_202604
```

Share the `summary`, `learning_gate`, `runs`, `summary.by_research_stance`,
`summary.by_price_ticker`, `summary.quality_warnings`, and `recommendations` sections.

Replay price-quality investigation:

```powershell
python run_agent_replay_price_quality_investigation.py --report-json reports\dean_os\replay_price_normalizer\latest.json --report-json reports\dean_os\historical_replay_batch\latest.json --report-json reports\dean_os\historical_research_replay_batch_202603_202604\latest.json --report-json reports\dean_os\historical_research_replay_20260301_filtered\latest.json --price-data data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --price-data data\dean_os\replay_prices\replay_prices_1d_normalized_20260612_073159.parquet --output-dir reports\dean_os\replay_price_quality_investigation_current_v2
```

Share the `summary`, `warning_summary`, `hypotheses`, `artifact_diagnostics`,
`window_diagnostics`, `operator_tasks`, and `recommendations` sections.

Replay price artifact repair:

```powershell
python run_agent_replay_price_artifact_repair.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --benchmark-ticker SPY --write-artifact --output-dir reports\dean_os\replay_price_artifact_repair_current --artifact-dir data\dean_os\replay_prices
python run_agent_replay_price_quality_investigation.py --artifact-only --price-data data\dean_os\replay_prices\replay_prices_1d_repaired_20260613_135839.parquet --benchmark-ticker SPY --output-dir reports\dean_os\replay_price_quality_investigation_repaired_artifact_only_v2
```

Share the `summary`, `artifact`, `quality`, `quarantine`, `learning_gate`,
`artifact_diagnostics`, and `recommendations` sections.

Interpretation:

- Repair is non-destructive: it writes a new candidate artifact and never edits the raw cache.
- Use `--artifact-only` when verifying the repaired parquet; otherwise older default reports can reintroduce historical warnings into the investigation summary.
- A clean repaired artifact is still diagnostic-only until replay batches have enough clean samples and evidence coverage.

Replay calibration readiness:

```powershell
python run_agent_replay_calibration_readiness.py --replay-batch-json reports\dean_os\historical_replay_batch_repaired_expanded\latest.json --research-batch-json reports\dean_os\historical_research_replay_batch_repaired_expanded_step14\latest.json --output-dir reports\dean_os\replay_calibration_readiness_gate_after_step14_research
```

Share the `summary`, `gate`, `checks`, `commands`, and `recommendations` sections.

Interpretation:

- This is read-only and never writes learning records, calibration proposals, config, pipeline outputs, or broker actions.
- `price_quality` passing means the repaired artifact can be used for diagnostics; it does not authorize calibration.
- `need_evidence_backfill` means replay mechanics are working, but analysts still lack enough pre-`as_of` evidence to judge research skill.

Latest real replay result:

- latest `ReplayPriceArtifactRepairPlan` created `data\dean_os\replay_prices\replay_prices_1d_repaired_20260613_135839.parquet`;
- repair input rows: 8376; candidate rows: 3501; quarantined rows: 4875 across 256 ticker/date pairs;
- quarantine reasons: `same_day_anchor_deviation` 4232, `daily_anchor_preferred` 570, `unanchored_price_level_outlier` 73;
- artifact-only quality investigation on the repaired parquet returned `clear`: 0 warning records, 0 extreme benchmark warnings, max rows per ticker/day 1, largest SPY one-step move about 5.85%;
- historical replay mini-batch on the repaired artifact for `2026-03-01` and `2026-04-01` returned 2 evaluated, `quality_blocked_runs=0`, hit rate `0.5`, clear hit rate `0.5`, average return about `0.316651`, learning gate `insufficient_sample`;
- historical research replay mini-batch on the repaired artifact returned 2 evaluated, `quality_blocked_runs=0`, research stance `mixed` in both, weak evidence in 1 run, hit rate `0.5`, average return about `0.316651`, learning gate `blocked_weak_evidence`;
- expanded historical replay batch on the repaired artifact returned 26 evaluated, `quality_blocked_runs=0`, clear hit rate `0.576923`, average return about `0.03175`, and learning gate `review_required`;
- expanded historical research replay batch with 14-day steps returned 13 evaluated, `quality_blocked_runs=0`, hit rate `0.615385`, average return about `0.036421`, but `weak_evidence_runs=13` and `research_inconclusive_runs=13`;
- latest `ReplayCalibrationReadinessGate` on the expanded batches returned `need_evidence_backfill`, with `price_quality`, `replay_sample`, and `research_sample` passing, but `evidence_coverage` blocked and `research_directionality` caution;
- price-quality is no longer the immediate blocker on the repaired candidate artifact; evidence coverage and neutral/inconclusive research are now the blockers before analyst calibration or learning promotion.
- older raw/normalized replay reports remain diagnostic history only and should not be used as learning truth.
- latest `PipelineControlSurface` run predates the repaired artifact and remains blocked; rerun it only after a larger clean replay batch exists.
- latest `TuningAgent` run with that blocked surface produced `tuning.status=control_surface_blocked` and only `validate -> pipeline_control_surface`; no tune proposal was created.
- latest `AnalystEvidencePackRunner` real smoke on cached news/macro with `--max-rows-per-table 5` produced 10 documents, source types `news` and `report`, quality `partial`, base analyst ready, and manager plan `single_base_then_specialize`.
- Agent Lab can now consume evidence packs via `run_agent_lab.py --evidence-pack-json ...`; smoke produced 10 documents and 4 notes from the cached evidence pack with learning/proposals disabled.
- latest `AnalystProfileOrchestrator` smoke ran `generalist_base_analyst` from the cached evidence pack.
- gating smoke skipped `news_catalyst`, `macro_policy`, and `sector_cycle` without explicit permission, then ran them when `--allow-candidate-profiles` was passed.
- latest `AnalystProfileScorecard` smoke kept base/candidate profiles gated under default thresholds; permissive diagnostic thresholds can mark candidate profiles ready, but that is not a default-promotion signal.
- latest `AnalystLearningPromotionBridge` smoke on `analyst_profiles_real_smoke/latest.json` found 4 candidates and correctly blocked all because the source Agent Lab run was not marked reviewed.
- latest `ReviewApprovedLearningLoop` preview smoke on the same profile run stayed blocked with 4 unreviewed candidates.
- latest isolated `ReviewApprovedLearningLoop --mark-reviewed --apply` smoke wrote 1 review action and 4 pending learning records into `reports\dean_os\review_approved_learning_apply_smoke`, not production stores.
- latest `AnalystOutcomeEvaluationLoop` smoke on that isolated learning store checked 4 pending records and correctly returned `blocked_need_newer_prices` because the normalized replay prices end before the records were created.
- latest `AnalystCalibrationGate` smoke correctly blocked `generalist_base_analyst`, `macro_policy`, `news_catalyst`, and `sector_cycle` because completed outcomes are still zero.
- latest `CalibrationProposalAgent` smoke on the blocked calibration gate returned `no_ready_profiles`, created 0 proposals, and did not enqueue anything.
- latest `CalibrationReviewLifecycle` smoke on the calibration proposal queue returned `no_calibration_proposals`, because no ready profile had been enqueued.
- latest `ManualImplementationBacklog` smoke returned `operation_queue_empty`, because there are no approved calibration proposals awaiting manual implementation.
- latest `AgentLearningLoopRunbook` smoke found all 10 expected artifacts and stopped at `learning_bridge` with status `blocked`; next safe command is `run_agent_analyst_learning_bridge.py` after resolving review blockers.
- latest `AnalystLoopDailyCheck` smoke returned `decision=blocked`, current stage `learning_bridge`, 1 blocker, 4 warnings, and stale market data at about 170 hours.
- latest `AnalystReviewInbox` smoke returned `ready_for_manual_review`, 1 source ready for human review, 0 needs-more-data candidates, and 0 not-reviewable-yet sources.
- latest `ReviewDecisionPacket` smoke on that source returned `manual_review_with_warnings`, recommended `operator_decides`, with 4 passing checks, 2 warnings, and 0 failures.
- latest `ReviewActionDryRun` smoke blocked `mark_reviewed` without warning acknowledgement and allowed `needs_more_data` for the same warning-heavy packet.
- latest `ReviewActionApplyCeremony` smoke blocked no-flag apply, recorded one isolated `needs_more_data` action with `--apply-review-action`, and blocked the duplicate rerun.
- latest `EvidenceGapResolutionPlan` smoke returned `ready_to_collect`, 8 tasks, and missing tickers `AAPL`, `AMD`, `NVDA`, `TSM`.
- refreshed evidence pack with `--max-rows-per-table 200` used existing cached parquet data and produced `strong` quality, 158 documents, all requested tickers covered, and no dropped rows.
- fixed `ReviewDecisionPacket` so `strong` evidence quality is a passing check; refreshed decision packet is now `reviewable`, recommended `mark_reviewed_candidate`, with 5 pass / 0 warn / 0 fail.
- isolated refreshed mark-reviewed apply ceremony plus learning bridge dry-run returned `dry_run_ready`, 4 promotable, 0 blocked, 0 promoted.
- latest `AnalystLearningApplyCeremony` smoke blocked without `--apply-learning`, applied 4 records into an isolated learning store, and blocked duplicate re-apply by note id.
- isolated applied learning records are pending only; no outcomes, calibration changes, config writes, pipeline runs, or broker access occurred.
- latest `OutcomeReadinessGate` smoke on those 4 pending records returned `blocked_need_newer_prices`, with all 4 records in `no_price_after_created_at`.
- The pending records were created on 2026-06-13, while the local stage2 price parquet ends on 2026-05-04; no outcome evaluation or calibration should run yet.
- latest `OutcomePriceCoveragePlan` smoke returned `needs_price_refresh_after_record_creation`.
- It found all 5 requested tickers in the local price file, but AAPL/AMD/MSFT/NVDA/TSM still need prices strictly after `2026-06-13T07:06:38.358225+00:00`.
- The true production outcome horizon remains around `2027-06-13`, so price-after-creation only clears the immediate data impossibility; it does not authorize outcome apply or calibration.
- latest `MarketDataRefreshRunbook` smoke returned `refresh_runbook_ready`.
- It identified `yahoo_finance` as the enabled local price feed candidate, but `can_refresh_automatically=false`; it did not run collectors, network calls, pipeline stages, config writes, outcome writes, or broker actions.
- It recommends creating/providing a separate refreshed CSV/parquet artifact, then running market freshness, outcome readiness, and price coverage checks against that artifact.
- fixed `AnalystEvidencePackRunner` to recognize `published_date`, `publication_date`, `pub_date`, `publishedAt`, and `time_published` as date columns; this prevents future news rows from bypassing `as_of` filtering.
- refreshed `HistoricalResearchReplayRunner` real run for `as_of=2026-03-01` returned `research_stance=mixed`, `research_expected_direction=neutral`, `ticker_specificity=basket_or_sector`.
- After the `published_date` fix, evidence documents dropped from the earlier contaminated 150-document smoke to 5 documents: 2 news and 3 macro; data quality is `partial`, and only AMD had matched ticker evidence.
- The same run attached price replay `candidate_long` on `TSM`; post-hoc 60-day evaluation was `miss` with realized return about `-0.0361`.
- The run is diagnostic only: `research_exam.learning_gate.status=blocked` because the price replay still has the extreme SPY lookback warning.
- latest `EvidenceTimestampAudit` on the filtered evidence pack returned `timestamp_ready`: news primary timestamp is `published_date`, macro primary timestamp is `date`, and future raw rows are correctly treated as rows that must be filtered, not as immediate leakage.
- latest `HistoricalResearchReplayBatchRunner` mini-batch over `2026-03-01` and `2026-04-01` returned 2 evaluated runs, research stance `mixed` in both, 1 weak-evidence run, hit rate `0.5`, average realized return about `-0.030151`, and `learning_gate.status=blocked`.
- In that mini-batch, `2026-03-01` had only 5 evidence documents and missed on TSM over 30 days; `2026-04-01` had 83 evidence documents and hit on TSM over 30 days.
- Both mini-batch runs remained quality-blocked by the persistent extreme SPY lookback warning, so no analyst-weight changes or learning promotion are justified.
- latest `ReplayPriceQualityInvestigationPlan` returned `blocked_price_quality`: 4 reports loaded, 2 price artifacts inspected, 16 warning records, 14 extreme benchmark warnings, and 9 replay-window diagnostics.
- Hypotheses are `interval_mixing_or_daily_label_issue` and `benchmark_window_price_anomaly`.
- Concrete SPY anomaly: around `2026-02-27`, SPY jumps/falls from about `684-687` to `148.91`; around `2026-03-02`, it jumps back from `148.91` to about `684.51`; later artifact diagnostics also show about `152.21 -> 711+`.
- Treat these as non-market-like price-series anomalies. Replay hit/miss, clear_hit_rate, analyst calibration, and learning promotion remain blocked until a refreshed/repaired price artifact clears these windows.

Restore audit note:

- Optimized cleanup audit is saved at `reports/dean_os/restore_audit/latest.json`.
- Cleanup commit `8d94503d9f511d84cbf4999690d36876b1df181a` deleted 17,597 tracked paths relative to restore base `9e89a426`.
- Current `HEAD` restores 17,594 of those paths.
- The 3 still-missing tracked paths are old Markdown files with encoded Cyrillic names, not DEAN-OS code or current agent logic.
- All 51 `run_agent_*.py` wrappers now exist in the root working tree.
- Restored wrappers are thin and safe: no heavy pipeline run, no broker access, no production config writes.
- Verification: all previous 43 `run_agent_*.py --help` checks passed; newer wrapper help checks for outcome coverage, market-data refresh, historical research replay, evidence timestamp audit, historical research replay batch, replay price-quality investigation, replay price artifact repair, and replay calibration readiness passed.
- Verification: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_replay_calibration_readiness_full` -> 105 passed.
- Added `tests/dean_os/test_agent_cli_restore.py` to guard against documented CLI wrappers disappearing again.

Replay price normalization:

```powershell
python run_agent_replay_price_normalizer.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --compare-replay --as-of 2026-03-01T00:00:00+00:00 --lookback-days 180 --horizon-days 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet
```

Share the `artifact`, `quality`, `learning_gate`, `replay_comparison`, and `recommendations` sections.

Historical replay batch:

```powershell
python run_agent_historical_replay_batch.py data\dean_os\replay_prices\replay_prices_1d_normalized_20260612_073159.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --start-as-of 2025-09-01T00:00:00+00:00 --end-as-of 2026-03-01T00:00:00+00:00 --step-days 30 --lookback-days 180 --horizon-days 30 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet
```

Share the `summary`, `learning_gate`, `summary.by_ticker`, `summary.by_horizon`, `summary.quality_warnings`, and `recommendations` sections.

Pipeline control surface:

```powershell
python run_agent_pipeline_control_surface.py --model-performance performance_data.json --replay-batch reports\dean_os\historical_replay_batch\latest.json --data-quality diagnostic_reports\feature_lineage_report.json
```

Share the `surface.status`, `surface.axes`, `surface.allowed_variation`, `proposal_gate`, and `recommendations` sections.

Tuning with surface gate:

```powershell
python run_agent_tuning.py performance_data.json --control-surface-json reports\dean_os\pipeline_control_surface\latest.json --tickers AMD NVDA --timeframes 1d --require-control-surface
```

Share the `tuning.status`, `tuning.control_surface_gate`, `tuning.allowed_variation`, and `action_proposals` sections.

Source routing:

```powershell
python run_agent_source_routing.py docs/research --collector-inventory reports/dean_os/collector_inventory/latest.json --output reports/dean_os/source_routing/latest.json
```

Analyst evidence pack:

```powershell
python run_agent_analyst_evidence_pack.py --materials docs/research --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --source-routing-json reports/dean_os/source_routing/latest.json --tickers AMD NVDA --sectors semiconductor --tags ai_cycle
```

Agent Lab from evidence pack:

```powershell
python run_agent_lab.py --evidence-pack-json reports/dean_os/analyst_evidence_pack/latest.json --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle
```

Analyst profile manager:

```powershell
python run_agent_analyst_profiles.py reports/dean_os/analyst_evidence_pack/latest.json --output-dir reports/dean_os/analyst_profiles
```

Candidate specialist preview:

```powershell
python run_agent_analyst_profiles.py reports/dean_os/analyst_evidence_pack/latest.json --profiles generalist_base_analyst news_catalyst macro_policy sector_cycle value_screening --no-review-snapshot
```

Analyst profile scorecard:

```powershell
python run_agent_analyst_scorecard.py --profile-runs-dir reports/dean_os/analyst_profiles --output-dir reports/dean_os/analyst_profile_scorecard
```

Analyst learning bridge:

```powershell
python run_agent_analyst_learning_bridge.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite
```

Review-approved learning loop:

```powershell
python run_agent_review_approved_learning.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite
python run_agent_review_approved_learning.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite --mark-reviewed --review-notes "Reviewed citations and accepted for pending outcome tracking."
python run_agent_review_approved_learning.py --profile-run-json reports/dean_os/analyst_profiles/latest.json --learning-store data/dean_os/agent_learning.sqlite --review-actions-store data/dean_os/review_actions.sqlite --apply
```

Analyst outcome evaluation loop:

```powershell
python run_agent_analyst_outcome_loop.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --latest-processed-prices 1d
python run_agent_analyst_outcome_loop.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --latest-processed-prices 1d --apply
python run_agent_analyst_outcome_loop.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --market-data-path data/dean_os/replay_prices/replay_prices_1d_normalized_20260612_073159.parquet --historical-diagnostic --as-of 2026-03-01T00:00:00+00:00
```

Outcome readiness and price coverage:

```powershell
python run_agent_outcome_readiness.py --learning-store reports\dean_os\analyst_learning_apply_ceremony_apply_smoke\learning.sqlite --memory-store reports\dean_os\analyst_learning_apply_ceremony_apply_smoke\memory.sqlite --market-data-path data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AAPL AMD MSFT NVDA TSM --output-dir reports\dean_os\outcome_readiness_gate_smoke
python run_agent_outcome_price_coverage.py --readiness-json reports\dean_os\outcome_readiness_gate_smoke\latest.json --output-dir reports\dean_os\outcome_price_coverage_plan_smoke
python run_agent_market_data_refresh_runbook.py --coverage-plan-json reports\dean_os\outcome_price_coverage_plan_smoke\latest.json --collector-inventory-json reports\dean_os\collector_inventory\latest.json --output-dir reports\dean_os\market_data_refresh_runbook_smoke
```

Analyst calibration gate:

```powershell
python run_agent_analyst_calibration_gate.py --profile-scorecard-json reports/dean_os/analyst_profile_scorecard/latest.json --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite
```

Calibration proposals:

```powershell
python run_agent_calibration_proposals.py reports/dean_os/analyst_calibration_gate/latest.json --operations-store data/dean_os/operation_queue.sqlite
python run_agent_calibration_proposals.py reports/dean_os/analyst_calibration_gate/latest.json --operations-store data/dean_os/operation_queue.sqlite --enqueue
```

Calibration review lifecycle:

```powershell
python run_agent_calibration_review_lifecycle.py --operations-store data/dean_os/operation_queue.sqlite
python run_agent_calibration_review_lifecycle.py --operations-store data/dean_os/operation_queue.sqlite --dry-run-proposals
python run_agent_calibration_review_lifecycle.py --operations-store data/dean_os/operation_queue.sqlite --approve PROPOSAL_ID_HERE --dry-run-proposals
python run_agent_calibration_review_lifecycle.py --operations-store data/dean_os/operation_queue.sqlite --reject PROPOSAL_ID_HERE
```

Manual implementation backlog:

```powershell
python run_agent_manual_implementation_backlog.py --operations-store data/dean_os/operation_queue.sqlite
python run_agent_manual_implementation_backlog.py --operations-store data/dean_os/operation_queue.sqlite --include-proposed --include-rejected
```

Agent learning loop runbook:

```powershell
python run_agent_learning_loop_runbook.py
```

Share the `summary.current_stage`, `summary.current_status`, `loop_position.stop_reason`, `loop_position.next_command`, `stages`, and `safety_contract` sections.

Analyst loop daily check:

```powershell
python run_agent_analyst_loop_daily_check.py --tickers AMD NVDA --max-age-hours 24
```

Share the `summary.decision`, `summary.current_stage`, `summary.current_status`, `checks.learning_loop`, `checks.market_freshness`, `blockers`, `warnings`, and `operator_actions` sections.

Analyst review inbox:

```powershell
python run_agent_analyst_review_inbox.py --learning-bridge-json reports/dean_os/analyst_learning_bridge/latest.json --profile-run-json reports/dean_os/analyst_profiles/latest.json
```

Share the `summary.status`, `groups.ready_for_manual_review`, `groups.needs_more_data_candidate`, `groups.not_reviewable_yet`, `items[].suggested_commands`, and `recommendations` sections.

Review decision packet:

```powershell
python run_agent_review_decision_packet.py --inbox-json reports/dean_os/analyst_review_inbox/latest.json
```

Share the `summary.packet_status`, `summary.recommended_review_action`, `source`, `evidence_pack`, `notes`, `review_checks`, `decision_guidance`, and `source.suggested_commands` sections.

Review action dry-run:

```powershell
python run_agent_review_action_dry_run.py --packet-json reports/dean_os/review_decision_packet/latest.json --intent needs_more_data --review-notes "Evidence coverage is too thin." --data-request "Add missing ticker/source coverage before learning promotion."
```

Share the `summary.dry_run_status`, `summary.can_record_review_action`, `validation`, `would_record_review_action`, `commands`, `bridge_expectation`, and `recommendations` sections.

Review snapshot:

```powershell
python run_agent_review.py --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --review-actions-store data/dean_os/review_actions.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --log-path logs/dean_os/events.jsonl
```

Context performance:

```powershell
python run_agent_context_performance.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite
```

Full DEAN-OS tests:

```powershell
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_manual
```

## Suggested Next Module

`HistoricalEvidenceBackfillPlan`

Scope:

- read `ReplayCalibrationReadinessGate` and `HistoricalResearchReplayBatchRunner` outputs;
- list which replay `as_of` dates/tickers have weak evidence or missing requested tickers;
- identify whether cached news/macro/material rows exist for those windows but are not being matched, or whether new historical source artifacts are required;
- propose backfill tasks by source type: news, macro, filings, transcripts, sector reports, or local materials;
- produce rerun commands for evidence pack / historical research replay after backfill;
- write only JSON/Markdown reports; no collector runs, no network calls, no data mutation, no pipeline execution, no learning records, no config writes, and no broker action.

Why this next:

- `ReplayCalibrationReadinessGate` now shows price quality and replay sample size are no longer the main blockers;
- expanded historical research replay has 13 clean price evaluations but `weak_evidence_runs=13` and `research_inconclusive_runs=13`;
- the system should improve historical evidence coverage before judging analyst skill or changing analyst weights.

## Chat Strategy

It is safe to start a new chat after this handoff. In the new chat, ask Codex to
read:

```text
dean_os/NEXT_CHAT_HANDOFF.md
dean_os/IMPLEMENTATION_STATUS.md
dean_os/COMMAND_CHECKLIST.md
reports/dean_os/collector_inventory/latest.json
```

If continuing in the current chat, proceed to `OutcomeEvaluationApplyCeremony` as a blocked-until-ready write gate, or pause to let the user refresh/provide newer price-cache logs first.
