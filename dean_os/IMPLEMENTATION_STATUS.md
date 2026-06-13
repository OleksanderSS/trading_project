# DEAN-OS Implementation Status

Last updated (Codex readable): 2026-06-13

Останнє оновлення: 2026-06-08

Цей файл фіксує фактичний стан реалізації агентського шару, окремо від великої архітектурної ідеї в `Agents_architecture.md`.

## Що вже зроблено

Створено окремий пакет `dean_os/`, який працює як контрольована оболонка поверх існуючого trading pipeline.

Компоненти:
- `schemas.py` - контракти для evidence, reports, market context, consensus decision.
- `base.py` - базовий агент і capabilities.
- `registry.py` - завантаження агентів із YAML.
- `branches.py` - pipeline branch та analytical branch.
- `consensus.py` - deterministic consensus engine.
- `orchestrator.py` - DEANOrchestrator, який спочатку запускає pipeline agents, потім pipeline runner, потім analytical agents.
- `pipeline_adapter.py` - міст до існуючого `HybridOrchestrator`.
- `agent_lab.py` - ізольований runner для research-material ingestion, specialist agents і reviewable reports без запуску pipeline.
- `financial_nlp.py` - FinBERT-ready financial NLP contract із deterministic rule-based fallback.
- `synthesis.py` - evidence-bound synthesis contract для майбутнього GPT-шару.
- `learning.py` - outcome tracking store для `AgentLearningRecord`, hit-rate і suggested agent weight.
- `operation_queue.py` - durable review queue для operations proposals: proposed/approved/rejected/dry-run.
- `event_log.py` - append-only JSONL event log для Agent Lab runs і operation queue actions.
- `review.py` - review snapshot builder для зведення Agent Lab report, learning, operation queue і logs.
- `review_actions.py` - durable review lifecycle store: mark-reviewed, needs-more-data, promote-to-watchlist-proposal.
- `recommendation_memory.py` - case memory для правильних/хибних рекомендацій, context tags, lessons і hit/miss summaries.
- `regime_context.py` - bridge from market-regime outputs or OHLCV CSV into stable DEAN-OS context tags such as `calm_market`, `rising_market`, `crisis`, `volatility_spike`.
- `agents/regime.py` - pipeline soft-agent that converts a `MarketRegimeSnapshot`, local OHLCV frame, or latest processed prices into a consensus-ready `PipelineReport` without running the heavy pipeline.
- `context_performance.py` - агрегує learning records і recommendation memory у hit/miss buckets за `agent`, `context_tag`, `regime_tag`, `agent+context`, `agent+regime`.
- `outcome_evaluation.py` - dry-run/apply evaluator для pending learning records проти локальних price CSV/parquet.
- `agents/market_data_freshness.py` - pipeline-agent для перевірки свіжості локальних market price даних і заповнення `context.metadata["data_freshness"]`.
- `agents/model_performance.py` - pipeline soft-agent that reads local evaluation/backtest metrics and creates review evidence before promotion or tuning.
- `agents/tuning.py` - proposal-only pipeline agent that creates guarded walk-forward tuning experiment proposals without training or production config writes.
- `paper_portfolio.py` - deterministic paper-only portfolio simulator over logged paper decisions, local OHLCV, sizing, slippage, commission, exposure, PnL, and drawdown.
- `agents/paper_portfolio.py` - pipeline soft-agent wrapper that writes `context.metadata["paper_portfolio"]` and returns a consensus-ready `PipelineReport`.
- `paper_autonomy.py` - supervised paper-autonomy runner that combines market freshness, regime, chief review, paper portfolio, diary bridge inspection, DEAN logs, and pipeline experience diary summaries without broker access.
- `agents/diary_bridge.py` - review-only bridge inspector between evaluated DEAN paper outcomes and the existing pipeline experience diary; detects schema compatibility and creates proposals without writing.
- `historical_replay.py` - old-data replay runner, deterministic replay analyst, leakage guard, daily-bar normalization option, and post-thesis outcome evaluation.
- `replay_price_normalizer.py` - data-only normalizer that creates reusable daily OHLCV replay artifacts, records quality warnings, compares raw vs normalized replay when requested, and blocks learning-memory promotion while warnings remain.
- `historical_replay_batch.py` - multi-slice replay evaluator that summarizes hit/miss by ticker, horizon, and price-quality state without writing learning memory.
- `historical_research_replay.py` - unified old-data research exam that builds a pre-`as_of` evidence pack, runs Agent Lab in isolated stores, attaches price replay outcome, and keeps learning/pipeline/broker writes disabled.
- `evidence_timestamp_audit.py` - read-only timestamp gate for cached news/macro/material tables and evidence packs before scaling historical research replay.
- `historical_research_replay_batch.py` - multi-slice research replay evaluator that summarizes research stance, evidence coverage, price outcome, and quality gates across dates/horizons.
- `replay_price_quality_investigation.py` - read-only forensic plan for replay price-quality blockers; inspects replay reports, price artifacts, benchmark windows, large one-step moves, and interval-mixing warnings.
- `replay_price_artifact_repair.py` - non-destructive candidate repair builder for mixed replay price artifacts; prefers midnight daily anchors, quarantines anomalous daily-like rows, writes a new artifact plus audit report, and never mutates source caches.
- `replay_calibration_readiness_gate.py` - read-only gate before analyst calibration; checks repaired price quality, replay sample size, research replay sample size, evidence coverage, and research directionality.
- `pipeline_control_surface.py` - bounded tuning surface builder that intersects profitability, risk, validation, feature stability, data quality, and replay repeatability axes before allowing tuning proposals.
- `analyst_evidence_pack.py` - local-only evidence-pack runner that normalizes materials, cached news, and macro tables into citable `ResearchDocument` payloads for Agent Lab.
- `analyst_profile_orchestrator.py` - central analyst profile manager that reads evidence-pack manager plans, runs the base analyst first, and gates candidate specialist profiles behind explicit approval.
- `analyst_profile_scorecard.py` - activation scorecard that aggregates saved profile runs, skipped reasons, confidence/citation proxies, and promotion blockers.
- `analyst_learning_promotion_bridge.py` - conservative bridge from reviewed analyst notes/profile runs into durable learning records; dry-run by default.
- `review_approved_learning_loop.py` - auditable review ceremony around analyst learning promotion: preview, mark-reviewed/needs-more-data, apply, and context-performance snapshot.
- `analyst_outcome_evaluation_loop.py` - review-friendly outcome evaluator for promoted analyst theses; filters analyst learning records, supports dry-run/apply, and records evaluation audit metadata.
- `analyst_calibration_gate.py` - proposal-only calibration gate that combines profile scorecards, evaluated outcomes, and context performance before any analyst weight/default recommendation.
- `calibration_proposal_agent.py` - proposal-only bridge from calibration gate readiness into `OperationQueue` review items.
- `calibration_review_lifecycle.py` - review-only lifecycle manager for calibration proposals: snapshot, operation dry-run, approve/reject status, and approved-waiting-manual-implementation reporting.
- `manual_implementation_backlog.py` - read-only backlog for approved calibration proposals awaiting separate manual PR/config implementation.
- `agent_learning_loop_runbook.py` - read-only operator runbook that shows the full safe analyst-learning loop position, stop reason, next command, and safety contract.
- `analyst_loop_daily_check.py` - read-only daily operator check that combines the learning-loop runbook, market freshness, evidence coverage, profile scorecard state, and DEAN logs.
- `analyst_review_inbox.py` - read-only inbox for Agent Lab/profile reports that need human review before learning promotion.
- `review_decision_packet.py` - read-only compact packet for deciding whether one inbox source should be marked reviewed or needs more data.
- `review_action_dry_run.py` - read-only preview of the selected review action intent before any review action is recorded.
- `agents/collector_inventory.py` - local-only preflight agent that reads `src/config/collectors.yaml` and collector classes without running network calls; classifies feeds as `pipeline_price_feed`, `pipeline_news_feed`, `pipeline_macro_feed`, `pipeline_context_feed`, or `research_specialist_feed`.
- `sample_materials.py` - deterministic smoke-test corpus для перевірки Agent Lab без реальних матеріалів.
- `research_corpus.py` - локальний SQLite-corpus для книжок, статей, звітів, новинних матеріалів, chunks і research notes.
- `material_loaders.py` - loaders для `.txt`, `.md`, `.html`, `.json`, optional `.pdf`, optional `.docx`.
- `decision_logger.py` - JSONL-лог рішень.
- `execution_gateway.py` - paper/live gate, live execution вимкнений за замовчуванням.
- `factory.py` - фабрики `create_dean_orchestrator()` і `create_hybrid_dean_orchestrator()`.

Підключення до існуючого pipeline:
- `HybridPipelineAdapter(mode="local")` запускає `HybridOrchestrator.run_local_pipeline()`.
- `HybridPipelineAdapter(mode="light")` запускає `HybridOrchestrator.run_light_models()`.
- `HybridPipelineAdapter(mode="prepare")` використовує наявний `PipelineExecutor.execute_prepare_mode()`.
- `HybridPipelineAdapter(mode="full")` використовує наявний `PipelineExecutor.execute_full_mode()`.
- `run_dean_os.py` дає CLI-вхід у DEAN-OS поверх існуючого pipeline.
- `run_research_ingest.py` дає CLI-вхід для ingestion матеріалів у `ResearchCorpus`.
- `run_agent_lab.py` дає CLI-вхід для повного isolated Agent Lab run.
- `run_agent_learning.py` дає CLI-вхід для list/update/score learning records.
- `run_agent_ops.py` дає CLI-вхід для import/list/approve/reject/dry-run operations proposals без запуску pipeline.
- `run_agent_logs.py` дає CLI-вхід для summary/tail structured event logs.
- `run_agent_review.py` дає CLI-вхід для JSON/Markdown review summary після Agent Lab run.
- `run_agent_review_actions.py` дає CLI-вхід для фіксації review-рішень без запуску pipeline.
- `run_agent_memory.py` дає CLI-вхід для add/list/update/summary recommendation memory cases.
- `run_regime_context.py` дає CLI-вхід для безпечного regime-context snapshot із CSV/parquet, latest processed prices, manual regime або існуючого project analyzer bridge.
- `run_agent_regime.py` gives a CLI entry for RegimeAgent as a consensus-style pipeline report without starting the trading pipeline.
- `run_agent_model_performance.py` gives a CLI entry for model evaluation/backtest preflight without training, tuning, or running the pipeline.
- `run_agent_tuning.py` gives a CLI entry for review-only tuning experiment proposals from model metrics and optional regime context.
- `run_agent_chief_review.py` gives a CLI entry for top-level review synthesis from saved review/model/regime/tuning outputs.
- `run_agent_paper_trades.py` gives a CLI entry for recording, listing, summarizing, and evaluating autonomous paper decisions.
- `run_agent_paper_portfolio.py` gives a CLI entry for paper-only portfolio simulation from logged decisions without broker access or store mutation.
- `run_agent_paper_autonomy.py` gives a CLI entry for the safe paper-autonomy loop, diary bridge inspection, and journal summary without creating new paper decisions.
- `run_agent_diary_bridge.py` gives a CLI entry for checking whether evaluated DEAN paper outcomes can safely be mapped into the pipeline experience diary.
- `run_agent_historical_replay.py` gives a CLI entry for safe historical replay with `as_of` cutoff, future/target leakage guard, optional daily-bar normalization, and report-only evaluation.
- `run_agent_replay_price_normalizer.py` gives a CLI entry for creating normalized daily OHLCV replay artifacts before batch replay or learning-memory review.
- `run_agent_historical_replay_batch.py` gives a CLI entry for repeated replay slices across dates/horizons without learning writes or pipeline execution.
- `run_agent_historical_research_replay.py` gives a CLI entry for the combined evidence-pack + Agent Lab + price-outcome old-data exam.
- `run_agent_evidence_timestamp_audit.py` gives a CLI entry for read-only source/evidence-pack timestamp checks before old-data research replay.
- `run_agent_historical_research_replay_batch.py` gives a CLI entry for repeated old-data research replay slices across dates/horizons.
- `run_agent_replay_price_quality_investigation.py` gives a CLI entry for read-only diagnosis of replay price-quality blockers.
- `run_agent_replay_price_artifact_repair.py` gives a CLI entry for creating and auditing non-destructive candidate repaired price artifacts.
- `run_agent_replay_calibration_readiness.py` gives a CLI entry for read-only replay calibration readiness checks.
- `run_agent_pipeline_control_surface.py` gives a CLI entry for building the safe variation area for proposal-only pipeline tuning.
- `run_agent_analyst_evidence_pack.py` gives a CLI entry for creating analyst-ready evidence packs from local materials/news/macro sources.
- `run_agent_lab.py --evidence-pack-json ...` feeds a saved evidence pack directly into Agent Lab.
- `run_agent_analyst_profiles.py` gives a CLI entry for centrally managed analyst profile runs from an evidence pack.
- `run_agent_analyst_scorecard.py` gives a CLI entry for scoring profile readiness before changing analyst defaults.
- `run_agent_analyst_learning_bridge.py` gives a CLI entry for dry-run/apply promotion of reviewed analyst notes into `LearningStore`.
- `run_agent_review_approved_learning.py` gives a CLI entry for the explicit preview -> review action -> apply learning ceremony.
- `run_agent_analyst_outcome_loop.py` gives a CLI entry for evaluating reviewed analyst learning outcomes against local prices.
- `run_agent_analyst_calibration_gate.py` gives a CLI entry for proposal-only analyst calibration guidance.
- `run_agent_calibration_proposals.py` gives a CLI entry for dry-run/enqueued calibration review proposals.
- `run_agent_calibration_review_lifecycle.py` gives a CLI entry for review-only calibration proposal lifecycle management.
- `run_agent_manual_implementation_backlog.py` gives a CLI entry for read-only approved-calibration manual implementation tasks.
- `run_agent_learning_loop_runbook.py` gives a CLI entry for the operator-facing safe analyst-learning loop runbook without executing any stage.
- `run_agent_analyst_loop_daily_check.py` gives a CLI entry for the read-only daily analyst-loop operator check.
- `run_agent_analyst_review_inbox.py` gives a CLI entry for the read-only analyst report review inbox.
- `run_agent_review_decision_packet.py` gives a CLI entry for the read-only per-source review decision packet.
- `run_agent_review_action_dry_run.py` gives a CLI entry for read-only review action intent validation before recording a review action.
- `run_agent_review_action_apply_ceremony.py` gives a CLI entry for the explicit one-action review write gate after a successful dry-run.
- `run_agent_evidence_gap_plan.py` gives a CLI entry for read-only needs-more-data source/task planning.
- `run_agent_learning_apply_ceremony.py` gives a CLI entry for applying pending analyst learning records from a validated bridge dry-run.
- `run_agent_outcome_readiness.py` gives a CLI entry for read-only analyst outcome maturity and price-coverage checks.
- `run_agent_outcome_price_coverage.py` gives a CLI entry for read-only price coverage planning after an outcome-readiness blocker.
- `run_agent_market_data_refresh_runbook.py` gives a CLI entry for read-only market-data refresh runbooks after price coverage blockers.
- `run_agent_context_performance.py` дає CLI-вхід для weak/strong context report по агентам.
- `run_agent_outcome_evaluation.py` дає CLI-вхід для безпечного outcome evaluation; за замовчуванням нічого не оновлює.
- `run_agent_market_freshness.py` дає CLI-вхід для freshness preflight без запуску trading pipeline.
- `run_agent_collector_inventory.py` gives a safe collector map before isolated collector health tests; it does not import collector modules, instantiate clients, call APIs, or run the pipeline.

Adapter після запуску намагається збагачувати `MarketContext`:
- `features_df` -> `context.dataframes["features"]`
- `targets_df` -> `context.dataframes["targets"]`
- `news_data` -> `context.dataframes["news"]` і `context.news`
- `macro_data` / `economic_data` -> `context.dataframes["macro"]` і `context.macro`
- `market_data` / `features_df.close` -> `context.returns`, якщо можна вивести returns

## Які агенти вже є

Pipeline agents:
- `PipelineAuditAgent` - читає `audit_reports/findings.json`; `P0` блокує, `P1` дає caution.
- `DataQualityAgent` - перевіряє порожні DataFrame, missing ratio, synthetic markers.
- `RiskAgent` - перевіряє gross exposure, max drawdown, daily VaR95.
- `ModelPerformanceAgent` - soft gate for supplied model evaluation/backtest metrics.
- `RegimeAgent` - soft regime context report for consensus.
- `TuningAgent` - proposal-only walk-forward tuning experiment planner.
- `ChiefReviewAgent` - supervised-autonomy review synthesizer over pipeline state, specialist notes, memory, and operation proposals.
- `PaperTradeStore` / `PaperTradeEvaluationRunner` - autonomous paper decision log and outcome evaluator.
- `PaperPortfolioAgent` - paper-only portfolio simulation over logged decisions, with explicit sizing and cost assumptions.
- `PaperAutonomyRunner` - safe paper loop report over freshness, regime, review, portfolio, diary bridge, DEAN logs, and pipeline experience diary.
- `DiaryBridgeAgent` - review-only inspector for paper-outcome-to-diary compatibility; never writes to pipeline memory automatically.
- `HistoricalReplayRunner` / `HistoricalReplayAnalyst` - safe old-data reasoning exam; forms a thesis from pre-`as_of` data and evaluates only afterward.
- `ReplayPriceNormalizer` - data-only normalized price artifact builder for replay; no paper trades, no learning writes, no heavy pipeline.
- `HistoricalReplayBatchRunner` - batch replay exam over normalized artifacts; summarizes repeatability while keeping learning promotion blocked by quality gates.
- `HistoricalResearchReplayRunner` - combined research evidence replay; uses raw/cached news and macro before `as_of`, Agent Lab analysis, and post-hoc price outcome in one report.
- `PipelineControlSurface` - control surface for bounded tuning proposals; never changes production config.
- `SourceRoutingAgent` - local source/material routing map for pipeline feeds and specialist-agent intake.

Domain / analytical MVP agents:
- `MacroPolicyAgent`
- `GeoPoliticalAgent`
- `NewsCatalystAgent`
- `SectorCycleAgent`
- `IndustryMapAgent`
- `HistoricalAnalogiesAgent`
- `ValueScreeningAgent`
- `ContrarianThesisAgent`
- `ResearchIngestionAgent`
- `FinancialNLPAgent`
- `SpecialistResearchAgent`
- `EvidenceSynthesisAgent`

Operations / automation agents:
- `OperationsProposalAgent`

Важливо: domain agents зараз навмисно прості. Вони не є повноцінними “економістами” або “Баффетами”. На цьому етапі вони:
- працюють без мережі;
- не використовують LLM;
- не вигадують факти;
- читають тільки переданий `MarketContext`;
- формують `AnalyticalReport`, а не trade signal.

Це правильний стартовий стан: спочатку контракти, gates, logging, tests; потім інтелект і навчання.

## Research Lab

Додано перший шар для “агент вчиться як спеціаліст”.

Нові артефакти:
- `ResearchDocument` - новина, стаття, книга, звіт, filing, transcript.
- `ResearchChunk` - фрагмент документа з citation.
- `SourceCitation` - посилання на джерело / chunk / excerpt.
- `ResearchNote` - структурована нотатка агента: thesis, patterns, catalysts, risks, blind spots.
- `AgentLearningRecord` - майбутній запис для перевірки тези проти outcome.
- `PipelineActionProposal` - пропозиція операційної дії без автоматичного запуску.

Поточна логіка:
- `material_loaders.py` читає локальні файли/папки та створює `ResearchDocument`.
- `AgentLabRunner` бере папку матеріалів, інжестить corpus, запускає `ResearchIngestionAgent`, `SpecialistResearchAgent`, опційно `OperationsProposalAgent`, і пише JSON/Markdown-звіт.
- `AgentLabRunner` також може створювати pending `AgentLearningRecord` для кожної якісної `ResearchNote`.
- `FinancialNLPAgent` аналізує tone, risk tone, event types і key terms. За замовчуванням працює rule-based; локальний FinBERT можна підключити через `finbert_model`, без завантаження з мережі.
- `ResearchIngestionAgent` бере `context.research_documents` і `context.news`, ріже матеріали на chunks, кладе в `ResearchCorpus`, створює `ResearchNote`.
- `SpecialistResearchAgent` читає research materials, news, fundamentals, macro і шукає патерни: `defense_rearmament`, `ai_compute_cycle`, `energy_security`, `policy_easing`, `supply_chain_reshoring`, `value_margin_safety`, `pricing_power`, `regulatory_risk`, `balance_sheet_stress`, `capacity_pressure`.
- `EvidenceSynthesisAgent` робить фінальну cited thesis тільки з `ResearchNote`, `FinancialNLPResult` і citations. Це місце для майбутнього GPT synthesis.
- `OperationsProposalAgent` лише пропонує дії: parse, enrich, accumulate, validate. Він нічого не запускає сам.
- `OperationQueue` зберігає ці пропозиції окремо від звіту, щоб ми могли переглядати, approve/reject і робити dry-run перед будь-якою автоматизацією.
- `EventLog` пише події: run started, materials loaded, agents finished, learning records created, proposals queued, run completed, proposal saved/status changed/dry-run previewed.
- `run_agent_lab.py --sample` запускає smoke-test loop без `docs/research`, щоб перевірити механіку агентів до появи реального research corpus.
- `AgentReviewBuilder` збирає latest Agent Lab report, pending learning records, operation proposals і останні log events в один review snapshot з next actions.
- `ReviewActionStore` зберігає review-рішення і може створити watchlist proposal у `OperationQueue`, але не trade і не pipeline execution.
- Для sample-run зафіксовано `mark_reviewed` і `needs_more_data`; sample-тези не промотяться у watchlist.
- `ReviewActionStore` тепер блокує placeholder source ids на кшталт `RUN_ID_HERE`; помилкові записи треба `void-action`, а не видаляти.
- `RecommendationMemoryStore` зберігає історичні кейси: теза, контекстні теги, expected direction, outcome, lesson, hit/miss по тегах.
- Agent Lab now injects relevant recommendation memory into `MarketContext.metadata`, and specialist/synthesis agents add memory warnings to evidence, risks, and blind spots.
- Agent Lab now accepts `--regime-tags` and `--regime-context-json`; these are merged with theme/event tags for memory lookup without starting the trading pipeline.
- `run_regime_context.py` can produce a saved context JSON from OHLCV CSV/parquet or `data/processed/prices_*`; Agent Lab can consume it to make agents regime-aware.
- `AgentPerformanceByContext` combines completed/pending learning records with manual recommendation memory and surfaces `weak_contexts`, `strengths`, `recent_miss_lessons`, and review recommendations.
- `AgentReviewBuilder` now includes context performance in review snapshots, so weak regimes can become review guardrails.
- `OutcomeEvaluationRunner` checks whether pending learning records can be evaluated from local prices; it reports `not_due`, `no_price_after_created_at`, `missing_price_window`, `evaluable`, or `updated`.
- `MarketDataFreshnessAgent` checks latest local market prices by ticker and age; stale results feed `OperationsProposalAgent` as refresh proposals.
- `CollectorInventoryAgent` maps configured collectors to discovered local classes without network calls, separates pipeline news/macro/context feeds from research-specialist feeds, and surfaces enabled class gaps before isolated health tests.
- `COMMAND_CHECKLIST.md` contains the safe command set for logs, review, learning, operation queue, review actions, and recommendation memory.

Це ще не повний FinBERT + GPT. Це rule-based skeleton + FinBERT-ready interface, який створює правильні структури для наступного шару:
- Локальний FinBERT / financial NLP для tone, event sentiment, risk tone.
- GPT / LLM для synthesis, аналогій, пояснень, секторних playbooks.
- GPT має працювати через `EvidenceSynthesisAgent`: input = citations/evidence package, output = cited thesis без claims поза джерелами.
- outcome tracking для того, щоб агент не просто писав тези, а перевіряв їх на майбутніх результатах.

Приклад ingestion:

```text
python run_research_ingest.py docs/research --corpus data/dean_os/research_corpus.sqlite --source-type report --tickers AMD NVDA --sectors semiconductor --tags ai_cycle
```

Приклад Agent Lab run:

```text
python run_agent_lab.py docs/research --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle
```

Smoke-test без реальних матеріалів:

```text
python run_agent_lab.py --sample --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle
```

Прапорці:
- `--no-financial-nlp` вимикає NLP-шар.
- `--no-operations-proposals` вимикає operations proposal report.
- `--no-learning-records` вимикає створення pending learning records.

Learning CLI:

```text
python run_agent_learning.py --store data/dean_os/agent_learning.sqlite list
python run_agent_learning.py --store data/dean_os/agent_learning.sqlite update RECORD_ID_HERE --realized-return 0.08
python run_agent_learning.py --store data/dean_os/agent_learning.sqlite score evidence_synthesis
```

Operations proposal CLI:

```text
python run_agent_ops.py --store data/dean_os/operation_queue.sqlite import-report reports/dean_os/agent_lab/RUN_ID.json
python run_agent_ops.py --store data/dean_os/operation_queue.sqlite list
python run_agent_ops.py --store data/dean_os/operation_queue.sqlite approve PROPOSAL_ID_HERE
python run_agent_ops.py --store data/dean_os/operation_queue.sqlite dry-run PROPOSAL_ID_HERE
```

Logs CLI:

```text
python run_agent_logs.py summary
python run_agent_logs.py tail --limit 5
```

Review CLI:

```text
python run_agent_review.py --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --log-path logs/dean_os/events.jsonl
```

Review lifecycle CLI:

```text
python run_agent_review_actions.py list
python run_agent_review_actions.py mark-reviewed --source-type agent_lab_report --source-id ACTUAL_RUN_ID_FROM_REVIEW --notes "Reviewed"
python run_agent_review_actions.py needs-more-data --source-type agent_lab_report --source-id ACTUAL_RUN_ID_FROM_REVIEW --data-request "Add filings and transcripts"
python run_agent_review_actions.py promote-watchlist --source-type agent_lab_report --source-id ACTUAL_RUN_ID_FROM_REVIEW --tickers AMD NVDA --thesis "Evidence-bound thesis" --reason "Review-approved research direction"
python run_agent_review_actions.py void-action ACTION_ID_HERE --reason "Mistyped placeholder id"
```

Важливо: `promote-watchlist` створює лише `PipelineActionProposal(action_type="report")`; це не trade signal і не запуск pipeline.

Recommendation memory CLI:

```text
python run_agent_memory.py add-case --source-id fuel-crisis-case --agent-name macro_policy --topic "fuel crisis" --thesis "Fuel stress would be short-lived" --expected-direction neutral --outcome-label miss --context-tags fuel_crisis energy_shock --lesson "Fuel shock persistence was underestimated"
python run_agent_memory.py list --context-tag fuel_crisis
python run_agent_memory.py summary
```

Regime context CLI:

```text
python run_regime_context.py --latest-processed-prices 1d --ticker AMD --output reports/dean_os/regime_context/amd_latest.json
python run_agent_lab.py docs/research --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle --regime-context-json reports/dean_os/regime_context/amd_latest.json
```

Context performance CLI:

```text
python run_agent_context_performance.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite
python run_agent_context_performance.py --learning-store data/dean_os/agent_learning.sqlite --memory-store data/dean_os/recommendation_memory.sqlite --context-tag crisis
```

Outcome evaluation CLI:

```text
python run_agent_outcome_evaluation.py --learning-store data/dean_os/agent_learning.sqlite --latest-processed-prices 1d
python run_agent_outcome_evaluation.py --learning-store data/dean_os/agent_learning.sqlite --latest-processed-prices 1d --limit 5
```

Поточний dry-run показує `no_price_after_created_at`, бо latest local price data ends before current Agent Lab records were created. Це правильний захист від фальшивої оцінки.

Market data freshness CLI:

```text
python run_agent_market_freshness.py --latest-processed-prices 1d --tickers AMD NVDA --max-age-hours 24
python run_agent_market_freshness.py --latest-processed-prices 1d --tickers AMD NVDA --max-age-hours 24 --include-operation-proposal
```

Поточний run показує stale local market prices і створює proposal `accumulate -> market_prices`; це не запускає pipeline і не оновлює дані автоматично.

Collector inventory CLI:

```text
python run_agent_collector_inventory.py --output reports/dean_os/collector_inventory/latest.json
```

This reads only local config/source files. RSS/Google News/NewsAPI are treated as pipeline news feeds because their `data_type: news` outputs can be aligned to candles and sentiment/event studies. SEC filings and similar slow evidence sources should first feed ResearchCorpus/Agent Lab, not daily model input.

Diary bridge CLI:

```text
python run_agent_diary_bridge.py --experience-diary logs/experience_diary.csv --paper-store data/dean_os/paper_trades.sqlite --output reports/dean_os/diary_bridge/latest.json
```

Поточний run показує `schema_mismatch`: `logs/experience_diary.csv` має schema kind `modeling_champion`, а не trade/outcome diary. Це правильний guardrail: DEAN paper outcomes не мають записуватись у цей CSV, доки не визначено цільовий contract.

Historical replay CLI:

```text
python run_agent_historical_replay.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --as-of 2026-03-01T00:00:00+00:00 --lookback-days 180 --horizon-days 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet --normalize-daily-bars
```

Поточний normalized replay вибрав `TSM` як `candidate_long`, але post-thesis evaluation дала `miss` на 60d. Це не paper trade і не learning truth: report також показує `price_quality.warnings`, включно з extreme SPY lookback return, тому наступний правильний крок - стабілізувати normalized replay price artifact.

Replay price normalizer CLI:

```text
python run_agent_replay_price_normalizer.py data\colab\backup_20260510_153551\stage2_prices_1d_20260505_151233.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --compare-replay --as-of 2026-03-01T00:00:00+00:00 --lookback-days 180 --horizon-days 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet
```

Use the saved normalized artifact for future batch replay. If `learning_gate.status` is `blocked`, replay hit/miss remains diagnostic and must not be written to learning memory.

Latest real normalizer run:
- artifact: `data\dean_os\replay_prices\replay_prices_1d_normalized_20260612_073159.parquet`;
- normalized artifact rows: 3506 across 7 tickers;
- global normalized price warnings: 0;
- replay comparison still has an extreme SPY window warning;
- `learning_gate.status`: `blocked`.

Historical replay batch CLI:

```text
python run_agent_historical_replay_batch.py data\dean_os\replay_prices\replay_prices_1d_normalized_20260612_073159.parquet --tickers AMD NVDA MSFT AAPL TSM QQQ SPY --start-as-of 2025-09-01T00:00:00+00:00 --end-as-of 2026-03-01T00:00:00+00:00 --step-days 30 --lookback-days 180 --horizon-days 30 60 --news-data data\colab\backup_20260510_153551\stage2_news_20260505_151233.parquet --macro-data data\colab\backup_20260510_153551\stage2_macro_20260507_191104.parquet
```

Latest real batch run:
- total runs: 14;
- evaluated runs: 14;
- quality-blocked runs: 2;
- hit rate: 0.642857;
- clear hit rate: 0.75;
- clear average realized return: 0.117827;
- `learning_gate.status`: `blocked`;
- main warning: extreme SPY lookback return in two replay slices.

Pipeline control surface CLI:

```text
python run_agent_pipeline_control_surface.py --model-performance performance_data.json --replay-batch reports\dean_os\historical_replay_batch\latest.json --data-quality diagnostic_reports\feature_lineage_report.json
```

Latest real control-surface run:
- `surface.status`: `blocked`;
- `proposal_gate.status`: `blocked`;
- profitability axis: `clear` via replay proxy return;
- risk, validation, feature stability axes: `caution` because required metrics are missing;
- data-quality axis: `blocked` because feature lineage contains `TARGET_*` leakage-looking columns;
- replay axis: `blocked` because 2 replay windows remain quality-blocked;
- allowed tuning trials: 0.

TuningAgent surface gate:
- `TuningAgent` now reads `context.metadata["pipeline_control_surface"]`;
- if `proposal_gate.can_propose_tuning=false`, it blocks tuning proposals and creates a validation proposal for `pipeline_control_surface`;
- if the surface is caution/clear, it includes `allowed_variation` bounds in the tuning command preview;
- latest real tuning run with the blocked surface produced `tuning.status=control_surface_blocked` and only `validate -> pipeline_control_surface`.

Restore audit note:
- optimized cleanup audit saved to `reports/dean_os/restore_audit/latest.json`;
- cleanup commit `8d94503d9f511d84cbf4999690d36876b1df181a` deleted 17,597 tracked paths relative to restore base `9e89a426`;
- current `HEAD` restores 17,594 of those paths;
- the 3 still-missing tracked paths are old Markdown files with encoded Cyrillic names, not DEAN-OS code or current agent logic;
- all 43 `run_agent_*.py` wrappers now exist in the root working tree;
- restored CLI wrappers remain thin and safe: no heavy pipeline run, no broker access, no production config writes;
- verification: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_restore_cli_final` -> 12 passed;
- verification: all 43 `run_agent_*.py --help` checks passed;
- smoke: `run_agent_model_performance.py`, `run_agent_collector_inventory.py`, `run_agent_review.py`, and `run_agent_chief_review.py` ran successfully.

AnalystEvidencePackRunner:
- `analyst_evidence_pack.py` builds local-only evidence packs from materials, cached news, cached macro tables, and optional source-routing output;
- output includes full normalized `ResearchDocument` payloads, `coverage`, `warnings`, `dropped`, `recommendations`, and `analyst_inputs`;
- `analyst_inputs.manager_plan` starts with one active `generalist_base_analyst` and lists candidate specialist profiles such as `news_catalyst`, `macro_policy`, and `sector_cycle`;
- `run_agent_lab.py --evidence-pack-json ...` can consume the saved pack directly;
- latest real smoke on cached news/macro with `--max-rows-per-table 5` produced 10 documents, source types `news` and `report`, quality `partial`, and base analyst ready;
- end-to-end smoke fed that pack into Agent Lab and produced 10 documents, 4 notes, and 0 proposals with learning/proposals disabled;
- verification: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_manual_backlog_full` -> 44 passed.

AnalystProfileOrchestrator:
- reads `analyst_inputs.manager_plan` from an evidence pack;
- default run executes only `generalist_base_analyst` through Agent Lab;
- candidate profiles such as `news_catalyst`, `macro_policy`, and `sector_cycle` are skipped unless `--allow-candidate-profiles` is explicit;
- unsupported or evidence-blocked profiles, such as `value_screening` without filings/fundamentals, remain skipped with reasons;
- can optionally build a linked review snapshot after the base Agent Lab run;
- latest real smoke on cached evidence pack ran `generalist_base_analyst` successfully;
- gating smoke skipped candidate profiles without permission and ran `macro_policy`, `news_catalyst`, and `sector_cycle` when explicitly allowed;
- verification: `python -m pytest tests\dean_os\test_analyst_profile_orchestrator.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_analyst_profiles` -> 3 passed.

AnalystProfileScorecard:
- reads saved `analyst_profile_orchestrator` outputs;
- aggregates completed/skipped counts, confidence proxy, citation proxy, note counts, verdict counts, skipped reasons, and activation blockers;
- default thresholds keep profiles as candidates until there are enough completed, cited runs;
- real smoke on `analyst_profiles_gate_smoke` kept `generalist_base_analyst` as candidate and blocked skipped specialists;
- permissive diagnostic smoke on candidate profiles marked `macro_policy`, `news_catalyst`, and `sector_cycle` ready only because thresholds were intentionally lowered;
- verification: `python -m pytest tests\dean_os\test_analyst_profile_scorecard.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_analyst_scorecard` -> 2 passed.

AnalystLearningPromotionBridge:
- reads `AnalystProfileOrchestrator` output or direct Agent Lab report JSON;
- requires a non-voided `mark_reviewed` action for the source Agent Lab report by default;
- blocks weak notes and duplicate note IDs by default;
- dry-run is default; `--apply` is required to write `AgentLearningRecord` rows;
- promoted records include evidence-pack/profile/profile-run/review-action metadata for audit;
- real smoke on `reports\dean_os\analyst_profiles_real_smoke\latest.json` found 4 candidates and correctly blocked all because the source was not reviewed;
- verification: `python -m pytest tests\dean_os\test_analyst_learning_promotion_bridge.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_learning_bridge` -> 3 passed.

ReviewApprovedLearningLoop:
- wraps `AnalystLearningPromotionBridge` with a preview -> review action -> apply ceremony;
- can record explicit `mark_reviewed` actions only with review notes;
- can record `needs_more_data` actions that keep promotion blocked;
- final apply still goes through bridge gates, so unreviewed/weak/duplicate notes remain blocked;
- includes a context-performance snapshot after promotion so pending records are visible immediately without changing weights;
- isolated preview smoke on `reports\dean_os\analyst_profiles_real_smoke\latest.json` stayed blocked with 4 unreviewed candidates;
- isolated review/apply smoke wrote 1 review action and 4 pending learning records into `reports\dean_os\review_approved_learning_apply_smoke`, not the production `data/dean_os` stores;
- verification: `python -m pytest tests\dean_os\test_review_approved_learning_loop.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_review_learning_loop` -> 4 passed;
- full verification after outcome loop: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_analyst_outcome_full` -> 30 passed;
- CLI verification after outcome loop: all 30 `run_agent_*.py --help` checks passed.

AnalystOutcomeEvaluationLoop:
- evaluates only promoted analyst learning records by default via metadata flag `analyst_learning_bridge=True`;
- wraps `OutcomeEvaluationRunner` with dry-run/apply reporting, profile outcome summaries, and context-performance refresh;
- writes evaluation audit metadata back into updated learning records after apply;
- blocks historical diagnostic apply by default, because early/old-data checks are mechanics tests rather than production learning truth;
- real smoke on the isolated review/apply learning store checked 4 pending analyst records and correctly returned `blocked_need_newer_prices` because the normalized replay prices end before the records were created;
- verification: `python -m pytest tests\dean_os\test_analyst_outcome_evaluation_loop.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_analyst_outcome_loop` -> 4 passed;
- full verification before calibration gate: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_analyst_outcome_full` -> 30 passed.

AnalystCalibrationGate:
- combines `AnalystProfileScorecard`, analyst learning outcomes, and `AgentPerformanceByContext`;
- returns per-profile `blocked`, `keep_candidate`, `ready_with_caution`, or `ready_for_review`;
- suggests only small reviewable weight deltas and never writes production config;
- requires scorecard readiness, minimum completed profile runs, minimum completed outcomes, minimum hit rate, and bounded miss rate;
- smoke on the isolated review/apply learning store correctly blocked `generalist_base_analyst`, `macro_policy`, `news_catalyst`, and `sector_cycle` because completed outcomes are still zero;
- verification: `python -m pytest tests\dean_os\test_analyst_calibration_gate.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_analyst_calibration_gate` -> 3 passed;
- full verification before calibration proposals: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_analyst_calibration_full` -> 33 passed.

CalibrationProposalAgent:
- reads `AnalystCalibrationGate` reports;
- creates `PipelineActionProposal` review items only for `ready_for_review` profiles by default;
- dry-run is default and writes only report artifacts;
- `--enqueue` writes proposals to `OperationQueue` as `proposed`, `dry_run=True`, `requires_human_approval=True`;
- proposals are `action_type=report` and include profile, suggested weight delta, completed outcomes, hit rate, scorecard activation status, risks, and command preview;
- smoke on blocked calibration gate correctly returned `no_ready_profiles` and created zero proposals;
- verification: `python -m pytest tests\dean_os\test_calibration_proposal_agent.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_calibration_proposals` -> 3 passed;
- full verification before review lifecycle: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_calibration_proposals_full` -> 36 passed.

CalibrationReviewLifecycle:
- reads calibration proposals from `OperationQueue`;
- can run operation dry-run previews without changing proposal status or config;
- can explicitly approve/reject calibration proposal IDs in the queue;
- approval means `approved_waiting_manual_implementation`, not config mutation;
- skips non-calibration proposals unless `--include-non-calibration` is explicit;
- smoke on the current calibration proposal queue returned `no_calibration_proposals`, because the previous calibration gate was blocked and proposal agent enqueued nothing;
- verification: `python -m pytest tests\dean_os\test_calibration_review_lifecycle.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_calibration_review_lifecycle` -> 4 passed;
- full verification before manual backlog: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_calibration_review_full` -> 40 passed.

ManualImplementationBacklog:
- reads approved calibration proposals from `OperationQueue`;
- creates manual implementation tasks with target profile, suggested delta, risks, evidence, and checklist;
- read-only: no config writes, no code edits, no queue status changes, no consensus/weight/default mutation;
- approved proposals become `waiting_manual_implementation`;
- proposed/rejected items are hidden by default but can be included for visibility;
- smoke on the isolated calibration proposal queue returned `operation_queue_empty`, because no ready/approved calibration proposals exist yet;
- verification: `python -m pytest tests\dean_os\test_manual_implementation_backlog.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_manual_implementation_backlog` -> 4 passed;
- full verification: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_manual_backlog_full` -> 44 passed;
- pre-runbook CLI verification: 34 `run_agent_*.py --help` checks passed.

AgentLearningLoopRunbook:
- reads the expected artifacts from evidence pack through manual implementation backlog;
- reports the current loop position, current status, stop reason, next safe command, and shareable sections;
- read-only: no pipeline run, no broker access, no production config writes, and no stage execution;
- `manual_implementation_required` is treated as a separate PR/config-change boundary, not an auto-apply signal;
- smoke with the existing saved artifacts found all 10 artifacts and stopped at `learning_bridge` with status `blocked`;
- verification: `python -m pytest tests\dean_os\test_agent_learning_loop_runbook.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_learning_loop_runbook_retry` -> 3 passed;
- full verification after market data refresh runbook: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_market_data_refresh_runbook_full` -> 85 passed;
- CLI wrapper count after market data refresh runbook: 45 `run_agent_*.py` wrappers; new `run_agent_market_data_refresh_runbook.py --help` passed. Previous wrappers remain thin/safe by contract.
- current verification after evidence timestamp audit and historical research replay batch: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_historical_research_replay_batch_full` -> 94 passed;
- current CLI wrapper count: 48 `run_agent_*.py` wrappers, including historical research replay, evidence timestamp audit, and historical research replay batch.
- current verification after replay price-quality investigation: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_replay_price_quality_investigation_full` -> 96 passed;
- current CLI wrapper count after replay price artifact repair: 50 `run_agent_*.py` wrappers.
- current verification after replay price artifact repair: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_replay_price_artifact_repair_full_v3` -> 100 passed;
- current repaired artifact: `data\dean_os\replay_prices\replay_prices_1d_repaired_20260613_135839.parquet`; artifact-only investigation returned `clear`, historical replay mini-batch had `quality_blocked_runs=0`, and historical research replay mini-batch had `quality_blocked_runs=0`.
- current CLI wrapper count after replay calibration readiness: 51 `run_agent_*.py` wrappers.
- current verification after replay calibration readiness: `python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_replay_calibration_readiness_full` -> 105 passed;
- expanded repaired historical replay batch: 26 evaluated, `quality_blocked_runs=0`, clear hit rate `0.576923`, average return about `0.03175`, learning gate `review_required`.
- expanded repaired historical research replay batch: 13 evaluated, `quality_blocked_runs=0`, hit rate `0.615385`, average return about `0.036421`, but `weak_evidence_runs=13` and `research_inconclusive_runs=13`.
- current `ReplayCalibrationReadinessGate` status: `need_evidence_backfill`; price quality, replay sample, and research sample pass, but evidence coverage blocks and research directionality is caution.
- replay price-quality investigation found 16 warning records, including 14 extreme benchmark warnings; SPY has non-market-like jumps around 2026-02-27 / 2026-03-02 and later 2026-04-28 / 2026-04-29.

AnalystLoopDailyCheck:
- wraps `AgentLearningLoopRunbook` in a cheap daily operator check;
- reads local market freshness, evidence-pack coverage, profile scorecard state, and recent DEAN event logs;
- returns `blocked`, `needs_operator_review`, or `safe_to_continue` plus blockers, warnings, and operator actions;
- read-only: no analyst/profile run, no outcome evaluation, no proposal enqueue, no config write, no heavy pipeline run, no broker access;
- smoke on saved analyst artifacts returned `blocked` at `learning_bridge` and also warned that market prices were stale at about 170 hours;
- verification: `python -m pytest tests\dean_os\test_analyst_loop_daily_check.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_analyst_loop_daily_check` -> 3 passed.

AnalystReviewInbox:
- reads a learning-bridge report or falls back to a profile-run report;
- reads review actions in SQLite read-only mode when the store already exists;
- groups sources into `ready_for_manual_review`, `needs_more_data_candidate`, and `not_reviewable_yet`;
- extracts source id, profile, evidence-pack path, report path, note/candidate counts, blockers, and suggested review command previews;
- read-only: no review action writes, no learning writes, no proposal enqueue, no config write, no pipeline run, no broker access;
- smoke on the saved unreviewed learning-bridge artifact returned 1 source in `ready_for_manual_review`;
- verification: `python -m pytest tests\dean_os\test_analyst_review_inbox.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_analyst_review_inbox` -> 3 passed.

ReviewDecisionPacket:
- reads one source from `AnalystReviewInbox`;
- loads the selected Agent Lab report and optional evidence pack;
- summarizes report agents, research notes, citations, risks, blind spots, evidence coverage, and command previews;
- returns `reviewable`, `manual_review_with_warnings`, or `needs_more_data_recommended`;
- read-only: no review action writes, no learning writes, no proposal enqueue, no config write, no pipeline run, no broker access;
- smoke on the saved review inbox returned `manual_review_with_warnings`, recommended `operator_decides`, with 4 passing checks, 2 warnings, and 0 failures;
- verification: `python -m pytest tests\dean_os\test_review_decision_packet.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_review_decision_packet` -> 3 passed.

ReviewActionDryRun:
- reads a `ReviewDecisionPacket` and an explicit operator intent: `mark_reviewed` or `needs_more_data`;
- validates whether the packet status supports the intent;
- blocks `mark_reviewed` on warning-heavy packets unless `acknowledge_warnings` is explicit;
- previews the review action payload, the real review command, and the next learning-bridge dry-run command;
- read-only: no review action writes, no learning writes, no proposal enqueue, no config write, no pipeline run, no broker access;
- smoke on the warning-heavy packet blocked `mark_reviewed` without acknowledgement and allowed `needs_more_data`;
- verification: `python -m pytest tests\dean_os\test_review_action_dry_run.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_review_action_dry_run` -> 4 passed.

ReviewActionApplyCeremony:
- reads a `ReviewActionDryRun` artifact and an explicit `apply_review_action` flag;
- records exactly one `mark_reviewed` or `needs_more_data` action through `ReviewActionStore` only when the dry-run is recordable;
- blocks no-flag runs, non-recordable dry-runs, duplicate active actions, and `mark_reviewed` while active `needs_more_data` is unresolved;
- never writes learning records, enqueues proposals, changes config, runs the pipeline, or accesses a broker;
- smoke on the warning-heavy `needs_more_data` dry-run blocked without `--apply-review-action`, applied once to an isolated review store, and blocked the duplicate rerun;
- verification: `python -m pytest tests\dean_os\test_review_action_apply_ceremony.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_review_action_apply_ceremony` -> 5 passed.

EvidenceGapResolutionPlan:
- reads an active `needs_more_data` review action, the related decision packet, evidence pack, and optional source-routing report;
- converts missing tickers, truncated cached tables, weak/partial quality, missing source routing, and short date windows into concrete source/data tasks;
- read-only: no review action writes, no learning writes, no proposal enqueue, no config write, no pipeline run, no broker access, and no network fetch;
- smoke on the warning-heavy needs-more-data action returned `ready_to_collect`, 8 tasks, and missing tickers `AAPL`, `AMD`, `NVDA`, `TSM`;
- evidence refresh with `--max-rows-per-table 200` on existing cached parquet data produced `strong` quality, 158 documents, no missing requested tickers, and no dropped rows;
- fixed `ReviewDecisionPacket` quality compatibility so `strong` evidence is a pass (`evidence_quality_strong`) instead of a warning;
- refreshed decision packet became `reviewable`, recommended `mark_reviewed_candidate`, with 5 pass / 0 warn / 0 fail;
- isolated refreshed mark-reviewed ceremony wrote one review action and the learning bridge dry-run returned `dry_run_ready` with 4 promotable and 0 blocked records;
- verification: `python -m pytest tests\dean_os\test_evidence_gap_resolution_plan.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_evidence_gap_resolution_plan_rerun` -> 3 passed.

AnalystLearningApplyCeremony:
- reads a saved `AnalystLearningPromotionBridge` dry-run artifact and resolved learning/review/operations store paths;
- requires explicit `--apply-learning` before writing pending analyst learning records;
- verifies bridge status `dry_run_ready`, zero blocked candidates, active `mark_reviewed` review actions, no active `needs_more_data`, and no duplicate note ids in the target learning store;
- writes only pending learning records through the existing bridge apply path; no review action writes, proposal enqueue, config write, pipeline run, or broker access;
- smoke on the refreshed bridge dry-run blocked without `--apply-learning`, applied 4 records into an isolated learning store, and blocked duplicate re-apply;
- isolated learning store now has 4 pending records: research_ingestion neutral, financial_nlp neutral, specialist_research neutral, and evidence_synthesis bearish;
- verification: `python -m pytest tests\dean_os\test_analyst_learning_apply_ceremony.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_analyst_learning_apply_ceremony` -> 4 passed.

OutcomeReadinessGate:
- reads pending analyst learning records and local market price coverage;
- calls the existing outcome evaluator only in dry-run mode and never updates outcomes;
- returns `ready_for_outcome_dry_run`, `waiting_for_horizon`, `blocked_need_newer_prices`, `blocked_missing_inputs`, or `blocked_missing_market_data`;
- smoke on the isolated 4-record learning store returned `blocked_need_newer_prices`, with all 4 records in `no_price_after_created_at`;
- the records were created on 2026-06-13, while the local stage2 price parquet ends on 2026-05-04, so outcome evaluation is not meaningful yet;
- verification: `python -m pytest tests\dean_os\test_outcome_readiness_gate.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_outcome_readiness_gate` -> 4 passed.

OutcomePriceCoveragePlan:
- reads an `OutcomeReadinessGate` artifact and local CSV/parquet price metadata;
- converts `blocked_need_newer_prices` into concrete ticker/date coverage tasks;
- reports per-ticker latest price timestamps, required timestamps after learning record creation, and due-at horizon coverage;
- read-only: no price fetching, outcome writes, learning writes, review action writes, proposal enqueue, config write, pipeline run, or broker access;
- smoke on `reports\dean_os\outcome_readiness_gate_smoke\latest.json` returned `needs_price_refresh_after_record_creation`;
- current smoke requires AAPL, AMD, MSFT, NVDA, and TSM prices strictly after `2026-06-13T07:06:38.358225+00:00`;
- current production outcome horizon remains around `2027-06-13`, so price-after-creation is only a sanity prerequisite, not a final outcome-learning signal;
- verification: `python -m pytest tests\dean_os\test_outcome_price_coverage_plan.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_outcome_price_coverage` -> 4 passed.

MarketDataRefreshRunbook:
- reads an `OutcomePriceCoveragePlan`, optional `CollectorInventoryAgent` artifact, and known local price-cache paths;
- maps required tickers/date windows to safe operator tasks and command templates;
- identifies enabled local pipeline price feeds such as `yahoo_finance`, but never runs collectors or network calls;
- recommends a separate refreshed CSV/parquet artifact before overwriting old/stale price files;
- read-only: no collector run, network access, outcome writes, learning writes, review action writes, proposal enqueue, config write, pipeline run, or broker access;
- smoke on the current outcome price coverage plan plus fresh collector inventory returned `refresh_runbook_ready`;
- current smoke primary price feed is `yahoo_finance`, required tickers are AAPL, AMD, MSFT, NVDA, and TSM, and the minimum timestamp remains strictly after `2026-06-13T07:06:38.358225+00:00`;
- verification: `python -m pytest tests\dean_os\test_market_data_refresh_runbook.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_market_data_refresh_runbook` -> 4 passed.

Useful command checklist:

```text
dean_os/COMMAND_CHECKLIST.md
```

Опційно Agent Lab може одразу класти proposals у queue:

```text
python run_agent_lab.py docs/research --corpus data/dean_os/research_corpus.sqlite --learning-store data/dean_os/agent_learning.sqlite --operations-store data/dean_os/operation_queue.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle
```

## Що протестовано

Додано тести в `tests/dean_os/`.

Покрито:
- audit hard veto;
- data quality block;
- risk exposure block;
- macro/news domain report;
- value screening по фундаментальних метриках;
- consensus hard veto;
- consensus watchlist з bullish analytical report;
- orchestrator flow;
- decision logger;
- hybrid pipeline adapter.
- research corpus storage.
- research ingestion.
- specialist pattern synthesis.
- operations action proposals.
- material loaders and corpus ingestion CLI.
- Agent Lab runner and JSON/Markdown report generation.
- controlled missing-materials handling without traceback.
- FinancialNLPAgent and rule-based FinBERT-ready analysis.
- EvidenceSynthesisAgent and evidence-bound GPT-ready synthesis.
- LearningStore, learning CLI, pending thesis records, and agent scoring.
- OperationQueue and run_agent_ops CLI for proposal import/list/approve/reject/dry-run.
- EventLog and run_agent_logs CLI for structured run/action observability.
- Agent Lab sample mode for end-to-end smoke tests without external materials.
- Learning score now separates completed outcomes from pending records via `total_record_count` and `pending_record_count`.
- AgentReviewBuilder and run_agent_review CLI for post-run JSON/Markdown review summaries.
- ReviewActionStore and run_agent_review_actions CLI for review lifecycle decisions and watchlist proposals.
- RecommendationMemoryStore and run_agent_memory CLI for historical recommendation memory and lessons.
- Memory-aware Agent Lab, SpecialistResearchAgent, and EvidenceSynthesisAgent.
- RegimeContextBuilder, run_regime_context CLI, and regime-aware Agent Lab memory lookup.
- AgentPerformanceByContext, run_agent_context_performance CLI, and review snapshot context-performance section.
- OutcomeEvaluationRunner and run_agent_outcome_evaluation CLI for safe pending learning record evaluation.
- MarketDataFreshnessAgent and run_agent_market_freshness CLI for market data freshness preflight and refresh proposals.
- CollectorInventoryAgent and run_agent_collector_inventory CLI for local-only collector mapping before isolated health checks.
- PaperPortfolioSimulator, PaperPortfolioAgent, and run_agent_paper_portfolio CLI for paper-only portfolio PnL/exposure/drawdown diagnostics from logged decisions.
- PaperAutonomyRunner and run_agent_paper_autonomy CLI for the supervised paper loop and journal summary.
- DiaryBridgeAgent and run_agent_diary_bridge CLI for review-only paper-outcome-to-diary compatibility checks.
- HistoricalReplayRunner, replay leakage guard, normalized daily-bar option, and run_agent_historical_replay CLI for old-data reasoning exams.
- HistoricalResearchReplayRunner and run_agent_historical_research_replay CLI for combined evidence-pack + Agent Lab + price outcome exams.
- EvidenceTimestampAudit and run_agent_evidence_timestamp_audit CLI for source/evidence-pack timestamp gates before old-data research replay.
- HistoricalResearchReplayBatchRunner and run_agent_historical_research_replay_batch CLI for multi-date research replay summaries.
- ReplayPriceQualityInvestigationPlan and run_agent_replay_price_quality_investigation CLI for replay benchmark/interval anomaly diagnosis.
- ReplayPriceArtifactRepairPlan and run_agent_replay_price_artifact_repair CLI for non-destructive repaired candidate artifacts with quarantine metadata.
- ReplayCalibrationReadinessGate and run_agent_replay_calibration_readiness CLI for read-only calibration readiness after repaired replay batches.
- AnalystEvidencePackRunner now recognizes `published_date` / publication-date style columns, so future news rows are filtered correctly by `as_of`.
- ReviewApprovedLearningLoop and run_agent_review_approved_learning CLI for explicit reviewed promotion into pending learning records.
- AnalystOutcomeEvaluationLoop and run_agent_analyst_outcome_loop CLI for reviewed analyst outcome evaluation and audit metadata.
- AnalystCalibrationGate and run_agent_analyst_calibration_gate CLI for proposal-only profile/weight calibration guidance.
- CalibrationProposalAgent and run_agent_calibration_proposals CLI for OperationQueue-bound calibration review proposals.
- CalibrationReviewLifecycle and run_agent_calibration_review_lifecycle CLI for review-only approval/rejection of calibration proposals.
- ManualImplementationBacklog and run_agent_manual_implementation_backlog CLI for read-only approved calibration implementation tasks.
- AgentLearningLoopRunbook and run_agent_learning_loop_runbook CLI for read-only operator guidance across the safe analyst learning loop.
- AnalystLoopDailyCheck and run_agent_analyst_loop_daily_check CLI for read-only daily blocker/warning review across the analyst loop.
- AnalystReviewInbox and run_agent_analyst_review_inbox CLI for read-only human review queue construction.
- ReviewDecisionPacket and run_agent_review_decision_packet CLI for read-only per-source review decision support.
- ReviewActionDryRun and run_agent_review_action_dry_run CLI for read-only review action intent validation.
- ReviewActionApplyCeremony and run_agent_review_action_apply_ceremony CLI for the explicit one-action review write gate.
- EvidenceGapResolutionPlan and run_agent_evidence_gap_plan CLI for read-only needs-more-data source/task planning.
- AnalystLearningApplyCeremony and run_agent_learning_apply_ceremony CLI for applying pending analyst learning records from a validated bridge dry-run.
- OutcomeReadinessGate and run_agent_outcome_readiness CLI for read-only analyst outcome maturity and price-coverage checks.
- OutcomePriceCoveragePlan and run_agent_outcome_price_coverage CLI for read-only price coverage planning after outcome readiness blockers.
- MarketDataRefreshRunbook and run_agent_market_data_refresh_runbook CLI for read-only price refresh runbooks after coverage blockers.

Остання перевірка:

```text
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_market_data_refresh_runbook_full
85 passed
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_historical_research_replay_full
87 passed
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_evidence_timestamp_full_after_fix
92 passed
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_historical_research_replay_batch_full
94 passed
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_replay_price_quality_investigation_full
96 passed
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_replay_price_artifact_repair_full_v3
100 passed
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_replay_calibration_readiness_full
105 passed
```

CLI missing-path check:

```text
python run_agent_lab.py docs/research --corpus data/dean_os/research_corpus.sqlite --tickers AMD NVDA --sectors semiconductor --tags ai_cycle
```

Якщо `docs/research` ще не існує, команда повертає контрольований JSON із `load_error_count=1`, а не traceback.

Стабілізаційні виправлення після ручної перевірки:
- `ResearchCorpus` і `LearningStore` тепер явно закривають SQLite connections після кожної операції. Це прибирає Windows `WinError 32` під час pytest cleanup.
- `run_agent_learning.py update` тепер повертає контрольований JSON із підказкою, якщо `record_id` ще не існує.
- Приклади для PowerShell використовують `RECORD_ID_HERE`, а не `<record_id>`, бо кутові дужки PowerShell читає як оператор.

Остання перевірка після стабілізації:

```text
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_market_data_refresh_runbook_full
85 passed
```

## Що далі

Етап 1: Guardian MVP
- Залишити увімкненими `pipeline_audit`, `data_quality`, `risk`.
- Прогнати `create_hybrid_dean_orchestrator(mode="local")` на малому наборі tickers.
- Або CLI: `python run_dean_os.py --mode local --tickers AMD --timeframes 1d --enable-logging`.
- Перевірити, що audit/data/risk gates не блокують помилково.
- Увімкнути decision logging.

Етап 2: Pipeline learning / tuning
- Додати `ModelPerformanceAgent`, який читає результати backtest/model evaluation.
- Додати `RegimeAgent`, який бере market regime з існуючих модулів.
- `TuningAgent` is implemented as `proposal_only`.
- Навчання не має напряму міняти production config.
- Tuning proposals мають проходити walk-forward validation, risk constraints і human approval.

Етап 3: Domain intelligence
- Підключити стабільні feeds для fundamentals, sector ETF, macro releases, news timestamps.
- Підключити corpus ingestion для книжок, статей, звітів, transcripts і SEC filings.
- Додати нормальні PDF/DOCX dependencies у requirements, якщо corpus ingestion буде активно використовувати ці формати.
- Додати FinBERT/financial NLP шар для sentiment/event tone.
- Підключити локальний FinBERT model path у `FinancialNLPAgent`, коли модель буде доступна локально.
- Додати GPT/LLM шар для synthesis тільки з citations/evidence.
- Додати GPT client до `EvidenceSynthesisAgent`, але залишити deterministic fallback і citation guard.
- Зробити `ValueScreeningAgent` не keyword/rule MVP, а нормальний value screener.
- Зробити `SectorCycleAgent` через relative strength, breadth, earnings revisions, capex/order indicators.
- Зробити `HistoricalAnalogiesAgent` через базу історичних подій і схожих режимів.
- Додати citation/evidence requirements для кожної тези.

Етап 4: Agent learning loop
- Зберігати кожен якісний `ResearchNote` як pending `AgentLearningRecord`.
- Зберігати кожен `AnalyticalReport` і `ConsensusDecision`.
- Порівнювати thesis з майбутнім outcome на відповідному горизонті.
- Рахувати hit rate, calibration, false positive rate, false negative rate по кожному агенту.
- Зменшувати вагу агентів, які стабільно помиляються.
- Підвищувати вагу агентів тільки після out-of-sample підтвердження.

Етап 5: Production gates
- Live execution лишається вимкненим.
- Paper trading має враховувати slippage, commission, market impact.
- Будь-яка зміна model/config проходить proposal -> review -> approved experiment -> promotion.

## Принцип навчання агентів

Агенти не “навчаються” як автономні трейдери, які самі міняють систему. Вони навчаються як контрольовані аналітики:
- покращують власну калібровку;
- вчаться, які типи evidence реально працювали;
- отримують вагу в consensus залежно від історичної точності;
- не мають права самостійно виконувати trades або переписувати production config.

Ціль: не магічний автотрейдер, а disciplined research-and-governance layer.
