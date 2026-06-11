# DEAN-OS Next Chat Handoff

Last updated: 2026-06-08

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
- `SourceRoutingAgent`: local source/material routing map for pipeline feeds and specialist-agent intake.
- `OperationQueue` and `OperationsProposalAgent`: proposal-only automation loop.
- `ReviewActionStore` and `AgentReviewBuilder`: durable human review actions.
- `EventLog`: JSONL observability for agent runs and operation actions.

## Latest Verified State

Latest full DEAN-OS test command:

```powershell
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_historical_replay_full_final
```

Latest result:

```text
115 passed
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
14. Let source-specific collector fixes happen separately from the DEAN-OS agent-system layer.

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

Latest real replay result:

- raw replay selected `TSM` and evaluated as `hit`;
- normalized daily replay also selected `TSM`, but evaluated as `miss`;
- `price_quality.warnings` still reports an extreme SPY lookback return, so the replay result is diagnostic, not learning truth yet.

Source routing:

```powershell
python run_agent_source_routing.py docs/research --collector-inventory reports/dean_os/collector_inventory/latest.json --output reports/dean_os/source_routing/latest.json
```

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

`ReplayPriceNormalizer`

Scope:

- create a reusable normalized daily OHLCV artifact for replay from cached Yahoo/collector data;
- deduplicate ticker/date rows and preserve explicit aggregation assumptions;
- compare raw vs normalized replay outcomes;
- block learning-memory writes when price-quality warnings remain.

Why this next:

- `HistoricalReplayRunner` found that raw vs normalized daily bars can flip hit/miss;
- cached Yahoo/collector data can extend beyond Yahoo's current intraday window, but it must be normalized before being used as replay truth;
- the system should not learn from replay outcomes until price aggregation is stable.

## Chat Strategy

It is safe to start a new chat after this handoff. In the new chat, ask Codex to
read:

```text
dean_os/NEXT_CHAT_HANDOFF.md
dean_os/IMPLEMENTATION_STATUS.md
dean_os/COMMAND_CHECKLIST.md
reports/dean_os/collector_inventory/latest.json
```

If continuing in the current chat, proceed directly to `ReplayPriceNormalizer`.
