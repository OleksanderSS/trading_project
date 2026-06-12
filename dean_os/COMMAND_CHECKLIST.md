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

Create a review-only tuning experiment proposal from model metrics and optional regime context:

```powershell
python run_agent_tuning.py PATH_TO_EVALUATION_JSON_OR_CSV --regime-context-json reports/dean_os/regime_context/amd_latest.json --tickers AMD --timeframes 1d
```

Share the `tuning.status`, `tuning.model_failures`, `tuning.regime_tags`, `tuning.guardrails`, and `action_proposals` sections.

This does not train, tune, run Optuna, or write production config. It only proposes a guarded walk-forward experiment.

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
