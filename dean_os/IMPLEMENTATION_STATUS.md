# DEAN-OS Implementation Status

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

Остання перевірка:

```text
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_historical_replay_full_final
115 passed
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
python -m pytest tests\dean_os -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_historical_replay_full_final
115 passed
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
