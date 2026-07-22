# DEAN-OS Next Chat Prompt

Last updated: 2026-06-27 after historical price recovery and timeframe-aware target integration.

Immediate continuation: first finish isolated integration verification of `TargetOrchestrator`, then implement the causal backward-only multi-timeframe context assembler. Do not restart template harvesting, data collection, or model-family training before those two temporal boundaries are closed.

Ти продовжуєш роботу в `D:\trading_project`. Не починай архітектуру з нуля і не копай нові шаблони без конкретної executable цінності.

Спочатку прочитай:

1. `dean_os/NEXT_CHAT_PROMPT.md`
2. `dean_os/IMPLEMENTATION_STATUS.md` лише верхні актуальні секції за 2026-06-27
3. `reports/dean_os/pipeline_control_historical_price_recovery_current/latest.json`
4. `reports/dean_os/pipeline_control_feature_causality_audit_current/latest.json`
5. `reports/dean_os/pipeline_control_bounded_evidence_batch_current/latest.json`
6. `Agents_architecture.md`, якщо треба перевірити межі гілок

## Мета системи

Будуємо три модульні частини:

1. Доменний аналітик. Повний екземпляр для semiconductor, який потім клонується на інший сектор заміною domain profile, показників і джерел, а не копіюванням коду.
2. Pipeline-control агент. Керує лише дозволеними площинами даних, ознак, моделей і метрик; пропонує зміни, але не торгує і не просуває їх сам.
3. Оркестратор. Поєднує аналітика і pipeline-control через evidence/review contracts.

Аналітик уже структурно близький до завершення. GPT і FinBERT є майбутніми optional enrichers, а не умовою правильної архітектури. Багатий аналіз, прогнози, рекомендаційні висновки, розбір причин успіху/помилки та learning proposals не треба видаляти. Заборонені лише самостійне виконання, неперевірене promotion і торгівля.

Human gate означає прийняття reusable template/learning/config change, а не ручне схвалення кожного детального звіту аналітика.

## Критичне уточнення про “0/4”

Не пиши “0 із 4 моделей”.

Коректно:

- corrected bounded batch перевірив чотири ticker contexts: `NVDA/15m`, `INTC/15m`, `TSM/15m`, `SPY/15m`;
- усі чотири використали один review-only `RandomForest` baseline;
- 0 із 4 ticker-context candidates пройшли всі metric planes;
- це не model-family benchmark.

Stage 4 окремо налаштований тренувати сім light model types:

- CatBoost
- LightGBM
- XGBoost
- RandomForest
- linear
- SVM
- KNN

Жодного чесного порівняння цієї сімки на нових multi-timeframe partitions ще не було.

## Що виправлено в Stage 3

- Знайдено й виправлено відрив `datetime` від OHLCV через index/suffix/sort/positional restore.
- Service columns відновлюються до guard-ів.
- Без temporal key сортування не виконується; із key використовується stable sort.
- Exact `datetime` має пріоритет над suffixed date-like columns.
- Market context, regime та significance thresholds стали causal point-in-time, а не final-window broadcast назад.
- Strict offline macro більше не домішує cache/FRED до explicit supplied macro.

Реальний causality audit:

- `reports/dean_os/pipeline_control_feature_causality_audit_current/latest.json`
- status: `feature_prefix_invariance_passed`
- NVDA: 0/229 noncausal numeric features
- SPY: 0/230
- 758/758 compared rows зберегли OHLCV identity
- audit не тренував модель і не читав test metrics

Старі bounded batch, diagnostic і feature-selection experiment, створені до row-identity fix, superseded. Не використовуй їхні числа.

## Corrected baseline

Актуальний post-fix batch:

- `reports/dean_os/pipeline_control_bounded_evidence_batch_current/latest.json`
- 4/4 real locked pairs completed
- 0 cautions cleared
- mean validation: 0.6842
- mean test: 0.5895
- mean balanced test: 0.5509
- mean feature stability: 0.5548

Це лише один frozen RF baseline у чотирьох контекстах. Не тюнити повторно на його test windows.

## Реальні price contexts, знайдені 2026-06-27

Новий код:

- `dean_os/pipeline_control_historical_price_recovery.py`
- `run_agent_pipeline_control_historical_price_recovery.py`
- `tests/dean_os/test_pipeline_control_historical_price_recovery.py`

Актуальний report:

- `reports/dean_os/pipeline_control_historical_price_recovery_current/latest.json`
- status: `historical_context_partitions_ready`

Trusted development sources:

- 15m: `data/colab/backup_20260510_153551/stage2_prices_15m_20260507_161411.parquet`
- 1d: `data/colab/backup_20260510_153551/stage2_prices_1d_20260426_083142.parquet`

Trusted later 15m source:

- бери точний artifact path із `reports/dean_os/pipeline_control_saved_price_repair_current/latest.json`
- на момент handoff це `reports/dean_os/pipeline_control_saved_price_repair_current/pipeline_control_saved_price_repair_20260627T120633739200+0000/artifacts/prices_15m_clean.parquet`

Фактичне development coverage для всіх 18 тікерів:

- 15m: 18,433 rows, 1,008-1,045 на тікер
- derived 60m: 4,090 rows, 220-237 на тікер
- direct 1d: 8,914 rows, 492-498 на тікер

Окрема later past-evaluation partition:

- 15m: 10,868 rows, 534-649 на тікер
- derived 60m: 2,326 rows, 115-142 на тікер
- derived 1d context tail: 422 rows, 21-25 на тікер

Отже твердження “60m/1d не мають достатньо даних” було правильним лише для короткого current-only repair artifact. Для development є достатньо чистих 15m, 60m і 1d.

Daily cross-check:

- 548 overlapping ticker-days
- p95 close relative error між direct 1d і derived-from-15m: 0.24%
- max: 0.67%
- consistency gate passed

## Небезпечні старі artifacts

Не відкривай accumulated `main_database_stage*_*.parquet` через pickle loader. Частина цих файлів має pickle protocol bytes, хоча suffix `.parquet`; writer використовував `pickle.dump`. Новий recovery runner перевіряє `PAR1` magic і відмовляється їх десеріалізувати.

Не використовуй як daily source:

- `data/colab/backup_20260510_153551/stage2_prices_1d_20260505_151233.parquet`

Він забруднений mixed cadence: 24,790 rows, intraday timestamps під label `1d`, 2,747 cross-ticker identical groups і тисячі extreme returns.

Не об’єднуй development та later partitions в один contiguous ряд. Між ними реальна часова прогалина; rolling features і targets не мають переходити через неї.

## Target/context contract

Recovery report уже фіксує:

- на 15m input one-hour target = shift 4 bars;
- на 60m input one-hour target = shift 1 bar;
- на 1d input one-day target = shift 1 bar;
- context join лише backward/as-of;
- future context заборонений;
- target не може перетинати partition/source boundary;
- development і past evaluation не конкатенуються;
- після model selection потрібен новий forward holdout.

Later partition називай `past_evaluation`, а не virgin locked holdout: частину її даних уже бачили попередні diagnostics.

## Timeframe-aware target implementation

Після recovery додано активну реалізацію:

- `src/targets/timeframe_contract.py`
- оновлено `src/targets/target_orchestrator.py`
- semantic `horizon` додано до intraday/hourly targets у `src/config/targets.yaml`
- тести додано в `tests/unit/test_target_orchestrator_alignment.py`

Тепер:

- `horizon=1h` резолвиться у shift 4 bars на 15m та 1 bar на 60m;
- target generation групує за `ticker+interval`, а не лише ticker;
- labels стають null, якщо future endpoint переходить через abnormal time gap;
- labels стають null при переході через `partition_id`, `source_partition`, `data_partition` або `segment_id`;
- chronological sort stable.

Перевірено: Python compilation passed, targets YAML parsed, direct contract smoke passed. Повний pytest-файл двічі вичерпав timeout під час старої важкої ініціалізації config stack без assertion output. Не називати цей pytest-набір успішним, поки його не буде завершено; спочатку ізолювати config initialization або запустити з достатнім контрольованим timeout.

## Наступна executable задача

Не запускай одразу всі сім моделей. Timeframe-aware target contract уже реалізовано; спочатку заверши його integration verification, потім:

1. Додай causal multi-timeframe context assembler:
   - anchor на prediction frame;
   - 60m і 1d тільки через backward `merge_asof`;
   - зберігати source timestamp, age/freshness і staleness flag;
   - не переносити контекст із майбутнього;
   - не приховувати великі gaps через interpolation/forward fill.
2. Покрий assembler synthetic unit tests для boundaries, але реальний report будуй лише з observed artifacts вище.
3. Потім зроби walk-forward train/validation на development partitions без читання past-evaluation metrics під час вибору.
4. Predeclare невеликий model-family comparison із configured Stage 4 models і naive/majority baseline. Не дозволяй model failures тихо перетворюватися на fallback RandomForest під чужою назвою.
5. Після freeze features/model/hyperparameters один раз оціни past-evaluation partition; далі накопичуй новий forward holdout.

Для контексту потрібні не лише price timeframes. Пізніше backward-as-of підключити causal macro/vintage, regime vector, news event context, sector/ticker state і evidence freshness. Не синтезувати macro signal, якщо реальні ряди stale/constant.

## Analyst branch state

Структурна оцінка: приблизно 97-98%, але це не означає predictive quality.

Уже є:

- modular domain instance/profile contracts;
- strict evidence pack and source gates;
- thesis, forecast, event interpretation, regime/scenario packets;
- outcome taxonomy, де окремо `correct_for_stated_reasons`, `correct_but_lucky_or_wrong_reason`, `incorrect_forecast`, `inconclusive`, `underspecified`, `data_unavailable`;
- case registry, feedback loop, learning proposal/promotion boundaries;
- portability review для clone-by-profile;
- review-only execution boundaries.

Manual template acceptance ще не записаний. Це не причина збіднювати звіт. Це лише gate перед клонуванням домену або автоматичним promotion.

Draft/thinking використовуй як source of candidate ideas. Інтегруй лише schemas, validators, deterministic builders, CLI, tests, audit/lineage hooks, human-review contracts і справді корисну domain logic. Довгі metadata ladders та повторні contract-fixture-checkpoint blocks лишай audit history.

## Safety boundaries

Без прямої команди користувача не запускати:

- live fetch або external API;
- важкий повний pipeline;
- autonomous tuning loop;
- learning/config promotion;
- broker routing, orders або trades;
- unreviewed dashboard publication.

Агенти можуть робити детальні висновки, прогнози, recommendation-like research conclusions і improvement proposals. Вони не можуть самостійно виконувати чи просувати їх.

## Verification

Новий recovery код перевірено:

```powershell
python -m pytest tests\dean_os\test_pipeline_control_historical_price_recovery.py tests\dean_os\test_pipeline_control_saved_price_repair.py -q -o addopts="" -p no:cacheprovider --basetemp reports\dean_os\pytest_tmp_historical_price_recovery
```

Результат: `3 passed`.

Команда реального recovery:

```powershell
python run_agent_pipeline_control_historical_price_recovery.py --historical-15m data\colab\backup_20260510_153551\stage2_prices_15m_20260507_161411.parquet --current-15m reports\dean_os\pipeline_control_saved_price_repair_current\pipeline_control_saved_price_repair_20260627T120633739200+0000\artifacts\prices_15m_clean.parquet --historical-1d data\colab\backup_20260510_153551\stage2_prices_1d_20260426_083142.parquet
```

## Робочий стиль

- Worktree shared і може бути dirty. Не revert чужі зміни.
- Спочатку читай наявний код і звіти.
- Пиши короткі progress updates користувачу.
- Не зациклюйся на нових шаблонах, відсотках або metadata.
- Реалізуй наступний важливий executable boundary, тести й реальний offline artifact.
- У звітах чітко розділяй structural readiness, data readiness, model evidence і execution authorization.
