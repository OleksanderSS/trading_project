# Deep Audit Report: `algorithms`, `analytics`, `core`, `data`

Проєкт: trading pipeline / DEAN-OS  
Формат: консолідований звіт для подальшої передачі в Codex / 5.5-модуль / agent-mode.  
Обсяг: усе, що було знайдено під час аудиту наданих папок:

- `algorithms`
- `analytics`
- `core`
- `data`

Мета аудиту:

1. Перевірити, чи папки реально залучені в систему.
2. Знайти помилки, runtime-баги, неправильні імпорти, дублювання.
3. Оцінити правильність розрахунків.
4. Визначити, що треба фіксити першим.
5. Дати готовий список задач для Codex.

---

# 0. Загальний висновок

Надані папки не виглядають як “мертвий код”. Це ядро системи:

- `algorithms` — алгоритмічний шар: backtest, walk-forward, position sizing, risk parity, bias detection.
- `analytics` — аналітичний шар: market context, evaluation, model arena, reporting, causal/event analytics.
- `core` — фундамент: logging, exceptions, cache, security, HTTP, files, secrets, validation, batch-processing.
- `data` — Stage 1 / ingestion: collectors, DuckDB, data quality, temporal alignment, validation, synthetic data.

Головна проблема не в тому, що папки “не потрібні”, а в тому, що є **розриви між шарами**:

- API одного модуля не збігається з API іншого.
- Частина optional dependencies може ламати імпорт навіть якщо модуль вимкнений.
- Є дублювання класів/менеджерів/validator-ів.
- Деякі компоненти ніби інтегровані, але фактично не виконують очікувану дію.
- Частина помилок не падає явно, а створює “тихі” проблеми: не накопичуються дані, не працює дедуп, не працює optimizer, не запускається analyzer, stale cache тощо.

Найвищий пріоритет стабілізації:

1. `core`
2. `data`
3. `analytics`
4. `algorithms`

Логіка така: агенти й аналітика спираються на `core` і `data`. Якщо там нестабільність, розумні агенти будуть робити висновки на поганому фундаменті.

---

# 1. Audit: `algorithms`

## 1.1. Загальний стан

`algorithms` — корисна папка. Вона містить інструменти для етапів, близьких до Stage 5 / Stage 5.5 / Stage 6:

- walk-forward optimization;
- backtesting;
- bias detection;
- adaptive position sizing;
- risk parity allocation;
- transaction cost model;
- regime detection.

Модулі не виглядають мертвими. Частина вже реально підключена в інших частинах системи.

## 1.2. Реальна залученість

### Реально використовується

`AdaptivePositionSizer` використовується в:

```text
src/trading/portfolio_manager.py
```

Там він застосовується для розрахунку розміру BUY-позиції на основі:

- total equity;
- volatility;
- confidence;
- active positions;
- market regime;
- current price.

`RiskParityAllocator` також підключений у `PortfolioManager`.

`BiasDetector`, `TransactionCostModel`, `WalkForwardOptimizer` використовуються в:

```text
src/algorithms/advanced_backtest_engine.py
```

Тобто `algorithms` не є ізольованим архівом — він реально бере участь у торговій і backtesting-логіці.

## 1.3. Основні проблеми

## 1.3.1. Дублювання метрик

Є міксин:

```text
src/algorithms/metrics_mixin.py
```

У ньому вже є:

- `_calculate_sharpe`
- `_calculate_max_drawdown`
- `_calculate_stability_score`
- `_calculate_average_performance`

Але ті ж або дуже схожі методи повторюються в:

```text
src/algorithms/advanced_backtest_engine.py
src/algorithms/walk_forward_optimizer.py
```

Це створює ризик:

- одна формула зміниться, інша ні;
- різні модулі будуть рахувати Sharpe / drawdown / stability по-різному;
- тести можуть проходити в одному місці, але production-логіка відрізнятиметься.

Рекомендація:

- залишити метрики в `PerformanceMetricsMixin`;
- `AdvancedBacktestEngine` і `WalkForwardOptimizer` мають викликати спільні методи;
- прибрати дублювання, крім випадків, де справді потрібна інша формула.

## 1.3.2. Walk-forward optimizer фактично не оцінює параметри

У `WalkForwardOptimizer._select_best_params()` є логіка перебору кандидатів:

```python
candidates = list(self._iter_param_candidates(param_space))
return max(candidates, key=evaluate_candidate)
```

Але `evaluate_candidate(params)` викликає:

```python
return self._evaluate_with_params(data, params).get(metric, 0.0)
```

А `_evaluate_with_params()` зараз просто робить:

```python
return self._evaluate_parameters(data)
```

Тобто `params` не використовуються.

Наслідок:

- всі кандидати оцінюються однаково;
- optimizer не вибирає реально кращий набір параметрів;
- `best_params` може бути випадковим або залежним від порядку;
- Stage 5.5 / оптимізація параметрів зараз більше схожа на заглушку, ніж на реальну оптимізацію.

Це важлива проблема.

Рекомендація:

- або передавати `params` у модель/стратегію;
- або вимагати `optimization_func(data, params)`;
- або чітко назвати поточний режим `placeholder_optimizer`.

Краще API:

```python
optimization_func: Callable[[pd.DataFrame, dict[str, Any]], dict[str, float]]
```

і тоді:

```python
score = optimization_func(train_data, params).get(metric, 0.0)
```

## 1.3.3. Backtest engine частково дублює optimizer і mixin

`AdvancedBacktestEngine` має власні:

- `_calculate_sharpe`
- `_calculate_max_drawdown`
- `_calculate_stability_score`
- `_calculate_average_performance`
- `_evaluate_parameters`

Аналогічні речі вже є у `PerformanceMetricsMixin` та `WalkForwardOptimizer`.

Рекомендація:

- визначити один canonical path для метрик;
- `AdvancedBacktestEngine` має бути orchestration layer, а не дублювати math;
- залишити local override тільки якщо формула навмисно інша.

## 1.3.4. BiasDetector: логіка загалом правильна, але треба обережно інтерпретувати

`BiasDetector.detect_look_ahead_bias()`:

- бере сигнали;
- бере future returns через `shift(-lag)`;
- порівнює кореляції;
- шукає підозріло високі correlation values.

Це правильно як heuristic.

Але:

- висока кореляція не завжди означає leakage;
- низька кореляція не гарантує відсутність leakage;
- метод корисний як warning, не як absolute proof.

Плюс негативний зсув там intentional, і це нормально для detector-а.

## 1.3.5. AdaptivePositionSizer: логіка в цілому правильна

`AdaptivePositionSizer` враховує:

- base position size;
- VaR adjustment;
- Kelly adjustment;
- confidence adjustment;
- volatility adjustment;
- drawdown adjustment;
- active positions adjustment;
- regime multiplier;
- liquidity adjustment;
- min/max position limits.

Позитивно:

- є fallback при помилках;
- Kelly обмежується;
- position size обрізається min/max limits;
- якщо historical_returns немає, VaR adjustment = 1.0.

Потенційні ризики:

- Kelly formula на основі одного confidence score досить груба;
- confidence має бути відкалібрований, інакше sizing буде агресивним;
- VaRCalculator має повертати loss у зрозумілому знаку;
- `min_position_size_pct` може створити позицію навіть там, де multiplier дуже малий.

Рекомендація:

- додати режим `allow_zero_position=True`, щоб дуже слабкий сигнал міг давати 0, а не мінімальну позицію;
- тестувати з low confidence / crisis / high volatility.

## 1.3.6. RiskParityAllocator: корисний, але залежить від scipy/sklearn

`RiskParityAllocator` реалізує:

- ERC;
- HRP;
- MDP;
- MVP;
- equal weight;
- fallback risk parity.

Позитивно:

- є fallback;
- є validation inputs;
- якщо correlation matrix нема, метод може перейти в простіший режим.

Ризики:

- `scipy` і `sklearn` мають бути встановлені;
- HRP без кореляційної матриці має бути вимкнений або fallback;
- потрібно тестувати behavior при singular correlation matrix / NaN correlations.

## 1.4. Що дати Codex по `algorithms`

```text
Deep-fix src/algorithms without changing public architecture.

Tasks:

1. Unify performance metric calculations:
   - keep canonical Sharpe, max_drawdown, stability and average performance in PerformanceMetricsMixin;
   - remove or delegate duplicate methods from AdvancedBacktestEngine and WalkForwardOptimizer.

2. Fix WalkForwardOptimizer parameter evaluation:
   - _select_best_params must evaluate each candidate with its own params;
   - support optimization_func(data, params);
   - if params cannot be applied, return clear warning/status instead of fake optimization.

3. Improve AdvancedBacktestEngine:
   - delegate metric math to shared mixin;
   - keep _simulate_returns with signal shift(1) because it is temporal-safe;
   - add tests for transaction costs and no-position rows.

4. Add tests for BiasDetector:
   - aligned signal vs future return should trigger;
   - lagged safe signal should not trigger;
   - multi-column mismatch should test all pairs.

5. Add tests for AdaptivePositionSizer:
   - low confidence + crisis + high volatility should reduce size strongly;
   - optional allow_zero_position mode;
   - liquidity constraint should cap position.

6. Add tests for RiskParityAllocator:
   - no correlations fallback;
   - NaN/singular correlation handling;
   - weights sum to 1 after constraints.
```

---

# 2. Audit: `analytics`

## 2.1. Загальний стан

`analytics` — центральний аналітичний шар. Він не мертвий і не допоміжний.

Він відповідає за:

- market context;
- market phase;
- macro context;
- prediction adjustment;
- anomaly detection;
- critical signal detection;
- evaluation;
- model arena;
- reporting;
- causal/event analytics;
- performance analytics.

Синтаксично файли компілюються, але є runtime/import проблеми.

## 2.2. Реальна залученість

### Реально залучено

`UnifiedAnalyticsEngine` використовується у:

```text
EvaluationStage
```

`PredictionStage` імпортує:

- `MarketContextAnalyzer`
- `PredictionAdjuster`
- `AnomalyDetector`
- `CriticalSignalDetector`
- `signal_analytics`
- `significance_detector`

Feature enrichers використовують:

- `MacroScoreCalculator`
- `SentimentStatsCalculator`
- `MarketPhaseAnalyzer`
- `VolatilityCalculator`
- `DrawdownCalculator`
- `FamaFrenchFactors`

Meta-learning / arena використовують:

- `arena_battle`
- `performance_tracker`
- `ensemble_performance_bridge`

Тобто `analytics` реально вплетений у prediction, feature engineering, evaluation і meta-learning.

## 2.3. Критичні проблеми

## 2.3.1. UnifiedAnalyticsEngine може валити Stage 7 через missing keys

`EvaluationStage._run_deep_analysis()` передає обмежений `data_map`, приблизно:

```python
{
    "price_data": ...,
    "portfolio_data": ...,
    "signals": ...
}
```

А `UnifiedAnalyticsEngine` через конфіг може очікувати:

```text
economic_data
historical_economic_data
news_data
market_indicators
predictions
market_context
returns
performance_metrics
feature_space
macro_data
```

Якщо якогось ключа немає, `_get_data_for_analyzer()` кидає `ConfigurationError`.

Наслідок:

- один analyzer без потрібного input може зламати весь deep analysis;
- Stage 7 працює тільки в ідеальному випадку повного data_map.

Правильна поведінка:

- analyzer з missing input має бути `skipped`;
- весь engine має продовжити виконання.

Рекомендований патерн:

```python
try:
    input_data = self._get_data_for_analyzer(...)
except ConfigurationError as e:
    results[name] = {"status": "skipped", "reason": str(e)}
    continue
```

## 2.3.2. Causal analyzer залежить від `dowhy` і може ламати registry

Є два різні `CausalEngine`:

```text
src.analytics.context.causal_engine.CausalEngine
src.analytics.engines.causal_engine.CausalEngine
```

Перший — легший, pattern/context engine.

Другий — залежить від `dowhy`.

`causal_event_finder.py` імпортує саме `src.analytics.engines.causal_engine.CausalEngine`. Якщо `dowhy` не встановлений, виникає:

```text
ModuleNotFoundError: No module named 'dowhy'
```

Наслідок:

- `causal_events` не реєструється;
- `analyzer_registry` може падати;
- optional causal module ламає не-causal analytics.

Рекомендація:

- зробити `dowhy` optional;
- lazy import всередині конкретного методу;
- якщо dependency відсутня, повертати `status="skipped_missing_dependency"`.

## 2.3.3. Reporting orchestrator має неправильний імпорт

Файл:

```text
analytics/reporting/reporting_orchestrator.py
```

Імпортує:

```python
from src.analytics.reporting.results_manager import ModelResultsManager
```

Але у:

```text
analytics/reporting/results_manager.py
```

клас називається:

```python
ResultsManager
```

`ModelResultsManager` є в іншому місці:

```text
analytics/data_managers/model_results_manager.py
```

Наслідок:

```text
ImportError: cannot import name 'ModelResultsManager'
```

Рекомендація:

- або замінити на `ResultsManager`;
- або імпортувати `ModelResultsManager` з правильного модуля;
- зафіксити naming, щоб reporting layer мав один source of truth.

## 2.3.4. analytics/backtesting/engine.py імпортує неіснуючий модуль

Є імпорт:

```python
from src.metrics.financial_metrics import calculate_performance_metrics
```

А фактична структура має:

```text
src/metrics/financial/financial_metrics_library.py
src/metrics/financial/portfolio_metrics.py
```

Модуля `src.metrics.financial_metrics` немає.

Наслідок:

```text
ModuleNotFoundError: No module named 'src.metrics.financial_metrics'
```

Рекомендація:

- оновити імпорт на реальний шлях;
- або створити compatibility shim `src/metrics/financial_metrics.py`.

## 2.3.5. RiskRewardCalculator викликає неіснуючий `calculate_atr`

У `risk_reward_calculator.py` викликається:

```python
VolatilityCalculator.calculate_atr(...)
```

А в `volatility_calculator.py` є тільки:

- `calculate_rolling_volatility`
- `calculate_realized_volatility`

Методу `calculate_atr` немає.

Наслідок:

```text
AttributeError: type object 'VolatilityCalculator' has no attribute 'calculate_atr'
```

Мінімальний фікс:

```python
@staticmethod
def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(window=window, min_periods=1).mean()
```

## 2.3.6. run_full_analysis ловить занадто вузькі винятки

`UnifiedAnalyticsEngine.run_full_analysis()` ловить тільки:

```text
ValueError
TypeError
AttributeError
KeyError
ZeroDivisionError
```

А у вашому проєкті є доменні помилки:

- `DataProcessingError`
- `ConfigurationError`
- `ImportError`
- optional dependency errors

Приклад:

```text
DataProcessingError: Input data must be a non-empty DataFrame with a 'text' column.
```

Наслідок:

- один analyzer може зламати весь engine.

Рекомендація:

на рівні одного analyzer-а ловити ширше:

```python
except Exception as e:
    results[name] = {
        "status": "failed",
        "error_type": type(e).__name__,
        "error": str(e),
    }
```

Це нормально саме для orchestrator-а.

## 2.4. Частково залучене або проблемне

### `reporting/`

Є, але `reporting_orchestrator.py` має зламаний імпорт.

### `backtesting/engine.py`

Виглядає як старий або паралельний backtester. У Stage 7 використовується інший:

```text
src.backtesting.advanced.advanced_engine
```

А тут є зламаний імпорт financial metrics.

### `analyzer_registry.py`

Дублює ідею dynamic registration через `UnifiedAnalyticsEngine`.

### `analytics/engines/causal_engine.py` vs `analytics/context/causal_engine.py`

Однакові назви класу, різна логіка. Це небезпечно для агентів/рефакторингу.

## 2.5. Правильність розрахунків

### Нормально

- `DrawdownCalculator` — price-based drawdown виглядає адекватно.
- `VolatilityCalculator` — rolling/realized volatility загалом коректні.
- `RiskRewardCalculator` — Sharpe, Sortino, beta, Treynor, VaR/CVaR зроблені обережно.
- `MarketContextAnalyzer` — повертає очікуваний context vector на нормальному OHLCV.
- `CriticalSignalDetector` — проста і зрозуміла логіка price shock / volume spike / volatility explosion.

### Потребує правок

- `calculate_trade_parameters()` не працює без ATR.
- `_generate_data_hash()` у `UnifiedAnalyticsEngine` хешує малий sample DataFrame (`head(10).tail(5)`), що може давати stale cache.
- `MarketContextAnalyzer.analyze()` може падати замість часткового fallback.
- `TradingModelArena.run_champion_challenge()` може мати mismatch між зареєстрованим ім’ям champion і ім’ям, яке потім шукає `_get_model_predictions()`.

## 2.6. Дублювання / архітектурні розриви

1. Два causal engines:

```text
analytics/context/causal_engine.py
analytics/engines/causal_engine.py
```

2. Два results managers:

```text
analytics/data_managers/model_results_manager.py
analytics/reporting/results_manager.py
```

3. Два способи реєстрації analyzers:

```text
analyzer_registry.py
UnifiedAnalyticsEngine через YAML config
```

4. Два конфіги аналізаторів:

```text
src/config/analysis.yaml
unified_config.yaml / analysis.engine
```

Потрібен один source of truth.

## 2.7. Що дати Codex по `analytics`

```text
Fix analytics integration without changing architecture.

Tasks:
1. Make UnifiedAnalyticsEngine resilient:
   - missing data_mapping keys should skip only that analyzer, not fail the whole run;
   - analyzer exceptions should be captured per analyzer into {"status": "failed", "error_type", "error"}.

2. Fix reporting_orchestrator import:
   - use ResultsManager from analytics.reporting.results_manager or ModelResultsManager from analytics.data_managers.model_results_manager consistently.

3. Fix analytics/backtesting/engine.py import:
   - replace src.metrics.financial_metrics with the correct module path or add compatibility shim.

4. Add VolatilityCalculator.calculate_atr and make RiskRewardCalculator.calculate_trade_parameters pass.

5. Make causal_events optional:
   - lazy import dowhy;
   - if dowhy is unavailable, return status="skipped_missing_dependency" instead of breaking registry/engine.

6. Unify analysis config:
   - decide whether source of truth is src/config/unified_config.yaml or src/config/analysis.yaml;
   - remove stale analyzer entries or sync them.

7. Fix TradingModelArena champion naming:
   - ensure the name used in register_model is the same name used in _get_model_predictions.
```

---

# 3. Audit: `core`

## 3.1. Загальний стан

`core` — фундамент системи. Тут:

- логування;
- exceptions;
- error handling;
- cache;
- HTTP client;
- secrets;
- path security;
- file manager;
- batch processor;
- memory profiler;
- integrations base.

Синтаксично файли компілюються, але імпорти можуть падати через optional dependencies:

```text
duckdb
google.cloud
```

Це не завжди bug, якщо у локальному середовищі вони встановлені. Але для `core` погано, коли optional dependency ламає unrelated import.

## 3.2. Реальна залученість

### Сильно залучено

`ProjectLogger` використовується майже всюди:

- algorithms;
- collectors;
- pipeline;
- analytics;
- monitoring;
- config.

`ErrorHandler` використовується в:

- `pipeline_orchestrator`;
- `pipeline_factory`;
- `base_stage`;
- stages 0/1/7;
- data manager;
- optimization;
- models;
- meta_learning/security.

`CacheManager` використовується в:

- collectors;
- `CollectorFactory`;
- `HealthHub`;
- hybrid pipeline;
- feature cache.

`HttpClientFactory` використовується в collectors:

- AAII;
- FRED;
- Fear & Greed;
- CFTC;
- Google News;
- NewsAPI;
- Put/Call;
- SEC filings;
- Reddit sentiment;
- VIX.

`FileManager` використовується в:

- config manager;
- system validator;
- task manager;
- processing stage;
- analytics reporting.

`MemoryProfiler` підключений до:

- `pipeline_factory`;
- `pipeline_orchestrator`.

## 3.3. Слабо або майже не залучено

### `core/validation/validators.py`

Майже не використовується. Є окремий шар:

```text
src/validation/validators.py
```

Це дублювання.

### `ObjectCache` / `QueryCache`

Експортуються, але основний кеш — `CacheManager`.

### `VersionManager`

Майже ніде не використовується. Потенційна проблема: може писати version history у source tree:

```text
src/config/version_history.json
```

Runtime не має без потреби мутувати source tree.

## 3.4. Критичні проблеми

## 3.4.1. HttpClientFactory має async/API mismatch

`HttpClientFactory.get_http_client()`:

```python
async def get_http_client(...) -> httpx.AsyncClient:
```

Треба викликати:

```python
client = await factory.get_http_client()
```

А частина collectors робить:

```python
async with self.http_client_factory.get_http_client(timeout=self.timeout) as http_client:
```

Це помилка, бо `get_http_client()` повертає coroutine, а не async context manager.

Проблемні collectors:

```text
aaii_sentiment_collector.py
put_call_ratio_collector.py
sec_filings_collector.py
```

Правильний варіант:

```python
client = await self.http_client_factory.get_http_client(timeout=self.timeout)
async with client as http_client:
    ...
```

Або зробити окремий contextmanager API:

```python
async with factory.client(timeout=...) as http_client:
    ...
```

## 3.4.2. Retry config вводить в оману

Конфіг має:

```text
status_forcelist
backoff_factor
```

Але реально використовується:

```python
httpx.AsyncHTTPTransport(retries=retries)
```

У `httpx` це не повноцінний retry для HTTP статусів 429/500/502/503/504.

Наслідок:

- collectors можуть не повторювати 429/500;
- конфіг обіцяє більше, ніж виконує.

Рекомендація:

- або реалізувати status-code retry;
- або прибрати misleading config fields.

## 3.4.3. CacheManager має неправильну fallback-ініціалізацію DataManager

У `CacheManager.__init__`:

```python
self.db = data_manager or DataManager(self.config.get('paths.raw_db', 'data/raw_data.duckdb'))
```

А `DataManager` очікує `config_manager`, не path string.

Правильно:

```python
self.config = config_manager or get_current_config()
self.db = data_manager or DataManager(self.config)
```

## 3.4.4. CacheManager має ризик stale cache

`db_salt` будується із:

```text
table_name + row_count + max_date
```

Якщо значення у таблиці змінилися, але:

- row_count такий самий;
- max_date такий самий;

то cache key не зміниться.

Для фінансових даних це ризик.

Рекомендація:

- додати checksum;
- або updated_at;
- або snapshot/version id;
- або явну invalidation після оновлення stage.

## 3.4.5. Видалення кешу через `ttl=0` накопичує мертві rows

Через DuckDB issue замість `DELETE` використовується:

```python
UPDATE cache_metadata SET ttl = 0, timestamp = 0 WHERE key_hash = ?
```

Наслідки:

- metadata rows ростуть;
- `get_stats()` може рахувати мертві записи;
- cleanup може бути неточним.

Рекомендація:

- додати `is_active`;
- або окрему compaction;
- або явно виключати `ttl=0` rows зі статистики.

## 3.4.6. validate_safe_path неправильно працює з relative paths

Поточна логіка:

```python
base_path = Path(base_dir).resolve()
target_path = Path(path).resolve()
```

Якщо `path = "data/file.csv"`, він рахується від process cwd, а не від `base_dir`.

Також string `startswith` небезпечний:

```text
/tmp/base_evil
```

може пройти для:

```text
/tmp/base
```

Правильний варіант:

```python
base_path = Path(base_dir).resolve()
raw_path = Path(path)
target_path = raw_path if raw_path.is_absolute() else base_path / raw_path
target_path = target_path.resolve()

if not target_path.is_relative_to(base_path):
    raise PathValidationError(...)
```

## 3.4.7. FileManager залежить від помилки path validator

Якщо `validate_safe_path` неправильно рахує relative paths, то `FileManager(base_dir=...)` теж може працювати неправильно.

Ще проблема: `_atomic_write()` ловить вузький список exceptions, але реальні файлові помилки:

```text
OSError
PermissionError
FileNotFoundError
```

Наслідок:

- `.tmp` файли можуть залишатися;
- помилка може не логуватись нормально.

## 3.4.8. secure_secrets_manager має side effects на import

У кінці файлу:

```python
_secrets_manager_instance = SecretsManager()
```

Тобто при імпорті:

- шукається `.env`;
- читається environment;
- завантажуються encrypted secrets;
- логиться warning, якщо `.env` немає.

Для `core/security` це небажано.

Краще lazy singleton:

```python
_secrets_manager_instance = None

def get_secrets_manager():
    global _secrets_manager_instance
    if _secrets_manager_instance is None:
        _secrets_manager_instance = SecretsManager()
    return _secrets_manager_instance
```

## 3.4.9. SecretsManager може перезаписувати environment

`load_dotenv()` робить:

```python
os.environ[key] = value
```

Тобто `.env` перезаписує вже існуючі env vars.

У production зазвичай реальні env vars мають пріоритет.

Краще:

```python
if key not in os.environ or override:
    os.environ[key] = value
```

## 3.4.10. Шифрування секретів слабке

`CRYPTO_KEY` обрізається або доповнюється нулями до 32 байт:

```python
key_bytes = crypto_key.encode()
key_bytes = key_bytes.ljust(32, b'\0')
fernet_key = base64.urlsafe_b64encode(key_bytes)
```

Це слабкий key derivation.

Рекомендація:

- вимагати справжній Fernet key;
- або PBKDF2/HKDF із salt;
- або інтегрувати нормальний secret manager.

## 3.4.11. Дублювання exception hierarchy

Є:

```text
src/core/exceptions.py
src/core/error_handling/error_handler.py
```

І в обох є схожі:

```text
PipelineError
ConfigurationError
```

Це погано, бо `except ConfigurationError` може не зловити “інший” `ConfigurationError`.

Рекомендація:

- canonical exceptions залишити в `src/core/exceptions.py`;
- `error_handler.py` має імпортувати їх, а не оголошувати дублікати.

## 3.4.12. BaseIntegration.get_status падає замість повернення offline status

Метод має повертати статус:

```python
{
  "status": "online/offline",
  "error": ...
}
```

Але якщо `ping()` падає, він робить `raise`.

Для health-check це погано. Має бути:

```python
{
  "status": "offline",
  "reachable": False,
  "error": str(e)
}
```

## 3.4.13. UniversalNotifier має bug з aiofiles

Є патерн:

```python
data.add_field('photo', await aiofiles.open(image_path, 'rb').read())
```

Правильно:

```python
async with aiofiles.open(image_path, "rb") as f:
    image_bytes = await f.read()
```

Те саме для Discord file upload.

## 3.4.14. prediction_utils.normalize_prediction неправильно обробляє numpy arrays

Поточний порядок:

```python
if hasattr(pred, 'item'):
    return float(pred.item())

if isinstance(pred, np.ndarray):
    ...
```

Але `np.ndarray` теж має `.item()`. Для масиву з кількома елементами:

```python
np.array([0.1, 0.2]).item()
```

дає:

```text
ValueError: can only convert an array of size 1 to a Python scalar
```

Правильно:

```python
if isinstance(pred, np.ndarray):
    if pred.size == 0:
        return 0.0
    return float(pred.flat[-1])

if hasattr(pred, "item"):
    return float(pred.item())
```

Також batch normalizer має ловити `ValueError`.

## 3.4.15. BatchProcessor не скидає статистику між запусками

`process_batches()` оновлює:

```text
total_tickers
start_time
```

але не скидає:

```text
processed_tickers
failed_tickers
batches_processed
```

Якщо викликати двічі на одному instance, статистика забруднюється.

## 3.4.16. BatchProcessor ловить занадто вузькі exceptions

Batch processor має ізолювати падіння окремого batch. Тому краще ловити `Exception` на рівні конкретного batch, а не тільки вузький список.

## 3.5. Дублювання в `core`

```text
core/cache/cache_manager.py
pipeline/hybrid/cache_manager.py
```

```text
core/system/batch_processor.py
training/batch/batch_processor.py
```

```text
core/validation/validators.py
validation/validators.py
```

```text
core/base_integration.py
integrations/base.py
```

```text
core/utils/path_utils.py
utils/path_utils.py
```

```text
core/utils/math_utils.py
utils/math_utils.py
```

Це не означає, що треба все видаляти. Але треба визначити:

- `core/*` — низькорівневий фундамент;
- `src/validation/*`, `src/utils/*`, `pipeline/hybrid/*` — доменні фасади.

## 3.6. Що працює добре

- `MemoryProfiler` загалом адекватний.
- `DataValidator.validate_df()` швидко перевіряє OHLCV.
- `QueryCache` нормальний для DataFrame cache.
- `ProjectLogger` підтримує console + rotating file + CSV queue.

## 3.7. Що дати Codex по `core`

```text
Perform a deep stabilization pass for src/core without changing public architecture.

Tasks:

1. Fix src/core/security/path_validator.py:
   - relative paths must resolve against base_dir, not process cwd;
   - replace string startswith check with Path.is_relative_to;
   - keep symlink protection.

2. Fix FileManager:
   - ensure all relative paths are interpreted under base_dir;
   - _atomic_write must catch OSError/PermissionError and always cleanup temp files;
   - async_save should return Future or log background exceptions explicitly.

3. Fix HttpClientFactory and collectors:
   - decide one API: either get_http_client is async and all callers await it, or it returns an async context manager directly;
   - fix collectors using `async with factory.get_http_client(...)`;
   - implement status-code retry for 429/500/502/503/504 or remove misleading config fields.

4. Fix CacheManager:
   - use passed config_manager instead of always get_current_config();
   - instantiate DataManager with config_manager, not raw db path;
   - add active/expired metadata handling instead of counting ttl=0 rows as valid cache stats;
   - improve cache invalidation salt beyond count + max_date, or document explicit namespace invalidation.

5. Fix prediction_utils:
   - handle np.ndarray before `.item()`;
   - normalize_predictions_batch must catch ValueError as well as TypeError.

6. Unify exception hierarchy:
   - keep canonical exceptions in src/core/exceptions.py;
   - import them into error_handler.py instead of redefining PipelineError/ConfigurationError.

7. Make optional dependencies lazy:
   - google.cloud.storage import should be inside GCSManager initialization;
   - if google-cloud-storage is unavailable, GCS should be disabled gracefully;
   - duckdb-dependent modules should fail with a clear dependency error, not break unrelated imports.

8. Fix UniversalNotifier image/file sending:
   - use `async with aiofiles.open(...) as f: data = await f.read()`;
   - avoid manual JSON string concatenation for Discord payload.

9. Clarify validation layer:
   - decide whether src/core/validation/validators.py is active or legacy;
   - avoid duplicate TradeOrder/DataValidationError definitions;
   - if active, wire it into processing/prediction stages.

10. Fix BatchProcessor:
   - reset processing_stats at start of each process_batches call;
   - isolate RuntimeError/OSError/custom project exceptions per batch.
```

---

# 4. Audit: `data`

## 4.1. Загальний стан

`data` — Stage 1 / raw ingestion + DuckDB + data quality + temporal safety.

Вона реально залучена в:

- `CollectionStage`;
- `PipelineFactory`;
- `PipelineOrchestrator`;
- dashboard;
- monitoring;
- scripts;
- meta-learning;
- evaluation recovery.

`DataManager` і collectors — центральні компоненти.

## 4.2. Реальна залученість

- `CollectorFactory` створюється в `CollectionStage`.
- `CollectionManager` запускає collectors.
- `DataManager` використовується у багатьох частинах проєкту.
- `TemporalAlignmentChecker` вже інтегрований у `CollectionStage._combine_news_data`.
- Активні або потенційно активні collectors:
  - `YFCollector`
  - `FredCollector`
  - `GoogleNewsCollector`
  - `RSSCollector`
  - `NewsAPICollector`
  - `HuggingfaceCollector`
  - `AlternativeMeCollector`
  - `VIXCollector`

## 4.3. Критичні проблеми

## 4.3.1. Частина collectors не може створитися через abstract `run()`

`BaseCollector` має abstract method:

```python
async def run(...)
```

А такі collectors реалізують тільки `fetch_raw_data`:

```text
BigQueryCollector
CustomCSVCollector
FreeGoogleTrendsCollector
LocalFileCollector
SyntheticGenerator
```

Якщо їх увімкнути, буде:

```text
TypeError: Can't instantiate abstract class ... with abstract method run
```

Фікс:

- або дефолтний `run()` у `BaseCollector`;
- або `run()` у кожному collector.

Краще:

```python
async def run(self, *args, **kwargs):
    return await self.fetch_raw_data(*args, **kwargs)
```

## 4.3.2. CollectorFactory може падати через optional dependencies

`CollectorFactory` імпортує всі collector modules.

А деякі мають heavy imports на рівні модуля:

```python
from gnews import GNews
from pytrends.request import TrendReq
import yfinance as yf
import feedparser
from bs4 import BeautifulSoup
```

Якщо dependency немає, discovery може зламатися навіть для disabled collector.

А `_process_module_for_collectors()` не ловить `ImportError`.

Фікс:

```python
except ImportError as e:
    logger.warning(...)
    continue
```

І переносити optional imports всередину collector-а.

## 4.3.3. Async HTTP mismatch у collectors

Проблемні файли:

```text
aaii_sentiment_collector.py
put_call_ratio_collector.py
sec_filings_collector.py
```

Поточний неправильний патерн:

```python
async with self.http_client_factory.get_http_client(...) as http_client:
```

Фікс:

```python
client = await self.http_client_factory.get_http_client(timeout=self.timeout)
async with client as http_client:
    ...
```

Або змінити API `HttpClientFactory`.

## 4.3.4. Частина collectors не зберігає дані в БД

Деякі collectors самі роблять `db_manager.upsert(...)`, наприклад:

- `YFCollector`
- `FredCollector`
- `GoogleNewsCollector`
- `RSSCollector`
- `NewsAPICollector`
- `SEC`
- `Insider`
- `HuggingFace`
- `EconomicCalendar`

А інші переважно повертають DataFrame:

```text
AIISentimentCollector
CFTCCollector
FearGreedCollector
PutCallRatioCollector
RedditSentimentCollector
VIXCollector
AlternativeMeCollector
```

`CollectionStage.run()` зараз бере `raw_data`, потім `db_data`, але централізовано не зберігає всі результати.

Наслідок:

- дані можуть бути в памʼяті одного запуску;
- але не накопичуватись у DuckDB.

Рішення:

- або всі collectors persist-ять;
- або Stage 1 централізовано persist-ить усе.

Краще централізовано: collector збирає, stage зберігає.

## 4.3.5. Непослідовність `hash` vs `record_hash`

Частина collectors створює:

```text
hash
```

А частина:

```text
record_hash
```

`DataManager.filter_new_records()` шукає `hash`.

Приклади `record_hash`:

```text
AAII
CFTC
FearGreed
PutCallRatio
RedditSentiment
VIX
```

Наслідок:

- дедуп може не працювати;
- можливі дублікати.

Рекомендація:

- стандартний ключ `hash`;
- `record_hash` тільки alias;
- перед upsert робити:

```python
if "hash" not in df.columns and "record_hash" in df.columns:
    df["hash"] = df["record_hash"]
```

## 4.3.6. DataLoader має runtime-bug

`ColabDataLoader.check_cache()` і `save_cache_signature()` викликають:

```python
self._compute_signature()
```

А методу немає.

Наслідок:

```text
AttributeError
```

Потрібно додати `_compute_signature()`.

## 4.3.7. EventDatasetValidator має кілька помилок

Файл:

```text
data/validation/event_dataset_validator.py
```

Проблеми:

1. `validate(None)` падає, бо `_make_report()` робить `len(df)` і `len(df.columns)`.
2. `self.logger` не існує.
3. `_check_datetime_columns()` ловить помилку, додає issue, але потім робить `raise`.

Validator має повертати invalid report, не падати.

## 4.3.8. DataManager._clean_numeric_data() робить глобальний forward-fill

В `DataManager.upsert()`:

```python
df = self._clean_numeric_data(df, table_name)
```

А `_clean_numeric_data()` робить:

```python
df[numeric_cols] = df[numeric_cols].ffill()
```

Це для всіх таблиць однаково.

Ризик:

- sentiment NaN не має заповнюватися попереднім;
- macro release NaN не має заповнюватися попереднім;
- targets не можна заповнювати;
- news numeric fields теж можуть мати інший сенс.

Рекомендація:

- у `DataManager` тільки прибирати `inf`;
- fill/drop policy — на рівні конкретного stage/table.

## 4.3.9. DataManager dedup працює тільки по першій колонці `unique_on`

У `_prepare_upsert_df()`:

```python
key_col = valid_unique_in_table[0]
```

Якщо `unique_on = ["ticker", "datetime", "interval"]`, реально використовується тільки `ticker`.

Для market data це може бути катастрофа: нові дати для існуючого ticker можуть бути відфільтровані неправильно.

Рішення:

- або hash-only unique key everywhere;
- або composite dedup через tuple/join.

## 4.3.10. DataManager тихо відкидає нові колонки

Якщо DataFrame має нові колонки, яких немає у таблиці:

```python
extra = set(df_insert.columns) - existing_cols
df_insert = df_insert[common_cols]
```

Нові колонки silently dropped.

Ризик:

- нові фічі не потрапляють у БД;
- collector виглядає успішним, але дані втрачаються.

Рекомендація:

- `ALTER TABLE ADD COLUMN`;
- або хоча б warning `schema_mismatch`.

## 4.3.11. DataManager має shared connections без реального lock

Є `_connections` і `_connection_lock`, але lock фактично не захищає write operations.

Ризик:

- паралельні collectors;
- DuckDB locks;
- race conditions.

Рішення:

- write queue;
- або threading.Lock навколо writes.

## 4.3.12. VIXCollector має lookahead у derived features

Поточна логіка для історичних рядків використовує `.iloc[-1]` по всьому `hist`.

Ризик:

- `vix_sma_20`;
- `vix_percentile_20`;
- `vix_percentile_80`;
- `vix_change`;

можуть містити future leakage.

Правильно:

```python
hist["vix_sma_20"] = hist["Close"].rolling(20).mean().shift(1)
hist["vix_change"] = hist["Close"] - hist["Close"].shift(1)
hist["vix_percentile_20"] = hist["Close"].rolling(60, min_periods=20).quantile(0.2).shift(1)
hist["vix_percentile_80"] = hist["Close"].rolling(60, min_periods=20).quantile(0.8).shift(1)
```

## 4.3.13. TemporalAlignmentChecker має слабкий контракт з CollectionStage

`TemporalAlignmentChecker.check_news_alignment()` повертає:

```python
{
  "status": "VIOLATION",
  "violations": [...]
}
```

А `CollectionStage._check_news_temporal_alignment()` очікує:

```python
result.get("future_news_count")
result["future_indices"]
```

Тобто інтеграція ніби є, але фільтр може не працювати.

Рекомендація:

- або checker повертає expected keys;
- або stage читає `violations`.

Також checker може давати false positives, бо шукає будь-які future news, а не actual joined future news.

## 4.3.14. DataFreshnessChecker може давати хибні статуси

Проблеми:

- timezone stripping;
- negative lag при future timestamps.

Якщо timestamp у майбутньому, checker може сказати “fresh”.

Фікс:

```python
if lag_hours < -tolerance:
    status = "ERROR"
    message = "Data timestamp is in the future"
```

## 4.3.15. quick_filter_news_by_data_availability може відкидати корисні новини

Він робить buffer:

```text
min_price_date + 1 hour
max_price_date - 1 hour
```

Для daily data це не завжди логічно.

Рекомендація:

- зробити buffer параметризованим;
- timeframe-aware.

## 4.3.16. HuggingFaceCollector має config/code mismatch

Config:

```yaml
hash_keys:
  - content
```

Code default:

```python
self.hash_keys = ["text", "timestamp"]
```

Потім:

```python
df[self.hash_keys]
```

Якщо колонок нема, буде `KeyError`.

Рекомендація:

- перевіряти hash_keys;
- fallback на `content`, `text`, `title`, `body`;
- нормалізувати schema перед hash.

## 4.3.17. GoogleNewsCollector ігнорує keywords

Є рядок:

```python
list(keywords.keys()) if isinstance(keywords, dict) else (keywords or [])
```

але результат не присвоюється.

Потім:

```python
all_terms = list(set(tickers or []))
```

Тобто generic keywords ігноруються.

Може бути свідомо, але треба прибрати мертвий рядок і явно назвати behavior.

## 4.4. Дублювання / архітектурні розриви

### Два synthetic layers

```text
data/synthetic/data_generator.py
data/collectors/synthetic_generator.py
```

Перший — synthetic training dataset.

Другий — stress market scenarios.

Рекомендація:

- перейменувати/документувати:
  - `SyntheticTrainingDataGenerator`
  - `SyntheticScenarioGenerator`
- всюди ставити:
  - `is_synthetic`
  - `eligible_for_training`

### Два data managers / connection handlers

```text
data/management/data_manager.py
data/management/handlers/connection_handler.py
```

`DataManager` уже має connection lifecycle. `ConnectionHandler` виглядає старим/паралельним.

### LocalFileCollector і CustomCSVCollector дублюють ідею

`LocalFileCollector` кращий, бо має path validation і parquet.

`CustomCSVCollector` слабший, бо читає шлях напряму.

Рекомендація:

- залишити `LocalFileCollector`;
- `CustomCSVCollector` зробити alias/deprecated.

## 4.5. Що працює добре

- `YFCollector` генерує нормальний `hash`.
- `YFCollector` обмежує intraday до 58 днів.
- `FredCollector` зберігає macro data у DB.
- `RSSCollector` добре зроблений: semaphore, timeouts, `return_exceptions`, cache, DB upsert.
- `GoogleNewsCollector` має hard timeout 120 секунд і per-term timeout 30 секунд.
- `DataManager._quote_identifier()` захищає table/column names.
- Ідея `TemporalAlignmentChecker` правильна.
- `NewsPriceAvailabilityFilter` правильно позначений як quick Stage 2 filter.

## 4.6. Що дати Codex по `data`

```text
Deep-fix src/data without changing public architecture.

Tasks:

1. Fix BaseCollector contract:
   - either add default run() delegating to fetch_raw_data();
   - or implement run() in BigQueryCollector, CustomCSVCollector, FreeGoogleTrendsCollector, LocalFileCollector, SyntheticGenerator.
   - ensure disabled collectors with missing optional dependencies do not break CollectorFactory discovery.

2. Fix CollectorFactory:
   - catch ImportError/ModuleNotFoundError during collector discovery;
   - log skipped collector modules with missing dependency;
   - do not fail whole factory because one optional collector cannot import.
   - respect config enabled=false before importing heavy optional modules if possible.

3. Fix HTTP async usage:
   - replace `async with self.http_client_factory.get_http_client(...)` with awaited client context in AAII, PutCallRatio, SEC.
   - optionally add a proper async contextmanager to HttpClientFactory.

4. Standardize collector persistence:
   - choose one policy: collectors return data only, CollectionStage persists all; or collectors persist themselves.
   - avoid mixed behavior.
   - re-enable or remove CollectionStage.process_and_save_results() accordingly.

5. Standardize dedup key:
   - use `hash` everywhere.
   - replace `record_hash` with `hash` or create alias `df["hash"] = df["record_hash"]`.
   - update configs where hash_keys reference non-existing columns, especially VIX and FearGreed.

6. Fix DataManager composite dedup:
   - filter_new_records and _prepare_upsert_df must support composite unique_on, not only first column.
   - alternatively enforce hash-only unique keys everywhere.

7. Fix DataManager missing policy:
   - do not forward-fill all numeric columns globally.
   - replace inf with NaN in DataManager, but table-specific fill/drop policy should live in processing/feature stages.

8. Fix schema evolution:
   - when new columns appear in df, either ALTER TABLE ADD COLUMN or log clear schema_mismatch warning.
   - do not silently drop new columns.

9. Fix DataLoader:
   - implement _compute_signature();
   - include file size/mtime/hash for enriched_features.parquet and targets.parquet.

10. Fix EventDatasetValidator:
   - validate(None) must return invalid report, not crash;
   - add self.logger;
   - do not raise during datetime parsing; return issues.

11. Fix TemporalAlignmentChecker integration:
   - either return future_indices/future_news_count expected by CollectionStage;
   - or update CollectionStage to use result["violations"].
   - avoid false positives by validating actual joined news features, not any future news globally.

12. Fix VIXCollector derived features:
   - compute vix_sma_20, vix_change, percentiles row-wise/rolling with shift(1);
   - avoid full-window future leakage.

13. Fix HuggingFaceCollector:
   - validate hash_keys against actual columns;
   - fallback to content/text/title/body;
   - catch KeyError during hash creation.

14. Clarify synthetic policy:
   - all synthetic outputs must include is_synthetic and eligible_for_training;
   - production collectors must not silently use synthetic fallback unless explicitly enabled.
```

---

# 5. Cross-folder findings

## 5.1. Найважливіші наскрізні проблеми

### 1. Optional dependencies мають бути lazy

Проблемні приклади:

- `dowhy`
- `google.cloud`
- `duckdb`
- `pytrends`
- `gnews`
- `feedparser`
- `bs4`
- `yfinance`

Правило:

- якщо dependency потрібна тільки одному optional module, вона не має ламати імпорт всієї папки.

### 2. Exception hierarchy має бути єдина

Зараз є дублювання `PipelineError`, `ConfigurationError` тощо.

Потрібен один canonical file:

```text
src/core/exceptions.py
```

І всі інші мають імпортувати звідти.

### 3. Orchestrators мають ізолювати падіння підмодулів

Це стосується:

- `UnifiedAnalyticsEngine`
- `CollectorFactory`
- `CollectionManager`
- `BatchProcessor`
- agent orchestrator / DEAN-OS

Правило:

- один analyzer/collector/batch не має валити весь процес, якщо це не hard-veto dependency.

### 4. Temporal safety треба робити table/stage-specific

Проблеми:

- VIX derived features мають leakage.
- global forward-fill у DataManager небезпечний.
- news temporal alignment checker інтегрований не повністю.

### 5. Hash / dedup має бути стандартизовано

Потрібен один стандарт:

```text
hash
```

`record_hash` може бути alias, але не primary.

### 6. Конфіги мають мати один source of truth

Проблеми:

- `analysis.yaml` vs `unified_config.yaml`;
- collector config vs code defaults;
- hash_keys у config не завжди відповідають реальним columns.

---

# 6. Рекомендований порядок фіксів

## Phase 1 — Foundation stabilization

1. `core/security/path_validator.py`
2. `core/FileManager`
3. `core/HttpClientFactory`
4. exception hierarchy
5. `prediction_utils.normalize_prediction`

## Phase 2 — Data ingestion correctness

1. `BaseCollector.run()`
2. `CollectorFactory ImportError handling`
3. HTTP async collectors
4. collector persistence policy
5. hash vs record_hash
6. DataManager composite dedup
7. DataManager missing policy
8. VIX leakage

## Phase 3 — Analytics execution stability

1. `UnifiedAnalyticsEngine` skip/fail per analyzer
2. `dowhy` optional
3. reporting import
4. backtesting import
5. ATR method
6. config source of truth

## Phase 4 — Algorithms correctness

1. WalkForwardOptimizer real param evaluation
2. metrics mixin unification
3. tests for backtest / bias / sizing / risk parity

## Phase 5 — Agents integration

Після стабілізації foundation/data/analytics:

1. Підключати DEAN-OS agents до реальних artifacts.
2. Дати агентам читати health snapshots.
3. Додати agent gates:
   - data freshness;
   - temporal leakage;
   - risk gate;
   - model performance;
   - pipeline supervisor.
4. Заборонити агентам live actions без human approval.

---

# 7. One-shot Codex prompt

Нижче готовий великий prompt, який можна дати Codex / агенту-розробнику.

```text
You are working on a trading ML pipeline repository. Perform a stabilization pass without deleting useful code and without changing public architecture unless required.

Scope:
- src/algorithms
- src/analytics
- src/core
- src/data

General rules:
- Preserve useful legacy code unless it is clearly broken and replaced.
- Prefer compatibility shims over destructive deletion.
- Make optional dependencies lazy and non-fatal.
- One failing collector/analyzer/batch must not crash the whole orchestrator unless configured as hard blocker.
- Avoid lookahead leakage and global forward-fill in data ingestion.
- Add or update tests where possible.

Tasks:

CORE:
1. Fix path_validator: relative paths resolve against base_dir; use Path.is_relative_to; keep symlink protection.
2. Fix FileManager atomic writes: catch OSError/PermissionError/FileNotFoundError; cleanup temp files.
3. Fix HttpClientFactory API mismatch and all collectors using async with get_http_client without await.
4. Implement real status-code retry or remove misleading retry config fields.
5. Fix CacheManager DataManager initialization; do not pass raw db path as config_manager.
6. Improve cache invalidation salt or document explicit namespace invalidation.
7. Avoid counting ttl=0 metadata rows as active cache.
8. Fix prediction_utils.normalize_prediction: handle np.ndarray before .item(); catch ValueError in batch.
9. Unify exception hierarchy: canonical exceptions in src/core/exceptions.py.
10. Make GCS / duckdb optional imports graceful where possible.
11. Fix UniversalNotifier image/file sending with aiofiles async context manager.
12. Reset BatchProcessor stats per run and isolate broad per-batch exceptions.

DATA:
1. Add default BaseCollector.run() delegating to fetch_raw_data or implement run() in abstract collectors.
2. Fix CollectorFactory to catch ImportError/ModuleNotFoundError during discovery.
3. Make optional collector dependencies lazy.
4. Fix async HTTP usage in AAII, PutCallRatio, SEC collectors.
5. Standardize collector persistence policy.
6. Standardize dedup key to hash; keep record_hash only as alias.
7. Fix DataManager composite unique_on dedup or enforce hash-only.
8. Stop global numeric ffill in DataManager; keep table-specific missing policy.
9. Add schema evolution handling or clear schema_mismatch warning.
10. Add _compute_signature to ColabDataLoader.
11. Fix EventDatasetValidator validate(None), missing self.logger, and no raise during datetime parsing.
12. Fix TemporalAlignmentChecker contract with CollectionStage.
13. Fix VIXCollector rolling features with shift(1), avoiding future leakage.
14. Fix HuggingFaceCollector hash key validation and fallbacks.
15. Clarify synthetic data policy with is_synthetic and eligible_for_training.

ANALYTICS:
1. Make UnifiedAnalyticsEngine skip analyzers with missing required data instead of failing full run.
2. Capture analyzer exceptions per analyzer into result status failed/skipped.
3. Fix reporting_orchestrator import for ResultsManager / ModelResultsManager.
4. Fix analytics/backtesting/engine.py import of financial metrics.
5. Add VolatilityCalculator.calculate_atr and ensure RiskRewardCalculator.calculate_trade_parameters works.
6. Make dowhy dependency optional/lazy.
7. Unify analysis config source of truth.
8. Fix TradingModelArena champion naming mismatch.

ALGORITHMS:
1. Unify Sharpe/drawdown/stability/average metrics through PerformanceMetricsMixin.
2. Fix WalkForwardOptimizer: _select_best_params must evaluate each candidate with its own params.
3. Support optimization_func(data, params) or equivalent strategy/model callback.
4. Add tests for BiasDetector, AdvancedBacktestEngine, AdaptivePositionSizer, RiskParityAllocator.
5. Ensure signal shift(1) remains in backtest simulation to avoid lookahead.
```

---

# 8. Final priority list

## P0 / must fix first

- `HttpClientFactory` async mismatch
- `validate_safe_path`
- `DataManager` composite dedup
- `BaseCollector.run()` abstract issue
- `CollectorFactory` optional import failure
- `VIXCollector` future leakage
- `UnifiedAnalyticsEngine` full-run crash on missing analyzer input
- `RiskRewardCalculator` missing `calculate_atr`
- exception hierarchy duplication

## P1 / high priority

- global ffill in `DataManager`
- hash vs record_hash
- collector persistence inconsistency
- `DataLoader._compute_signature`
- `EventDatasetValidator`
- `reporting_orchestrator` import
- `analytics/backtesting/engine.py` import
- `prediction_utils.normalize_prediction`
- cache stale salt
- secrets manager import side effects

## P2 / cleanup

- duplicate managers / validators / utils
- config source of truth
- HRP/scipy/sklearn dependency handling
- BatchProcessor stats reset
- GoogleNews keywords behavior
- synthetic layer naming and policy

---

# 9. Short conclusion

Усі чотири папки корисні й важливі. Найбільша небезпека зараз не в “поганих алгоритмах”, а в інтеграційних розривах:

- collector повернув, але не зберіг;
- analyzer існує, але engine падає на missing key;
- optimizer перебирає params, але не застосовує їх;
- core має helper-и, але їх API не збігається з викликами;
- path/security/cache/data-manager мають edge cases, які можуть зламати pipeline непомітно.

Перший стабілізаційний спринт має бути не про нові фічі, а про:

1. core stability;
2. data correctness;
3. analytics resilience;
4. algorithms metric consistency;
5. після цього — DEAN-OS agents поверх стабільного фундаменту.

---

# 10. Audit: `devtools`

## 10.1. Загальний стан

`devtools` — це невелика, але корисна папка для допоміжних інструментів розробки. Вона не є production pipeline layer, і це прямо написано в `devtools/README.md`: інструменти для розробки, тестування, аналізу, експериментів і прототипів.

У наданому архіві є:

```text
devtools/
  README.md
  __init__.py
  rule_generator.py
  system_validator.py
  task_manager.py
  experimentation/
    README.md
    run_hyperparameter_tuning.py
  prototypes/
    live_trading_ticker_manager.py
```

Синтаксис Python-файлів компілюється. Але є кілька важливих runtime/import проблем, які можуть ламати використання `devtools` як пакета.

Головний висновок: **папка корисна, але має бути чітко ізольована від production, а імпорти треба зробити lazy/безпечними.** Найгірше місце — `src/devtools/__init__.py`, бо воно тягне важкий `ContextRuleGenerator`, а той тягне `DataManager`/DuckDB.

---

## 10.2. Залученість у проєкті

`devtools` не виглядає повністю мертвим.

### Реально згадується

`ContextRuleGenerator` використовується або згадується в:

```text
src/meta_learning/evolution/dual_loops.py
src/scripts/analysis/generate_context_rules.py
src/meta_learning/META_LEARNING_ANALYSIS.md
```

`TaskManager` згадується в `SystemValidator` як критичний файл:

```text
src/devtools/task_manager.py
```

`run_hyperparameter_tuning.py` описаний як canonical приклад у:

```text
devtools/experimentation/README.md
```

`live_trading_ticker_manager.py` зараз є prototype і не має бути залучений у production.

### Важливе уточнення

`devtools` — це не core production dependency. Тому він не має ламати запуск production pipeline, якщо у dev-середовищі немає якоїсь experimental залежності.

---

## 10.3. Критичні проблеми

## 10.3.1. `src/devtools/__init__.py` має важкий side effect

Файл:

```python
from src.devtools.rule_generator import ContextRuleGenerator
```

Це означає: будь-який імпорт `src.devtools.*` спочатку виконує `src/devtools/__init__.py`, а він тягне `rule_generator`.

`rule_generator.py` імпортує:

```python
from src.data.management.data_manager import DataManager
```

А `DataManager` тягне `duckdb`.

Наслідок: якщо `duckdb` не встановлений або недоступний, може впасти навіть такий імпорт:

```python
from src.devtools.task_manager import TaskManager
```

У моєму середовищі імпорт усіх `src.devtools.*` падав через:

```text
ModuleNotFoundError: No module named 'duckdb'
```

Це не означає, що у вас локально немає DuckDB. Але архітектурно це погано: `TaskManager` не має залежати від DuckDB тільки через `__init__.py`.

Правильніше:

```python
# src/devtools/__init__.py
__all__ = []
```

або lazy export:

```python
def __getattr__(name):
    if name == "ContextRuleGenerator":
        from src.devtools.rule_generator import ContextRuleGenerator
        return ContextRuleGenerator
    raise AttributeError(name)
```

А ще краще — не експортувати важкі інструменти через package-level `__init__`.

### Пріоритет

P0/P1, бо це ламає імпорт devtools submodules.

---

## 10.3.2. `TaskManager` має неправильний імпорт логера

У `task_manager.py`:

```python
from src.core.logging.logger import Logger as ProjectLogger
```

А в актуальному `src/core/logging/logger.py` клас називається:

```python
class ProjectLogger:
```

Тобто правильний імпорт:

```python
from src.core.logging.logger import ProjectLogger
```

Зараз ця помилка може маскуватися попередньою проблемою з `src/devtools/__init__.py`, але після фіксу `__init__` вона проявиться як:

```text
ImportError: cannot import name 'Logger'
```

### Пріоритет

P0/P1, бо `TaskManager` фактично не імпортується коректно.

---

## 10.3.3. `ContextRuleGenerator` рахує forward returns неправильно для multi-day windows

У `rule_generator.py`:

```python
target_returns = data[self.target_asset].pct_change(fill_method=None)

for window in effect_windows:
    data[f'target_return_{window}d'] = target_returns.shift(-window)
```

Для `window = 5` це не 5-денна forward return від `t` до `t+5`. Це **одноденна return у день `t+5`**.

Тобто для rules типу “що сталося через 5/20 днів після події” зараз рахується не те.

Правильніше:

```python
data[f"target_return_{window}d"] = (
    data[self.target_asset].shift(-window) / data[self.target_asset] - 1
)
```

або:

```python
data[f"target_return_{window}d"] = (
    data[self.target_asset].pct_change(periods=window, fill_method=None).shift(-window)
)
```

Поточний негативний shift тут intentional, бо це генерація label/ефекту після події, але формула має бути cumulative forward return, а не shifted one-day return.

### Пріоритет

P1, бо rule generation може створювати неправильні trading/context rules.

---

## 10.3.4. `ContextRuleGenerator` залежить від конфігу, якого може не бути

У `__init__`:

```python
self.analysis_config = self.config_manager.get_config('context_rule_generation')

if not self.analysis_config:
    raise ValueError("'context_rule_generation' section not found in configuration.")
```

Це нормально для explicit runner, але проблема в тому, що `ContextRuleGenerator` імпортується через `devtools/__init__.py`.

Якщо хтось просто імпортує `src.devtools`, він може підтягнути важкий generator і далі зачепити config/data dependencies.

Рекомендація:

- прибрати package-level import;
- робити ініціалізацію тільки в runner/script;
- не створювати `ContextRuleGenerator` автоматично.

---

## 10.3.5. `ContextRuleGenerator` використовує `DataManager.load_data_for_tickers`, але це жорстко прив’язано до `market_data_raw`

`DataManager.load_data_for_tickers()` бере:

```sql
SELECT datetime, ticker, close
FROM market_data_raw
WHERE ticker IN (...) AND interval = ?
```

А `ContextRuleGenerator` очікує, що target asset і всі indicators будуть columns у pivot DataFrame.

Це працює, якщо indicators теж збережені як ticker-like series у `market_data_raw`.

Але для багатьох context indicators це може бути не так:

- VIX може бути в окремій таблиці;
- macro може бути в `macro_data`;
- sentiment у `news_data` або іншій таблиці;
- breadth/put-call/fear-greed можуть бути alternative tables.

Тобто generator зараз добре працює тільки для “усі indicator-и є market_data_raw ticker series”.

Рекомендація:

- явно назвати це обмеження;
- або додати data source mapping для indicators:
  - table;
  - date column;
  - value column;
  - frequency;
  - alignment policy.

---

## 10.3.6. `ContextRuleGenerator` не має statistical significance / baseline

Він рахує:

- mean return;
- median return;
- win rate.

Але не рахує:

- sample size threshold;
- baseline return for same windows;
- t-stat / p-value / bootstrap CI;
- multiple testing correction;
- regime split;
- effect size vs unconditional distribution.

Наслідок: rules можуть виглядати “цікавими”, але бути випадковими.

Рекомендація:

- додати `min_event_count`;
- додати baseline comparison;
- додати confidence interval;
- позначати rules як `experimental`, не production.

---

## 10.3.7. `ContextRuleGenerator._save_rules_to_yaml()` може впасти, якщо path без директорії

Код:

```python
os.makedirs(os.path.dirname(path), exist_ok=True)
```

Якщо `path = "generated_rules.yaml"`, тоді:

```python
os.path.dirname(path) == ""
```

і `os.makedirs("")` може впасти.

Default path має директорію, але краще зробити безпечно:

```python
dir_name = os.path.dirname(path)
if dir_name:
    os.makedirs(dir_name, exist_ok=True)
```

Також краще використовувати `FileManager`, щоб не обходити security/path policy.

---

## 10.3.8. `SystemValidator` занадто жорстко перевіряє secrets

`SystemValidator._check_secrets()` перевіряє:

```text
NEWS_API_KEY
FRED_API_KEY
TELEGRAM_TOKEN
```

І якщо їх немає — додає errors.

Для devtools health check це може бути занадто суворо:

- Telegram може бути не потрібен для локального data/dev запуску;
- News API може бути вимкнений;
- FRED може бути optional.

Зараз відсутність optional secrets робить весь validator failed.

Рекомендація:

- розділити secrets на:
  - required;
  - optional;
  - required_if_enabled.
- брати список із config, а не hardcode.

---

## 10.3.9. `SystemValidator` створює `SecretsManager()` у `__init__`

```python
self.secrets_manager = SecretsManager()
```

Як уже було знайдено в `core`, `SecretsManager` має side effects: читає `.env`, environment, encrypted secrets, логить warnings.

Для validator-а це логічно, але краще робити lazy:

```python
self.secrets_manager = None
```

і створювати тільки якщо реально викликається `_check_secrets`.

---

## 10.3.10. `SystemValidator` має hardcoded paths і dependencies

Перевіряються шляхи:

```text
src/analytics/context
src/analytics/reporting
src/features/enrichers
src/data/collectors
src/pipeline/stages
src/devtools
src/core/logging
```

і файли:

```text
src/core/file_management/file_manager.py
src/utils/trading_calendar.py
src/utils/rate_limiter.py
src/core/logging/logger.py
src/pipeline/pipeline_orchestrator.py
src/config/unified_config_manager.py
src/devtools/task_manager.py
```

Це може бути нормально як quick check, але система у вас уже змінюється. Якщо структура зміниться, validator буде давати false failures.

Рекомендація:

- винести expected dirs/files у YAML config;
- мати профілі:
  - `minimal`;
  - `data_collection`;
  - `training`;
  - `full_pipeline`;
  - `devtools`.

---

## 10.3.11. `SystemValidator` рахує WARNING дивно

У `_check_system_resources()` RAM < 2GB дає:

```python
results["ram"] = {"status": "WARNING", ...}
```

А `_summarize_results()` рахує як passed тільки:

```python
["PASSED", "INFO"]
```

Тобто warning не є error, але знижує success_rate. Це може бути нормально, але варто явно рахувати:

- passed;
- warnings;
- failed.

Зараз summary має тільки failed_checks = len(errors), але warnings count відсутній.

---

## 10.3.12. `TaskManager` loading може падати через один поганий task

`_load_tasks()`:

```python
tasks = {task_id: Task.from_dict(task_data) for task_id, task_data in data.items()}
```

Якщо один task має:

- invalid status;
- invalid priority;
- bad created_at;
- bad due_date;

то весь task manager не завантажиться.

Рекомендація:

- load per task in try/except;
- зіпсовані tasks класти в quarantine або log warning;
- не валити весь task store.

---

## 10.3.13. `TaskManager.update_task()` не ловить invalid enum strings

```python
if key == 'status' and isinstance(value, str):
    value = TaskStatus(value)
```

Якщо value = `"done"` замість `"completed"`, буде `ValueError`.

Краще:

```python
try:
    value = TaskStatus(value.lower())
except ValueError:
    logger.warning(...)
    return None
```

Так само для priority.

---

## 10.3.14. `TaskManager` генерує нестабільні task IDs

```python
task_id = f"task_{int(time.time() * 1000)}_{len(self.tasks) + 1}"
```

Проблеми:

- можливий collision при паралельному створенні;
- ID залежить від кількості tasks;
- немає стабільного ID для TODO з конкретного file/line/text.

Для scanned TODO краще:

```python
task_id = sha256(f"{relative_path}:{line_num}:{todo_text}")
```

Для ручних tasks — `uuid4`.

---

## 10.3.15. `TaskManager.consolidate_codebase_todos()` може створювати дублікати або пропуски

Дедуп зараз:

```python
if any(task.title.startswith(...) and todo_text[:60] in task.title for task in self.tasks.values()):
    continue
```

Проблеми:

- два однакові TODO в різних файлах будуть вважатися одним;
- якщо текст трохи зміниться — буде новий task;
- якщо line moved — немає stable identity;
- не враховується file path.

Краще:

```text
source_path + line_number + normalized_text
```

або hash без line number, якщо треба survive line shifts.

---

## 10.3.16. `TaskManager.consolidate_codebase_todos()` не має exclude rules

Він сканує:

```python
for py_file in project_path.rglob("*.py"):
```

Без виключень:

- `.venv`
- `venv`
- `.git`
- `build`
- `dist`
- `__pycache__`
- notebooks checkpoint dirs
- generated files
- downloaded archives

Це може створити багато шуму.

Рекомендація:

```python
exclude_dirs = {".git", ".venv", "venv", "__pycache__", "build", "dist", ".mypy_cache", ".pytest_cache"}
```

---

## 10.3.17. `TaskManager` зберігає файл після кожного знайденого TODO

`consolidate_codebase_todos()` викликає `create_task()`, а той одразу `_save_tasks()`.

На великому repo це багато дискових записів.

Краще:

- створити всі tasks in-memory;
- один раз `_save_tasks()` в кінці;
- або `bulk_create_tasks`.

---

## 10.3.18. `run_hyperparameter_tuning.py` неправильно трактує return value optimizer-а

У скрипті:

```python
best_params = optimizer.optimize(X, y)

if not best_params:
    ...

final_model = RandomForestRegressor(**best_params, random_state=42)
```

А актуальний `BayesianOptimizer.optimize()` повертає:

```python
{
    "best_params": self.best_params,
    "best_score": self.best_score
}
```

Тобто `best_params` у скрипті насправді буде dict із ключами:

```text
best_params
best_score
```

І тоді:

```python
RandomForestRegressor(**best_params, random_state=42)
```

передасть у модель неіснуючі параметри `best_params` і `best_score`.

Правильно:

```python
result = optimizer.optimize(X, y)
best_params = result.get("best_params", result)
best_score = result.get("best_score")
```

Після цього:

```python
final_model = RandomForestRegressor(**best_params, random_state=42)
```

Це P1, бо experimentation template зараз показує неправильний usage canonical optimizer-а.

---

## 10.3.19. `run_hyperparameter_tuning.py` — хороша ідея, але має бути clearly demo-only

Позитивно:

- використовує synthetic dataset;
- показує `OptimizationFactory`;
- ловить `ImportError` для Optuna;
- добре підходить як приклад.

Але:

- це не має запускатися production pipeline;
- synthetic data має бути явно demo;
- `n_trials=25` нормально для прикладу, але не для реальної задачі;
- треба поправити return shape.

---

## 10.3.20. `live_trading_ticker_manager.py` — безпечний як prototype, але імпорти зламані/ризикові

Файл чесно попереджає:

```text
TODO: [IMPORTANT] This entire module is a non-functional prototype.
```

Методи піднімають `NotImplementedError`, тобто він не робить live trading actions. Це добре.

Але на рівні модуля є імпорти:

```python
from config.enhanced_sector_tickers import enhanced_sector_manager
from features.nlp.extractors.news_ticker_detector import NewsTickerDetector
```

Проблеми:

- імпорти без `src.` можуть не працювати;
- ці залежності навіть не використовуються, бо рядки в `__init__` закоментовані;
- через них prototype може не імпортуватися взагалі;
- є global instance:

```python
live_trading_manager = LiveTradingTickerManager()
```

що створює side effect на import.

Рекомендація:

- прибрати unused imports;
- не створювати global instance;
- залишити файл як README/prototype або guarded import;
- додати warning у назві/шляху: `prototypes/nonfunctional_live_trading_ticker_manager.py`.

---

## 10.4. Дублювання / архітектурні розриви

## 10.4.1. Rule generator існує у двох різних місцях/шляхах

У `devtools` є:

```text
src/devtools/rule_generator.py
```

А runner:

```text
src/scripts/analysis/generate_context_rules.py
```

у проєкті імпортує старі/неправильні шляхи:

```python
from src.core.analysis.rule_generator import ContextRuleGenerator
from src.utils.config_manager import UnifiedConfigManager
from src.core.data.data_manager import DataManager
```

Поточні правильні шляхи мають бути ближче до:

```python
from src.devtools.rule_generator import ContextRuleGenerator
from src.config.unified_config_manager import UnifiedConfigManager
from src.data.management.data_manager import DataManager
```

Тобто runner виглядає застарілим після рефакторингу.

Це важливо, бо фактичний CLI для rule generation може не працювати.

---

## 10.4.2. Experimentation README згадує майбутні scripts, яких немає

README пропонує:

```text
feature_importance_analysis.py
model_comparison.py
alternative_data_validation.py
```

Це нормально як roadmap, але краще позначити як “planned”, щоб не здавалося, ніби файли загублені.

---

## 10.4.3. `experiments/compare_layers.py` імпортує неіснуючий `devtools.experimentation.base`

Хоч це не всередині `devtools.zip`, але воно посилається на devtools:

```python
from devtools.experimentation.base import BaseExperiment
```

У наданій `devtools/experimentation` є тільки:

```text
README.md
run_hyperparameter_tuning.py
```

`base.py` немає.

Також там є старий імпорт:

```python
from src.metrics.financial_metrics import calculate_performance_metrics
```

який уже був знайдений як проблемний в `analytics`.

Отже `experiments/compare_layers.py` зараз, ймовірно, не працює.

Рішення:

- або додати `devtools/experimentation/base.py`;
- або оновити `compare_layers.py` на актуальний experimentation framework;
- або позначити `experiments/compare_layers.py` як legacy.

---

## 10.5. Що працює добре

### Позитивні моменти

1. `devtools/README.md` правильно відділяє devtools від production pipeline.
2. `live_trading_ticker_manager.py` не робить небезпечних live actions, а явно кидає `NotImplementedError`.
3. `ContextRuleGenerator` має корисну ідею: автоматично генерувати context rules з історичних data.
4. `SystemValidator` корисний як quick environment/project health check.
5. `TaskManager` може бути корисним для консолідації TODO/FIXME у structured tasks.
6. `run_hyperparameter_tuning.py` корисний як приклад використання `OptimizationFactory`.

---

## 10.6. Що дати Codex по `devtools`

```text
Deep-fix src/devtools without changing public architecture.

Tasks:

1. Fix src/devtools/__init__.py:
   - remove eager import of ContextRuleGenerator;
   - do not import DataManager/DuckDB just by importing src.devtools;
   - optionally use lazy __getattr__ for ContextRuleGenerator.

2. Fix TaskManager import:
   - replace `from src.core.logging.logger import Logger as ProjectLogger`
     with `from src.core.logging.logger import ProjectLogger`.

3. Harden TaskManager:
   - use uuid4 for manual task IDs;
   - use stable hash IDs for scanned TODOs based on path + normalized text;
   - validate status/priority strings safely;
   - load tasks one-by-one and quarantine invalid records instead of failing whole store;
   - add exclude dirs for TODO scanning;
   - batch-save tasks after consolidation instead of saving after every TODO.

4. Fix ContextRuleGenerator forward return calculation:
   - for N-day effect use price[t+N] / price[t] - 1;
   - keep negative shift only as intentional label generation;
   - add min_event_count, baseline return comparison and optional significance stats.

5. Make ContextRuleGenerator data inputs explicit:
   - document that indicators must currently exist as market_data_raw ticker-like series;
   - or add indicator source mapping: table/date/value/frequency/alignment.

6. Fix ContextRuleGenerator save path:
   - handle output path without dirname;
   - consider FileManager for safe writes.

7. Fix SystemValidator:
   - make SecretsManager lazy;
   - split required/optional/required_if_enabled secrets;
   - move expected dirs/files/dependencies into config profiles;
   - count warnings separately from failed checks.

8. Fix run_hyperparameter_tuning.py:
   - handle optimizer.optimize() return shape:
     result = optimizer.optimize(X, y)
     best_params = result.get("best_params", result)
   - do not pass `best_score` to RandomForestRegressor.

9. Fix live_trading_ticker_manager.py prototype:
   - remove unused module-level imports with wrong non-src paths;
   - remove global live_trading_manager instance;
   - keep NotImplementedError safeguards;
   - clearly mark as non-production prototype.

10. Fix stale runner paths:
   - update src/scripts/analysis/generate_context_rules.py imports to current paths:
     src.devtools.rule_generator,
     src.config.unified_config_manager,
     src.data.management.data_manager.
   - or mark runner as legacy.

11. Add missing experimentation base or update experiments:
   - either create devtools/experimentation/base.py;
   - or update src/experiments/compare_layers.py to current framework;
   - also fix old financial metrics import there.
```

---

## 10.7. Priority list for `devtools`

### P0 / must fix

- `src/devtools/__init__.py` eager import of `ContextRuleGenerator`.
- `TaskManager` wrong logger import.
- stale runner imports in `src/scripts/analysis/generate_context_rules.py`, if this script is still used.

### P1 / high priority

- `ContextRuleGenerator` forward return calculation.
- `run_hyperparameter_tuning.py` optimizer return handling.
- `TaskManager` robust loading and TODO stable IDs.
- `SystemValidator` required vs optional secrets.
- prototype import side effects.

### P2 / cleanup

- README planned scripts labeling.
- `experiments/compare_layers.py` missing experimentation base.
- remove unused imports.
- add config-driven validator profiles.

---

## 10.8. Summary for `devtools`

`devtools` — корисна папка, але її треба тримати окремо від production. Найбільша проблема зараз — не якість самих ідей, а те, що devtools має важкі імпорти й side effects на package import.

Правильний напрям:

- `devtools` не має ламати production imports;
- усі experimental/prototype речі мають бути lazy або guarded;
- rule generation має рахувати forward returns правильно;
- task manager має стати стійким до поганих записів і великих repo;
- system validator має бути config-driven, а не hardcoded;
- live trading prototype має залишатися безпечним і не створювати global runtime objects.

Після цих фіксів `devtools` може стати корисним контуром для Codex/агентів: генерувати rules, збирати TODO, запускати експерименти й валідатор середовища без ризику зламати основний пайплайн.

---

# 11. Audit: `calibration`

## 11.1. Загальний стан

`calibration` — невелика папка, але концептуально важлива. Вона відповідає за два різні типи калібрування:

1. **Калібрування гіперпараметрів DEAN / trading model** через `CalibrationEngine` + Optuna.
2. **Калібрування confidence score / probability calibration** через `AdaptiveConfidenceCalibrator`.

У папці є:

```text
calibration/
  README.md
  __init__.py
  adaptive_confidence_calibrator.py
  calibration_engine.py
```

Синтаксично файли компілюються, імпорт модулів проходить. Якщо `optuna` не встановлена, `calibration_engine.py` імпортується, але пише warning. Це краще, ніж падіння на import.

Головний висновок: **папка корисна, але зараз майже не залучена в основний pipeline і має кілька важливих API/логічних розривів.** Найкритичніше: `CalibrationEngine` не використовується ніде, README показує неправильний programmatic API, а `AdaptiveConfidenceCalibrator` не експортований через `__init__.py` і теж не використовується.

---

## 11.2. Залученість у проєкті

### Поточна залученість слабка

Пошук по проєкту показує, що:

```text
CalibrationEngine
AdaptiveConfidenceCalibrator
src.calibration
```

фактично згадуються тільки в самій папці `src/calibration` і README.

Тобто:

- `CalibrationEngine` не підключений до `PipelineOrchestrator`;
- не видно активного `run_hybrid_pipeline.py`, який реально викликає calibrate mode;
- `scripts/calibrate_dean.py`, згаданий у README, у поточному дереві не знайдений;
- `AdaptiveConfidenceCalibrator` не підключений до ensemble/model output layer;
- `AdaptiveConfidenceCalibrator` навіть не експортується у `src/calibration/__init__.py`.

### Є інші calibration-модулі в проєкті

Окремо існує:

```text
src/models/ensemble/calibration/
  base.py
  strategies.py
```

Там є:

- `CalibrationStrategy`
- `PlattScalingStrategy`
- `IsotonicRegressionStrategy`

Також у `dean_os` є інший контур:

```text
dean_os/analyst_calibration_gate.py
dean_os/calibration_proposal_agent.py
dean_os/calibration_review_lifecycle.py
```

Це не те саме, що `src/calibration`:

- `src/calibration` — model/hyperparameter/confidence calibration.
- `dean_os/*calibration*` — review-only calibration of analyst profiles / operation proposals.

Потрібно явно розвести ці поняття в документації, інакше Codex/агент може переплутати.

---

## 11.3. Позитивні моменти

## 11.3.1. `CalibrationEngine` має правильну загальну ідею

Ідея корисна:

- брати real data з DuckDB;
- брати synthetic scenarios;
- запускати Optuna;
- використовувати chronological split;
- комбінувати real metric і synthetic metric;
- зберігати `calibration_results.json`.

Це підходить для Stage 5.5 / model tuning / DEAN tuning.

## 11.3.2. Є chronological split

У `evaluate_hyperparameters()` використовується:

```python
X_train, X_val, y_train, y_val = self._chronological_split(X, y)
```

Це добре, бо для trading/time-series не можна робити shuffle split.

## 11.3.3. Fallback evaluation deterministic

Якщо real data немає, є `_fallback_evaluation()`.

Це краще, ніж випадковий mock score. Але є важлива проблема: у `run_calibration()` при empty real data процес завершується failed і fallback не використовується як повний calibration mode. Тобто fallback використовується тільки всередині trial evaluation, але не при повній відсутності real data.

## 11.3.4. `AdaptiveConfidenceCalibrator` має корисну ідею

Він реалізує:

- Platt scaling;
- isotonic regression;
- simple binning fallback;
- online history;
- exponential decay;
- distribution shift detection;
- save/load.

Це дуже корисно для системи, яка має `confidence` в сигналах, consensus engine, risk sizing і agent verdicts.

---

# 11.4. Критичні проблеми

## 11.4.1. `src/calibration/__init__.py` експортує тільки `CalibrationEngine`

Поточний `__init__.py`:

```python
from src.calibration.calibration_engine import CalibrationEngine

__all__ = ['CalibrationEngine']
```

`AdaptiveConfidenceCalibrator` не експортується.

Наслідок:

```python
from src.calibration import AdaptiveConfidenceCalibrator
```

не працює.

Рекомендація:

```python
from src.calibration.calibration_engine import CalibrationEngine
from src.calibration.adaptive_confidence_calibrator import AdaptiveConfidenceCalibrator

__all__ = ["CalibrationEngine", "AdaptiveConfidenceCalibrator"]
```

Але з урахуванням optional dependencies краще робити lazy export, щоб calibration import не тягнув важкі модулі без потреби.

---

## 11.4.2. README показує неправильний programmatic API

README показує:

```python
engine = CalibrationEngine(
    real_data_path="data/duckdb/trading.db",
    synthetic_data_path="data/synthetic/",
    n_trials=50,
    metric="sharpe_ratio",
    batch_name="my_calibration"
)
```

А фактичний `CalibrationEngine.__init__`:

```python
def __init__(self, config_manager: UnifiedConfigManager, n_trials=50, metric='sharpe_ratio', batch_name='calibration'):
```

Тобто README-код не запуститься.

Потрібно або:

1. Оновити README:

```python
from src.config.unified_config_manager import UnifiedConfigManager
from src.calibration import CalibrationEngine

config = UnifiedConfigManager()
engine = CalibrationEngine(
    config_manager=config,
    n_trials=50,
    metric="sharpe_ratio",
    batch_name="my_calibration",
)
```

або

2. Розширити `CalibrationEngine.__init__`, щоб він підтримував і `config_manager`, і explicit paths:

```python
def __init__(
    self,
    config_manager=None,
    real_data_path=None,
    synthetic_data_path=None,
    ...
):
```

---

## 11.4.3. README згадує scripts/files, яких у поточному дереві не видно

README згадує:

```text
python run_hybrid_pipeline.py --mode calibrate
python scripts/calibrate_dean.py
docs/CALIBRATION_GUIDE.md
scripts/data_accumulation_strategy.md
src/pipeline/hybrid_orchestrator.py
```

У наданому дереві частина цих шляхів або відсутня, або називається інакше.

Наслідок:

- документація може вести користувача/агента до неіснуючих entrypoints;
- Codex може намагатися фіксити не ті файли;
- calibration виглядає інтегрованим, але фактичний runner відсутній.

Рекомендація:

- або відновити entrypoints;
- або оновити README під актуальну структуру;
- або позначити ці команди як legacy/planned.

---

## 11.4.4. `CalibrationEngine.__init__` робить `run_calibration()` fallback unreachable без Optuna

У `calibration_engine.py`:

```python
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
```

А в `__init__`:

```python
if not OPTUNA_AVAILABLE:
    raise ImportError("optuna is required for calibration...")
```

При цьому в `run_calibration()` є:

```python
if not OPTUNA_AVAILABLE:
    return {
        "status": "failed",
        "reason": "optuna_not_installed",
        ...
    }
```

Але до `run_calibration()` неможливо дійти, бо instance не створиться.

Це логічний розрив.

Рішення:

- або залишити raise в `__init__` і прибрати unreachable check у `run_calibration`;
- або краще не падати в `__init__`, а повертати structured failed result у `run_calibration`.

Для pipeline/agent orchestration краще другий варіант:

```python
self.optuna_available = OPTUNA_AVAILABLE

def run_calibration(...):
    if not self.optuna_available:
        return {"status": "failed", "reason": "optuna_not_installed"}
```

---

## 11.4.5. `load_real_data()` має SQL injection / unsafe string interpolation

У `load_real_data()`:

```python
query = 'SELECT * FROM enriched_features'
if test_ticker:
    query += f" WHERE ticker = '{test_ticker}'"
```

Те саме для targets.

Навіть якщо ticker приходить від користувача або CLI, це небезпечно.

Потрібно використовувати параметризований запит DuckDB:

```python
features_df = conn.execute(
    "SELECT * FROM enriched_features WHERE ticker = ?",
    [test_ticker],
).fetchdf()
```

або без WHERE, якщо `test_ticker is None`.

Також потрібно quoting/validation ticker format.

---

## 11.4.6. `load_real_data()` ловить не ті винятки

DuckDB errors часто будуть:

- `duckdb.Error`
- `duckdb.CatalogException`
- `duckdb.IOException`
- `ImportError`
- `OSError`

А код ловить тільки:

```python
ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError
```

Наслідок:

- якщо DuckDB file відсутній;
- таблиця `enriched_features` відсутня;
- `duckdb` не встановлений;
- query падає;

то помилка може не бути коректно перетворена в empty DataFrame.

Рекомендація:

```python
except Exception as e:
    logger.exception(...)
    return {"features": pd.DataFrame(), "targets": pd.DataFrame()}
```

Саме для loader-а це нормально, бо він має повертати controlled status.

---

## 11.4.7. DuckDB connection не гарантовано закривається

У `load_real_data()`:

```python
conn = duckdb.connect(...)
...
conn.close()
```

Якщо помилка станеться між connect і close, connection може не закритися.

Краще:

```python
conn = None
try:
    conn = duckdb.connect(...)
    ...
finally:
    if conn is not None:
        conn.close()
```

або context manager, якщо підтримується.

---

## 11.4.8. `evaluate_hyperparameters()` не вирівнює `features` і `targets`

Код:

```python
X = real_data['features']
y = real_data['targets']
...
X_train, X_val, y_train, y_val = self._chronological_split(X, y)
```

Проблема:

- `features` і `targets` можуть мати різні індекси;
- різну довжину;
- різні datetime;
- різні ticker rows;
- targets можуть бути multi-row/multi-ticker не aligned.

Зараз split бере `len(X)` для split index і застосовує до `y`, але не перевіряє, що `len(X) == len(y)`.

Наслідок:

- model.fit може впасти;
- або гірше — features/targets можуть бути зміщені.

Потрібно:

- join/merge by stable keys:
  - `ticker`
  - `datetime`
  - target horizon
- або хоча б assert length/index equality.

---

## 11.4.9. У features передаються неочищені нечислові колонки

`X = real_data['features']` передається в:

```python
RandomForestRegressor.fit(X_train, y_train)
```

А `enriched_features` може містити:

- `datetime`
- `ticker`
- strings/categories
- hash
- source
- booleans/object columns
- NaN/inf

RandomForestRegressor не прийме object columns.

Треба:

```python
X = X.select_dtypes(include=[np.number])
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
X = X.fillna(...)
```

Але missing policy має бути обережна й не створювати leakage.

---

## 11.4.10. `_calculate_sharpe_ratio()` трактує `y_pred` як directional signal

У `_calculate_sharpe_ratio()`:

```python
returns = np.sign(y_pred) * y_true
```

Це означає:

- якщо prediction > 0, long;
- якщо prediction < 0, short;
- якщо prediction = 0, no position.

Це нормально як proxy metric, але треба явно назвати: це не Sharpe моделі, а Sharpe directional strategy proxy.

Ризики:

- не враховує transaction costs;
- не враховує position sizing;
- не враховує confidence threshold;
- не враховує long-only constraints;
- annualization always `sqrt(252)`, навіть якщо target horizon не daily.

Рекомендація:

- перейменувати на `_calculate_directional_strategy_sharpe`;
- додати horizon/frequency parameter;
- за потреби додати cost penalty.

---

## 11.4.11. `_evaluate_on_synthetic()` фактично не використовує модель

Функція приймає:

```python
def _evaluate_on_synthetic(self, model, synthetic_scenarios):
```

Але всередині модель не використовується. Вона просто читає:

```python
metrics = scenario.get('metrics', {})
sharpe = metrics.get('sharpe_ratio', 0)
sharpe_ratios.append(abs(sharpe))
```

Тобто synthetic score не залежить від hyperparams/model. Усі trials отримають однаковий synthetic component, якщо scenarios однакові.

Наслідок:

- 30% combined metric є константою;
- Optuna реально оптимізує тільки 70% real metric;
- документація каже “Evaluate model on synthetic scenarios”, але цього не відбувається.

Рішення:

- або реально генерувати synthetic features і проганяти model.predict;
- або прибрати synthetic metric з optimization;
- або назвати це `synthetic_prior_score` і зробити залежним від hyperparams.

---

## 11.4.12. `_evaluate_on_synthetic()` бере `abs(sharpe)`

```python
sharpe_ratios.append(abs(sharpe))
```

Це небезпечно.

Якщо synthetic scenario має `sharpe_ratio = -2`, `abs()` перетворить це на +2 і зробить поганий сценарій хорошим.

Для trading metric знак важливий.

Рекомендація:

```python
sharpe_ratios.append(float(sharpe))
```

або якщо це “stress severity”, перейменувати metric.

---

## 11.4.13. `self.metric` майже не впливає на objective

`CalibrationEngine` має:

```python
metric: str = 'sharpe_ratio'
```

Але `evaluate_hyperparameters()` завжди рахує:

```python
real_metric = self._calculate_sharpe_ratio(...)
synthetic_metric = self._evaluate_on_synthetic(...)
combined_metric = 0.7 * real_metric + 0.3 * synthetic_metric
```

Тобто якщо користувач передасть:

```text
metric="max_drawdown"
metric="win_rate"
metric="profit_factor"
```

це майже ніде не буде використано як інша objective function.

README каже, що supported metrics є:

- `sharpe_ratio`
- `max_drawdown`
- `win_rate`
- `profit_factor`
- `calmar_ratio`

А код фактично оптимізує Sharpe proxy.

Потрібно або реалізувати ці метрики, або не писати, що вони підтримуються.

---

## 11.4.14. `define_hyperparameter_space()` змішує RL parameters і RandomForest proxy parameters

У hyperparams є:

- `actor_lr`
- `critic_lr`
- `hidden_dim`
- `num_layers`
- `batch_size`
- `replay_buffer_size`
- `gamma`
- `tau`
- `exploration_noise`
- `dropout`
- `weight_decay`

А реально в `evaluate_hyperparameters()` модель:

```python
RandomForestRegressor(
    n_estimators=actor_n_estimators,
    max_depth=actor_max_depth,
    ...
)
```

Тобто RL parameters не використовуються в реальному model.fit. Вони впливають тільки на `_fallback_evaluation()`.

Наслідок:

- Optuna шукає багато параметрів, які не впливають на objective;
- study results misleading;
- `best_params` може містити RL params, які насправді не були перевірені.

Рекомендація:

розділити режими:

```text
mode="rf_proxy"
mode="dean_rl"
```

Для `rf_proxy` тільки RF params.

Для `dean_rl` треба реальний DEAN train/eval callback.

---

## 11.4.15. `run_calibration()` не зберігає study/trials/history

Зараз зберігається тільки:

```text
calibration_results.json
```

з best params/value.

Не зберігаються:

- всі trials;
- trial params;
- trial values;
- failed trials;
- Optuna study database;
- environment/config snapshot;
- data hash;
- feature columns;
- target name;
- train/val split metadata.

Для reproducibility цього недостатньо.

Рекомендація:

- `study.trials_dataframe().to_csv(...)`;
- Optuna storage SQLite;
- save config snapshot;
- save data fingerprint;
- save selected numeric features;
- save target/horizon/frequency.

---

## 11.4.16. `AdaptiveConfidenceCalibrator` не використовується в системі

Він не підключений до:

- prediction stage;
- consensus engine;
- portfolio manager;
- risk sizing;
- DEAN agents;
- ensemble calibration strategies.

Тобто корисний модуль є, але наразі це isolated utility.

Рекомендація:

- інтегрувати в місце, де формується final signal confidence;
- або створити adapter між `src/models/ensemble/calibration` і `AdaptiveConfidenceCalibrator`.

---

## 11.4.17. `AdaptiveConfidenceCalibrator` не валідує input range

`calibrate(raw_confidence)` очікує confidence 0..1, але не перевіряє:

- raw_confidence is finite;
- raw_confidence not NaN;
- raw_confidence within [0,1];
- actual_outcome is 0/1.

Зараз `np.clip()` частково рятує, але:

```python
np.clip(np.nan, 0.01, 0.99)
```

залишить `nan`.

Потрібно:

```python
if not np.isfinite(raw_confidence):
    return 0.5
raw_confidence = float(np.clip(raw_confidence, 0.0, 1.0))
```

Для outcome:

```python
if actual_outcome not in (0, 1, True, False):
    raise ValueError(...)
```

або мʼяко skip.

---

## 11.4.18. Platt scaling падає, якщо outcomes мають один клас

`LogisticRegression.fit()` не може навчитися, якщо всі `actual_outcome` однакові.

Наприклад, перші 50 результатів усі 1 або всі 0. Тоді `_execute_platt_training()` впаде.

Зараз `_retrain_models()` ловить exception і просто завершується. Але тоді:

- isotonic не тренується;
- simple fallback не тренується;
- calibrator залишається uncalibrated.

Краще:

```python
if len(np.unique(outcomes)) < 2:
    train simple binning / constant calibrator
    skip Platt
else:
    train Platt
```

І якщо Platt fails, не зупиняти весь retraining, а спробувати isotonic/simple.

---

## 11.4.19. `_compute_metrics()` рахує metrics по старих calibrated confidence

У `update_with_outcome()` зберігається:

```python
'calibrated_confidence': calibrated_conf
```

Це calibrated value на момент додавання outcome.

Після retrain моделей `_compute_metrics()` бере саме ці старі values:

```python
calibrated = np.array([e['calibrated_confidence'] for e in self.calibration_history])
```

Але після retrain новий calibrator може давати інші calibrated confidence для тих самих raw values.

Тому MAE/ECE показують не якість поточного calibrator-а, а історичні outputs старих calibrator states.

Краще після retrain рахувати:

```python
calibrated = np.array([self.calibrate(e["raw_confidence"]) for e in self.calibration_history])
```

Можна з guard, щоб не рекурсити/не логити зайве.

---

## 11.4.20. ECE не включає confidence == 1.0 в останній bin

У `_compute_metrics()`:

```python
mask = (calibrated >= bin_edges[i]) & (calibrated < bin_edges[i + 1])
```

Для останнього bin `[0.9, 1.0]` значення `1.0` не потрапляє.

Оскільки `calibrate()` clip до 0.99, це майже не проявляється. Але для loaded/history/simple map значення може бути 1.0.

Краще для останнього bin:

```python
if i == n_bins - 1:
    mask = (calibrated >= bin_edges[i]) & (calibrated <= bin_edges[i + 1])
```

---

## 11.4.21. Distribution shift detection через KS test на binary outcomes слабкий

`_check_distribution_shift()` порівнює дві половини outcomes через:

```python
stats.ks_2samp(old_outcomes, new_outcomes)
```

Для binary 0/1 це може працювати як heuristic, але це не найкращий тест для change in hit rate.

Краще:

- two-proportion z-test;
- rolling hit rate difference;
- PSI/JS divergence по raw_confidence distribution;
- calibration error drift;
- sample size thresholds.

Також `scipy` імпортується eagerly:

```python
from scipy import stats
```

Якщо scipy нема, весь calibrator не імпортується. Для fallback mode краще lazy/optional.

---

## 11.4.22. `save()` не використовує trusted path, а `load()` використовує

`load()` використовує:

```python
resolve_trusted_artifact_path(...)
```

Це добре.

А `save()` робить:

```python
Path(filepath).parent.mkdir(...)
joblib.dump(data, filepath)
```

Тобто save path не проходить через той самий trusted artifact policy.

Рекомендація:

- або використовувати аналогічний trusted output resolver;
- або явно обмежити save dir;
- або приймати тільки path у configured artifacts dir.

---

## 11.4.23. `save()`/`load()` ловлять занадто вузькі exceptions

`joblib.dump/load` можуть кидати:

- `OSError`
- `PermissionError`
- `FileNotFoundError`
- `ImportError`
- pickle/joblib errors

А код ловить тільки:

```text
ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError
```

Потрібно ловити ширше або конкретні файлові exceptions.

---

## 11.4.24. `CalibrationEngine` використовує synthetic data у score, але немає чіткого synthetic gate

У коді є audit-ignore comment:

```python
# audit-ignore: SYNTHETIC_SECONDARY — 30% weight only
```

Це добре, що synthetic має тільки 30% ваги. Але:

- нема поля `is_synthetic`;
- нема provenance;
- нема перевірки, що synthetic data не змішано з training real data;
- synthetic score фактично не залежить від model.

Рекомендація:

- explicit `synthetic_policy`;
- save in results:
  - synthetic_weight;
  - synthetic_files;
  - scenario counts;
  - whether synthetic was used;
- якщо synthetic data немає, combined metric має бути або real-only, або weight renormalized.

---

## 11.5. Дублювання / архітектурні розриви

## 11.5.1. Два confidence calibration шари

Є:

```text
src/calibration/adaptive_confidence_calibrator.py
src/models/ensemble/calibration/strategies.py
```

У другому вже є:

- Platt scaling;
- Isotonic regression.

Перший робить те саме + online learning/history/drift.

Не треба видаляти. Але треба визначити ролі:

- `src/models/ensemble/calibration/strategies.py` — low-level strategies.
- `AdaptiveConfidenceCalibrator` — stateful online calibrator / wrapper.
- `CalibrationEngine` — hyperparameter tuning, не probability calibration.

## 11.5.2. `calibration` vs `dean_os` calibration

`dean_os/analyst_calibration_gate.py` не про model hyperparameters, а про analyst profile promotion gate.

Потрібна документація:

```text
src/calibration = model and confidence calibration
dean_os/*calibration* = analyst profile review/proposal lifecycle
```

## 11.5.3. README описує інтеграцію, якої в дереві не видно

README говорить про hybrid pipeline calibrate mode, але активного entrypoint не видно.

Це або legacy docs, або відсутні файли.

---

## 11.6. Що працює добре

- `calibration_engine.py` імпортується без hard crash, навіть якщо Optuna не встановлена.
- DuckDB import зроблений всередині `load_real_data`, не на рівні модуля.
- Є chronological split, а не random split.
- `_calculate_sharpe_ratio()` чистить non-finite returns.
- `AdaptiveConfidenceCalibrator` має fallback, якщо sklearn відсутній.
- `load()` перевіряє trusted artifact path перед joblib load.
- Є `get_calibration_report()`, що корисно для monitoring/agents.
- `CalibrationEngine` зберігає результат у JSON.

---

## 11.7. Що дати Codex по `calibration`

```text
Deep-fix src/calibration without changing public architecture.

Tasks:

1. Fix src/calibration/__init__.py:
   - export AdaptiveConfidenceCalibrator too;
   - preferably use lazy exports to avoid unnecessary heavy imports.

2. Fix README:
   - update programmatic usage to match CalibrationEngine(config_manager=...);
   - remove or mark legacy commands that do not exist in current repo;
   - clearly distinguish model calibration from DEAN-OS analyst calibration gates.

3. Fix CalibrationEngine Optuna behavior:
   - do not raise in __init__ when optuna is missing;
   - let run_calibration() return structured failure reason optuna_not_installed;
   - or remove unreachable run_calibration optuna check.

4. Fix load_real_data:
   - use parameterized DuckDB queries for test_ticker;
   - validate ticker string;
   - catch duckdb/import/os errors safely;
   - ensure connection closes via finally/context manager.

5. Align features and targets:
   - merge/join features and targets by ticker + datetime or another stable key;
   - validate equal length/index after alignment;
   - fail with clear status if alignment is impossible.

6. Prepare numeric model matrix:
   - select numeric feature columns;
   - remove target/leakage/id columns;
   - handle NaN/inf with explicit policy;
   - save selected feature list in calibration results.

7. Clarify metric behavior:
   - rename _calculate_sharpe_ratio to directional strategy proxy Sharpe;
   - make self.metric actually select objective metric;
   - implement or remove documented metrics: max_drawdown, win_rate, profit_factor, calmar_ratio.

8. Fix synthetic evaluation:
   - either actually evaluate model on synthetic scenario features;
   - or remove synthetic score from objective;
   - do not use abs(sharpe);
   - save synthetic_weight and scenario provenance.

9. Split hyperparameter spaces:
   - mode="rf_proxy" should only tune RF params;
   - mode="dean_rl" should call actual DEAN train/eval callback;
   - do not include unused RL params in RF objective.

10. Improve reproducibility:
   - save all trials to CSV/JSON;
   - optionally use Optuna SQLite storage;
   - save config snapshot, data fingerprint, target, selected features, train/val ranges.

11. Integrate AdaptiveConfidenceCalibrator:
   - connect it to the place where final signal confidence is produced;
   - or bridge it with src/models/ensemble/calibration strategies;
   - add tests for calibration behavior.

12. Harden AdaptiveConfidenceCalibrator inputs:
   - validate raw_confidence finite and in [0, 1];
   - validate actual_outcome is binary;
   - handle NaN by fallback confidence 0.5.

13. Fix retraining edge cases:
   - if outcomes have one class, skip Platt and use simple/constant calibration;
   - if Platt fails, still attempt isotonic/simple fallback;
   - do not let one calibration method stop all retraining.

14. Fix metrics after retraining:
   - recompute calibrated confidences using current calibrator before MAE/ECE;
   - include confidence == 1.0 in final ECE bin.

15. Make scipy optional or lazy:
   - distribution shift detection should not break module import if scipy is unavailable;
   - use a simple rolling hit-rate fallback.

16. Fix save/load:
   - use trusted/safe output path for save too;
   - catch OSError/PermissionError/FileNotFoundError/ImportError around joblib.
```

---

## 11.8. Priority list for `calibration`

### P0 / must fix

- README/programmatic API mismatch.
- `CalibrationEngine.__init__` Optuna behavior vs unreachable `run_calibration()` fallback.
- SQL interpolation in `load_real_data`.
- features/targets not aligned.
- `_evaluate_on_synthetic()` does not use model and uses `abs(sharpe)`.
- unused RL hyperparams in RF objective.

### P1 / high priority

- export `AdaptiveConfidenceCalibrator`.
- numeric feature selection / leakage column filtering.
- implemented metrics must match README.
- save all trials / reproducibility metadata.
- input validation in `AdaptiveConfidenceCalibrator`.
- one-class outcome handling for Platt scaling.
- recompute ECE/MAE after retraining.

### P2 / cleanup

- lazy scipy import.
- trusted save path.
- docs distinguishing `src/calibration` and `dean_os` calibration.
- bridge with `src/models/ensemble/calibration`.
- create actual CLI/runner if calibration is intended to be active.

---

## 11.9. Summary for `calibration`

`calibration` — корисна папка, але зараз вона більше схожа на **потенційний tuning/calibration module**, ніж на активно підключений production-компонент.

Основні проблеми:

- майже немає реальної залученості;
- README описує API/entrypoints, які не збігаються з кодом;
- Optuna fallback у `run_calibration()` недосяжний через raise в `__init__`;
- synthetic score не залежить від model і ще й бере `abs(sharpe)`;
- RF proxy objective тюнить RL params, які не використовуються;
- features/targets не вирівнюються;
- confidence calibrator хороший, але ізольований.

Рекомендований напрям:

1. Спочатку зробити модуль чесним і безпечним: docs/API/Optuna/SQL/alignment.
2. Потім розділити режими `rf_proxy` і `dean_rl`.
3. Потім підключити `AdaptiveConfidenceCalibrator` до реального final confidence.
4. І тільки після цього використовувати calibration results як basis для production model tuning.

---

# 12. Audit: `advanced_engine.py`

## 12.1. Загальний стан

Наданий файл `advanced_engine.py` — це той самий файл, який у проєкті лежить як:

```text
src/backtesting/advanced/advanced_engine.py
```

Це **важливий файл**, бо він не ізольований і не мертвий. Він прямо використовується у кількох місцях:

```text
src/pipeline/stages/stage_7_evaluation.py
src/pipeline/stages/evaluation/backtest_analyzer.py
src/main/modes/backtest.py
src/trading/virtual_portfolio.py
```

Тобто це частина Stage 7 / evaluation / backtesting / paper trading.

Синтаксично файл компілюється. Але імпорт модуля в поточній структурі падає.

Критичний висновок: **цей файл зараз не production-ready і, скоріше за все, ламає Stage 7 / backtest mode / VirtualPortfolio import.** Найбільша проблема — змішані старі та нові імпорти плюс дублювання класів у самому файлі.

---

## 12.2. Реальна залученість

### Де використовується

`AdvancedBacktestEngine` імпортується в:

```text
src/pipeline/stages/stage_7_evaluation.py
```

Там він створюється приблизно так:

```python
from src.backtesting.advanced.advanced_engine import AdvancedBacktestEngine
self.backtester = AdvancedBacktestEngine(self.config_manager)
```

`BacktestAnalyzer` викликає:

```python
self.backtester.run_comprehensive_backtest(price_pivot, signal_pivot)
```

`BacktestMode` імпортує:

```python
from src.backtesting.advanced.advanced_engine import BiasDetector, WalkForwardOptimizer
```

`VirtualPortfolio` імпортує:

```python
from src.backtesting.advanced.advanced_engine import TransactionCostModel
```

Отже, якщо `advanced_engine.py` не імпортується, можуть падати:

- Stage 7 evaluation;
- backtest mode;
- virtual portfolio / paper trading;
- evaluation backtest analyzer.

---

# 12.3. Критичні проблеми

## 12.3.1. P0: модуль не імпортується через неправильний шлях `src.core.config.config_manager`

У файлі спочатку є правильний імпорт:

```python
from src.config.unified_config_manager import get_current_config
```

А потім нижче, після вже оголошених класів, файл знову імпортує:

```python
from src.core.config.config_manager import get_current_config
```

У поточному дереві немає:

```text
src/core/config/config_manager.py
```

Тому імпорт падає:

```text
ModuleNotFoundError: No module named 'src.core.config'
```

Це P0, бо файл не завантажується взагалі.

Правильний фікс:

```python
from src.config.unified_config_manager import get_current_config
```

і прибрати другий неправильний імпорт.

---

## 12.3.2. P0: файл має дубльовані класи, які потім перезаписуються імпортами

У верхній частині файлу оголошені локальні класи:

```text
TransactionCostModel
BiasDetector
WalkForwardOptimizer
```

А потім нижче файл робить:

```python
from src.algorithms.bias_detector import BiasDetector
from src.algorithms.transaction_cost_model import TransactionCostModel
from src.algorithms.walk_forward_optimizer import WalkForwardOptimizer
```

Це перезаписує локальні класи в namespace модуля.

Наслідок:

- локальні `TransactionCostModel`, `BiasDetector`, `WalkForwardOptimizer` фактично стають мертвим кодом для всього, що йде після цих імпортів;
- `AdvancedBacktestEngine` використовує вже не локальні класи, а імпортовані з `src.algorithms`;
- `VirtualPortfolio`, який імпортує `TransactionCostModel` з `advanced_engine.py`, фактично може отримувати не локальну модель, а re-export із `src.algorithms.transaction_cost_model`.

Це дуже небезпечно, бо API цих класів відрізняється.

Рішення:

- або видалити локальні дублікати і чесно re-export-ити `src.algorithms.*`;
- або прибрати імпорти з `src.algorithms.*` і використовувати локальні класи;
- краще: зробити `src.backtesting.advanced.advanced_engine` orchestration wrapper, а всі алгоритмічні класи тримати в `src.algorithms`.

---

## 12.3.3. P0: `TransactionCostModel` API mismatch

Локальний клас у цьому файлі має метод:

```python
calculate_execution_costs(
    trade_value: float,
    daily_volume: float,
    volatility: float,
    order_size_pct: float | None = None,
) -> dict[str, float]
```

А `src.algorithms.transaction_cost_model.TransactionCostModel` має іншу сигнатуру:

```python
calculate_execution_costs(
    trade_value: float,
    daily_volume: float = 1000000.0,
) -> float
```

Через re-import локальний клас перезаписується класом із `src.algorithms`.

Наслідки:

### У `AdvancedBacktestEngine._analyze_transaction_costs()`

Код викликає:

```python
cost_estimate = self.cost_model.calculate_execution_costs(
    100000,
    prices[col].mean() * 1000,
    avg_volatility
)
```

Якщо використовується `src.algorithms.TransactionCostModel`, буде:

```text
TypeError: calculate_execution_costs() takes from 2 to 3 positional arguments but 4 were given
```

Навіть якщо виклик пройде, `src.algorithms.TransactionCostModel` повертає `float`, а далі код очікує dict:

```python
cost_estimate['total']
```

Це дасть:

```text
TypeError: 'float' object is not subscriptable
```

### У `VirtualPortfolio`

`VirtualPortfolio` викликає:

```python
calculate_execution_costs(
    trade_value=trade_value,
    daily_volume=daily_volume,
    volatility=volatility,
    order_size_pct=order_size_pct
)
```

Якщо він отримує re-export із `src.algorithms`, буде:

```text
TypeError: unexpected keyword argument 'volatility'
```

Це критично для paper trading / virtual portfolio.

Рекомендація:

- стандартизувати один API transaction cost model.
- Найкраще зробити canonical `src.algorithms.transaction_cost_model.TransactionCostModel` таким, щоб він повертав dict і підтримував `volatility`/`order_size_pct`.
- Або створити adapter:

```python
class BacktestTransactionCostAdapter:
    def calculate_execution_costs(...) -> dict[str, float]:
        ...
```

---

## 12.3.4. P0: `BacktestMode` викликає локальний WalkForward API, але отримує інший клас

`BacktestMode` робить:

```python
from src.backtesting.advanced.advanced_engine import WalkForwardOptimizer
```

і далі викликає:

```python
walk_forward.walk_forward_optimization(
    data=historical_data,
    optimization_func=optimization_function,
    in_sample_months=in_sample_months,
    out_sample_months=out_sample_months,
)
```

Але через re-import `advanced_engine.WalkForwardOptimizer` — це фактично `src.algorithms.walk_forward_optimizer.WalkForwardOptimizer`.

Його метод має інший API:

```python
walk_forward_optimization(
    data,
    param_space=None,
    anchor_type="expanding",
    train_size=252,
    test_size=63,
    optimization_func=None,
    metric="sharpe",
)
```

Він не приймає:

```text
in_sample_months
out_sample_months
```

Наслідок:

```text
TypeError: got an unexpected keyword argument 'in_sample_months'
```

Це ламає walk-forward mode.

Рішення:

- або `BacktestMode` має імпортувати `src.algorithms.walk_forward_optimizer.WalkForwardOptimizer` і використовувати його API;
- або `advanced_engine.py` має експортувати compatibility wrapper зі старою сигнатурою;
- але не можна змішувати два API під одним ім’ям.

---

## 12.3.5. Є два AdvancedBacktestEngine у проєкті

У проєкті є:

```text
src/backtesting/advanced/advanced_engine.py
src/algorithms/advanced_backtest_engine.py
```

Обидва мають схожу роль:

- backtest;
- transaction costs;
- bias detection;
- walk-forward;
- metrics.

Це дублювання.

Ризик:

- Stage 7 використовує один engine;
- інші місця можуть використовувати інший;
- фікс в одному не фіксить другий;
- метрики/витрати/логіка симуляції можуть відрізнятись.

Рекомендація:

- визначити canonical engine.
- Я б залишив:
  - `src.algorithms.*` як низькорівневі алгоритми;
  - `src.backtesting.advanced.advanced_engine.AdvancedBacktestEngine` як orchestration facade;
  - `src.algorithms.advanced_backtest_engine` або deprecated wrapper, або merge into backtesting facade.

---

## 12.3.6. `_calculate_win_rate()` рахує win-rate неправильно

У `run_comprehensive_backtest()`:

```python
daily_returns = returns_series.pct_change(fill_method=None).dropna()
...
'win_rate': float(self._calculate_win_rate(daily_returns))
```

А `_calculate_win_rate()` робить ще раз:

```python
daily_returns = returns.pct_change(fill_method=None).dropna()
wins = (daily_returns > 0).sum()
```

Тобто метод отримує вже daily returns, але повторно рахує `pct_change()` від returns.

Це дає win-rate по “зміні returns”, а не по прибуткових днях.

Правильний варіант:

```python
def _calculate_win_rate(self, returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return float((clean > 0).mean()) if len(clean) else 0.0
```

Або передавати equity curve і явно назвати:

```python
_calculate_win_rate_from_equity(equity)
```

---

## 12.3.7. `BiasDetector` у верхній частині аналізує future prices, а не returns

Локальний `BiasDetector.detect_look_ahead_bias()` робить:

```python
signals[common_cols].corrwith(future_prices[common_cols].shift(-lag_periods))
```

Тобто порівнює сигнал із майбутнім рівнем ціни, а не майбутньою доходністю.

Для lookahead bias краще порівнювати з:

```python
future_returns = prices.pct_change(...).shift(-lag)
```

В `src.algorithms.bias_detector.BiasDetector` це зроблено краще. Але локальний клас все одно є в файлі й вводить в оману.

Рішення:

- прибрати локальний дубль;
- або привести його до returns-based логіки.

---

## 12.3.8. `_analyze_transaction_costs()` переоцінює кількість угод

Код:

```python
total_trades = (signals != signals.shift()).sum().sum()
trades = signals[col] != signals[col].shift()
n_trades = trades.sum()
```

Проблеми:

- перший рядок майже завжди вважається trade, бо `signal != NaN`;
- зміна `NaN -> HOLD` або invalid -> 0 може рахуватись як угода;
- не враховується actual position change після normalization;
- не враховується order size;
- для multi-asset weights краще рахувати turnover.

Рекомендація:

- рахувати turnover після `_prepare_signal_positions`;
- перший рядок або пропускати, або вважати initial allocation окремо;
- costs повинні бути пов’язані з реальним turnover у `_simulate_returns`.

---

## 12.3.9. `_analyze_transaction_costs()` оцінює кожну угоду як $100000

```python
cost_estimate = self.cost_model.calculate_execution_costs(
    100000,
    prices[col].mean() * 1000,
    avg_volatility
)
```

Це грубий placeholder.

Проблема:

- не використовує `initial_capital`;
- не використовує actual position weight;
- не використовує actual price/quantity;
- `daily_volume = mean price * 1000` — вигаданий proxy.

Це може бути прийнятно як rough report, але не як production transaction analysis.

Рекомендація:

- використовувати turnover * portfolio_value;
- якщо volume немає, позначати `liquidity_unknown=True`;
- не видавати placeholder costs як точні.

---

## 12.3.10. `_simulate_returns()` використовує fillna(0.0) для missing returns active positions

Код:

```python
missing_position_returns = asset_returns.isna() & lagged_weights.ne(0.0)
...
weighted_returns = weighted_returns.fillna(0.0)
```

Він логить warning, що добре. Але фінансово це може занижувати risk:

- якщо ціна відсутня для активної позиції, нульова return не завжди правильна;
- може приховати data gap;
- max drawdown і volatility можуть бути занижені.

Рекомендація:

- policy має бути configurable:
  - `missing_active_return_policy="error" | "skip_day" | "zero_with_warning" | "ffill_price"`
- для audit/evaluation default краще `error` або `skip_day`, а не zero.

---

## 12.3.11. `_prepare_signal_positions()` робить ffill сигналів

```python
aligned = aligned.apply(pd.to_numeric, errors='coerce').ffill()
```

Для trading positions це може бути нормальна логіка: сигнал діє до наступного сигналу.

Але це треба явно документувати як position state, не raw signal. Якщо сигнал має значити only today action, то ffill буде неправильним.

Рекомендація:

- перейменувати або документувати:
  - input `signals` are desired positions, not orders;
- або додати режим:
  - `signal_mode="position"` vs `signal_mode="orders"`.

---

## 12.3.12. `risk_metrics` у звіті завжди порожній

У report створюється:

```python
'risk_metrics': {}
```

Але далі не заповнюється.

Це mismatch між обіцяним report schema і фактичним результатом.

Рекомендація:

- заповнити:
  - volatility;
  - downside volatility;
  - VaR;
  - CVaR;
  - exposure;
  - turnover;
  - max consecutive losses;
- або прибрати порожній ключ до реалізації.

---

## 12.3.13. Header обіцяє Statistical Significance Testing, але його немає

Docstring каже:

```text
Statistical Significance Testing
```

У файлі немає:

- p-values;
- bootstrap CI;
- t-test;
- deflated Sharpe;
- probabilistic Sharpe ratio;
- multiple testing control.

Рекомендація:

- або додати статистичний блок;
- або прибрати з docstring, щоб не вводити в оману.

---

## 12.3.14. `run_comprehensive_backtest()` ловить занадто вузькі exceptions

Функція ловить:

```text
ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError
```

А реальні runtime issues можуть бути:

- `ImportError`;
- `OSError`;
- custom project exceptions;
- pandas errors;
- `RuntimeError`.

Для top-level backtest engine можна ловити ширше і повертати structured failure:

```python
except Exception as e:
    return {
        "status": "failed",
        "error_type": type(e).__name__,
        "error": str(e),
    }
```

Але не приховувати P0 під простим `{"error": ...}`.

---

## 12.3.15. `kwargs` у `run_comprehensive_backtest()` не використовуються

Параметр:

```python
**kwargs
```

є, але не використовується.

Це не критично, але створює ілюзію, що engine приймає додаткові options.

Рекомендація:

- або прибрати;
- або підтримати:
  - `initial_capital`;
  - `frequency`;
  - `cost_policy`;
  - `missing_data_policy`;
  - `signal_mode`.

---

# 12.4. Що працює добре

1. `_simulate_returns()` використовує `positions.shift(1)`, тобто сигнал не застосовується до того ж дня. Це правильно для temporal safety.
2. `_prepare_signal_positions()` нормалізує exposure, тобто portfolio не стає випадково 500% long через кілька BUY сигналів.
3. Є підтримка строкових сигналів:
   - BUY/LONG
   - SELL/SHORT
   - HOLD/FLAT/CLOSE
4. Є warning при missing returns для активних позицій.
5. Є приблизна модель turnover costs у `_simulate_returns()`.
6. Є alert, якщо bias detector знаходить lookahead.
7. `PerformanceMetricsMixin` використовується для Sharpe/drawdown, що добре, якщо прибрати дублювання.

---

# 12.5. Що дати Codex по `advanced_engine.py`

```text
Deep-fix src/backtesting/advanced/advanced_engine.py without changing public architecture.

Tasks:

1. Fix import failure:
   - remove `from src.core.config.config_manager import get_current_config`;
   - use only `from src.config.unified_config_manager import get_current_config`.

2. Remove class shadowing:
   - do not define TransactionCostModel/BiasDetector/WalkForwardOptimizer locally and then overwrite them via imports;
   - choose one canonical implementation.
   - Preferred architecture:
     src.algorithms.* = low-level algorithms
     src.backtesting.advanced.advanced_engine = orchestration facade.

3. Standardize TransactionCostModel API:
   - one canonical calculate_execution_costs signature;
   - must support VirtualPortfolio use case;
   - return either dict everywhere or float everywhere, not mixed.
   - Recommended return: dict with commission/spread/market_impact/slippage/total/total_pct.

4. Fix BacktestMode walk-forward compatibility:
   - either provide compatibility wrapper accepting in_sample_months/out_sample_months;
   - or update BacktestMode to use src.algorithms.walk_forward_optimizer API.

5. Fix win-rate calculation:
   - do not call pct_change() on already computed daily returns;
   - count positive returns directly.

6. Fix transaction analysis:
   - calculate trade count from position turnover;
   - exclude initial NaN transition unless explicitly treated as initial allocation;
   - do not estimate every trade as fixed $100000;
   - use portfolio value and actual turnover.

7. Add missing/risk metrics:
   - fill risk_metrics or remove placeholder;
   - add volatility, VaR/CVaR, turnover, exposure, downside volatility.

8. Clarify signal semantics:
   - document whether signals are desired positions or one-day orders;
   - if both needed, add signal_mode="position"|"orders".

9. Make missing active returns policy configurable:
   - error / skip_day / zero_with_warning / ffill_price;
   - default for evaluation should be conservative.

10. Remove or implement Statistical Significance Testing:
   - either add bootstrap CI / probabilistic Sharpe / p-values;
   - or remove from module docstring.

11. Improve top-level error reporting:
   - return structured status failed with error_type;
   - do not hide P0 as generic {"error": "..."}.

12. Add tests:
   - module import test;
   - VirtualPortfolio TransactionCostModel call;
   - BacktestMode walk_forward_optimization compatibility;
   - win_rate correctness;
   - no lookahead: positions must be shifted by 1;
   - cost model return shape.
```

---

# 12.6. Priority list for `advanced_engine.py`

## P0 / must fix first

- Wrong import `src.core.config.config_manager`.
- Local classes overwritten by imports from `src.algorithms`.
- TransactionCostModel API mismatch.
- BacktestMode walk-forward signature mismatch.
- Duplicate backtesting engines / unclear canonical implementation.

## P1 / high priority

- `_calculate_win_rate()` double `pct_change`.
- transaction count / fixed $100000 cost estimate.
- missing active returns fillna zero policy.
- risk_metrics empty.
- top-level error handling too weak.

## P2 / cleanup

- unused `kwargs`.
- docstring promises statistical testing but no implementation.
- clarify signal semantics.
- consolidate with `src.algorithms/advanced_backtest_engine.py`.

---

# 12.7. Summary for `advanced_engine.py`

`advanced_engine.py` — важливий файл, але зараз він має **критичні інтеграційні проблеми**.

Найгірше:

1. Він не імпортується через старий шлях `src.core.config.config_manager`.
2. У ньому є локальні класи, які потім перезаписуються імпортами.
3. Через це `TransactionCostModel` має невідповідний API для `VirtualPortfolio` і самого `AdvancedBacktestEngine`.
4. `BacktestMode` очікує старий walk-forward API, але фактично отримує інший optimizer.
5. Є дублювання з `src/algorithms/advanced_backtest_engine.py`.

Тобто цей файл треба стабілізувати раніше, ніж серйозно довіряти Stage 7 evaluation/backtest результатам.

Найкращий архітектурний напрям: зробити `src.backtesting.advanced.advanced_engine` фасадом, який використовує canonical low-level алгоритми з `src.algorithms`, але не дублює і не перезаписує їх. Після цього додати тести на import, cost model API, walk-forward compatibility і правильний win-rate.

---

# 13. Audit: `cli`

## 13.1. Загальний стан

`cli` — це шар запуску hybrid pipeline / local pipeline / Colab prepare-continue flow / calibration mode.

У наданій папці є:

```text
cli/
  argument_parser.py
  argument_validator.py
  batch_manager.py
  pipeline_data_loader.py
  pipeline_executor.py
```

Синтаксично всі Python-файли компілюються. Імпорт модулів `src.cli.*` у поточному дереві проходить.

Але функціонально є кілька важливих проблем:

- частина CLI-логіки існує, але немає видимого active entrypoint, який реально використовує `create_argument_parser()` і `ArgumentValidator`;
- `calibrate` mode оголошений у parser/executor, але `orchestrator.run_calibration()` у проєкті не знайдений;
- `BatchManager.generate_batch_name()` не використовує `--batch-name` і не санітизує raw args перед формуванням batch name;
- `PipelineDataLoader` дублює частину `PipelineExecutor`, але майже не використовується;
- `continue` mode сильно залежить від `orchestrator.config.output_dir`, а `args.batch_name` використовується не всюди;
- є дублювання між `src/cli/pipeline_executor.py` і `src/pipeline/hybrid/pipeline_executor.py`.

Головний висновок: **папка корисна, але зараз це радше набір helper-ів, ніж завершений CLI-контур.** Найбільше ризику в `continue`/`calibrate` режимах.

---

## 13.2. Залученість у проєкті

### Реально використовується

`PipelineExecutor` із `src.cli.pipeline_executor` використовується у:

```text
dean_os/pipeline_adapter.py
```

Там DEAN-OS adapter викликає:

```python
from src.cli.pipeline_executor import PipelineExecutor

await PipelineExecutor.execute_prepare_mode(...)
await PipelineExecutor.execute_full_mode(...)
```

Тобто `PipelineExecutor` реально залучений у DEAN-OS bridge.

### Слабо або неочевидно залучене

Пошук показує, що:

```text
create_argument_parser
ArgumentValidator
PipelineDataLoader
```

майже не використовуються поза власними файлами.

Це означає:

- parser є;
- validator є;
- executor є;
- але видимого CLI runner/main dispatcher, який усе це зшиває, у наданому дереві не видно.

Можливо, він був у старому `run_hybrid_pipeline.py`, але в поточному дереві такого файлу не знайдено. Це узгоджується з попереднім аудитом `calibration`, де README посилалась на `run_hybrid_pipeline.py`, але active entrypoint не було видно.

---

# 13.3. Критичні проблеми

## 13.3.1. P0/P1: `calibrate` mode оголошений, але `orchestrator.run_calibration()` не знайдений

У `argument_parser.py` mode має:

```python
choices=['local', 'full', 'prepare', 'light', 'continue', 'calibrate']
```

У `PipelineExecutor` є:

```python
async def execute_calibrate_mode(orchestrator, args):
    results = await orchestrator.run_calibration(...)
```

Але у поточному дереві пошук `run_calibration` показує тільки:

```text
src/calibration/calibration_engine.py
src/cli/pipeline_executor.py
```

Тобто у `HybridOrchestrator` / pipeline orchestrator метод `run_calibration()` не знайдений.

Наслідок:

```text
AttributeError: 'HybridOrchestrator' object has no attribute 'run_calibration'
```

Рекомендація:

- або додати `HybridOrchestrator.run_calibration()` як wrapper над `src.calibration.CalibrationEngine`;
- або змінити `execute_calibrate_mode()` так, щоб він напряму створював `CalibrationEngine`;
- або прибрати `calibrate` з parser, якщо mode ще не реалізований.

Це треба звʼязати з фіксами в `src/calibration`.

---

## 13.3.2. P1: немає видимого active CLI entrypoint

Є:

```text
argument_parser.py
argument_validator.py
pipeline_executor.py
batch_manager.py
```

Але немає очевидного файлу, який робить:

```python
parser = create_argument_parser()
args = parser.parse_args()
ArgumentValidator.validate_arguments(args, config_manager)
...
await PipelineExecutor.execute_*()
```

Пошук по `create_argument_parser` показує тільки сам `argument_parser.py`.

Наслідок:

- CLI helper-и можуть бути готові, але не використовуються;
- README/команди типу `python run_hybrid_pipeline.py --mode ...` можуть бути legacy;
- Codex може фіксити helper-и, але реальна команда все одно не працюватиме.

Рекомендація:

- або відновити `run_hybrid_pipeline.py`;
- або створити `src/cli/main.py`;
- або явно позначити CLI як library-only helper для DEAN adapter.

---

## 13.3.3. P1: `BatchManager.generate_batch_name()` не використовує `args.batch_name`

У parser є:

```python
--batch-name
```

Але `BatchManager.generate_batch_name(args)`:

- якщо є test params — генерує `test_...`;
- якщо немає — повертає `"main_database"`;
- у `continue` шукає існуючі batch dirs;
- але custom `args.batch_name` не враховує.

Наслідок:

- користувач може передати `--batch-name`, але batch name буде інший;
- `continue` mode може дивитися не туди;
- документація й поведінка CLI можуть розходитись.

Рекомендація:

```python
if getattr(args, "batch_name", None) and args.mode != "continue":
    return BatchManager.sanitize_path_input(args.batch_name)
```

А для `continue` — `--batch-name` має бути explicit target, а не тільки trigger для пошуку.

---

## 13.3.4. P1/P0: batch name формується із raw user input без sanitize

У `BatchManager._generate_test_batch_name()`:

```python
parts.append(f"ticker_{args.test_ticker}")
parts.append(target_name)
parts.append(f"model_{args.test_model}")
```

Санітизація є окремим методом:

```python
sanitize_path_input(...)
```

але при генерації batch name вона не використовується.

Наслідок:

- `test_ticker`, `test_target`, `test_model` можуть містити `/`, `..`, control chars, пробіли;
- якщо batch name потім використовується як шлях, можливий path traversal або invalid path;
- це особливо небезпечно для `--batch-name`, якщо його теж почнуть використовувати.

Рекомендація:

- санітизувати кожну частину перед join;
- whitelist для ticker/model/target:
  - letters;
  - numbers;
  - `_`;
  - `-`;
  - `.`;
- заборонити `/`, `\`, `..`, null/control chars.

---

## 13.3.5. P1: `BatchManager._handle_continue_mode()` вибирає `max(existing_batches)` лексикографічно

```python
return max(existing_batches)
```

Це не завжди останній batch.

Приклади:

```text
test_AMD_9
test_AMD_10
```

Лексикографічно може вибрати не той.

Рекомендація:

- сортувати за `mtime`;
- або парсити timestamp у назві;
- або вимагати explicit `--batch-name` для continue mode.

---

## 13.3.6. P1: `continue` mode залежить від `orchestrator.config.output_dir`, а не явно від `args.batch_name`

У `execute_continue_mode()`:

```python
val_report = PipelineExecutor._validate_batch_contract(orchestrator)
```

А `_validate_batch_contract()`:

```python
validate_batch_dir(orchestrator.config.output_dir)
```

Далі `_load_core_continue_data()`:

```python
batch_dir = orchestrator.config.output_dir
batch_name = getattr(args, 'batch_name', 'unknown')
colab_results = orchestrator.load_colab_results(batch_name)
features_path = batch_dir / FEATURES_FILE
targets_path = batch_dir / TARGETS_FILE
```

Тобто:

- `colab_results` завантажуються по `batch_name`;
- features/targets беруться з `orchestrator.config.output_dir`;
- contract validation теж бере `orchestrator.config.output_dir`.

Якщо orchestrator не був створений саме для цього batch_name, можна отримати mismatch:

- Colab results з одного batch;
- features/targets з іншої папки.

Рекомендація:

- один source of truth:
  - або `batch_dir = orchestrator.resolve_batch_dir(args.batch_name)`;
  - або orchestrator ініціалізується із `batch_name`;
  - або `execute_continue_mode()` явно отримує `batch_dir`.
- після цього всі файли беруться з одного batch dir.

---

## 13.3.7. P1: `PipelineExecutor._reconstruct_data_from_db()` може валити continue mode через optional news/macro fallback

У `_load_extra_continue_data()`:

```python
if news_data is None or economic_data is None:
    news_data, economic_data = PipelineExecutor._reconstruct_data_from_db(...)
```

А `_reconstruct_data_from_db()` при помилці:

```python
logger.exception(...)
raise
```

Тобто якщо fallback reconstruction не вдався, весь continue mode може впасти, хоча news/economic data можуть бути optional або recoverable.

Рекомендація:

- якщо news/economic required для final stages — повертати structured failure;
- якщо optional — не `raise`, а warning і повернути поточні `None`;
- явно вказати, що саме required для stage 4-7.

---

## 13.3.8. P1: `_safe_load_parquet()` і `PipelineDataLoader.load_parquet_from_path()` ловлять занадто вузькі exceptions

Зараз ловляться:

```text
ValueError
TypeError
AttributeError
KeyError
ZeroDivisionError
```

А `pd.read_parquet()` часто кидає:

- `ImportError` / missing `pyarrow` or `fastparquet`;
- `OSError`;
- `FileNotFoundError`;
- `PermissionError`;
- `pyarrow.lib.ArrowInvalid`;
- `pandas.errors.EmptyDataError`.

Наслідок:

- пошкоджений parquet або відсутній engine може валити CLI;
- замість controlled failure буде crash.

Рекомендація:

```python
except Exception as e:
    logger.exception(...)
    return None
```

На рівні file loader це нормально, бо далі є validation.

---

## 13.3.9. P1: `_reconstruct_category()` дедуплікує news/macro тільки по `timestamp`

У `PipelineExecutor`:

```python
deduplicate_dataframe(pd.concat(dfs, ignore_index=True), subset_cols=['timestamp'])
```

Це небезпечно для news:

- багато новин можуть мати один timestamp;
- різні джерела можуть публікувати одночасно;
- macro rows теж можуть мати однаковий timestamp для різних indicators.

Наслідок:

- корисні rows можуть бути видалені.

`PipelineDataLoader.reconstruct_from_db()` має іншу логіку — dedup по всіх hashable columns. Тобто є inconsistent behavior.

Рекомендація:

- стандарт:
  - news: `hash` або `url/title/published_at/source`;
  - macro: `indicator/timestamp`;
  - fallback: усі hashable columns.
- не дедуплікувати news тільки по timestamp.

---

## 13.3.10. P1/P2: `PipelineDataLoader` дублює логіку `PipelineExecutor` і майже не використовується

`PipelineDataLoader` має:

- `load_parquet_from_path`;
- `load_news_economic_data`;
- `load_from_db_fallback`;
- `reconstruct_from_db`.

А `PipelineExecutor` має майже те саме:

- `_safe_load_parquet`;
- `_load_news_data_with_fallbacks`;
- `_load_economic_data_with_fallbacks`;
- `_reconstruct_data_from_db`.

Пошук показує, що `PipelineDataLoader` майже не використовується.

Наслідок:

- два різні місця для одного процесу;
- фікс в одному не фіксить інше;
- різний dedup behavior.

Рекомендація:

- або PipelineExecutor має використовувати PipelineDataLoader;
- або видалити/позначити PipelineDataLoader як legacy;
- краще — зробити `PipelineDataLoader` canonical helper, а Executor тільки orchestration.

---

## 13.3.11. P1: `execute_prepare_mode()` не вимикає lineage tracking через `finally`

У `execute_prepare_mode()`:

```python
tracker = PipelineExecutor._enable_lineage_tracking_for_run()
...
PipelineExecutor._disable_lineage_tracking()
return result
```

Якщо `run_local_pipeline`, `prepare_colab_data` або lineage capture впаде до `_disable_lineage_tracking()`, tracking може залишитись увімкненим.

Рекомендація:

```python
tracker = None
try:
    tracker = ...
    ...
    return result
finally:
    PipelineExecutor._disable_lineage_tracking()
```

Також `enable_lineage_tracking` failure не обовʼязково має валити prepare mode. Це diagnostic feature.

---

## 13.3.12. P1/P2: `_get_timeframes()` очікує dict, але config може бути list

```python
yf_timeframes = collectors.get('yahoo_finance', {}).get('timeframes', {})
return list(yf_timeframes.keys()) if yf_timeframes else ['15m', '60m', '1d']
```

Якщо `timeframes` у config буде list:

```yaml
timeframes:
  - 15m
  - 60m
  - 1d
```

то буде:

```text
AttributeError: 'list' object has no attribute 'keys'
```

Рекомендація:

```python
if isinstance(yf_timeframes, dict):
    return list(yf_timeframes.keys())
if isinstance(yf_timeframes, list):
    return yf_timeframes
```

---

## 13.3.13. P1/P2: `ArgumentValidator` не валідовує `--tickers`

`ArgumentValidator` перевіряє тільки:

- `test_ticker`;
- `test_target`;
- `test_model`;
- mode;
- continue batch;
- numeric params;
- stages.

Але `--tickers` як список не перевіряється проти available tickers.

Наслідок:

- `--test-ticker BAD` буде error;
- `--tickers BAD1 BAD2` пройде validation.

Можливо, це свідомо, бо користувач може захотіти нові тикери не з config. Але тоді це треба документувати.

Рекомендація:

- або валідовувати `args.tickers`;
- або warning, якщо ticker не в config;
- або `--allow-external-tickers`.

---

## 13.3.14. P1/P2: `ArgumentValidator` не перевіряє `n_trials` і `epochs`

Parser має:

```python
--n-trials
--epochs
--max-iterations
```

Validator перевіряє тільки `max_iterations`.

Рекомендація:

- `n_trials >= 1`;
- `epochs is None or epochs >= 1`;
- можливо upper bounds, щоб випадково не запустити 100000 trials.

---

## 13.3.15. P2: `ArgumentValidator` логить user-controlled values без sanitization

Наприклад:

```python
errors.append("❌ Ticker '{}' not found...".format(args.test_ticker, ...))
```

Потім:

```python
logger.error("   %s", error)
```

Це краще, ніж f-string, але сам `error` може містити newline/control chars із `args.test_ticker`.

`PipelineExecutor` уже має `_sanitize()`, а validator — ні.

Рекомендація:

- додати спільний `sanitize_for_log`;
- застосувати до args values.

---

## 13.3.16. P2: `profile_execution` не логить час, якщо функція падає

Decorator:

```python
result = await func(...)
end_time = ...
logger.info(...)
return result
```

Якщо функція кидає exception, elapsed time не логується.

Рекомендація:

```python
try:
    return await func(...)
finally:
    logger.info(...)
```

Але не поглинати exception.

---

## 13.3.17. P2: `src/cli` не має `__init__.py`

У поточному дереві `src/cli` може імпортуватися як namespace package. Це працює в сучасному Python, але для частини tooling/packaging краще мати явний:

```text
src/cli/__init__.py
```

Не критично, бо імпорт `src.cli.pipeline_executor` проходить.

---

# 13.4. Дублювання / архітектурні розриви

## 13.4.1. Два PipelineExecutor

Є:

```text
src/cli/pipeline_executor.py
src/pipeline/hybrid/pipeline_executor.py
```

Вони мають різну роль:

- `src/cli/pipeline_executor.py` — static orchestration для modes: local/light/prepare/full/continue/calibrate.
- `src/pipeline/hybrid/pipeline_executor.py` — class-based executor для hybrid pipeline stages і saving outputs.

Назви однакові, але поведінка різна.

Ризик:

- плутанина для Codex/агентів;
- неправильний import;
- duplicate responsibility.

Рекомендація:

- перейменувати або задокументувати:
  - `CliPipelineExecutor`;
  - `HybridStageExecutor`;
- або експліцитно в README пояснити.

---

## 13.4.2. Два data loader / fallback implementations

```text
src/cli/pipeline_data_loader.py
src/cli/pipeline_executor.py internal static loaders
```

Рекомендація:

- зробити `PipelineDataLoader` єдиним implementation;
- `PipelineExecutor` має викликати його.

---

## 13.4.3. BatchManager vs DataBatchManager

Є:

```text
src/cli/batch_manager.py
src/pipeline/hybrid/data_batch_manager.py
```

Назви схожі, але відповідальність різна:

- CLI BatchManager генерує batch names;
- hybrid DataBatchManager, ймовірно, відповідає за data batch packaging/management.

Потрібна документація, щоб не переплутати.

---

# 13.5. Що працює добре

1. Parser чітко описує основні modes.
2. `continue` mode має contract validation через `validate_batch_dir`.
3. `PipelineExecutor` санітизує багато значень перед логуванням.
4. Є fallback пошук news/economic parquet у persistent paths.
5. Є спроба реконструювати missing news/macro із DuckDB.
6. `execute_prepare_mode()` інтегрує feature lineage tracking.
7. `execute_continue_mode()` правильно валідовує, що features/targets/colab_results існують перед final stages.
8. `_extract_target_columns()` шукає `target_` columns, що відповідає загальній target naming логіці.
9. `execute_calibrate_mode()` уже передбачений як режим, просто потребує реальної інтеграції з calibration engine.

---

# 13.6. Що дати Codex по `cli`

```text
Deep-fix src/cli without changing public architecture.

Tasks:

1. Add/restore active CLI entrypoint:
   - either run_hybrid_pipeline.py or src/cli/main.py;
   - wire create_argument_parser(), ArgumentValidator, BatchManager, PipelineExecutor dispatch.

2. Fix calibrate mode:
   - implement HybridOrchestrator.run_calibration() using src.calibration.CalibrationEngine;
   - or make PipelineExecutor.execute_calibrate_mode instantiate CalibrationEngine directly;
   - if not ready, remove calibrate from parser choices.

3. Fix BatchManager:
   - honor args.batch_name where appropriate;
   - sanitize every batch-name component before joining;
   - reject path traversal/control chars;
   - for continue mode, do not guess with max(existing_batches); require explicit batch_name or sort by mtime/timestamp.

4. Fix continue-mode batch source of truth:
   - resolve batch_dir from args.batch_name once;
   - use the same batch_dir for contract validation, features/targets, and colab results;
   - avoid mixing orchestrator.config.output_dir from one batch with args.batch_name from another.

5. Consolidate PipelineDataLoader:
   - make PipelineExecutor use PipelineDataLoader for parquet loading and DB fallback;
   - remove duplicated internal loader methods or mark PipelineDataLoader legacy.

6. Harden parquet loading:
   - catch ImportError/OSError/PermissionError/FileNotFoundError/ArrowInvalid/Exception in file loaders;
   - return None plus structured reason instead of crashing.

7. Fix DB reconstruction fallback:
   - do not raise for optional news/economic fallback unless final stages require it;
   - return current data with warning or structured failure;
   - standardize dedup keys:
     news: hash/url/title+published_at/source
     macro: indicator+timestamp
     fallback: all hashable columns.

8. Fix execute_prepare_mode lineage:
   - wrap lineage enable/disable in try/finally;
   - feature lineage failure should warn, not fail prepare mode.

9. Fix ArgumentValidator:
   - validate args.tickers or document allow-external behavior;
   - validate n_trials >= 1 and epochs >= 1;
   - sanitize user-controlled values before logging;
   - support warning instead of hard error for external tickers if desired.

10. Fix _get_timeframes:
   - support both dict and list config formats;
   - validate returned timeframes are non-empty strings.

11. Rename/document duplicate executors:
   - src.cli.pipeline_executor.PipelineExecutor => CliPipelineExecutor or document clearly;
   - src.pipeline.hybrid.pipeline_executor.PipelineExecutor => HybridStageExecutor or document clearly.

12. Add tests:
   - parser modes and numeric args;
   - continue mode requires batch name;
   - batch name sanitization and path traversal rejection;
   - calibrate mode behavior when run_calibration missing/available;
   - _get_timeframes with dict/list config;
   - parquet loader handles missing pyarrow/corrupt files;
   - DB reconstruction dedup does not drop distinct news with same timestamp.
```

---

# 13.7. Priority list for `cli`

## P0 / must fix

- `calibrate` mode calls missing `orchestrator.run_calibration()`.
- batch name path safety: raw test_ticker/test_target/test_model are used in generated batch names.
- continue mode can mix `args.batch_name` and `orchestrator.config.output_dir`.

## P1 / high priority

- no visible active CLI entrypoint / dispatcher.
- `BatchManager` ignores `--batch-name`.
- continue fallback DB reconstruction raises on optional data.
- parquet loaders catch too narrow exceptions.
- PipelineDataLoader duplicates PipelineExecutor logic and is unused.
- dedup by `timestamp` only in `_reconstruct_category`.
- lineage tracking not disabled in `finally`.

## P2 / cleanup

- validate `--tickers`, `--n-trials`, `--epochs`.
- `_get_timeframes` should support list and dict.
- add `src/cli/__init__.py`.
- rename/document duplicate PipelineExecutor classes.
- make `profile_execution` log elapsed time on failures.

---

# 13.8. Summary for `cli`

`cli` — корисна папка, але зараз вона виглядає як **частково зібраний command layer**, а не повністю завершений CLI.

Найважливіше:

1. `PipelineExecutor` реально використовується DEAN-OS adapter-ом.
2. Але parser/validator/BatchManager не видно в активному entrypoint.
3. `calibrate` mode наразі не підключений до реального orchestrator method.
4. `continue` mode може змішати batch_name і output_dir.
5. Batch name треба санітизувати до того, як він стане шляхом.
6. Data loading/fallback логіка продубльована й частково небезпечна через дедуп по timestamp.

Після фіксів `cli` може стати нормальним control surface для запуску:

- local pipeline;
- prepare for Colab;
- continue after Colab;
- light models;
- calibration;
- DEAN-OS triggered pipeline actions.

---

# 14. Audit: `colab`

## 14.1. Загальний стан

`colab` — це модуль для **Colab-side execution**, тобто код, який має виконуватися не в локальному production pipeline, а у Google Colab під час важкого тренування моделей.

У наданій папці є:

```text
colab/
  README.md
  __init__.py
  config/
    config_loader.py
    training_config.py
  environment/
    colab_environment.py
    setup.py
  memory/
    memory_monitor.py
  models/
    architectures.py
    model_factory.py
    sklearn_fallback.py
    torch_models.py
  utils/
    batch_size.py
    data_signature.py
    metrics.py
    retry.py
    utils.py
```

Синтаксично всі Python-файли компілюються.

Головний висновок: **модуль корисний, але зараз це радше автономний Colab toolkit, а не повністю інтегрована частина pipeline.** README прямо каже, що локальний pipeline його не імпортує. Це правильно для ізоляції. Але є критична помилка в `src.colab.utils`, дублювання файлів, path-safety ризики, а також моделі з назвами LSTM/GRU/Transformer фактично працюють із sequence length = 1, тобто не є повноцінними часовими моделями.

---

## 14.2. Залученість у проєкті

### Локальна залученість слабка

Пошук по проєкту показує:

```text
src.colab
ColabEnvironment
RuntimeConfigLoader
MemoryMonitor
```

здебільшого згадуються тільки всередині самої папки `src/colab`.

README прямо пише:

```text
This module is intentionally not imported by run_hybrid_pipeline.py or any stage.
It is copied/uploaded to Colab and executed there.
```

Тобто низька локальна залученість — не обов’язково проблема. Це може бути правильна архітектура: локальний pipeline готує batch, Colab module тренує, потім локальний `continue` підхоплює результати.

### Зв’язок із hybrid pipeline непрямий

У pipeline є:

```text
src/pipeline/hybrid/colab_manager.py
src/pipeline/hybrid/colab_workflow_manager.py
```

Вони готують batch folder, інструкції й очікують, що Colab щось натренує та збереже `colab_results_summary.json`.

Але `src/colab` не має видимого головного runner-а/notebook script у наданій папці. Тобто модулі є, але точка входу для повного Colab training flow не очевидна.

---

# 14.3. Критичні проблеми

## 14.3.1. P0/P1: `src.colab.utils` не імпортується через неправильний `__init__.py`

Файл:

```text
colab/utils/__init__.py
```

робить:

```python
from .utils import (
    compute_data_signature,
    compute_metrics,
    find_latest_checkpoint,
    get_optimal_batch_size,
    load_checkpoint,
    retry_on_timeout,
    save_checkpoint,
)
```

Але `retry_on_timeout` знаходиться не в `utils.py`, а в:

```text
colab/utils/retry.py
```

Наслідок:

```text
ImportError: cannot import name 'retry_on_timeout' from 'src.colab.utils.utils'
```

Це означає, що:

```python
import src.colab.utils
from src.colab import retry_on_timeout
```

може падати.

Правильний фікс:

```python
from .utils import (
    compute_data_signature,
    compute_metrics,
    find_latest_checkpoint,
    get_optimal_batch_size,
    load_checkpoint,
    save_checkpoint,
)
from .retry import retry_on_timeout
```

Або краще розвести імпорти по відповідних файлах:

```python
from .batch_size import get_optimal_batch_size
from .data_signature import compute_data_signature
from .metrics import compute_metrics
from .retry import retry_on_timeout
from .utils import save_checkpoint, load_checkpoint, find_latest_checkpoint
```

Це критично, бо top-level `src.colab.__getattr__` для `retry_on_timeout`, `compute_metrics`, `save_checkpoint` делегує саме в `src.colab.utils`.

---

## 14.3.2. P1: дублювання `colab_environment.py` і `setup.py`

У папці є два майже однакові файли:

```text
environment/colab_environment.py
environment/setup.py
```

В обох є клас:

```python
class ColabEnvironment
```

Код практично дублюється:

- пошук `PROJECT_PATH`;
- додавання `src` в `sys.path`;
- `setup_paths`;
- Google Drive mount;
- `setup_batch_directory`.

`environment/__init__.py` експортує:

```python
from .colab_environment import ColabEnvironment
```

Отже `setup.py` виглядає як старий дубль або legacy.

Ризик:

- фікс в одному файлі не фіксить другий;
- Codex/людина може імпортувати не той `ColabEnvironment`;
- поведінка розійдеться.

Рекомендація:

- залишити `colab_environment.py` як canonical;
- `setup.py` зробити compatibility shim:

```python
from .colab_environment import ColabEnvironment
```

або позначити як deprecated.

---

## 14.3.3. P1: `RuntimeConfigLoader` використовує raw `sys.argv` і не санітизує `--batch-name`

У `config_loader.py`:

```python
if '--batch-name' in sys.argv:
    idx = sys.argv.index('--batch-name')
    return sys.argv[idx + 1]
```

Потім batch name прямо потрапляє в шлях:

```python
batch_dir = self.project_path / 'data' / 'colab' / 'accumulated' / batch_name
```

Немає захисту від:

- `../`;
- `/`;
- `\`;
- control chars;
- дуже довгих значень;
- `target_target_` нормалізація є, але не security normalization.

Рекомендація:

- приймати `batch_name` явно як параметр методу або через argparse;
- використовувати той самий sanitizer, що й у CLI BatchManager;
- заборонити path traversal;
- після resolve перевіряти, що batch_dir всередині `data/colab/accumulated`.

---

## 14.3.4. P1: `_load_config_file()` має неправильну перевірку безпеки шляху

У `RuntimeConfigLoader._load_config_file()`:

```python
config_path = config_path.resolve()
if not str(config_path).startswith(str(Path.cwd().resolve())):
    raise ValueError('Config path outside working directory not allowed')
```

Проблеми:

1. Якщо `project_path` не дорівнює `cwd`, валідний `config.json` у `project_path` може бути відхилений.
2. `startswith` для шляхів небезпечний:

```text
/tmp/project_evil
```

може пройти перевірку для:

```text
/tmp/project
```

3. Перевіряти треба відносно дозволеного base dir, а не `cwd`.

Правильно:

```python
base = (self.project_path / "data" / "colab" / "accumulated").resolve()
config_path = config_path.resolve()
if not config_path.is_relative_to(base):
    raise ValueError(...)
```

---

## 14.3.5. P1: `RuntimeConfigLoader._handle_config_in_main_database()` використовує `self.logger`, якого немає

У методі:

```python
except (...) as e:
    self.logger.error(...)
```

А в `__init__` немає:

```python
self.logger = logger
```

Якщо видалення `config.json` із `main_database` не вдасться, отримаємо:

```text
AttributeError: 'RuntimeConfigLoader' object has no attribute 'logger'
```

Рекомендація:

```python
self.logger = logger
```

або використовувати module-level `logger`.

---

## 14.3.6. P1: `RuntimeConfigLoader` може видалити `main_database/config.json`

Код:

```python
if is_main_database and config_path and config_path.exists():
    print('🗑️ Видаляємо config.json з main_database')
    config_path.unlink()
```

Це агресивна поведінка. Навіть якщо `config.json` у `main_database` “не коректно”, runtime loader не має мовчки видаляти файл.

Ризики:

- випадкова втрата config;
- складно відновити стан;
- Codex/Colab може стерти файл у Drive.

Рекомендація:

- не видаляти автоматично;
- перейменовувати у `.bak`;
- або повертати warning/error;
- або вимагати explicit `--fix-main-config`.

---

## 14.3.7. P1: `ColabEnvironment.setup_batch_directory()` використовує raw `batch_name` у шляху

```python
self.batch_dir = self.PROJECT_PATH / "data" / "colab" / "accumulated" / self.BATCH_NAME
```

Немає sanitizer.

Це той самий ризик path traversal, що й у CLI/BatchManager.

Рекомендація:

- спільний `sanitize_batch_name`;
- перевірити `.resolve().is_relative_to(accumulated_base)`.

---

## 14.3.8. P1: `colab/utils/utils.py` дублює `batch_size.py`, `data_signature.py`, `metrics.py`

Є окремі файли:

```text
utils/batch_size.py
utils/data_signature.py
utils/metrics.py
utils/retry.py
```

Але `utils/utils.py` знову містить:

- `get_optimal_batch_size`
- `compute_data_signature`
- `compute_metrics`
- checkpoint helpers

Тобто частина функцій дублюється.

Наслідок:

- `compute_data_signature` можна виправити в одному місці, але не в іншому;
- `get_optimal_batch_size` може розійтись;
- `utils/__init__.py` зараз бере все з `utils.py`, тому окремі модулі майже не використовуються.

Рекомендація:

- зробити окремі файли canonical:
  - `batch_size.py`
  - `data_signature.py`
  - `metrics.py`
  - `retry.py`
  - `checkpoint.py`
- або навпаки залишити `utils.py`, а окремі файли зробити re-export wrappers.
- зараз треба хоча б синхронізувати `__init__.py`.

---

## 14.3.9. P1: `compute_data_signature()` хешує тільки `tail(100)` і shape

```python
feat_info = f"{df_feat.shape}_{hash_pandas_object(df_feat.tail(100)).sum()}"
targ_info = f"{df_targ.shape}_{hash_pandas_object(df_targ.tail(100)).sum()}"
```

Ризик:

- зміни в середині або на початку dataset не змінять signature, якщо shape і tail ті самі;
- cache/checkpoint logic може думати, що дані ті самі;
- для тренування це небезпечно.

Рекомендація:

- або хешувати весь DataFrame для невеликих batch;
- або sample head + middle + tail + schema + index range;
- або хешувати parquet file bytes/mtime/size;
- додати data schema/columns/order/dtypes.

---

## 14.3.10. P1: checkpoint paths будуються з raw `ticker`, `target_col`, `m_type`

У `save_checkpoint()`:

```python
checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{ticker}_{target_col}_{m_type}_ep{epoch}.pt"
torch.save(...)
```

У `find_latest_checkpoint()`:

```python
pattern = f"checkpoint_{ticker}_{target_col}_{m_type}_ep*.pt"
```

Ризики:

- якщо `ticker` або `target_col` містить `/`, `\`, `..`, wildcard chars — path traversal або неправильний glob;
- target column names часто можуть бути довгі або містити спецсимволи;
- checkpoint_dir parent не створюється;
- save не використовує trusted output path.

Рекомендація:

- sanitize filename components;
- create parent dir;
- resolve path і перевірити, що він всередині checkpoint_dir;
- для `find_latest_checkpoint` не використовувати raw glob pattern із unsanitized values.

---

## 14.3.11. P1/P2: `find_latest_checkpoint()` може впасти на нестандартних filenames

```python
checkpoints.sort(key=lambda x: int(x.stem.split('_ep')[-1]), reverse=True)
```

Якщо файл має `_epbad.pt`, або інша назва випадково match, буде `ValueError`.

Рекомендація:

- regex:

```python
r"_ep(\d+)$"
```

- пропускати invalid filenames.

---

## 14.3.12. P1: sklearn fallback / RandomForestWrapper не сумісні з типовим PyTorch training loop

`create_model()` якщо torch unavailable повертає sklearn wrapper.

Wrapper має:

```python
parameters() -> []
```

А типовий PyTorch код часто робить:

```python
optimizer = Adam(model.parameters())
```

Якщо parameters empty, optimizer може впасти:

```text
ValueError: optimizer got an empty parameter list
```

Також `RandomForestWrapper` для `model_type='random_forest'` у torch-available режимі теж має:

```python
parameters() -> []
```

і `forward()` викликає:

```python
self.model.predict(x_np)
```

але RandomForest не fitted.

Наслідок:

- якщо тренувальний loop сприйме random_forest як torch model, він зламається;
- sklearn fallback не можна тренувати через torch criterion/optimizer;
- треба окремий sklearn training path.

Рекомендація:

- `create_model()` має повертати разом із моделлю `backend="torch"|"sklearn"`;
- або wrapper має мати явний `.fit(X, y)`;
- training orchestrator має розгалужуватись;
- не передавати sklearn model у torch optimizer.

---

## 14.3.13. P1/P2: `FakeTensor` неповний

У sklearn fallback `FakeTensor` має тільки:

```python
numpy()
flatten()
```

А тренувальний або evaluation код може очікувати:

- `.detach()`
- `.cpu()`
- `.item()`
- `.shape`
- `.to()`
- `.float()`

Це може дати runtime errors.

Рекомендація:

- або повертати `np.ndarray`, не fake tensor;
- або повертати реальний `torch.tensor`, якщо torch є;
- або зробити evaluation path backend-aware.

---

## 14.3.14. P1/P2: LSTM/GRU/Transformer фактично не використовують часову послідовність

У `architectures.py`:

```python
out, _ = self.lstm(x.unsqueeze(1))
```

Якщо `x` має shape `[batch, features]`, після `unsqueeze(1)` отримуємо:

```text
[batch, 1, features]
```

Тобто sequence length = 1.

Так само Transformer:

```python
x = self.embedding(x.unsqueeze(1))
```

sequence length теж 1.

Наслідок:

- LSTM/GRU/Transformer не вчать часову динаміку;
- це фактично tabular nonlinear model із іншою архітектурою;
- назви можуть вводити в оману.

Це не обов’язково P0, але потрібно чесно документувати або змінити data shape:

```text
[batch, sequence_length, features]
```

Рекомендація:

- або назвати їх `TabularLSTMWrapper`, `SingleStepTransformer`;
- або зробити sequence dataset/windowing;
- або не порівнювати їх як справжні time-series sequence models.

---

## 14.3.15. P1/P2: `AutoencoderModel` не є справжнім autoencoder

У `architectures.py`:

```python
encoder: input -> 32
decoder: 32 -> 16 -> 1
```

Справжній autoencoder для реконструкції мав би повертати `input_sz`, а не 1.

Тут це bottleneck regressor. Якщо модель використовується як primary predictor, це нормально як архітектура, але не autoencoder.

Ризик:

- Codex/агент може думати, що це anomaly/reconstruction model;
- evaluation може неправильно інтерпретувати output.

Рекомендація:

- перейменувати на `BottleneckRegressor`;
- або зробити decoder output `input_sz` для autoencoder і окремий prediction head;
- у routing не використовувати autoencoder як primary predictor без явного рішення.

---

## 14.3.16. P2: `MemoryMonitor.save_log()` не створює parent dir і не ловить файлові помилки

```python
with open(filepath, 'w') as f:
    json.dump(...)
```

Рекомендація:

- `Path(filepath).parent.mkdir(parents=True, exist_ok=True)`;
- catch `OSError/PermissionError`;
- для datetime у memory_log зараз strings, окей.

---

## 14.3.17. P2: `retry_on_timeout()` тільки sync

`retry_on_timeout()` використовує `time.sleep()` і не підтримує async functions.

Якщо Colab training helper буде async, decorator не підійде.

Рекомендація:

- або документувати sync-only;
- або зробити async-aware decorator.

---

## 14.3.18. P2: `compute_metrics()` MAPE нестабільний для near-zero y_true

```python
mask = y_true != 0
mape = mean(abs((y_true - y_pred) / y_true)) * 100
```

Якщо y_true дуже близький до 0, MAPE вибухає.

Для returns це часта ситуація.

Рекомендація:

- epsilon threshold:

```python
mask = np.abs(y_true) > eps
```

- або додати sMAPE/MAE/RMSE без MAPE як основні.

---

# 14.4. Дублювання / архітектурні розриви

## 14.4.1. Два ColabEnvironment

```text
colab/environment/colab_environment.py
colab/environment/setup.py
```

Потрібен один canonical файл.

## 14.4.2. Дві реалізації model_factory

```text
colab/models/model_factory.py
colab/models/torch_models.py
```

`model_factory.py` уже містить torch model creation і sklearn fallback.  
`torch_models.py` дублює частину логіки.

Рекомендація:

- `model_factory.py` — canonical entrypoint;
- `torch_models.py` — або helper-only, або deprecated wrapper.

## 14.4.3. Дублювання utility functions

```text
colab/utils/utils.py
colab/utils/batch_size.py
colab/utils/data_signature.py
colab/utils/metrics.py
colab/utils/retry.py
```

Потрібен один source of truth.

## 14.4.4. `README.md` посилається на `run_hybrid_pipeline.py`, якого в поточному дереві не видно

Це вже було знайдено в `cli`/`calibration`.

Треба або відновити entrypoint, або оновити README.

---

# 14.5. Що працює добре

1. Top-level `colab/__init__.py` зроблений lazy — це добре.
2. README правильно каже, що Colab модуль не має імпортуватись локальним pipeline.
3. `model_factory.create_model()` не імпортує torch на module import, а перевіряє доступність у runtime.
4. `load_checkpoint()` використовує `resolve_trusted_artifact_path` і `torch.load(weights_only=True)`, це добре.
5. `get_optimal_batch_size()` проста й корисна.
6. `MemoryMonitor` простий і практичний.
7. `RuntimeConfigLoader` має force full mode і batch config detection — ідея правильна.
8. `ColabEnvironment` обережно fallback-иться на local mode, якщо Google Colab/Drive недоступні.

---

# 14.6. Що дати Codex по `colab`

```text
Deep-fix src/colab without changing public architecture.

Tasks:

1. Fix src/colab/utils/__init__.py:
   - import retry_on_timeout from .retry, not .utils;
   - re-export utilities from their canonical files.

2. Consolidate duplicate ColabEnvironment:
   - keep environment/colab_environment.py as canonical;
   - make environment/setup.py a compatibility shim or mark deprecated.

3. Harden batch path safety:
   - sanitize batch_name from sys.argv and setup_batch_directory;
   - reject ../, /, \, control chars and overly long names;
   - ensure resolved batch_dir stays inside project_path/data/colab/accumulated.

4. Fix RuntimeConfigLoader:
   - avoid raw sys.argv parsing if possible; accept batch_name param;
   - replace Path.cwd().startswith path check with Path.is_relative_to allowed base dir;
   - add self.logger or use module logger consistently;
   - do not auto-delete main_database/config.json; warn or move to .bak only with explicit fix flag.

5. Consolidate colab/utils:
   - define canonical modules for batch_size, data_signature, metrics, retry, checkpoint;
   - remove duplicated implementations or convert to wrappers.

6. Improve compute_data_signature:
   - do not hash only tail(100);
   - include columns, dtypes, index range, and broader content hash or file-level fingerprint.

7. Harden checkpoint helpers:
   - sanitize ticker/target_col/model_type before building filenames;
   - create checkpoint_dir;
   - ensure save path stays inside checkpoint_dir;
   - make find_latest_checkpoint robust with regex epoch parsing;
   - catch OSError/PermissionError around save/load.

8. Make model factory backend-aware:
   - return backend metadata or separate create_torch_model/create_sklearn_model;
   - do not pass sklearn wrappers into torch optimizers;
   - add explicit fit/predict path for sklearn fallback and random_forest.

9. Clarify sequence model semantics:
   - LSTM/GRU/Transformer currently use sequence length 1;
   - either document as tabular wrappers or implement windowed sequence input [batch, seq_len, features].

10. Fix Autoencoder naming:
   - either make decoder reconstruct input_size;
   - or rename to BottleneckRegressor / AutoencoderRegressor with explicit prediction head.

11. Harden MemoryMonitor:
   - save_log should create parent dirs and catch file errors.

12. Improve retry_on_timeout:
   - document sync-only or add async-aware support.

13. Improve compute_metrics:
   - protect MAPE from near-zero y_true;
   - add epsilon threshold or use sMAPE.

14. Update README:
   - if run_hybrid_pipeline.py is legacy/missing, update command;
   - document actual Colab entrypoint/notebook expected to call src.colab modules.
```

---

# 14.7. Priority list for `colab`

## P0 / must fix

- `src.colab.utils` import failure because `retry_on_timeout` is imported from wrong module.
- batch name/path traversal risk in `RuntimeConfigLoader` and `ColabEnvironment`.
- sklearn fallback/random_forest wrappers are not compatible with torch training loops if used as normal models.

## P1 / high priority

- duplicate `ColabEnvironment`.
- duplicate utils and model_factory logic.
- `RuntimeConfigLoader` unsafe path check via `startswith(Path.cwd())`.
- auto-delete of `main_database/config.json`.
- data signature hashes only tail(100).
- checkpoint filenames use raw ticker/target/model strings.
- LSTM/GRU/Transformer sequence length = 1.
- Autoencoder output is 1, not reconstructed input.

## P2 / cleanup

- `MemoryMonitor.save_log()` parent dir/error handling.
- sync-only retry decorator.
- MAPE near-zero instability.
- README command mismatch with missing `run_hybrid_pipeline.py`.
- add clear Colab runner/notebook entrypoint.

---

# 14.8. Summary for `colab`

`colab` — корисна папка для Colab-side heavy training, і її правильно тримати ізольованою від локального pipeline. Але зараз вона має кілька важливих проблем.

Найперше треба виправити:

1. `src.colab.utils` import failure.
2. batch path safety.
3. duplicate environment/utils/model factory code.
4. checkpoint filename/path safety.
5. backend-aware model training для sklearn fallback/random_forest.
6. чесно описати, що LSTM/GRU/Transformer зараз не використовують справжнє sequence window.
7. або зробити справжній autoencoder, або перейменувати його.

Після цих фіксів `colab` може бути нормальним автономним training toolkit: локальний pipeline готує batch, Colab тренує heavy models, зберігає результати, а локальний `continue` mode підхоплює artifacts.

---

# 15. Audit: `config`

## 15.1. Загальний стан

`config` — це **центральний control-plane** проєкту. У папці є YAML-конфіги для:

- assets / tickers;
- collectors;
- paths;
- models;
- targets;
- features;
- analytics;
- strategy/risk;
- monitoring;
- system;
- context/rules;
- unified config manager.

У наданій папці:

```text
config/
  README.md
  __init__.py
  unified_config_manager.py
  model_registry.py
  tickers.py
  sentiment_config.py
  *.yaml
```

Синтаксично Python-файли компілюються. YAML-файли парсяться. `UnifiedConfigManager` може ініціалізуватися на наданій папці, але при цьому видно кілька важливих архітектурних проблем:

1. Є **кілька джерел правди** для одних і тих самих речей.
2. `UnifiedConfigManager` мерджить усі YAML-файли в алфавітному порядку, що робить precedence неочевидним.
3. Є дубльовані/конфліктні top-level секції.
4. Частина README/API/інших модулів очікує keys, яких у конфігу немає.
5. Конфіг має прямі production-looking cloud values.
6. `UnifiedConfigManager` має runtime side effects: створює директорії, читає secrets, валідовує cloud storage.
7. Частина class paths у config вказує на модулі, які падають через optional dependencies або проблеми в коді.

Головний висновок: **config дуже важливий і залучений, але зараз він не є стабільним single source of truth.** Перед подальшим агентним шаром треба стабілізувати саме contract: що є canonical, які keys очікуються, які файли legacy, які секції використовуються реально.

---

## 15.2. Залученість у проєкті

### Реально залучено

`UnifiedConfigManager` і `get_current_config()` використовуються широко:

- pipeline stages;
- collectors;
- analytics;
- calibration;
- CLI;
- models;
- cache;
- data manager;
- devtools;
- backtesting.

`models.yaml` реально важливий, бо містить `models.model_definitions`, `models.dual_model_manager`, `models.categories`.

`collectors.yaml` реально важливий для Stage 1 collection.

`paths.yaml` реально важливий для DB, models, logs, outputs.

`targets.yaml` реально важливий для target generation.

`analysis.yaml` і `unified_config.yaml` важливі для `UnifiedAnalyticsEngine`, але між ними є дублювання.

### Слабо або неоднозначно залучено

`tickers.py` існує як Python single source of truth, але `assets.yaml` також задає assets/tickers.

`sentiment_config.py` задає `SENTIMENT_DEFAULTS`, але є також `sentiment.yaml` і `models.yaml -> models.sentiment`.

`analysis.yaml` має top-level `engine`, а `unified_config.yaml` має `analysis.engine`. У попередньому аудиті було видно, що `UnifiedAnalyticsEngine` орієнтується на `analysis.engine`, тому `analysis.yaml` може бути частково legacy або паралельним конфігом.

---

# 15.3. Що працює добре

## 15.3.1. YAML-файли валідні

Усі YAML-файли парсяться через `yaml.safe_load`.

## 15.3.2. Python-файли компілюються

Файли:

```text
unified_config_manager.py
model_registry.py
tickers.py
sentiment_config.py
__init__.py
```

компілюються без syntax errors.

## 15.3.3. `UnifiedConfigManager` має dotted access

Метод:

```python
get("paths.models")
```

дозволяє зручно читати nested config.

## 15.3.4. Є DynamicConfig wrapper

Можна звертатись до top-level секцій як до атрибутів:

```python
config.paths.models
```

Це зручно, хоча має свої ризики.

## 15.3.5. Є class path config для pipeline і analytics

`unified_config.yaml` містить:

```text
training_pipeline:
  - name: Stage_0_Setup
    class_path: ...
```

І `analysis.engine.analyzers` містить module/class для analyzers. Це добре для factory/dynamic loading.

## 15.3.6. `models.yaml` class paths зараз здебільшого валідні

Перевірка `models.model_definitions` показала, що model class paths імпортуються і класи існують:

```text
catboost
lightgbm
xgboost
random_forest
linear
svm
knn
mlp
cnn
lstm
gru
transformer
tabnet
autoencoder
ensemble
```

Це позитивно.

---

# 15.4. Критичні проблеми

## 15.4.1. P0/P1: `UnifiedConfigManager` мерджить усі YAML-файли алфавітно і має неочевидний precedence

`_load_and_resolve_configs()` бере всі `*.yaml`:

```python
config_files = self.file_manager.find_files("*.yaml", search_dir=self.config_dir)
```

Потім кожен файл мерджиться в один `merged_config`.

Функція merge:

```python
def _deep_merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
```

Тобто якщо два файли мають однаковий top-level key, вони не повністю перезаписуються, а глибоко змішуються.

Це зручно, але небезпечно:

- порядок файлів визначає precedence;
- файл `unified_config.yaml` може частково змішатися з `strategy.yaml`, `analysis.yaml`, `risk_management.yaml`;
- неочевидно, який config реально використовує pipeline.

Під час ініціалізації видно warning:

```text
Conflicting top-level key 'strategy' in unified_config.yaml.
Previous source: risk_management.yaml. Precedence given to latest.
```

Але формулювання misleading: через deep merge не завжди “latest wins” на всю секцію. Він може частково змішати стару й нову структуру.

Рекомендація:

- зробити явний порядок завантаження config files;
- або мати один `unified_config.yaml` як root, а інші файли include/override;
- або заборонити duplicate top-level keys без explicit override marker;
- логувати full source map для кожного key path, а не тільки top-level.

---

## 15.4.2. P0/P1: є два analysis config contracts

Є:

```text
analysis.yaml -> engine.analyzers
unified_config.yaml -> analysis.engine.analyzers
```

Обидва містять analyzers, але списки різні.

`analysis.yaml`:

```text
engine.analyzers:
  market_regime
  adaptive_confidence
  causal_event_finder
  knn_similarity
  news_impact
  hedge_fund_style
  macro_context
  critical_signals
  pattern_analysis
  risk_decomposition
```

`unified_config.yaml`:

```text
analysis.engine.analyzers:
  critical_signal_detector
  macro_context_analyzer
  news_impact_analyzer
  market_context_analyzer
  market_phase_analyzer
  prediction_adjuster
  hedge_fund_style
  adaptive_confidence
  causal_events
  knn_similarity
```

Наслідок:

- різні analyzer names;
- різні class names;
- різні data_mapping;
- один engine може читати `analysis.engine`, інший `engine`;
- Codex може фіксити не той config.

Приклад проблеми:

`analysis.yaml` має:

```yaml
causal_event_finder:
  module: src.analytics.analyzers.causal_event_finder
  class: CausalEngine
```

А в `unified_config.yaml`:

```yaml
causal_events:
  module: src.analytics.analyzers.causal_event_finder
  class: CausalEventFinder
```

Це два різні очікування для одного модуля.

Рекомендація:

- залишити тільки один canonical path: `analysis.engine.analyzers`;
- `engine.analyzers` або видалити, або зробити compatibility alias;
- створити test, який перевіряє, що всі configured analyzers importable і мають клас.

---

## 15.4.3. P0/P1: configured causal analyzers падають через `dowhy`

Перевірка class paths показала:

```text
analysis.engine.causal_events -> IMPORT_FAIL: No module named 'dowhy'
engine.causal_event_finder -> IMPORT_FAIL: No module named 'dowhy'
```

Це узгоджується з попереднім аудитом `analytics`.

Проблема саме в тому, що config вмикає analyzer, який залежить від optional dependency. Якщо `UnifiedAnalyticsEngine` без skip/fail isolation спробує підняти його, весь analysis може впасти.

Рекомендація:

- у config додати:
  - `enabled: false` для `causal_events`, якщо `dowhy` не встановлений;
  - або `optional_dependency: dowhy`;
  - або `on_missing_dependency: skip`.
- у engine реалізувати graceful skip.

---

## 15.4.4. P1: `training_pipeline` class paths показують реальні import failures

Перевірка `training_pipeline` із `unified_config.yaml` дала:

```text
Stage_0_Setup: OK
Stage_1_Collection: IMPORT_FAIL duckdb
Stage_2_Processing: IMPORT_FAIL SyntaxError in cleaners.py
Stage_3_Feature_Engineering: IMPORT_FAIL IndentationError in feature_orchestrator.py
Stage_4_Modeling: IMPORT_FAIL duckdb
Stage_5_Prediction: IMPORT_FAIL duckdb
Stage_6_Trading_Execution: IMPORT_FAIL duckdb
Stage_7_Evaluation: OK
```

Частина проблем — optional dependency `duckdb`, але частина — реальні syntax/indentation errors у проекті. Для config це означає:

- config вказує на stage-и, які можуть не імпортуватися;
- немає config-level health check, який зупиняє pipeline до запуску.

Рекомендація:

- додати `config validate --class-paths`;
- у CI перевіряти всі class paths із `training_pipeline`;
- optional dependency errors мають бути чітко відділені від syntax errors.

---

## 15.4.5. P1: `paths.trading_db`, `paths.synthetic_data`, `paths.results`, `paths.cache` відсутні, але код їх очікує

У `paths.yaml` є:

```yaml
paths:
  raw_db: data/trading_data.duckdb
  duckdb_path: data/trading_data.duckdb
  models: data/trained_models
  outputs: outputs
  reports: reports
```

А в інших модулях уже були очікування:

```python
paths.get('trading_db', 'data/duckdb/trading.db')
paths.get('synthetic_data', 'data/synthetic/')
paths.get('results', 'results')
paths.get('cache', 'data/cache')
```

Наприклад, `CalibrationEngine` використовує:

```python
paths.get('trading_db', 'data/duckdb/trading.db')
paths.get('synthetic_data', 'data/synthetic/')
paths.get('results', 'results')
```

У config цих keys немає, тому модуль піде на defaults, які не збігаються з actual `paths.raw_db = data/trading_data.duckdb`.

Наслідок:

- `DataManager` може писати в `data/trading_data.duckdb`;
- `CalibrationEngine` читатиме `data/duckdb/trading.db`;
- результати можуть зберігатися не там, де очікує pipeline.

Рекомендація:

- стандартизувати DB path:
  - або `paths.trading_db`;
  - або `paths.raw_db`;
  - але всі модулі мають використовувати один alias.
- додати compatibility aliases:

```yaml
paths:
  trading_db: data/trading_data.duckdb
  raw_db: data/trading_data.duckdb
  duckdb_path: data/trading_data.duckdb
  synthetic_data: data/synthetic
  results: results
  cache: data/cache
```

---

## 15.4.6. P1: cloud_storage має real-looking values і обов’язкову валідацію

`cloud_storage.yaml`:

```yaml
cloud_storage:
  project_id: monospace-13
  bucket_name: trading_multi_project
```

`UnifiedConfigManager.validate_configuration()` вимагає:

```python
required_sections = ['paths', 'assets', 'models', 'features', 'cloud_storage']
```

і потім:

```python
if not project_id: raise
if not bucket_name: raise
```

Проблема:

- cloud storage може бути optional для local/dev;
- real-looking project/bucket values не завжди мають бути в repo config;
- якщо bucket/project зміняться, локальна config validation може падати;
- якщо user/agent запускає tests без cloud, config не має вимагати cloud.

Рекомендація:

- `cloud_storage.enabled: false/true`;
- якщо `enabled: false`, не вимагати project_id/bucket_name;
- project_id/bucket_name краще брати з env/secrets у production;
- default dev config не має містити production-looking IDs.

---

## 15.4.7. P1: `UnifiedConfigManager` має heavy side effects при ініціалізації

Під час `UnifiedConfigManager(...)` він:

1. завантажує всі YAML;
2. ставить dynamic attributes;
3. валідовує cloud storage;
4. створює директорії з `paths`;
5. створює `SecretsManager`;
6. читає `.env`/environment;
7. резолвить secrets/placeholders.

Це забагато для простого “прочитати config”.

Наслідки:

- import/config access може створювати папки;
- tests можуть мутувати файлову систему;
- config load може логити secrets warnings;
- optional cloud/secrets можуть ламати local flow.

Рекомендація:

- розділити:
  - `load_config()`;
  - `validate_config()`;
  - `ensure_paths()`;
  - `resolve_secrets()`;
- зробити side effects opt-in:
  - `UnifiedConfigManager(..., create_paths=False, resolve_secrets=False)` для tests/devtools;
  - `create_paths=True` тільки для runtime.

---

## 15.4.8. P1: `_ensure_paths_exist()` створює директорії для всіх string values у `paths`

Зараз він проходить `paths` і для кожного string:

```python
dir_to_create = path_obj.parent if path_obj.suffix else path_obj
self.file_manager.ensure_directory(dir_to_create)
```

Це може створити:

- папки в неправильному project_root, якщо config_dir не той;
- parent dirs для файлів, які не мають створюватись у read-only режимі;
- неочікувані папки під час tests.

Рекомендація:

- додати `create: true/false` metadata для paths;
- або `ensure_paths_exist()` викликати явно з runtime;
- не робити при кожному config load.

---

## 15.4.9. P1: singleton `get_current_config(config_dir=None)` ігнорує новий config_dir після першого виклику

```python
if _config_instance is not None:
    return _config_instance
```

Якщо десь уже викликали:

```python
get_current_config()
```

то потім:

```python
get_current_config("/other/config")
```

поверне старий instance.

Це небезпечно для:

- tests;
- agent audits;
- multi-project workflows;
- staging/prod configs;
- notebooks.

Рекомендація:

- якщо `config_dir` передано і відрізняється — або створити новий instance, або кинути clear error;
- додати `reset_config()` для tests;
- або кешувати per `(env, config_dir)`.

---

## 15.4.10. P1: placeholder resolution не використовує `secrets` для `${...}`

`_resolve_placeholders()` приймає `secrets`, але фактично робить:

```python
resolved_placeholder = self.get(placeholder, "")
```

Тобто `${ENV_SECRET}` не буде взятий із `secrets`, якщо це не config key.

Є окрема логіка для keys, що закінчуються `_env`, але це інший контракт.

Рекомендація:

- визначити один контракт:
  - `${paths.root}` для config refs;
  - `${env:NEWS_API_KEY}` або `${secret:NEWS_API_KEY}` для secrets;
- не змішувати;
- якщо placeholder не resolved — не мовчки залишати/заміняти на empty string без warning.

---

## 15.4.11. P1/P2: `DynamicConfig.__getattr__` не може повернути `None` як легітимне значення

```python
value = self._get_attribute_value(name)
if value is not None:
    return value
raise AttributeError
```

Якщо в config ключ існує і значення `None`, доступ через attribute дасть `AttributeError`.

Наприклад:

```yaml
google_drive:
  folder_id: null
```

`config.system.google_drive.folder_id` може виглядати як відсутній, хоча ключ існує.

Рекомендація:

```python
if name in self._data:
    value = self._data[name]
    return DynamicConfig(value) if isinstance(value, dict) else value
raise AttributeError
```

---

## 15.4.12. P1/P2: `config/__init__.py` порожній

Файл порожній.

Це не критично, але незручно. У проекті багато хто очікує:

```python
from src.config import get_current_config
```

Краще явно експортувати:

```python
from .unified_config_manager import UnifiedConfigManager, get_current_config, Environment
```

Але з урахуванням side effects краще lazy exports.

---

## 15.4.13. P1/P2: `model_registry.py` має ще один source of truth для models

Є:

```text
config/model_registry.py
config/models.yaml
src/models/constants.py
src/factories/model_factory.py
src/config/unified_config.yaml? maybe model refs
```

`model_registry.py` містить:

```python
MODELS = {
  'lgbm': ...
  'rf': ...
  constants.XGBOOST: ...
}
```

`models.yaml` містить:

```yaml
lightgbm
random_forest
xgboost
catboost
...
```

Тобто alias-и різні:

```text
lgbm vs lightgbm
rf vs random_forest
```

Ризик:

- model selector може назвати `rf`, а config очікує `random_forest`;
- feature selection cache може мати одне ім’я, model factory інше;
- duplicate aliases вже були проблемою в попередніх аудитах.

Рекомендація:

- один model registry у YAML або Python, не два;
- якщо YAML canonical, Python `ModelRegistry` має завантажувати YAML;
- додати aliases:

```yaml
aliases:
  rf: random_forest
  lgbm: lightgbm
```

і єдину функцію normalize_model_name.

---

## 15.4.14. P1/P2: `tickers.py` дублює `assets.yaml`

`assets.yaml` має active preset і sectors:

```yaml
assets:
  active_preset: default_volatile
  presets:
  sectors:
  details:
```

`tickers.py` має великий hardcoded список:

```python
ETF_TICKERS
TECH_GIANTS
...
ALL_TICKERS
CORE_TICKERS
```

Ризик:

- CLI validator читає `assets`;
- інші модулі можуть читати `tickers.py`;
- списки розходяться;
- активний preset не синхронізований з `CORE_TICKERS`.

Рекомендація:

- зробити `assets.yaml` canonical;
- `tickers.py` має або читати YAML, або бути generated artifact;
- якщо залишити Python, додати test consistency:
  - all active preset tickers exist in details;
  - all details tickers in some sector/preset.

---

## 15.4.15. P1/P2: `sentiment_config.py`, `sentiment.yaml`, `models.yaml` дублюють sentiment config

Є:

```text
sentiment_config.py -> SENTIMENT_DEFAULTS
sentiment.yaml -> sentiment.*
models.yaml -> models.sentiment.model_name
```

Ризик:

- NLP/sentiment analyzer бере одну модель;
- config manager показує іншу;
- thresholds різні.

Рекомендація:

- canonical `sentiment.yaml`;
- `sentiment_config.py` або deprecated, або wrapper over config.

---

## 15.4.16. P1/P2: `targets.yaml` використовує negative shift, але потрібен явний temporal contract

У `targets.yaml` багато target definitions:

```yaml
shift: -1
shift: -5
```

Це нормально для target generation, але config має явно позначати:

```yaml
target_generation_only: true
requires_feature_shift_guard: true
```

І target generator має гарантувати:

- features at t;
- target at t+h;
- target rows із майбутніми NaN drop only after split-aware handling;
- no leakage into feature columns.

Рекомендація:

- додати `horizon_periods`;
- додати `target_timestamp_policy`;
- test для target generation.

---

## 15.4.17. P2: `context_rule_generation.output_path` пише в `src/config/generated_context_rules.yaml`

```yaml
output_path: src/config/generated_context_rules.yaml
```

Devtool rule generator може мутувати source tree.

У попередньому аудиті `devtools` це вже було видно.

Рекомендація:

- output у `outputs/context_rules/generated_context_rules.yaml`;
- source config має бути immutable;
- якщо потрібен promotion у config — через review/PR, не runtime write.

---

## 15.4.18. P2: `export_tickers_to_json()` пише в `config/tickers_export.json`

У `tickers.py`:

```python
export_tickers_to_json(filepath="config/tickers_export.json")
```

Це теж runtime write у config folder.

Рекомендація:

- default path у `outputs/config_exports/tickers_export.json`;
- або вимагати explicit path.

---

# 15.5. Дублювання / архітектурні розриви

## 15.5.1. Multiple config roots

Є окремі YAML:

```text
analysis.yaml
features.yaml
models.yaml
collectors.yaml
...
```

і великий:

```text
unified_config.yaml
```

Зараз `UnifiedConfigManager` просто зливає все. Немає явного “root config”.

## 15.5.2. Models duplicated

```text
model_registry.py
models.yaml
src/models/constants.py
factories/model_factory.py
```

## 15.5.3. Tickers duplicated

```text
assets.yaml
tickers.py
collectors.yaml ingestion reference tickers
unified_config.yaml ingestion reference tickers
```

## 15.5.4. Sentiment duplicated

```text
sentiment_config.py
sentiment.yaml
models.yaml -> models.sentiment
```

## 15.5.5. Analysis config duplicated

```text
analysis.yaml -> engine
unified_config.yaml -> analysis.engine
```

## 15.5.6. Risk/strategy duplicated

```text
risk_management.yaml -> strategy.risk_management + analytics + trading
strategy.yaml -> risk_management + backtesting
unified_config.yaml -> strategy.trading_advisor + strategy.strategies
```

Результат — `strategy` як top-level key конфліктує і змішується.

---

# 15.6. Що дати Codex по `config`

```text
Deep-fix src/config without changing public architecture.

Tasks:

1. Define config source-of-truth:
   - decide whether unified_config.yaml is root or whether separate YAML files are canonical;
   - document load order and override rules;
   - avoid accidental duplicate top-level keys.

2. Fix UnifiedConfigManager loading:
   - use explicit file order;
   - detect duplicate top-level keys and require explicit override marker;
   - track source file per dotted key, not just top-level key.

3. Split side effects:
   - config load should not automatically create directories or resolve secrets unless requested;
   - add flags: create_paths=False, resolve_secrets=False, validate_cloud=False for tests/devtools;
   - runtime can opt in.

4. Fix singleton behavior:
   - get_current_config(config_dir=...) must not silently ignore a different config_dir after first initialization;
   - add reset_config() for tests;
   - optionally cache per (env, config_dir).

5. Fix cloud storage validation:
   - add cloud_storage.enabled;
   - if disabled, do not require project_id/bucket_name;
   - do not hardcode production-looking cloud project/bucket in dev defaults.

6. Standardize paths:
   - add compatibility aliases:
     paths.trading_db
     paths.raw_db
     paths.duckdb_path
     paths.synthetic_data
     paths.results
     paths.cache
   - make all modules use one canonical DB path.

7. Unify analysis config:
   - keep canonical analysis.engine.analyzers;
   - remove or alias engine.analyzers;
   - add enabled/on_missing_dependency/optional_dependency fields;
   - set causal_events to skip when dowhy is unavailable.

8. Add config class-path validation:
   - validate training_pipeline class_path imports;
   - validate analysis analyzer module/class imports;
   - validate models.model_definitions imports;
   - distinguish optional dependency failures from syntax errors.

9. Fix placeholders/secrets:
   - support explicit ${config:paths.root}, ${env:NEWS_API_KEY}, ${secret:...};
   - warn on unresolved placeholders;
   - do not silently replace with empty string.

10. Fix DynamicConfig:
   - allow existing keys with None values to be returned via attribute access;
   - __getattr__ should check key existence, not value is not None.

11. Add lazy exports to config/__init__.py:
   - expose get_current_config, UnifiedConfigManager, Environment safely.

12. Unify model registry:
   - make models.yaml canonical;
   - ModelRegistry should load YAML or be generated from it;
   - add aliases rf->random_forest, lgbm->lightgbm;
   - one normalize_model_name function.

13. Unify tickers/assets:
   - make assets.yaml canonical;
   - tickers.py should read/generate from YAML or be clearly legacy;
   - test active preset consistency with details/sectors.

14. Unify sentiment config:
   - make sentiment.yaml canonical;
   - sentiment_config.py should be wrapper/deprecated;
   - ensure model_name/thresholds have one source of truth.

15. Fix generated config outputs:
   - context_rule_generation.output_path should not write into src/config by default;
   - use outputs/context_rules/generated_context_rules.yaml;
   - export_tickers_to_json default should not write into config folder.

16. Strengthen target config contract:
   - explicitly mark negative shifts as target-generation-only;
   - add horizon metadata;
   - require target generation tests for no feature leakage.

17. Add config tests:
   - YAML parse test;
   - required sections test;
   - no duplicate top-level keys unless explicit override;
   - class paths import test;
   - paths aliases test;
   - model aliases test;
   - analyzer dependency skip behavior test.
```

---

# 15.7. Priority list for `config`

## P0 / must fix

- Decide single source of truth for `analysis.engine` vs `engine`.
- Fix config path aliases so modules do not read different DB/result/cache paths.
- Add class-path validation for `training_pipeline` and analyzers.
- Add dependency skip mechanism for configured analyzers such as `causal_events`.
- Stop treating cloud storage as mandatory in local/dev unless enabled.

## P1 / high priority

- Explicit config load order and override rules.
- Split `UnifiedConfigManager` side effects from plain load.
- Fix singleton ignoring new config_dir.
- Unify model registry aliases.
- Unify assets/tickers.
- Fix placeholder/secrets resolution.
- Fix `DynamicConfig` None handling.
- Add `config/__init__.py` exports.

## P2 / cleanup

- Move generated context rules output out of `src/config`.
- Move ticker export output out of `config`.
- Unify sentiment config.
- Make target negative shift contract explicit.
- Reduce duplicate hardcoded reference tickers/keywords in collectors/unified config.

---

# 15.8. Summary for `config`

`config` — один із найважливіших шарів проєкту. Він реально залучений всюди, але зараз у ньому є ризикова суміш:

- багато YAML-файлів;
- великий `unified_config.yaml`;
- deep merge без явного root/override policy;
- дублювання `analysis`, `strategy`, `models`, `tickers`, `sentiment`;
- відсутні path aliases, які очікують інші модулі;
- configured class paths можуть падати;
- cloud storage обов’язковий навіть для dev;
- config load має побічні ефекти.

Після стабілізації `config` стане реальною основою для агентів: агенти зможуть читати один source of truth, а не гадати, який YAML зараз працює.

---

# 16. Audit: `dashboard`

## 16.1. Загальний стан

`dashboard` — це Streamlit UI шар для human-in-the-loop monitoring: signals, performance, news, risk, system status.

У наданій папці є:

```text
dashboard/
  README.md
  __init__.py
  main_app.py
```

Синтаксично Python-файл компілюється. Але імпорт у моєму середовищі падає через відсутній `streamlit`:

```text
ModuleNotFoundError: No module named 'streamlit'
```

Це не обов’язково помилка коду, якщо в dashboard environment `streamlit` встановлений. Але архітектурно важливо: `dashboard` має бути optional UI dependency і не має ламати production/pipeline imports.

Головний висновок: **ідея корисна, але `main_app.py` зараз виглядає як старий/прямий Streamlit app, який не синхронізований з актуальним DataManager/config/schema.** Найкритичніше: dashboard викликає неіснуючий `DataManager.load_data(query)`, неправильно дістає tickers із `assets`, і напряму запитує таблиці/колонки, яких може не бути.

---

## 16.2. Залученість у проєкті

### Слабка локальна залученість

Пошук показує, що:

```text
UnifiedDashboard
src.dashboard
```

майже не згадуються поза самою папкою `src/dashboard`.

Тобто dashboard не є production pipeline dependency. Це нормально для UI.

### Є кращий bridge, який `main_app.py` не використовує

У проєкті є:

```text
src/integration/dashboard_data_bridge.py
```

Він якраз створений для зв’язку pipeline data ↔ dashboard UI, має cache, sample/real data markers і fallback.

А `dashboard/main_app.py` напряму імпортує:

```python
DataManager
UnifiedAnalyticsEngine
FamaFrenchFactors
HedgeFundAnalyzer
```

і напряму робить SQL-запити.

Рекомендація: dashboard має використовувати `DashboardDataBridge`, а не напряму `DataManager`.

---

# 16.3. Критичні проблеми

## 16.3.1. P0: `get_data_from_db()` викликає неіснуючий `DataManager.load_data(query)`

У `main_app.py`:

```python
@st.cache_data(ttl=60)
def get_data_from_db(_db_manager, query):
    return _db_manager.load_data(query)
```

А актуальний `DataManager` має:

```python
fetch_df(query, params=None)
fetch_all(...)
fetch_one(...)
fetch_data_from_table(...)
```

Методу `load_data(query)` у concrete `DataManager` не видно.

Наслідок:

```text
AttributeError: 'DataManager' object has no attribute 'load_data'
```

і dashboard падає при першому запиті.

Правильний фікс:

```python
def get_data_from_db(_db_manager, query):
    if hasattr(_db_manager, "fetch_df"):
        return _db_manager.fetch_df(query)
    if hasattr(_db_manager, "query_data"):
        return _db_manager.query_data(query)
    return pd.DataFrame()
```

Але краще — винести це в `DashboardDataBridge`.

---

## 16.3.2. P0/P1: `get_all_configured_tickers()` неправильно читає `assets`

Код:

```python
assets = config_manager.get("assets", {})
tickers = list(assets.keys()) if isinstance(assets, dict) else []
```

А в `assets.yaml` структура така:

```yaml
assets:
  presets:
  active_preset:
  sectors:
  details:
```

Тобто `list(assets.keys())` поверне:

```text
["presets", "active_preset", "sectors", "details"]
```

а не tickers.

Потім додається `SPY`, і sidebar матиме дивні “tickers”:

```text
active_preset
details
presets
sectors
SPY
```

Рекомендація:

```python
assets = config_manager.get("assets", {})
active = assets.get("active_preset")
preset = assets.get("presets", {}).get(active, {})
tickers = preset.get("tickers", [])
```

Fallback:

```python
if not tickers:
    tickers = list(assets.get("details", {}).keys())
```

---

## 16.3.3. P1: dashboard напряму запитує hardcoded tables, яких може не бути

У `main_app.py` є SQL до:

```text
model_performance
factor_exposures
adaptive_thresholds
trading_signals
news_data
```

README при цьому каже, що потрібні таблиці:

```text
trading_signals
model_performance
news
evaluation_summary
```

Тобто вже є mismatch:

```text
README: news
code: news_data
```

Раніше в інших аудитах було видно, що дані можуть зберігатись у:

```text
google_news
rss_news
newsapi_articles
fred_data
vix_data
fear_greed_data
```

а не в `news_data`.

Наслідок:

- якщо хоча б однієї таблиці немає, dashboard падає;
- UI не показує graceful empty state;
- schema evolution ламає dashboard.

Рекомендація:

- використовувати `DataManager.table_exists()`;
- мати `DashboardDataBridge`, який знає fallback/schemas;
- кожен query має бути обгорнутий у safe query:

```python
try:
    df = db.fetch_df(query)
except Exception:
    return pd.DataFrame()
```

---

## 16.3.4. P1: SQL columns можуть не відповідати актуальним schemas

Приклади:

```python
SELECT AVG(profit_factor) as avg_pf, AVG(win_rate) as avg_wr, MAX(max_drawdown) as max_dd FROM model_performance
```

А `DashboardDataBridge` очікує інші назви:

```text
avg_win_rate
avg_sharpe_ratio
avg_precision
total_trades
last_updated
```

У `render_risk_management_tab()`:

```python
SELECT model_name, sharpe_ratio, max_drawdown, win_rate FROM model_performance
```

Якщо таблиця має `avg_sharpe_ratio` замість `sharpe_ratio`, query впаде.

Рекомендація:

- створити canonical dashboard views:
  - `dashboard_model_performance_view`
  - `dashboard_signals_view`
  - `dashboard_news_view`
- або bridge має адаптувати різні schema versions.

---

## 16.3.5. P1: dashboard має heavy imports на module import

У `main_app.py` на module import:

```python
import streamlit as st
import plotly...
import psutil
from src.data.management.data_manager import DataManager
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
from src.analytics.calculators.fama_french_factors import FamaFrenchFactors
from src.analytics.analyzers.hedge_fund_analyzer import HedgeFundAnalyzer
```

Частина імпортів не використовується:

```text
UnifiedAnalyticsEngine
FamaFrenchFactors
HedgeFundAnalyzer
json
Optional
Dict
Any
np
go? частково go використовується
```

Ризик:

- dashboard import падає через optional dependencies, які UI прямо не використовує;
- streamlit app стартує повільніше;
- аналітичні залежності тягнуть `dowhy`/інші optional bugs.

Рекомендація:

- прибрати unused imports;
- імпортувати важкі речі lazy всередині tab/component, якщо вони реально потрібні;
- основний dashboard має залежати від bridge, а не від analytics engine.

---

## 16.3.6. P1: import error одразу робить `st.stop()`

```python
except ImportError as e:
    st.error(...)
    st.stop()
```

Це зручно для Streamlit, але краще розділити:

- якщо немає `DataManager` — показати sample dashboard;
- якщо немає analytics extras — вимкнути конкретний tab;
- якщо немає Streamlit — це CLI/environment issue.

У `DashboardDataBridge` вже є pattern:

```python
DataManager not available -> using sample dashboard data
```

`main_app.py` має використовувати цей pattern.

---

## 16.3.7. P1: `render_header()` запитує DB без error handling

```python
model_perf = get_data_from_db(...)
active_models = model_perf.iloc[0]['count'] if not model_perf.empty else 0
```

Якщо таблиці немає, query падає до того, як `empty` буде перевірено.

Те саме в інших tabs.

Рекомендація:

- `get_data_from_db()` має завжди повертати DataFrame;
- або `safe_metric_query(name, query, default)`.

---

## 16.3.8. P1/P2: `st.set_page_config()` викликається всередині `render_header()`

```python
def render_header(self):
    st.set_page_config(...)
```

Streamlit рекомендує викликати `st.set_page_config()` як першу Streamlit-команду і тільки один раз.

Зараз до `render_header()` не має бути інших `st.*`, але якщо в майбутньому `__init__`, sidebar або error path викличе Streamlit раніше — буде помилка.

Рекомендація:

- винести `st.set_page_config()` на початок `run()` перед будь-яким UI;
- або в `main()` до створення dashboard.

---

## 16.3.9. P2: auto refresh не реалізований

```python
if st.session_state.auto_refresh:
    st.empty() # Placeholder for refresh logic
```

Checkbox є, slider є, але фактичного refresh немає.

Рекомендація:

- використати `st_autorefresh` package, якщо дозволено;
- або `time.sleep()` + `st.rerun()` дуже обережно;
- або прибрати auto-refresh UI до реалізації.

---

## 16.3.10. P2: `dashboard/__init__.py` порожній

Не критично. Але можна експортувати `UnifiedDashboard` lazy, або залишити порожнім, якщо dashboard не має імпортуватись як library.

---

# 16.4. Що працює добре

1. Dashboard відокремлений від pipeline.
2. Використовується Streamlit cache:
   - `st.cache_data`
   - `st.cache_resource`
3. Є базова структура tabs:
   - Overview
   - Signals
   - News
   - Risk
   - System
4. Є CPU/RAM monitoring через `psutil`.
5. Є спроба показувати factor exposures і adaptive thresholds.
6. README правильно описує роль dashboard як human-in-the-loop layer.
7. SQL-запити статичні, тобто немає прямої SQL injection через UI input у поточному коді.

---

# 16.5. Що дати Codex по `dashboard`

```text
Deep-fix src/dashboard without changing public UI intent.

Tasks:

1. Fix DataManager call:
   - replace _db_manager.load_data(query) with fetch_df/query_data compatibility;
   - get_data_from_db must always return pd.DataFrame and never crash UI.

2. Fix ticker loading:
   - read tickers from assets.active_preset -> assets.presets[active].tickers;
   - fallback to assets.details keys;
   - do not use list(assets.keys()) as tickers.

3. Use DashboardDataBridge:
   - move DB/schema/fallback logic from main_app.py into src.integration.dashboard_data_bridge;
   - dashboard UI should call bridge.get_dashboard_data(...);
   - support real data and sample data explicitly.

4. Make table/query access safe:
   - check table_exists before querying;
   - handle missing tables/columns gracefully;
   - show st.info instead of crashing.

5. Unify dashboard schema:
   - define canonical dashboard views or adapters for:
     model_performance
     trading_signals
     news
     evaluation_summary
     factor_exposures
     adaptive_thresholds
   - align README and code: news vs news_data.

6. Remove unused heavy imports:
   - remove UnifiedAnalyticsEngine/FamaFrenchFactors/HedgeFundAnalyzer from module import unless actually used;
   - import optional analytics lazily per tab.

7. Improve dependency handling:
   - if DataManager unavailable, show sample dashboard via bridge;
   - if optional analytics unavailable, disable only affected tab.

8. Move st.set_page_config:
   - call once at the start of run/main before any other Streamlit output.

9. Implement or remove auto-refresh placeholder:
   - either use proper rerun/autorefresh;
   - or remove checkbox/slider until implemented.

10. Add tests/smoke checks:
   - import dashboard with streamlit mocked;
   - get_all_configured_tickers returns real tickers from assets.yaml;
   - get_data_from_db handles missing table;
   - dashboard bridge returns sample data when DB unavailable.
```

---

## 16.6. Priority list for `dashboard`

### P0 / must fix

- `DataManager.load_data(query)` call is wrong.
- `get_all_configured_tickers()` returns config keys instead of actual tickers.
- Missing table/column queries can crash dashboard.

### P1 / high priority

- Use `DashboardDataBridge` instead of direct DB calls.
- Align README schema with actual code.
- Remove heavy unused analytics imports.
- Make DataManager/optional dependencies graceful.
- Move `st.set_page_config`.

### P2 / cleanup

- Implement auto refresh or remove placeholder.
- Add lazy export in `dashboard/__init__.py` if needed.
- Add dashboard smoke tests.

---

## 16.7. Summary for `dashboard`

`dashboard` — корисний UI-шар, але зараз він не синхронізований з актуальним backend contract.

Найголовніше:

1. Він викликає неіснуючий `DataManager.load_data`.
2. Він неправильно читає tickers із `assets`.
3. Він напряму запитує таблиці/колонки, які можуть не існувати.
4. Він не використовує вже наявний `DashboardDataBridge`.
5. README і code мають schema mismatch.

Після фіксів dashboard може стати нормальним human-in-the-loop монітором, але зараз його не варто вважати надійним джерелом production visibility.

---

# 17. Audit: `data_sources`

## 17.1. Загальний стан

`data_sources` — дуже маленька папка з adapter-ом для config-driven local file loading.

У наданій папці є:

```text
data_sources/
  __init__.py
  local_file_data_source.py
```

Синтаксично файли компілюються. Імпорт працює.

`LocalFileDataSource` підтримує:

- CSV;
- Parquet;
- optional `date_col`;
- aliases:
  - `load()`
  - `read()`
  - `fetch()`

Головний висновок: **модуль корисний як мінімальний adapter, але зараз він майже не залучений і має path-safety проблеми.** Це має бути безпечний config-driven loader, але поки він читає будь-який шлях напряму.

---

## 17.2. Залученість у проєкті

### Реально згадується в config

`src/config/data_sources.yaml` містить:

```yaml
data_sources:
  local_market_data:
    module: "src.data_sources.local_file_data_source"
    class: "LocalFileDataSource"
    params:
      file_path: "data/raw/market_data.csv"
      file_type: "csv"
      date_col: "date"
```

Тобто config знає про цей adapter.

### Але активного loader/factory майже не видно

Пошук показує, що `LocalFileDataSource` згадується тільки в:

```text
src/config/data_sources.yaml
src/data_sources/local_file_data_source.py
src/data_sources/__init__.py
```

Тобто поки що немає очевидного `DataSourceFactory`, який читає `data_sources.yaml` і створює adapter-и.

Наслідок:

- модуль виглядає підготовленим, але не залученим;
- config-driven idea є, execution layer відсутній або неочевидний.

---

# 17.3. Критичні проблеми

## 17.3.1. P1: немає safe path validation

У `LocalFileDataSource.__init__`:

```python
self.file_path = Path(file_path)
```

У `load()`:

```python
file_path = Path(overrides.get("file_path", self.file_path))
if not file_path.exists():
    raise FileNotFoundError(...)
...
pd.read_csv(file_path)
```

Немає:

- trusted base dir;
- `resolve()`;
- `is_relative_to`;
- заборони `..`;
- allowed suffixes;
- separation between config paths and user override paths.

Якщо file_path приходить із config або runtime overrides, можна прочитати будь-який локальний CSV/parquet, доступний процесу.

Рекомендація:

- приймати `base_dir`;
- resolve path відносно base_dir;
- перевіряти, що path усередині allowed roots;
- дозволити тільки `.csv`, `.parquet`;
- для overrides теж застосовувати same validation.

Приклад:

```python
base = Path(base_dir).resolve()
candidate = Path(file_path)
resolved = (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

if not resolved.is_relative_to(base):
    raise PathValidationError(...)
```

---

## 17.3.2. P1: config comment каже “date column used as index”, але код index не ставить

У `data_sources.yaml`:

```yaml
# Specify the date column to be used as the DataFrame index. If null, no index is set.
date_col: "date"
```

А код робить тільки:

```python
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
```

Індекс не встановлюється.

Наслідок:

- модулі, які очікують datetime index, можуть отримати звичайну колонку;
- коментар у config вводить в оману.

Рішення:

- або оновити comment;
- або додати option:

```python
set_index: bool = False
```

і якщо true:

```python
df = df.set_index(date_col)
```

---

## 17.3.3. P1: немає schema validation

`LocalFileDataSource` не перевіряє:

- required columns;
- дата не повністю NaT;
- empty file;
- duplicate timestamps;
- ticker column;
- numeric columns;
- OHLCV мінімум для market_data.

Рекомендація:

- додати optional `required_columns`;
- додати `schema_name`;
- інтегрувати з існуючим validation layer;
- хоча б повертати warning metadata.

---

## 17.3.4. P1/P2: read_parquet може падати через optional dependencies

```python
pd.read_parquet(file_path)
```

може кинути:

- `ImportError` якщо немає pyarrow/fastparquet;
- `OSError`;
- `PermissionError`;
- Arrow errors.

Зараз exceptions не ловляться. Це може бути нормально для low-level loader, але config-driven ingestion краще має повертати structured error або кидати project-specific `DataSourceError`.

Рекомендація:

- ловити `Exception` і кидати `DataLoadError`/`DataSourceError`;
- логувати error_type.

---

## 17.3.5. P2: `file_type` довіряється config/override, а не suffix

Зараз:

```python
file_type = overrides.get("file_type", self.file_type).lower()
```

Якщо `file_path="x.parquet"`, але `file_type="csv"`, loader спробує `read_csv`.

Рекомендація:

- якщо `file_type="auto"`, визначати по suffix;
- якщо file_type заданий і не збігається з suffix — warning/error;
- allowed suffixes мають відповідати file_type.

---

## 17.3.6. P2: лог може розкривати повний локальний шлях

```python
logger.info(f"Loaded local data source {file_path}: {df.shape}")
```

Для локального dev нормально. Для shared logs краще логувати relative path або sanitized path.

---

## 17.3.7. P2: немає metadata/provenance

Loader повертає тільки DataFrame.

Для pipeline краще мати:

```python
{
  "data": df,
  "source": str(path),
  "file_type": file_type,
  "loaded_at": timestamp,
  "row_count": len(df),
  "schema_warnings": [...]
}
```

Або окремий method `load_with_metadata()`.

---

## 17.4. Що працює добре

1. Мінімальний adapter простий і зрозумілий.
2. Підтримує CSV і Parquet.
3. Має aliases `read()` і `fetch()`, що зручно для різних loader APIs.
4. `date_col` парситься через `pd.to_datetime(errors="coerce")`, тобто не падає на поганих датах.
5. Імпорт модуля легкий, без heavy dependencies окрім pandas.

---

## 17.5. Що дати Codex по `data_sources`

```text
Deep-fix src/data_sources without changing public intent.

Tasks:

1. Add DataSourceFactory:
   - read config/data_sources.yaml;
   - dynamically import module/class;
   - instantiate enabled data sources;
   - expose load/read/fetch interface.

2. Harden LocalFileDataSource path handling:
   - accept base_dir/allowed_roots;
   - resolve relative paths against base_dir;
   - reject path traversal and absolute paths outside allowed roots;
   - allow only .csv/.parquet.

3. Support file_type="auto":
   - infer from suffix;
   - validate file_type matches suffix if explicitly provided.

4. Fix date_col behavior:
   - either update config comment;
   - or add set_index option and implement it.

5. Add schema validation:
   - optional required_columns;
   - optional schema_name;
   - empty DataFrame warning/error;
   - date_col all-NaT warning;
   - duplicate timestamp warning.

6. Improve error handling:
   - catch pandas/parquet/OS errors;
   - raise project-specific DataLoadError/DataSourceError with error_type.

7. Add provenance:
   - provide load_with_metadata();
   - include source path, row_count, columns, loaded_at, schema warnings.

8. Add tests:
   - csv load;
   - parquet load if pyarrow available;
   - missing file;
   - path traversal rejection;
   - date_col parse/set_index;
   - required columns validation.
```

---

## 17.6. Priority list for `data_sources`

### P0 / must fix

- Немає P0, якщо module не використовується production.

### P1 / high priority

- Safe path validation.
- Date index contract mismatch.
- Add DataSourceFactory or mark adapter as unused.
- Schema validation for local market data.
- Parquet/OS error handling.

### P2 / cleanup

- file_type auto/suffix validation.
- sanitized logging.
- provenance metadata.

---

## 17.7. Summary for `data_sources`

`data_sources` — маленький, корисний adapter-шар, але зараз він більше схожий на заготовку.

Найважливіше:

1. Він згаданий у `config/data_sources.yaml`, але активного factory/loader майже не видно.
2. Він читає локальні файли без safe path validation.
3. Коментар у config обіцяє datetime index, але код тільки парсить колонку.
4. Немає schema validation і provenance.

Якщо цей шар планується використовувати для config-driven ingestion, його треба зробити безпечним до підключення в pipeline.

---

# 18. Audit: `ensembling`

## 18.1. Загальний стан

`ensembling` — це стратегічно важлива папка, бо вона має стояти між Stage 5 Prediction і Stage 6 Trading/Signal generation. Її роль — об’єднувати прогнози моделей, рахувати фінальний signal/confidence/divergence і давати consensus для торгового шару.

У наданій папці є:

```text
ensembling/
  README.md
  ensemble.py
  stacked_ensemble.py
  ensemble/
    __init__.py
    ensemble_model.py
    archive/
      adaptive_ensemble.py
```

Синтаксично всі Python-файли компілюються.

Але імпорт модулів у поточному дереві падає:

```text
src.ensembling.stacked_ensemble -> ModuleNotFoundError: No module named 'duckdb'
src.ensembling.ensemble -> ModuleNotFoundError: No module named 'duckdb'
src.ensembling.ensemble.ensemble_model -> ModuleNotFoundError: No module named 'duckdb'
src.ensembling.ensemble.archive.adaptive_ensemble -> ModuleNotFoundError: No module named 'duckdb'
```

Коренева причина для `stacked_ensemble.py`: він на module import тягне:

```python
from src.meta_learning.memory.diary_engine import DiaryEngine
```

а імпорт `src.meta_learning` тягне важкий meta-learning stack, який тягне DuckDB.

Головний висновок: **папка дуже важлива, але зараз має P0-проблеми з імпортами, дублюванням і API-конфліктами.** Найгірше — `Stage 5` і `ConsensusEngine` прямо імпортують `StackedEnsemble`, тому відсутній DuckDB або важкий meta-learning dependency може зламати prediction/trading шар ще до запуску логіки.

---

## 18.2. Залученість у проєкті

`ensembling` реально залучений.

### Прямі згадки

`StackedEnsemble` використовується в:

```text
src/pipeline/stages/stage_5_prediction.py
src/trading/consensus_engine.py
src/models/loader.py
src/analytics/context/ensemble_selector.py
src/scripts/modeling/train_consensus_model.py
```

`ensemble_forecast` використовується в:

```text
src/predictions/models_predict.py
src/experiments/compare_layers.py
src/predictions/prediction_utils.py
```

`EnsembleModel` є в:

```text
src/config/models.yaml
src/factories/model_factory.py
src/models/factory.py
src/models/ensemble/ensemble_model.py
src/ensembling/ensemble/ensemble_model.py
```

`AdaptiveEnsemble` згадується через:

```text
src/integration/ensemble_performance_bridge.py
src/trading/live_adaptive_ensemble.py
src/analytics/context/ensemble_selector.py
```

Отже, це не мертва папка. Вона реально впливає на Stage 5 / consensus / trading.

---

# 18.3. Критичні проблеми

## 18.3.1. P0: `StackedEnsemble` не має ламати import через DuckDB/meta_learning

У `stacked_ensemble.py`:

```python
from src.meta_learning.memory.diary_engine import DiaryEngine
```

Це відбувається на module import.

`DiaryEngine` тягне:

```python
from src.data.management.data_manager import DataManager
```

а DataManager тягне DuckDB.

Наслідок:

- якщо DuckDB не встановлений, `StackedEnsemble` не імпортується;
- `Stage 5 Prediction` не імпортується;
- `ConsensusEngine` не імпортується;
- навіть якщо ensemble хоче просто зробити median/weighted_average без diary, він усе одно падає.

Рекомендація:

- зробити `DiaryEngine` lazy/optional;
- не створювати diary у `__init__` без потреби;
- для method != `stacked` або якщо diary unavailable — fallback to neutral contextual weights.

Приклад:

```python
def _get_diary_engine(self):
    if self.diary_engine is None:
        try:
            from src.meta_learning.memory.diary_engine import DiaryEngine
            self.diary_engine = DiaryEngine()
        except Exception as e:
            logger.warning("DiaryEngine unavailable; using neutral ensemble weights: %s", e)
            self.diary_engine = False
    return self.diary_engine
```

---

## 18.3.2. P0: конфлікт імен `ensemble.py` і папки `ensemble/`

У `src/ensembling` одночасно є:

```text
ensemble.py
ensemble/
  __init__.py
  ensemble_model.py
```

Це небезпечно, бо `src.ensembling.ensemble` може резолвитись як package `ensemble/`, а не як файл `ensemble.py`.

Фактично `ensemble/ __init__.py` робить:

```python
from ..stacked_ensemble import EnsembleResult, StackedEnsemble, ensemble_forecast
```

Тобто `ensemble.py` може бути практично shadowed/legacy, попри те що README називає його “математичним core”.

Наслідок:

- різні імпорти можуть отримати різний код;
- README misleading;
- Codex може фіксити `ensemble.py`, але runtime використовує `stacked_ensemble.py`;
- існують дві реалізації `StackedEnsemble` і `ensemble_forecast`.

Рекомендація:

- перейменувати файл `ensemble.py` у `ensemble_core.py` або `legacy_ensemble.py`;
- або перейменувати папку `ensemble/` у `models/` чи `standard_models/`;
- зробити один canonical import path:
  - `src.ensembling.stacked_ensemble.StackedEnsemble`
  - `src.ensembling.stacked_ensemble.ensemble_forecast`
- README оновити відповідно.

---

## 18.3.3. P0: `ensemble.py` імпортує неіснуючий `ExperienceDiaryEngine`

У `ensemble.py`:

```python
from src.meta_learning.memory.diary_engine import ExperienceDiaryEngine
```

А в `diary_engine.py` є:

```python
class DiaryEngine
```

але немає:

```python
ExperienceDiaryEngine
```

Тобто якщо імпортувати саме файл `ensemble.py`, буде:

```text
ImportError: cannot import name 'ExperienceDiaryEngine'
```

Це ще один доказ, що `ensemble.py` або legacy, або не проходить runtime.

Рекомендація:

- якщо файл legacy — позначити deprecated;
- якщо active — замінити на `DiaryEngine`;
- або зробити compatibility alias у `diary_engine.py`:

```python
ExperienceDiaryEngine = DiaryEngine
```

але краще не плодити aliases без потреби.

---

## 18.3.4. P0/P1: `StackedEnsemble` має train/test leakage risk

`StackedEnsemble.train(X, y)` просто:

```python
self.meta_model.fit(X, y)
```

Це нормально тільки якщо `X` — out-of-fold predictions базових моделей.

Але якщо `X` містить in-sample predictions від моделей, які тренувались на тих самих targets, то stacked meta-model отримає leakage і завищить якість.

У коді немає:

- перевірки, що `X` — OOF predictions;
- timestamp split;
- walk-forward stacking;
- embargo/purge;
- metadata про train windows.

Рекомендація:

- явно перейменувати input contract:
  - `X_oof_predictions`;
- додати `fit_oof()` або `train_from_oof_predictions()`;
- заборонити training на in-sample predictions без `allow_in_sample=True`;
- зберігати metadata:
  - base model training window;
  - prediction window;
  - target horizon.

---

## 18.3.5. P1: `StackedEnsemble._predict_stacked()` не перевіряє колонки

```python
preds_matrix = X[self.feature_names].to_numpy()
```

Якщо у live prediction немає однієї колонки або є зайві/інший порядок:

- KeyError;
- або silent mismatch, якщо до цього X був сформований неочевидно.

Рекомендація:

```python
missing = set(self.feature_names) - set(X.columns)
extra = set(X.columns) - set(self.feature_names)
if missing:
    return structured failure or fill according to policy
X = X.reindex(columns=self.feature_names)
```

Для live trading краще не fill unknown model predictions автоматично без warning.

---

## 18.3.6. P1: weight normalization через `sum(abs(weights))` дозволяє short/negative weights, але `active_weights` misleading

У `_predict_stacked()`:

```python
base_weights = self.meta_model.coef_
adjusted_weights = np.array(base_weights, copy=True)
...
weight_sum = np.sum(np.abs(adjusted_weights))
adjusted_weights /= weight_sum
final_preds = np.dot(preds_matrix, adjusted_weights)
```

Це дозволяє negative meta weights. Це може бути нормально для Ridge stacking.

Але `active_weights_map` заповнюється **до normalization**:

```python
active_weights_map[model_name] = float(adjusted_weights[i])
```

Тобто у результаті `active_weights` не відповідає фактичним normalized weights, використаним для prediction.

Наслідок:

- consensus/explainability бачить неправильні contributions;
- risk/logging може неправильно інтерпретувати модельний внесок.

Фікс:

- заповнювати `active_weights_map` після normalization;
- або повертати окремо:
  - raw coefficients;
  - normalized effective weights.

---

## 18.3.7. P1: weighted_average може давати NaN/inf weights

У `_predict_weighted_average()`:

```python
rmse_scores = [...]
inverse_rmse = [1 / (rmse + 1e-6) for rmse in rmse_scores]
```

Якщо `rmse` NaN або inf, weights стануть NaN.

Для `mape` те саме.

Для `r2`:

```python
max(0, r2)
```

але якщо r2 is NaN — `max(0, np.nan)` може дати NaN/неочікувану поведінку.

Рекомендація:

- sanitize metrics:
  - non-finite -> default;
  - negative R2 -> 0;
  - all weights zero -> equal weights.

---

## 18.3.8. P1: confidence/divergence thresholds hardcoded і scale-dependent

У кількох місцях:

```python
base_confidence = 0.8
extreme_mask = divergence > 0.7
final_confidence[extreme_mask] *= 0.3
```

Але prediction scale може бути:

- returns ~ 0.001;
- normalized signals [-1, 1];
- probabilities [0, 1];
- price values.

`divergence > 0.7` має сенс тільки для normalized signals, але не для returns.

Рекомендація:

- config-driven threshold;
- normalize divergence by signal scale;
- або рахувати disagreement через sign disagreement / directional disagreement;
- для returns додати threshold типу `divergence > k * median_abs_signal`.

---

## 18.3.9. P1: `ensemble_forecast()` padding shorter predictions зліва може створити temporal alignment risk

У `_align_predictions_and_confidences()`:

```python
if len(p) < max_len:
    p = np.pad(p, (max_len - len(p), 0), constant_values=np.nan)
```

Це left-padding. Іноді це правильно, якщо коротший ряд — це пізніший start. Але без datetime index це припущення.

Якщо коротший ряд насправді обрізаний з кінця або має інший timestamp coverage, alignment буде неправильним.

Рекомендація:

- приймати predictions як `pd.Series` з datetime index;
- align by index, не by length;
- якщо масиви без index — вимагати однакову довжину або explicit alignment policy.

---

## 18.3.10. P1: `ensemble_forecast()` не нормалізує constrained base weights

```python
constrained_weights = {m: max(min_weight, min(w, max_weight)) for ...}
```

Далі effective weights нормалізуються per time step через confidence. Це ок для final weighted prediction.

Але `active_weights` у результаті повертає саме `constrained_weights`, не normalized/effective weights.

Наслідок:

- якщо weights `{a: 0.8, b: 0.8}`, active_weights sum = 1.6;
- якщо confidence змінює ваги per time step, active_weights не показує реальні weights.

Рекомендація:

- повертати:
  - base_weights;
  - constrained_weights;
  - average_effective_weights;
- `active_weights` краще зробити average effective normalized weights.

---

## 18.3.11. P1: якщо всі confidence = 0, final signal/confidence стають нулями без clear warning

У `_calculate_effective_weights()`:

```python
normalized_weights = np.divide(..., out=np.zeros_like(...), where=weight_sums != 0)
```

Якщо всі confidence нульові, weights нульові, final_signal через `np.nansum` стане 0.

Це може виглядати як реальний HOLD/signal 0, але насправді це “немає valid confidence”.

Рекомендація:

- якщо `weight_sums == 0`, використовувати equal weights або return low-confidence status;
- stats має містити `zero_weight_steps`.

---

## 18.3.12. P1/P2: `save()` у `stacked_ensemble.py` небезпечніший за `load()`

`load()` використовує:

```python
resolve_trusted_artifact_path(...)
```

і allowed suffixes.

А `save()`:

```python
with open(path, 'wb') as f:
    joblib.dump(state, f)
```

Немає:

- safe output path;
- allowed suffix;
- parent dir creation;
- path traversal guard.

Рекомендація:

- save теж має використовувати trusted output resolver / configured model path;
- створювати parent dir;
- не зберігати `config_manager` object у state без потреби.

---

## 18.3.13. P1/P2: `StackedEnsemble.load()` path check може блокувати валідний configured relative path

```python
base_model_path = config.get('models.dual_model_manager.base_path', 'data/models')

if not trusted_path.resolve().is_relative_to(Path(base_model_path).resolve()):
    raise ValueError(...)
```

Якщо `base_model_path` відносний, `Path(base_model_path).resolve()` рахується від process cwd, а не від project root/config root.

Це типова path bug.

Рекомендація:

- resolve base model path через config manager helper;
- або `project_root / base_model_path`.

---

## 18.3.14. P1: `EnsembleModel` у `src/ensembling/ensemble/ensemble_model.py` дублює `src/models/ensemble/ensemble_model.py`

Є два файли:

```text
src/ensembling/ensemble/ensemble_model.py
src/models/ensemble/ensemble_model.py
```

Обидва мають `EnsembleModel`, але різні реалізації.

`config/models.yaml` вказує на:

```yaml
ensemble:
  module: src.models.ensemble.ensemble_model
  class: EnsembleModel
```

Тобто active model registry використовує `src/models/ensemble/ensemble_model.py`, не `src/ensembling/ensemble/ensemble_model.py`.

Наслідок:

- `src/ensembling/ensemble/ensemble_model.py` може бути legacy/unused;
- фікси в ньому не вплинуть на model factory;
- README згадує `ensemble_model.py`, але неясно який саме.

Рекомендація:

- зробити `src/models/ensemble/ensemble_model.py` canonical model interface;
- `src/ensembling` має містити signal-level ensembling/composition;
- `src/ensembling/ensemble/ensemble_model.py` або прибрати/депрекейтнути, або перетворити на wrapper.

---

## 18.3.15. P1: `EnsembleModel._get_default_models()` очікуує config structure, якої немає в `models.yaml`

У `src/ensembling/ensemble/ensemble_model.py`:

```python
model_config = config_manager.get_config('models')
ensemble_config = model_config.get('ensemble', {})
model_set_key = 'classification_models' if ... else 'regression_models'
models_to_load = ensemble_config.get(model_set_key, {})
```

А `models.yaml` має:

```yaml
models:
  model_definitions:
  dual_model_manager:
  trained_models_registry:
  categories:
  per_model:
  sentiment:
```

Немає:

```yaml
models:
  ensemble:
    classification_models:
    regression_models:
```

Отже `_get_default_models()` поверне порожній список, ensemble буде empty.

Рекомендація:

- або оновити config;
- або видалити цей legacy file;
- або адаптувати до `models.model_definitions` / `models.categories`.

---

## 18.3.16. P1: `EnsembleModel.load_model()` використовує unsafe `joblib.load` без trusted path

У `src/ensembling/ensemble/ensemble_model.py`:

```python
metadata = joblib.load(path)
```

Немає trusted path validation, на відміну від інших модулів.

Так само у `save_model()` path не валідується.

Рекомендація:

- використовувати `resolve_trusted_artifact_path` для load;
- safe output path для save;
- suffix whitelist;
- не load arbitrary pickle/joblib з user path.

---

## 18.3.17. P1/P2: `AdaptiveEnsemble` в archive має старі absolute imports і leakage

`ensemble/archive/adaptive_ensemble.py` імпортує:

```python
from utils.logger import ProjectLogger
from models.ensemble_model import EnsembleModel
```

Це старі non-`src.` imports і в поточній структурі не працюють.

Також методи adaptive/meta weighting приймають `y_true` і одразу в тому ж методі використовують його для створення predictions:

```python
results['weighted_average'] = self._weighted_average(model_predictions, y_true)
results['adaptive_weighted'] = self._adaptive_weighted_average(model_predictions, y_true)
results['meta_learner'] = self._meta_learner_ensemble(model_predictions, y_true)
```

А `_meta_learner_ensemble()` для першої половини повертає actual y:

```python
full_pred[:split_point] = y_train_meta
```

Це прямий leakage, якщо ці predictions підуть у evaluation як модельний результат.

Файл у `archive`, тому не треба фіксити першим. Але його не можна випадково підключати production.

Рекомендація:

- залишити `archive` як non-production;
- додати README/guard;
- або перенести в `legacy/` і прибрати з import discovery.

---

## 18.3.18. P2: `ensemble_forecast()` має duplicated implementation у `ensemble.py` і `stacked_ensemble.py`

Є дві схожі функції:

```text
ensembling/ensemble.py -> ensemble_forecast
ensembling/stacked_ensemble.py -> ensemble_forecast
```

Через name conflict невідомо, яку реально використовують старі імпорти.

Рекомендація:

- одна canonical function;
- інша — wrapper з deprecation warning.

---

# 18.4. Що працює добре

1. `ensemble_forecast()` має хорошу ідею:
   - model_predictions;
   - optional confidences;
   - regime-based base weights;
   - max/min weight constraints;
   - divergence shrinkage;
   - optional rolling smoothing.

2. `StackedEnsemble` підтримує кілька методів:
   - stacked;
   - weighted_average;
   - median;
   - voting.

3. Є `EnsembleResult` з нормальним contract:
   - final_signal;
   - confidence;
   - divergence;
   - active_weights;
   - stats.

4. `load()` у `stacked_ensemble.py` вже має partial trusted path validation.

5. `median` і `voting` — корисні robust fallback-и.

6. README правильно описує роль ensembling як decision merger між prediction і trading.

---

# 18.5. Що дати Codex по `ensembling`

```text
Deep-fix src/ensembling without changing public intent.

Tasks:

1. Fix import stability:
   - StackedEnsemble must not import DiaryEngine/meta_learning/DuckDB on module import;
   - make DiaryEngine lazy/optional;
   - if unavailable, use neutral contextual weights and warn.

2. Resolve ensemble.py vs ensemble/ package conflict:
   - choose canonical module path;
   - rename ensemble.py to ensemble_core.py or legacy_ensemble.py;
   - update README and imports;
   - keep compatibility wrapper if needed.

3. Fix legacy ExperienceDiaryEngine import:
   - ensemble.py imports non-existing ExperienceDiaryEngine;
   - replace with DiaryEngine or deprecate file.

4. Define canonical ensembling architecture:
   - src.ensembling.stacked_ensemble = signal-level ensemble composer;
   - src.models.ensemble.ensemble_model = sklearn/model-interface ensemble;
   - archive/adaptive_ensemble.py = legacy only.

5. Guard stacking against leakage:
   - require OOF/walk-forward predictions for StackedEnsemble.train;
   - add metadata/flag allow_in_sample=False by default;
   - document X as base model OOF predictions.

6. Validate prediction columns:
   - check missing/extra model prediction columns before predict;
   - reindex by feature_names explicitly;
   - return structured failure or warning for missing models.

7. Fix active_weights reporting:
   - for stacked method, return normalized effective weights, not pre-normalization raw values;
   - optionally include raw coefficients separately.

8. Sanitize model metrics for weighted_average:
   - handle NaN/inf/negative values;
   - fallback to equal weights when invalid.

9. Make divergence threshold scale-aware:
   - move threshold to config;
   - support directional disagreement metric;
   - avoid hardcoded 0.7 for all prediction scales.

10. Fix alignment:
   - prefer pd.Series/DataFrame with datetime index;
   - align predictions by index;
   - if arrays are used, require equal length or explicit alignment policy;
   - do not silently left-pad without metadata.

11. Handle zero confidence case:
   - if all weights/confidences are zero for a time step, fallback to equal weights or mark low-confidence/no_valid_weights;
   - include zero_weight_steps in stats.

12. Harden save/load:
   - save should use safe/trusted output path and create parent dirs;
   - load base path should resolve relative to project root, not cwd;
   - do not serialize config_manager unless necessary.

13. Fix or remove duplicate EnsembleModel:
   - src/models/ensemble/ensemble_model.py is canonical per models.yaml;
   - src/ensembling/ensemble/ensemble_model.py should be deprecated or adapted;
   - if kept, update config contract for default models.

14. Secure EnsembleModel load/save:
   - use resolve_trusted_artifact_path for joblib.load;
   - whitelist suffixes;
   - safe output path for save.

15. Quarantine archive/adaptive_ensemble.py:
   - mark non-production;
   - fix old imports only if needed;
   - do not use y_true to generate live predictions;
   - remove direct leakage behavior if restored.

16. Add tests:
   - import src.ensembling.stacked_ensemble without duckdb installed;
   - ensemble_forecast equal-length arrays;
   - ensemble_forecast mismatched lengths with explicit policy;
   - zero confidence fallback;
   - active_weights sum/shape;
   - StackedEnsemble missing columns;
   - save/load safe path;
   - no accidental import of archive module.
```

---

## 18.6. Priority list for `ensembling`

### P0 / must fix

- `StackedEnsemble` import depends on `DiaryEngine`/DuckDB.
- `ensemble.py` vs `ensemble/` name conflict.
- `ensemble.py` imports non-existing `ExperienceDiaryEngine`.
- Stage 5 and ConsensusEngine depend on `StackedEnsemble`, so import failure can block prediction/trading.

### P1 / high priority

- Stacking leakage risk if trained on in-sample base predictions.
- Missing/extra prediction columns not handled.
- `active_weights` reported before normalization.
- hardcoded divergence threshold is scale-dependent.
- array left-padding can misalign time series.
- duplicate `EnsembleModel` with config mismatch.
- unsafe joblib load/save in duplicate `EnsembleModel`.

### P2 / cleanup

- archive/adaptive_ensemble old imports and leakage.
- duplicate `ensemble_forecast`.
- save path hardening.
- better stats for zero-confidence/zero-weight cases.
- README update.

---

## 18.7. Summary for `ensembling`

`ensembling` — не просто допоміжний модуль, а критичний шар між прогнозом і торговим рішенням. Але зараз він має “авгієві” проблеми саме інтеграційного типу:

1. import `StackedEnsemble` може падати через DuckDB/meta_learning;
2. є конфлікт `ensemble.py` vs `ensemble/`;
3. є legacy import `ExperienceDiaryEngine`, якого немає;
4. є дубльовані EnsembleModel-и;
5. archive-код має старі imports і leakage;
6. stacking може бути leakage-prone, якщо не OOF;
7. final weights/confidence пояснюються не зовсім чесно.

Правильний напрям: зробити `src.ensembling.stacked_ensemble` легким, стабільним, без важких import side effects, а meta-learning/diary weighting — optional adapter. Після цього вже можна підключати ensemble до Stage 5/ConsensusEngine як production-safe компонент.

---

# 23. Audit: `features`

## 23.1. Загальний стан

`features` — це один із найважливіших шарів проєкту. Він відповідає за:

- Stage 3 feature engineering;
- enrichers;
- technical indicators;
- macro/news/sentiment/NLP enrichment;
- feature selection;
- feature cache;
- leakage guard;
- news event datasets;
- decay modeling;
- feature drift/redundancy/regime importance analysis.

У наданому архіві:

```text
features/
  analysis/
  builders/
  enrichers/
  monitoring/
  nlp/
  selection/
  utils/
  validation/
  feature_orchestrator.py
  feature_selector.py
  feature_selection_cache.py
  ...
```

Синтаксична перевірка дала **2 критичні помилки компіляції**:

```text
features/feature_orchestrator.py
  IndentationError: unindent does not match any outer indentation level, line 400

features/enrichers/technical_analysis_enricher.py
  IndentationError: unindent does not match any outer indentation level, line 411
```

Це P0, бо `FeatureOrchestrator` — центральний компонент Stage 3, а `TechnicalAnalysisEnricher` — базовий enricher для технічних індикаторів.

Головний висновок: **папка дуже корисна й жива, але зараз Stage 3 може бути фактично заблокований через syntax/indentation errors.** Після цього треба фіксити temporal leakage/missing policy/heavy imports/config side effects.

---

## 23.2. Залученість у проєкті

`features` реально залучений.

### Прямі залежності

`FeatureOrchestrator` згадується в:

```text
src/pipeline/stages/features/orchestrator_manager.py
src/pipeline/stages/feature_engineering/enricher.py
src/pipeline/stages/prediction/data_preparer.py
src/main/modes/training_data_pipeline.py
src/features/enrichers/*
```

`FeatureEngineeringStage` згадується в:

```text
src/pipeline/pipeline_orchestrator.py
src/pipeline/stages/stage_3_feature_engineering.py
src/pipeline/stages/feature_engineering/orchestrator.py
```

`FeatureLeakageGuard` згадується в:

```text
src/pipeline/hybrid/colab_manager.py
src/features/validation/feature_leakage_guard.py
```

`SmartFeatureSelector` згадується в:

```text
src/features/feature_selector.py
src/features/feature_selection_cache.py
src/features/selection/smart_selector.py
src/features/selection/enhanced_smart_selector.py
src/features/colab_context_integration.py
```

`NewsEventDatasetBuilder` згадується в:

```text
src/data/validation/event_dataset_validator.py
src/features/builders/news_event_dataset_builder.py
```

Тобто це не мертвий код. Це основа feature layer.

---

# 23.3. Критичні проблеми

## 23.3.1. P0: `feature_orchestrator.py` не компілюється

Помилка:

```text
IndentationError: unindent does not match any outer indentation level
feature_orchestrator.py, line 400
```

Фрагмент:

```python
0396:         except (...) as e:
0397:             logger.error(...)
0399:         return df
0400:       return df
```

На line 400 є зайвий/криво відступлений:

```python
return df
```

Наслідок:

- `src.features.feature_orchestrator` не імпортується;
- Stage 3 feature engineering може не стартувати;
- будь-який manager, який імпортує orchestrator, падає.

Рекомендація:

- прибрати line 400 або вирівняти правильно;
- додати compile/import test для `src.features.feature_orchestrator`;
- після фіксу прогнати stage 3 smoke test.

---

## 23.3.2. P0: `technical_analysis_enricher.py` не компілюється

Помилка:

```text
IndentationError: unindent does not match any outer indentation level
technical_analysis_enricher.py, line 411
```

Фрагмент:

```python
0409:         except Exception:
0410:             return 0.5
0411:      return 0.5
```

На line 411 є зайвий/криво відступлений:

```python
return 0.5
```

Наслідок:

- `TechnicalAnalysisEnricher` не імпортується;
- `AdvancedAnalyticsEnricher`, який імпортує `TechnicalAnalysisEnricher`, теж може впасти;
- Stage 3 може втратити технічні індикатори.

Рекомендація:

- видалити зайвий line 411;
- додати compile/import test;
- перевірити, що `_calculate_hurst_exponent` має один fallback return.

---

## 23.3.3. P0/P1: `features/selection/__init__.py` імпортує важкий Enhanced selector на package import

`features/selection/__init__.py` робить:

```python
from .enhanced_smart_selector import EnhancedSmartFeatureSelector, ...
from .smart_selector import SmartFeatureSelector
```

`EnhancedSmartFeatureSelector` імпортує:

```python
get_feature_drift_monitor
get_data_freshness_monitor
get_redundancy_detector
get_regime_importance_tracker
get_news_decay_modeler
```

У моєму середовищі імпорт:

```python
src.features.selection.smart_selector
```

падав через:

```text
ModuleNotFoundError: No module named 'duckdb'
```

через важкий monitoring/data path.

Наслідок:

- простий імпорт `src.features.selection` може тягнути monitoring/DuckDB;
- feature selection стає залежним від optional runtime infrastructure;
- пакетний import може ламати легкі задачі.

Рекомендація:

- прибрати eager imports із `selection/__init__.py`;
- зробити lazy `__getattr__`;
- `EnhancedSmartFeatureSelector` має бути optional;
- `SmartFeatureSelector` не має тягнути monitoring/DuckDB на import.

---

## 23.3.4. P1: `FeatureOrchestrator` має dynamic context selection, але повертає `df`, не enriched df

У кінці `_run_dynamic_context_selection()`:

```python
if dynamic_context_features:
    run_kwargs['selected_features'] = dynamic_context_features
...
return df
```

Це може бути нормально, якщо метод тільки мутує `run_kwargs`. Але назва й інтеграція можуть створювати очікування, що він повертає змінений DataFrame.

Також через syntax error line 400 зараз не працює взагалі.

Рекомендація:

- явно документувати, що method mutates `run_kwargs`;
- або повернути `(df, run_kwargs)`;
- додати test, що selected_features реально доходять до `ContextMapEnricher`.

---

## 23.3.5. P1: `TechnicalAnalysisEnricher` має potential lookahead/rolling issues, але частково вже виправлений

Позитивно: видно, що частина rolling metrics рахується rolling-window style, наприклад:

```python
rolling autocorrelation
rolling Hurst
rolling skew/kurtosis
rolling Sharpe/Sortino
```

Але треба перевірити весь файл після виправлення syntax error:

- чи всі rolling features використовують тільки past/current values;
- чи немає full-series statistics;
- чи Fama-French proxy не використовує майбутні дані;
- чи `fillna(0)` не маскує missing active indicators.

Рекомендація:

- після syntax fix прогнати targeted leakage audit саме для technical indicators;
- додати unit tests для rolling features: value at t не має змінюватися при додаванні future rows.

---

## 23.3.6. P1: global/local fill policy ризикована

Пошук показав багато місць:

```text
.ffill()
.bfill()
fillna(0)
dropna()
```

Приклади:

```text
enrichers/context_map_enricher.py
enrichers/improved_sentiment_enricher.py
enrichers/macro_features_enricher.py
enrichers/sentiment_features_enricher.py
selection/volatility_driver_selector.py
utils/*adaptive_technical_indicators.py
validation/redundancy_detector.py
```

Не кожен `ffill` поганий. Для macro/sentiment іноді forward-fill має сенс. Але має бути:

- per ticker;
- per timeframe;
- з limit;
- без перетікання між assets;
- без fill майбутніми значеннями;
- з provenance flags.

Ризики:

- cross-ticker leakage;
- stale sentiment/macro values без позначки;
- validation/test contamination;
- zero-fill може створити “сигнал” замість missing.

Рекомендація:

- централізувати missing policy;
- кожен enricher має повертати `*_is_missing`, `*_age`, `*_source`;
- не використовувати broad `fillna(0)` без audit-ignore і пояснення.

---

## 23.3.7. P1: `ImprovedSentimentEnricher` має potential stale sentiment leakage/incorrect decay

У `_handle_missing_sentiment()`:

```python
filled_sentiment = sentiment_series.ffill(limit=5)
last_known = filled_sentiment.dropna().iloc[-1]
...
for i, (idx, val) in enumerate(filled_sentiment.items()):
    if pd.isna(val):
        days_since_last = i
        decayed_value = last_known * (decay_factor ** days_since_last)
```

Проблема: `last_known` береться як останнє відоме значення у всій ticker series, тобто може бути майбутнім відносно ранніх NaN.

Якщо на початку series є NaN, `last_known` може бути з майбутнього.

Рекомендація:

- decay має бути strictly forward-only;
- ітерувати по часу й тримати `last_seen_value` тільки з минулого;
- не брати `.iloc[-1]` для заповнення попередніх NaN.

---

## 23.3.8. P1: `NewsEventDatasetBuilder` додає after candles і target columns у record

У `NewsEventDatasetBuilder`:

```python
candles_before = get_candles_before(...)
candles_after = get_candles_after(..., n=2)
...
record.update(... candle_before ...)
self._add_targets_from_candle(record, candle_before, ticker)
record.update(... candles_after[0], suffix='_+1')
record.update(... candles_after[1], suffix='_+2')
```

Це може бути правильно для **event dataset / supervised target generation**, але дуже небезпечно, якщо такий dataset потрапить у feature training як звичайні features.

Потрібен чіткий contract:

- columns із suffix `_+1`, `_+2` — це future outcome features/labels, не inference features;
- вони мають бути або targets, або excluded from feature matrix;
- `FeatureLeakageGuard` має це ловити.

Рекомендація:

- додати `is_future_feature=True` metadata;
- називати future columns `target_*` або `future_*`;
- заборонити `_+1/_+2` columns у inference feature set.

---

## 23.3.9. P1: `FeatureLeakageGuard` корисний, але має бути обов’язковим gate

`FeatureLeakageGuard` існує й згадується в `colab_manager`, але треба перевірити, чи він реально запускається перед training/modeling для всіх paths.

Рекомендація:

- Stage 3/4 має мати hard gate:
  - suspicious future suffixes;
  - target columns in features;
  - columns with `target_`, `future_`, `+1`, `+2`;
  - shift-generated labels accidentally in features;
- guard має писати report artifact.

---

## 23.3.10. P1: `FeatureSelectionCache` cache key може бути слабким

`FeatureSelectionCache` кешує selected features на основі:

- model_type;
- target_name;
- market_regime;
- available_features.

Але якщо target values/data distribution змінилися, а список features той самий, cache може повернути старі selected features.

Рекомендація:

- додати data fingerprint:
  - target hash;
  - feature sample hash;
  - train period;
  - ticker/timeframe;
  - feature version;
- або invalidation after Stage 3/target changes.

---

## 23.3.11. P1: `SmartFeatureSelector._clean_data()` median-imputation може leakage-нути, якщо fitted на full data

```python
features_clean.fillna(features_clean.median())
```

Якщо selector запускається на full train+val+test, median бачить майбутні дані.

Рекомендація:

- feature selection має виконуватись тільки на train split;
- imputer statistics мають зберігатись;
- для time-series validation потрібен walk-forward feature selection або pre-split gate.

---

## 23.3.12. P1: `NewsDecayModeler` вирівнює returns у вікні до і після новини

У `_prepare_modeling_data()`:

```python
start_time = news_time - time_window
end_time = news_time + time_window
relevant_returns = market_returns[(timestamp >= start_time) & (timestamp <= end_time)]
relevant_returns['hours_since_news'] = ...
```

Для decay model fitting це може бути нормально, якщо мета — дослідити before/after pattern. Але для production feature generation не можна використовувати post-event future returns як input.

Рекомендація:

- чітко позначити `NewsDecayModeler` як offline analysis/training tool;
- не використовувати його output для live features без frozen parameters;
- fitted decay params мають бути trained on past only.

---

## 23.3.13. P1: `DecayModelFitter.select_best_overall_model()` має division-by-zero risk

```python
mse_norm = (mse - min_mse) / (max_mse - min_mse)
mae_norm = ...
r2_norm = ...
```

Якщо всі MSE однакові або всі MAE/R2 однакові, denominator = 0.

Наслідок:

- NaN scores;
- wrong best model;
- RuntimeWarning.

Рекомендація:

- safe normalization:

```python
den = max_val - min_val
if den == 0:
    return 0.0
```

---

## 23.3.14. P1/P2: heavy NLP imports/downloads мають бути optional і cached

`finbert_pipeline.py` lazy-loads:

```python
torch
transformers
AutoTokenizer.from_pretrained("ProsusAI/finbert")
AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

Це добре, що lazy. Але `except` ловить тільки:

```text
ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError
```

Не ловить:

- `ImportError`;
- network/download errors;
- `OSError`;
- HF auth/cache errors.

Те саме стосується summarizer/roberta/spacy.

Рекомендація:

- ловити `Exception` на boundary NLP loader-а;
- повертати graceful fallback simple sentiment;
- явно конфігурувати offline mode/cache dir;
- не блокувати Stage 3 через HF download failure.

---

## 23.3.15. P1/P2: `news_harmonizer` використовує `pd.Timestamp.now()` при поганій даті

У `harmonize_entry()`:

```python
if pd.isna(pub_dt):
    pub_dt = pd.Timestamp.now()
```

Це небезпечно:

- погана дата перетворюється в “поточну”;
- може створити future/now leakage;
- news temporal alignment стане неправильним.

Рекомендація:

- якщо дата невалідна — повертати invalid row / skip;
- або додавати `date_parse_failed=True`, але не підставляти now як published_at.

---

## 23.3.16. P1/P2: `monitoring/feature_drift_detector.py` створює output dir на init

```python
self.output_dir = Path(output_dir)
self.output_dir.mkdir(parents=True, exist_ok=True)
```

Для monitoring це нормально в runtime, але import/constructor у tests/devtools може мутувати filesystem.

Рекомендація:

- create dirs lazy only when saving report;
- або configurable `create_output_dir`.

---

## 23.3.17. P1/P2: `analysis/__init__.py` eager imports heavy analysis modules

`features/analysis/__init__.py` імпортує:

```python
NewsDecayModeler
RegimeImportanceTracker
```

Ці модулі створюють/використовують storage paths, scipy/sklearn і можуть тягнути side effects.

Рекомендація:

- lazy exports;
- не тягнути full analysis stack при `import src.features.analysis`.

---

## 23.3.18. P2: багато generated/cache `.pyc` у архіві

Архів містить `__pycache__/*.pyc`.

Це не runtime bug, але не треба тримати в repo/archive:

- шумить аудит;
- може плутати з актуальним source;
- збільшує архів.

Рекомендація:

- `.gitignore` / clean archives;
- не передавати `__pycache__`.

---

# 23.4. Дублювання / legacy

## 23.4.1. Багато technical indicator implementations

Є кілька файлів:

```text
utils/adaptive_technical_indicators.py
utils/hybrid_adaptive_technical_indicators.py
utils/modular_adaptive_technical_indicators.py
utils/simple_adaptive_technical_indicators.py
utils/ultimate_adaptive_technical_indicators.py
utils/technical_indicators_lib.py
enrichers/technical_analysis_enricher.py
```

Це може бути історія експериментів. Але зараз треба визначити canonical.

Рекомендація:

- `technical_indicators_lib.py` як low-level formulas;
- `technical_analysis_enricher.py` як orchestrator;
- інші adaptive/ultimate/simple — або strategies, або legacy with clear docs.

## 23.4.2. improved vs non-improved enrichers

Є:

```text
improved_macro_enricher.py
macro_features_enricher.py
improved_sentiment_enricher.py
sentiment_features_enricher.py
improved_news_impact_enricher.py? / news_impact_enricher.py
```

У наданому архіві є improved macro/sentiment, але не всі improved файли, які згадуються в pycache.

Потрібен мапінг:

- active enricher;
- deprecated enricher;
- compatibility alias.

## 23.4.3. Multiple selectors

```text
feature_selector.py
context_aware_feature_selector.py
selection/smart_selector.py
selection/enhanced_smart_selector.py
selection/volatility_driver_selector.py
feature_selection_cache.py
```

Корисно, але потрібно чітко розділити:

- simple selector;
- context-aware selector;
- train-time selector;
- live volatility/context selector;
- cache.

---

# 23.5. Що працює добре

1. Є багато корисних модулів, і більшість компілюється.
2. NLP loaders частково lazy.
3. Є `FeatureLeakageGuard` — правильна ідея.
4. Є `RedundancyDetector` — корисно перед modeling.
5. Є `FeatureDriftDetector` — корисно для monitoring.
6. Є `NewsEventDatasetBuilder` — корисно для event-based analysis.
7. Є modular `news_event` builder components:
   - filter;
   - candle seeker;
   - enricher.
8. `TechnicalAnalysisEnricher` має спробу rolling/windowed calculations.
9. `EnhancedSmartFeatureSelector` задуманий як інтеграція drift/redundancy/regime/news decay, тобто напрям хороший.
10. `feature` шар має багато hooks для Colab/context selection.

---

# 23.6. Що дати Codex по `features`

```text
Deep-fix src/features without changing public architecture.

Tasks:

1. Fix syntax errors first:
   - remove/fix stray return in src/features/feature_orchestrator.py line ~400;
   - remove/fix stray return in src/features/enrichers/technical_analysis_enricher.py line ~411;
   - add compile/import tests for all src.features modules.

2. Make feature selection imports lazy:
   - features/selection/__init__.py must not eagerly import EnhancedSmartFeatureSelector;
   - Enhanced selector must not pull monitoring/DuckDB on package import;
   - SmartFeatureSelector should be importable without DuckDB.

3. Stabilize FeatureOrchestrator:
   - document _run_dynamic_context_selection mutates run_kwargs;
   - ensure selected_features reach ContextMapEnricher;
   - add Stage 3 smoke test.

4. Add mandatory leakage gate:
   - run FeatureLeakageGuard before training/Colab export;
   - block target_, future_, _+1, _+2 columns from feature matrix;
   - write leakage report artifact.

5. Fix sentiment missing policy:
   - ImprovedSentimentEnricher must be strictly forward-only;
   - do not use last known value from future rows to fill earlier NaNs;
   - add sentiment_age/source/missing flags.

6. Audit all ffill/bfill/fillna(0):
   - group by ticker/timeframe;
   - limit forward fill;
   - add provenance flags;
   - avoid broad zero-fill unless explicitly justified.

7. Harden NewsEventDatasetBuilder:
   - mark after-candle columns as future/target;
   - ensure _+1/_+2 columns are never used as inference features;
   - document event-dataset-only contract.

8. Fix DecayModelFitter normalization:
   - handle zero denominators when all mse/mae/r2 equal;
   - avoid NaN combined scores.

9. Treat NewsDecayModeler as offline training:
   - do not use post-news future returns in live features;
   - freeze learned decay parameters before production use;
   - save training period/provenance.

10. Harden NLP loaders:
   - catch ImportError/OSError/HF download errors;
   - support offline/cache mode;
   - fallback to simple sentiment/scoring without blocking Stage 3.

11. Fix news_harmonizer invalid date policy:
   - do not replace invalid published_at with now;
   - skip invalid rows or mark date_parse_failed.

12. Reduce side effects:
   - FeatureDriftDetector should create output dirs lazily;
   - analysis/__init__.py should use lazy exports;
   - avoid get_current_config() side effects in simple feature helpers.

13. Clean legacy/duplicates:
   - choose canonical technical indicator implementation;
   - classify improved vs old enrichers;
   - document selector roles.

14. FeatureSelectionCache:
   - include data fingerprint/target hash/train period in cache key;
   - invalidate when features/target distribution changes.

15. Add tests:
   - compile/import all features modules;
   - Stage 3 minimal run;
   - technical rolling features no-lookahead test;
   - sentiment missing forward-only test;
   - event dataset future columns blocked;
   - feature cache invalidates on target change;
   - NLP loader missing dependency fallback.
```

---

## 23.7. Priority list for `features`

### P0 / must fix

- `feature_orchestrator.py` indentation error.
- `technical_analysis_enricher.py` indentation error.
- Stage 3 import/compile test.

### P1 / high priority

- `features.selection` eager imports heavy monitoring/DuckDB stack.
- Mandatory leakage guard before training/Colab export.
- Sentiment forward-fill can use future `last_known`.
- Event dataset after-candle columns must be blocked from feature matrix.
- Broad ffill/fillna policies need per ticker/timeframe/provenance.
- FeatureSelectionCache needs data/target fingerprint.
- NLP loaders need ImportError/OSError/HF fallback.
- `news_harmonizer` must not replace invalid dates with now.

### P2 / cleanup

- duplicate technical indicator implementations.
- improved vs old enricher naming.
- lazy exports in `analysis/__init__.py`.
- FeatureDriftDetector output dir side effects.
- remove `__pycache__` from archive/repo.

---

## 23.8. Summary for `features`

`features` — це не просто допоміжна папка, а серце Stage 3. Там багато корисної роботи вже зроблено: enrichers, NLP, leakage guard, feature selection, drift/redundancy, news event datasets. Але прямо зараз є дві P0 syntax помилки, які можуть зупинити весь feature engineering layer.

Після P0-фіксів головний фокус має бути не “додати ще фіч”, а зробити шар безпечним:

1. no lookahead;
2. no future event columns in features;
3. no cross-ticker fill;
4. no stale cache;
5. no heavy optional imports on package import;
6. strict train-only fitting for selectors/scalers.

Тільки після цього можна довіряти Stage 4/5 моделям, бо моделі настільки хороші, наскільки чистий їх feature matrix.

---

# 24. Audit: `integration`

## 24.1. Загальний стан

`integration` — це внутрішній bridge-шар, не зовнішні API. Він з’єднує вже наявні компоненти проєкту:

```text
integration/
  README.md
  dashboard_data_bridge.py
  ensemble_performance_bridge.py
  ensemble_selector.py
```

Синтаксично всі Python-файли компілюються. Імпорти проходять:

```text
src.integration.dashboard_data_bridge      OK
src.integration.ensemble_performance_bridge OK
src.integration.ensemble_selector           OK
```

Головний висновок: **це корисний bridge-шар, але він поки слабко залучений. `DashboardDataBridge` якраз треба використати для dashboard, бо в аудиті dashboard було видно, що dashboard напряму ламається через неправильний DataManager API.**

---

## 24.2. Залученість у проєкті

### `DashboardDataBridge`

Пошук показує майже тільки:

```text
src/integration/dashboard_data_bridge.py
src/integration/README.md
```

Тобто bridge існує, але dashboard його не використовує. Це важливо, бо в попередньому аудиті `dashboard/main_app.py` було знайдено:

```python
_db_manager.load_data(query)
```

а це неактуальний API. `DashboardDataBridge` уже має compatibility method:

```python
if hasattr(self.data_manager, 'fetch_df'):
    return self.data_manager.fetch_df(query, params)
if hasattr(self.data_manager, 'query_data'):
    return self.data_manager.query_data(query)
```

Отже правильний напрям: **dashboard має йти через `DashboardDataBridge`, а не напряму в DataManager.**

### `EnsemblePerformanceBridge`

Згадується в:

```text
src/integration/ensemble_performance_bridge.py
src/analytics/arena/arena_orchestrator.py
src/analytics/arena/ensemble_performance_bridge.py
```

Тобто є дубль/аналог у `analytics/arena`.

### `ensemble_selector.py`

Це маленький compatibility wrapper:

```python
from src.analytics.context.ensemble_selector import EnsembleContext, EnsembleSelector
```

Корисно для backward compatibility, але треба не плодити ще один active selector.

---

# 24.3. Критичні проблеми

## 24.3.1. P1: `DashboardDataBridge` не перевіряє існування таблиць/колонок

Bridge робить hardcoded SQL:

```sql
FROM model_performance
FROM trading_signals
FROM portfolio_performance
FROM market_data
FROM model_arena_results
```

і очікує конкретні колонки:

```text
avg_win_rate
avg_sharpe_ratio
avg_precision
total_trades
last_updated
signal_type
confidence
timestamp
pnl
date/open/high/low/close/volume
```

Якщо таблиці/колонки немає, `fetch_df()` може кинути DB exception.

Bridge ловить тільки:

```text
ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError
```

Не ловить:

- DuckDB errors;
- pandas SQL errors;
- OSError;
- custom DB errors.

Наслідок:

- dashboard може падати замість sample fallback;
- bridge не виконує свою головну роль “safe adapter”.

Рекомендація:

- додати `safe_query_df()`:
  - catch broad `Exception` на DB boundary;
  - логувати `error_type`;
  - повертати empty DataFrame.
- якщо `data.empty` або query failed — повертати sample data з `is_sample_data=True`.

---

## 24.3.2. P1: `DashboardDataBridge` має schema mismatch із реальними таблицями

У попередніх аудитах data/dashboard було видно, що реальні таблиці можуть бути:

```text
market_data_raw
google_news
rss_news
newsapi_articles
fred_data
vix_data
fear_greed_data
model_results
evaluation_summary
```

А bridge очікує:

```text
model_performance
portfolio_performance
market_data
model_arena_results
```

Це не обов’язково погано, якщо є окремі dashboard summary tables. Але тоді їх треба явно створювати stage-ом або materialized views.

Рекомендація:

- визначити canonical dashboard schema;
- або створити `DashboardViewBuilder`, який наповнює:
  - `dashboard_model_performance`
  - `dashboard_trading_activity`
  - `dashboard_portfolio_metrics`
  - `dashboard_market_data`
- або bridge має fallback adapters до raw tables.

---

## 24.3.3. P1: cache key через `hash(str(sorted(kwargs.items())))` нестабільний між процесами

```python
cache_key = f"{data_type}_{hash(str(sorted(kwargs.items())))}"
```

Python `hash()` randomized per process. Для in-memory cache в межах одного процесу це не критично. Але:

- cache key не відтворюваний;
- якщо колись буде persistent cache — не підходить;
- kwargs із несеріалізованими об’єктами можуть давати нестабільний string.

Рекомендація:

```python
json.dumps(kwargs, sort_keys=True, default=str)
sha256(...)
```

---

## 24.3.4. P1/P2: sample data не позначене достатньо жорстко для UI

Bridge додає:

```python
data_source = "sample"
is_sample_data = True
```

Це добре. Але dashboard має явно показувати banner:

```text
Sample data — DB unavailable or query failed
```

Інакше користувач може сприйняти mock signals як реальні.

Рекомендація:

- dashboard UI має показувати sample warning;
- bridge має додавати `sample_reason`.

---

## 24.3.5. P1/P2: `DashboardDataBridge` не параметризує частину запитів і не має date/ticker controls

Market data використовує parameterized ticker:

```python
WHERE ticker = ?
```

Це добре.

Але інші queries жорстко привʼязані до:

```text
last 7 days
last 1 day
last 30 days
```

Рекомендація:

- `days`, `start_date`, `end_date` як kwargs;
- параметризувати date windows, де можливо;
- унеможливити arbitrary SQL із UI.

---

## 24.3.6. P1: `EnsemblePerformanceBridge` очікує неформалізований API live_ensemble/performance_tracker

Методи очікують, що об’єкти мають:

```text
live_ensemble.get_performance_summary()
live_ensemble.get_current_weights()
live_ensemble.update_model_weights()
performance_tracker.update_performance(record)
performance_tracker.get_all_performance()
performance_tracker.get_model_performance(model_name)
```

Але немає Protocol/interface.

Наслідок:

- runtime AttributeError;
- складно тестувати;
- adapters можуть розʼїхатись.

Рекомендація:

- додати `Protocol` або abstract interface;
- validate required methods у `__init__`;
- повертати structured error, якщо методів немає.

---

## 24.3.7. P1: `EnsemblePerformanceBridge` може втрачати/перезаписувати дані через merge by `model_name`

`_merge_performance_data()` об’єднує ensemble і tracker data по model_name. Якщо різні моделі мають однаковий короткий alias, або якщо model version/target/timeframe різні — вони зіллються.

Рекомендація:

- ключ має бути:
  - model_name;
  - model_version;
  - ticker;
  - target;
  - timeframe;
  - regime/context.
- або bridge має явно aggregate.

---

## 24.3.8. P1/P2: `ensemble_selector.py` як wrapper правильний, але треба не плутати active selector-и

Є вже:

```text
src.analytics.context.ensemble_selector
src.ensembling.*
src.integration.ensemble_selector
```

Wrapper нормальний для compatibility. Але README/архітектура мають пояснити:

- `analytics.context.ensemble_selector` — selector logic;
- `integration.ensemble_selector` — compatibility import;
- `ensembling` — signal composer.

---

# 24.4. Що працює добре

1. `DashboardDataBridge` вже виправляє проблему `load_data` vs `fetch_df`.
2. Є sample fallback і маркування `is_sample_data`.
3. DataManager import optional.
4. Dashboard cache TTL простий і зрозумілий.
5. `ensemble_selector.py` як compatibility wrapper мінімальний і чистий.
6. `EnsemblePerformanceBridge` має правильну ідею: синхронізувати live ensemble weights/performance з tracker/arena.

---

# 24.5. Що дати Codex по `integration`

```text
Deep-fix src/integration without changing public intent.

Tasks:

1. Wire DashboardDataBridge into dashboard:
   - dashboard/main_app.py should use DashboardDataBridge instead of direct DataManager queries;
   - show sample-data banner when is_sample_data=True.

2. Harden DashboardDataBridge queries:
   - add safe_query_df() that catches DB/pandas/DuckDB exceptions;
   - missing tables/columns should return sample fallback, not crash;
   - include error_type/sample_reason.

3. Define dashboard schema:
   - either create dashboard summary tables/views;
   - or implement fallback adapters from raw tables:
     market_data_raw, model_results, evaluation_summary, news tables.

4. Fix cache key:
   - replace Python hash() with stable sha256(json.dumps(kwargs, sort_keys=True, default=str)).

5. Parameterize date windows:
   - support days/start_date/end_date kwargs;
   - keep SQL parameterized.

6. Add protocols for EnsemblePerformanceBridge:
   - LiveEnsembleProtocol;
   - PerformanceTrackerProtocol;
   - validate required methods in __init__.

7. Improve merge keys:
   - merge performance by model_name + version + ticker + target + timeframe + regime if available;
   - avoid collapsing different contexts into one model row.

8. Document selector roles:
   - integration.ensemble_selector is compatibility wrapper only;
   - active logic lives in analytics.context.ensemble_selector.

9. Add tests:
   - bridge returns sample when DataManager unavailable;
   - bridge handles missing table exception;
   - cache key stable;
   - dashboard consumes bridge data;
   - ensemble bridge validates interfaces.
```

---

## 24.6. Priority list for `integration`

### P0 / must fix

- Немає P0 всередині `integration`, бо файли імпортуються.

### P1 / high priority

- DashboardDataBridge треба реально підключити до dashboard.
- DB query exceptions мають давати sample fallback.
- Dashboard schema/table mismatch.
- EnsemblePerformanceBridge needs interfaces/protocols.
- Merge key має враховувати context/version, не тільки model_name.

### P2 / cleanup

- stable cache key.
- date window kwargs.
- sample_reason.
- documentation for compatibility wrappers.

---

## 24.7. Summary for `integration`

`integration` — корисний bridge-шар. Найбільша користь прямо зараз: використати `DashboardDataBridge`, щоб полагодити dashboard без дублювання DB/schema логіки.

Але bridge треба зробити справді безпечним: якщо таблиця/колонка відсутня, UI має бачити sample/empty state, а не crash. Для ensemble bridge треба formal interface, бо зараз він працює на duck-typing і може впасти на AttributeError.

---

# 25. Audit: `integrations`

## 25.1. Загальний стан

`integrations` — це зовнішній gateway-шар для Cloud/CI/CD/інфраструктурних інтеграцій.

У наданій папці:

```text
integrations/
  README.md
  base.py
  data/
    bigquery_client.py
  infra/
    github_actions.py
```

Синтаксично всі Python-файли компілюються.

Імпорт:

```text
src.integrations.base                 OK
src.integrations.data.bigquery_client FAIL: No module named 'google'
src.integrations.infra.github_actions FAIL: No module named 'src.core.reporting'
```

Головний висновок: **папка частково legacy/production-aspirational. BigQuery integration має бути optional/lazy; GitHubActionsClient зараз не імпортується через старий шлях `src.core.reporting.results_manager` і містить багато mock/stub metrics.**

---

## 25.2. Залученість у проєкті

`BigQueryClient` реально використовується:

```text
src/data/management/connectors/bigquery_connector.py
```

Тобто `bigquery_client.py` може впливати на data collection/management.

`GitHubActionsClient` майже не згадується поза своїм файлом. Це більше devops utility/prototype.

`BaseIntegration` дублюється:

```text
src/core/base_integration.py
src/integrations/base.py
```

І це вже було знайдено в аудиті `core`.

---

# 25.3. Критичні проблеми

## 25.3.1. P0/P1: `BigQueryClient` не імпортується без google dependencies

На module import:

```python
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import bigquery
from pandas_gbq.gbq import GenericGBQException
```

Якщо `google-cloud-bigquery` або `pandas-gbq` не встановлені:

```text
ModuleNotFoundError: No module named 'google'
```

Навіть якщо користувач хоче `BIGQUERY_SIMULATOR_MODE=true`, до `__init__` він не дійде, бо import вже впав.

Це P1/P0 для optional integration.

Рекомендація:

- перенести google imports всередину real-mode branches;
- module import має працювати без GCP dependencies;
- якщо dependency missing:
  - `use_simulator=True`;
  - або structured unavailable status.

---

## 25.3.2. P1: `BigQueryClient` імпортує `BaseIntegration` з `src.core.base_integration`, а не `src.integrations.base`

У `bigquery_client.py`:

```python
from src.core.base_integration import BaseIntegration
```

А поруч є:

```text
src/integrations/base.py
```

Це дублювання. У `github_actions.py` використовується:

```python
from src.integrations.base import BaseIntegration
```

Тобто різні integrations наслідуються від різних `BaseIntegration`.

Наслідок:

- різна behavior;
- різний error handling;
- `core.base_integration.get_status()` теж re-raises ping errors;
- consistency немає.

Рекомендація:

- один canonical `BaseIntegration`, бажано `src.integrations.base`;
- або `src.core.base_integration` залишити compatibility wrapper;
- всі integrations мають наслідувати той самий base.

---

## 25.3.3. P1: `BaseIntegration.get_status()` re-raises exception замість повернути offline status

У `integrations/base.py`:

```python
try:
    is_alive = self.ping()
except (...) as e:
    error = str(e)
    raise
return {...}
```

Тобто якщо `ping()` падає, `get_status()` не повертає:

```python
{"status": "offline", "error": "..."}
```

а падає сам.

Цю проблему ми вже бачили в `core`.

Рекомендація:

```python
except Exception as e:
    error = str(e)
    is_alive = False
return {
  "status": "offline",
  "reachable": False,
  "error": error
}
```

---

## 25.3.4. P1: `BigQueryClient` `execute_query()` не валідовує query перед виконанням

Є метод:

```python
validate_query(query)
```

але `execute_query()` його не викликає.

Рекомендація:

- або caller має явно викликати validation;
- або `execute_query(validate=True)` by default;
- для production BigQuery потрібно dry run / cost guard перед real query.

---

## 25.3.5. P1: BigQuery cost formula неправильна або принаймні hardcoded/stale

```python
gb_processed = bytes_processed / 1024 ** 3
cost_usd = gb_processed * (6.25 / 1024)
```

Це фактично `$6.25 per TB`, converted to GB. Але pricing змінюється, залежить від region/pricing model/free tier. Hardcoded value небезпечне.

Рекомендація:

- винести `cost_per_tb_usd` у config;
- назвати estimate approximate;
- не використовувати як billing truth.

---

## 25.3.6. P1: simulator mock data використовує `datetime.now()` і random seed з full config

`_generate_gdelt_mock_data()`:

```python
config = get_current_config()
seed = config.get('performance.random_seed', 42)
random.seed(seed)
dates = [(datetime.now() - timedelta(days=i)).strftime('%Y%m%d') for i in range(30)]
```

`_generate_generic_mock_data()`:

```python
config = get_current_config()
seed = config.get('performance.random_seed', 42)
np.random.seed(seed)
pd.date_range(end=datetime.now(), ...)
```

Проблеми:

- full config side effects for simple simulator;
- dates change every run;
- global random seed mutation;
- mock data can look like real current data.

Рекомендація:

- accept `seed` and `reference_date` in constructor/config;
- use local RNG;
- mark mock data with `is_simulated=True`;
- do not call `get_current_config()` inside simulator helper.

---

## 25.3.7. P1: `GitHubActionsClient` не імпортується через `src.core.reporting.results_manager`

У `github_actions.py`:

```python
from src.core.reporting.results_manager import ResultsManager
```

А такого шляху в проекті немає. Є:

```text
src/analytics/reporting/results_manager.py
src/analytics/data_managers/model_results_manager.py
```

Наслідок:

```text
ModuleNotFoundError: No module named 'src.core.reporting'
```

Рекомендація:

- оновити import на актуальний ResultsManager;
- або передавати будь-який object із `save_results_to_output()` protocol;
- або зробити цей client legacy/devtools, а не production integration.

---

## 25.3.8. P1: `GitHubActionsClient` запускає subprocess команди в поточній директорії без sandbox/path config

Приклади:

```python
subprocess.run(["python", "-m", "pytest", "tests/", "--cov=.", ...], timeout=300)
subprocess.run(["python", "-m", "flake8", ".", "--format=json"], timeout=60)
```

Проблеми:

- запускається від поточного робочого каталогу, не project_root;
- може зачепити не той repo;
- може бути важко/довго;
- немає allowlist;
- може створювати/видаляти `test_results.json` у cwd;
- flake8 `--format=json` не є standard для flake8 без plugin.

Рекомендація:

- приймати `project_root`;
- всі команди запускати з `cwd=project_root`;
- output files у temp dir;
- timeout/configurable;
- tool availability check.

---

## 25.3.9. P1: `GitHubActionsClient` містить багато hardcoded/mock metrics

Методи повертають статичні значення:

```python
test_pipeline_performance -> status passed, execution_time 125.5
test_model_inference_speed -> hardcoded
test_database_performance -> hardcoded
test_memory_usage -> hardcoded
run_simple_tests -> 25 tests, 23 passed
check_licenses -> hardcoded
analyze_dependency_tree -> hardcoded
```

Це може бути корисним як prototype, але небезпечно для CI/CD:

- виглядає як реальний CI status;
- може створити false confidence;
- `overall_status=passed` може бути на mock data.

Рекомендація:

- додати `mode: "simulated" | "real"`;
- у simulated mode результат має явно мати `is_simulated=True`;
- production CI має не приймати simulated pass як real pass.

---

## 25.3.10. P1/P2: security scan дуже поверхневий

`scan_secrets`, `scan_vulnerabilities`, `scan_code_injection` треба перевірити глибше, але вже видно, що client більше схожий на демонстраційний quality report, ніж на реальний security scanner.

Рекомендація:

- інтегрувати конкретні tools:
  - ruff/flake8;
  - pytest;
  - bandit;
  - pip-audit;
  - detect-secrets/gitleaks;
  - mypy optional;
- явно зберігати raw tool outputs.

---

## 25.3.11. P2: README не відповідає фактичній структурі

README каже:

```text
bigquery/
ci_cd/
```

А фактично:

```text
data/bigquery_client.py
infra/github_actions.py
```

Рекомендація:

- оновити README під реальні шляхи;
- або перейменувати папки під README.

---

# 25.4. Що працює добре

1. `BaseIntegration` дає єдиний interface:
   - `name`
   - `ping`
   - `get_status`
2. BigQuery має simulator mode.
3. BigQuery має basic query validation і dry-run cost estimate.
4. BigQuery `execute_query()` повертає DataFrame.
5. `BigQueryConnector` у data management використовує centralized BigQueryClient.
6. GitHubActionsClient має правильну ідею як CI quality aggregator.
7. У GitHubActionsClient є timeouts на pytest/flake8.

---

# 25.5. Що дати Codex по `integrations`

```text
Deep-fix src/integrations without changing public intent.

Tasks:

1. Make BigQueryClient optional-import safe:
   - move google/cloud/pandas_gbq imports inside real-mode code;
   - module import should work without google dependencies;
   - if deps missing, return unavailable/simulator status.

2. Unify BaseIntegration:
   - choose one canonical BaseIntegration;
   - update BigQueryClient and GitHubActionsClient to inherit the same base;
   - core/base_integration can be compatibility wrapper.

3. Fix BaseIntegration.get_status:
   - do not re-raise ping errors;
   - return offline/reachable=false/error.

4. Add BigQuery query guards:
   - execute_query(validate=True) should validate before running;
   - optionally dry-run/cost estimate before real execution;
   - configurable max estimated GB/cost.

5. Fix BigQuery simulator:
   - do not call get_current_config() in mock generation;
   - use local RNG with seed;
   - support fixed reference_date;
   - mark DataFrame attrs or columns with is_simulated/source.

6. Fix BigQuery cost estimate:
   - move cost_per_tb_usd to config;
   - label as approximate estimate.

7. Fix GitHubActionsClient imports:
   - replace src.core.reporting.results_manager with actual results manager path;
   - or accept a ResultsSink protocol with save_results_to_output().

8. Harden subprocess execution:
   - accept project_root;
   - run subprocess with cwd=project_root;
   - use temp output dir for json reports;
   - tool availability checks;
   - configurable timeouts.

9. Separate real vs simulated CI:
   - all hardcoded/mock metrics must have is_simulated=True;
   - overall_status from simulated checks must not be treated as real CI pass.

10. Integrate real tools:
   - pytest;
   - ruff/flake8;
   - bandit;
   - pip-audit;
   - gitleaks/detect-secrets if available;
   - save raw outputs.

11. Update README:
   - actual structure is data/bigquery_client.py and infra/github_actions.py;
   - not bigquery/ and ci_cd/.
```

---

## 25.6. Priority list for `integrations`

### P0 / must fix

- `GitHubActionsClient` import path is broken.
- BigQueryClient cannot import without google dependencies, even in simulator mode.

### P1 / high priority

- Duplicate BaseIntegration.
- get_status re-raises instead of returning offline.
- BigQuery execute_query lacks validation/cost guard.
- BigQuery mock data uses full config and datetime.now.
- GitHubActionsClient uses cwd subprocess and many hardcoded simulated metrics.

### P2 / cleanup

- README path mismatch.
- cost formula config.
- raw tool outputs.
- protocol for results manager.

---

## 25.7. Summary for `integrations`

`integrations` — це зовнішній gateway, але зараз він ще не production-ready.

BigQuery-клієнт має правильну ідею й реально використовується через data connector, але має бути optional-import safe: якщо google libs нема, simulator/unavailable status має працювати без падіння на import.

GitHubActionsClient поки більше схожий на prototype/devtools CI summary: import broken, багато hardcoded метрик, subprocess запускається від cwd. Його не можна вважати реальним CI proof, поки не буде розділення real/simulated і реальні tool outputs.

---

# 26. Note: `feature_engineering.transformers`

Ти уточнив, що `transformers.py` і відповідні `__init__.py` були з папки `feature_engineering`. Це вже враховано в попередньому блоці:

```text
# 22. Audit: feature_engineering.transformers
```

Його залишаю як окремий маленький блок, а не змішую з великим `features`, бо це інша папка/compatibility namespace. Головні висновки там лишаються:

- scaler-и прості й корисні;
- `fit_transform()` має leakage-ризик, якщо викликати до train/test split;
- `transform()` до `fit()` мовчки no-op;
- треба додати fitted-state, strict columns, metadata/save-load.

---

# 27. Audit: `main`

## 27.1. Загальний стан

`main` — це intended entry/control layer: SystemOrchestrator + operational modes.

У наданому архіві:

```text
main/
  README.md
  MAIN_MODULE_ANALYSIS.md
  __init__.py
  system_orchestrator.py
  modes/
    __init__.py
    base.py
    train.py
    predict.py
    backtest.py
    intelligent.py
    monster_test.py
    training_data_pipeline.py
    web_ui.py
```

Синтаксично всі `.py` файли компілюються. Але імпорт у поточному дереві показав критичні runtime/import проблеми:

```text
src.main                              OK
src.main.system_orchestrator          FAIL: ModuleNotFoundError: No module named 'duckdb'
src.main.modes                        FAIL: ModuleNotFoundError: No module named 'src.core.config'
src.main.modes.base                   FAIL: ModuleNotFoundError: No module named 'src.core.config'
src.main.modes.train                  FAIL: ModuleNotFoundError: No module named 'src.core.config'
src.main.modes.predict                FAIL: ModuleNotFoundError: No module named 'src.core.config'
src.main.modes.backtest               FAIL: ModuleNotFoundError: No module named 'src.core.config'
src.main.modes.intelligent            FAIL: ModuleNotFoundError: No module named 'src.core.config'
src.main.modes.training_data_pipeline FAIL: ModuleNotFoundError: No module named 'src.core.config'
src.main.modes.web_ui                 FAIL: ModuleNotFoundError: No module named 'src.core.config'
src.main.modes.monster_test           FAIL: ModuleNotFoundError: No module named 'src.core.config'
```

Частина падінь є наслідком уже знайдених проблем в інших модулях:

- `duckdb` тягнеться через `DataManager`;
- `src.core.config` тягнеться через старий import у `advanced_engine.py`;
- `modes/__init__.py` eager-imports `BacktestMode`, а `BacktestMode` імпортує `advanced_engine`.

Головний висновок: **`main` зараз не є надійним entrypoint. Він компілюється, але фактично не може стабільно імпортуватися/запускатися через eager imports, abstract BaseMode contract mismatch, stale mode APIs і дублювання Web UI/dashboard paths.**

---

## 27.2. Залученість у проєкті

`main` виглядає як центральний entry layer, але пошук показує, що `SystemOrchestrator` майже не використовується зовні:

```text
src/main/system_orchestrator.py
src/main/modes/intelligent.py
src/main/MAIN_MODULE_ANALYSIS.md
```

Тобто зараз це радше potential/legacy control surface, а не явно активний production entrypoint.

Водночас `main` тягне дуже важкі залежності на import:

```python
DataManager
BacktestMode
PredictMode
TrainMode
HybridOrchestrator
get_dean_system
```

Тому навіть якщо `main` не використовується часто, будь-який імпорт може зламатися через downstream dependency.

---

# 27.3. Критичні проблеми

## 27.3.1. P0: `modes/__init__.py` eager-imports `BacktestMode`, що ламає весь package import

`modes/__init__.py`:

```python
from .backtest import BacktestMode
from .base import BaseMode
from .monster_test import MonsterTestMode
from .train import TrainMode
```

`BacktestMode` імпортує:

```python
from src.backtesting.advanced.advanced_engine import BiasDetector, WalkForwardOptimizer
```

А `advanced_engine.py` має старий import:

```python
from src.core.config.config_manager import get_current_config
```

Тому:

```python
import src.main.modes
```

падає.

Рекомендація:

- прибрати eager imports із `modes/__init__.py`;
- зробити lazy `__getattr__`;
- `BacktestMode` має імпортувати advanced backtesting components lazy всередині `run`;
- після фіксу `advanced_engine.py` прибрати старий `src.core.config` import.

---

## 27.3.2. P0: `BaseMode` має abstract `cleanup()`, але mode-класи його не реалізують

`BaseMode`:

```python
@abstractmethod
def cleanup(self) -> None:
    pass
```

`TrainMode`, `PredictMode`, `BacktestMode`, `MonsterTestMode`, `WebUIMode` не мають власного `cleanup()`.

Наслідок:

```text
TypeError: Can't instantiate abstract class TrainMode with abstract method cleanup
```

Тобто навіть якщо імпорт полагодити, `SystemOrchestrator._run_single_instance()`:

```python
instance = mode_class(self.config_manager)
```

може падати при створенні mode instance.

Рішення:

- або зробити `cleanup()` у BaseMode не abstract, default no-op;
- або додати `cleanup()` у всі modes.

Найпростіше:

```python
def cleanup(self) -> None:
    pass
```

і прибрати `@abstractmethod`.

---

## 27.3.3. P0/P1: `BacktestMode.run()` не приймає kwargs, але `SystemOrchestrator` завжди передає tickers/timeframes

`BacktestMode`:

```python
def run(self) -> dict[str, Any]:
```

`SystemOrchestrator._run_single_instance()`:

```python
result = instance.run(tickers=tickers or [], timeframes=timeframes or [], **kwargs)
```

Наслідок:

```text
TypeError: BacktestMode.run() got an unexpected keyword argument 'tickers'
```

Рекомендація:

- усі mode-и мають мати однаковий contract:

```python
def run(
    self,
    tickers: list[str] | None = None,
    timeframes: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
```

- `BaseMode.run` вже задає `**kwargs`, треба дотримуватись.

---

## 27.3.4. P0/P1: `SystemOrchestrator._run_monster_test()` запускає `TrainMode`, не `MonsterTestMode`

У `system_orchestrator.py`:

```python
async def _run_monster_test(...):
    ...
    config = ExecutionConfig(mode='monster_test', ...)
    return await self._dispatch(TrainMode, config, **kwargs)
```

Це очевидний bug. Він має dispatch-ити `MonsterTestMode`.

Наслідок:

- user просить monster_test;
- система фактично запускає training;
- stress-test logic не виконується.

Фікс:

```python
from src.main.modes.monster_test import MonsterTestMode
...
return await self._dispatch(MonsterTestMode, config, **kwargs)
```

Але спершу треба полагодити `MonsterTestMode` contract і dependencies.

---

## 27.3.5. P1: `SystemOrchestrator` imports heavy runtime dependencies на module import

На верхньому рівні:

```python
from src.data.management.data_manager import DataManager
from src.main.modes.backtest import BacktestMode
from src.main.modes.predict import PredictMode
from src.main.modes.train import TrainMode
from src.models.dean.dean_bootstrap_system import get_dean_system
from src.pipeline.hybrid_orchestrator import HybridOrchestrator
```

Це робить `SystemOrchestrator` дуже крихким:

- немає DuckDB → `system_orchestrator` не імпортується;
- advanced backtest broken → `system_orchestrator` не імпортується;
- DEAN dependency broken → `system_orchestrator` не імпортується;
- hybrid dependency broken → `system_orchestrator` не імпортується.

Рекомендація:

- lazy import per mode:
  - TrainMode тільки для train;
  - BacktestMode тільки для backtest;
  - HybridOrchestrator тільки для hybrid;
  - DataManager тільки для training_data_pipeline;
  - DEAN тільки для intelligent.
- `SystemOrchestrator` import має бути light and safe.

---

## 27.3.6. P1: `TrainMode` і `PredictMode` викликають `asyncio.run()` всередині async orchestrator context

`TrainMode.run()`:

```python
final_results = orchestrator.run(**initial_data)
if inspect.isawaitable(final_results):
    final_results = asyncio.run(final_results)
```

`PredictMode.run()` аналогічно.

А `SystemOrchestrator._run_single_instance()` викликається з async context і сам уміє await-ити результат:

```python
result = instance.run(...)
if inspect.isawaitable(result):
    return await result
```

Якщо `PipelineOrchestrator.run()` поверне coroutine, `TrainMode.run()` зробить `asyncio.run()` всередині вже запущеного event loop:

```text
RuntimeError: asyncio.run() cannot be called from a running event loop
```

Рекомендація:

- або modes мають бути async:

```python
async def run(...):
    result = orchestrator.run(...)
    if inspect.isawaitable(result):
        result = await result
```

- або `TrainMode.run()` не має запускати coroutine, а має повертати її наверх.
- один рівень має відповідати за await, не два.

---

## 27.3.7. P1: parallel execution через ProcessPoolExecutor майже напевно нестабільний

`SystemOrchestrator._run_parallel_execution()`:

```python
executor.submit(self._run_single_instance_sync, mode_class, [ticker], ...)
```

Проблеми:

- передається bound method `self._run_single_instance_sync`, треба pickle-ити `self`;
- `self.config_manager` може бути непіклябельним;
- logger/resources можуть бути непіклябельними;
- child process заново імпортує весь stack;
- `_run_single_instance_sync()` якщо result awaitable, просто повертає None:

```python
if inspect.isawaitable(result):
    return None
```

- результати ticker execution не агрегуються, тільки tickers_processed.

Рекомендація:

- для parallel tickers використовувати top-level worker function;
- передавати serializable config snapshot;
- або поки default `parallel=False`;
- якщо result awaitable — у sync worker треба створювати event loop і виконувати coroutine;
- збирати per-ticker results/errors.

---

## 27.3.8. P1: `_run_web_ui()` запускає Streamlit dashboard, але існує окремий `WebUIMode`

`SystemOrchestrator._run_web_ui()`:

```python
subprocess.run(['streamlit', 'run', dashboard_path], check=True)
```

А в `main/modes/web_ui.py` є цілий `WebUIMode` із власним HTTP server.

Це два різні UI paths:

1. Streamlit dashboard:
   ```text
   src/dashboard/main_app.py
   ```
2. custom HTTP server:
   ```text
   src/main/modes/web_ui.py
   ```

Наслідок:

- неясно, який UI production;
- `WebUIMode` може бути legacy;
- docs `main/README.md` каже `web_ui.py` entrypoint, але orchestrator запускає Streamlit dashboard.

Рекомендація:

- вибрати canonical UI:
  - краще Streamlit dashboard + `DashboardDataBridge`;
- `WebUIMode` позначити legacy або перенести в devtools;
- `SystemOrchestrator` має не тримати два UI contracts.

---

## 27.3.9. P1: `_run_web_ui()` ловить не ті exceptions

`subprocess.run(..., check=True)` може кинути:

- `FileNotFoundError` якщо streamlit не встановлений;
- `subprocess.CalledProcessError`;
- `OSError`.

Код ловить:

```text
ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError
```

Тобто реальний startup failure не буде перетворений у structured result.

Рекомендація:

```python
except (FileNotFoundError, subprocess.CalledProcessError, OSError) as e:
    return {"status": "error", "error_type": type(e).__name__, ...}
```

---

## 27.3.10. P1: `IntelligentMode` має окремий несумісний contract

`modes/intelligent.py`:

```python
class IntelligentMode:
    def __init__(self, args):
        self.orchestrator = SystemOrchestrator()
```

Він:

- не наслідує `BaseMode`;
- очікує `args`, а не `config_manager`;
- створює новий `SystemOrchestrator` всередині;
- запускає `mode='intelligent'`, який у `SystemOrchestrator` знову йде в `_run_intelligent_mode`.

Це потенційна рекурсія/дублювання control flow.

При цьому `SystemOrchestrator` для intelligent mode не використовує `IntelligentMode`, а напряму:

```python
dean_brain = get_dean_system()
mode_type = PredictMode or TrainMode
return await self._dispatch(mode_type, ...)
```

Рекомендація:

- або видалити/legacy `IntelligentMode`;
- або привести до BaseMode contract;
- один intelligent flow, не два.

---

## 27.3.11. P1: `MAIN_MODULE_ANALYSIS.md` явно stale/misleading

Файл називає багато компонентів `Production Ready`, але фактичний імпорт показує:

- package import падає;
- BaseMode abstract cleanup не реалізовано;
- monster_test dispatch неправильний;
- BacktestMode run signature mismatch;
- web UI дублюється.

Документ уже сам містить частину issues, наприклад:

```text
MonsterTestMode: Method signature mismatch
WebUIMode: Missing return statement
IntelligentMode: Missing UnifiedTradingSystem import
```

Але при цьому багато компонентів позначені як `Production Ready`.

Рекомендація:

- позначити `MAIN_MODULE_ANALYSIS.md` як historical/legacy;
- або оновити після фактичного smoke/import tests;
- не використовувати як source of truth.

---

## 27.3.12. P1/P2: `TrainingDataPipeline` тягне broken FeatureOrchestrator

`training_data_pipeline.py`:

```python
from src.features.feature_orchestrator import FeatureOrchestrator
```

А `FeatureOrchestrator` зараз має indentation error.

Отже training_data_pipeline mode не працюватиме до P0 fix у `features`.

Також:

```python
targets_list = config_manager.get_config('targets', [])
target_orchestrator = TargetOrchestrator(targets_list=targets_list)
```

Треба перевірити, чи `targets` config дійсно list, бо в попередньому аудиті config мав nested YAML structure.

`to_parquet()` ловить тільки вузькі exceptions, але може кинути `ImportError`/`OSError`/pyarrow errors.

---

## 27.3.13. P1/P2: `BacktestMode` залежить від уже broken `advanced_engine.py` і має API mismatch

`BacktestMode` імпортує:

```python
BiasDetector, WalkForwardOptimizer
```

із `advanced_engine.py`, де раніше вже знайдено:

- wrong import `src.core.config.config_manager`;
- shadowed classes;
- WalkForward API mismatch.

`BacktestMode._run_walk_forward_validation()` викликає:

```python
walk_forward.walk_forward_optimization(
    data=historical_data,
    optimization_func=optimization_function,
    in_sample_months=in_sample_months,
    out_sample_months=out_sample_months,
)
```

Це той самий старий API, який може не відповідати actual `src.algorithms.walk_forward_optimizer`.

Рекомендація:

- спочатку стабілізувати `advanced_engine.py`;
- або BacktestMode має використовувати canonical `src.algorithms.walk_forward_optimizer` API.

---

## 27.3.14. P1/P2: `BacktestMode._detect_biases()` re-raises після warning

```python
except (...) as e:
    warning...
    bias_results['warnings'].append(...)
    raise
```

Bias detection не має валити весь backtest, якщо це diagnostic step.

Рекомендація:

- повернути `bias_results` із warning;
- fatal тільки якщо `strict_bias_checks=True`.

---

## 27.3.15. P1/P2: `WebUIMode` повертає mock/simulation portfolio/market data без явного hard gate

`get_portfolio_status()` повертає hardcoded:

```python
total_value: 125000
positions: TSLA, NVDA, AAPL
```

`get_market_data()` генерує random simulated prices і ставить:

```python
source: simulation
```

Це добре, що source є, але UI має явно показувати simulation banner. І цей mode не має виглядати як live trading UI.

Також `random.seed(seed)` викликається на кожен request, тому market data повторюється однаково.

Рекомендація:

- позначити `WebUIMode` as demo/legacy;
- якщо залишити, додати `simulation_mode=True` banner;
- не використовувати для live/paper trading без real data bridge.

---

## 27.3.16. P2: `WebUIMode.serve_file()` рахує Content-Length по символах, а пише bytes

```python
self.send_header('Content-Length', str(len(content)))
self.wfile.write(content.encode('utf-8'))
```

Для Unicode `len(content)` може не дорівнювати кількості bytes.

Рекомендація:

```python
payload = content.encode("utf-8")
send_header("Content-Length", str(len(payload)))
wfile.write(payload)
```

---

## 27.3.17. P2: `modes/__init__.py` не експортує всі modes

`__all__`:

```python
BaseMode
TrainMode
MonsterTestMode
BacktestMode
```

Не експортує:

```text
PredictMode
WebUIMode
IntelligentMode
```

Але вони існують.

Рекомендація:

- або lazy export all active modes;
- або позначити non-exported modes as legacy.

---

# 27.4. Що працює добре

1. `main` має правильну ідею: один orchestrator + modes.
2. `ExecutionConfig` зменшує кількість параметрів.
3. Є mode separation:
   - train;
   - predict;
   - backtest;
   - hybrid;
   - training_data_pipeline;
   - dashboard;
   - intelligent;
   - monster_test.
4. `TrainMode`/`PredictMode` мають простий flow.
5. `SystemOrchestrator` уже має unknown mode handler.
6. `WebUIMode` має API endpoints і HTML templates, хоч і legacy/demo.
7. `MAIN_MODULE_ANALYSIS.md` містить корисні підказки, але потребує оновлення.

---

# 27.5. Що дати Codex по `main`

```text
Deep-fix src/main without changing public intent.

Tasks:

1. Make imports light and stable:
   - SystemOrchestrator should not import DataManager, BacktestMode, HybridOrchestrator, DEAN, etc. at module import;
   - lazy import per mode inside methods;
   - modes/__init__.py should not eager-import BacktestMode.

2. Fix BaseMode contract:
   - make cleanup() a default no-op, not abstract;
   - or implement cleanup() in every mode.

3. Standardize mode run signatures:
   - every mode should support:
     run(tickers=None, timeframes=None, **kwargs) -> dict
   - BacktestMode currently lacks kwargs.

4. Fix monster_test dispatch:
   - _run_monster_test must dispatch MonsterTestMode, not TrainMode.

5. Fix async ownership:
   - remove asyncio.run() from TrainMode/PredictMode when called under SystemOrchestrator;
   - make modes async-aware or let SystemOrchestrator await returned coroutine;
   - avoid asyncio.run inside running event loop.

6. Fix parallel execution:
   - avoid submitting bound self method to ProcessPoolExecutor;
   - use top-level worker;
   - pass serializable config snapshot;
   - execute awaitables inside worker event loop;
   - aggregate per-ticker results, not only tickers_processed.

7. Clarify UI path:
   - choose Streamlit dashboard as canonical or WebUIMode as canonical;
   - likely use Streamlit dashboard + DashboardDataBridge;
   - mark WebUIMode as demo/legacy if not production.

8. Fix _run_web_ui exceptions:
   - catch FileNotFoundError, subprocess.CalledProcessError, OSError;
   - return structured failure if streamlit unavailable.

9. Fix IntelligentMode:
   - either remove legacy args-based IntelligentMode;
   - or adapt it to BaseMode;
   - avoid recursive SystemOrchestrator -> IntelligentMode -> SystemOrchestrator flow.

10. Stabilize BacktestMode:
   - lazy import advanced backtesting components;
   - update WalkForwardOptimizer API;
   - bias detection should warn, not fatal, unless strict mode enabled.

11. Stabilize TrainingDataPipeline:
   - wait for FeatureOrchestrator syntax fix;
   - validate targets config shape;
   - catch parquet engine/OSError errors;
   - return structured result.

12. WebUIMode cleanup:
   - mark hardcoded portfolio/market data as demo/simulated;
   - fix Content-Length bytes;
   - avoid re-raising after send_error;
   - or move to devtools/demo.

13. Update documentation:
   - MAIN_MODULE_ANALYSIS.md is stale;
   - mark as historical or regenerate after import/smoke tests.
```

---

## 27.6. Priority list for `main`

### P0 / must fix

- `modes/__init__.py` eager import chain breaks package import.
- `BaseMode.cleanup` abstract but modes do not implement it.
- `BacktestMode.run()` signature mismatch with SystemOrchestrator.
- `_run_monster_test()` dispatches `TrainMode`, not `MonsterTestMode`.

### P1 / high priority

- SystemOrchestrator has heavy imports at module import.
- Train/Predict use `asyncio.run()` inside possible running event loop.
- ProcessPoolExecutor design likely not pickle/async safe.
- Web UI has two competing implementations.
- `_run_web_ui` catches wrong exceptions.
- IntelligentMode has incompatible contract/possible recursion.
- TrainingDataPipeline depends on broken FeatureOrchestrator.

### P2 / cleanup

- `MAIN_MODULE_ANALYSIS.md` stale.
- `modes/__init__.py` incomplete exports.
- WebUIMode mock data banner.
- Content-Length bytes.
- Better structured results per mode.

---

## 27.7. Summary for `main`

`main` має правильну ідею: бути control center. Але зараз це не production-ready entrypoint.

Найгірше:

1. package imports падають через eager imports;
2. mode classes не можна інстанціювати через abstract `cleanup`;
3. `BacktestMode.run()` не сумісний із `SystemOrchestrator`;
4. `monster_test` запускає TrainMode;
5. async execution ownership розмазаний між modes і orchestrator;
6. parallel execution через process pool небезпечний;
7. UI path дублюється;
8. docs кажуть “production ready”, але smoke tests показують протилежне.

Після фіксів `main` може стати нормальним control plane, але зараз краще не запускати production pipeline через нього, доки не буде import/smoke test suite для всіх modes.

---

# 44. Audit: `dean_os (2)`

## 44.1. Загальний стан

`dean_os (2)` — це окремий DEAN-OS control/review/autonomy layer: агенти, review gates, paper trading store, replay/evidence workflows, calibration proposals, operation queue, pipeline adapter/control surface.

У наданому архіві:

```text
Agents_architecture.md
dean_os/
  COMMAND_CHECKLIST.md
  IMPLEMENTATION_STATUS.md
  NEXT_CHAT_HANDOFF.md
  config/
    agent_registry.yaml
    horizon_policy.yaml
    logging_policy.yaml
    paper_trading.yaml
  agents/
  *.py
```

Стан:

```text
files: 96
python files: 88
compile failures: 0
sample import failures: 0
YAML configs parse OK
```

Порівняння з попередніми `dean_os.zip` / `dean_os(1).zip`:

```text
new archive has +38 extra Python files
diffs vs previous: __init__.py, agents/tuning.py, outcome_evaluation.py
missing from new vs previous: 0
```

Тобто це не дубль: `dean_os (2)` — розширена версія з review/calibration/replay/control-surface workflows.

Головний висновок: **DEAN-OS технічно сильно покращився: компілюється, імпортується, має paper/review-first дизайн і багато явних human approval/dry-run barriers. Але це все ще має бути “advisor/review/paper-only control layer”, не execution authority. Основні P1: default ExecutionGateway може ставити `paper_traded` до human approval, pipeline_audit hard gate може silent-skip при missing required_inputs, AgentRegistry динамічно імпортує class_path без allowlist, багато stores/reports пишуть у relative paths без safe root/atomic writes, і replay/learning/calibration gates треба не змішувати з production/paper execution без immutable manifest.**

---

## 44.2. Залученість у проєкті

DEAN-OS — не просто одна папка, а окремий control-plane поверх основного pipeline.

Активні/важливі ролі:

```text
Pipeline gates:
  pipeline_audit
  data_quality
  risk
  model_performance
  regime

Review/autonomy:
  chief_review
  review_action_dry_run
  review_action_apply_ceremony
  review_approved_learning_loop
  operation_queue
  calibration_review_lifecycle

Paper/replay:
  paper_trading
  paper_portfolio
  paper_autonomy
  historical_replay
  historical_replay_batch
  historical_research_replay
  replay_calibration_readiness_gate

Pipeline bridge:
  pipeline_adapter
  pipeline_control_surface

Memory/evidence:
  recommendation_memory
  event_log
  decision_logger
  evidence_timestamp_audit
  evidence_gap_resolution_plan
```

У `agent_registry.yaml` за замовчуванням enabled:

```text
pipeline_audit        hard, enabled, block
data_quality          hard, enabled, block
risk                  hard, enabled, block
macro_policy          analytical, enabled
geopolitical          analytical, enabled
news_catalyst         analytical, enabled
sector_cycle          analytical, enabled
```

Smoke test із чистим project root показав:

```text
loaded pipeline agents: data_quality, risk
```

`pipeline_audit` не завантажився, бо його `required_inputs` були відсутні. Це важлива деталь: hard gate with missing required files currently може бути skipped at registry load, а не returned as blocked.

---

# 44.3. Що працює добре

1. Усі 88 Python files компілюються.
2. Sample imports пройшли без помилок:
   - `dean_os.orchestrator`
   - `dean_os.pipeline_adapter`
   - `dean_os.pipeline_control_surface`
   - `dean_os.execution_gateway`
   - `dean_os.paper_trading`
   - `dean_os.paper_portfolio`
   - core agents.
3. Config YAML parse OK.
4. `HybridPipelineAdapter` imports heavy project pipeline lazily. Це правильно.
5. `ExecutionPolicy` default:
   ```yaml
   live_execution_enabled: false
   paper_trading_enabled: true
   require_human_approval: true
   ```
   Це правильний safety direction.
6. `OperationQueue.dry_run()` preview-only і каже, що нічого не виконано.
7. `ReviewActionDryRun` explicitly sets:
   ```text
   learning_write_performed: false
   proposal_enqueue_performed: false
   config_write_performed: false
   pipeline_run_performed: false
   broker_access_performed: false
   ```
8. `ReviewActionApplyCeremony` потребує explicit `--apply-review-action`.
9. `ChiefReviewAgent` прямо каже, що не approve-ить trades/config, а тільки summarizes evidence.
10. `PaperPortfolioAgent` прямо маркує results as simulated diagnostics only.
11. `PipelineControlSurface` — good concept: not tuner, but variation permission surface.
12. `ReplayCalibrationReadinessGate` явно рекомендує review packet, not automatic learning/config/live.
13. `RiskAgent`, `DataQualityAgent`, `PipelineAuditAgent` мають hard-veto semantics in registry.
14. `paper_trading` / `recommendation_memory` stores use parameterized SQLite queries.
15. Нові replay/focused/ticker-specific tools корисні для evidence-driven calibration, якщо лишаються offline/review-only.

---

# 44.4. Критичні проблеми

## 44.4.1. P1: `ExecutionGateway` returns `paper_traded` before human approval check

`execution_gateway.py`:

```python
if not self.policy.live_execution_enabled and self.policy.paper_trading_enabled:
    return ExecutionOutcome(status="paper_traded", ...)
if self.policy.require_human_approval or decision.requires_human_approval:
    return ExecutionOutcome(status="queued_for_review", ...)
```

Smoke test:

```text
decision.requires_human_approval = True
ExecutionGateway().process(decision) -> status='paper_traded'
```

Можливо, це intended, якщо `paper_traded` означає simulation-only. Але config назва `require_human_approval` тоді misleading.

Рекомендація:

- або approval check before paper trade;
- або rename status to `paper_trade_preview` / `paper_trade_candidate`;
- додати policy split:
  ```yaml
  require_human_approval_for_paper: true/false
  require_human_approval_for_live: true
  ```

---

## 44.4.2. P1: `ExecutionGateway` can return `executed` without execution adapter

Якщо:

```python
live_execution_enabled=True
paper_trading_enabled=False
require_human_approval=False
decision.requires_human_approval=False
```

то gateway повертає:

```text
status='executed'
```

Без broker adapter / paper store / operation queue.

Рекомендація:

- default live path should return `blocked_no_execution_adapter`;
- only adapter can return `executed`;
- gateway should never claim execution by itself.

---

## 44.4.3. P1: `AgentRegistry` dynamic imports `class_path` without allowlist

`registry.py`:

```python
module_name, class_name = cfg["class_path"].split(":", maxsplit=1)
module = importlib.import_module(module_name)
agent_cls = getattr(module, class_name)
```

Config local, але це все одно code execution surface.

Рекомендація:

- allowlist prefixes:
  ```text
  dean_os.agents.
  dean_os.
  ```
- validate class subclass `BaseAgent`;
- reject malformed class_path with clear error.

---

## 44.4.4. P1: hard gate agents with missing required_inputs can be silently skipped

`BaseAgent.check_prerequisites()` returns False if required files missing.

`AgentRegistry.load_all()` only appends if prerequisites pass. Якщо hard-veto `pipeline_audit` required files are missing, it is not loaded and does not produce a blocked report.

Рекомендація:

- for hard/block agents, missing required_inputs should create synthetic blocked report;
- registry should not silently skip hard gates;
- at least return `unavailable_hard_gate` report.

---

## 44.4.5. P1: hard agents return `caution` on missing core inputs, not blocked

Smoke test із пустим контекстом:

```text
data_quality -> caution: No DataFrame inputs supplied
risk -> caution: No returns or positions supplied
```

Для pre-pipeline це може бути OK. Для pre-trade це має бути fail-closed.

Рекомендація:

- add context phase:
  ```text
  phase=pre_pipeline / post_pipeline / pre_trade
  ```
- missing data/risk in pre_trade phase must block.

---

## 44.4.6. P1: `HybridPipelineAdapter._extract_returns()` may use target labels as returns

```python
for name in ("return", "returns", "target_return_1d", "close_return", "pct_change"):
    if name in columns:
        return frame[columns[name]]
```

`target_return_1d` — supervised label, not live realized return. For offline review it can be acceptable, but must be marked.

Рекомендація:

- prefer realized return columns only;
- if using `target_return_*`, mark `returns_source="target_label"` and `offline_only=True`;
- never use target labels for live/paper execution gating.

---

## 44.4.7. P1: context enrichment catches broad exceptions and silently degrades

Examples:

```python
_records_from_dataframe -> except Exception: return []
_extract_returns -> except Exception: return None
_safe_len -> except Exception: return 0
```

Рекомендація:

- accumulate `context_enrichment_warnings`;
- pass warnings into `context.metadata`;
- hard gates react to missing critical context.

---

## 44.4.8. P1: many report/store writers use relative paths and non-atomic writes

Examples:

```text
reports/dean_os/...
data/dean_os/*.sqlite
json_path.write_text(...)
latest_json.write_text(...)
```

Risks:

- cwd-dependent outputs;
- partial writes;
- latest overwrite race;
- accidental writes in wrong project root.

Рекомендація:

- central `DeanPaths(project_root, reports_root, data_root)`;
- safe path resolver;
- atomic temp write + replace;
- run_id scoped directories.

---

## 44.4.9. P1: `OperationQueue.approve()` changes status but does not require reviewer identity/reason

```python
def approve(self, proposal_id):
    return self.set_status(proposal_id, "approved")
```

Рекомендація:

- `approve(proposal_id, reviewer, reason, evidence_ref)`;
- same for reject/expire/execute;
- immutable audit event.

---

## 44.4.10. P1: `OperationQueue` has status `"executed"` but no execution receipt

Queue can set status to executed, but actual execution is not tied to adapter/evidence.

Рекомендація:

- require execution receipt/reference for executed;
- split:
  - approved;
  - manually_executed;
  - execution_confirmed.

---

## 44.4.11. P1: `ReviewActionApplyCeremony` may write review actions without cross-store transaction

It writes action through `ReviewActionStore`; if event log/operation log is also used, consistency may not be atomic.

Рекомендація:

- one transactional store or outbox event pattern;
- return action_id + event_id.

---

## 44.4.12. P1: stores use `INSERT OR REPLACE`

`PaperTradeStore`, `RecommendationMemoryStore`, `OperationQueue` use replacement style writes.

Risk:

- historical payload overwritten for same id;
- transition history lost.

Рекомендація:

- immutable event log + current-state view;
- explicit transitions;
- version/revision.

---

## 44.4.13. P1: `PaperPortfolioSimulator` may not enforce aggregate portfolio cash/exposure across overlapping records

Interface suggests per-record sizing from `initial_cash * position_size_pct`, then aggregate equity curve. Need ensure overlapping records do not over-allocate.

Рекомендація:

- enforce cash/exposure over time;
- position sizing based on current equity at entry;
- max gross/net exposure;
- skip/reduce positions if exposure cap breached.

---

## 44.4.14. P1: `PaperPortfolioSimulator` imports private outcome helpers

```python
from dean_os.outcome_evaluation import _frame_latest_datetime, _parse_datetime, ...
```

Private helper imports make APIs fragile.

Рекомендація:

- expose public `MarketDataFrameLoader` / `OutcomePriceService`;
- avoid importing `_private` functions across modules.

---

## 44.4.15. P1: evidence/replay tools read arbitrary provided paths without central trusted root

Many tools accept paths and do:

```python
Path(path).read_text(...)
json.loads(path.read_text(...))
pd.read_csv(path)
```

Recommendation:

- `DeanPaths.resolve_input_artifact(path, allowed_roots=...)`;
- structured input error if path invalid/missing.

---

## 44.4.16. P1: learning/calibration workflows need central approval invariant

Many files already say review/dry-run/no automatic learning. Good. But enforce with one object:

```text
ApprovalReceipt
  reviewer
  action_id
  evidence_refs
  gate_status
  timestamp
```

No learning memory write, config write, paper trade, or operation execution unless an `ApprovalReceipt` exists.

---

## 44.4.17. P1/P2: `PipelineControlSurface` thresholds are fixed defaults

Defaults such as:

```python
min_sharpe=0.0
max_drawdown=0.25
min_clear_replay_hit_rate=0.55
```

Need config profiles, uncertainty bounds, and sample size interpretation.

---

## 44.4.18. P1/P2: `RiskAgent` simple snapshot lacks liquidity/execution risk

The agent transparently says broker-side liquidity/execution impact is a blind spot. If used as hard gate, add:

- liquidity;
- slippage;
- turnover;
- concentration;
- correlation cluster;
- pending orders.

---

## 44.4.19. P2: wall-clock `datetime.now()` in replay/review tools

For replay/backtesting/review reproducibility, pass `as_of`.

---

# 44.5. Дублювання / canonical decisions

## 44.5.1. Execution authority

Canonical boundary:

```text
DEAN-OS may propose / review / paper-simulate.
DEAN-OS must not be broker execution authority.
Only ExecutionAdapter can execute, and only with ApprovalReceipt + RiskDecision.
```

## 44.5.2. Evidence and path roots

Need one canonical path service:

```text
data/dean_os
reports/dean_os
audit_reports
market_data inputs
```

## 44.5.3. Learning promotion

Canonical lifecycle:

```text
replay/evidence -> review packet -> dry-run action -> human action -> apply ceremony -> learning memory update
```

No direct shortcuts.

---

# 44.6. Що дати Codex

```text
Deep-fix dean_os (2) without changing public intent.

1. ExecutionGateway:
   - check human approval before paper_traded if require_human_approval_for_paper is true;
   - rename paper_traded to paper_trade_preview unless a PaperTradeStore record is written;
   - never return executed without injected execution adapter and receipt.

2. Add policy split:
   - require_human_approval_for_paper;
   - require_human_approval_for_live;
   - live_execution_enabled remains false by default.

3. AgentRegistry:
   - add allowlist for class_path prefixes;
   - validate class is BaseAgent;
   - malformed class_path returns blocked config error.

4. Hard gate prerequisites:
   - missing required_inputs for hard/block agents should produce blocked report, not silently skip.

5. Context phase:
   - distinguish pre_pipeline / post_pipeline / pre_trade.
   - missing data/risk in pre_trade phase must block.

6. HybridPipelineAdapter:
   - do not use target_return_* as live returns unless offline_only flag is set;
   - attach context_enrichment_warnings to context.metadata;
   - mark returns_source.

7. Path safety and writes:
   - central DeanPaths;
   - trusted input/output roots;
   - atomic JSON/markdown writes;
   - run_id-scoped outputs;
   - safe sqlite paths.

8. OperationQueue:
   - approve/reject require reviewer, reason, evidence_ref;
   - executed status requires execution receipt;
   - no INSERT OR REPLACE without audit versioning.

9. ReviewActionApplyCeremony:
   - transactional write of review action + event log if both are used;
   - return action_id/event_id.

10. Stores:
   - PaperTradeStore, RecommendationMemoryStore, OperationQueue should be append-auditable;
   - current state can be view, but preserve transitions.

11. PaperPortfolioSimulator:
   - enforce aggregate portfolio cash/exposure over time;
   - no independent initial_cash per overlapping record;
   - skip/reduce positions if exposure cap breached.

12. Public APIs:
   - replace imports of private outcome_evaluation helpers with public service.

13. ApprovalReceipt:
   - central object required for learning/config/paper-write/apply ceremonies.
   - tests for all blocked paths.

14. PipelineControlSurface:
   - config profiles for thresholds;
   - sample-size/CI-aware replay hit rate;
   - keep conservative default.

15. RiskAgent:
   - add liquidity/slippage/concentration/correlation/pending-order blind spots or metrics.

16. Clock:
   - pass as_of through replay/calibration/review.
   - wall clock only for artifact created_at.

17. Docs:
   - authority map: advisory vs hard gate vs paper simulator vs review store vs live-execution forbidden.

18. Tests:
   - default gateway with require_human_approval does not mark paper_traded without policy allowing it;
   - live execution cannot return executed without adapter;
   - missing pipeline_audit required files yields blocked report;
   - class_path outside dean_os rejected;
   - target_return_* returns source marked offline_only;
   - OperationQueue approval records reviewer/reason;
   - PaperPortfolio respects aggregate exposure.
```

---

## 44.7. Priority list

### P0 / must fix

- No syntax/import P0: all compile, sample imports pass.

### P1 / high priority

- ExecutionGateway paper path bypasses human approval check.
- ExecutionGateway can return `executed` without adapter if live flags are opened.
- AgentRegistry lacks class_path allowlist/subclass validation.
- Hard gate missing required_inputs can be silently skipped.
- Missing data/risk hard agents return caution; should block in pre_trade phase.
- HybridPipelineAdapter may use `target_return_*` as returns source without offline marking.
- Relative/non-atomic report/store writes.
- OperationQueue approve/reject lacks reviewer/reason/evidence_ref.
- Queue/store `INSERT OR REPLACE` loses transition history.
- PaperPortfolioSimulator may not enforce aggregate portfolio cash/exposure across records.
- Replay/learning/calibration apply paths need central ApprovalReceipt invariant.
- Path input safety for arbitrary report/data paths.

### P2 / cleanup

- Replace private helper imports with public services.
- `datetime.now()` clock injection for replay.
- Config profiles for control-surface thresholds.
- Authority/advisory docs.
- Store file locks/concurrency.
- More unit tests for blocked paths.

---

## 44.8. Summary

`dean_os (2)` is a strong step forward. It compiles, imports, has explicit dry-run/review patterns, and is clearly moving toward a safe “advisor + paper simulation + review gate” architecture.

The main thing to protect now is **authority boundaries**:

1. DEAN can propose.
2. DEAN can review.
3. DEAN can simulate paper outcomes.
4. DEAN can write review records only after explicit apply.
5. DEAN must not claim real execution without an execution adapter and approval receipt.

Fix `ExecutionGateway`, hard-gate missing prerequisites, path/write safety, and append-only audit stores. After that, DEAN-OS can be a reliable control plane over the trading pipeline, not a risky autonomous executor.
