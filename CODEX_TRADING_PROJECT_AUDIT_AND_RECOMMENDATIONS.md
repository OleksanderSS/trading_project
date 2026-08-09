# Codex: аудит та повний перелік рекомендацій для `trading_project`

**Дата аудиту:** 2026-07-28  
**Призначення документа:** зібрати в одному місці конкретні знахідки аудиту, пріоритетні виправлення та потенційно корисні напрями розвитку проєкту.  
**Важливо:** це не список для одночасного впровадження. Кожен пункт потрібно спочатку класифікувати як `already_exists`, `partial`, `missing`, `obsolete`, `archive` або `not_needed`.

---

## 1. Головний висновок

Проєкт уже є сильною дослідницькою платформою, але поки не є надійною торговою системою.

Основний потенціал зростання зараз міститься не в додаванні нових індикаторів або складніших моделей, а у створенні послідовного **шару правди**:

```text
валідні point-in-time дані
→ правильна семантика targets
→ чесні train/validation/test
→ єдині фінансові метрики
→ реалістична симуляція витрат
→ model promotion gate
→ prospective shadow outcomes
→ лише потім paper/live-like рішення
```

Проєкту потрібна менша кількість активних шляхів, більше доказів і чітка межа між Research Lab та Stable Core.

---

## 2. Конкретні знахідки аудиту

### 2.1. Остання збережена база не може бути еталонною

В останньому `features.parquet`:

- 1 601 рядок;
- 1 030 колонок;
- 225 дубльованих ключів `datetime + ticker + interval`;
- 450 рядків входять до дубльованих пар;
- календарні ознаки для `60m` збігаються з реальним timestamp лише приблизно у 11–27% випадків;
- 181 числова ознака константна;
- 209 числових ознак майже константні;
- 337 ознак мають понад 50% пропусків;
- metadata заявляє `AAPL` і лише `1d`, хоча фактичний файл містить також 1 103 рядки `60m`;
- `batch_metadata.json` має порожній `stages_completed`, хоча інший metadata-файл заявляє завершені Stage 0–3;
- raw/cleaned файли з розширенням `.parquet` у частині маршруту фактично записуються як pickle-словники.

Остання база була створена до низки feature-leakage виправлень від 2026-07-26. Моделі, навчені на старих даних, не стають валідними лише через виправлення коду.

**Рекомендація:** позначити відповідні batches і всі похідні моделі як `quarantined` та перевипустити Stage 1–4 після виправлення data identity.

### 2.2. Справжній holdout/test фактично не використовується

Stage 4 створює `train`, `validation` і `test`, але у звичайний training manager передає validation під назвою `X_test`. Реальний підготовлений test залишається невикористаним.

Після вибору переможця trainer повторно оцінює його на тому самому validation-наборі та безумовно створює файл `CHAMP_*`.

У збережених Stage 4 artifacts:

- 831 model candidates;
- медіана `train_sample_count = 186`;
- медіана `validation_sample_count = 18`;
- `test_sample_count = 0` у всіх 831;
- усі 831 мають статус `partial_model_evaluation_candidate`;
- усі 831 не мають `max_drawdown` та `train_score`;
- feature-importance count дорівнює нулю у всіх 831 feature artifacts.

Водночас prediction path може споживати моделі, названі champions.

**Рекомендація:** файл або alias `CHAMP` може створювати лише окремий promotion service після завершення locked test, walk-forward, baseline, cost і stability gates.

### 2.3. Семантика частини targets не відповідає їх назвам

Приклади:

- `target_daily_trend_strength_1d`;
- `target_daily_momentum_score_1d`;
- `target_return_1d`.

Перші два targets у поточній базі повністю ідентичні, тому що regression calculator ігнорує `method` і `window` та завжди рахує майбутню відсоткову зміну базової колонки.

Також:

- `target_hourly_volume_spike_1h` ігнорує `compare_to: average`;
- `target_hourly_breakout_1h` ігнорує `indicator_col: BB_Upper`;
- `target_intraday_volatility_15m` фактично рахує зміну `high`, а не майбутній high-low range;
- weekly targets використовують неоднозначний `shift: -7`, що не обов’язково відповідає одному торговому тижню.

**Рекомендація:** кожен target повинен мати версію, формальну математичну специфікацію, правильний торговий горизонт і golden tests із відомою відповіддю.

### 2.4. Фінансові метрики мають активні суперечності

Виявлено:

- `win_rate` у comprehensive backtest рахується через повторний `pct_change()` уже готових returns;
- `regime_scorecard.avg_return` насправді використовує середнє значення predictions, а не реалізованих returns;
- `expertise_map` визначає «кращу» модель за частотою використання, а не за її результатом;
- `chaos_efficiency` декларує avoided drawdown, але переважно рахує зміну exposure/confidence;
- один summary одночасно містить Sharpe приблизно `-3131.87` і `-1.28`;
- той самий summary містить різні значення max drawdown;
- `avg_return` у regime scorecard може набувати економічно абсурдних значень на кшталт `47.24`.

**Рекомендація:** один canonical equity curve, один canonical return series, одна frequency/annualization policy та один набір фінансових калькуляторів.

### 2.5. Співвідношення features до samples надто ризикове

У типовому training run спостерігалося приблизно:

- 192 train rows;
- 19 validation rows;
- 58 test rows, які не доходять до normal trainer;
- приблизно 237 model features.

Stage 3 має feature selection, але звичайний Stage 4 знову бере практично всі числові non-target колонки.

**Рекомендація:** train-only feature selection усередині кожного fold, обмеження 20–40 features для першого контрольного контуру та ablation за feature families.

### 2.6. Поточний drift evidence недостатній

На диску є prediction drift report:

- лише 10 current predictions;
- PSI із 10 bins;
- KS не підтверджує drift;
- PSI та Wasserstein заявляють drift;
- підсумок рекомендує retraining протягом кількох годин.

Поточний default у коді вимагає більше samples, тому такий старий або спеціально налаштований artifact не повинен використовуватися як production evidence.

### 2.7. DEAN-OS має сильні safety boundaries, але ще не має closed loop

Позитивно:

- review-only межі;
- заборона автоматичного production config write;
- заборона autonomous trading;
- provenance та evidence contracts;
- перспективна архітектура для hypothesis/outcome memory.

Фактичний стан:

- у research corpus багато документів;
- в outcome tracker лише 5 predictions;
- 0 completed outcomes;
- recommendation memory порожня;
- значна частина агентної оцінки ще не може бути відкалібрована.

**Рекомендація:** пріоритетом DEAN-OS зробити prospective outcome collection, а не додавання нових агентів.

### 2.8. Є constructed-but-unused і advertised-but-missing компоненти

Серед прикладів:

- advertised CLI mode `calibrate`, але відповідний executor method відсутній;
- actor/critic filtering може не мати реальних зареєстрованих ACTOR/CRITIC агентів;
- data freshness та feature drift monitors конструюються, але їх checks можуть не викликатися;
- model pool та quality controller створюються у CLI, але не підключені до реального маршруту;
- частина hybrid components створюється, але ніколи не викликається;
- LLM cognitive lenses можуть залишатися інертними через `llm_client=None`;
- adaptive parameter manager може створюватися без споживання його результатів;
- risk/train-test/config policies розпорошені між кількома YAML та defaults.

---

## 3. Пріоритетний план

### P0 — довіра до даних і результатів

1. Data identity та duplicate gate.
2. Правильна нормалізація `1h/60m`.
3. Перевірка calendar/timezone/session semantics.
4. Виправлення service-column alignment.
5. Immutable batch/run manifests.
6. Target semantic fixes.
7. Справжній untouched test.
8. Заборона unconditional `CHAMP`.
9. Canonical metrics.
10. Quarantine старих datasets/models.

### P1 — компактний доказовий pipeline

1. Один timeframe, бажано спочатку `1d`.
2. Малий, ліквідний та чітко визначений universe.
3. Два основні targets: net return і direction після costs.
4. Три моделі: linear/logistic baseline, LightGBM, Random Forest.
5. 20–40 features.
6. Чотири walk-forward folds.
7. Окремий фінальний locked period.
8. Cost stress ×1.5 та ×2.
9. MLflow tracking/registry.
10. Prediction/outcome ledger.

### P2 — якість рішень і portfolio layer

1. `NO TRADE` як first-class decision.
2. Uncertainty intervals.
3. Pooled/cross-sectional models.
4. Portfolio construction.
5. Event-driven simulator.
6. Champion–challenger shadow testing.
7. Prospective calibration.

### R&D — лише після стабільного baseline

1. Multi-task learning.
2. Triple-barrier labels.
3. Meta-labeling.
4. Regime transitions.
5. Options features.
6. Causal agents.
7. Adaptive ensembles.
8. Online learning.

---

## 4. Data contracts і point-in-time архітектура

### 4.1. Обов’язкова identity-схема

Для кожного market row:

- `ticker`;
- `datetime`;
- `interval`;
- `source`;
- `source_symbol`;
- `event_time`;
- `available_at`;
- `ingested_at`;
- `revision_id`;
- `source_hash`.

Комбінація логічних ключів повинна бути унікальною.

### 4.2. As-of firewall

Кожне зовнішнє значення повинно мати:

- коли подія сталася;
- коли інформація стала доступною;
- коли система її отримала;
- чи була вона пізніше переглянута.

Потрібен автоматичний time-travel test:

> Запуск станом на історичну дату не має права прочитати жодного факту, revision або artifact, доступного після цієї дати.

### 4.3. Immutable run layout

Рекомендований формат:

```text
runs/<run_id>/
  raw/
  cleaned/
  features/
  targets/
  models/
  evaluation/
  manifest.json
```

У manifest:

- git commit;
- config hash;
- dataset hashes;
- schema versions;
- target versions;
- feature schema hash;
- cost model version;
- training cutoff;
- environment/dependency lock;
- random seeds.

`latest` має бути лише pointer, а не mutable artifact.

### 4.4. Atomic writes

Критичний файл:

1. записується у temporary path;
2. перевіряється schema/hash;
3. атомарно перейменовується;
4. лише потім оновлюється latest pointer.

### 4.5. Quarantine workflow

При data incident:

1. позначити batch як invalid;
2. знайти всі похідні features, targets і models за hash;
3. відкликати champion aliases;
4. заблокувати prediction;
5. створити incident report;
6. перевипустити batch;
7. повторити validation;
8. лише після цього дозволити promotion.

### 4.6. Timeframe і calendar contracts

Перевіряти:

- UTC та exchange-local time;
- daylight saving;
- holidays;
- early-close sessions;
- pre/after-market;
- auction sessions;
- bar-start проти bar-end;
- calendar day проти trading session;
- `15m → 60m → 1d` aggregation;
- відсутність mixed cadence;
- availability OHLC лише після закриття бару.

### 4.7. Cross-source validation

Для підозрілих або контрольних OHLCV:

- порівняння двох джерел;
- split/dividend reconciliation;
- missing sessions;
- abnormal jumps;
- volume disagreement.

### 4.8. Survivorship/universe contracts

Зберігати історичні:

- index constituents;
- sector classification;
- delisted companies;
- ticker changes;
- mergers;
- bankruptcies;
- IPO dates;
- borrow availability.

### 4.9. Corporate actions

Окремо обробляти:

- splits;
- reverse splits;
- dividends;
- special dividends;
- spin-offs;
- rights offerings;
- mergers;
- delistings;
- ticker changes.

Не змішувати:

- raw execution prices;
- split-adjusted prices;
- dividend-adjusted total-return series.

---

## 5. Targets і labels

### 5.1. Target contract

Кожен target повинен мати:

- унікальну назву;
- версію;
- формулу;
- horizon;
- timeframe applicability;
- `known_at`;
- required columns;
- transaction-cost policy;
- missing/outlier policy;
- boundary masking;
- classification balance expectations;
- unit tests;
- golden example.

### 5.2. Потенційно корисні targets

- gross future return;
- net future return;
- probability of return exceeding costs;
- direction після costs;
- market residual return;
- sector residual return;
- cross-sectional rank;
- top/bottom quantile;
- future realized volatility;
- downside volatility;
- maximum adverse excursion;
- maximum favorable excursion;
- probability of drawdown before profit;
- time-to-hit-profit;
- time-to-hit-stop;
- regime transition;
- liquidity deterioration;
- spread expansion;
- gap risk;
- earnings reaction;
- abnormal volume;
- multi-horizon labels;
- survival/time-to-event labels.

### 5.3. Triple-barrier labeling

Результат визначається першою подією:

- profit barrier;
- stop barrier;
- time barrier.

### 5.4. Meta-labeling

Перша модель знаходить candidate setup або напрям. Друга вирішує, чи варто виконувати сигнал з урахуванням:

- regime;
- liquidity;
- uncertainty;
- costs;
- event risk.

### 5.5. Multi-task targets

Одна модель потенційно може спільно прогнозувати:

- return;
- direction;
- volatility;
- downside risk;
- кілька горизонтів.

Це R&D після якісного single-task baseline.

---

## 6. Feature engineering

### 6.1. Feature economy

Кожна feature family повинна довести:

- OOS improvement;
- stability;
- coverage;
- economic relevance;
- latency;
- collection cost;
- maintenance cost.

Приблизна оцінка:

```text
feature_value =
OOS improvement
× stability
× coverage
÷ collection cost
÷ latency
÷ operational risk
```

### 6.2. Train-only selection

Feature selection виконується:

- лише на train частині;
- окремо всередині кожного fold;
- без доступу до locked test;
- з логуванням selected-feature hash;
- з перевіркою stability.

### 6.3. Feature redundancy graph

Виявляти:

- correlation;
- mutual information;
- однакові формули;
- однакові source columns;
- різні назви одного сигналу;
- майже однакові windows;
- однакові missingness patterns.

З кожного кластера залишати представника.

### 6.4. Stability of explanations

Вимірювати:

- importance across folds;
- importance across seeds;
- importance across regimes;
- coefficient sign stability;
- permutation importance;
- SHAP stability;
- importance після видалення корельованих ознак.

### 6.5. Potential price/volatility features

- Parkinson volatility;
- Garman–Klass;
- Rogers–Satchell;
- upside/downside semivariance;
- realized skewness;
- realized kurtosis;
- jump proxy;
- bipower variation;
- volatility-of-volatility;
- drawdown duration;
- trend persistence;
- gap decomposition;
- overnight return;
- open-to-close return;
- close-to-open return.

### 6.6. Liquidity/microstructure proxies

- Amihud illiquidity;
- turnover;
- dollar volume;
- volume surprise;
- price-impact proxy;
- high-low spread proxy;
- zero-return frequency;
- intraday volume seasonality;
- relative volume by time of day;
- close-location value;
- auction imbalance, якщо доступний.

### 6.7. Cross-sectional features

- market beta;
- sector beta;
- idiosyncratic volatility;
- market residual;
- sector residual;
- sector rank;
- universe momentum rank;
- breadth;
- dispersion;
- rolling correlation;
- correlation centrality;
- factor exposure;
- crowding proxy.

### 6.8. Fundamental features

- gross profitability;
- margin changes;
- free-cash-flow yield;
- accruals;
- working-capital change;
- leverage;
- interest coverage;
- dilution;
- buybacks;
- earnings quality;
- revenue acceleration;
- inventory growth;
- capex intensity.

### 6.9. Event features

- earnings surprise;
- estimate revisions;
- guidance changes;
- filing amendments;
- 8-K categories;
- insider transactions;
- buybacks;
- secondary offerings;
- debt issuance;
- management changes;
- litigation/regulatory events;
- product launches;
- supply-chain disruptions.

### 6.10. Volatility/options features

Після появи надійного джерела:

- VIX term-structure slope;
- VIX curvature;
- VVIX;
- implied-volatility rank;
- IV minus realized volatility;
- skew;
- put/call ratios;
- open-interest concentration;
- expected move;
- volatility risk premium;
- gamma-exposure proxy.

### 6.11. Flow features

- ETF flows;
- sector ETF flows;
- fund flows;
- short interest;
- FINRA short-sale volume з правильними обмеженнями;
- insider purchases/sales;
- institutional holdings changes;
- issuance;
- buybacks;
- options positioning.

### 6.12. Signal half-life

Для кожної feature family визначати:

- horizon;
- decay;
- результат після затримки;
- реальну можливість виконання;
- залежність від market regime.

---

## 7. Models

### 7.1. Сильні baselines

- zero-return;
- previous return;
- rolling mean;
- historical mean;
- majority class;
- logistic regression;
- linear/ridge/elastic-net;
- simple momentum;
- simple mean reversion;
- sector-relative momentum;
- volatility-scaled buy-and-hold;
- random predictions із тією ж частотою trades.

### 7.2. Інтерпретовані кандидати

- GAM;
- Explainable Boosting Machine;
- monotonic gradient boosting;
- quantile regression;
- robust regression;
- ranking models.

### 7.3. Pooled/cross-sectional models

Замість багатьох маленьких ticker-target моделей:

- спільна модель на кількох тикерах;
- ticker/sector як context;
- relative/residual targets;
- cross-sectional ranking;
- leave-period-out validation;
- іноді leave-ticker-out validation.

### 7.4. Uncertainty

Prediction має повертати:

- point estimate;
- uncertainty interval;
- probability edge exceeds costs;
- uncertainty source;
- calibration status.

Можливі підходи:

- quantile regression;
- ensemble variance;
- bootstrap intervals;
- rolling conformal prediction.

### 7.5. Monotonic/economic constraints

Наприклад:

- більші costs не можуть збільшувати expected net edge;
- більша uncertainty не повинна збільшувати position size;
- більший spread не має покращувати execution score;
- більший downside не має збільшувати long allocation.

### 7.6. Ensemble diversity

Вибирати ensemble members не за різними назвами алгоритмів, а за:

- prediction correlation;
- residual correlation;
- disagreement in stress regimes;
- marginal OOS contribution;
- tail-risk reduction;
- ablation without each member.

### 7.7. Regime transitions

Дослідити не лише current regime, а:

- probability of transition;
- change points;
- volatility expansion/contraction;
- correlation breakdown;
- liquidity deterioration.

За недостатньої вибірки повертати `unknown`.

### 7.8. Cold start

Новий ticker/model/target:

- статус `unmeasured`;
- нульовий trading authority;
- shadow only;
- мінімум prospective outcomes;
- окрема calibration;
- жодного автоматичного champion.

### 7.9. Model retirement

Модель відкликається, якщо:

- schema несумісна;
- source зник;
- target semantics змінилися;
- calibration зламалась;
- performance деградувала;
- turnover надто великий;
- новий champion стабільно кращий;
- serialization/library version небезпечна.

---

## 8. Validation і статистична надійність

### 8.1. Correct split policy

- chronological splits;
- purge;
- embargo;
- expanding або rolling windows;
- locked final test;
- nested validation для tuning;
- жодного selection по test.

### 8.2. Walk-forward evidence

Зберігати:

- кожен fold;
- train/validation dates;
- sample counts;
- selected features;
- model params;
- baseline delta;
- fold metrics;
- worst fold;
- median fold;
- fold dispersion.

### 8.3. Negative controls

- shuffled target;
- random labels;
- features shifted backward;
- artificial future feature, яку guard має заблокувати;
- signal delay;
- double costs;
- remove top features;
- exclude feature families;
- alternative universe;
- alternative period.

### 8.4. Research trial ledger

Рахувати всі:

- targets;
- feature sets;
- models;
- parameter trials;
- horizons;
- universes;
- thresholds;
- cost variants;
- rejected experiments.

### 8.5. Statistical tests

- block bootstrap;
- stationary bootstrap;
- confidence intervals;
- White’s Reality Check;
- SPA test;
- false discovery rate;
- minimum track-record length;
- Deflated Sharpe;
- Probability of Backtest Overfitting;
- sensitivity surfaces;
- performance across seeds.

### 8.6. Prequential evaluation

1. Спрогнозувати.
2. Зафіксувати prediction.
3. Дочекатися outcome.
4. Оцінити.
5. Лише потім оновити модель.

### 8.7. Implausibility alarms

Результати на кшталт:

- R² близько 0.99;
- майже 100% accuracy;
- величезний Sharpe;
- відсутній drawdown;
- perfect performance на малій вибірці

повинні викликати leakage/reproducibility audit, а не автоматичний promotion.

---

## 9. Model promotion lifecycle

Рекомендовані стани:

```text
experimental
→ candidate
→ validation_passed
→ locked_test_passed
→ shadow
→ paper_candidate
→ champion
→ retired
```

### Promotion gate

Champion дозволяється лише за умов:

- data contract green;
- target contract verified;
- baseline beaten;
- locked test measured;
- walk-forward stable;
- costs included;
- stressed costs acceptable;
- no negative-control failure;
- feature schema locked;
- model/data/code provenance complete;
- uncertainty measured;
- shadow evidence mature;
- risk approval.

### Champion–challenger

- однаковий as-of context;
- однакові costs;
- однакові outcome rules;
- prospective comparison;
- statistical та economic significance;
- rollback package.

---

## 10. Decision layer

### 10.1. `NO TRADE` як first-class result

```text
expected_edge =
predicted_return
- estimated_costs
- uncertainty_buffer
- regime_penalty
- liquidity_penalty
```

Якщо edge недостатній — `NO TRADE`.

### 10.2. Прогнозувати корисність рішення

Фінальний objective має бути ближчим до:

- net return;
- probability of covering costs;
- expected utility;
- risk-adjusted edge;
- ranking;
- allocation;
- abstention.

### 10.3. Людська картка рішення

Для кожної потенційної угоди:

- decision;
- expected edge;
- uncertainty;
- costs;
- position risk;
- regime;
- supporting evidence;
- contradictory evidence;
- invalidation conditions;
- model/data hashes;
- `NO TRADE` reason.

### 10.4. Why-changed explanation

Якщо BUY змінився на NO TRADE:

- які дані змінилися;
- які features змінилися;
- чи змінилася модель;
- regime;
- costs;
- risk;
- uncertainty.

---

## 11. Portfolio construction

Potential approaches:

- rank-based allocation;
- volatility targeting;
- equal risk contribution;
- sector-neutral;
- beta-neutral;
- hierarchical risk parity;
- turnover-penalized optimization;
- robust covariance;
- maximum diversification;
- fractional Kelly з жорстким cap;
- uncertainty-aware sizing.

Кожен optimizer порівнювати з equal-weight або simple risk-scaled baseline.

### Portfolio attribution

Розкладати результат на:

- market beta;
- sector beta;
- factor exposures;
- selection;
- timing;
- leverage;
- costs;
- slippage;
- idiosyncratic alpha;
- ticker;
- regime;
- feature family;
- event type.

---

## 12. Risk management

### 12.1. Position sizing

Базувати насамперед на:

- volatility;
- liquidity/capacity;
- portfolio risk budget;
- correlation;
- max loss;
- uncertainty.

Raw confidence не повинен прямо визначати позицію.

### 12.2. Concentration risk

Перевіряти:

- sector concentration;
- factor concentration;
- ETF overlap;
- correlated signals;
- common news;
- supply-chain overlap;
- currency/rate exposure;
- correlation spikes.

### 12.3. Pre-trade checks

- data freshness;
- model status;
- market open;
- halt;
- price sanity;
- spread;
- slippage;
- liquidity;
- exposure;
- concentration;
- daily loss;
- drawdown;
- event blackout;
- earnings;
- borrow;
- participation limit;
- stale prediction;
- duplicate order;
- kill switch.

### 12.4. Stress scenarios

- flash crash;
- gap open;
- volatility spike;
- liquidity collapse;
- correlation to one;
- missing data source;
- stale model;
- spread expansion;
- delayed execution;
- rate shock;
- sector shock;
- overnight halt.

---

## 13. Backtest і execution realism

### 13.1. Два рівні simulator

- vectorized research simulator для швидкості;
- event-driven validation simulator для фінального evidence.

### 13.2. Event-driven order lifecycle

- submitted;
- accepted;
- partially filled;
- filled;
- rejected;
- canceled;
- expired;
- delayed;
- gap;
- halt;
- insufficient liquidity;
- insufficient cash;
- short unavailable.

### 13.3. Order types

- market;
- limit;
- stop;
- stop-limit;
- cancel/replace;
- auction participation.

### 13.4. Costs

- commission;
- bid-ask spread;
- slippage;
- market impact;
- borrow fee;
- locate;
- financing;
- margin;
- cash drag;
- short dividends;
- regulatory fees;
- currency conversion;
- partial-fill opportunity cost.

### 13.5. Latency tests

- same close лише за реальної доступності;
- next open;
- next bar;
- 1/5/15-minute delay;
- random latency;
- missed first fill.

### 13.6. Missing-return policy

Не використовувати silent zero-fill для active position. Можливі:

- fail;
- quarantine;
- conservative mark;
- explicit missing state;
- known fallback contract.

---

## 14. Monitoring і drift

Розділяти:

- schema drift;
- data-quality drift;
- covariate drift;
- label/prior drift;
- prediction drift;
- calibration drift;
- performance/concept drift;
- execution-cost drift.

Retraining не повинен запускатися лише через PSI.

### Operational metrics

- data freshness SLA;
- collector success;
- duplicate rate;
- schema failure;
- feature latency;
- inference latency;
- missing model rate;
- fallback rate;
- no-trade rate;
- stale prediction rate;
- artifact mismatch;
- recovery time;
- cost per experiment.

### Retraining protocol

- trigger;
- frozen cutoff;
- validation;
- old champion;
- challenger;
- shadow comparison;
- approval;
- rollback.

---

## 15. MLOps і reproducibility

### 15.1. MLflow

Використати один tracking/registry для local і Colab:

- dataset hash;
- feature schema;
- target version;
- params;
- all trials;
- metrics;
- artifacts;
- candidate/champion aliases;
- pre-deploy gate tags.

### 15.2. Dependency lock

- зафіксовані версії;
- hashes;
- окремі research/runtime dependencies;
- library version у model metadata;
- контроль major upgrades.

### 15.3. Reproducibility

Повторний frozen run повинен давати:

- однакові splits;
- однакові features;
- однакові IDs;
- predictions у межах tolerance;
- однаковий backtest.

Фіксувати seeds та відомий GPU/parallel nondeterminism.

### 15.4. Schema migrations

- schema version;
- migration scripts;
- backups;
- backward compatibility;
- rollback;
- old-version fixtures.

### 15.5. Idempotency

Повторний запуск не має:

- дублювати rows;
- дублювати trades;
- повторно списувати costs;
- перезаписувати immutable artifacts;
- повторно надсилати notification.

---

## 16. Testing і QA

### 16.1. Golden datasets

Малі deterministic datasets із:

- двома tickers;
- кількома timeframes;
- duplicate aliases;
- missing bar;
- split/dividend;
- breakout;
- volume spike;
- exact trades;
- exact PnL.

### 16.2. Property-based testing

- різні tickers/timeframes;
- timezone;
- NaN/inf;
- duplicates;
- shuffled rows;
- invariants targets;
- permutation invariance;
- scale invariance, де доречно.

### 16.3. Mutation testing

Перевірити, чи тест падає, якщо навмисно:

- змінити знак return;
- видалити shift;
- пропустити transaction costs;
- змішати ticker;
- повторно виконати `pct_change`;
- використати test для selection.

### 16.4. Differential testing

Порівнювати:

- два metric calculators;
- vectorized та event-driven simulators;
- independent target implementation;
- alternative data source.

### 16.5. Chaos testing

- API timeout;
- rate limit;
- corrupt file;
- incomplete write;
- missing model;
- stale cache;
- disk full;
- process crash;
- duplicate message;
- unavailable LLM;
- malformed agent output.

---

## 17. Architecture і codebase governance

### 17.1. Stable Core та Research Lab

Stable Core:

- data contracts;
- targets;
- splits;
- baselines;
- backtest;
- risk;
- registry;
- prediction ledger.

Research Lab:

- experimental enrichers;
- new agents;
- Optuna;
- deep learning;
- alternative targets;
- new data sources.

Перехід між ними лише через promotion evidence.

### 17.2. Component lifecycle registry

Статуси:

- `core`;
- `active`;
- `shadow`;
- `experimental`;
- `constructed_but_unused`;
- `broken`;
- `deprecated`;
- `archived`.

Для кожного:

- purpose;
- real caller;
- config;
- outputs;
- tests;
- last successful run;
- owner/review date.

### 17.3. Runtime call map

Автоматично показувати:

- що імпортується;
- що конструюється;
- що реально викликається;
- що пише state;
- які configs читаються;
- які outputs споживаються.

### 17.4. Complexity budgets

Обмежити:

- active targets;
- feature families;
- models per target;
- agents per default cycle;
- orchestration paths;
- config sources;
- fallback layers.

### 17.5. Single Policy Manager

Централізувати:

- train/validation/test policy;
- purge/embargo;
- promotion;
- risk limits;
- transaction costs;
- retraining;
- shadow/paper permissions;
- adaptive-parameter outputs.

### 17.6. Error taxonomy

Розділяти:

- unavailable data;
- invalid data;
- schema mismatch;
- insufficient evidence;
- model incompatible;
- risk blocked;
- execution unavailable;
- programming bug.

Programming bugs не повинні тихо перетворюватися на fallback.

### 17.7. Architecture Decision Records

Фіксувати:

- рішення;
- альтернативи;
- обґрунтування;
- ризики;
- verification;
- умови перегляду.

---

## 18. DEAN-OS та LLM/agent governance

### 18.1. Consensus independence

П’ять агентів із тими самими джерелами не є п’ятьма незалежними доказами.

Враховувати:

- source overlap;
- model/prompt overlap;
- causal dependence;
- error correlation;
- duplicated evidence.

### 18.2. Separation of duties

- proposer;
- verifier;
- risk reviewer;
- promotion gate;
- human approval.

Агент не затверджує власну пропозицію.

### 18.3. Hypothesis registry

Для кожної тези:

- hypothesis;
- as-of;
- sources/hashes;
- tickers;
- horizon;
- expected direction;
- confirmation;
- falsification;
- outcome dates;
- final evaluation.

### 18.4. Strategy Graveyard

Зберігати:

- rejected hypotheses;
- failed experiments;
- причини;
- trials;
- regimes;
- умови повторного перегляду.

### 18.5. Agent calibration

Для кожного agent/domain/horizon:

- Brier;
- calibration;
- direction accuracy;
- abstention;
- evidence quality;
- contradiction rate;
- citation validity;
- cost;
- latency;
- error correlation.

Не змінювати вагу до достатньої кількості reviewed outcomes.

### 18.6. RAG evaluation

- retrieval recall;
- retrieval precision;
- freshness;
- duplicates;
- contradictory versions;
- correct as-of;
- citation validity;
- claim support;
- source diversity.

### 18.7. Prompt-injection boundary

Новини, filings і web pages є недовіреним evidence.

LLM не має права:

- виконувати інструкції з джерела;
- читати secrets;
- змінювати production config;
- затверджувати модель;
- самостійно рахувати final metrics;
- надсилати orders.

### 18.8. Deterministic core

Числові:

- returns;
- metrics;
- splits;
- risk;
- promotion;
- execution

мають залишатися у deterministic code. LLM може пояснювати, пропонувати й перевіряти evidence, але не бути джерелом фінальної математики.

---

## 19. Security

### 19.1. Secrets

- не логувати API keys;
- redact headers/query params;
- scan repo/history/logs;
- rotate exposed keys;
- separate dev/test credentials;
- мінімальні permissions;
- не зберігати secrets у artifacts.

### 19.2. Model serialization

Оскільки pickle/joblib небезпечні для недовірених файлів:

- hash verification;
- allowlisted directories;
- provenance;
- жодних зовнішніх pickle;
- isolation legacy loading;
- безпечніший формат, де можливо.

### 19.3. Supply chain

- dependency lock;
- vulnerability scan;
- SBOM;
- hashed packages;
- minimal runtime;
- version check при model load.

---

## 20. Operations і recovery

### 20.1. Окремі середовища

- development;
- test;
- research;
- shadow;
- paper;
- production-like.

Окремі:

- configs;
- credentials;
- databases;
- registries;
- portfolios;
- notifications;
- permissions.

### 20.2. Backup/restore

- backup scope;
- schedule;
- retention;
- encryption;
- restore test;
- RPO/RTO;
- recovery runbook;
- champion rollback.

### 20.3. Incident response

- severity;
- affected artifacts;
- automatic quarantine;
- owner;
- timeline;
- root cause;
- remediation;
- regression test;
- postmortem.

### 20.4. Fail-safe matrix

Приклади:

- немає news → модель без news або NO TRADE;
- stale macro → macro model blocked;
- missing spread → не використовувати нуль;
- schema mismatch → quarantine;
- model load failure → NO TRADE;
- unknown regime → conservative policy.

---

## 21. Research governance

### 21.1. Preregistration

Перед експериментом:

- hypothesis;
- target;
- features;
- period;
- universe;
- metrics;
- baseline;
- success threshold;
- stop rule;
- allowed trials.

### 21.2. Value of Information

Перед новим джерелом/agent/model:

- яке рішення покращиться;
- яка uncertainty зменшиться;
- шанс змінити decision;
- cost;
- latency;
- maintenance;
- possible financial impact.

### 21.3. Research circuit breakers

Зупиняти experiment, якщо:

- shuffled target не знищує performance;
- features > train samples;
- validation занадто мала;
- baseline не переможений;
- delay знищує edge;
- cost stress знищує edge;
- folds нестабільні;
- data contract не пройдений;
- результат неправдоподібний.

### 21.4. Anti-goals

До P0/P1 не варто:

- додавати Transformer/LSTM;
- тренувати всі targets;
- запускати великий Optuna search;
- додавати сотні RSI/SMA;
- будувати нові агентні ієрархії;
- створювати великий dashboard;
- автоматизувати trading authority.

---

## 22. Metrics, які варто мати

### Data

- duplicate-key rate;
- missingness;
- constant/near-constant features;
- cadence match;
- point-in-time violations;
- source disagreement;
- freshness;
- schema version;
- quarantine count.

### Classification

- majority baseline;
- balanced accuracy;
- MCC;
- precision/recall;
- PR-AUC;
- ROC-AUC з обережністю;
- Brier score;
- log loss;
- ECE;
- calibration slope;
- top-decile lift.

### Regression/ranking

- MAE;
- RMSE;
- OOS R²;
- baseline delta;
- Spearman/rank IC;
- directional accuracy;
- quantile loss;
- calibration by prediction buckets.

### Trading

- gross/net return;
- turnover;
- trade hit rate;
- expectancy;
- profit factor;
- Sharpe;
- Sortino;
- Calmar;
- max drawdown;
- drawdown duration;
- CVaR;
- exposure;
- concentration;
- capacity;
- slippage;
- cost contribution.

### Robustness

- worst fold;
- median fold;
- fold dispersion;
- bootstrap intervals;
- Deflated Sharpe;
- PBO;
- stressed costs;
- delay sensitivity;
- regime coverage;
- seed stability.

### Operations

- collector success;
- data latency;
- inference latency;
- fallback/no-trade rate;
- stale predictions;
- artifact mismatch;
- recovery time;
- compute cost.

---

## 23. Інструменти

Перед додаванням нових залежностей варто підключити вже наявні або близькі до наявних:

- **MLflow:** experiment tracking, lineage, registry, aliases.
- **Optuna:** лише всередині правильного nested/walk-forward validation.
- **Evidently:** drift evidence після достатньої кількості samples.
- **Ruff:** static lint.
- **mypy:** type contracts критичного core.
- **pytest:** unit/contract/integration/golden/property tests.
- **pre-commit:** швидкі обов’язкові перевірки.
- **Pandera або аналогічний власний contract layer:** DataFrame schemas.
- **DVC/lakeFS або immutable SHA manifests:** data versioning.
- **secret scanner та dependency vulnerability scanner:** security.

Необхідно також:

- прибрати tracked `.pyc`;
- не зберігати archives/binaries у source tree без потреби;
- зафіксувати dependencies;
- узгодити `pytest.ini` і `pyproject.toml`, щоб один config не ігнорував інший.

---

## 24. Формат triage для Claude

Для кожної рекомендації Claude має повернути:

1. `status`: `already_exists / partial / missing / obsolete / not_needed`.
2. Реальний active caller.
3. Який ризик усувається.
4. Очікувана користь.
5. Вартість і складність.
6. Залежності.
7. Як перевірити результат.
8. Пріоритет: `P0 / P1 / P2 / R&D / archive`.
9. Які старі компоненти можна прибрати.
10. Чи змінює пункт trading authority.

Рекомендований фінальний результат triage:

| Компонент/ідея | Поточний стан | Рішення | Пріоритет | Verification |
|---|---|---|---|---|
| Data identity gate | Partial | Fix | P0 | Golden + real batch |
| Locked test | Broken path | Fix | P0 | Test artifact |
| Target semantics | Partial/incorrect | Fix | P0 | Contract tests |
| MLflow registry | Partial | Wire | P1 | End-to-end run |
| New options features | Missing | Research later | R&D | Ablation |
| Unused hybrid components | Constructed-unused | Archive/wire | P1 | Runtime map |

---

## 25. Мінімальний контрольний експеримент

Рекомендована перша доказова конфігурація:

- timeframe: `1d`;
- невеликий визначений universe ліквідних акцій;
- targets:
  - future net return;
  - direction after costs;
- models:
  - linear/logistic;
  - LightGBM;
  - Random Forest;
- 20–40 train-only selected features;
- 4 walk-forward folds;
- purge/embargo за target horizon;
- один locked final test;
- baseline suite;
- costs base/×1.5/×2;
- uncertainty;
- no-trade threshold;
- immutable run manifest;
- MLflow record;
- shadow-only prospective ledger.

Модель не отримує статус champion, доки не:

- переможе baseline;
- пройде негативні controls;
- покаже стабільність folds;
- переживе stressed costs;
- отримає mature shadow outcomes.

---

## 26. Зовнішні методологічні джерела

- Time-series split із gap:  
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

- Balanced accuracy, Brier та інші model metrics:  
  https://scikit-learn.org/stable/modules/model_evaluation.html

- Probability calibration:  
  https://scikit-learn.org/stable/modules/calibration.html

- Deflated Sharpe Ratio:  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551

- Probability of Backtest Overfitting:  
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

- ALFRED point-in-time macro data:  
  https://fred.stlouisfed.org/docs/api/fred/alfred.html

- SEC XBRL/Companyfacts API:  
  https://www.sec.gov/search-filings/edgar-application-programming-interfaces

- Cboe VIX term structure:  
  https://www.cboe.com/tradable-products/vix/term-structure

- FINRA short-sale volume limitations:  
  https://www.finra.org/finra-data/browse-catalog/short-sale-volume

- MLflow Model Registry:  
  https://mlflow.org/docs/latest/ml/model-registry/

---

## 27. Остаточний принцип

Для кожної нової ідеї потрібно ставити питання:

> Чи підвищує вона достовірну out-of-sample економічну цінність після витрат, невизначеності та multiple-testing correction, чи лише збільшує складність системи?

Найсильніша потенційна комбінація для цього проєкту:

1. Якісний point-in-time data engine.
2. Чесний research/validation/promotion pipeline.
3. `NO TRADE` та uncertainty-aware decision layer.
4. Portfolio/risk engine із реалістичними costs.
5. DEAN-OS як пам’ять перевірюваних гіпотез, невдалих експериментів і prospective outcomes.

Саме наявність доказів, negative memory, provenance та здатності відмовитися від угоди відрізнятиме зрілу систему від великого набору моделей, індикаторів і агентів.
