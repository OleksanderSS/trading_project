# Каталог ролей: що виміряно про кожну величину

**Файл будується скриптом, не рукою.** Джерело — `diagnostic_reports/feature_roles_1d.csv`, який лишає по собі `leading_feature_report.py`. Щоб оновити — перезапустити вимір.

- виміряно: **2026-09-07**
- ціль: `target_relative_return_5d`
- денний кадр, запечатано з **2023-09-01**
- гіпотез перевірено: **253**
- величин у каталозі: **1390**, з них **388** справді виміряно, а **1002** не мають у дослідній частині нічого, про що можна судити

> **Про ці 1002.** Це твердження про ВІДСУТНІСТЬ даних, а не про ряд. До 07.09 їх тут не було видно взагалі: 935 колонок не лишали жодного рядка у звіті, бо цикл їх мовчки пропускав, а 67 діставали вирок «market-wide: use as interaction» — тобто пораду будувати взаємодію зі сталою. Читач не міг відрізнити «виміряли й нічого не знайшли» від «до виміру не дійшло».

## Скільки чого

| роль | скільки | що це означає |
|---|---|---|
| absent before the seal: nothing to judge | 935 | жодного значення до дати сілу. Колонка існує, збирач звітує про успіх, виміряти на ній нічого не можна — це твердження про ВІДСУТНІСТЬ даних, а не про ряд |
| inside the noise for this many tests | 155 | не проходить поправку Бенджаміні-Хохберга на кількість перевірок |
| market-wide: use as interaction | 135 | одне значення на дату для всіх імен, АЛЕ воно змінюється від дати до дати. Ранжувати НЕ МОЖЕ за конструкцією; входить лише як взаємодія з чутливістю імені |
| sign flipped out of sample | 79 | напрямок не втримався поза вибіркою — монета |
| one value everywhere: a filled default | 67 | одне-єдине значення на всіх рядках до сілу — заповнювач, а не дані. Раніше такі колонки діставали вирок «market-wide», що пропонувало будувати взаємодію зі сталою |
| labels the name, not the moment | 17 | ранжує імена сталою величиною; прибери середнє тікера — і нічого не лишиться. Одна ставка, а не передбачення |
| too thin to judge | 1 | покриття замале, щоб щось стверджувати |
| survives, worth testing | 1 | не спростовано жодною перевіркою. Наступний крок — дохідність на одиницю ризику й беззбитковість проти витрат |

## Що вижило

| величина | ic/date | t | останній квартал | дат | покриття |
|---|---|---|---|---|---|
| `peer_divergence_1d` | -0.0124 | -3.98 | -2.03 | 1,763 | 99% |

## Найсильніші за кожною роллю

Найбільший |ic_out| у групі — щоб було видно, чого саме коштувала кожна відмова.

**absent before the seal: nothing to judge** (935)

- `hash` — ic_out +nan, ic/date —, t —, покриття 0%
- `fear_greed_index_1d` — ic_out +nan, ic/date —, t —, покриття 0%
- `filing_days_since_last_1d` — ic_out +nan, ic/date —, t —, покриття 0%
- `filing_count_30d_1d` — ic_out +nan, ic/date —, t —, покриття 0%
- `filing_material_30d_1d` — ic_out +nan, ic/date —, t —, покриття 0%

**inside the noise for this many tests** (155)

- `AUTOCORR_1d` — ic_out +0.0232, ic/date +0.0058, t +1.79, покриття 100%
- `HURST_EXPONENT_1d` — ic_out -0.0231, ic/date -0.0101, t -2.65, покриття 100%
- `SKEWNESS_1d` — ic_out -0.0184, ic/date -0.0096, t -2.87, покриття 100%
- `significant_events_30d_1d` — ic_out -0.0175, ic/date -0.0163, t -3.60, покриття 100%
- `SHARPE_RATIO_1d` — ic_out +0.0156, ic/date +0.0160, t +2.52, покриття 100%

**market-wide: use as interaction** (135)

- `FRED_SAHMREALTIME_1d` — ic_out -0.0413, ic/date —, t —, покриття 17%
- `FRED_ICSA_1d` — ic_out -0.0374, ic/date —, t —, покриття 58%
- `FRED_CCSA_1d` — ic_out -0.0318, ic/date —, t —, покриття 57%
- `FRED_DCOILWTICO_1d` — ic_out +0.0317, ic/date —, t —, покриття 100%
- `cftc_gold_net_pct_1d` — ic_out -0.0289, ic/date —, t —, покриття 30%

**sign flipped out of sample** (79)

- `market_context_fed_funds_trend_1d` — ic_out +0.0175, ic/date +0.0057, t +0.93, покриття 100%
- `VOLATILITY_50_1d` — ic_out -0.0158, ic/date -0.0037, t -0.55, покриття 100%
- `market_context_volatility_ratio_1d` — ic_out +0.0126, ic/date +0.0098, t +2.59, покриття 100%
- `market_context_volatility_20d_1d` — ic_out -0.0116, ic/date +0.0008, t +0.13, покриття 100%
- `ROLLING_VOL_20_1d` — ic_out -0.0116, ic/date +0.0008, t +0.13, покриття 100%

**one value everywhere: a filled default** (67)

- `hour_1d` — ic_out +nan, ic/date —, t —, покриття 100%
- `market_session_1d` — ic_out +nan, ic/date —, t —, покриття 100%
- `hour_sin_1d` — ic_out +nan, ic/date —, t —, покриття 100%
- `hour_cos_1d` — ic_out +nan, ic/date —, t —, покриття 100%
- `FRED_BAMLC0A0CM_1d` — ic_out +nan, ic/date —, t —, покриття 0%

**labels the name, not the moment** (17)

- `MAX_DRAWDOWN_1d` — ic_out -0.0299, ic/date -0.0343, t -7.39, покриття 100%
- `CURRENT_DRAWDOWN_1d` — ic_out -0.0296, ic/date -0.0340, t -7.32, покриття 100%
- `fund_return_on_equity_1d` — ic_out +0.0144, ic/date +0.0198, t +4.62, покриття 38%
- `state_SMA_200_1d` — ic_out +0.0138, ic/date +0.0146, t +2.72, покриття 100%
- `insider_net_value_30d_1d` — ic_out -0.0131, ic/date -0.0194, t -6.26, покриття 17%

**too thin to judge** (1)

- `context_pattern_id_1d` — ic_out -0.0093, ic/date —, t —, покриття 3%

**survives, worth testing** (1)

- `peer_divergence_1d` — ic_out -0.0118, ic/date -0.0124, t -3.98, покриття 99%
