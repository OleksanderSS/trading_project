# Каталог ролей: що виміряно про кожну величину

**Файл будується скриптом, не рукою.** Джерело — `diagnostic_reports/feature_roles_1d.csv`, який лишає по собі `leading_feature_report.py`. Щоб оновити — перезапустити вимір.

- виміряно: **2026-08-29**
- ціль: `target_relative_return_5d`
- денний кадр, запечатано з **2023-09-01**
- гіпотез перевірено: **245**
- величин у каталозі: **450**

## Скільки чого

| роль | скільки | що це означає |
|---|---|---|
| market-wide: use as interaction | 203 | одне значення на дату для всіх імен. Ранжувати НЕ МОЖЕ за конструкцією; входить лише як взаємодія з чутливістю імені |
| inside the noise for this many tests | 155 | не проходить поправку Бенджаміні-Хохберга на кількість перевірок |
| sign flipped out of sample | 74 | напрямок не втримався поза вибіркою — монета |
| labels the name, not the moment | 14 | ранжує імена сталою величиною; прибери середнє тікера — і нічого не лишиться. Одна ставка, а не передбачення |
| too thin to judge | 3 | покриття замале, щоб щось стверджувати |
| faded: held once, gone in the latest quarter | 1 | ефект БУВ і згас. Пулована статистика його ще показує, останній квартал — ні |

## Що вижило

**Нічого.** Жодна з величин не проходить весь ланцюжок перевірок на цьому всесвіті.

Це вимір, а не поразка: 110 великих американських імен відповіли, що тут нема чого ловити. Запис існує саме для того, щоб цього не перевіряли вдруге.

## Найсильніші за кожною роллю

Найбільший |ic_out| у групі — щоб було видно, чого саме коштувала кожна відмова.

**market-wide: use as interaction** (203)

- `FRED_SAHMREALTIME_1d` — ic_out -0.0420, ic/date —, t —, покриття 17%
- `FRED_ICSA_1d` — ic_out -0.0382, ic/date —, t —, покриття 58%
- `FRED_DCOILWTICO_1d` — ic_out +0.0317, ic/date —, t —, покриття 100%
- `FRED_CCSA_1d` — ic_out -0.0308, ic/date —, t —, покриття 57%
- `cftc_gold_ls_ratio_1d` — ic_out -0.0289, ic/date —, t —, покриття 30%

**inside the noise for this many tests** (155)

- `AUTOCORR_1d` — ic_out +0.0232, ic/date +0.0058, t +1.79, покриття 100%
- `HURST_EXPONENT_1d` — ic_out -0.0231, ic/date -0.0101, t -2.65, покриття 100%
- `SKEWNESS_1d` — ic_out -0.0184, ic/date -0.0096, t -2.87, покриття 100%
- `peer_volatility_1d` — ic_out -0.0179, ic/date -0.0054, t -1.60, покриття 100%
- `SHARPE_RATIO_1d` — ic_out +0.0156, ic/date +0.0160, t +2.52, покриття 100%

**sign flipped out of sample** (74)

- `market_context_fed_funds_trend_1d` — ic_out +0.0175, ic/date +0.0057, t +0.92, покриття 100%
- `VOLATILITY_50_1d` — ic_out -0.0158, ic/date -0.0037, t -0.55, покриття 100%
- `peer_count_1d` — ic_out -0.0144, ic/date -0.0167, t -4.65, покриття 100%
- `market_context_volatility_ratio_1d` — ic_out +0.0126, ic/date +0.0098, t +2.59, покриття 100%
- `market_context_volatility_20d_1d` — ic_out -0.0116, ic/date +0.0008, t +0.13, покриття 100%

**labels the name, not the moment** (14)

- `MAX_DRAWDOWN_1d` — ic_out -0.0299, ic/date -0.0343, t -7.39, покриття 100%
- `CURRENT_DRAWDOWN_1d` — ic_out -0.0296, ic/date -0.0340, t -7.32, покриття 100%
- `fund_return_on_equity_1d` — ic_out +0.0144, ic/date +0.0198, t +4.62, покриття 38%
- `state_SMA_200_1d` — ic_out +0.0141, ic/date +0.0156, t +2.89, покриття 100%
- `insider_net_value_30d_1d` — ic_out -0.0131, ic/date -0.0194, t -6.26, покриття 17%

**too thin to judge** (3)

- `context_pattern_id_1d` — ic_out +0.0288, ic/date —, t —, покриття 3%
- `FRED_BAMLC0A0CM_1d` — ic_out +nan, ic/date —, t —, покриття 0%
- `FRED_BAMLH0A0HYM2_1d` — ic_out +nan, ic/date —, t —, покриття 0%

**faded: held once, gone in the latest quarter** (1)

- `significant_events_30d_1d` — ic_out -0.0175, ic/date -0.0163, t -3.60, покриття 100%
