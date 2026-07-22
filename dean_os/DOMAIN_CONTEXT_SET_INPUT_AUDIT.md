# Domain Context Set — Input Audit

Дата актуалізації: 2026-07-22. Цільовий домен: `semiconductor_ai_infrastructure`. Аудит і підготовка виконані офлайн: без мережі, LLM, pipeline regeneration, learning або trading.

## Підсумок

Для першого DomainContextSet зараз є **5 валідних domain-scoped кандидатів із 6**. Єдина відсутня готова родина — `sector_market`.

| Family | Domain envelope | Recursive loader | Реальний стан | Придатність |
|---|---|---|---|---|
| macro | `dean_domain_scoped_macro_evidence_envelope_v1` | так | ready, 9/9 configured series | кандидат |
| fundamentals | `dean_domain_scoped_fundamentals_envelope_v1` | build-time recursive verification | ready_with_gaps, 4/12 profile coverage | кандидат із прогалинами |
| sector_market | `dean_domain_scoped_sector_market_envelope_v1` | так | blocked: universe і benchmark mismatch | не кандидат |
| news | `dean_domain_scoped_news_envelope_v1` | так | ready_with_gaps, 4/5 lanes | кандидат із прогалинами |
| official_policy | `dean_domain_scoped_official_policy_envelope_v1` | так | ready_with_gaps, registry acceptance pending | кандидат із прогалинами |
| pipeline_context | `dean_domain_scoped_pipeline_context_envelope_v1` | так | ready, 3/3 lanes і 12/12 lineage references | кандидат |

## Виправлення початкового Gemini-аудиту

Початковий висновок `3/6` був неточний. Код `DomainScopedSectorMarketEnvelope` і `DomainScopedPipelineContextEnvelope` уже існував; не існували лише їхні canonical saved outputs. Pipeline bundle успішно пройшов наявний адаптер. Sector source пройшов адаптер як заблокований, що дало точну діагностику замість припущення про відсутній код.

Наявний energy macro envelope не використовувався для semiconductor. Для semiconductor profile додано власний macro scope. Новий envelope побудовано з локального point-in-time macro source; усі 9 configured series присутні.

## Готові входи

### Macro

- Artifact: `reports/dean_os/domain_scoped_macro_envelope_current/latest.json`.
- As-of: `2026-06-30T21:00:00+00:00`.
- Scope: INDPRO, DGORDER, PPIACO, FEDFUNDS, DGS10, T10Y2Y, BAMLH0A0HYM2, VIXCLS, DEXCHUS.
- Structural blockers: none.
- Recursive loader повторно будує offline macro core та звіряє source/registry/dispatch SHA, selected observations і fragment.

### Fundamentals

- Artifact: `reports/dean_os/domain_scoped_fundamentals_envelope_current/latest.json`.
- As-of: `2026-06-30T21:00:00+00:00`.
- Status: `domain_fundamentals_candidate_ready_with_gaps`.
- Прогалини: 4/12 profile issuer coverage, неповна cohort comparability, pending issuer-registry acceptance.

### News

- Artifact: `reports/dean_os/domain_scoped_news_envelope_current/latest.json`.
- As-of: `2026-06-30T21:00:00+00:00`.
- Status: `domain_news_candidate_ready_with_gaps`.
- Прогалини: відсутня news lane `policy_or_geopolitical`; news залишається trigger-only.

### Official policy

- Artifact: `reports/dean_os/domain_scoped_official_policy_envelope_current/latest.json`.
- As-of: `2026-06-30T21:00:00+00:00`.
- Status: `domain_official_policy_candidate_ready_with_gaps`.
- Прогалина: official-source registry pending operator acceptance.

### Pipeline context

- Artifact: `reports/dean_os/domain_scoped_pipeline_context_envelope_current/latest.json`.
- Envelope cutoff: `2026-07-10T19:50:45.683169+00:00`.
- Status: `domain_pipeline_context_candidate_ready`.
- Tickers: ASML, MU, NVDA, TSM — усі належать semiconductor universe.
- Coverage: 3/3 timeframe lanes; 12/12 declared lineage references verified.
- Dry-run state transition: ready, not recorded.

## Єдиний заблокований вхід: sector_market

- Envelope report: `reports/dean_os/domain_scoped_sector_market_envelope_current/latest.json`.
- Source містить AMD, INTC, NVDA, TSM і benchmark QQQ.
- Domain contract вимагає 12 тикерів: NVDA, AMD, TSM, ASML, AMAT, LRCX, KLAC, AVGO, MU, ARM, INTC, QCOM та benchmark SOXX.
- Blockers: `sector_market_universe_mismatch`, `sector_market_benchmark_mismatch`.

Наявні raw/repaired price artifacts також не містять повного 12-ticker universe і SOXX. Контракт не послаблено, source не перелабельовано, synthetic market data не створено. Потрібен новий справжній point-in-time price artifact із повною конфігурованою когортою та SOXX.

## Контрактна та orchestration перевірка

Binding policy тепер вимагає domain-envelope contracts для macro, sector_market і pipeline_context, а не їхні legacy producer contracts. Planner бачить 5 валідних кандидатів і одну collection proposal для sector_market.

State machine підтримує дві безпечні macro routes: повний collection lifecycle та прямий reuse уже перевіреного локального candidate. Обидві дороги монотонні; один виклик виконує не більше одного stage. Macro і pipeline dry-run transitions готові, але не записані. Canonical transition ledger лишається порожнім.

Перевірки після змін: **59 passed**.
