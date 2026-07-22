# Sector Market Acquisition Compatibility Audit

## Поточний висновок

Кодовий ланцюг для одного bounded `sector_market` збору тепер підготовлений і перевірений офлайн. Реальні дані ще не повні: наявний clean snapshot містить лише `ASML`, `MU`, `NVDA`, `TSM`, тобто **4 з 13** потрібних market identities (12 секторних тикерів плюс `SOXX`).

Тому система правильно залишається в стані очікування. Recommendation/analyst branch не запускається, а repair відхиляє поточний coverage artifact.

## Результат перевірки змін Gemini

Корисні зміни:

- coverage producer тепер розділяє один файл за фактичним `interval` і не змішує 15m/60m/1d;
- відновлені файли двох root CLI;
- тести mixed cadence і cross-ticker contamination були корисними.

Знайдені проблеми:

- обидва CLI імпортували неіснуючі compatibility modules і не запускали навіть `--help`;
- coverage використовував legacy `default_volatile` preset із 18 іншими тикерами, а не domain profile;
- clean snapshot manifest, Parquet і SHA lineage не перевірялися;
- repair приймав довільний непідписаний coverage JSON;
- `effective_start=None` дозволяв використати всю історію і фактично обходив point-in-time cutoff;
- відсутні clean-snapshot і saved-sector producer CLI;
- твердження «всі перешкоди усунуто» не відповідало фактичним контрактам.

## Що виправлено

1. `load_verified_clean_yahoo_market_snapshot` повторно перевіряє manifest contract, safety flags, immutable Parquet, dataframe fingerprint, file SHA-256, ticker/timeframe inventory і lanes.
2. `PipelineControlSavedDataCoverage` отримав канонічний contract, source hashes та явний ticker override.
3. Створено `DomainSectorMarketCoverageBridge`, який бере universe і benchmark лише з domain lifecycle profile.
4. Bridge вимагає точний набір 12 тикерів плюс `SOXX`, повний eligible 15m scope і ненульовий `effective_start`.
5. `PipelineControlSavedPriceRepair` для domain path приймає лише рекурсивно перевірений bridge artifact, звіряє coverage/source SHA та відхиляє null/NaT cutoff.
6. Виправлені root CLI imports. Додані bounded CLI для clean snapshot і saved sector-market evidence.
7. `DomainContextSet` і універсальний `DomainOrchestrator` інтегровані: incomplete 5/6 packet переходить у `domain_orchestrator_waiting_for_context_families`, а analyst/pipeline branches не запускаються.

## Канонічний ланцюг

```text
CleanYahooMarketSnapshot (network, explicit bounded run)
  -> verified immutable manifest + Parquet
  -> DomainSectorMarketCoverageBridge (offline, exact 12 + SOXX)
  -> PipelineControlSavedPriceRepair (offline, 15m -> 60m/1d)
  -> SavedSectorMarketEvidenceProducer (offline)
  -> DomainScopedSectorMarketEnvelope (offline)
  -> DomainContextSet rebuild (offline)
  -> DomainOrchestrator context gate
```

## Реальний current result

- source snapshot verified: `true`;
- required tickers: `13`;
- snapshot tickers: `4`;
- eligible 15m tickers: `4`;
- coverage status: `domain_sector_market_coverage_blocked`;
- blockers: `domain_market_ticker_scope_mismatch`, `domain_15m_eligible_scope_incomplete`;
- repair run: rejected before artifact creation;
- current-turn network access: `false`;
- analyst invocation: `false`;
- learning/trading: `false`.

Canonical review: `reports/dean_os/domain_sector_market_coverage_bridge_current/latest.md`.

## Наступне рішення

Щоб перейти з 5/6 до 6/6, потрібен один окремо дозволений bounded Yahoo run для exact domain scope. Для економії ресурсів достатньо збирати тільки native `15m`: чинний repair детерміновано виводить 60m та 1d, а sector evidence потребує 20 common sessions, не 180 native daily rows.

До такого дозволу рекомендації та domain analyst залишаються закритими. Синтетика, relabel старих чотирьох тикерів або послаблення benchmark/universe contract заборонені.
