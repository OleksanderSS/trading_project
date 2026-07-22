# Build Session — System Context

> Цей файл — єдине джерело контексту для AI-агента. Оновлювати при зміні фокусу.

---

## 1. Що ми будуємо

**Dean OS** — review-only financial analysis platform. Аналізує ринкові дані (news, macro, fundamentals, price), будує evidence-backed thesis, класифікує події, оцінює materiality/sentiment/directness.

**Key constraint**: review-only — ніколи не створює trade signals, не виконує угоди, не дає buy/sell рекомендацій. Вся аналітика проходить через human review.

**Tech stack**: Python 3.12, Pydantic v2, YAML config, rule-based (no LLM calls in core pipeline).

---

## 2. Архітектура

```
config/
├── domain_profiles/          ← YAML-конфіги доменів
│   ├── <id>.yaml             ← profile (horizons, keywords, tickers, rules)
│   └── <id>/evidence_keywords.yaml  ← keyword map
└── agent_registry.yaml       ← Реєстрація всіх агентів (BaseAgent)

dean_os/
├── analyst_core/             ← Фреймворк (lenses, orchestrator, contracts)
│   ├── lenses/               ← Аналітичні модулі (event_classifier, transmission_mapper, ...)
│   ├── lens_contract.py      ← AnalystLens ABC, LensRegistry, ModuleDelta
│   ├── lens_orchestrator.py  ← Запускає лензи послідовно
│   ├── sector_analyst.py     ← Збирає SectorAnalyst з профайла
│   ├── pipeline_manager.py   ← SectorPipelineManager (discover, run, save)
│   ├── domain_analyst_runtime.py ← DomainAnalystRuntime (обгортка над SectorAnalyst)
│   └── artifact_evidence_loader.py
├── analysts/                 ← Доменна логіка
│   ├── profiles.py           ← Завантажує YAML-профайли
│   ├── schemas.py            ← DomainProfile, AnalystEvidenceItem, BaseAnalystAgent
│   ├── context_adapter.py    ← MarketContext → AnalystEvidenceItem (keyword-based)
│   └── base.py               ← BaseAnalystAgent (не плутати з BaseAgent!)
├── agents/                   ← Виконавчі агенти (BaseAgent subclasses)
│   ├── domain_analyst.py     ← DomainAnalystAgent (SectorAnalyst wrapper)
│   ├── pipeline_manager.py   ← PipelineManagerAgent (SectorPipelineManager wrapper)
│   └── ... (chief_review, data_quality, risk, tuning, etc.)
├── base.py                   ← BaseAgent ABC (async run → PipelineReport)
├── orchestrator.py           ← DEANOrchestrator (registry → branches → consensus)
├── registry.py               ← AgentRegistry (YAML → BaseAgent instances)
├── schemas.py                ← MarketContext, PipelineReport, ConsensusDecision, etc.
├── saved_*.py                ← Legacy "saved producers"
└── semiconductor_analyst_runtime.py
```

### Key data flow:
```
MarketContext
  → MarketContextEvidenceAdapter (keyword classification, evidence extraction)
  → AnalysisPacket (event_records + evidence)
  → LensOrchestrator (runs registered lenses with domain config)
    → EventClassifierLens (classify events, detect sectors/directness/sentiment)
    → Other lenses (transmission mapper, historical analog, etc.)
  → ModuleDelta (what each lens contributed)
  → SectorAnalyst (aggregates, produces report)
```

### Domain config now comes from:
1. `config/domain_profiles/<id>.yaml` — profile (horizons, keywords, tickers, rules)
2. `config/domain_profiles/<id>/evidence_keywords.yaml` — keyword map for context adapter

---

## 3. Ключові патерни та конвенції

- **Lenses** — модульні аналітичні юніти. Кожен extends `AnalystLens`, реалізує `analyze(packet, config) → ModuleDelta`. Конфіг = словник, який передає LensOrchestrator.
- **DomainProfile** — Pydantic модель. Нові поля додаються в `schemas.py`. Профайли зберігаються в YAML, завантажуються через `get_domain_profile(id)`.
- **BaseAgent** (base.py) — ABC для всіх агентів. Контракт: `async run(context: MarketContext) → PipelineReport`. Реєстрація через YAML в `agent_registry.yaml`, завантаження через `AgentRegistry`.
- **DomainAnalystAgent** (agents/domain_analyst.py) — extends BaseAgent. Обгортка над `SectorAnalyst`. Приймає домен через config (`domain_id`). Можна створити екземпляр для будь-якого сектора: змінити `domain_id` в YAML.
- **PipelineManagerAgent** (agents/pipeline_manager.py) — extends BaseAgent. Обгортка над `SectorPipelineManager`. Керує: discover artifacts → run analysis → save report.
- **DEANOrchestrator** (orchestrator.py) — топ-оркестратор. Завантажує агенти з registry за branch, виконує pre-pipeline → pipeline → post-pipeline, збирає ConsensusDecision.
- **Test framework**: pytest. Команда: `python -m pytest tests/...`
- **Стиль коду**: жодних коментарів (крім docstring для публічних методів), зрозумілі імена змінних, type hints
- **Мови**: комунікація — українська, код — англійська, змінні/назви — snake_case

---

## 4. Поточний статус (03 Jul 2026)

### Що зроблено
- **P0-P2 complete**: всі хардкодні значення винесено в YAML-конфіг.
  - profiles.py → YAML
  - context_adapter _DOMAIN_EVIDENCE_KEYWORDS → YAML
  - event_classifier _DOMAIN_SECTOR_KEYWORDS → profile config-driven
  - 7 файлів DEFAULT_TICKERS → profile-driven
  - domain_id тепер обов'язковий скрізь
  - 2 hardcoded if-branch виправлено на config fields
- **P3: Система агентів** (04 Jul 2026):
  - `agents/domain_analyst.py` — DomainAnalystAgent (BaseAgent wrapper над SectorAnalyst)
  - `agents/pipeline_manager.py` — PipelineManagerAgent (BaseAgent wrapper над SectorPipelineManager)
  - Зареєстровано в `agent_registry.yaml` (3 domain analysts + pipeline manager, disabled)
  - Всі агенти проходять через DEANOrchestrator + AgentRegistry

### Що будуємо далі (що скаже користувач)
- Нові лензи, доробка існуючих, тести
- Нові домени, pipeline agents
- Інтеграція DomainAnalystAgent в реальний пайплайн (enable в agent_registry.yaml)

---

## 5. Шпаргалка

- `get_domain_profile(id)` — завантажити профайл
- `MarketContextEvidenceAdapter(domain_id)` — адаптер контексту
- `profiles.py` imports YAML from `config/domain_profiles/*.yaml`
- `context_adapter.py` imports YAML from `config/domain_profiles/<id>/evidence_keywords.yaml`
- Синтаксис: `python -m pytest path/to/test.py -x -q`

---

## 6. Останні зміни (щоб не забути)

| Файл | Зміна |
|---|---|
| `config/domain_profiles/*.yaml` | 5 профайлів (semiconductor, macro, liquidity, geopolitics, energy) |
| `config/domain_profiles/*/evidence_keywords.yaml` | 5 файлів keyword maps |
| `profiles.py` | Переписано: завантаження з YAML замість hardcoded Python |
| `context_adapter.py` | `_DOMAIN_EVIDENCE_KEYWORDS` видалено, завантаження з YAML |
| `schemas.py` | Додано `sector_label`, `macro_evidence_type` |
| `event_classifier_lens.py` | config-driven keywords для unknown domain |
| `sector_analyst.py` | Передає profile keywords в lens config |
| `artifact_evidence_loader.py` | domain_id required (no default) |
| `vertical_slice_run.py` | `sector_label or domain_id` замість if-branch |
| `semiconductor_analyst_runtime.py` | tickers з профайла, relaxed validation |
| `saved_sector_market_evidence_producer.py` | tickers з профайла |
| `saved_ticker_specific_evidence_producer.py` | tickers з профайла |
| `saved_semiconductor_news_evidence_producer.py` | tickers з профайла |
| `saved_official_policy_evidence_producer.py` | tickers з профайла |
| `agents/domain_analyst.py` | Новий: DomainAnalystAgent (BaseAgent, обгортка SectorAnalyst) |
| `agents/pipeline_manager.py` | Новий: PipelineManagerAgent (BaseAgent, обгортка SectorPipelineManager) |
| `agents/__init__.py` | Експорт DomainAnalystAgent, PipelineManagerAgent |
| `config/agent_registry.yaml` | 4 нових agent entry (semiconductor, energy, macro analysts + pipeline manager) |
| `.agents/build_session.md` | Оновлено архітектуру, додано agent layer |
