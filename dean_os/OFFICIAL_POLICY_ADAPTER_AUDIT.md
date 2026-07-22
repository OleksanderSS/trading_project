# Аудит адаптера `official_policy`

Дата актуалізації: 2026-07-22. Режим аудиту: офлайн, без API, мережевих запитів, повторного збору даних або виконання торгового контуру.

## Висновок

Legacy producer `dean_os/analysts/_producers/policy.py` залишається semiconductor-specific: він містить фіксований `domain_id`, використовує semiconductor news loader і перевіряє BIS-джерела. Переписувати цей перевірений producer як універсальний ризиковано й непотрібно.

Універсалізацію реалізовано зовнішнім адаптером `DomainScopedOfficialPolicyEnvelope`. Він повторно використовує лише явний збережений policy artifact і не запускає producer. Domain identity, дозволені official hosts та source identities, registry, cutoff і corroborating news задаються та перевіряються на рівні domain profile і envelope.

## Наявні компоненти

- Legacy producer і verified loader: `dean_os/analysts/_producers/policy.py`.
- Compatibility CLI: `run_agent_saved_official_policy_evidence.py`.
- Domain envelope: `dean_os/domain_scoped_official_policy_envelope.py`.
- Domain CLI: `run_agent_domain_scoped_official_policy_envelope.py`.
- Official-source registry: `dean_os/config/official_policy_evidence_registry.yaml`.
- Domain news envelope: `dean_os/domain_scoped_news_envelope.py`.
- State-machine registry: `dean_os/config/context_acquisition_family_registry.json`.

`run_agent_bis_policy_snapshot.py` відсутній. Нову snapshot-команду не створено, оскільки у робочому дереві немає окремого безпечного snapshot producer, до якого вона могла б бути тонкою compatibility-обгорткою.

## Реальна збережена lineage

Поточний policy artifact має контракт `dean_saved_official_policy_evidence_producer_v1` і статус `official_policy_evidence_ready`. Він прив’язаний SHA-256 до:

- BIS policy snapshot;
- immutable raw PDF у `data/dean_os/policy_raw`;
- official-source registry;
- legacy saved semiconductor news artifact.

Official source identity — `us_bureau_industry_security`; фактичний host raw документа — `media.bis.gov`. Corroboration містить одне незалежне news-джерело та official source як окремі типи доказів; official source не зараховується як незалежне news-джерело.

## Розподіл відповідальності

Legacy producer перевіряє власний source/snapshot/news contract і формує policy evidence. Legacy verified loader повторно перевіряє контракт, статус, cutoff, SHA вхідних артефактів, raw document і fingerprint.

`DomainScopedOfficialPolicyEnvelope` додатково перевіряє:

- точний `domain_id` і timezone-aware `as_of`;
- domain-configured official registry, hosts та source identities;
- snapshot → raw PDF lineage і PDF header;
- registry mapping, publication time та freshness;
- verified domain news envelope;
- точний cross-binding між news source envelope і policy corroboration input;
- незалежну news corroboration без подвійного підрахунку official source;
- review-only та no-authority boundaries;
- відповідний dispatch task.

## Семантична межа

Official document може підтвердити факт існування політики та її зміст. Він не створює напрямний market forecast сам по собі, не підтверджує гіпотезу автоматично і не надає права на binding, analyst invocation, learning, training або trading.

News envelope зберігає роль `trigger_evidence_only`. News corroboration не замінює official source, а official source не перетворює новину на підтвердження секторної тези.

## Поточний реальний результат

Semiconductor policy packet пройшов structural verification без blocker’ів. Статус — `domain_official_policy_candidate_ready_with_gaps`, тому що registry має `agent_verified_official_source_review_only`, а не operator acceptance.

State machine пропонує один перехід `idle -> awaiting_binding_decision`, але він не записаний. Binding, journal append, analyst invocation, hypothesis approval, learning і trading залишаються `false`.

Канонічні звіти:

- `reports/dean_os/domain_scoped_official_policy_envelope_current/latest.md`;
- `reports/dean_os/official_policy_semis_binding_review_current/latest.md`;
- `reports/dean_os/official_policy_semis_candidate_binding_plan_current/latest.md`.

## Перевірки

- 7 focused news/policy envelope tests passed.
- 36 broad adapter/planner/dispatcher/profile/state-machine tests passed.
- Тести включають cross-domain/news-lineage rejection, raw PDF tamper rejection та single-stage state transition.
