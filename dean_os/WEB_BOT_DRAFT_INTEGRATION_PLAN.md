# Інтеграція напрацювань веб-версії бота (draft/dean_os_after_245)

Цей документ фіксує статус та плани щодо інтеграції коду, згенерованого ботом в веб-версії ізольовано (на базі `Agents_architecture.md`), який дійшов до блоку 245.

## Поточний стан (Where we are after block 245)

* Останній реалізований блок: `245_review_only_real_source_normalized_packet_fixture_v1`.
* Наявний пайплайн: `assets universe -> real source intake scaffold -> intake normalizer contract -> normalized packet fixture`.
* В блоці 245 створено `normalized packet fixtures` для всіх 12 типів джерел. 
* **Важливе зауваження:** Це лише `review-only` та `fixture-only` артефакти. Вони перевіряють форму майбутніх результатів нормалізації реальних джерел, але поки що не є реальними доказами і не здійснюють екстракцію сутностей (claims/events/entities). Фікстури потрібні для CI/тестування, а реальні дані будуть заходити згодом (через upload, paste, API, connector reference тощо) і нормалізовуватись за тією ж схемою.

## Стратегія інтеграції

Оскільки веб-бот працював ізольовано, його напрацювання (починаючи з блоку 121 і до 245) необхідно інтегрувати обережно і вибірково.

**Правила інтеграції (згідно CODEX_BRIDGE):**
1. Всі блоки від 121 до 245 розглядаються як `staged/review-only` матеріали (`assistant_workbench`).
2. Фікстури ніколи не повинні трактуватися як `production evidence`.
3. Інтеграцію слід починати з **контрактів та тестів** (contracts and tests), а не з кінцевих результатів виконання.
4. Всі `review-only` safety flags повинні зберігатись.

## Найвищий пріоритет для інтеграції (Highest-priority target)

* Побудова **інтерфейсу для прийому та нормалізації даних з реальних джерел** (real-source intake/normalization interface). 
* Цей інтерфейс має приймати дані від оператора (файли, текст, connector references, API snapshots) і видавати нормалізовані пакети із збереженням: provenance, хешів, anchors, quarantine partitions, quality precheck, та candidate routing.

## Подальші кроки (Future Sequence)

Рекомендована послідовність наступних блоків для інтеграції/реалізації:

1. **Блок 246:** `246_review_only_real_source_normalized_packet_validation_gate_v1`
   * *Мета:* Валідація фікстур нормалізованих пакетів із блоку 245. Без екстракції фактів (claims/events/entities).
2. **Блок 247:** `247_review_only_real_source_claim_event_entity_extraction_contract_v1`
   * *Мета:* Створення контракту екстракції для claims, events, entities, topics, sectors, assets, та financial implication candidates.
3. **Блок 248:** `248_review_only_real_source_claim_event_entity_extraction_fixture_v1`
   * *Мета:* Екстракція виключно на базі фікстур (over normalized packet fixtures).

## Суворі заборони (Forbidden)

Під час цієї інтеграції суворо забороняється впроваджувати:
* Автономний fetch (завантаження) даних наживо.
* Генерацію рекомендацій або company thesis.
* Інтерпретацію коефіцієнтів (ratio interpretation) або оцінку (valuation).
* Розрахунок price targets.
* Генерацію торгових сигналів (trade signals/outputs), роутинг до брокера або створення ордерів.
