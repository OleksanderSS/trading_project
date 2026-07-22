# Package Index

## 00_README_FOR_CODEX.md
Primary instructions for Codex.

## source_notes/
Human-readable design notes and journal files.

Recommended reading order:

1. `01_analyst_journal_latest_2026-06-26.md`
2. `02_daily_briefing_notes_2026-06-26.md`
3. `03_regime_context_scenario_graph_note_2026-06-24.md`
4. `04_modular_analyst_architecture_2026-06-24.md`
5. `05_pipeline_control_and_domain_analyst_branches_2026-06-24.md`
6. `06_data_pipeline_vs_agentic_analyst_layer_2026-06-24.md`
7. `07_additional_analyst_observations_2026-06-24.md`

## specs/
Machine-readable JSON specs extracted from the notes.

## codex_prompts/
Focused prompts that can be given to Codex as implementation tickets.

---

## Minimum viable integration target

Do not try to implement all notes at once.

First implementation target:

```text
1. RegimeContextVector schema
2. ScenarioOutcomeGraph schema
3. EvidenceGap schema
4. HypothesisLedgerEntry schema
5. HistoricalOutcomeCheck schema
6. DomainAnalystReport extension fields
7. validation tests
8. review-only analyst report extension
```

Then advanced modules can be added incrementally.
