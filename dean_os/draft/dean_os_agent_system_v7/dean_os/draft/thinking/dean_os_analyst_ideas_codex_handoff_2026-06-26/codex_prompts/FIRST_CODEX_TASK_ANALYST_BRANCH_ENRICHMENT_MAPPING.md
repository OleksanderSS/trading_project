# First Codex Task — Analyst Branch Enrichment Mapping

Task:
Inspect the existing DEAN-OS Analyst Branch and map this handoff package to the current codebase.

Output required:
1. Current Analyst Branch files/contracts found.
2. Existing report/output schema.
3. Where each new concept should integrate:
   - RegimeContextVector
   - ScenarioOutcomeGraph
   - EvidenceGap
   - HypothesisLedgerEntry
   - HistoricalOutcomeCheck
   - DomainAnalystReport extension fields
4. Missing schemas/modules/tests.
5. Proposed smallest safe implementation tickets.

Do not:
- rewrite the analyst branch;
- add live fetch;
- add external API calls;
- add trading outputs;
- modify production config;
- promote models;
- implement autonomous execution.

This first task is review/mapping only.
