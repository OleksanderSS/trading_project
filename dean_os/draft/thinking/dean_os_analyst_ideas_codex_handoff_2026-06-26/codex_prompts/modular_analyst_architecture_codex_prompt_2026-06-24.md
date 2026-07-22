# Codex Integration Prompt — Modular Analyst Architecture

Use this package as a design supplement for DEAN-OS.

Goal:
Convert the analyst layer into a modular workbench where news/text analysis is extended through pluggable modules rather than one monolithic prompt or predictor.

Implement first:
1. SourcePacket schema.
2. AnalysisPacket schema.
3. ModuleDelta schema.
4. RegimeContextVector schema.
5. ScenarioOutcomeGraph schema.
6. Module registry.
7. Orchestrator that routes packets and merges module deltas.
8. Review-only analyst report builder.

Strict boundaries:
- no live trading;
- no buy/sell/hold output;
- no position sizing;
- no broker routing;
- no autonomous execution;
- no production price targets.

Design rule:
Every module must:
- declare input/output contracts;
- write only deltas;
- attach evidence IDs;
- expose confidence and evidence gaps;
- preserve as_of_date;
- be testable independently.
