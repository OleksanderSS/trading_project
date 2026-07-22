# Recommended Codex Integration Order After 385

## Phase 0 — Read-only repository map

Goal: map existing repo modules to the kits without changing behavior.

Outputs:

- module-to-kit map;
- current collectors map;
- current pipeline stages map;
- current guards/eval map;
- current trading/execution boundaries map.

No code mutation beyond docs unless explicitly approved.

## Phase 1 — Provenance and manifests

Goal: make the system reproducible.

Harvest from:

- Advanced Governance Kit
- Eval/Audit Kit
- Automation Governance Kit

Add/adapt:

- source manifest;
- run manifest;
- feature snapshot manifest;
- model state manifest;
- decision lineage skeleton.

## Phase 2 — Daily Pipeline Controller governor

Goal: automate simple daily run governance.

Harvest from:

- Automation / Execution Governance Kit v2

Add/adapt:

- daily run state machine;
- collector health;
- source quality;
- dedupe status;
- blocked state;
- daily audit log.

No analyst trading outputs.

## Phase 3 — Normalized event packets and routing

Goal: convert raw news/data into structured event inputs.

Harvest from:

- Domain Learning Kit v3
- Macro Regime Kit v3
- Automation Kit v2

Add/adapt:

- normalized event packet schema;
- source quality flags;
- sector/domain routing;
- tag-only analyst output.

## Phase 4 — Macro regime + hypothesis ledger

Goal: implement the user's event-graph learning logic.

Harvest from:

- Macro Regime Kit v3
- Advanced Governance Kit
- Agent Memory Kit

Add/adapt:

- macro regime snapshots;
- risk archetypes;
- expectation gap;
- open hypothesis tokens;
- causal graph;
- scenario tree;
- watch metrics;
- outcome review.

## Phase 5 — Operator review and feedback learning

Goal: make review a learning surface.

Harvest from:

- Operator Review / Synchronous Learning Kit
- Agent Memory Kit
- Eval/Audit Kit

Add/adapt:

- daily operator report;
- review session state;
- feedback labels;
- report-to-training examples;
- learning candidates.

## Phase 6 — Eval / observability / stress

Goal: test reasoning and system behavior.

Harvest from:

- Eval/Audit Kit
- Scenario Stress Testing Kit
- Advanced Governance Kit

Add/adapt:

- grounding eval;
- leakage eval;
- unit/period traps;
- stress scenario seeds;
- incident/postmortem conversion.

## Phase 7 — Strategy/playbook layer

Goal: define strategies before trading simulation.

Harvest from:

- Strategy Library / Playbook Governance Kit
- Portfolio Governance section in Advanced Governance Kit

Add/adapt:

- strategy registry;
- input contracts;
- regime compatibility;
- eval requirements;
- promotion gates;
- block/deprecation policy.

## Phase 8 — Replay → paper → shadow → supervised

Goal: only after prior phases.

Harvest from:

- Automation Execution Governance Kit v2
- Strategy Library Kit
- Scenario Stress Kit

Add/adapt:

- replay gate;
- paper gate;
- shadow gate;
- supervised live gate;
- risk limits;
- kill switch;
- execution gateway.

## Explicitly out of first integration

- unrestricted autonomous execution;
- direct broker order integration;
- LLM-generated trading recommendations;
- production strategy promotion without gates.
