# CODEX_INTEGRATION_BRIEF_AFTER_385

Use `dean_os_after_385_full_context_bundle.zip` as the most complete staged design inventory for DEAN-OS / assistant_workbench.

This is not a production patch and not a blind merge instruction. The goal is to inspect, harvest, adapt, and integrate useful pieces into the current project.

## Canonical stance

- after-385 is the latest complete staged blueprint/design inventory.
- Earlier context up to around block 135 may still contain useful already-integrated work.
- Do not discard existing repo work mechanically.
- Reconcile current repo state against after-385 and prefer the latest lifecycle guidance where there is conflict.
- Treat staged rows, manifests, fixtures, contracts, and target paths as authoring guidance unless deliberately adapted.

## High-level architecture to harvest

### 1. Analyst Branch

Blocks: `376-378`.

Harvestable ideas:

- request intake boundary;
- evidence pack selection;
- evidence quality / quarantine / blocked states;
- macro, history, policy, geopolitics, sector, fundamentals lenses;
- sector profile binding;
- scenario/risk framing;
- review packet abstraction;
- human review queue;
- feedback backlog.

Do not generate production analyst reports, thesis, valuation, recommendation, price target, buy/sell/hold labels, or trading instructions.

### 2. Pipeline Controller Branch

Blocks: `379-381`.

Harvestable ideas:

- data availability checks;
- data quality blockers;
- feature window policy;
- train/test split governance;
- walk-forward policy;
- model candidate registry;
- model comparison plan;
- backtest hygiene validators;
- drift/overfit monitors;
- safe parameter proposal artifact;
- human review gate;
- rollback/monitoring feedback loop.

Do not execute model training, backtests, production config writes, runtime parameter mutation, model promotion, or trading.

### 3. Orchestrator

Blocks: `382-384`.

Harvestable ideas:

- branch routing between Analyst and Pipeline Controller;
- dependency ordering;
- review queues;
- state machine for pending/ready/blocked/reviewed/rejected/escalated;
- incident/blocker routing;
- cross-branch joins;
- final human decision boundary.

Do not implement autonomous execution loops by default. Orchestrator should coordinate tasks and review states, not trade or run live pipelines.

## Recommended implementation order

1. Repo inventory: identify existing agents, pipeline modules, configs, fixtures, tests, and safety boundaries.
2. Interfaces/contracts: add practical interfaces for Analyst Branch, Pipeline Controller Branch, and Orchestrator.
3. Offline fixtures: create small deterministic fixtures derived from after-385 ideas.
4. Unit tests: prove no live fetch, no external APIs, no trading, no production writes by default.
5. Analyst vertical slice: evidence -> lenses -> review packet.
6. Pipeline Controller vertical slice: data checks -> split/window policy -> proposal artifact.
7. Orchestrator vertical slice: route tasks, track dependencies, emit review queue items.
8. Observability hooks: queue counts, blocked states, validation status, drift/overfit flags.
9. Smoke commands: offline deterministic verification.
10. Integration notes: document what was harvested, rewritten, skipped, or deferred.

## Stop rule

Do not create block 386 or additional numbered assistant_workbench lifecycle templates. Next work should be repo integration or a concrete integration issue list.
