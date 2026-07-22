# CODEX_REPO_RECONCILIATION_CHECKLIST_AFTER_385

Use this checklist because earlier context up to around block 135 may already have been given to Codex or partly integrated.

## Step 1 — Inventory current repo

- [ ] Identify existing agent modules.
- [ ] Identify existing pipeline/backtest/model modules.
- [ ] Identify existing orchestration/task-routing code.
- [ ] Identify existing fixtures and tests.
- [ ] Identify existing config mutation paths.
- [ ] Identify existing safety guards for no trading/no broker/no live fetch.
- [ ] Identify any assistant_workbench-derived files already copied into the repo.

## Step 2 — Classify existing work

For each existing related component, classify:

- [ ] Keep as-is.
- [ ] Keep but adapt to after-385 lifecycle.
- [ ] Replace with cleaner interface.
- [ ] Move to docs/reference only.
- [ ] Remove only if clearly obsolete and covered by tests.

Do not delete useful older work mechanically.

## Step 3 — Compare against after-385

- [ ] Analyst Branch coverage: intake, evidence quality, lenses, sector binding, review packet, review queue, feedback backlog.
- [ ] Pipeline Controller coverage: data checks, feature windows, split governance, model comparison, backtest hygiene, drift/overfit monitors, safe proposal artifacts.
- [ ] Orchestrator coverage: branch routing, dependency ordering, review states, incident/blocker states, cross-branch joins.
- [ ] Safety coverage: no live fetch/API/trading/config mutation/runtime execution by default.
- [ ] Fixtures coverage: deterministic offline fixtures for each branch and orchestrator.
- [ ] Tests coverage: unit tests for boundaries and forbidden operations.

## Step 4 — Integration sequence

- [ ] Add/adjust interfaces first.
- [ ] Add deterministic fixtures.
- [ ] Add safety/contract tests.
- [ ] Implement minimal Analyst vertical slice.
- [ ] Implement minimal Pipeline Controller vertical slice.
- [ ] Implement minimal Orchestrator routing/state slice.
- [ ] Add observability counters/structured logs for blocked/review states.
- [ ] Run offline verification commands.
- [ ] Document harvested/adapted/skipped items.

## Step 5 — Safety verification

Tests should prove, by default:

- [ ] No live fetch.
- [ ] No external API calls.
- [ ] No source retrieval.
- [ ] No broker/order routing.
- [ ] No autonomous trading.
- [ ] No model training execution.
- [ ] No backtest execution.
- [ ] No production config writes.
- [ ] No runtime parameter mutation.
- [ ] No model promotion.
- [ ] No analyst thesis/valuation/recommendation/trading output.
- [ ] No autonomous orchestration loop.

## Step 6 — Output expected from Codex

Codex should produce one of these:

1. Practical adapted repo changes with tests and fixtures; or
2. A concrete integration issue list if implementation is not possible immediately.

Codex should not produce another assistant_workbench numbered block.
