# DEAN-OS after-385 Codex Integration Kit

This kit is a compact human/Codex-facing companion for:

`dean_os_after_385_full_context_bundle.zip`

Bundle SHA-256:

`22d204a363d1e2a243f5b17641a176b55ae41d5f90d46ff153a3996fb80ca06d`

Bundle size bytes: `56913887`

Generated: `2026-06-20T21:24:32Z`

## Purpose

The after-385 bundle is a staged design inventory, not a blind merge patch. It contains architecture ideas, contracts, lifecycle templates, tests/fixture ideas, safety guards, and integration guidance for DEAN-OS.

This kit explains how Codex should harvest useful artifacts from that bundle and adapt them to the current main repository.

## Closed staged inventory

- `375_review_only_dean_os_final_codex_ready_authoring_materialization_handoff_bundle_v1` — target/module/test/fixture/verification/guard inventory.
- `376-378` — Analyst Branch full-cycle template, closed.
- `379-381` — Pipeline Controller Branch full-cycle template, closed.
- `382-384` — Orchestrator full-cycle template, closed.
- `385` — final Codex harvest/adapt integration brief.

## What Codex should do

1. Inspect the current repo first.
2. Use after-385 as a design inventory and idea bank.
3. Preserve useful work already integrated from earlier context, including the old ~135 bundle.
4. Prefer the after-385 lifecycle templates when deciding workflow architecture.
5. Cherry-pick/adapt/rewrite useful artifacts into the current repo structure.
6. Add offline tests and fixtures before runtime wiring.
7. Keep human-review boundaries and no-autonomous-trading constraints explicit.

## What Codex should not do

- Do not copy the assistant_workbench tree into production.
- Do not blind-merge staged metadata.
- Do not continue the numbered assistant_workbench ladder after 385.
- Do not run live fetches, external APIs, model training, backtests, broker/order routing, or trading flows during integration tests.
- Do not emit analyst thesis/valuation/recommendation/trading outputs from these templates.

## Files in this kit

- `CODEX_INTEGRATION_BRIEF_AFTER_385.md`
- `CODEX_HARVEST_PRIORITY_MATRIX_AFTER_385.md`
- `CODEX_REPO_RECONCILIATION_CHECKLIST_AFTER_385.md`
- `CODEX_PROMPT_AFTER_385.txt`
- `MANIFEST_AFTER_385_CODEX_INTEGRATION_KIT.json`
