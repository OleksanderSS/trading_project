# CODEX_HARVEST_PRIORITY_MATRIX_AFTER_385

Use this matrix to decide what to harvest from `dean_os_after_385_full_context_bundle.zip`.

| Area | Priority | Harvest | Suggested repo target | Integration risk | Notes |
|---|---:|---|---|---|---|
| Analyst Branch interfaces | High | Evidence intake, lens runner, sector binding, review packet contracts | `dean_os/analyst/`, existing agent modules, or closest current structure | Medium | Implement as small interfaces before runtime wiring. |
| Evidence quality / blocked states | High | Quality scoring, quarantine, source trust, blocked escalation | `dean_os/evidence/`, validation modules, fixtures | Low-Medium | Highly reusable and testable offline. |
| Sector profile adapters | High | Domain profiles for semiconductors, tech, energy, healthcare, financials | `dean_os/sectors/` or profile registry | Medium | Avoid duplicating Analyst Core per sector. Profiles/adapters should bind to shared core. |
| Analyst review queue | High | Review packet, approval/rejection/defer states, feedback backlog | `dean_os/review/` or queue abstractions | Low | Should not produce production reports automatically. |
| Pipeline Controller interfaces | High | Data check, feature-window proposal, split governance, model comparison plan | `dean_os/pipeline_controller/` or pipeline governance module | Medium | Keep proposal artifacts separate from runtime config mutation. |
| Backtest hygiene checks | High | Leakage checks, walk-forward policy, overfit guard, benchmark comparison plan | existing backtest/testing modules | Medium | Validators can be offline/unit-tested; do not run real backtests by default. |
| Drift / overfit monitors | High | Monitor contract, alert row, review-state mapping | monitoring/observability modules | Medium | Start with review-only signals and fixtures. |
| Safe parameter proposals | High | Reviewable proposal object, rationale, constraints, rollback notes | config governance or pipeline controller module | Medium | Never mutate production parameters directly. |
| Orchestrator routing | High | Branch routing, dependency ordering, review queue routing | `dean_os/orchestrator/` or task routing module | Medium-High | Must not become autonomous runtime executor by default. |
| Blocked / incident state model | High | Pending/blocked/escalated/rejected/approved states | shared state machine or review module | Low-Medium | Useful safety primitive across branches. |
| Offline fixtures | High | Analyst, pipeline, orchestrator fixture examples | `tests/fixtures/` | Low | Small deterministic fixtures should come first. |
| Offline tests | High | No live fetch/API/trading/config-write tests | `tests/` | Low | Required safety coverage. |
| Verification commands | Medium-High | py_compile, pytest, smoke validation commands | docs / CI notes | Low | Keep deterministic and offline. |
| Observability hooks | Medium-High | queue counts, blocked counts, validation results, drift flags | monitoring/logging modules | Medium | Review-only metrics first. |
| Target file inventory from 375 | Medium | Candidate target paths/module boundaries | repo planning only | Medium | Use as hints, not mandatory paths. |
| Old assistant_workbench manifests | Low | Historical traceability | docs only | Low | Do not copy into production. |
| Numbered validation ladder after 385 | None | Nothing | none | High if continued | Stop. Do not create more template/gate/checkpoint blocks. |
| Staged metadata rows as production models | Low | Only if rewritten deliberately | selected modules | High | Avoid one-to-one production copy. |
| Patch/package artifacts | None | Nothing | none | High | after-385 is not a patch package for blind application. |

## Priority rule

Start with small vertical slices that are easy to test offline:

1. Analyst evidence-quality/review packet slice.
2. Pipeline data/split/window/proposal slice.
3. Orchestrator routing/state-machine slice.
4. Safety tests proving forbidden operations remain impossible by default.
