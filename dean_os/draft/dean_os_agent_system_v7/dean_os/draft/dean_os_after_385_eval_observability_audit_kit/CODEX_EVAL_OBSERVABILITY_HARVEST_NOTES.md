# Codex Eval / Observability / Audit Harvest Notes

Use this kit as a practical template library. Do not blind-merge it.

## High-priority harvest targets

1. Daily run audit log schema
2. Safety counters
3. Source-grounding evals
4. Numeric/unit trap tests
5. Time-leakage tests
6. Causal-pattern false positive tests
7. Agent output quality metrics
8. Human review feedback labels
9. Alerting rules
10. Regression test runbook

## Integration guidance

Codex should adapt these templates to the current repo structure and existing test stack.

Possible implementation areas:

- tests/evals/
- tests/fixtures/
- dean_os/observability/
- dean_os/evidence/
- dean_os/domain_learning/
- dean_os/orchestration/
- docs/dean_os/
- configs/eval/

## Important

This kit does not authorize:
- live trading;
- paper trading;
- broker calls;
- autonomous recommendations;
- production config mutation;
- model promotion without review.

It supports measuring quality of data accumulation, retrieval, analyst reasoning, causal patterns,
and review-only outputs.
