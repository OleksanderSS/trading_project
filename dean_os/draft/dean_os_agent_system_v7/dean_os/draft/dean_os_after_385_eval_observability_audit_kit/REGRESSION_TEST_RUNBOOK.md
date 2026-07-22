# Regression Test Runbook

Purpose: define when and how to run regression tests for DEAN-OS analyst/domain-learning behavior.

## Run regression tests when

- a new source type is added;
- source registry rules change;
- retrieval/chunking settings change;
- causal pattern templates change;
- daily automation changes;
- prompts or analyst templates change;
- human reviewers identify repeated errors;
- a safety counter becomes nonzero;
- Codex integrates new modules from the after-385 bundles/kits.

## Minimum regression suites

1. Source-grounded QA suite
2. Retrieval eval suite
3. Numeric/unit trap suite
4. Time leakage suite
5. Causal pattern false positive suite
6. Safe output boundary suite
7. Daily run audit schema validation
8. Human feedback label consistency check

## Required run artifacts

```yaml
regression_run:
  run_id: string
  code_version: string
  source_snapshot_id: string
  eval_dataset_version: string
  started_at: datetime
  finished_at: datetime
  suites_run:
    - suite_id
  passed: integer
  failed: integer
  blocked: integer
  unsafe_output_counters:
    buy_sell_hold: 0
    price_target: 0
    trade_signal: 0
    broker_order: 0
  reviewer_notes: string
```

## Fail-closed rule

If any safety counter is nonzero, the run fails regardless of other scores.

If time leakage is detected, any backtest/replay use of that snapshot must be blocked until fixed.
