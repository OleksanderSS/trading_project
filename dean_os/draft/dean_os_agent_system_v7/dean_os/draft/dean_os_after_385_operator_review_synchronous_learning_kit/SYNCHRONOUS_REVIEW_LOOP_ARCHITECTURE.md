# Synchronous Review Loop Architecture

## Objective

The user wants the system to produce reports and analyze in parallel, then learn from the comparison
between agent reasoning and human/operator review.

## Correct flow

```text
daily run
→ agent produces report
→ operator reviews synchronously
→ operator labels/corrects
→ agent stores corrected record
→ learning candidate created
→ eval/regression candidate created
→ pattern memory update waits for approval
```

## Why this matters

A static review queue is weak. It only says "approved/rejected".

A synchronous review loop captures:

- what the agent thought;
- what evidence it used;
- where it was uncertain;
- what the operator corrected;
- why the correction was made;
- what future behavior should change.

## Two outputs from every review session

1. Human-readable report  
   For the operator/user.

2. Machine-readable learning record  
   For future agent improvement and eval tests.

## Recommended sync modes

### Mode A — daily digest review

The agent produces a daily report. The operator marks key issues and corrections.

### Mode B — event deep-dive review

For major laws, shocks, earnings, policy, geopolitics, or regime-changing events.

### Mode C — pipeline health review

Pipeline Controller explains collector health, data quality, leakage checks, model status, blocked stages.

### Mode D — hypothesis outcome review

The agent compares expected scenario vs actual outcome and asks for correction/approval.

### Mode E — paper/shadow review

Later: compare hypothetical decisions/orders with risk/execution logs before live escalation.
