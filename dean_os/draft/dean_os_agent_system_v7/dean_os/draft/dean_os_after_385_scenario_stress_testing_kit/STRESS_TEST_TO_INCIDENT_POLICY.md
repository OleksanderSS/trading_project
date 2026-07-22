# Stress Test to Incident Policy

Stress tests should create incidents when system behavior violates expected safety behavior.

## Incident triggers

- broker order generated in non-live mode;
- buy/sell/hold output produced by analyst;
- risk limit breach ignored;
- kill switch unavailable;
- strategy runs in forbidden regime;
- time leakage detected;
- missing decision lineage;
- stale data allowed downstream;
- model promotion allowed after failed gate;
- unsupported asset allowed.

## Incident severity

Critical:
- broker/order safety failure;
- risk engine bypass;
- kill switch failure;
- future leakage in replay;
- direct trading output from analyst.

High:
- missing decision lineage;
- bad model promotion gate;
- strategy regime block failure;
- major data-quality failure.

Medium:
- weak source used for material claim;
- review queue item not created;
- missing evidence gap.

Low:
- noncritical metadata issue;
- minor warning missing.

## Required output

Every material stress failure creates:

```text
incident report
→ root cause label
→ remediation item
→ regression test
→ retest requirement
```
