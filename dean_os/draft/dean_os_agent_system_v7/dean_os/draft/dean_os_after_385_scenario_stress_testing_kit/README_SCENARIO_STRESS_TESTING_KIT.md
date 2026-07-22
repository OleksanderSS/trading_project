# DEAN-OS After 385 — Scenario Stress Testing Kit

Non-numbered practical Codex-facing kit.

Purpose: define system-level stress testing for DEAN-OS before replay, paper, shadow, supervised live,
or constrained autonomous execution.

This kit tests not only a model, but the full system:

```text
collectors
→ normalized event packets
→ macro regime logic
→ analyst agents
→ hypothesis ledger
→ strategy playbooks
→ pipeline controller
→ portfolio/risk state
→ execution gates
→ audit/review/learning loop
```

## Core principle

```text
A strategy passing normal backtest is insufficient.
The system must survive stress scenarios, bad data, regime shifts, and conflicting signals.
```

## Boundary

Allowed:
- scenario definitions;
- synthetic stress events;
- replay/paper/shadow simulation;
- risk gate checks;
- blocked-state checks;
- review and incident creation.

Forbidden by default:
- direct live orders;
- buy/sell/hold;
- price targets;
- bypassing risk engine;
- treating stress output as a trade signal.
