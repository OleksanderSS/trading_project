# Codex Automation / Execution Harvest Notes

Use this kit as practical design inventory. Do not blind-merge.

## High-priority harvest targets

1. Existing pipeline automation map
2. Automation maturity ladder
3. Simple process automation backlog
4. Daily automation run template
5. Normalized event packet schema
6. Source quality/dedupe template
7. Analyst routing + review queue template
8. Agent learning automation template
9. Pipeline Controller automation template
10. Model promotion + replay gates
11. Paper/shadow/supervised live gates
12. Execution gateway template
13. Risk limits + kill switch template
14. Order decision lineage schema

## Integration principle

The existing pipeline already has collectors, news processing, causal/context methods, leakage guards,
training managers, risk/kill switch, and trading/virtual portfolio components. Codex should add
governance, scheduling, contracts, gates, review queues, and audit lineage around those modules.

## Hard rules

- Start with data accumulation and review queues.
- No LLM direct order.
- No broker order without execution gateway.
- No model promotion without gates.
- No execution without risk engine, kill switch, and order lineage.

## v2 harvest targets

Codex should additionally harvest/adapt:

- `PIPELINE_DAILY_RUN_GOVERNOR_TEMPLATE.yaml`
- `COLLECTOR_HEALTH_AND_RUN_MANIFEST_TEMPLATE.yaml`
- `PIPELINE_CONTROLLER_AUTOMATION_BOUNDARY_NOTE.md`

Important v2 rule:

```text
Pipeline Controller Agent = daily automation governor.
Collectors = data acquisition.
Analyst Agents = interpretation.
Orchestrator = cross-agent coordination.
Risk/Execution = later gated order authority.
```
