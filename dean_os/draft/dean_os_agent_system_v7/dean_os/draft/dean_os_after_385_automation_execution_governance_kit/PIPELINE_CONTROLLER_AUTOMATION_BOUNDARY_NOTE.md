# Pipeline Controller as Daily Automation Governor

The user intent is that daily automation should be managed by the Pipeline Controller Agent.

This does not mean the Pipeline Controller becomes the analyst or the trader.

## Correct separation

```text
Collectors
→ fetch and store raw/normalized data

Pipeline Controller Agent
→ governs the daily run, validates health, blocks bad data, controls state machine, triggers gates

Analyst Agents
→ interpret normalized events, classify risk archetypes, create hypotheses and evidence gaps

Orchestrator
→ coordinates cross-agent routing, dependencies, review queues, and escalation

Risk Engine / Execution Gateway
→ only later, after replay/paper/shadow/supervised gates, can authorize constrained execution
```

## Why this matters

Daily automation governance is not just cron.

It includes:

- run IDs;
- collector health;
- source hashes;
- data freshness;
- deduplication status;
- schema validation;
- analyst output validation;
- eval/audit checks;
- blocked-state reasons;
- review queues;
- daily digest;
- reproducibility manifest.

## Controller powers

Allowed:

- trigger collectors;
- stop a run;
- retry safe collectors;
- block downstream stages;
- request analyst review;
- request replay/paper test;
- write audit logs;
- create review queue items.

Forbidden:

- direct macro/sector thesis as final output;
- direct probability without source;
- direct buy/sell/hold;
- broker order;
- bypass risk engine;
- promote model without gate.
