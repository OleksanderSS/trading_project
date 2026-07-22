# Strategy Library Architecture

## Objective

Create a governed library of strategies/playbooks that can later be tested in replay, paper, shadow,
and supervised/constrained execution.

## Strategy lifecycle

```text
idea
→ draft playbook
→ data requirements
→ regime compatibility
→ feature/input validation
→ replay
→ paper
→ shadow
→ supervised live
→ constrained autonomous candidate
→ active / deprecated
```

## What a strategy must declare

- strategy ID and version;
- intended market/asset universe;
- time horizon;
- supported regimes;
- forbidden regimes;
- required inputs;
- forbidden inputs;
- feature freshness requirements;
- model dependencies;
- risk budget;
- expected failure modes;
- evaluation criteria;
- promotion gate;
- rollback/deprecation policy.

## What a strategy must not do

- rely on unverifiable LLM output;
- use ungrounded news interpretation as direct signal;
- bypass risk engine;
- bypass execution gateway;
- promote itself;
- trade outside allowed assets or regimes.
