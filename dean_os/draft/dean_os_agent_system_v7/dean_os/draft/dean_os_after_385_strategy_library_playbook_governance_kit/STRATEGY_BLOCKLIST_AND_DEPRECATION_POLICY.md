# Strategy Blocklist and Deprecation Policy

## Block conditions

A strategy must be blocked when:

- regime is forbidden;
- market data is stale;
- model state is unknown;
- feature drift exceeds threshold;
- decision lineage is missing;
- time leakage is detected;
- risk engine is unavailable;
- kill switch is unavailable;
- portfolio limit is breached;
- unsupported asset appears;
- execution gateway is bypassed;
- operator marks strategy unsafe.

## Deprecation reasons

- repeated incident;
- degraded performance across regimes;
- overfit detected;
- source dependency invalid;
- market structure changed;
- strategy cannot pass replay/paper/shadow gates;
- strategy depends on ungrounded LLM output;
- better replacement exists.

## Deprecation actions

- mark status deprecated;
- block further promotion;
- preserve historical audit;
- create postmortem;
- create regression tests if failure was material.
