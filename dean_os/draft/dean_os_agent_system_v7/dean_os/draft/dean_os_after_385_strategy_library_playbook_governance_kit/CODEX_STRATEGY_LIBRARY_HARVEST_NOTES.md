# Codex Strategy Library / Playbook Governance Harvest Notes

Use this kit as a practical design inventory. Do not blind-merge.

## High-priority harvest targets

1. Strategy playbook schema
2. Regime compatibility matrix
3. Strategy input contract
4. Strategy eval requirements
5. Strategy promotion gate
6. Strategy blocklist/deprecation policy
7. Strategy-to-agent role map

## Suggested repo areas

- dean_os/strategies/
- dean_os/playbooks/
- dean_os/pipeline_controller/
- dean_os/risk/
- dean_os/execution/
- configs/strategies/
- configs/risk/
- tests/fixtures/strategies/
- docs/dean_os/

## Core rule

```text
Strategy output is not execution authority.
Execution authority belongs only to the execution gateway after risk and maturity gates pass.
```

## Integration order

1. Create strategy registry.
2. Add input contracts.
3. Add regime compatibility.
4. Add eval requirements.
5. Add promotion gates.
6. Add block/deprecation policy.
7. Only then connect to replay/paper/shadow.
