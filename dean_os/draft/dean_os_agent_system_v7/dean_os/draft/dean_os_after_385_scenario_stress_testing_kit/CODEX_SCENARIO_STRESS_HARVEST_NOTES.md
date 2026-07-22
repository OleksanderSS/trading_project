# Codex Scenario Stress Testing Harvest Notes

Use this kit as a practical design inventory. Do not blind-merge.

## High-priority harvest targets

1. Stress scenario schema
2. Stress scenario seed library
3. System stress run manifest
4. Component stress check matrix
5. Expected behavior template
6. Stress outcome review template
7. Stress-to-incident policy
8. Stress test schedule

## Suggested repo areas

- dean_os/stress_testing/
- dean_os/eval/
- dean_os/risk/
- dean_os/strategies/
- dean_os/pipeline_controller/
- dean_os/execution/
- configs/stress_scenarios/
- tests/fixtures/stress/
- docs/dean_os/

## Integration rule

Stress tests should verify system behavior, not generate trading recommendations.

## Relationship to other kits

- Macro kit supplies regimes and archetypes.
- Strategy kit supplies regime compatibility.
- Automation kit supplies maturity gates.
- Advanced Governance kit supplies incident/postmortem and portfolio governance.
- Operator Review kit consumes stress reports.
- Agent Memory kit decides if stress learnings can become durable memory.
