# Verification — Vertical Slice 7

## Static/runtime checks

```text
python -m compileall -q dean_os tests: PASS
pytest -q: 49 passed
```

## New regression coverage

- topology contains every canonical branch;
- topology rejects cycles;
- stage-0–3 bridge excludes later stages;
- saved pipeline news is normalized;
- future evidence is removed from `MarketContext.news` before agents execute;
- full orchestrator emits all branch records in dependency order;
- stage-0–3 pipeline control does not require model/PnL metrics;
- allowed stage policy is restricted to `[0, 1, 2, 3]`;
- daily audit persistence and manifest hashing remain active.

## CLI smoke result

A stage-0–3 fixture produced:

```text
system status: partial
pipeline_stage03_intake: partial
pipeline_control: completed
evidence_intelligence: completed
domain_analysis: completed
world_model: partial
replay_evaluation: completed
governance_review: completed
daily_audit: partial
final decision: watchlist
allowed stages: [0, 1, 2, 3]
```

`partial` is expected in the no-persistence smoke because the fixture lacks a
complete stage-0 setup artifact and no immutable daily/world-state store was
enabled.

## Patch application verification

```text
V6 -> V7 patch apply --check: PASS
V6 -> V7 post-apply tests: 49 passed
Original snapshots -> V7 patch apply --check: PASS
Original snapshots -> V7 post-apply tests: 49 passed
```
