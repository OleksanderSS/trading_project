# Codex Integration Guide — Vertical Slice 7

## Preferred handoff

For a branch that has not integrated V6, use only:

```text
patch_cumulative_original_to_v7.diff
```

For a branch that already contains V6, use only:

```text
patch_v7.diff
```

Do not apply both patches.

## Integration sequence

1. Apply the appropriate patch.
2. Preserve the existing `src` pipeline implementation.
3. Do not route `DEANAgentSystemOrchestrator` to stages above 3 yet.
4. Locate the real saved outputs from completed stages 0–3.
5. Map the live result/artifact format into `PipelineStage03Bridge` rather than
   changing analytical contracts.
6. Run the verification commands.
7. Run one review-only system cycle from real parsed news.
8. Inspect branch records and the operator review package.

## Files introduced

```text
dean_os/system_topology.py
dean_os/config/system_topology.yaml
dean_os/pipeline_stage03_bridge.py
dean_os/full_system_orchestrator.py
dean_os/full_system_cli.py
tests/test_full_system_orchestrator_v7.py
README_FULL_AGENT_SYSTEM_V7.md
```

## Files modified

```text
dean_os/__init__.py
dean_os/daily_agent_run.py
dean_os/agents/pipeline_control.py
dean_os/agents/data_quality.py
dean_os/agents/risk.py
```

## Required invariants

- stages 0–3 may be consumed without enabling model/trading stages;
- model/PnL thresholds must not block a stage-0–3-only analytical run;
- rejected evidence must be removed from downstream agent context;
- every branch must appear in the system manifest;
- no branch may trade, promote models, or write learning memory;
- domain instances must be registered/configured, not implemented by copying
  the orchestrator.

## Verification

```bash
python -m compileall -q dean_os tests
pytest -q
```

Expected:

```text
49 passed
```

CLI smoke:

```bash
PYTHONPATH=. python -m dean_os.full_system_cli \
  --project-root . \
  --domain-id semiconductor_ai_infrastructure \
  --as-of 2026-07-12T12:00:00+00:00 \
  --knowledge-cutoff 2026-07-12T12:00:00+00:00 \
  --pipeline-stage03-json <real_or_fixture_stage03_result.json> \
  --soft-mode \
  --no-persistence
```
