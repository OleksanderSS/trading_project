# DEAN-OS Full Agent-System Skeleton — Vertical Slice 7

Vertical Slice 7 establishes the canonical orchestration skeleton. It does not
run the expensive pipeline. It consumes already-produced pipeline stages 0–3
outputs and routes them through a review-only agent system.

## Canonical topology

```text
DEANAgentSystemOrchestrator
├── Pipeline / control trunk
│   ├── pipeline_stage03_intake
│   └── pipeline_control
│
├── Analytical / world-model trunk
│   ├── evidence_intelligence
│   ├── domain_analysis
│   ├── world_model
│   └── replay_evaluation
│
└── Cross-cutting governance trunk
    ├── governance_review
    └── daily_audit
```

The dependency graph is stored in:

```text
dean_os/config/system_topology.yaml
```

The root orchestrator is:

```text
dean_os.full_system_orchestrator.DEANAgentSystemOrchestrator
```

## Current pipeline boundary

The active integration boundary is explicitly limited to stages 0–3:

```text
0 — setup / runtime profile
1 — collection, including parsed news
2 — processing / normalization
3 — feature engineering and saved upstream artifacts
```

Stages 4+ are not started by this runtime. The `PipelineStage03Bridge` consumes
an existing result dictionary or explicit artifact references and creates an
immutable `PipelineStage03Packet`.

In this operating profile, pipeline control evaluates upstream data/news
availability and quality. PnL, train/test gaps, model stability, replay hit rate,
and trade-risk metrics are marked not applicable until model/evaluation stages
exist.

## Domain instances

The orchestrator is not copied per domain. The first configured instance is:

```text
semiconductor_ai_infrastructure
```

Later instances are added through the agent registry and domain profile files.
They reuse the same branch graph, evidence contracts, world-state store, replay
contracts, review gates, and audit layer.

## Run from saved stages 0–3 JSON

```bash
PYTHONPATH=. python -m dean_os.full_system_cli \
  --project-root . \
  --domain-id semiconductor_ai_infrastructure \
  --as-of 2026-07-12T12:00:00+00:00 \
  --knowledge-cutoff 2026-07-12T12:00:00+00:00 \
  --pipeline-stage03-json path/to/stage03_result.json \
  --tickers NVDA AMD AVGO \
  --timeframes 1d \
  --soft-mode \
  --output reports/dean_os/latest_full_system_run.json
```

## Safety boundary

The full skeleton remains unable to:

- place orders;
- write production pipeline configuration;
- promote models;
- write learning memory automatically;
- use evidence after the knowledge cutoff;
- execute stages above 3 through the stage-0–3 bridge.

## Verification

```bash
python -m compileall -q dean_os tests
pytest -q
```

Expected result:

```text
49 passed
```
