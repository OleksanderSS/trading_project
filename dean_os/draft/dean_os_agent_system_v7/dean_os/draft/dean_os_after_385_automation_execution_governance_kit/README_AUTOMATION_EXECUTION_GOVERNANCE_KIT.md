# DEAN-OS After 385 — Automation / Execution Governance Kit

Non-numbered practical Codex-facing kit.

Purpose: define a staged automation path for DEAN-OS: data/news collection, normalized event packets,
source quality/dedupe, analyst routing, hypothesis/evidence gaps, daily digest, audit logs, review queue,
automated learning, pipeline control, replay, paper/shadow trading, supervised live, and later constrained
autonomous execution.

This kit is not a production patch. Codex should harvest/adapt useful ideas into the existing repository.

Core rule:

```text
Automate simple low-risk processes first.
Escalate autonomy only after gates pass.
LLM/Analyst does not send orders.
Execution is only through risk engine + execution gateway.
```

## v2 clarification

Daily automation governance belongs primarily to the Pipeline Controller Agent.

Collectors collect. Analysts interpret. Orchestrator coordinates. The Pipeline Controller governs the daily
run state machine: start, health checks, validation, blocking, retry, audit, and downstream gate control.

New v2 files:

- `PIPELINE_DAILY_RUN_GOVERNOR_TEMPLATE.yaml`
- `COLLECTOR_HEALTH_AND_RUN_MANIFEST_TEMPLATE.yaml`
- `PIPELINE_CONTROLLER_AUTOMATION_BOUNDARY_NOTE.md`
