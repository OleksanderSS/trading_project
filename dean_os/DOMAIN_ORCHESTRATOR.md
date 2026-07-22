# DEAN-OS Domain Orchestrator

`DomainOrchestrator` is a review-only diagnostic composer for one configured
economic domain. It is not a replacement for `DEANOrchestrator` and it is not
the active hash-bound `FullSystemReviewCycle`.

It loads one registry snapshot, separates generic pipeline diagnostics from
the matching `DomainAnalystAgent`, runs the domain analyst exactly once, and
runs any configured composite pipeline manager after the analyst runtime
artifact exists. Pipeline agents execute through `PipelineBranch`, so timeout
and report-schema contracts remain active. The output always has
`can_trade=false` and cannot write learning memory or production config.

Canonical domain definitions remain in `config/domain_profiles/*.yaml`.
`dean_os/domain_profiles.py` is only a compatibility accessor and does not
duplicate those profiles. A domain analyst must be explicitly enabled and
fully configured in `dean_os/config/agent_registry.yaml`; the orchestrator no
longer creates an unconfigured fallback analyst.

Optional profile agents are off by default. `--include-profile-agents` is an
explicit diagnostic opt-in and is not equivalent to the normal
evidence-pack/manager-plan gate in `AnalystProfileOrchestrator`.

Example:

```powershell
python run_agent_domain_orchestrator.py `
  --domain semiconductor_ai_infrastructure `
  --as-of 2026-07-13T08:53:14Z
```

For the current semiconductor system assessment, use
`run_agent_full_system_review_cycle.py` and its hash-bound world-model and
governance closure steps instead.
