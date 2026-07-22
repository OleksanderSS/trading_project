# DEAN-OS Agent System — Vertical Slices 1–7

This package is a cumulative, runnable composition layer over the supplied
`dean_os` and `src` snapshots. The current build is **agent-first**: the heavy
pipeline is deliberately deferred and can be disabled while the analytical,
world-model, replay, and governance planes are developed and tested.

## Canonical runtime shape

```text
DEANMinimalSystem
└── DEANOrchestrator
    ├── Pipeline branch — prepared boundary, currently deferrable
    │   ├── PipelineControlAgent
    │   ├── DataQualityAgent
    │   ├── RiskAgent
    │   ├── bounded HybridPipelineAdapter
    │   └── proposal-only TuningAgent
    │
    ├── Analytical branch — current implementation focus
    │   └── DomainAnalyticalAgent
    │       └── semiconductor_ai_infrastructure profile
    │
    ├── World-model closure
    │   ├── ContextGrid v1
    │   ├── IndicatorStateGrid v1
    │   ├── point-in-time evidence audit
    │   ├── event classification
    │   ├── falsifiable hypothesis ledger
    │   ├── ScenarioOutcomeGraph
    │   ├── historical World-State retrieval
    │   └── immutable WorldStateSnapshot
    │
    └── Outcome / replay lifecycle
        ├── fixed-horizon OutcomeSnapshot
        ├── scenario probability scoring
        ├── append-only human review decisions
        ├── false-analogy scoring
        ├── calibration proposal generation
        └── review-gated manual learning-promotion packet
```

Every path remains review-only. The package cannot trade, mutate production
pipeline configuration, promote a model, or write learning memory.

## Current readiness

Run:

```bash
PYTHONPATH=. python -m dean_os.agent_system_readiness \
  --package-root . \
  --domain semiconductor_ai_infrastructure
```

The supplied package currently reports approximately:

```text
structural readiness: 0.73
operational readiness: 0.47
status: runnable structural MVP with major operational gaps
```

These values are not a certification. Structural readiness measures executable
contracts, composition, persistence, and tests. Operational readiness measures
real recurring sources, repeated reviewed outcomes, calibrated probabilities,
scheduling, and production operations.

## Run the agent system without the heavy pipeline

```bash
PYTHONPATH=. python -m dean_os.minimal_cli \
  --project-root . \
  --domain semiconductor_ai_infrastructure \
  --disable-pipeline \
  --input-json examples/semiconductor_minimal_context.json
```

This runs the agent and world-model branches while preserving the pipeline
integration boundary for later work.

## World-State persistence and analog retrieval

The system stores immutable point-in-time snapshots:

```text
ContextGrid
+ IndicatorStateGrid
+ ScenarioOutcomeGraph
+ evidence IDs / evidence gaps
+ as_of / knowledge_cutoff
+ content hash / parent snapshot
```

Inspect them with:

```bash
PYTHONPATH=. python -m dean_os.world_state_cli \
  --store reports/dean_os/world_state/world_states.sqlite3 \
  list --domain semiconductor_ai_infrastructure
```

Historical retrieval excludes future states and treats analogs as review-only.
Unknown context values do not count as similarity matches.

## Fixed-horizon outcome and replay lifecycle

Outcome evaluation is separate from the daily analytical run. It happens only
when a fixed horizon is due and outcome evidence is available.

```bash
PYTHONPATH=. python -m dean_os.world_state_outcome_cli evaluate \
  --world-state-store reports/dean_os/world_state/world_states.sqlite3 \
  --outcome-store reports/dean_os/world_state/world_states.sqlite3 \
  --snapshot-id <world_state_snapshot_id> \
  --input-json <outcome_evidence_packet.json>
```

Review an immutable outcome:

```bash
PYTHONPATH=. python -m dean_os.world_state_outcome_cli review \
  --outcome-store reports/dean_os/world_state/world_states.sqlite3 \
  --outcome-snapshot-id <outcome_snapshot_id> \
  --decision approved \
  --reviewer <reviewer_id> \
  --rationale "Evidence and resolution checked."
```

Generate a calibration proposal from approved, scored outcomes:

```bash
PYTHONPATH=. python -m dean_os.world_state_outcome_cli calibrate \
  --outcome-store reports/dean_os/world_state/world_states.sqlite3 \
  --domain semiconductor_ai_infrastructure \
  --horizon 20 \
  --min-approved-samples 20
```

A ready proposal still cannot change probabilities or write learning memory. It
only enters a separate human-reviewed shadow implementation process.

## Canonical contracts

### `PipelineMetricSnapshot v1`

The compatibility boundary between heterogeneous pipeline output and DEAN-OS.
It separates profitability, risk, validation, feature stability, data quality,
replay, and run lineage. Missing metric families remain missing.

### `ContextGrid v1`

Qualitative world-state representation across global, regional, country,
sector, adjacent-sector, and company scopes. The canonical global dimensions
include economic phase, market phase, credit phase, inflation phase, AI cycle,
and geopolitical phase.

### `IndicatorStateGrid v1`

Quantitative counterpart containing point-in-time macro, market, sector,
fundamental, and optional pipeline observations.

### `WorldStateSnapshot v1`

Immutable atomic binding of Context Grid, Indicator Grid, and Scenario Graph.
`knowledge_cutoff` cannot exceed `as_of`; observations after the cutoff are
rejected.

### `WorldStateOutcomeSnapshot v1`

Immutable fixed-horizon evaluation linked to one World State. It stores outcome
evidence, scenario/hypothesis resolutions, probability scores, and evidence
gaps. Human review is a separate append-only record.

## Domain portability

Do not copy the orchestrator to add a domain. Add a new profile under:

```text
dean_os/config/domain_profiles/<domain_id>.yaml
```

The profile should define evidence lanes, event mappings, adjacent sectors,
contradiction rules, required source types, and domain questions. A second
domain still needs to pass the same end-to-end tests before portability can be
considered demonstrated.

## Verification

```bash
python -m compileall -q dean_os src
pytest -q
```

Expected in this package:

```text
49 passed
```

The heavy pipeline and its analyzer chain were not run. That is intentional for
this build phase.

## Evidence intelligence and operator review (Vertical Slice 6)

The daily analytical layer now adds a conservative evidence-intelligence stage
before interpretation:

```text
bounded source payloads
→ domain-aware credibility assessment
→ conservative semantic deduplication
→ immutable Evidence Catalog
→ analytical runtime
→ coverage-aware Evidence Gap Plan
→ Markdown/HTML briefing
→ immutable operator review inbox
```

Credibility is a provenance score, not a truth label. Weak sources can create
lead-generation or evidence-gap tasks but cannot close a material coverage lane.
Exact or same-source near duplicates are suppressed. Similar evidence from an
independent source is retained when it represents genuine corroboration.

Collector routing remains configuration-only. Routes contain query terms,
source types, priorities, and authority boundaries; they never execute network
calls from `DailyAgentRun`.

The domain profile owns the mandatory coverage gate. Adding logistics or another
domain therefore requires a profile and source rules, not a copied orchestrator.

Operator inbox inspection:

```bash
PYTHONPATH=. python -m dean_os.operator_review_cli \
  --store reports/dean_os/operator_review/inbox.sqlite3 \
  --domain-id semiconductor_ai_infrastructure
```

## Codex handoff

For a fresh integration, use only the latest V6 full package and
`patch_cumulative_original_to_v6.diff`. Do not apply V1–V5 first.

If the target branch already contains V5, use only `patch_v6.diff`.
Never apply both patches.


## Full branch topology and stages 0–3 intake (Vertical Slice 7)

The canonical entry point is now `DEANAgentSystemOrchestrator`. It exposes the
full pipeline/control, analytical/world-model, replay, review, and audit branch
graph while consuming already-produced pipeline stages 0–3 outputs. See
`README_FULL_AGENT_SYSTEM_V7.md` for the current topology and CLI.
