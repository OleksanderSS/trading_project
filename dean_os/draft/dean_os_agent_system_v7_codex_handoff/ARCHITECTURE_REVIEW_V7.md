# DEAN-OS Architecture Review — Vertical Slice 7

## Decision

The system should keep two primary trunks rather than creating one independent
orchestrator per sector:

1. pipeline/control;
2. analytical/world-model.

Evidence, replay/evaluation, and governance are cross-cutting branches with
explicit dependencies. Domain analysts are instances registered under the
analytical trunk.

## Canonical branches

| Branch | Parent trunk | Purpose | Current execution |
|---|---|---|---|
| `pipeline_stage03_intake` | pipeline | Read saved stages 0–3 outputs and parsed news | implemented |
| `pipeline_control` | pipeline | Stage-aware data/model guardrails | implemented |
| `evidence_intelligence` | analytical | point-in-time, credibility, dedup, provenance | implemented |
| `domain_analysis` | analytical | domain interpretation, hypotheses, transmissions | implemented for semiconductor instance |
| `world_model` | analytical | grids, scenarios, immutable World State | implemented |
| `replay_evaluation` | analytical | fixed-horizon replay scheduling and outcome path | implemented as review-gated framework |
| `governance_review` | governance | briefing, evidence gaps, operator review | implemented |
| `daily_audit` | governance | cross-artifact immutable manifest | implemented |

## Corrections made in V7

### 1. Pipeline stages 0–3 are now a first-class operating profile

The previous control surface expected model and trading metrics even when only
collection/processing/feature stages existed. That created false hard blocks.
V7 introduces `stage03_data_only`:

- stages above 3 are excluded;
- data/news availability is evaluated;
- model axes are explicitly not applicable;
- tuning and promotion remain disabled.

### 2. Rejected future evidence is removed from downstream context

Earlier code rejected an item in the Evidence Catalog but could leave the raw
item in `MarketContext.news`. An analytical agent could therefore still read
future evidence. V7 rebuilds the downstream news/document set from accepted
catalog inputs only.

### 3. Topology is now explicit and hash-bound

`system_topology.yaml` defines dependencies, inputs, outputs, allowed actions,
and forbidden actions. The topology and every system run receive content hashes.

### 4. One full-system manifest links all branches

`SystemRunManifest` records branch status, input/output hashes, warnings,
required/optional semantics, blocked branches, and the global authority boundary.

## Remaining architectural limitations

1. Several inner branches are still executed inside the existing composite
   `DailyAgentRun` / `DEANMinimalSystem`; their V7 branch records are honest
   projections from that composite execution, not fully independent workers.
2. `PipelineStage03Bridge` normalizes common in-memory result keys. Exact saved
   filenames and database tables from the live environment still require a
   repository-specific artifact resolver.
3. Only one domain instance is enabled. Portability is structurally prepared
   but not empirically demonstrated with a second domain.
4. The orchestrator is synchronous within one process. Durable scheduling,
   leasing, retry, and recovery remain a later operational slice.
5. Scenario probabilities remain structured analyst estimates until enough
   reviewed outcomes exist for empirical calibration.

## Recommended next step

Do not add another domain yet. Integrate V7, point the stage-0–3 bridge at the
actual saved news/processed/features artifacts, and run repeated semiconductor
daily cycles. After the contracts stabilize, separate composite projections into
independent branch executors and add the second domain instance.
