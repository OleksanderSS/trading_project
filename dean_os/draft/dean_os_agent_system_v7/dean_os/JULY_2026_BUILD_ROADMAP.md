# DEAN-OS July 2026 Build Roadmap

Target window: 2026-07-01 through 2026-07-31.

The realistic July target is a reproducible source-first research and isolated
paper-ready prototype. Production live trading is not a July acceptance target:
it requires enough genuinely forward outcomes and operational observation time,
not only completed code.

## Week 1: Real Evidence Producers

- Complete saved FRED macro producer and manually confirm the series registry.
- Build filing/fundamental producer with exact metric, unit, fiscal period,
  availability timestamp, filing locator, accession/source hash, and gate
  fingerprint.
- Route cached news through the existing point-in-time evidence contract.
- Define source-bound sector aggregates; document counts and model outputs remain
  inventory, not sector evidence.

Acceptance:

- At least one real saved artifact per implemented source family produces a
  verified MarketContext fragment.
- Changed source, changed registry, future vintage, missing unit/period, and
  payload tampering fail closed.
- No collector, pipeline, learning, paper execution, or trading side effect is
  needed to review the producer artifacts.

## Week 2: Reproducible End-To-End Saved-Data Case

- Select one trustworthy saved ticker/timeframe window.
- Run only the required pipeline stages under existing causality and output
  contracts.
- Join Stage5 prediction review, Stage7 regime review, specialist context, and
  synthesis on one exact ticker/timeframe/target/context fingerprint.
- Save one reproducible case manifest with immutable source and artifact hashes.

Acceptance:

- One command or bounded runbook reproduces the same review artifacts from the
  same saved inputs.
- No target-label return is used as pre-trade evidence.
- No sector context is promoted to ticker evidence.
- Any missing real metric remains unavailable rather than synthesized.

## Week 3: Outcomes, Calibration, And Isolated Paper Executor

- Register prospective outcome windows before observation.
- Accumulate only matured, source-bound outcomes.
- Feed exact-context cases into deterministic shadow diagnostics.
- Implement an isolated paper simulator behind the existing
  receipt-plan-external-result-post-review lineage.

Acceptance:

- Forward case count is reported honestly; the 30-case threshold is not lowered
  or filled with fixtures.
- Paper execution cannot reach a broker, production portfolio, Stage6 live path,
  learning memory, or production config.
- Re-running the same paper plan is idempotent or explicitly rejected.

## Week 4: Operational Hardening

- Add bounded scheduling, failure recovery, artifact retention, and review inbox
  visibility.
- Exercise source tampering, stale data, partial pipeline failure, duplicate
  events, restart, and interrupted-write scenarios.
- Consolidate commands and remove obsolete duplicate entrypoints only after
  compatibility review.
- Produce a July readiness report with explicit blocked, partial, and ready
  capabilities.

Acceptance:

- A human can inspect source lineage, pipeline lineage, agent evidence,
  prediction semantics, outcomes, and paper results without reading raw code.
- All automatic learning, weight changes, production config writes, and live
  execution remain blocked unless separately reviewed and authorized.

## July Success Definition

By 2026-07-31 the system should have:

- real macro and fundamental evidence producers;
- one reproducible saved-data end-to-end review case;
- a running prospective outcome registry with whatever genuinely matured cases
  time permits;
- deterministic shadow diagnostics over real cases when thresholds permit;
- an isolated no-broker paper simulator and post-paper review;
- bounded operations/recovery documentation and tests.

Not promised by 2026-07-31:

- statistically sufficient forward evidence if time has not produced it;
- approved calibration or automatic agent/model weight changes;
- production capital allocation;
- live broker execution.

The main anti-loop rule for July: do not add another agent or review packet when
the next missing deliverable is a real producer, exact-context case, matured
outcome, or operational test.

## Current Progress — 2026-07-01

- Saved FRED macro producer: implemented and verified on 454 rows / 27 series.
- Real macro Agent Lab smoke: completed; MacroPolicyAgent returned neutral
  without fabricated policy direction, learning records, or proposals.
- Saved SEC filing index: implemented over 10,191 DuckDB metadata rows.
- AMD periodic filing request: one verified 10-Q with canonical SEC locator and
  pending immutable content/XBRL acquisition.
- Active semiconductor pipeline cohort resolved as NVDA/AMD/INTC/TSM; current
  periodic-filing coverage is 3/4 with NVDA explicitly missing.
- Tuning is exact-context scoped. One ticker/model/target failure cannot broaden
  into sector or multi-ticker tuning.
- Fundamental fact values: not present in the saved database and therefore still
  blocked in DuckDB, but official companyfacts snapshots are now stored.
- Accession-bound SEC fact producers: 14 Company Facts observations for
  AMD/INTC plus 8 consolidated inline-XBRL observations for TSM.
- An immutable SEC submissions snapshot recovered NVDA's latest 10-Q after the
  local collector window missed it. NVDA adds 7 quarterly USD facts.
- The merged artifact has 29 facts and `4/4` source coverage. Raw comparison
  remains blocked because quarter/USD and annual/TWD observations are not
  comparable without reviewed transformations.
- Verified merged fragment -> readiness gate -> Agent Lab path works with one
  matching fingerprint. Raw statement facts correctly return
  `needs_more_data` until reviewed ratios and price alignment exist.
- A verified sector-market producer now consumes the existing non-destructive
  price-repair artifact. Current `NVDA/AMD/INTC/TSM + QQQ` data provides 22
  common sessions and closes only the `market_confirmation` lane.
- The first combined semiconductor runtime is operational. It joins verified
  fundamental, macro, sector-market, and saved-news fragments at one cutoff and
  correctly returns `needs_more_data`, confidence `0.0`, with `4/5` required
  lanes satisfied.
- AMD's negative model case remains an exact ticker/model pipeline review case
  and is explicitly excluded from sector evidence.
- The strict saved-news producer excludes 9,209 orphan rows and routes 63
  candidates. Demand, capex cycle, and supply chain have independent
  strong-source corroboration; policy/geopolitical remains open.
- Immediate next deliverable is one additional independent official or tier-2
  policy source, without lowering source or corroboration thresholds.
- Formula-bound SEC ratio context is now active: 21 ratios and five
  multi-ticker same-period lanes. TSM annual ratios remain separate from US
  issuer Q1 ratios, so full-cohort comparability remains blocked.
