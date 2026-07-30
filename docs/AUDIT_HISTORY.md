# Audit history (2026-07-24 → 2026-07-28)

Full chronological record of the project-wide audit-and-fix initiative on branch
`analyst-core-phase1`: every pass, root cause, and commit hash. Moved here from
Claude's memory directory on 2026-07-30 because it had grown to 248 KB and was
being loaded into context every session. The compacted memory entry now holds
current state + open work only and points here for detail.

---

---
name: project-colab-pipeline-audit
description: "Ongoing incremental audit of the WHOLE project (pipeline stages 0-7, dean_os agent system, and eventually all of src/), fixing real bugs found along the way rather than just reporting them"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7c714356-0914-4c25-a794-b016ec635ea8
  modified: 2026-07-28T10:39:27.607Z
---

Standing initiative on branch `analyst-core-phase1`: walk the pipeline
(stages 0-7) and the dean_os analyst agent system incrementally, verify
results at each step, and fix what's found rather than just flagging it.
[[feedback-audit-and-fix-standing-mandate]] covers the collaboration rule
this runs under.

**Why:** the user discovered real Colab training was silently broken by a
stale `runtime_params.json` (epochs=1 leaking into full runs) and by deeper
methodology bugs in `scripts/colab/colab_clean_cell.py` (target-type
blindness, missing `return` statements silently discarding successful
training metrics, fake sequence-length-1 for recurrent/attention models,
random instead of chronological train/val split). Once those surfaced, the
user asked for a full incremental audit rather than a one-off fix, on the
premise that "there are probably many more bugs like this."

**State as of 2026-07-24 (commit `76d74a69` on `analyst-core-phase1`):**
- Fixed: exit-code-1 atexit bug, stale `runtime_params.json`, target-type
  blindness, missing-return bug, val_loss-vs-training-loss bug, fake
  sequence length, random val split → chronological split with purge.
- Fixed: dean_os macro evidence-provenance chain (4 layered bugs: registry
  path, `EVENT_CLASS_TO_DIMENSION` routing keyed by FRED series_id instead
  of context_key, `context_adapter.py` key mismatch, a `_first_required_or`
  sentinel-defeating override).
- Built: empirical champion selector (`src/pipeline/hybrid/champion_selector.py`,
  `src/config/target_type_registry.py`) wired as a hard filter into Stage 5
  via `ResultsProcessor.build_models_metadata()` — champion picked per
  (ticker, target, horizon) from real training metrics, separate from the
  static prior in `model_competence_map.json` (deliberately untouched).
  `scripts/colab/select_champions.py` is now a thin CLI wrapper over the
  same `champion_selector.py` the live pipeline uses — no duplication.
- Audited stages 5-7: fixed `path`→`model_path` key mismatch (5 downstream
  readers), silent Stage-5 skip now logs a warning, `_drop_incomplete_model_rows`
  now actually drops rows instead of zero-filling, archived 2 confirmed-dead
  files (`final_stages_executor.py`, `orchestrator_context.py`) to
  `archive/dead_pipeline_code/` via `git mv` rather than deleting.
- Just completed (commit `76d74a69`): every trainer (mlp, cnn, lstm, gru,
  transformer, tabnet, autoencoder) now reports `validation_windows` (3
  chronological metric windows instead of 1 aggregate number) and
  `context_windows` (market-context snapshot per window, reused from
  existing `market_context_*` feature columns — no new context-computation
  path built). This is metadata only, not yet a selection axis, and is
  designed to plug into the *existing* `ModelSelectionService` for future
  regime-conditioned selection rather than a new parallel mechanism — a
  `quant_context` bridge idea was explicitly built then reverted this same
  session once the real bug in the existing system was found instead. See
  [[feedback-audit-and-fix-standing-mandate]] for why "don't build parallel
  systems, fix the existing one" is a hard rule here, not just this one
  decision.

**How to apply:** before proposing a new mechanism for anything in this
pipeline, check whether an existing one (ModelSelectionService, the static
competence map, the champion selector, market_context_* columns) already
covers it and is just broken/underused — fix/extend that first. This
project has a documented history of the same bug (macro context-key
mapping) being independently mis-fixed twice by different agents/sessions
before someone found the actual root cause.

**Commit pass done (2026-07-24):** all previously-uncommitted work from
this audit is now committed on `analyst-core-phase1` as 6 logical commits
on top of `76d74a69`: `d729c8fd` (dean_os macro evidence-provenance chain),
`29357bec` (champion selector), `1d61f0ee` (Stage 5 drop-incomplete-rows +
component_factory cleanup), `51a4aeb8` (archive dead
final_stages_executor/orchestrator_context), `83ed8383` (atexit exit-code
broad-catch fix), `469ce31a` (stale runtime_params.json deletion). All 55
relevant tests passed before committing. Working tree is clean except:
`dean_os/analyst_core/lens_contract.py` (pure CRLF/LF line-ending noise,
zero content diff — safe to ignore), `diagnostic_reports/feature_lineage_report.json`
and several `.pyc` files (auto-generated artifacts, not source), and
`reports/*.txt` (untracked scratch run logs).

**Scope widened (2026-07-24):** user explicitly asked for this same
audit-and-fix treatment across the *entire* project, not just the Colab
pipeline. Confirmed project scale: `dean_os/` has 1349 `.py` files but 982
of those are `dean_os/draft/` (design notes/prototypes, not live code —
only referenced from docstrings/comments and one already-archived
`archive_v1/daily_governor.py`); real live dean_os code is ~189 files.
`src/` has ~30 top-level subdirectories beyond what the pipeline audit
already covered. This is a multi-session effort — process module by module
(recon pass via a research subagent first to survey a directory and flag
concrete candidates, then personally verify + fix each one, then commit,
then update this memory) rather than trying to cover it in one sitting.

**dean_os/agents/ pass complete (2026-07-24, commits `6316f8e4`..`ca2ab94c`):**
- Fixed `CoherenceScanAgent` (cross-agent contradiction scanner): (1) crashed
  on every real run — `context._agent_reports` holds raw pydantic model
  instances, agent called `.get()` on them expecting dicts; `error_behavior:
  skip` meant the crash was never even logged. (2) Deeper: even once fixed,
  the agent ran inside the same `asyncio.gather()` batch as the analytical
  peers whose verdicts it's designed to reconcile, so the merged report set
  didn't exist yet when it ran — it always saw 0 reports and always
  returned "nothing found". Added `DEANOrchestrator.PEER_SYNTHESIS_AGENTS`
  (currently just `coherence_scan`) as an explicit second orchestration pass
  run after `pipeline_reports + analytical_reports` is known. This is the
  same "layered silent-failure" pattern as the earlier macro
  evidence-provenance bugs — confirmed by adding a regression test with real
  pydantic model instances and an orchestrator-level end-to-end test, since
  every existing test had only ever exercised the dict-shaped fallback.
- Fixed `NewsEventAnalyzerAgent`: `NewsEvent(**item)` splatted raw news-record
  dicts into a constructor that only accepts headline/source/published_at —
  real collectors use a `title` column, not `headline`, so any real record
  raised `TypeError`. The VIX-injection path had the mirror bug (passed 7
  kwargs the constructor didn't accept, silently swallowed by `except
  Exception: pass`). Agent is `enabled: false` in the registry so no current
  blast radius, but was guaranteed to break the moment someone re-enabled it.
- Fixed `EventCausalGraphBuilder.build()` (`dean_os/event_causal_graph.py`):
  referenced an undefined `watch_list` name (copy-paste slip from the sibling
  `all_sectors` accumulator) — `NameError` on any event with non-neutral
  shock or `|impact| >= 0.2`. No existing test exercised `build()` end-to-end
  (they construct `CausalGraph`/`CausalNode` directly), so this had never
  been caught; found only because fixing `NewsEventAnalyzerAgent` needed a
  real end-to-end test that happened to go through this path.
- Archived confirmed-dead `CollectorHealthAgent`/`CollectorInventoryAgent` —
  never instantiated anywhere outside their own class definitions, absent
  from `agent_registry.yaml` and every test. Also removed a now-dangling
  `dean_os/__init__.py` lazy-export entry for `CollectorInventoryAgent` that
  would have raised `ModuleNotFoundError` if ever triggered post-archival.
- Removed one stale test in `test_historical_research_replay.py` that
  imported `_research_stance`/`_research_direction` — functions removed in
  commit `787dc294`'s rewrite (replaced by reading `report.position_bias`
  directly). That broken import blocked *collection* of the entire file,
  silently disabling 4 other, still-relevant tests alongside it.

**Two things discovered but deliberately NOT fixed in this pass (flagged for
follow-up, not silently left):**
1. Unblocking `test_historical_research_replay.py` revealed the other 4
   tests now fail for real: `HistoricalReplayRunner.run()`'s actual
   signature no longer accepts `news_data_paths`/`macro_data_paths`/
   `focused_overlay_path`/`apply_focused_overlay`. This is a separate,
   substantial investigation (API drift vs. real regression — needs
   comparing current `historical_research_replay.py` against what the tests
   assume). Spawned as background task `task_6a4b1774`.
2. **70 pre-existing test failures across `tests/dean_os/`** discovered via
   a full-suite run — confirmed via `git stash` + isolated run that these
   fail identically on the pre-session code, i.e. NOT caused by this
   session's changes.

**Also noticed, not touched:** two pre-existing `git stash` entries from
branch `production` (commit `8d94503d`, "Refactor: Cleanup repository -
track only src/ and dean_os/") were already sitting in the stash before
this session started. Not created by this session, not touched — but the
user may not remember they're there; worth asking about if a clean stash
matters.

**70-failure triage complete, first batch fixed (2026-07-24, commits
`113851da`..`a9d0b91d`):** categorized all 70 into: ~30 tests failing
because a referenced `run_agent_<name>.py` CLI wrapper script simply
doesn't exist at the repo root (the underlying class/logic is real,
tested separately, and passes — someone wrote the module + its test
including a CLI-smoke-test expectation, but never wrote the wrapper
script; mechanical, templatable, not yet done), a handful of
likely-stale-assertion tests needing individual judgment (`test_clone_to_energy`-
style: real data legitimately grew/changed and the hardcoded expected
value is just old), and ~14 genuine bugs — user chose to fix the real
bugs first. All 14 fixed and verified (70 → 56 failures, confirmed via
diff against the original failure list: zero new failures, zero
regressions):
- `dean_os/analysts/_producers/ticker.py`'s `DEFAULT_ISSUER_REGISTRY` had
  the exact same `.parent` (one level) vs `.parents[2]` (three levels)
  path bug as the earlier macro.py fix — landed at a non-existent
  directory. Once fixed, `semiconductor_issuer_identity_registry.yaml`
  turned out to only cover 4 of the domain's 12 tickers (NVDA/AMD/INTC/TSM);
  filled in the missing 8 (ASML/AMAT/LRCX/KLAC/AVGO/MU/ARM/QCOM), CIK
  numbers verified against SEC's `company_tickers.json` rather than
  trusted from memory. This is the second time in one session the exact
  same path-depth mistake was found in a `_producers/*.py` file — worth
  grepping `dean_os/analysts/_producers/*.py` for `Path(__file__)` again
  if any new producer file shows up.
- `dean_os/recommendation_memory.py`: `_init_db()`'s `CREATE TABLE IF NOT
  EXISTS` can never migrate a pre-existing table with a stale schema —
  the real `data/dean_os/recommendation_memory.sqlite` predated the
  `revision` column and broke every read. Root cause was compounded by
  `tests/dean_os/test_review_approved_learning_loop.py` never passing an
  isolated `memory_path`, so tests were silently touching that real file.
  Added a proper migration (rename/recreate/backfill) plus test isolation.
- `dean_os/agents/domain_research.py`'s `ValueScreeningAgent`: three
  layered bugs, same "contract changed on one side, reader wasn't
  updated" pattern as everything else this session — `_fundamental_gate()`
  read a nested `gate["summary"][...]` shape that no real caller
  (`agent_lab.py`'s `_fundamental_gate_summary()`) ever produces (it's
  flat); `_score_fundamentals()` didn't descend into the structured
  `fundamentals[ticker]["metrics"][name]["value"]` shape
  `structured_context_provenance.py`'s `_candidate_map()` already handles;
  a computed `gate_fingerprint` was never actually compared against
  anything, so a gate reviewed against different data than what's
  currently attached would be silently accepted.
- `dean_os/analyst_core/domain_analyst_runtime.py`'s `clone()` applied
  `ticker_universe`/etc. overrides to `self.analyst` but left the separate
  `self.profile` attribute as the unmodified registry default — the two
  silently disagreed after any override. Also two hand-rolled test
  fixtures missing the producer-contract wrapper (`created_at`/`status`/
  `safety.review_only`) that `_validated_producer` requires unconditionally.
- **Caught a self-inflicted regression before committing**: completing the
  issuer registry (4→12 tickers) broke `test_domain_scoped_fundamentals_envelope.py`,
  which did an exact-set-match against the registry's issuer keys and
  had coincidentally been relying on the *shared production registry*
  having exactly 4 entries to match its own 4-ticker fixture — a real
  test-isolation gap (same class of bug as the recommendation_memory.sqlite
  one). Gave it its own isolated registry YAML fixture instead of
  depending on the shared file's current size. Full before/after diff of
  the 70→56 failure list confirmed this was the only regression and it's
  now resolved with zero net-new failures.

**CLI wrapper pass complete (2026-07-24, commits `4d783f63`, `9318ea87`):**
Wrote 33 missing `run_agent_<name>.py` scripts (deduplicated from the ~30
failing `*_saves_markdown_and_cli_runs`/`*_cli_runs`/`*_cli_smoke` tests) —
argparse → kwargs → `Class(output_dir).build(**kwargs)` → print the
module's own `render_*_markdown()` output with backticks stripped, matching
`run_agent_saved_macro_evidence_producer.py`'s existing template. One
exception: `run_review_only_real_source_normalized_packet_validation_gate.py`
wraps a plain function (`build_validation_gate`), not a class, so it writes
`latest.json`/`latest.md` itself.

Along the way, discovered the project's documentation (`COMMAND_CHECKLIST.md`,
`IMPLEMENTATION_STATUS.md`, `NEXT_CHAT_HANDOFF.md`, `Agents_architecture.md`)
references **192 `run_agent_*.py` wrappers total** — only 20 existed before
this session, now 53. The other ~139 are genuinely aspirational/undone
work, not something to build reflexively; `test_agent_cli_restore.py::
test_documented_run_agent_wrappers_exist` wants all 192 to exist and will
keep failing until someone either builds the rest or trims the docs to
match reality — explicit user decision this session was to only build the
~30 that real tests exercise, not chase that count.

Fixing the wrappers surfaced ~7 small `render_*_markdown()` label
mismatches (e.g. "Status:" vs "Architecture:", "Compatible:" vs "Can route
to analyst apply loop:") that had never been checked against real CLI
output before (no working wrapper existed to run the check). Verified each
had exactly one dependent test before renaming, to avoid breaking a
different caller of the same label.

**Result: 56 → 23 failures** (confirmed via full `tests/dean_os` diff,
zero new regressions). The remaining 23 are five separate,
already-diagnosed issues, explicitly deferred (not silently dropped):
1. **`test_agent_cli_restore.py::test_documented_run_agent_wrappers_exist`**
   (1) — wants all 192 documented wrappers; out of scope per above.
2. **`_validated_producer` `as_of`-defaulting bug** (7 tests: `test_analyst_knowledge_pack_builder.py`
   x4, `test_analyst_core_pipeline_manager.py` x2, `test_analyst_core_cli.py`
   x1) — `build_knowledge_pack`/`load_evidence_from_artifacts` and similar
   callers don't default `as_of` to `utc_now_iso()` the way
   `domain_analyst_intake_packet.py` already does, so
   `artifact_evidence_loader.py`'s `_validated_producer` rejects them with
   "requires an analysis as_of". Single root cause, likely a quick fix,
   but a different code path from today's work — good next-session target.
3. **`DomainAnalystPortabilityReview` logic bug** (2 tests,
   `test_domain_analyst_portability_review.py`) — `.build()` computes
   `review_status: "domain_analyst_portability_blocked"` when tests expect
   `"...ready"`; a real bug in the class itself, not a CLI-script or label
   issue (confirmed: fails identically whether called directly or via the
   now-working CLI wrapper).
4. **`test_staged_workbench_integration_review.py`** (1) — `draft_bundle`
   path `dean_os/draft/dean_os_after_245_full_context_bundle` does not
   exist on disk at all (confirmed), so `staged_block_count` is always 0
   instead of the expected `>= 30`. Needs the actual bundle restored or the
   test's expectation revisited — not a code bug per se.
5. **Untouched from the original 70**: `test_agent_capability_matrix.py`
   (`matrix_complete` logic bug), `test_collector_synthetic_production_boundary.py`,
   `test_parallel_scaffold_safety.py`, `test_saved_sec_filing_index_producer.py`,
   `test_pipeline_control_metric_artifact_candidates.py` (1 each) — all
   flagged in the original triage as needing individual judgment calls, not
   yet looked at.
6. **`test_historical_research_replay.py` + `test_historical_research_replay_batch.py`**
   (7) — the API-drift issue already spawned as background task
   `task_6a4b1774`, running independently in a separate session as of this
   writing.

**as_of-defaulting bug fixed (2026-07-24, commit `466bff7a`):** before
fixing, checked whether this code path was real/live vs. leftover agent
experimentation — confirmed `SectorAnalyst`/`DomainAnalystRuntime` (which
`load_evidence_from_artifacts` underpins) is exactly what
`dean_os.agents.domain_analyst:DomainAnalystAgent` runs, and that agent is
registered 5+ times in `agent_registry.yaml` — genuinely live, not
orphaned. Root cause confirmed via `run_analyst.py` (the real CLI entry
point), which already worked around the bug at its own call site with
`args.as_of or utc_now_iso()` — pushed that same fallback into
`load_evidence_from_artifacts()`/`build_knowledge_pack()` themselves.
Along the way fixed 3 more real gaps in `build_knowledge_pack.py`'s
`_evidence_to_knowledge_items()` found via the same test file:
`content_sha256`/`known_limitations` were never populated, and
`required_lane_eligible` (an existing `AnalystEvidenceItem.provenance`
field read elsewhere in this codebase by `sector_analyst.py`) was dropped
on the floor instead of propagated. **Result: 23 → 16 failures**, zero
regressions (clean diff, only removals).

**Standalone governance/audit tools — lower priority, not on the live
execution path:** `DomainAnalystPortabilityReview`, `StagedWorkbenchIntegrationReview`,
`AgentCapabilityMatrixBuilder` are top-level `dean_os/` modules with no
other live caller (confirmed via grep — only their own CLI wrapper/tests
reference them). They look like one-off self-audit reports rather than
pipeline execution code: `staged_workbench`'s failing test references a
draft-bundle directory (`dean_os/draft/dean_os_after_245_full_context_bundle`)
that doesn't exist on disk at all — plausibly a one-time migration tool
whose job is already done. Lower priority than anything on the
DomainAnalystAgent/PipelineManagerAgent execution path.

**Remaining 16 failures, categorized:**
1. `test_agent_cli_restore.py::test_documented_run_agent_wrappers_exist` (1)
   — wants all 192 documented wrappers; explicitly out of scope.
2. `DomainAnalystPortabilityReview` logic bug (2) — `review_status` computes
   "blocked" when tests expect "ready"; confirmed same failure whether
   called directly or via CLI, not a wrapper/label issue.
3. `test_staged_workbench_integration_review.py` (1) — missing draft
   bundle directory, see above.
4. Not yet individually triaged (1 each): `test_agent_capability_matrix.py`
   (`matrix_complete` logic bug), `test_collector_synthetic_production_boundary.py`
   (reddit_sentiment), `test_parallel_scaffold_safety.py`,
   `test_saved_sec_filing_index_producer.py`,
   `test_pipeline_control_metric_artifact_candidates.py`.
5. `test_historical_research_replay.py` + `_batch.py` (7) — API-drift
   issue, running as background task `task_6a4b1774` in a separate
   session as of this writing.

**Category-4 triage complete, 4 of 5 fixed (2026-07-24, commits `ba565b0a`,
`ec549322`):**
- **Real bug found and fixed**: `pipeline_manager` (composite
  `PipelineManagerAgent` for `domain_id=semiconductor_ai_infrastructure`)
  had no `execution_group`/`run_phases`, so it ran unrestricted in every
  phase *alongside* the standalone `semiconductor_analyst` for the same
  domain — silently duplicating analysis every `pre_trade` cycle. The
  registry's own `_validate_exclusive_groups` check exists exactly to
  catch this but never saw the conflict since `pipeline_manager` was never
  given the matching group. Gave it `execution_group:
  semiconductor_domain_analysis` + `run_phases: [pre_trade]` (matching
  `semiconductor_analyst`) and set `enabled: false` — `semiconductor_analyst`
  stays active since it has real evidence artifact paths wired and
  `pipeline_manager` has none configured; `pipeline_manager` is now a
  documented, registry-protected opt-in swap, not a silent duplicate.
- `test_parallel_scaffold_safety.py` asserted 6 of 7 named agents should
  be inactive by default (`pipeline_manager`, `semiconductor_analyst`,
  `agriculture_analyst`, `historical_analogies`, `coherence_scan`,
  `freshness_audit`) — confirmed only `pipeline_manager`'s inactivity was
  a real bug; the other 5 are genuinely live (independently confirmed
  throughout this session). Updated the test to assert reality, and added
  a real regression test for the exclusive-group conflict logic itself.
- `test_agent_capability_matrix.py`'s `CAPABILITY_CONTRACTS` dict (in
  `dean_os/agent_capability_matrix.py`) was stale — missing 10 registry
  agents added since it was written (`agent_count` 28 → 39, matching the
  same "reference list didn't track registry growth" pattern as the
  192-vs-20 CLI-wrapper docs and the 4-vs-12-ticker issuer registry).
  Added their contract entries.
- `test_reddit_sentiment_stays_disabled_until_real_adapter_exists`: the
  real adapter now exists (`reddit_sentiment_collector.py` fetches real
  posts from Reddit's public RSS feeds, no API key, no synthetic data)
  and was legitimately enabled in `src/config/collectors.yaml` since this
  test was written — updated the test to assert the collector is enabled
  with `use_synthetic_data` still `False` (the actual invariant that must
  never regress), not that it stays off forever.
- `test_current_database_has_verified_amd_periodic_filing`: asserted an
  exact row count (10191) against the real, growing
  `data/trading_data.duckdb` (currently 19371). Changed to a floor (`>=`)
  — every other assertion in the test already passed unchanged.

**Result: 16 → 12 failures**, zero regressions (clean diff, only removals).

**One item in category 4 turned out to be genuinely deep, not a quick
fix — deferred, not silently dropped:**
`test_pipeline_control_metric_artifact_candidates.py::test_metric_materializer_expands_pipeline_manifest_when_locked_pair_exists`
traces to a real, unresolved architecture question: `build_model_evaluation_candidate`
(`src/pipeline/stages/modeling/pipeline_control_artifacts.py`, real
training-output builder) sets `artifact_class:
"pipeline_control_model_evaluation_candidate"` and
`contract_status: "ready_locked_model_evaluation_candidate"`, but
dean_os's classifier (`verify_locked_model_evaluation` in
`pipeline_control_evidence_inventory.py`) requires `artifact_class ==
"locked_model_evaluation"` plus `joined_lineage`/`join_contract`/
`materialization_contract` proof fields that the builder never produces.
There IS a "candidate → locked" promotion mechanism in this codebase
(`PipelineControlLockedEvaluationAssembler`), but it's wired for a
*different* pairing (training candidate + Stage-7 evaluation candidate,
via `artifact_types=("model_evaluation_json"/"model_evaluation_candidate")`
+ `("evaluation_metric_candidate"/"evaluation_metric_json")`) — not the
model_evaluation+feature_stability pairing this test builds via
`write_pipeline_control_metric_artifact_candidates`. Two questions before
touching this: (1) should model_evaluation+feature_stability candidates
from the real training pipeline go through some assembler at all, or is
the dean_os classifier's bar simply wrong for this specific pairing (i.e.
should it accept `contract_status == "ready_locked_model_evaluation_candidate"`
as an alternative to the `artifact_class`/lineage check)? (2) does a
"real" locked pair ever actually reach this materializer in production
today, or has this path never worked end-to-end? Asked the user via
AskUserQuestion whether to dig in now or defer; no response was given, so
deferred per the recommended option. Worth a dedicated session — likely
touches `pipeline_control_evidence_inventory.py`,
`pipeline_control_locked_evaluation_assembler.py`, and
`pipeline_control_artifacts.py` together.

**`DomainAnalystPortabilityReview` bug fixed (2026-07-24):** root cause
was a real, live gap — `EVIDENCE_TYPE_ALIASES` (in
`dean_os/analyst_core/domain_analyst_intake_packet.py`, the keyword-alias
table `DomainAnalystIntakePacket` uses to classify incoming news/documents
into a domain's `required_evidence_types` lanes) was missing entries for
7 of 15 real domain profiles (`communication_services`,
`consumer_discretionary`, `consumer_staples`, `healthcare`, `industrials`,
`metals_mining`, `utilities_power`) — 19 evidence-type keys total (e.g.
`trial_readouts`, `fda_decisions`, `pmi`, `electricity_load`). Same
"reference dict didn't track registry/profile growth" pattern as
`CAPABILITY_CONTRACTS` and the issuer identity registry. This wasn't just
a portability-review test failure — it meant these 7 domains could
*structurally never* auto-classify evidence into their own required
lanes. Added all 19 alias entries. Also fixed one more duplicate-phrasing
test bug in the same file (CLI-stdout check wanted different wording than
the file-content check for the same field — same class of bug fixed
earlier in `test_build_focus_review_packet.py`).

**`test_pipeline_control_metric_artifact_candidates.py` — resolved as
"leave failing, do not weaken the safety gate":** traced the full chain
and found `dean_os/IMPLEMENTATION_STATUS.md:586` explicitly documents this
exact behavior as intentional: *"Closed a provenance hole: complete-looking
JSON with familiar metric names is no longer classified or materialized
as locked evidence. Accepted artifacts must prove the exact locked
artifact class, same-window model lineage or measured feature-stability
assembly, complete lineage, and non-synthetic origin."* `write_pipeline_control_metric_artifact_candidates`
(called for real by `src/pipeline/stages/modeling/orchestrator.py` —
confirmed live) deliberately writes *candidates*, not locked artifacts;
line 813 of the same doc says cautions "remain until real locked artifacts
exist" by design. The only sanctioned candidate→locked promotion path
(`PipelineControlLockedEvaluationAssembler`) is wired for a *different*
pairing (Stage-4 training candidate + Stage-7 evaluation candidate), not
model_evaluation+feature_stability. **Conclusion: do not loosen
`verify_locked_model_evaluation`/`_classification` to accept
`contract_status` as an alternative proof — that would reopen the exact
provenance hole this project already deliberately closed.** This test is
either testing a fixture that should itself simulate a real assembled/
locked artifact (not raw candidates), or testing a model_evaluation+
feature_stability-specific assembler that was never built. Left failing,
not silently patched around.

**Remaining 12 failures were 12; now 10 after this pass:**
1. `test_agent_cli_restore.py::test_documented_run_agent_wrappers_exist` (1)
   — wants all 192 documented wrappers; explicitly out of scope.
2. `test_staged_workbench_integration_review.py` (1) — missing draft
   bundle directory on disk.
3. `test_pipeline_control_metric_artifact_candidates.py` (1) — resolved
   as "leave failing", see above — not a bug, don't touch the classifier.
4. `test_historical_research_replay.py` + `_batch.py` (7) — API-drift
   issue, running as background task `task_6a4b1774` in a separate
   session as of this writing.

**Next steps:** move to the next `dean_os/` module per the standing
module-by-module plan (analyst_core/, analysts/, pipeline_control/,
replays/, world_model/, packets/, risk/, strategies/, stress/,
observability/, execution/, evals/, pipeline_tuning/, research_corpus/,
plus the many top-level `dean_os/*.py` review/governance modules not yet
individually audited), then `src/`'s ~30 subdirectories once dean_os is
done. Given the scale, continue the established rhythm: recon pass
(subagent or direct grep/read) on the next unreviewed directory, verify +
fix findings personally, commit, update this memory, repeat.

**Foundational-modules pass (schemas/consensus/anxiety_kill_switch/world_state)
complete, 2 of 8 subagent findings fixed (2026-07-25, commits `a66de3dc`,
`13d3858a`):**
- **Anxiety kill-switch Trigger 5 was effectively defeated**: it read
  `len(decision.agent_report_hashes)`, which counts every review-only
  domain-analyst report that always runs and never moves `final_score`.
  With 8+ such agents enabled, `min_active_agents` cleared even when zero
  decision-relevant guardians (risk/regime/etc.) responded — the
  "too few agents" safety trigger could almost never fire in practice.
  Added `ConsensusDecision.decision_influencing_agent_count` (computed in
  `ConsensusEngine.combine()` from reports passing `_has_decision_influence()`)
  and switched the kill-switch to read that instead of the raw hash count.
- **`world_state.py`**: `geopolitics_analyst`/`liquidity_credit_analyst`
  are real, enabled `DomainAnalystAgent` registry entries but were entirely
  absent from `DOMAIN_SECTOR_MAP`/`_sector_id_from_agent` — their
  stance/confidence/thesis were silently dropped from the Stage 7 world
  state snapshot instead of appearing as their own sector. Also removed a
  dead `"macro_policy"` mapping entry: `macro_analyst` returns early into
  `global_state.macro_stance` before that mapping is ever consulted, so it
  could never fire — leftover from before that early-return existed.
- Added `tests/dean_os/test_world_state_builder.py` and
  `tests/dean_os/test_consensus_decision_influencing_count.py` (4 new
  tests), full `tests/dean_os` suite re-run afterward: still exactly the
  same known 10 failures, zero regressions.
- **Finding #6 (hardcoded `hard_veto_agents` classvar duplicated in
  `consensus.py` vs. registry-driven set wired only in `factory.py`):
  confirmed via grep — `DEANOrchestrator(...)` is only constructed
  directly in test files today, zero live production impact. Deferred,
  not fixed.**
**Findings #2, #3, #7, #8 triaged (2026-07-25) — one real fix, three
confirmed non-issues:**
- **#8 REAL, fixed**: `AnalyticalBranch._run_agent` in `dean_os/branches.py`
  only logged a failed agent when `error_behavior == "warn"`; for the far
  more common `error_behavior: skip` (every enabled `branch: analytical`
  agent in the registry uses `skip`), an exception was swallowed with zero
  logging at all — unlike `PipelineBranch._handle_error`, which logs at
  `info` level even for `skip`. Same "error_behavior: skip means the crash
  is never even logged" pattern as the earlier `CoherenceScanAgent` bug
  this session. Added the matching `else: logger.info(...)` branch.
- **#2 confirmed NOT live**: `ConsensusEngine._find_hard_veto` does return
  only the first blocked hard-veto report, but this can never matter in
  practice — `PipelineBranch.run()` (`branches.py`) breaks immediately on
  the first `can_veto` agent that returns `verdict == "blocked"`, and
  `can_veto` is `True` exactly when `veto_level == "hard"` (`base.py`).
  So at most one hard-veto agent's blocked report can ever reach
  `pipeline_reports` in a real run.
- **#3 confirmed NOT live**: `registry.hard_veto_agent_names()` (used by
  `factory.py` to build `ConsensusEngine`) filters on `veto_level=="hard"`
  + `enabled`; `_is_hard_block_agent()` (used internally by the registry
  for synthetic blocked reports) additionally checks
  `error_behavior=="block"`. Checked every `veto_level: hard` entry in
  `agent_registry.yaml`: all of them (`pipeline_audit`, `data_quality`,
  `risk`, plus disabled `agent_evaluation_controller`) already have
  `error_behavior: block`, so the two criteria always agree today. Latent
  risk if a future hard-veto agent is configured without
  `error_behavior: block`, but not worth a speculative registry-validation
  fix for a scenario that doesn't exist yet.
- **#7 confirmed NOT live / working as designed**: `_report_score`'s
  hardcoded default only fires for `"risk"` (never reachable — `risk` is
  enabled, hard-veto, `error_behavior: block`, so it always produces a
  report or a synthetic blocked one via `registry.get_synthetic_reports()`
  merged in `orchestrator.py`) or `"regime"` (reachable whenever
  `context.phase != "pre_trade"`, since `regime`'s `run_phases:
  [pre_trade]` excludes it from other phases — but `regime` is
  `veto_level: soft`/`shadow_mode: true`, so defaulting its 25%-weighted
  score contribution to neutral (0.0) when it didn't run this phase is a
  reasonable, intentional default, not a masked guardian failure).

**`historical_research_replay.py` — real, substantial fix, not test drift
(2026-07-25):** the API-drift question from the earlier background task
(no longer trackable — the task ID was lost across the compaction) turned
out to be a genuinely broken, half-finished migration, not stale test
assertions. Root cause: `dean_os/historical_research_replay.py` was still
the OLD pre-"research exam" file — a near-byte-identical duplicate of
`dean_os/historical_replay.py` (simple price-only replay). The *actual*
richer "research exam" version (wrapping `AnalystEvidencePackRunner` +
`AgentLabRunner` + price replay + an optional ticker-focused overlay) only
existed in `dean_os/draft/dean_os_agent_system_v7/dean_os/historical_research_replay.py`
under the class name `HistoricalResearchReplayRunner` — it was drafted but
never promoted to the live file. Three independent pieces of evidence
proved this was live/real, not aspirational: (1) `tests/dean_os/test_historical_research_replay.py`
imports `HistoricalReplayRunner` from the live module and calls it with
the rich signature (`news_data_paths`, `macro_data_paths`,
`focused_overlay_path`, etc.); (2) the already-live
`dean_os/historical_research_replay_batch.py` (`HistoricalResearchReplayBatchRunner`)
*already calls* `HistoricalReplayRunner` from the live module expecting
the rich payload shape (`research_exam`, `evidence_pack`, `price_replay`
keys) — meaning the batch runner itself was silently broken too; (3)
`dean_os/__init__.py`'s lazy-export table already had an entry for
`dean_os.historical_research_replay.HistoricalResearchReplayRunner`,
which would `AttributeError` today since only the old `HistoricalReplayRunner`
class existed. All the draft version's real dependencies
(`dean_os.agent_lab.AgentLabRunner`, `dean_os.analyst_core.analyst_evidence_pack.AnalystEvidencePackRunner`,
`dean_os.dean_paths.DeanPaths`, `dean_os.regime_context.normalize_context_tags`,
`dean_os.market_data_api.parse_datetime`) already exist live with matching
signatures — this was a promotion someone did 90% of (all the supporting
modules) and then missed the one file that ties them together. Ported the
draft's `HistoricalResearchReplayRunner` into the live file, renamed to
`HistoricalReplayRunner` (matching the two real live callers), with
`HistoricalResearchReplayRunner = HistoricalReplayRunner` kept as an alias
so `__init__.py`'s lazy export also resolves. Confirmed via grep that
nothing else imports the old simple-replay class names
(`ReplayDataGuardResult`/`HistoricalReplayAnalyst`/`guard_replay_frame`)
from this specific module — they still live on unchanged in
`dean_os/historical_replay.py`. **All 7 previously-failing tests now
pass** (`test_historical_research_replay.py` x4,
`test_historical_research_replay_batch.py` x3).

**Result: 10 → 3 known dean_os failures, confirmed clean (2026-07-25,
commits `e6f626a2` branches.py, `bb4f40c0` historical_research_replay.py):**
full suite: `3 failed, 1211 passed` (was `10 failed, 1204 passed`) — zero
regressions, only the 7 historical_research_replay tests newly passing.
Remaining 3, all previously triaged and deliberately left:
`test_agent_cli_restore.py::test_documented_run_agent_wrappers_exist`
(out of scope — wants all 192 documented wrappers),
`test_staged_workbench_integration_review.py` (missing draft-bundle
directory on disk, `staged_block_count` is 0 vs expected `>=30`),
`test_pipeline_control_metric_artifact_candidates.py` (resolved as "leave
failing, don't weaken the safety gate" — see above).

**dean_os/ core audit is essentially done for now** — schemas/consensus/
anxiety_kill_switch/world_state/branches/registry/factory/agents/ all
reviewed, all real findings fixed, remaining 3 failures are deliberate
non-fixes with documented reasoning.

**dean_os/analyst_core/ pass complete (2026-07-25, commits `a820e9a7`,
`0889b07c`, `ddb5aff2`):** recon subagent surveyed all 39 files, returned
3 confirmed-live findings (all fixed) + 3 lower-priority/unconfirmed ones
(left as-is, documented below). Full suite re-confirmed clean after each:
`3 failed, 1211 passed` — same 3 known non-fixes, zero regressions.
- **`artifact_evidence_loader.py`**: `load_evidence_from_artifacts()`
  computes/defaults `as_of` specifically so `_validated_producer`'s
  as-of-consistency check has something to compare against, but only
  forwarded it into the `from_producer_artifacts` branch — the
  `from_runtime_artifact` branch call never passed `as_of` at all, so its
  own `expected_as_of` consistency check (catching a `--as-of` that
  doesn't match the runtime artifact's baked-in cutoff) silently no-opped
  for every real `run_analyst.py --runtime-artifact` invocation. Fixed by
  passing `as_of` through.
- **`domain_analyst_intake_packet.py`**: `_domain_relevant()` — the
  fallback check deciding whether an unclassified document should be
  dropped as `outside_domain_scope` — matched document text against
  `EVIDENCE_TYPE_ALIASES.values()`, i.e. the union of every alias across
  all 15 domain profiles (including ultra-generic single words:
  "demand", "market", "stock", "supply", "policy", "china", "inflation").
  Virtually any financial document matches one of those, so the
  domain-scope filter almost never fired — a document that already failed
  to match the *current* domain's own required/useful evidence types got
  kept anyway because it hit some *unrelated* domain's alias term.
  Live: `DomainAnalystIntakePacket` is used by
  `run_agent_domain_analyst_intake_packet.py` and lazily exported from
  `dean_os/__init__.py`; no existing test exercised this path. Fixed by
  scoping the check to just `[*profile_required_types, *profile_useful_types]`
  (same candidate list `_classify_evidence_type` already uses).
- **`cross_domain_signal_bus.py`**: `CROSS_DOMAIN_PROPAGATION`'s
  `target_domains` used `"industrial"`, `"consumer"`, `"financials"` —
  none match a real `domain_id` from
  `dean_os.domain_profiles.list_domain_ids()` (real ones:
  `"industrials"`, `"consumer_discretionary"`/`"consumer_staples"`, no
  `"financials"` domain at all — closest is `"liquidity_credit"`).
  `from_signal_bus()` silently drops any signal whose domain doesn't
  match a real one. Confirmed dormant today (`SectorAnalyst.enable_cross_domain_signal_bus`
  defaults `False`, nothing in config enables it) but fixed anyway since
  it's a cheap, low-risk string correction that prevents a guaranteed
  silent misfire the moment someone flips that flag on — same
  "reference list out of sync with the real domain registry" pattern as
  the issuer registry / capability-contracts / evidence-type-aliases bugs
  found earlier this session.
- **Not fixed, documented for awareness only** (per subagent's own
  "unconfirmed/lower priority" flagging, independently spot-checked):
  `lens_orchestrator.py`'s `expectation_gap` field lacks the
  single-owner-invariant guard other overwrite fields have, but
  `ExpectationGapLens` is deliberately excluded from
  `sector_analyst.py`'s production lens registry, so unreachable today;
  `sector_analyst.py:576-577`'s duplicate `payload.pop(field, None)` line
  (harmless — pop with default is idempotent, but looks like a
  copy-paste leftover, worth a second look if a different field was
  meant to be popped); `analyst_learning_apply_ceremony.py`'s
  `status="applied"` + `can_apply=False` combination — no caller reads
  `can_apply` for that status today, plausibly intentional.

**dean_os/analysts/ pass complete (2026-07-25, commit `6af2a3f5`):** recon
subagent read all 25 files + `_producers/` subpackage in full, ran
`pyflakes` for undefined-name bugs (zero hits), and specifically
re-checked every `_producers/*.py` file for the same path-depth mistake
already fixed in `ticker.py` this session (`macro.py`/`news.py`/`policy.py`
all correct; `sec/companyfacts.py` correct but missing `.resolve()` —
harmless, not fixed). Found one real, live bug:
- **`context_adapter.py`**: `_structured_context_evidence()`'s hardcoded
  `macro_domains` gate only allowed `{macro_policy, liquidity_credit,
  energy, semiconductor_ai_infrastructure}` through before a macro
  observation could even reach `_macro_series_evidence_type()` — but
  `MACRO_SERIES_EVIDENCE_MAP` (same file) has explicit, intentional
  entries for `real_estate`, `agriculture`, `logistics`, and `geopolitics`
  too (e.g. `cpi`→`real_estate`, `wti_crude_oil`→`agriculture`/`geopolitics`).
  Every macro observation for those 4 domains was silently excluded at
  the gate, making those map entries dead code. Reproduced live:
  `MarketContextEvidenceAdapter("real_estate").adapt(...)` on a `cpi`
  observation returned 0 evidence items with reason
  `structured_family_not_relevant_to_domain`, despite the map having a
  `cpi`→`real_estate` entry. Live because `MarketContextEvidenceAdapter(domain_id)`
  is instantiated generically by `SectorAnalyst`/`DomainAnalystRuntime`
  for every registered domain analyst, including `agriculture_analyst`/
  `logistics_analyst`/`real_estate_analyst`/`geopolitics_analyst` in
  `agent_registry.yaml`; no existing test covered these 4 domains for
  this path (`test_context_adapter_macro_evidence_type.py` only tests
  `macro_policy`/`semiconductor_ai_infrastructure`). Same "stale
  hand-maintained reference list" pattern as several fixes this session
  — this file's own comments even document two *previous* fixes of this
  exact same failure mode in this exact same function. Fixed by deriving
  `MACRO_RELEVANT_DOMAINS` directly from `MACRO_SERIES_EVIDENCE_MAP`'s own
  keys instead of a hand-maintained duplicate, so it can't drift again.
  Full suite re-confirmed clean: `3 failed, 1211 passed`, same 3 known
  non-fixes.
- Everything else in `dean_os/analysts/` (`base.py`, `profiles.py`,
  `schemas.py`, `quality_gates.py`, `sector_bridge_adapter.py`,
  `ticker_bridge.py`, `domain_feeder.py`, `outcome_tracking.py`,
  `review_packet.py`, `markdown.py`) read in full — no other
  contract mismatches found.

**dean_os/pipeline_control/ pass complete (2026-07-25, commit `50ad7f85`):**
recon subagent read all 22 files, ran `pyflakes` (zero hits), and
specifically re-confirmed (without re-litigating) that
`test_pipeline_control_metric_artifact_candidates.py` remains a deliberate
non-fix per last round's reasoning. Cross-checked every producer/consumer
boundary in the chain (evidence_inventory ↔ materializer ↔ both locked
assemblers ↔ real_metric_evidence_run ↔ surface/instance/caution_review,
plus the forward-data-accrual and saved-price-repair sub-chains) — all
consistent, no other contract mismatches found. One real, live bug:
- **`pipeline_control_instance_contract.py` + `pipeline_control_caution_review_packet.py`**:
  both files' `_load_json()` (required-input loader, not the optional-input
  one) raised `FileNotFoundError`/`json.JSONDecodeError` uncaught, unlike
  every other stage in the same fixed chain (`evidence_inventory.py`,
  both locked assemblers, `metric_artifact_materializer.py`,
  `real_metric_evidence_run.py`, `pipeline_metric_input_readiness_gate.py`,
  `pipeline_control_surface.py`), which all treat a missing/corrupt
  artifact as a normal "blocked"/"caution" condition. Live: both are
  wired directly in `pipeline_control_real_metric_evidence_run.py` (itself
  called from `pipeline_control_bounded_evidence_run.py`) and via their
  own `run_agent_*.py` CLI wrappers, whose default paths
  (`reports/dean_os/.../latest.json`) don't exist on a fresh checkout.
  No existing test covered the missing/corrupt-input path for either file.
  Fixed by catching the load error and returning a degraded dict — every
  downstream reader already uses `.get()` defensively, so this alone makes
  both stages fall through to their existing "blocked" status logic with
  zero other changes needed. Verified live: reproduced both crashing
  before the fix, both gracefully returning `blocked_pipeline_control_instance`/
  `pipeline_caution_review_blocked_by_hard_planes` after. Full suite
  reconfirmed clean: `3 failed, 1211 passed`, same 3 known non-fixes.

**dean_os/world_model/ pass complete (2026-07-25, commit `ae2fbd7f`):**
recon subagent read all 8 files in full (small directory, full coverage
expected and delivered). Confirmed this directory has **no** hardcoded
domain/sector mapping table of its own (unlike `world_state.py`) — every
file resolves domain context dynamically via `get_domain_profile(domain_id)`,
so no separate reference-list-drift risk here. One real, live bug found:
- **`world_model_pipeline_context.py`'s `_ticker_matches()`**: classic
  `[x] or y` truthiness trap — `scope.get("tickers") or [scope.get("ticker")]
  or payload.get("tickers") or []`. `[scope.get("ticker")]` is a list
  literal, always truthy even when it wraps `None`, so the third fallback
  (`payload.get("tickers")`) was **provably unreachable dead code**
  regardless of its contents. Real stage4 producers
  (`pipeline_stage4_exact_context_review.py`) always populate
  `scope.ticker` with a real value, which is why this happened to work in
  practice — but a malformed/incomplete stage4 artifact missing
  `scope.ticker` would silently synthesize a bogus `"NONE"` ticker via
  `_normalize_tickers([None])` instead of falling back to any top-level
  `tickers` list on the payload. Live: feeds
  `WorldModelPipelineContextDiscovery.build()`, exported via
  `dean_os/__init__.py`. No existing test exercised a payload missing both
  `scope.tickers` and `scope.ticker`. Fixed by only wrapping
  `scope.get("ticker")` in a list when it's actually truthy. Full suite
  reconfirmed clean: `3 failed, 1211 passed`.
- Two lower-confidence/dormant findings noted but not fixed (cosmetic-only
  impact, no observed wrong behavior today): `world_model_replay_registration.py`'s
  `_sectors()` colon-tag parsing (`sector:`/`domain:` prefix convention)
  is dead code because this pipeline's own tag generators never emit that
  format; `hypothesis_ledger_lens.py`'s `default_horizon_days` config key
  is never set by its only real caller, always silently using the
  hardcoded default of 20 (which happens to already match another
  hardcoded 20 elsewhere in the same file, so no visible defect).

**CORRECTION (2026-07-26) to the replays/ pass below**: the "Half B is
dead, zero references outside its own tests" conclusion was wrong —
confirmed by finding `reports/dean_os/chief_review_index/` etc. contain
real, dated artifacts from 2026-06-28 through 2026-07-13 (13 days before
this correction), proving the chain has actually run in practice, not
just in tests. Root cause of the wrong conclusion: this whole "chief
review cycle" governance layer (`dean_os/chief_review_index.py`,
`full_system_cycle_closure.py`, `current_architecture_map.py`,
`current_cycle_journal.py`, plus CLI wrappers
`run_agent_current_architecture_map.py`, `run_agent_current_cycle_journal.py`,
`run_agent_replay_calibration_readiness.py`,
`run_agent_historical_evidence_backfill.py`) is wired via **file-path
artifact handoff on disk** (e.g. `ChiefReviewIndexBuilder` reads
`reports/dean_os/replay_checkpoint_due_router_current/latest.json` by
default path), not via Python imports — so grepping for Python-level
callers (the method used both by the earlier recon subagent and by
`diagnostics/config_reachability_checker.py`/`dead_code_classifier.py`,
[[project-colab-pipeline-audit]]'s reachability tool fixed 2026-07-26)
systematically cannot see this reachability pattern at all. **This is a
real blind spot in that diagnostics toolkit, not just this one past
conclusion** — any dean_os subsystem wired by CLI-script-writes-JSON /
next-CLI-script-reads-JSON handoff will show as a false-positive orphan
in `orphan_modules.txt`/`dead_code_classification.csv`. Do not trust
those reports alone for dean_os's CLI-chained governance layer; check
`reports/dean_os/<name>_current/` for dated real artifacts before
concluding something is dead. **Asked the user directly (2026-07-26):
confirmed they still run this chief-review cycle manually today** — it
is live, not a candidate for archival. Documented this blind spot
directly in `diagnostic_reports/AUDIT_GUIDE.md` so future sessions (and
the diagnostics toolkit's own users) don't repeat this exact false
"dead code" conclusion for this layer.

**dean_os/replays/ pass complete (2026-07-25, commit `7cb520a5`):** recon
subagent read all 14 files. Key structural discovery: this directory
splits into two halves. Half A (`historical_replay_batch.py`,
`replay_price_normalizer.py`, `replay_price_artifact_repair.py`,
`replay_price_quality_investigation.py`, `replay_evidence_window_selector.py`)
is genuinely live — `resolve_as_of_dates()` is imported directly by
`dean_os/historical_research_replay_batch.py`. Half B (`replay_checkpoint_due_router.py`,
`replay_checkpoint_monitor.py`, `replay_evaluation_router.py`,
`replay_evidence_refresh_controller.py`, `replay_lifecycle_journal_bridge.py`,
`replay_outcome_evidence_plan.py`, `replay_outcome_lifecycle_orchestrator.py`,
`replay_calibration_readiness_gate.py`, `historical_replay_outcome_review.py`,
plus `world_model/world_model_replay_registration.py`) is a complete,
heavily-tested "world-model replay lifecycle" chain with **zero**
references in `orchestrator.py`/`factory.py`/`agent_registry.yaml`/any
`run_agent_*.py` wrapper — confirmed by grep. It only exercises itself via
its own tests. Worth noting for future rounds: this explains why bugs can
sit undetected in fully-tested code — nothing outside the tests exercises
these paths yet.
- **Fixed**: `replay_lifecycle_journal_bridge.py`'s `domain_id` fallback —
  `lifecycle["inputs"]` never actually contains a `"domain_id"` key (its
  real shape only has `as_of/registration_json/review_gate_json/packet_json/
  verified_price_paths/pipeline_paths/prior_outcome_json_paths/journal_path`),
  so every journaled event always fell through to the hardcoded
  `"semiconductor_ai_infrastructure"` default regardless of the actual
  domain. The real value was already loaded in memory
  (`registration["source_packet"]["domain_id"]`) but only used for
  `artifact_binding()`, never read for `domain_id`. Fixed anyway despite
  the chain being dormant today (cheap, low-risk, prevents a guaranteed
  mis-tag the moment this gets wired up) — same reasoning as the
  `cross_domain_signal_bus` fix.
- **Deferred, documented, not fixed** (per standing "check if live"
  rule — genuinely ambiguous or out of established scope):
  (a) `replay_checkpoint_due_router.py:79-81`'s gate-SHA256 verification
  is silently skipped whenever `source_gate.sha256` is `None`/empty
  (`if bound_gate_sha and bound_gate_sha != _sha256(gate_path): raise` —
  the `bound_gate_sha and` guard means a missing sha bypasses the check
  entirely), inconsistent with `historical_replay_outcome_review.py`'s
  `_verify()` which raises unconditionally on any mismatch including
  `None`. Not fixed: whole chain is dormant, and it's unclear whether "no
  bound sha" is meant to mean "trust it" (current behavior, matches an
  existing test that constructs registration without a file path) or
  "reject it" (tightening this could break that test's assumption) —
  a genuine design-intent question, not a clear-cut bug, left for
  whoever wires this chain up to resolve deliberately.
  (b) Missing `run_agent_*.py` CLI wrappers for 6 replay tools
  (`historical_replay_batch`, `historical_research_replay_batch`,
  `replay_price_normalizer`, `replay_price_quality_investigation`,
  `replay_price_artifact_repair`, `evidence_gap_plan`) whose own
  self-documented "next command" text points at scripts that don't
  exist — same class as the ~139 already-deferred wrappers from the
  70-failure triage earlier this session; explicit prior user decision
  was to only build wrappers real tests exercise, not chase doc-completeness.
  (c) `replay_evidence_window_selector.py`'s `DEFAULT_PRICE_ARTIFACT`
  hardcodes a dated filename that exists today but has no "latest"
  glob-and-sort fallback the way `historical_replay_outcome_review.py`'s
  `_default_price_paths()` does — will `FileNotFoundError` the moment
  that specific file is rotated out. Low urgency, noted for whoever next
  touches replay artifact rotation.
- Full suite reconfirmed clean: `3 failed, 1211 passed`.

**dean_os/packets/ pass complete (2026-07-25, commit `5585b0bc`):** recon
subagent read all 11 files plus traced every packet class to its real
caller (or lack thereof). Found the directory splits similarly to
replays/ — several packet classes (`RealSourceNormalizedPacketBuilder`,
`ContextEvidenceReviewPacket`, `SourceExtractionFixturePacket`/
`SourceExtractionReviewPacket`, `SpecialistContextReviewPacket`,
`SectorToTickerReviewPacket`, `DomainSpecialistReviewPacket`,
`ReviewDecisionPacket`) have zero live callers outside their own tests —
no `run_agent_*.py` wrapper exists for them. `pipeline_model_case_packet.py`/
`pipeline_model_feedback_packet.py` are genuinely live and were checked
thoroughly — no bugs found.
- **Fixed**: `staged_workbench_integration_review.py`'s `_main_repo_alignment()` —
  `required_paths` hardcoded pre-refactor locations for 7 modules directly
  under `dean_os/` (e.g. `dean_os/real_source_normalized_packet.py`,
  `dean_os/analyst_evidence_pack.py`, `dean_os/domain_analyst_intake_packet.py`)
  that actually live under `dean_os/packets/` or `dean_os/analyst_core/`
  now — every one of these real, live modules was falsely reported as
  missing in `missing_target_path_ids`. Live:
  `run_agent_staged_workbench_integration_review.py` calls this function
  directly; verified via direct execution that `missing_target_path_ids`
  went from 10 false positives down to just the 3 genuinely-missing CLI
  wrappers after the fix. This is a *different* bug from the file's own
  known, deliberately-left test failure (`staged_block_count` — missing
  draft-bundle directory on disk) — that one is untouched and still fails
  identically. Full suite reconfirmed clean: `3 failed, 1211 passed`.
- **Deferred, documented, not fixed**: `specialist_context_review_packet.py`
  reads `candidate.get("manual_review_decision")`, a field no real
  producer (`sector_to_ticker_review_packet.py`'s `_ticker_review_item()`)
  ever writes — making the packet's designed "success" status
  (`specialist_context_exact_match_ready`) permanently unreachable dead
  code by construction. Confirmed dormant: `SpecialistContextReviewPacket`
  has no live caller anywhere. Not fixed because the correct fix requires
  a design judgment (which field should represent "manually approved",
  not just a reference-path correction) rather than a mechanical fix —
  left for whoever wires this packet up to resolve deliberately, same
  category as the `replay_checkpoint_due_router.py` gate-SHA asymmetry
  from last round.
- Several packets' own self-documented "next command" suggestions point
  at CLI wrappers that don't exist (`run_agent_real_source_normalized_packet.py`,
  `run_agent_review_decision_packet.py`, `run_agent_sector_to_ticker_review_packet.py`,
  `run_agent_domain_specialist_review_packet.py`) — same already-deferred
  "~139 undone wrappers, don't chase doc-completeness" category from
  earlier this session.

**dean_os/{risk,strategies,stress,observability,execution,evals}/ pass
complete (2026-07-25, commit `349b1a49`):** recon subagent covered all 6
small directories (~12 files) in one pass, confirmed reachability for
each: only `strategy_playbook.py`/`maturity_gates.py` (via
`strategy_maturity_operations.py` → 2 real `run_agent_*.py` CLIs) have
live callers outside their own tests; `risk_engine.py`, `strategy_registry.py`,
`scenario_library.py`, both `observability/` files, most `evals/` files,
and `execution/execution_gateway.py` itself are dormant (their sole
non-test consumer, `dean_os/archive_v1/daily_governor.py`, is itself dead
— imports from `dean_os/draft/`, zero callers anywhere).
- **Fixed**: `dean_os/__init__.py`'s lazy-export table —
  `"dean_os.execution.execution_gateway": ("ExecutionGateway", "ExecutionPolicy")`
  — but the new `execution/execution_gateway.py` (the live "fail-closed
  rewrite", confirmed used by `dean_os/stress/test_phase8.py`) only
  defines `ExecutionGateway`/`OrderRequest`/`OrderResult`/`OrderDecision`;
  `ExecutionPolicy` only exists in the old, separate root-level
  `dean_os/execution_gateway.py` (superseded API,
  `process(ConsensusDecision)` vs the new `submit(OrderRequest, ...)`,
  referenced only by its own test). Reproduced live:
  `dean_os.ExecutionPolicy` raised a confusing AttributeError pointing at
  the wrong module. No current caller uses it, but this is the package's
  public lazy-export surface — same dangling-export-after-a-module-split
  pattern as the `CollectorInventoryAgent` fix earlier this session.
  Removed the stale entry. Full suite reconfirmed clean:
  `3 failed, 1211 passed`.
- **Deferred, documented, not fixed** (design-judgment questions, not
  mechanical fixes): (a) `StrategyStatus` (in `strategy_playbook.py`) and
  `MaturityLevel` enums are meant to track the same maturity concept but
  their string values never actually match (e.g.
  `"constrained_autonomous_candidate"` vs `"constrained_autonomous"`),
  and `StrategyStatus` has 4 terminal values with no `MaturityLevel`
  counterpart at all — reachable via the live
  `StrategyMaturityDailyReconciler.build()` but latent today because
  fresh playbooks start at `research`/`research` (which happen to
  match); will spuriously flag `playbook_status_does_not_match_registry_maturity`
  the moment a strategy's playbook is hand-edited past that stage. (b)
  The maturity ladder (`MATURITY_ORDER`/`GATE_CHECKS` in
  `maturity_gates.py`, `level_order` in `strategy_maturity_operations.py`)
  is hardcoded to 5 stages and is missing `constrained_autonomous` (a
  legitimate `MaturityLevel` value) in all 3 places — if ever passed as
  `target_gate`, returns an error dict with no `"receipt"` key, which
  would `KeyError` in `StrategyReplayCandidateAssessment.build()`; no
  live call site currently passes that target, so this is real-but-latent,
  not actively firing. Both require deciding what the new maturity
  stage's actual promotion criteria should be (business logic), not just
  a reference-list correction — left for whoever owns strategy-maturity
  design to resolve deliberately.

**dean_os/pipeline_tuning/ + dean_os/research_corpus/ pass complete
(2026-07-25, no code changes — valid "nothing live to fix" result):**
- **`pipeline_tuning/` re-investigated in depth (2026-07-25) — NOT dead
  code, an unfinished better redesign, deliberately left alone (not
  archived).** Initial recon called it "superseded by TuningAgent", but
  checking git history changed the read: `dean_os/agents/tuning.py:TuningAgent`
  was restored from git history 2026-06-11 (older) and is itself currently
  `enabled: false` in `agent_registry.yaml` — i.e. neither implementation
  is live in production today. `pipeline_tuning/` was added in commit
  `787dc294` (2026-07-22), the SAME commit that added `pipeline_control/`,
  `world_model/`, `replays/` — subsystems that *did* get wired up. Its
  design is meaningfully more rigorous than `TuningAgent`'s single flat
  status/proposal flow: explicit `TuningPlaneProfile` objects
  (`model_selection`, `feature_space`, `hyperparameters`,
  `ensemble_weights`, `risk_thresholds`), each with its own
  `allowed_parameters`/`max_change_pct`/`required_preconditions`/`blocked_if`
  — a real, structured model for bounded per-plane tuning experiments that
  `TuningAgent` doesn't have. The only thing missing is the integration
  "glue": `PipelineTuningPlanner` is not a `BaseAgent` (no `async run()`,
  no `PipelineReport` output), so it was never hooked into the
  orchestrator/registry. **Conclusion: leave both files in place,
  untouched** — archiving would destroy real, better-designed unfinished
  work; finishing the integration (deciding whether to wrap
  `PipelineTuningPlanner` in a `BaseAgent` and replace/merge with
  `TuningAgent`) is an architecture decision for the eventual
  architecture-review phase, not a bug-audit fix. Also still true: docs
  (`system_audit_summary.py`/`review_index.py`/`chief_review_index.py`)
  reference a nonexistent `dean_os/agents/pipeline_tuning_controller.py`
  and a stale report path — confirmed dormant-on-dormant, not fixed.
- **`research_corpus/` (4 of 6 files live, all sound)**: traced
  `hypothesis_measurement_policy_preparer.py`,
  `hypothesis_quality_assessment.py`, `hypothesis_learning_review.py`,
  `hypothesis_reverse_analysis.py` to real callers in
  `world_model_hypothesis_lifecycle_orchestrator.py`/
  `world_model_replay_review_gate.py`/`replay_outcome_lifecycle_orchestrator.py`
  and cross-checked every field contract end-to-end — no live defect
  found (this area already had careful scrutiny in prior passes). Found
  one dead-code duplication (`hypothesis_learning_review.py`'s
  `_diagnose_errors`/`_ASSESSMENT_ERRORS` — never called, superseded by
  `hypothesis_reverse_analysis.py`'s live equivalent) — cleanup
  opportunity, not a bug, left alone. `hypothesis_evidence_gap_review.py`/
  `hypothesis_gap_replay_packet.py` (top-level compat re-export shims)
  are dormant (tested, unwired) but internally sound.

**MAJOR correction to earlier session's "192 documented wrappers, only
~53 exist, 139 out of scope" conclusion (2026-07-25, commit `53e01c3e`):**
that framing was wrong for at least 52 of them. Commit `e34650e0`
("chore: repo root cleanup — remove stray audit/analysis artifacts",
2026-07-22, 3 days before this discovery) deleted 52 real, working
`run_agent_*.py` CLI wrappers under the same commit as genuine junk
(`scratch/`, `audit_reports/`, `mlruns/`, `category_*_analysis.md`) — its
own message mischaracterized them as "stray scripts." These were **not**
"never built" (aspirational docs) — they were built, worked, and got
swept up by an overly-broad cleanup. Discovered via a recon subagent
finding `outcome_readiness_gate.py`/`outcome_price_coverage_plan.py`
(confirmed-live modules) still recommending commands like `python
run_agent_outcome_readiness.py ...` that no longer exist. Verified via
`git show e34650e0 --stat` that 52 `run_agent_*.py` files were deleted in
that commit, and confirmed none had been recreated since (diffed against
today's file list).
- Asked the user whether to restore all 52, restore only the subset with
  currently-live "next command" references, or just document — question
  went unanswered; proceeded with the stated recommended default
  (restore all 52) since it's purely additive/reversible (git rm undoes
  it trivially) and directly fixes broken operator guidance in live
  modules.
- Restored all 52 from `e34650e0^` (the parent commit, i.e. their
  last-good state). One, `run_agent_collector_inventory.py`, was
  deliberately left out: it wraps `CollectorInventoryAgent`, which this
  session independently confirmed dead and archived earlier (zero
  registry/orchestrator references) — no reason to resurrect a wrapper
  for already-archived dead code.
- 10 of the remaining 51 failed on `--help` after restoration: they
  imported from the old flat `dean_os.<module>` path for classes that
  moved into `dean_os/analyst_core/` during a later refactor
  (`analyst_calibration_gate`, `analyst_evidence_pack`,
  `analyst_learning_promotion_bridge`, `analyst_loop_daily_check`,
  `analyst_outcome_evaluation_loop`, `analyst_profile_orchestrator`,
  `analyst_review_inbox`, `analyst_profile_scorecard`,
  `analyst_learning_apply_ceremony`). Fixed each import path. Verified
  all 51 restored scripts run `--help` cleanly against the current
  codebase before committing. Full suite reconfirmed clean:
  `3 failed, 1211 passed`.
- **Correction to record**: the earlier "~139 undone, out of scope,
  don't chase" framing from the 70-failure triage pass should now read
  as "~139 minus these 52 (now restored) = ~87 still genuinely
  never-built/aspirational, still out of scope"; `test_agent_cli_restore.py::test_documented_run_agent_wrappers_exist`
  still fails (93 wrappers still missing) — unchanged verdict on that
  specific test, still deliberately out of scope, but the count moved in
  the right direction as a side effect of this fix.

**Top-level dean_os/*.py sweep started (2026-07-25) — 2 more real bugs
found and fixed, commits `85132b11`, `b2b013fb`:**
- **`review_actions.py`'s `void_action()` crashed on every call with a
  linked proposal**: `OperationQueue.reject()` requires non-empty
  `reviewer`/`reason` (raises `ValueError` otherwise), but `void_action()`
  called `.reject(action.linked_proposal_id)` with neither — a `TypeError`
  every time, not the intended guard. Live-caller angle: zero code callers
  exist anywhere (not even tests), but `review_action_apply_ceremony.py`'s
  own `_recommendations()` tells the human operator to "void an old
  action" as the standard remediation for a duplicate-action conflict —
  the system's own documented recovery path led straight into this crash
  for any human who actually followed it. Added a `reviewer` parameter
  (matching the pattern every other `ReviewActionStore` method already
  uses) and reused the already-computed non-empty `reason_text`. Verified
  end-to-end with a manual repro (promote → void, no crash, status
  becomes "voided").
- **`context_performance.py`'s `weak_contexts`/`strengths` were truncated
  by volume before miss-rate ever mattered**: `by_agent_context`/
  `by_agent_regime` were built via `_bucket_by_agent_and_tags(...,
  limit=limit)`, which sorts by `(completed_count, record_count,
  miss_count, hit_count)` — i.e. by volume — and truncates to the top-N
  *before* `_weak_contexts()`/`_strengths()` ever run their own
  miss-rate/hit-rate filter+sort on top. A low-volume-but-100%-miss-rate
  agent/context combo was silently invisible whenever `limit` or more
  other combos had higher raw counts — exactly the outlier this feature
  exists to surface. Live: `build_summary()` is called by `review.py`,
  `review_approved_learning_loop.py`,
  `analyst_core/analyst_outcome_evaluation_loop.py`,
  `analyst_core/analyst_calibration_gate.py` — all real, non-test
  callers; zero test file exercises `build_summary()` at all. Fixed by
  making `_bucket_by_agent_and_tags`'s `limit` optional, computing the
  full untruncated bucket set once, feeding that into weak/strength
  detection while still truncating separately for the
  `by_agent_context`/`by_agent_regime` display fields. Verified with a
  synthetic repro (12 high-volume 75%-hit buckets + 1 low-volume
  100%-miss bucket): invisible to `weak_contexts` before the fix, found
  after. Full suite reconfirmed clean both times: `3 failed, 1211 passed`.
- **Deferred, documented, not fixed**: two recon subagents each surfaced
  a lower-confidence "dormant, unwired subsystem" pattern again
  (`review_decision_state.py`'s `ALLOWED_TRANSITIONS` gap — confirmed
  intentional fail-closed by an existing test, not a bug;
  `industry_operational_metrics.py` never validates a record's own
  `domain_id` against the requested one, but is dormant/only used by its
  own test; the entire `domain_macro_collection_*` chain is internally
  consistent but has zero live callers outside tests). Also confirmed:
  the domain-scoped-envelope batch specifically re-checked for the
  "hardcoded wrong domain_id" bug class already found twice this session
  and found **zero** new instances — every file in that batch threads
  `domain_id` as a parameter and cross-verifies it recursively, no
  hardcoded lists to drift.

**CRITICAL, UNRESOLVED finding for the future `src/` audit phase
(2026-07-25) — same commit `e34650e0` also deleted 100+ `src/` files,
NOT yet investigated or touched:** while investigating the 52 deleted
`run_agent_*.py` wrappers above, `git show e34650e0 --stat --name-only`
revealed the same commit deleted well over 100 `src/*.py` files under
the same "cleanup" banner — including safety-critical modules:
`src/risk/risk_manager.py`, `src/risk/kill_switch/{manager,calculator,executor,alerts,config}.py`,
`src/risk/exposure_calculator.py`, `src/risk/metrics.py`,
`src/calibration/calibration_engine.py`,
`src/calibration/adaptive_confidence_calibrator.py`,
`src/optimization/hyperparameter_searcher.py`, `src/models/factory.py`,
plus dozens more across backtesting, feature engineering, meta-learning,
ensembling, validation. **Confirmed real, current impact**: `tests/test_risk_manager.py`,
`tests/unit/test_kill_switch_calculator.py`,
`tests/unit/test_calibration_engine.py` all currently fail with
`ModuleNotFoundError` (verified via `pytest --collect-only`) — this is a
live, present-day broken state in the general test suite (outside
`tests/dean_os/`, which is why this session's dean_os-scoped test runs
never surfaced it).
**Unlike the run_agent_*.py wrapper case, this does NOT look like a clean
"mistake" — there is real evidence of a mix**: e.g. `src/archive/risk/exposure_calculator.py`
already exists, suggesting someone deliberately archived that specific
module *before* this commit deleted the stale original from `src/risk/` —
i.e. at least some of these 100+ deletions may have been correct cleanup
of already-superseded duplicates, not accidental loss like the CLI
wrappers were. Determining which of the 100+ files are "correctly
removed, superseded elsewhere" vs. "mistakenly deleted, still needed"
requires per-file investigation, not a blanket restore.
**Deliberately NOT investigated or restored this session** — this is
`src/` scope (the actual trading pipeline / risk management / kill-switch
safety systems), explicitly out of this session's `dean_os`-focused sweep
per the standing plan ("then src/'s ~30 subdirectories once dean_os is
done"). **This should be the FIRST task when the src/ phase begins** —
given it touches kill-switch/risk-manager code, treat as high-priority,
not routine cleanup-review. Do not assume "cleanup commit = safe to
ignore" the way the run_agent_*.py case turned out; also do not assume
"restore everything" the way that case was resolved — investigate each
module's disposition (archived-elsewhere vs. lost) before acting.

**dean_os paper trading / outcome tracking pass complete (2026-07-25):**
the finding above (52 deleted wrappers) came from this pass — recon
subagent otherwise found the receipt→plan→result→review lifecycle chain
(`review_decision.py` → `paper_simulation_plan.py` →
`paper_simulation_result.py` → `post_paper_simulation_review.py`, gated
by `paper_lifecycle_contract.py`) unusually rigorous, no live
safety-weakening defect found there. Other findings, documented not
fixed:
- **`OutcomeTracker`'s paper-trade bridge is dormant/unintegrated**:
  `register_paper_trade()`/`check_paper_trades()` implement a full
  "register → check interval outcomes → hit/miss" bridge, but grepping
  the whole repo shows `register_paper_trade` is never called by
  `paper_trading.py`/`paper_portfolio.py`/`paper_autonomy.py` — the real
  paper-trading pipeline uses a separate, actually-wired mechanism
  (`PaperTradeStore.update_outcome()`/`PaperTradeEvaluationRunner` in
  `paper_trading.py`). Two parallel, non-integrated outcome-tracking
  mechanisms exist for paper trades; only one is live. Some docs
  (`TEMPLATE_KIT.md`, `.agents/deepseek_session.md`) describe the dormant
  one as if it fires automatically — inaccurate for current wiring. Not
  fixed: deciding which mechanism should be canonical (or whether to
  merge them) is an architecture decision, not a bug fix.
- Cosmetic, not fixed: three separately-maintained copies of the same
  `_record_tickers()` intersection logic (`paper_trading.py`,
  `outcome_evaluation.py`, `paper_portfolio.py`) produce a misleading
  `"...has no tickers."` message when a record's tickers exist but just
  don't match a caller's `--tickers` filter; a dead fallback branch in
  `paper_simulation_plan.py` (`source_ready_decisions` set) that
  `paper_lifecycle_contract.py`'s stricter upstream check makes
  unreachable in practice.

**dean_os world_model_*/hypothesis_*/replay_*/historical_* top-level
batch complete (2026-07-25) — 18 of 39 files confirmed as correct thin
compat shims, 2 findings, both deferred (design-judgment required, not
mechanical fixes):**
- **`sector_thesis_to_ticker_basket_bridge.py`'s `_current_bridge_summary()`
  hardcodes `bridge_status: "ticker_pipeline_inputs_incomplete"` and
  `sector_stance: "mixed"` as permanent literals**, never varying despite
  computing rich per-ticker readiness data (`ticker_evidence_ready_count`,
  `negative_case_count`, `timeframe_mismatch_tickers`, etc.) right above.
  Live: this is the "current runtime-linked mode" per
  `current_architecture_map.py`, consumed by `packets/sector_to_ticker_review_packet.py`
  which surfaces it verbatim to a human review packet. HOWEVER: traced
  the actual `candidate_status` enum this specific mode's ticker
  candidates can take (`_current_ticker_candidates()`, ~line 1278-1285)
  and found it only ever produces two values —
  `"ticker_evidence_ready_pipeline_blocked"` or
  `"blocked_missing_ticker_evidence"` — **there is no "fully ready"
  terminal candidate_status defined anywhere in this code path**, unlike
  the sibling older `build()` path's 4-state enum
  (`direct_ticker_thesis_ready`/`ticker_context_ready`/
  `blocked_missing_ticker_evidence`/`sector_context_only`) that the
  dynamic `_summary()` correctly maps to a varying `bridge_status`. So
  it's genuinely ambiguous whether the hardcoded literal is (a) a bug —
  someone stubbed placeholder values (`direct_ticker_thesis_ready_count: 0`,
  `evidence_limited_direct_candidate_count: 0` are ALSO hardcoded to 0)
  intending to add a "ready" detection later and never did, or (b) an
  intentional design choice — this "current" review mode may be
  structurally meant to never self-declare "ready," always requiring a
  human to promote it after reviewing the blocked/negative-case
  breakdown, since ticker-level replay evidence alone is documented
  elsewhere in the same function as unable to "override pipeline
  blocks." Determining which requires either the original author's
  intent or a real design decision about what "exact_pipeline_case_count
  > 0 with 0 negative cases" should mean for readiness — not a safe
  mechanical fix. **Deferred, not fixed** — flagged as high-priority for
  whoever owns this bridge's design to resolve deliberately (operator-
  facing status is either permanently misleading, or correctly
  conservative — someone with the original intent needs to decide which).
- `pipeline_timeframe_lane_readiness.py` imports
  `WORLD_MODEL_PIPELINE_CONTEXT_CONTRACT` but never uses it to validate
  the pipeline-context artifact's `contract`/`schema_version` field
  (only displays it) — a missing-validation gap, not a functional break
  (a wrong-shaped artifact degrades to empty/missing-lane defaults rather
  than crashing or falsely reporting readiness). Confirmed live via
  `dean_os/__init__.py`'s lazy-export table. Deferred: adding real
  validation here is a scope decision (how strict should the check be?),
  not a reference-fix.
- Full-batch `pyflakes` run across all 39 files: zero undefined-name
  issues. Batch was notably cleaner than prior ones — most large files
  read/spot-checked with no new findings beyond the two above.

**dean_os top-level *.py sweep COMPLETE (2026-07-25) — final 2 batches,
9 more real bugs found and fixed, commits `5d05a59a` through `500db570`:**
- **`agent_learning_loop_runbook.py`**: `_stop_reason()`'s `blocking_statuses`
  was missing `"gated"`/`"no_profiles"` — statuses `_status_for_stage()`
  can genuinely return for `profile_scorecard`/`calibration_gate`/
  `calibration_proposals` — so the loop silently walked past a
  not-actually-ready stage instead of stopping there. Confirmed live:
  `analyst_loop_daily_check.py`'s own `soft_loop_statuses` already
  included both values, proving its author expected them to surface.
  Fixing this exposed a test fixture bug (`test_daily_check_blocks_on_learning_loop_gate`
  had `profile_scorecard` accidentally in a "gated" state, masking the
  stage it meant to test) — fixed the fixture too.
- **`populate_research_corpus.py`**: dead `citations=[]` kwarg + unused
  `SourceCitation` import — `ResearchDocument` has no `citations` field,
  silently dropped by pydantic. Cosmetic, cleaned up.
- **`pipeline_adapter.py`**: `HybridPipelineAdapter._get_orchestrator()`
  set `_src_unavailable = True` then re-raised the `ImportError` anyway —
  `__call__` only checks that flag *before* calling `_get_orchestrator()`,
  so on the very first invocation (flag still `False`) the exception
  propagated straight out, crashing `DEANOrchestrator.run()` (no
  try/except anywhere around the pipeline_runner call) — defeating the
  adapter's own documented "degrade to no-op instead of crashing" design
  on exactly the case it exists for. Live:
  `create_hybrid_dean_orchestrator` (public `__init__.py` export) wires a
  bare adapter with no override. Wrapped the call in `try/except
  ImportError`; verified with a simulated missing-dependency repro
  (crashed before, degrades to `"pipeline_skipped"` after).
- **`dean_paths.py`**: the "optional import with local fallback" pattern
  for `PathValidationError`/`validate_safe_path` was broken — the local
  `class`/`def` executed *unconditionally* after the `try/except`,
  always overwriting a successful import of the real, symlink-checking
  `src.core.security.path_validator` with the weaker local version (no
  symlink check at all). Live: `DeanPaths` (used by ~29 files for
  essentially every dean_os artifact read/write) calls
  `validate_safe_path` via `resolve_input_artifact`. Guarded the fallback
  behind a flag; verified `validate_safe_path` now actually resolves to
  the hardened module.
- **`preload_risk.py`**: `preload_risk_data()`'s `base_cols` never
  included `"datetime"` (the real time column in `features.parquet`), so
  it was dropped by the timeframe filter before the chronological sort
  ever ran; the sort's own fallback key (`"timestamp"`) never exists
  either, so it silently sorted by an arbitrary column, making
  `pct_change()`-derived returns wrong instead of crashing. Confirmed via
  `git stash` A/B comparison against the real parquet: return series
  values genuinely differ before/after. Reachability: zero live callers
  today (only caller lives in untracked `.archive_temp/`) — fixed anyway
  since it's cheap and will be wrong the moment it's wired back up
  (plausible given the `run_agent_orchestrator.py` restoration question
  already flagged from the `e34650e0` investigation).
- **`shadow_calibration_readiness.py`**: unguarded
  `self.policy_path.read_text()` — crashes on a caller-supplied
  nonexistent path (default path exists, so narrow trigger). Sibling
  `ShadowCalibrationDiagnostics` already has the guarded pattern; matched
  it.
- **`data_inventory.py`**: silently reported timezone-naive datetimes
  (common from DuckDB) as `"latest: 0d ago"` instead of computing real
  staleness — a diagnostic tool that could mask genuinely stale market
  data. Live via `dean_domain_scaffold.py`'s `search` CLI command.
- Full suite reconfirmed clean after every commit in this batch:
  `3 failed, 1211 passed`, same 3 known non-fixes throughout. Two
  spurious NEW failures appeared in one intermediate run
  (`test_cli_smoke.py::test_search`,
  `test_saved_sec_filing_index_producer.py::test_current_database_has_verified_amd_periodic_filing`)
  — both confirmed as transient DuckDB file-lock contention (re-ran each
  in isolation immediately after, both passed cleanly), not real
  regressions.
- **Deferred, documented, not fixed** (per subagent's own reachability/
  confidence caveats, spot-checked): `fact_extractor.py`'s dedup-scope
  comment mismatch (dedupes across all chunks, not just current one —
  no live caller found); `preload_regime.py`'s `context_key` case
  inconsistency (doesn't currently break its one confirmed consumer);
  `data_loader.py`'s narrow `except PermissionError` (its one live caller
  already wraps it in a broader try/except, so contained).

**Result: full `dean_os/` module-by-module sweep is now complete** —
every subdirectory (agents/, analyst_core/, analysts/, pipeline_control/,
replays/, world_model/, packets/, risk/strategies/stress/observability/
execution/evals/, pipeline_tuning/, research_corpus/) and every top-level
`dean_os/*.py` file has been read and audited at least once this session.
**Total: 30 real bugs found and fixed** across all passes, plus the
52-file CLI-wrapper restoration and the flagged-but-untouched 100+-file
`src/` deletion discovery (see above, still open). Test suite holds
steady at `3 failed, 1211 passed` throughout — same 3 pre-existing,
deliberately-left non-fixes (documented reasoning above for each).

**dean_os module-by-module sweep: major subsystems now covered.** This
session's sweep has gone through agents/, analyst_core/, analysts/,
pipeline_control/, replays/, world_model/, packets/, risk/+strategies/+
stress/+observability/+execution/+evals/, pipeline_tuning/, and
research_corpus/ — 16 real bugs found and fixed, all verified live before
fixing, all tested, all committed with zero regressions (full suite
holds at `3 failed, 1211 passed` throughout, same 3 pre-existing
deliberate non-fixes). Diminishing returns are visible in the last two
passes (mostly dormant subsystems, no live bugs). Remaining unreviewed:
~170 top-level `dean_os/*.py` governance/report modules (mostly one-off
packet/review builders, similar in nature to what packets/ and replays/
already showed — likely a similar mix of a few live bugs among many
dormant modules), then `src/`'s ~30 subdirectories (the actual trading
pipeline — arguably higher marginal value than continuing to comb through
more one-off dean_os report modules, per the diminishing-returns signal).
**Next session should decide: one or two more recon passes on top-level
dean_os/*.py modules to close out dean_os, or pivot to src/ now.**

---

## src/ audit phase started (2026-07-25)

**User explicitly requested a systematic pass through `src/` (~712 .py
files, ~32 subdirectories) after dean_os's top-level sweep completed**,
starting with the previously-flagged critical, unresolved finding: commit
`e34650e0` (2026-07-22, "repo root cleanup") had deleted 100+ `src/`
files including safety-critical `risk_manager.py`/`kill_switch/*` — this
was flagged but deliberately NOT investigated during the dean_os phase.

**RESOLVED: the `src/` deletion was legitimate, not a mistake (mostly).**
Found commit `16b207494` (same day, right after the deletion) with
message: *"Superseded/retired implementations kept for reference:
backtesting, data sources, features, meta_learning, models, monitoring,
patterns, processing, reporting, risk, utils, validation. Same-session
leakage audit confirmed none of this is imported anywhere in the active
src/ tree — it's inert."* Personally verified this claim via grep across
the whole non-test, non-archive, non-draft codebase for every affected
module name — confirmed zero live callers anywhere. The content itself
was never lost; it's preserved at `src/archive/<original path>`.

**What was actually still broken (not the archived code itself, but
stale references to it) — all fixed, commits `573ad98b`, `155c85ca`:**
- **8 modules were deleted but never archived at all** (an incomplete
  archival pass): `calibration_engine.py`, `adaptive_confidence_calibrator.py`,
  `walk_forward_optimizer.py`, `hybrid_adaptive_technical_indicators.py`,
  `simple_adaptive_technical_indicators.py`,
  `modular_adaptive_technical_indicators.py`, `pattern_aware_training.py`,
  `real_time_learning.py`, `signal_processor.py`, `stage_3_improvements.py`
  (confirmed dormant the same way — zero live callers). Restored each
  from `e34650e0^` into `src/archive/<original relative path>`, matching
  the treatment already given to their siblings.
- **Some already-archived modules' own internal imports still pointed at
  the old (now-deleted) `src.*` paths for OTHER archived siblings**
  (e.g. `src/archive/risk/metrics.py` importing `from src.utils.data_safety`
  instead of `from src.archive.utils.data_safety`) — fixed each
  cross-import found.
- **This left 14 test files broken** (`ModuleNotFoundError` at collection
  time, or deferred inside individual test functions), which was blocking
  pytest collection for the **entire test suite**, not just the affected
  modules. Redirected each to import from `src.archive.*` instead.
- **Found and fixed a genuinely serious, separate, LIVE bug while doing
  this**: `test_stage4_active_training_contract.py`'s `PredictionResultRequest`
  import was pointed (by an earlier fix this same session) at
  `src.pipeline.stages.prediction.result_request` — a module that turned
  out to be a **fully orphaned, diverged duplicate** with zero real
  callers anywhere. The REAL, live `PredictionResultRequest` is defined
  *inline* inside `src/pipeline/stages/prediction/orchestrator.py` (Stage
  5's actual prediction-result contract, with a `models: dict[str, Any]`
  field) — `orchestrator.py` never imports the separate module at all.
  Fixed the test to import from `orchestrator.py` instead, and archived
  the orphaned duplicate to `src/archive/pipeline/stages/prediction/`
  with an explanatory comment, to prevent the next person (human or
  agent) from making the same mistake I initially did.
- Also fixed 2 more test-only issues surfaced by this: a `monkeypatch`
  target that patched the thin `stage_4_modeling.py` facade instead of
  `modeling/orchestrator.py` (where the patched name is actually resolved
  at call time — same "patch the real module, not the facade" lesson as
  the `PredictionResultRequest` case), and a test bypassing `BatchTrainer.__init__`
  via `object.__new__()` that never set the (newer) `self.artifact_store`
  attribute the constructor normally provides.

**Verified: full `tests/` tree (dean_os + everything else) now collects
with zero errors — 1860 tests, was 13 collection errors across 2
different failure classes before this pass.**

**Separately discovered and diagnosed a real performance bug, NOT
fixed**: `tests/contracts/test_config_reachability.py::test_no_obvious_missing_class_paths_in_config_files`
does `Path(".").rglob("*")` — an unscoped, unfiltered walk of the ENTIRE
repository root (data/, reports/, all archive directories, everything)
looking for config files, with only `.git`/`.venv`/`venv` excluded. This
single test caused what looked like two multi-hour "hangs" of the full
test suite (confirmed via `wmic`: the process was genuinely at 87-99%
CPU the whole time, not deadlocked — just doing an enormous, unscoped
filesystem walk). Excluding just this one test dropped a full-suite run
from 2h20m+ (killed, unclear if it would ever finish in reasonable time)
to under 5 minutes for everything else. Not fixed (would need scoping
the `rglob` to `configs/`+`src/` type directories only, or adding a
`--co` fast-path) — flagged as a concrete, high-value fix for next
session since it makes the ENTIRE test suite impractical to run in one
sitting today.

**Full non-dean_os suite baseline established (excluding the slow test
above): `28 failed, 614 passed, 3 skipped` in ~5 minutes.** Of the 14
test files touched in this pass, only 2 have any failing tests, and both
are explained (not caused by these fixes): a legacy math bug in the
already-archived `hybrid_adaptive_technical_indicators.py`
(`test_indicators_causality.py`), and the `test_stage4_active_training_contract.py`
issue described above (now fully fixed, 6/6 pass). **The other 26
failures are pre-existing, unrelated to this session's work, and have
NOT yet been triaged** — this is the immediate next task for the src/
audit: go through the 26 remaining failures the same way dean_os's 70
were triaged (categorize into real bugs vs. stale test assertions vs.
out-of-scope), starting from the list:
`test_data_source_contracts.py::test_no_network_calls_at_import_time_by_source_scan`,
`test_enrichers_correctness.py::test_feature_enricher_modules_do_not_emit_target_columns_by_source_scan`,
`test_static_trading_ml_contracts.py` (4 sub-failures),
`test_synthetic_data_gates.py::test_sample_fallback_requires_opt_in_by_source_scan`,
`test_target_calculators_correctness.py` (2 sub-failures),
`test_minimal_pipeline_smoke.py::test_run_hybrid_pipeline_help_if_available`,
`test_bias_detector.py::test_look_ahead_bias_detection`,
`test_advanced_engine.py::TestBiasDetector::test_detect_survivorship_bias_with_delisted`,
`test_config_integrations.py` (4 sub-failures),
`test_lazy_heavy_import_policy.py::test_enhanced_ensemble_import_does_not_import_torch`,
`test_nlp_optimization.py::test_keyword_entity_enricher_batch_processing`,
`test_p1_missing_policy_math.py` (3 sub-failures),
`test_reddit_sentiment_collector.py` (3 sub-failures). Also noted:
`test_monoliths.py::test_imports` fails on a pytest-asyncio config gap
(missing `@pytest.mark.asyncio`/mode config), unrelated to anything this
session touched.

**src/ pre-existing test-failure triage COMPLETE (2026-07-25) — 28 → 7
failures, commits `fe6eaf0b`, `7bf7d53f`, `fe6d0cb9`:**

**Highest-severity finding of this whole round — REAL, live data leakage,
now fixed:** `src/targets/calculators/{regression_calculator,
classification_calculator, indicator_prediction_calculator}.py` all did
`df[col].shift(shift)` directly on the whole input frame with no
per-ticker grouping. On a multi-ticker concatenated dataframe, the last
`abs(shift)` rows of every ticker except the final one would silently
pick up the *following* ticker's price/indicator value as their "future"
target — textbook cross-ticker data leakage, the exact class of bug this
project's whole audit culture exists to catch. Confirmed via
`tests/contracts/test_target_calculators_correctness.py`. **Not currently
exploited**: the one confirmed live caller, `TargetOrchestrator`
(`src/targets/target_orchestrator.py`), already does
`df.groupby(['ticker','interval'])` before invoking any calculator — but
the calculators themselves had zero defense-in-depth, so any future or
alternate caller skipping that grouping would silently corrupt training
labels. Added an internal `groupby('ticker')` to all three calculators
(falls back to the previous global-shift behavior when no `ticker`
column is present, e.g. genuine single-series callers). Verified against
`TargetOrchestrator`'s own test suite that double-grouping an
already-single-ticker sub-frame is a no-op with identical results.

**Second major finding — another confirmed-live, non-archived diverged
duplicate class**, same shape as the `PredictionResultRequest` case:
none this round, but see below for the general pattern continuing.

**Root-cause pattern for most of the remaining failures, confirmed
again**: several more modules were deleted by `e34650e0` but never
archived (an incomplete sweep, matching the earlier finding) —
`src/algorithms/advanced_backtest_engine.py`, `src/algorithms/bias_detector.py`
(a dependency of the former), `src/integration/ensemble_selector.py`,
plus (already covered in the prior commit) `calibration_engine.py`,
`adaptive_confidence_calibrator.py`, `walk_forward_optimizer.py`, the 3
`*_adaptive_technical_indicators.py` files, `pattern_aware_training.py`,
`real_time_learning.py`, `signal_processor.py`, `stage_3_improvements.py`.
All confirmed zero live callers, all restored to
`src/archive/<original path>`, cross-imports fixed.

**A THIRD, entirely separate archival event was discovered**: commit
`7f8f1cd7` ("chore: archive confirmed-dead code, not delete; fix stale
imports uncovered along the way") — an EARLIER, unrelated commit that did
the exact same kind of work this session has been doing, for a different
batch of files under `src/archive/models_dead/`,
`src/archive/model_selector_dead/` (e.g. `enhanced_ensemble.py`). This
confirms the "archive dead code with a clear paper trail, fix stale
references" pattern is an established, recurring practice in this
codebase across multiple past sessions/agents — worth remembering when
hunting for "missing module" test failures: **always check
`src/archive/**` (all subdirectories, there are at least 3 separate
archival waves) before concluding a module was truly lost.**

**A live (non-test) config file was also found stale and fixed**:
`src/config/data_sources.yaml`'s `local_file_source` entry pointed at
`src.data_sources.local_file_data_source` (archived). Confirmed nothing
in production dynamically reads this specific config entry today (only
the test does) — inert drift, not a live crash, but fixed to keep the
config honest.

**3 test files rewritten for genuine API drift (not archival)**:
- `test_reddit_sentiment_collector.py` — fully rewritten. Confirms and
  extends the earlier dean_os-session finding that this collector was
  rewritten to use real Reddit RSS feeds with **no synthetic-data
  fallback at all** anymore (`use_synthetic_data` attribute and
  `_fetch_reddit_sentiment_data` method don't exist). Verified real
  current behavior empirically: disabled by default (unchanged); enabled
  without a real `http_client_factory` raises `RuntimeError` (not a
  silent `None` return, as the old test assumed). Replaced 3 obsolete
  synthetic-data tests with 2 matching real behavior.
- `test_advanced_engine.py::test_detect_survivorship_bias_with_delisted` —
  `BiasDetector.detect_survivorship_bias()` takes 2 args (`historical`,
  `current`), not 3; no `delisted_dates` param and different return keys
  (`potential_bias`/`missing_assets_count`/`missing_assets`, not
  `has_survivorship_bias`/`delisted_count`/`delisted_tickers`) exist in
  the current, real, live implementation. Updated to match.
- `tests/test_bias_detector.py::test_look_ahead_bias_detection` —
  `detect_look_ahead_bias()`'s second argument is a raw **price** series
  (it derives future returns internally via `pct_change().shift()`), not
  a pre-computed returns series — the test was passing returns directly
  as if they were prices, silently defeating its own deliberate "signal =
  future return" leak scenario (the double pct_change transform hid the
  correlation). Fixed to pass a real price series.

**1 test skipped with clear reason**:
`test_calibration_synthetic_not_primary_score_by_default` checks a safety
property of now-archived, zero-live-caller `calibration_engine.py` —
nothing left to enforce since it never executes in production.

**Remaining 7 failures in the non-dean_os suite — all understood,
documented, deliberately not touched:**
- **5 are false positives from the same 3 `*_by_source_scan`-style
  static contract tests** (`test_no_network_calls_at_import_time_by_source_scan`,
  `test_feature_enricher_modules_do_not_emit_target_columns_by_source_scan`,
  `test_feature_enrichers_do_not_emit_target_columns`,
  `test_sample_fallback_requires_opt_in_by_source_scan`, plus one
  duplicate) — these do naive substring matching over file text (e.g.
  `'target_' in text`, `'sample data' in text.lower()`) with no
  understanding of code semantics. Manually inspected every flagged file
  (`context_map_enricher.py`, `derived_features_enricher.py`,
  `feature_orchestrator.py`, `feature_selector.py`,
  `correlation_engine.py`, and others): every single one is either
  *excluding* target columns defensively, using `target_column`/`target_col`
  as a generic parameter name unrelated to ML label leakage, or
  describing ordinary statistical subsampling ("Sample data if
  requested") — not one is an actual violation. This is a test-quality
  gap (the scan needs AST-based column-assignment detection, not text
  search) worth fixing eventually, but not a code bug — confirmed by
  hand, not fixed.
- **1 real, confirmed, but architectural finding, not fixed**:
  `test_model_factory_import_does_not_top_level_import_neural_models` —
  `src/factories/model_factory.py` genuinely does top-level `import` all
  6 heavy neural model classes (LSTM/GRU/CNN/Transformer/TabNet/Autoencoder),
  eagerly loading torch etc. on every import instead of lazy-loading.
  Real performance/architecture issue, but fixing it means restructuring
  a central factory's import strategy — bigger and riskier than warranted
  in this pass. Flagged for a dedicated future task.
- **1 pre-existing legacy-code math bug**: `test_indicators_causality.py`
  — a Bollinger Bands NaN/type-conversion bug inside the already-archived
  `hybrid_adaptive_technical_indicators.py`. Dead code, not fixed.
- **1 confirmed flaky** (passes standalone, intermittently fails under
  full-suite load — likely spaCy model loading/thread contention, not a
  real bug): `test_nlp_optimization.py::test_keyword_entity_enricher_batch_processing`.

**Separately, still open from the previous round, NOT touched**:
`tests/contracts/test_config_reachability.py::test_no_obvious_missing_class_paths_in_config_files`'s
unscoped `Path(".").rglob("*")` full-repo scan (the pathological
2h20m+ "hang" cause) — still needs its own fix (scope the walk to
`configs/`+`src/` only), flagged as high-value for next session.

**Final verified state**: full non-dean_os suite (excluding the one
pathological test, deselected):
`7 failed, 633 passed, 4 skipped, 1 deselected` in ~80 seconds. Combined
with dean_os's `3 failed, 1211 passed`, this session leaves the ENTIRE
`tests/` tree (1860 tests) collecting with zero errors and only 10 total
known, individually-documented, deliberately-left failures across the
whole project.

**`test_config_reachability.py` performance fix COMPLETE (2026-07-25,
commit `226e2162`).** Root cause was more fundamental than first thought:
`Path(".").rglob("*")` cannot skip a subtree once it starts descending
into it — it only filters the *yielded* Path objects afterward, by which
point the expensive directory-listing/stat I/O for `data/`/`models/`/
`reports/` (>65k files combined) had already happened. A first attempt at
this fix (adding exclusion checks on the yielded paths, same session)
looked reasonable but **did not actually help** — confirmed by direct
profiling that even a fully unrestricted `os.walk('.')` with zero pruning
was equally slow (>120s and still climbing), while `os.walk(topdown=True)`
with directories removed from `dirnames` *before* the walk descends into
them finished 46,848 files in 0.3 seconds. Rewrote the scan using
`os.walk(topdown=True)` with pruning (excludes `data/`, `models/`,
`reports/`, `logs/`, `outputs/`, `mlruns/`, `archive/`, `audit/`,
`node_modules/`, `.git/`, `.venv/`, `venv/`, and `.trunk/` — the last one
a symlink to an external trunk.io tool cache, confirmed via `readlink`,
though at only 79 files it wasn't itself the bottleneck). **Verified: full
test run now takes ~30s standalone (was 2h20m+, possibly never
finishing) and still correctly PASSES** — the underlying config-to-class
mapping was fine all along; only the scan itself was pathological. Full
non-dean_os suite (this test no longer needs to be deselected):
`7 failed, 634 passed, 4 skipped` in **2 minutes 45 seconds** (was
2+ hours). The whole `tests/` tree (dean_os + everything else) is now
practical to run in one sitting.

**Lesson for future sessions searching for "why is this hanging":
`Path.rglob()`/`Path.glob()` can never prune a directory subtree —
if a repo has any large generated-output directory (data, models,
checkpoints, node_modules, a symlinked external cache, etc.), any
`rglob("*")` starting above it will silently pay the full I/O cost of
walking that entire subtree even if the results are filtered afterward.
Use `os.walk(topdown=True)` with in-place `dirnames` pruning instead
whenever scanning from a root that might contain such directories.**

**src/archive/MANIFEST.md created (2026-07-26), commit `2ad6c1c2`.**
Documents all 3 archival waves (`16b207494` 2026-07-22, `7f8f1cd7`
undated-earlier, and this session's completion of `e34650e0`'s
incomplete sweep) so future sessions don't have to re-derive this from
git archaeology. While writing it, found 3 files (`pattern_aware_training.py`,
`real_time_learning.py`, `signal_processor.py`) existed as byte-identical
duplicates at both a flattened Wave-1 path and a nested path this
session created (not knowing Wave 1 already had them) — removed the 3
redundant flattened copies, kept the nested ones this session's tests
reference. **Read `src/archive/MANIFEST.md` before concluding any module
is "genuinely lost" in future src/ work — check it first.**

## Strategic architecture review (2026-07-25/26)

At the user's request, stepped back from line-by-line bug-hunting to
identify systemic patterns across the whole audit. Full discussion is in
the conversation; key findings and what was actioned:

1. **No single source of truth for "is this code live."** The single
   most repeated question this whole audit was "does anything real call
   this?" — always answered by manual grep, never by a queryable
   artifact. Costed a large fraction of total audit time. **Not yet
   actioned** — would need a lint rule / manifest generator (e.g. a
   script that flags modules with zero non-test importers) — a good
   candidate for a dedicated future session.
2. **Reference-list drift is endemic**, not a one-off bug class:
   capability matrices, evidence-alias tables, issuer registries,
   domain-sector maps, CLI-wrapper doc counts all independently drifted
   from the real registry they were meant to mirror. Standing principle
   going forward (already applied in several fixes this session, e.g.
   `MACRO_RELEVANT_DOMAINS` derived from `MACRO_SERIES_EVIDENCE_MAP`):
   **never hand-maintain a second copy of a set that already exists
   elsewhere — derive it.**
3. **Duplicate-class-divergence from incomplete refactors** (the
   `PredictionResultRequest` case, `historical_replay.py` vs
   `historical_research_replay.py` before this session's fix). Standing
   principle: when extracting a class to a new module, the old location
   must become a re-export (`from .new import X`), never a second
   independent definition, even temporarily.
4. **CI pipeline could not fail on test/lint status — ACTIONED, commit
   `a1cd5647`.** Found `.github/workflows/ci.yml`'s pytest step ends in
   `|| true` (always "succeeds" regardless of failures), and every
   linter (Ruff/Black/Bandit/mypy) has `continue-on-error: true`. This
   is concretely why the `e34650e0` incident (52 CLI wrappers + 100+
   src/ modules deleted, 13 collection errors) went unnoticed for days —
   nothing could fail loudly. Added a new, genuinely blocking
   `pytest --collect-only` step ahead of the existing coverage step
   (safe today: collection is 100% clean, 1860 tests). Left the
   coverage step's `|| true` in place with a comment recommending the
   next step: mark the 10 known pre-existing failures as explicit
   `pytest.mark.xfail` with reasons, then remove `|| true` so that step
   also actually gates merges. **Not yet actioned** (the xfail marking).
5. **Multiple archival waves, no shared manifest — ACTIONED**, see
   `src/archive/MANIFEST.md` above.
6. **`dean_os`'s ~189 real files, only a fraction wired into
   `orchestrator.py`.** Not necessarily bad (looks like an evolving
   "instrument panel" built ahead of need), but worth periodically
   asking per-subsystem: wire in now, or archive as a completed
   exploration? **Not actioned, ongoing judgment call per subsystem.**
7. **`model_factory.py` eagerly imports all 6 heavy neural models** —
   symptomatic of a likely-broader "import-time cost" pattern across
   `src/`. **Not yet actioned** — flagged as its own dedicated future
   task (restructuring a central factory's imports safely needs its own
   focused session, not a quick fix).

**User confirmed**: wants all of this addressed incrementally over
future sessions, not necessarily in one sitting. **Session ending here
deliberately** (very long single session, already recovered from one
mid-session compaction) — user was advised to start a fresh chat next;
this memory file has everything needed to resume cleanly.

**Next steps for src/ phase (in rough priority order)**: (1) mark the 10
known pre-existing test failures as `pytest.mark.xfail` with reasons to
enable removing `|| true` from CI's coverage step, (2) improve the 3
naive `*_by_source_scan` contract tests to use AST-based checks instead
of substring matching (removes 5 permanent false-positive failures), (3)
`model_factory.py` lazy-import refactor as its own dedicated task, (4)
~~consider a "zero live importers" lint/manifest tool~~ **already exists,
see below — don't build a new one**, (5) then begin the module-by-module
recon sweep of `src/`'s ~32 subdirectories the same way dean_os was
covered (agents, algorithms, analytics, backtesting, cli, colab, config,
core, dashboard, data, devtools, ensembling, factories, features,
integrations, main, meta_learning, metrics, models, monitoring, patterns,
pipeline, processing, risk, scripts, sentiment, simulation, targets,
trading, training, utils, validation — `src/archive/` itself is out of
scope, confirmed dead-by-design, manifest now exists).

**Post-audit architecture review + reachability-tool fix (2026-07-26,
commits `3ece95b8`, `ab21153c`):** after the src/ + dean_os module sweep
above, gave the user a standalone architecture assessment (strengths:
provenance/point-in-time discipline, propose-then-act separation,
traceable archival via git mv; systemic issues: no single reachability
source of truth, "reference list drift" as a recurring pattern, orphaned
refactor twins, a CI coverage step that couldn't ever fail a build, no
central archive registry, well-written dean_os logic never wired to
orchestrator.py). Two items from that review were **already fixed
earlier the same session** before this write-up (verify via git log if
picking this up cold): commit `a1cd5647` added a blocking
`pytest --collect-only` CI gate, commit `2ad6c1c2` added
`src/archive/MANIFEST.md`. Then worked the remaining items:
- **Orphaned twin, fixed**: `src/pipeline/stages/prediction/result_builder.py`
  (`PredictionResultBuilder`) was a complete, never-wired-in parallel
  implementation of the same "build Stage 5 result" job
  `orchestrator.py`'s own `_create_prediction_result`/
  `_prepare_final_results`/`_save_stage_5_results` already do live — same
  "diverged twin" pattern as the `PredictionResultRequest` dataclass
  archived earlier this session, except this was a whole module, and its
  own `from .result_request import PredictionResultRequest` was already
  broken (ImportError) because that dataclass file had since been
  archived out from under it. Zero real callers anywhere (grep found only
  comments naming the file). User chose "archive now" over "port the one
  unique capability it has" when asked. Archived via `git mv` into
  `src/archive/pipeline/stages/prediction/`, documented in MANIFEST.md
  including the one thing it had that live code doesn't — autoencoder
  reconstruction-error anomaly-score blending
  (`_integrate_autoencoder_anomaly`) — in case that signal is ever wanted
  in production later.
- **The "no reachability source of truth" finding was wrong as stated —
  the tool already exists, don't rebuild it.** `diagnostics/` (a whole
  toolkit: `module_diagnostic.py`, `dead_code_classifier.py`,
  `config_reachability_checker.py`, `registry_consistency_checker.py`,
  `domain_rule_scanner.py`, `pipeline_stage_checker.py`,
  `component_engagement_audit.py`, `component_harness_runner.py`,
  orchestrated by `diagnostics/run_all_diagnostics.py`) plus its output
  in `diagnostic_reports/` (`orphan_modules.txt`,
  `dead_code_classification.csv`, `component_engagement.csv`,
  `FULL_DIAGNOSTIC_REPORT.md`, etc.) is exactly this capability, and
  predates this whole audit. **The real problem was staleness, not
  absence**: the reports were dated Jun 11/Jul 5, i.e. before this entire
  dean_os audit and all 3 archival waves, and nothing regenerates them —
  no CI step, no habit of re-running before an audit pass. This is the
  same "silently drifts because nothing keeps it current" pattern as
  every reference-list bug found this session, just applied to the
  meta-tool meant to catch that pattern. **Lesson for future sessions:
  before doing a manual grep-for-callers investigation, run
  `python diagnostics/run_all_diagnostics.py` (plus
  `config_reachability_checker.py` — see below, it's separate) and check
  `diagnostic_reports/orphan_modules.txt` /
  `dead_code_classification.csv` / `FULL_DIAGNOSTIC_REPORT.md` first.**
  This session burned real time re-deriving reachability by hand for
  nearly every dean_os module because this existed but wasn't checked.
- **Found and fixed while regenerating: `diagnostics/config_reachability_checker.py`
  had the exact same pathological-scan bug already fixed once in
  `tests/contracts/test_config_reachability.py` (commit `226e2162`,
  same session) — `iter_config_files()`'s `Path(".").rglob("*")` can't
  prune a subtree once it descends into it, so it did the full I/O for
  `data/`, `models/`, `reports/`, `mlruns/` (tens of thousands of files)
  before any exclusion filter ran. That earlier fix only patched the
  test, not this sibling script. Confirmed live: running
  `run_all_diagnostics.py` end-to-end hung past 10 minutes and had to be
  killed mid-scan on this exact step. Applied the identical
  `os.walk(topdown=True)`-with-dirname-pruning fix; now finishes in
  ~1 minute. This is a second, independent instance of "the same bug
  exists in two sibling files and only one got the memo" — worth
  grepping for other `rglob("*")` calls without directory pruning if this
  pattern shows up again.
- Full static diagnostics regenerated and committed: `module_inventory.csv`,
  `orphan_modules.txt`, `dead_code_classification.csv`, `static_imports.csv`,
  `risk_findings.csv`, `config_reachability.csv`, `pipeline_stage_report.csv`,
  `FULL_DIAGNOSTIC_REPORT.md`. Confirmed archived files (e.g.
  `result_builder.py`) now correctly show up under `archive/` instead of
  being flagged as live orphans.
- **Decided and done (commit `5bb61e7d`)**: wired `run_all_diagnostics.py`
  into `ci.yml` as an informational step — `continue-on-error: true` +
  `timeout-minutes: 5`, uploads `diagnostic_reports/` as a build artifact.
  Deliberately does NOT auto-commit the regenerated reports back (that
  stays a manual "refresh before an audit pass" step) — the value of the
  CI step is purely as a smoke test, so the diagnostics tooling itself
  can't silently rot/hang again for weeks unnoticed the way it just did.
- **"Connect-vs-archive" review done for the biggest candidate (2026-07-26)**:
  see the CORRECTION note above the `dean_os/replays/` section — the
  "chief review cycle" chain looked dead by import-grep but the user
  confirmed they still run it manually; not archived. Also fixed the
  diagnostics blind spot this exposed (`AUDIT_GUIDE.md`, commit `b1031969`).
- **"Reference list drift" systemic lint rule — deliberately scoped down,
  done (2026-07-26, commit `a32685b8`)**: considered building a generic
  AST-heuristic scanner ("find hardcoded collections that look like they
  duplicate a registry"), rejected as over-engineering — no reliable way
  to infer intent generically, high false-positive risk, and this
  session already found every concrete instance by hand rather than by a
  generic tool. Instead locked in the two live regression gaps found:
  (1) `test_agent_capability_matrix.py` asserted a hardcoded
  `agent_count == 39` that had *already* drifted once (28->39) with no
  test failure until manually caught — removed the magic number, kept
  `matrix_complete` (the real invariant: every registry agent has a
  non-stale contract) plus a live-computed count. (2)
  `CROSS_DOMAIN_PROPAGATION` (`cross_domain_signal_bus.py`) had zero test
  coverage at all despite its `target_domains` lists having already
  drifted once this session (bogus `"industrial"`/`"consumer"`/
  `"financials"` strings, fixed earlier with no regression lock) — added
  `tests/dean_os/test_cross_domain_signal_bus.py` asserting every
  `target_domains` entry is a real `list_domain_ids()` value. Did NOT
  chase a third candidate (issuer-registry ticker-universe coverage) —
  traced far enough to see the "true" ticker universe isn't cleanly
  defined in one place (`agent_registry.yaml`'s `semiconductor_analyst`
  entry has no `ticker_universe` key), and decided that was diminishing
  returns for this pass; the original gap (4->12 tickers) is already
  fixed, just without a regression test.
**`src/risk/` pass complete (2026-07-26, commit `27dccb5e`):** first
`src/` module-by-module pass after resuming the standing plan (picked
`risk/` over the other candidates in the priority list below because
it's small — 5 files — and directly safety-critical for a real-money
system). Read all 5 files in full personally (no subagent needed at this
size): `max_exposure_monitor.py`, `analyzers/var_calculator.py`,
`analyzers/concentration_analyzer.py`, `analyzers/correlation_analyzer.py`,
`elite_risk_metrics.py` (531 lines, the real substantive VaR/CVaR/stress-
test engine). Mostly clean — `elite_risk_metrics.py`'s own
`check_limits()` docstring already documents a real prior-session fix
(a hardcoded flat 2% VaR estimate that could never trigger the 5% limit).
One small live-but-unreachable bug fixed: `check_limits()`'s
concentration-limits loop read `pos_data['value']` directly while the
VaR loop two lines above it in the same method already defends with
`.get('value', 0.0)` — inconsistent, and `check_limits()` itself has zero
live callers today (confirmed via grep; only `get_risk_report()`/
`compute_comprehensive_risk_metrics()` are exercised by
`MaxExposureMonitor` and tests) so this was dormant, not urgent, but
cheap/safe to fix now rather than leave as a footgun for whenever it's
wired in. Two other dead-but-harmless findings, deliberately not touched
(no correctness risk, just clutter): `max_exposure_monitor.py`'s
`_check_exposure_breaches`/`_get_most_frequent_breach` are unused private
methods (monitor_exposure() does its own inline breach check instead);
`VaRCalculator.calculate()` (the class's top-level wrapper, distinct from
the actually-used `calculate_var_historical()`) has zero live callers and
a `result.get('var', 0.0)` pattern that wouldn't actually catch a NaN
result (dict key exists even when value is NaN) — harmless because
nothing calls it, not fixed since it's speculative effort for dead code.
`tests/unit/test_var_loss_policy.py` + `tests/smoke_test_system.py`: 7
passed, zero regressions.

- **This completes the 2026-07-26 architecture-review punch list**
  (6 systemic findings from that review, all addressed or explicitly
  triaged: CI blocking-collection gate, archive MANIFEST, orphaned
  `result_builder.py` twin, reachability-tool staleness + its own
  `config_reachability_checker.py` bug, CI diagnostics wiring, chief-review
  false-dead-code correction, 2 registry-drift regression tests). Next
  session should resume the plain module-by-module `src/` sweep (see the
  priority list above this section) rather than continuing architecture
  meta-work.

**`src/targets/` pass complete (2026-07-26, commit `0f0aa460`):** second
`src/` module in the module-by-module sweep (user picked this over
`trading/`/`backtesting/` since it's small and we'd already found the
cross-ticker leakage bug in `calculators/{regression,classification,
indicator_prediction}_calculator.py` there the prior round). Recon
subagent covered the remaining files: `base_news_target_calculator.py`,
`post_news_target_calculator.py`, `pre_news_target_calculator.py`,
`target_orchestrator.py`, `timeframe_contract.py`,
`calculators/__init__.py`.
- **Archived** (not a fix — confirmed dead): `base_news_target_calculator.py`/
  `post_news_target_calculator.py`/`pre_news_target_calculator.py` were
  never wired into `TargetOrchestrator.CALCULATOR_MAPPING` (only has
  `regression`/`classification_binary`/`classification_multiclass`/
  `indicator_prediction`) and no config anywhere names a `post_news`/
  `pre_news` target type — confirmed via repo-wide grep, zero callers
  outside their own 3 files (other hits were only `dean_os/draft/` and
  `audit/legacy/quarantine/`, both already-known non-live). Moved via
  `git mv` to `src/archive/targets/calculators/`, documented as "Wave 4"
  in `src/archive/MANIFEST.md`. Worth remembering the bug documented
  there if anyone re-wires these: `news_df.get('news_type', 'general')`
  is not a per-row default (that's not how `DataFrame.get()` works) — if
  the `news_type` column is absent, the OR clause becomes `True` for
  every row, silently matching ALL tickers' news instead of just the
  target ticker's. Same cross-ticker-contamination failure class as the
  `shift()`-without-`groupby` bug fixed in the sibling calculators the
  prior round, just manifesting through a news-join instead of a shift.
  Also neither subclass actually used the shared `BaseNewsTargetCalculator`
  they were meant to extend — reimplemented a diverged copy inline
  instead (another instance of this project's "drift from the thing that
  was supposed to be the shared source of truth" pattern).
- **`target_orchestrator.py` and `timeframe_contract.py` confirmed
  clean**: `_process_by_ticker_groups()` groups by `['ticker','interval']`
  and sorts chronologically before any shift-based calculation;
  `mask_targets_across_time_boundaries()` (the one place in
  `timeframe_contract.py` that does `.shift()`) only ever runs on
  already-per-ticker-grouped frames. No leakage risk in the live path.
- Verified via `pytest -k target`: `2 failed, 102 passed` — the 2
  failures are the already-known pre-existing `*_by_source_scan`
  false-positive contract tests (documented earlier this session under
  the src/ pre-existing-failure triage), unrelated to this change, zero
  new regressions.

**`src/trading/` pass complete (2026-07-26, commit `0264494b`):** third
`src/` module. Recon subagent covered all 9 files
(`adaptive_parameter_manager.py`, `consensus_engine.py`,
`elite_risk_sizer.py`, `live_adaptive_ensemble.py`, `portfolio_manager.py`,
`post_inference_filter.py`, `trader.py`, `trading_orchestrator.py`,
`virtual_portfolio.py`). Key structural discovery, same shape as the
dean_os `replays/` split: **two separate live entry points into this
package that diverge sharply.**
- **Fixed (real, live)**: `VirtualPortfolio.buy_stock`/`sell_stock`
  (`virtual_portfolio.py:208,276`) — the except block referenced
  `self.logger`, which is never set (only a module-level `logger`
  exists) — so any real error during a buy/sell raised a fresh
  `AttributeError` from inside the except clause itself, propagating
  uncaught instead of the intended graceful `{'success': False, 'error':
  ...}`. Live via `SimulationEngine.run_monte_carlo_for_strategy`
  (`monster_test.py`/`shadow_battle.py`). Zero test coverage existed for
  either `virtual_portfolio.py` or `simulation_engine.py`
  (`tests/trading/` is an empty directory) — verified the fix manually
  with a smoke script (malformed order now returns a graceful error dict
  instead of crashing).
- **`src/main/modes/backtest.py` — FIXED (2026-07-26, commit
  `a478d6c0`)**, after tracing the real data flow rather than guessing.
  `BacktestMode` is live (wired into `system_orchestrator.py`'s mode
  dispatch as `'backtest'`), had zero test coverage, and
  `_run_portfolio_simulation()` was broken on 3 independent axes:
  `VirtualPortfolio(initial_capital=...)` (real kwarg is
  `initial_balance`), `.run_simulation()`/`.get_equity_curve()` (neither
  exists on the class), `MetricsCalculator(equity_curve)`/
  `.calculate_all_metrics()` (neither is real — the ctor only takes an
  optional `config_manager`; the real getter is
  `.get_portfolio_metrics(equity_curve)`). Deeper still: `_align_data`
  did a flat `price_data['close'].align(signals_df['signal'])` with no
  `groupby('ticker')`, even though `processed_data` (confirmed via
  `src/pipeline/modeling_context.py::iter_model_contexts` — accepts
  `DataFrame | dict[str, DataFrame]`, and in the live single-frame case
  requires a `'ticker'` column) is multi-ticker concatenated, same
  convention as everywhere else in this project — real cross-ticker
  mixing on top of the API breaks. Root cause of the original
  uncertainty: `_extract_predictions_and_signals` assumed a `'signal'`
  column would already exist in pipeline output, but Stage 5 Prediction
  never produces one (only raw `'predictions'`/`'raw_forecast'` values)
  — **there was no signal-generation step in this file at all.**
  **Resolution, not a rewrite-from-scratch**: traced that
  `Stage_7_Evaluation` (`src/pipeline/stages/evaluation/orchestrator.py`,
  which already runs as part of `execute_full_pipeline()` —
  `BacktestMode._execute_pipeline()` already calls it) does this EXACT
  job correctly and completely: `_prediction_to_signal()` converts raw
  predictions to BUY/SELL/HOLD, `BacktestAnalyzer.prepare_pivot()` pivots
  the long-format signals into the wide per-ticker-column shape
  `AdvancedBacktestEngine.run_comprehensive_backtest()` expects (which
  itself already runs `BiasDetector.detect_look_ahead_bias` internally),
  and `final_data['evaluation_summary']['metrics']` already carries the
  exact flat `final_equity`/`total_return_pct`/`sharpe_ratio` shape
  `_log_results()` expects (via the same `PortfolioMetricsCalculator`
  chain). **This is the same "duplicate-class-divergence from an
  incomplete refactor" pattern as `PredictionResultBuilder`/
  `historical_research_replay.py` from earlier sessions, just at the
  mode level** — `BacktestMode` was a redundant, never-finished parallel
  reimplementation of what Stage 7 already does live. Removed
  `_extract_predictions_and_signals`/`_validate_price_data`/
  `_detect_biases`/`_apply_embargo_period`/`_align_data`/
  `_run_portfolio_simulation` (~120 lines) and replaced with reading
  `evaluation_summary` directly; also hardened `_log_results` against
  Stage 7's basic-evaluation fallback shape (missing `final_equity`
  would have crashed a format string). Verified: imports cleanly,
  `pytest -k "backtest or trading"` → `2 failed, 12 passed, 1 skipped`
  (the 2 failures are the already-known pre-existing
  `*_by_source_scan`/`model_factory` findings, unrelated). **Minor,
  not-yet-fixed side finding surfaced along the way**: Stage 7's own
  `create_evaluation_summary()` drops `backtest_results['bias_analysis']`
  when building `final_summary` (only keeps `.get('performance', {})`)
  — the look-ahead-bias check runs but its result is silently discarded
  before the summary is saved. Low severity (informational-only field),
  not fixed this pass, worth a look if `src/pipeline/stages/evaluation/`
  is ever audited directly.
- **Found real bugs, confirmed dormant BY DESIGN (not an accidental gap
  — do not "fix" by wiring them up without a deliberate decision)**:
  `TradingExecutionStage.run()` (`src/pipeline/stages/trading/orchestrator.py:78-128`)
  explicitly, deliberately never calls its own `_initialize_trading_stack()`
  — it always returns `status='blocked_paper_execution_requires_isolated_executor'`,
  with methods explicitly docstringed "RESERVED for the isolated
  paper-executor workflow" (a review-receipt -> paper-simulation-plan ->
  isolated-external-executor -> paper-result-review boundary, same
  intentional-safety-boundary shape as `Trader.execute_order`'s live-trading
  block). This means `trading_orchestrator.py`, `portfolio_manager.py`,
  `consensus_engine.py`, `trader.py`, `elite_risk_sizer.py`,
  `post_inference_filter.py` are all currently unreachable in production
  — confirmed real bugs in them are documented here for whoever builds
  the isolated executor, not fixed now:
  - `portfolio_manager.py::check_risk_exits` reads `position.get('stop_loss')`/
    `('take_profit')`, but `virtual_portfolio.py::_process_buy_order`
    (the only place positions are created) never writes those keys —
    `VirtualPortfolio.__init__` even computes `self.stop_loss_pct`/
    `self.take_profit_pct` from config but nothing in the file ever
    consumes them. The SL/TP kill-switch this code appears to promise
    is a permanent no-op.
  - `trading_orchestrator.py:67-73` hardcodes `regime = 'ranging'`
    regardless of `self.regime_detector` (only used to gate a log
    message, its actual detection method is never called) — clobbers
    the real per-ticker `market_regime` already present on each
    prediction, making `EnhancedConsensusEngine.regime_weights`'s
    `trending_up`/`trending_down`/`volatile` branches unreachable through
    this path.
  - `portfolio_manager.py:109,115` reads `signal.get('selected_primary_model')`/
    `('model_id')`/`('cognitive_scenarios')` off the signal dict, but
    `trading_orchestrator.py:234-237` only ever builds
    `{'ticker','final_signal','confidence','report'}` — none of those
    keys are ever present, so `EliteRiskSizer`'s real-measured-win-rate
    lookup (Sources 1/2) and its "Cognitive Risk Penalty" block both
    permanently fall through to crude heuristic defaults.
  - `trader.py`'s `TradeOrder` dataclass has no `confidence` field, but
    `trading_orchestrator.py:311-316` reads
    `getattr(order, 'confidence', 0.8)` — always the hardcoded default.
  - **Position-size cap ("max_position_size_pct") drifts across 4
    independent sources** — same reference-list-drift class as several
    fixes earlier this session: `portfolio_manager.py` reads a top-level
    `risk_config.get('max_position_size_pct', 0.1)` (only used by its
    last-resort BASIC sizing tier); `virtual_portfolio.py` computes the
    same value but never uses it anywhere (dead read, in the LIVE
    class); `elite_risk_sizer.py:354` hardcodes an unrelated literal
    `0.15` for its own position-value cap; `src/algorithms/adaptive_position_sizer.py`
    (the actual PRIMARY sizing path when `PortfolioManager` succeeds)
    reads its own default from a *differently-nested* config key
    (`risk_management.position_sizer.max_position_size_pct`, not the
    top-level `risk_management.max_position_size_pct` an operator would
    naturally edit). An operator tightening the "obvious" config key
    would believe they'd tightened the cap without actually affecting
    the primary sizing path.
  - `elite_risk_sizer.py:343-344` — genuine silent `except Exception: pass`
    (no logging) around a diary win/loss-ratio lookup; low severity
    (falls back to a reasonable heuristic) but matches this project's
    "error_behavior: skip means the crash is never even logged" pattern.
  - Lower priority, noted only: `adaptive_parameter_manager.py`'s
    `_apply_config_overrides` only overrides `regime_presets`, never the
    5 hardcoded `asset_presets` dicts — this class IS live (via
    `recommendation_engine.py:297`), but this is a config-completeness
    gap, not a correctness bug.
  - `src/archive/risk/kill_switch/manager.py` has a fuller
    `KillSwitchManager` with reset semantics that diverges from
    `portfolio_manager.py`'s single-flag kill switch — already in
    `archive/`, lower priority, just noted for whoever eventually
    reconnects real risk-exit logic.

**`backtest.py` FIXED** (see above — resolved, not deferred, once the
real Stage 7 data flow was traced). `src/backtesting/` itself needed no
separate pass — its one real file, `AdvancedBacktestEngine`
(`src/backtesting/advanced/advanced_engine.py`), was already fully
examined and confirmed live/correct as part of that investigation.

**`src/algorithms/` pass complete (2026-07-26, commits `9187fa22`,
`f4d38e03`, `d4a1b732`, `c9bbc134`, `c93ec8c8`):** fourth `src/` module —
picked because `adaptive_position_sizer.py` (flagged in the `trading/`
pass as the PRIMARY live position-sizing path) lives here. Recon subagent
covered all 8 real files. This pass surfaced the **highest-leverage
finding of the whole `src/` sweep so far** — a systemic config-accessor
bug, found while verifying a config fix actually worked end-to-end rather
than just trusting the YAML looked right.

- **`UnifiedConfigManager.get_config()` vs `.get()` — 9 silently-broken
  call sites across 6 files, all fixed.** `.get(key, default)` does real
  hierarchical dotted-path traversal (`_traverse_nested_keys`);
  `.get_config(name, default)` is docstringed "Legacy access interface"
  and does a **flat** `self.merged_config.get(name, default)` — zero
  dot-splitting. Any call passing a dotted key to `get_config()` silently
  returns the default every time, forever, regardless of what's actually
  configured. Found this by writing the config-drift fix below, then
  testing it end-to-end and discovering the real values still weren't
  reaching the classes — traced to the accessor itself, not the config
  files. Ran an AST scan (not just grep — needed to handle multi-line
  calls reliably) for every `get_config()` call whose first arg is a
  dotted string literal: exactly 9, all real, all now fixed to `.get()`:
  `virtual_portfolio.py` (`strategy.risk_management`,
  `backtest.transaction_costs` — the latter also had a second bug, see
  below), `base_trainer.py` (`models.enabled_types`),
  `system_orchestrator.py` (`execution.max_workers`,
  `execution.parallel_tickers`, `monster_test.tickers`),
  `pipeline_factory.py` + `pipeline_orchestrator.py` (both
  `performance.memory_warn_gb`), `backtest.py` (`backtest.walk_forward`).
  Concretely this meant: `VirtualPortfolio.stop_loss_pct`/
  `take_profit_pct`/transaction costs always ran on hardcoded defaults no
  matter what config said; `system_orchestrator`'s worker
  count/parallel-tickers/monster-test-tickers config was always ignored;
  the pipeline's memory-profiler warning threshold was always 10.0GB
  regardless of config. **Verified end-to-end after the fix** (not just
  "should work now"): `VirtualPortfolio.stop_loss_pct`/`take_profit_pct`
  now read the real `0.10`/`0.20` instead of the old hardcoded
  `0.05`/`0.1`. Full test suite: `1846 passed, 10 failed` — same 10
  already-known pre-existing failures documented earlier this session,
  zero regressions. **Given this bug's blast radius, if any future
  session finds a config value that "looks right in the YAML but doesn't
  seem to take effect," check whether the reading code used
  `get_config()` with a dot in the key before assuming the bug is
  elsewhere.**
- **Config-key drift, root-caused and fixed alongside the accessor bug**:
  `strategy.yaml` had a top-level `risk_management:` block
  (`max_position_size_pct: 0.15`, `stop_loss_pct: 0.10`,
  `take_profit_pct: 0.20`) that landed at `merged_config['risk_management']`
  — a sibling of `merged_config['strategy']`, never read by any code
  (confirmed via repo-wide grep for the bare key). Moved these settings
  into `risk_management.yaml`'s `strategy.risk_management` block (the
  path every real reader actually queries), aligning
  `max_position_size_pct` to `0.10` to match the existing,
  already-correctly-wired `max_single_position_pct` (used by
  `elite_risk_metrics.py`) rather than introducing a second, conflicting
  number. Also added `position_sizer`/`risk_allocator` sub-keys, which
  **no config file defined anywhere** — `AdaptivePositionSizer`/
  `RiskParityAllocator` have always silently run on class-internal
  hardcoded defaults regardless of any config edit, since
  `PortfolioManager` passes them `risk_config.get('position_sizer', {})`
  and that sub-key never existed. Also fixed a second, independent bug at
  the `virtual_portfolio.py` transaction-costs call site while there:
  read `'backtest.transaction_costs'` but the real top-level config key
  is `backtesting` (with "-ing") — confirmed `'backtest'` has never
  existed as a top-level key in any config file.
- **`DataProcessingError` from regime detection escaped every real
  caller — fixed.** `RegimeClusteringEngine.detect_regime_ml()`/
  `RegimeRulesEngine.detect_regime_rules()` wrap internal failures and
  re-raise as the custom `DataProcessingError`, but
  `MarketRegimeDetector.detect_regime()` itself and all 3 real callers
  (`technical_analysis_enricher.py`'s per-row regime-feature loop,
  `market_regime_analyzer.py`, `recommendation_engine.py`) only caught
  the standard `(ValueError, TypeError, AttributeError, KeyError,
  ZeroDivisionError)` tuple — not `DataProcessingError` — so any internal
  ML-clustering or rules-engine failure crashed feature enrichment or the
  trading recommendation stage outright, instead of degrading gracefully
  like every sibling try/except in the same files already does. Widened
  all 3 callers' except tuples; added a per-row try/except inside
  `technical_analysis_enricher.py`'s regime loop (matching its existing
  `history.empty` graceful-degradation branch, so one bad historical row
  marks `'UNKNOWN'` instead of aborting the whole ticker's feature
  computation); added the missing `self.logger.error()` call before
  `clustering.py`'s re-raise (matching `rules.py`'s already-correct
  equivalent).
- **`AdaptivePositionSizer` — two money-sizing safety gaps, fixed**:
  `_apply_position_limits()`'s `np.clip(position_size,
  portfolio_value*min_pct, portfolio_value*max_pct)` had no floor on
  `portfolio_value` — with a negative value (blown/underwater account),
  the min bound exceeds the max bound and `np.clip` silently returns the
  (negative) upper bound instead of raising, producing a negative dollar
  position size. Floored `portfolio_value` at 0 first (verified:
  negative portfolio now returns `position_size=0.0`, not negative).
  `conf_adjustment` was the one multiplier in the position-size formula
  that flowed straight from caller-supplied `confidence` unclamped, unlike
  every sibling adjustment factor (volatility/drawdown/kelly/liquidity),
  all of which are `np.clip`-bounded. Clamped to `[0, 1]` (verified:
  `confidence=1.5` now clamps to `conf_adjustment=1.0`). Both changes are
  in the confirmed-live `AdaptivePositionSizer` class (via
  `PortfolioManager._calculate_position_size`'s ADAPTIVE tier, currently
  reachable only through the dormant `TradingExecutionStage` boundary —
  same as most of `trading/`'s other findings — but the class itself, its
  math, and this fix are real regardless of current reachability).
- **`src/algorithms/transaction_cost_model.py` — archived** (Wave 5 in
  `src/archive/MANIFEST.md`). A second, diverged `TransactionCostModel`
  — the live one is in `src/backtesting/advanced/advanced_engine.py`
  (imported by `virtual_portfolio.py`). Same `__init__` config keys, but
  `calculate_execution_costs()` diverged: this one took
  `(trade_value, daily_volume)` and returned a `float`; the live one
  requires an extra positional `volatility` and returns a `dict`. Same
  duplicate-class-divergence pattern as `PredictionResultRequest`/
  `result_builder.py`/`backtest.py`-vs-Stage-7. Confirmed zero real
  callers outside its own file and the package `__init__.py`'s re-export
  (removed); only other reference was already-archived (Wave 3)
  `advanced_backtest_engine.py`, whose own import is now fixed to the new
  archive path.
- **Not fixed, documented for awareness only** (per recon's own lower-
  confidence flags, independently reviewed):
  1. `RiskParityAllocator` — its one live caller
     (`portfolio_manager.py::optimize_allocation`, itself only reachable
     through the dormant `TradingExecutionStage` boundary) feeds it
     `correlations = np.eye(len(target_assets))` — a fabricated identity
     matrix (zero cross-correlation for every pair), never computed from
     real return data, on every rebalance call. Since the matrix isn't
     `None`, the allocator doesn't fall back to
     `AllocationMethod.RISK_PARITY` — it silently proceeds with fake
     "everything is uncorrelated" data, defeating the algorithm's whole
     premise with no error or warning. Not fixed: doing this properly
     means plumbing real historical multi-ticker return data into
     `PortfolioManager` (which currently receives no price-history
     provider at all in its constructor) — a real architecture decision,
     not a quick fix, and the caller is dormant today anyway. Worth a
     dedicated look whenever `TradingExecutionStage`'s isolated-executor
     boundary gets built out.
  2. `metrics_mixin.py`'s `_calculate_max_drawdown` (used by the live
     `AdvancedBacktestEngine`) is a separate, un-unified reimplementation
     of `PortfolioMetricsCalculator.calculate_drawdown` (used by the live
     `VirtualPortfolio`) — same core ratio, but the canonical version
     additionally computes `avg_drawdown`/`recovery_time_days`. Sharpe
     was already unified between the two (`metrics_mixin.py` explicitly
     delegates to `FinancialMetricsLibrary`); drawdown wasn't. Neither
     guards a zero/near-zero rolling-max denominator (both would silently
     produce `inf`/`nan`). Low severity, flagged as a maintenance-drift
     risk, not touched.
  3. `RegimeClusteringEngine._initialize_cluster_centers` fits `KMeans`
     against 8 hardcoded, unlabeled 7-value rows, then classifies real
     feature vectors against those fixed centers — not genuinely trained
     on returns data despite being reported as `method: 'ml_clustering'`.
     The 7-feature ordering between `_extract_ml_features` and the
     hardcoded centers matrix is coupled only by construction discipline,
     not an enforced contract. Design smell, not a currently-triggered
     bug; not touched.

**`src/config/` pass complete (2026-07-26, commit `64057e39`):** fifth
`src/` module — small (`unified_config_manager.py`, `target_type_registry.py`
already vetted this session from the champion-selector work,
`__init__.py` empty), picked because the `get_config()` accessor bug just
found lives here. Read `unified_config_manager.py` in full personally (no
subagent needed at this size).
- **Fixed**: `DynamicConfig.__getattr__` checked `if value is not None:
  return value`, so a config key whose YAML value is explicitly `null`
  (e.g. `experiments.yaml`'s `max_workers: null`) raised `AttributeError`
  via attribute-style access — indistinguishable from the key not
  existing at all. Switched to a dict-key-membership check (matching how
  `_get_nested_value` already correctly does it for the dotted-path
  `.get()` accessor) so a `None` value now returns `None` and only a
  genuinely absent key raises. Verified directly; `tests/ -k config`: 34
  passed.
- **`get_config()` itself deliberately left as-is, not deprecated**:
  considered whether to migrate its ~64 remaining (non-dotted-key) call
  sites to `.get()` for consistency now that its "legacy" docstring is
  known to hide a real footgun, but decided against a sweeping rename —
  every remaining call site already uses single-level keys, where
  `get_config()`'s flat lookup is behaviorally identical to `.get()`'s
  traversal (confirmed: `.get()` splits on `.` and a dot-less key is a
  no-op split). No live bug left to fix there; a rename would be pure
  churn. The real fix (this session's `f4d38e03`) was catching every
  *dotted*-key misuse, which is now zero (re-confirmed via the same AST
  scan).
- **`_generate_feature_lists()`/`self.feature_sets`** — a stub that
  always returns `{}`, assigned to an attribute with zero external
  readers anywhere in `src/` (confirmed via grep). Genuinely dead, not a
  behavioral bug (nothing reads it, so nothing is wrong), left as-is —
  removing it is small, speculative cleanup with no correctness benefit.
- Verified the file's precedence/merge logic (`_sort_config_by_precedence`
  + `_deep_merge`) is actually correct despite looking suspicious at
  first read: files are merged in ascending-precedence order and each
  new file is the `source` argument to `_deep_merge(source, destination)`
  (source wins), so the last-processed (highest-precedence) file's keys
  correctly win on conflict — `_track_key_source`'s "precedence given to
  latest" warning message is accurate, not a bug.

**`src/analytics/` pass complete (2026-07-26, commits `3d97e7a4`,
`dabe5540`, `65414e91`):** sixth `src/` module — 57 files, split across 2
parallel recon subagents (calculators/analyzers/arena/data_managers vs.
context/detectors/engines/signals/utils). Highest-value fix was a
confirmed live cross-ticker + lookahead leak, found in the same family as
this whole project's core bug class.

- **Fixed — real, live, cross-ticker + lookahead leak**:
  `src/features/enrichers/advanced_analytics_enricher.py::_add_market_phase_detection`
  passed the WHOLE (potentially multi-ticker, multi-date) `df_enriched`
  to `MarketPhaseAnalyzer.analyze()`, which computes exactly one phase
  from `market_data.iloc[-1]` (the physically last row of the batch),
  then broadcast that single value via scalar assignment to
  `df_enriched['market_phase']` — i.e. literally every row. Since
  `FeatureOrchestrator` never splits by ticker before handing data to
  enrichers (only by interval), every ticker in a batch got the same
  phase, computed from whichever ticker's row happened to be physically
  last — both cross-ticker contamination and lookahead (every historical
  row got a feature derived from the dataset's last row, not its own
  point in time). `advanced_analytics` is enabled by default
  (`features.yaml:19`). Fixed by calling `analyze()` per-row (a 1-row
  slice each time) — `_determine_market_phase` only ever needs one row's
  own indicator values, no trailing window, so this is both leak-free and
  needs no explicit `groupby('ticker')` (each row's own already-per-ticker
  indicator columns are enough). Verified with synthetic two-ticker data
  producing distinct per-row phases instead of one shared value.
- **Fixed — real, live (currently masked by call pattern)**:
  `ModelComparisonAnalyzer._build_model_cohort` read
  `ticker_data.get('metrics', {})`, but `base_trainer.py`'s
  `results['metrics']` is keyed BY MODEL TYPE, not a flat dict with a
  top-level `accuracy` key — so `_extract_performance_metric` could never
  find it, and every cohort entry's `performance_score` silently
  defaulted to `0.0`. `base_trainer.py` already computes exactly the
  right flat shape under a *different* key,
  `results['winner_metrics'] = results['metrics'].get(winner, {})`
  (`base_trainer.py:402`) — read that instead. Not currently flipping any
  real champion selection (today's single-ticker call pattern means
  `_arbitrate_champion` always hits the "defaulted, no alternatives"
  branch rather than the real `&gt;=` comparison), but the comparison
  mechanism itself is now actually functional if ever called with a
  multi-model/multi-ticker cohort. Verified with synthetic data:
  `performance_score` now correctly shows `0.72` instead of `0.0`.
- **Archived** (Wave 6 in `src/archive/MANIFEST.md`):
  `analyzer_registry.py` (stale static registry, missing 6 of 11 real
  analyzer classes, unrelated to the real live registration mechanism
  `UnifiedAnalyticsEngine._register_analyzers_from_config()`; only
  referenced by `tests/smoke_test_system.py`, a standalone diagnostic
  script, now fixed to check the real `UnifiedAnalyticsEngine.analyzers`
  dict instead — correctly reports 2 live analyzers, not the stale
  registry's fake 8); `critical_signal_detector.py`/`signal_analytics.py`/
  `significance_detector.py` (all 3 initialized in
  `PredictionStage.__init__` behind a misleading "✅ ... initialized" log
  but never called anywhere else — confirmed zero callers AND zero test
  coverage). **Deliberately kept live despite also having zero production
  callers**: `analyzers/wrappers.py`, `detectors/anomaly_detector.py`,
  `utils/analytics_math.py` — all three have real, passing test coverage
  (`test_wrappers.py`, `test_p1_missing_policy_math.py`) that would
  otherwise be discarded. **New standing rule for archival decisions in
  this project: orphaned-but-tested code stays; only orphaned-and-
  zero-test-coverage code gets archived.**
- **Found but NOT fixed — deliberately, needs a dedicated session, HIGH
  PRIORITY for `src/features/` audit**: while verifying the market_phase
  fix, discovered `FeatureOrchestrator._instantiate_enricher()`
  (`src/features/feature_orchestrator.py:190-195`) computes each
  enricher's constructor config via
  `config_manager.get_config('features', {}).get('enrichers', {}).get(enricher_id, {})`.
  Confirmed **11 enrichers** have a `def __init__(self, config...)`
  constructor that would receive whatever this computes (grep:
  `advanced_analytics`, `context_map`, `decay_features`, `hype`,
  `keyword_entity`, `macro_features`, `market_context`, `news_impact`,
  `news_quality`, `volatility`, `volume`). The real `features.yaml` only
  has `features.enabled_enrichers` (boolean on/off flags, not settings);
  a real `features.enrichers.<id>` block DOES exist but only in
  `unified_config.yaml` (highest merge precedence), and it only carries
  `{enabled: true}` per entry for **13** enrichers (not real settings,
  and `advanced_analytics` isn't even among those 13). Confirmed
  end-to-end for `AdvancedAnalyticsEnricher` specifically: it always
  receives `config={}` in production, so `phase_config =
  self.config.get('market_phase', &lt;hardcoded default&gt;)` always falls
  back to a **hardcoded default** market-phase rule set
  (`volatility`/`trend`/`regime` indicators) that is completely different
  from the human-authored `market_phase_definition` block in
  `strategy.yaml` (`price`/`short_term_ma`/`long_term_ma` indicators,
  moving-average-crossover rules) — two semantically different phase
  models that were probably meant to be the same feature, disconnected by
  both a wrong config path AND a wrong key name (`market_phase` vs
  `market_phase_definition`). **Why not fixed now**: (1) unclear whether
  the other 10 affected enrichers rely on their constructor's hardcoded
  defaults being correct already (would need per-enricher verification
  before touching the shared orchestrator wiring, to avoid an
  unintended behavior ripple), (2) bridging `market_phase_definition`'s
  schema to `AdvancedAnalyticsEnricher`'s expected shape is a genuine
  design decision (which phase model does the user actually want?), not
  a mechanical key-rename. `TechnicalAnalysisEnricher` is NOT affected —
  it takes zero constructor args and self-fetches its own config
  directly via `get_current_config().get_config('technical_analysis', {})`
  inside `__init__`, bypassing this wiring entirely; worth checking
  whether any of the other 10 do the same self-fetch pattern before
  concluding they're all affected. **Next session: audit
  `FeatureOrchestrator._instantiate_enricher`'s config-wiring path first,
  as the entry point into the `src/features/` module-by-module pass.**
- **Other findings from recon, documented but not touched** (all
  confirmed dormant/orphaned, no live path found):
  `arena_battle.py`'s `run_battle()` — `UnifiedTrainingManager` calls it
  after every training cycle but nothing ever populates the shared
  arena's `current_battles` first (`register_model()`/`create_battle()`
  have no real callers), so it's a guaranteed silent no-op returning
  `{'battles_completed': 0, ...}`, logged misleadingly as "✅ Arena Battle
  completed" — but `results['arena_rankings']` (where this gets stored)
  has zero downstream readers, so zero practical impact today; would need
  real model-registration plumbing to fix, not a quick patch.
  `SyntheticControlMethods.calculate_treatment_effects` is a stub that
  always returns no-effect regardless of input — whole causal-inference
  stack (`counterfactual_generator.py` and siblings) is orphaned, zero
  production callers, only a unit test exercises it.
  `RiskParityAllocator`'s fake-correlation-matrix issue (found in the
  `src/algorithms/` pass) is corroborated here from the caller side.
  `MacroContextAnalyzer`, `CausalRippleEngine`, `analytics_math.py`'s
  three functions — fully orphaned, no callers anywhere.
  `MarketContextAnalyzer.analyze()` has no live callers at either of its
  2 instantiation sites, and would itself silently default 3 of 3
  requested context features to `0.0` on a caller/method-name mismatch if
  ever invoked (`context_features=['volatility','trend','momentum']` vs.
  the class's real methods `_calculate_volatility_5d` etc.) — same
  reference-list-drift-adjacent pattern as elsewhere. `MetaPatternMiner`
  writes `routing_rules.json`, which IS consumed live by `DynamicRouter`
  — worth flagging that nothing checks staleness if the offline miner
  stops being re-run.

**`FeatureOrchestrator` config-wiring bug FIXED (2026-07-26, commit
`f9cd7348`)**, same session, immediately after the finding above — traced
to completion rather than left as "needs a dedicated session." Confirmed
10 enrichers have a real `config` constructor param (`advanced_analytics`,
`context_map`, `decay_features`, `hype`, `keyword_entity`, `market_context`,
`news_impact`, `news_quality`, `volatility`, `volume`) and would all
receive whatever `_instantiate_enricher` computed;
`macro_features_enricher.py` self-fetches its own config directly (like
`technical_analysis_enricher.py` does) and is unaffected. Of the 10:
`hype`/`news_quality`/`volatility`/`volume` never actually read anything
from `self.config` at all (config injection is fully decorative for
them — bug had zero effect either way); `context_map`/`decay_features`'s
config keys (`champion_ticker`, `velocity_window`, `pattern_length`,
`half_life_periods`, `event_columns`) have no real YAML counterpart
anywhere (bug had zero practical effect, just always ran on hardcoded
defaults, which is all that was ever going to happen regardless);
`market_context`/`news_impact` DO have real YAML settings in
`enrichment.yaml`, but their hardcoded fallback defaults happened to
already match those real values (no behavior change, just newly
`really` configurable instead of accidentally-correct);
**`keyword_entity` is the one with confirmed real production impact**:
its `keyword_config` default is `{}` (not a safe fallback list like its
siblings), so `KeywordExtractor` has been running with **zero configured
keywords** for as long as this bug has existed — the real
`enrichment.keyword_entity.keywords` block (9 tickers, 6 tech terms, 8
financial terms) never reached it. `advanced_analytics`'s `market_phase`
key mismatch (see above) is a separate, deeper issue than this specific
path bug — even with the path fixed, `market_phase` still doesn't exist
as its own key under `enrichment.advanced_analytics` (no such block
exists at all), so that one's fix is genuinely a different, still-open
question (which phase-detection schema is actually wanted).

Root cause: `_instantiate_enricher` looked up
`features.enrichers.<id>` (only ever `{enabled: true}` stubs for 13
enrichers in `unified_config.yaml`, never real settings, and
`advanced_analytics` isn't even among those 13) instead of the real
location, `enrichment.<id>` in `enrichment.yaml` — sometimes one level
deeper under `.params` (`market_context`), sometimes not
(`keyword_entity`, `news_impact`). Added
`_resolve_enricher_config()` trying `.params` first, falling back to the
flat shape. **Verified thoroughly, not just "should work now"**:
`FeatureOrchestrator.create_from_config()` instantiates all 17 real,
currently-enabled enrichers successfully with the fix (no crashes, no
regressions); `_resolve_enricher_config()` directly confirmed to return
the real `keywords.tickers`/`context_features` (18 items)/
`half_life_hours` values instead of `{}`. `tests/ -k "feature or
enrich"`: 73 passed, same 2 known pre-existing false-positive failures.

**Separate, pre-existing, NOT caused by this fix, worth its own look**:
`significance_features`'s `enrichment.yaml` block uses a
`module`/`class`/`params` wrapper shape, but
`SignificanceFeaturesEnricher.__init__(self, significance_col=...,
min_events_per_ticker=..., mode=...)` takes individual keyword params,
not a single config dict — the resolved `params` sub-dict gets passed
*positionally* into `significance_col` instead of being unpacked, so
that constructor arg silently ends up holding a dict instead of the
intended string. Equally broken before this fix (received `{}`
instead) and after (receives the wrong-shaped dict) — confirmed via the
full orchestrator construction test that it doesn't crash either way
(Python doesn't runtime-check type hints), just silently wrong.
`economic_calendar_enricher.py`/`time_features_enricher.py` have the same
`module`/`class`/`params` YAML shape but weren't checked in depth this
round (`economic_calendar` isn't even in `enabled_enrichers` today, so
it's currently moot; `time_features_enricher.py.__init__(self)` takes no
args at all, so it's unaffected regardless).

**Rest of `src/features/` pass complete (2026-07-26, commits `cb067b7c`,
`c6626a9c`, `01832cf7`, `3c808ba5`):** 78 files, split across 3 parallel
recon subagents (enrichers/ remaining files, nlp/, and
analysis+builders+selection+validation+utils+monitoring+top-level). This
was the single highest-yield recon batch of the whole `src/` sweep —
**6 confirmed-live cross-ticker leakage bugs fixed in one pass**, plus 2
bugs discovered mid-fix by the fixes themselves (see below). This
directly validates the standing hypothesis that cross-ticker leakage is
this codebase's dominant bug class: `FeatureOrchestrator.run()` only
ever splits batches by `interval`, never by `ticker`, so every enricher
must defend against multi-ticker-concatenated data itself, and most of
them simply hadn't.

**Fixed — 6 confirmed live cross-ticker leaks, all enabled by default**:
1. `volume_enricher.py` — **worst of the batch**: OBV is `cumsum()` with
   no groupby, so once it crosses a ticker boundary the contamination
   never resets and corrupts every subsequent row for that ticker.
   volume_sma/roc/price_volume_trend/volume_rs all similarly ungrouped.
2. `volatility_enricher.py` — same pattern: returns, volatility_5/10/20,
   ATR, Garman-Klass all ungrouped. Feeds risk sizing/target generation
   downstream, large blast radius.
3. `derived_features_enricher.py` — LAG_*/VELOCITY_*/ACCELERATION_*/
   rolling_skew/rolling_kurtosis/rolling_volatility, all ungrouped.
4. `context_map_enricher.py::_process_numeric_column` — the adaptive
   noise-filter's pct_change/rolling had no groupby, inconsistent with
   the SAME file's `_generate_pattern_sequences`/`_calculate_context_velocity`
   a few lines below, which already correctly group by ticker — one spot
   simply missed during a partial fix at some point in the past.
5. `decay_features_enricher.py::_apply_decay_to_column` — the
   exponential-decay state is a genuinely sequential loop; a recent
   event in one ticker's last rows leaked a nonzero decayed value into
   the next ticker's first rows.
6. `keyword_entity_enricher.py::_aggregate_by_time`/`_merge_with_main_df`
   — aggregated keyword/entity counts across ALL news regardless of
   which company the article was about, then merged that single global
   series onto every ticker. **Newly exercised for the first time** this
   session — the earlier `FeatureOrchestrator` config-wiring fix is what
   made this enricher's config (and therefore its real behavior) reach
   production for the first time; this bug was presumably always there
   but effectively dormant while the enricher ran on empty config.

**2 bugs found mid-fix, by the fixes themselves — worth remembering as a
pattern**: fixing #4 and #5 above with a naive `.loc[boolean_mask]` /
`.reindex(df.index)` approach crashed (`"indices are out-of-bounds"` and
`"cannot reindex on an axis with duplicate labels"` respectively) against
a full `FeatureOrchestrator.run()` repro. Root cause: **multiple tickers
legitimately share the same trading dates**, and by the time these
enrichers run, something upstream has set `datetime` as the index — so
a duplicate-labeled index is the *normal*, expected shape for real
multi-ticker data, not an edge case. Both rewritten to use positional
numpy arrays (`.to_numpy()` / boolean-mask array assignment) instead of
any index-based reassembly, sidestepping the issue entirely. **Lesson:
any per-ticker groupby-fix in this codebase that reassembles results via
`.loc[]`/`.reindex()`/index-alignment must be verified against a
duplicate-index scenario (shared trading dates across tickers), not just
a clean-RangeIndex synthetic test — the synthetic tests that looked fine
in isolation this session all happened to use non-overlapping index
ranges per ticker and would NOT have caught this.** Confirmed via a full
`FeatureOrchestrator.run()` call with 2 tickers sharing 40 identical
trading dates before considering either fix done.

**Also fixed, smaller**: `technical_analysis_enricher.py`'s
`.replace([float('inf'), float('inf')], float('nan'))` — positive
infinity listed twice instead of `[inf, -inf]`, so negative-infinite
returns were never cleaned; `news_clusterer.py` — same `self.logger`
doesn't-exist bug pattern already fixed once this session in
`virtual_portfolio.py` (only a module-level `logger` exists), dormant
chain but cheap to fix; `volatility_driver_selector.py.select()` —
another cross-ticker leak (pct_change + ffill ungrouped), reachable from
`FeatureOrchestrator` but gated behind a config flag (`features.context_selection.enabled`)
that's off in every config file today — fixed anyway, cheap.

**Found, NOT fixed — documented for awareness, all confirmed via
repo-wide grep**:
- **`FeatureLeakageGuard` — FIXED (2026-07-26, commit `919cda10`), after a
  dry-run risk assessment and explicit user sign-off.** Was never
  actually blocking: `ColabManager._check_feature_leakage()` constructed
  it with `block_on_forbidden=False`, and even flipping that alone would
  have been neutered by the same method's own `except (ValueError, ...)`
  swallowing the guard's raise. Before touching anything, ran the guard
  directly (read-only, no code changes) against **7 real production
  batches** — `data/colab/accumulated/main_database` (1601 rows, 1030
  feature cols, 22 targets, updated the day before) plus 6
  `regenerated`/`accumulated` batches across different tickers
  (NVDA/semiconductor) and timeframes (15m/1d/60m) — all came back
  `status: clean`, zero forbidden columns, zero high-correlation
  features. This confirmed enabling blocking wouldn't halt anything
  currently passing. User approved after seeing this evidence. Fixed:
  `block_on_forbidden=True` + removed `ValueError` from the method's own
  except tuple (kept `TypeError`/`AttributeError`/`KeyError`/
  `ZeroDivisionError` as non-blocking internal-error cases) so the raise
  genuinely propagates out and the batch save is skipped. Verified both
  directions directly: real clean data still passes through unchanged;
  a synthetic injected forbidden column now raises `ValueError` that
  propagates all the way out (previously silently swallowed). Also
  corrected the module's docstring (falsely claimed Stage-3 integration)
  and `get_leakage_guard()`'s docstring (singleton factory has zero
  callers anywhere, corrected rather than removed). `tests/ -k
  "colab_manager or leakage or hybrid"`: 21 passed, only the
  already-known unrelated `--help` subprocess-timeout flake failed.
  **This is the kind of decision (production-behavior change, not a
  pure bug fix) that should always go through this dry-run-then-ask
  pattern** — verify real-world impact first, present the evidence, let
  the user decide, don't silently flip a policy flag even when the code
  fix itself is small and well-understood.
- `FeatureCache` (`src/features/feature_cache.py`) — wired into the live
  `FeatureEnricher.__init__` (`get_feature_cache(...)`), creates/prunes
  a cache dir every run, but its only two functional methods
  (`get_features`/`save_features`) have zero callers anywhere — the
  promised "60-80% speedup" never happens, pure overhead today. Separate
  duplicate: `src/features/monitoring/feature_drift_detector.py`
  (Evidently-AI based) has zero callers, a second drift-detector
  implementation alongside the also-orphaned `src/monitoring/feature_drift_monitor.py`.
- `EnhancedSmartFeatureSelector` (live, via `FeatureEngineeringStage`) —
  3 of its 5 constructed sub-components
  (`drift_monitor`/`freshness_monitor`/`regime_tracker`/`news_decay_modeler`)
  are assigned and never referenced again. This orphans an entire chain:
  `regime_importance_tracker.py`, `news_decay_modeler.py`, all of
  `analysis/decay/*`, and `pipeline/stages/monitoring/feature_monitoring.py::FeatureEngineeringMonitor`
  (itself never instantiated anywhere).
- Entire `NewsEventDatasetBuilder`/`NewsContextDatasetBuilder` chain
  (`news_dataset_builder.py`, `builders/news_event/*`,
  `news_impact_classifier.py`, `news_clusterer.py`) is orphaned — driven
  only by `FeatureEngineeringNewsManager`, which is never instantiated
  anywhere in `src/`. Two separate, similarly-named classes
  (`NewsContextDatasetBuilder` vs. `builders/news_event_dataset_builder.py::NewsEventDatasetBuilder`)
  implement essentially the same concept, both unused — another
  duplicate-implementation pair.
- Stale hardcoded ticker/company lists (pattern b, same class as several
  fixes earlier this session): `news_impact_classifier.py`'s
  `company_to_ticker` (19 companies) and `entity_linker.py`'s
  `entity_graph` (6 tickers) — both in low-priority/dormant code
  (`entity_linker.py` is test/dev-script only).
- `candle_seeker.py::get_candles_before()` docstring says "strictly
  BEFORE publication" but the filter is `<= pub_at` (mild look-ahead,
  includes the exact-timestamp candle) — dormant chain, not fixed.

**`FeatureLeakageGuard` block-vs-warn — RESOLVED (2026-07-26, commit
`919cda10`)**: raised directly with the user as planned, did a dry-run
risk assessment against 7 real production batches (all clean), got
explicit approval, then fixed `block_on_forbidden=True` +
stopped catching `ValueError` in `ColabManager._check_feature_leakage`'s
except tuple (it was swallowing the guard's own raise even before this,
so simply flipping the constructor flag alone would NOT have been
enough). See full detail earlier in this file. **This dry-run-then-ask
pattern is now the established playbook for any "safety net that looks
wired but isn't" finding** — used again immediately below.

**`src/data/` pass complete (2026-07-26, commits `dda7def8`, `6641fce3`,
`f2abf7e7`, `76bcae3a`, `74874170`, `55a3fa20`):** eighth `src/` module,
41 files, 2 parallel recon subagents (collectors/ vs.
management+quality+validation+synthetic). Second-highest-yield batch of
the whole sweep after `src/features/` — multiple confirmed-live crashing
bugs, a real synthetic-data-integrity violation, and (again) a whole
safety layer that was never wired in.

- **Fixed — 3 enabled collectors crashed on every single run**:
  `reddit_sentiment_collector.py`/`wikimedia_attention_collector.py`/
  `sdmx_macro_collector.py` all call
  `db_manager.filter_new_records(table, df, unique_cols=["record_hash"])`,
  but the real method only ever accepted `(table_name, df)` and hardcoded
  dedup on a column literally named `'hash'` — not `'record_hash'`, the
  column these 3 collectors actually produce. All 3 are `enabled: true`;
  every run paid the network cost (Reddit RSS / Wikipedia pageviews /
  World Bank-ECB-IMF-OECD-BIS SDMX) then crashed with `TypeError` right
  before persisting — **none of them had ever successfully written a row
  to the DB.** Added an optional `unique_cols` param to
  `filter_new_records` (both the abstract `IDatabaseManager` declaration
  and `DataManager`'s real implementation), defaulting to `['hash']` so
  the other 15+ callers are unaffected. Verified directly (no crash) and
  via `tests/ -k "data or collector"`: 155 passed.
- **Fixed — real synthetic-data-integrity violation**:
  `put_call_ratio_collector.py` fabricates 59 of 60 "historical" days via
  a deterministic sawtooth formula (CBOE only ever exposes the *current*
  ratio) and stored them **without** `is_synthetic`/`eligible_for_training`
  flags — unlike the collector's own explicit sample-data fallback
  (`_create_sample_put_call_data`), which correctly sets them. This
  happened on the "success" code path regardless of the
  `allow_sample_fallback` setting — an operator who disabled sample
  fallback believing fabrication was off was still getting 59/60 days of
  unflagged fabricated data. Directly violates this project's own
  established synthetic-data rule. Flagged the 59 fabricated rows and the
  1 real row correctly, matching the existing fallback's convention.
- **Fixed — lookahead bug**: `vix_collector.py`'s `vix_change` used
  `hist['Close'].shift(1).iloc[-1]` (the second-to-last close of the
  *entire* 60-day frame) instead of `hist_up_to_now` (which every sibling
  calculation in the same loop already correctly uses) — every historical
  row got the identical value, computed from data that hadn't happened
  yet relative to that row. `vix` is enabled by default. Verified with
  synthetic multi-day data that per-row changes now match real
  day-over-day deltas.
- **Fixed — dead duplicate method + documented a real, deliberately
  unfixed gap**: `economic_calendar_collector.py` had two `run()`
  definitions (Python keeps only the last); the dead one was what
  `collectors.yaml`'s ~50-line Investing.com config block was written
  for, and was also architecturally wrong regardless of shadowing
  (duplicated the orchestrator's own generic hash/filter/upsert handling
  inside the collector, unlike every other collector). Deleted it + now-
  unused imports. **Left deliberately unfixed**: `hash_keys =
  (timestamp, country, event)` means an event first stored before its
  release (`actual` empty) can never be updated once the real value
  arrives — hashes identically, filtered out as a duplicate, the actual
  print is lost forever. Confirmed this is NOT the same "point-in-time
  safety" rationale that makes `DataManager.upsert`'s insert-if-absent
  semantics deliberate elsewhere (that docstring is about preventing
  retroactive rewrite of what was known at an earlier point in time —
  here the data is just lost, not protected). Real fix needs a
  data-model change (e.g. a `collected_at` dimension distinguishing
  pre/post-release snapshots as two legitimate historical facts), which
  is a feature decision, not a quick patch.
- **Archived** (Wave 7, `src/archive/MANIFEST.md`): `alternative_me_collector.py`/
  `market_data_collector.py` (neither collector_type is a key in
  `collectors.yaml`, confirmed zero callers, `alternative_me` is a
  near-duplicate of live `fear_greed_collector.py`) — **plus an entire
  point-in-time-leakage-prevention layer**: `temporal_alignment_checker.py`,
  `news_price_availability_filter.py`, `data_freshness_checker.py`,
  `event_dataset_validator.py` (fixed its own `self.logger` bug before
  archiving), `data_versioning.py` (documented, not fixed: a
  partial-completion bug in `cleanup_stale_files`), and
  `management/handlers/connection_handler.py` (near-verbatim duplicate of
  `DataManager`'s own connection pooling). All confirmed zero test
  coverage. `data/management/data_cleaner.py` was deliberately **kept
  live** despite zero production callers — same "orphaned but tested"
  rule as `src/analytics/`'s Wave 6 (`tests/unit/test_p1_missing_policy_math.py`
  exercises `clean_numeric_data()` directly) — added a docstring warning
  instead of a rename, since it shares its exact class name with the
  actually-live `DataCleaner` in `src/processing/cleaners.py`, a real
  landmine for a future edit.
- **Found, NOT fixed — same shape as `FeatureLeakageGuard`, needs the
  same dry-run-then-ask treatment, HIGH PRIORITY for next session**:
  while tracing `temporal_alignment_checker.py`'s live equivalent,
  discovered `FeatureGuards._initialize_guards()`
  (`src/pipeline/stages/feature_engineering/guards.py`, confirmed live
  via `FeatureEngineeringStage.__init__`) constructs 5 guards but
  `apply_guards()` only ever invokes ONE of them
  (`temporal_leakage_guard.validate_rolling_windows`).
  `timeframe_guard`/`safe_combiner`/`macro_guard`/`temporal_target_guard`
  are all constructed and never invoked — confirmed via grep that none of
  their real validation methods (`validate_macro_data_timing`/
  `combine_features_safe`/`generate_targets_safe`) are called from
  anywhere except their own defining files.
  **`macro_guard` (`MacroReleaseTimingGuard`) is the one to prioritize**
  — it checks that macro-economic data wasn't used before its actual
  official release time, exactly the point-in-time bug class this
  project has been bitten by before (the macro evidence-provenance chain
  bugs fixed in `dean_os` earlier this session). `safe_combiner`
  (`SafeFeatureCombiner`/`TimeframeAlignmentGuard`) is lower-confidence
  severity — `FeatureEngineeringStage._combine_timeframes()` has its own
  separate, seemingly-actively-maintained logic
  (`BackwardTimeframeContextAssembler`), so this may just be legacy code
  superseded by a newer implementation rather than a live gap. Same for
  `temporal_target_guard` — `TargetGenerator.generate_targets()` is
  called directly, bypassing `TemporalTargetGuard.generate_targets_safe()`'s
  wrapper entirely. **Next session: do the same dry-run-against-real-data
  assessment used for `FeatureLeakageGuard`, starting with
  `MacroReleaseTimingGuard.validate_macro_data_timing()` against real
  macro data, then present findings and ask before wiring anything in.**
- **Other findings, lower priority, documented only**: `huggingface_collector.py`
  ignores most of its own `collectors.yaml` config block
  (`filter_by_keywords`/`keywords_categories`/`max_days`/`max_rows` all
  unread — loads the entire configured HF split unfiltered every run);
  `newsapi_collector.py` reads `api_key_name` but config sets
  `api_key_env` — currently masked (both resolve to the same default
  env-var name), `newsapi` is disabled anyway; `synthetic_generator.py`'s
  `run()`/`collect_historical_data()` call `generate_scenarios(scenario_name=...)`
  — wrong kwarg name (real one is plural `scenario_names`) and
  `'neutral_regime'` isn't a real scenario key — would `TypeError`
  immediately, but unreachable (`"synthetic"` isn't a `collectors.yaml`
  key; the only real callers call `generate_scenarios()` directly with
  correct args); `fear_greed_collector.py`'s `base_url` looked
  potentially stale (couldn't verify without network access) — if it
  404s, the collector silently produces zero data with only a log line,
  no alarm.

**FeatureLeakageGuard dry-run-then-ask resolved (2026-07-26): ENABLED.**
Dry-ran the guard against 7 real feature batches from the live pipeline.
All 7 came back clean (zero violations) — proving the guard's leakage
checks are compatible with real data shape/content, not just theoretically
correct. Presented the finding, user asked "як правильно?" (what's the
right call?), given a direct recommendation to fully enable it, user
confirmed with "+". Wired `temporal_leakage_guard` — wait, this refers to
enabling `FeatureLeakageGuard` itself (a separate class from the 5 guards
inside `FeatureGuards` described above) into the live path with blocking
behavior rather than warn-only. This established the reusable playbook:
**dry-run against real data → present findings → get explicit user sign-off
before wiring in any dormant safety/blocking mechanism** — used again below
for `MacroReleaseTimingGuard` with the opposite conclusion.

**MacroReleaseTimingGuard dry-run complete (2026-07-26) — recommended NOT
to wire in, structurally incompatible with real data:** ran
`MacroReleaseTimingGuard.validate_macro_data_timing()` directly against
real `data/processed/features/macro_data.parquet` (long-format, keyed by
a `series_id` column — one row per FRED series per date). Result: all
200/200 sampled rows hit "Could not determine macro type" and the guard
returned `status: valid` with 0 issues — a false "all clear", not a real
validation. Root cause: `_infer_macro_type_from_columns()` looks for
macro-type names (cpi, gdp, etc.) as **column names** in a wide-format
frame, but the real data is long-format with the macro type encoded in
`series_id` values instead. This is not a simple enable/flag flip like
`FeatureLeakageGuard` was — the guard's core type-inference logic would
need to be rewritten to understand `series_id`-based long format before
it could ever validate anything real. Recommendation given to the user:
do not wire this into `FeatureGuards.apply_guards()` as-is; needs real
logic changes first. `safe_combiner`/`temporal_target_guard` remain
undiagnosed (lower priority per the reasoning already documented above —
likely superseded by newer live logic rather than a live gap).

**Collector external-failure triage (2026-07-26):** tested network
reachability directly (curl -v + Python httpx) for every collector
suspected of being broken, to separate "our bug" from "genuinely
external, not fixable by us" per the standing rule to never attempt to
bypass anti-bot protections.
- **Genuinely dead, external, not fixable**: `fear_greed_collector.py`'s
  `production.datapoint.cloud` endpoint — confirmed dead via both
  `curl -v` (`SEC_E_ILLEGAL_MESSAGE` at the TLS layer) and Python `httpx`
  (`SSL: TLSV1_UNRECOGNIZED_NAME`). The server itself no longer answers
  for this hostname; would need an entirely new data source, not a fix.
- **Bot-blocked, external, not fixable**: CBOE (`put_call_ratio_collector.py`)
  returns HTTP 403 for automated requests even after the domain-typo fix
  below — confirmed the domain now resolves correctly, the block is
  CBOE's own anti-scraping layer. Per standing rule, not circumvented.
- **Was actually our bug, now fixed**: CBOE URL was
  `https://www.cboe.org/...`, which doesn't resolve (DNS failure) at all
  — typo for `https://www.cboe.com/...`, which does resolve (then hits
  the 403 above). Fixed in commit `0bc95ec4`.
- **Confirmed working, not broken**: ForexFactory (200), Reddit RSS with
  the real collector User-Agent `"DEAN_OS_Agent research@example.com"`
  (200), Wikimedia pageviews (200), World Bank SDMX (200). FRED's 400 was
  from an intentionally-fake test API key in the smoke test itself, not a
  real collector problem.

**`src/models/` pass complete (2026-07-26, commits `2f059f30`, `a8d115b5`,
`87b18975`, `4444339a`, `f8f0d96a`, `5385ccc6`):** 3 recon batches covering
neural models, loader/pooling, and ensemble/calibration. Real, live bugs
found and fixed:
- **`transformer_model.py`**: `TransformerModel(BaseModel)` implemented
  `fit()`/`predict()` instead of the abstract `train()`/`save_model()`/
  `load_model()` `BaseModel` actually requires — could not be
  instantiated at all (`TypeError`). Live: `transformer` is in
  `DEFAULT_ENABLED_MODEL_TYPES` (`ModelFactory.get_available_models()`),
  trained by default via `base_trainer.py`/`light_model_trainer.py`'s
  loop over enabled model types — any real training run that reached
  `'transformer'` crashed. Added `train()`/`save_model()`/`load_model()`
  as thin wrappers delegating to the existing `fit()`/`predict()` logic.
  **Second, deeper bug found only by insisting on end-to-end
  verification** (train+predict, not just "instantiates"):
  `_create_transformer_model()` used `tf.reduce_mean(ff_output, axis=1)`
  — a raw TF op applied directly to a `KerasTensor`, which raises on the
  installed Keras/TF version ("A KerasTensor cannot be used as input to a
  TensorFlow function"). This was being silently caught by `fit()`'s own
  exception handling and falling back to a plain `RandomForest` every
  single time — **the real transformer architecture had never once
  successfully trained**, project-wide. Fixed by replacing with
  `tf.keras.layers.GlobalAveragePooling1D()` (the Keras-layer equivalent).
  Verified via `ModelFactory.create_model('transformer', ...)`:
  instantiates, trains, predicts, and now genuinely uses the TF model
  (`model.model is not None`, `model.fallback_model is None`).
- **`gru_model.py`**: same "fake sequence" anti-pattern already fixed in
  `LSTMModel` earlier this session — built RNN input via
  `np.reshape(X, (X.shape[0], X.shape[1], 1))`, treating each feature
  column as a length-1-timestep series instead of a real rolling window.
  Mirrored `LSTMModel`'s fix exactly: added
  `SequenceBuilder(strategy='sliding_window')`, `train()`/`predict()` now
  call `build_sequences(X, window_size=..., step_size=...)`. Verified:
  50 samples, window_size=10 → 41 real sequences (matches `50-10+1=41`).
- **`loader.py`**: `KerasPredictor.predict()`'s CNN branch produced
  `(n, features, 1)` — transposed vs. CNN's actual trained shape
  `(n, 1, features)` — silently feeding wrong-shaped input into live CNN
  predictions instead of erroring. Merged CNN into the existing correct
  branch. Also simplified 4 redundant
  `except (ValueError, TypeError, Exception)`-style tuples to
  `except Exception as e:  # noqa: BLE001 - ... deliberately broad, always logged`,
  matching this file's own existing noqa convention (legitimate broad
  multi-library-exception catches, not silent).
- **`interfaces.py`**: `BaseModel.evaluate()` had a bare
  `except Exception: pass` around `predict_proba` — any failure vanished
  with zero trace. Narrowed to the concrete exception types and added a
  warning log (confirmed `self.logger` always exists via `__init__`).
- **`constants.py`**: `RANDOM_FOREST = "randomforest"` didn't match the
  canonical `"random_forest"` string used everywhere else. Zero live
  behavioral effect today (only referenced by already-dead
  `unified_model_adapter.py` and archived code) but was a landmine for
  if that constant ever gets a real caller. Fixed.
- **`correlation_engine.py`**: `adjust_weights_by_correlation()` computed
  `np.mean([...])` over a list that's empty whenever a model is perfectly
  correlated with every other model — `np.mean([])` silently returns NaN,
  corrupting every downstream ensemble weight via
  max()/sum()/normalization. Fixed to default to the max correlation
  penalty (1.0) in that case, since perfect correlation is the worst case
  for diversity, not "no data". Also tightened 2 overly-broad except
  tuples and removed 2 small dead-code lines. Verified via direct repro
  (pre-fix: confirmed NaN + RuntimeWarning; post-fix: correct equal
  weights). `tests/ -k "correlation or ensemble"` → 22 passed.
- Combined test run after all 6 fixes:
  `tests/ -k "neural or transformer or gru_model or cnn_model or loader
  or model_pool or interfaces or correlation or ensemble or constants"`
  → **78 passed, 1 pre-existing failure** (unrelated:
  `test_model_factory_import_does_not_top_level_import_neural_models`,
  a lazy-import contract test that already failed before this session's
  changes — `model_factory.py` top-level-imports all neural model
  classes; not touched, out of scope for this pass), zero regressions.
**Second recon batch, remaining 22 `src/models/` files, resolved
(2026-07-26, commits `49b4fe2d`, `1c649d56`, `6fe58dc3`, `e1335bf9`,
`5ddbf481`, `2928e80`, `4ad21b7e`)** — this closes out the
"model-health/drift/overfitting stack" and orphan items flagged as
documented-only in the paragraph above; superseded by what actually
happened:
- **Fixed, live bug**: `ModelHealthAnalyzer` called non-existent method
  names on all 4 sub-components; 2 of the 4 also needed a `model_results`
  dict it never built. Found `ModelAnalyzer` (a second, correct,
  previously-dead implementation of the same composition) already solves
  this — rewired `ModelHealthAnalyzer` to delegate to it, fixing both
  classes' orphan status at once.
- **Fixed, live bug**: `RegimeWinnerAnalyzer`'s constructor imported a
  class name (`MarketRegimeDetector`) that never existed (real name:
  `RegimeDetector`) — crashed on every instantiation, zero test coverage
  existed to catch it.
- **Fixed, live bug**: `PredictionDriftMonitor` read
  `self.reference_predictions`/`self.performance_history`/
  `self.drift_history`/`self.retraining_history` directly, but `__init__`
  only sets `self.history_manager` (which actually holds all four) —
  guaranteed `AttributeError` the first time enough samples accumulated
  for real drift detection. `self.drift_analyzer` already implements the
  correct modular equivalents; rewired to delegate to it. Zero test file
  existed for this class at all.
- **Fixed**: `src/models/__init__.py`'s stale `IntegratedModelManager`
  lazy-export (pointed at an already-archived module, raised
  `ModuleNotFoundError` instead of a clean `AttributeError`) — removed.
- **Fixed**: `PerformanceHistorySelector.critique_action()`
  (`smart_selector.py`) passed args in the wrong order into
  `_get_historical_reliability()`, silently defeating the key match and
  always returning the neutral 0.5 default regardless of real history.
- **Archived** (zero callers, zero tests, confirmed via grep):
  `ActionTrigger`, `ModelHealthEvaluator`, `ModelStatistics` (each the
  sole file in its directory — `actions/`/`health/`/`statistics/` removed);
  `LightModelInterface`/`HeavyModelInterface` in `adapters.py` (both also
  independently broken — wrong call signature into
  `LightModelTrainer.train_light_model`, and an import of a
  `ColabManager` path that's never existed); `UnifiedModelAdapter`;
  `handle_categorical_features_split`. `data_preparation.py`/
  `sentiment_integration.py` in the same `adapters/` directory are real,
  live, tested code — left untouched.
- **Found, documented only — real but low-priority, self-acknowledged
  incomplete feature**: `PrototypeRegistry._load_registry()` reconstructs
  every disk-persisted prototype with `model_class=Any` (the code's own
  comment admits "we need a way to resolve model_class from name... In
  production, we'd use a registry or importlib") — any prototype reloaded
  from JSON (vs. freshly registered in-process) is unclonable. Not fixed:
  the whole `EnhancedModelFactory`/`PrototypeRegistry` subsystem has zero
  live callers anywhere; its own test never exercises reload-then-clone
  together, so it currently passes despite the bug.
- **Found, NOT fixed — needs your input, this is live and risk-relevant,
  not a quick bug fix, HIGH PRIORITY for next discussion**:
  `DeanBootstrapSystem.bootstrap_action_critique()` requires at least one
  registered ACTOR and one registered CRITIC model, but **nothing in the
  live codebase ever calls `register_model()`** on the real singleton —
  confirmed via grep. This is called from a genuinely live path:
  `ConsensusEngine._apply_critic_filter()` (used by every real
  `ConsensusEngine.decide()`/`evaluate()` call), wrapped in a broad
  `except` that reduces it to `logger.warning(...)` +
  `critic_score = 0.0`. **This permanently-unconfigured safety filter has
  silently no-op'd on every single trade decision since it was written** —
  `ConsensusReport.blocked_by_critic`/`critic_score` look load-bearing but
  can never actually block a trade today, and there's no alert
  distinguishing "critic evaluated and passed" from "critic
  infrastructure was never initialized." Needs a deliberate decision:
  register real actor/critic models (and decide which ones), or
  explicitly disable/remove the feature rather than leave it silently
  inert. Full detail in `src/archive/MANIFEST.md`'s Wave 8 second batch.
- Combined verification across this whole batch: `tests/ -k "models"` →
  124 passed, 1 pre-existing unrelated failure (same
  `test_model_factory_import_does_not_top_level_import_neural_models` as
  before), zero regressions from any of the above.

**`src/models/` module is now fully audited, both recon batches closed
out.** Remaining loose ends, all deliberately deferred (not silently
dropped): `PersistentModelPool` disconnected duplicate cache (confirmed
harmless — `model_pool.py`'s real invalidation contract with
`base_trainer.py` is unaffected); duplicate dead `EnsembleModel` in
`src/ensembling/`; a dormant second `ConfidenceCalibrator` concept;
orphaned ensembling infra (`EnsembleComposer`, `DynamicWeightCalculator`,
`WeightStabilityMonitor` + 6 files, `ModelCorrelationAnalyzer` facade);
confidence calibration pooled across all tickers in
`adaptive_confidence_calibrator.py` (code's own comments call this
intentional — real design tradeoff, needs explicit discussion, not a
unilateral fix); the DEAN Critic gap above.

**`src/training/` pass complete (2026-07-26, commit `0d392bca`):** recon
subagent read all 11 files in full. Two real, live bugs found and fixed:
- `UnifiedTrainingManager._create_progressive_plan()` called
  `trainer.create_progressive_batches(tickers)` on a `ProgressiveTrainer` —
  that method only exists on `BatchProcessor`
  (`trainer.batch_processor.create_progressive_batches`) and needs 5 args,
  not 1. Guaranteed `AttributeError` whenever `modeling.strategy:
  "progressive"` is configured (one of 3 documented strategy choices) or
  whenever more than 5 tickers are trained with no strategy set
  (`_analyze_ticker_set` recommends `"progressive"` for `count > 5`).
  Fixed by delegating to `ProgressiveTrainer._prepare_ticker_groups({'tickers':
  tickers})`, mirroring the exact pattern this same file already uses for
  the `BATCH` branch two lines above. Verified: 7 synthetic tickers → real
  `ProgressiveTrainer()` now correctly splits into batches (5, then 2)
  instead of crashing.
- `base_trainer.py`'s `_train_individual_model()` and
  `light_model_trainer.py`'s `train_light_model()` both called
  `self.config_manager.get_config(f"models.{m_type}", {})` — same
  flat-lookup-on-a-dotted-string bug as the 9 other call sites fixed
  earlier this session, silently always returning `{}`. Switched to the
  hierarchical `.get()` accessor — but then verifying against the real
  config revealed the key path itself was ALSO wrong: real per-model
  hyperparameters live under `models.per_model.<type>` (confirmed
  `get('models.per_model.xgboost')` → `{'max_features': 48}`), not
  `models.<type>` directly. Fixed the path too. **Every model trained via
  either trainer had been silently training with empty/default
  hyperparameters this whole time**, regardless of what's actually tuned
  in config — this is exactly the kind of bug the "verify against real
  data, don't stop at the mock passing" discipline this session has
  repeatedly caught (found only because I checked the fixed value against
  real config instead of assuming the first fix was sufficient).
- Verified: `tests/ -k "base_trainer or light_model_trainer or
  training_manager or unified_training or progressive_trainer or
  batch_trainer or stage4"` → 21 passed, zero regressions.
- Recon read the other 9 files (`adaptive_training_manager.py`,
  `batch/batch_processor.py`, `batch_trainer.py`, `constants.py`,
  `run_training.py`, `security/path_security_validator.py`,
  `state/training_state_manager.py`) fully and found nothing else
  provably wrong — no path-traversal bypass, no state-persistence race,
  no chronological-ordering bug.

**`src/ensembling/` pass complete (2026-07-26, commits `25c30eba`,
`4841d512`):** closes out the "duplicate dead `EnsembleModel` in
`src/ensembling/`" and "orphaned ensembling infra" items flagged earlier.
- Archived 3 confirmed-dead files: `ensemble/ensemble_model.py` (broken
  import, not even exported by its own package, a separate correctly-
  tested live `EnsembleModel` already exists at
  `src/models/ensemble/ensemble_model.py`); `base_ensemble.py` (confirmed
  superseded duplicate of `stacked_ensemble.py`'s same classes — older,
  `pickle`-based, no path-security check, plus its own broken
  `ExperienceDiaryEngine` import — renamed to `DiaryEngine` long ago);
  `ensemble/archive/adaptive_ensemble.py` (was already informally set
  aside inside live `src/`, moved to the real archive convention).
- **Real bug found and fixed**: `src/scripts/modeling/train_consensus_model.py`
  (docstring: trains the meta-model "used by the real-time
  ConsensusEngine") imported `StackedEnsemble` from the stale
  `base_ensemble.py`, but the real live consumer
  (`src/trading/consensus_engine.py`) loads via
  `stacked_ensemble.StackedEnsemble.load()`, which expects a completely
  different on-disk format (joblib state dict vs. plain
  `pickle.dump(self)`). Even past the trainer's own import crash,
  whatever it produced would never have loaded in production — a genuine
  trainer/consumer format mismatch, no blast radius today only because
  the model file doesn't exist yet (graceful fallback). Fixed by
  redirecting the import to the real, live `stacked_ensemble` module.
- Same stale-import pattern also fixed in `compare_layers.py` for
  consistency, but that script has other unrelated bugs (missing
  `devtools.experimentation.base` module, a 5-field NamedTuple unpacked
  into 2 vars, a nonexistent `DiaryEngine.add_entry()` call) — documented
  in `MANIFEST.md`, not chased further; needs a full rewrite, belongs to
  a future `scripts/`/`devtools` pass.
- `caching.py` read in full, confirmed live and correct, no bugs found.
- Verified: `tests/ -k "ensemble or consensus"` → 27 passed, zero
  regressions.

**Decision (2026-07-26): user wants `src/pipeline/` (the core, ~117
files) finished completely before moving to peripheral directories**
(scripts/devtools/cli/dashboard etc. explicitly deprioritized — those are
auxiliary tools that call INTO the pipeline, not part of its execution).
Plan: sweep `src/pipeline/` in sub-batches — spine first (top-level
orchestration + stage 0-7 entry points + guards/), then `stages/<name>/`
subdirectories by size (modeling, evaluation, prediction, processing,
trading, feature_engineering, utils, the five 1-file dirs), then
`hybrid/` (36 files, partially already touched in the original Colab
pipeline audit at the top of this memory file — champion_selector.py,
results_processor.py, component_factory.py, path→model_path fix, dead
final_stages_executor.py/orchestrator_context.py already archived).

**`src/pipeline/` spine pass complete (2026-07-26, commit `f2574613`):**
recon subagent read all 23 files (top-level orchestration: constants.py,
hybrid_orchestrator.py, modeling_context.py, pipeline_factory.py,
pipeline_orchestrator.py, stage_loader.py, target_column_utils.py,
timeframe_lineage.py; stage 0-7 entry points; the 5 guard classes in
`guards/`). Confirmed clean: all 8 stage entry points correctly delegate
to their real `stages/<name>/` implementations, `pipeline_orchestrator.py`'s
sequencing/dependency/error-handling logic has no confirmed defect. 3
real bugs found in the guard classes themselves — these guards are
dormant (constructed by `FeatureGuards` but never invoked, a known gap
already documented for `macro_release_timing_guard.py`'s type-inference
incompatibility), so these 3 are SEPARATE bugs that would fire the
moment any of them gets wired in, independent of that known gap:
- `TemporalTargetGuard._process_target_config()` called
  `calc.calculate(df, **params)` uniformly for every target type, but
  `ClassificationCalculator` only exposes `calculate_binary()`/
  `calculate_multiclass()` — every classification target config raised
  `AttributeError`, silently caught and dropped. The real live
  `target_orchestrator.py` already solves this via a `METHOD_MAPPING`
  dict; mirrored that pattern. Verified: a `classification_binary` config
  now returns a real Series instead of silently `None`.
- `TimeframeAlignmentGuard.validate_timeframe_compatibility()` re-read
  `df['datetime'].max()` on the ORIGINAL per-timeframe frame after
  `_validate_single_timeframe()` had already fabricated a `'datetime'`
  column from a DatetimeIndex on its own *local* copy — any frame indexed
  by a bare DatetimeIndex (a completely normal shape elsewhere in this
  codebase) raised `KeyError('datetime')`. Fixed by applying the same
  column-fabrication in the caller's loop too. Verified against a
  DatetimeIndex-only frame.
- `MacroReleaseTimingGuard.get_safe_macro_data()` collected row labels via
  `.iterrows()` (`idx` is an index LABEL) into `valid_data`, then filtered
  with `.iloc[valid_indices]` — `.iloc` expects positions, not labels.
  Only worked by coincidence on a fresh default `RangeIndex`; on any
  non-default index (filtered/concatenated data, a realistic real-world
  shape) this either raises `IndexError` or silently returns the *wrong*
  rows as "safe" — exactly the failure mode this guard exists to
  prevent. Fixed `.iloc` → `.loc`. Verified against a non-default index.
- Verified: `tests/unit/test_stage3_data_contracts.py` → 10 passed,
  `tests/ -k "guard"` → 12 passed, zero regressions.
- Usage audit confirmed (table from recon): of the 5 guards, only
  `TemporalLeakageGuard` is actually invoked from `FeatureGuards.apply_guards()`
  (live). `TemporalTargetGuard` is dormant-but-tested (default-target path
  only). `MacroReleaseTimingGuard`/`SafeFeatureCombiner`/
  `TimeframeAlignmentGuard` are fully orphaned — constructed but zero
  methods ever called anywhere, zero test coverage. This matches the
  standing top-priority dry-run-then-ask item; the 3 bug fixes above
  don't change that recommendation (still don't wire these in without
  the separate dry-run-against-real-data step already planned).

**`stages/modeling/` and `stages/evaluation/` both complete** — full
detail in `src/archive/MANIFEST.md` Wave 10 (not duplicated here).
Highlights: modeling — archived a 5-file dead alternate training chain,
fixed a silent-except that could under-purge label leakage in
walk-forward validation. Evaluation — **most severe finding of this
pass**: `AdvancedBacktestEngine` never exposed its real simulated equity
curve, so every backtest evaluation report was silently computed from a
fabricated straight-line curve instead of the real path (max_drawdown
reported as ~0 regardless of real volatility) — fixed. Also fixed: silent
fallback to random fake data on thin input with zero downstream trace
(added `is_simulated_data` flag), a permanently-dead stress-test scenario
(key-name mismatch), a Series-vs-DataFrame shape bug that would silently
zero out financial metrics the moment a real `PortfolioMetricsCalculator`
is wired in, and a cross-ticker leak in a dormant analytics helper.
Archived one more confirmed-dead file (`data_recovery.py`).

**`stages/prediction/` complete** — full detail in `src/archive/MANIFEST.md`
Wave 10. Two silent wrong-prediction bugs fixed (the most dangerous class
for a live prediction stage): `ModelResolver`'s fallback model-loading
path collapsed its cache key to the literal string `"model"` for every
ticker matching the standard filename convention, so the first ticker
resolved through that path silently poisoned the shared `ModelPool`
cache for every other ticker afterward (each got served the FIRST
ticker's model). `PredictionGenerator`'s ensemble `context_params` had a
permanently-broken `ticker` (already-dropped column) and no `tf` key at
all, collapsing per-ticker live-performance-weighting/routing into one
shared bucket across every ticker. Both fixed and verified. Archived one
more dead duplicate (`data_preparer.py`, contained the OLD unfixed
zero-fill bug its live sibling `DataPreparationService` already fixed).

**`stages/processing/` (Stage 2) and `stages/trading/` (Stage 6) complete**
— full detail in `src/archive/MANIFEST.md` Wave 10. One genuine data-
corruption bug: the "persistent" macro-data parquet was silently
overwritten with just each cycle's incremental delta every run,
destroying all prior history (fixed with proper read-merge-dedupe-write).
Also fixed: reddit_sentiment data collected in Stage 1 silently vanishing
in Stage 2 (never copied into the dict the downstream filter reads);
Stage 2's "validation"/"quality metrics" were non-functional stubs always
reporting perfect data (wired in the real, already-constructed, non-
blocking validator); Stage 6's batch-lookup fallback missing the
`main_database` default check, silently returning no predictions on any
CLI run without an explicit batch_name.

**`stages/feature_engineering/` (remaining files) and `stages/utils/`
complete** — full detail in `src/archive/MANIFEST.md` Wave 10. Notably
clean batch: no leakage/lookahead bugs found despite maximum scrutiny
(this is the pipeline's most point-in-time-sensitive stage). One real
finding documented but not fixed: `FeatureEnricher` constructs a real
`FeatureCache` (genuine disk I/O, advertised 60-80% speedup) that's
never actually read anywhere — every Stage 3 run recomputes every
feature from scratch. Not fixed because wiring it in correctly needs a
real per-ticker/per-date batch-shape understanding and cache-key design,
not a one-line connection — doing it wrong risks silently serving stale
features, worse than the current "just slow" state. Archived 2 more dead
files (`stages/utils/` had no `__init__.py` at all — never even a real
package).

**Five 1-file stage dirs complete** — full detail in `src/archive/MANIFEST.md`
Wave 10. **Most severe finding of the entire `src/pipeline/` sweep**:
`CollectionStage._normalize_data()` (Stage 1, the very first stage) called
a `generate_hash` method that doesn't exist on most collectors — **4
currently-enabled collectors (cftc, fear_greed, put_call_ratio,
economic_calendar) were silently discarding every collected record on
every run**, never reaching the database, while logging what looked like
success. Fixed by computing the shared hash formula directly instead of
depending on an inconsistent per-collector method name. Also fixed in the
same file: collector crashes/timeouts were converted to `None` and
counted as "ran fine, nothing new" — a real failure was indistinguishable
from a benign empty result. Archived 3 more dead files.

**Self-corrected incident during this pass**: accidentally deleted
`stages/cache/feature_cache_manager.py` via `rm -rf` after `git mv`
silently failed — turned out `.gitignore` has a blanket `cache/` pattern
that had been hiding this real source file from version control the
whole time (only `src/core/cache/` is excepted). Recovered by finding an
identical copy under `.archive_docs/draft/...` and force-adding the
restored file. Confirmed no other `.py` files in `src/` are hidden by
this same pattern. **The gitignore pattern itself is still overly broad
and untouched — worth a deliberate fix in a future session.**

**`src/pipeline/` core sweep is now complete for spine + all `stages/`
subdirectories.** Only `hybrid/` (36 files, partially pre-covered by the
original Colab pipeline audit) remains before `src/pipeline/` can be
called fully closed out.

**Next steps**: `hybrid/` (36 files) — the last big chunk before
`src/pipeline/` is fully closed out. Then move to the deprioritized
peripheral list: agents, cli, colab, core, dashboard, devtools,
factories, integrations, main, meta_learning, metrics, monitoring
(top-level `src/monitoring/`, distinct from `src/models/monitoring/`
already covered), patterns, processing (top-level `src/processing/`),
scripts (has at least one confirmed-broken file, `compare_layers.py`),
sentiment, simulation, utils, validation. `training` and `ensembling`
are fully covered. Also worth a deliberate look: the `cache/` gitignore
overly-broad pattern flagged above.

**`src/pipeline/hybrid/` pass complete (2026-07-27, commits `415ce5ff`,
`d251dd59`, `192a4912`) — `src/pipeline/` core sweep is now FULLY closed
out (spine + all `stages/` + `hybrid/`).** Full detail in
`src/archive/MANIFEST.md` Wave 10. Recon subagent read all 36 files.
Key structural finding: ~15 of the ~20 components
`OrchestratorComponentFactory.initialize_components()` builds and attaches
to the orchestrator via `setattr` are never called by any live path —
`HybridOrchestrator`'s own public API only touches `pipeline_runner`,
`pipeline_manager`, `colab_manager`, `light_models_trainer`. Zero test
coverage for the whole dormant cluster (one apparent hit,
`test_pipeline_executor.py`, is a false positive — tests the unrelated
`src.cli.pipeline_executor`, which just shares a class name).
- **Fixed** (all verified, `tests/ -k "hybrid or colab_manager or
  selected_features or model_training_orchestrator or component_factory
  or pipeline_runner"` → 17 passed, zero regressions; one pre-existing
  unrelated smoke-test failure confirmed via `git stash` to predate this
  session): `colab_manager._load_single_file`'s wrapped-`models_metadata`
  overwrite-instead-of-merge bug (real shape confirmed against
  `scripts/colab/colab_clean_cell.py`'s writer; added a regression test,
  zero prior coverage existed); `selected_features_processor`'s
  leading-underscore method-name typo calling the feature-selection
  validator; `model_training_orchestrator`'s `context_data['features']`
  vs. the real producer's `'selected_features'` key (this one silently
  made `train_models_for_contexts` always train 0 models).
- **Archived** 4 confirmed fully-dead files (zero references anywhere,
  not even instantiation): `hybrid_dataclasses.py`, `storage_helpers.py`,
  `data_components_context.py`, `feature_loader.py`.
- **Checked, confirmed NOT a bug**: `component_factory.py` doesn't pass
  the shared `db_data_manager` into `PipelineRunner` (unlike
  `light_models_trainer`, which does get it) — `PipelineOrchestrator`
  falls back to building its own separate `DataManager`. Traced
  `DataManager.__init__`: the only real shared state (`_connections`,
  the DuckDB connection cache) is a classvar keyed by db_path, not
  per-instance, so a second instance still resolves to the same
  connection. Harmless redundant construction, not a correctness issue.
- **Deferred, asked the user, no response given (2026-07-27)**: whether
  to archive the entire ~15-component dormant cluster (each one
  duplicates a responsibility a working live component already handles —
  e.g. `colab_workflow_manager.py` superseded by inline logic already in
  `pipeline_manager.py`; `pipeline_executor.py`'s own stage methods are
  literal `# Implementation would go here` stubs, fully superseded by the
  real `pipeline_runner.py`), fix bugs cheaply and leave wired, or just
  document. Took the lower-risk, reversible path this session (fixed the
  3 confirmed contract bugs above in place, left everything else wired
  but untouched) rather than the bigger, harder-to-reverse factory
  rewrite. **Still an open question for you to weigh in on in a future
  session** — see `src/archive/MANIFEST.md`'s Wave 10 hybrid/ entry for
  the full component-by-component breakdown if you want to revisit.

**`src/pipeline/` is now fully audited end-to-end** (spine, all
`stages/<name>/` subdirectories, and `hybrid/`). Next: the deprioritized
peripheral `src/` directory list above (agents, cli, colab, core,
dashboard, devtools, factories, integrations, main, meta_learning,
metrics, monitoring, patterns, processing, scripts, sentiment,
simulation, utils, validation).

**Peripheral `src/` sweep begun (2026-07-27, commits `5c137c43`..`775ffb33`)
— first batch complete: `patterns/`, `sentiment/`, `factories/`,
`integrations/`, `simulation/`, `dashboard/` (11 files, batched together
since each dir was tiny).** Full detail in `src/archive/MANIFEST.md`.
Unusually severe batch — 2 of the 5 fixes were live, currently-broken,
high-impact bugs:
- **`src/dashboard/main_app.py` — the actual live dashboard entry point
  — was broken in 4 of its 6 tabs**: called `DataManager.load_data()`,
  a method that doesn't exist (real one: `fetch_df`). Fixed.
- **Every Stage 5 prediction with real news data present was silently
  failing**: `prediction/orchestrator.py` did `if news_data:` on a real
  `pd.DataFrame`, raising `ValueError` (ambiguous truth value), caught by
  a broad except that silently dropped the whole prediction for that
  context — not just the NLP adjustment step. Also fixed a second,
  deeper format bug in the same code path: the receiving function
  (`pattern_recognition_adjustment.py`) expects `list[dict]`, not a raw
  DataFrame. Fixed both (truthiness + `.to_dict('records')`).
- Also fixed: `ModelFactory` silently dropped per-model hyperparameter
  config for every model except KNN (LSTM/GRU/CNN/Transformer/TabNet/
  MLP/Autoencoder/SVM/Linear all trained with constructor defaults
  regardless of tuned config); 2 `ModelRegistry` entries
  (`dean_ensemble`/`sentiment`) referencing classes that don't exist
  anywhere in the live codebase, removed; `dashboard_data_bridge.py`'s
  SQLite-dialect `datetime()` calls against the real DuckDB backend
  (dormant — this bridge isn't wired into `main_app.py` yet, but the
  project's own prior audit notes recommend it should be — the correct
  long-term fix for the `main_app.py` bug above is probably wiring this
  bridge in rather than patching raw `fetch_df` calls forever).
- Archived 1 more confirmed-dead file: `GitHubActionsClient`
  (`src/integrations/infra/github_actions.py`), zero real callers,
  already independently flagged in this project's own
  `diagnostic_reports/orphan_modules.txt`.
- Verified: `tests/ -k "model_factory or model_registry or dashboard or
  pattern_recognition or prediction_orchestrator or stage5 or
  dashboard_data_bridge or tree_model_factory"` → 25 passed, zero
  regressions (1 pre-existing unrelated failure, already documented).

**Peripheral `src/` sweep, second batch complete (2026-07-27, commits
`a715b741`, `94be8bfc`) — `validation/`, `cli/`, `metrics/`,
`devtools/` (22 files).** Full detail in `src/archive/MANIFEST.md`.
Another severe batch:
- **Fixed, critical, two stacked bugs, zero prior test coverage**:
  `TimeSeriesValidator.validate_time_gaps()` called a nonexistent
  `calendar.get_trading_days()` (real API: `.trading_days` attribute,
  sliced by date range) — crashed every call. Even past that,
  `UnifiedValidator._check_time_continuity` (runs on every pipeline
  execution's Stage 2 validation) read dict keys
  (`has_gaps`/`gap_count`) the function never produces — silent no-op
  regardless of real gaps. Fixed both, added 4 regression tests.
- **Documented, NOT fixed, still broken**: `--mode calibrate` calls
  `PipelineExecutor.execute_calibrate_mode()`, which has never existed —
  a fully-advertised CLI mode, dead on arrival since it was written. No
  real calibration pipeline exists to wire in (only a synthetic-data
  demo script). Asked the user (document/build/remove) — no response,
  documented only per the lower-risk default. **Open decision for you.**
- **Documented, NOT fixed**: `dual_loops.py`'s default meta-learning
  update path calls `rule_generator.generate_rules_from_context(...)`,
  a method that was never built — the caller's own comments admit it's
  a "temporary compatibility layer" needing a real refactor. Needs
  design work (how to derive rule conditions/actions from trade
  records), not a mechanical fix.
- Archived 3 more confirmed-dead files: `task_manager.py` (also
  independently broken — imports a `Logger` class that doesn't exist),
  `system_validator.py`, `pipeline_data_loader.py` (superseded by inline
  reimplementation in `pipeline_executor.py`).
- Verified: `tests/ -k "stage2 or processing_stage or time_series or
  walk_forward or cross_val or purged"` → 38 passed, zero regressions.

**Peripheral `src/` sweep, third batch complete (2026-07-27, commits
`9b73bf28`..`2520ad8a`) — `utils/`, `monitoring/` top-level (25 files).**
Full detail in `src/archive/MANIFEST.md`. Another severe batch:
- **Fixed, LIVE**: `health_hub.py` (real pipeline component) fed the
  wrong disk-usage value (always 0.0, verified live against real 14.0%
  usage) into live ML risk-prediction models — wrong dict-key path vs.
  `ResourceMonitor`'s real schema. Also fixed `_load_performance_data`
  calling 2 nonexistent `DataManager` methods (real one: `fetch_df`),
  which always failed financial drift detection. 3 new regression tests
  (zero prior coverage).
- **Fixed, dormant-but-tested**: `ml_analytics.py` had the identical
  nonexistent-`DataManager`-method bug — undetected because its own
  test's fake stub matched the bug's method name instead of the real
  class. Also fixed `datetime.now().dayofweek` (pandas-only attribute,
  not on stdlib datetime) silently defeating feature extraction every
  call.
- **Fixed, cheap**: `monitoring/config.py` `NameError` on any non-numeric
  env var; `data_freshness_monitor.py` broken import path (module
  doesn't exist); `performance_reports.py` parsing a string format
  `ResourceMonitor` never produces (alerts could never fire).
- **Architectural finding, not fixed** (same shape as the earlier
  time-gap-detection bug): `DataFreshnessMonitor`/`FeatureDriftMonitor`
  are constructed live in the real feature-engineering pipeline but
  their check methods are never actually called after construction —
  silently does nothing every run. Added to the architectural-review list
  below.
- Archived 6 more confirmed-dead files (zero callers anywhere, including
  tests): `checkpoint_manager.py`, `json_utils.py`, `math_utils.py`
  (2 name-collision risks with unrelated live modules of similar name),
  `monitoring/base.py` (duplicate orphaned `BaseMonitor`),
  `drift_detector.py`, `performance_monitor.py`.
- Verified: 21 passed, zero regressions.

**Peripheral `src/` sweep, fourth batch complete (2026-07-27, commits
`69c2858c`..`d923520c`) — `main/`, `processing/` top-level (26 files).**
Full detail in `src/archive/MANIFEST.md`. Biggest structural finding of
the whole peripheral sweep so far:
- **`SystemOrchestrator` ("Central Control Center", docs call it "the
  primary hub"/"Production Ready") has ZERO live callers anywhere** —
  confirmed via repo-wide grep across all 110 root scripts. The real
  production path (`run_hybrid_pipeline.py` → `HybridOrchestrator` →
  `PipelineRunner`/`PipelineManager`) bypasses it entirely. Zero test
  coverage for `SystemOrchestrator`/`TrainMode`/`PredictMode`/
  `BacktestMode`/`IntelligentMode`/`WebUIMode`. **Deliberately NOT
  archived** (unlike smaller dead utilities this session) — flagged for
  your holistic project review instead, since this contradicts what the
  module's own documentation claims about it. 2 real bugs live entirely
  inside this dead dispatch path (moot unless it gets reconnected):
  `MonsterTestMode`'s `ticker`/`tickers` signature mismatch, and DEAN's
  "self-diagnosis retraining" guarding on an attribute
  (`experience_diary`) that's never set anywhere — permanent no-op.
  3 Mode classes DO have real standalone entry points bypassing
  `SystemOrchestrator` (`MonsterTestMode`, `ShadowBattleMode`,
  `HistoricalEventReplayMode` via their own `run_*.py` scripts).
- **Fixed, LIVE**: `historical_replay.py`'s 2 silent except blocks
  (zero logging) hiding real prediction failures behind a generic "no
  successful predictions" message; `shadow_battle.py`'s dead
  simulator/context construction; `price_preprocessor.py`'s unguarded
  `KeyError` on malformed input (live on every ingested price
  dataframe).
- Archived 2 more dead code blocks: 6 standalone functions from
  `processing/cleaners.py` (abandoned "unified schema" effort),
  `processing/parallel_processor.py` (whole file).
- Verified: 10 passed, zero regressions.

**SystemOrchestrator archived, user-confirmed (2026-07-27, commit
`792f4793`):** user confirmed they run via `run_hybrid_pipeline.py`, so
`system_orchestrator.py` + its 2 zero-caller-outside-it dependents
(`modes/intelligent.py`, `modes/web_ui.py`) archived to
`src/archive/main/`. `TrainMode`/`PredictMode`/`BacktestMode` left in
place (share the live `BaseMode` framework with 3 confirmed-live
standalone modes). 247 tests passed, zero regressions.

**Peripheral `src/` sweep, fifth batch complete (2026-07-27, commits
`7ced90e8`..`9bca7ca4`) — `meta_learning/` (19), `colab/` (20).** Full
detail in `src/archive/MANIFEST.md`.
- **Fixed, LIVE**: `diary_engine.py`'s `record_decision()` (runs on
  every real trading decision) computed a stable UUID `decision_id`
  (schema even migrated INTEGER→VARCHAR to support it) but never
  actually used it — still inserted a fresh collision-prone truncated
  int on every call. Fixed, added a regression test.
- **Fixed, dormant, cheap**: `dual_loops.py`'s `get_state()` crashed on
  a fresh arena with zero battles (`.get(key, default)` doesn't help
  when the key IS present but its value is `[]`).
- **Major finding, clarified user question, NOT archived**: `src/colab/`
  (20 files) has zero real callers and was completely unimportable
  (`ImportError`) — but its own README says this is *intentional*: meant
  to be uploaded manually into a Google Colab notebook, not imported
  locally. The real Colab-side script that runs today
  (`scripts/colab/colab_clean_cell.py`) doesn't import it either — it
  reimplements the same logic from scratch. Fixed the 2 confirmed bugs
  (the `ImportError`, a `self.logger`-doesn't-exist bug) but did NOT
  archive, since manual-upload usage can't be verified from this repo.
  **You clarified**: root `scripts/` (has `colab_clean_cell.py`) is real
  production tooling already covered by the original Colab audit; the 6
  standalone root `test_*.py` files are genuine ad-hoc test scripts,
  correctly out of scope; `src/scripts/` (22 files, part of `src/`) is
  NOT test-only — contains real operational tooling
  (`train_consensus_model.py`, `run_dashboard.py`, `hyperparameter_searcher.py`,
  etc.) — still in scope for the peripheral sweep.
- Verified: 51 passed, zero regressions.

**Peripheral `src/` sweep, sixth batch complete (2026-07-27, commits
`0fdfb308`, `a72a8b53`, `7d234224`) — `src/scripts/` (22 files).** Full
detail in `src/archive/MANIFEST.md`. Another batch of severe,
currently-broken live tools:
- **Fixed, LIVE**: `run_health_check.py` (docs call it "a key tool for
  ensuring stability") crashed with `TypeError` before ever producing a
  report — wrong-type constructor arg. Fixed, verified end-to-end
  (runs, produces a real health report against the live system).
- **Fixed, dormant**: `train_consensus_model.py` (trains the real-time
  `ConsensusEngine`'s meta-model) had 3 nonexistent `DataManager` method
  calls, including a `finally: data_manager.close()` that would have
  overridden even the successful/graceful-fallback path with a fresh
  crash on every single call. Fixed, verified end-to-end against the
  real database.
- Archived 2 confirmed dead+broken scripts: `generate_context_rules.py`
  (superseded by a working root-level `scripts/core/` equivalent),
  `ticker_config_updater.py` (broken import + path-depth bug + wrong
  config format reference, no working equivalent needed).
- **Documented, NOT fixed — needs a genuine rewrite**:
  `auto_accumulator.py` has multiple deeply stacked bugs (wrong module
  path AND the target function doesn't exist there anymore — real API
  is a class+method, not a function; a config double-extraction bug;
  another nonexistent-method bug) — even its own dedicated test can't
  collect and tests a completely different, nonexistent API shape than
  the real file. Same deferral class as `compare_layers.py` from the
  earlier `src/ensembling/` pass. The near-identical root-level copy
  (`scripts/core/auto_accumulator.py`) has the same broken import.
- No test coverage existed for any of the fixed/archived files in this
  batch (confirmed via search before each change).

**Peripheral `src/` sweep, seventh batch complete (2026-07-27, commits
`d2dd68af`, `cfce47f2`) — `src/agents/` (24 files). MOST SEVERE FINDING
OF THE ENTIRE PERIPHERAL SWEEP:**
- **The Investment Committee veto safety layer has been a permanent
  no-op in production since the day it was written (2026-07-22)**.
  `trading_orchestrator.py`'s `_apply_veto_committee` (runs on every
  real trading cycle, Stage 6) imported a module path that has never
  existed (`src.agents.veto_system` instead of the real
  `src.agents.archive.veto_system`) — `git log -S` shows the bug was
  introduced 36 seconds after the real singleton was created. Every
  call silently fell back to unvetoed consensus signals via a broad
  except. Even fixing that path alone wasn't enough — the target module
  itself had 2 more stale imports from an earlier repo reorg. **This
  also explains a false-dead-code classification from an earlier audit
  session**: `KnowledgeIngestor` was archived as dead code at the time
  because it genuinely had zero reachable callers then — entirely
  because of this same broken chain. Confirmed all deps (faiss,
  sentence-transformers, pypdf) installed and the knowledge-base data
  exists on disk — this is now a fully live, functional safety system,
  not just import-safe. Fixed all 3 imports, added 2 regression tests
  (zero prior coverage existed).
- Archived 1 more confirmed-dead file (`cognitive_pipeline.py`, same
  stale-import pattern, zero callers).
- Noted, not touched: 5 correctly-written but currently-dormant tool
  files (`comtrade_tool.py` etc.) whose only callers are now-archived
  dead code; the 12-lens "Cognitive Pipeline" system is live but always
  inert by design (every real caller passes `llm_client=None`).
- Verified: 19 passed, zero regressions.

**Peripheral `src/` sweep, EIGHTH AND FINAL batch complete (2026-07-27,
commits `1862bcf4`..`fd849de1`) — `src/core/` (33 files). PERIPHERAL
SWEEP NOW FULLY COMPLETE.** Full detail in `src/archive/MANIFEST.md`.
Foundational infrastructure, wide blast radius:
- **Fixed, LIVE, security**: `path_validator.py` — the sole containment
  gate for `FileManager`/`SecretsManager` — had the same sibling-
  directory boundary bug found earlier in `src/utils/path_safety.py`
  (there dormant, here live and reachable). Fixed, added a regression
  test proving the escape.
- **Fixed, LIVE**: `file_manager.py`'s `_atomic_write` couldn't catch
  its own deliberately-raised `OSError` — `.tmp` files never cleaned up,
  intended error log never fired, on the dominant real file-I/O failure
  mode (disk full, permission denied).
- **Fixed, LIVE**: `cache_manager.py` had 3 bugs: wrong constructor arg
  type into `DataManager` (masked today, every real caller happens to
  guard it), the same path-boundary bug as above, and except tuples
  missing `OSError` around real parquet/pickle/DuckDB I/O.
- **Fixed a REAL near-loss incident, not hypothetical**: the same
  overly-broad `cache/` gitignore pattern flagged earlier this session
  (after the `feature_cache_manager.py` incident) had already silently
  hidden `src/pipeline/cache/results_cache_manager.py` from git
  entirely (confirmed empty, zero callers — no content actually lost
  this time, but the exposure was real) and was blocking this session's
  own new test file from being staged. **Finally fixed**: anchored to
  `/cache/` (root-only), verified via `git check-ignore` that the real
  runtime cache dir is still ignored while all source `cache/`
  directories are not.
- **Fixed, cheap**: 2 smaller bugs in `http_client_factory.py` (a sync
  method wrapping an async one; a truthiness trap) and one in
  `base_integration.py` (unreachable graceful-degradation dict).
- Archived 9 more confirmed-dead files (zero test coverage): including
  a second, unrelated, same-named `validators.py`
  (`src/core/validation/`) distinct from the real, live
  `src/validation/validators.py::UnifiedValidator` — same
  duplicate-name-confusion pattern as several earlier findings.
- Left in place, dormant-but-tested (has real test coverage, so not
  archived per this sweep's established rule): `llm_client.py`,
  `core/utils/math_utils.py`.
- Verified: 199 passed, 2 skipped, zero regressions.

**THE ENTIRE PERIPHERAL `src/` SWEEP IS NOW COMPLETE.** Every directory
outside `src/pipeline/`, `src/models/`, `src/training/`, `src/ensembling/`
(all fully audited in earlier passes) has been read in full at least
once: patterns, sentiment, factories, integrations, simulation,
dashboard, validation, cli, metrics, devtools, utils, monitoring, main,
processing, meta_learning, colab, scripts, agents, core.

**Post-sweep session (2026-07-27): user reviewed the open architectural
decisions and made calls on all 4. Also found + fixed one more real bug
(commit `64fea9f0`) while investigating decision #3. This session's
work is done; a NEW chat continues execution — this memory entry is the
handoff.**

1. **DEAN Critic — GREENLIT, build it.** User: "we'll trade virtual
   assets anyway, gain experience, lose nothing — do it." Investigation
   found a real, well-built implementation already exists but is
   archived: `src/archive/meta_learning/dean_trading_models.py` —
   `DeanActor` (wraps any sklearn-style classifier into `decide_action()`)
   and `DeanCritic` (rules + a trained meta-model predicting the actor's
   expected error + macro/pattern-regime penalties like "TECH BUBBLE" +
   buying NVDA/TSLA + paradoxical-confidence detection). Matches
   `DeanBootstrapSystem`'s expected interface exactly
   (`src/models/dean/dean_bootstrap_system.py`). The whole
   actor/critic/reward loop (`register_model`, `bootstrap_action_critique`,
   `calculate_reward` — the "+1 to critic if it correctly warned"
   mechanism the user described) is fully coded but has literally never
   run: `register_model()` is never called anywhere, so
   `ConsensusEngine._apply_critic_filter()` (`src/trading/consensus_engine.py:204`,
   live on every real consensus decision) always hits the "no models
   registered" exception path and no-ops. **Next-session task**:
   un-archive `dean_trading_models.py`, train `DeanCritic`'s meta-model
   on historical (features, actual, actor-prediction) triples (can
   bootstrap from existing walk-forward/backtest data — doesn't need
   live trading first), register both models via
   `get_dean_system().register_model(...)`, and wire `calculate_reward()`
   to fire after each (virtual-portfolio, per user) trade outcome is
   known — likely via `DiaryEngine` or wherever Stage 6 outcomes are
   already tracked.
2. **`src/pipeline/hybrid/` dormant cluster — GREENLIT, archive it.**
   User: "do your recommendation." **Next-session task**: archive the
   ~15 components listed in the `src/archive/MANIFEST.md` hybrid/ entry
   (`cache_manager.py`, `orchestrator_interface.py`,
   `feature_selection_manager.py`, `feature_selection_validator.py`,
   `test_mode_manager.py`, `context_builder.py`,
   `data_manager.py`/`HybridDataManager`, `data_processor.py`,
   `data_utils.py`, `data_batch_manager.py`,
   `pipeline_metadata_manager.py`, `pipeline_executor.py`,
   `colab_workflow_manager.py`, `model_training_orchestrator.py`,
   `selected_features_processor.py`) and trim
   `component_factory.py` to stop constructing them.
3. **Build a `PipelinePolicyManager`-style consolidation layer —
   GREENLIT.** User asked whether an existing "pipeline-manager agent
   with pnl/train-test bounds" could host hyperparameter calibration.
   Investigation (this session) found **no such coherent layer exists**
   — PnL/train-test bounds are scattered across 5 disconnected places,
   several dead (full detail in `src/archive/MANIFEST.md`'s "Investigation
   finding" entry, same date). Key sub-finding, already fixed
   (commit `64fea9f0`): `portfolio_manager.py`'s kill-switch read the
   wrong config key (`max_daily_drawdown_pct` instead of the real
   `max_daily_loss_pct`), silently 67% more permissive than the
   configured 3% limit. User's decision: build one real
   `PipelinePolicyManager`-style component consolidating (a) risk
   limits, (b) train/test/validation split ratios — currently a
   hardcoded `DEFAULT_TEST_SIZE = 0.2` Python constant, config is dead
   duplicate — and (c) calibrated hyperparameters, once `--mode
   calibrate` is built for real (using the already-live, tested
   `BayesianOptimizer`/`OptimizationFactory` in `src/scripts/optimization/`,
   which the earlier `ModelFactory` config-passthrough fix this session
   now makes actually consumable). **Critical design constraint**: wire
   in the already-built `AdaptiveParameterManager`
   (`src/trading/adaptive_parameter_manager.py` — regime-aware drawdown
   limits, fully coded, currently instantiated with no config and its
   output never read by anything) rather than duplicating its regime
   logic — matches this audit's standing "fix/extend the existing
   mechanism, don't build a parallel one" rule. **Next-session task**:
   design and build this component from scratch — genuine architecture
   work, not a mechanical fix.
**MAJOR DISCOVERY (2026-07-27, same discussion): `Agents_architecture.md`
(1133 lines, root of repo, "DEAN-OS v4.2 — Фінальна архітектура
мультиагентної системи") already documents almost exactly the
multi-agent vision the user was describing, and large parts of it are
already REALLY implemented in `dean_os/` (confirmed via
`dean_os/IMPLEMENTATION_STATUS.md` — extensive real audit history, not
just a plan on paper).** This should be the starting point for all of
decisions #1/#3/#4 below and for the DEAN Critic work (decision, top of
this section) — read this doc + `dean_os/IMPLEMENTATION_STATUS.md` +
`dean_os/NEXT_CHAT_HANDOFF.md` in full before building anything new.

Key mapping (user's questions → what already exists):
- **"Agent per domain/sphere"** → `dean_os/`'s `SectorAgents`/domain
  analysts (semiconductor, agriculture, geopolitics, energy,
  macro_policy, liquidity_credit, logistics, real_estate + more per
  `domain_profiles.py`) — mature, already-audited implementation of
  exactly this idea. **`src/agents/modular_pipeline/`'s 12 "lens" files
  are very likely a separate, much thinner, disconnected duplicate of
  the same idea** (built without full awareness this architecture +
  dean_os implementation already existed) — recommend consolidating
  onto `dean_os`, retiring/repurposing the lenses.
- **"MLflow automation"** → maps directly to `Agents_architecture.md`
  section on `ModelPerformanceAgent` ("reads MLflow/Arena results,
  Soft gate"), already real in `dean_os` per IMPLEMENTATION_STATUS.md
  (extensive locked-evidence/lineage validation history). MLflow itself
  is already a real dependency (`requirements.txt`) and already
  partially wired into the live Colab training script
  (`scripts/colab/colab_clean_cell.py:898` `_log_mlflow_run` — logs
  ticker/target/model_type params + metrics + artifacts per run) —
  nobody has built the agent that *consumes* those logs for decisions
  yet.
- **`--mode calibrate` / hyperparameter tuning** → maps directly to
  `Agents_architecture.md` section 10, "TuningAgent — proposal
  lifecycle": a fully-specified Optuna-style multi-objective tuning
  objective (Sharpe minus cross-regime instability/drawdown/turnover/
  cost/complexity penalties), hard constraints (drawdown limit,
  unresolved P0 findings, synthetic-data guard), and a proper
  human-approval proposal lifecycle
  (`pending/approved/rejected/expired`, TTL, `allowed_for_production:
  bool = False` until approved). Already real in `dean_os` per
  IMPLEMENTATION_STATUS.md ("TuningAgent is implemented as
  proposal_only"). **Build calibration as this TuningAgent, not a
  bespoke standalone thing.**
- **Manager vs. Critic separation (user asked, confirmed)** — actually
  THREE layers, not two, once reconciled with this doc:
  1. `dean_os`'s `RiskAgent`/`TuningAgent`/`ModelPerformanceAgent` —
     review-only "analytical staff" (`Agents_architecture.md`'s own
     stated principle: "Агенти — аналітичний штаб, не автономні
     трейдери" — proposals go through human review, never auto-applied).
  2. DEAN Critic (`src/models/dean/dean_bootstrap_system.py` +
     `src/archive/meta_learning/dean_trading_models.py`) — explicitly
     meant to be the fast, automated, real-time gate wired directly
     into `ConsensusEngine._apply_critic_filter()`, deliberately
     separate from the slower human-reviewed analytical staff above.
     This is the one from the top of this section — still greenlit,
     still the next-session task.
  3. Deterministic real-time risk limits — `PortfolioManager`/
     `AdaptiveParameterManager` in `src/trading/` (already fixed this
     session, commit `64fea9f0`) — the actual automated kill-switch on
     live/virtual trades, distinct from `dean_os`'s `RiskAgent` (which
     per the stated principle is review-only, not a live auto-block).

**User confirmed**: not using any of these agents in production yet —
"at the construction stage." This means there's real freedom to
converge on ONE coherent design (per the user's own stated goal:
"уніфікувати, прибрати дублі" — unify, remove duplicates) rather than
needing to preserve any currently-relied-upon behavior.

**Two more strong matches found (2026-07-27, same discussion) for the
user's "historical horizon + world→sector/region→event probable-scenario
modeling" idea — both already substantially built in `dean_os/`:**
- `dean_os/event_causal_graph.py`'s `EventCausalGraphBuilder` — exactly
  the "a news event models probable downstream effects across
  sectors, with probabilities, not a prediction" idea (docstring's own
  example: "Earthquake hits Taiwan near TSMC fabs" → production halt →
  chip shortage → NVDA/AAPL effects, each step probabilistic). Its
  `build()` bug (undefined `watch_list` NameError) was already fixed
  earlier in this same standing audit (dean_os/agents/ pass). **Real
  gap found this session**: `CAUSAL_RULES` only operates at the sector
  level (semiconductor, logistics, energy, finance, etc.) — there's no
  country/region dimension at all, so the user's "steel sector in
  Germany vs. Japan vs. USA" granularity doesn't exist yet. Confirmed
  gap, not a guess.
- `dean_os/agents/historical_analogies.py`'s `HistoricalAnalogiesAgent`
  — exactly the "compare today against historical regimes since the
  Great Depression (tulip mania, etc.)" idea: matches structured
  world-state tags (regime verdict + per-sector tags) against a seed
  list of historical periods, not raw keyword matching against news
  text. **Confirmed permanently broken, but CORRECTED FINDING (found
  the real file after a slower background search completed)**:
  `_load_historical_periods()` looks for
  `dean_os/draft/dean_os_after_385_macro_regime_historical_hypothesis_kit/HISTORICAL_PERIODS_SEED_LIST.yaml`
  — that exact path doesn't exist, BUT **the real, substantive 56-line
  seed list already exists**, just one level too deep, inside an old
  full-snapshot draft folder:
  `dean_os/draft/dean_os_agent_system_v7/dean_os/draft/dean_os_after_385_macro_regime_historical_hypothesis_kit/HISTORICAL_PERIODS_SEED_LIST.yaml`.
  Confirmed real content, already well-curated: `late_1920s_pre_depression`,
  `great_depression` (1929-1939, tags: depression/deflation/banking_crisis/
  unemployment_shock/policy_experimentation), `wwii_mobilization`,
  `postwar_reconstruction_boom`, both 1970s oil shocks, `volcker_disinflation`,
  `globalization_1990s`, `dotcom_boom_bust`, `global_financial_crisis`, and
  more. **This is the exact same path-depth mismatch bug pattern found
  dozens of times throughout this whole audit.** Next-session task is
  now much cheaper than originally stated: copy/restore this file one
  level up to the path the live agent actually expects (or fix the
  agent's path computation to match where the file already lives) —
  not a content-curation task, a path fix + verification that the
  seed list's shape matches what `HistoricalAnalogiesAgent` actually
  parses (`historical_periods_seed_list` top-level key — already
  confirmed matching `_load_historical_periods()`'s expected shape).

**Book/literature knowledge base — how it works, and what's missing
(2026-07-27, same discussion).** User is actively adding books (Dalio,
Acemoglu, more planned) to `data/knowledge_base/books/` and asked how
chunking/search should be implemented. Answer: it already is, in the
same `KnowledgeIngestor` un-archived for the DEAN Critic fix above
(`src/archive/models_dead/knowledge_ingestor.py`):
- Extraction: `pypdf` per-page text extraction.
- Chunking: word-based sliding window, `chunk_size=1000` words,
  `overlap=200` words.
- Embedding: `sentence-transformers` `all-MiniLM-L6-v2`.
- Index: FAISS `IndexFlatL2` (exact search) at
  `data/memory/faiss_index/knowledge.index` + `metadata.json`
  (confirmed real data already present: Fukuyama, Superforecasters, a
  resource-wars paper, etc.).
- `ingest_new_books()` already exists and incrementally indexes any new
  PDF dropped into the books dir (skips already-indexed filenames by
  name) — but has no exposed entry point today (only reachable by
  uncommenting the `__main__` guard); a trivial one-line script would
  fix that.
**Real gap, not yet built**: `search()` only does semantic
retrieval (nearest-neighbor chunks) — there is no reasoning/synthesis
step. `AgenticVetoSystem` already wires the retrieved `context_chunks`
into its critique flow, but the actual decision is currently computed
by a stand-in (`_simulate_llm_decision`), not a real LLM call — so
"compare what Dalio vs. Acemoglu would say about this situation" isn't
actually happening yet; only the raw retrieval half is real. Wiring a
real LLM call (Anthropic/OpenAI) over the retrieved chunks + news is
the missing piece for genuine synthesis/comparison. Also worth noting
for later (not urgent at current library size): `IndexFlatL2` is exact
but O(n) per search — fine for thousands of chunks, would want an
approximate index (IVF/HNSW) if the library grows to hundreds of books.

**DATA-COLLECTION AUDIT (2026-07-28) — verified against the real
`data/trading_data.duckdb`, not config alone. 9 of 17 enabled collectors
have produced ZERO rows (no table in the DB at all).** Triggered by the
user asking whether to parse a Telegram channel (`t.me/Capitalizator_UA`)
for macro indicators. Conclusion on that: **don't parse it** — the
numbers it cites (CPI, PPI) are already collected from FRED directly, so
parsing is strictly worse (later, second-hand, prose-to-parse, typo
risk). Its real value is *indicator selection* — a one-time read of its
archive to find what to collect from primary sources. User agreed:
periodic manual review, no parser.

**Cross-referencing `collectors.yaml` (enabled flag + table_name) against
actual DB tables:**

| Collector | Table | Status |
|---|---|---|
| fred | fred_data (31839) | ✅ works |
| yahoo_finance | market_data_raw (123153) | ✅ works |
| huggingface | huggingface_data (999396) | ✅ works |
| sec_filings | sec_filings (19371) | ✅ works |
| rss | rss_news (7914) | ✅ works |
| google_news | google_news (5180) | ✅ works |
| vix | vix_data (**62**) | ⚠️ suspiciously few rows — worth checking |
| cftc | cftc_data | ❌ empty — **fixed this session (`2479709a`), not yet re-run** |
| economic_calendar | economic_calendar | ❌ empty — **same fix, not yet re-run** |
| fear_greed | fear_greed_data | ❌ empty — **source genuinely dead** (endpoint `production.datapoint.cloud` fails at the TLS layer; verified by both curl and httpx earlier this session). Needs a new data source or removal. |
| put_call_ratio | put_call_ratio_data | ❌ empty — **CBOE 403-blocks automated requests** (verified; the domain typo was separately fixed in `0bc95ec4`). Per standing rule, not circumvented. |
| reddit_sentiment | sociological_sentiment_data | ❌ empty — **but the source is ALIVE**: Reddit RSS returned 200 with the real collector User-Agent (verified earlier this session). So this is OUR bug, not a dead source. |
| wikimedia_attention | wikipedia_attention_data | ❌ empty — **source ALIVE** (Wikimedia pageviews returned 200, verified). Our bug. |
| sdmx_macro | macro_sdmx_data | ❌ empty — **source ALIVE** (World Bank SDMX returned 200, verified). Our bug. |
| aaii_sentiment | aaii_sentiment_data | ❌ empty — never investigated |
| insider | insider_trades | ❌ empty — never investigated |
| newsapi | newsapi_articles (2510) | disabled; rows are leftovers — fine |
| bigquery / custom_csv / free_google_trends / local_file | — | disabled, no data expected — fine |

**IMPORTANT — do NOT bulk-remove the empty ones.** Three of them
(`reddit_sentiment`, `wikimedia_attention`, `sdmx_macro`) have
independently-verified-live endpoints, so removing them would delete
working data sources over what is our own collector bug. Two more
(`cftc`, `economic_calendar`) were already fixed this session and simply
haven't been re-run. Only `fear_greed` (dead endpoint) and
`put_call_ratio` (deliberate CBOE block) are genuine remove-or-replace
candidates.

**Also confirmed dead — a FRED series with 0 rows:** `TEDRATE` is in the
`fred` collector's `series_ids` but has **zero rows** in `fred_data` (35
of 36 configured series present). Cause: FRED **discontinued TEDRATE in
January 2022** — the TED spread is computed from 3-month LIBOR, which
was phased out. Safe to delete from the config; the intent (interbank
credit stress) is better served by the credit-spread additions below.

**Indicator gaps found while analyzing the channel — worth ADDING (from
primary sources, not the channel):**
- **Inflation expectations — the biggest conceptual gap.** The pipeline
  collects what inflation *was* (`CPIAUCSL`/`PCEPI`/`PPIACO`) but not
  what the market *expects*: `T5YIE`, `T10YIE` (breakevens), `T5YIFR`
  (5y5y forward). For a forecasting system, expectations matter more
  than realized values.
- **`T10Y3M`** — the 10y-3m spread. Estrella/Mishkin research shows it
  predicts recessions better than the `T10Y2Y` already collected. One
  config line.
- **Net liquidity — currently only 1/3 of the picture.** `WALCL` (Fed
  balance sheet) is collected, but the measure macro traders actually
  watch is `WALCL − TGA − RRP`. Missing: `WTREGEN` (Treasury General
  Account), `RRPONTSYD` (reverse repo). Without them `WALCL` alone is
  misleading.
- **`NFCI`** (Chicago Fed financial conditions, weekly, aggregates 100+
  inputs) and **`BAMLC0A0CM`** (investment-grade spread — the pipeline
  has only the high-yield `BAMLH0A0HYM2`; together they show whether
  stress is confined to junk or has spread to quality).
- **`ICSA`** (initial jobless claims, weekly) — the pipeline has `CCSA`
  (continuing claims) but not the more timely initial series.
- **`SAHMREALTIME`** — real-time recession trigger.
- **FINRA margin debt** — cited by the channel, not on FRED; FINRA
  publishes monthly. Classic leverage/froth gauge. Needs its own
  collector.

**Caveat recorded for the user (raised, understood): the pipeline
already has 35 macro series. Mechanically adding 10 more as *model
features* risks overfitting on limited training samples.** Suggested
split: only the ones with real predictive logic (inflation expectations,
`T10Y3M`, net liquidity) as model features; the rest (`NFCI`,
`SAHMREALTIME`, margin debt) as *regime context* for the dean_os
analysts/lenses rather than raw gradient-boosting inputs.

**Recommended first task for the new chat**: reconcile
`Agents_architecture.md` + `dean_os/IMPLEMENTATION_STATUS.md` +
`dean_os/NEXT_CHAT_HANDOFF.md` against real current code (the doc's own
header explicitly says to do this before implementing any next step) —
produce one clear picture of what's real vs. aspirational vs.
duplicated, THEN proceed with the DEAN Critic wiring, TuningAgent-based
calibration, and ModelPerformanceAgent-based MLflow consumption as
parts of this one architecture, not as separate ad-hoc builds.

4. **`auto_accumulator.py` rewrite — GREENLIT.** User: doesn't check
   data gaps manually, considers the auto-heal tool valuable. **Next-
   session task**: the real rewrite deferred earlier this sweep — fix
   the wrong module path (`src.data.collector_factory` →
   `src.data.collectors.collector_factory`), adapt the call site to the
   real `CollectorFactory(configs, http_client_factory).get_all_collectors()`
   class+method API (not the old standalone-function shape), fix the
   `AssetUniverseManager` double-config-extraction + nonexistent
   `'day_trading_tech'` preset, fix `get_all_tables()` →
   `get_all_table_names()`. Also fix or rewrite the near-identical root
   copy `scripts/core/auto_accumulator.py` and reconcile
   `tests/scripts/data/test_auto_accumulator.py`, which currently tests
   a completely different, nonexistent API shape than the real file.
   `compare_layers.py` was NOT discussed this session — still an open,
   separate "needs a rewrite" item, not yet greenlit either way.

**Architectural / design-decision items accumulated so far (not fixed by
this audit's normal fast-path — each needs a deliberate decision, not a
mechanical fix). User asked for a holistic project-level review after
the peripheral sweep finishes; this list is the input for that:**
1. ~~`--mode calibrate`~~ — **DECISION 2026-07-27**: build it for real,
   folded into decision #3's `PipelinePolicyManager` (calibration output
   becomes one of the things that component serves). See the "Post-sweep
   session" entry above for full detail. Not yet implemented.
2. ~~`src/pipeline/hybrid/`'s ~15-component dormant cluster~~ —
   **GREENLIT 2026-07-27**: archive it. See "Post-sweep session" entry
   above for the full component list. Not yet implemented.
3. `dual_loops.py`'s rule-generation call — self-admitted "temporary
   compatibility layer" against a method that was never built; needs
   real design work on how to derive rule conditions/actions from trade
   records.
4. ~~DEAN Critic actor/critic never registered~~ — **GREENLIT 2026-07-27**:
   build it for real. A complete, well-built implementation already
   exists archived at `src/archive/meta_learning/dean_trading_models.py`
   (`DeanActor`/`DeanCritic` — rules + trained meta-model + macro-regime
   penalties). See "Post-sweep session" entry above for the full plan.
   Not yet implemented.
5. `FeatureCache` in Stage 3 built but never read — every run
   recomputes every feature from scratch (60-80% speedup left unused).
6. 3 orphaned pipeline guards (`MacroReleaseTimingGuard`,
   `SafeFeatureCombiner`, `TimeframeAlignmentGuard`) — built, tested,
   never wired into `FeatureGuards.apply_guards()`. Needs a dry-run
   against real data before wiring in.
7. `DataFreshnessMonitor`/`FeatureDriftMonitor` — constructed live in the
   feature-engineering pipeline, check methods never actually called.
8. ~~`.gitignore`'s overly-broad `cache/` pattern~~ — **RESOLVED
   2026-07-27**: anchored to `/cache/` during the `src/core/` batch,
   after it caused a second near-loss incident (silently hid
   `src/pipeline/cache/results_cache_manager.py` from git entirely,
   though that file turned out to be empty/dead so no content was
   actually lost).
9. Adaptive confidence calibration pooled across all tickers in
   `adaptive_confidence_calibrator.py` — code's own comments call this
   intentional, but flagged as a real design tradeoff worth explicit
   discussion.
10. ~~`SystemOrchestrator`~~ — **RESOLVED 2026-07-27**: user confirmed
    they run via `run_hybrid_pipeline.py`; archived (commit `792f4793`).
11. ~~`src/colab/`~~ — **RESOLVED 2026-07-27**: user confirmed they still
    manually move databases to disk and train via the Colab workflow —
    correctly left unarchived; the 2 confirmed bugs (ImportError,
    self.logger) are already fixed.
12. **Tools needing a genuine rewrite, not a mechanical fix** (deferred,
    documented, not silently dropped): `src/scripts/experiments/compare_layers.py`
    (from the `src/ensembling/` pass — stale import, NamedTuple unpacking
    bug, nonexistent method call — **NOT discussed 2026-07-27, still an
    open item either way**) and ~~`src/scripts/data/auto_accumulator.py`~~
    / `scripts/core/auto_accumulator.py` — **GREENLIT 2026-07-27**: user
    finds the auto-heal-data-gaps concept valuable and doesn't check
    manually today; worth the rewrite. (wrong-module-path import where
    the target function no longer exists there either — real API is a
    class+method now — plus a config double-extraction bug and another
    nonexistent-method bug; its own dedicated test can't even collect
    and tests a completely different API shape than the real file).
    Worth a dedicated session each.
