from __future__ import annotations

import json
from pathlib import Path

from dean_os.domain_analyst_binding_planner import DomainAnalystBindingPlanner
from dean_os.domain_binding_task_dispatcher import DomainBindingTaskDispatcher
from dean_os.domain_scoped_macro_envelope import (
    DomainScopedMacroEnvelopeCeremony,
    load_verified_domain_macro_context_fragment,
)


AS_OF = "2026-07-14T12:00:00Z"


def test_no_source_waits_without_running_adapter(tmp_path):
    dispatch = _dispatch(tmp_path)
    payload = DomainScopedMacroEnvelopeCeremony().build(
        dispatch_path=dispatch,
        as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["summary"]["status"] == "awaiting_explicit_local_macro_source"
    assert payload["summary"]["adapter_run_performed"] is False
    assert payload["summary"]["candidate_ready_for_binding_review"] is False
    assert payload["journal"]["events_proposed"] == 2
    assert payload["summary"]["can_trade"] is False


def test_explicit_source_runs_once_and_builds_domain_candidate(tmp_path):
    source = _macro_csv(tmp_path)
    payload = DomainScopedMacroEnvelopeCeremony().build(
        dispatch_path=_dispatch(tmp_path),
        source_path=source,
        as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["summary"]["status"] == "domain_macro_binding_candidate_ready_with_scope_gaps"
    assert payload["summary"]["adapter_run_performed"] is True
    assert payload["summary"]["candidate_ready_for_binding_review"] is True
    assert payload["summary"]["binding_accepted"] is False
    assert payload["domain_id"] == "energy"
    assert payload["domain_binding"]["present_series_scope"] == ["DCOILWTICO", "INDPRO"]
    assert payload["safety"]["single_adapter_run_limit"] == 1
    assert payload["safety"]["automatic_retry_allowed"] is False


def test_saved_envelope_is_accepted_as_candidate_not_binding(tmp_path):
    source = _macro_csv(tmp_path)
    envelope = DomainScopedMacroEnvelopeCeremony(tmp_path / "envelope").build(
        dispatch_path=_dispatch(tmp_path),
        source_path=source,
        as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
    )
    plan = DomainAnalystBindingPlanner(tmp_path / "replan").build(
        as_of=AS_OF,
        candidate_artifacts={"macro": [envelope["saved_paths"]["latest_json"]]},
        save=False,
    )
    macro = next(item for item in plan["family_plans"] if item["context_family"] == "macro")

    assert macro["status"] == "reuse_candidate_ready_for_review"
    assert macro["binding_written"] is False
    assert plan["summary"]["can_invoke_domain_analysis_now"] is False


def test_recursive_loader_rebuilds_saved_macro_core(tmp_path):
    envelope = DomainScopedMacroEnvelopeCeremony(tmp_path / "envelope").build(
        dispatch_path=_dispatch(tmp_path),
        source_path=_macro_csv(tmp_path),
        as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
    )
    fragment = load_verified_domain_macro_context_fragment(
        envelope["saved_paths"]["latest_json"],
        expected_domain_id="energy",
        expected_as_of=AS_OF,
    )

    assert fragment["metadata"]["domain_macro_envelope_verified"] is True
    assert "industrial_production" in fragment["macro"]


def test_journal_apply_is_idempotent_for_same_preview(tmp_path):
    source = _macro_csv(tmp_path)
    dispatch = _dispatch(tmp_path)
    journal = tmp_path / "journal.jsonl"
    ceremony = DomainScopedMacroEnvelopeCeremony(tmp_path / "envelope")
    first = ceremony.build(
        dispatch_path=dispatch,
        source_path=source,
        as_of=AS_OF,
        journal_path=journal,
        apply_journal=True,
    )
    second = ceremony.build(
        dispatch_path=dispatch,
        source_path=source,
        as_of=AS_OF,
        journal_path=journal,
        apply_journal=True,
    )

    assert first["journal"]["appended_count"] == 2
    assert second["journal"]["appended_count"] == 0
    assert second["journal"]["existing_count"] == 2
    assert second["journal"]["record_count"] == 2
    assert second["journal"]["chain_valid"] is True


def test_future_only_source_cannot_create_candidate(tmp_path):
    source = tmp_path / "future.csv"
    source.write_text(
        "datetime,series_id,value,available_at\n"
        "2026-07-15T00:00:00Z,DCOILWTICO,80,2026-07-15T01:00:00Z\n",
        encoding="utf-8",
    )
    payload = DomainScopedMacroEnvelopeCeremony().build(
        dispatch_path=_dispatch(tmp_path),
        source_path=source,
        as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["summary"]["status"] in {
        "blocked_macro_core_not_ready",
        "blocked_no_domain_relevant_macro_observations",
    }
    assert payload["summary"]["candidate_ready_for_binding_review"] is False


def test_pipeline_shape_without_availability_is_explicitly_blocked(tmp_path):
    source = tmp_path / "pipeline_macro.csv"
    source.write_text(
        "datetime,series,value,hash\n"
        "2026-07-10,DCOILWTICO,78.5,abc\n",
        encoding="utf-8",
    )
    payload = DomainScopedMacroEnvelopeCeremony().build(
        dispatch_path=_dispatch(tmp_path),
        source_path=source,
        as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["summary"]["status"] == "blocked_macro_core_not_ready"
    assert payload["core_preview"]["schema_mapping"]["availability_column"] is None
    assert "macro_schema_missing_availability_column" in payload["core_preview"]["exclusion_reasons"]
    assert payload["summary"]["candidate_ready_for_binding_review"] is False


def test_report_exposes_scope_gaps_and_boundaries(tmp_path):
    payload = DomainScopedMacroEnvelopeCeremony(tmp_path / "envelope").build(
        dispatch_path=_dispatch(tmp_path),
        source_path=_macro_csv(tmp_path),
        as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
    )
    markdown = (tmp_path / "envelope" / "latest.md").read_text(encoding="utf-8")

    assert "Candidate ready for binding review: True" in markdown
    assert "Binding accepted: False" in markdown
    assert "One explicit local source, one offline adapter pass" in markdown
    assert payload["saved_paths"]["latest_json"].endswith("latest.json")


def test_sha_bound_execution_gate_allows_offline_envelope_after_replacement_collection(
    tmp_path,
):
    dispatch_path = Path(_dispatch(tmp_path))
    dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
    macro = next(
        item for item in dispatch["task_dispatches"] if item["context_family"] == "macro"
    )
    macro["recommended_action"] = "review_sha_bound_candidate_without_rebuilding_source_artifact"
    dispatch_path.write_text(json.dumps(dispatch), encoding="utf-8")
    scope = [
        "CPIAUCSL",
        "DCOILWTICO",
        "DGS10",
        "FEDFUNDS",
        "INDPRO",
        "PPIACO",
        "VIXCLS",
    ]
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "contract": "dean_domain_macro_collection_execution_gate_v1",
                "mode": "domain_macro_collection_execution_gate",
                "domain_id": "energy",
                "summary": {
                    "status": "macro_collection_execution_ready_single_run",
                    "single_run_authorized": True,
                },
                "execution_ticket": {
                    "domain_id": "energy",
                    "request_as_of": AS_OF,
                    "series_scope": scope,
                    "maximum_collection_runs": 1,
                    "automatic_retry_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = DomainScopedMacroEnvelopeCeremony().build(
        dispatch_path=dispatch_path,
        execution_gate_path=gate_path,
        source_path=_macro_csv(tmp_path),
        as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["summary"]["status"] == "domain_macro_binding_candidate_ready_with_scope_gaps"
    assert payload["inputs"]["execution_gate_sha256"]


def _dispatch(tmp_path: Path) -> str:
    plan = DomainAnalystBindingPlanner(tmp_path / "plan").build(as_of=AS_OF)
    dispatch = DomainBindingTaskDispatcher(tmp_path / "dispatch").build(
        binding_plan_path=plan["saved_paths"]["latest_json"]
    )
    return dispatch["saved_paths"]["latest_json"]


def _macro_csv(tmp_path: Path) -> Path:
    path = tmp_path / "macro.csv"
    path.write_text(
        "datetime,series_id,value,available_at\n"
        "2026-07-10T00:00:00Z,DCOILWTICO,78.5,2026-07-10T23:59:00Z\n"
        "2026-07-09T00:00:00Z,INDPRO,101.2,2026-07-10T12:00:00Z\n",
        encoding="utf-8",
    )
    return path
