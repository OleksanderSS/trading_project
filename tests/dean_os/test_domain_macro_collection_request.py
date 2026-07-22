from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from dean_os.domain_analyst_lifecycle_profile import DomainAnalystLifecycleProfileCompiler
from dean_os.domain_macro_binding_quality_review import CONTRACT as QUALITY_CONTRACT
from dean_os.domain_macro_collection_request import DomainMacroCollectionRequest
from dean_os.system_journal import SystemJournal

AS_OF = "2026-07-14T09:50:00+00:00"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path, recommendation: str = "replace_candidate") -> tuple[Path, Path]:
    profile = DomainAnalystLifecycleProfileCompiler().compile("energy")
    policy = profile["domain_overlay"]["macro_binding_quality_policy"]
    candidate = tmp_path / "candidate.json"
    _write_json(
        candidate,
        {
            "domain_id": "energy",
            "inputs": {"as_of": AS_OF},
            "domain_binding": {
                "requested_series_scope": sorted(policy["required_series"] + policy["supporting_series"]),
                "present_series_scope": ["DGS10"],
            },
        },
    )
    quality = tmp_path / "quality.json"
    _write_json(
        quality,
        {
            "mode": "domain_macro_binding_quality_review",
            "contract": QUALITY_CONTRACT,
            "domain_id": "energy",
            "inputs": {
                "candidate_sha256": _sha(candidate),
                "profile_domain_overlay_sha256": profile["domain_overlay_sha256"],
                "review_as_of": AS_OF,
            },
            "summary": {
                "status": "quality_review_ready_recommendation_only",
                "recommendation": recommendation,
                "structural_blockers": [],
                "decision_recorded": False,
                "binding_accepted": False,
            },
            "series_assessment": {
                "required_series": sorted(policy["required_series"]),
                "supporting_series": sorted(policy["supporting_series"]),
            },
        },
    )
    return candidate, quality


def _build(tmp_path: Path, **kwargs) -> dict:
    candidate, quality = _artifacts(tmp_path, kwargs.pop("recommendation", "replace_candidate"))
    return DomainMacroCollectionRequest(tmp_path / "report").build(
        domain_id="energy",
        quality_review_path=quality,
        candidate_path=candidate,
        request_as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
        **kwargs,
    )


def test_builds_full_replacement_and_exact_gap_scope(tmp_path: Path) -> None:
    payload = _build(tmp_path)

    assert payload["summary"]["status"] == "macro_collection_request_ready"
    assert payload["summary"]["replacement_scope_count"] == 7
    assert payload["summary"]["gap_series_count"] == 6
    assert payload["collection_request"]["missing_required_series"] == ["DCOILWTICO", "INDPRO"]
    assert payload["collection_request"]["refresh_existing_series"] == ["DGS10"]
    assert payload["summary"]["execution_authorized"] is False
    assert payload["summary"]["network_access_performed"] is False


def test_request_encodes_point_in_time_and_single_pass_contract(tmp_path: Path) -> None:
    payload = _build(tmp_path)

    assert payload["point_in_time_contract"]["availability_field"] == "realtime_start"
    assert payload["point_in_time_contract"]["observation_date_is_not_availability"] is True
    assert payload["collection_request"]["runtime_parameters"]["maximum_collection_runs"] == 1
    assert payload["collection_request"]["runtime_parameters"]["automatic_retry_allowed"] is False
    assert payload["collection_request"]["runtime_parameters"]["runtime_override_supported"] is True
    assert payload["collection_request"]["runtime_parameters"]["fred_vintage_dates_parameter_required"] is True


def test_accept_recommendation_does_not_create_collection_request(tmp_path: Path) -> None:
    payload = _build(tmp_path, recommendation="accept_binding")

    assert payload["summary"]["status"] == "macro_collection_not_required"
    assert payload["summary"]["request_required"] is False
    assert payload["journal"]["events_proposed"] == 0


def test_candidate_sha_mismatch_blocks_request(tmp_path: Path) -> None:
    candidate, quality = _artifacts(tmp_path)
    candidate.write_text(candidate.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    payload = DomainMacroCollectionRequest(tmp_path / "report").build(
        quality_review_path=quality,
        candidate_path=candidate,
        request_as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["summary"]["status"] == "macro_collection_request_blocked"
    assert "candidate_sha_mismatch" in payload["summary"]["structural_blockers"]


def test_registry_mapping_gap_blocks_request(tmp_path: Path) -> None:
    candidate, quality = _artifacts(tmp_path)
    registry = yaml.safe_load(Path("dean_os/config/macro_series_registry.yaml").read_text(encoding="utf-8"))
    del registry["series"]["DCOILWTICO"]
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    payload = DomainMacroCollectionRequest(tmp_path / "report").build(
        quality_review_path=quality,
        candidate_path=candidate,
        registry_path=registry_path,
        request_as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert "registry_mapping_missing:DCOILWTICO" in payload["summary"]["structural_blockers"]


def test_journal_is_idempotent_and_report_exposes_boundary(tmp_path: Path) -> None:
    candidate, quality = _artifacts(tmp_path)
    builder = DomainMacroCollectionRequest(tmp_path / "report")
    args = {
        "quality_review_path": quality,
        "candidate_path": candidate,
        "request_as_of": AS_OF,
        "journal_path": tmp_path / "journal.jsonl",
        "apply_journal": True,
        "save": True,
    }
    first = builder.build(**args)
    second = builder.build(**args)

    assert first["journal"]["appended_count"] == 1
    assert second["journal"]["appended_count"] == 0
    assert SystemJournal(tmp_path / "journal.jsonl").status()["chain_valid"] is True
    report = Path(second["saved_paths"]["latest_markdown"]).read_text(encoding="utf-8")
    assert "This artifact prepares one request only" in report
