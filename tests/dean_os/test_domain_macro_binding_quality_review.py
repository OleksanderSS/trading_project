from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dean_os.domain_macro_binding_quality_review import (
    DomainMacroBindingQualityReview,
)
from dean_os.system_journal import SystemJournal


REQUESTED = [
    "DCOILWTICO",
    "INDPRO",
    "CPIAUCSL",
    "PPIACO",
    "FEDFUNDS",
    "DGS10",
    "VIXCLS",
]
AS_OF = "2026-05-07T19:11:04+00:00"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path: Path, present: list[str]) -> tuple[Path, Path]:
    candidate_path = tmp_path / "candidate.json"
    _write_json(
        candidate_path,
        {
            "run_id": "candidate_fixture",
            "created_at": AS_OF,
            "producer_contract": "dean_domain_scoped_macro_envelope_v1",
            "domain_id": "energy",
            "inputs": {"as_of": AS_OF},
            "domain_binding": {
                "requested_series_scope": REQUESTED,
                "present_series_scope": present,
            },
            "summary": {
                "candidate_ready_for_binding_review": True,
                "binding_accepted": False,
            },
            "safety": {"review_only": True},
        },
    )
    plan_path = tmp_path / "plan.json"
    _write_json(
        plan_path,
        {
            "summary": {"domain_id": "energy"},
            "family_plans": [
                {
                    "context_family": "macro",
                    "status": "reuse_candidate_ready_for_review",
                    "proposed_candidate": {"sha256": _sha256(candidate_path)},
                }
            ],
        },
    )
    return candidate_path, plan_path


def _build(tmp_path: Path, present: list[str], **kwargs) -> dict:
    candidate, plan = _artifacts(tmp_path, present)
    return DomainMacroBindingQualityReview(tmp_path / "report").build(
        domain_id="energy",
        candidate_path=candidate,
        binding_plan_path=plan,
        review_as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
        **kwargs,
    )


def test_rates_only_candidate_is_replaced(tmp_path: Path) -> None:
    payload = _build(tmp_path, ["DGS10"])

    assert payload["summary"]["recommendation"] == "replace_candidate"
    assert payload["summary"]["quality_score"] == 0.2
    assert payload["series_assessment"]["required_missing"] == [
        "DCOILWTICO",
        "INDPRO",
    ]
    assert payload["summary"]["decision_recorded"] is False
    assert payload["summary"]["can_invoke_domain_analysis"] is False


def test_required_but_thin_context_is_deferred(tmp_path: Path) -> None:
    payload = _build(tmp_path, ["DCOILWTICO", "INDPRO"])

    assert payload["summary"]["recommendation"] == "defer"
    assert payload["summary"]["required_coverage"] == 1.0
    assert payload["summary"]["supporting_coverage"] == 0.0


def test_threshold_candidate_recommends_acceptance_without_deciding(tmp_path: Path) -> None:
    payload = _build(
        tmp_path,
        ["DCOILWTICO", "INDPRO", "DGS10", "FEDFUNDS"],
    )

    assert payload["summary"]["recommendation"] == "accept_binding"
    assert payload["summary"]["quality_score"] == 0.85
    assert payload["summary"]["total_coverage"] == 0.571429
    assert payload["manual_gate"]["status"] == "pending_explicit_binding_decision"
    assert payload["summary"]["binding_accepted"] is False


def test_sha_mismatch_blocks_coverage_recommendation(tmp_path: Path) -> None:
    candidate, plan = _artifacts(
        tmp_path,
        ["DCOILWTICO", "INDPRO", "DGS10", "FEDFUNDS"],
    )
    plan_payload = json.loads(plan.read_text(encoding="utf-8"))
    plan_payload["family_plans"][0]["proposed_candidate"]["sha256"] = "0" * 64
    _write_json(plan, plan_payload)

    payload = DomainMacroBindingQualityReview(tmp_path / "report").build(
        domain_id="energy",
        candidate_path=candidate,
        binding_plan_path=plan,
        review_as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["summary"]["recommendation"] == "defer"
    assert "candidate_sha_not_bound_to_current_plan" in payload["summary"][
        "structural_blockers"
    ]
    assert payload["summary"]["quality_score"] == 0.7


def test_journal_append_is_idempotent_and_hash_chain_remains_valid(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path, ["DGS10"], apply_journal=True)
    second = _build(tmp_path, ["DGS10"], apply_journal=True)

    assert first["journal"]["appended_count"] == 1
    assert second["journal"]["appended_count"] == 0
    assert second["journal"]["existing_count"] == 1
    assert SystemJournal(tmp_path / "journal.jsonl").status()["chain_valid"] is True


def test_saved_markdown_exposes_review_boundary(tmp_path: Path) -> None:
    candidate, plan = _artifacts(tmp_path, ["DGS10"])
    payload = DomainMacroBindingQualityReview(tmp_path / "report").build(
        domain_id="energy",
        candidate_path=candidate,
        binding_plan_path=plan,
        review_as_of=AS_OF,
        journal_path=tmp_path / "journal.jsonl",
        save=True,
    )

    latest = Path(payload["saved_paths"]["latest_markdown"])
    report = latest.read_text(encoding="utf-8")
    assert "Machine recommendation: `replace_candidate`" in report
    assert "This packet recommends only" in report
