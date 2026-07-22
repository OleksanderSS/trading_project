from __future__ import annotations

import json
from pathlib import Path

import pytest

from dean_os.domain_context_set import (
    CONTRACT,
    FAMILY_ORDER,
    REQUIRED_CONTRACTS,
    DomainContextSetAssembler,
    load_verified_domain_context_set,
)


DOMAIN = "semiconductor_ai_infrastructure"
CUTOFF = "2026-07-10T19:50:45.683169+00:00"
FAMILY_AS_OF = "2026-06-30T21:00:00+00:00"


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidates(tmp_path: Path) -> dict[str, Path]:
    return {
        family: _write(
            tmp_path / f"{family}.json",
            {
                "contract": REQUIRED_CONTRACTS[family],
                "domain_id": DOMAIN,
                "inputs": {"as_of": FAMILY_AS_OF},
                "summary": {
                    "structural_blockers": [],
                    "coverage_gaps": [],
                },
            },
        )
        for family in FAMILY_ORDER
    }


def _install_loaders(monkeypatch, *, blocked_family: str | None = None) -> None:
    loaders = {}
    for family in FAMILY_ORDER:
        if family == blocked_family:
            def _blocked(*args, **kwargs):
                raise ValueError("domain sector-market envelope is not ready")

            loaders[family] = _blocked
        else:
            def _verified(path, *, expected_domain_id, expected_as_of, family=family):
                return {
                    "domain_id": expected_domain_id,
                    "as_of": expected_as_of,
                    family: {"verified": True},
                }

            loaders[family] = _verified
    monkeypatch.setattr("dean_os.domain_context_set._LOADERS", loaders)


def test_complete_set_preserves_family_timestamps_but_does_not_invoke(
    tmp_path: Path, monkeypatch
) -> None:
    _install_loaders(monkeypatch)
    payload = DomainContextSetAssembler(tmp_path / "reports").build(
        domain_id=DOMAIN,
        analysis_cutoff=CUTOFF,
        candidate_artifacts=_candidates(tmp_path),
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["contract"] == CONTRACT
    assert payload["status"] == "domain_context_set_candidate_ready"
    assert payload["summary"]["verified_family_count"] == 6
    assert payload["summary"][
        "family_timestamp_alignment_required"
    ] is False
    assert payload["summary"]["can_invoke_domain_analysis"] is False
    assert payload["binding_gate"]["decision_recorded"] is False


def test_partial_set_keeps_five_verified_families_and_proposes_no_execution(
    tmp_path: Path, monkeypatch
) -> None:
    _install_loaders(monkeypatch, blocked_family="sector_market")
    payload = DomainContextSetAssembler(tmp_path / "reports").build(
        domain_id=DOMAIN,
        analysis_cutoff=CUTOFF,
        candidate_artifacts=_candidates(tmp_path),
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["status"] == "domain_context_set_incomplete"
    assert payload["summary"]["verified_family_count"] == 5
    assert payload["summary"]["missing_families"] == ["sector_market"]
    assert "sector_market" not in payload["verified_family_fragments"]
    proposal = payload["collection_proposals"][0]
    assert proposal["execution_authorized"] is False
    assert proposal["automatic_collection_allowed"] is False
    assert "pipeline_control_saved_price_repair" in proposal[
        "required_upstream_chain"
    ]


def test_future_family_artifact_fails_closed(tmp_path: Path, monkeypatch) -> None:
    _install_loaders(monkeypatch)
    candidates = _candidates(tmp_path)
    future = json.loads(candidates["macro"].read_text(encoding="utf-8"))
    future["inputs"]["as_of"] = "2026-07-11T00:00:00+00:00"
    _write(candidates["macro"], future)

    payload = DomainContextSetAssembler().build(
        domain_id=DOMAIN,
        analysis_cutoff=CUTOFF,
        candidate_artifacts=candidates,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    macro = next(
        item
        for item in payload["family_receipts"]
        if item["context_family"] == "macro"
    )
    assert macro["verification_status"] == "blocked"
    assert macro["not_future_data"] is False
    assert payload["summary"]["can_invoke_domain_analysis"] is False


def test_unknown_family_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported domain context families"):
        DomainContextSetAssembler().build(
            domain_id=DOMAIN,
            analysis_cutoff=CUTOFF,
            candidate_artifacts={"made_up": tmp_path / "x.json"},
            journal_path=tmp_path / "journal.jsonl",
            save=False,
        )


def test_saved_partial_set_is_recursively_verified(
    tmp_path: Path, monkeypatch
) -> None:
    _install_loaders(monkeypatch, blocked_family="sector_market")
    payload = DomainContextSetAssembler(tmp_path / "reports").build(
        domain_id=DOMAIN,
        analysis_cutoff=CUTOFF,
        candidate_artifacts=_candidates(tmp_path),
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )
    saved = _write(tmp_path / "context_set.json", payload)

    verified = load_verified_domain_context_set(
        saved,
        expected_domain_id=DOMAIN,
        expected_analysis_cutoff=CUTOFF,
    )

    assert verified["status"] == "domain_context_set_incomplete"
    assert verified["missing_families"] == ["sector_market"]
    assert verified["complete"] is False
    assert verified["can_invoke_domain_analysis"] is False


def test_saved_set_loader_rejects_fragment_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    _install_loaders(monkeypatch)
    payload = DomainContextSetAssembler(tmp_path / "reports").build(
        domain_id=DOMAIN,
        analysis_cutoff=CUTOFF,
        candidate_artifacts=_candidates(tmp_path),
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )
    payload["verified_family_fragments"]["news"]["news"] = {
        "verified": False
    }
    saved = _write(tmp_path / "tampered_context_set.json", payload)

    with pytest.raises(ValueError, match="saved content mismatch"):
        load_verified_domain_context_set(saved)
