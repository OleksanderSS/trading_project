from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from dean_os.context_acquisition_state_machine import ContextAcquisitionStateMachine
from dean_os.analyst_core.domain_analyst_binding_planner import DomainAnalystBindingPlanner
from dean_os.domain_scoped_fundamentals_envelope import (
    CONTRACT,
    DomainScopedFundamentalsEnvelope,
    load_verified_domain_fundamentals_context_fragment,
)


AS_OF = "2026-06-30T21:00:00+00:00"
DOMAIN = "semiconductor_ai_infrastructure"
CIKS = {
    "AMD": "0000002488",
    "INTC": "0000050863",
    "NVDA": "0001045810",
    "TSM": "0001046179",
}


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merger(tmp_path: Path, *, wrong_cik: bool = False) -> Path:
    facts = [
        {
            "ticker": ticker,
            "cik": "0000000001" if wrong_cik and ticker == "AMD" else cik,
            "metric_name": "revenue",
        }
        for ticker, cik in CIKS.items()
    ]
    return _write(
        tmp_path / "merged.json",
        {
            "run_id": "merged_fixture",
            "producer_contract": "dean_saved_sec_fundamental_evidence_merger_v1",
            "status": "merged_fundamental_evidence_ready_with_gaps",
            "inputs": {
                "as_of": AS_OF,
                "requested_tickers": list(CIKS),
            },
            "summary": {
                "accepted_fact_tickers": list(CIKS),
                "can_claim_complete_sector_fundamentals": False,
            },
            "facts": facts,
            "source_artifacts": [{"family": "sec_companyfacts"}],
            "safety": {
                "review_only": True,
                "network_access_performed": False,
                "valuation_performed": False,
                "pipeline_run_performed": False,
                "training_run_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "live_execution_performed": False,
            },
        },
    )


def _ratio(tmp_path: Path, merger: Path) -> Path:
    return _write(
        tmp_path / "ratio.json",
        {
            "run_id": "ratio_fixture",
            "producer_contract": "dean_saved_sec_derived_ratio_evidence_v1",
            "status": "derived_ratio_evidence_ready_with_gaps",
            "inputs": {
                "as_of": AS_OF,
                "merged_fundamental_artifact_path": str(merger),
                "merged_fundamental_artifact_sha256": _sha(merger),
            },
            "summary": {
                "derived_tickers": list(CIKS),
                "can_claim_full_cohort_comparability": False,
            },
            "integration_boundary": {"review_only": True},
            "safety": {
                "network_access_performed": False,
                "pipeline_run_performed": False,
                "training_run_performed": False,
                "tuning_run_performed": False,
                "learning_write_performed": False,
                "live_execution_performed": False,
            },
        },
    )


def _dispatch(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "dispatch.json",
        {
            "mode": "domain_binding_task_dispatch",
            "summary": {"domain_id": DOMAIN},
            "task_dispatches": [
                {
                    "context_family": "fundamentals",
                    "recommended_action": (
                        "domain_scoped_fundamentals_evidence_envelope"
                    ),
                }
            ],
        },
    )


def _issuer_registry(tmp_path: Path) -> Path:
    # A self-contained snapshot matching exactly the CIKS tickers this fixture
    # uses -- not the shared dean_os/config/semiconductor_issuer_identity_registry.yaml,
    # whose issuer count this test's exact-scope-match assertions must not be
    # coupled to (that file legitimately grew to cover the domain's full
    # ticker_universe_hint; this test intentionally exercises a *narrower*
    # registry scope than the full domain universe to hit the "with_gaps"
    # partial-coverage path).
    path = tmp_path / "issuer_identity_registry.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "registry_id": "dean_semiconductor_issuer_identity_registry_v1",
                "domain_id": DOMAIN,
                "review_status": "initial_mapping_requires_manual_acceptance",
                "eligibility": {
                    "strong_source_tiers": [
                        "tier_1_primary_or_wire",
                        "tier_2_strong_context",
                    ],
                    "minimum_independent_strong_sources": 2,
                    "require_consistent_directional_stance": True,
                    "plain_substring_match_allowed": False,
                    "raw_fundamental_fact_is_directional_evidence": False,
                    "sector_context_can_close_ticker_lane": False,
                },
                "issuers": {
                    ticker: {
                        "cik": cik,
                        "legal_name": f"{ticker} Test Issuer",
                        "aliases": [ticker],
                    }
                    for ticker, cik in CIKS.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _build(tmp_path: Path, monkeypatch, *, wrong_cik: bool = False) -> dict:
    merger = _merger(tmp_path, wrong_cik=wrong_cik)
    ratio = _ratio(tmp_path, merger)
    registry_path = _issuer_registry(tmp_path)
    monkeypatch.setattr(
        "dean_os.domain_scoped_fundamentals_envelope._configured_registry_path",
        lambda policy: registry_path,
    )
    monkeypatch.setattr(
        "dean_os.domain_scoped_fundamentals_envelope."
        "load_verified_merged_fundamental_context_fragment",
        lambda path, expected_as_of=None: {
            "as_of": expected_as_of,
            "fundamentals": {ticker: {"revenue": 1.0} for ticker in CIKS},
            "metadata": {"merged_verified": True},
        },
    )
    monkeypatch.setattr(
        "dean_os.domain_scoped_fundamentals_envelope."
        "load_verified_derived_ratio_context_fragment",
        lambda path, expected_as_of=None: {
            "as_of": expected_as_of,
            "fundamentals": {
                ticker: {"operating_margin": 0.2} for ticker in CIKS
            },
            "metadata": {"ratios_verified": True},
        },
    )
    return DomainScopedFundamentalsEnvelope(tmp_path / "reports").build(
        domain_id=DOMAIN,
        as_of=AS_OF,
        source_path=ratio,
        dispatch_path=_dispatch(tmp_path),
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )


def test_verified_partial_fundamentals_candidate_uses_shared_state_machine(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _build(tmp_path, monkeypatch)

    assert payload["contract"] == CONTRACT
    assert payload["summary"]["status"] == (
        "domain_fundamentals_candidate_ready_with_gaps"
    )
    assert payload["summary"]["source_lineage_verified"] is True
    assert payload["summary"]["issuer_identity_verified"] is True
    assert payload["coverage"]["profile_ticker_coverage_ratio"] == 0.333333
    assert payload["summary"]["producer_run_performed"] is False
    assert payload["summary"]["binding_accepted"] is False
    assert payload["summary"]["can_trade"] is False

    envelope = _write(tmp_path / "envelope.json", payload)
    machine = ContextAcquisitionStateMachine(
        ledger_path=tmp_path / "transition_ledger.jsonl",
        journal_path=tmp_path / "state_journal.jsonl",
        output_dir=tmp_path / "machine",
    )
    result = machine.advance(
        acquisition_id="fundamentals_semis_1",
        domain_id=DOMAIN,
        context_family="fundamentals",
        stage_id="candidate_verified",
        artifact_path=envelope,
        evaluated_at=AS_OF,
        apply_transition=True,
        apply_journal=True,
        save=False,
    )

    assert result["summary"]["status"] == "transition_recorded"
    assert result["summary"]["persisted_state"] == "awaiting_binding_decision"
    assert result["summary"]["binding_accepted"] is False


def test_tampered_upstream_merger_hash_fails_before_recursive_loading(
    tmp_path: Path, monkeypatch
) -> None:
    merger = _merger(tmp_path)
    ratio = _ratio(tmp_path, merger)
    merger.write_text("{}", encoding="utf-8")
    called = False

    def _unexpected(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        "dean_os.domain_scoped_fundamentals_envelope."
        "load_verified_merged_fundamental_context_fragment",
        _unexpected,
    )
    payload = DomainScopedFundamentalsEnvelope().build(
        domain_id=DOMAIN,
        as_of=AS_OF,
        source_path=ratio,
        dispatch_path=_dispatch(tmp_path),
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert called is False
    assert "merged_fundamental_sha256_mismatch" in payload["summary"][
        "structural_blockers"
    ]
    assert payload["summary"]["candidate_ready_for_binding_review"] is False


def test_fact_ticker_cik_mismatch_fails_closed(tmp_path: Path, monkeypatch) -> None:
    payload = _build(tmp_path, monkeypatch, wrong_cik=True)

    assert "fact_ticker_cik_mismatch:AMD" in payload["summary"][
        "structural_blockers"
    ]
    assert payload["summary"]["issuer_identity_verified"] is False
    assert payload["summary"]["can_invoke_domain_analysis"] is False


def test_binding_planner_accepts_only_the_domain_envelope(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _build(tmp_path, monkeypatch)
    envelope = _write(tmp_path / "envelope_for_binding.json", payload)
    plan = DomainAnalystBindingPlanner(tmp_path / "binding_plan").build(
        domain_id=DOMAIN,
        candidate_artifacts={"fundamentals": [envelope]},
        as_of=AS_OF,
        save=False,
    )
    family = next(
        item
        for item in plan["family_plans"]
        if item["context_family"] == "fundamentals"
    )

    assert family["status"] == "reuse_candidate_ready_for_review"
    assert family["proposed_candidate"]["contract"] == CONTRACT
    assert family["binding_written"] is False


def test_saved_fundamentals_envelope_is_recursively_verified(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _build(tmp_path, monkeypatch)
    envelope = _write(tmp_path / "verified_envelope.json", payload)

    fragment = load_verified_domain_fundamentals_context_fragment(
        envelope,
        expected_domain_id=DOMAIN,
        expected_as_of=AS_OF,
    )

    assert sorted(fragment["fundamentals"]) == sorted(CIKS)
    assert sorted(fragment["derived_fundamental_ratios"]) == sorted(CIKS)
    assert fragment["metadata"][
        "domain_fundamentals_envelope_verified"
    ] is True


def test_saved_fundamentals_loader_rejects_fragment_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _build(tmp_path, monkeypatch)
    payload["market_context_fragment"]["fundamentals"]["AMD"][
        "revenue"
    ] = 999.0
    envelope = _write(tmp_path / "tampered_envelope.json", payload)

    try:
        load_verified_domain_fundamentals_context_fragment(envelope)
    except ValueError as exc:
        assert "fragment mismatch" in str(exc)
    else:
        raise AssertionError("tampered fundamentals fragment was accepted")
