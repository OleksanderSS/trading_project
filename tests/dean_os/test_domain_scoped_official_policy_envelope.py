from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dean_os.context_acquisition_state_machine import ContextAcquisitionStateMachine
from dean_os.domain_scoped_official_policy_envelope import (
    CONTRACT,
    DomainScopedOfficialPolicyEnvelope,
    load_verified_domain_official_policy_context_fragment,
)


AS_OF = "2026-06-30T21:00:00+00:00"
DOMAIN = "semiconductor_ai_infrastructure"
URL = "https://media.bis.gov/media/documents/policy.pdf"
IDENTITY = "us_bureau_industry_security"


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixtures(tmp_path: Path, *, cross_bind_news: bool = False) -> dict[str, Path]:
    raw = tmp_path / "policy.pdf"
    raw.write_bytes(b"%PDF-fixture")
    raw_sha = _sha(raw)
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "registry_version: fixture",
                "review_status: agent_verified_official_source_review_only",
                "documents:",
                f"  {raw_sha}:",
                "    title: Policy fixture",
                f'    published_at: "2026-05-31T00:00:00+00:00"',
                f"    source_url: {URL}",
                f"    source_identity: {IDENTITY}",
                "    source_tier: tier_1_core_evidence",
                "    evidence_type: policy_or_geopolitical",
                "    semantic_claim: A policy exists.",
            ]
        ),
        encoding="utf-8",
    )
    snapshot = _write(
        tmp_path / "snapshot.json",
        {
            "snapshot_contract": "dean_official_policy_source_snapshot_v1",
            "status": "official_policy_snapshot_ready",
            "source": {
                "source_identity": IDENTITY,
                "source_tier": "tier_1_core_evidence",
                "final_url": URL,
                "sha256": raw_sha,
                "immutable_path": str(raw),
            },
        },
    )
    legacy_news = _write(tmp_path / "legacy-news.json", {"fixture": True})
    news_source = tmp_path / ("other-news.json" if cross_bind_news else "legacy-news.json")
    if cross_bind_news:
        _write(news_source, {"fixture": "other"})
    news = _write(
        tmp_path / "news-envelope.json",
        {
            "contract": "dean_domain_scoped_news_envelope_v1",
            "mode": "domain_scoped_news_envelope",
            "domain_id": DOMAIN,
            "status": "domain_news_candidate_ready_with_gaps",
            "inputs": {
                "domain_id": DOMAIN,
                "as_of": AS_OF,
                "source_path": str(news_source),
                "source_sha256": _sha(news_source),
            },
            "summary": {
                "source_lineage_verified": True,
                "trigger_semantics_preserved": True,
                "binding_accepted": False,
                "can_trade": False,
            },
        },
    )
    source = _write(
        tmp_path / "policy-source.json",
        {
            "run_id": "policy_fixture",
            "producer_contract": "dean_saved_official_policy_evidence_producer_v1",
            "status": "official_policy_evidence_ready",
            "inputs": {
                "snapshot_artifact_path": str(snapshot),
                "snapshot_artifact_sha256": _sha(snapshot),
                "corroborating_news_artifact_path": str(legacy_news),
                "corroborating_news_artifact_sha256": _sha(legacy_news),
                "registry_path": str(registry),
                "registry_sha256": _sha(registry),
                "as_of": AS_OF,
            },
            "source_provenance": {
                "source_identity": IDENTITY,
                "source_tier": "tier_1_core_evidence",
                "final_url": URL,
                "sha256": raw_sha,
                "published_at": "2026-05-31T00:00:00+00:00",
            },
            "corroboration": {
                "existing_independent_strong_sources": ["bloomberg"],
                "official_source_identity": IDENTITY,
                "combined_independent_sources": ["bloomberg", IDENTITY],
            },
            "summary": {
                "policy_lane_ready": True,
                "can_enter_market_context_review": True,
                "can_trade": False,
            },
            "integration_boundary": {
                "review_only": True,
                "official_source_hash_bound": True,
                "independent_corroboration_required": True,
                "plain_text_ticker_promotion_allowed": False,
                "automatic_prediction_influence": False,
                "automatic_trading_allowed": False,
            },
            "safety": {
                "review_only": True,
                "network_access_performed": False,
                "pipeline_run_performed": False,
                "training_run_performed": False,
                "learning_write_performed": False,
                "live_execution_performed": False,
            },
        },
    )
    dispatch = _write(
        tmp_path / "dispatch.json",
        {
            "mode": "domain_binding_task_dispatch",
            "summary": {"domain_id": DOMAIN},
            "task_dispatches": [
                {
                    "context_family": "official_policy",
                    "recommended_action": "domain_scoped_official_policy_envelope",
                }
            ],
        },
    )
    return {
        "raw": raw,
        "registry": registry,
        "snapshot": snapshot,
        "legacy_news": legacy_news,
        "news": news,
        "source": source,
        "dispatch": dispatch,
    }


def _patch_profile(monkeypatch, registry: Path) -> None:
    monkeypatch.setattr(
        "dean_os.domain_scoped_official_policy_envelope."
        "DomainAnalystLifecycleProfileCompiler.compile",
        lambda self, domain_id: {
            "contract": "profile-contract",
            "fixed_contract_sha256": "a" * 64,
            "domain_overlay_sha256": "b" * 64,
            "readiness": {"schema_valid": True},
            "domain_overlay": {
                "official_policy_binding_policy": {
                    "source_registry_path": str(registry),
                    "allowed_official_hosts": ["media.bis.gov"],
                    "allowed_source_identities": [IDENTITY],
                    "max_source_age_days": 120,
                    "minimum_independent_news_sources": 1,
                    "accepted_registry_review_statuses": ["operator_accepted"],
                }
            },
        },
    )


def test_verified_policy_candidate_uses_shared_state_machine(
    tmp_path: Path, monkeypatch
) -> None:
    files = _fixtures(tmp_path)
    _patch_profile(monkeypatch, files["registry"])
    monkeypatch.setattr(
        "dean_os.domain_scoped_official_policy_envelope."
        "load_verified_official_policy_context_fragment",
        lambda path, expected_as_of=None: {
            "as_of": expected_as_of,
            "news": [{"title": "official policy", "published_at": AS_OF}],
            "metadata": {"saved_official_policy_verified": True},
        },
    )
    monkeypatch.setattr(
        "dean_os.domain_scoped_official_policy_envelope."
        "load_verified_domain_news_context_fragment",
        lambda path, expected_domain_id=None, expected_as_of=None: {
            "as_of": expected_as_of,
            "domain_id": expected_domain_id,
            "news": [{"title": "corroboration"}],
            "metadata": {"domain_news_envelope_sha256": "c" * 64},
        },
    )
    payload = DomainScopedOfficialPolicyEnvelope(tmp_path / "reports").build(
        domain_id=DOMAIN,
        as_of=AS_OF,
        source_path=files["source"],
        news_envelope_path=files["news"],
        dispatch_path=files["dispatch"],
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["contract"] == CONTRACT
    assert payload["summary"]["status"] == (
        "domain_official_policy_candidate_ready_with_gaps"
    )
    assert payload["summary"]["source_lineage_verified"] is True
    assert payload["summary"]["policy_fact_established"] is True
    assert payload["summary"]["directional_market_claim_created"] is False
    assert payload["summary"]["hypothesis_confirmed"] is False
    assert "official_policy_registry_pending_operator_acceptance" in payload[
        "quality"
    ]["quality_gaps"]

    envelope = _write(tmp_path / "policy-envelope.json", payload)
    verified = load_verified_domain_official_policy_context_fragment(
        envelope,
        expected_domain_id=DOMAIN,
        expected_as_of=AS_OF,
    )
    assert verified["metadata"]["domain_official_policy_envelope_verified"] is True
    machine = ContextAcquisitionStateMachine(
        ledger_path=tmp_path / "transition-ledger.jsonl",
        journal_path=tmp_path / "state-journal.jsonl",
        output_dir=tmp_path / "machine",
    )
    transition = machine.advance(
        acquisition_id="official_policy_semis_1",
        domain_id=DOMAIN,
        context_family="official_policy",
        stage_id="candidate_verified",
        artifact_path=envelope,
        evaluated_at=AS_OF,
        apply_transition=True,
        apply_journal=True,
        save=False,
    )
    assert transition["summary"]["status"] == "transition_recorded"
    assert transition["summary"]["persisted_state"] == "awaiting_binding_decision"
    assert transition["summary"]["binding_accepted"] is False


def test_news_cross_binding_mismatch_fails_before_recursive_loaders(
    tmp_path: Path, monkeypatch
) -> None:
    files = _fixtures(tmp_path, cross_bind_news=True)
    _patch_profile(monkeypatch, files["registry"])
    called = False

    def _unexpected(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        "dean_os.domain_scoped_official_policy_envelope."
        "load_verified_official_policy_context_fragment",
        _unexpected,
    )
    monkeypatch.setattr(
        "dean_os.domain_scoped_official_policy_envelope."
        "load_verified_domain_news_context_fragment",
        _unexpected,
    )
    payload = DomainScopedOfficialPolicyEnvelope().build(
        domain_id=DOMAIN,
        as_of=AS_OF,
        source_path=files["source"],
        news_envelope_path=files["news"],
        dispatch_path=files["dispatch"],
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert called is False
    assert "official_policy_news_lineage_cross_binding_failed" in payload[
        "summary"
    ]["structural_blockers"]
    assert payload["summary"]["candidate_ready_for_binding_review"] is False


def test_tampered_raw_pdf_fails_closed(tmp_path: Path, monkeypatch) -> None:
    files = _fixtures(tmp_path)
    _patch_profile(monkeypatch, files["registry"])
    files["raw"].write_bytes(b"not-a-pdf")
    payload = DomainScopedOfficialPolicyEnvelope().build(
        domain_id=DOMAIN,
        as_of=AS_OF,
        source_path=files["source"],
        news_envelope_path=files["news"],
        dispatch_path=files["dispatch"],
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert "official_policy_raw_source_not_pdf" in payload["summary"][
        "structural_blockers"
    ]
    assert "official_policy_raw_pdf_sha256_mismatch" in payload["summary"][
        "structural_blockers"
    ]
    assert payload["summary"]["can_trade"] is False


def test_verified_loader_rejects_directional_boundary_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    files = _fixtures(tmp_path)
    _patch_profile(monkeypatch, files["registry"])
    monkeypatch.setattr(
        "dean_os.domain_scoped_official_policy_envelope."
        "load_verified_official_policy_context_fragment",
        lambda path, expected_as_of=None: {
            "as_of": expected_as_of,
            "news": [{"title": "official policy", "published_at": AS_OF}],
            "metadata": {"saved_official_policy_verified": True},
        },
    )
    monkeypatch.setattr(
        "dean_os.domain_scoped_official_policy_envelope."
        "load_verified_domain_news_context_fragment",
        lambda path, expected_domain_id=None, expected_as_of=None: {
            "as_of": expected_as_of,
            "domain_id": expected_domain_id,
            "news": [{"title": "corroboration"}],
            "metadata": {
                "domain_news_envelope_sha256": _sha(Path(path).resolve())
            },
        },
    )
    payload = DomainScopedOfficialPolicyEnvelope().build(
        domain_id=DOMAIN,
        as_of=AS_OF,
        source_path=files["source"],
        news_envelope_path=files["news"],
        dispatch_path=files["dispatch"],
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )
    tampered = _write(tmp_path / "tampered-envelope.json", payload)
    changed = json.loads(tampered.read_text(encoding="utf-8"))
    changed["summary"]["directional_market_claim_created"] = True
    _write(tampered, changed)

    import pytest

    with pytest.raises(ValueError, match="forbidden flag invalid"):
        load_verified_domain_official_policy_context_fragment(tampered)
