from __future__ import annotations

import json
from pathlib import Path

from dean_os.context_acquisition_state_machine import ContextAcquisitionStateMachine
from dean_os.domain_scoped_news_envelope import (
    CONTRACT,
    DomainScopedNewsEnvelope,
    load_verified_domain_news_context_fragment,
)


AS_OF = "2026-06-30T21:00:00+00:00"
DOMAIN = "semiconductor_ai_infrastructure"
REGISTRY = Path("dean_os/config/semiconductor_news_source_registry.yaml").resolve()
LANES = [
    "sector_demand",
    "capex_cycle",
    "supply_chain",
    "policy_or_geopolitical",
    "market_confirmation",
]


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _source(tmp_path: Path, *, domain_id: str = DOMAIN) -> Path:
    return _write(
        tmp_path / "news.json",
        {
            "run_id": "news_fixture",
            "mode": "saved_semiconductor_news_evidence_producer",
            "producer_contract": "dean_saved_semiconductor_news_evidence_producer_v1",
            "status": "semiconductor_news_evidence_ready_with_gaps",
            "inputs": {
                "as_of": AS_OF,
                "domain_id": domain_id,
                "registry_path": str(REGISTRY),
            },
            "registry": {"path": str(REGISTRY), "sha256": "a" * 64},
            "source_provenance": {"path": "saved.parquet", "sha256": "b" * 64},
            "summary": {
                "accepted_news_record_count": 1,
                "ready_required_lanes": [
                    "sector_demand",
                    "capex_cycle",
                    "supply_chain",
                    "market_confirmation",
                ],
                "missing_required_lanes": ["policy_or_geopolitical"],
            },
            "integration_boundary": {
                "keyword_hit_is_lane_completion": False,
                "independent_strong_sources_required": True,
                "plain_text_ticker_promotion_allowed": False,
                "pipeline_feature_promotion_allowed": False,
                "training_allowed": False,
                "automatic_trading_allowed": False,
            },
            "safety": {
                "review_only": True,
                "network_access_performed": False,
                "collector_run_performed": False,
                "pipeline_run_performed": False,
                "training_run_performed": False,
                "learning_write_performed": False,
                "production_config_write_performed": False,
                "broker_access_performed": False,
                "live_execution_performed": False,
            },
        },
    )


def _dispatch(tmp_path: Path, *, domain_id: str = DOMAIN) -> Path:
    return _write(
        tmp_path / "dispatch.json",
        {
            "mode": "domain_binding_task_dispatch",
            "summary": {"domain_id": domain_id},
            "task_dispatches": [
                {
                    "context_family": "news",
                    "recommended_action": "domain_scoped_news_envelope",
                }
            ],
        },
    )


def test_verified_news_is_trigger_only_and_uses_shared_state_machine(
    tmp_path: Path, monkeypatch
) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(
        "dean_os.domain_scoped_news_envelope."
        "load_verified_semiconductor_news_context_fragment",
        lambda path, expected_as_of=None: {
            "as_of": expected_as_of,
            "news": [{"title": "event", "published_at": AS_OF}],
            "metadata": {"saved_semiconductor_news_verified": True},
        },
    )
    payload = DomainScopedNewsEnvelope(tmp_path / "reports").build(
        domain_id=DOMAIN,
        as_of=AS_OF,
        source_path=source,
        dispatch_path=_dispatch(tmp_path),
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["contract"] == CONTRACT
    assert payload["summary"]["status"] == "domain_news_candidate_ready_with_gaps"
    assert payload["summary"]["source_lineage_verified"] is True
    assert payload["summary"]["trigger_semantics_preserved"] is True
    assert payload["summary"]["hypothesis_confirmed"] is False
    assert payload["summary"]["llm_call_performed"] is False
    assert payload["domain_binding"]["context_role"] == "trigger_evidence_only"
    assert "required_news_lane_missing:policy_or_geopolitical" in payload[
        "quality"
    ]["quality_gaps"]

    verified_envelope = _write(tmp_path / "verified-envelope.json", payload)
    verified = load_verified_domain_news_context_fragment(
        verified_envelope,
        expected_domain_id=DOMAIN,
        expected_as_of=AS_OF,
    )
    assert verified["domain_id"] == DOMAIN
    assert verified["metadata"]["domain_news_envelope_verified"] is True

    envelope = _write(tmp_path / "envelope.json", payload)
    machine = ContextAcquisitionStateMachine(
        ledger_path=tmp_path / "transition_ledger.jsonl",
        journal_path=tmp_path / "state_journal.jsonl",
        output_dir=tmp_path / "machine",
    )
    transition = machine.advance(
        acquisition_id="news_semis_1",
        domain_id=DOMAIN,
        context_family="news",
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
    assert transition["summary"]["can_trade"] is False


def test_cross_domain_relabel_is_blocked_before_recursive_loading(
    tmp_path: Path, monkeypatch
) -> None:
    called = False

    def _unexpected(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        "dean_os.domain_scoped_news_envelope."
        "load_verified_semiconductor_news_context_fragment",
        _unexpected,
    )
    payload = DomainScopedNewsEnvelope().build(
        domain_id=DOMAIN,
        as_of=AS_OF,
        source_path=_source(tmp_path, domain_id="energy"),
        dispatch_path=_dispatch(tmp_path),
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert called is False
    assert "news_source_domain_mismatch" in payload["summary"][
        "structural_blockers"
    ]
    assert payload["summary"]["candidate_ready_for_binding_review"] is False
    assert payload["summary"]["can_approve_hypothesis"] is False


def test_domain_news_loader_rejects_semantic_boundary_tampering(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "dean_os.domain_scoped_news_envelope."
        "load_verified_semiconductor_news_context_fragment",
        lambda path, expected_as_of=None: {
            "as_of": expected_as_of,
            "news": [{"title": "event", "published_at": AS_OF}],
            "metadata": {},
        },
    )
    payload = DomainScopedNewsEnvelope().build(
        domain_id=DOMAIN,
        as_of=AS_OF,
        source_path=_source(tmp_path),
        dispatch_path=_dispatch(tmp_path),
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )
    payload["domain_binding"]["news_is_hypothesis_confirmation"] = True
    tampered = _write(tmp_path / "tampered-envelope.json", payload)

    try:
        load_verified_domain_news_context_fragment(tampered)
    except ValueError as exc:
        assert "semantic boundary" in str(exc)
    else:
        raise AssertionError("tampered news semantics were accepted")
