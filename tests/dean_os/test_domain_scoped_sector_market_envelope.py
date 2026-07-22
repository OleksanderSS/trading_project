from __future__ import annotations

import json

from dean_os.context_acquisition_state_machine import ContextAcquisitionStateMachine
from dean_os.domain_scoped_sector_market_envelope import (
    DomainScopedSectorMarketEnvelope,
    load_verified_domain_sector_market_context_fragment,
)


AS_OF = "2026-06-30T21:00:00+00:00"
DOMAIN_ID = "semiconductor_ai_infrastructure"
UNIVERSE = [
    "NVDA",
    "AMD",
    "TSM",
    "ASML",
    "AMAT",
    "LRCX",
    "KLAC",
    "AVGO",
    "MU",
    "ARM",
    "INTC",
    "QCOM",
]


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source(path, *, benchmark="SOXX", universe=None):
    payload = {
        "run_id": "sector_source_1",
        "created_at": AS_OF,
        "producer_contract": "dean_saved_sector_market_evidence_producer_v1",
        "status": "sector_market_evidence_ready",
        "inputs": {
            "as_of": AS_OF,
            "sector_tickers": universe or UNIVERSE,
            "benchmark": benchmark,
        },
        "metrics": [{"name": "sector_median_excess_return", "value": 1.0}],
        "lineage": {"daily_artifact": {"sha256": "a" * 64}},
        "safety": {
            "review_only": True,
            "pipeline_run_performed": False,
            "training_run_performed": False,
            "tuning_run_performed": False,
            "learning_write_performed": False,
            "production_config_write_performed": False,
            "broker_access_performed": False,
            "live_execution_performed": False,
        },
    }
    _write(path, payload)
    return path


def _dispatch(path):
    _write(
        path,
        {
            "mode": "domain_binding_task_dispatch",
            "summary": {"domain_id": DOMAIN_ID},
            "task_dispatches": [
                {
                    "context_family": "sector_market",
                    "recommended_action": (
                        "domain_scoped_sector_market_evidence_producer"
                    ),
                }
            ],
        },
    )
    return path


def test_verified_sector_market_candidate_uses_shared_state_machine(
    tmp_path, monkeypatch
):
    source = _source(tmp_path / "source.json")
    dispatch = _dispatch(tmp_path / "dispatch.json")

    monkeypatch.setattr(
        "dean_os.domain_scoped_sector_market_envelope."
        "load_verified_sector_market_context_fragment",
        lambda path, expected_as_of=None: {
            "as_of": expected_as_of,
            "sector_data": {"relative_strength": {"value": 1.0}},
            "metadata": {"saved_sector_market_verified": True},
        },
    )
    payload = DomainScopedSectorMarketEnvelope(tmp_path / "reports").build(
        domain_id=DOMAIN_ID,
        as_of=AS_OF,
        source_path=source,
        dispatch_path=dispatch,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert payload["summary"]["status"] == "domain_sector_market_candidate_ready"
    assert payload["summary"]["source_lineage_verified"] is True
    assert payload["summary"]["producer_run_performed"] is False
    assert payload["summary"]["binding_accepted"] is False
    assert payload["market_context_fragment"]["domain_id"] == DOMAIN_ID

    envelope = tmp_path / "envelope.json"
    _write(envelope, payload)
    verified = load_verified_domain_sector_market_context_fragment(
        envelope,
        expected_domain_id=DOMAIN_ID,
        expected_as_of=AS_OF,
    )
    assert verified["metadata"]["domain_sector_market_envelope_verified"] is True
    machine = ContextAcquisitionStateMachine(
        ledger_path=tmp_path / "transition_ledger.jsonl",
        journal_path=tmp_path / "system_journal.jsonl",
        output_dir=tmp_path / "machine",
    )
    transition = machine.advance(
        acquisition_id="sector_market_semis_1",
        domain_id=DOMAIN_ID,
        context_family="sector_market",
        stage_id="candidate_verified",
        artifact_path=envelope,
        apply_transition=True,
        apply_journal=True,
        save=False,
    )

    assert transition["summary"]["status"] == "transition_recorded"
    assert transition["summary"]["persisted_state"] == "awaiting_binding_decision"
    assert transition["summary"]["binding_accepted"] is False
    assert transition["summary"]["can_trade"] is False


def test_sector_market_scope_mismatch_fails_before_source_binding(
    tmp_path, monkeypatch
):
    source = _source(tmp_path / "source.json", benchmark="QQQ", universe=["NVDA"])
    dispatch = _dispatch(tmp_path / "dispatch.json")
    called = False

    def _unexpected(*args, **kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        "dean_os.domain_scoped_sector_market_envelope."
        "load_verified_sector_market_context_fragment",
        _unexpected,
    )
    payload = DomainScopedSectorMarketEnvelope().build(
        domain_id=DOMAIN_ID,
        as_of=AS_OF,
        source_path=source,
        dispatch_path=dispatch,
        journal_path=tmp_path / "journal.jsonl",
        save=False,
    )

    assert called is False
    assert payload["summary"]["candidate_ready_for_binding_review"] is False
    assert "sector_market_universe_mismatch" in payload["summary"][
        "structural_blockers"
    ]
    assert "sector_market_benchmark_mismatch" in payload["summary"][
        "structural_blockers"
    ]
    assert payload["summary"]["can_invoke_domain_analysis"] is False
