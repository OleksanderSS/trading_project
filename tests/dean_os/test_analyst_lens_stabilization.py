from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from dean_os.analyst_core.artifact_evidence_loader import ArtifactEvidenceLoader
from dean_os.analyst_core.lens_contract import AnalysisPacket
from dean_os.analyst_core.lenses.event_classifier_lens import EventClassifierLens
from dean_os.analyst_core.lenses.hypothesis_ledger_lens import HypothesisLedgerLens
from dean_os.analyst_core.schemas import HypothesisLedgerEntry, HypothesisStatus
from dean_os.analyst_core.sector_analyst import SectorAnalyst
from dean_os.analysts.profiles import get_domain_profile
from dean_os.analysts.schemas import AnalystEvidenceItem
from dean_os.utils import sha256_json


AS_OF = "2026-07-01T00:00:00+00:00"


@pytest.mark.parametrize(
    "text",
    [
        "Warehouses are full after demand slowed.",
        "The supplier was awarded a new contract.",
        "A ransomware incident affected one website.",
        "Management warns about ordinary execution risk.",
    ],
)
def test_war_keyword_does_not_match_inside_other_words(text: str) -> None:
    packet = AnalysisPacket(
        packet_id="packet_word_boundary",
        as_of_date=AS_OF,
        event_records=[{"event_id": "ev_1", "title": text}],
    )

    delta = EventClassifierLens().analyze(packet)

    assert delta.classified_events_added[0]["event_class"] != "war_escalation"
    assert delta.as_of == AS_OF


def test_hypothesis_lens_proposes_review_without_mutating_status() -> None:
    hypothesis = HypothesisLedgerEntry(
        hypothesis_id="hypothesis_existing",
        as_of=AS_OF,
        hypothesis="Supply constraints will persist",
        invalidation_signals=["capacity expansion completed"],
        expected_observations=["lead times remain elevated"],
        status=HypothesisStatus.OPEN,
    )
    packet = AnalysisPacket(
        packet_id="packet_hypothesis_review",
        as_of_date=AS_OF,
        classified_events=[
            {
                "event_id": "ev_capacity",
                "evidence_id": "ev_capacity",
                "event_class": "supply_disruption",
                "text_preview": "Capacity expansion completed ahead of schedule.",
            }
        ],
        hypotheses=[hypothesis],
    )

    delta = HypothesisLedgerLens().analyze(packet)

    assert packet.hypotheses[0].status == HypothesisStatus.OPEN
    assert len(delta.hypothesis_review_proposals_added) == 1
    proposal = delta.hypothesis_review_proposals_added[0]
    assert proposal["proposal_type"] == "candidate_contradiction"
    assert proposal["status_changed"] is False
    assert proposal["requires_manual_review"] is True
    assert proposal["evidence_ids"] == ["ev_capacity"]


def test_cross_domain_bus_rejects_future_signal() -> None:
    payload = {
        "contract": "dean_cross_domain_signal_v1",
        "source_domain": "energy",
        "event_class": "oil_shock",
        "event_id": "event_oil",
        "source_evidence_id": "evidence_oil",
        "source_evidence_sha256": "a" * 64,
        "title": "Verified oil shock",
        "text": "Verified source context",
        "materiality": 0.8,
        "as_of": "2026-07-02T00:00:00+00:00",
        "available_at": "2026-07-02T00:00:00+00:00",
        "source_reliability": 0.8,
        "propagation_rules": {
            "target_domains": ["semiconductor_ai_infrastructure"],
            "evidence_type": "macro_context",
            "stance_hint": "negative",
            "strength_multiplier": 0.7,
        },
    }
    signal_hash = sha256_json(payload)
    payload["signal_sha256"] = signal_hash
    with TemporaryDirectory(
        prefix="dean_os_signal_bus_test_", dir=Path.cwd()
    ) as temporary_dir:
        bus_dir = Path(temporary_dir)
        (bus_dir / f"signal_{signal_hash}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="future evidence"):
            ArtifactEvidenceLoader().from_signal_bus(
                "semiconductor_ai_infrastructure",
                AS_OF,
                bus_dir=bus_dir,
            )


def test_profile_trusted_sources_are_retained() -> None:
    profile = get_domain_profile("energy")

    assert profile.trusted_sources["tier_1_apis"]
    assert profile.trusted_sources["tier_2_reports"]


def test_sector_report_is_deterministic_and_serializes_audit_trail() -> None:
    evidence = AnalystEvidenceItem(
        evidence_id="evidence_demand",
        source_type="news",
        source="verified_fixture",
        published_at="2026-06-30T00:00:00+00:00",
        as_of=AS_OF,
        domain_id="semiconductor_ai_infrastructure",
        tickers=[],
        sectors=["semiconductor_ai_infrastructure"],
        evidence_type="sector_demand",
        summary="AI infrastructure demand accelerates",
        stance_hint="positive",
        strength=0.7,
        freshness_score=0.9,
        directness="sector",
        reliability_score=0.8,
        provenance={"source_sha256": "b" * 64},
        point_in_time={
            "status": "point_in_time_compatible",
            "available_at": "2026-06-30T00:00:00+00:00",
        },
    )
    analyst = SectorAnalyst("semiconductor_ai_infrastructure")

    first = analyst.run_from_evidence([evidence], as_of=AS_OF).to_dict()
    second = analyst.run_from_evidence([evidence], as_of=AS_OF).to_dict()

    assert first == second
    assert first["audit"]["analysis_input_sha256"]
    assert first["audit"]["analysis_output_sha256"]
    assert first["audit"]["source_evidence_ids"] == ["evidence_demand"]
    assert first["audit"]["delta_trail"]
    assert all(
        delta["delta_sha256"] for delta in first["audit"]["delta_trail"]
    )
