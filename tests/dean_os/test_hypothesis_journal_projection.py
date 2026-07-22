from __future__ import annotations

from dean_os.analyst_core.sector_analyst import SectorAnalyst
from dean_os.hypothesis_journal_projection import project_active_hypotheses
from dean_os.system_journal import SystemJournal


DOMAIN = "semiconductor_ai_infrastructure"
AS_OF = "2026-07-10T12:00:00+00:00"


def _created(hypothesis_id, *, effective_at="2026-07-01T12:00:00+00:00"):
    return {
        "event_type": "hypothesis_created",
        "effective_at": effective_at,
        "actor": "test",
        "domain_id": DOMAIN,
        "entity_type": "hypothesis",
        "entity_id": hypothesis_id,
        "payload": {
            "hypothesis_id": hypothesis_id,
            "as_of": effective_at,
            "hypothesis": "Equipment orders will strengthen",
            "confidence": 0.5,
            "trigger_evidence_ids": ["trigger_1"],
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
            "expected_observations": ["rising equipment orders"],
            "invalidation_signals": ["equipment orders fall"],
            "horizons_to_check": [30, 90, 180],
            "status": "open",
        },
    }


def test_projection_reuses_active_journal_hypothesis_in_sector_analysis(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    events = [
        _created("active_hypothesis"),
        _created("rejected_hypothesis"),
        {
            "event_type": "hypothesis_reviewed",
            "effective_at": "2026-07-02T12:00:00+00:00",
            "actor": "reviewer",
            "domain_id": DOMAIN,
            "entity_type": "hypothesis",
            "entity_id": "rejected_hypothesis",
            "payload": {"disposition": "reject"},
        },
        _created(
            "future_hypothesis",
            effective_at="2026-07-11T12:00:00+00:00",
        ),
    ]
    SystemJournal(journal_path).append_many(events)

    projection = project_active_hypotheses(
        journal_path,
        domain_id=DOMAIN,
        as_of=AS_OF,
    )

    assert projection["active_hypothesis_count"] == 1
    assert projection["active_hypotheses"][0].hypothesis_id == "active_hypothesis"
    assert any(
        item["hypothesis_id"] == "rejected_hypothesis"
        for item in projection["exclusions"]
    )

    report = SectorAnalyst(domain_id=DOMAIN).run_from_evidence(
        [
            {
                "evidence_id": "evidence_orders_1",
                "source_type": "official",
                "source": "company filing",
                "published_at": "2026-07-09T12:00:00+00:00",
                "as_of": AS_OF,
                "domain_id": DOMAIN,
                "tickers": [],
                "sectors": [DOMAIN],
                "evidence_type": "capex_cycle",
                "summary": "Rising equipment orders were reported.",
                "stance_hint": "positive",
                "strength": 0.8,
                "freshness_score": 0.9,
                "directness": "sector",
                "reliability_score": 0.9,
            }
        ],
        as_of=AS_OF,
        prior_hypotheses=projection["active_hypotheses"],
    )

    proposals = report.hypothesis_review_proposals
    assert len(proposals) == 1
    assert proposals[0]["hypothesis_id"] == "active_hypothesis"
    assert proposals[0]["proposal_type"] == "candidate_support"
    assert proposals[0]["requires_manual_review"] is True
    assert proposals[0]["status_changed"] is False
