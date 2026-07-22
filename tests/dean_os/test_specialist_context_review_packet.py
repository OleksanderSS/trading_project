from __future__ import annotations

import json

from dean_os.specialist_context_review_packet import (
    SpecialistContextReviewPacket,
)


def _write_sources(tmp_path, supporting_as_of=None):
    supporting_as_of = supporting_as_of or [
        "2026-03-25T00:00:00+00:00",
        "2026-04-01T00:00:00+00:00",
    ]
    review_path = tmp_path / "sector_to_ticker.json"
    review_path.write_text(
        json.dumps({
            "run_id": "sector_review:test",
            "mode": "sector_to_ticker_review_packet",
            "summary": {
                "manual_review_required": True,
                "domain_profile": (
                    "semiconductor_ai_infrastructure"
                ),
                "sector": "semiconductor",
                "sector_stance": "evidence_limited",
            },
            "ticker_review_map": [
                {
                    "ticker": "AMD",
                    "candidate_status": (
                        "direct_ticker_thesis_ready"
                    ),
                    "review_status": "review_ready",
                    "thesis_level": (
                        "direct_ticker_thesis_candidate_for_"
                        "manual_review"
                    ),
                    "allowed_use": (
                        "manual_review_of_ticker_candidate_only"
                    ),
                    "direct_evidence": {
                        "overlay_ready_runs": 2,
                        "directional_ready_runs": 1,
                        "neutral_ready_runs": 1,
                        "supporting_as_of": supporting_as_of,
                    },
                    "blocked_evidence": {
                        "blocked_runs": 0,
                        "blocked_as_of": [],
                    },
                    "risk_and_counter_thesis_flags": [
                        "mixed_windows"
                    ],
                }
            ],
        }),
        encoding="utf-8",
    )
    thesis_path = tmp_path / "domain_thesis.json"
    thesis_path.write_text(
        json.dumps({
            "run_id": "domain_thesis:test",
            "mode": "domain_analyst_thesis_review_packet",
            "summary": {
                "domain_id": (
                    "semiconductor_ai_infrastructure"
                ),
                "ticker_direct_count": 0,
            },
        }),
        encoding="utf-8",
    )
    return review_path, thesis_path


def test_specialist_context_separates_domain_and_amd_candidate(
    tmp_path,
):
    review_path, thesis_path = _write_sources(tmp_path)

    payload = SpecialistContextReviewPacket().build(
        sector_to_ticker_review_path=review_path,
        domain_thesis_path=thesis_path,
        ticker="AMD",
        timeframe="15m",
        context_as_of="2026-06-29T12:00:00+00:00",
        max_evidence_age_days=30,
        save=False,
    )

    assert payload["domain_scope"]["sector"] == "semiconductor"
    assert payload["domain_scope"]["domain_ticker_direct_count"] == 0
    assert payload["domain_scope"][
        "domain_thesis_is_ticker_evidence"
    ] is False
    assert payload["ticker_scope"]["ticker"] == "AMD"
    assert payload["ticker_scope"]["evidence_scope"] == (
        "direct_ticker_review_candidate"
    )
    assert payload["ticker_scope"][
        "eligible_as_direct_ticker_review_context"
    ] is True
    assert payload["ticker_scope"][
        "eligible_as_approved_ticker_thesis"
    ] is False
    assert payload["point_in_time"]["status"] == (
        "older_than_review_window"
    )
    assert payload["timeframe_alignment"]["status"] == (
        "unverified_source_timeframe_not_declared"
    )
    assert payload["safety"][
        "eligible_for_exact_pipeline_context"
    ] is False
    assert payload["safety"]["sector_context_promoted_to_ticker"] is False
    assert payload["safety"]["decision_influence"] is False
    assert payload["safety"]["can_trade"] is False


def test_specialist_context_missing_ticker_stays_sector_only(tmp_path):
    review_path, thesis_path = _write_sources(tmp_path)

    payload = SpecialistContextReviewPacket().build(
        sector_to_ticker_review_path=review_path,
        domain_thesis_path=thesis_path,
        ticker="NVDA",
        timeframe="15m",
        context_as_of="2026-04-02T00:00:00+00:00",
        save=False,
    )

    assert payload["ticker_scope"]["evidence_scope"] == (
        "sector_context_only"
    )
    assert payload["ticker_scope"]["candidate_found"] is False
    assert {
        item["code"] for item in payload["review_issues"]
    } >= {
        "direct_ticker_review_missing",
        "direct_ticker_as_of_missing",
        "specialist_timeframe_unaligned",
    }
    assert payload["safety"][
        "eligible_for_exact_pipeline_context"
    ] is False


def test_specialist_context_rejects_future_evidence(tmp_path):
    review_path, thesis_path = _write_sources(
        tmp_path,
        supporting_as_of=["2026-07-01T00:00:00+00:00"],
    )

    payload = SpecialistContextReviewPacket().build(
        sector_to_ticker_review_path=review_path,
        domain_thesis_path=thesis_path,
        ticker="AMD",
        timeframe="1d",
        context_as_of="2026-06-29T00:00:00+00:00",
        save=False,
    )

    assert payload["point_in_time"]["status"] == (
        "future_evidence_conflict"
    )
    assert "future_specialist_evidence" in {
        item["code"] for item in payload["review_issues"]
    }


def test_specialist_context_preserves_source_hashes(tmp_path):
    review_path, thesis_path = _write_sources(tmp_path)

    payload = SpecialistContextReviewPacket().build(
        sector_to_ticker_review_path=review_path,
        domain_thesis_path=thesis_path,
        ticker="AMD",
        timeframe=None,
        context_as_of="2026-04-02T00:00:00+00:00",
        save=False,
    )

    provenance = payload["source_provenance"]
    assert len(provenance["sector_to_ticker_review_sha256"]) == 64
    assert len(provenance["domain_thesis_sha256"]) == 64
    assert len(payload["packet_fingerprint"]) == 64
