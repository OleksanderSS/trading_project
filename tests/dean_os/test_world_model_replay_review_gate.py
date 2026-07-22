from __future__ import annotations

from dean_os.schemas import MarketContext
from dean_os.world_model_event_learning import WorldModelEventLearningPacket
from dean_os.world_model_replay_review_gate import (
    WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT,
    WorldModelReplayReviewGate,
)

AS_OF = "2026-07-01T12:00:00+00:00"
DOMAIN_ID = "semiconductor_ai_infrastructure"


def _semantic_news():
    return {
        "title": (
            "Nvidia AI demand growth confirms semiconductor memory shortage "
            "and data center capex pressure"
        ),
        "summary": (
            "AI demand growth is increasing HBM memory shortage risk, "
            "supporting data center capex but raising supply-chain constraints."
        ),
        "published_at": "2026-07-01T10:00:00+00:00",
        "url": "https://example.test/news/ai-memory-shortage",
        "tickers": ["NVDA"],
        "_dean_semantic_evidence": {
            "producer_contract": "test_news_contract",
            "evidence_type": "sector_demand",
            "matched_terms": ["ai demand", "data center demand"],
            "required_lane_eligible": True,
            "source_tier": "tier_2_strong_context",
            "source_identity": "reuters",
            "candidate_sha256": "abc123",
        },
    }


def _event_packet():
    context = MarketContext(
        as_of=AS_OF,
        tickers=["NVDA"],
        news=[_semantic_news()],
    )
    return WorldModelEventLearningPacket().build(
        context,
        domain_id=DOMAIN_ID,
        save=False,
    )


def test_world_model_replay_review_gate_requires_manual_review():
    packet = _event_packet()

    payload = WorldModelReplayReviewGate().build(packet, save=False)

    assert payload["contract"] == WORLD_MODEL_REPLAY_REVIEW_GATE_CONTRACT
    assert payload["summary"]["gate_status"] == (
        "manual_review_required_for_replay_registration"
    )
    assert payload["summary"]["can_register_replay_tasks"] is False
    assert payload["registration_bundle"] is None
    assert payload["summary"]["replay_task_registration_performed"] is False
    assert payload["safety"]["replay_task_registration_performed"] is False
    assert payload["safety"]["learning_memory_write_performed"] is False
    assert payload["summary"]["can_trade"] is False


def test_world_model_replay_review_gate_blocks_approval_without_reviewer():
    packet = _event_packet()

    payload = WorldModelReplayReviewGate().build(
        packet,
        approve=True,
        save=False,
    )

    assert payload["summary"]["gate_status"] == (
        "blocked_missing_reviewer_for_replay_approval"
    )
    assert payload["summary"]["can_register_replay_tasks"] is False
    assert "reviewer_required_for_replay_registration_approval" in payload[
        "summary"
    ]["issues"]
    assert payload["registration_bundle"] is None


def test_world_model_replay_review_gate_creates_approved_registration_bundle():
    packet = _event_packet()

    payload = WorldModelReplayReviewGate().build(
        packet,
        approve=True,
        reviewer="operator",
        review_notes="Hypotheses look coherent for replay registration.",
        save=False,
    )

    assert payload["summary"]["gate_status"] == (
        "replay_tasks_approved_for_registration"
    )
    assert payload["summary"]["can_register_replay_tasks"] is True
    assert payload["summary"]["registration_bundle_created"] is True
    assert payload["summary"]["replay_task_registration_performed"] is False
    assert payload["safety"]["learning_memory_write_performed"] is False
    assert payload["safety"]["outcome_registration_performed"] is False
    assert payload["safety"]["replay_task_registration_performed"] is False

    bundle = payload["registration_bundle"]
    assert bundle["approved_by"] == "operator"
    assert bundle["task_count"] == len(packet["replay_tasks"])
    assert bundle["tasks"][0]["manual_review_gate_required"] is True
    assert "trade_signal" in bundle["forbidden_next_steps"]


def test_cycle_bound_gate_blocks_approval_with_unaligned_upstream_hypothesis():
    packet = _event_packet()
    packet["cycle_binding_contract"] = "dean_full_system_cycle_world_model_bridge_v1"
    packet["hypothesis_alignment_review"] = {
        "contract": "dean_cycle_hypothesis_alignment_review_v1",
        "summary": {
            "unaligned_upstream_hypothesis_count": 1,
            "horizon_substitution_allowed": False,
        },
    }

    payload = WorldModelReplayReviewGate().build(
        packet,
        approve=True,
        reviewer="operator",
        save=False,
    )

    assert payload["summary"]["can_register_replay_tasks"] is False
    assert "cycle_bound_upstream_hypothesis_alignment_incomplete" in payload[
        "summary"
    ]["issues"]


def test_cycle_bound_gate_requires_explicit_hypothesis_disposition():
    packet = _event_packet()
    hypothesis_id = packet["hypotheses"][0]["hypothesis_id"]
    packet["cycle_binding_contract"] = "dean_full_system_cycle_world_model_bridge_v1"
    packet["hypothesis_alignment_review"] = {
        "contract": "dean_cycle_hypothesis_alignment_review_v1",
        "summary": {
            "status": "all_upstream_mechanisms_mapped_pending_manual_review",
            "unaligned_upstream_hypothesis_count": 0,
            "horizon_substitution_allowed": False,
        },
        "alignments": [
            {
                "upstream_hypothesis_id": "upstream_1",
                "upstream_horizons_days": [30, 90, 180],
                "world_hypothesis_ids": [hypothesis_id],
            }
        ],
    }

    blocked = WorldModelReplayReviewGate().build(
        packet,
        approve=True,
        reviewer="operator",
        review_notes="Reviewed source and horizon roles.",
        save=False,
    )
    assert blocked["summary"]["can_register_replay_tasks"] is False
    assert any(
        issue.startswith("cycle_bound_hypothesis_dispositions_missing:")
        for issue in blocked["summary"]["issues"]
    )

    approved = WorldModelReplayReviewGate().build(
        packet,
        approve=True,
        reviewer="operator",
        review_notes="Reviewed source and horizon roles.",
        hypothesis_dispositions={hypothesis_id: "accept_for_replay"},
        save=False,
    )
    assert approved["summary"]["can_register_replay_tasks"] is True
    assert approved["summary"]["approved_replay_task_count"] == len(
        packet["replay_tasks"]
    )
    assert approved["hypothesis_review"][0]["disposition"] == (
        "accept_for_replay"
    )
    assert approved["hypothesis_review"][0]["trigger_event"]["source_id"]


def test_cycle_bound_gate_rejects_packet_anchored_due_dates():
    packet = _event_packet()
    hypothesis_id = packet["hypotheses"][0]["hypothesis_id"]
    packet["cycle_binding_contract"] = "dean_full_system_cycle_world_model_bridge_v1"
    packet["hypothesis_alignment_review"] = {
        "contract": "dean_cycle_hypothesis_alignment_review_v1",
        "summary": {
            "status": "all_upstream_mechanisms_mapped_pending_manual_review",
            "unaligned_upstream_hypothesis_count": 0,
            "horizon_substitution_allowed": False,
        },
        "alignments": [
            {
                "upstream_hypothesis_id": "upstream_1",
                "upstream_horizons_days": [30, 90, 180],
                "world_hypothesis_ids": [hypothesis_id],
            }
        ],
    }
    packet["replay_tasks"][0]["as_of"] = packet["summary"]["as_of"]

    payload = WorldModelReplayReviewGate().build(packet, save=False)

    assert payload["summary"]["can_register_replay_tasks"] is False
    assert any(
        issue.startswith("cycle_bound_task_as_of_not_event_anchor:")
        for issue in payload["summary"]["issues"]
    )


def test_cycle_bound_gate_records_structured_reformulation_without_approval():
    packet = _event_packet()
    hypothesis_id = packet["hypotheses"][0]["hypothesis_id"]
    packet["cycle_binding_contract"] = "dean_full_system_cycle_world_model_bridge_v1"
    packet["hypothesis_alignment_review"] = {
        "contract": "dean_cycle_hypothesis_alignment_review_v1",
        "summary": {
            "status": "all_upstream_mechanisms_mapped_pending_manual_review",
            "unaligned_upstream_hypothesis_count": 0,
            "horizon_substitution_allowed": False,
        },
        "alignments": [
            {
                "upstream_hypothesis_id": "upstream_1",
                "upstream_horizons_days": [30, 90, 180],
                "world_hypothesis_ids": [hypothesis_id],
            }
        ],
    }

    payload = WorldModelReplayReviewGate().build(
        packet,
        reviewer="content_reviewer",
        review_notes="Trigger supports a narrower claim only.",
        hypothesis_dispositions={
            hypothesis_id: {
                "disposition": "reformulate",
                "rationale": "The trigger is narrower than the generated claim.",
                "proposed_hypothesis": "A narrower event response will be observed.",
                "source_assessment": "coherent_trigger_but_claim_scope_too_broad",
            }
        },
        save=False,
    )

    assert payload["summary"]["gate_status"] == (
        "hypothesis_review_complete_reformulation_required"
    )
    assert payload["summary"]["manual_hypothesis_review_complete"] is True
    assert payload["summary"]["can_register_replay_tasks"] is False
    assert payload["hypothesis_review"][0]["proposed_hypothesis"] == (
        "A narrower event response will be observed."
    )


def test_cycle_bound_gate_blocks_accepted_claim_below_quality_floor():
    packet = _event_packet()
    hypothesis = packet["hypotheses"][0]
    hypothesis_id = hypothesis["hypothesis_id"]
    hypothesis["expected_observations"] = []
    hypothesis["invalidation_signals"] = []
    packet["cycle_binding_contract"] = "dean_full_system_cycle_world_model_bridge_v1"
    packet["hypothesis_alignment_review"] = {
        "contract": "dean_cycle_hypothesis_alignment_review_v1",
        "summary": {
            "status": "all_upstream_mechanisms_mapped_pending_manual_review",
            "unaligned_upstream_hypothesis_count": 0,
            "horizon_substitution_allowed": False,
        },
        "alignments": [
            {
                "upstream_hypothesis_id": "upstream_1",
                "upstream_horizons_days": [30, 90, 180],
                "world_hypothesis_ids": [hypothesis_id],
            }
        ],
    }

    payload = WorldModelReplayReviewGate().build(
        packet,
        approve=True,
        reviewer="operator",
        review_notes="Content and replay registration reviewed.",
        hypothesis_dispositions={hypothesis_id: "accept_for_replay"},
        save=False,
    )

    assert payload["summary"]["can_register_replay_tasks"] is False
    assert any(
        issue.startswith("cycle_bound_accepted_hypotheses_fail_quality_floor:")
        for issue in payload["summary"]["issues"]
    )
    quality = payload["hypothesis_review"][0]["quality_assessment"]
    assert quality["replay_eligible"] is False
    assert quality["confidence_probability"] is None
