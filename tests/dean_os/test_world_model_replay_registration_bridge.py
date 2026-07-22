from __future__ import annotations

import json

from dean_os.outcome_tracker import OutcomeTracker
from dean_os.schemas import MarketContext
from dean_os.world_model_event_learning import WorldModelEventLearningPacket
from dean_os.world_model_replay_registration import (
    WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT,
    WorldModelReplayRegistrationBridge,
)
from dean_os.world_model_replay_review_gate import WorldModelReplayReviewGate

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


def _approved_gate(packet: dict):
    return WorldModelReplayReviewGate().build(
        packet,
        approve=True,
        reviewer="operator",
        review_notes="Approved for tracker registration test.",
        save=False,
    )


def test_world_model_replay_registration_bridge_dry_run_plan_only():
    packet = _event_packet()
    gate = _approved_gate(packet)

    payload = WorldModelReplayRegistrationBridge().build(
        gate,
        source_packet_json=packet,
        save=False,
    )

    assert payload["contract"] == WORLD_MODEL_REPLAY_REGISTRATION_CONTRACT
    assert payload["summary"]["bridge_status"] == (
        "dry_run_ready_for_outcome_tracker_registration"
    )
    assert payload["summary"]["dry_run"] is True
    assert payload["summary"]["registered_count"] == 0
    assert payload["summary"]["planned_registration_count"] == len(packet["replay_tasks"])
    assert payload["summary"]["outcome_tracker_registration_performed"] is False
    assert payload["safety"]["outcome_tracker_write_performed"] is False
    assert payload["safety"]["learning_memory_write_performed"] is False
    assert payload["summary"]["can_trade"] is False

    first_plan = payload["registration_plan"][0]
    assert first_plan["event_type"] == "world_model_replay_task"
    assert first_plan["predicted_direction"] == "neutral"
    assert first_plan["predicted_direction_source"] == (
        "neutral_projection_no_explicit_direction"
    )
    assert first_plan["tracker_intervals"] == [first_plan["horizon_days"]]
    assert first_plan["event_anchor_at"] == packet["replay_tasks"][0][
        "trigger_event_at"
    ]
    assert first_plan["source"].startswith("world_model_replay|bundle=")
    assert DOMAIN_ID in first_plan["sectors"]


def test_registration_bridge_accepts_reformulated_task_after_new_manual_review():
    packet = _event_packet()
    gate = _approved_gate(packet)
    for task in gate["registration_bundle"]["tasks"]:
        task["registration_status"] = "candidate_pending_new_manual_review"

    payload = WorldModelReplayRegistrationBridge().build(
        gate,
        source_packet_json=packet,
        save=False,
    )

    assert payload["summary"]["issue_count"] == 0
    assert payload["summary"]["bridge_status"] == (
        "dry_run_ready_for_outcome_tracker_registration"
    )


def test_world_model_replay_registration_bridge_blocks_unapproved_gate():
    packet = _event_packet()
    gate = WorldModelReplayReviewGate().build(packet, save=False)

    payload = WorldModelReplayRegistrationBridge().build(gate, save=False)

    assert payload["summary"]["bridge_status"] == (
        "blocked_world_model_replay_registration_bridge"
    )
    assert "source_gate_not_approved_for_registration" in payload["summary"]["issues"]
    assert payload["summary"]["registered_count"] == 0
    assert payload["safety"]["outcome_tracker_write_performed"] is False


def test_world_model_replay_registration_bridge_apply_to_temp_tracker_and_dedupes(tmp_path):
    packet = _event_packet()
    gate = _approved_gate(packet)
    db_path = tmp_path / "outcome_tracker.sqlite"

    payload = WorldModelReplayRegistrationBridge().build(
        gate,
        source_packet_json=packet,
        tracker_db_path=db_path,
        apply=True,
        save=False,
    )

    assert payload["summary"]["bridge_status"] == "outcome_tracker_registration_applied"
    assert payload["summary"]["registered_count"] == len(packet["replay_tasks"])
    assert payload["summary"]["skipped_existing_count"] == 0
    assert payload["summary"]["outcome_tracker_registration_performed"] is True
    assert payload["summary"]["outcome_scoring_performed"] is False
    assert payload["safety"]["learning_memory_write_performed"] is False

    stats = OutcomeTracker(db_path).stats()
    assert stats["events"] == len(packet["replay_tasks"])
    assert stats["predictions"] == len(packet["replay_tasks"])
    assert stats["outcomes"] == 0

    repeated = WorldModelReplayRegistrationBridge().build(
        gate,
        source_packet_json=packet,
        tracker_db_path=db_path,
        apply=True,
        save=False,
    )

    assert repeated["summary"]["bridge_status"] == (
        "outcome_tracker_registration_already_applied"
    )
    assert repeated["summary"]["registered_count"] == 0
    assert repeated["summary"]["skipped_existing_count"] == len(packet["replay_tasks"])
    assert OutcomeTracker(db_path).stats()["events"] == len(packet["replay_tasks"])


def test_registration_bridge_rejects_source_packet_changed_after_review(tmp_path):
    packet = _event_packet()
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    gate = WorldModelReplayReviewGate().build(
        packet_path,
        approve=True,
        reviewer="operator",
        save=False,
    )
    packet["tampered_after_review"] = True
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    payload = WorldModelReplayRegistrationBridge().build(
        gate,
        source_packet_json=packet_path,
        save=False,
    )

    assert payload["summary"]["bridge_status"] == (
        "blocked_world_model_replay_registration_bridge"
    )
    assert "source_packet_sha256_mismatch" in payload["summary"]["issues"]


def test_registration_bridge_defers_matured_checkpoint_from_live_tracker(tmp_path):
    packet = _event_packet()
    packet["replay_tasks"][0]["checkpoint_state_at_packet"] = "matured"
    gate = _approved_gate(packet)
    db_path = tmp_path / "outcome_tracker.sqlite"

    payload = WorldModelReplayRegistrationBridge().build(
        gate,
        source_packet_json=packet,
        tracker_db_path=db_path,
        apply=True,
        save=False,
    )

    assert payload["summary"]["bridge_status"] == (
        "outcome_tracker_registration_partially_applied_historical_review_required"
    )
    assert payload["summary"]["deferred_historical_count"] == 1
    assert payload["summary"]["registered_count"] == len(packet["replay_tasks"]) - 1
    assert OutcomeTracker(db_path).stats()["predictions"] == len(
        packet["replay_tasks"]
    ) - 1

    repeated = WorldModelReplayRegistrationBridge().build(
        gate,
        source_packet_json=packet,
        tracker_db_path=db_path,
        apply=True,
        save=False,
    )
    assert repeated["summary"]["bridge_status"] == (
        "outcome_tracker_registration_already_applied_historical_review_required"
    )
    assert repeated["summary"]["registered_or_existing_count"] == (
        len(packet["replay_tasks"]) - 1
    )
