from __future__ import annotations

import json

from dean_os.chief_review_index import ChiefReviewIndexBuilder


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _review_index(path):
    _write(
        path,
        {
            "run_id": "review_1",
            "summary": {
                "available_count": 1,
                "missing_count": 0,
                "ready_for_chief_review": True,
            },
            "entries": [
                {
                    "source_name": "other_review",
                    "available": True,
                    "status": "ready",
                    "recommendation": "ready_for_review",
                    "safety": {"can_trade": False},
                }
            ],
        },
    )


def test_chief_review_surfaces_contract_and_pending_decision(tmp_path):
    review = tmp_path / "review.json"
    lifecycle = tmp_path / "lifecycle.json"
    _review_index(review)
    _write(
        lifecycle,
        {
            "run_id": "life_1",
            "review_inbox": {
                "status": "pending_hypothesis_decisions",
                "blockers": [],
                "proposed_contracts": [
                    {
                        "hypothesis_id": "h1",
                        "expected_direction": "negative",
                        "horizon_days": 20,
                        "neutral_band_absolute_return": 0.04,
                    }
                ],
                "pending_decisions": [
                    {
                        "decision_type": "hypothesis_disposition",
                        "hypothesis_id": "h1",
                        "allowed_decisions": ["accept_for_replay", "defer"],
                    }
                ],
            },
            "safety": {"can_trade": False},
        },
    )

    payload = ChiefReviewIndexBuilder(
        review,
        hypothesis_lifecycle_path=lifecycle,
        checkpoint_due_router_path=None,
        outcome_lifecycle_path=None,
        evidence_refresh_path=None,
        verified_source_router_path=None,
        output_dir=tmp_path / "out",
    ).build(save=False)

    assert payload["decision"]["decision"] == "hypothesis_review_required"
    assert payload["decision"]["hypothesis_lifecycle_state"][
        "proposed_contract_count"
    ] == 1
    assert len(payload["hypothesis_lifecycle_inbox"]["pending_decisions"]) == 1


def test_chief_review_prioritizes_measurement_blocker(tmp_path):
    review = tmp_path / "review.json"
    lifecycle = tmp_path / "lifecycle.json"
    _review_index(review)
    _write(
        lifecycle,
        {
            "run_id": "life_2",
            "review_inbox": {
                "status": "blocked_measurement_inputs",
                "blockers": [
                    {
                        "hypothesis_id": "h2",
                        "blockers": ["relative_return_benchmark_missing"],
                    }
                ],
                "proposed_contracts": [],
                "pending_decisions": [],
            },
            "safety": {"can_trade": False},
        },
    )

    payload = ChiefReviewIndexBuilder(
        review,
        hypothesis_lifecycle_path=lifecycle,
        checkpoint_due_router_path=None,
        outcome_lifecycle_path=None,
        evidence_refresh_path=None,
        verified_source_router_path=None,
        output_dir=tmp_path / "out",
    ).build(save=False)

    assert payload["decision"]["decision"] == "hypothesis_measurement_blocked"
    assert payload["decision"]["verdict"] == "blocked"
    assert payload["decision"]["hypothesis_lifecycle_state"]["blocker_count"] == 1


def test_chief_review_surfaces_only_matured_checkpoint_actions(tmp_path):
    review = tmp_path / "review.json"
    router = tmp_path / "router.json"
    _review_index(review)
    _write(
        router,
        {
            "run_id": "router_1",
            "chief_review_inbox": {
                "status": "matured_checkpoints_require_outcome_review",
                "matured_checkpoints": [
                    {
                        "task_id": "task_20",
                        "hypothesis_id": "h1",
                        "horizon_days": 20,
                        "checkpoint_session": "2026-07-15",
                    }
                ],
                "data_accrual_actions": [],
                "pending_decisions": [{"task_id": "task_20"}],
                "due_soon_silent_count": 3,
                "future_checkpoints_are_operator_actions": False,
            },
            "safety": {"can_trade": False},
        },
    )

    payload = ChiefReviewIndexBuilder(
        review,
        hypothesis_lifecycle_path=None,
        checkpoint_due_router_path=router,
        outcome_lifecycle_path=None,
        evidence_refresh_path=None,
        verified_source_router_path=None,
        output_dir=tmp_path / "out",
    ).build(save=False)

    assert payload["decision"]["decision"] == (
        "checkpoint_outcome_review_required"
    )
    assert payload["decision"]["checkpoint_due_state"][
        "matured_checkpoint_count"
    ] == 1
    assert payload["decision"]["checkpoint_due_state"][
        "future_checkpoints_are_operator_actions"
    ] is False


def test_chief_review_does_not_promote_future_checkpoint_to_action(tmp_path):
    review = tmp_path / "review.json"
    router = tmp_path / "router.json"
    _review_index(review)
    _write(
        router,
        {
            "run_id": "router_2",
            "chief_review_inbox": {
                "status": "no_checkpoint_action_required",
                "matured_checkpoints": [],
                "data_accrual_actions": [],
                "pending_decisions": [],
                "due_soon_silent_count": 2,
                "future_checkpoints_are_operator_actions": False,
            },
            "safety": {"can_trade": False},
        },
    )

    payload = ChiefReviewIndexBuilder(
        review,
        hypothesis_lifecycle_path=None,
        checkpoint_due_router_path=router,
        outcome_lifecycle_path=None,
        evidence_refresh_path=None,
        verified_source_router_path=None,
        output_dir=tmp_path / "out",
    ).build(save=False)

    assert payload["decision"]["decision"] != (
        "checkpoint_outcome_review_required"
    )
    assert payload["checkpoint_due_inbox"]["pending_decisions"] == []


def test_chief_review_surfaces_primary_outcome_causal_review(tmp_path):
    review = tmp_path / "review.json"
    lifecycle = tmp_path / "outcome_lifecycle.json"
    _review_index(review)
    _write(
        lifecycle,
        {
            "run_id": "outcome_life_1",
            "review_inbox": {
                "status": "primary_outcome_packet_pending_causal_review",
                "data_actions": [],
                "outcome_packets": [
                    {"task_id": "task_20", "result_label": "unobservable"}
                ],
                "learning_proposals": [],
                "pending_decisions": [
                    {
                        "task_id": "task_20",
                        "decision_type": "primary_checkpoint_causal_disposition",
                    }
                ],
            },
            "safety": {"can_trade": False},
        },
    )

    payload = ChiefReviewIndexBuilder(
        review,
        hypothesis_lifecycle_path=None,
        checkpoint_due_router_path=None,
        outcome_lifecycle_path=lifecycle,
        evidence_refresh_path=None,
        verified_source_router_path=None,
        output_dir=tmp_path / "out",
    ).build(save=False)

    assert payload["decision"]["decision"] == (
        "primary_outcome_causal_review_required"
    )
    assert payload["decision"]["outcome_lifecycle_state"][
        "pending_decision_count"
    ] == 1


def test_chief_review_surfaces_refresh_failure_without_outcome_judgment(tmp_path):
    review = tmp_path / "review.json"
    refresh = tmp_path / "refresh.json"
    _review_index(review)
    _write(
        refresh,
        {
            "run_id": "refresh_1",
            "summary": {
                "status": "single_refresh_pass_failed",
                "refresh_executed": False,
                "lifecycle_rerun": False,
            },
            "refresh_failure": {
                "error_type": "RuntimeError",
                "error": "no rows",
                "next_action": "Use alternate verified source.",
            },
            "safety": {"can_trade": False},
        },
    )

    payload = ChiefReviewIndexBuilder(
        review,
        hypothesis_lifecycle_path=None,
        checkpoint_due_router_path=None,
        outcome_lifecycle_path=None,
        evidence_refresh_path=refresh,
        verified_source_router_path=None,
        output_dir=tmp_path / "out",
    ).build(save=False)

    assert payload["decision"]["evidence_refresh_state"]["failure_recorded"] is True
    assert payload["decision"]["evidence_refresh_state"][
        "automatic_retry_allowed"
    ] is False
    assert payload["decision"]["decision"] != "hypothesis_review_required"


def test_chief_review_surfaces_next_bounded_verified_source(tmp_path):
    review = tmp_path / "review.json"
    router = tmp_path / "source_router.json"
    _review_index(review)
    _write(
        router,
        {
            "run_id": "source_router_1",
            "summary": {
                "status": "awaiting_operator_supplied_verified_snapshot",
                "ready_local_snapshot_count": 0,
            },
            "next_system_actions": [
                {
                    "action_type": "supply_local_verified_market_snapshot",
                    "task_id": "task_60",
                    "required_tickers": ["AMAT"],
                }
            ],
            "safety": {"can_trade": False},
        },
    )

    payload = ChiefReviewIndexBuilder(
        review,
        hypothesis_lifecycle_path=None,
        checkpoint_due_router_path=None,
        outcome_lifecycle_path=None,
        evidence_refresh_path=None,
        verified_source_router_path=router,
        output_dir=tmp_path / "out",
    ).build(save=False)

    state = payload["decision"]["verified_source_router_state"]
    assert state["status"] == "awaiting_operator_supplied_verified_snapshot"
    assert state["automatic_provider_loop_allowed"] is False
    assert any("AMAT" in item for item in payload["decision"]["next_actions"])
