from __future__ import annotations

from datetime import UTC, datetime

from dean_os.replays.replay_evaluation_router import ReplayEvaluationRouter


def test_hypothesis_task_is_not_misrouted_to_event_study() -> None:
    task = {
        "task_id": "h1-30d",
        "hypothesis_id": "h1",
        "as_of": "2026-07-01T00:00:00+00:00",
        "due_at": "2026-07-31T00:00:00+00:00",
        "expected_observations": ["orders rise"],
        "invalidation_signals": ["capex cut"],
    }
    route = ReplayEvaluationRouter().route_task(
        task,
        evaluation_as_of=datetime(2026, 7, 11, tzinfo=UTC),
    )

    assert route.route == "hypothesis_outcome_replay"
    assert route.evaluation_status == "waiting"
    assert route.event_study_eligible_to_check is False
    assert route.hypothesis_outcome_eligible_to_check is False
    assert "task_as_of_not_treated_as_event_timestamp" in route.warnings
    assert "market_price_response_context_only" in route.secondary_outcomes


def test_verified_timestamped_event_routes_to_event_study() -> None:
    task = {
        "task_id": "event-1",
        "event_id": "event-1",
        "event_timestamp": "2026-07-01T14:00:00+00:00",
        "release_timestamp_verified": True,
    }
    route = ReplayEvaluationRouter().route_task(
        task,
        evaluation_as_of=datetime(2026, 7, 11, tzinfo=UTC),
    )

    assert route.route == "event_study"
    assert route.evaluation_status == "ready_for_event_study_eligibility"
    assert route.event_study_eligible_to_check is True
    assert "abnormal_return" in route.secondary_outcomes


def test_unverified_event_timestamp_is_blocked() -> None:
    task = {
        "task_id": "event-2",
        "event_id": "event-2",
        "event_timestamp": "2026-07-01T14:00:00+00:00",
    }
    route = ReplayEvaluationRouter().route_task(
        task,
        evaluation_as_of=datetime(2026, 7, 11, tzinfo=UTC),
    )

    assert route.route == "blocked_unroutable"
    assert route.evaluation_status == "blocked"
    assert "event_release_timestamp_not_verified" in route.blockers
