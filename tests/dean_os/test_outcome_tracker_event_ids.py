"""Event ids must be unique even when registrations share a timestamp.

`register()` used to derive event_id purely from
`datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')`. Two registrations inside the
same microsecond therefore produced the same id and the second one died with
"UNIQUE constraint failed: tracked_events.event_id". Registering several
replay tasks in a loop hits that regularly -- and on Windows, where the system
clock is coarse, it was the common case, which is why
test_world_model_replay_registration_bridge.py failed intermittently and
looked like a test-isolation problem.
"""
from __future__ import annotations

from dean_os.outcome_tracker import OutcomeTracker


def test_rapid_successive_registrations_get_distinct_ids(tmp_path):
    tracker = OutcomeTracker(tmp_path / "outcome_tracker.sqlite")

    ids = [
        tracker.register(f"headline {i}", event_type="replay", shock="positive")
        for i in range(50)
    ]

    assert len(ids) == 50
    assert len(set(ids)) == 50, "event ids collided within the same microsecond"
    assert tracker.stats()["events"] == 50


def test_ids_keep_a_sortable_timestamp_prefix(tmp_path):
    tracker = OutcomeTracker(tmp_path / "outcome_tracker.sqlite")
    event_id = tracker.register("headline", event_type="replay")
    assert event_id.startswith("evt_")
    # evt_ + YYYYMMDD_HHMMSS_ffffff + _ + 8 hex chars
    parts = event_id.split("_")
    assert len(parts) == 5, event_id
    assert len(parts[1]) == 8 and parts[1].isdigit()
    assert len(parts[-1]) == 8
