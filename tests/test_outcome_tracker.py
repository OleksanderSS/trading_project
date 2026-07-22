"""Full lifecycle test: register → force outcomes → calibrate."""
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dean_os.outcome_tracker import OutcomeTracker, REPLAY_INTERVALS


def _simulate_days(n: int) -> str:
    return (datetime.now(UTC) - timedelta(days=n + 1)).isoformat()


def test_lifecycle():
    db = Path("data/dean_os/_test_outcome_tracker.sqlite")
    if db.exists():
        db.unlink()

    tracker = OutcomeTracker(db)

    # 1. Register events — force old timestamps so intervals are "passed"
    old = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    tracker.register(
        headline="Test geopolitical crisis — sanctions imposed",
        event_type="geopolitical",
        shock="negative",
        impact_estimate=-0.7,
        confidence=0.8,
        sectors=["semiconductor", "energy"],
        source="test",
    )
    tracker.register(
        headline="Test positive earnings beat expectations",
        event_type="corporate",
        shock="positive",
        impact_estimate=0.5,
        confidence=0.75,
        sectors=["technology"],
        source="test",
    )
    import sqlite3
    with sqlite3.connect(str(db)) as con:
        con.execute("UPDATE tracked_events SET registered_at = ?", (old,))

    stats = tracker.stats()
    assert stats["events"] == 2, f"Expected 2 events, got {stats['events']}"
    assert stats["predictions"] == len(REPLAY_INTERVALS) * 2, f"Expected {len(REPLAY_INTERVALS)*2} predictions"

    # 2. Check outcomes
    stances = {"semiconductor": "bearish", "energy": "bearish", "technology": "bullish"}
    outcomes = tracker.check_due(current_stance_by_sector=stances)
    assert len(outcomes) > 0, "Should have due outcomes"

    # 3. Calibrate
    cal = tracker.calibrate()
    assert cal.total_events == 2
    assert cal.total_outcomes > 0
    assert 0 <= cal.brier_score <= 1
    assert 0 <= cal.accuracy_rate <= 1

    print(f"Events: {cal.total_events}, Outcomes: {cal.total_outcomes}")
    print(f"Brier: {cal.brier_score}, Accuracy: {cal.accuracy_rate:.2%}")
    print(f"By interval: {cal.by_interval}")

    # 4. List events
    events = tracker.list_events()
    assert len(events) >= 2
    print(f"Latest event: {events[0]['headline'][:60]}")

    if db.exists():
        try:
            db.unlink()
        except PermissionError:
            pass
    print("PASSED")


def test_exact_custom_interval_uses_explicit_event_origin(tmp_path):
    db = tmp_path / "custom_interval_tracker.sqlite"
    tracker = OutcomeTracker(db)
    origin = "2026-05-14T20:13:54+00:00"

    tracker.register(
        headline="Exact 20-day event-response checkpoint",
        event_type="world_model_replay_task",
        source="test_custom_interval",
        directions={20: "neutral"},
        intervals=[20],
        registered_at=origin,
    )

    import sqlite3

    with sqlite3.connect(str(db)) as con:
        event_origin = con.execute(
            "SELECT registered_at FROM tracked_events"
        ).fetchone()[0]
        intervals = [
            row[0]
            for row in con.execute(
                "SELECT interval_days FROM predictions ORDER BY interval_days"
            ).fetchall()
        ]

    assert event_origin == origin
    assert intervals == [20]


if __name__ == "__main__":
    test_lifecycle()
