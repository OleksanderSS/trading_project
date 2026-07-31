from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

REPLAY_INTERVALS: list[int] = [1, 5, 30, 60, 120]
_DEFAULT_DB = Path("data/dean_os/outcome_tracker.sqlite")


class TrackedEvent(BaseModel):
    event_id: str = ""
    headline: str = ""
    event_type: str = "uncategorized"
    shock: str = "neutral"
    impact_estimate: float = 0.0
    confidence: float = 0.5
    sectors: list[str] = []
    source: str = ""
    registered_at: str = ""
    predictions: dict[str, Prediction] = {}  # interval_days → prediction


class Prediction(BaseModel):
    interval_days: int
    predicted_direction: str  # bullish, bearish, neutral
    confidence: float = 0.5
    narrative: str = ""


class Outcome(BaseModel):
    event_id: str
    interval_days: int
    checked_at: str
    current_stance: str = ""
    accuracy_score: float = 0.0
    notes: str = ""


class CalibrationSnapshot(BaseModel):
    calculated_at: str = ""
    total_events: int = 0
    total_outcomes: int = 0
    brier_score: float = 0.0
    accuracy_rate: float = 0.0
    by_interval: dict[str, dict[str, float]] = {}


class OutcomeTracker:
    """Tracks events + predictions and checks outcomes at fixed intervals.

    The core loop:
        1. Register event with predictions for each interval
        2. At each interval, check current reality against prediction
        3. Store outcome, calculate calibration scores
        4. Feed calibration back into agent confidence
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS tracked_events (
                    event_id TEXT PRIMARY KEY,
                    headline TEXT,
                    event_type TEXT,
                    shock TEXT,
                    impact_estimate REAL,
                    confidence REAL,
                    sectors TEXT,
                    source TEXT,
                    registered_at TEXT
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT REFERENCES tracked_events(event_id),
                    interval_days INTEGER,
                    predicted_direction TEXT,
                    confidence REAL,
                    narrative TEXT
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT REFERENCES tracked_events(event_id),
                    interval_days INTEGER,
                    checked_at TEXT,
                    current_stance TEXT,
                    accuracy_score REAL,
                    notes TEXT
                );
                CREATE TABLE IF NOT EXISTS calibration_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    calculated_at TEXT,
                    total_events INTEGER,
                    total_outcomes INTEGER,
                    brier_score REAL,
                    accuracy_rate REAL,
                    by_interval TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_outcomes_event ON outcomes(event_id);
                CREATE INDEX IF NOT EXISTS idx_outcomes_due ON outcomes(checked_at);
                CREATE INDEX IF NOT EXISTS idx_predictions_event ON predictions(event_id);
            """)

    # ── Register ──────────────────────────────────────────────────────────

    def register(
        self,
        headline: str,
        *,
        event_type: str = "uncategorized",
        shock: str = "neutral",
        impact_estimate: float = 0.0,
        confidence: float = 0.5,
        sectors: list[str] | None = None,
        source: str = "",
        directions: dict[int, str] | None = None,
        intervals: list[int] | tuple[int, ...] | None = None,
        registered_at: str | None = None,
    ) -> str:
        """Register an event with predictions for each interval.

        Args:
            directions: Override predicted direction per interval.
                        If None, infers from shock: positive→bullish, negative→bearish.

            intervals: Exact checkpoint horizons. Defaults to the legacy
                       REPLAY_INTERVALS set for backward compatibility.
            registered_at: Timezone-aware event origin. Defaults to now; replay
                           callers should pass the trigger time, not review time.

        Returns:
            event_id
        """
        now = datetime.now(UTC)
        event_origin = now
        if registered_at is not None:
            try:
                parsed_origin = datetime.fromisoformat(
                    str(registered_at).replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise ValueError("registered_at must be a valid ISO timestamp") from exc
            if parsed_origin.tzinfo is None or parsed_origin.utcoffset() is None:
                raise ValueError("registered_at must be timezone-aware")
            event_origin = parsed_origin.astimezone(UTC)
        checkpoint_intervals = sorted(
            {
                int(interval)
                for interval in (intervals if intervals is not None else REPLAY_INTERVALS)
                if int(interval) > 0
            }
        )
        if not checkpoint_intervals:
            raise ValueError("intervals must contain at least one positive horizon")
        # The timestamp prefix is kept because it makes ids readable and
        # sortable, but it CANNOT be the whole id: two registrations landing in
        # the same microsecond produce the same string and the second one dies
        # on "UNIQUE constraint failed: tracked_events.event_id". That is not
        # theoretical -- registering several replay tasks in a loop hits it
        # regularly, and Windows' coarse system-clock resolution makes it the
        # common case rather than a rare race. Nothing parses this format back
        # into a timestamp (verified: it is constructed here and nowhere else),
        # so a random suffix is free.
        event_id = f"evt_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"

        with sqlite3.connect(str(self.db_path)) as con:
            con.execute(
                """INSERT INTO tracked_events
                   (event_id, headline, event_type, shock, impact_estimate, confidence, sectors, source, registered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, headline, event_type, shock, impact_estimate, confidence,
                 json.dumps(sectors or []), source, event_origin.isoformat()),
            )

            for interval in checkpoint_intervals:
                if directions and interval in directions:
                    pred_dir = directions[interval]
                else:
                    pred_dir = {"positive": "bullish", "negative": "bearish"}.get(shock, "neutral")

                pred_conf = confidence * (1.0 - (interval / 200.0))  # decay confidence with horizon
                con.execute(
                    """INSERT INTO predictions (event_id, interval_days, predicted_direction, confidence, narrative)
                       VALUES (?, ?, ?, ?, ?)""",
                    (event_id, interval, pred_dir, max(pred_conf, 0.1),
                     f"Predicted {pred_dir} at {interval}d based on {event_type} event ({shock})"),
                )

        return event_id

    # ── Check Outcomes ────────────────────────────────────────────────────

    def check_due(self, current_stance_by_sector: dict[str, str] | None = None) -> list[Outcome]:
        """Check all predictions whose interval has elapsed but have no outcome yet.

        Args:
            current_stance_by_sector: Dict mapping sector → current stance.
                                       If None, outcomes are registered without accuracy.

        Returns:
            List of new Outcome records.
        """
        now = datetime.now(UTC)
        new_outcomes: list[Outcome] = []

        with sqlite3.connect(str(self.db_path)) as con:
            due = con.execute(
                """SELECT p.event_id, e.registered_at, e.sectors, p.interval_days,
                          p.predicted_direction, p.confidence
                   FROM predictions p
                   JOIN tracked_events e ON p.event_id = e.event_id
                   LEFT JOIN outcomes o ON p.event_id = o.event_id AND p.interval_days = o.interval_days
                   WHERE o.id IS NULL
                     AND datetime(e.registered_at, '+' || p.interval_days || ' days') <= datetime(?)
                """,
                (now.isoformat(),),
            ).fetchall()

            for row in due:
                event_id, registered_at, sectors_json, interval, pred_dir, pred_conf = row
                sectors: list[str] = json.loads(sectors_json) if sectors_json else []

                accuracy = 0.5  # neutral default
                notes = ""

                if current_stance_by_sector:
                    matching_stances = [
                        s for s in sectors
                        if s in current_stance_by_sector
                    ]
                    if matching_stances:
                        actual_stances = [current_stance_by_sector[s] for s in matching_stances]
                        avg_direction = self._stance_to_direction(actual_stances)
                        accuracy = 1.0 if avg_direction == pred_dir else 0.0
                        notes = f"Sectors: {matching_stances}, actual: {actual_stances}"

                outcome = Outcome(
                    event_id=event_id,
                    interval_days=interval,
                    checked_at=now.isoformat(),
                    current_stance=notes,
                    accuracy_score=accuracy,
                    notes=notes,
                )
                con.execute(
                    """INSERT INTO outcomes (event_id, interval_days, checked_at, current_stance, accuracy_score, notes)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (event_id, interval, outcome.checked_at, outcome.current_stance,
                     outcome.accuracy_score, outcome.notes),
                )
                new_outcomes.append(outcome)

        return new_outcomes

    # ── Calibration ───────────────────────────────────────────────────────

    def calibrate(self) -> CalibrationSnapshot:
        """Calculate calibration metrics: Brier score, accuracy rate by interval."""
        with sqlite3.connect(str(self.db_path)) as con:
            outcomes = con.execute(
                "SELECT interval_days, accuracy_score FROM outcomes"
            ).fetchall()

        if not outcomes:
            return CalibrationSnapshot(calculated_at=datetime.now(UTC).isoformat())

        total = len(outcomes)
        by_interval: dict[int, list[float]] = {}
        for interval, score in outcomes:
            by_interval.setdefault(interval, []).append(score)

        brier = sum((1.0 - s) ** 2 for _, s in outcomes) / total
        accurate = sum(1 for _, s in outcomes if s >= 0.5)
        accuracy_rate = accurate / total

        interval_breakdown: dict[str, dict[str, float]] = {}
        for interval, scores in sorted(by_interval.items()):
            n = len(scores)
            correct = sum(1 for s in scores if s >= 0.5)
            interval_breakdown[str(interval)] = {
                "count": n,
                "accuracy": round(correct / n, 3),
                "avg_score": round(sum(scores) / n, 3),
            }

        snapshot = CalibrationSnapshot(
            calculated_at=datetime.now(UTC).isoformat(),
            total_events=len(set(o[0] for o in con.execute("SELECT DISTINCT event_id FROM outcomes").fetchall())),
            total_outcomes=total,
            brier_score=round(brier, 4),
            accuracy_rate=round(accuracy_rate, 4),
            by_interval=interval_breakdown,
        )

        # Log calibration
        with sqlite3.connect(str(self.db_path)) as con:
            con.execute(
                """INSERT INTO calibration_log
                   (calculated_at, total_events, total_outcomes, brier_score, accuracy_rate, by_interval)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (snapshot.calculated_at, snapshot.total_events, snapshot.total_outcomes,
                 snapshot.brier_score, snapshot.accuracy_rate, json.dumps(interval_breakdown)),
            )

        return snapshot

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with sqlite3.connect(str(self.db_path)) as con:
            events = con.execute("SELECT count(*) FROM tracked_events").fetchone()[0]
            preds = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
            outs = con.execute("SELECT count(*) FROM outcomes").fetchone()[0]
            due = con.execute(
                """SELECT count(*)
                   FROM predictions p
                   JOIN tracked_events e ON p.event_id = e.event_id
                   LEFT JOIN outcomes o ON p.event_id = o.event_id AND p.interval_days = o.interval_days
                   WHERE o.id IS NULL
                     AND datetime(e.registered_at, '+' || p.interval_days || ' days') <= datetime('now')
                """
            ).fetchone()[0]
        return {
            "events": events,
            "predictions": preds,
            "outcomes": outs,
            "due": due,
            "intervals": REPLAY_INTERVALS,
        }

    # ── Event List ────────────────────────────────────────────────────────

    def list_events(self, limit: int = 20) -> list[dict[str, Any]]:
        with sqlite3.connect(str(self.db_path)) as con:
            rows = con.execute(
                """SELECT e.event_id, e.headline, e.event_type, e.shock, e.impact_estimate,
                          e.registered_at,
                          (SELECT count(*) FROM outcomes o WHERE o.event_id = e.event_id) as outcome_count,
                          (SELECT count(*) FROM predictions p WHERE p.event_id = e.event_id) as pred_count
                   FROM tracked_events e
                   ORDER BY e.registered_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "event_id": r[0],
                "headline": r[1][:80],
                "event_type": r[2],
                "shock": r[3],
                "impact": r[4],
                "registered_at": r[5],
                "outcomes": r[6],
                "predictions": r[7],
            }
            for r in rows
        ]

    # ── Helpers ───────────────────────────────────────────────────────────

    # ── Paper Trade Bridge ────────────────────────────────────────────────

    def register_paper_trade(self, trade: dict[str, Any]) -> str | None:
        """Register a PaperTradeRecord in the outcome tracker.

        Expects a dict with keys matching PaperTradeRecord fields.
        Returns event_id or None if skipped.
        """
        headline = trade.get("thesis", trade.get("trade_id", "paper_trade"))
        direction = trade.get("expected_direction", "neutral")
        horizon = trade.get("horizon_days")
        if not horizon or horizon <= 0:
            return None

        shock_map = {"bullish": "positive", "bearish": "negative", "neutral": "neutral"}
        sectors: list[str] = []
        context_tags: list = trade.get("context_tags") or []
        regime_tags: list = trade.get("regime_tags") or []
        for tag in context_tags + regime_tags:
            if isinstance(tag, str) and ":" in tag:
                parts = tag.split(":")
                if len(parts) >= 2 and parts[0] in ("sector", "domain"):
                    sectors.append(parts[1])
        if not sectors:
            sectors = ["global"]

        directions: dict[int, str] = {}
        for interval in REPLAY_INTERVALS:
            if interval <= horizon:
                directions[interval] = direction

        return self.register(
            headline=headline,
            event_type="paper_trade",
            shock=shock_map.get(direction, "neutral"),
            impact_estimate=0.0,
            confidence=trade.get("confidence", 0.5),
            sectors=sectors,
            source=trade.get("source_type", "paper_trade"),
            directions=directions if directions else {horizon: direction},
        )

    def check_paper_trades(
        self,
        horizon_map: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Check paper trade outcomes and return update instructions.

        Args:
            horizon_map: Trade ID → actual horizon_days override.

        Returns:
            List of dicts with trade_id, interval, accuracy, label.
        """
        results: list[dict[str, Any]] = []
        with sqlite3.connect(str(self.db_path)) as con:
            out = con.execute(
                """SELECT o.event_id, o.interval_days, o.accuracy_score,
                          e.headline
                   FROM outcomes o
                   JOIN tracked_events e ON o.event_id = e.event_id
                   WHERE e.event_type = 'paper_trade'
                """
            ).fetchall()

        for event_id, interval, accuracy, headline in out:
            label = "hit" if accuracy >= 0.5 else "miss"
            results.append({
                "event_id": event_id,
                "headline": headline[:80],
                "interval_days": interval,
                "accuracy": accuracy,
                "label": label,
            })
        return results

    @staticmethod
    def _stance_to_direction(stances: list[str]) -> str:
        bullish = sum(1 for s in stances if s in ("bullish", "clear"))
        bearish = sum(1 for s in stances if s in ("bearish", "blocked"))
        if bullish > bearish:
            return "bullish"
        elif bearish > bullish:
            return "bearish"
        return "neutral"


__all__ = [
    "CalibrationSnapshot", "Outcome", "OutcomeTracker",
    "Prediction", "REPLAY_INTERVALS", "TrackedEvent",
]
