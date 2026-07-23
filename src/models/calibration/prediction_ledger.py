"""Prediction ledger: the missing second half of confidence-calibrator wiring.

get_confidence_calibrator() (adaptive_confidence_calibrator.py) is already
applied at prediction time (Stage 5), but its online-learning half —
update_with_outcome(raw_confidence, actual_outcome) — needs to know, for
each past prediction, what actually happened once its horizon has elapsed.
Investigation during the same session that wired in the calibrator found
no existing per-signal outcome-reconciliation mechanism in the active
src/ pipeline (Stage 7 evaluates portfolio-level performance, not
individual predictions against later-realized prices).

This module is that mechanism, kept deliberately simple and decoupled:

1. record_prediction() — append-only JSONL journal, called from Stage 5
   right after a prediction is made. No database dependency, no coupling
   to whichever DataManager backend is configured.
2. PredictionOutcomeReconciler — a separate, explicitly-invoked process
   (not run automatically inside the live Stage 5->7 flow, since horizons
   elapse asynchronously, on their own calendar) that:
   - finds ledger entries whose horizon has elapsed and are unresolved,
   - looks up the realized price at the horizon date via an injected
     price_lookup callable (kept injectable, not hardcoded to one data
     source, for testability and because this repo has more than one
     price-loading path),
   - derives directional correctness (predicted up/down vs. what
     actually happened) the same way
     dean_os/analyst_core/outcome_evaluator.py already does for the
     analyst layer,
   - feeds the calibrator and persists both the calibrator state and the
     now-resolved ledger entries.

Intentionally NOT wired into any scheduler here — how often to reconcile
(daily cron, manual invocation, a dean_os review cycle) is an operational
decision for the project owner, not something to guess at. See
scripts/reconcile_prediction_outcomes.py for a ready-to-run CLI entrypoint.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from src.core.logging.logger import ProjectLogger
from src.models.calibration.adaptive_confidence_calibrator import (
    DEFAULT_CALIBRATOR_PATH,
    get_confidence_calibrator,
)

logger = ProjectLogger.get_logger("PredictionLedger")

DEFAULT_LEDGER_PATH = "data/models/prediction_ledger.jsonl"

# Matches the trailing horizon token in target names like "target_up_1d",
# "target_return_4h", "target_up_1w". Falls back to 1 calendar day when a
# target name doesn't carry a recognizable horizon — better to reconcile
# late/approximately than to silently never reconcile at all.
_HORIZON_PATTERN = re.compile(r"_(\d+)(m|h|d|w)(?:_|$)")
_HORIZON_DAYS_PER_UNIT = {"m": 1 / (24 * 60), "h": 1 / 24, "d": 1.0, "w": 7.0}


def parse_horizon_days(target_name: str, default_days: float = 1.0) -> float:
    """Best-effort horizon-in-days extraction from a target column name."""
    match = _HORIZON_PATTERN.search(target_name or "")
    if not match:
        return default_days
    count, unit = match.groups()
    return float(count) * _HORIZON_DAYS_PER_UNIT[unit]


@dataclass
class PredictionLedgerEntry:
    prediction_id: str
    ticker: str
    target_name: str
    timeframe: str
    as_of: str  # ISO timestamp when the prediction was made
    horizon_days: float
    predicted_value: float
    last_price: float
    raw_confidence: float
    calibrated_confidence: float
    resolved: bool = False
    actual_outcome: int | None = None
    resolved_at: str | None = None

    @property
    def due_at(self) -> datetime:
        return datetime.fromisoformat(self.as_of) + timedelta(days=self.horizon_days)


def record_prediction(
    *,
    ticker: str,
    target_name: str,
    timeframe: str,
    predicted_value: float,
    last_price: float | None,
    raw_confidence: float,
    calibrated_confidence: float,
    as_of: datetime | None = None,
    ledger_path: str = DEFAULT_LEDGER_PATH,
) -> str | None:
    """Append one prediction to the ledger. Returns the prediction_id, or
    None if last_price is unavailable (direction can't be derived later
    without an anchor price, so there's nothing useful to reconcile)."""
    if last_price is None:
        return None

    entry = PredictionLedgerEntry(
        prediction_id=uuid4().hex,
        ticker=ticker,
        target_name=target_name,
        timeframe=timeframe,
        as_of=(as_of or datetime.now()).isoformat(),
        horizon_days=parse_horizon_days(target_name),
        predicted_value=float(predicted_value),
        last_price=float(last_price),
        raw_confidence=float(raw_confidence),
        calibrated_confidence=float(calibrated_confidence),
    )

    try:
        path = Path(ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        return entry.prediction_id
    except (OSError, TypeError, ValueError) as e:
        logger.warning(f"Failed to record prediction to ledger: {e}")
        return None


class PredictionOutcomeReconciler:
    """Resolves due predictions against realized prices and feeds the
    confidence calibrator's online-learning loop."""

    def __init__(
        self,
        price_lookup: Callable[[str, datetime], float | None],
        ledger_path: str = DEFAULT_LEDGER_PATH,
        calibrator_path: str | None = None,
    ):
        """
        Args:
            price_lookup: (ticker, as_of_date) -> close price at/after that
                date, or None if not yet available. Injected rather than
                hardcoded so callers can point this at whichever price
                source/DataManager instance they already have configured.
            ledger_path: JSONL file written by record_prediction().
            calibrator_path: passed through to get_confidence_calibrator();
                None uses its default.
        """
        self.price_lookup = price_lookup
        self.ledger_path = ledger_path
        self.calibrator_path = calibrator_path
        self.logger = logger

    def _load_entries(self) -> list[PredictionLedgerEntry]:
        path = Path(self.ledger_path)
        if not path.exists():
            return []
        entries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(PredictionLedgerEntry(**json.loads(line)))
        return entries

    def _save_entries(self, entries: list[PredictionLedgerEntry]) -> None:
        path = Path(self.ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(asdict(entry)) + "\n")

    def reconcile_due_predictions(self, as_of: datetime | None = None) -> dict[str, Any]:
        """Resolves every unresolved entry whose horizon has elapsed and a
        realized price is available. Entries whose horizon has elapsed but
        have no price yet are left unresolved (tried again next run)."""
        now = as_of or datetime.now()
        entries = self._load_entries()
        resolved_calibrator_path = self.calibrator_path or DEFAULT_CALIBRATOR_PATH
        calibrator = get_confidence_calibrator(resolved_calibrator_path)

        resolved_count = 0
        skipped_no_price = 0
        for entry in entries:
            if entry.resolved or entry.due_at > now:
                continue

            realized_price = self.price_lookup(entry.ticker, entry.due_at)
            if realized_price is None:
                skipped_no_price += 1
                continue

            predicted_direction = 1 if entry.predicted_value > entry.last_price else 0
            actual_direction = 1 if realized_price > entry.last_price else 0
            actual_outcome = 1 if predicted_direction == actual_direction else 0

            calibrator.update_with_outcome(entry.raw_confidence, actual_outcome)

            entry.resolved = True
            entry.actual_outcome = actual_outcome
            entry.resolved_at = now.isoformat()
            resolved_count += 1

        if resolved_count:
            self._save_entries(entries)
            calibrator.save(resolved_calibrator_path)

        self.logger.info(
            f"Reconciliation: {resolved_count} resolved, {skipped_no_price} "
            f"awaiting price data, {len(entries) - resolved_count - skipped_no_price} not yet due."
        )
        return {
            "resolved": resolved_count,
            "awaiting_price": skipped_no_price,
            "total_entries": len(entries),
        }
