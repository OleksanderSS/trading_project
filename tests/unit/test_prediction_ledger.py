"""Tests for src/models/calibration/prediction_ledger.py — the outcome
feedback loop that completes the confidence-calibrator wiring started in
the same session (see test_confidence_calibrator_wiring.py for the
calibrate()-at-prediction-time half)."""
import os
from datetime import datetime, timedelta

import pytest

import src.models.calibration.adaptive_confidence_calibrator as calibrator_module
from src.models.calibration.prediction_ledger import (
    PredictionOutcomeReconciler,
    parse_horizon_days,
    record_prediction,
)

LEDGER_PATH = "data/models/_test_prediction_ledger.jsonl"
CALIBRATOR_PATH = "data/models/_test_ledger_calibrator.joblib"


@pytest.fixture(autouse=True)
def _isolated_files():
    calibrator_module._calibrator_instance = None
    os.makedirs("data/models", exist_ok=True)
    yield
    calibrator_module._calibrator_instance = None
    for path in (LEDGER_PATH, CALIBRATOR_PATH):
        if os.path.exists(path):
            os.remove(path)


def test_parse_horizon_days_recognizes_common_suffixes():
    assert parse_horizon_days("target_up_1d") == pytest.approx(1.0)
    assert parse_horizon_days("target_up_1w") == pytest.approx(7.0)
    assert parse_horizon_days("target_return_4h") == pytest.approx(4 / 24)
    assert parse_horizon_days("target_return_15m") == pytest.approx(15 / (24 * 60))


def test_parse_horizon_days_falls_back_to_default_for_unrecognized_names():
    assert parse_horizon_days("some_unusual_target_name") == pytest.approx(1.0)
    assert parse_horizon_days("") == pytest.approx(1.0)


def test_record_prediction_returns_none_without_a_last_price():
    result = record_prediction(
        ticker="AAPL", target_name="target_up_1d", timeframe="1d",
        predicted_value=101.0, last_price=None,
        raw_confidence=0.6, calibrated_confidence=0.55,
        ledger_path=LEDGER_PATH,
    )
    assert result is None
    assert not os.path.exists(LEDGER_PATH)


def test_record_prediction_writes_a_resolvable_ledger_entry():
    prediction_id = record_prediction(
        ticker="AAPL", target_name="target_up_1d", timeframe="1d",
        predicted_value=105.0, last_price=100.0,
        raw_confidence=0.7, calibrated_confidence=0.65,
        as_of=datetime(2026, 1, 1),
        ledger_path=LEDGER_PATH,
    )
    assert prediction_id is not None
    assert os.path.exists(LEDGER_PATH)


def test_reconciler_resolves_correct_up_prediction_and_updates_calibrator():
    record_prediction(
        ticker="AAPL", target_name="target_up_1d", timeframe="1d",
        predicted_value=105.0, last_price=100.0,  # predicted UP
        raw_confidence=0.8, calibrated_confidence=0.75,
        as_of=datetime(2026, 1, 1),
        ledger_path=LEDGER_PATH,
    )

    def price_lookup(ticker, as_of_date):
        return 110.0  # price went UP — prediction was correct

    reconciler = PredictionOutcomeReconciler(
        price_lookup=price_lookup, ledger_path=LEDGER_PATH, calibrator_path=CALIBRATOR_PATH,
    )
    summary = reconciler.reconcile_due_predictions(as_of=datetime(2026, 1, 5))

    assert summary["resolved"] == 1
    assert summary["awaiting_price"] == 0

    entries = reconciler._load_entries()
    assert entries[0].resolved is True
    assert entries[0].actual_outcome == 1  # predicted direction matched realized direction


def test_reconciler_resolves_incorrect_prediction_as_zero():
    record_prediction(
        ticker="AAPL", target_name="target_up_1d", timeframe="1d",
        predicted_value=105.0, last_price=100.0,  # predicted UP
        raw_confidence=0.8, calibrated_confidence=0.75,
        as_of=datetime(2026, 1, 1),
        ledger_path=LEDGER_PATH,
    )

    def price_lookup(ticker, as_of_date):
        return 90.0  # price went DOWN — prediction was wrong

    reconciler = PredictionOutcomeReconciler(
        price_lookup=price_lookup, ledger_path=LEDGER_PATH, calibrator_path=CALIBRATOR_PATH,
    )
    reconciler.reconcile_due_predictions(as_of=datetime(2026, 1, 5))

    entries = reconciler._load_entries()
    assert entries[0].actual_outcome == 0


def test_reconciler_leaves_entries_unresolved_when_horizon_not_yet_elapsed():
    record_prediction(
        ticker="AAPL", target_name="target_up_1d", timeframe="1d",
        predicted_value=105.0, last_price=100.0,
        raw_confidence=0.8, calibrated_confidence=0.75,
        as_of=datetime(2026, 1, 1),
        ledger_path=LEDGER_PATH,
    )

    reconciler = PredictionOutcomeReconciler(
        price_lookup=lambda t, d: 110.0, ledger_path=LEDGER_PATH, calibrator_path=CALIBRATOR_PATH,
    )
    # Only a few hours later — the 1-day horizon hasn't elapsed yet.
    summary = reconciler.reconcile_due_predictions(as_of=datetime(2026, 1, 1, 2, 0, 0))

    assert summary["resolved"] == 0
    entries = reconciler._load_entries()
    assert entries[0].resolved is False


def test_reconciler_leaves_entries_unresolved_when_price_not_yet_available():
    record_prediction(
        ticker="AAPL", target_name="target_up_1d", timeframe="1d",
        predicted_value=105.0, last_price=100.0,
        raw_confidence=0.8, calibrated_confidence=0.75,
        as_of=datetime(2026, 1, 1),
        ledger_path=LEDGER_PATH,
    )

    reconciler = PredictionOutcomeReconciler(
        price_lookup=lambda t, d: None,  # data not caught up yet
        ledger_path=LEDGER_PATH, calibrator_path=CALIBRATOR_PATH,
    )
    summary = reconciler.reconcile_due_predictions(as_of=datetime(2026, 1, 5))

    assert summary["resolved"] == 0
    assert summary["awaiting_price"] == 1


def test_reconciler_does_not_double_resolve_already_resolved_entries():
    record_prediction(
        ticker="AAPL", target_name="target_up_1d", timeframe="1d",
        predicted_value=105.0, last_price=100.0,
        raw_confidence=0.8, calibrated_confidence=0.75,
        as_of=datetime(2026, 1, 1),
        ledger_path=LEDGER_PATH,
    )

    reconciler = PredictionOutcomeReconciler(
        price_lookup=lambda t, d: 110.0, ledger_path=LEDGER_PATH, calibrator_path=CALIBRATOR_PATH,
    )
    first = reconciler.reconcile_due_predictions(as_of=datetime(2026, 1, 5))
    second = reconciler.reconcile_due_predictions(as_of=datetime(2026, 1, 6))

    assert first["resolved"] == 1
    assert second["resolved"] == 0  # already resolved, not reprocessed
