from datetime import datetime, timezone

import pytest

from src.meta_learning.memory.diary_engine import DecisionOutcome, DecisionType
from src.pipeline.stages.stage_6_trading_execution import TradingExecutionStage


class DiaryStub:
    def __init__(self):
        self.records = []

    def record_decision(self, record):
        self.records.append(record)


class LoggerStub:
    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _stage():
    stage = object.__new__(TradingExecutionStage)
    stage.diary_engine = DiaryStub()
    stage.logger = LoggerStub()
    return stage


def test_stage6_records_profitable_sell_with_pattern_context():
    stage = _stage()
    timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    transaction = {
        "timestamp": timestamp,
        "type": "SELL",
        "ticker": "aapl",
        "quantity": 2,
        "price": 110.0,
        "trade_value": 220.0,
        "net_revenue": 220.0,
        "pnl": 20.0,
        "pnl_pct": 10.0,
        "reason": "Take-Profit",
    }
    prediction = {
        "ticker": "AAPL",
        "selected_primary_model": "catboost_model",
        "predictions": [0.01, 0.03],
        "raw_forecast": 0.025,
        "confidence": 0.7,
        "context_fingerprint": "fp-123",
        "context_pattern_id": "pattern-1",
        "context_pattern_seq": "1|1>>1|0>>0|1",
        "context_velocity": "0.42",
    }

    written = stage._record_transactions_to_diary([transaction], [prediction])

    assert written == 1
    record = stage.diary_engine.records[0]
    assert record.agent_id == "catboost_model"
    assert record.ticker == "AAPL"
    assert record.decision_type == DecisionType.SELL
    assert record.outcome == DecisionOutcome.PROFITABLE
    assert record.profit_loss == pytest.approx(0.10)
    assert record.entry_price == pytest.approx(100.0)
    assert record.exit_price == pytest.approx(110.0)
    assert record.context_fingerprint == "fp-123"
    assert record.context_pattern_seq == "1|1>>1|0>>0|1"
    assert record.model_prediction == pytest.approx(0.03)
    assert record.model_confidence == pytest.approx(0.7)
    # Seconds, not milliseconds: this assertion used to pin the very
    # inconsistency it was meant to guard. Stage 6 was the only writer putting
    # milliseconds into experience_diary.decision_timestamp, a column every
    # other writer filled with seconds and which orders the Critic's "recent
    # decisions" query. See tests/unit/test_diary_timestamp_units.py.
    assert record.decision_timestamp == int(timestamp.timestamp())
    assert record.market_context["context_pattern_id"] == "pattern-1"
    assert record.market_context["context_velocity"] == pytest.approx(0.42)


def test_stage6_records_buy_as_pending_without_prediction_context():
    stage = _stage()
    transaction = {
        "timestamp": "2026-01-01T12:00:00+00:00",
        "type": "BUY",
        "ticker": "MSFT",
        "quantity": 3,
        "price": 50.0,
        "trade_value": 150.0,
        "reason": "New signal",
        "confidence": 0.91,
    }

    written = stage._record_transactions_to_diary([transaction], [])

    assert written == 1
    record = stage.diary_engine.records[0]
    assert record.agent_id == "stage6_execution"
    assert record.decision_type == DecisionType.BUY
    assert record.outcome == DecisionOutcome.PENDING
    assert record.entry_price == pytest.approx(50.0)
    assert record.exit_price is None
    assert record.context_fingerprint == "unknown_context"
    assert record.model_confidence == pytest.approx(0.91)


def test_context_rules_handle_string_velocity_and_list_predictions():
    stage = _stage()
    signals = [
        {
            "ticker": "AAPL",
            "context_velocity": "0.9",
            "confidence": "0.8",
            "predictions": [0.02],
        }
    ]

    result = stage._apply_context_rules(signals)

    assert result[0]["confidence"] == 0.0
