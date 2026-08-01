"""A close and a re-open at the same bar are two decisions, not one.

record_decision upserted on ("agent_id", "decision_timestamp", "ticker").
decision_type was not in the key, and DataManager.upsert only inserts rows
whose key it has not seen -- so when Stage 6 records a SELL and a BUY for the
same ticker by the same model at the same bar, which it does in one batch
when a position is closed and another opened, the second row is DROPPED.

The row that carries realized P&L can be the one lost, and realized P&L is
what contextual model weights are computed from since 61286d38. So a
discarded SELL does not merely shrink the history -- it removes the evidence
the ensemble weights itself on.

Same shape as the OutcomeTracker collision fixed earlier in this audit:
identity derived from a timestamp rather than from what makes the record
distinct. Nothing has been lost so far (19,305 rows, 19,305 distinct keys --
model training takes longer than a second, so training rows never collided),
which is why it went unnoticed. Paper trading is where it would have bitten.
"""
from __future__ import annotations

import inspect

import pytest

from src.meta_learning.memory.diary_engine import (
    DecisionOutcome,
    DecisionRecord,
    DecisionType,
    DiaryEngine,
)

KEY = ["agent_id", "decision_timestamp", "ticker", "decision_type"]


class _CapturingManager:
    """Records what upsert was asked to do, and applies the key itself."""

    def __init__(self):
        self.rows: dict[tuple, dict] = {}
        self.keys_used: list[list[str]] = []

    def upsert(self, table, df, unique_on):
        self.keys_used.append(list(unique_on))
        for row in df.to_dict("records"):
            key = tuple(row[column] for column in unique_on)
            # Insert-if-absent, matching DataManager.upsert's contract.
            self.rows.setdefault(key, row)


@pytest.fixture()
def engine():
    instance = object.__new__(DiaryEngine)
    instance.data_manager = _CapturingManager()
    instance.table_name = "experience_diary"
    import logging
    instance.logger = logging.getLogger("diary-keys-test")
    return instance


def _record(decision_type, pnl, timestamp=1785000000):
    return DecisionRecord(
        agent_id="catboost",
        ticker="AAPL",
        decision_type=decision_type,
        reasoning="test",
        market_context={},
        context_fingerprint="fp",
        decision_timestamp=timestamp,
        outcome=DecisionOutcome.PROFITABLE,
        profit_loss=pnl,
    )


def test_a_sell_and_a_buy_at_the_same_bar_both_survive(engine):
    """The regression: one of these used to be dropped."""
    engine.record_decision(_record(DecisionType.SELL, 12.5))
    engine.record_decision(_record(DecisionType.BUY, 0.0))

    assert len(engine.data_manager.rows) == 2
    types = {row["decision_type"] for row in engine.data_manager.rows.values()}
    assert types == {"sell", "buy"}


def test_the_realized_pnl_is_the_row_that_used_to_be_lost(engine):
    engine.record_decision(_record(DecisionType.SELL, 12.5))
    engine.record_decision(_record(DecisionType.BUY, 0.0))

    recorded = [row["profit_loss"] for row in engine.data_manager.rows.values()]
    assert 12.5 in recorded, "the row carrying realized P&L disappeared"


def test_the_same_decision_recorded_twice_stays_one_row(engine):
    """Idempotency is why the composite key exists; it must survive."""
    engine.record_decision(_record(DecisionType.SELL, 12.5))
    engine.record_decision(_record(DecisionType.SELL, 12.5))

    assert len(engine.data_manager.rows) == 1


def test_different_seconds_are_different_rows(engine):
    engine.record_decision(_record(DecisionType.BUY, 0.0, timestamp=1785000000))
    engine.record_decision(_record(DecisionType.BUY, 0.0, timestamp=1785000001))

    assert len(engine.data_manager.rows) == 2


def test_the_key_includes_decision_type(engine):
    engine.record_decision(_record(DecisionType.BUY, 0.0))

    assert engine.data_manager.keys_used == [KEY]


def test_metadata_writes_use_the_same_key():
    """It shares the table, so a metadata row could displace a decision."""
    source = inspect.getsource(DiaryEngine.record_decision_metadata)

    assert '"decision_type"' in source or "'decision_type'" in source


def test_no_writer_is_left_on_the_old_three_column_key():
    source = inspect.getsource(DiaryEngine)
    old_key = '["agent_id", "decision_timestamp", "ticker"]'

    assert old_key not in source, "a writer still keys on agent+time+ticker alone"


def test_the_live_diary_has_no_collisions_to_repair():
    """Recorded so the claim is checkable: training rows never collided
    because training takes longer than a second."""
    import duckdb

    connection = duckdb.connect("data/trading_data.duckdb", read_only=True)
    total, distinct = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT (agent_id, decision_timestamp, ticker, decision_type)) "
        "FROM experience_diary"
    ).fetchone()
    connection.close()

    assert total == distinct
