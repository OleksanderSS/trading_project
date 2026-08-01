"""Every writer of experience_diary.decision_timestamp must use one unit.

One BIGINT column had three writers in two units:

    DecisionRecord default      int(datetime.now(UTC).timestamp())        seconds
    log_training_event          int(datetime.now(UTC).timestamp())        seconds
    consensus metadata          int(pd.Timestamp.now().timestamp()*1000)  ms
    Stage 6 _transaction_timestamp   int(ts.timestamp() * 1000)           ms

The column drives `ORDER BY decision_timestamp DESC` (knn_context_finder,
get_recent_decisions) and is part of the upsert key
["agent_id", "decision_timestamp", "ticker"]. Mixing units means every
millisecond row (~1.7e12) sorts above every second row (~1.7e9) no matter
when either actually happened -- so "the most recent decisions" would be
whichever writer used milliseconds.

Latent until now only because the two millisecond writers had not run: all
19,305 rows in the live table are seconds. It would have fired the moment
paper trading started writing Stage 6 decisions.

Seconds was chosen so the existing rows need no migration.
"""
from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from src.meta_learning.memory import diary_engine
from src.meta_learning.memory.diary_engine import DecisionRecord, diary_timestamp

SECONDS_CEILING = 10_000_000_000  # ~year 2286 in seconds; anything above is ms


def _looks_like_seconds(value: int) -> bool:
    return 1_000_000_000 < value < SECONDS_CEILING


def _record() -> DecisionRecord:
    from src.meta_learning.memory.diary_engine import DecisionType
    return DecisionRecord(
        agent_id="a",
        ticker="AAPL",
        decision_type=DecisionType.BUY,
        reasoning="test",
        market_context={},
        context_fingerprint="ctx",
    )


def test_the_helper_returns_seconds():
    assert _looks_like_seconds(diary_timestamp())


def test_the_helper_round_trips_a_given_moment():
    moment = datetime(2026, 3, 14, 15, 9, 26, tzinfo=UTC)
    assert diary_timestamp(moment) == int(moment.timestamp())


def test_the_record_default_is_seconds():
    assert _looks_like_seconds(_record().decision_timestamp)


def test_no_writer_multiplies_by_a_thousand():
    """The regression itself: a `* 1000` next to a timestamp in this module."""
    source = inspect.getsource(diary_engine)
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    offenders = [
        line.strip() for line in code.splitlines()
        if "timestamp" in line and "* 1000" in line
    ]
    assert not offenders, offenders


class _Stage6Timestamps:
    """The Stage 6 conversion, lifted out of the orchestrator so this test
    does not have to build a pipeline stage with its collaborators."""

    def __init__(self):
        import logging
        self.logger = logging.getLogger("stage6-test")

    from src.pipeline.stages.trading.orchestrator import TradingExecutionStage
    _transaction_timestamp = TradingExecutionStage._transaction_timestamp


@pytest.fixture()
def stage():
    return _Stage6Timestamps()


def test_stage6_emits_seconds_for_a_datetime(stage):
    moment = datetime(2026, 3, 14, 15, 9, 26, tzinfo=UTC)
    assert stage._transaction_timestamp(moment) == int(moment.timestamp())


def test_stage6_emits_seconds_for_an_iso_string(stage):
    assert stage._transaction_timestamp("2026-03-14T15:09:26+00:00") == int(
        datetime(2026, 3, 14, 15, 9, 26, tzinfo=UTC).timestamp()
    )


def test_stage6_accepts_either_unit_on_the_way_in(stage):
    """Upstream transaction dicts are not all ours; a millisecond epoch must
    not be stored as if it were seconds."""
    moment = datetime(2026, 3, 14, 15, 9, 26, tzinfo=UTC)
    seconds = int(moment.timestamp())

    assert stage._transaction_timestamp(seconds) == seconds
    assert stage._transaction_timestamp(seconds * 1000) == seconds


def test_stage6_agrees_with_the_diary_default(stage):
    """Both writers land in the same table; they must land in the same range."""
    from_stage = stage._transaction_timestamp(datetime.now(UTC))
    from_record = _record().decision_timestamp

    assert abs(from_stage - from_record) < 5


def test_an_unparseable_timestamp_is_reported_not_quietly_replaced(stage, caplog):
    """Substituting "now" stamps a historical trade with the moment the
    pipeline happened to run, and the diary is what the Critic learns from."""
    import logging

    with caplog.at_level(logging.WARNING):
        result = stage._transaction_timestamp("not-a-date")

    assert _looks_like_seconds(result)
    assert any("could not be parsed" in r.getMessage() for r in caplog.records)


def test_a_missing_timestamp_falls_back_without_crying_wolf(stage, caplog):
    """None means "the transaction carried no time", which is ordinary; only
    an unusable VALUE deserves a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert _looks_like_seconds(stage._transaction_timestamp(None))

    assert not [r for r in caplog.records if "unusable type" in r.getMessage()]
