"""A collector that delivers nothing was counted as a collector that worked.

`process_and_save_results` kept one `successful` counter and incremented it
for three different outcomes: rows saved, `None` returned, and an empty frame
returned. The 2026-08-11 run had 16 collectors enabled and 10 delivering
rows; aaii_sentiment and put_call_ratio were answering HTTP 403, fear_greed
and wikimedia_attention were raising. The summary line said all of them were
processed successfully, because "returned no new data" and "returned data"
increment the same integer.

The 403s were visible -- inside each collector's own log, one line among
thousands. Nothing at the orchestrator level ever said "four of your sources
gave you nothing today", so four dead feeds could persist indefinitely while
every run reported success.

The second half of this is the error text. `f"Error in '{name}': {res}"`
printed

    Error in 'fear_greed':

with nothing after the colon, because `str()` on some exceptions is empty --
the same signature that made 54 drift timeouts indistinguishable from
crashes. An exception whose message is empty still has a type, and the type
is what makes it findable.
"""
import logging

import pandas as pd
import pytest

from src.pipeline.stages.collection.orchestrator import CollectionStage


class _Collector:
    def __init__(self, collector_type):
        self.collector_type = collector_type


@pytest.fixture
def stage():
    instance = object.__new__(CollectionStage)
    instance.logger = logging.getLogger("CollectionStageTest")
    return instance


def _run(stage, results, names):
    return stage.process_and_save_results(
        results, [_Collector(name) for name in names]
    )


def test_a_silent_collector_is_named_rather_than_counted_as_a_success(stage, caplog):
    with caplog.at_level(logging.WARNING, logger="CollectionStageTest"):
        _run(stage, [None, None], ["aaii_sentiment", "put_call_ratio"])

    warnings = "\n".join(
        record.message for record in caplog.records
        if record.levelno == logging.WARNING
    )
    assert "aaii_sentiment" in warnings
    assert "put_call_ratio" in warnings


def test_an_exception_with_an_empty_message_still_reports_its_type(stage, caplog):
    """`Error in 'fear_greed': ` names nothing. The type always does."""
    with caplog.at_level(logging.ERROR, logger="CollectionStageTest"):
        _run(stage, [KeyError()], ["fear_greed"])

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, "a failed collector must be logged as an error"
    assert "KeyError" in errors[0].message
    assert "fear_greed" in errors[0].message


def test_the_summary_counts_only_collectors_that_delivered_rows(stage, caplog):
    """The number in the summary has to mean rows arrived, or it means nothing."""
    with caplog.at_level(logging.INFO, logger="CollectionStageTest"):
        _run(stage, [None, RuntimeError("boom"), None], ["a", "b", "c"])

    summaries = [
        r.message for r in caplog.records if "collectors delivered rows" in r.message
    ]
    assert summaries, "every collection run must state how many sources delivered"
    assert "0/3" in summaries[0], (
        "none of the three delivered rows; a summary that says otherwise is the "
        "bug this pins"
    )


def test_an_empty_frame_is_silence_not_delivery(stage, caplog):
    """An empty DataFrame is what a rate-limited source returns."""
    with caplog.at_level(logging.INFO, logger="CollectionStageTest"):
        _run(stage, [pd.DataFrame()], ["rss"])

    summaries = [
        r.message for r in caplog.records if "collectors delivered rows" in r.message
    ]
    assert "0/1" in summaries[0]
