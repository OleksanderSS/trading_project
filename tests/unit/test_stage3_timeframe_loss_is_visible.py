"""A whole timeframe must not leave Stage 3 without a word.

The 2026-08-04 batch recorded timeframes ['15m', '1d', '1h'] and delivered
1d and 60m. features.parquet had no 15m rows, targets.parquet had no 15m
column, and 0 of 506 champions were 15m -- although targets.yaml declares
target_intraday_up_15m, collectors.yaml requests 15m at period 60d, and the
collector was observed querying Yahoo for '15m' with a valid window.

Two places in Stage 3 could drop a timeframe in silence:

- _validate_and_prepare_market_data skipped an empty frame with a bare
  `continue`;
- _combine_timeframes filtered with `df['interval'] == tf`, a raw string
  comparison one spelling away from selecting nothing. This project writes
  '1h' in config and '60m' in data, and that pair has already cost it a
  rolling-window budget lookup and a macro availability column.

Neither is proven to be where 15m went -- the prepare-run log has rotated
away. Both now say so, which is what the next run needs.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.pipeline.stages.feature_engineering.orchestrator import (
    FeatureEngineeringStage,
)


@pytest.fixture()
def stage():
    instance = object.__new__(FeatureEngineeringStage)
    instance.logger = logging.getLogger("stage3-timeframe-test")
    return instance


def _frame(interval, rows=3):
    return pd.DataFrame({
        "ticker": ["AAPL"] * rows,
        "interval": [interval] * rows,
        "datetime": pd.date_range("2026-08-01", periods=rows, freq="D"),
        "close": [1.0] * rows,
    })


def test_an_empty_timeframe_is_announced(stage, caplog):
    with caplog.at_level(logging.ERROR):
        stage._validate_and_prepare_market_data(
            cleaned_data={"prices": {
                "15m": pd.DataFrame(),
                "1d": _frame("1d"),
            }}
        )

    assert any("15m" in str(record.args) for record in caplog.records)
    assert any("dropped" in record.message for record in caplog.records)


def test_a_delivered_timeframe_survives(stage):
    _, market = stage._validate_and_prepare_market_data(
        cleaned_data={"prices": {"1d": _frame("1d", rows=5)}}
    )

    assert "1d" in market


def _rows(combined):
    """Rows across every timeframe.

    `_combine_timeframes` used to hand back ONE concatenated frame, so these
    tests said `len(combined)`. It hands back a dict of frames per timeframe
    since 2026-08-26, and `len` of that counts TIMEFRAMES -- one, always. The
    assertions passed the wrong number to compare and would have gone green on
    an empty frame.
    """
    return sum(len(frame) for frame in combined.values())


def test_the_interval_filter_compares_normalised_names(stage):
    """'1h' and '60m' are the same timeframe. A raw == selects nothing."""
    stage.timeframe_context_assembler = _PassthroughAssembler()

    combined = stage._combine_timeframes({"60m": _frame("1h", rows=4)})

    assert _rows(combined) == 4, "1h rows were not recognised as 60m"


def test_a_timeframe_whose_rows_all_mismatch_is_announced(stage, caplog):
    stage.timeframe_context_assembler = _PassthroughAssembler()

    with caplog.at_level(logging.ERROR):
        stage._combine_timeframes({"15m": _frame("1d", rows=4)})

    assert any("none whose interval matches" in r.message for r in caplog.records)


def test_matching_rows_are_kept(stage):
    stage.timeframe_context_assembler = _PassthroughAssembler()

    combined = stage._combine_timeframes({"1d": _frame("1d", rows=6)})

    assert _rows(combined) == 6


class _PassthroughAssembler:
    """Returns the frames unchanged; these tests are about the filter.

    It carried only `assemble`, which returned ONE concatenated frame. Stage 3
    stopped calling that on 2026-08-26: the union of timeframes is ~80% NaN by
    construction and came to about 11 GiB at 110 tickers, so the orchestrator
    now calls `assemble_by_timeframe` and keeps the frames separate. The stub
    was never updated, so all three tests here died on an AttributeError
    inside the orchestrator and stopped watching the interval filter -- which
    is the only thing they were ever about.
    """

    def assemble_by_timeframe(self, filtered_data):
        frames = {name: frame for name, frame in filtered_data.items()
                  if not frame.empty}
        return frames, {"summary": {
            "base_context_count": len(frames),
            "output_rows": sum(len(frame) for frame in frames.values()),
        }}
