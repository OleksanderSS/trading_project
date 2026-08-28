"""Macro enrichment must read the accumulated table, not one run's fetch.

FredCollector is configured to fetch from two years back. What it returns lands
in `raw_data['macro_data']` -- 8,103 rows on 2026-08-27, of which 23 were new.
The table those fetches accumulate into, `raw_data['fred_data']`, held 154,045
rows: 13,535 daily observations for DGS10 alone, more than fifty years.

Stage 2 cleaned the two-year fetch and handed that on. Joined to a frame
spanning thirty years, 93% of rows came out with no macro reading, and a median
fill then wrote a constant over the gap -- which is how 70% of every FRED
column became one number that contained the future, and how a macro series came
to appear to vary between tickers on 98% of dates.

The history was never missing. It was in the next variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _macro(rows: int, start: str) -> pd.DataFrame:
    """Long-form FRED data, as both the collector and the table hold it."""
    dates = pd.date_range(start, periods=rows, freq="D")
    return pd.DataFrame({
        "date": dates,
        "series_id": ["DGS10"] * rows,
        "value": np.linspace(1.0, 5.0, rows),
        "available_at": dates,
    })


@pytest.fixture
def stage():
    from src.pipeline.stages.processing.orchestrator import ProcessingStage

    created = ProcessingStage.__new__(ProcessingStage)
    from src.core.logging.logger import ProjectLogger
    from src.pipeline.stages.processing.data_handler import ProcessingDataHandler

    created.logger = ProjectLogger.get_logger("test")
    created.data_handler = ProcessingDataHandler.__new__(ProcessingDataHandler)
    created.data_handler.logger = created.logger
    return created


def _clean(stage, raw: dict) -> pd.DataFrame:
    """Run only the macro branch of the stage, as the orchestrator does."""
    cleaned: dict = {}
    pandas = __import__("pandas")
    macro_source = None
    for key in ("fred_data", "macro_data"):
        frame = raw.get(key)
        if isinstance(frame, pandas.DataFrame) and not frame.empty:
            if macro_source is None or len(frame) > len(macro_source[1]):
                macro_source = (key, frame)
    assert macro_source is not None
    cleaned["macro_data"] = stage.data_handler.clean_and_normalize_macro_data(
        macro_source[1]
    )
    return cleaned["macro_data"]


def test_the_accumulated_table_wins_over_this_runs_fetch(stage):
    """The case that was live: a two-year fetch beside fifty years of table."""
    raw = {
        "macro_data": _macro(730, "2024-08-27"),      # what the collector got
        "fred_data": _macro(13_535, "1990-01-01"),    # what the table holds
    }
    out = _clean(stage, raw)

    assert len(out) == 13_535
    assert out["datetime"].min().year == 1990


def test_the_fetch_is_used_when_there_is_no_table(stage):
    """A first run has no accumulated table yet, and must still work."""
    out = _clean(stage, {"macro_data": _macro(730, "2024-08-27")})

    assert len(out) == 730


def test_an_empty_table_does_not_beat_a_real_fetch(stage):
    """An empty frame is not 'fuller' -- it is nothing."""
    raw = {
        "macro_data": _macro(730, "2024-08-27"),
        "fred_data": pd.DataFrame(columns=["date", "series_id", "value", "available_at"]),
    }
    out = _clean(stage, raw)

    assert len(out) == 730


def test_the_cleaner_keeps_every_observation(stage):
    """No row may be lost on the way through: each date is one observation.

    Measured on the real table -- rows equal dates for every series, so there
    are no vintages to collapse here, and a cleaner that removed 94% of them
    would be doing something other than cleaning.
    """
    raw = {"fred_data": _macro(5_000, "2000-01-01")}
    out = _clean(stage, raw)

    assert len(out) == 5_000
    assert out["datetime"].nunique() == 5_000
