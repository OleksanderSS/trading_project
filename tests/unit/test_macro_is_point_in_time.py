"""A bar saw July's inflation on the first of July.

The macro pivot keyed on the observation date -- the period a figure
describes -- and `merge_asof(direction='backward')` then attached it to every
bar from that date onward. CPI for July 2025 is filed under 2025-07-01 and
was published on 2025-08-12, so 44 macro columns were visible six weeks
before they existed.

Walk-forward validation cannot catch this. The leak is identical in every
fold, so it raises the score everywhere and looks like signal.

`available_at` was already derived in Stage 1 and validated in Stage 2 -- it
was simply not the column the pivot used. And it could not have helped on
its own: FRED, asked without realtime parameters, returns the current
revision of every observation stamped with the request date. 7,939 stored
rows carried 21 distinct realtime_start values, all within days of the last
collection. The check passed on a value that did not mean what the check
assumed.

Asked properly, the API gives the truth:

    date=2025-07-01  realtime_start=2025-08-12  value=322.132   first release
    date=2025-07-01  realtime_start=2026-02-13  value=322.169   revision

Both facts are needed: the first so a figure is not visible before it
existed, the second so a bar sees the number that was current then rather
than today's restatement.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.macro_features_enricher import MacroFeaturesEnricher


OBSERVED = pd.Timestamp("2025-07-01")
PUBLISHED = pd.Timestamp("2025-08-12")
REVISED = pd.Timestamp("2026-02-13")


@pytest.fixture
def enricher():
    return MacroFeaturesEnricher()


def _macro():
    return pd.DataFrame({
        "series_id": ["CPIAUCSL"] * 3,
        "datetime": pd.to_datetime(["2025-06-01", "2025-07-01", "2025-07-01"]),
        "available_at": pd.to_datetime(["2025-07-15", PUBLISHED, REVISED]),
        "realtime_start": pd.to_datetime(["2025-07-15", PUBLISHED, REVISED]),
        "value": [321.500, 322.132, 322.169],
    })


def test_the_pivot_is_keyed_on_publication_not_on_the_period(enricher):
    pivoted = enricher._pivot_macro_data(_macro())

    assert list(pivoted.index) == [
        pd.Timestamp("2025-07-15"), PUBLISHED, REVISED
    ], "the index must be when each figure became known"
    assert OBSERVED not in pivoted.index, (
        "1 July is the month CPI describes, not a date on which anything "
        "was published"
    )


def test_a_figure_is_invisible_before_it_is_published(enricher):
    pivoted = enricher._pivot_macro_data(_macro())
    column = pivoted.columns[0]

    # What a bar on 1 August could know: June's number, published 15 July.
    known_on_first_august = pivoted.loc[pivoted.index <= pd.Timestamp("2025-08-01"), column]
    assert known_on_first_august.iloc[-1] == pytest.approx(321.500)
    assert 322.132 not in set(known_on_first_august), (
        "July's CPI reached bars six weeks before its release"
    )


def test_a_bar_after_publication_sees_the_figure_current_then(enricher):
    pivoted = enricher._pivot_macro_data(_macro())
    column = pivoted.columns[0]

    on_20_august = pivoted.loc[pivoted.index <= pd.Timestamp("2025-08-20"), column].iloc[-1]
    assert on_20_august == pytest.approx(322.132), "the first release, not the revision"

    on_1_march = pivoted.loc[pivoted.index <= pd.Timestamp("2026-03-01"), column].iloc[-1]
    assert on_1_march == pytest.approx(322.169), "after February, the revised value"


def test_falling_back_to_the_observation_date_is_announced(enricher, caplog):
    """Old data has no availability column. Using it must not be silent."""
    import logging

    without_availability = _macro().drop(columns=["available_at", "realtime_start"])

    with caplog.at_level(logging.WARNING):
        enricher._pivot_macro_data(without_availability)

    warnings = "\n".join(r.message for r in caplog.records)
    assert "not when it was published" in warnings


def test_the_collector_requests_every_vintage():
    """Without these parameters FRED stamps every row with the request date."""
    import inspect

    from src.data.collectors.fred_collector import FredCollector

    source = inspect.getsource(FredCollector._fetch_series)
    assert '"realtime_start": "1776-07-04"' in source
    assert '"realtime_end": "9999-12-31"' in source


def test_macro_values_arrive_as_numbers(enricher):
    """FRED sends strings, and "." for a print it does not have.

    Pivoted unconverted they stay object columns, and the macro cache write
    failed outright on one:

        Could not convert '2.85' with type str: tried to convert to double
        ... column FRED_BAMLH0A0HYM2 with type object

    Anything that is not a number is a missing observation, not a category.
    """
    macro = pd.DataFrame({
        "series_id": ["BAMLH0A0HYM2"] * 3,
        "datetime": pd.to_datetime(["2025-06-01", "2025-07-01", "2025-08-01"]),
        "available_at": pd.to_datetime(["2025-06-02", "2025-07-02", "2025-08-02"]),
        "realtime_start": pd.to_datetime(["2025-06-02", "2025-07-02", "2025-08-02"]),
        "value": ["2.85", ".", "3.10"],
    })

    pivoted = enricher._pivot_macro_data(macro)

    column = pivoted["FRED_BAMLH0A0HYM2"]
    assert str(column.dtype).startswith("float"), (
        "an object column here is what broke the cache write"
    )
    assert column.tolist() == pytest.approx([2.85, 3.10])
    # The "." row leaves no entry at all, which is right: nothing was
    # published on that date, so there is no publication event to record.
    assert pd.Timestamp("2025-07-02") not in pivoted.index


def test_the_pivot_can_be_written_to_parquet(enricher, tmp_path):
    """The failure that surfaced this: object columns break the cache write."""
    macro = pd.DataFrame({
        "series_id": ["CPIAUCSL"] * 2,
        "datetime": pd.to_datetime(["2025-06-01", "2025-07-01"]),
        "available_at": pd.to_datetime(["2025-07-15", "2025-08-12"]),
        "realtime_start": pd.to_datetime(["2025-07-15", "2025-08-12"]),
        "value": ["321.500", "322.132"],
    })

    target = tmp_path / "macro.parquet"
    enricher._pivot_macro_data(macro).to_parquet(target)

    assert target.exists()
