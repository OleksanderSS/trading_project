"""Nothing said which bars could have had news at all.

Every news flag in the frame answers "is there a signal on this bar", and
answers 0 both for a quiet hour inside our coverage and for a bar from before
we collected anything. Those are different facts and nothing separated them.

That gap had a price, recorded in collectors.yaml: hourly price history was
pinned to 180 days while Yahoo serves 730, because extending it would add ~17
months of bars whose 144 news and sentiment features are zero -- teaching the
models that the past was newsless, which is worse than not having the bars.

`news_coverage` closes it. The earliest story we hold is a property of our
collection rather than of the market, and it is historical, so using it is
metadata and not look-ahead. Measured on the 2026-08-15 batch, with the
earliest story at 2026-03-14:

    15m  100.0% covered   (66 days of bars, all inside the window)
    60m   83.9%           3,155 bars before it
    1d    20.4%           9,070 bars before it

Note what it is NOT: as things stand it equals news_quality_available exactly,
because merge_asof forward-fills and every covered bar therefore has a value.
That equality is a property of the merge, not of the meaning, and it will stop
holding the moment price history outruns news history -- which is the change
it exists to permit.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.news_quality_enricher import NewsQualityEnricher


@pytest.fixture
def enricher():
    return NewsQualityEnricher()


def test_bars_before_the_first_story_are_marked_uncovered(enricher):
    bars = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    news = pd.Series(pd.date_range("2026-01-06", periods=3, freq="D", tz="UTC"))

    flag = enricher._coverage_flag(bars, news)

    assert flag.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]


def test_a_quiet_day_inside_the_window_still_counts_as_covered(enricher):
    """The distinction the whole feature exists to make."""
    bars = pd.date_range("2026-01-10", periods=5, freq="D", tz="UTC")
    # One story at the start, then silence.
    news = pd.Series([pd.Timestamp("2026-01-10", tz="UTC")])

    flag = enricher._coverage_flag(bars, news)

    assert flag.tolist() == [1, 1, 1, 1, 1], (
        "silence inside the collected window is an observation, not a gap"
    )


def test_the_earliest_story_sets_the_boundary_not_the_nearest(enricher):
    bars = pd.date_range("2026-01-05", periods=3, freq="D", tz="UTC")
    news = pd.Series(pd.to_datetime(
        ["2026-01-01", "2026-06-01"], utc=True
    ))

    assert enricher._coverage_flag(bars, news).tolist() == [1, 1, 1]


def test_no_usable_timestamps_means_nothing_is_covered(enricher):
    bars = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")

    flag = enricher._coverage_flag(bars, pd.Series([pd.NaT, pd.NaT]))

    assert flag.tolist() == [0, 0, 0]


def test_a_naive_news_stamp_does_not_break_a_tz_aware_index(enricher):
    """The frames disagree on tz often enough that this is not hypothetical."""
    bars = pd.date_range("2026-01-05", periods=3, freq="D", tz="UTC")
    news = pd.Series(pd.to_datetime(["2026-01-01"]))  # tz-naive

    assert enricher._coverage_flag(bars, news).tolist() == [1, 1, 1]


def test_the_column_reaches_the_enriched_frame(enricher):
    bars = pd.DataFrame({
        "ticker": ["AAPL"] * 12,
        "datetime": pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC"),
        "close": np.linspace(100, 110, 12),
    })
    news = pd.DataFrame({
        "published_at": pd.date_range("2026-01-07", periods=5, freq="D", tz="UTC"),
        "title": ["a story about markets"] * 5,
        "source": ["rss"] * 5,
    })

    enriched = enricher._enrich_impl(bars, news=news)

    assert "news_coverage" in enriched.columns
    coverage = pd.to_numeric(enriched["news_coverage"], errors="coerce")
    assert coverage.nunique() == 2, "both eras are present in this fixture"
