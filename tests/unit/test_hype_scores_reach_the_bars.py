"""An enricher counted 15,274 articles and added nothing.

Every run logged, on every timeframe:

    HypeEnricher - Calculating hype scores using window: 1h from 15274 news items
    HypeEnricher - Found time column 'published_at' with 15274 valid timestamps
    FeatureOrchestrator - Enricher 'hype_features' completed: +0 columns in 0.17s

Three defects, each on its own sufficient to produce that:

1. `_normalize_news_count_datetime` opened with
   `if 'datetime' not in news_count.columns: return news_count` -- returning
   early in exactly the case its rename existed to serve. The aggregation
   emits [ticker, published_at, news_count], so 'datetime' was never there,
   the rename never ran, and the merge asked for a column that did not
   exist. The rename was wrong anyway: `columns[0]` is 'ticker'.

2. `_merge_hype_scores` returned the frame untouched when 'datetime' was not
   a column -- and earlier enrichers leave it as the index.

3. The tz strip only ran on a 'datetime' column, so an index kept its tz and
   merge_asof refused to join tz-aware bars to tz-naive windows, raising
   into a handler that returned the frame unchanged.

And the window itself: `pd.Grouper(freq='1h')` labels a bucket with its
start, so counts from articles up to 14:59 would have reached a bar at
14:00. Same correction as the sentiment and keyword paths.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.hype_enricher import HypeEnricher


@pytest.fixture
def enricher():
    return HypeEnricher()


def _bars():
    return pd.DataFrame({
        "ticker": ["AAPL"] * 12,
        "datetime": pd.date_range("2026-07-01 13:00", periods=12,
                                  freq="15min", tz="UTC"),
        "close": np.linspace(100, 101, 12),
    })


def _news():
    return pd.DataFrame({
        "ticker": ["AAPL"] * 5,
        "published_at": pd.to_datetime([
            "2026-07-01 13:05", "2026-07-01 13:20",
            "2026-07-01 14:05", "2026-07-01 14:30",
            "2026-07-01 15:05",
        ], utc=True),
        "title": ["x"] * 5,
    })


@pytest.mark.parametrize("as_index", [False, True])
def test_the_score_actually_reaches_the_frame(enricher, as_index):
    """Both placements: earlier enrichers leave datetime as the index."""
    bars = _bars()
    frame = bars.set_index("datetime") if as_index else bars

    enriched = enricher._enrich_impl(frame.copy(), news=_news())

    added = {c for c in enriched.columns if c not in frame.columns}
    assert {"hype_score", "hype_available"} <= added, (
        f"enricher added {added or 'nothing'} — this is the +0 columns case"
    )


def test_the_index_is_handed_back_the_way_it_arrived(enricher):
    frame = _bars().set_index("datetime")

    enriched = enricher._enrich_impl(frame.copy(), news=_news())

    assert isinstance(enriched.index, pd.DatetimeIndex)


def test_a_bar_never_counts_articles_published_after_it(enricher):
    enriched = enricher._enrich_impl(_bars(), news=_news())

    stamps = pd.to_datetime(enriched["datetime"], utc=True)
    scores = pd.to_numeric(enriched["hype_score"], errors="coerce")

    # The first article is at 13:05; nothing before its window closes at
    # 14:00 may carry a count.
    before = scores[stamps < pd.Timestamp("2026-07-01 14:00", tz="UTC")]
    assert (before == 0).all(), "bars counted articles from their own hour"


def test_the_count_arrives_once_the_window_closes(enricher):
    enriched = enricher._enrich_impl(_bars(), news=_news())

    stamps = pd.to_datetime(enriched["datetime"], utc=True)
    scores = pd.to_numeric(enriched["hype_score"], errors="coerce")
    at_14 = scores[stamps == pd.Timestamp("2026-07-01 14:00", tz="UTC")]

    assert float(at_14.iloc[0]) == 2.0, (
        "two articles fell in the 13:00-13:59 window"
    )


def test_intermediate_bars_inherit_the_last_closed_window(enricher):
    """An exact merge on the hour left 14:15 and 14:30 with nothing."""
    enriched = enricher._enrich_impl(_bars(), news=_news())

    stamps = pd.to_datetime(enriched["datetime"], utc=True)
    scores = pd.to_numeric(enriched["hype_score"], errors="coerce")
    quarter_past = scores[stamps == pd.Timestamp("2026-07-01 14:15", tz="UTC")]

    assert float(quarter_past.iloc[0]) == 2.0
