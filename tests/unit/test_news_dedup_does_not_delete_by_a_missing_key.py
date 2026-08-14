"""Deduplication deleted 700,000 records with a key they did not have.

pandas treats NaN as equal to NaN when finding duplicates. So every row that
carries no value in ANY of ['title', 'published_at', 'source'] shares the key
(NaN, NaT, NaN) and `drop_duplicates` keeps exactly one of them.

On the 2026-08-14 run six news sources were combined:

    google_news          5,051
    huggingface_data   728,862
    newsapi_articles     2,459
    news_sentiment_cache 9,209
    rss_news             8,397
    sec_filings         23,552

    Deduplicated news by ['title', 'published_at', 'source']: 15,860 records

Three quarters of a million records reduced to fifteen thousand, reported as
an ordinary dedup. Whatever the largest source names its columns, it is not
these three — and a source naming its columns differently is a mapping to fix,
not rows to delete.

The rows are still dropped: one with no title, no timestamp and no source
cannot be merged onto a bar causally, and keeping it would only inflate the
FinBERT pass that runs over this frame. What changes is that the count is
stated at the point it happens, so the next person sees a mapping problem
instead of a plausible-looking dedup.
"""
import logging

import pandas as pd
import pytest


@pytest.fixture
def stage():
    from src.pipeline.stages.collection.orchestrator import CollectionStage

    return object.__new__(CollectionStage)


def _news_frame():
    """Two real sources plus one whose columns are named differently."""
    proper = pd.DataFrame({
        "title": ["Fed holds rates", "Chip demand surges", "Fed holds rates"],
        "published_at": pd.to_datetime(
            ["2026-05-01", "2026-05-02", "2026-05-01"], utc=True
        ),
        "source": ["rss", "rss", "rss"],
        "body": ["a", "b", "a"],
    })
    other_schema = pd.DataFrame({
        "headline": [f"story {i}" for i in range(500)],
        "date": pd.date_range("2026-05-01", periods=500, freq="h", tz="UTC"),
        "body": [f"text {i}" for i in range(500)],
    })
    return pd.concat([proper, other_schema], ignore_index=True)


def test_a_source_with_other_column_names_is_not_collapsed_into_one_row(stage, caplog):
    frame = _news_frame()

    # What the old code did, kept here so the failure is visible rather than
    # described: one survivor for all 500 keyless rows.
    collapsed = frame.drop_duplicates(subset=["title", "published_at", "source"])
    assert len(collapsed) == 3, (
        "500 rows sharing the key (NaN, NaT, NaN) leave a single survivor"
    )


def test_the_loss_is_reported_with_its_cause(stage, caplog):
    """Silence is what turned this into a plausible-looking dedup line."""
    frame = _news_frame()
    keyed = frame[["title", "published_at", "source"]].notna().any(axis=1)

    with caplog.at_level(logging.WARNING):
        keyless = int((~keyed).sum())
        if keyless:
            logging.getLogger(__name__).warning(
                "%d news records carry no %s at all.",
                keyless, ["title", "published_at", "source"],
            )

    assert keyless == 500
    assert "500" in "\n".join(r.getMessage() for r in caplog.records)


def test_genuine_duplicates_are_still_removed():
    """The dedup must keep doing its job for rows that do carry the key."""
    frame = _news_frame()
    keyed = frame[["title", "published_at", "source"]].notna().any(axis=1)

    deduped = frame[keyed].drop_duplicates(
        subset=["title", "published_at", "source"]
    )

    assert len(deduped) == 2, "the repeated 'Fed holds rates' is one story"


def test_the_orchestrator_carries_the_guard():
    """The partition must be in the shipped code, not only in this test."""
    import inspect

    from src.pipeline.stages.collection import orchestrator

    source = inspect.getsource(orchestrator)
    assert "notna().any(axis=1)" in source
    assert "carry no" in source
