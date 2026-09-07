"""
HuggingFace collector must record the absence of a time axis, not hide it.

The configured dataset (m-ric/financial-news-2024, table huggingface_data) was
persisted with hash_keys=[content] and no time handling at all, so rows landed
in storage with no usable time. Rows like that cannot be aligned to a price
series, split chronologically, or checked for lookahead — the table is not
incomplete, it is unusable and unverifiable.
"""
import pandas as pd

from src.data.collectors.huggingface_collector import HuggingfaceCollector


def _collector(**configs):
    """A collector with only the attributes the time-axis helpers touch."""
    c = object.__new__(HuggingfaceCollector)
    c.configs = configs
    c.dataset_name = configs.get("dataset_name", "test/dataset")
    c.hash_keys = configs.get("hash_keys", ["content"])
    c.logger = _QuietLogger()
    return c


class _QuietLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


def test_missing_time_axis_marks_rows_ineligible_instead_of_storing_them_as_usable():
    c = _collector()
    df = pd.DataFrame({"content": ["a", "b"]})

    assert c._resolve_time_column(df) is None
    out = c._apply_time_axis(df, None)

    assert out["eligible_for_training"].tolist() == [False, False]
    assert out["timestamp"].isna().all()


def test_time_column_is_detected_and_normalised():
    c = _collector()
    df = pd.DataFrame({"content": ["a", "b"], "published_at": ["2024-01-01", "2024-01-02"]})

    time_column = c._resolve_time_column(df)
    assert time_column == "published_at"

    out = c._apply_time_axis(df, time_column)
    assert out["eligible_for_training"].tolist() == [True, True]
    assert str(out["timestamp"].dtype).startswith("datetime64")


def test_rows_with_unparseable_time_are_marked_ineligible_individually():
    c = _collector()
    df = pd.DataFrame({"content": ["a", "b"], "date": ["2024-01-01", "not a date"]})

    out = c._apply_time_axis(df, c._resolve_time_column(df))

    assert out["eligible_for_training"].tolist() == [True, False]
    assert pd.isna(out["timestamp"].iloc[1])


def test_configured_time_column_that_is_absent_is_reported_not_guessed():
    """A configured column that the dataset does not have is an error, not a cue to fall back."""
    c = _collector(time_column="event_time")
    df = pd.DataFrame({"content": ["a"], "published_at": ["2024-01-01"]})

    assert c._resolve_time_column(df) is None


def test_hash_keys_always_include_time_so_dedup_does_not_collapse_distinct_times():
    """
    hash_keys=[content] alone folds the same text observed at two different
    times into one row. Time must take part in the identity of a record.
    """
    c = _collector(hash_keys=["content"])
    df = pd.DataFrame({"content": ["a"], "published_at": ["2024-01-01"]})

    keys = c._effective_hash_keys(df, "published_at")

    assert "published_at" in keys
    assert "content" in keys


def test_same_text_at_different_times_hashes_differently():
    c = _collector(hash_keys=["content"])
    df = pd.DataFrame(
        {"content": ["same story", "same story"],
         "published_at": ["2024-01-01", "2024-06-01"]}
    )
    keys = c._effective_hash_keys(df, "published_at")

    joined = df[keys].astype(str).agg("|".join, axis=1)
    assert joined.iloc[0] != joined.iloc[1]


def test_unknown_hash_keys_fall_back_to_all_columns_rather_than_raising():
    c = _collector(hash_keys=["nonexistent"])
    df = pd.DataFrame({"content": ["a"], "published_at": ["2024-01-01"]})

    keys = c._effective_hash_keys(df, "published_at")

    assert "content" in keys and "published_at" in keys
