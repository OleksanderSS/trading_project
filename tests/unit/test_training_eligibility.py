"""
Rows a collector could not vouch for must not reach training.

put_call_ratio and cftc mark fabricated stand-in rows, and huggingface marks
rows with no usable time axis, with eligible_for_training=False. Nothing
outside those collectors ever read the flag, so the gate was decorative: the
rows were labelled and then loaded anyway.
"""
import pandas as pd

from src.data.management.eligibility import (
    ELIGIBILITY_COLUMN,
    filter_eligible_rows,
    has_eligibility_opinion,
)


def test_frame_without_the_column_is_returned_untouched():
    """
    Six of the eight enabled collectors never set the flag. Treating their
    silence as ineligibility would silently destroy the dataset.
    """
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert not has_eligibility_opinion(df)
    assert len(filter_eligible_rows(df, "market_data")) == 3


def test_explicit_false_rows_are_dropped():
    df = pd.DataFrame({"a": [1, 2, 3], ELIGIBILITY_COLUMN: [True, False, True]})
    assert filter_eligible_rows(df, "cftc_data")["a"].tolist() == [1, 3]


def test_null_means_unknown_and_the_row_survives():
    """Absence of a verdict is not a negative verdict."""
    df = pd.DataFrame({"a": [1, 2, 3], ELIGIBILITY_COLUMN: [True, None, False]})
    assert filter_eligible_rows(df, "huggingface_data")["a"].tolist() == [1, 2]


def test_all_ineligible_yields_an_empty_frame_rather_than_the_original():
    df = pd.DataFrame({"a": [1, 2], ELIGIBILITY_COLUMN: [False, False]})
    assert filter_eligible_rows(df, "put_call_ratio_data").empty


def test_index_and_columns_survive_filtering():
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], ELIGIBILITY_COLUMN: [True, False, True]},
        index=[10, 11, 12],
    )
    out = filter_eligible_rows(df, "t")
    assert out.index.tolist() == [10, 12]
    assert list(out.columns) == ["a", "b", ELIGIBILITY_COLUMN]


def test_empty_frame_is_handled():
    assert filter_eligible_rows(pd.DataFrame(), "t").empty


def test_huggingface_rows_without_a_time_axis_are_filtered_out():
    """End-to-end of the two halves: the collector marks, the loader drops."""
    from src.data.collectors.huggingface_collector import HuggingfaceCollector

    collector = object.__new__(HuggingfaceCollector)
    collector.configs = {}
    collector.dataset_name = "test/dataset"
    collector.hash_keys = ["content"]
    collector.logger = type(
        "L", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None,
                  "error": lambda *a, **k: None}
    )()

    timeless = pd.DataFrame({"content": ["a", "b"]})
    marked = collector._apply_time_axis(timeless, None)

    assert marked[ELIGIBILITY_COLUMN].tolist() == [False, False]
    assert filter_eligible_rows(marked, "huggingface_data").empty
