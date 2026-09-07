"""
Absence of data must be reported as absence, not as a bad score.

A quality report that prints "0 features, 0.00% nulls, alignment ❌" for a
timeframe that produced no rows is making a claim about data that does not
exist. It reads as "measured, and bad", which is the one thing it cannot mean.
"""
import pandas as pd

from src.pipeline.stages.stage_3_improvements import (
    MEASURED,
    NO_EVIDENCE,
    calculate_data_quality_metrics,
)


def _frame(rows=3):
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "feature_a": range(rows),
            "feature_b": range(rows),
        }
    )


def _targets(rows=3):
    return pd.DataFrame({"target_up_1d": [0] * rows})


def test_empty_timeframe_reports_no_evidence_not_zero():
    metrics = calculate_data_quality_metrics(
        enriched_prices={"1d": pd.DataFrame()},
        all_targets={"1d": pd.DataFrame()},
        enrichers_count=5,
    )

    tf = metrics["timeframes"]["1d"]
    assert tf["status"] == NO_EVIDENCE
    # None, not 0 — a zero here would read as a measured result.
    assert tf["features_count"] is None
    assert tf["null_percentage"] is None
    assert tf["alignment_valid"] is None


def test_empty_timeframe_is_still_listed_rather_than_dropped():
    """It used to `continue`, so an empty timeframe vanished from the report."""
    metrics = calculate_data_quality_metrics(
        enriched_prices={"1d": pd.DataFrame(), "1h": _frame()},
        all_targets={"1d": pd.DataFrame(), "1h": _targets()},
        enrichers_count=5,
    )

    assert set(metrics["timeframes"]) == {"1d", "1h"}
    assert metrics["timeframes"]["1d"]["status"] == NO_EVIDENCE
    assert metrics["timeframes"]["1h"]["status"] == MEASURED


def test_overall_status_is_no_evidence_when_nothing_was_measured():
    metrics = calculate_data_quality_metrics(
        enriched_prices={"1d": pd.DataFrame()},
        all_targets={"1d": pd.DataFrame()},
        enrichers_count=5,
    )

    overall = metrics["overall"]
    assert overall["status"] == NO_EVIDENCE
    assert overall["timeframes_with_evidence"] == 0
    assert overall["total_features"] is None
    assert overall["avg_features_per_timeframe"] is None


def test_averages_are_taken_over_timeframes_that_had_data():
    """
    An empty timeframe used to divide the average down as if it had been
    measured and found to hold nothing.
    """
    metrics = calculate_data_quality_metrics(
        enriched_prices={"1d": _frame(), "1h": pd.DataFrame()},
        all_targets={"1d": _targets(), "1h": pd.DataFrame()},
        enrichers_count=5,
    )

    overall = metrics["overall"]
    assert overall["total_timeframes"] == 2
    assert overall["timeframes_with_evidence"] == 1
    # 3 feature columns measured in one timeframe -> average is 3, not 1.5.
    assert overall["avg_features_per_timeframe"] == overall["total_features"]


def test_missing_targets_are_no_evidence_not_a_misalignment_verdict():
    """
    alignment_valid used to be False when targets_df was empty. "Nothing to
    align against" and "aligned incorrectly" are different claims.
    """
    metrics = calculate_data_quality_metrics(
        enriched_prices={"1d": _frame()},
        all_targets={"1d": pd.DataFrame()},
        enrichers_count=5,
    )

    assert metrics["timeframes"]["1d"]["alignment_valid"] is None


def test_real_misalignment_is_still_reported_as_false():
    """Guard: the tri-state must not swallow a genuine mismatch."""
    metrics = calculate_data_quality_metrics(
        enriched_prices={"1d": _frame(rows=5)},
        all_targets={"1d": _targets(rows=3)},
        enrichers_count=5,
    )

    assert metrics["timeframes"]["1d"]["alignment_valid"] is False


def test_aligned_data_is_still_reported_as_true():
    metrics = calculate_data_quality_metrics(
        enriched_prices={"1d": _frame(rows=4)},
        all_targets={"1d": _targets(rows=4)},
        enrichers_count=5,
    )

    assert metrics["timeframes"]["1d"]["alignment_valid"] is True
