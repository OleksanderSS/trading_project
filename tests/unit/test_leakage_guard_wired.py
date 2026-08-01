"""Feature-vs-target leakage detection on the main Stage 3 path.

FeatureLeakageGuard existed and worked, but was reachable only from the
Colab/hybrid branch (colab_manager.py). A normal training run got no runtime
leakage check: the other guard, TemporalLeakageGuard, matches 0 of the 713
real feature names because its patterns describe a naming convention this
project does not use.

This one detects by CORRELATION against the actual targets, which does not
care what anything is called, plus the project's own is_target_like_column
rule.

One defect had to be fixed before wiring it in. The correlation check began
with

    sample_df = df[numeric_features + numeric_targets].dropna()

which drops a row if ANY of ~700 columns is null there. Measured on a frame
shaped like ours, a single sparse column -- a news or macro feature present
in 3% of rows, which we have -- cut 5,000 rows to 150. Two such columns empty
the frame, and the check then reports "clean" having compared nothing: the
same failure mode as the guard it is replacing. corrwith aligns pairwise, so
the dropna was both unnecessary and destructive.

Removing it needed a companion: a correlation from a handful of overlapping
points is noise, so a feature must share at least `min_overlap` rows with a
target before its correlation counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.validation.feature_leakage_guard import FeatureLeakageGuard
from src.pipeline.stages.feature_engineering.guards import FeatureGuards


def _frame(rows=600, features=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(rows, features)),
        columns=[f"FEATURE_{i}" for i in range(features)],
    )
    df["datetime"] = pd.date_range("2026-01-01", periods=rows, freq="h")
    df["ticker"] = "AAPL"
    df["target_up_1d"] = rng.normal(size=rows)
    return df


@pytest.fixture()
def guard(tmp_path):
    return FeatureLeakageGuard(block_on_forbidden=False, report_dir=str(tmp_path))


def test_a_feature_copied_from_the_target_is_caught(guard):
    df = _frame()
    df["LEAKY"] = df["target_up_1d"] * 1.001

    report = guard.check(df, ticker="AAPL")

    assert "LEAKY" in report.high_corr_cols


def test_ordinary_features_are_not_flagged(guard):
    assert not guard.check(_frame(), ticker="AAPL").high_corr_cols


def test_one_sparse_column_no_longer_blinds_the_check(guard):
    """The regression: dropna over every column let a single sparse feature
    delete the evidence."""
    df = _frame()
    sparse = np.full(len(df), np.nan)
    rng = np.random.default_rng(1)
    picked = rng.choice(len(df), size=int(len(df) * 0.03), replace=False)
    sparse[picked] = rng.normal(size=len(picked))
    df["news_sentiment_daily"] = sparse
    df["LEAKY"] = df["target_up_1d"] * 1.001

    assert len(df.dropna()) < 50, "fixture must reproduce the collapse"
    assert "LEAKY" in guard.check(df, ticker="AAPL").high_corr_cols


def test_two_sparse_columns_would_have_emptied_the_frame(guard):
    df = _frame()
    rng = np.random.default_rng(2)
    for name, share in (("news_a", 0.03), ("macro_b", 0.02)):
        column = np.full(len(df), np.nan)
        picked = rng.choice(len(df), size=int(len(df) * share), replace=False)
        column[picked] = rng.normal(size=len(picked))
        df[name] = column
    df["LEAKY"] = df["target_up_1d"] * 1.001

    assert df.dropna().empty, "fixture must reproduce the total collapse"
    assert "LEAKY" in guard.check(df, ticker="AAPL").high_corr_cols


def test_a_sparse_feature_is_not_condemned_on_a_few_points(guard):
    """A correlation from a handful of overlapping rows is not evidence."""
    df = _frame()
    column = np.full(len(df), np.nan)
    column[:5] = df["target_up_1d"].iloc[:5].to_numpy()  # perfect, on 5 rows
    df["rare_but_innocent"] = column

    assert "rare_but_innocent" not in guard.check(df, ticker="AAPL").high_corr_cols


def test_a_target_column_among_the_features_blocks(tmp_path):
    """The unambiguous case, and the only one that stops the pipeline."""
    strict = FeatureLeakageGuard(block_on_forbidden=True, report_dir=str(tmp_path))
    df = _frame()
    df["state_TARGET_RETURN_1P"] = df["target_up_1d"]

    with pytest.raises(ValueError, match="FORBIDDEN"):
        strict.check(df, feature_cols=["state_TARGET_RETURN_1P"], ticker="AAPL")


def test_a_frame_without_targets_says_it_checked_nothing(guard):
    """It used to return status 'clean' here -- the default -- so a report
    from the early return, where nothing is compared at all, read exactly
    like one that looked and found nothing."""
    df = _frame().drop(columns=["target_up_1d"])
    assert guard.check(df, ticker="AAPL").status == "not_checked"


def test_a_real_check_says_clean_only_after_comparing(guard):
    assert guard.check(_frame(), ticker="AAPL").status == "clean"


def test_stage3_now_runs_the_check():
    """The wiring itself: FeatureGuards must hold and use the guard."""
    guards = FeatureGuards(mode="prepare")
    assert hasattr(guards, "leakage_guard")

    import inspect
    source = inspect.getsource(FeatureGuards.apply_guards)
    assert "leakage_guard.check" in source


def test_stage3_passes_a_clean_frame_through():
    guards = FeatureGuards(mode="prepare")
    df = _frame()
    result = guards.apply_guards(df)

    assert len(result) == len(df)
    assert "target_up_1d" in result.columns
