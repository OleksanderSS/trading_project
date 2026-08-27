"""A feature that is constant while a model learns cannot be learned.

Measured on the 110-ticker batch of 2026-08-27, split at 2018-11-19: 65
features are constant across the training window and only start varying after
it -- 14 sentiment, 12 FRED, 9 news, 15 state, plus filings, nlp, fear/greed
and keywords. News exists only for 2026, at 69.8% coverage that year and zero
for 2015 through 2025, and the macro series do not reach back either.

That is worse than a useless column. A model gives no weight to something that
never moved, and then the column starts moving in the holdout -- a
distribution shift arriving exactly when the feature finally means something.

The project has met this shape three times by hand now: attention collected 30
days deep against a frame spanning decades, then news, then macro. The macro
case stayed hidden for months because a median fill wrote a plausible number
over the gap, so the fabrication concealed the shallowness. These tests exist
so the fourth time is announced by the pipeline instead of found by someone
reading a report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.feature_engineering.guards import FeatureGuards


@pytest.fixture
def guards() -> FeatureGuards:
    return FeatureGuards(mode="prepare")


def _frame(**columns) -> pd.DataFrame:
    rows = len(next(iter(columns.values())))
    return pd.DataFrame({
        "datetime": pd.date_range("2015-01-01", periods=rows, freq="D"),
        **columns,
    })


def test_a_source_collected_only_recently_is_named(guards):
    """The real case: zero for the first years, real values at the end."""
    rows = 100
    sentiment = np.zeros(rows)
    sentiment[90:] = np.linspace(-1, 1, 10)     # arrives only at the very end

    dead = guards.report_features_dead_in_training(_frame(sentiment=sentiment))

    assert dead == ["sentiment"]


def test_a_feature_that_varies_throughout_is_left_alone(guards):
    rng = np.random.default_rng(0)
    dead = guards.report_features_dead_in_training(
        _frame(close=rng.standard_normal(100))
    )
    assert dead == []


def test_a_column_constant_everywhere_is_not_reported_here(guards):
    """A column that never varies at all is a different problem.

    `hour` on a daily frame is legitimately zero throughout, and the empty and
    constant filters already remove such columns. This check is specifically
    about the ones that come alive after training, because those are the ones
    that look fine.
    """
    dead = guards.report_features_dead_in_training(_frame(hour=np.zeros(100)))
    assert dead == []


def test_missing_values_do_not_count_as_variation(guards):
    """NaN in training is absence, not a second value.

    After the median fill was removed, shallow sources read as NaN in the early
    years rather than as a fabricated constant -- so the check has to see
    through NaN or it would stop noticing exactly the case that motivated it.
    """
    rows = 100
    values = np.full(rows, np.nan)
    values[:70] = 0.0                # training: one value and a lot of nothing
    values[70:] = np.linspace(1, 5, 30)

    assert guards.report_features_dead_in_training(_frame(macro=values)) == ["macro"]


def test_a_frame_with_no_timestamps_is_skipped_quietly(guards):
    """Without time there is no training window, so there is nothing to say."""
    frame = pd.DataFrame({"sentiment": np.zeros(10)})
    assert guards.report_features_dead_in_training(frame) == []


def test_text_columns_are_ignored(guards):
    dead = guards.report_features_dead_in_training(
        _frame(ticker=["AAPL"] * 90 + ["MSFT"] * 10)
    )
    assert dead == []


def test_it_runs_inside_apply_guards(guards):
    """Reported where the features are produced, not in a report a week later."""
    rows = 100
    sentiment = np.zeros(rows)
    sentiment[90:] = 1.0
    frame = _frame(sentiment=sentiment, ticker=["AAPL"] * rows)

    out = guards.apply_guards(frame)

    # The guard reports; it must not quietly drop the column.
    assert "sentiment" in out.columns
    assert len(out) == rows
