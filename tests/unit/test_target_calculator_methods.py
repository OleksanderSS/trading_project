"""Forward-window target methods: shape contract and semantics.

The shape tests exist because of a real regression: `_calculate_by_method`
originally assembled its per-ticker result with `groupby().apply()`, which
concatenates Series when there are several groups but treats a Series as a
ROW when there is exactly one -- returning a 1xN DataFrame that then
reindexed into an NxN square. Single-ticker frames are the real call path,
since TargetOrchestrator._process_by_ticker_groups splits by ticker before
calling the calculator, so the pipeline broke while a multi-ticker ad-hoc
check passed.
"""
import numpy as np
import pandas as pd
import pytest

from src.targets.calculators.classification_calculator import ClassificationCalculator
from src.targets.calculators.regression_calculator import RegressionCalculator


def _frame(tickers=("AAA",), n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in tickers:
        close = 100 + np.cumsum(rng.normal(0, 1, n))
        rows.append(pd.DataFrame({
            "ticker": t,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="D"),
            "close": close,
            "high": close + rng.uniform(0.1, 1.0, n),
            "low": close - rng.uniform(0.1, 1.0, n),
            "volume": rng.uniform(1e6, 5e6, n),
        }))
    return pd.concat(rows, ignore_index=True)


METHOD_CASES = [
    ("slope_strength", 20),
    ("rate_of_change", 10),
    ("high_low_range", 3),
]


@pytest.mark.parametrize("method,window", METHOD_CASES)
@pytest.mark.parametrize("tickers", [("AAA",), ("AAA", "BBB", "CCC")])
def test_method_returns_one_dimensional_series(method, window, tickers):
    df = _frame(tickers)
    out = RegressionCalculator().calculate(
        df, base_col="close", shift=-1, method=method, window=window
    )
    assert isinstance(out, pd.Series)
    assert out.ndim == 1
    assert len(out) == len(df)
    assert out.index.equals(df.index)


@pytest.mark.parametrize("method,window", METHOD_CASES)
def test_single_ticker_result_fits_in_a_dataframe(method, window):
    """pd.DataFrame(targets_dict) is what TargetOrchestrator builds."""
    df = _frame(("AAA",))
    series = RegressionCalculator().calculate(
        df, base_col="close", shift=-1, method=method, window=window
    )
    built = pd.DataFrame({"target_x": series})
    assert built.shape == (len(df), 1)


def test_slope_strength_is_bounded_zero_to_one():
    df = _frame(("AAA", "BBB"))
    out = RegressionCalculator().calculate(
        df, base_col="close", shift=-1, method="slope_strength", window=20
    ).dropna()
    assert not out.empty
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_high_low_range_is_never_negative():
    df = _frame(("AAA",))
    out = RegressionCalculator().calculate(
        df, base_col="close", shift=-1, method="high_low_range", window=3
    ).dropna()
    assert not out.empty
    assert (out >= 0).all()


def test_methods_are_not_all_the_same_signal():
    """The bug that started this: method/window were ignored, so
    slope_strength and rate_of_change both collapsed to a plain return."""
    df = _frame(("AAA",))
    calc = RegressionCalculator()
    trend = calc.calculate(df, base_col="close", shift=-1, method="slope_strength", window=20)
    momentum = calc.calculate(df, base_col="close", shift=-1, method="rate_of_change", window=10)
    plain = calc.calculate(df, base_col="close", shift=-1)
    assert not trend.equals(momentum)
    assert not trend.equals(plain)
    assert not momentum.equals(plain)


def test_unknown_method_is_rejected_loudly():
    with pytest.raises(ValueError, match="Unknown regression target method"):
        RegressionCalculator().calculate(
            _frame(), base_col="close", shift=-1, method="not_a_real_method"
        )


def test_compare_to_average_reads_threshold_as_a_multiple():
    """`threshold: 2.0` with compare_to must mean 2x the average, not +200%."""
    df = _frame(("AAA",), n=80)
    # Force one unmistakable spike four bars ahead of index 40.
    df.loc[44, "volume"] = df["volume"].iloc[:44].mean() * 5

    out = ClassificationCalculator().calculate_binary(
        df, base_col="volume", shift=-4, threshold=2.0,
        compare_to="average", window=20,
    )
    assert out.iloc[40] == 1.0
    assert out.ndim == 1


def test_indicator_col_builds_a_crossing_target_not_a_direction_target():
    df = _frame(("AAA",), n=80)
    df["BAND"] = df["close"].rolling(20, min_periods=20).mean() + 5

    calc = ClassificationCalculator()
    crossing = calc.calculate_binary(
        df, base_col="close", shift=-4, threshold=0.0, indicator_col="BAND"
    )
    direction = calc.calculate_binary(df, base_col="close", shift=-4, threshold=0.0)
    assert not crossing.equals(direction)
    assert crossing.ndim == 1


def test_missing_indicator_column_raises_instead_of_returning_nans():
    with pytest.raises(ValueError, match="Indicator column"):
        ClassificationCalculator().calculate_binary(
            _frame(), base_col="close", shift=-4, threshold=0.0,
            indicator_col="DOES_NOT_EXIST",
        )
