import pandas as pd

from src.archive.meta_learning.real_time_learning import RealTimeLearning


def test_calculate_sharpe_ratio_returns_zero_for_constant_returns():
    learner = object.__new__(RealTimeLearning)

    assert learner._calculate_sharpe_ratio(pd.Series([1.0, 1.0, 1.0])) == 0.0


def test_calculate_sharpe_ratio_returns_zero_for_all_nan_returns():
    learner = object.__new__(RealTimeLearning)

    assert learner._calculate_sharpe_ratio(pd.Series([None, None])) == 0.0
