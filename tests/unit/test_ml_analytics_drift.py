from datetime import datetime, timedelta

import pandas as pd

from src.monitoring.ml_analytics import MLAnalytics


class _DataManagerStub:
    def __init__(self, df):
        self.df = df

    def load_data(self, *args, **kwargs):
        return self.df.copy()


def test_model_drift_handles_single_historical_point_as_insufficient_variability():
    now = datetime.now()
    df = pd.DataFrame(
        {
            "accuracy": [0.8] + [0.7] * 9,
            "timestamp": [now - timedelta(days=30)] + [now - timedelta(days=1)] * 9,
        }
    )
    analyzer = object.__new__(MLAnalytics)
    analyzer.data_manager = _DataManagerStub(df)
    analyzer.logger = type("Logger", (), {"warning": lambda *args, **kwargs: None})()

    result = analyzer.check_model_drift("model-a", window_days=7)

    assert result["status"] == "insufficient_variability"


def test_problem_predictor_split_is_chronological():
    analyzer = object.__new__(MLAnalytics)
    X = pd.DataFrame({"feature": range(10)})
    y = pd.Series(range(10))

    X_train, X_test, y_train, y_test = analyzer._chronological_split(X, y)

    assert X_train["feature"].tolist() == list(range(8))
    assert X_test["feature"].tolist() == [8, 9]
    assert y_train.tolist() == list(range(8))
    assert y_test.tolist() == [8, 9]
