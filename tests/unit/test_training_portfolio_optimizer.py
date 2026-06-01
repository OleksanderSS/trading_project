import numpy as np
import pandas as pd

from src.training.portfolio_optimizer import PortfolioOptimizer


def _optimizer():
    optimizer = object.__new__(PortfolioOptimizer)
    optimizer.logger = type(
        "Logger",
        (),
        {
            "info": lambda *args, **kwargs: None,
            "warning": lambda *args, **kwargs: None,
        },
    )()
    return optimizer


def test_global_portfolio_model_uses_row_aligned_returns_without_target_feature():
    optimizer = _optimizer()
    market_features = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 8 + ["MSFT"] * 8,
            "feature_a": np.linspace(0.0, 1.0, 16),
            "feature_b": np.linspace(1.0, 2.0, 16),
            "returns": np.linspace(-0.02, 0.03, 16),
        }
    )

    X, y = optimizer._prepare_global_training_frame(market_features)
    model_info = optimizer._train_global_model(market_features)
    prediction_frame = optimizer._select_model_features(
        market_features.head(3),
        model_info["feature_columns"],
    )

    assert "returns" not in X.columns
    assert "ticker" not in X.columns
    assert len(X) == len(y) == len(market_features)
    assert model_info["feature_columns"] == ["feature_a", "feature_b"]
    assert len(model_info["model"].predict(prediction_frame)) == 3


def test_training_portfolio_optimizer_split_is_chronological():
    optimizer = _optimizer()
    X = pd.DataFrame({"feature": range(10)})
    y = pd.Series(range(10))

    X_train, X_test, y_train, y_test = optimizer._chronological_split(X, y)

    assert X_train["feature"].tolist() == list(range(8))
    assert X_test["feature"].tolist() == [8, 9]
    assert y_train.tolist() == list(range(8))
    assert y_test.tolist() == [8, 9]
