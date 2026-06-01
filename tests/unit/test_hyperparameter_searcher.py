import numpy as np

from src.optimization.hyperparameter_searcher import HyperparameterSearcher


def make_searcher():
    searcher = HyperparameterSearcher.__new__(HyperparameterSearcher)
    searcher.random_seed = 42
    return searcher


def test_mlp_param_evaluation_uses_validation_data():
    searcher = make_searcher()
    x_train = np.arange(20, dtype=float).reshape(-1, 1)
    y_train = 2.0 * x_train.reshape(-1) + 1.0
    x_val = np.arange(20, 25, dtype=float).reshape(-1, 1)
    y_val = 2.0 * x_val.reshape(-1) + 1.0
    params = {
        "hidden_size_1": 8,
        "hidden_size_2": 4,
        "learning_rate": 0.01,
        "dropout": 0.0,
        "batch_size": 8,
        "epochs": 20,
    }

    score = searcher._evaluate_mlp_params(
        x_train, y_train, x_val, y_val, params,
        metric_func=lambda y_true, pred: -float(np.mean((y_true - pred) ** 2)),
    )

    assert np.isfinite(score)
    assert score < 0.0


def test_lstm_param_evaluation_is_deterministic_for_same_data():
    searcher = make_searcher()
    x_train = np.arange(30, dtype=float).reshape(-1, 1)
    y_train = np.sin(x_train.reshape(-1) / 10.0)
    x_val = np.arange(30, 35, dtype=float).reshape(-1, 1)
    y_val = np.sin(x_val.reshape(-1) / 10.0)
    params = {
        "hidden_size": 8,
        "n_layers": 1,
        "learning_rate": 0.01,
        "dropout": 0.0,
        "batch_size": 8,
        "epochs": 20,
    }

    score_one = searcher._evaluate_lstm_params(x_train, y_train, x_val, y_val, params)
    score_two = searcher._evaluate_lstm_params(x_train, y_train, x_val, y_val, params)

    assert np.isfinite(score_one)
    assert score_one == score_two
