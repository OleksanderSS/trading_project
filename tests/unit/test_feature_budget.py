"""A model gets a feature budget, and the budget comes from one place.

Three copies of this number existed and the configured one lost. Measured
across 4,613 heavy-model artifacts before the fix: cnn trained on exactly 64
features, lstm/gru/transformer/autoencoder on 128, mlp/tabnet up to 256 —
matching a hardcoded map in the Colab script on all seven types and
`models.yaml` on none. The light branch had no budget at all: a median of 388
features against roughly 308 training rows.
"""
import numpy as np
import pandas as pd
import pytest

from src.config.feature_budget import DEFAULT_MAX_FEATURES, get_model_max_features
from src.training.base_trainer import BaseTrainer


class _StubConfig:
    def __init__(self, values=None):
        self._values = values or {}

    def get(self, key, default=None):
        return self._values.get(key, default)


class _Host(BaseTrainer):
    def __init__(self, config=None):
        from src.metrics.model.ml_evaluator import MLEvaluator
        self.evaluator = MLEvaluator()
        self.config_manager = config or _StubConfig()
        self.logger = _NullLogger()

    def train(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def _prepare_ticker_groups(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def _train_ticker_group(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


class _NullLogger:
    def warning(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def isEnabledFor(self, _level):
        return False


def test_budget_comes_from_config():
    config = _StubConfig({'models.per_model.lightgbm.max_features': 12})
    assert get_model_max_features('lightgbm', config) == 12


def test_missing_or_nonsense_config_falls_back_rather_than_going_unlimited():
    assert get_model_max_features('brand_new_model', _StubConfig()) == DEFAULT_MAX_FEATURES
    assert get_model_max_features('x', _StubConfig({'models.per_model.x.max_features': 0})) == DEFAULT_MAX_FEATURES
    assert get_model_max_features('', _StubConfig()) == DEFAULT_MAX_FEATURES


def _data(n_rows=200, n_features=100, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = rng.normal(size=n_rows)
    # Make three columns genuinely informative.
    for i, weight in ((7, 3.0), (23, 2.0), (61, 1.5)):
        X[f"f{i}"] = y * weight + rng.normal(scale=0.1, size=n_rows)
    return {'X_train': X, 'y_train': y}


def test_selection_respects_the_budget_and_keeps_the_informative_columns():
    host = _Host(_StubConfig({'models.per_model.lightgbm.max_features': 5}))
    data = _data()

    chosen = host._select_features_for_model('lightgbm', data, is_classif=False)

    assert len(chosen) == 5
    # The planted signal must survive a cut from 100 columns to 5.
    for name in ("f7", "f23", "f61"):
        assert name in chosen


def test_selection_uses_training_rows_only():
    """Validation and holdout must not influence which columns exist."""
    host = _Host(_StubConfig({'models.per_model.linear.max_features': 4}))
    data = _data()
    # A holdout whose "signal" is in different columns entirely.
    data['X_holdout'] = data['X_train'].copy()
    data['y_holdout'] = -data['y_train']
    data['X_val'] = data['X_train'].copy()
    data['y_val'] = data['y_train'][::-1].copy()

    chosen = host._select_features_for_model('linear', data, is_classif=False)
    baseline = _Host(_StubConfig({'models.per_model.linear.max_features': 4}))._select_features_for_model(
        'linear', _data(), is_classif=False
    )

    assert chosen == baseline


def test_a_frame_already_within_budget_is_untouched():
    host = _Host(_StubConfig({'models.per_model.linear.max_features': 500}))
    data = _data(n_features=10)

    chosen = host._select_features_for_model('linear', data, is_classif=False)

    assert chosen == list(data['X_train'].columns)


def test_projection_only_keeps_requested_columns():
    frame = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

    assert list(BaseTrainer._project(frame, ['a', 'c']).columns) == ['a', 'c']
    # Unknown names are ignored rather than raising.
    assert list(BaseTrainer._project(frame, ['a', 'zzz']).columns) == ['a']
    # Nothing requested, or nothing addressable: pass through.
    assert BaseTrainer._project(frame, None) is frame
    assert BaseTrainer._project(np.zeros((2, 2)), ['a']) is not None


class _ColumnStrictModel:
    """Refuses a frame whose columns differ from the fitted ones, as sklearn does."""

    def __init__(self):
        self.fitted_columns = None

    def train(self, X, y):
        self.fitted_columns = list(X.columns)

    def predict(self, X):
        if list(X.columns) != self.fitted_columns:
            raise ValueError(
                "The feature names should match those that were passed during fit."
            )
        return np.zeros(len(X))


def test_the_holdout_is_scored_on_the_columns_the_winner_was_fitted_on():
    """The regression that killed the first budgeted run.

    _record_winner_test_score predicted on the full holdout frame while the
    model had been fitted on 35 of 388 columns, so sklearn refused it and the
    modelling stage died with "Critical error in stage 'ModelingStage'".
    """
    host = _Host(_StubConfig({'models.per_model.linear.max_features': 5}))
    data = _data(n_rows=120, n_features=60)
    columns = host._select_features_for_model('linear', data, is_classif=False)

    model = _ColumnStrictModel()
    model.train(BaseTrainer._project(data['X_train'], columns), data['y_train'])

    data['X_holdout'] = data['X_train'].copy()
    data['y_holdout'] = data['y_train'].copy()
    results = {}

    host._record_winner_test_score(model, data, False, results, columns=columns)

    assert results['winner_holdout_metrics']['status'] == 'measured'


def test_the_winner_columns_reach_the_results_for_stage5():
    host = _Host(_StubConfig({'models.per_model.linear.max_features': 4}))
    data = _data(n_rows=120, n_features=40)
    columns = host._select_features_for_model('linear', data, is_classif=False)

    assert columns is not None and len(columns) == 4
    # Stage 5 rebuilds its input frame from exactly this list.
    assert set(columns) <= set(data['X_train'].columns)


def test_constant_columns_lose_to_informative_ones():
    host = _Host(_StubConfig({'models.per_model.linear.max_features': 3}))
    data = _data(n_features=20)
    data['X_train']['dead'] = 1.0  # zero variance -> correlation is NaN

    chosen = host._select_features_for_model('linear', data, is_classif=False)

    assert 'dead' not in chosen
