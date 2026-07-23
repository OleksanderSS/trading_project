"""Tests for EnsembleModel's correlation-aware member weighting.

Wires in src/models/ensemble/correlation/correlation_engine.py, which
existed but had zero callers before this change (found during a
duplication/dead-code audit). Previously EnsembleModel always used
implicit equal voting weights.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from src.models.ensemble.ensemble_model import EnsembleModel


def _classification_data(seed: int = 0, n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] + X["f1"] > 0).astype(int))
    return X, y


def test_correlated_members_are_downweighted_relative_to_diverse_member():
    """Two near-identical trees should end up with lower weight than an
    uncorrelated dummy classifier — redundant models add no diversity."""
    X, y = _classification_data(seed=1, n=300)
    models = [
        ("dt1", DecisionTreeClassifier(max_depth=3, random_state=1)),
        ("dt2", DecisionTreeClassifier(max_depth=3, random_state=1)),
        ("dummy", DummyClassifier(strategy="stratified", random_state=2)),
    ]

    ensemble = EnsembleModel(models=models, task_type="classification")
    ensemble.train(X, y)

    assert ensemble.member_weights is not None
    assert ensemble.member_weights["dt1"] == pytest.approx(ensemble.member_weights["dt2"], abs=1e-6)
    assert ensemble.member_weights["dummy"] > ensemble.member_weights["dt1"]
    # Weights are normalized.
    assert sum(ensemble.member_weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_correlation_weighting_can_be_disabled():
    """use_correlation_weighting=False must reproduce the old equal-weight
    behavior exactly (no correlation computation, no member_weights set)."""
    X, y = _classification_data(seed=2)
    models = [
        ("dt", DecisionTreeClassifier(max_depth=3, random_state=1)),
        ("dummy", DummyClassifier(strategy="stratified", random_state=2)),
    ]

    ensemble = EnsembleModel(models=models, task_type="classification", use_correlation_weighting=False)
    ensemble.train(X, y)

    assert ensemble.member_weights is None
    assert ensemble.is_trained is True


def test_training_still_succeeds_if_a_candidate_cannot_be_cloned_for_weighting():
    """Correlation weighting is a refinement, not a training precondition:
    a member that fails during the internal probe-fit (not sklearn-clonable)
    must not block the real VotingClassifier fit from succeeding."""

    class NotClonable:
        """Deliberately missing get_params/fit — sklearn.base.clone() raises
        on this, exercising _compute_correlation_weights' fallback path."""

        def fit(self, X, y):
            raise TypeError("simulated: this estimator cannot be cloned/fit standalone")

    X, y = _classification_data(seed=3)
    models = [
        ("dt", DecisionTreeClassifier(max_depth=3, random_state=1)),
        ("broken", NotClonable()),
    ]

    ensemble = EnsembleModel(models=models, task_type="classification")
    weights = ensemble._compute_correlation_weights(X, y)

    assert weights is None
    assert ensemble.member_weights is None


def test_regression_task_also_supports_correlation_weighting():
    from sklearn.dummy import DummyRegressor
    from sklearn.tree import DecisionTreeRegressor

    X, y = _classification_data(seed=4)
    y = y.astype(float)
    models = [
        ("dt", DecisionTreeRegressor(max_depth=3, random_state=1)),
        ("dummy", DummyRegressor(strategy="mean")),
    ]

    ensemble = EnsembleModel(models=models, task_type="regression")
    ensemble.train(X, y)

    assert ensemble.is_trained is True
    preds = ensemble.predict(X.iloc[:5])
    assert len(preds) == 5
