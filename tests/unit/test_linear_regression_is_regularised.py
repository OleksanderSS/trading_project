"""`linear` was unregularised OLS, and it won 135 of 354 champion slots.

Measured on the 2026-08-12 batch, 22 daily contexts, target_return_1d,
holdout R2 (median across contexts):

    OLS,        35 features   -7.35
    RidgeCV,    35 features   -0.85
    RidgeCV,    10 features   -0.25
    ElasticNet, 35 features   -0.14
    baseline (predict the training mean)  -0.01

Through the factory, with the pipeline's own scaler in front, RidgeCV lands
at -0.089 against OLS's -7.35. A model at -7.35 is not a weak model; it is
worse than a constant by a factor of hundreds, produced by fitting 35
correlated features to ~306 rows of financial noise.

The asymmetry this closes: the classification branch has always been
regularised, because sklearn's LogisticRegression applies L2 by default.
Only regression targets were exposed, which is a large part of why every
return target lost to its baseline.

What this does NOT do is make returns predictable. Even ElasticNet stays
below the baseline on the median. It stops the model from fabricating fits,
which is a different and smaller claim.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV

from src.models.linear.linear_model import LinearModel


def test_regression_uses_a_regularised_estimator():
    model = LinearModel(task_type="regression")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(rng.normal(size=60))

    model.train(X, y)

    assert isinstance(model.model, RidgeCV), (
        "plain LinearRegression here scored a median holdout R2 of -7.35 on "
        "real return targets; the estimator must carry a penalty"
    )
    assert model.model.alpha_ > 0


def test_classification_keeps_the_balanced_logistic_model():
    """That branch was never the problem and must not move."""
    model = LinearModel(task_type="classification")
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    y = pd.Series([0, 1] * 30)

    model.train(X, y)

    assert isinstance(model.model, LogisticRegression)
    assert model.model.class_weight == "balanced"


def test_it_does_not_fit_noise_worse_than_a_constant():
    """The shape of the real failure: many correlated features, few rows.

    With pure noise there is nothing to learn, so the honest outcome is a
    holdout R2 near zero -- what predicting the training mean would score.
    Unregularised OLS went sharply negative here, which is what -7.35 looked
    like in production.
    """
    rng = np.random.default_rng(7)
    n_train, n_hold, n_features = 120, 60, 35
    base = rng.normal(size=(n_train + n_hold, 5))
    # Correlated columns, as a feature budget of 35 technical indicators is.
    blocks = [base + rng.normal(scale=0.1, size=base.shape) for _ in range(7)]
    X = pd.DataFrame(
        np.hstack(blocks)[:, :n_features],
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(rng.normal(scale=0.01, size=len(X)))

    model = LinearModel(task_type="regression")
    model.train(X.iloc[:n_train], y.iloc[:n_train])
    predicted = model.predict(X.iloc[n_train:])

    actual = y.iloc[n_train:].to_numpy()
    ss_total = ((actual - actual.mean()) ** 2).sum()
    r_squared = 1 - ((actual - predicted) ** 2).sum() / ss_total

    assert r_squared > -1.0, (
        f"holdout R2 {r_squared:.2f} on pure noise: the model is inventing a "
        f"fit rather than shrinking toward the mean"
    )
