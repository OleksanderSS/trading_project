"""The winner's feature importances were extracted by nobody.

All 3,207 feature-stability artifacts written by the 2026-08-14 batch read:

    "feature_importance_status": "not_available_from_model"
    "feature_importance_count": 0

from RandomForest, CatBoost, LightGBM and XGBoost winners, every one of which
exposes importances. The artifact was honest — it declared the absence rather
than inventing a number — but the absence was manufactured one call earlier:
Stage 4 passed a literal `{}`.

`extract_native_feature_importance` unwraps the wrapper and reads
`feature_importances_`, `coef_` or `get_feature_importance`. It works and it
is covered by tests. Its only caller lives in
src/archive/dead_pipeline_code/modeling/training.py:120 — the pipeline this
one replaced. The replacement kept the artifact and dropped the call.

The columns are half the fix. A winner is fitted on its own budget (5 to 35 of
388 after per-model budgets came in), and `_importance_dict` returns {} on a
length mismatch — silently, and indistinguishably from a model that genuinely
has nothing to give. Handing it the full prepared frame would have restored
the call and changed the output not at all.
"""
import numpy as np
import pytest

from src.pipeline.stages.modeling import pipeline_control_artifacts
from src.training.base_trainer import BaseTrainer


class _Tree:
    """A wrapper of the shape src/models/tree/* produce."""

    def __init__(self, importances):
        self.model = type("Inner", (), {
            "feature_importances_": np.asarray(importances, dtype=float)
        })()


class _Linear:
    def __init__(self, coefs):
        self.model = type("Inner", (), {"coef_": np.asarray(coefs, dtype=float)})()


BUDGET = ["rsi_14", "volume_z", "macd", "atr_14", "nlp_sentiment_score"]


def test_a_tree_winners_importances_are_read():
    importance = BaseTrainer._winner_feature_importance(
        _Tree([0.4, 0.3, 0.2, 0.1, 0.0]), BUDGET
    )

    assert importance, "the batch that surfaced this reported 0 for 3,207 winners"
    assert set(importance) == set(BUDGET)
    assert importance["rsi_14"] > importance["macd"]


def test_a_linear_winners_coefficients_count_too():
    importance = BaseTrainer._winner_feature_importance(
        _Linear([-0.9, 0.1, 0.5, 0.0, 0.2]), BUDGET
    )

    assert importance
    assert importance["rsi_14"] == pytest.approx(max(importance.values())), (
        "a large negative coefficient is a strong feature, not a weak one"
    )


def test_the_full_frame_instead_of_the_fitted_columns_yields_nothing():
    """Why passing the wrong names is not a harmless near-miss.

    `_importance_dict` compares lengths and returns {} when they differ. The
    result is byte-identical to a model with no importances, so restoring the
    call while passing the prepared frame's 388 columns would have looked like
    a fix and changed nothing.
    """
    full_frame = BUDGET + [f"unused_{i}" for i in range(383)]

    assert BaseTrainer._winner_feature_importance(
        _Tree([0.4, 0.3, 0.2, 0.1, 0.0]), full_frame
    ) == {}


def test_a_model_with_nothing_to_give_reports_empty_not_invented():
    class _Opaque:
        pass

    assert BaseTrainer._winner_feature_importance(_Opaque(), BUDGET) == {}
    assert BaseTrainer._winner_feature_importance(None, BUDGET) == {}
    assert BaseTrainer._winner_feature_importance(_Tree([0.5, 0.5]), []) == {}


def test_the_artifact_reports_measured_once_the_importances_arrive():
    """The status field is derived from the dict, so the dict is the fix."""
    measured = pipeline_control_artifacts.build_feature_stability_candidate(
        ticker="AAPL", target_name="target_hourly_breakout_1h",
        model_type="random_forest", timeframe="1h",
        context_fingerprint="fp", market_regime="NORMAL", volatility_regime="low",
        feature_importance=BaseTrainer._winner_feature_importance(
            _Tree([0.4, 0.3, 0.2, 0.1, 0.0]), BUDGET
        ),
        stability_analysis={},
    )

    assert measured["feature_importance_status"] == "measured_from_trained_model"
    assert measured["feature_importance_count"] == len(BUDGET)
