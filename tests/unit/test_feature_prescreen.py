"""Columns are ranked BEFORE they are imputed, and the ranking is the same one.

The pooled run of 2026-08-31 died with a MemoryError in the median imputer:
474 columns across 490,799 rows, all of them filled and scaled so that the
largest model budget could spend 35. `SimpleImputer(strategy='median')` sorts
through a masked array, which asked for a `(490799, 200)` int64 index -- 749
MiB for one block -- after eight hours of run time.

The fix is an ordering, not a new selector: rank first, impute what survives.
That is only safe if the pre-screen ranks by the SAME statistic the per-model
budget spends, which is what these tests pin. If the two ever diverge, the
pre-screen stops being a memory optimisation and starts being an undocumented
second feature selector.
"""
import numpy as np
import pandas as pd
import pytest

from src.config.feature_budget import (
    DEFAULT_MAX_FEATURES,
    get_preselection_ceiling,
)
from src.models.adapters.data_preparation import (
    _BlockImputer,
    _preselect_features,
    _target_correlation_ranking,
)
from sklearn.preprocessing import StandardScaler


def _budget_pick(frame, target, budget):
    """Exactly `BaseTrainer._select_features_for_model`'s ranking."""
    numeric = frame.select_dtypes(include=[np.number])
    series = pd.Series(np.asarray(target).ravel(), index=numeric.index).astype(float)
    ranked = (
        numeric.corrwith(series).abs()
        .fillna(-1.0)
        .sort_values(ascending=False, kind="mergesort")
    )
    return list(ranked.index[:budget])


def _prepared(frame, target):
    """What the models actually see: median-filled, then standardised."""
    imputer, scaler = _BlockImputer(), StandardScaler()
    values = scaler.fit_transform(imputer.fit_transform(frame))
    return pd.DataFrame(values, columns=imputer.get_feature_names_out(),
                        index=frame.index)


@pytest.fixture
def messy():
    """A frame with the shapes that actually break things."""
    rng = np.random.default_rng(7)
    rows, columns = 3000, 300
    frame = pd.DataFrame(
        rng.normal(size=(rows, columns)),
        columns=[f"f{i:03d}" for i in range(columns)],
    )
    target = pd.Series(
        ((frame["f005"] * 0.9 + frame["f200"] * 0.6
          + rng.normal(size=rows)) > 0).astype(int)
    )
    frame = frame.mask(rng.random((rows, columns)) < 0.15)
    frame["f100"] = np.nan          # no median exists
    frame["f101"] = 3.0             # no variance exists
    return frame, target


@pytest.mark.parametrize("budget", [5, 12, 35])
def test_the_prescreen_does_not_change_what_a_model_is_trained_on(messy, budget):
    """The top 35 of the top 70 are the top 35 -- so the models are unchanged.

    This is the whole safety argument for the reordering. It holds only
    because both steps rank by |Pearson r| against the training target, and
    standardisation is affine and positive, which leaves correlation alone.
    """
    frame, target = messy
    ceiling = get_preselection_ceiling()
    assert ceiling >= budget, "the ceiling must exceed every model budget"

    kept = _preselect_features(frame, target.to_numpy(), ceiling)

    everything = _budget_pick(_prepared(frame, target), target, budget)
    survivors = _budget_pick(_prepared(frame[kept], target), target, budget)
    assert survivors == everything


def test_columns_with_no_median_or_no_variance_sort_last(messy):
    """They carry no information; they must lose deterministically, not by luck."""
    frame, target = messy

    ranking = _target_correlation_ranking(frame, target.to_numpy())
    assert np.isnan(ranking["f100"])
    assert np.isnan(ranking["f101"])

    kept = _preselect_features(frame, target.to_numpy(), 70)
    assert "f100" not in kept and "f101" not in kept


def test_the_survivors_keep_the_frame_order(messy):
    """The downstream sort is STABLE, so equal scores are broken by order.

    Re-ordering here would break a tie differently and hand a model a
    different -- equally good, but different -- feature set, which is exactly
    the kind of silent divergence this reordering must not introduce.
    """
    frame, target = messy
    kept = _preselect_features(frame, target.to_numpy(), 70)
    assert kept == [column for column in frame.columns if column in set(kept)]


def test_nothing_happens_when_there_is_nothing_to_gain(messy):
    """A frame already inside the ceiling must not pay for a copy."""
    frame, target = messy
    assert _preselect_features(frame.iloc[:, :40], target.to_numpy(), 70) is None


def test_an_unrankable_target_keeps_every_column(messy):
    """Refusing to guess beats making alphabetical order the selector."""
    frame, _target = messy
    constant = np.zeros(len(frame))
    assert _preselect_features(frame, constant, 70) is None


def test_the_ceiling_is_derived_from_the_budgets_not_invented():
    """One number, read from the same config the budgets live in."""
    class _Config:
        def get(self, key, default=None):
            assert key == "models.per_model"
            return {"linear": {"max_features": 5},
                    "catboost": {"max_features": 35}}

    assert get_preselection_ceiling(_Config()) >= 35

    class _Broken:
        def get(self, key, default=None):
            raise KeyError(key)

    # A config that cannot be read must not produce a ceiling BELOW the
    # default budget, or the fallback would silently starve every model.
    assert get_preselection_ceiling(_Broken()) >= DEFAULT_MAX_FEATURES
