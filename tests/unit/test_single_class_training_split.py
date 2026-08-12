"""Six models trained on one class and reported 80% accuracy.

AAPL 60m target_hourly_breakout_1h: 11 positives in 278 rows, and the
split is chronological, so all 11 landed inside the final 20%. The
training portion was 223 rows of a single class.

Six of the seven trainers accepted it. A classifier fitted to one class
can only ever predict that class, and on a validation window that is
mostly the same class it scored 44/55 = 80% -- a respectable-looking
number for a model that learned nothing, and one that would then compete
for champion against models that did.

Only TabNet refused:

    Valid set -- {0, 1} -- contains unkown targets from training set

That crash was the single honest report in the group. The temptation was
to "fix TabNet"; the defect was the six that stayed quiet.

Scope, measured across the batch: 1 of 308 trainable classification
contexts (0.3%). Isolated, not systemic -- which is why it needed a guard
rather than a redesign of the split.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _controller():
    path = Path("scripts/colab/colab_clean_cell.py")
    if not path.exists():
        pytest.skip("colab trainer script not present")
    spec = importlib.util.spec_from_file_location("colab_clean_cell", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"colab trainer imports unavailable here: {exc}")
    return module.ColabTrainingController


def test_the_real_case_is_refused():
    """267 zeros then 11 ones -- the AAPL 60m breakout target's shape."""
    verdict = _controller()._classification_split_verdict(
        pd.Series([0.0] * 267 + [1.0] * 11)
    )

    assert verdict is not None
    assert "один клас" in verdict


def test_the_refusal_names_the_numbers():
    """A refusal that does not say how imbalanced is a refusal nobody can
    act on."""
    verdict = _controller()._classification_split_verdict(
        pd.Series([0.0] * 267 + [1.0] * 11)
    )

    assert "267" in verdict and "11" in verdict


def test_a_balanced_target_still_trains():
    assert _controller()._classification_split_verdict(
        pd.Series([0.0, 1.0] * 100)
    ) is None


def test_a_rare_but_early_class_still_trains():
    """Rarity alone is not the problem; the problem is a training portion
    with nothing to contrast. Where "too rare to be useful" begins is a
    modelling decision, not one to make silently inside a guard.
    """
    assert _controller()._classification_split_verdict(
        pd.Series([1.0] * 5 + [0.0] * 200)
    ) is None


def test_too_few_samples_is_reported_separately():
    verdict = _controller()._classification_split_verdict(
        pd.Series([0.0, 1.0] * 10)
    )

    assert verdict is not None
    assert "зразк" in verdict


def test_the_guard_uses_the_same_split_as_training():
    """A guard that draws its own boundary is a guard that will disagree
    with the thing it protects."""
    import inspect

    source = inspect.getsource(
        _controller()._classification_split_verdict.__func__
    )

    assert "_chronological_split" in source


def test_the_guard_runs_before_any_model_is_trained():
    """Once per target, not once per architecture: the condition is a
    property of the data and the split, not of any model."""
    import inspect

    controller = _controller()
    source = inspect.getsource(controller._process_target)

    guard = source.index("_classification_split_verdict")
    training = source.index("for model_type in heavy_models")
    assert guard < training


def test_the_batch_on_disk_does_not_grow_new_single_class_contexts():
    """Pins the measured scope as a ceiling. A rise means the split or the
    targets moved and this needs looking at again, not silencing.

    Measured 2026-08-12, after the batch was rebuilt on repaired bar dates:
    zero contexts, down from the one this file was written for. The case did
    not get fixed, it dissolved -- AAPL 60m target_hourly_breakout_1h now
    holds 895 labelled rows with 91 positives (10.2%), where it held 278 rows
    with 11 positives, all of which had landed inside the final 20%. With
    three times the history the positives no longer sit entirely in the
    validation window.

    The behaviour of the guard is pinned by the tests above, which build the
    pathological series directly and do not depend on what is on disk. This
    one watches the data, so it is an inequality: zero is the good outcome,
    and anything above the historical one is the signal.
    """
    features = Path("data/colab/accumulated/main_database/features.parquet")
    targets = Path("data/colab/accumulated/main_database/targets.parquet")
    if not (features.exists() and targets.exists()):
        pytest.skip("no prepared batch on disk")

    from src.config.target_type_registry import load_target_types

    controller = _controller()
    types = load_target_types()
    frame = pd.read_parquet(targets)
    classification = [
        c for c in frame.columns
        if c.startswith("target_") and "classification" in str(types.get(c, ""))
    ]

    refused = [
        (ticker, tf, col)
        for ticker in frame.ticker.unique()
        for tf, rows in frame[frame.ticker == ticker].groupby("interval")
        for col in classification
        if controller._classification_split_verdict(
            rows.sort_values("datetime")[col].dropna()
        ) and rows[col].notna().sum() >= controller._MIN_TRAINING_SAMPLES
    ]

    assert len(refused) <= 1, (
        f"single-class training contexts rose above the one historical case: "
        f"{refused}. The guard will refuse them, so nothing trains on one "
        f"class -- but a rise means the split or the target thresholds moved, "
        f"and that is worth understanding rather than accepting."
    )
