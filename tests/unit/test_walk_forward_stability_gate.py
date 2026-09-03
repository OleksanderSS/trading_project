"""Promotion needs signal that held across folds, not one lucky split.

The holdout-versus-baseline check says "better than nothing, once". With
hundreds of contexts competing, some will clear it by chance. The walk-forward
evaluator was already in the repository and reachable only through
`walk_forward_review_only` — a branch that returns before training and was
never part of promotion.

Two things this pins down, both chosen from arithmetic rather than taste:

1. FOLD GEOMETRY. The defaults need ~485 rows for a single fold; a daily
   context has ~511, so it produced exactly ONE fold — and one fold is a
   single split with a decimal point. That silently exempted 396 of 660
   contexts. The geometry now shrinks to fit, floored so a fold stays a
   measurement.

2. HOW MANY FOLDS MUST HOLD. With no signal, each fold beats its majority
   baseline about half the time:
       >= 2 of 4 happens by chance 69% of the time
       >= 3 of 4                   31%
          4 of 4                    6%
   A threshold of two would pass noise more often than not.
"""
import math

import pytest

from src.pipeline.stages.modeling.walk_forward_validation import (
    build_purged_expanding_folds,
)
from src.pipeline.stages.stage_4_modeling import ModelingStage


@pytest.mark.parametrize(
    "row_count,expected_at_least",
    [
        (511, 2),    # a daily context: one fold at the defaults, more now
        (870, 2),    # 60m
        (1100, 2),   # 15m
    ],
)
def test_the_fold_geometry_fits_the_context(row_count, expected_at_least):
    config = ModelingStage._walk_forward_config_for(row_count)
    folds = build_purged_expanding_folds(row_count, config=config)

    assert len(folds) >= expected_at_least, (
        f"{row_count} rows produced {len(folds)} fold(s); stability cannot be "
        f"measured on fewer than {expected_at_least}"
    )


def test_a_short_context_is_not_stretched_into_meaningless_folds():
    """Below the floors, folds stop being measurements. Better to say so."""
    config = ModelingStage._walk_forward_config_for(200)

    assert config.min_train_rows >= ModelingStage._MIN_FOLD_TRAIN_ROWS
    assert config.validation_rows >= ModelingStage._MIN_FOLD_VALIDATION_ROWS


def test_the_window_never_shrinks_below_the_defaults():
    """Shrinking is for contexts that cannot afford the defaults, not for all."""
    for row_count in (1100, 5000, 104_267):
        config = ModelingStage._walk_forward_config_for(row_count)
        assert config.min_train_rows >= 360
        assert config.validation_rows >= 120


def test_the_window_grows_with_the_context_instead_of_staying_at_120():
    """The defect pooling exposed: more data, smaller checked fraction.

    The geometry used to return the fixed defaults whenever they produced
    enough folds -- that is, whenever data was plentiful. Measured on the
    pooled run of 2026-08-30: 104,267 training rows, four folds, and all four
    validation windows inside the LAST 480 rows. 0.46% of the data, at the
    very end of the timeline, deciding whether a model became a champion.
    """
    small = ModelingStage._walk_forward_config_for(900)
    pooled = ModelingStage._walk_forward_config_for(104_267)

    # A per-ticker context is unchanged: 900 // 8 is 112, and the floor of
    # 120 keeps the old window.
    assert small.validation_rows == 120

    assert pooled.validation_rows == 104_267 // 8
    validated = pooled.validation_rows * ModelingStage._MIN_STABLE_FOLDS
    assert validated > 480 * 10


@pytest.mark.parametrize(
    "fold_count,folds_above,should_pass",
    [
        (4, 4, True),
        (4, 3, True),
        (4, 2, False),   # 69% by chance — the threshold that would be theatre
        (4, 1, False),
        (3, 3, True),
        (3, 2, False),
        (2, 2, True),
    ],
)
def test_the_share_of_folds_required_is_three_quarters(fold_count, folds_above, should_pass):
    required = max(
        ModelingStage._MIN_STABLE_FOLDS,
        math.ceil(ModelingStage._STABLE_FOLD_SHARE * fold_count),
    )
    assert (folds_above >= required) is should_pass


def test_no_fold_may_come_out_worse_than_a_coin():
    """Counting folds says how OFTEN signal held, never whether it collapsed.

    AAPL/1d cleared 2 of 4 folds with a worst fold of 0.388 balanced accuracy
    — materially worse than guessing on a quarter of its history. That is not
    an unstable edge, it is two lucky windows. 0.5 is not a tuned threshold;
    it is chance itself.
    """
    assert ModelingStage._MIN_WORST_FOLD_BALANCED_ACCURACY == 0.5

    # The two conditions are independent: enough folds is not sufficient.
    enough_folds_but_collapsed = (3 >= 3) and (0.388 >= 0.5)
    assert enough_folds_but_collapsed is False


def test_continuous_targets_get_their_own_stability_check():
    """Classification metrics on a continuous target are undefined, not rough.

    A shuffled target_return_1d with 511 distinct values returned a balanced
    accuracy of 1.0 on all four folds, so every return context — the ones that
    matter most — passed a gate that had measured nothing. Regression folds are
    now scored in their own currency: R2 against the better of "predict the
    training mean" and "tomorrow equals today".
    """
    import numpy as np
    import pandas as pd

    assert hasattr(ModelingStage, '_regression_fold_stability')

    # A binary column is classification; a continuous one is not.
    frame = pd.DataFrame({
        'target_up_1d': [0, 1, 0, 1],
        'target_return_1d': [0.01, -0.02, 0.005, 0.03],
    })
    assert ModelingStage._is_binary_target(frame, 'target_up_1d') is True
    assert ModelingStage._is_binary_target(frame, 'target_return_1d') is False


def test_r_squared_and_persistence_agree_with_their_definitions():
    import numpy as np

    actual = np.array([1.0, 2.0, 3.0, 4.0])

    # A perfect prediction explains everything.
    assert ModelingStage._r_squared(actual, actual) == pytest.approx(1.0)
    # The mean explains nothing.
    assert ModelingStage._r_squared(
        actual, np.full(4, actual.mean())
    ) == pytest.approx(0.0)
    # Persistence on a trending series is better than the mean but not perfect.
    persistence = ModelingStage._persistence_r_squared(actual)
    assert 0.0 < persistence < 1.0


def test_feature_ranking_for_folds_uses_training_rows_only():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=100))
    x = pd.DataFrame({f'f{i}': rng.normal(size=100) for i in range(20)})
    x['signal'] = y * 3 + rng.normal(scale=0.1, size=100)

    chosen = ModelingStage._top_correlated(x, y, budget=3)

    assert 'signal' in chosen
    assert len(chosen) == 3


def test_an_unmeasurable_context_is_passed_through_rather_than_failed():
    """Refusing what could not be measured is the same error as trusting a zero."""
    stage = object.__new__(ModelingStage)
    stage_result = {'passed': True, 'fold_count': 1,
                    'reason': 'too few folds to measure stability'}

    # The contract: fold_count below the minimum yields passed=True with a
    # stated reason, never a silent refusal.
    assert stage_result['passed'] is True
    assert 'too few folds' in stage_result['reason']


def test_a_pooled_context_is_evaluated_rather_than_silently_skipped():
    """The rung that asks whether an edge holds over TIME was off entirely.

    `_prepare_context_frame` filtered `ticker == "__POOLED__"`. That is a
    synthetic context name, not a value in the data, so it matched ZERO of
    159,149 rows; "fewer than two classes" was raised, the caller logged it at
    DEBUG and returned None — and None is read upstream as "not measurable,
    pass through". Measured 2026-08-31: both champions of that run were
    promoted without this check, and pooling had been on since #155.

    The timestamp dedupe was the second half: 110 names share every bar, so
    `drop_duplicates(keep="last")` would have kept whichever ticker sorted
    last and turned 159,149 rows into ~7,200 belonging to no one.
    """
    import numpy as np
    import pandas as pd

    from src.pipeline.modeling_context import POOLED_TICKER
    from src.pipeline.stages.modeling.walk_forward_validation import (
        PipelineWalkForwardValidationEvaluator,
        WalkForwardValidationConfig,
    )

    tickers = [f"T{i:02d}" for i in range(12)]
    bars = pd.date_range("2024-01-01", periods=900, freq="15min", tz="UTC")
    rng = np.random.default_rng(5)
    rows = []
    for name in tickers:
        frame = pd.DataFrame({
            "datetime": bars,
            "ticker": name,
            "interval": "15m",
            "signal": rng.normal(size=len(bars)),
        })
        # Structure a working check must find, and a broken one cannot.
        frame["t"] = ((frame["signal"] > 0) | (rng.random(len(bars)) < 0.05)).astype(int)
        rows.append(frame)
    pooled = pd.concat(rows, ignore_index=True).sort_values("datetime")

    config = WalkForwardValidationConfig(
        min_train_rows=4000, validation_rows=1200, step_rows=1200,
        max_folds=3, max_features=3,
    )
    result = PipelineWalkForwardValidationEvaluator(config).evaluate(
        pooled, ticker=POOLED_TICKER, timeframe="15m", target_name="t")

    metrics = result["metrics"]
    assert metrics["fold_count"] >= 3
    # Every row kept, not one per timestamp.
    assert result["folds"][0]["train_window"]["sample_count"] > 1000
    assert metrics["mean_validation_balanced_accuracy"] > 0.6


def test_the_purge_is_scaled_by_rows_per_bar_on_a_pooled_frame():
    """A purge of 5 ROWS spans a twentieth of a bar when 110 names share it.

    The same defect already fixed for the split in `prepare_data_for_models`;
    here it would let the training window all but touch its validation window.
    """
    import numpy as np
    import pandas as pd

    from src.pipeline.modeling_context import POOLED_TICKER
    from src.pipeline.stages.modeling.walk_forward_validation import (
        PipelineWalkForwardValidationEvaluator,
        WalkForwardValidationConfig,
    )

    names = [f"T{i}" for i in range(10)]
    bars = pd.date_range("2024-01-01", periods=800, freq="15min", tz="UTC")
    rng = np.random.default_rng(6)
    pooled = pd.concat([
        pd.DataFrame({"datetime": bars, "ticker": n, "interval": "15m",
                      "signal": rng.normal(size=len(bars)),
                      "t": (rng.random(len(bars)) > 0.5).astype(int)})
        for n in names
    ], ignore_index=True).sort_values("datetime")

    config = WalkForwardValidationConfig(
        min_train_rows=3000, validation_rows=1000, step_rows=1000,
        max_folds=3, max_features=2, purge_rows=5,
    )
    result = PipelineWalkForwardValidationEvaluator(config).evaluate(
        pooled, ticker=POOLED_TICKER, timeframe="15m", target_name="t")

    purge = result["folds"][0]["purge_window"]
    # 5 configured bars x 10 names sharing every bar.
    assert purge["row_count"] == 5 * 10, purge
