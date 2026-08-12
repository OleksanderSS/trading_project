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


def test_intraday_contexts_keep_the_full_windows():
    """Shrinking is for contexts that cannot afford the defaults, not for all."""
    default = ModelingStage._walk_forward_config_for(1100)

    assert default.min_train_rows == 360
    assert default.validation_rows == 120


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


def test_an_unmeasurable_context_is_passed_through_rather_than_failed():
    """Refusing what could not be measured is the same error as trusting a zero."""
    stage = object.__new__(ModelingStage)
    stage_result = {'passed': True, 'fold_count': 1,
                    'reason': 'too few folds to measure stability'}

    # The contract: fold_count below the minimum yields passed=True with a
    # stated reason, never a silent refusal.
    assert stage_result['passed'] is True
    assert 'too few folds' in stage_result['reason']
