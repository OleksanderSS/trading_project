"""A fold that produced no comparison must not vanish into the count.

REGISTER #199 and #191, and this file exists because BOTH were wrong about the
same number.

#191 observed `fold_count = 2` on a pooled run and concluded the GEOMETRY left
room for only two folds. #199 was filed on that reading and proposed shrinking
the training window to get more folds and, as a bonus, a faster run.

Measured 2026-09-05, and it refutes both halves:

    rows      folds built by the real config path
       900      3
    30,494      4
   127,424      4
   352,000      4
   623,398      4

The geometry gives EXACTLY FOUR folds at every pooled size, by construction:
`min_train = n/2` and `validation = n/8` walk from n/2 to n in four steps of
n/8. And shrinking to n/3 and n/12 was measured too -- same four folds (the cap
is `max_folds=4`), and MORE total training rows fitted, 403k against 350k at
127,424 rows, because the last four folds then sit closer to the end with
larger expanding train sets. Slower, not faster.

So `fold_count = 2` never meant "two folds fit". It meant two of the four were
DROPPED inside the loop, by one of three silent `continue` branches -- too few
labelled rows, too few train rows complete on the chosen features, too few
validation rows complete on them. A dropped fold and a passed fold looked
identical from outside, in the check whose entire job is to say whether a
signal is stable. That is family B (#202) in the stability gate.

The fix is not a new geometry. It is that the count now travels with
`folds_built` and `folds_skipped`, and a drop is logged with its reason.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.modeling.orchestrator import ModelingStage
from src.pipeline.stages.modeling.walk_forward_validation import (
    build_purged_expanding_folds,
)


def _stage() -> ModelingStage:
    """Bypass __init__: the method under test reads no instance state beyond
    the class constants, and building a real stage would drag the whole
    config stack into a contract about fold bookkeeping."""
    return ModelingStage.__new__(ModelingStage)


def _frame(rows: int, blank_target_before: int = 0, seed: int = 0):
    rng = np.random.default_rng(seed)
    feature = rng.normal(size=rows)
    target = feature * 0.4 + rng.normal(scale=0.5, size=rows)
    if blank_target_before:
        target[:blank_target_before] = np.nan
    return pd.DataFrame({
        "feature_a": feature,
        "feature_b": rng.normal(size=rows),
        "target_x": target,
    })


def test_the_geometry_gives_four_folds_at_every_pooled_size():
    """The measurement that refuted #199. Pinned so the entry cannot be
    re-filed on the same false premise."""
    for rows, expected in [(30_494, 4), (127_424, 4), (352_000, 4), (623_398, 4)]:
        config = ModelingStage._walk_forward_config_for(rows)
        folds = build_purged_expanding_folds(rows, config=config)
        assert len(folds) == expected, (
            f"{rows:,} rows built {len(folds)} folds, not {expected}; the "
            "claim that the pooled geometry only fits two or three folds was "
            "measured false on 2026-09-05"
        )


def test_a_full_frame_measures_every_fold_it_builds():
    result = _stage()._regression_fold_stability(
        _frame(5_000), target_name="target_x", context_key="TEST/1d/target_x")
    assert result["fold_count"] == result["folds_built"]
    assert result["folds_skipped"] == []


def test_a_dropped_fold_is_counted_separately_and_named():
    """The real shape behind `fold_count = 2`: the target is missing over the
    early rows, so the first fold has nothing to learn from and disappears."""
    result = _stage()._regression_fold_stability(
        _frame(5_000, blank_target_before=3_000),
        target_name="target_x", context_key="TEST/1d/target_x")

    assert result["folds_built"] > result["fold_count"], (
        "every fold was measured, so this frame no longer exercises the drop"
    )
    assert result["folds_skipped"], "a fold vanished without a record"
    reasons = {item["reason"] for item in result["folds_skipped"]}
    assert reasons <= {
        "too few labelled rows",
        "too few rows complete on the chosen features",
        "too few validation rows complete on the chosen features",
    }
    assert all("fold" in item for item in result["folds_skipped"]), (
        "the record does not say WHICH fold, so it cannot be chased"
    )


def test_a_drop_is_audible(caplog):
    with caplog.at_level(logging.WARNING):
        _stage()._regression_fold_stability(
            _frame(5_000, blank_target_before=3_000),
            target_name="target_x", context_key="TEST/1d/target_x")
    said = " ".join(record.getMessage() for record in caplog.records
                    if record.levelno >= logging.WARNING)
    assert "walk-forward folds produced no comparison" in said
    assert "TEST/1d/target_x" in said, "the warning does not name the context"
    assert "#199" in said or "#202" in said, (
        "the warning does not point at why this is recorded at all"
    )


def test_the_refusal_path_also_carries_the_breakdown():
    """When too few folds could be measured the stage returns `measured:
    False`. That is the honest answer, and it must still say what was built
    and what was lost -- otherwise the caller sees "not measured" with no way
    to tell a small frame from a broken one."""
    result = _stage()._regression_fold_stability(
        _frame(1_200, blank_target_before=1_100),
        target_name="target_x", context_key="TEST/1d/target_x")
    assert result["measured"] is False
    assert "folds_built" in result
    assert "folds_skipped" in result


@pytest.mark.parametrize("rows", [900, 5_000])
def test_the_count_never_exceeds_what_was_built(rows):
    result = _stage()._regression_fold_stability(
        _frame(rows), target_name="target_x", context_key="TEST/1d/target_x")
    assert result["fold_count"] <= result["folds_built"]
