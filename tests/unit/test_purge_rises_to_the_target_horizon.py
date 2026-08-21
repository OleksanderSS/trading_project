"""The purge nobody configures, and therefore nobody notices going wrong.

`purge_rows` is 5 everywhere and appears in no config file. That is the right
design -- a purge hardcoded per target would drift away from the target
definitions the moment one changed -- but it means the whole guarantee rests on
`enforce_horizon_purge` raising it at runtime, and nothing pinned that.

The failure it prevents is not subtle. `target_hourly_volume_spike_1h` reaches
23 bars ahead. With a purge of 5, the labels on the last 18 training rows are
computed from prices inside the validation window, and the hyperparameters
that win are the ones best at exploiting the overlap.

Existing tests pin that a GIVEN purge is honoured by the splitter. These pin
that the right purge is chosen in the first place.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.pipeline.stages.modeling.walk_forward_validation import (
    PipelineWalkForwardValidationEvaluator,
    WalkForwardValidationConfig,
    _get_target_horizon_rows,
)


def _frame(rows: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=rows, freq="h"),
        "ticker": ["AAPL"] * rows,
        "interval": ["60m"] * rows,
        "f1": rng.standard_normal(rows),
        "f2": rng.standard_normal(rows),
        "target_hourly_volume_spike_1h": rng.integers(0, 2, rows),
    })


def test_a_multi_bar_horizon_is_read_as_more_than_one_row():
    """Falling back to 1 silently is how the gap gets closed to nothing."""
    horizon = _get_target_horizon_rows("target_hourly_volume_spike_1h")
    assert horizon > 1


def test_an_unknown_target_never_yields_a_zero_purge():
    assert _get_target_horizon_rows("target_that_does_not_exist") >= 1


def test_the_evaluator_raises_the_purge_and_says_so():
    """The default 5 must not be what a 23-bar target gets."""
    target = "target_hourly_volume_spike_1h"
    horizon = _get_target_horizon_rows(target)
    if horizon <= 5:
        pytest.skip(f"{target} no longer reaches beyond the default purge")

    evaluator = PipelineWalkForwardValidationEvaluator(
        WalkForwardValidationConfig(purge_rows=5, max_folds=2)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = evaluator.evaluate(
            _frame(), ticker='AAPL', timeframe='60m', target_name=target
        )

    messages = [str(w.message) for w in caught]
    assert any("Automatically raising purge_rows" in m for m in messages), messages

    contract = _first_contract(result)
    assert contract is not None, "no fold reported a temporal contract"
    assert contract["purge_rows"] >= horizon


def test_turning_the_guarantee_off_is_visible_in_the_contract():
    """If this ever flips to False, a test says what it costs, not a silence."""
    target = "target_hourly_volume_spike_1h"
    horizon = _get_target_horizon_rows(target)
    if horizon <= 5:
        pytest.skip(f"{target} no longer reaches beyond the default purge")

    evaluator = PipelineWalkForwardValidationEvaluator(
        WalkForwardValidationConfig(purge_rows=5, max_folds=2,
                                    enforce_horizon_purge=False)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluator.evaluate(
            _frame(), ticker='AAPL', timeframe='60m', target_name=target
        )

    contract = _first_contract(result)
    assert contract is not None
    assert contract["purge_rows"] == 5 < horizon


def test_the_default_keeps_the_guarantee_on():
    assert WalkForwardValidationConfig().enforce_horizon_purge is True


def _first_contract(result):
    folds = (result or {}).get("folds") or []
    for fold in folds:
        contract = fold.get("temporal_contract")
        if contract and "purge_rows" in contract:
            return contract
    return None
