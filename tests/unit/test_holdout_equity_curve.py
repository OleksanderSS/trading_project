"""The equity curve comes from out-of-sample bars, not from three live signals.

Stage 7 computed its financial metrics from Stage 5, which predicts the
LATEST bar of each context — one point apiece. 540 predictions therefore
pivoted to a (3, 22) table, and the Sharpe of -329.82 at a volatility of
8.46e-05 in summary_20260812_020842.json came from a three-point curve. No
arithmetic fix downstream could have helped that.

Holdout predictions are ~100-220 purged bars per context that the model never
saw and was never selected on, and for a return target the realised value
stored beside each prediction IS the return — so an honest curve needs no
price series at all.
"""
import numpy as np
import pandas as pd

from src.pipeline.stages.evaluation.holdout_equity import (
    build_holdout_equity,
    is_return_target,
)


def _predictions(n=50, target='target_return_1d', seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.normal(0.001, 0.01, n)
    return pd.DataFrame({
        'context': ['AAPL_1d_target_return_1d_normal'] * n,
        'ticker': ['AAPL'] * n,
        'timeframe': ['1d'] * n,
        'target': [target] * n,
        'model_type': ['linear'] * n,
        'datetime': pd.date_range('2026-05-01', periods=n, freq='D', tz='UTC'),
        'prediction': actual + rng.normal(0, 0.002, n),
        'actual': actual,
    })


def test_the_curve_has_one_point_per_out_of_sample_bar():
    result = build_holdout_equity(_predictions(n=120))

    assert result['status'] == 'built'
    assert result['bar_count'] == 120, "a curve of three points is what this replaces"
    assert len(result['portfolio_history']) == 120
    assert isinstance(result['portfolio_history'].index, pd.DatetimeIndex)


def test_a_perfect_forecaster_makes_money_and_a_reversed_one_loses_it():
    """position = sign(prediction), return = position * actual — no prices."""
    frame = _predictions(n=80)
    frame['prediction'] = frame['actual']            # perfect sign every bar

    good = build_holdout_equity(frame)
    assert good['returns'].sum() > 0

    frame['prediction'] = -frame['actual']           # always wrong
    bad = build_holdout_equity(frame)
    assert bad['returns'].sum() < 0


def test_contexts_are_equal_weighted_per_bar_not_summed():
    """Summing would make the return depend on how many contexts were trained."""
    one = _predictions(n=30)
    two = pd.concat([one, one.assign(context='AAPL_1d_second', ticker='MSFT')])

    single = build_holdout_equity(one)
    doubled = build_holdout_equity(two)

    assert np.allclose(single['returns'].to_numpy(), doubled['returns'].to_numpy())


def test_classification_targets_are_refused_rather_than_multiplied():
    """A position times a 0/1 label is not a return."""
    frame = _predictions(n=40, target='target_up_1d')

    result = build_holdout_equity(frame)

    assert result['status'] == 'no_return_targets'
    assert 'portfolio_history' not in result


def test_an_empty_artifact_yields_no_curve_rather_than_a_flat_one():
    result = build_holdout_equity(pd.DataFrame())

    assert result['status'] == 'no_holdout_predictions'
    assert 'portfolio_history' not in result


def test_rows_without_a_timestamp_or_a_value_are_dropped():
    frame = _predictions(n=20)
    frame.loc[0, 'datetime'] = None
    frame.loc[1, 'actual'] = np.nan

    result = build_holdout_equity(frame)

    assert result['bar_count'] == 18


def test_return_targets_are_recognised_by_name():
    assert is_return_target('target_return_1d')
    assert is_return_target('target_hourly_return_1h')
    assert not is_return_target('target_up_1d')
    assert not is_return_target('target_volatility_spike_15m')
    assert not is_return_target(None)
