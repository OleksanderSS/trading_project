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


def test_cost_stress_charges_only_the_increment_not_the_whole_fee_again():
    """All five return targets already have 0.5% round-trip subtracted.

    Charging it a second time would understate the edge as badly as omitting
    it overstates it, so 1.0x must leave the curve exactly as built.
    """
    from src.pipeline.stages.evaluation.holdout_equity import (
        BASELINE_ROUND_TRIP_COST,
        stress_costs,
    )

    frame = _predictions(n=100)
    base = build_holdout_equity(frame)
    stressed = stress_costs(frame)

    assert stressed['baseline_already_in_target'] is True
    assert stressed['levels']['x1']['incremental_charged'] == 0.0
    assert np.isclose(
        stressed['levels']['x1']['total_return'],
        float(base['portfolio_history']['total_value'].iloc[-1] / 100_000.0 - 1.0),
    )
    assert stressed['levels']['x2']['incremental_charged'] == BASELINE_ROUND_TRIP_COST


def test_higher_costs_never_improve_the_result():
    from src.pipeline.stages.evaluation.holdout_equity import stress_costs

    stressed = stress_costs(_predictions(n=100))
    returns = [stressed['levels'][k]['total_return'] for k in ('x1', 'x1.5', 'x2')]

    assert returns[0] >= returns[1] >= returns[2]


def test_a_signal_that_flips_every_bar_reports_full_turnover():
    """Turnover is usually the number that decides whether costs kill an edge."""
    from src.pipeline.stages.evaluation.holdout_equity import stress_costs

    frame = _predictions(n=60)
    frame['prediction'] = [1.0, -1.0] * 30      # flips on every bar

    stressed = stress_costs(frame)

    assert stressed['turnover'] > 0.95
    # And that turnover must cost something by x2.
    assert stressed['levels']['x2']['total_return'] < stressed['levels']['x1']['total_return']


def test_a_held_position_pays_once_for_entering_and_no_more():
    """Buy and hold still pays the entry. It just does not pay it 60 times."""
    from src.pipeline.stages.evaluation.holdout_equity import stress_costs

    held = _predictions(n=60)
    held['prediction'] = 1.0                     # one entry, then held

    flipping = _predictions(n=60)
    flipping['prediction'] = [1.0, -1.0] * 30    # a round trip every bar

    held_stress = stress_costs(held)
    flip_stress = stress_costs(flipping)

    assert held_stress['turnover'] < 0.05        # 1 trade in 60 bars
    assert flip_stress['turnover'] > 0.95

    def cost_of_doubling(result):
        return result['levels']['x1']['total_return'] - result['levels']['x2']['total_return']

    # Both pay, but turnover decides by how much — the point of the check.
    assert cost_of_doubling(flip_stress) > 10 * cost_of_doubling(held_stress)


def test_return_targets_are_recognised_by_name():
    assert is_return_target('target_return_1d')
    assert is_return_target('target_hourly_return_1h')
    assert not is_return_target('target_up_1d')
    assert not is_return_target('target_volatility_spike_15m')
    assert not is_return_target(None)
