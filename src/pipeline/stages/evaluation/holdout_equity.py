"""Build an equity curve from out-of-sample holdout predictions.

Stage 7 has been computing its financial metrics from Stage 5, which answers
a different question. Stage 5 predicts the LATEST bar of each context — one
point apiece — so 540 predictions pivoted to a `(3, 22)` table. Three time
points. The Sharpe of -329.82 at a volatility of 8.46e-05 in
summary_20260812_020842.json was not a nearly flat curve, it was a
three-point one, and no arithmetic fix downstream could have helped.

The holdout is the honest source: ~100-220 purged bars per context that the
model never saw and was never selected on. For a return target the realised
value stored beside each prediction IS the return, so the strategy return of
a bar is simply

    position(prediction) * actual

and no price series is needed. Positions are the sign of the prediction, so a
model that forecasts a negative return goes short and one that forecasts zero
stands aside.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("HoldoutEquity")

#: Only these targets carry a realised RETURN in `actual`. A classification
#: target stores a 0/1 label and a spike target a flag; multiplying a position
#: by either produces a number with no monetary meaning, which is the kind of
#: quantity this whole audit has been removing.
_RETURN_TARGET_MARKERS = ('return',)


def is_return_target(target: str | None) -> bool:
    if not target:
        return False
    return any(marker in str(target).lower() for marker in _RETURN_TARGET_MARKERS)


def build_holdout_equity(
    predictions: pd.DataFrame,
    *,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """Turn holdout predictions into a portfolio equity curve.

    Returns a dict with `portfolio_history` (a DataFrame with total_value and
    a DatetimeIndex), the per-bar strategy returns, and the counts behind
    them, or a `status` explaining why no curve could be built. Nothing is
    fabricated: a batch with no return targets yields no curve rather than a
    curve of zeros.
    """
    if predictions is None or predictions.empty:
        return {'status': 'no_holdout_predictions'}

    frame = predictions.copy()
    frame = frame[frame['target'].map(is_return_target)]
    if frame.empty:
        return {
            'status': 'no_return_targets',
            'reason': (
                'Holdout predictions exist, but none is a return target. A '
                'position multiplied by a 0/1 label is not a return.'
            ),
        }

    frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True, errors='coerce')
    frame = frame.dropna(subset=['datetime', 'prediction', 'actual'])
    if frame.empty:
        return {'status': 'no_usable_rows'}

    frame['position'] = np.sign(frame['prediction'].astype(float))
    frame['strategy_return'] = frame['position'] * frame['actual'].astype(float)

    # NET EXPOSURE AND THE NAIVE OPPONENT, both measured, neither changed.
    #
    # `sign(prediction)` says nothing about how one-sided the book is. A model
    # that predicts up for almost every name produces positions of +1 almost
    # everywhere, and this curve is then the MARKET rather than the strategy.
    # That is not hypothetical: on 2026-09-04 seven features cleared a
    # Bonferroni correction at net Sharpe 1.016 in the diagnostics, and the
    # constant opponent -- hold everything, same clock, same friction -- scored
    # 1.018. All seven were that opponent (CLAIMS R28).
    #
    # The strategy definition is NOT changed here. Demeaning the position would
    # convert this into a dollar-neutral book, which is a decision about what
    # the pipeline trades and belongs to the owner, not to a fix. What is added
    # is the two numbers that make the substitution visible: how one-sided the
    # book is, and what owning everything would have returned over the same
    # bars.
    mean_position = float(frame['position'].mean())
    constant_per_bar = (
        frame.groupby('datetime')['actual'].apply(lambda s: s.astype(float).mean())
        .sort_index()
    )

    # Average across whatever contexts hold a position on that bar: an
    # equal-weight portfolio. Summing instead would make the return depend on
    # how many contexts happened to be trained, which is a property of the
    # pipeline rather than of the strategy.
    per_bar = (
        frame.groupby('datetime')['strategy_return']
        .mean()
        .sort_index()
    )
    if per_bar.empty:
        return {'status': 'no_usable_rows'}

    equity = initial_capital * (1.0 + per_bar).cumprod()
    portfolio_history = pd.DataFrame({'total_value': equity})
    portfolio_history.index.name = 'datetime'

    constant_equity = initial_capital * (1.0 + constant_per_bar).cumprod()
    constant_return = (float(constant_equity.iloc[-1] / initial_capital - 1.0)
                       if len(constant_equity) else 0.0)
    strategy_return = (float(equity.iloc[-1] / initial_capital - 1.0)
                       if len(equity) else 0.0)

    logger.info(
        'Built holdout equity curve: %d bars across %d contexts, %d tickers',
        len(per_bar), frame['context'].nunique(), frame['ticker'].nunique(),
    )
    if abs(mean_position) >= ONE_SIDED_WARNING:
        logger.warning(
            'The holdout book is %.0f%% one-sided (mean position %+.3f). Owning '
            'everything over the same bars returned %+.2f%%; this curve returned '
            '%+.2f%%. A curve this directional is mostly the market, and the '
            'difference is the only part the model earned.',
            abs(mean_position) * 100, mean_position,
            constant_return * 100, strategy_return * 100,
        )
    else:
        logger.info(
            'Holdout book net exposure %+.3f; buy-everything returned %+.2f%% '
            'over the same bars, against %+.2f%% for this curve.',
            mean_position, constant_return * 100, strategy_return * 100,
        )
    return {
        'status': 'built',
        'portfolio_history': portfolio_history,
        'returns': per_bar,
        'bar_count': int(len(per_bar)),
        'context_count': int(frame['context'].nunique()),
        'ticker_count': int(frame['ticker'].nunique()),
        'source': 'holdout_predictions',
        'mean_position': mean_position,
        'total_return': strategy_return,
        'constant_opponent_return': constant_return,
        'excess_over_constant': strategy_return - constant_return,
    }


#: How one-sided a book may be before the curve is reported as mostly market
#: exposure. At 0.5 the book is three-quarters on one side; below that the
#: constant opponent is still reported, just without the warning.
ONE_SIDED_WARNING = 0.5

#: Round-trip cost already subtracted from every return target by
#: targets.yaml (commission 0.1% + spread 0.05% + slippage 0.1%, doubled).
#: Verified against the config for all five return targets before use —
#: charging it a second time would understate the edge as badly as omitting
#: it overstates it.
BASELINE_ROUND_TRIP_COST = 0.005

#: Codex §21.3: an edge that only survives at the assumed cost level is not an
#: edge. 1.0 is what the data already carries.
COST_STRESS_MULTIPLIERS = (1.0, 1.5, 2.0)


def stress_costs(
    predictions: pd.DataFrame,
    *,
    multipliers: tuple[float, ...] = COST_STRESS_MULTIPLIERS,
    initial_capital: float = 100_000.0,
) -> dict[str, Any]:
    """Does the edge survive higher trading costs than assumed?

    The five return targets already have a 0.5% round trip subtracted by
    targets.yaml, so 1.0x is the curve as built and each higher multiplier
    charges only the INCREMENT — (m - 1) x 0.5% — never the whole amount
    again.

    The increment is charged on position CHANGES rather than on every bar,
    which is what actually costs money. Note the asymmetry that creates and
    that this function cannot fix: the baseline 0.5% was subtracted from every
    bar by the target definition regardless of whether a trade happened, which
    is conservative. Reported turnover makes the difference visible instead of
    burying it.

    Turnover is the number that usually decides. A signal that flips position
    every bar pays the round trip every bar, and at 0.5% a 15m strategy would
    need to earn that back 26 times a day.
    """
    base = build_holdout_equity(predictions, initial_capital=initial_capital)
    if base.get('status') != 'built':
        return base

    frame = predictions.copy()
    frame = frame[frame['target'].map(is_return_target)]
    frame['datetime'] = pd.to_datetime(frame['datetime'], utc=True, errors='coerce')
    frame = frame.dropna(subset=['datetime', 'prediction', 'actual'])
    frame['position'] = np.sign(frame['prediction'].astype(float))

    # A position change within one context, in time order.
    frame = frame.sort_values(['context', 'datetime'])
    previous = frame.groupby('context')['position'].shift(1).fillna(0.0)
    frame['traded'] = (frame['position'] != previous).astype(float)

    turnover_per_bar = frame.groupby('datetime')['traded'].mean().sort_index()
    overall_turnover = float(frame['traded'].mean())

    results = {}
    for multiplier in multipliers:
        increment = (float(multiplier) - 1.0) * BASELINE_ROUND_TRIP_COST
        frame['stressed_return'] = (
            frame['position'] * frame['actual'].astype(float)
            - frame['traded'] * increment
        )
        per_bar = frame.groupby('datetime')['stressed_return'].mean().sort_index()
        equity = initial_capital * (1.0 + per_bar).cumprod()
        total_return = float(equity.iloc[-1] / initial_capital - 1.0) if len(equity) else 0.0
        results[f"x{multiplier:g}"] = {
            'round_trip_cost': BASELINE_ROUND_TRIP_COST * float(multiplier),
            'incremental_charged': increment,
            'total_return': total_return,
            'final_equity': float(equity.iloc[-1]) if len(equity) else initial_capital,
            'mean_bar_return': float(per_bar.mean()),
        }

    survives = all(entry['total_return'] > 0 for entry in results.values())
    return {
        'status': 'built',
        'bar_count': base['bar_count'],
        'context_count': base['context_count'],
        'turnover': overall_turnover,
        'mean_turnover_per_bar': float(turnover_per_bar.mean()),
        'baseline_round_trip_cost': BASELINE_ROUND_TRIP_COST,
        'baseline_already_in_target': True,
        'levels': results,
        'survives_double_costs': survives,
    }


def load_holdout_predictions(path: str | Path | None) -> pd.DataFrame | None:
    """Read the artifact Stage 4 wrote, if it is there."""
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        logger.warning(f'No holdout predictions artifact at {file_path}')
        return None
    try:
        return pd.read_parquet(file_path)
    except (OSError, ValueError) as e:
        logger.error(f'Could not read holdout predictions from {file_path}: {e}')
        return None
