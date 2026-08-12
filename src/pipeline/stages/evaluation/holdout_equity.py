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

    logger.info(
        'Built holdout equity curve: %d bars across %d contexts, %d tickers',
        len(per_bar), frame['context'].nunique(), frame['ticker'].nunique(),
    )
    return {
        'status': 'built',
        'portfolio_history': portfolio_history,
        'returns': per_bar,
        'bar_count': int(len(per_bar)),
        'context_count': int(frame['context'].nunique()),
        'ticker_count': int(frame['ticker'].nunique()),
        'source': 'holdout_predictions',
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
