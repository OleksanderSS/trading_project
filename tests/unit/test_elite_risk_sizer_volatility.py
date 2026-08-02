"""Position size must respond to how risky the instrument is.

EliteRiskSizer is live: Stage 6 builds it (trading/orchestrator.py:53) and
PortfolioManager calls compute_optimal_position_size
(portfolio_manager.py:249) as the fallback when AdaptivePositionSizer fails
-- that is, exactly when something has already gone wrong.

Per-ticker volatility reached the arithmetic on neither route:

- compute_optimal_position_size accepted `ticker_volatility` as a parameter
  and never used it, so the caller's figure was discarded;
- calculate_optimal_position_size computed its own via
  _estimate_ticker_volatility, which reads self.historical_returns, which
  update_returns_data fills -- and update_returns_data has ZERO callers. It
  therefore always returned its 0.2 fallback.

With portfolio_volatility at the caller's hardcoded 0.15, vol_factor was a
constant 0.75 for every instrument: NVDA sized exactly like KO.

Measured after wiring it through, on identical inputs:

    volatility not supplied (0.2 fallback)   46 shares
    calm ticker      0.12                    78
    average          0.20                    46
    volatile         0.60                    31
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.trading.elite_risk_sizer import EliteRiskSizer


@pytest.fixture()
def sizer():
    EliteRiskSizer._reported_missing_returns = False
    return EliteRiskSizer()


BASE = dict(
    entry_price=100.0,
    win_rate=0.55,
    avg_win_loss_ratio=1.5,
    current_positions={},
    total_equity=100_000.0,
    position_value_limit=0.15,
    portfolio_volatility=0.15,
    cash_available=100_000.0,
)


def _size(sizer, volatility):
    return sizer.calculate_optimal_position_size(
        ticker="X", ticker_volatility=volatility, **BASE
    )


def test_a_calmer_instrument_gets_a_bigger_position(sizer):
    assert _size(sizer, 0.12) > _size(sizer, 0.20)


def test_a_wilder_instrument_gets_a_smaller_one(sizer):
    assert _size(sizer, 0.60) < _size(sizer, 0.20)


def test_size_is_monotonic_in_volatility(sizer):
    sizes = [_size(sizer, volatility) for volatility in (0.10, 0.20, 0.40, 0.80)]

    assert sizes == sorted(sizes, reverse=True)


def test_the_supplied_figure_beats_the_internal_estimate(sizer):
    """The estimate has no data; a caller that knows better must win."""
    sizer.update_returns_data("X", pd.Series(np.full(50, 0.001)))

    assert _size(sizer, 0.60) != _size(sizer, None)


def test_without_a_figure_the_fallback_still_produces_a_position(sizer):
    assert _size(sizer, None) > 0


def test_the_missing_history_is_reported_once(sizer, caplog):
    with caplog.at_level(logging.WARNING):
        _size(sizer, None)
        _size(sizer, None)
        _size(sizer, None)

    warnings = [r for r in caplog.records if "update_returns_data" in r.getMessage()]
    assert len(warnings) == 1


def test_the_live_entry_point_passes_it_down():
    """compute_optimal_position_size used to accept and discard it."""
    import inspect

    source = inspect.getsource(EliteRiskSizer.compute_optimal_position_size)

    assert "ticker_volatility=ticker_volatility" in source


def test_a_zero_volatility_does_not_divide_by_zero(sizer):
    assert _size(sizer, 0.0) > 0


def test_the_factor_stays_within_its_documented_clamp(sizer):
    """vol_factor is clipped to [0.5, 1.5], so the extremes must saturate
    rather than run away."""
    tiny = _size(sizer, 0.001)
    huge = _size(sizer, 100.0)

    assert tiny == _size(sizer, 0.01)
    assert huge == _size(sizer, 10.0)
