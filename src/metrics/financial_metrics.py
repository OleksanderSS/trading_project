"""
Backward-compatible adapter for the former `calculate_performance_metrics`
function.

The financial-metrics implementation was migrated into
``FinancialMetricsLibrary`` (``src/metrics/financial/financial_metrics_library.py``)
during the metrics consolidation refactor. Several modules still import the
legacy ``calculate_performance_metrics`` entry point; this thin shim preserves
that contract by delegating to the static library methods.

Do not add new logic here — new metrics belong on ``FinancialMetricsLibrary``.
This file exists only to keep the old call sites working.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary


def calculate_performance_metrics(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.02,
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    """Aggregate the standard performance metrics into a single dict.

    This mirrors the historical signature and return shape used across the
    codebase (backtest engine, comparison scripts, diary, etc.). All heavy
    lifting is delegated to ``FinancialMetricsLibrary``.

    Args:
        returns: Periodic returns (e.g. daily). Series or 1-D ndarray.
        risk_free_rate: Annualized risk-free rate used for Sharpe/Sortino.
        trading_days_per_year: Annualization factor (252 for daily).

    Returns:
        Dict with total_return, sharpe_ratio, sortino_ratio, max_drawdown,
        volatility, var, cvar, etc. Missing/non-finite values become 0.0
        so downstream consumers never crash on degenerate input.
    """
    if returns is None:
        return {}

    returns_series = pd.Series(returns).dropna()
    if returns_series.empty:
        return {}

    # Equity curve reconstructed from periodic returns (start = 1.0)
    equity = (1.0 + returns_series).cumprod()

    Lib = FinancialMetricsLibrary

    def _safe(fn, *args, **kwargs) -> Any:
        try:
            value = fn(*args, **kwargs)
        except (ValueError, TypeError, AttributeError, KeyError,
                ZeroDivisionError, OverflowError):
            return 0.0
        if isinstance(value, float):
            return value if np.isfinite(value) else 0.0
        return value

    total_return = _safe(Lib.calculate_total_return, equity)
    volatility = _safe(
        Lib.calculate_annualized_volatility,
        returns_series,
        trading_days_per_year,
    )
    sharpe = _safe(
        Lib.calculate_sharpe_ratio,
        returns_series,
        risk_free_rate,
        trading_days_per_year,
    )
    sortino = _safe(
        Lib.calculate_sortino_ratio,
        returns_series,
        risk_free_rate,
        trading_days_per_year,
    )
    max_drawdown = _safe(Lib.calculate_max_drawdown, equity)
    calmar = _safe(Lib.calculate_calmar_ratio, equity, trading_days_per_year)
    underwater = _safe(Lib.calculate_underwater_duration, equity)

    try:
        risk = Lib.calculate_var_cvar(returns_series, confidence_level=0.95)
    except (ValueError, TypeError, AttributeError, KeyError,
            ZeroDivisionError, OverflowError):
        risk = {"var_95": 0.0, "cvar_95": 0.0}

    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(max_drawdown),
        "volatility": float(volatility),
        "calmar_ratio": float(calmar),
        "underwater_duration": int(underwater) if underwater else 0,
        "var_95": float(risk.get("var_95", 0.0)),
        "cvar_95": float(risk.get("cvar_95", 0.0)),
    }


__all__ = ["calculate_performance_metrics"]
