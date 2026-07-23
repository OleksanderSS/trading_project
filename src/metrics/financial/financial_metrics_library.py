"""
Financial Metrics Library
Unified collection of financial, statistical, and econometric metrics.
Provides core calculation logic for portfolios, assets, and benchmarks.
"""

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from src.core.logging.logger import ProjectLogger

# ✅ Re-export calculation tools so callers can use a single import
from src.metrics.utils.calculation_tools import (  # noqa: F401
    adjust_for_risk_free_rate,
    annualize_returns,
)


def infer_periods_per_year(returns: pd.Series) -> int:
    """Infer annualisation factor from the DatetimeIndex of *returns*.

    Falls back to 252 (daily) when the index is not a DatetimeIndex or the
    median gap cannot be determined reliably. Moved here from
    src/algorithms/metrics_mixin.py so a single canonical Sharpe
    implementation (below) can offer cadence-aware annualisation to every
    caller, not just the backtest engine — this project runs 15m/1h/1d
    timeframes side by side, and a fixed 252 assumes daily bars regardless
    of what's actually being measured.
    """
    if not isinstance(returns.index, pd.DatetimeIndex) or len(returns) < 2:
        return 252

    gaps = returns.index.to_series().diff().dropna()
    if gaps.empty:
        return 252

    median_seconds = gaps.median().total_seconds()
    if median_seconds <= 90:           # ≤ 1.5 min → 1-minute bars
        return 252 * 390
    if median_seconds <= 1200:         # ≤ 20 min → 15-minute bars
        return 252 * 26
    if median_seconds <= 5400:         # ≤ 90 min → 1-hour bars
        return 252 * 7
    if median_seconds <= 100_000:      # ≤ ~1.15 days → daily
        return 252
    if median_seconds <= 800_000:      # ≤ ~9 days → weekly
        return 52
    if median_seconds <= 2_800_000:    # ≤ ~32 days → monthly
        return 12
    return 4                           # quarterly

logger = ProjectLogger.get_logger("FinancialMetricsLibrary")


class FinancialMetricsLibrary:
    """
    Core library for financial calculations.
    Stateless methods for maximum reusability.
    """

    @staticmethod
    def calculate_total_return(equity_curve: pd.Series) -> float:
        """Calculates total percentage return."""
        if equity_curve.empty:
            return 0.0
        initial_equity = equity_curve.iloc[0]
        # Validate initial equity
        if not np.isfinite(initial_equity) or initial_equity <= 0:
            logger.warning(f"Invalid initial equity: {initial_equity}. Cannot calculate total return.")
            return 0.0
        return float((equity_curve.iloc[-1] - initial_equity) / initial_equity)

    @staticmethod
    def calculate_cagr(equity_curve: pd.Series, trading_days_per_year: int = 252) -> float:
        """Calculates Compound Annual Growth Rate."""
        if equity_curve.empty:
            return 0.0
        initial_equity = equity_curve.iloc[0]
        # Validate initial equity
        if not np.isfinite(initial_equity) or initial_equity <= 0:
            logger.warning(f"Invalid initial equity: {initial_equity}. Cannot calculate CAGR.")
            return 0.0
        years = len(equity_curve) / trading_days_per_year
        if years <= 0:
            return 0.0
        total_return_factor = equity_curve.iloc[-1] / initial_equity
        return float(total_return_factor ** (1 / years) - 1)

    @staticmethod
    def calculate_annualized_volatility(returns: pd.Series, trading_days_per_year: int = 252) -> float:
        """Calculates annualized standard deviation of returns."""
        if returns.empty:
            return 0.0
        return float(returns.std() * np.sqrt(trading_days_per_year))

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int | None = 252,
        on_error: float = np.nan,
    ) -> float:
        """Calculates annualized Sharpe Ratio.

        This is the canonical implementation — src/analytics/calculators/
        risk_reward_calculator.py and src/algorithms/metrics_mixin.py both
        delegate here instead of maintaining their own copy of the formula,
        after a same-session audit found three independently-maintained
        Sharpe implementations that could silently disagree (different
        risk-free-rate defaults, different NaN-vs-0.0 failure behavior).

        Args:
            trading_days_per_year: pass None to auto-infer the
                annualisation factor from `returns`' DatetimeIndex cadence
                (see infer_periods_per_year) instead of assuming daily
                bars — useful for callers measuring 15m/1h/1d returns.
            on_error: value returned when there's insufficient data or the
                excess-return std is zero/non-finite. Defaults to NaN
                (explicit "could not compute"); some callers prefer 0.0 to
                avoid propagating NaN through downstream aggregations —
                pass on_error=0.0 to preserve that behavior.
        """
        clean_returns = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_returns) < 2:
            return on_error
        periods = trading_days_per_year if trading_days_per_year is not None else infer_periods_per_year(clean_returns)
        periods = max(int(periods), 1)
        excess_returns = clean_returns - risk_free_rate / periods
        excess_std = excess_returns.std()
        if not np.isfinite(excess_std) or excess_std <= 1e-12:
            return on_error
        sharpe = excess_returns.mean() / excess_std * np.sqrt(periods)
        return float(sharpe) if np.isfinite(sharpe) else on_error

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series, risk_free_rate: float = 0.0, trading_days_per_year: int = 252
    ) -> float:
        """Calculates annualized Sortino Ratio (downside risk only)."""
        clean_returns = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_returns) < 2:
            return np.nan
        periods = max(int(trading_days_per_year), 1)
        target_return = risk_free_rate / periods
        downside_returns = clean_returns[clean_returns < target_return]
        if len(downside_returns) < 2:
            return np.nan
        downside_std = downside_returns.std() * np.sqrt(periods)
        if not np.isfinite(downside_std) or downside_std <= 1e-12:
            return np.nan
        annual_return = (1 + clean_returns.mean()) ** periods - 1
        sortino = (annual_return - risk_free_rate) / downside_std
        return float(sortino) if np.isfinite(sortino) else np.nan

    @staticmethod
    def calculate_drawdowns(equity_curve: pd.Series) -> pd.Series:
        """
        Calculates percentage drawdown series.
        Drawdowns are always <= 0 (negative for losses, 0 at new highs).
        """
        if equity_curve.empty:
            return pd.Series([], dtype=float)
        rolling_max = equity_curve.expanding(min_periods=1).max().shift(1)
        drawdowns = (equity_curve - rolling_max) / rolling_max
        # Ensure drawdowns are never positive (0 at new highs, negative for losses)
        return drawdowns.clip(upper=0)

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculates maximum percentage drawdown."""
        drawdowns = FinancialMetricsLibrary.calculate_drawdowns(equity_curve)
        return float(drawdowns.min()) if not drawdowns.empty else 0.0

    @staticmethod
    def calculate_underwater_duration(equity_curve: pd.Series) -> int:
        """Calculates maximum duration (in periods) spent underwater."""
        drawdowns = FinancialMetricsLibrary.calculate_drawdowns(equity_curve)
        is_underwater = drawdowns < 0
        if not is_underwater.any():
            return 0
        dd_groups = (is_underwater != is_underwater.shift()).cumsum()
        return int(is_underwater[is_underwater].groupby(dd_groups).size().max())

    @staticmethod
    def run_granger_causality(df: pd.DataFrame, target_col: str, predictor_col: str, maxlag: int = 5) -> dict[str, Any]:
        """Runs Granger causality test and checks for stationarity."""
        if target_col not in df.columns or predictor_col not in df.columns:
            return {"error": "Missing columns"}
        data = df[[target_col, predictor_col]].dropna()
        if len(data) < maxlag + 5:
            return {"error": "Insufficient data"}
        try:
            target_stationary = adfuller(data[target_col])[1] < 0.05
            predictor_stationary = adfuller(data[predictor_col])[1] < 0.05
            test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0]["ssr_ftest"][1], 4) for i in range(maxlag)]
            min_p = float(min(p_values))
            correlation = data[target_col].corr(data[predictor_col])
            return {
                "p_value": min_p,
                "is_causal": min_p < 0.05,
                "correlation": correlation,
                "target_stationary": target_stationary,
                "predictor_stationary": predictor_stationary,
                "is_spurious": abs(correlation) > 0.7 and min_p >= 0.05,
            }
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Виникла помилка: {e}")
            return {"error": str(e)}

    @staticmethod
    def get_var_forecast(df: pd.DataFrame, columns: list[str], steps: int = 5) -> pd.DataFrame:
        """Vector Auto Regression (VAR) baseline forecast."""
        data = df[columns].dropna()
        if len(data) < 20:
            return pd.DataFrame()
        try:
            model = VAR(data)
            results = model.fit(maxlags=min(len(data) // 5, 10))
            forecast = results.forecast(y=data.values[-results.k_ar :], steps=steps)
            return pd.DataFrame(
                forecast,
                columns=columns,
                index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps),
            )
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"VAR forecast failed: {e}")
            return pd.DataFrame()

    @staticmethod
    def calculate_calmar_ratio(equity_curve: pd.Series, trading_days_per_year: int = 252) -> float:
        """Calculates Calmar Ratio (Annualized Return / Max Drawdown)."""
        cagr = FinancialMetricsLibrary.calculate_cagr(equity_curve, trading_days_per_year)
        mdd = abs(FinancialMetricsLibrary.calculate_max_drawdown(equity_curve))
        return float(cagr / mdd) if mdd > 0 else 0.0

    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculates the Beta of an asset relative to the market."""
        common = asset_returns.dropna().index.intersection(market_returns.dropna().index)
        if len(common) < 5:
            return 0.0
        a_ret = asset_returns.loc[common]
        m_ret = market_returns.loc[common]
        m_var = m_ret.var()
        if m_var == 0:
            return 0.0
        return float(a_ret.cov(m_ret) / m_var)

    @staticmethod
    def calculate_treynor_ratio(
        asset_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int = 252,
    ) -> float:
        """Calculates the Treynor Ratio."""
        beta = FinancialMetricsLibrary.calculate_beta(asset_returns, market_returns)
        if beta == 0:
            return 0.0
        annual_excess_return = asset_returns.mean() * trading_days_per_year - risk_free_rate
        return float(annual_excess_return / beta)

    @staticmethod
    def calculate_var_cvar(returns: pd.Series, confidence_level: float = 0.95) -> dict[str, Any]:
        """Calculates loss-positive VaR/CVaR plus raw return thresholds."""
        clean_returns = pd.Series(returns, dtype=float).dropna()
        if clean_returns.empty:
            return {"var": np.nan, "cvar": np.nan, "status": "insufficient_data"}
        quantile = 1 - confidence_level
        var_return_threshold = clean_returns.quantile(quantile)  # audit-ignore: VAR_SIGN_OR_EMPTY_DATA_REVIEW
        tail_returns = clean_returns[clean_returns <= var_return_threshold]
        cvar_return_threshold = tail_returns.mean()
        var_loss_positive = max(0.0, float(-var_return_threshold))
        cvar_loss_positive = max(0.0, float(-cvar_return_threshold))
        return {
            "var": var_loss_positive,
            "cvar": cvar_loss_positive,
            "var_return_threshold": float(var_return_threshold),
            "cvar_return_threshold": float(cvar_return_threshold),
            "confidence_level": float(confidence_level),
            "status": "ok",
        }

    @staticmethod
    def calculate_information_ratio(
        asset_returns: pd.Series, benchmark_returns: pd.Series, trading_days_per_year: int = 252
    ) -> float:
        """Calculates the annualized Information Ratio."""
        if asset_returns.empty or benchmark_returns.empty:
            return np.nan
        active_returns = pd.Series(asset_returns - benchmark_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(active_returns) < 2:
            return np.nan
        tracking_error = active_returns.std()
        if not np.isfinite(tracking_error) or tracking_error <= 1e-12:
            return np.nan
        ir = active_returns.mean() / tracking_error
        annualized_ir = ir * np.sqrt(max(int(trading_days_per_year), 1))
        return float(annualized_ir) if np.isfinite(annualized_ir) else np.nan
