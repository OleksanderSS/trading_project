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

logger = ProjectLogger.get_logger("FinancialMetricsLibrary")

class FinancialMetricsLibrary:
    """
    Core library for financial calculations.
    Stateless methods for maximum reusability.
    """

    # === PnL & Return Metrics ===

    @staticmethod
    def calculate_total_return(equity_curve: pd.Series) -> float:
        """Calculates total percentage return."""
        if equity_curve.empty: return 0.0
        return float((equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0])

    @staticmethod
    def calculate_cagr(equity_curve: pd.Series, trading_days_per_year: int = 252) -> float:
        """Calculates Compound Annual Growth Rate."""
        if equity_curve.empty: return 0.0
        years = len(equity_curve) / trading_days_per_year
        if years <= 0: return 0.0
        total_return_factor = equity_curve.iloc[-1] / equity_curve.iloc[0]
        return float(total_return_factor ** (1 / years) - 1)

    # === Risk & Volatility Metrics ===

    @staticmethod
    def calculate_annualized_volatility(returns: pd.Series, trading_days_per_year: int = 252) -> float:
        """Calculates annualized standard deviation of returns."""
        if returns.empty: return 0.0
        return float(returns.std() * np.sqrt(trading_days_per_year))

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, trading_days_per_year: int = 252) -> float:
        """Calculates annualized Sharpe Ratio."""
        if returns.empty or returns.std() == 0: return 0.0
        excess_returns = returns - (risk_free_rate / trading_days_per_year)
        return float((excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days_per_year))

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, trading_days_per_year: int = 252) -> float:
        """Calculates annualized Sortino Ratio (downside risk only)."""
        if returns.empty: return 0.0
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(trading_days_per_year)
        if downside_std == 0: return 0.0

        annual_return = (1 + returns.mean()) ** trading_days_per_year - 1
        return float((annual_return - risk_free_rate) / downside_std)

    # === Drawdown Metrics ===

    @staticmethod
    def calculate_drawdowns(equity_curve: pd.Series) -> pd.Series:
        """Calculates percentage drawdown series."""
        if equity_curve.empty: return pd.Series([], dtype=float)
        rolling_max = equity_curve.expanding(min_periods=1).max()
        return (equity_curve - rolling_max) / rolling_max

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
        if not is_underwater.any(): return 0

        # Calculate blocks of underwater periods
        dd_groups = (is_underwater != is_underwater.shift()).cumsum()
        return int(is_underwater[is_underwater].groupby(dd_groups).size().max())

    # === Econometric & Statistical Tests ===

    @staticmethod
    def run_granger_causality(df: pd.DataFrame, target_col: str, predictor_col: str, maxlag: int = 5) -> dict[str, Any]:
        """Runs Granger causality test and checks for stationarity."""
        if target_col not in df.columns or predictor_col not in df.columns:
            return {"error": "Missing columns"}

        data = df[[target_col, predictor_col]].dropna()
        if len(data) < maxlag + 5:
            return {"error": "Insufficient data"}

        try:
            # ADF Stationarity Check
            target_stationary = adfuller(data[target_col])[1] < 0.05
            predictor_stationary = adfuller(data[predictor_col])[1] < 0.05

            # Granger Test
            test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(maxlag)]
            min_p = float(min(p_values))

            correlation = data[target_col].corr(data[predictor_col])

            return {
                "p_value": min_p,
                "is_causal": min_p < 0.05,
                "correlation": correlation,
                "target_stationary": target_stationary,
                "predictor_stationary": predictor_stationary,
                "is_spurious": abs(correlation) > 0.7 and min_p >= 0.05
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_var_forecast(df: pd.DataFrame, columns: list[str], steps: int = 5) -> pd.DataFrame:
        """Vector Auto Regression (VAR) baseline forecast."""
        data = df[columns].dropna()
        if len(data) < 20: return pd.DataFrame()

        try:
            model = VAR(data)
            results = model.fit(maxlags=min(len(data)//5, 10))
            forecast = results.forecast(y=data.values[-results.k_ar:], steps=steps)

            return pd.DataFrame(
                forecast,
                columns=columns,
                index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps)
            )
        except Exception as e:
            logger.error(f"VAR forecast failed: {e}")
            return pd.DataFrame()

    # === Portfolio & Performance ===

    @staticmethod
    def calculate_calmar_ratio(equity_curve: pd.Series, trading_days_per_year: int = 252) -> float:
        """Calculates Calmar Ratio (Annualized Return / Max Drawdown)."""
        cagr = FinancialMetricsLibrary.calculate_cagr(equity_curve, trading_days_per_year)
        mdd = abs(FinancialMetricsLibrary.calculate_max_drawdown(equity_curve))
        return float(cagr / mdd) if mdd > 0 else 0.0

    # === Advanced Risk & Portfolio Metrics ===

    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculates the Beta of an asset relative to the market."""
        common = asset_returns.dropna().index.intersection(market_returns.dropna().index)
        if len(common) < 5: return 0.0

        a_ret = asset_returns.loc[common]
        m_ret = market_returns.loc[common]

        m_var = m_ret.var()
        if m_var == 0: return 0.0
        return float(a_ret.cov(m_ret) / m_var)

    @staticmethod
    def calculate_treynor_ratio(asset_returns: pd.Series, market_returns: pd.Series,
                               risk_free_rate: float = 0.0, trading_days_per_year: int = 252) -> float:
        """Calculates the Treynor Ratio."""
        beta = FinancialMetricsLibrary.calculate_beta(asset_returns, market_returns)
        if beta == 0: return 0.0

        annual_excess_return = (asset_returns.mean() * trading_days_per_year) - risk_free_rate
        return float(annual_excess_return / beta)

    @staticmethod
    def calculate_var_cvar(returns: pd.Series, confidence_level: float = 0.95) -> dict[str, float]:
        """Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR)."""
        if returns.empty: return {'var': 0.0, 'cvar': 0.0}

        quantile = 1 - confidence_level
        var = returns.quantile(quantile)
        cvar = returns[returns <= var].mean()
        return {'var': float(var), 'cvar': float(cvar)}

    @staticmethod
    def calculate_information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series,
                                   trading_days_per_year: int = 252) -> float:
        """Calculates the annualized Information Ratio."""
        if asset_returns.empty or benchmark_returns.empty: return 0.0

        active_returns = asset_returns - benchmark_returns
        tracking_error = active_returns.std()
        if tracking_error == 0: return 0.0

        ir = active_returns.mean() / tracking_error
        return float(ir * np.sqrt(trading_days_per_year))
