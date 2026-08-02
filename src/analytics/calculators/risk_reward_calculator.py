import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.utils.technical_indicators_lib import TechnicalIndicators
from src.metrics.financial.financial_metrics_library import FinancialMetricsLibrary

logger = logging.getLogger(__name__)

#: Floor for an ATR-derived stop or target, as a fraction of entry price.
#: 1% of entry, so a stop can imply risking at most ~99% of the position --
#: still extreme, but a price rather than an impossibility.
MINIMUM_PRICE_FRACTION = 0.01


@dataclass
class TradeConfig:
    """Configuration for trade parameter calculations."""
    atr_multiplier: float = 2.0
    tp_multiplier: float = 3.0
    risk_free_rate: float = 0.0
    periods_per_year: int = 252
    confidence_level: float = 0.95

@dataclass
class TradeParameters:
    """Parameters for a specific trade."""
    df: pd.DataFrame
    signal_type: str
    entry_price: float

class RiskRewardCalculator:
    """A collection of static methods for calculating standard risk-reward metrics and dynamic trade parameters."""

    @staticmethod
    def _clean_return_series(returns: pd.Series) -> pd.Series:
        """Return finite numeric observations only."""
        return pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def calculate_trade_parameters(trade_params: TradeParameters, config: TradeConfig | None = None) -> dict[str, float]:
        """
        Calculates dynamic Stop Loss and Take Profit levels based on market volatility (ATR).

        Args:
            trade_params (TradeParameters): Trade parameters including data, signal type, and entry price.
            config (Optional[TradeConfig]): Configuration for multipliers and settings.

        Returns:
            Dict[str, float]: Dictionary containing stop_loss, take_profit, and risk_reward_ratio.
        """
        if config is None:
            config = TradeConfig()

        return RiskRewardCalculator._calculate_trade_with_config(trade_params, config)

    @staticmethod
    def _calculate_trade_with_config(trade_params: TradeParameters, config: TradeConfig) -> dict[str, float]:
        """Calculate trade parameters with given configuration."""
        # VolatilityCalculator has no calculate_atr -- this raised
        # AttributeError on every call, so the stop-loss/take-profit
        # machinery could not be used at all. The real implementation is in
        # TechnicalIndicators, with a per-series signature.
        frame = trade_params.df
        atr_series = TechnicalIndicators.calculate_atr(
            frame['high'], frame['low'], frame['close'], period=14
        )
        atr = atr_series.iloc[-1] if len(atr_series) else float('nan')

        if pd.isna(atr) or atr <= 0:
            atr = trade_params.entry_price * 0.01 # Fallback to 1% of price

        risk = atr * config.atr_multiplier
        entry = trade_params.entry_price

        if trade_params.signal_type == 'BUY':
            sl = entry - risk
            tp = entry + (risk * config.tp_multiplier)
        elif trade_params.signal_type == 'SELL':
            sl = entry + risk
            tp = entry - (risk * config.tp_multiplier)
        else:
            return {'stop_loss': 0.0, 'take_profit': 0.0, 'risk_reward_ratio': 0.0}

        # Nothing kept these above zero. A BUY whose ATR is large relative to
        # price -- a low-priced or violently moving instrument -- produced a
        # NEGATIVE stop loss, and a SELL produced a negative take profit.
        # Neither is a price. Worse, the negative number is not obviously
        # wrong downstream: it flows into risk sizing as a stop "distance"
        # wider than the entire position is worth.
        #
        # Floored at a small positive fraction of entry rather than at zero:
        # a stop AT zero implies risking 100% of the position, which is not
        # meaningfully better than a negative one.
        floor = entry * MINIMUM_PRICE_FRACTION
        clamped = False
        if sl <= floor:
            sl, clamped = floor, True
        if tp <= floor:
            tp, clamped = floor, True

        if clamped:
            logger.warning(
                "ATR-derived level fell to or below zero for a %s at %.4f "
                "(ATR %.4f x %.1f = %.4f risk); clamped to %.4f. The "
                "volatility estimate is large relative to the price, so the "
                "reward:risk below is the CLAMPED ratio, not %.1f.",
                trade_params.signal_type, entry, atr, config.atr_multiplier,
                risk, floor, config.tp_multiplier,
            )

        # Derived from the final levels, not from the multipliers. Before the
        # clamp above this was tautological -- reward is risk*tp_multiplier
        # and risk is risk, so the ratio was exactly tp_multiplier (3.0) for
        # every trade ever evaluated, and anything filtering on it was
        # filtering on a constant. It now differs from tp_multiplier exactly
        # when a level had to be clamped, which is the case worth seeing.
        reward_distance = abs(tp - entry)
        risk_distance = abs(entry - sl)
        rr_ratio = reward_distance / risk_distance if risk_distance != 0 else 0.0

        return {
            'stop_loss': float(sl),
            'take_profit': float(tp),
            'risk_reward_ratio': float(rr_ratio),
            'risk_amount': float(risk_distance),
            # The caller cannot otherwise tell a 3.0 that was designed from a
            # 3.0 that survived a clamp.
            'levels_clamped': clamped,
        }

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, config: TradeConfig | None = None) -> float:
        """Calculates the annualized Sharpe Ratio.

        Delegates to FinancialMetricsLibrary.calculate_sharpe_ratio (the
        canonical implementation) — this wrapper only exists to keep the
        TradeConfig-based call signature existing callers already use.
        Behavior is unchanged: same formula, same defaults
        (risk_free_rate=0.0, periods_per_year=252 via TradeConfig), NaN on
        insufficient data or zero/non-finite excess-return std.
        """
        if config is None:
            config = TradeConfig()
        return FinancialMetricsLibrary.calculate_sharpe_ratio(
            returns,
            risk_free_rate=config.risk_free_rate,
            trading_days_per_year=config.periods_per_year,
        )

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, config: TradeConfig | None = None) -> float:
        """Calculates the annualized Sortino Ratio.

        Delegates to FinancialMetricsLibrary.calculate_sortino_ratio (the
        canonical implementation), exactly as calculate_sharpe_ratio does.

        This used to carry its own formula, dividing by the standard deviation
        of the losing subset rather than the downside deviation
        `sqrt(mean(min(0, r - target)^2))`. Measured against the definition it
        overstated by 1.151x / 1.175x / 1.277x on three samples -- and it
        overstated MOST on downside-skewed returns, i.e. precisely the case
        Sortino exists to penalise. Three separate Sortino implementations
        existed in this codebase and all three disagreed; Sharpe had already
        been consolidated for the same reason.
        """
        if config is None:
            config = TradeConfig()
        return FinancialMetricsLibrary.calculate_sortino_ratio(
            returns,
            risk_free_rate=config.risk_free_rate,
            trading_days_per_year=config.periods_per_year,
        )

    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculates the Beta of an asset relative to the market."""
        common_index = asset_returns.dropna().index.intersection(market_returns.dropna().index)
        if len(common_index) < 2:
            return np.nan

        asset_returns = asset_returns.loc[common_index]
        market_returns = market_returns.loc[common_index]

        market_variance = market_returns.var()
        if market_variance == 0:
            return np.nan

        covariance = asset_returns.cov(market_returns)
        beta = covariance / market_variance
        return float(beta)

    @staticmethod
    def calculate_treynor_ratio(asset_returns: pd.Series, market_returns: pd.Series, config: TradeConfig | None = None) -> float:
        """Calculates the Treynor Ratio."""
        if config is None:
            config = TradeConfig()

        beta = RiskRewardCalculator.calculate_beta(asset_returns, market_returns)
        if beta == 0 or pd.isna(beta):
            return np.nan

        excess_return = (asset_returns.mean() * config.periods_per_year) - config.risk_free_rate
        treynor_ratio = excess_return / beta
        return float(treynor_ratio)

    @staticmethod
    def calculate_var_cvar(returns: pd.Series, config: TradeConfig | None = None) -> dict:
        """Calculates loss-positive VaR/CVaR plus raw return thresholds."""
        if config is None:
            config = TradeConfig()

        clean_returns = pd.Series(returns, dtype=float).dropna()
        if clean_returns.empty:
            return {'var': np.nan, 'cvar': np.nan, 'status': 'insufficient_data'}

        quantile = 1 - config.confidence_level
        var_return_threshold = clean_returns.quantile(quantile)  # audit-ignore: VAR_SIGN_OR_EMPTY_DATA_REVIEW
        tail_returns = clean_returns[clean_returns <= var_return_threshold]
        cvar_return_threshold = tail_returns.mean()
        var_loss_positive = max(0.0, float(-var_return_threshold))
        cvar_loss_positive = max(0.0, float(-cvar_return_threshold))
        return {
            'var': var_loss_positive,
            'cvar': cvar_loss_positive,
            'var_return_threshold': float(var_return_threshold),
            'cvar_return_threshold': float(cvar_return_threshold),
            'confidence_level': float(config.confidence_level),
            'status': 'ok',
        }

    @staticmethod
    def calculate_information_ratio(asset_returns: pd.Series, benchmark_returns: pd.Series, config: TradeConfig | None = None) -> float:
        """Calculates the annualized Information Ratio."""
        if config is None:
            config = TradeConfig()

        active_returns = RiskRewardCalculator._clean_return_series(asset_returns - benchmark_returns)
        if len(active_returns) < 2:
            return np.nan

        tracking_error = active_returns.std()

        if not np.isfinite(tracking_error) or tracking_error <= 1e-12:
            return np.nan

        information_ratio = active_returns.mean() / tracking_error
        annualized_ir = information_ratio * np.sqrt(max(int(config.periods_per_year), 1))
        return float(annualized_ir) if np.isfinite(annualized_ir) else np.nan
