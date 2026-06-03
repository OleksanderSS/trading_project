from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("BiasDetector")


class BiasDetector:
    """Detect common backtest biases such as look-ahead and survivorship bias."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = logger

    def detect_look_ahead_bias(
        self,
        signals: pd.DataFrame | pd.Series,
        future_prices: pd.DataFrame | pd.Series,
        threshold: float = 0.9,
        lag_periods: int = 1,
    ) -> dict[str, Any]:
        """Check whether current signals are suspiciously correlated with future returns."""
        try:
            signals_df = self._as_numeric_frame(signals, "signal")
            prices_df = self._as_numeric_frame(future_prices, "price")
            future_returns = self._future_returns(prices_df, lag_periods)
            pairs = self._matching_column_pairs(signals_df, future_returns)

            correlations: dict[str, float] = {}
            suspicious_signals: list[dict[str, Any]] = []
            for signal_col, return_col in pairs:
                aligned = pd.concat(
                    [signals_df[signal_col], future_returns[return_col]],
                    axis=1,
                    join="inner",
                ).dropna()
                if len(aligned) < 3:
                    continue
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                if pd.isna(corr):
                    continue
                key = signal_col if signal_col == return_col else f"{signal_col}->{return_col}"
                correlations[key] = float(corr)
                if abs(corr) >= threshold:
                    suspicious_signals.append(
                        {
                            "signal": signal_col,
                            "target": return_col,
                            "correlation": float(corr),
                            "threshold": threshold,
                        }
                    )

            detected = bool(suspicious_signals)
            return {
                "lookahead_bias_detected": detected,
                "has_look_ahead_bias": detected,
                "correlations": correlations,
                "suspicious_signals": suspicious_signals,
                "threshold": threshold,
                "lag_periods": lag_periods,
            }
        except Exception as e:
            self.logger.error(f"Look-ahead bias detection failed: {e}", exc_info=True)
            return {
                "lookahead_bias_detected": False,
                "has_look_ahead_bias": False,
                "correlations": {},
                "suspicious_signals": [],
                "error": str(e),
            }

    def detect_survivorship_bias(
        self,
        historical_universe: list[str],
        current_universe: list[str],
    ) -> dict[str, Any]:
        """Compare historical and current asset universes for missing assets."""
        historical_set = set(historical_universe)
        current_set = set(current_universe)
        missing_assets = sorted(historical_set - current_set)
        missing_count = len(missing_assets)
        bias_score = missing_count / len(historical_set) if historical_set else 0.0
        return {
            "survivorship_bias_score": bias_score,
            "missing_assets_count": missing_count,
            "missing_assets": missing_assets,
            "potential_bias": bias_score > 0.1,
        }

    @staticmethod
    def _as_numeric_frame(data: pd.DataFrame | pd.Series, default_name: str) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            return data.rename(data.name or default_name).to_frame()
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame or Series")
        numeric = data.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("No numeric columns available for bias detection")
        return numeric

    @staticmethod
    def _future_returns(prices: pd.DataFrame, lag_periods: int) -> pd.DataFrame:
        lag = max(1, int(lag_periods))
        returns = prices.pct_change(periods=lag, fill_method=None).shift(-lag)
        returns = returns.replace([np.inf, -np.inf], np.nan)
        if returns.dropna(how="all").empty:
            return prices.shift(-lag)
        return returns

    @staticmethod
    def _matching_column_pairs(signals: pd.DataFrame, returns: pd.DataFrame) -> list[tuple[str, str]]:
        common_cols = [col for col in signals.columns if col in returns.columns]
        if common_cols:
            return [(col, col) for col in common_cols]
        if len(signals.columns) == 1 and len(returns.columns) == 1:
            return [(signals.columns[0], returns.columns[0])]
        return [(sig, ret) for sig in signals.columns for ret in returns.columns]
