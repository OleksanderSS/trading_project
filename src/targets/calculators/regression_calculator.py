import logging

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

# src/targets/calculators/regression_calculator.py
"""
Regression Calculator Module.
Computes future percentage returns for asset labels.
Supports transaction cost adjustments to ensure model training accounts for market friction.
"""

logger = ProjectLogger.get_logger("RegressionCalculator")

class RegressionCalculator:
    """
    Calculates regression targets based on normalized future returns.
    """

    def calculate(self, df: pd.DataFrame, base_col: str, shift: int, **kwargs) -> pd.Series:
        """
        Calculates the future percentage return relative to the current timestamp.

        Args:
            df (pd.DataFrame): Input market data.
            base_col (str): Source column for calculations (typically 'close').
            shift (int): Lookahead horizon (must be negative for future values).
            adjust_for_costs (bool): If True, subtracts estimated friction from the targets.
            transaction_costs (dict): Configuration containing commission, spread, and slippage percentages.

        Returns:
            pd.Series: Vector of calculated future returns.
        """
        if base_col not in df.columns:
            logger.error(f"Integrity Error: Mapping column '{base_col}' is absent from the input DataFrame.")
            raise ValueError(f"Mapping column '{base_col}' not found.")

        if shift >= 0:
            raise ValueError(f"shift must be negative for future targets, got {shift}")

        method = kwargs.get('method')
        if method:
            return self._calculate_by_method(df, base_col, shift, method, kwargs)

        # Standard lookahead return: (Price[T+n] - Price[T]) / Price[T]
        # Shift per-ticker so a multi-ticker frame never leaks the next
        # ticker's price into the previous ticker's future-return target.
        # TargetOrchestrator already groups by ticker before calling this,
        # but the calculator must not depend on that -- a caller passing a
        # multi-ticker frame directly would otherwise silently leak.
        if "ticker" in df.columns:
            future_price = df.groupby("ticker")[base_col].shift(shift)
        else:
            future_price = df[base_col].shift(shift)
        target_series = (future_price - df[base_col]) / df[base_col]

        # TRANSACTION COST ADJUSTMENT
        # Subtracting friction (margin) from the target forces the model to ignore
        # signals where the predicted reward doesn't cover execution overhead.
        adjust_for_costs = kwargs.get('adjust_for_costs', False)
        transaction_costs = kwargs.get('transaction_costs', {})

        if adjust_for_costs and transaction_costs:
            commission_pct = transaction_costs.get('commission_pct', 0.0)
            spread_pct = transaction_costs.get('spread_pct', 0.0)
            slippage_pct = transaction_costs.get('slippage_pct', 0.0)

            # Total estimated round-trip overhead (buy sequence + sell sequence)
            total_cost = (commission_pct + spread_pct + slippage_pct) * 2

            # Prune target returns by the cost factor
            target_series = target_series - total_cost

            logger.info(f"Target Sanitization: Adjusted for round-trip friction ({total_cost:.4%})")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Breakdown: Comm={commission_pct:.4%}, Spread={spread_pct:.4%}, Slip={slippage_pct:.4%}")

        return target_series

    # ------------------------------------------------------------------
    # Forward-window methods
    #
    # `method` and `window` used to be accepted and silently dropped, so
    # target_daily_trend_strength_1d and target_daily_momentum_score_1d both
    # collapsed to a plain next-bar return -- byte-identical to each other,
    # and differing from target_return_1d only by its cost constant. Three
    # configured targets carrying one signal. These implement what the
    # config always claimed.
    #
    # All three look FORWARD from `shift` over `window` bars, so they are
    # genuine targets, never features. Grouped per ticker so a multi-ticker
    # frame cannot leak across boundaries.
    # ------------------------------------------------------------------

    _METHODS = ("slope_strength", "rate_of_change", "high_low_range")

    def _calculate_by_method(self, df: pd.DataFrame, base_col: str, shift: int,
                            method: str, params: dict) -> pd.Series:
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown regression target method '{method}'. "
                f"Supported: {', '.join(self._METHODS)}."
            )
        window = int(params.get('window', 20))
        if window < 2:
            raise ValueError(f"window must be >= 2 for method '{method}', got {window}")

        if method == "high_low_range":
            missing = [c for c in ("high", "low") if c not in df.columns]
            if missing:
                raise ValueError(
                    f"method 'high_low_range' needs columns {missing}, which are absent."
                )

        def per_ticker(group: pd.DataFrame) -> pd.Series:
            return self._forward_metric(group, base_col, shift, method, window)

        if "ticker" in df.columns:
            # Select only the columns the metric needs, so the grouping column
            # is never handed to apply() (pandas 2.2 deprecates that).
            needed = ["high", "low", "close"] if method == "high_low_range" else [base_col]
            needed = [c for c in dict.fromkeys(needed) if c in df.columns]
            out = df.groupby("ticker", group_keys=False)[needed].apply(per_ticker)
            return out.reindex(df.index)
        return self._forward_metric(df, base_col, shift, method, window)

    @staticmethod
    def _forward_metric(g: pd.DataFrame, base_col: str, shift: int,
                        method: str, window: int) -> pd.Series:
        """Compute the metric over the `window` bars starting at `shift`."""
        if method == "high_low_range":
            # Realized volatility proxy: mean bar range relative to close.
            bar_range = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)
            # rolling().mean() looks BACKWARD; shifting by -(window-1) turns it
            # into the forward window, then `shift` moves its start point.
            forward = bar_range.rolling(window, min_periods=window).mean().shift(
                -(window - 1)
            )
            return forward.shift(shift + 1)

        series = g[base_col].astype(float)

        if method == "rate_of_change":
            start = series.shift(shift)
            end = series.shift(shift - (window - 1))
            return (end - start) / start.replace(0, np.nan)

        # slope_strength: R^2 of an OLS fit of the forward window. Naturally
        # in [0, 1] -- "how trend-like is the next stretch", which is what the
        # config's "score (0-1)" describes. Magnitude of the move is already
        # covered by rate_of_change, so this is deliberately unsigned.
        x = np.arange(window, dtype=float)
        x_centered = x - x.mean()
        denom_x = float((x_centered ** 2).sum())

        def r_squared(values: np.ndarray) -> float:
            y = np.asarray(values, dtype=float)
            if np.isnan(y).any():
                return np.nan
            y_centered = y - y.mean()
            ss_tot = float((y_centered ** 2).sum())
            if ss_tot <= 0.0 or denom_x <= 0.0:
                return 0.0
            cov = float((x_centered * y_centered).sum())
            return float(min(1.0, max(0.0, (cov ** 2) / (denom_x * ss_tot))))

        backward = series.rolling(window, min_periods=window).apply(r_squared, raw=True)
        forward = backward.shift(-(window - 1))
        return forward.shift(shift + 1)
