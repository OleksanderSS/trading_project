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

    #: Params this calculator honours, declared explicitly so
    #: TargetOrchestrator can flag config keys nothing reads. Declared rather
    #: than introspected: params consumed inside helper methods are invisible
    #: to a scan of `calculate`'s own source, and that produced false
    #: positives.
    SUPPORTED_PARAMS = frozenset({
        "base_col", "shift",
        "adjust_for_costs", "transaction_costs",
        "method", "window",
    })

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
            total_cost = self._round_trip_cost(df[base_col], transaction_costs)

            # Prune target returns by the cost factor
            target_series = target_series - total_cost

            shown = float(np.nanmean(total_cost)) if hasattr(total_cost, '__len__') \
                else float(total_cost)
            logger.info(f"Target Sanitization: Adjusted for round-trip friction ({shown:.4%})")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Cost config: {transaction_costs}")

        return target_series

    @staticmethod
    def _round_trip_cost(price: pd.Series, costs: dict):
        """Round-trip friction as a fraction of the position's value.

        Two models, and which one applies is the point of this method.

        `flat` is what this project assumed until 2026-08-17: one
        `commission_pct` for every trade. Measured against a real schedule it
        is wrong in both directions at once, because a broker charges per
        SHARE with a minimum per ORDER. IBKR Pro tiered is $0.0035/share, min
        $0.35: a $1,000 order pays 7 bp round trip and a $10,000 order in the
        same stock pays 0.8 bp, so the cost in basis points depends on the
        order value and on the share price. A $20 stock cannot go below 3.5 bp
        at any size; a $230 stock reaches 0.3 bp.

        `per_share` computes that per row, from the bar's own price.

        `spread_pct` and `slippage_pct` are PER SIDE, as they have always been
        here -- the total doubles them. A 2 bp quoted spread costs one
        half-spread per side, so `spread_pct: 0.0001`.

        Returns a scalar for the flat model and a per-row Series for
        `per_share`; both subtract correctly from the target series, which
        shares this frame's index.
        """
        spread = float(costs.get('spread_pct', 0.0) or 0.0)
        slippage = float(costs.get('slippage_pct', 0.0) or 0.0)
        model = str(costs.get('model', 'flat')).lower()

        if model == 'per_share':
            order_value = float(costs.get('order_value', 0.0) or 0.0)
            if order_value <= 0:
                raise ValueError(
                    "transaction_costs model 'per_share' requires a positive "
                    "order_value -- the cost in basis points is a function of "
                    "it, so a missing value has no safe default."
                )
            per_share = float(costs.get('per_share_fee', 0.0) or 0.0)
            min_fee = float(costs.get('min_fee_per_order', 0.0) or 0.0)
            max_pct = float(costs.get('max_fee_pct_of_order', 1.0) or 1.0)

            prices = pd.to_numeric(price, errors='coerce')
            shares = order_value / prices.where(prices > 0)
            fee = (per_share * shares).clip(lower=min_fee,
                                            upper=max_pct * order_value)
            # A bar with no usable price cannot size an order; charging the
            # per-order minimum is the conservative reading, and a NaN here
            # would silently void the whole target for that row.
            commission = (fee / order_value).fillna(min_fee / order_value)
        elif model == 'flat':
            commission = float(costs.get('commission_pct', 0.0) or 0.0)
        else:
            raise ValueError(
                f"Unknown transaction_costs model '{model}'. "
                f"Expected 'flat' or 'per_share'."
            )

        return (commission + spread + slippage) * 2

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

        if "ticker" not in df.columns:
            return self._forward_metric(df, base_col, shift, method, window)

        # Assemble per ticker explicitly rather than via groupby().apply().
        # apply() is ambiguous here: with several groups it concatenates the
        # returned Series, but with a SINGLE group it treats the Series as a
        # row and hands back a 1xN DataFrame, which then reindexes into an NxN
        # square. That is the real call path -- TargetOrchestrator
        # ._process_by_ticker_groups already splits by ticker, so this
        # calculator normally sees one-ticker frames.
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for _, idx in df.groupby("ticker", sort=False).groups.items():
            values = self._forward_metric(
                df.loc[idx], base_col, shift, method, window
            )
            out.loc[idx] = pd.Series(values, index=idx).astype(float)
        return out

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
