import logging

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
