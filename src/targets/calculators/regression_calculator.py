
import pandas as pd
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("RegressionCalculator")

class RegressionCalculator:
    """
    Calculates regression targets based on future returns.
    """
    def calculate(self, df: pd.DataFrame, base_col: str, shift: int, **kwargs) -> pd.Series:
        """
        Calculates the future percentage return.

        Args:
            df (pd.DataFrame): The input DataFrame.
            base_col (str): The column to use for calculation (e.g., 'close').
            shift (int): The number of periods to look into the future (should be negative).
            adjust_for_costs (bool): Whether to subtract transaction costs from returns.
            transaction_costs (dict): Dict with commission_pct, spread_pct, slippage_pct.

        Returns:
            pd.Series: A series with the calculated future returns.
        """
        if base_col not in df.columns:
            logger.error(f"Base column '{base_col}' not found in DataFrame.")
            raise ValueError(f"Base column '{base_col}' not found.")
            
        future_price = df[base_col].shift(shift)
        target_series = (future_price - df[base_col]) / df[base_col]
        
        # ✅ КРИТИЧНО: Віднімаємо маржу (transaction costs) з таргету
        # Це навчає модель враховувати реальні витрати на торгівлю
        adjust_for_costs = kwargs.get('adjust_for_costs', False)
        transaction_costs = kwargs.get('transaction_costs', {})
        
        if adjust_for_costs and transaction_costs:
            commission_pct = transaction_costs.get('commission_pct', 0.0)
            spread_pct = transaction_costs.get('spread_pct', 0.0)
            slippage_pct = transaction_costs.get('slippage_pct', 0.0)
            
            # Загальна маржа на round trip (buy + sell)
            total_cost = (commission_pct + spread_pct + slippage_pct) * 2
            
            # Віднімаємо маржу з таргету
            target_series = target_series - total_cost
            
            logger.info(f"✅ Adjusted target for transaction costs: {total_cost:.4%} per round trip")
            logger.debug(f"   Commission: {commission_pct:.4%}, Spread: {spread_pct:.4%}, Slippage: {slippage_pct:.4%}")
        
        return target_series
