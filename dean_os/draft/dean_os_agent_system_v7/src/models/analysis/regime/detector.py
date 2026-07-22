
import numpy as np
import pandas as pd

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class RegimeDetector:
    def __init__(self, regime_types):
        self.regime_types = regime_types

    def detect_regime(self, market_data: pd.DataFrame) -> str:
        """Визначає поточний ринковий режим."""
        try:
            volatility = self._calculate_volatility(market_data)
            trend = self._calculate_trend(market_data)
            for regime_name, regime_config in self.regime_types.items():
                vol_range = regime_config['volatility_range']
                trend_range = regime_config['trend_strength']
                if vol_range[0] <= volatility <= vol_range[1] and trend_range[0
                    ] <= trend <= trend_range[1]:
                    return regime_name
            return 'normal'
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error detecting regime: {e}")
            raise DataProcessingError(f"Regime detection failed: {e}") from e

    def _calculate_volatility(self, market_data: pd.DataFrame) ->float:
        """Розраховує волатильність ринку."""
        try:
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change(fill_method=None).dropna()
                return float(returns.std() * np.sqrt(252))
            price_cols = [col for col in market_data.columns if 'price' in
                col.lower() or col in ['open', 'high', 'low', 'close']]
            if price_cols:
                returns = market_data[price_cols[0]].pct_change(fill_method=None).dropna()
                return float(returns.std() * np.sqrt(252))
            return 0.0
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error calculating volatility: {e}")
            raise DataProcessingError(f"Volatility calculation failed: {e}") from e

    def _calculate_trend(self, market_data: pd.DataFrame) ->float:
        """Розраховує силу тренду ринку."""
        try:
            if 'close' in market_data.columns:
                recent_prices = market_data['close'].tail(20)
                if len(recent_prices) >= 2:
                    x = np.arange(len(recent_prices))
                    slope = np.polyfit(x, recent_prices, 1)[0]
                    normalized_trend = slope / recent_prices.mean()
                    return float(normalized_trend)
            return 0.0
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f"Error calculating trend: {e}")
            raise DataProcessingError(f"Trend calculation failed: {e}") from e
