from typing import Any

import pandas as pd

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

from .base import BaseBaseline

logger = ProjectLogger.get_logger("BaselineStrategies")


class BuyAndHoldBaseline(BaseBaseline):
    """Стратегія 'Купуй та тримай'."""

    def __init__(self, complexity_score: float=0.1):
        super().__init__(complexity_score)

    def train_and_evaluate(self, market_data: pd.DataFrame, features_df:
        pd.DataFrame | None=None, target_series: pd.Series | None=None
        ) ->dict[str, Any]:
        try:
            if target_series is not None:
                predictions = target_series.shift(1).fillna(target_series.mean()
                    )
                actual = target_series
            elif 'close' in market_data.columns:
                prices = market_data['close']
                predictions = prices.shift(1).fillna(prices.mean())
                actual = prices
            else:
                return {'model_type': 'buy_and_hold', 'status': 'no_data'}
            return {'model_type': 'buy_and_hold', 'predictions':
                predictions, 'metrics': self._calculate_metrics(actual,
                predictions), 'complexity_score': self.complexity_score}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error in BuyAndHoldBaseline: {e}", exc_info=True)
            raise DataProcessingError(f"BuyAndHoldBaseline evaluation failed: {e}") from e


class MovingAverageBaseline(BaseBaseline):
    """Стратегія на основі ковзного середнього."""

    def __init__(self, complexity_score: float=0.5, windows: list[int] | None=None):
        super().__init__(complexity_score)
        self.windows = windows or [5, 10, 20]

    def train_and_evaluate(self, market_data: pd.DataFrame, features_df:
        pd.DataFrame | None=None, target_series: pd.Series | None=None
        ) ->dict[str, Any]:
        try:
            if 'close' not in market_data.columns:
                return {'model_type': 'moving_average', 'status': 'no_data'}
            prices = market_data['close']
            best_score = float('inf')
            best_predictions = None
            best_window = None
            actual = target_series if target_series is not None else prices
            for window in self.windows:
                if len(prices) >= window:
                    ma = prices.rolling(window=window, min_periods=1).mean()
                    predictions = ma.shift(1).fillna(ma.mean())
                    from sklearn.metrics import mean_squared_error
                    mse = mean_squared_error(actual, predictions)
                    if mse < best_score:
                        best_score = mse
                        best_predictions = predictions
                        best_window = window
            if best_predictions is not None:
                return {'model_type': 'moving_average', 'predictions':
                    best_predictions, 'metrics': self._calculate_metrics(
                    actual, best_predictions), 'complexity_score': self.
                    complexity_score, 'best_window': best_window}
            return {'model_type': 'moving_average', 'status': 'no_data'}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error in MovingAverageBaseline: {e}", exc_info=True)
            raise DataProcessingError(f"MovingAverageBaseline evaluation failed: {e}") from e


class MeanReversionBaseline(BaseBaseline):
    """Стратегія повернення до середнього."""

    def __init__(self, complexity_score: float=2.0, lookback_window: int=20):
        super().__init__(complexity_score)
        self.lookback_window = lookback_window

    def train_and_evaluate(self, market_data: pd.DataFrame, features_df:
        pd.DataFrame | None=None, target_series: pd.Series | None=None
        ) ->dict[str, Any]:
        try:
            if 'close' not in market_data.columns:
                return {'model_type': 'mean_reversion', 'status': 'no_data'}
            prices = market_data['close']
            if len(prices) < self.lookback_window:
                return {'model_type': 'mean_reversion', 'status': 'no_data'}
            mean_price = prices.rolling(window=self.lookback_window,
                min_periods=1).mean()
            price_diff = prices - mean_price
            reversion_factor = 0.5
            predictions = prices - reversion_factor * price_diff
            actual = target_series if target_series is not None else prices
            return {'model_type': 'mean_reversion', 'predictions':
                predictions, 'metrics': self._calculate_metrics(actual,
                predictions), 'complexity_score': self.complexity_score,
                'lookback_window': self.lookback_window}
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error in MeanReversionBaseline: {e}", exc_info=True)
            raise DataProcessingError(f"MeanReversionBaseline evaluation failed: {e}") from e
