
import numpy as np

from src.core.logging.logger import ProjectLogger


class RegimeMetricsCalculator:
    """Розрахунок базових ринкових метрик."""
    logger = ProjectLogger.get_logger(__name__)

    @staticmethod
    def calculate_basic_metrics(returns: np.ndarray) ->tuple[float, float,
        float]:
        """Розраховує волатильність, середню дохідність та ADX."""
        volatility = float(np.std(returns))
        mean_return = float(np.mean(returns))
        adx = RegimeMetricsCalculator._calculate_adx(returns)
        return volatility, mean_return, adx

    @staticmethod
    def _calculate_adx(returns: np.ndarray, period: int=14) ->float:
        """Розрахунок ADX."""
        if len(returns) < period + 1:
            return 0.0
        try:
            up_move = np.maximum(returns, 0)
            down_move = np.maximum(-returns, 0)
            tr = np.maximum(up_move, down_move)
            tr_sum = np.sum(tr[-period:])
            if tr_sum == 0:
                return 0.0
            di_plus = 100 * np.sum(up_move[-period:]) / tr_sum
            di_minus = 100 * np.sum(down_move[-period:]) / tr_sum
            adx = abs(di_plus - di_minus)
            return float(adx)
        except Exception as e:
            RegimeMetricsCalculator.logger.error(f'Помилка розрахунку ADX: {e}'
                , exc_info=True)
            return 0.0

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int=14) ->float:
        """Розрахунок RSI."""
        if len(prices) < period + 1:
            return 50.0
        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        avg_gain = np.mean(gains[-period:]) if gains else 0
        avg_loss = np.mean(losses[-period:]) if losses else 0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = float(100 - 100 / (1 + rs))
        return float(np.clip(rsi, 0, 100))

    @staticmethod
    def calculate_z_score(returns: np.ndarray) ->float:
        """Розрахунок Z-score для mean reversion."""
        if len(returns) < 20:
            return 0.0
        recent_returns = returns[-20:]
        std = np.std(recent_returns)
        if std == 0:
            return 0.0
        z_score = (recent_returns[-1] - np.mean(recent_returns)) / std
        return float(z_score)
