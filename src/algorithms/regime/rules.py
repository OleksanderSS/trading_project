from typing import Any

import numpy as np

from src.algorithms.regime.metrics import RegimeMetricsCalculator
from src.algorithms.regime.types import MarketRegime
from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger


class RegimeRulesEngine:
    """Логіка виявлення режимів ринку на основі правил."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.adx_threshold = config.get('adx_threshold', 25)
        self.volatility_threshold_high = config.get('volatility_threshold_high'
            , 0.03)
        self.volatility_threshold_low = config.get('volatility_threshold_low',
            0.01)
        self.mean_reversion_threshold = config.get('mean_reversion_threshold',
            2.0)
        self.momentum_window = config.get('momentum_window', 20)
        self.momentum_threshold = config.get('momentum_threshold', 0.02)
        self.breakout_threshold = config.get('breakout_threshold', 0.05)
        self.breakout_volume_multiplier = config.get(
            'breakout_volume_multiplier', 1.5)

    def detect_regime_rules(self, metrics: Any) ->dict[str, Any]:
        """Rule-based regime detection."""
        try:
            mr = self._check_mean_reversion(metrics.returns)
            if mr:
                return mr
            mom = self._check_momentum(metrics.returns)
            if mom:
                return mom
            brk = self._check_breakout(metrics.prices, metrics.volume)
            if brk:
                return brk
            return self._detect_standard_regimes(metrics.adx, metrics.
                volatility, metrics.mean_return)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            if not hasattr(self, 'logger'):
                self.logger = ProjectLogger.get_logger('RegimeRulesEngine')
            self.logger.error('Rule-based detection failed', exc_info=True)
            raise DataProcessingError('Rule-based detection failed') from e

    def _check_mean_reversion(self, returns: np.ndarray) ->dict[str, Any] | None:
        if self._is_mean_reversion(returns):
            return {'regime': MarketRegime.MEAN_REVERSION.value,
                'confidence': 0.8, 'reason': 'statistical_mean_reversion',
                'metrics': {'z_score': RegimeMetricsCalculator.
                calculate_z_score(returns)}}
        return None

    def _check_momentum(self, returns: np.ndarray) ->dict[str, Any] | None:
        if self._is_momentum(returns):
            direction = 'up' if np.mean(returns[-self.momentum_window:]
                ) > 0 else 'down'
            return {'regime': MarketRegime.MOMENTUM.value, 'confidence':
                0.75, 'reason': f'strong_{direction}_momentum', 'metrics':
                {'momentum_strength': abs(float(np.mean(returns[-self.
                momentum_window:])))}}
        return None

    def _check_breakout(self, prices: np.ndarray | None, volume:
        np.ndarray | None) ->dict[str, Any] | None:
        if prices is not None and volume is not None and self._is_breakout(
            prices, volume):
            return {'regime': MarketRegime.BREAKOUT.value, 'confidence':
                0.85, 'reason': 'price_volume_breakout', 'metrics': {
                'breakout_size': float(abs(prices[-1] - prices[-2]) /
                prices[-2])}}
        return None

    def _detect_standard_regimes(self, adx: float, volatility: float,
        mean_return: float) ->dict[str, Any]:
        if adx > self.adx_threshold:
            regime, confidence = self._detect_trending_regime(adx, mean_return)
        elif volatility > self.volatility_threshold_high:
            regime = MarketRegime.VOLATILE
            confidence = min(0.9, volatility / 0.05)
        elif volatility < self.volatility_threshold_low:
            regime = MarketRegime.RANGING
            confidence = 0.7
        else:
            regime = MarketRegime.NORMAL
            confidence = 0.6
        return {'regime': regime.value, 'confidence': float(confidence),
            'reason': 'rule_based'}

    def _detect_trending_regime(self, adx: float, mean_return: float) ->tuple[
        MarketRegime, float]:
        regime = (MarketRegime.TRENDING_UP if mean_return > 0 else
            MarketRegime.TRENDING_DOWN)
        confidence = min(0.9, adx / 50)
        return regime, confidence

    def _is_mean_reversion(self, returns: np.ndarray) ->bool:
        if len(returns) < 50:
            return False
        try:
            from statsmodels.tsa.stattools import adfuller
            prices = np.cumprod(1 + returns)
            p_value = adfuller(prices, maxlag=10)[1]
            return bool(p_value < 0.05)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            if not hasattr(self, 'logger'):
                self.logger = ProjectLogger.get_logger('RegimeRulesEngine')
            self.logger.error(f'Error calculating mean reversion stats: {e}',
                exc_info=True)
            recent_returns = returns[-50:]
            if len(recent_returns) == 0:
                return False
            std = np.std(recent_returns)
            z_score = abs(np.mean(recent_returns)) / (std / np.sqrt(len(
                recent_returns))) if std != 0 else 0
            return bool(z_score < self.mean_reversion_threshold)

    def _is_momentum(self, returns: np.ndarray) ->bool:
        if len(returns) < self.momentum_window * 2:
            return False
        recent_avg = np.mean(returns[-self.momentum_window:])
        previous_avg = np.mean(returns[-2 * self.momentum_window:-self.
            momentum_window])
        momentum = abs(float(recent_avg) - float(previous_avg))
        return bool(momentum > self.momentum_threshold)

    def _is_breakout(self, prices: np.ndarray, volume: np.ndarray) ->bool:
        if len(prices) < 20 or len(volume) < 20:
            return False
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        current_price = prices[-1]
        price_range = float(recent_high - recent_low)
        if price_range == 0:
            return False
        breakout_up = (current_price - recent_low
            ) / price_range > self.breakout_threshold
        breakout_down = (recent_high - current_price
            ) / price_range > self.breakout_threshold
        avg_volume = float(np.mean(volume[-20:]))
        volume_spike = float(volume[-1]
            ) > avg_volume * self.breakout_volume_multiplier
        return bool((breakout_up or breakout_down) and volume_spike)
