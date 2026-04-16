"""
Виявлення режимів ринку (Market Regime Detection) - Покращена версія.

Виявляє режими:
1. Trending Up (тренд вгору)
2. Trending Down (тренд вниз)
3. Ranging (бічний рух)
4. Volatile (волатильний)
5. Crisis (криза)
6. Mean Reversion (середнє повернення)
7. Momentum (моментум)
8. Breakout (прорив)

Використовує:
- Multiple timeframe analysis
- Machine learning clustering
- Statistical tests
- Volume analysis
- Sentiment integration
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from src.core.logging.logger import ProjectLogger

class MarketRegime(Enum):
    """Режими ринку з додатковими станами"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM = "MOMENTUM"
    BREAKOUT = "BREAKOUT"
    NORMAL = "NORMAL"

class MarketRegimeDetector:
    """Виявляє режими ринку з використанням ML та статистичних методів"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = ProjectLogger.get_logger("MarketRegimeDetector")
        self.config = config or {}

        # Параметри для виявлення режимів
        self.adx_threshold = self.config.get('adx_threshold', 25)
        self.volatility_threshold_high = self.config.get('volatility_threshold_high', 0.03)
        self.volatility_threshold_low = self.config.get('volatility_threshold_low', 0.01)
        self.crisis_threshold = self.config.get('crisis_threshold', -0.05)

        # Нові параметри
        self.use_ml_clustering = self.config.get('use_ml_clustering', True)
        self.n_clusters = self.config.get('n_clusters', 8)  # Для кожного regime
        self.min_samples_for_clustering = self.config.get('min_samples_for_clustering', 252)  # 1 рік

        # Mean reversion parameters
        self.mean_reversion_threshold = self.config.get('mean_reversion_threshold', 2.0)  # Standard deviations

        # Momentum parameters
        self.momentum_window = self.config.get('momentum_window', 20)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.02)

        # Breakout parameters
        self.breakout_threshold = self.config.get('breakout_threshold', 0.05)  # 5% move
        self.breakout_volume_multiplier = self.config.get('breakout_volume_multiplier', 1.5)

        # Multi-timeframe analysis
        self.timeframes = self.config.get('timeframes', ['1d', '1w', '1M'])

        # Scaler для ML
        self.scaler = StandardScaler()
        self.cluster_model = None

    def detect_regime(self,
                     returns: np.ndarray,
                     prices: Optional[np.ndarray] = None,
                     volume: Optional[np.ndarray] = None,
                     multi_timeframe_data: Optional[Dict[str, np.ndarray]] = None,
                     sentiment_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Виявляє режим ринку з використанням всіх доступних даних

        Args:
            returns: Масив повернень
            prices: Масив цін (опціонально)
            volume: Масив обсягів (опціонально)
            multi_timeframe_data: Дані по різних таймфреймах
            sentiment_data: Дані sentiment (опціонально)

        Returns:
            Dict з режимом та детальними метриками
        """
        try:
            if len(returns) < 30:
                return {'regime': MarketRegime.NORMAL.value, 'confidence': 0.5, 'reason': 'insufficient_data'}

            # Розрахуємо базові метрики
            volatility = np.std(returns)
            mean_return = np.mean(returns)
            adx = self._calculate_adx(returns)

            # 1. Crisis detection (highest priority)
            if mean_return < self.crisis_threshold:
                return {
                    'regime': MarketRegime.CRISIS.value,
                    'confidence': 0.95,
                    'reason': 'extreme_negative_returns',
                    'metrics': {'mean_return': mean_return, 'crisis_threshold': self.crisis_threshold}
                }

            # 2. ML-based clustering (якщо достатньо даних)
            if self.use_ml_clustering and len(returns) >= self.min_samples_for_clustering:
                ml_regime = self._detect_regime_ml(returns, prices, volume, sentiment_data)
                if ml_regime['confidence'] > 0.7:
                    return ml_regime

            # 3. Rule-based detection
            rule_regime = self._detect_regime_rules(returns, prices, volume, adx, volatility, mean_return)

            # 4. Multi-timeframe consensus
            if multi_timeframe_data:
                consensus_regime = self._multi_timeframe_consensus(
                    rule_regime, multi_timeframe_data
                )
                if consensus_regime['confidence'] > rule_regime['confidence']:
                    return consensus_regime

            return rule_regime

        except Exception as e:
            self.logger.error(f"Помилка виявлення режиму: {e}")
            return {
                'regime': MarketRegime.NORMAL.value,
                'confidence': 0.5,
                'error': str(e)
            }

    def _detect_regime_ml(self, returns: np.ndarray, prices: Optional[np.ndarray],
                         volume: Optional[np.ndarray], sentiment: Optional[np.ndarray]) -> Dict[str, Any]:
        """ML-based regime detection using clustering"""
        try:
            # Створюємо features для кластеризації
            features = []

            # Return-based features
            features.extend([
                np.mean(returns), np.std(returns), np.skew(returns),
                np.kurtosis(returns), np.min(returns), np.max(returns)
            ])

            # Price-based features (якщо доступні)
            if prices is not None and len(prices) > 20:
                # Trend strength
                trend = np.polyfit(np.arange(len(prices)), prices, 1)[0]
                features.append(trend)

                # RSI
                rsi = self._calculate_rsi(prices)
                features.append(rsi)

                # Bollinger Bands position
                sma = np.mean(prices[-20:])
                std = np.std(prices[-20:])
                bb_position = (prices[-1] - sma) / (2 * std) if std > 0 else 0
                features.append(bb_position)

            # Volume-based features
            if volume is not None and len(volume) > 20:
                volume_sma = np.mean(volume[-20:])
                volume_ratio = volume[-1] / volume_sma if volume_sma > 0 else 1
                features.append(volume_ratio)

            # Sentiment features
            if sentiment is not None and len(sentiment) > 0:
                features.append(np.mean(sentiment))

            # Normalize features
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.fit_transform(features_array)

            # Fit cluster model if not fitted
            if self.cluster_model is None:
                # Use predefined cluster centers based on regime characteristics
                self._initialize_cluster_centers()

            # Predict cluster
            cluster = self.cluster_model.predict(features_scaled)[0]
            regime = self._cluster_to_regime(cluster)

            # Calculate confidence based on distance to cluster center
            distances = self.cluster_model.transform(features_scaled)[0]
            min_distance = np.min(distances)
            confidence = max(0.5, 1.0 - min_distance)  # Higher confidence for closer points

            return {
                'regime': regime.value,
                'confidence': float(confidence),
                'method': 'ml_clustering',
                'cluster': int(cluster),
                'features_used': len(features)
            }

        except Exception as e:
            self.logger.warning(f"ML regime detection failed: {e}")
            return {'regime': MarketRegime.NORMAL.value, 'confidence': 0.5, 'method': 'ml_fallback'}

    def _initialize_cluster_centers(self):
        """Ініціалізує центри кластерів на основі характеристик режимів"""
        # Predefined centers based on typical regime characteristics
        centers = np.array([
            # TRENDING_UP: positive mean, moderate volatility, positive skew
            [0.001, 0.015, 0.2, 0.1, -0.03, 0.03, 0.001, 60, 0.5, 1.0, 0.1],
            # TRENDING_DOWN: negative mean, moderate volatility, negative skew
            [-0.001, 0.015, -0.2, 0.1, -0.03, 0.03, -0.001, 40, -0.5, 1.0, -0.1],
            # RANGING: low mean, low volatility, low skew
            [0.000, 0.008, 0.0, -0.5, -0.01, 0.01, 0.000, 50, 0.0, 0.8, 0.0],
            # VOLATILE: any mean, high volatility
            [0.000, 0.035, 0.0, 1.0, -0.08, 0.08, 0.000, 70, 0.0, 1.2, 0.0],
            # CRISIS: very negative mean, high volatility
            [-0.005, 0.040, -0.5, 2.0, -0.15, 0.05, -0.003, 30, -1.0, 0.7, -0.3],
            # MEAN_REVERSION: oscillating around mean
            [0.000, 0.012, 0.0, -0.8, -0.025, 0.025, 0.000, 45, 0.0, 0.9, 0.0],
            # MOMENTUM: strong directional movement
            [0.003, 0.025, 0.8, 0.5, -0.02, 0.06, 0.002, 75, 1.0, 1.5, 0.2],
            # BREAKOUT: sudden large moves with volume
            [0.002, 0.030, 0.3, 1.2, -0.04, 0.08, 0.001, 80, 0.8, 2.0, 0.3]
        ])

        self.cluster_model = KMeans(n_clusters=self.n_clusters, init=centers, n_init=1, random_state=42)

    def _cluster_to_regime(self, cluster: int) -> MarketRegime:
        """Перетворює номер кластера в regime"""
        cluster_regime_map = {
            0: MarketRegime.TRENDING_UP,
            1: MarketRegime.TRENDING_DOWN,
            2: MarketRegime.RANGING,
            3: MarketRegime.VOLATILE,
            4: MarketRegime.CRISIS,
            5: MarketRegime.MEAN_REVERSION,
            6: MarketRegime.MOMENTUM,
            7: MarketRegime.BREAKOUT
        }
        return cluster_regime_map.get(cluster, MarketRegime.NORMAL)

    def _detect_regime_rules(self, returns: np.ndarray, prices: Optional[np.ndarray],
                           volume: Optional[np.ndarray], adx: float, volatility: float,
                           mean_return: float) -> Dict[str, Any]:
        """Rule-based regime detection"""

        # Mean Reversion detection
        if self._is_mean_reversion(returns):
            return {
                'regime': MarketRegime.MEAN_REVERSION.value,
                'confidence': 0.8,
                'reason': 'statistical_mean_reversion',
                'metrics': {'z_score': self._calculate_z_score(returns)}
            }

        # Momentum detection
        if self._is_momentum(returns):
            direction = 'up' if np.mean(returns[-self.momentum_window:]) > 0 else 'down'
            return {
                'regime': MarketRegime.MOMENTUM.value,
                'confidence': 0.75,
                'reason': f'strong_{direction}_momentum',
                'metrics': {'momentum_strength': abs(np.mean(returns[-self.momentum_window:]))}
            }

        # Breakout detection
        if prices is not None and volume is not None and self._is_breakout(prices, volume):
            return {
                'regime': MarketRegime.BREAKOUT.value,
                'confidence': 0.85,
                'reason': 'price_volume_breakout',
                'metrics': {'breakout_size': abs(prices[-1] - prices[-2]) / prices[-2]}
            }

        # Original logic for other regimes
        if adx > self.adx_threshold:
            if mean_return > 0:
                regime = MarketRegime.TRENDING_UP
                confidence = min(0.9, adx / 50)
            else:
                regime = MarketRegime.TRENDING_DOWN
                confidence = min(0.9, adx / 50)
        elif volatility > self.volatility_threshold_high:
            regime = MarketRegime.VOLATILE
            confidence = min(0.9, volatility / 0.05)
        elif volatility < self.volatility_threshold_low:
            regime = MarketRegime.RANGING
            confidence = 0.7
        else:
            regime = MarketRegime.NORMAL
            confidence = 0.6

        return {
            'regime': regime.value,
            'confidence': float(confidence),
            'reason': 'rule_based',
            'metrics': {
                'adx': adx,
                'volatility': volatility,
                'mean_return': mean_return,
                'adx_threshold': self.adx_threshold
            }
        }

    def _is_mean_reversion(self, returns: np.ndarray) -> bool:
        """Перевіряє чи є mean reversion"""
        if len(returns) < 50:
            return False

        # Augmented Dickey-Fuller test for stationarity
        try:
            from statsmodels.tsa.stattools import adfuller
            prices = np.cumprod(1 + returns)  # Convert to prices
            adf_result = adfuller(prices, maxlag=10)
            p_value = adf_result[1]

            # If p-value < 0.05, series is stationary (mean reversion)
            return p_value < 0.05
        except:
            # Fallback: check if returns oscillate around zero
            recent_returns = returns[-50:]
            z_score = abs(np.mean(recent_returns)) / (np.std(recent_returns) / np.sqrt(len(recent_returns)))
            return z_score < self.mean_reversion_threshold

    def _is_momentum(self, returns: np.ndarray) -> bool:
        """Перевіряє чи є momentum"""
        if len(returns) < self.momentum_window * 2:
            return False

        recent_avg = np.mean(returns[-self.momentum_window:])
        previous_avg = np.mean(returns[-2*self.momentum_window:-self.momentum_window])

        momentum = abs(recent_avg - previous_avg)
        return momentum > self.momentum_threshold

    def _is_breakout(self, prices: np.ndarray, volume: np.ndarray) -> bool:
        """Перевіряє чи є breakout"""
        if len(prices) < 20 or len(volume) < 20:
            return False

        # Price breakout
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        current_price = prices[-1]

        price_range = recent_high - recent_low
        if price_range == 0:
            return False

        breakout_up = (current_price - recent_low) / price_range > self.breakout_threshold
        breakout_down = (recent_high - current_price) / price_range > self.breakout_threshold

        # Volume confirmation
        avg_volume = np.mean(volume[-20:])
        current_volume = volume[-1]
        volume_spike = current_volume > avg_volume * self.breakout_volume_multiplier

        return (breakout_up or breakout_down) and volume_spike

    def _calculate_adx(self, returns: np.ndarray, period: int = 14) -> float:
        """Розраховує ADX індикатор"""
        try:
            if len(returns) < period:
                return 0.0

            # Розраховуємо true range
            prices = np.cumsum(returns)  # Наближена ціна

            # Розраховуємо directional movement
            up_move = np.maximum(np.diff(prices), 0)
            down_move = np.maximum(-np.diff(prices), 0)

            # Розраховуємо DI+ і DI-
            tr = np.maximum(up_move, down_move)
            tr_sum = np.sum(tr[-period:])

            if tr_sum == 0:
                return 0.0

            di_plus = 100 * np.sum(up_move[-period:]) / tr_sum
            di_minus = 100 * np.sum(down_move[-period:]) / tr_sum

            # ADX = середнє абсолютне значення DI+ - DI-
            adx = abs(di_plus - di_minus)

            return float(adx)

        except Exception as e:
            self.logger.warning(f"Помилка розрахунку ADX: {e}")
            return 0.0

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Розрахунок RSI"""
        if len(prices) < period + 1:
            return 50

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains[-period:]) if gains else 0
        avg_loss = np.mean(losses[-period:]) if losses else 0

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_z_score(self, returns: np.ndarray) -> float:
        """Розрахунок Z-score для mean reversion"""
        if len(returns) < 20:
            return 0

        recent_returns = returns[-20:]
        z_score = (recent_returns[-1] - np.mean(recent_returns)) / np.std(recent_returns)
        return z_score

    def _multi_timeframe_consensus(self, base_regime: Dict[str, Any],
                                 multi_timeframe_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Multi-timeframe regime consensus"""
        try:
            regimes = [base_regime['regime']]
            confidences = [base_regime['confidence']]

            for tf, tf_returns in multi_timeframe_data.items():
                if len(tf_returns) >= 30:
                    tf_regime = self._detect_regime_rules(tf_returns, None, None,
                                                        self._calculate_adx(tf_returns),
                                                        np.std(tf_returns),
                                                        np.mean(tf_returns))
                    regimes.append(tf_regime['regime'])
                    confidences.append(tf_regime['confidence'] * 0.8)  # Lower weight for higher TFs

            # Find consensus regime
            from collections import Counter
            regime_counts = Counter(regimes)
            consensus_regime = regime_counts.most_common(1)[0][0]

            # Average confidence
            avg_confidence = np.mean(confidences)

            # Boost confidence if consensus
            if len(set(regimes)) == 1:  # All timeframes agree
                avg_confidence = min(avg_confidence * 1.2, 0.95)

            return {
                'regime': consensus_regime,
                'confidence': float(avg_confidence),
                'method': 'multi_timeframe_consensus',
                'timeframes_agree': len(set(regimes)),
                'total_timeframes': len(regimes)
            }

        except Exception as e:
            self.logger.warning(f"Multi-timeframe consensus failed: {e}")
            return base_regime
