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

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger


@dataclass
class RegimeMetrics:
    """Метрики для визначення режиму ринку"""
    returns: np.ndarray
    prices: np.ndarray | None
    volume: np.ndarray | None
    adx: float
    volatility: float
    mean_return: float


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

    def __init__(self, config: dict[str, Any] | None = None):
        self.logger = ProjectLogger.get_logger("MarketRegimeDetector")
        self.config_manager = get_current_config()
        self.config = config or self.config_manager.get('logic.regime_detection', {})

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
        self.cluster_model: KMeans | None = None

    def detect_regime(self,
                      returns: np.ndarray,
                      data_bundle: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Виявляє режим ринку з використанням всіх доступних даних

        Args:
            returns: Масив повернень
            data_bundle: Словник з опціональними даними (prices, volume, multi_timeframe_data, sentiment_data)

        Returns:
            Dict з режимом та детальними метриками
        """
        try:
            if len(returns) < 30:
                return self._create_insufficient_data_result()

            bundle = data_bundle or {}
            prices = bundle.get('prices')
            volume = bundle.get('volume')
            sentiment = bundle.get('sentiment_data')

            # Розрахуємо базові метрики
            volatility, mean_return, adx = self._calculate_basic_metrics(returns)

            # 1. Crisis detection
            crisis_result = self._check_crisis_regime(mean_return)
            if crisis_result:
                return crisis_result

            # 2. ML-based clustering
            ml_result = self._try_ml_detection(returns, prices, volume, sentiment)
            if ml_result:
                return ml_result

            # 3. Rule-based detection
            metrics = RegimeMetrics(
                returns=returns, prices=prices, volume=volume,
                adx=adx, volatility=volatility, mean_return=mean_return
            )
            rule_regime = self._detect_regime_rules(metrics)

            # 4. Multi-timeframe consensus
            consensus_regime = self._try_multi_timeframe_consensus(rule_regime, bundle.get('multi_timeframe_data'))
            return consensus_regime or rule_regime

        except Exception as e:
            return self._create_error_result(e)

    def _create_insufficient_data_result(self) -> dict[str, Any]:
        """Створює результат при недостатньо даних"""
        return {
            'regime': MarketRegime.NORMAL.value,
            'confidence': 0.5,
            'reason': 'insufficient_data'
        }

    def _calculate_basic_metrics(self, returns: np.ndarray) -> tuple[float, float, float]:
        """Розраховує базові метрики"""
        volatility = np.std(returns)
        mean_return = np.mean(returns)
        adx = self._calculate_adx(returns)
        return volatility, mean_return, adx

    def _check_crisis_regime(self, mean_return: float) -> dict[str, Any] | None:
        """Перевіряє crisis regime"""
        if mean_return < self.crisis_threshold:
            return {
                'regime': MarketRegime.CRISIS.value,
                'confidence': 0.95,
                'reason': 'extreme_negative_returns',
                'metrics': {'mean_return': mean_return, 'crisis_threshold': self.crisis_threshold}
            }
        return None

    def _try_ml_detection(self, returns: np.ndarray, prices: np.ndarray | None,
                        volume: np.ndarray | None, sentiment: np.ndarray | None) -> dict[str, Any] | None:
        """Намагається виявити regime за допомогою ML"""
        if self.use_ml_clustering and len(returns) >= self.min_samples_for_clustering:
            ml_regime = self._detect_regime_ml(returns, prices, volume, sentiment)
            if ml_regime['confidence'] > 0.7:
                return ml_regime
        return None

    def _try_multi_timeframe_consensus(self, rule_regime: dict[str, Any],
                                     multi_timeframe_data: dict[str, np.ndarray] | None) -> dict[str, Any] | None:
        """Намагається отримати consensus з різних таймфреймів"""
        if multi_timeframe_data:
            consensus_regime = self._multi_timeframe_consensus(rule_regime, multi_timeframe_data)
            if consensus_regime['confidence'] > rule_regime['confidence']:
                return consensus_regime
        return None

    def _create_error_result(self, error: Exception) -> dict[str, Any]:
        """Створює результат при помилці"""
        return {
            'regime': MarketRegime.NORMAL.value,
            'confidence': 0.5,
            'error': str(error)
        }

    def _detect_regime_ml(self, returns: np.ndarray, prices: np.ndarray | None,
                         volume: np.ndarray | None, sentiment: np.ndarray | None) -> dict[str, Any]:
        """ML-based regime detection using clustering"""
        try:
            features = self._extract_ml_features(returns, prices, volume, sentiment)
            features_scaled = self._normalize_features(features)

            self._ensure_cluster_model_fitted()
            if self.cluster_model is None:
                raise RuntimeError("Cluster model initialization failed")

            model = cast(KMeans, self.cluster_model)  # Type narrowing with cast
            cluster = model.predict(features_scaled)[0]
            regime = self._cluster_to_regime(cluster)
            confidence = self._calculate_ml_confidence(features_scaled)

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

    def _extract_ml_features(self, returns: np.ndarray, prices: np.ndarray | None,
                            volume: np.ndarray | None, sentiment: np.ndarray | None) -> list[float]:
        """Витягує features для ML кластеризації"""
        features = []

        # Return-based features
        features.extend([
            np.mean(returns), np.std(returns), np.skew(returns),
            np.kurtosis(returns), np.min(returns), np.max(returns)
        ])

        # Price-based features
        if prices is not None and len(prices) > 20:
            features.extend(self._extract_price_features(prices))

        # Volume-based features
        if volume is not None and len(volume) > 20:
            features.append(self._extract_volume_feature(volume))

        # Sentiment features
        if sentiment is not None and len(sentiment) > 0:
            features.append(np.mean(sentiment))

        return features

    def _extract_price_features(self, prices: np.ndarray) -> list[float]:
        """Витягує price-based features"""
        features = []

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

        return features

    def _extract_volume_feature(self, volume: np.ndarray) -> float:
        """Витягує volume feature"""
        volume_sma = np.mean(volume[-20:])
        return volume[-1] / volume_sma if volume_sma > 0 else 1

    def _normalize_features(self, features: list[float]) -> np.ndarray:
        """Нормалізує features"""
        features_array = np.array(features).reshape(1, -1)
        return self.scaler.fit_transform(features_array)

    def _ensure_cluster_model_fitted(self):
        """Переконується що cluster model навчена"""
        if self.cluster_model is None:
            self._initialize_cluster_centers()

    def _calculate_ml_confidence(self, features_scaled: np.ndarray) -> float:
        """Розраховує confidence для ML методу"""
        assert self.cluster_model is not None, "Cluster model must be initialized"
        distances = self.cluster_model.transform(features_scaled)[0]
        min_distance = float(np.min(distances))
        return max(0.5, 1.0 - min_distance)  # Higher confidence for closer points

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

        seed = self.config_manager.get('performance.random_seed', 42)
        self.cluster_model = KMeans(n_clusters=self.n_clusters, init=centers, n_init=1, random_state=seed)

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

    def _detect_regime_rules(self, metrics: RegimeMetrics) -> dict[str, Any]:
        """Rule-based regime detection"""

        # Check special regimes first
        mean_reversion_result = self._check_mean_reversion_regime(metrics.returns)
        if mean_reversion_result:
            return mean_reversion_result

        momentum_result = self._check_momentum_regime(metrics.returns)
        if momentum_result:
            return momentum_result

        breakout_result = self._check_breakout_regime(metrics.prices, metrics.volume)
        if breakout_result:
            return breakout_result

        # Standard regime detection
        return self._detect_standard_regimes(metrics.adx, metrics.volatility, metrics.mean_return)

    def _check_mean_reversion_regime(self, returns: np.ndarray) -> dict[str, Any] | None:
        """Перевіряє mean reversion regime"""
        if self._is_mean_reversion(returns):
            return {
                'regime': MarketRegime.MEAN_REVERSION.value,
                'confidence': 0.8,
                'reason': 'statistical_mean_reversion',
                'metrics': {'z_score': self._calculate_z_score(returns)}
            }
        return None

    def _check_momentum_regime(self, returns: np.ndarray) -> dict[str, Any] | None:
        """Перевіряє momentum regime"""
        if self._is_momentum(returns):
            direction = 'up' if np.mean(returns[-self.momentum_window:]) > 0 else 'down'
            return {
                'regime': MarketRegime.MOMENTUM.value,
                'confidence': 0.75,
                'reason': f'strong_{direction}_momentum',
                'metrics': {'momentum_strength': abs(np.mean(returns[-self.momentum_window:]))}
            }
        return None

    def _has_breakout_data(self, prices: np.ndarray | None, volume: np.ndarray | None) -> bool:
        """Перевіряє чи є необхідні дані для breakout аналізу"""
        return prices is not None and volume is not None

    def _check_breakout_regime(self, prices: np.ndarray | None,
                              volume: np.ndarray | None) -> dict[str, Any] | None:
        """Перевіряє breakout regime"""
        if self._has_breakout_data(prices, volume) and self._is_breakout(prices, volume):  # type: ignore[arg-type]
            return {
                'regime': MarketRegime.BREAKOUT.value,
                'confidence': 0.85,
                'reason': 'price_volume_breakout',
                'metrics': {'breakout_size': abs(prices[-1] - prices[-2]) / prices[-2]}  # type: ignore[index]
            }
        return None

    def _detect_standard_regimes(self, adx: float, volatility: float,
                                mean_return: float) -> dict[str, Any]:
        """Виявляє стандартні режими ринку"""
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

    def _detect_trending_regime(self, adx: float, mean_return: float) -> tuple[MarketRegime, float]:
        """Виявляє trending regime"""
        if mean_return > 0:
            regime = MarketRegime.TRENDING_UP
        else:
            regime = MarketRegime.TRENDING_DOWN
        confidence = min(0.9, adx / 50)
        return regime, confidence

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
            return bool(p_value < 0.05)
        except Exception as e:
            self.logger.debug(f"ADF test failed, using fallback mean-reversion check: {e}")
            # Fallback: check if returns oscillate around zero
            recent_returns = returns[-50:]
            z_score = abs(np.mean(recent_returns)) / (np.std(recent_returns) / np.sqrt(len(recent_returns)))
            return bool(z_score < self.mean_reversion_threshold)

    def _is_momentum(self, returns: np.ndarray) -> bool:
        """Перевіряє чи є momentum"""
        if len(returns) < self.momentum_window * 2:
            return False

        recent_avg = np.mean(returns[-self.momentum_window:])
        previous_avg = np.mean(returns[-2*self.momentum_window:-self.momentum_window])

        momentum = abs(float(recent_avg) - float(previous_avg))
        return bool(momentum > self.momentum_threshold)

    def _is_breakout(self, prices: np.ndarray, volume: np.ndarray) -> bool:
        """Перевіряє чи є breakout"""
        if len(prices) < 20 or len(volume) < 20:
            return False

        # Price breakout
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        current_price = prices[-1]

        price_range = float(recent_high - recent_low)
        if price_range == 0:
            return False

        breakout_up = bool((current_price - recent_low) / price_range > self.breakout_threshold)
        breakout_down = bool((recent_high - current_price) / price_range > self.breakout_threshold)

        # Volume confirmation
        avg_volume = float(np.mean(volume[-20:]))
        current_volume = float(volume[-1])
        volume_spike = bool(current_volume > avg_volume * self.breakout_volume_multiplier)

        return bool((breakout_up or breakout_down) and volume_spike)

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
            return 0.0

        recent_returns = returns[-20:]
        z_score = (recent_returns[-1] - np.mean(recent_returns)) / np.std(recent_returns)
        return float(z_score)

    def _multi_timeframe_consensus(self, base_regime: dict[str, Any],
                                 multi_timeframe_data: dict[str, np.ndarray]) -> dict[str, Any]:
        """Multi-timeframe regime consensus"""
        try:
            regimes = [base_regime['regime']]
            confidences = [base_regime['confidence']]

            for _tf, tf_returns in multi_timeframe_data.items():
                if len(tf_returns) >= 30:
                    tf_metrics = RegimeMetrics(
                        returns=tf_returns,
                        prices=None,
                        volume=None,
                        adx=self._calculate_adx(tf_returns),
                        volatility=float(np.std(tf_returns)),
                        mean_return=float(np.mean(tf_returns))
                    )
                    tf_regime = self._detect_regime_rules(tf_metrics)
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

    @staticmethod
    def calculate_entropy(price_series: 'pd.Series', window: int = 50, num_bins: int = 10) -> 'pd.Series':
        """
        Calculates rolling Shannon entropy to measure market uncertainty/disorder.
        
        Integrated from MarketRegimeCalculator for unified regime analysis.

        Args:
            price_series: Historical price sequence.
            window: Rolling window for distribution estimation.
            num_bins: Histogram bins for return discretization.

        Returns:
            Series of entropy coefficients (measured in bits).
        """
        import pandas as pd
        from scipy.stats import entropy as scipy_entropy
        
        if not isinstance(price_series, pd.Series) or price_series.empty:
            return pd.Series(dtype=float)
            
        returns = price_series.pct_change().dropna()
        
        def _compute_entropy(window_slice):
            if len(window_slice) < window * 0.8:
                return np.nan
            hist, _ = np.histogram(window_slice, bins=num_bins, density=True)
            # Normalize to probability distribution
            prob_dist = hist * np.diff(np.histogram_bin_edges(window_slice, bins=num_bins))
            return scipy_entropy(prob_dist, base=2)

        result = returns.rolling(window=window).apply(_compute_entropy, raw=True)
        return result

    @staticmethod
    def calculate_reversal_probability(price_series: 'pd.Series', down_day_threshold: float = -0.01, window: int = 5) -> 'pd.Series':
        """
        Estimates local reversal probability following sequences of expansionary/contractionary days.
        
        Integrated from MarketRegimeCalculator for unified regime analysis.

        Args:
            price_series: Historical price sequence.
            down_day_threshold: Return threshold defining a 'down' state.
            window: Lookback for streak identification.

        Returns:
            Series of probabilities [0, 1].
        """
        import pandas as pd
        
        if not isinstance(price_series, pd.Series) or price_series.empty:
            return pd.Series(dtype=float)

        returns = price_series.pct_change()
        is_down = (returns < down_day_threshold).astype(int)

        # Identify consecutive streak length
        consecutive_down = is_down.rolling(window=window).sum()

        # Heuristic probabilistic model
        base_probability = 0.1
        probability_estimate = base_probability + (consecutive_down / window) * 0.5
        
        # Isolate probabilities for relevant streaks only
        reversal_series = probability_estimate.where(consecutive_down > 1, 0.0)
        
        return reversal_series.clip(0, 1)
