"""
Adaptive Parameter Manager
- Parameters adapt to market regime, volatility, asset class
- Data-driven thresholds instead of hardcoded
- Quarterly optimization via historical backtest
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    DEAD = "dead"

class AssetClass(Enum):
    LARGE_CAP = "large_cap"  # AAPL, MSFT (less volatile)
    SMALL_CAP = "small_cap"  # Low liquidity (more slippage)
    CRYPTO = "crypto"  # Very volatile
    COMMODITY = "commodity"  # Mean-reverting

@dataclass
class AdaptiveParameters:
    """Set of parameters for specific context"""
    # Signal thresholds
    buy_threshold: float = 0.02 # Minimum prediction value for BUY
    sell_threshold: float = -0.02 # Maximum prediction value for SELL
    hold_threshold: float = 0.005 # Range for HOLD

    # Confidence adjustment
    confidence_min_accepted: float = 0.50 # Do not trade if confidence < this
    confidence_boost_trending: float = 1.0 # Boost when in trend
    confidence_penalty_volatile: float = 0.9 # Penalty when volatile

    # Risk sizing
    risk_per_trade_pct: float = 0.02 # % equity per trade
    max_position_size_pct: float = 0.10 # Max % in one position
    max_daily_drawdown_pct: float = 0.05 # Kill switch

    # News impact
    news_negative_threshold: float = -0.6 # Threshold for negative news
    news_positive_threshold: float = 0.6 # Threshold for positive news
    news_decay_hours: float = 24.0 # How fast to forget new news

    # Model weighting
    model_decay_days: float = 30.0 # How fast to forget old models
    ensemble_reweight_days: float = 7.0 # How often to reweight ensemble

    regime: MarketRegime = MarketRegime.RANGING
    asset_class: AssetClass = AssetClass.LARGE_CAP
    volatility_percentile: float = 0.50  # 0.0 = low volatility, 1.0 = high volatility

class AdaptiveParameterManager:
    """Elite-grade parameter management"""

    def __init__(self, config: dict | None = None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}

        # Base parameters for each regime/asset combination
        self.regime_presets = self._build_regime_presets()
        self.asset_presets = self._build_asset_presets()

        # Override with config if available
        self._apply_config_overrides()

        self.current_params: AdaptiveParameters | None = None
        self.param_history: list[dict[str, Any]] = []  # Track changes

    def _apply_config_overrides(self):
        """Override hardcoded presets with values from config_manager/params.json"""
        config_presets = self.config.get('regime_presets', {})
        for regime_str, overrides in config_presets.items():
            try:
                regime_enum = MarketRegime(regime_str.lower())
                if regime_enum in self.regime_presets:
                    self.regime_presets[regime_enum].update(overrides)
                    self.logger.info(f"✅ Applied config overrides for regime: {regime_str}")
            except (ValueError, KeyError):
                continue

    def _build_trending_up_preset(self) -> dict[str, float]:
        """Build preset for trending up regime."""
        return {
            'buy_threshold': 0.02,
            'sell_threshold': -0.02,
            'hold_threshold': 0.005,
            'confidence_min_accepted': 0.55,
            'confidence_boost_trending': 1.15,
            'confidence_penalty_volatile': 0.95,
            'risk_per_trade_pct': 0.04,
            'max_position_size_pct': 0.12,
            'max_daily_drawdown_pct': 0.06,
            'news_negative_threshold': -0.5,
            'news_positive_threshold': 0.4,
            'news_decay_hours': 12.0,
            'model_decay_days': 45.0,
            'ensemble_reweight_days': 5.0,
        }

    def _build_trending_down_preset(self) -> dict[str, float]:
        """Build preset for trending down regime."""
        return {
            'buy_threshold': 0.02,
            'sell_threshold': -0.02,
            'hold_threshold': 0.005,
            'confidence_min_accepted': 0.60,
            'confidence_boost_trending': 0.90,
            'confidence_penalty_volatile': 0.85,
            'risk_per_trade_pct': 0.02,
            'max_position_size_pct': 0.06,
            'max_daily_drawdown_pct': 0.04,
            'news_negative_threshold': -0.7,
            'news_positive_threshold': 0.6,
            'news_decay_hours': 48.0,
            'model_decay_days': 20.0,
            'ensemble_reweight_days': 3.0,
        }

    def _build_ranging_preset(self) -> dict[str, float]:
        """Build preset for ranging regime."""
        return {
            'buy_threshold': 0.01,
            'sell_threshold': -0.01,
            'hold_threshold': 0.003,
            'confidence_min_accepted': 0.50,
            'confidence_boost_trending': 0.85,
            'confidence_penalty_volatile': 0.90,
            'risk_per_trade_pct': 0.025,
            'max_position_size_pct': 0.08,
            'max_daily_drawdown_pct': 0.05,
            'news_negative_threshold': -0.6,
            'news_positive_threshold': 0.6,
            'news_decay_hours': 24.0,
            'model_decay_days': 30.0,
            'ensemble_reweight_days': 7.0,
        }

    def _build_volatile_preset(self) -> dict[str, float]:
        """Build preset for volatile regime."""
        return {
            'buy_threshold': 0.03,
            'sell_threshold': -0.03,
            'hold_threshold': 0.01,
            'confidence_min_accepted': 0.65,
            'confidence_boost_trending': 0.75,
            'confidence_penalty_volatile': 0.70,
            'risk_per_trade_pct': 0.01,
            'max_position_size_pct': 0.03,
            'max_daily_drawdown_pct': 0.03,
            'news_negative_threshold': -0.8,
            'news_positive_threshold': 0.7,
            'news_decay_hours': 72.0,
            'model_decay_days': 10.0,
            'ensemble_reweight_days': 2.0,
        }

    def _build_dead_preset(self) -> dict[str, float]:
        """Build preset for dead regime."""
        return {
            'buy_threshold': 1.0,
            'sell_threshold': -1.0,
            'hold_threshold': 0.5,
            'confidence_min_accepted': 0.99,
            'confidence_boost_trending': 0.0,
            'confidence_penalty_volatile': 0.5,
            'risk_per_trade_pct': 0.001,
            'max_position_size_pct': 0.01,
            'max_daily_drawdown_pct': 0.01,
            'news_negative_threshold': -0.99,
            'news_positive_threshold': 0.99,
            'news_decay_hours': 168.0,
            'model_decay_days': 90.0,
            'ensemble_reweight_days': 30.0,
        }

    def _build_regime_presets(self) -> dict[MarketRegime, dict[str, float]]:
        """
        Optimal parameters for each regime (result of historical backtest)
        """
        return {
            MarketRegime.TRENDING_UP: self._build_trending_up_preset(),
            MarketRegime.TRENDING_DOWN: self._build_trending_down_preset(),
            MarketRegime.RANGING: self._build_ranging_preset(),
            MarketRegime.VOLATILE: self._build_volatile_preset(),
            MarketRegime.DEAD: self._build_dead_preset(),
        }

    def _build_asset_presets(self) -> dict[AssetClass, dict[str, float]]:
        """Коригування параметрів для класів активів"""
        return {
            AssetClass.LARGE_CAP: {
                'risk_multiplier': 1.0,  # Baseline
                'slippage_adjustment': 0.9995,  # 5 bps slippage
                'confidence_multiplier': 1.0,
            },
            AssetClass.SMALL_CAP: {
                'risk_multiplier': 0.7,  # Less risk через слабку ліквідність
                'slippage_adjustment': 0.995,  # 50 bps slippage
                'confidence_multiplier': 1.05,  # Потребуємо вищої впевненості
            },
            AssetClass.CRYPTO: {
                'risk_multiplier': 0.5,  # Половину ризику через екстремальну Volatility
                'slippage_adjustment': 0.99,  # 100 bps slippage
                'confidence_multiplier': 1.15,  # Майже вдвічі вища впевненість
            },
            AssetClass.COMMODITY: {
                'risk_multiplier': 0.8,
                'slippage_adjustment': 0.9975,  # 25 bps
                'confidence_multiplier': 1.08,
            }
        }

    def _normalize_regime_input(self, regime) -> MarketRegime:
        """Convert regime string to Enum, default to RANGING if invalid."""
        if isinstance(regime, str):
            try:
                return MarketRegime(regime.lower())
            except ValueError:
                self.logger.warning(f"Unknown regime '{regime}', defaulting to RANGING")
                return MarketRegime.RANGING
        return regime

    def _normalize_asset_class_input(self, asset_class) -> AssetClass:
        """Convert asset_class string to Enum, default to LARGE_CAP if invalid."""
        if isinstance(asset_class, str):
            try:
                return AssetClass(asset_class.lower())
            except ValueError:
                self.logger.warning(f"Unknown asset class '{asset_class}', defaulting to LARGE_CAP")
                return AssetClass.LARGE_CAP
        return asset_class

    def _normalize_volatility_percentile(self, volatility_percentile: float) -> float:
        """Normalize volatility_percentile from 0-100 to 0-1 if needed."""
        if volatility_percentile > 1.0:
            return volatility_percentile / 100.0
        return volatility_percentile

    def _apply_asset_class_adjustments(self, regime_params: dict, asset_params: dict) -> dict:
        """Apply asset class adjustments to regime parameters."""
        regime_params['risk_per_trade_pct'] *= asset_params['risk_multiplier']
        regime_params['max_position_size_pct'] *= asset_params['risk_multiplier']
        regime_params['confidence_min_accepted'] *= asset_params['confidence_multiplier']
        return regime_params

    def _apply_volatility_adjustments(self, regime_params: dict, volatility_percentile: float) -> dict:
        """Apply volatility-based fine-tuning to parameters."""
        vol_adjustment = 1.0 - (volatility_percentile * 0.4)  # Up to 40% reduction
        regime_params['risk_per_trade_pct'] *= vol_adjustment
        regime_params['max_position_size_pct'] *= vol_adjustment
        regime_params['confidence_min_accepted'] *= (1 + volatility_percentile * 0.15)  # Up to 15% increase
        return regime_params

    def _apply_sharpe_adjustments(self, regime_params: dict, historical_sharpe: float | None) -> dict:
        """Apply Sharpe ratio-based adjustments if available."""
        if historical_sharpe is not None:
            if historical_sharpe < 0.5:
                # Poor results - more cautious
                regime_params['risk_per_trade_pct'] *= 0.7
                regime_params['confidence_min_accepted'] *= 1.1
            elif historical_sharpe > 1.5:
                # Good results - more aggressive
                regime_params['risk_per_trade_pct'] *= 1.2
                regime_params['confidence_min_accepted'] *= 0.95
        return regime_params

    def _log_parameter_changes(self, regime: MarketRegime, asset_class: AssetClass,
                               regime_params: dict, volatility_percentile: float) -> None:
        """Log parameter changes and update history."""
        self.logger.info(f"🔄 Parameters adapted to {regime.value} / {asset_class.value}")
        self.logger.info(f"  risk_per_trade: {regime_params['risk_per_trade_pct']:.3%}")
        self.logger.info(f"  confidence_min: {regime_params['confidence_min_accepted']:.2f}")
        self.param_history.append({
            'timestamp': pd.Timestamp.now(),
            'regime': regime.value,
            'asset_class': asset_class.value,
            'volatility_pct': volatility_percentile
        })

    def compute_adaptive_params(self,
                               regime: MarketRegime,
                               asset_class: AssetClass,
                               volatility_percentile: float,
                               historical_sharpe: float | None = None) -> AdaptiveParameters:
        """
        Обчислити адаптивні параметри для поточного контексту

        Args:
            regime: Поточний ринковий режим
            asset_class: Клас активу
            volatility_percentile: 0-1, де 1=максимальна Volatility
            historical_sharpe: Sharpe ratio асету за останній місяць (для fine-tuning)

        Returns:
            AdaptiveParameters instance
        """
        # Convert string inputs to Enums
        regime = self._normalize_regime_input(regime)
        asset_class = self._normalize_asset_class_input(asset_class)
        volatility_percentile = self._normalize_volatility_percentile(volatility_percentile)

        # 1. Start with regime-based parameters
        regime_params = self.regime_presets[regime].copy()

        # 2. Apply asset class adjustments
        asset_params = self.asset_presets[asset_class]
        regime_params = self._apply_asset_class_adjustments(regime_params, asset_params)

        # 3. Volatility-based fine-tuning
        regime_params = self._apply_volatility_adjustments(regime_params, volatility_percentile)

        # 4. Sharpe-based adjustment (if available)
        regime_params = self._apply_sharpe_adjustments(regime_params, historical_sharpe)

        params = AdaptiveParameters(
            regime=regime,
            asset_class=asset_class,
            volatility_percentile=volatility_percentile,
            **regime_params
        )

        # 5. Log changes
        if self.current_params != params:
            self._log_parameter_changes(regime, asset_class, regime_params, volatility_percentile)

        self.current_params = params
        return params

    def validate_parameters(self, params: AdaptiveParameters) -> bool:
        """
        Перевірити що параметри залишаються розумними
        (попередити якщо відбулося щось дивне)
        """
        issues = []

        if params.risk_per_trade_pct > 0.05:
            issues.append(f"Risk per trade very high: {params.risk_per_trade_pct:.2%}")

        if params.confidence_min_accepted > 0.85:
            issues.append(f"Confidence threshold too high: {params.confidence_min_accepted:.2f}")

        if params.confidence_min_accepted < 0.3:
            issues.append(f"Confidence threshold too low: {params.confidence_min_accepted:.2f}")

        if params.buy_threshold <= params.sell_threshold:
            issues.append("Buy threshold <= sell threshold (logic error)")

        if issues:
            for issue in issues:
                self.logger.warning(f"⚠️ {issue}")
            return False

        return True
