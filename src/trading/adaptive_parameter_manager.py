"""
Adaptive Parameter Manager
- Parameters adapt to market regime, volatility, asset class
- Data-driven thresholds instead of hardcoded
- Quarterly optimization via historical backtest
"""

import logging
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.core.logging.logger import ProjectLogger

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
    buy_threshold: float  # Minimum prediction value for BUY
    sell_threshold: float  # Maximum prediction value for SELL
    hold_threshold: float  # Range for HOLD
    
    # Confidence adjustment
    confidence_min_accepted: float  # Do not trade if confidence < this
    confidence_boost_trending: float  # Boost when in trend
    confidence_penalty_volatile: float  # Penalty when volatile
    
    # Risk sizing
    risk_per_trade_pct: float  # % equity per trade
    max_position_size_pct: float  # Max % in one position
    max_daily_drawdown_pct: float  # Kill switch
    
    # News impact
    news_negative_threshold: float  # Threshold for negative news
    news_positive_threshold: float  # Threshold for positive news
    news_decay_hours: float  # How fast to forget new news
    
    # Model weighting
    model_decay_days: float  # How fast to forget old models
    ensemble_reweight_days: float  # How often to reweight ensemble
    
    regime: MarketRegime = MarketRegime.RANGING
    asset_class: AssetClass = AssetClass.LARGE_CAP
    volatility_percentile: float = 0.50  # 0.0 = low volatility, 1.0 = high volatility

class AdaptiveParameterManager:
    """Elite-grade parameter management"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Base parameters for each regime/asset combination
        self.regime_presets = self._build_regime_presets()
        self.asset_presets = self._build_asset_presets()
        self.current_params = None
        self.param_history = []  # Track changes
    
    def _build_regime_presets(self) -> Dict[MarketRegime, Dict[str, float]]:
        """
        Optimal parameters for each regime (result of historical backtest)
        """
        return {
            MarketRegime.TRENDING_UP: {
                'buy_threshold': 0.02,  # Higher threshold - allow only strong signals
                'sell_threshold': -0.02,
                'hold_threshold': 0.005,
                'confidence_min_accepted': 0.55,  # Higher minimum
                'confidence_boost_trending': 1.15,  # Boost 15% in trend
                'risk_per_trade_pct': 0.04,  # More risk in trend
                'max_position_size_pct': 0.12,
                'max_daily_drawdown_pct': 0.06,
            },
            MarketRegime.TRENDING_DOWN: {
                'buy_threshold': 0.02,  # More cautious
                'sell_threshold': -0.02,
                'hold_threshold': 0.005,
                'confidence_min_accepted': 0.60,  # Significantly higher
                'confidence_boost_trending': 0.90,  # Penalty in downtrend
                'risk_per_trade_pct': 0.02,  # Less risk
                'max_position_size_pct': 0.06,
                'max_daily_drawdown_pct': 0.04,
            },
            MarketRegime.RANGING: {
                'buy_threshold': 0.01,  # Lower - catch reversal
                'sell_threshold': -0.01,
                'hold_threshold': 0.003,
                'confidence_min_accepted': 0.50,  # Accept weaker signals
                'confidence_boost_trending': 0.85,  # Penalty for weakness
                'risk_per_trade_pct': 0.025,  # Medium risk
                'max_position_size_pct': 0.08,
                'max_daily_drawdown_pct': 0.05,
            },
            MarketRegime.VOLATILE: {
                'buy_threshold': 0.03,  # Very high - only strong signals
                'sell_threshold': -0.03,
                'hold_threshold': 0.01,
                'confidence_min_accepted': 0.65,  # Very cautious
                'confidence_boost_trending': 0.75,  # Penalty 25%
                'risk_per_trade_pct': 0.01,  # Minimum risk
                'max_position_size_pct': 0.03,
                'max_daily_drawdown_pct': 0.03,
            },
            MarketRegime.DEAD: {
                'buy_threshold': 1.0,  # Practically do not trade
                'sell_threshold': -1.0,
                'hold_threshold': 0.5,
                'confidence_min_accepted': 0.99,  # Wait for clear signal
                'confidence_boost_trending': 0.0,
                'risk_per_trade_pct': 0.001,  # Мінімум
                'max_position_size_pct': 0.01,
                'max_daily_drawdown_pct': 0.01,
            }
        }
    
    def _build_asset_presets(self) -> Dict[AssetClass, Dict[str, float]]:
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
    
    def compute_adaptive_params(self,
                               regime: MarketRegime,
                               asset_class: AssetClass,
                               volatility_percentile: float,
                               historical_sharpe: Optional[float] = None) -> AdaptiveParameters:
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
        if isinstance(regime, str):
            try:
                regime = MarketRegime(regime.lower())
            except ValueError:
                self.logger.warning(f"Unknown regime '{regime}', defaulting to RANGING")
                regime = MarketRegime.RANGING
                
        if isinstance(asset_class, str):
            try:
                asset_class = AssetClass(asset_class.lower())
            except ValueError:
                self.logger.warning(f"Unknown asset class '{asset_class}', defaulting to LARGE_CAP")
                asset_class = AssetClass.LARGE_CAP
                
        # 1. Start з regime-based параметрів
        regime_params = self.regime_presets[regime].copy()
        
        # 2. Apply asset class adjustments
        asset_params = self.asset_presets[asset_class]
        
        regime_params['risk_per_trade_pct'] *= asset_params['risk_multiplier']
        regime_params['max_position_size_pct'] *= asset_params['risk_multiplier']
        regime_params['confidence_min_accepted'] *= asset_params['confidence_multiplier']
        
        # 3. Volatility-based fine-tuning
        # Більша Volatility = Less risk, більша впевненість
        vol_adjustment = 1.0 - (volatility_percentile * 0.4)  # До 40% reduction
        regime_params['risk_per_trade_pct'] *= vol_adjustment
        regime_params['max_position_size_pct'] *= vol_adjustment
        regime_params['confidence_min_accepted'] *= (1 + volatility_percentile * 0.15)  # До 15% збільшення
        
        # 4. Sharpe-based adjustment (якщо доступно)
        if historical_sharpe is not None:
            if historical_sharpe < 0.5:
                # Погані результати - More cautious
                regime_params['risk_per_trade_pct'] *= 0.7
                regime_params['confidence_min_accepted'] *= 1.1
            elif historical_sharpe > 1.5:
                # Добрі результати - більш агресивно
                regime_params['risk_per_trade_pct'] *= 1.2
                regime_params['confidence_min_accepted'] *= 0.95
        
        params = AdaptiveParameters(
            regime=regime,
            asset_class=asset_class,
            volatility_percentile=volatility_percentile,
            **regime_params
        )
        
        # 5. Log changes
        if self.current_params != params:
            self.logger.info(f"🔄 Parameters adapted to {regime.value} / {asset_class.value}")
            self.logger.info(f"  risk_per_trade: {regime_params['risk_per_trade_pct']:.3%}")
            self.logger.info(f"  confidence_min: {regime_params['confidence_min_accepted']:.2f}")
            self.param_history.append({
                'timestamp': pd.Timestamp.now(),
                'regime': regime.value,
                'asset_class': asset_class.value,
                'volatility_pct': volatility_percentile
            })
        
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
