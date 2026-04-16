"""
Адаптивний розмір позиції (Adaptive Position Sizing).

Розраховує розмір позиції на основі:
1. Волатильності (VaR-based)
2. Confidence score моделі
3. Максимального drawdown портфеля
4. Кількості активних позицій
5. Risk-adjusted sizing (Kelly Criterion)
6. Liquidity constraints
7. Market regime adaptation
"""

import numpy as np
from typing import Dict, Any, Optional, List
from src.core.logging.logger import ProjectLogger
from src.risk_management import VaRCalculator

class AdaptivePositionSizer:
    """Адаптивно розраховує розмір позиції з використанням сучасних методів"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = ProjectLogger.get_logger("AdaptivePositionSizer")
        self.config = config or {}
        
        # Параметри
        self.base_position_size_pct = self.config.get('base_position_size_pct', 0.02)  # 2%
        self.max_position_size_pct = self.config.get('max_position_size_pct', 0.10)    # 10%
        self.min_position_size_pct = self.config.get('min_position_size_pct', 0.005)   # 0.5%
        self.max_active_positions = self.config.get('max_active_positions', 10)
        
        # Kelly Criterion параметри
        self.use_kelly = self.config.get('use_kelly_criterion', True)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.5)  # Conservative Kelly
        
        # Liquidity constraints
        self.liquidity_threshold = self.config.get('liquidity_threshold', 0.01)  # 1% of daily volume
        
        # Market regime adaptation
        self.regime_multipliers = {
            'TRENDING_UP': 1.2,      # Increase size in uptrends
            'TRENDING_DOWN': 0.8,    # Decrease size in downtrends  
            'RANGING': 1.0,          # Normal size in ranging markets
            'VOLATILE': 0.6,         # Reduce size in high volatility
            'CRISIS': 0.3            # Minimal size in crisis
        }
        
        # VaR calculator для risk-based sizing
        self.var_calculator = VaRCalculator()
    
    def calculate_position_size(self,
                               portfolio_value: float,
                               volatility: float,
                               confidence: float,
                               max_drawdown: float = 0.0,
                               active_positions: int = 0,
                               market_regime: str = 'NORMAL',
                               daily_volume: Optional[float] = None,
                               current_price: Optional[float] = None,
                               historical_returns: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Розраховує розмір позиції з використанням всіх факторів
        
        Args:
            portfolio_value: Вартість портфеля
            volatility: Волатильність активу
            confidence: Confidence score моделі (0-1)
            max_drawdown: Максимальний drawdown портфеля (0-1)
            active_positions: Кількість активних позицій
            market_regime: Поточний ринковий режим
            daily_volume: Добовий обсяг торгів (для liquidity check)
            current_price: Поточна ціна активу
            historical_returns: Історичні повернення для VaR
            
        Returns:
            Dict з розміром позиції та детальними розрахунками
        """
        try:
            # 1. Базовий розмір
            base_size = portfolio_value * self.base_position_size_pct
            
            # 2. VaR-based sizing (якщо є історичні дані)
            var_adjustment = 1.0
            if historical_returns is not None and len(historical_returns) > 30:
                try:
                    var_result = self.var_calculator.calculate_var_historical(
                        historical_returns, confidence=0.95, time_horizon=1
                    )
                    if 'var' in var_result:
                        # VaR-based position sizing: position = portfolio_value * (target_risk / VaR)
                        target_risk_pct = self.base_position_size_pct
                        var_pct = abs(var_result['var'])
                        if var_pct > 0:
                            var_adjustment = min(target_risk_pct / var_pct, 2.0)  # Max 2x adjustment
                except Exception as e:
                    self.logger.debug(f"VaR calculation failed, using base size: {e}")
            
            # 3. Kelly Criterion sizing (якщо включено)
            kelly_adjustment = 1.0
            if self.use_kelly and confidence is not None:
                # Simplified Kelly: f = (p - q) / b, де p=confidence, q=1-p, b=1 (even odds)
                kelly_f = (confidence - (1 - confidence)) / 1.0
                kelly_adjustment = max(0.1, min(kelly_f * self.kelly_fraction, 2.0))
            
            # 4. Confidence adjustment
            conf_adjustment = confidence if confidence is not None else 1.0
            
            # 5. Volatility adjustment (inverse relationship)
            vol_adjustment = 1.0 / (1.0 + volatility * 5)  # Reduce size with volatility
            vol_adjustment = np.clip(vol_adjustment, 0.3, 1.0)
            
            # 6. Drawdown adjustment
            dd_adjustment = 1.0 - max_drawdown * 2  # Reduce size with drawdown
            dd_adjustment = np.clip(dd_adjustment, 0.3, 1.0)
            
            # 7. Active positions adjustment
            if active_positions > 0:
                pos_adjustment = 1.0 / (1.0 + active_positions / self.max_active_positions)
                pos_adjustment = np.clip(pos_adjustment, 0.5, 1.0)
            else:
                pos_adjustment = 1.0
            
            # 8. Market regime adjustment
            regime_adjustment = self.regime_multipliers.get(market_regime, 1.0)
            
            # 9. Liquidity adjustment
            liquidity_adjustment = 1.0
            if daily_volume and current_price and current_price > 0:
                position_value = base_size * var_adjustment * kelly_adjustment
                max_safe_position = daily_volume * current_price * self.liquidity_threshold
                if position_value > max_safe_position:
                    liquidity_adjustment = max_safe_position / position_value
                    liquidity_adjustment = np.clip(liquidity_adjustment, 0.1, 1.0)
            
            # 10. Комбінуємо всі фактори
            position_size = (base_size * 
                           var_adjustment * 
                           kelly_adjustment * 
                           conf_adjustment * 
                           vol_adjustment * 
                           dd_adjustment * 
                           pos_adjustment * 
                           regime_adjustment * 
                           liquidity_adjustment)
            
            # 11. Обмежуємо розмір
            position_size = np.clip(
                position_size,
                portfolio_value * self.min_position_size_pct,
                portfolio_value * self.max_position_size_pct
            )
            
            # 12. Розраховуємо відсоток від портфеля
            position_size_pct = position_size / portfolio_value
            
            # 13. Risk metrics
            expected_risk = volatility * position_size_pct
            risk_adjusted_return = confidence * position_size_pct
            
            return {
                'position_size': float(position_size),
                'position_size_pct': float(position_size_pct),
                'base_size': float(base_size),
                
                # Individual adjustments
                'var_adjustment': float(var_adjustment),
                'kelly_adjustment': float(kelly_adjustment),
                'conf_adjustment': float(conf_adjustment),
                'vol_adjustment': float(vol_adjustment),
                'dd_adjustment': float(dd_adjustment),
                'pos_adjustment': float(pos_adjustment),
                'regime_adjustment': float(regime_adjustment),
                'liquidity_adjustment': float(liquidity_adjustment),
                
                # Combined multiplier
                'effective_multiplier': float(var_adjustment * kelly_adjustment * conf_adjustment * 
                                            vol_adjustment * dd_adjustment * pos_adjustment * 
                                            regime_adjustment * liquidity_adjustment),
                
                # Risk metrics
                'expected_risk_pct': float(expected_risk),
                'risk_adjusted_return': float(risk_adjusted_return),
                'sharpe_contribution': float(risk_adjusted_return / expected_risk) if expected_risk > 0 else 0.0,
                
                # Market context
                'market_regime': market_regime,
                'liquidity_constrained': liquidity_adjustment < 1.0,
                'kelly_fraction_used': self.kelly_fraction,
                'var_based_sizing': historical_returns is not None
            }
        
        except Exception as e:
            self.logger.error(f"Помилка розрахунку розміру позиції: {e}")
            # Fallback to simple calculation
            fallback_size = portfolio_value * self.base_position_size_pct
            return {
                'position_size': float(fallback_size),
                'position_size_pct': float(self.base_position_size_pct),
                'error': str(e),
                'fallback_used': True
            }
    
    def calculate_position_size(self,
                               portfolio_value: float,
                               volatility: float,
                               confidence: float,
                               max_drawdown: float = 0.0,
                               active_positions: int = 0) -> Dict[str, Any]:
        """
        Розраховує розмір позиції
        
        Args:
            portfolio_value: Вартість портфеля
            volatility: Волатильність активу
            confidence: Confidence score (0-1)
            max_drawdown: Максимальний drawdown портфеля (0-1)
            active_positions: Кількість активних позицій
        
        Returns:
            Dict з розміром позиції та параметрами
        """
        try:
            # Базовий розмір
            base_size = portfolio_value * self.base_position_size_pct
            
            # 1. Коригування за волатильністю
            # Вища волатильність = менша позиція
            vol_factor = 1.0 / (1.0 + volatility * 10)
            vol_factor = np.clip(vol_factor, 0.3, 1.0)
            
            # 2. Коригування за confidence
            # Нижча впевненість = менша позиція
            conf_factor = confidence
            
            # 3. Коригування за drawdown
            # Більший drawdown = менша позиція
            dd_factor = 1.0 - max_drawdown
            dd_factor = np.clip(dd_factor, 0.3, 1.0)
            
            # 4. Коригування за кількістю позицій
            # Більше позицій = менша позиція на кожну
            if active_positions > 0:
                pos_factor = 1.0 / (1.0 + active_positions / self.max_active_positions)
            else:
                pos_factor = 1.0
            
            # Комбінуємо всі фактори
            position_size = base_size * vol_factor * conf_factor * dd_factor * pos_factor
            
            # Обмежуємо розмір
            position_size = np.clip(
                position_size,
                portfolio_value * self.min_position_size_pct,
                portfolio_value * self.max_position_size_pct
            )
            
            # Розраховуємо відсоток від портфеля
            position_size_pct = position_size / portfolio_value
            
            return {
                'position_size': float(position_size),
                'position_size_pct': float(position_size_pct),
                'base_size': float(base_size),
                'vol_factor': float(vol_factor),
                'conf_factor': float(conf_factor),
                'dd_factor': float(dd_factor),
                'pos_factor': float(pos_factor),
                'effective_multiplier': float(vol_factor * conf_factor * dd_factor * pos_factor)
            }
        
        except Exception as e:
            self.logger.error(f"Помилка розрахунку розміру позиції: {e}")
            return {
                'position_size': portfolio_value * self.base_position_size_pct,
                'position_size_pct': self.base_position_size_pct,
                'base_size': portfolio_value * self.base_position_size_pct,
                'vol_factor': 1.0,
                'conf_factor': 1.0,
                'dd_factor': 1.0,
                'pos_factor': 1.0,
                'effective_multiplier': 1.0
            }
    
    def calculate_kelly_fraction(self,
                                win_rate: float,
                                avg_win: float,
                                avg_loss: float) -> float:
        """
        Розраховує Kelly Fraction для оптимального розміру позиції
        
        Kelly % = (bp - q) / b
        де:
        - b = avg_win / avg_loss (коефіцієнт виграшу)
        - p = win_rate (ймовірність виграшу)
        - q = 1 - p (ймовірність програшу)
        
        Args:
            win_rate: Відсоток виграшних торгів (0-1)
            avg_win: Середній виграш
            avg_loss: Середній програш
        
        Returns:
            Kelly fraction (0-1)
        """
        try:
            if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0
            
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            
            kelly = (b * p - q) / b
            
            # Обмежуємо Kelly fraction (зазвичай використовуємо 25% від Kelly)
            kelly = np.clip(kelly, 0, 0.25)
            
            return float(kelly)
        
        except Exception as e:
            self.logger.warning(f"Помилка розрахунку Kelly fraction: {e}")
            return 0.0
