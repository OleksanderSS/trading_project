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
import logging
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.core.logging.logger import ProjectLogger
from src.risk_management import VaRCalculator


@dataclass
class PositionSizingParams:
    """Parameters for position sizing calculation"""
    portfolio_value: float
    volatility: float
    confidence: float
    max_drawdown: float = 0.0
    active_positions: int = 0
    market_regime: str = 'NORMAL'
    daily_volume: Optional[float] = None
    current_price: Optional[float] = None
    historical_returns: Optional[np.ndarray] = None


@dataclass
class LiquidityParams:
    """Parameters for liquidity adjustment calculation"""
    base_size: float
    var_adjustment: float
    kelly_adjustment: float
    daily_volume: Optional[float]
    current_price: Optional[float]


class AdaptivePositionSizer:
    """Адаптивно розраховує розмір позиції з використанням сучасних методів"""

    def __init__(self, config: Optional[Dict[str, Any]]=None):
        self.logger = ProjectLogger.get_logger('AdaptivePositionSizer')
        self.config = config or {}
        self._initialize_position_parameters()
        self._initialize_kelly_parameters()
        self._initialize_liquidity_parameters()
        self._initialize_regime_multipliers()
        self._initialize_var_calculator()

    def _initialize_position_parameters(self):
        """Initialize basic position sizing parameters"""
        self.base_position_size_pct = self.config.get('base_position_size_pct',
            0.02)
        self.max_position_size_pct = self.config.get('max_position_size_pct',
            0.1)
        self.min_position_size_pct = self.config.get('min_position_size_pct',
            0.005)
        self.max_active_positions = self.config.get('max_active_positions', 10)

    def _initialize_kelly_parameters(self):
        """Initialize Kelly Criterion parameters"""
        self.use_kelly = self.config.get('use_kelly_criterion', True)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.5)

    def _initialize_liquidity_parameters(self):
        """Initialize liquidity constraint parameters"""
        self.liquidity_threshold = self.config.get('liquidity_threshold', 0.01)

    def _initialize_regime_multipliers(self):
        """Initialize market regime adaptation multipliers"""
        self.regime_multipliers = {'TRENDING_UP': 1.2, 'TRENDING_DOWN': 0.8,
            'RANGING': 1.0, 'VOLATILE': 0.6, 'CRISIS': 0.3}

    def _initialize_var_calculator(self):
        """Initialize VaR calculator for risk-based sizing"""
        self.var_calculator = VaRCalculator()

    def calculate_position_size(self, params: PositionSizingParams) ->Dict[
        str, Any]:
        """Calculate position size using parameter object"""
        return self._calculate_position_size_from_params(params)

    @staticmethod
    def create_params(portfolio_value: float, volatility: float, confidence:
        float, **kwargs) ->PositionSizingParams:
        """
        Factory method to create PositionSizingParams with required params only.
        Optional params passed as keyword arguments.
        
        Args:
            portfolio_value: Required portfolio value
            volatility: Required volatility
            confidence: Required confidence score
            **kwargs: Optional parameters (max_drawdown, active_positions, etc.)
        """
        return PositionSizingParams(portfolio_value=portfolio_value,
            volatility=volatility, confidence=confidence, max_drawdown=
            kwargs.get('max_drawdown', 0.0), active_positions=kwargs.get(
            'active_positions', 0), market_regime=kwargs.get(
            'market_regime', 'NORMAL'), daily_volume=kwargs.get(
            'daily_volume', None), current_price=kwargs.get('current_price',
            None), historical_returns=kwargs.get('historical_returns', None))

    def calculate_position_size_legacy(self, portfolio_value: float,
        volatility: float, confidence: float, max_drawdown: float=0.0,
        active_positions: int=0, market_regime: str='NORMAL', daily_volume:
        Optional[float]=None, current_price: Optional[float]=None,
        historical_returns: Optional[np.ndarray]=None) ->Dict[str, Any]:
        """
        @deprecated: Use calculate_position_size(params) instead.
        This method has too many parameters and is kept for backward compatibility only.
        
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
        params = self.create_params(portfolio_value=portfolio_value,
            volatility=volatility, confidence=confidence, max_drawdown=
            max_drawdown, active_positions=active_positions, market_regime=
            market_regime, daily_volume=daily_volume, current_price=
            current_price, historical_returns=historical_returns)
        return self._calculate_position_size_from_params(params)

    def _calculate_position_size_from_params(self, params: PositionSizingParams
        ) ->Dict[str, Any]:
        """Розраховує розмір позиції з параметрів"""
        try:
            base_size = params.portfolio_value * self.base_position_size_pct
            var_adjustment = self._calculate_var_adjustment(params.
                historical_returns)
            kelly_adjustment = self._calculate_kelly_adjustment(params.
                confidence)
            conf_adjustment = (params.confidence if params.confidence is not
                None else 1.0)
            vol_adjustment = self._calculate_volatility_adjustment(params.
                volatility)
            dd_adjustment = self._calculate_drawdown_adjustment(params.
                max_drawdown)
            pos_adjustment = self._calculate_positions_adjustment(params.
                active_positions)
            regime_adjustment = self.regime_multipliers.get(params.
                market_regime, 1.0)
            liquidity_params = LiquidityParams(base_size=base_size,
                var_adjustment=var_adjustment, kelly_adjustment=
                kelly_adjustment, daily_volume=params.daily_volume,
                current_price=params.current_price)
            liquidity_adjustment = self._calculate_liquidity_adjustment(
                liquidity_params)
            position_size = (base_size * var_adjustment * kelly_adjustment *
                conf_adjustment * vol_adjustment * dd_adjustment *
                pos_adjustment * regime_adjustment * liquidity_adjustment)
            position_size = self._apply_position_limits(position_size,
                params.portfolio_value)
            position_size_pct = position_size / params.portfolio_value
            expected_risk = params.volatility * position_size_pct
            risk_adjusted_return = params.confidence * position_size_pct
            return {'position_size': float(position_size),
                'position_size_pct': float(position_size_pct), 'base_size':
                float(base_size), 'var_adjustment': float(var_adjustment),
                'kelly_adjustment': float(kelly_adjustment),
                'conf_adjustment': float(conf_adjustment), 'vol_adjustment':
                float(vol_adjustment), 'dd_adjustment': float(dd_adjustment
                ), 'pos_adjustment': float(pos_adjustment),
                'regime_adjustment': float(regime_adjustment),
                'liquidity_adjustment': float(liquidity_adjustment),
                'effective_multiplier': float(var_adjustment *
                kelly_adjustment * conf_adjustment * vol_adjustment *
                dd_adjustment * pos_adjustment * regime_adjustment *
                liquidity_adjustment), 'expected_risk_pct': float(
                expected_risk), 'risk_adjusted_return': float(
                risk_adjusted_return), 'sharpe_contribution': float(
                risk_adjusted_return / expected_risk) if expected_risk > 0 else
                0.0, 'market_regime': params.market_regime,
                'liquidity_constrained': liquidity_adjustment < 1.0,
                'kelly_fraction_used': self.kelly_fraction,
                'var_based_sizing': params.historical_returns is not None}
        except (ValueError, TypeError, ZeroDivisionError, AttributeError) as e:
            self.logger.error(f'Помилка розрахунку розміру позиції: {e}',
                exc_info=True)
            fallback_size = (params.portfolio_value * self.
                base_position_size_pct)
            return {'position_size': float(fallback_size),
                'position_size_pct': float(self.base_position_size_pct),
                'error': str(e), 'fallback_used': True}

    def _calculate_var_adjustment(self, historical_returns: Optional[np.
        ndarray]) ->float:
        """Calculate VaR-based adjustment factor"""
        if historical_returns is None or len(historical_returns) <= 30:
            return 1.0
        try:
            var_result = self.var_calculator.calculate_var_historical(
                historical_returns, confidence=0.95, time_horizon=1)
            if 'var' in var_result:
                target_risk_pct = self.base_position_size_pct
                var_pct = abs(var_result['var'])
                if var_pct > 0:
                    return min(target_risk_pct / var_pct, 2.0)
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'VaR calculation failed: {e}')
            raise Exception(f'VaR calculation failed: {e}') from e
        return 1.0

    def _calculate_kelly_adjustment(self, confidence: float) ->float:
        """Calculate Kelly Criterion adjustment factor"""
        if not self.use_kelly or confidence is None:
            return 1.0
        kelly_f = (confidence - (1 - confidence)) / 1.0
        return max(0.1, min(kelly_f * self.kelly_fraction, 2.0))

    def _calculate_volatility_adjustment(self, volatility: float) ->float:
        """Calculate volatility-based adjustment factor"""
        vol_adjustment = 1.0 / (1.0 + volatility * 5)
        return np.clip(vol_adjustment, 0.3, 1.0)

    def _calculate_drawdown_adjustment(self, max_drawdown: float) ->float:
        """Calculate drawdown-based adjustment factor"""
        dd_adjustment = 1.0 - max_drawdown * 2
        return np.clip(dd_adjustment, 0.3, 1.0)

    def _calculate_positions_adjustment(self, active_positions: int) ->float:
        """Calculate adjustment based on number of active positions"""
        if active_positions <= 0:
            return 1.0
        pos_adjustment = 1.0 / (1.0 + active_positions / self.
            max_active_positions)
        return np.clip(pos_adjustment, 0.5, 1.0)

    def _calculate_liquidity_adjustment(self, params: LiquidityParams) ->float:
        """Calculate liquidity-based adjustment factor"""
        if not self._has_valid_liquidity_data(params):
            return 1.0
        position_value = self._calculate_position_value(params)
        max_safe_position = self._calculate_max_safe_position(params)
        if position_value <= max_safe_position:
            return 1.0
        return self._compute_liquidity_adjustment(position_value,
            max_safe_position)

    def _has_valid_liquidity_data(self, params: LiquidityParams) ->bool:
        """Check if liquidity data is valid"""
        return (params.daily_volume and params.current_price and params.
            current_price > 0)

    def _calculate_position_value(self, params: LiquidityParams) ->float:
        """Calculate position value"""
        return (params.base_size * params.var_adjustment * params.
            kelly_adjustment)

    def _calculate_max_safe_position(self, params: LiquidityParams) ->float:
        """Calculate maximum safe position based on liquidity"""
        return (params.daily_volume * params.current_price * self.
            liquidity_threshold)

    def _compute_liquidity_adjustment(self, position_value: float,
        max_safe_position: float) ->float:
        """Compute final liquidity adjustment"""
        liquidity_adjustment = max_safe_position / position_value
        return np.clip(liquidity_adjustment, 0.1, 1.0)

    def _apply_position_limits(self, position_size: float, portfolio_value:
        float) ->float:
        """Apply min/max position size limits"""
        return np.clip(position_size, portfolio_value * self.
            min_position_size_pct, portfolio_value * self.max_position_size_pct
            )

    def calculate_kelly_fraction(self, win_rate: float, avg_win: float,
        avg_loss: float) ->float:
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
            if self._is_invalid_kelly_input(avg_loss, win_rate):
                return 0.0
            kelly = self._compute_kelly_fraction(avg_win, avg_loss, win_rate)
            return float(np.clip(kelly, 0, 0.25))
        except Exception as e:
            self.logger.error(f'Error computing Kelly fraction: {e}',
                exc_info=True)
            return 0.0

    def _is_invalid_kelly_input(self, avg_loss: float, win_rate: float) ->bool:
        """Check if Kelly calculation inputs are invalid"""
        return avg_loss == 0 or win_rate <= 0 or win_rate >= 1

    def _compute_kelly_fraction(self, avg_win: float, avg_loss: float,
        win_rate: float) ->float:
        """Compute Kelly fraction"""
        try:
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            return (b * p - q) / b
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(f'Помилка розрахунку Kelly fraction: {e}')
            return 0.0
