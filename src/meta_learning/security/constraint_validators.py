#!/usr/bin/env python3
"""
Constraint Validators - Validation Logic for Security Constraints

This module contains the validation logic for various security constraint types.
Each validator checks specific conditions and returns True if the constraint is satisfied.
"""

from typing import Dict, Any
from datetime import datetime, timedelta

from src.core.logging.logger import ProjectLogger
from src.core.error_handling.error_handler import IErrorHandler, ErrorHandler


class ConstraintValidators:
    """
    Collection of constraint validation methods.
    
    This class encapsulates the validation logic for different types of security constraints,
    making it easier to maintain and test individual validators.
    """
    
    def __init__(self, logger=None, error_handler: IErrorHandler = None):
        """
        Initialize the constraint validators.
        
        Args:
            logger: Logger instance
            error_handler: Error handler instance
        """
        self.logger = logger or ProjectLogger.get_logger(self.__class__.__name__)
        self.error_handler = error_handler or ErrorHandler()
    
    def validate_position_size(self, context: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """Validate position size constraints."""
        try:
            position_size = context.get('position_size', 0)
            portfolio_value = context.get('portfolio_value', 0)
            
            # Check percentage of portfolio
            if portfolio_value > 0:
                position_pct = position_size / portfolio_value
                max_pct = constraints.get('max_position_size', {}).get('max_position_pct', 0.1)
                if position_pct > max_pct:
                    return False
            
            # Check absolute position size
            max_absolute = constraints.get('max_position_size', {}).get('max_absolute_position', 1000000)
            if position_size > max_absolute:
                return False
            
            return True
            
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Position size validation error: {e}", exc_info=True)
            self.error_handler.handle_error(e, context={'context': context})
            raise RuntimeError(f"Position size validation failed: {e}") from e
    
    def validate_risk_exposure(self, context: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """Validate risk exposure constraints."""
        try:
            current_risk = context.get('current_risk_exposure', 0)
            portfolio_value = context.get('portfolio_value', 0)
            trade_risk = context.get('trade_risk', 0)
            
            # Check portfolio risk
            if portfolio_value > 0:
                portfolio_risk_pct = current_risk / portfolio_value
                max_portfolio_risk = constraints.get('max_risk_exposure', {}).get('max_portfolio_risk', 0.3)
                if portfolio_risk_pct > max_portfolio_risk:
                    return False
            
            # Check single trade risk
            max_single_risk = constraints.get('max_risk_exposure', {}).get('max_single_trade_risk', 0.02)
            if trade_risk > max_single_risk:
                return False
            
            return True
            
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Risk exposure validation error: {e}", exc_info=True)
            self.error_handler.handle_error(e, context={'context': context})
            raise RuntimeError(f"Risk exposure validation failed: {e}") from e
    
    def validate_trading_frequency(
        self, 
        context: Dict[str, Any], 
        constraints: Dict[str, Any],
        violation_history: Dict[str, list]
    ) -> bool:
        """Validate trading frequency constraints."""
        try:
            agent_id = context.get('agent_id', '')
            current_time = datetime.now()
            
            # Initialize agent tracking if needed
            if agent_id not in violation_history:
                violation_history[agent_id] = []
            
            # Get recent trades for this agent
            recent_trades = [
                ts for ts in violation_history[agent_id]
                if ts > current_time - timedelta(hours=1)
            ]
            
            max_per_hour = constraints.get('trading_frequency_limit', {}).get('max_trades_per_hour', 50)
            if len(recent_trades) >= max_per_hour:
                return False
            
            # Check minimum time between trades
            min_time_between = constraints.get('trading_frequency_limit', {}).get('min_time_between_trades', 60)
            if recent_trades:
                time_since_last = (current_time - recent_trades[-1]).total_seconds()
                if time_since_last < min_time_between:
                    return False
            
            return True
            
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Trading frequency validation error: {e}", exc_info=True)
            self.error_handler.handle_error(e, context={'context': context})
            raise RuntimeError(f"Trading frequency validation failed: {e}") from e
    
    def validate_consecutive_losses(self, context: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """Validate consecutive losses constraint."""
        try:
            recent_results = context.get('recent_trade_results', [])
            max_consecutive = constraints.get('consecutive_losses_limit', {}).get('max_consecutive_losses', 5)
            loss_threshold = constraints.get('consecutive_losses_limit', {}).get('loss_threshold_pct', 0.01)
            
            consecutive_losses = 0
            for result in reversed(recent_results[-max_consecutive-1:]):  # Check last N+1 results
                if result < -loss_threshold:  # Loss exceeds threshold
                    consecutive_losses += 1
                else:
                    break
            
            return bool(consecutive_losses < max_consecutive)
            
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Consecutive losses validation error: {e}", exc_info=True)
            self.error_handler.handle_error(e, context={'context': context})
            raise RuntimeError(f"Consecutive losses validation failed: {e}") from e
    
    def validate_volatility_limits(self, context: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """Validate volatility constraints."""
        try:
            current_volatility = context.get('current_volatility', 0)
            
            max_volatility = constraints.get('volatility_limit', {}).get('max_volatility_threshold', 0.05)
            
            return bool(current_volatility <= max_volatility)
            
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Volatility validation error: {e}")
            self.error_handler.handle_error(e, context={'context': context})
            raise RuntimeError(f"Volatility validation failed: {e}") from e
    
    def validate_liquidity_requirements(self, context: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """Validate liquidity requirements."""
        try:
            daily_volume = context.get('daily_volume_usd', 0)
            avg_spread = context.get('avg_spread_bps', 0)
            
            min_volume = constraints.get('liquidity_requirement', {}).get('min_daily_volume_usd', 1000000)
            max_spread = constraints.get('liquidity_requirement', {}).get('min_avg_spread_bps', 50)
            
            return bool(daily_volume >= min_volume and avg_spread <= max_spread)
            
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Liquidity validation error: {e}")
            self.error_handler.handle_error(e, context={'context': context})
            raise RuntimeError(f"Liquidity validation failed: {e}") from e
    
    def validate_correlation_limits(self, context: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """Validate correlation constraints."""
        try:
            current_positions = context.get('current_positions', {})
            symbol_correlations = context.get('symbol_correlations', {})
            
            max_correlation = constraints.get('correlation_limit', {}).get('max_correlation', 0.8)
            
            # Check correlation with existing positions
            for existing_symbol, correlation in symbol_correlations.items():
                if existing_symbol in current_positions and correlation > max_correlation:
                    return False
            
            return True
            
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Correlation validation error: {e}")
            self.error_handler.handle_error(e, context={'context': context})
            raise RuntimeError(f"Correlation validation failed: {e}") from e
