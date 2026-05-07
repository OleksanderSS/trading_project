"""
Security Constraint Engine - Real-time Safety Validation for Meta-Learning Agents

Provides comprehensive constraint validation and enforcement for all agent actions.
Critical for preventing unsafe agent behavior in production environments.
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config

logger = ProjectLogger.get_logger(__name__)

class ConstraintType(Enum):
    """Types of security constraints."""
    POSITION_SIZE = "position_size"
    RISK_EXPOSURE = "risk_exposure"
    TRADING_FREQUENCY = "trading_frequency"
    RESOURCE_USAGE = "resource_usage"
    MARKET_CONDITIONS = "market_conditions"
    CORRELATION_LIMITS = "correlation_limits"
    VOLATILITY_LIMITS = "volatility_limits"
    LIQUIDITY_REQUIREMENTS = "liquidity_requirements"
    TIME_RESTRICTIONS = "time_restrictions"
    CONSECUTIVE_LOSSES = "consecutive_losses"

class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Constraint:
    """Individual constraint definition."""
    name: str
    constraint_type: ConstraintType
    validator: Callable[[Dict[str, Any]], bool]
    severity: ConstraintSeverity
    description: str
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""
    timestamp: datetime
    constraint_name: str
    severity: ConstraintSeverity
    agent_id: str
    action_context: Dict[str, Any]
    violation_details: str
    recommended_action: str

class SecurityConstraintEngine:
    """
    Comprehensive security constraint validation engine.
    
    Features:
    - Real-time constraint validation
    - Multiple constraint types
    - Configurable severity levels
    - Violation tracking and reporting
    - Emergency stop triggers
    - Adaptive constraint tuning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Security Constraint Engine.
        
        Args:
            config: Configuration dictionary for constraints
        """
        self.config = config or {}
        self.logger = logger
        
        # Constraint storage
        self._constraints: Dict[str, Constraint] = {}
        self._constraint_violations: List[ConstraintViolation] = []
        
        # Engine settings
        self.enabled = self.config.get('enabled', True)
        self.strict_mode = self.config.get('strict_mode', True)
        self.emergency_stop_enabled = self.config.get('emergency_stop_enabled', True)
        
        # Violation tracking
        self.violation_history: Dict[str, List[datetime]] = {}
        self.max_violations_per_hour = self.config.get('max_violations_per_hour', 10)
        self.critical_violation_threshold = self.config.get('critical_violation_threshold', 3)
        
        # Market data cache
        self.market_data_cache: Dict[str, Any] = {}
        self.market_data_ttl = self.config.get('market_data_ttl', 300)  # 5 minutes
        
        # Thread safety
        self._lock = RLock()
        
        # Initialize default constraints
        self._initialize_default_constraints()
        
        self.logger.info("✅ SecurityConstraintEngine initialized")
    
    def validate_action(self, agent_id: str, action_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an agent action against all security constraints.
        
        Args:
            agent_id: Agent identifier
            action_context: Context information about the action
            
        Returns:
            Dictionary with validation results
        """
        if not self.enabled:
            return {
                'allowed': True,
                'reason': 'Constraint engine disabled',
                'violations': [],
                'warnings': []
            }
        
        with self._lock:
            try:
                violations = []
                warnings = []
                
                # Check all enabled constraints
                for constraint_name, constraint in self._constraints.items():
                    if not constraint.enabled:
                        continue
                    
                    try:
                        # Validate constraint
                        is_valid = constraint.validator(action_context)
                        
                        if not is_valid:
                            violation = self._create_violation(
                                constraint, agent_id, action_context
                            )
                            
                            if constraint.severity == ConstraintSeverity.WARNING:
                                warnings.append(violation)
                            else:
                                violations.append(violation)
                                self._record_violation(violation)
                    
                    except Exception as e:
                        self.logger.error(f"Error validating constraint {constraint_name}: {e}")
                        # Create a violation for the validation error itself
                        error_violation = ConstraintViolation(
                            timestamp=datetime.now(),
                            constraint_name=constraint_name,
                            severity=ConstraintSeverity.ERROR,
                            agent_id=agent_id,
                            action_context=action_context,
                            violation_details=f"Constraint validation error: {str(e)}",
                            recommended_action="Review constraint implementation"
                        )
                        violations.append(error_violation)
                
                # Determine if action is allowed
                critical_violations = [v for v in violations if v.severity == ConstraintSeverity.CRITICAL]
                error_violations = [v for v in violations if v.severity == ConstraintSeverity.ERROR]
                
                if critical_violations:
                    allowed = False
                    reason = f"Critical constraint violations: {[v.constraint_name for v in critical_violations]}"
                    
                    # Trigger emergency stop if enabled
                    if self.emergency_stop_enabled:
                        self._trigger_emergency_stop(agent_id, critical_violations)
                
                elif error_violations and self.strict_mode:
                    allowed = False
                    reason = f"Error constraint violations in strict mode: {[v.constraint_name for v in error_violations]}"
                
                elif error_violations and not self.strict_mode:
                    allowed = True
                    reason = f"Error violations present but strict mode disabled: {[v.constraint_name for v in error_violations]}"
                
                else:
                    allowed = True
                    reason = "All constraints satisfied" if not warnings else "Constraints satisfied with warnings"
                
                result = {
                    'allowed': allowed,
                    'reason': reason,
                    'violations': [self._violation_to_dict(v) for v in violations],
                    'warnings': [self._violation_to_dict(w) for w in warnings],
                    'constraint_engine': {
                        'enabled': self.enabled,
                        'strict_mode': self.strict_mode,
                        'total_constraints': len(self._constraints),
                        'enabled_constraints': len([c for c in self._constraints.values() if c.enabled])
                    }
                }
                
                self.logger.info(f"Constraint validation for {agent_id}: {'ALLOWED' if allowed else 'DENIED'} - {reason}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error in constraint validation for {agent_id}: {e}")
                return {
                    'allowed': False,
                    'reason': f'Constraint validation error: {str(e)}',
                    'violations': [],
                    'warnings': []
                }
    
    def add_constraint(self, constraint: Constraint) -> bool:
        """
        Add a new constraint to the engine.
        
        Args:
            constraint: Constraint to add
            
        Returns:
            True if constraint added successfully
        """
        with self._lock:
            try:
                self._constraints[constraint.name] = constraint
                self.logger.info(f"✅ Added constraint: {constraint.name}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to add constraint {constraint.name}: {e}")
                return False
    
    def remove_constraint(self, constraint_name: str) -> bool:
        """
        Remove a constraint from the engine.
        
        Args:
            constraint_name: Name of constraint to remove
            
        Returns:
            True if constraint removed successfully
        """
        with self._lock:
            try:
                if constraint_name in self._constraints:
                    del self._constraints[constraint_name]
                    self.logger.info(f"✅ Removed constraint: {constraint_name}")
                    return True
                else:
                    self.logger.warning(f"Constraint not found: {constraint_name}")
                    return False
            except Exception as e:
                self.logger.error(f"Failed to remove constraint {constraint_name}: {e}")
                return False
    
    def get_constraint_status(self) -> Dict[str, Any]:
        """Get current status of all constraints."""
        with self._lock:
            return {
                'total_constraints': len(self._constraints),
                'enabled_constraints': len([c for c in self._constraints.values() if c.enabled]),
                'constraints': {
                    name: {
                        'type': constraint.constraint_type.value,
                        'severity': constraint.severity.value,
                        'enabled': constraint.enabled,
                        'description': constraint.description
                    }
                    for name, constraint in self._constraints.items()
                },
                'recent_violations': len([
                    v for v in self._constraint_violations
                    if v.timestamp > datetime.now() - timedelta(hours=1)
                ]),
                'total_violations': len(self._constraint_violations),
                'engine_settings': {
                    'enabled': self.enabled,
                    'strict_mode': self.strict_mode,
                    'emergency_stop_enabled': self.emergency_stop_enabled
                }
            }
    
    def get_violation_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get violation history for specified time period."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_violations = [
                self._violation_to_dict(v) for v in self._constraint_violations
                if v.timestamp > cutoff_time
            ]
            
            return recent_violations
    
    def _initialize_default_constraints(self):
        """Initialize default security constraints."""
        
        # Position size constraint
        self.add_constraint(Constraint(
            name="max_position_size",
            constraint_type=ConstraintType.POSITION_SIZE,
            validator=self._validate_position_size,
            severity=ConstraintSeverity.ERROR,
            description="Limit maximum position size",
            parameters={
                'max_position_pct': 0.1,  # 10% of portfolio
                'max_absolute_position': 1000000  # $1M max
            }
        ))
        
        # Risk exposure constraint
        self.add_constraint(Constraint(
            name="max_risk_exposure",
            constraint_type=ConstraintType.RISK_EXPOSURE,
            validator=self._validate_risk_exposure,
            severity=ConstraintSeverity.CRITICAL,
            description="Limit total risk exposure",
            parameters={
                'max_portfolio_risk': 0.3,  # 30% of portfolio
                'max_single_trade_risk': 0.02  # 2% per trade
            }
        ))
        
        # Trading frequency constraint
        self.add_constraint(Constraint(
            name="trading_frequency_limit",
            constraint_type=ConstraintType.TRADING_FREQUENCY,
            validator=self._validate_trading_frequency,
            severity=ConstraintSeverity.WARNING,
            description="Limit trading frequency",
            parameters={
                'max_trades_per_hour': 50,
                'max_trades_per_day': 200,
                'min_time_between_trades': 60  # seconds
            }
        ))
        
        # Consecutive losses constraint
        self.add_constraint(Constraint(
            name="consecutive_losses_limit",
            constraint_type=ConstraintType.CONSECUTIVE_LOSSES,
            validator=self._validate_consecutive_losses,
            severity=ConstraintSeverity.CRITICAL,
            description="Limit consecutive losses",
            parameters={
                'max_consecutive_losses': 5,
                'loss_threshold_pct': 0.01  # 1%
            }
        ))
        
        # Volatility constraint
        self.add_constraint(Constraint(
            name="volatility_limit",
            constraint_type=ConstraintType.VOLATILITY_LIMITS,
            validator=self._validate_volatility_limits,
            severity=ConstraintSeverity.ERROR,
            description="Limit trading in high volatility periods",
            parameters={
                'max_volatility_threshold': 0.05,  # 5% daily volatility
                'volatility_lookback_days': 20
            }
        ))
        
        # Liquidity constraint
        self.add_constraint(Constraint(
            name="liquidity_requirement",
            constraint_type=ConstraintType.LIQUIDITY_REQUIREMENTS,
            validator=self._validate_liquidity_requirements,
            severity=ConstraintSeverity.ERROR,
            description="Require minimum liquidity for trades",
            parameters={
                'min_daily_volume_usd': 1000000,  # $1M daily volume
                'min_avg_spread_bps': 50  # Maximum 50 bps spread
            }
        ))
        
        # Correlation constraint
        self.add_constraint(Constraint(
            name="correlation_limit",
            constraint_type=ConstraintType.CORRELATION_LIMITS,
            validator=self._validate_correlation_limits,
            severity=ConstraintSeverity.WARNING,
            description="Limit highly correlated positions",
            parameters={
                'max_correlation': 0.8,
                'correlation_lookback_days': 30
            }
        ))
    
    def _validate_position_size(self, context: Dict[str, Any]) -> bool:
        """Validate position size constraints."""
        try:
            position_size = context.get('position_size', 0)
            portfolio_value = context.get('portfolio_value', 0)
            
            # Check percentage of portfolio
            if portfolio_value > 0:
                position_pct = position_size / portfolio_value
                max_pct = self._constraints['max_position_size'].parameters.get('max_position_pct', 0.1)
                if position_pct > max_pct:
                    return False
            
            # Check absolute position size
            max_absolute = self._constraints['max_position_size'].parameters.get('max_absolute_position', 1000000)
            if position_size > max_absolute:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Position size validation error: {e}")
            return False
    
    def _validate_risk_exposure(self, context: Dict[str, Any]) -> bool:
        """Validate risk exposure constraints."""
        try:
            current_risk = context.get('current_risk_exposure', 0)
            portfolio_value = context.get('portfolio_value', 0)
            trade_risk = context.get('trade_risk', 0)
            
            # Check portfolio risk
            if portfolio_value > 0:
                portfolio_risk_pct = current_risk / portfolio_value
                max_portfolio_risk = self._constraints['max_risk_exposure'].parameters.get('max_portfolio_risk', 0.3)
                if portfolio_risk_pct > max_portfolio_risk:
                    return False
            
            # Check single trade risk
            max_single_risk = self._constraints['max_risk_exposure'].parameters.get('max_single_trade_risk', 0.02)
            if trade_risk > max_single_risk:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Risk exposure validation error: {e}")
            return False
    
    def _validate_trading_frequency(self, context: Dict[str, Any]) -> bool:
        """Validate trading frequency constraints."""
        try:
            agent_id = context.get('agent_id', '')
            current_time = datetime.now()
            
            # Initialize agent tracking if needed
            if agent_id not in self.violation_history:
                self.violation_history[agent_id] = []
            
            # Get recent trades for this agent
            recent_trades = [
                ts for ts in self.violation_history[agent_id]
                if ts > current_time - timedelta(hours=1)
            ]
            
            max_per_hour = self._constraints['trading_frequency_limit'].parameters.get('max_trades_per_hour', 50)
            if len(recent_trades) >= max_per_hour:
                return False
            
            # Check minimum time between trades
            min_time_between = self._constraints['trading_frequency_limit'].parameters.get('min_time_between_trades', 60)
            if recent_trades:
                time_since_last = (current_time - recent_trades[-1]).total_seconds()
                if time_since_last < min_time_between:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trading frequency validation error: {e}")
            return False
    
    def _validate_consecutive_losses(self, context: Dict[str, Any]) -> bool:
        """Validate consecutive losses constraint."""
        try:
            recent_results = context.get('recent_trade_results', [])
            max_consecutive = self._constraints['consecutive_losses_limit'].parameters.get('max_consecutive_losses', 5)
            loss_threshold = self._constraints['consecutive_losses_limit'].parameters.get('loss_threshold_pct', 0.01)
            
            consecutive_losses = 0
            for result in reversed(recent_results[-max_consecutive-1:]):  # Check last N+1 results
                if result < -loss_threshold:  # Loss exceeds threshold
                    consecutive_losses += 1
                else:
                    break
            
            return bool(consecutive_losses < max_consecutive)
            
        except Exception as e:
            self.logger.error(f"Consecutive losses validation error: {e}")
            return False
    
    def _validate_volatility_limits(self, context: Dict[str, Any]) -> bool:
        """Validate volatility constraints."""
        try:
            symbol = context.get('symbol', '')
            current_volatility = context.get('current_volatility', 0)
            
            max_volatility = self._constraints['volatility_limit'].parameters.get('max_volatility_threshold', 0.05)
            
            return bool(current_volatility <= max_volatility)
            
        except Exception as e:
            self.logger.error(f"Volatility validation error: {e}")
            return False
    
    def _validate_liquidity_requirements(self, context: Dict[str, Any]) -> bool:
        """Validate liquidity requirements."""
        try:
            symbol = context.get('symbol', '')
            daily_volume = context.get('daily_volume_usd', 0)
            avg_spread = context.get('avg_spread_bps', 0)
            
            min_volume = self._constraints['liquidity_requirement'].parameters.get('min_daily_volume_usd', 1000000)
            max_spread = self._constraints['liquidity_requirement'].parameters.get('min_avg_spread_bps', 50)
            
            return bool(daily_volume >= min_volume and avg_spread <= max_spread)
            
        except Exception as e:
            self.logger.error(f"Liquidity validation error: {e}")
            return False
    
    def _validate_correlation_limits(self, context: Dict[str, Any]) -> bool:
        """Validate correlation constraints."""
        try:
            symbol = context.get('symbol', '')
            current_positions = context.get('current_positions', {})
            symbol_correlations = context.get('symbol_correlations', {})
            
            max_correlation = self._constraints['correlation_limit'].parameters.get('max_correlation', 0.8)
            
            # Check correlation with existing positions
            for existing_symbol, correlation in symbol_correlations.items():
                if existing_symbol in current_positions and correlation > max_correlation:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Correlation validation error: {e}")
            return False
    
    def _create_violation(self, constraint: Constraint, agent_id: str, 
                          action_context: Dict[str, Any]) -> ConstraintViolation:
        """Create a constraint violation record."""
        return ConstraintViolation(
            timestamp=datetime.now(),
            constraint_name=constraint.name,
            severity=constraint.severity,
            agent_id=agent_id,
            action_context=action_context,
            violation_details=f"Constraint '{constraint.name}' violated: {constraint.description}",
            recommended_action=self._get_recommended_action(constraint)
        )
    
    def _record_violation(self, violation: ConstraintViolation):
        """Record a constraint violation."""
        self._constraint_violations.append(violation)
        
        # Add to agent violation history
        if violation.agent_id not in self.violation_history:
            self.violation_history[violation.agent_id] = []
        self.violation_history[violation.agent_id].append(violation.timestamp)
        
        # Trim violation history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.violation_history[violation.agent_id] = [
            ts for ts in self.violation_history[violation.agent_id] if ts > cutoff_time
        ]
        
        # Trim violations list
        if len(self._constraint_violations) > 10000:
            self._constraint_violations = self._constraint_violations[-10000:]
        
        self.logger.warning(f"⚠️ Constraint violation: {violation.constraint_name} by agent {violation.agent_id}")
    
    def _get_recommended_action(self, constraint: Constraint) -> str:
        """Get recommended action for constraint violation."""
        recommendations = {
            ConstraintType.POSITION_SIZE: "Reduce position size or increase portfolio value",
            ConstraintType.RISK_EXPOSURE: "Reduce risk exposure or close some positions",
            ConstraintType.TRADING_FREQUENCY: "Wait before placing next trade",
            ConstraintType.CONSECUTIVE_LOSSES: "Stop trading and review strategy",
            ConstraintType.VOLATILITY_LIMITS: "Wait for volatility to decrease",
            ConstraintType.LIQUIDITY_REQUIREMENTS: "Choose more liquid instruments",
            ConstraintType.CORRELATION_LIMITS: "Avoid highly correlated positions"
        }
        
        return recommendations.get(constraint.constraint_type, "Review and adjust action")
    
    def _trigger_emergency_stop(self, agent_id: str, violations: List[ConstraintViolation]):
        """Trigger emergency stop for critical violations."""
        try:
            self.logger.critical(f"🚨 EMERGENCY STOP triggered for agent {agent_id}")
            
            # Log all critical violations
            for violation in violations:
                self.logger.critical(f"Critical violation: {violation.constraint_name} - {violation.violation_details}")
            
            # This would integrate with the emergency stop system
            # For now, just log the event
            
        except Exception as e:
            self.logger.error(f"Error triggering emergency stop: {e}")
    
    def _violation_to_dict(self, violation: ConstraintViolation) -> Dict[str, Any]:
        """Convert violation to dictionary format."""
        return {
            'timestamp': violation.timestamp.isoformat(),
            'constraint_name': violation.constraint_name,
            'severity': violation.severity.value,
            'agent_id': violation.agent_id,
            'violation_details': violation.violation_details,
            'recommended_action': violation.recommended_action
        }


# Singleton instance
_security_constraint_engine_instance: Optional[SecurityConstraintEngine] = None


def get_security_constraint_engine(config: Optional[Dict[str, Any]] = None) -> SecurityConstraintEngine:
    """Get or create singleton SecurityConstraintEngine instance."""
    global _security_constraint_engine_instance
    
    if _security_constraint_engine_instance is None:
        _security_constraint_engine_instance = SecurityConstraintEngine(config)
    
    return _security_constraint_engine_instance


def validate_agent_action(agent_id: str, action_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate agent action against constraints.
    
    Args:
        agent_id: Agent identifier
        action_context: Context information about the action
        
    Returns:
        Validation result
    """
    engine = get_security_constraint_engine()
    return engine.validate_action(agent_id, action_context)
