"""
Security Constraint Engine - Real-time Safety Validation for Meta-Learning Agents

Provides comprehensive constraint validation and enforcement for all agent actions.
Critical for preventing unsafe agent behavior in production environments.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import RLock
from typing import Any

from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
from src.core.logging.logger import ProjectLogger
from src.meta_learning.security.constraint_validators import ConstraintValidators

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
    validator: Callable[[dict[str, Any]], bool]
    severity: ConstraintSeverity
    description: str
    enabled: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)

@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""
    timestamp: datetime
    constraint_name: str
    severity: ConstraintSeverity
    agent_id: str
    action_context: dict[str, Any]
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

    def __init__(self, config: dict[str, Any] | None = None, error_handler: IErrorHandler | None = None):
        """
        Initialize the Security Constraint Engine.

        Args:
            config: Configuration dictionary for constraints
            error_handler: Error handler instance
        """
        self.config = config or {}
        self.logger = logger
        self.error_handler = error_handler or ErrorHandler()

        # Initialize constraint validators
        self.validators = ConstraintValidators(self.logger, self.error_handler)

        # Constraint storage
        self._constraints: dict[str, Constraint] = {}
        self._constraint_violations: list[ConstraintViolation] = []

        # Engine settings
        self.enabled = self.config.get('enabled', True)
        self.strict_mode = self.config.get('strict_mode', True)
        self.emergency_stop_enabled = self.config.get('emergency_stop_enabled', True)

        # Violation tracking
        self.violation_history: dict[str, list[datetime]] = {}
        self.max_violations_per_hour = self.config.get('max_violations_per_hour', 10)
        self.critical_violation_threshold = self.config.get('critical_violation_threshold', 3)

        # Market data cache
        self.market_data_cache: dict[str, Any] = {}
        self.market_data_ttl = self.config.get('market_data_ttl', 300)  # 5 minutes

        # Thread safety
        self._lock = RLock()

        # Initialize default constraints
        self._initialize_default_constraints()

        self.logger.info("✅ SecurityConstraintEngine initialized")

    def _check_constraint(self, constraint_name: str, constraint, agent_id: str, action_context: dict[str, Any]) -> tuple:
        """Check a single constraint and return (violations, warnings)."""
        violations = []
        warnings = []

        try:
            is_valid = constraint.validator(action_context)

            if not is_valid:
                violation = self._create_violation(constraint, agent_id, action_context)

                if constraint.severity == ConstraintSeverity.WARNING:
                    warnings.append(violation)
                else:
                    violations.append(violation)
                    self._record_violation(violation)

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error validating constraint {constraint_name}: {e}")
            self.error_handler.handle_error(
                e,
                context={
                    "agent_id": agent_id,
                    "constraint_name": constraint_name,
                    "action_context": action_context,
                },
            )
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

        return violations, warnings

    def _determine_action_allowed(self, violations: list, warnings: list) -> tuple:
        """Determine if action is allowed based on violations and warnings."""
        critical_violations = [v for v in violations if v.severity == ConstraintSeverity.CRITICAL]
        error_violations = [v for v in violations if v.severity == ConstraintSeverity.ERROR]

        if critical_violations:
            allowed = False
            reason = f"Critical constraint violations: {[v.constraint_name for v in critical_violations]}"
        elif error_violations and self.strict_mode:
            allowed = False
            reason = f"Error constraint violations in strict mode: {[v.constraint_name for v in error_violations]}"
        elif error_violations and not self.strict_mode:
            allowed = True
            reason = f"Error violations present but strict mode disabled: {[v.constraint_name for v in error_violations]}"
        else:
            allowed = True
            reason = "All constraints satisfied" if not warnings else "Constraints satisfied with warnings"

        return allowed, reason, critical_violations

    def _build_validation_result(self, allowed: bool, reason: str, violations: list, warnings: list) -> dict[str, Any]:
        """Build the validation result dictionary."""
        return {
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

    def validate_action(self, agent_id: str, action_context: dict[str, Any]) -> dict[str, Any]:
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

                    constraint_violations, constraint_warnings = self._check_constraint(
                        constraint_name, constraint, agent_id, action_context
                    )
                    violations.extend(constraint_violations)
                    warnings.extend(constraint_warnings)

                # Determine if action is allowed
                allowed, reason, critical_violations = self._determine_action_allowed(violations, warnings)

                # Trigger emergency stop if enabled and critical violations exist
                if critical_violations and self.emergency_stop_enabled:
                    self._trigger_emergency_stop(agent_id, critical_violations)

                # Build result
                result = self._build_validation_result(allowed, reason, violations, warnings)

                self.logger.info(f"Constraint validation for {agent_id}: {'ALLOWED' if allowed else 'DENIED'} - {reason}")

                return result

            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
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
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
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
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(f"Failed to remove constraint {constraint_name}: {e}")
                return False

    def get_constraint_status(self) -> dict[str, Any]:
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

    def get_violation_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get violation history for specified time period."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_violations = [
                self._violation_to_dict(v) for v in self._constraint_violations
                if v.timestamp > cutoff_time
            ]

            return recent_violations

    def _add_position_size_constraint(self) -> None:
        """Add position size constraint."""
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

    def _add_risk_exposure_constraint(self) -> None:
        """Add risk exposure constraint."""
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

    def _add_trading_frequency_constraint(self) -> None:
        """Add trading frequency constraint."""
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

    def _add_consecutive_losses_constraint(self) -> None:
        """Add consecutive losses constraint."""
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

    def _add_volatility_constraint(self) -> None:
        """Add volatility constraint."""
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

    def _add_liquidity_constraint(self) -> None:
        """Add liquidity constraint."""
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

    def _add_correlation_constraint(self) -> None:
        """Add correlation constraint."""
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

    def _initialize_default_constraints(self):
        """Initialize default security constraints."""
        self._add_position_size_constraint()
        self._add_risk_exposure_constraint()
        self._add_trading_frequency_constraint()
        self._add_consecutive_losses_constraint()
        self._add_volatility_constraint()
        self._add_liquidity_constraint()
        self._add_correlation_constraint()

    def _validate_position_size(self, context: dict[str, Any]) -> bool:
        """Validate position size constraints."""
        constraints_dict = {name: constraint.parameters for name, constraint in self._constraints.items()}
        return self.validators.validate_position_size(context, constraints_dict)

    def _validate_risk_exposure(self, context: dict[str, Any]) -> bool:
        """Validate risk exposure constraints."""
        constraints_dict = {name: constraint.parameters for name, constraint in self._constraints.items()}
        return self.validators.validate_risk_exposure(context, constraints_dict)

    def _validate_trading_frequency(self, context: dict[str, Any]) -> bool:
        """Validate trading frequency constraints."""
        constraints_dict = {name: constraint.parameters for name, constraint in self._constraints.items()}
        return self.validators.validate_trading_frequency(context, constraints_dict, self.violation_history)

    def _validate_consecutive_losses(self, context: dict[str, Any]) -> bool:
        """Validate consecutive losses constraint."""
        constraints_dict = {name: constraint.parameters for name, constraint in self._constraints.items()}
        return self.validators.validate_consecutive_losses(context, constraints_dict)

    def _validate_volatility_limits(self, context: dict[str, Any]) -> bool:
        """Validate volatility constraints."""
        constraints_dict = {name: constraint.parameters for name, constraint in self._constraints.items()}
        return self.validators.validate_volatility_limits(context, constraints_dict)

    def _validate_liquidity_requirements(self, context: dict[str, Any]) -> bool:
        """Validate liquidity requirements."""
        constraints_dict = {name: constraint.parameters for name, constraint in self._constraints.items()}
        return self.validators.validate_liquidity_requirements(context, constraints_dict)

    def _validate_correlation_limits(self, context: dict[str, Any]) -> bool:
        """Validate correlation constraints."""
        constraints_dict = {name: constraint.parameters for name, constraint in self._constraints.items()}
        return self.validators.validate_correlation_limits(context, constraints_dict)

    def _create_violation(self, constraint: Constraint, agent_id: str,
                          action_context: dict[str, Any]) -> ConstraintViolation:
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

    def _trigger_emergency_stop(self, agent_id: str, violations: list[ConstraintViolation]):
        """Trigger emergency stop for critical violations."""
        try:
            self.logger.critical(f"🚨 EMERGENCY STOP triggered for agent {agent_id}")

            # Log all critical violations
            for violation in violations:
                self.logger.critical(f"Critical violation: {violation.constraint_name} - {violation.violation_details}")

            # This would integrate with the emergency stop system
            # For now, just log the event

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error triggering emergency stop: {e}")

    def _violation_to_dict(self, violation: ConstraintViolation) -> dict[str, Any]:
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
_security_constraint_engine_instance: SecurityConstraintEngine | None = None


def get_security_constraint_engine(config: dict[str, Any] | None = None) -> SecurityConstraintEngine:
    """Get or create singleton SecurityConstraintEngine instance."""
    global _security_constraint_engine_instance

    if _security_constraint_engine_instance is None:
        _security_constraint_engine_instance = SecurityConstraintEngine(config)

    return _security_constraint_engine_instance


def validate_agent_action(agent_id: str, action_context: dict[str, Any]) -> dict[str, Any]:
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
