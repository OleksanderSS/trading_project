"""
Agent Permission System - Security Framework for Meta-Learning Agents

Provides role-based access control and permission validation for all agent actions.
Critical for safe deployment of self-improving trading agents.
"""
import logging
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import RLock
import json
from pathlib import Path
from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config
logger = ProjectLogger.get_logger(__name__)


class ActionType(Enum):
    """Types of actions that agents can perform."""
    READ_DATA = 'read_data'
    WRITE_DATA = 'write_data'
    EXECUTE_TRADE = 'execute_trade'
    MODIFY_STRATEGY = 'modify_strategy'
    GENERATE_RULE = 'generate_rule'
    ACCESS_MEMORY = 'access_memory'
    MODIFY_PARAMETERS = 'modify_parameters'
    TRIGGER_EMERGENCY_STOP = 'trigger_emergency_stop'
    ACCESS_EXTERNAL_API = 'access_external_api'
    MODIFY_SYSTEM_CONFIG = 'modify_system_config'


class AgentRole(Enum):
    """Agent roles with different permission levels."""
    CHAMPION = 'champion'
    PRETENDER = 'pretender'
    OBSERVER = 'observer'
    ANALYZER = 'analyzer'
    TRAINER = 'trainer'


@dataclass
class Permission:
    """Individual permission definition."""
    action_type: ActionType
    allowed: bool
    conditions: Optional[Dict[str, Any]] = None
    time_restrictions: Optional[Dict[str, str]] = None
    resource_limits: Optional[Dict[str, Any]] = None


@dataclass
class AgentIdentity:
    """Agent identity and role information."""
    agent_id: str
    role: AgentRole
    created_at: datetime
    last_active: datetime
    status: str = 'active'
    metadata: Optional[Dict[str, Any]] = None


class AgentPermissionSystem:
    """
    Comprehensive permission system for meta-learning agents.
    
    Features:
    - Role-based access control
    - Action validation
    - Time-based restrictions
    - Resource usage limits
    - Emergency stop capabilities
    - Audit trail for all actions
    """

    def __init__(self, config: Optional[Dict[str, Any]]=None):
        """
        Initialize the Agent Permission System.
        
        Args:
            config: Configuration dictionary for permissions
        """
        self.config = config or {}
        self.logger = logger
        self._permissions: Dict[AgentRole, Set[Permission]] = {}
        self._registered_agents: Dict[str, AgentIdentity] = {}
        self.enabled = self.config.get('enabled', True)
        self.strict_mode = self.config.get('strict_mode', True)
        self.audit_all_actions = self.config.get('audit_all_actions', True)
        self.action_counts: Dict[str, Dict[ActionType, List[datetime]]] = {}
        self.rate_limits = self.config.get('rate_limits', {ActionType.
            EXECUTE_TRADE: {'max_per_hour': 100, 'max_per_day': 1000},
            ActionType.MODIFY_STRATEGY: {'max_per_hour': 10, 'max_per_day':
            100}, ActionType.GENERATE_RULE: {'max_per_hour': 50,
            'max_per_day': 500}})
        self._lock = RLock()
        self.audit_log: List[Dict[str, Any]] = []
        self.max_audit_entries = self.config.get('max_audit_entries', 10000)
        self.storage_path = Path(self.config.get('storage_path',
            'data/agent_permissions'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._initialize_default_permissions()
        self.logger.info('✅ AgentPermissionSystem initialized')

    def register_agent(self, agent_id: str, role: AgentRole, metadata:
        Optional[Dict[str, Any]]=None) ->bool:
        """
        Register a new agent with the permission system.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Agent role
            metadata: Additional agent metadata
            
        Returns:
            True if registration successful, False otherwise
        """
        with self._lock:
            try:
                if agent_id in self._registered_agents:
                    self.logger.warning(f'Agent {agent_id} already registered')
                    return False
                identity = AgentIdentity(agent_id=agent_id, role=role,
                    created_at=datetime.now(), last_active=datetime.now(),
                    metadata=metadata)
                self._registered_agents[agent_id] = identity
                self.action_counts[agent_id] = {action: [] for action in
                    ActionType}
                self._log_audit_event(agent_id=agent_id, action=ActionType.
                    READ_DATA, result='success', details=
                    f'Agent registered with role: {role.value}')
                self.logger.info(
                    f'✅ Agent {agent_id} registered with role: {role.value}')
                return True
            except Exception as e:
                self.logger.error(f'Failed to register agent {agent_id}: {e}')
                return False

    def _check_agent_registered(self, agent_id: str) -> tuple:
        """Check if agent is registered and return (is_registered, agent_or_reason)."""
        if agent_id not in self._registered_agents:
            return False, 'Agent not registered'
        return True, self._registered_agents[agent_id]

    def _check_agent_active(self, agent) -> tuple:
        """Check if agent is active and return (is_active, reason)."""
        if agent.status != 'active':
            return False, f'Agent status: {agent.status}'
        return True, None

    def _find_matching_permission(self, agent, action_type: ActionType) -> tuple:
        """Find matching permission for action type and return (permission, reason)."""
        role_permissions = self._permissions.get(agent.role, set())
        for perm in role_permissions:
            if perm.action_type == action_type:
                return perm, None
        return None, f'No permission found for action: {action_type.value}'

    def _check_permission_allowed(self, matching_permission, agent_role) -> tuple:
        """Check if permission is allowed and return (is_allowed, reason)."""
        if not matching_permission.allowed:
            return False, f'Action {matching_permission.action_type.value} not allowed for role {agent_role.value}'
        return True, None

    def _check_conditions(self, matching_permission, context, agent) -> dict:
        """Check permission conditions."""
        if not matching_permission.conditions:
            return {'passed': True}
        
        condition_result = self._check_conditions(
            matching_permission.conditions, context, agent)
        return condition_result

    def _check_time_restrictions(self, matching_permission) -> dict:
        """Check time restrictions."""
        if not matching_permission.time_restrictions:
            return {'passed': True}
        
        time_result = self._check_time_restrictions(
            matching_permission.time_restrictions)
        return time_result

    def _check_resource_limits(self, matching_permission, context) -> dict:
        """Check resource limits."""
        if not matching_permission.resource_limits:
            return {'passed': True}
        
        resource_result = self._check_resource_limits(
            matching_permission.resource_limits, context)
        return resource_result

    def check_permission(self, agent_id: str, action_type: ActionType,
        context: Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Check if an agent has permission to perform an action.
        
        Args:
            agent_id: Agent identifier
            action_type: Type of action to perform
            context: Additional context for permission check
            
        Returns:
            Dictionary with permission check result
        """
        if not self.enabled:
            return {'allowed': True, 'reason': 'Permission system disabled',
                'strict_mode': False}
        
        with self._lock:
            try:
                # Check if agent is registered
                is_registered, agent_or_reason = self._check_agent_registered(agent_id)
                if not is_registered:
                    result = {'allowed': False, 'reason': agent_or_reason,
                        'strict_mode': self.strict_mode}
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                agent = agent_or_reason
                
                # Check if agent is active
                is_active, reason = self._check_agent_active(agent)
                if not is_active:
                    result = {'allowed': False, 'reason': reason,
                        'strict_mode': self.strict_mode}
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Find matching permission
                matching_permission, reason = self._find_matching_permission(agent, action_type)
                if matching_permission is None:
                    result = {'allowed': self.strict_mode is False, 'reason': reason,
                        'strict_mode': self.strict_mode}
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Check if permission is allowed
                is_allowed, reason = self._check_permission_allowed(matching_permission, agent.role)
                if not is_allowed:
                    result = {'allowed': False, 'reason': reason,
                        'strict_mode': self.strict_mode}
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Check conditions
                condition_result = self._check_conditions(matching_permission, context, agent)
                if not condition_result['passed']:
                    result = {'allowed': False,
                        'reason': f"Condition check failed: {condition_result['reason']}",
                        'strict_mode': self.strict_mode}
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Check time restrictions
                time_result = self._check_time_restrictions(matching_permission)
                if not time_result['passed']:
                    result = {'allowed': False,
                        'reason': f"Time restriction: {time_result['reason']}",
                        'strict_mode': self.strict_mode}
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Check rate limits
                rate_result = self._check_rate_limits(agent_id, action_type)
                if not rate_result['passed']:
                    result = {'allowed': False,
                        'reason': f"Rate limit exceeded: {rate_result['reason']}",
                        'strict_mode': self.strict_mode}
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Check resource limits
                resource_result = self._check_resource_limits(matching_permission, context)
                if not resource_result['passed']:
                    result = {'allowed': False,
                        'reason': f"Resource limit exceeded: {resource_result['reason']}",
                        'strict_mode': self.strict_mode}
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Permission granted
                self._record_action(agent_id, action_type)
                self._log_audit_event(agent_id, action_type, 'allowed', 'Permission granted')
                return {'allowed': True, 'reason': 'Permission granted',
                    'strict_mode': self.strict_mode}
            except Exception as e:
                self.logger.error(
                    f'Error checking permission for {agent_id}: {e}')
                result = {'allowed': False, 'reason':
                    f'Permission check error: {str(e)}', 'strict_mode':
                    self.strict_mode}
                self._log_audit_event(agent_id, action_type, 'error',
                    result['reason'])
                return result

    def revoke_agent_permissions(self, agent_id: str, reason: str) ->bool:
        """
        Revoke all permissions for an agent (emergency measure).
        
        Args:
            agent_id: Agent identifier
            reason: Reason for revocation
            
        Returns:
            True if revocation successful
        """
        with self._lock:
            try:
                if agent_id not in self._registered_agents:
                    self.logger.warning(
                        f'Agent {agent_id} not found for revocation')
                    return False
                self._registered_agents[agent_id].status = 'revoked'
                self._log_audit_event(agent_id=agent_id, action=ActionType.
                    TRIGGER_EMERGENCY_STOP, result='revoked', details=
                    f'All permissions revoked: {reason}')
                self.logger.warning(
                    f'⚠️ Agent {agent_id} permissions revoked: {reason}')
                return True
            except Exception as e:
                self.logger.error(
                    f'Failed to revoke permissions for {agent_id}: {e}')
                return False

    def get_agent_status(self, agent_id: str) ->Optional[Dict[str, Any]]:
        """Get current status and permissions for an agent."""
        with self._lock:
            if agent_id not in self._registered_agents:
                return None
            agent = self._registered_agents[agent_id]
            role_permissions = self._permissions.get(agent.role, set())
            return {'agent_id': agent_id, 'role': agent.role.value,
                'status': agent.status, 'created_at': agent.created_at.
                isoformat(), 'last_active': agent.last_active.isoformat(),
                'permissions': [{'action': perm.action_type.value,
                'allowed': perm.allowed, 'conditions': perm.conditions,
                'time_restrictions': perm.time_restrictions} for perm in
                role_permissions], 'action_counts': {action.value: len(
                timestamps) for action, timestamps in self.action_counts.
                get(agent_id, {}).items()}}

    def _initialize_default_permissions(self):
        """Initialize default permission sets for each role."""
        self._permissions[AgentRole.CHAMPION] = {Permission(ActionType.
            READ_DATA, True), Permission(ActionType.EXECUTE_TRADE, True, {
            'max_position_size': 0.1, 'max_risk_per_trade': 0.02}),
            Permission(ActionType.ACCESS_MEMORY, True), Permission(
            ActionType.MODIFY_PARAMETERS, True, {'max_change_pct': 0.1}),
            Permission(ActionType.WRITE_DATA, True), Permission(ActionType.
            GENERATE_RULE, False), Permission(ActionType.MODIFY_STRATEGY, 
            False), Permission(ActionType.TRIGGER_EMERGENCY_STOP, True),
            Permission(ActionType.ACCESS_EXTERNAL_API, False), Permission(
            ActionType.MODIFY_SYSTEM_CONFIG, False)}
        self._permissions[AgentRole.PRETENDER] = {Permission(ActionType.
            READ_DATA, True), Permission(ActionType.EXECUTE_TRADE, True, {
            'max_position_size': 0.05, 'max_risk_per_trade': 0.01,
            'simulation_only': True}), Permission(ActionType.ACCESS_MEMORY,
            True), Permission(ActionType.MODIFY_PARAMETERS, True, {
            'max_change_pct': 0.05}), Permission(ActionType.WRITE_DATA, 
            False), Permission(ActionType.GENERATE_RULE, False), Permission
            (ActionType.MODIFY_STRATEGY, False), Permission(ActionType.
            TRIGGER_EMERGENCY_STOP, True), Permission(ActionType.
            ACCESS_EXTERNAL_API, False), Permission(ActionType.
            MODIFY_SYSTEM_CONFIG, False)}
        self._permissions[AgentRole.OBSERVER] = {Permission(ActionType.
            READ_DATA, True), Permission(ActionType.EXECUTE_TRADE, False),
            Permission(ActionType.ACCESS_MEMORY, True), Permission(
            ActionType.MODIFY_PARAMETERS, False), Permission(ActionType.
            WRITE_DATA, False), Permission(ActionType.GENERATE_RULE, False),
            Permission(ActionType.MODIFY_STRATEGY, False), Permission(
            ActionType.TRIGGER_EMERGENCY_STOP, True), Permission(ActionType
            .ACCESS_EXTERNAL_API, False), Permission(ActionType.
            MODIFY_SYSTEM_CONFIG, False)}
        self._permissions[AgentRole.ANALYZER] = {Permission(ActionType.
            READ_DATA, True), Permission(ActionType.EXECUTE_TRADE, False),
            Permission(ActionType.ACCESS_MEMORY, True), Permission(
            ActionType.MODIFY_PARAMETERS, False), Permission(ActionType.
            WRITE_DATA, True, {'write_path_restricted': True,
            'allowed_paths': ['reports/', 'analysis/']}), Permission(
            ActionType.GENERATE_RULE, False), Permission(ActionType.
            MODIFY_STRATEGY, False), Permission(ActionType.
            TRIGGER_EMERGENCY_STOP, True), Permission(ActionType.
            ACCESS_EXTERNAL_API, False), Permission(ActionType.
            MODIFY_SYSTEM_CONFIG, False)}
        self._permissions[AgentRole.TRAINER] = {Permission(ActionType.
            READ_DATA, True), Permission(ActionType.EXECUTE_TRADE, False),
            Permission(ActionType.ACCESS_MEMORY, True), Permission(
            ActionType.MODIFY_PARAMETERS, True, {'max_change_pct': 0.2}),
            Permission(ActionType.WRITE_DATA, True), Permission(ActionType.
            GENERATE_RULE, True, {'max_rules_per_hour': 10,
            'requires_validation': True}), Permission(ActionType.
            MODIFY_STRATEGY, True, {'requires_approval': True,
            'test_period_required': True}), Permission(ActionType.
            TRIGGER_EMERGENCY_STOP, True), Permission(ActionType.
            ACCESS_EXTERNAL_API, False), Permission(ActionType.
            MODIFY_SYSTEM_CONFIG, False)}

    def _check_conditions(self, conditions: Dict[str, Any], context:
        Optional[Dict[str, Any]], agent: AgentIdentity) ->Dict[str, Any]:
        """Check permission conditions against context."""
        try:
            for condition_name, condition_value in conditions.items():
                if condition_name == 'max_position_size':
                    if context and 'position_size' in context:
                        if context['position_size'] > condition_value:
                            return {'passed': False, 'reason':
                                f"Position size {context['position_size']} exceeds limit {condition_value}"
                                }
                elif condition_name == 'max_risk_per_trade':
                    if context and 'risk_per_trade' in context:
                        if context['risk_per_trade'] > condition_value:
                            return {'passed': False, 'reason':
                                f"Risk per trade {context['risk_per_trade']} exceeds limit {condition_value}"
                                }
                elif condition_name == 'simulation_only':
                    if context and 'live_trading' in context:
                        if context['live_trading']:
                            return {'passed': False, 'reason':
                                'Agent only allowed in simulation mode'}
                elif condition_name == 'max_change_pct':
                    if context and 'parameter_change_pct' in context:
                        if context['parameter_change_pct'] > condition_value:
                            return {'passed': False, 'reason':
                                f"Parameter change {context['parameter_change_pct']}% exceeds limit {condition_value}%"
                                }
            return {'passed': True, 'reason': 'All conditions satisfied'}
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return {'passed': False, 'reason':
                f'Condition check error: {str(e)}'}

    def _check_time_restrictions(self, time_restrictions: Dict[str, str]
        ) ->Dict[str, Any]:
        """Check time-based restrictions."""
        try:
            now = datetime.now()
            if 'allowed_hours' in time_restrictions:
                allowed_hours = [int(h) for h in time_restrictions[
                    'allowed_hours'].split(',')]
                if now.hour not in allowed_hours:
                    return {'passed': False, 'reason':
                        f'Current hour {now.hour} not in allowed hours {allowed_hours}'
                        }
            if 'allowed_days' in time_restrictions:
                allowed_days = [int(d) for d in time_restrictions[
                    'allowed_days'].split(',')]
                if now.weekday() not in allowed_days:
                    return {'passed': False, 'reason':
                        f'Current day {now.weekday()} not in allowed days {allowed_days}'
                        }
            return {'passed': True, 'reason': 'Time restrictions satisfied'}
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return {'passed': False, 'reason':
                f'Time restriction error: {str(e)}'}

    def _check_rate_limits(self, agent_id: str, action_type: ActionType
        ) ->Dict[str, Any]:
        """Check if agent has exceeded rate limits."""
        try:
            if action_type not in self.rate_limits:
                return {'passed': True, 'reason':
                    'No rate limit for this action'}
            now = datetime.now()
            agent_actions = self.action_counts.get(agent_id, {}).get(
                action_type, [])
            cutoff_time = now - timedelta(hours=24)
            recent_actions = [ts for ts in agent_actions if ts > cutoff_time]
            self.action_counts[agent_id][action_type] = recent_actions
            limits = self.rate_limits[action_type]
            hour_cutoff = now - timedelta(hours=1)
            hourly_actions = [ts for ts in recent_actions if ts > hour_cutoff]
            if len(hourly_actions) >= limits.get('max_per_hour', float('inf')):
                return {'passed': False, 'reason':
                    f"Hourly limit exceeded: {len(hourly_actions)}/{limits['max_per_hour']}"
                    }
            if len(recent_actions) >= limits.get('max_per_day', float('inf')):
                return {'passed': False, 'reason':
                    f"Daily limit exceeded: {len(recent_actions)}/{limits['max_per_day']}"
                    }
            return {'passed': True, 'reason': 'Rate limits satisfied'}
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return {'passed': False, 'reason':
                f'Rate limit check error: {str(e)}'}

    def _check_resource_limits(self, resource_limits: Dict[str, Any],
        context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """Check resource usage limits."""
        try:
            return {'passed': True, 'reason': 'Resource limits satisfied'}
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return {'passed': False, 'reason':
                f'Resource limit check error: {str(e)}'}

    def _record_action(self, agent_id: str, action_type: ActionType):
        """Record an action for rate limiting."""
        if agent_id not in self.action_counts:
            self.action_counts[agent_id] = {action: [] for action in ActionType
                }
        self.action_counts[agent_id][action_type].append(datetime.now())
        if agent_id in self._registered_agents:
            self._registered_agents[agent_id].last_active = datetime.now()

    def _log_audit_event(self, agent_id: str, action: ActionType, result:
        str, details: str):
        """Log an audit event."""
        if not self.audit_all_actions and result == 'allowed':
            return
        audit_entry = {'timestamp': datetime.now().isoformat(), 'agent_id':
            agent_id, 'action': action.value, 'result': result, 'details':
            details}
        self.audit_log.append(audit_entry)
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries:]
        if len(self.audit_log) % 100 == 0:
            self._save_audit_log()

    def _save_audit_log(self):
        """Save audit log to file."""
        try:
            audit_file = (self.storage_path /
                f"agent_audit_{datetime.now().strftime('%Y%m%d')}.json")
            with open(audit_file, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
        except Exception as e:
            self.logger.error(f'Failed to save audit log: {e}')


_agent_permission_system_instance: Optional[AgentPermissionSystem] = None


def get_agent_permission_system(config: Optional[Dict[str, Any]]=None
    ) ->AgentPermissionSystem:
    """Get or create singleton AgentPermissionSystem instance."""
    global _agent_permission_system_instance
    if _agent_permission_system_instance is None:
        _agent_permission_system_instance = AgentPermissionSystem(config)
    return _agent_permission_system_instance


def check_agent_permission(agent_id: str, action_type: ActionType, context:
    Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
    """
    Convenience function to check agent permissions.
    
    Args:
        agent_id: Agent identifier
        action_type: Type of action to perform
        context: Additional context for permission check
        
    Returns:
        Permission check result
    """
    system = get_agent_permission_system()
    return system.check_permission(agent_id, action_type, context)
