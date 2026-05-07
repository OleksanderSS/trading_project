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
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    EXECUTE_TRADE = "execute_trade"
    MODIFY_STRATEGY = "modify_strategy"
    GENERATE_RULE = "generate_rule"
    ACCESS_MEMORY = "access_memory"
    MODIFY_PARAMETERS = "modify_parameters"
    TRIGGER_EMERGENCY_STOP = "trigger_emergency_stop"
    ACCESS_EXTERNAL_API = "access_external_api"
    MODIFY_SYSTEM_CONFIG = "modify_system_config"

class AgentRole(Enum):
    """Agent roles with different permission levels."""
    CHAMPION = "champion"          # Main trading agent
    PRETENDER = "pretender"        # Experimental agent
    OBSERVER = "observer"          # Read-only agent
    ANALYZER = "analyzer"          # Analysis-only agent
    TRAINER = "trainer"           # Meta-learning agent

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
    status: str = "active"
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Agent Permission System.
        
        Args:
            config: Configuration dictionary for permissions
        """
        self.config = config or {}
        self.logger = logger
        
        # Permission storage
        self._permissions: Dict[AgentRole, Set[Permission]] = {}
        self._registered_agents: Dict[str, AgentIdentity] = {}
        
        # Security settings
        self.enabled = self.config.get('enabled', True)
        self.strict_mode = self.config.get('strict_mode', True)
        self.audit_all_actions = self.config.get('audit_all_actions', True)
        
        # Rate limiting
        self.action_counts: Dict[str, Dict[ActionType, List[datetime]]] = {}
        self.rate_limits = self.config.get('rate_limits', {
            ActionType.EXECUTE_TRADE: {"max_per_hour": 100, "max_per_day": 1000},
            ActionType.MODIFY_STRATEGY: {"max_per_hour": 10, "max_per_day": 100},
            ActionType.GENERATE_RULE: {"max_per_hour": 50, "max_per_day": 500}
        })
        
        # Thread safety
        self._lock = RLock()
        
        # Audit trail
        self.audit_log: List[Dict[str, Any]] = []
        self.max_audit_entries = self.config.get('max_audit_entries', 10000)
        
        # Storage
        self.storage_path = Path(self.config.get('storage_path', 'data/agent_permissions'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize default permissions
        self._initialize_default_permissions()
        
        self.logger.info("✅ AgentPermissionSystem initialized")
    
    def register_agent(self, agent_id: str, role: AgentRole, 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
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
                    self.logger.warning(f"Agent {agent_id} already registered")
                    return False
                
                identity = AgentIdentity(
                    agent_id=agent_id,
                    role=role,
                    created_at=datetime.now(),
                    last_active=datetime.now(),
                    metadata=metadata
                )
                
                self._registered_agents[agent_id] = identity
                self.action_counts[agent_id] = {action: [] for action in ActionType}
                
                self._log_audit_event(
                    agent_id=agent_id,
                    action=ActionType.READ_DATA,  # Registration action
                    result="success",
                    details=f"Agent registered with role: {role.value}"
                )
                
                self.logger.info(f"✅ Agent {agent_id} registered with role: {role.value}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to register agent {agent_id}: {e}")
                return False
    
    def check_permission(self, agent_id: str, action_type: ActionType, 
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            return {
                'allowed': True,
                'reason': 'Permission system disabled',
                'strict_mode': False
            }
        
        with self._lock:
            try:
                # Check if agent is registered
                if agent_id not in self._registered_agents:
                    result = {
                        'allowed': False,
                        'reason': 'Agent not registered',
                        'strict_mode': self.strict_mode
                    }
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                agent = self._registered_agents[agent_id]
                
                # Check if agent is active
                if agent.status != 'active':
                    result = {
                        'allowed': False,
                        'reason': f'Agent status: {agent.status}',
                        'strict_mode': self.strict_mode
                    }
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Get role permissions
                role_permissions = self._permissions.get(agent.role, set())
                
                # Find matching permission
                matching_permission = None
                for perm in role_permissions:
                    if perm.action_type == action_type:
                        matching_permission = perm
                        break
                
                if matching_permission is None:
                    result = {
                        'allowed': self.strict_mode is False,
                        'reason': f'No permission found for action: {action_type.value}',
                        'strict_mode': self.strict_mode
                    }
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Check if action is allowed
                if not matching_permission.allowed:
                    result = {
                        'allowed': False,
                        'reason': f'Action {action_type.value} not allowed for role {agent.role.value}',
                        'strict_mode': self.strict_mode
                    }
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Check conditions
                if matching_permission.conditions:
                    condition_result = self._check_conditions(
                        matching_permission.conditions, context, agent
                    )
                    if not condition_result['passed']:
                        result = {
                            'allowed': False,
                            'reason': f'Condition check failed: {condition_result["reason"]}',
                            'strict_mode': self.strict_mode
                        }
                        self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                        return result
                
                # Check time restrictions
                if matching_permission.time_restrictions:
                    time_result = self._check_time_restrictions(
                        matching_permission.time_restrictions
                    )
                    if not time_result['passed']:
                        result = {
                            'allowed': False,
                            'reason': f'Time restriction: {time_result["reason"]}',
                            'strict_mode': self.strict_mode
                        }
                        self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                        return result
                
                # Check rate limits
                rate_result = self._check_rate_limits(agent_id, action_type)
                if not rate_result['passed']:
                    result = {
                        'allowed': False,
                        'reason': f'Rate limit exceeded: {rate_result["reason"]}',
                        'strict_mode': self.strict_mode
                    }
                    self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                    return result
                
                # Check resource limits
                if matching_permission.resource_limits:
                    resource_result = self._check_resource_limits(
                        matching_permission.resource_limits, context
                    )
                    if not resource_result['passed']:
                        result = {
                            'allowed': False,
                            'reason': f'Resource limit exceeded: {resource_result["reason"]}',
                            'strict_mode': self.strict_mode
                        }
                        self._log_audit_event(agent_id, action_type, 'denied', result['reason'])
                        return result
                
                # Permission granted
                self._record_action(agent_id, action_type)
                self._log_audit_event(agent_id, action_type, 'allowed', 'Permission granted')
                
                return {
                    'allowed': True,
                    'reason': 'Permission granted',
                    'strict_mode': self.strict_mode
                }
                
            except Exception as e:
                self.logger.error(f"Error checking permission for {agent_id}: {e}")
                result = {
                    'allowed': False,
                    'reason': f'Permission check error: {str(e)}',
                    'strict_mode': self.strict_mode
                }
                self._log_audit_event(agent_id, action_type, 'error', result['reason'])
                return result
    
    def revoke_agent_permissions(self, agent_id: str, reason: str) -> bool:
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
                    self.logger.warning(f"Agent {agent_id} not found for revocation")
                    return False
                
                self._registered_agents[agent_id].status = 'revoked'
                
                self._log_audit_event(
                    agent_id=agent_id,
                    action=ActionType.TRIGGER_EMERGENCY_STOP,
                    result='revoked',
                    details=f"All permissions revoked: {reason}"
                )
                
                self.logger.warning(f"⚠️ Agent {agent_id} permissions revoked: {reason}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to revoke permissions for {agent_id}: {e}")
                return False
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current status and permissions for an agent."""
        with self._lock:
            if agent_id not in self._registered_agents:
                return None
            
            agent = self._registered_agents[agent_id]
            role_permissions = self._permissions.get(agent.role, set())
            
            return {
                'agent_id': agent_id,
                'role': agent.role.value,
                'status': agent.status,
                'created_at': agent.created_at.isoformat(),
                'last_active': agent.last_active.isoformat(),
                'permissions': [
                    {
                        'action': perm.action_type.value,
                        'allowed': perm.allowed,
                        'conditions': perm.conditions,
                        'time_restrictions': perm.time_restrictions
                    }
                    for perm in role_permissions
                ],
                'action_counts': {
                    action.value: len(timestamps)
                    for action, timestamps in self.action_counts.get(agent_id, {}).items()
                }
            }
    
    def _initialize_default_permissions(self):
        """Initialize default permission sets for each role."""
        
        # Champion agent - full trading permissions
        self._permissions[AgentRole.CHAMPION] = {
            Permission(ActionType.READ_DATA, True),
            Permission(ActionType.EXECUTE_TRADE, True, {
                'max_position_size': 0.1,
                'max_risk_per_trade': 0.02
            }),
            Permission(ActionType.ACCESS_MEMORY, True),
            Permission(ActionType.MODIFY_PARAMETERS, True, {
                'max_change_pct': 0.1
            }),
            Permission(ActionType.WRITE_DATA, True),
            Permission(ActionType.GENERATE_RULE, False),  # Only trainer can generate rules
            Permission(ActionType.MODIFY_STRATEGY, False),  # Only trainer can modify strategy
            Permission(ActionType.TRIGGER_EMERGENCY_STOP, True),
            Permission(ActionType.ACCESS_EXTERNAL_API, False),
            Permission(ActionType.MODIFY_SYSTEM_CONFIG, False)
        }
        
        # Pretender agent - experimental with limited permissions
        self._permissions[AgentRole.PRETENDER] = {
            Permission(ActionType.READ_DATA, True),
            Permission(ActionType.EXECUTE_TRADE, True, {
                'max_position_size': 0.05,  # Smaller positions
                'max_risk_per_trade': 0.01,
                'simulation_only': True  # Paper trading only
            }),
            Permission(ActionType.ACCESS_MEMORY, True),
            Permission(ActionType.MODIFY_PARAMETERS, True, {
                'max_change_pct': 0.05  # More conservative
            }),
            Permission(ActionType.WRITE_DATA, False),  # Cannot write data
            Permission(ActionType.GENERATE_RULE, False),
            Permission(ActionType.MODIFY_STRATEGY, False),
            Permission(ActionType.TRIGGER_EMERGENCY_STOP, True),
            Permission(ActionType.ACCESS_EXTERNAL_API, False),
            Permission(ActionType.MODIFY_SYSTEM_CONFIG, False)
        }
        
        # Observer agent - read-only
        self._permissions[AgentRole.OBSERVER] = {
            Permission(ActionType.READ_DATA, True),
            Permission(ActionType.EXECUTE_TRADE, False),
            Permission(ActionType.ACCESS_MEMORY, True),
            Permission(ActionType.MODIFY_PARAMETERS, False),
            Permission(ActionType.WRITE_DATA, False),
            Permission(ActionType.GENERATE_RULE, False),
            Permission(ActionType.MODIFY_STRATEGY, False),
            Permission(ActionType.TRIGGER_EMERGENCY_STOP, True),
            Permission(ActionType.ACCESS_EXTERNAL_API, False),
            Permission(ActionType.MODIFY_SYSTEM_CONFIG, False)
        }
        
        # Analyzer agent - analysis permissions
        self._permissions[AgentRole.ANALYZER] = {
            Permission(ActionType.READ_DATA, True),
            Permission(ActionType.EXECUTE_TRADE, False),
            Permission(ActionType.ACCESS_MEMORY, True),
            Permission(ActionType.MODIFY_PARAMETERS, False),
            Permission(ActionType.WRITE_DATA, True, {
                'write_path_restricted': True,
                'allowed_paths': ['reports/', 'analysis/']
            }),
            Permission(ActionType.GENERATE_RULE, False),
            Permission(ActionType.MODIFY_STRATEGY, False),
            Permission(ActionType.TRIGGER_EMERGENCY_STOP, True),
            Permission(ActionType.ACCESS_EXTERNAL_API, False),
            Permission(ActionType.MODIFY_SYSTEM_CONFIG, False)
        }
        
        # Trainer agent - meta-learning permissions
        self._permissions[AgentRole.TRAINER] = {
            Permission(ActionType.READ_DATA, True),
            Permission(ActionType.EXECUTE_TRADE, False),  # Cannot trade directly
            Permission(ActionType.ACCESS_MEMORY, True),
            Permission(ActionType.MODIFY_PARAMETERS, True, {
                'max_change_pct': 0.2
            }),
            Permission(ActionType.WRITE_DATA, True),
            Permission(ActionType.GENERATE_RULE, True, {
                'max_rules_per_hour': 10,
                'requires_validation': True
            }),
            Permission(ActionType.MODIFY_STRATEGY, True, {
                'requires_approval': True,
                'test_period_required': True
            }),
            Permission(ActionType.TRIGGER_EMERGENCY_STOP, True),
            Permission(ActionType.ACCESS_EXTERNAL_API, False),
            Permission(ActionType.MODIFY_SYSTEM_CONFIG, False)
        }
    
    def _check_conditions(self, conditions: Dict[str, Any], 
                          context: Optional[Dict[str, Any]], 
                          agent: AgentIdentity) -> Dict[str, Any]:
        """Check permission conditions against context."""
        try:
            for condition_name, condition_value in conditions.items():
                if condition_name == 'max_position_size':
                    if context and 'position_size' in context:
                        if context['position_size'] > condition_value:
                            return {
                                'passed': False,
                                'reason': f'Position size {context["position_size"]} exceeds limit {condition_value}'
                            }
                
                elif condition_name == 'max_risk_per_trade':
                    if context and 'risk_per_trade' in context:
                        if context['risk_per_trade'] > condition_value:
                            return {
                                'passed': False,
                                'reason': f'Risk per trade {context["risk_per_trade"]} exceeds limit {condition_value}'
                            }
                
                elif condition_name == 'simulation_only':
                    if context and 'live_trading' in context:
                        if context['live_trading']:
                            return {
                                'passed': False,
                                'reason': 'Agent only allowed in simulation mode'
                            }
                
                elif condition_name == 'max_change_pct':
                    if context and 'parameter_change_pct' in context:
                        if context['parameter_change_pct'] > condition_value:
                            return {
                                'passed': False,
                                'reason': f'Parameter change {context["parameter_change_pct"]}% exceeds limit {condition_value}%'
                            }
            
            return {'passed': True, 'reason': 'All conditions satisfied'}
            
        except Exception as e:
            return {'passed': False, 'reason': f'Condition check error: {str(e)}'}
    
    def _check_time_restrictions(self, time_restrictions: Dict[str, str]) -> Dict[str, Any]:
        """Check time-based restrictions."""
        try:
            now = datetime.now()
            
            if 'allowed_hours' in time_restrictions:
                allowed_hours = [int(h) for h in time_restrictions['allowed_hours'].split(',')]
                if now.hour not in allowed_hours:
                    return {
                        'passed': False,
                        'reason': f'Current hour {now.hour} not in allowed hours {allowed_hours}'
                    }
            
            if 'allowed_days' in time_restrictions:
                allowed_days = [int(d) for d in time_restrictions['allowed_days'].split(',')]
                if now.weekday() not in allowed_days:
                    return {
                        'passed': False,
                        'reason': f'Current day {now.weekday()} not in allowed days {allowed_days}'
                    }
            
            return {'passed': True, 'reason': 'Time restrictions satisfied'}
            
        except Exception as e:
            return {'passed': False, 'reason': f'Time restriction error: {str(e)}'}
    
    def _check_rate_limits(self, agent_id: str, action_type: ActionType) -> Dict[str, Any]:
        """Check if agent has exceeded rate limits."""
        try:
            if action_type not in self.rate_limits:
                return {'passed': True, 'reason': 'No rate limit for this action'}
            
            now = datetime.now()
            agent_actions = self.action_counts.get(agent_id, {}).get(action_type, [])
            
            # Clean old actions (older than 24 hours)
            cutoff_time = now - timedelta(hours=24)
            recent_actions = [ts for ts in agent_actions if ts > cutoff_time]
            self.action_counts[agent_id][action_type] = recent_actions
            
            limits = self.rate_limits[action_type]
            
            # Check hourly limit
            hour_cutoff = now - timedelta(hours=1)
            hourly_actions = [ts for ts in recent_actions if ts > hour_cutoff]
            if len(hourly_actions) >= limits.get('max_per_hour', float('inf')):
                return {
                    'passed': False,
                    'reason': f'Hourly limit exceeded: {len(hourly_actions)}/{limits["max_per_hour"]}'
                }
            
            # Check daily limit
            if len(recent_actions) >= limits.get('max_per_day', float('inf')):
                return {
                    'passed': False,
                    'reason': f'Daily limit exceeded: {len(recent_actions)}/{limits["max_per_day"]}'
                }
            
            return {'passed': True, 'reason': 'Rate limits satisfied'}
            
        except Exception as e:
            return {'passed': False, 'reason': f'Rate limit check error: {str(e)}'}
    
    def _check_resource_limits(self, resource_limits: Dict[str, Any], 
                              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check resource usage limits."""
        try:
            # This would integrate with system monitoring
            # For now, return passed as placeholder
            return {'passed': True, 'reason': 'Resource limits satisfied'}
            
        except Exception as e:
            return {'passed': False, 'reason': f'Resource limit check error: {str(e)}'}
    
    def _record_action(self, agent_id: str, action_type: ActionType):
        """Record an action for rate limiting."""
        if agent_id not in self.action_counts:
            self.action_counts[agent_id] = {action: [] for action in ActionType}
        
        self.action_counts[agent_id][action_type].append(datetime.now())
        
        # Update agent last active
        if agent_id in self._registered_agents:
            self._registered_agents[agent_id].last_active = datetime.now()
    
    def _log_audit_event(self, agent_id: str, action: ActionType, 
                        result: str, details: str):
        """Log an audit event."""
        if not self.audit_all_actions and result == 'allowed':
            return
        
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': agent_id,
            'action': action.value,
            'result': result,
            'details': details
        }
        
        self.audit_log.append(audit_entry)
        
        # Trim audit log if necessary
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries:]
        
        # Log to file periodically
        if len(self.audit_log) % 100 == 0:
            self._save_audit_log()
    
    def _save_audit_log(self):
        """Save audit log to file."""
        try:
            audit_file = self.storage_path / f"agent_audit_{datetime.now().strftime('%Y%m%d')}.json"
            with open(audit_file, 'w') as f:
                json.dump(self.audit_log, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save audit log: {e}")


# Singleton instance
_agent_permission_system_instance: Optional[AgentPermissionSystem] = None


def get_agent_permission_system(config: Optional[Dict[str, Any]] = None) -> AgentPermissionSystem:
    """Get or create singleton AgentPermissionSystem instance."""
    global _agent_permission_system_instance
    
    if _agent_permission_system_instance is None:
        _agent_permission_system_instance = AgentPermissionSystem(config)
    
    return _agent_permission_system_instance


def check_agent_permission(agent_id: str, action_type: ActionType, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
