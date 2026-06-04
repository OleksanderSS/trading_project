from typing import Dict, Any, Optional
from pathlib import Path

class KillSwitchConfig:
    """Configuration and thresholds for the Kill Switch system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Risk levels and triggers
        self.RISK_LEVELS = {
            'normal': {
                'description': 'Normal market conditions',
                'color': 'green',
                'portfolio_var_threshold': 0.15,
                'position_var_threshold': 0.25,
                'max_drawdown_threshold': 0.10,
                'correlation_threshold': 0.7,
                'action': 'monitor'
            },
            'elevated': {
                'description': 'Elevated risk conditions',
                'color': 'yellow',
                'portfolio_var_threshold': 0.20,
                'position_var_threshold': 0.30,
                'max_drawdown_threshold': 0.05,
                'correlation_threshold': 0.5,
                'action': 'reduce_positions'
            },
            'high': {
                'description': 'High risk conditions',
                'color': 'orange',
                'portfolio_var_threshold': 0.25,
                'position_var_threshold': 0.40,
                'max_drawdown_threshold': 0.02,
                'correlation_threshold': 0.3,
                'action': 'reduce_positions_moderate'
            },
            'critical': {
                'description': 'Critical risk conditions',
                'color': 'red',
                'portfolio_var_threshold': 0.30,
                'position_var_threshold': 0.50,
                'max_drawdown_threshold': 0.01,
                'correlation_threshold': 0.1,
                'action': 'emergency_closure'
            },
            'emergency': {
                'description': 'Emergency conditions',
                'color': 'darkred',
                'portfolio_var_threshold': 0.40,
                'position_var_threshold': 0.60,
                'max_drawdown_threshold': 0.005,
                'correlation_threshold': 0.05,
                'action': 'immediate_closure'
            }
        }
        
        self.EMERGENCY_TRIGGERS = {
            'portfolio_var_exceeded': {
                'description': 'Portfolio variance exceeded threshold',
                'action': 'reduce_all_positions',
                'cooldown_minutes': 30
            },
            'max_drawdown_exceeded': {
                'description': 'Maximum drawdown exceeded threshold',
                'action': 'reduce_all_positions',
                'cooldown_minutes': 15
            },
            'position_var_exceeded': {
                'description': 'Position variance exceeded threshold',
                'action': 'reduce_position_risk',
                'cooldown_minutes': 10
            },
            'correlation_spike': {
                'description': 'Correlation spike detected',
                'action': 'reduce_correlated_positions',
                'cooldown_minutes': 5
            },
            'market_volatility_spike': {
                'description': 'Market volatility spike detected',
                'action': 'reduce_all_positions',
                'cooldown_minutes': 20
            },
            'liquidity_crisis': {
                'description': 'Liquidity crisis detected',
                'action': 'emergency_closure',
                'cooldown_minutes': 60
            }
        }
        
        self.risk_limits = self.config.get('risk_limits', {})
        self.custom_emergency_triggers = self.config.get('emergency_triggers', {})
        self.storage_path = Path(self.config.get('storage_path', 'data/risk/kill_switch'))
