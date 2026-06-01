#!/usr/bin/env python3
"""
Kill Switch Manager - Facade for the modular Kill-Switch System.
This module maintains backward compatibility while delegating to the new modular structure.
"""

from typing import Dict, Any, Optional
import pandas as pd
from .kill_switch.manager import KillSwitchManager as ModularKillSwitchManager
from .kill_switch.alerts import AlertManager as ModularAlertManager

# Re-exporting the main class under the same name for backward compatibility
class KillSwitchManager(ModularKillSwitchManager):
    """
    Facade for KillSwitchManager.
    Maintains the original API but uses the modular components internally.
    """
    pass

# Alert Manager for backward compatibility
class AlertManager(ModularAlertManager):
    """Facade for AlertManager."""
    pass

# Factory function for easy instantiation
def get_kill_switch_manager(config: Optional[Dict[str, Any]] = None) -> KillSwitchManager:
    """Factory function to get KillSwitchManager instance."""
    return KillSwitchManager(config)

# Convenience function for quick emergency monitoring
async def monitor_risk_emergency_quick(portfolio_data: Dict[str, Any],
                                 market_data: pd.DataFrame,
                                 config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Quick emergency risk monitoring.
    
    Args:
        portfolio_data: Current portfolio positions
        market_data: Current market data
        config: Configuration dictionary
        
    Returns:
        Risk monitoring result dictionary
    """
    manager = get_kill_switch_manager(config)
    return await manager.monitor_and_execute(portfolio_data, market_data)
