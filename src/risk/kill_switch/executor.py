from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("KillSwitchExecutor")

class KillSwitchExecutor:
    """Executes risk management actions for Kill Switch."""
    
    def __init__(self, config_manager: Any):
        self.logger = logger
        self.config_manager = config_manager

    async def execute_emergency_actions(self,
                                     triggers: Dict[str, Any],
                                     portfolio_data: Dict[str, Any],
                                     market_data: Any) -> List[Dict[str, Any]]:
        """Execute emergency actions based on triggered events."""
        actions_taken = []
        
        for trigger_name in triggers.get('active_triggers', []):
            trigger_info = self.config_manager.EMERGENCY_TRIGGERS.get(trigger_name, {})
            action_type = trigger_info.get('action')
            
            self.logger.warning(f"🚨 Emergency trigger activated: {trigger_name}. Action: {action_type}")
            
            if action_type == 'emergency_closure' or action_type == 'immediate_closure':
                await self.emergency_closure(portfolio_data)
                actions_taken.append({'trigger': trigger_name, 'action': action_type, 'status': 'executed'})
            elif action_type == 'reduce_all_positions':
                await self.reduce_all_positions(portfolio_data, reduction_factor=0.5)
                actions_taken.append({'trigger': trigger_name, 'action': action_type, 'status': 'executed'})
                
        return actions_taken

    async def execute_normal_risk_management(self,
                                           risk_analysis: Dict[str, Any],
                                           portfolio_data: Dict[str, Any],
                                           market_data: Any) -> List[Dict[str, Any]]:
        """Execute normal (non-emergency) risk management actions."""
        actions_taken = []
        risk_level = risk_analysis.get('portfolio_level', 'normal')
        
        if risk_level == 'elevated':
            await self.reduce_all_positions(portfolio_data, reduction_factor=0.2)
            actions_taken.append({'risk_level': risk_level, 'action': 'reduce_positions_20%', 'status': 'executed'})
        elif risk_level == 'high':
            await self.reduce_all_positions(portfolio_data, reduction_factor=0.4)
            actions_taken.append({'risk_level': risk_level, 'action': 'reduce_positions_40%', 'status': 'executed'})
            
        return actions_taken

    async def emergency_closure(self, portfolio_data: Dict[str, Any]) -> None:
        """Close all positions immediately."""
        self.logger.critical("🚨 EMERGENCY CLOSURE INITIATED - CLOSING ALL POSITIONS")
        # Logic to call trading API would go here
        for symbol in portfolio_data:
            self.logger.info(f"Closing position for {symbol}")

    async def reduce_all_positions(self, portfolio_data: Dict[str, Any], reduction_factor: float) -> None:
        """Reduce all positions by a given factor."""
        self.logger.warning(f"📉 REDUCING ALL POSITIONS BY {reduction_factor*100}%")
        for symbol in portfolio_data:
            self.logger.info(f"Reducing position for {symbol} by {reduction_factor*100}%")
