import json
from datetime import datetime
from typing import Any

from src.core.logging.logger import ProjectLogger

from .alerts import AlertManager
from .calculator import KillSwitchCalculator
from .config import KillSwitchConfig
from .executor import KillSwitchExecutor

logger = ProjectLogger.get_logger("KillSwitchManager")

class KillSwitchManager:
    """
    Orchestrator for the Kill Switch system.
    Delegates calculation, analysis, and execution to specialized components.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.logger = logger
        self.config_manager = KillSwitchConfig(config)
        self.calculator = KillSwitchCalculator(self.config_manager)
        self.executor = KillSwitchExecutor(self.config_manager)
        self.alert_manager = AlertManager()

        # State tracking
        self.current_risk_level = 'normal'
        self.kill_switch_active = False
        self.risk_events = []

        self.logger.info("✅ KillSwitchManager (Modular) initialized")

    async def monitor_and_execute(self,
                                portfolio_data: dict[str, Any],
                                market_data: Any) -> dict[str, Any]:
        """Monitor portfolio and execute risk management actions."""
        self.logger.info("🛡️ Starting kill-switch monitoring cycle")

        results = {
            'timestamp': datetime.now(),
            'current_risk_level': self.current_risk_level,
            'risk_analysis': {},
            'actions_taken': []
        }

        try:
            # 1. Calculate risk metrics & Determine risk level
            risk_analysis = self.calculator.calculate_risk_metrics(
                portfolio_data, market_data, self.current_risk_level
            )
            results['risk_analysis'] = risk_analysis
            self.current_risk_level = risk_analysis.get('portfolio_level', 'normal')

            # 2. Check emergency triggers
            emergency_triggers = self.calculator.check_emergency_triggers(risk_analysis)

            # 3. Execute actions
            if emergency_triggers['any_triggered']:
                actions = await self.executor.execute_emergency_actions(
                    emergency_triggers, portfolio_data, market_data
                )
                results['actions_taken'].extend(actions)
                self.kill_switch_active = True
            else:
                actions = await self.executor.execute_normal_risk_management(
                    risk_analysis, portfolio_data, market_data
                )
                results['actions_taken'].extend(actions)

            # 4. Store event if significant
            if results['actions_taken'] or self.current_risk_level != 'normal':
                self._record_risk_event(results)

            return results

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Error in kill-switch cycle: {e}", exc_info=True)
            results['error'] = str(e)
            return results

    def _record_risk_event(self, event: dict[str, Any]):
        """Record risk event for historical analysis."""
        self.risk_events.append(event)
        # Persistent storage logic
        try:
            save_path = self.config_manager.storage_path / f"risk_event_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # Ensure path is serializable
            event_to_save = event.copy()
            event_to_save['timestamp'] = event_to_save['timestamp'].isoformat()

            with open(save_path, 'w') as f:
                json.dump(event_to_save, f, indent=4, default=str)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f"Failed to save risk event: {e}")

    def get_risk_summary(self) -> dict[str, Any]:
        """Get summary of recent risk state and events."""
        return {
            'current_risk_level': self.current_risk_level,
            'kill_switch_active': self.kill_switch_active,
            'event_count': len(self.risk_events)
        }
