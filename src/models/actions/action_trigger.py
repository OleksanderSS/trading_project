#!/usr/bin/env python3
"""
Action Trigger - Model Action Management
Handles action triggering and alert sending.
"""

from typing import Dict, Any, List
import asyncio

from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError

logger = ProjectLogger.get_logger("ActionTrigger")


class ActionTrigger:
    """
    Action trigger for model management.
    
    Handles:
    - Retraining trigger
    - Critical alert sending
    - Action execution
    """
    
    def __init__(self, drift_monitor):
        """
        Initialize Action Trigger.
        
        Args:
            drift_monitor: PredictionDriftMonitor instance for retraining
        """
        self.logger = logger
        self.drift_monitor = drift_monitor
        self.retraining_history: List[Dict[str, Any]] = []
        
        self.logger.info("✅ ActionTrigger initialized")
    
    async def trigger_actions(self, results: Dict[str, Any]) -> None:
        """Trigger required actions based on analysis results."""
        
        try:
            model_name = results['model_name']
            
            # Trigger retraining if needed
            if results['retraining_recommended']:
                retraining_reason = f"Comprehensive analysis indicated retraining needed for {model_name}"
                severity = 'high' if results['overall_health_score'] < 0.4 else 'medium'
                
                retraining_record = self.drift_monitor.trigger_retraining(
                    reason=retraining_reason,
                    severity=severity
                )
                
                self.retraining_history.append(retraining_record)
                self.logger.info(f"🔄 Retraining triggered for {model_name}: {retraining_reason}")
            
            # Send alerts for critical issues
            if results['action_required']:
                await self._send_critical_alert(results)
            
        except Exception as e:
            self.logger.error(f"Error triggering actions: {e}", exc_info=True)
            raise DataProcessingError(f"Action triggering failed: {e}") from e
    
    async def _send_critical_alert(self, results: Dict[str, Any]) -> None:
        """Send critical alert for model issues."""
        
        try:
            model_name = results['model_name']
            health_score = results['overall_health_score']
            
            alert_message = (
                f"🚨 CRITICAL MODEL ALERT 🚨\n"
                f"Model: {model_name}\n"
                f"Health Score: {health_score:.3f}\n"
                f"Time: {results['timestamp']}\n"
                f"Recommendations:\n"
            )
            
            for i, recommendation in enumerate(results['recommendations'][:5], 1):
                alert_message += f"{i}. {recommendation}\n"
            
            self.logger.critical(alert_message)
            
            # Here you could integrate with alert systems like:
            # - Email notifications
            # - Slack notifications
            # - PagerDuty alerts
            # - SMS alerts
            
        except Exception as e:
            self.logger.error(f"Error sending critical alert: {e}", exc_info=True)
            raise DataProcessingError(f"Critical alert sending failed: {e}") from e
    
    def get_retraining_history(self) -> List[Dict[str, Any]]:
        """Get retraining history."""
        return self.retraining_history
