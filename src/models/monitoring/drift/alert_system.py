#!/usr/bin/env python3
"""
Alert System - Alert generation and retraining recommendations
Handles alert logic and retraining trigger recommendations.
"""

from datetime import datetime
from typing import Any

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("AlertSystem")


class AlertSystem:
    """
    Alert system for drift monitoring.

    Handles:
    - Alert generation based on drift analysis
    - Retraining recommendations
    - Cooldown management
    - Alert severity classification
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Alert System.

        Args:
            config: Configuration dictionary for alert system
        """
        self.logger = logger
        self.config = config or {}

        # Retraining triggers configuration
        self.retraining_triggers = {
            'critical_drift': {
                'description': 'Critical drift detected',
                'action': 'immediate_retraining',
                'cooldown_hours': 1
            },
            'high_drift': {
                'description': 'High drift detected',
                'action': 'scheduled_retraining',
                'cooldown_hours': 4
            },
            'medium_drift': {
                'description': 'Medium drift detected',
                'action': 'monitor_and_alert',
                'cooldown_hours': 12
            },
            'low_drift': {
                'description': 'Low drift detected',
                'action': 'log_only',
                'cooldown_hours': 24
            }
        }

        # Update with custom config
        self.retraining_triggers.update(self.config.get('retraining_triggers', {}))

        # State tracking
        self.last_retraining_time = None
        self.drift_cooldowns = {}

        self.logger.info("✅ AlertSystem initialized")

    def _check_drift_analysis(self, drift_analysis: dict[str, Any], recommendations: list) -> None:
        """Check drift analysis and add recommendations based on severity."""
        if not drift_analysis.get('drift_detected', False):
            return

        drift_severity = drift_analysis.get('drift_severity', 'low')
        drift_score = drift_analysis.get('overall_drift_score', 0)

        if drift_severity == 'critical':
            recommendations.append(
                f"🚨 CRITICAL: Critical prediction drift detected (score: {drift_score:.3f}). "
                "Immediate retraining required."
            )
            recommendations.append("   → Action: Stop current model and retrain immediately.")
        elif drift_severity == 'high':
            recommendations.append(
                f"⚠️ HIGH: High prediction drift detected (score: {drift_score:.3f}). "
                "Retraining recommended."
            )
            recommendations.append("   → Action: Schedule retraining within next 4 hours.")
        elif drift_severity == 'medium':
            recommendations.append(
                f"⚠️ MEDIUM: Medium prediction drift detected (score: {drift_score:.3f}). "
                "Monitor closely."
            )
            recommendations.append("   → Action: Increase monitoring frequency, prepare for retraining.")
        else:
            recommendations.append(
                f"📊 LOW: Low prediction drift detected (score: {drift_score:.3f}). "
                "Continue monitoring."
            )

    def _check_performance_degradation(self, performance_analysis: dict[str, Any], recommendations: list) -> None:
        """Check performance degradation and add recommendations."""
        if not performance_analysis.get('degradation_detected', False):
            return

        degradation_score = performance_analysis.get('degradation_score', 0.0)
        recommendations.append(
            f"⚠️ PERFORMANCE: Performance degradation detected (score: {degradation_score:.3f}). "
            "Retraining recommended."
        )

        # Add specific metric information
        metrics_trends = performance_analysis.get('performance_trend', {}).get('metrics_trends', {})
        for metric_name, trend_info in metrics_trends.items():
            if trend_info.get('trend') == 'degrading':
                recommendations.append(
                    f"   • {metric_name.upper()} is degrading (slope: {trend_info.get('slope', 0):.6f})"
                )

    def _check_confidence_drift(self, confidence_analysis: dict[str, Any], recommendations: list) -> None:
        """Check confidence drift and add recommendations."""
        if not confidence_analysis.get('confidence_drift_detected', False):
            return

        drift_score = confidence_analysis.get('drift_score', 0.0)
        recommendations.append(
            f"⚠️ CONFIDENCE: Confidence distribution drift detected (score: {drift_score:.3f}). "
            "Model calibration may be affected."
        )
        recommendations.append("   → Action: Consider recalibration or retraining.")

    def _check_cooldowns(self, timestamp: datetime, recommendations: list) -> None:
        """Check retraining cooldown and add recommendation if needed."""
        if not self.last_retraining_time:
            return

        hours_since_retraining = (timestamp - self.last_retraining_time).total_seconds() / 3600
        if hours_since_retraining < 24:
            recommendations.append(
                f"⏰ COOLDOWN: Last retraining was {hours_since_retraining:.1f} hours ago. "
                "Consider waiting before next retraining."
            )

    def generate_retraining_recommendations(self,
                                         drift_analysis: dict[str, Any],
                                         performance_analysis: dict[str, Any],
                                         confidence_analysis: dict[str, Any],
                                         timestamp: datetime | None = None) -> list[str]:
        """
        Generate retraining recommendations based on analysis.

        Args:
            drift_analysis: Drift analysis results
            performance_analysis: Performance analysis results
            confidence_analysis: Confidence analysis results
            timestamp: Current timestamp

        Returns:
            List of recommendation strings
        """
        if timestamp is None:
            timestamp = datetime.now()

        recommendations = []

        try:
            # Check drift analysis
            self._check_drift_analysis(drift_analysis, recommendations)

            # Check performance degradation
            self._check_performance_degradation(performance_analysis, recommendations)

            # Check confidence drift
            self._check_confidence_drift(confidence_analysis, recommendations)

            # Check cooldowns
            self._check_cooldowns(timestamp, recommendations)

            # No issues detected
            if not recommendations:
                recommendations.append(
                    "✅ STABLE: No significant drift or degradation detected. "
                    "Model performance is stable."
                )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating retraining recommendations: {e}")
            return [f"❌ Error generating recommendations: {str(e)}"]

    def classify_alert_severity(self, drift_analysis: dict[str, Any]) -> str:
        """
        Classify alert severity based on drift analysis.

        Args:
            drift_analysis: Drift analysis results

        Returns:
            Severity level: 'critical', 'high', 'medium', 'low', 'none'
        """
        if drift_analysis.get('drift_detected', False):
            return drift_analysis.get('drift_severity', 'low')
        return 'none'

    def should_trigger_retraining(self,
                                  drift_analysis: dict[str, Any],
                                  performance_analysis: dict[str, Any],
                                  timestamp: datetime | None = None) -> bool:
        """
        Determine if retraining should be triggered.

        Args:
            drift_analysis: Drift analysis results
            performance_analysis: Performance analysis results
            timestamp: Current timestamp

        Returns:
            True if retraining should be triggered, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        try:
            # Check cooldown
            if self.last_retraining_time:
                hours_since_retraining = (timestamp - self.last_retraining_time).total_seconds() / 3600
                if hours_since_retraining < 1:  # Minimum 1 hour cooldown
                    return False

            # Check for critical drift
            if drift_analysis.get('drift_severity') == 'critical':
                return True

            # Check for high drift
            if drift_analysis.get('drift_severity') == 'high':
                return True

            # Check for performance degradation
            if performance_analysis.get('degradation_detected', False):
                degradation_score = performance_analysis.get('degradation_score', 0.0)
                if degradation_score > 0.1:  # 10% degradation threshold
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking retraining trigger: {e}")
            return False

    def get_retraining_action(self, severity: str) -> str:
        """
        Get recommended retraining action based on severity.

        Args:
            severity: Drift severity level

        Returns:
            Action string: 'immediate_retraining', 'scheduled_retraining', 'monitor_and_alert', 'log_only'
        """
        trigger_key = f'{severity}_drift'
        return self.retraining_triggers.get(trigger_key, {}).get('action', 'log_only')

    def get_cooldown_hours(self, severity: str) -> int:
        """
        Get cooldown hours for retraining based on severity.

        Args:
            severity: Drift severity level

        Returns:
            Cooldown hours
        """
        trigger_key = f'{severity}_drift'
        return self.retraining_triggers.get(trigger_key, {}).get('cooldown_hours', 24)

    def record_retraining(self,
                        reason: str,
                        severity: str,
                        timestamp: datetime | None = None) -> dict[str, Any]:
        """
        Record retraining event.

        Args:
            reason: Reason for retraining
            severity: Drift severity level
            timestamp: Timestamp of retraining

        Returns:
            Retraining record dictionary
        """
        if timestamp is None:
            timestamp = datetime.now()

        retraining_record = {
            'timestamp': timestamp,
            'reason': reason,
            'severity': severity,
            'status': 'triggered',
            'cooldown_hours': self.get_cooldown_hours(severity)
        }

        self.last_retraining_time = timestamp

        self.logger.info(f"🔄 Retraining triggered: {reason} (severity: {severity})")

        return retraining_record

    def generate_alert_message(self,
                             drift_analysis: dict[str, Any],
                             performance_analysis: dict[str, Any],
                             confidence_analysis: dict[str, Any]) -> str:
        """
        Generate comprehensive alert message.

        Args:
            drift_analysis: Drift analysis results
            performance_analysis: Performance analysis results
            confidence_analysis: Confidence analysis results

        Returns:
            Formatted alert message string
        """
        try:
            severity = self.classify_alert_severity(drift_analysis)

            message_parts = [
                f"🔔 DRIFT ALERT [{severity.upper()}]",
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ]

            # Drift information
            if drift_analysis.get('drift_detected', False):
                message_parts.append("📊 DRIFT ANALYSIS:")
                message_parts.append(f"  - Overall Score: {drift_analysis.get('overall_drift_score', 0):.3f}")
                message_parts.append(f"  - Severity: {drift_analysis.get('drift_severity', 'none')}")

                # Method details
                drift_methods = drift_analysis.get('drift_methods', {})
                for method_name, method_result in drift_methods.items():
                    if method_result.get('drift_detected', False):
                        message_parts.append(f"  - {method_name}: {method_result.get('drift_score', 0):.3f}")
                message_parts.append("")

            # Performance information
            if performance_analysis.get('degradation_detected', False):
                message_parts.append("📈 PERFORMANCE ANALYSIS:")
                message_parts.append(f"  - Degradation Score: {performance_analysis.get('degradation_score', 0):.3f}")

                metrics_trends = performance_analysis.get('performance_trend', {}).get('metrics_trends', {})
                for metric_name, trend_info in metrics_trends.items():
                    if trend_info.get('trend') == 'degrading':
                        message_parts.append(f"  - {metric_name.upper()}: {trend_info.get('slope', 0):.6f}")
                message_parts.append("")

            # Confidence information
            if confidence_analysis.get('confidence_drift_detected', False):
                message_parts.append("🎯 CONFIDENCE ANALYSIS:")
                message_parts.append(f"  - Drift Score: {confidence_analysis.get('drift_score', 0):.3f}")
                message_parts.append("")

            # Recommendations
            recommendations = self.generate_retraining_recommendations(
                drift_analysis, performance_analysis, confidence_analysis
            )
            message_parts.append("💡 RECOMMENDATIONS:")
            for rec in recommendations:
                message_parts.append(f"  {rec}")

            return "\n".join(message_parts)

        except Exception as e:
            self.logger.error(f"Error generating alert message: {e}")
            return f"❌ Error generating alert message: {str(e)}"

    def check_cooldown_status(self, severity: str, timestamp: datetime | None = None) -> bool:
        """
        Check if cooldown period has passed for given severity.

        Args:
            severity: Drift severity level
            timestamp: Current timestamp

        Returns:
            True if cooldown has passed, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        if not self.last_retraining_time:
            return True

        cooldown_hours = self.get_cooldown_hours(severity)
        hours_since_retraining = (timestamp - self.last_retraining_time).total_seconds() / 3600

        return hours_since_retraining >= cooldown_hours


# Factory function
def get_alert_system(config: dict[str, Any] | None = None) -> AlertSystem:
    """Factory function to get AlertSystem instance."""
    return AlertSystem(config)
