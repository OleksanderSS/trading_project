import json
import os
from datetime import datetime
from typing import Any

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.core.logging.logger import ProjectLogger
from src.monitoring.infrastructure.resource_monitor import get_resource_monitor

logger = ProjectLogger.get_logger(__name__)

class ComprehensiveReporter:
    """
    Implements a system-wide reporting engine for monitoring health,
    performance, and model integrity, as outlined in the architectural blueprints.
    """

    def __init__(self, config_manager: UnifiedConfigManager | None = None):
        self.config_manager = config_manager or get_current_config()
        self.config = self.config_manager.get_specific_config('monitoring', 'reporter') or {}

        self.thresholds = self.config.get('thresholds', {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'drift_threshold': 0.15
        })

        self.stage_timings: dict[str, float] = {}
        self.model_accuracies: list[dict[str, Any]] = []
        self.alerts: list[str] = []
        self.resource_monitor = get_resource_monitor()

    def record_stage_time(self, stage_name: str, duration: float):
        """Records the execution time for a specific pipeline stage."""
        self.stage_timings[stage_name] = round(duration, 4)
        logger.info(f"Report: Stage '{stage_name}' took {duration:.2f}s")

    def record_model_accuracy(self, model_name: str, accuracy: float, timestamp: datetime | None = None):
        """Logs model accuracy for drift detection analysis."""
        self.model_accuracies.append({
            'model': model_name,
            'accuracy': accuracy,
            'timestamp': (timestamp or datetime.now()).isoformat()
        })

    def _check_system_status(self) -> dict[str, Any]:
        """Gathers real-time OS resource metrics from ResourceMonitor and triggers alerts."""
        health = self.resource_monitor.get_health_status()

        # Map values from ResourceMonitor format (e.g. '45.2%') back to float for thresholds
        cpu_val = float(health.get('cpu', '0%').replace('%', ''))
        mem_val = float(health.get('memory', '0%').replace('%', ''))

        status = {
            'cpu': {'percent': cpu_val, 'status': 'OK' if cpu_val < self.thresholds['cpu_percent'] else 'HIGH_LOAD'},
            'memory': {
                'percent': mem_val,
                'status': 'OK' if mem_val < self.thresholds['memory_percent'] else 'LOW_MEMORY'
            }
        }

        for key, val in status.items():
            if val['status'] != 'OK':
                self.alerts.append(f"SYSTEM_ALERT: {key.upper()} is at {val['percent']}% ({val['status']})")

        return status

    def _analyze_model_drift(self) -> dict[str, Any]:
        """Calculates drift by comparing recent accuracy vs historical average."""
        if len(self.model_accuracies) < 5:
            return {"status": "INSUFFICIENT_DATA"}

        df = pd.DataFrame(self.model_accuracies)
        results = {}

        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            if len(model_df) < 2: continue

            baseline = model_df['accuracy'].iloc[:-1].mean()
            current = model_df['accuracy'].iloc[-1]
            drift = abs(current - baseline)

            is_drifting = drift > self.thresholds['drift_threshold']
            if is_drifting:
                self.alerts.append(f"DRIFT_ALERT: Model '{model}' accuracy dropped by {drift:.4f}")

            results[model] = {
                'baseline_avg': round(baseline, 4),
                'current_val': round(current, 4),
                'drift_delta': round(drift, 4),
                'is_drifting': is_drifting
            }

        return results

    def generate_report(self, output_path: str = "logs/comprehensive_report.json") -> dict[str, Any]:
        """Compiles all metrics into a single structured JSON report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self._check_system_status(),
            'pipeline_performance': {
                'stage_times_seconds': self.stage_timings,
                'total_time': sum(self.stage_timings.values())
            },
            'model_integrity': self._analyze_model_drift(),
            'alerts': self.alerts
        }

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Comprehensive report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

        self._print_console_summary(report)
        return report

    def _print_console_summary(self, report: dict[str, Any]):
        """Outputs a human-readable summary to the logs."""
        logger.info("\n" + "="*50)
        logger.info(f" SYSTEM HEALTH REPORT - {report['timestamp']}")
        logger.info("="*50)

        sys = report['system_status']
        logger.info(f"CPU:    {sys['cpu']['percent']}%  [{sys['cpu']['status']}]")
        logger.info(f"MEM:    {sys['memory']['percent']}%  [{sys['memory']['status']}]")

        logger.info("\nPIPELINE PERFORMANCE:")
        for stage, t in self.stage_timings.items():
            logger.info(f" - {stage:.<25} {t:>8.2f}s")

        if self.alerts:
            logger.info("\nACTIVE alerts:")
            for alert in self.alerts:
                logger.info(f" [!] {alert}")
        else:
            logger.info("\nNo critical issues detected.")
        logger.info("="*50 + "\n")

if __name__ == "__main__":
    # Mock usage
    cfg = get_current_config()
    reporter = ComprehensiveReporter(cfg)

    reporter.record_stage_time("Collection", 12.5)
    reporter.record_stage_time("Processing", 5.2)
    reporter.record_stage_time("Training", 120.8)

    reporter.record_model_accuracy("CatBoost_V1", 0.85)
    reporter.record_model_accuracy("CatBoost_V1", 0.84)
    reporter.record_model_accuracy("CatBoost_V1", 0.86)
    reporter.record_model_accuracy("CatBoost_V1", 0.85)
    reporter.record_model_accuracy("CatBoost_V1", 0.65) # Simulating drift

    reporter.generate_report()
