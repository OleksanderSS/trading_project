# src/scripts/monitoring/run_health_check.py

import json

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.analytics.data_managers.model_results_manager import ModelResultsManager
from src.data.management.data_manager import DataManager
from src.monitoring.health_hub import HealthHub


def main():
    """
    Initializes and runs a comprehensive system health check using the HealthHub.
    This script provides a snapshot of the system's operational status, including
    resource utilization, ML-based risk predictions, and anomaly detection.
    """
    logger = ProjectLogger.get_logger("HealthCheckScript")
    logger.info("--- Starting System Health Check ---")

    try:
        # Initialize dependencies for HealthHub
        config_manager = UnifiedConfigManager()
        data_manager = DataManager(config_manager)
        results_manager = ModelResultsManager()

        # Initialize the Health Hub
        health_hub = HealthHub(data_manager=data_manager, results_manager=results_manager)

        # Run the comprehensive health check
        logger.info("Running HealthHub.check_system_health()...")
        health_report = health_hub.check_system_health()

        if health_report.get("status") == "failed":
            logger.error(f"Health check failed: {health_report.get('error')}")
            return

        # Pretty-print the JSON report
        report_str = json.dumps(health_report, indent=2, ensure_ascii=False)

        print("\n--- System Health Report ---")
        print(report_str)
        print("--- End of Report ---\n")

        logger.info("Health check completed successfully.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the health check: {e}", exc_info=True)

if __name__ == "__main__":
    main()
