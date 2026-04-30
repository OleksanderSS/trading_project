from src.config.unified_config_manager import get_current_config
from src.monitoring.health_hub import HealthHub
from src.data.management.data_manager import DataManager
from src.core.reporting.results_manager import ResultsManager
from src.core.logging.logger import ProjectLogger

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
        config_manager = get_current_config()
        data_manager = DataManager(config_manager)
        results_manager = ResultsManager(data_manager)
        
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
        
        logger.info("\n--- System Health Report ---")
        logger.info(report_str)
        logger.info("--- End of Report ---\n")

        logger.info("Health check completed successfully.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the health check: {e}", exc_info=True)

if __name__ == "__main__":
    main()
