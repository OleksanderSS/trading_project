import os

from src.core.logging.logger import ProjectLogger
from src.monitoring.dashboard import MonitoringDashboardGenerator
from src.monitoring.monitoring_system import MonitoringSystem


def main() -> None:
    logger = ProjectLogger.get_logger("MonitoringDashboardRunner")

    host = os.getenv("MONITOR_HOST", "127.0.0.1")
    port = int(os.getenv("MONITOR_PORT", "8050"))
    update_interval = int(os.getenv("MONITOR_UPDATE_MS", "5000"))
    debug = os.getenv("MONITOR_DEBUG", "0").strip().lower() in {"1", "true", "yes"}

    monitoring_system = MonitoringSystem()
    monitoring_system.start()

    dashboard_config = {
        "web": {
            "host": host,
            "port": port,
            "update_interval": update_interval,
        }
    }
    dashboard = MonitoringDashboardGenerator(monitoring_system, dashboard_config)

    try:
        dashboard.run_web_dashboard(debug=debug)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    finally:
        monitoring_system.stop()


if __name__ == "__main__":
    main()
