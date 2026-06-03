from typing import Any, Dict

from src.analytics.reporting.automated_reports import HistoricalAnalytics
from src.analytics.reporting.model_analyzer import ModelAnalyzer
from src.analytics.reporting.results_manager import ModelResultsManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ReportingOrchestrator")


class ReportingOrchestrator:
    """
    Centralized orchestrator for system reporting and results management.
    Activates previously unused tools and provides a clean API for analytics.
    """

    def __init__(self, results_manager: ModelResultsManager):
        self.results_manager = results_manager
        self.historical_analyzer = HistoricalAnalytics(results_manager)
        self.model_analyzer = ModelAnalyzer({})

    def get_full_status_report(self) -> Dict[str, Any]:
        """Consolidates all reporting data."""
        return {
            "latest_results": self.results_manager.get_latest_results(),
            "recent_trends": self.historical_analyzer.analyze_trends(days=30),
            "model_stats": self.model_analyzer.get_light_vs_heavy_stats(),
            "cache_status": self.results_manager.get_cache_info(),
            "available_results": self.results_manager.list_results(),
        }

    def export_all(self, output_path: str):
        """Exports all results to CSV."""
        return self.results_manager.export_results_to_csv("summary.csv", output_path=output_path)
