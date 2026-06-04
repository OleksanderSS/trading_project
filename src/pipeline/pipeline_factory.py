from src.analytics.data_managers.model_results_manager import ModelResultsManager
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler
from src.core.monitoring.memory_profiler import get_memory_profiler
from src.data.management.data_manager import DataManager
from src.monitoring.health_hub import HealthHub
from src.processing.normalization_manager import NormalizationManager


class PipelineFactory:
    """Фабрика для створення залежностей PipelineOrchestrator."""

    @staticmethod
    def create_dependencies(config_manager: UnifiedConfigManager, error_handler: ErrorHandler):
        paths_config = config_manager.get_config("paths") or {}
        models_path = paths_config.get("models", "trained_models")
        scaler_path = paths_config.get("scalers")

        data_manager = DataManager(config_manager)
        results_manager = ModelResultsManager(models_path)
        http_client_factory = HttpClientFactory(config_manager, error_handler)
        normalizer = NormalizationManager(scaler_dir=scaler_path)
        health_hub = HealthHub(config_manager, data_manager, results_manager)

        memory_warn_gb = config_manager.get_config("performance.memory_warn_gb", 10.0)
        memory_profiler = get_memory_profiler(warn_threshold_gb=memory_warn_gb)

        return {
            "data_manager": data_manager,
            "results_manager": results_manager,
            "http_client_factory": http_client_factory,
            "normalizer": normalizer,
            "health_hub": health_hub,
            "memory_profiler": memory_profiler
        }
