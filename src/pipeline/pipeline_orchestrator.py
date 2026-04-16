
import asyncio
import inspect
import logging
import time
from typing import List, Dict, Optional, Any

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.utils.dynamic_module_loader import DynamicModuleLoader
from src.monitoring.health_hub import HealthHub
from src.data.management.data_manager import DataManager
from src.analytics.data_managers.model_results_manager import ModelResultsManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.processing.normalization_manager import NormalizationManager
from src.core.error_handling.error_handler import ErrorHandler

class PipelineOrchestrator:
    """
    Orchestrates the execution of pipeline stages, managing data flow and dependencies.
    """

    def __init__(
        self,
        config_manager: UnifiedConfigManager,
        brain: Optional[Dict[str, Any]] = None,
        stages_to_run: Optional[List[int]] = None
    ):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger(__name__)
        self.brain = brain or {}
        self.stages_to_run = stages_to_run
        self.error_handler = ErrorHandler(config_manager)

        # --- Diagnostic Change ---
        self.logger.info("Attempting to retrieve 'paths' configuration...")
        paths_config = self.config_manager.get_config('paths')
        self.logger.info(f"Retrieved 'paths' config: {paths_config}")

        db_path = paths_config.get('raw_db') if paths_config else 'data/raw_data.duckdb'
        models_path = paths_config.get('models') if paths_config else 'trained_models'
        scaler_path = paths_config.get('scalers') if paths_config else None

        self.logger.info(f"Resolved scaler_path: {scaler_path}")
        if not scaler_path:
             self.logger.error("Failed to resolve scaler_path, it is None.")
        # --- End Diagnostic Change ---

        self.data_manager = DataManager(self.config_manager)
        self.results_manager = ModelResultsManager(models_path)
        self.http_client_factory = HttpClientFactory(self.config_manager, self.error_handler)
        self.normalizer = NormalizationManager(scaler_dir=scaler_path)
        self.health_hub = HealthHub(self.config_manager, self.data_manager, self.results_manager)
        self.stages = self._load_stages()

    def _load_stages(self) -> List[Any]:
        """Loads pipeline stages from the configuration."""
        self.logger.info("Loading pipeline stages from configuration...")
        stages_config = self.config_manager.get_config('training_pipeline', [])
        
        loaded_stages = []
        for stage_info in stages_config:
            if not stage_info.get('enabled', False):
                self.logger.info(f"Stage '{stage_info.get('name')}' is disabled in config. Skipping.")
                continue

            try:
                dependencies = {
                    "config_manager": self.config_manager,
                    "db_manager": self.data_manager,
                    "http_client_factory": self.http_client_factory,
                    "normalizer": self.normalizer,
                    "error_handler": self.error_handler,
                    "brain": self.brain
                }
                stage_instance = DynamicModuleLoader.load_instance(stage_info, **dependencies)
                loaded_stages.append(stage_instance)
                self.logger.info(f"Stage '{stage_info.get('name')}' loaded successfully.")
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                self.logger.error(f"Failed to load stage '{stage_info.get('name')}': {e}", exc_info=True)
                raise
        
        return loaded_stages

    def _execute_sync(self, coro: Any) -> Any:
        if inspect.isawaitable(coro):
            return asyncio.run(coro)
        return coro

    def execute_full_pipeline(
        self,
        tickers: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        return self._execute_sync(
            self.run(tickers=tickers, timeframes=timeframes, run_mode='predict', **kwargs)
        )

    def execute_training_pipeline(
        self,
        tickers: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        return self._execute_sync(
            self.run(tickers=tickers, timeframes=timeframes, run_mode='train', **kwargs)
        )

    def run_incremental_pipeline(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_layers: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Legacy compatibility wrapper for older experiments.

        This method delegates to the modern async pipeline and preserves
        backward compatibility for scripts that still call the old interface.
        """
        params = {
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'feature_layers': feature_layers,
            **kwargs
        }
        return self._execute_sync(self.run(**params))

    async def run(self, tickers: Optional[List[str]] = None, timeframes: Optional[List[str]] = None, run_mode: str = 'train', **kwargs):
        """Runs the entire pipeline or specific stages if provided."""
        # ✅ FIX: Дозволяємо перевизначити stages_to_run через kwargs
        stages_to_run = kwargs.pop('stages_to_run', self.stages_to_run)
        
        num_stages = len(self.stages)
        self.logger.info(f"Starting pipeline with {num_stages} stages...")
        self.logger.info(f"Stages to run: {stages_to_run}")

        stage_outputs: Dict[str, Any] = {
            'tickers': tickers,
            'timeframes': timeframes,
            'run_mode': run_mode
        }
        
        # ✅ FIX: Додаємо всі kwargs в stage_outputs
        stage_outputs.update(kwargs)
        
        # ✅ DEBUG: Логуємо що передається в stage_outputs
        self.logger.info(f"📊 Initial stage_outputs keys: {list(stage_outputs.keys())}")
        if 'models_metadata' in stage_outputs:
            self.logger.info(f"📊 models_metadata count: {len(stage_outputs['models_metadata'])}")
            self.logger.info(f"📊 models_metadata keys (first 3): {list(stage_outputs['models_metadata'].keys())[:3]}")

        for i, stage in enumerate(self.stages):
            stage_name = type(stage).__name__
            self.logger.info(f"DEBUG: Checking stage {i}: {stage_name}. stages_to_run: {stages_to_run}")

            if stages_to_run and i not in stages_to_run:
                self.logger.info(f"Skipping Stage {i}: {stage_name} as it is not in the list of stages to run ({stages_to_run}).")
                continue

            self.logger.info(f"===== Executing Stage {i}: {stage_name} =====")
            start_time = time.time()
            initial_mem = self.health_hub.resource_monitor.get_health_status()['system']['memory']['used_gb'] * 1024


            try:
                stage_output = await stage.run(**stage_outputs)
                
                self.logger.info(f"Stage output type: {type(stage_output)}, keys: {stage_output.keys() if stage_output else 'None'}")
                
                if stage_output:
                    stage_outputs.update(stage_output)
                    self.logger.info(f"Updated stage_outputs with {len(stage_output)} keys")
                    
                    # ✅ DEBUG: Логуємо models_metadata після кожної стадії
                    if 'models_metadata' in stage_outputs:
                        self.logger.info(f"📊 models_metadata still present: {len(stage_outputs['models_metadata'])} моделей")
                    else:
                        self.logger.warning(f"⚠️ models_metadata NOT in stage_outputs after {stage_name}")

                end_time = time.time()
                final_mem = self.health_hub.resource_monitor.get_health_status()['system']['memory']['used_gb'] * 1024
                self.logger.info(
                    f"===== Stage {stage_name} finished in {end_time - start_time:.2f}s. "
                    f"Memory: {final_mem:.1f}MB (\u0394 {final_mem - initial_mem:.1f}MB) ====="
                )
            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    context=f"PipelineOrchestrator:{stage_name}",
                    severity="critical"
                )
                self.logger.critical(f"Critical error in stage '{stage_name}', stopping pipeline.", exc_info=True)
                stage_outputs['pipeline_status'] = 'failed'
                stage_outputs['failed_stage'] = stage_name
                break
        
        self.logger.info("Pipeline execution completed successfully.")
        return stage_outputs
