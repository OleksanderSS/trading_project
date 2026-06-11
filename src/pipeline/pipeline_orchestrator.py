# src/pipeline/pipeline_orchestrator.py

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, cast

from src.analytics.data_managers.model_results_manager import ModelResultsManager
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.core.monitoring.memory_profiler import get_memory_profiler, get_memory_stats
from src.data.management.data_manager import DataManager
from src.monitoring.health_hub import HealthHub
from src.processing.normalization_manager import NormalizationManager
from src.utils.dynamic_module_loader import DynamicModuleLoader
from src.validation.pipeline_schemas import (
    validate_stage_output,
)


@dataclass
class StageLoggingRequest:
    """Request for stage completion logging."""
    stage_name: str
    start_time: float
    end_time: float
    initial_mem: float
    final_mem: float

class PipelineOrchestrator:
    """
    Orchestrates the execution of pipeline stages, managing data flow and dependencies.
    """

    def __init__(
        self,
        config_manager: UnifiedConfigManager,
        brain: dict[str, Any] | None = None,
        stages_to_run: list[int] | None = None
    ):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger(__name__)
        self.brain = brain or {}
        self.stages_to_run = stages_to_run
        self.error_handler = ErrorHandler(config_manager)

        paths_config = self.config_manager.get_config('paths') or {}
        models_path = paths_config.get('models', 'trained_models')
        scaler_path = paths_config.get('scalers', 'data/scalers')

        if not scaler_path:
            self.logger.warning("scaler_path not resolved from config 'paths.scalers'.")
            scaler_path = 'data/scalers'

        self.data_manager = DataManager(self.config_manager)
        self.results_manager = ModelResultsManager(models_path)
        self.http_client_factory = HttpClientFactory(self.config_manager, self.error_handler)
        self.normalizer = NormalizationManager(scaler_dir=scaler_path)
        self.health_hub = HealthHub(self.config_manager, self.data_manager, self.results_manager)

        # ✅ Phase 3 Optimization: Initialize memory profiler
        memory_warn_gb = self.config_manager.get_config('performance.memory_warn_gb', 10.0)
        self.memory_profiler = get_memory_profiler(warn_threshold_gb=memory_warn_gb)
        self.logger.info(f"✅ Memory profiler enabled (warning threshold: {memory_warn_gb}GB)")

        self.stages = self._load_stages()

    def _load_stages(self) -> list[Any]:
        """Loads pipeline stages from the configuration."""
        self.logger.info("Loading pipeline stages from configuration...")
        stages_config = self.config_manager.get_config('training_pipeline', [])

        loaded_stages = []
        for i, stage_info in enumerate(stages_config):
            if not stage_info.get('enabled', False):
                self.logger.info(f"Stage '{stage_info.get('name')}' is disabled in config. Skipping.")
                continue

            # ✅ Phase 3 Optimization: Only load stages we actually intend to run
            if self.stages_to_run is not None and i not in self.stages_to_run:
                self.logger.debug(f"Stage {i} ('{stage_info.get('name')}') not in stages_to_run. Skipping load.")
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
                stage_instance._pipeline_stage_index = i
                loaded_stages.append(stage_instance)
                self.logger.info(f"Stage '{stage_info.get('name')}' loaded successfully.")
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                self.logger.error(f"Failed to load stage '{stage_info.get('name')}': {e}", exc_info=True)
                raise

        return loaded_stages

    def _execute_sync(self, coro: Any) -> dict[str, Any]:
        if inspect.isawaitable(coro):
            return asyncio.run(coro)
        return cast(dict[str, Any], coro)

    def execute_full_pipeline(
        self,
        tickers: list[str] | None = None,
        timeframes: list[str] | None = None,
        **kwargs
    ) -> dict[str, Any]:
        return self._execute_sync(
            self.run(tickers=tickers, timeframes=timeframes, run_mode='predict', **kwargs)
        )

    def execute_training_pipeline(
        self,
        tickers: list[str] | None = None,
        timeframes: list[str] | None = None,
        **kwargs
    ) -> dict[str, Any]:
        return self._execute_sync(
            self.run(tickers=tickers, timeframes=timeframes, run_mode='train', **kwargs)
        )

    def run_incremental_pipeline(
        self,
        execution_params: dict[str, Any] | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Legacy compatibility wrapper for older experiments.

        Args:
            execution_params: Dictionary containing tickers, start_date, end_date, feature_layers
            **kwargs: Additional parameters
        """
        params = execution_params or {}
        params.update(kwargs)
        return self._execute_sync(self.run(**params))

    async def run(self, tickers: list[str] | None = None, timeframes: list[str] | None = None, run_mode: str = 'train', **kwargs) -> dict[str, Any]:
        """Runs the entire pipeline or specific stages if provided."""
        self.logger.debug("PipelineOrchestrator.run() called")

        stages_to_run = kwargs.pop('stages_to_run', self.stages_to_run)

        # Build execution context from parameters
        execution_context = {
            'tickers': tickers,
            'timeframes': timeframes,
            'run_mode': run_mode,
            'stages_to_run': stages_to_run,
            **kwargs
        }

        # ✅ ENHANCED: Update brain with context data to ensure all stages can access it
        for key in ['news_data', 'economic_data', 'market_indicators', 'macro_data']:
            if key in kwargs and kwargs[key] is not None:
                self.brain[key] = kwargs[key]
                self.logger.info(f"🧠 Updated brain with {key} ({len(kwargs[key]) if hasattr(kwargs[key], '__len__') else 'exists'})")

        stage_outputs = self._initialize_stage_outputs(execution_context)

        for stage in self.stages:
            i = getattr(stage, '_pipeline_stage_index', -1)
            if not self._should_run_stage(i, stage, stages_to_run):
                continue

            stage_result = await self._execute_stage(i, stage, stage_outputs)
            if stage_result['status'] == 'failed':
                return stage_outputs

            stage_outputs.update(stage_result['outputs'])

        self._log_memory_statistics()
        self.logger.info("Pipeline execution completed successfully.")
        return stage_outputs

    def _initialize_stage_outputs(self, execution_context: dict[str, Any]) -> dict[str, Any]:
        """Initialize stage outputs with basic parameters.

        Args:
            execution_context: Dictionary containing tickers, timeframes, run_mode, and additional kwargs
        """
        tickers = execution_context.get('tickers')
        timeframes = execution_context.get('timeframes')
        run_mode = execution_context.get('run_mode', 'train')

        num_stages = len(self.stages)
        self.logger.info(f"Starting pipeline with {num_stages} stages...")
        self.logger.info(f"Stages to run: {self.stages_to_run if execution_context.get('stages_to_run') is None else execution_context.get('stages_to_run')}")
        self.logger.info(f"Context keys: {list(execution_context.keys())}")

        stage_outputs = {
            'tickers': tickers,
            'timeframes': timeframes,
            'run_mode': run_mode
        }
        # Merge in all additional context parameters
        stage_outputs.update({k: v for k, v in execution_context.items()
                             if k not in ('tickers', 'timeframes', 'run_mode')})

        self.logger.info(f"📊 Initial stage_outputs keys: {list(stage_outputs.keys())}")
        if 'models_metadata' in stage_outputs:
            self.logger.info(f"📊 models_metadata count: {len(stage_outputs['models_metadata'])}")

        return stage_outputs

    def _should_run_stage(self, stage_index: int, stage: Any, stages_to_run: list[int] | None) -> bool:
        """Check if stage should be executed."""
        stage_name = type(stage).__name__

        if stages_to_run and stage_index not in stages_to_run:
            self.logger.info(f"SKIPPING Stage {stage_index}: {stage_name} - NOT in stages_to_run {stages_to_run}")
            return False

        return True

    async def _run_stage_with_memory_tracking(self, stage_index: int, stage_name: str, stage: Any, stage_outputs: dict[str, Any]) -> dict[str, Any] | None:
        """Run stage with memory tracking."""
        with self.memory_profiler.track(f"stage_{stage_index}_{stage_name}"):
            return await stage.run(**stage_outputs)

    async def _execute_stage(self, stage_index: int, stage: Any, stage_outputs: dict[str, Any]) -> dict[str, Any]:
        """Executes a single pipeline stage with logging and error handling."""
        stage_name = type(stage).__name__
        self.logger.info(f"🚀 Starting Stage {stage_index}: {stage_name}")

        start_time = time.time()
        initial_mem = self._get_memory_usage() * 1024  # MB

        try:
            # Execute the stage using memory tracking
            output = await self._run_stage_with_memory_tracking(stage_index, stage_name, stage, stage_outputs)

            if output is None:
                output = {}

            # Validate output
            validated_output = self._validate_stage_output(stage_name, output)
            if validated_output:
                output = validated_output

            end_time = time.time()
            final_mem = self._get_memory_usage() * 1024  # MB

            self._log_stage_completion(StageLoggingRequest(
                stage_name=stage_name,
                start_time=start_time,
                end_time=end_time,
                initial_mem=initial_mem,
                final_mem=final_mem
            ))

            self._log_models_metadata(stage_name, output)

            return {'status': 'success', 'outputs': output}

        except Exception as e:
            self._handle_stage_error(e, stage_name, stage_outputs)
            return {'status': 'failed', 'outputs': {}}

    def _validate_stage_output(self, stage_name: str, stage_output: dict[str, Any] | None) -> dict[str, Any] | None:
        """Validate stage output against schema."""
        if not stage_output:
            return None

        # Dynamically import schemas to avoid circular dependencies and hardcoding
        try:
            from src.validation.pipeline_schemas import (
                EnrichedDataSchema,
                EvaluationDataSchema,
                ModelingDataSchema,
                PredictionDataSchema,
                ProcessedDataSchema,
                RawDataSchema,
                TradingDataSchema,
            )
        except ImportError:
            self.logger.warning("Could not import all schemas, using basic validation.")
            return stage_output

        stage_schema_map = {
            "CollectionStage": RawDataSchema,
            "ProcessingStage": ProcessedDataSchema,
            "FeatureEngineeringStage": EnrichedDataSchema,
            "ModelingStage": ModelingDataSchema,
            "PredictionStage": PredictionDataSchema,
            "TradingStage": TradingDataSchema,
            "EvaluationStage": EvaluationDataSchema
        }

        try:
            schema = stage_schema_map.get(stage_name)
            if schema:
                return validate_stage_output(stage_name, stage_output, schema)

            self.logger.debug(f"[OK] Stage {stage_name} (no schema validation defined)")
            return stage_output
        except Exception as e:
            self.logger.warning(f"[WARNING] Stage {stage_name} output validation failed: {e}")
            return stage_output

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return self.health_hub.resource_monitor.get_health_status()['system']['memory']['used_gb']

    def _log_models_metadata(self, stage_name: str, stage_outputs: dict[str, Any]) -> None:
        """Log models metadata presence after stage execution."""
        self.logger.info("Updated stage_outputs with keys")

        if 'models_metadata' in stage_outputs:
            self.logger.info(f"📊 models_metadata still present: {len(stage_outputs['models_metadata'])} models")
        else:
            self.logger.warning(f"⚠️ models_metadata NOT in stage_outputs after {stage_name}")

    def _log_stage_completion(self, request: StageLoggingRequest) -> None:
        """Log stage completion with timing and memory info."""
        self.logger.info(
            f"===== Stage {request.stage_name} finished in {request.end_time - request.start_time:.2f}s. "
            f"Memory: {request.final_mem:.1f}MB (Delta {request.final_mem - request.initial_mem:.1f}MB) ====="
        )

    def _handle_stage_error(self, error: Exception, stage_name: str, stage_outputs: dict[str, Any]) -> None:
        """Handle stage execution error."""
        self.error_handler.handle_error(
            error,
            context=f"PipelineOrchestrator:{stage_name}",
            severity="critical"
        )
        self.logger.critical(f"Critical error in stage '{stage_name}', stopping pipeline.", exc_info=True)
        stage_outputs['pipeline_status'] = 'failed'
        stage_outputs['failed_stage'] = stage_name

    def _log_memory_statistics(self) -> None:
        """Log memory profiling statistics."""
        memory_stats = get_memory_stats() or {}
        peak_memory_gb = memory_stats.get('peak_memory_gb', 0.0)
        operations_tracked = memory_stats.get('operations_tracked', 0)
        warnings_issued = memory_stats.get('warnings_issued', 0)
        cleanups_performed = memory_stats.get('cleanups_performed', 0)
        memory_freed_mb = memory_stats.get('memory_freed_mb', 0.0)
        self.logger.info(
            f"🧠 Pipeline memory profile - Peak: {peak_memory_gb:.2f}GB, "
            f"Operations: {operations_tracked}, Warnings: {warnings_issued}, "
            f"Cleanups: {cleanups_performed} ({memory_freed_mb:.1f}MB freed)"
        )
