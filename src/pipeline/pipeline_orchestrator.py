import asyncio
import inspect
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.analytics.data_managers.model_results_manager import ModelResultsManager
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler
from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger
from src.core.monitoring.memory_profiler import get_memory_profiler, get_memory_stats
from src.data.management.data_manager import DataManager
from src.monitoring.health_hub import HealthHub
from src.processing.normalization_manager import NormalizationManager
from src.utils.dynamic_module_loader import DynamicModuleLoader
from src.validation.pipeline_schemas import (
    EnrichedDataSchema,
    ProcessedDataSchema,
    RawDataSchema,
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

    def __init__(self, config_manager: UnifiedConfigManager, brain:
        dict[str, Any] | None=None, stages_to_run: list[int] | None=None,
        data_manager=None
        ):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger(__name__)
        self.brain = brain or {}
        self.stages_to_run = stages_to_run
        self.error_handler = ErrorHandler(config_manager)
        paths_config = self.config_manager.get_config('paths') or {}
        models_path = paths_config.get('models', 'trained_models')
        scaler_path = paths_config.get('scalers')
        if not scaler_path:
            self.logger.warning(
                "scaler_path not resolved from config 'paths.scalers'.")
        self.data_manager = data_manager if data_manager else DataManager(self.config_manager)
        self.results_manager = ModelResultsManager(models_path)
        self.http_client_factory = HttpClientFactory(self.config_manager,
            self.error_handler)
        self.normalizer = NormalizationManager(scaler_dir=scaler_path)
        self.health_hub = HealthHub(self.config_manager, self.data_manager,
            self.results_manager)
        memory_warn_gb = self.config_manager.get_config(
            'performance.memory_warn_gb', 10.0)
        self.memory_profiler = get_memory_profiler(warn_threshold_gb=
            memory_warn_gb)
        self.logger.info(
            f'✅ Memory profiler enabled (warning threshold: {memory_warn_gb}GB)'
            )
        self.stages = self._load_stages()

    def _load_stages(self) ->list[Any]:
        """Loads pipeline stages from the configuration.

        Only instantiates stages that are in stages_to_run (if specified).
        This prevents unnecessary initialization (HTTP clients, DB connections)
        for stages like CollectionStage when running --mode continue.
        """
        self.logger.info('Loading pipeline stages from configuration...')
        stages_config = self.config_manager.get_config('training_pipeline', [])
        loaded_stages = []
        global_index = 0  # tracks index across all stages (enabled + disabled)
        for stage_info in stages_config:
            if not stage_info.get('enabled', False):
                self.logger.info(
                    f"Stage '{stage_info.get('name')}' is disabled in config. Skipping."
                    )
                global_index += 1
                continue
            # Skip instantiation if stage index is not in stages_to_run
            if self.stages_to_run is not None and global_index not in self.stages_to_run:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"Stage {global_index} '{stage_info.get('name')}' not in "
                        f"stages_to_run={self.stages_to_run} — skipping load."
                    )
                global_index += 1
                continue
            try:
                dependencies = {'config_manager': self.config_manager,
                    'db_manager': self.data_manager, 'http_client_factory':
                    self.http_client_factory, 'normalizer': self.normalizer,
                    'error_handler': self.error_handler, 'brain': self.brain}
                stage_instance = DynamicModuleLoader.load_instance(stage_info,
                    **dependencies)
                loaded_stages.append(stage_instance)
                self.logger.info(
                    f"Stage {global_index} '{stage_info.get('name')}' loaded successfully.")
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                self.logger.exception(
                    f"Failed to load stage '{stage_info.get('name')}': {e}")
                raise RuntimeError(f"Failed to load stage '{stage_info.get('name')}'") from e
            global_index += 1
        return loaded_stages

    def _execute_sync(self, coro: Any) ->Any:
        if inspect.isawaitable(coro):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)
            timeout_seconds = self.config_manager.get(
                'pipeline.sync_timeout_seconds', 300)
            result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

            def _runner() ->None:
                try:
                    result_queue.put(('value', asyncio.run(coro)))
                except BaseException as exc:
                    result_queue.put(('error', exc))

            worker = threading.Thread(target=_runner, daemon=True)
            worker.start()
            worker.join(timeout=timeout_seconds)
            if worker.is_alive():
                raise TimeoutError(
                    f'Pipeline execution timed out after {timeout_seconds}s')
            try:
                result_type, payload = result_queue.get_nowait()
            except queue.Empty as exc:
                raise RuntimeError(
                    'Pipeline execution finished without returning a result'
                    ) from exc
            if result_type == 'error':
                raise payload
            return payload
        return coro

    def execute_full_pipeline(self, tickers: list[str] | None=None,
        timeframes: list[str] | None=None, **kwargs) ->dict[str, Any]:
        return self._execute_sync(self.run(tickers=tickers, timeframes=
            timeframes, run_mode='predict', **kwargs))

    def execute_training_pipeline(self, tickers: list[str] | None=None,
        timeframes: list[str] | None=None, **kwargs) ->dict[str, Any]:
        return self._execute_sync(self.run(tickers=tickers, timeframes=
            timeframes, run_mode='train', **kwargs))

    def run_incremental_pipeline(self, execution_params: dict[str, Any] | None=None, **kwargs) ->dict[str, Any]:
        """Legacy compatibility wrapper for older experiments.

        Args:
            execution_params: Dictionary containing tickers, start_date, end_date, feature_layers
            **kwargs: Additional parameters
        """
        params = execution_params or {}
        params.update(kwargs)
        return self._execute_sync(self.run(**params))

    async def run(self, tickers: list[str] | None=None, timeframes:
        list[str] | None=None, run_mode: str='train', **kwargs):
        """Runs the entire pipeline or specific stages if provided."""
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug('PipelineOrchestrator.run() called')
        # stages_to_run from kwargs takes priority, then from constructor
        stages_to_run = kwargs.pop('stages_to_run', self.stages_to_run)
        execution_context = {'tickers': tickers, 'timeframes': timeframes,
            'run_mode': run_mode, 'stages_to_run': stages_to_run, **kwargs}
        stage_outputs = self._initialize_stage_outputs(execution_context)
        # When stages are filtered at load time, enumerate starts from 0
        # but the logical index corresponds to the original config position.
        # We use the stage's position in self.stages (already filtered).
        for idx, stage in enumerate(self.stages):
            stage_result = await self._execute_stage(stage, stage_outputs, stage_index=idx)
            if stage_result['status'] == 'failed':
                return stage_outputs
            stage_outputs.update(stage_result['outputs'])
        self._log_memory_statistics()
        self.logger.info('Pipeline execution completed successfully.')
        return stage_outputs

    def _initialize_stage_outputs(self, execution_context: dict[str, Any]
        ) ->dict[str, Any]:
        """Initialize stage outputs with basic parameters.

        Args:
            execution_context: Dictionary containing tickers, timeframes, run_mode, and additional kwargs
        """
        tickers = execution_context.get('tickers')
        timeframes = execution_context.get('timeframes')
        run_mode = execution_context.get('run_mode', 'train')
        num_stages = len(self.stages)
        self.logger.info(f'Starting pipeline with {num_stages} stages...')
        self.logger.info(
            f"Stages to run: {self.stages_to_run if execution_context.get('stages_to_run') is None else execution_context.get('stages_to_run')}"
            )
        self.logger.info(f'Context keys: {list(execution_context.keys())}')
        stage_outputs = {'tickers': tickers, 'timeframes': timeframes,
            'run_mode': run_mode}

        # Merge enriched_data and targets_df if both are provided
        enriched_data = execution_context.get('enriched_data')
        targets_df = execution_context.get('targets_df')
        if enriched_data is not None and targets_df is not None:
            # Merge targets into enriched_data using concat on index
            self.logger.info(f"Merging enriched_data ({enriched_data.shape}) with targets_df ({targets_df.shape})")
            # Remove duplicate columns from targets_df (ticker, datetime, interval)
            target_cols_only = targets_df.drop(columns=['ticker', 'datetime', 'interval'], errors='ignore')
            # Reset index to ensure proper alignment
            enriched_data_reset = enriched_data.reset_index(drop=True)
            targets_df_reset = target_cols_only.reset_index(drop=True)
            merged_df = pd.concat([enriched_data_reset, targets_df_reset], axis=1)
            self.logger.info(f"Merged DataFrame shape: {merged_df.shape}")
            self.logger.info(f"Target columns in merged DataFrame: {[col for col in merged_df.columns if col.startswith('target_')]}")
            stage_outputs['enriched_data'] = merged_df
            # Remove separate targets_df and old enriched_data to avoid confusion
            execution_context.pop('targets_df', None)
            execution_context.pop('enriched_data', None)

        stage_outputs.update({k: v for k, v in execution_context.items() if
            k not in ('tickers', 'timeframes', 'run_mode')})
        self.logger.info(
            f'📊 Initial stage_outputs keys: {list(stage_outputs.keys())}')
        if 'models_metadata' in stage_outputs:
            self.logger.info(
                f"📊 models_metadata count: {len(stage_outputs['models_metadata'])}"
                )
        return stage_outputs

    def _should_run_stage(self, stage_index: int, stage: Any, stages_to_run:
        list[int] | None) ->bool:
        """Check if stage should be executed."""
        stage_name = type(stage).__name__
        if stages_to_run and stage_index not in stages_to_run:
            self.logger.info(
                f'SKIPPING Stage {stage_index}: {stage_name} - NOT in stages_to_run {stages_to_run}'
                )
            return False
        return True

    async def _execute_stage(self, stage: Any,
        stage_outputs: dict[str, Any], stage_index: int=0) ->dict[str, Any]:
        """Execute a single stage and return results."""
        stage_name = type(stage).__name__
        self.logger.info(
            f'===== Executing Stage {stage_index}: {stage_name} =====' if stage_index > 0
            else f'===== Executing {stage_name} ====='
        )
        start_time = time.time()
        initial_mem = self._get_memory_usage()
        try:
            stage_output = await self._run_stage_with_memory_tracking(
                stage_index, stage_name, stage, stage_outputs)
            validated_output = self._validate_stage_output(stage_name,
                stage_output)
            if validated_output:
                if stage_name == 'CollectionStage':
                    stage_outputs['raw_data'] = validated_output
                else:
                    stage_outputs.update(validated_output)
                self._log_models_metadata(stage_name, stage_outputs)
            end_time = time.time()
            final_mem = self._get_memory_usage()
            logging_request = StageLoggingRequest(stage_name=stage_name,
                start_time=start_time, end_time=end_time, initial_mem=
                initial_mem, final_mem=final_mem)
            self._log_stage_completion(logging_request)
            return {'status': 'success', 'outputs': validated_output or {}}
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f'Помилка виконання стадії {stage_name}: {e}', exc_info=True)
            self._handle_stage_error(e, stage_name, stage_outputs)
            raise RuntimeError(f"Stage {stage_name} execution failed: {e}") from e

    async def _run_stage_with_memory_tracking(self, stage_index: int,
        stage_name: str, stage: Any, stage_outputs: dict[str, Any]) ->dict[str, Any] | None:
        """Run stage with memory tracking."""
        with self.memory_profiler.track(f'stage_{stage_index}_{stage_name}'):
            # Log enriched_data shape for ModelingStage
            if stage_name == 'ModelingStage':
                enriched_data = stage_outputs.get('enriched_data')
                if enriched_data is not None:
                    self.logger.info(f"ModelingStage receiving enriched_data with shape: {enriched_data.shape}")
                    target_cols = [col for col in enriched_data.columns if col.startswith('target_')]
                    self.logger.info(f"ModelingStage enriched_data target columns: {len(target_cols)}")
            return await stage.run(**stage_outputs)

    def _validate_stage_output(self, stage_name: str, stage_output:
        dict[str, Any] | None) ->dict[str, Any] | None:
        """Validate stage output against schema."""
        if not stage_output:
            return None
        stage_schema_map = {'CollectionStage': RawDataSchema,
            'ProcessingStage': ProcessedDataSchema,
            'FeatureEngineeringStage': EnrichedDataSchema}

        schema = stage_schema_map.get(stage_name)
        if not schema:
            return stage_output

        try:
            return validate_stage_output(stage_name, stage_output, schema)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Помилка валідації виходу стадії {stage_name}: {e}', exc_info=True)
            raise DataProcessingError(f"Stage {stage_name} output validation failed: {e}") from e

    def _get_memory_usage(self) ->float:
        """Get current memory usage in GB."""
        return self.health_hub.resource_monitor.get_health_status()['system'][
            'memory']['used_gb']

    def _log_models_metadata(self, stage_name: str, stage_outputs: dict[str,
        Any]) ->None:
        """Log models metadata presence after stage execution."""
        self.logger.info('Updated stage_outputs with keys')
        metadata = stage_outputs.get('models_metadata')
        if metadata is not None:
            self.logger.info(
                f"📊 models_metadata still present: {len(metadata)} models"
                )
        else:
            self.logger.warning(
                f'⚠️ models_metadata NOT in stage_outputs after {stage_name}')

    def _log_stage_completion(self, request: StageLoggingRequest) ->None:
        """Log stage completion with timing and memory info."""
        self.logger.info(
            f'===== Stage {request.stage_name} finished in {request.end_time - request.start_time:.2f}s. Memory: {request.final_mem:.1f}MB (Delta {request.final_mem - request.initial_mem:.1f}MB) ====='
            )

    def _handle_stage_error(self, error: Exception, stage_name: str,
        stage_outputs: dict[str, Any]) ->None:
        """Handle stage execution error."""
        self.error_handler.handle_error(error, context=
            f'PipelineOrchestrator:{stage_name}', severity='critical')
        self.logger.critical(
            f"Critical error in stage '{stage_name}', stopping pipeline.",
            exc_info=True)
        stage_outputs['pipeline_status'] = 'failed'
        stage_outputs['failed_stage'] = stage_name

    def _log_memory_statistics(self) ->None:
        """Log memory profiling statistics."""
        memory_stats = get_memory_stats() or {}
        peak_memory_gb = memory_stats.get('peak_memory_gb', 0.0)
        operations_tracked = memory_stats.get('operations_tracked', 0)
        warnings_issued = memory_stats.get('warnings_issued', 0)
        cleanups_performed = memory_stats.get('cleanups_performed', 0)
        memory_freed_mb = memory_stats.get('memory_freed_mb', 0.0)
        self.logger.info(
            f'🧠 Pipeline memory profile - Peak: {peak_memory_gb:.2f}GB, Operations: {operations_tracked}, Warnings: {warnings_issued}, Cleanups: {cleanups_performed} ({memory_freed_mb:.1f}MB freed)'
            )
