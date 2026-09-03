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
        memory_warn_gb = self.config_manager.get(
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

        # Merge enriched_data and targets_df if both are provided.
        #
        # Both may now be DICTS keyed by timeframe rather than single frames.
        # The batch is written that way because the combined frame carries
        # every timeframe's columns on every row -- 154,069 daily rows holding
        # 1,836 unused columns -- and loading it costs 4.85 GiB of resident
        # memory against 0.27 for the daily slice alone. `iter_model_contexts`
        # already accepts either shape; this is the last place that did not.
        enriched_data = execution_context.get('enriched_data')
        targets_df = execution_context.get('targets_df')
        if enriched_data is not None and targets_df is not None:
            if isinstance(enriched_data, dict) and isinstance(targets_df, dict):
                merged: dict[str, Any] = {}
                for timeframe, features in enriched_data.items():
                    targets = targets_df.get(timeframe)
                    if targets is None or getattr(features, 'empty', True):
                        # A timeframe with features and no targets is not an
                        # error -- it simply cannot be trained on -- but it must
                        # not be passed on silently as if it could.
                        self.logger.warning(
                            'No targets for timeframe %s; it will not be '
                            'trained on.', timeframe,
                        )
                        continue
                    merged[timeframe] = self._merge_features_and_targets(
                        features, targets, label=timeframe,
                    )
                stage_outputs['enriched_data'] = merged
            else:
                stage_outputs['enriched_data'] = self._merge_features_and_targets(
                    enriched_data, targets_df, label='batch',
                )
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

    def _merge_features_and_targets(self, enriched_data, targets_df, *,
                                    label: str = 'batch'):
        """Pair each bar's features with THAT bar's outcome, not another's."""
        self.logger.info(
            "Merging %s: enriched_data %s with targets_df %s",
            label, enriched_data.shape, targets_df.shape,
        )
        # Remove duplicate columns from targets_df (ticker, datetime, interval)
        target_cols_only = targets_df.drop(
            columns=['ticker', 'datetime', 'interval'], errors='ignore'
        )

        # This pairs row i of the features with row i of the targets, and
        # nothing here checked that row i is the same BAR in both.
        #
        # Verified on the 18.08 batch: 256,062 rows on each side, ticker,
        # datetime and interval identical on every one of them. So the
        # assumption has been holding. It is unchecked, not wrong.
        #
        # It is one reordering away from silent garbage, and this pipeline
        # reorders: `Enricher 'nlp_features' returned the same 28856 rows
        # in a DIFFERENT ORDER` is a warning that appears in these logs.
        # Equal row counts would survive that; the pairing would not, and
        # a model trained on it learns the relationship between one bar's
        # features and another bar's outcome.
        #
        # So the invariant is asserted rather than assumed, and when it
        # fails the frames are merged on their keys instead. Checking three
        # columns costs nothing next to a 2,226-column concat.
        keys = [name for name in ('ticker', 'datetime', 'interval')
                if name in enriched_data.columns and name in targets_df.columns]
        aligned = len(enriched_data) == len(targets_df)
        if aligned and keys:
            for key in keys:
                left = enriched_data[key].to_numpy()
                right = targets_df[key].to_numpy()
                if not (left == right).all():
                    aligned = False
                    self.logger.error(
                        "Features and targets disagree on '%s': positional "
                        "concat would pair different bars. Merging on %s "
                        "instead.", key, keys,
                    )
                    break

        if aligned:
            merged_df = pd.concat(
                [enriched_data.reset_index(drop=True),
                 target_cols_only.reset_index(drop=True)], axis=1,
            )
        elif keys:
            merged_df = enriched_data.merge(
                targets_df[[*keys, *target_cols_only.columns]], on=keys, how='left',
            )
        else:
            raise ValueError(
                f"Cannot align features ({enriched_data.shape}) with targets "
                f"({targets_df.shape}): row counts differ and no shared key "
                f"columns exist to merge on. Concatenating positionally here "
                f"would pair each bar's features with another bar's outcome."
            )
        self.logger.info("Merged %s shape: %s", label, merged_df.shape)
        return merged_df

    def _release_database_cache(self, stage_name: str) -> None:
        """Shrink DuckDB's buffer pool once collection and processing are done.

        The connection is opened with `max_memory: 2GB` and fills that pool
        during collection -- 596k Wikipedia rows, 442k market rows, 141k from
        FRED. Feature engineering then never touches the database: there is no
        `db_manager`, no connection and no `duckdb` reference anywhere in
        stage 3. So two gigabytes of page cache were being carried through the
        longest stage in the pipeline by code that cannot use them.

        That was measured on 2026-08-24 and written down, and not acted on --
        the reasoning being that changing database behaviour late in a long
        session was careless. It then cost three consecutive failures. The
        third made the size of it plain: stage 3 entered `combine timeframes`
        holding 7.09 GiB with 0.27 GiB free, and the three frames it actually
        needed were 3.28 GiB of that. It was not short of memory; it was
        carrying memory it had no use for.

        Lowering the limit rather than closing the connection is deliberate.
        DuckDB evicts buffer-managed blocks to respect a new limit, so the
        pool shrinks immediately, while the connection stays open and valid
        for anything later that does want it -- and nothing has to know
        whether it was closed. Collection keeps its full 2 GB, because this
        runs after collection is finished.

        Failure here is logged and swallowed. A pipeline that dies while
        trying to free memory would be a worse bargain than one that keeps it.
        """
        if stage_name != 'ProcessingStage':
            return
        try:
            from src.data.management.data_manager import DataManager
            connections = getattr(DataManager, '_connections', None) or {}
            if not connections:
                return
            for name, connection in list(connections.items()):
                connection.execute("PRAGMA memory_limit='256MB'")
                self.logger.info(
                    "Shrank the DuckDB buffer pool for %s to 256MB after "
                    "%s; nothing downstream queries the database.",
                    name, stage_name,
                )
        except Exception as error:      # noqa: BLE001 - see the docstring
            self.logger.warning(
                "Could not shrink the DuckDB buffer pool (%s); "
                "the run continues carrying it.", error,
            )

    def _release_raw_data(self, stage_name: str, stage_outputs: dict) -> None:
        """Drop the raw tables once processing has consumed them.

        `stage_outputs['raw_data']` is every collected table, and NOTHING after
        stage 2 reads it -- verified across the whole of src/, where every other
        mention of the name is a local variable inside a collector. It is also
        already on disk: `main_database_stage1_raw_data_*.parquet`, 360 MiB
        compressed. So the in-memory copy was carried through stage 3's two and
        a quarter hours for nothing.

        Why this matters beyond tidiness. Stage 3 peaks at 2.67 GiB and 2.04 of
        that is held BEFORE its first phase -- left behind by collection and
        processing. Enrichment itself costs about 0.13 GiB. The part that scales
        with tickers is this one, so it is what stands between 22 names and 110.

        Released only after ProcessingStage has produced cleaned_data, because
        processing is the one thing that reads it. A run that skips stage 2
        keeps it.
        """
        if stage_name != 'ProcessingStage':
            return
        if 'raw_data' not in stage_outputs:
            return
        if not stage_outputs.get('cleaned_data'):
            # Processing ran and produced nothing usable; keeping the raw
            # tables is the only way anything downstream could recover.
            self.logger.warning(
                'Processing produced no cleaned_data; keeping raw_data.'
            )
            return

        raw = stage_outputs.pop('raw_data')
        try:
            import pandas as pd

            held = sum(
                frame.memory_usage(deep=False).sum()
                for frame in (raw.values() if isinstance(raw, dict) else [raw])
                if isinstance(frame, pd.DataFrame)
            ) / 2 ** 30
            self.logger.info(
                'Released raw_data after processing: %.2f GiB, read by nothing '
                'downstream and already on disk.', held,
            )
        except Exception:  # noqa: BLE001 - never let bookkeeping end a run
            self.logger.info('Released raw_data after processing.')

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
                self._release_raw_data(stage_name, stage_outputs)
                self._release_database_cache(stage_name)
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
                    # A DEBUG LINE MUST NOT BE ABLE TO KILL THE RUN.
                    #
                    # `_load_prepared_batch` deliberately returns a dict keyed
                    # by timeframe -- `iter_model_contexts` has always taken
                    # one, and keeping the slices apart is what stopped the
                    # union from costing 11 GiB. This line assumed a frame,
                    # called `.shape` on the dict, and the AttributeError
                    # propagated out of `_execute_stage` as
                    #
                    #   RuntimeError: Stage ModelingStage execution failed:
                    #   'dict' object has no attribute 'shape'
                    #
                    # Measured 2026-08-29: stage 4 never started. Not one row
                    # was trained on, and the cause was a log message about
                    # the data rather than anything done to it.
                    if isinstance(enriched_data, dict):
                        for key, frame in enriched_data.items():
                            shape = getattr(frame, 'shape', None)
                            columns = getattr(frame, 'columns', [])
                            targets = sum(1 for c in columns
                                          if str(c).startswith('target_'))
                            self.logger.info(
                                "ModelingStage receiving '%s': shape %s, "
                                "%d target column(s)", key, shape, targets,
                            )
                    else:
                        self.logger.info(
                            "ModelingStage receiving enriched_data with shape: %s",
                            getattr(enriched_data, 'shape', 'unknown'),
                        )
                        target_cols = [c for c in getattr(enriched_data, 'columns', [])
                                       if str(c).startswith('target_')]
                        self.logger.info(
                            "ModelingStage enriched_data target columns: %d",
                            len(target_cols),
                        )
            return await stage.run(**stage_outputs)

    # Stages whose output the rest of the pipeline cannot do without. When one
    # of these produces nothing, that is a failure, not an empty success.
    #
    # Deliberately excluded:
    #   Stage0Setup            - creates directories, has no data output at all
    #   PredictionStage        - an empty prediction set is a documented,
    #                            expected outcome (the champion filter can drop
    #                            every (ticker, target) group)
    #   TradingExecutionStage  - returns a structured result with an explicit
    #     EvaluationStage        status even when there is nothing to act on
    _STAGES_REQUIRING_OUTPUT = frozenset({
        'CollectionStage',
        'ProcessingStage',
        'FeatureEngineeringStage',
        'ModelingStage',
    })

    def _validate_stage_output(self, stage_name: str, stage_output:
        dict[str, Any] | None) ->dict[str, Any] | None:
        """Validate stage output against schema."""
        if not stage_output:
            if stage_name in self._STAGES_REQUIRING_OUTPUT:
                # These stages used to return {} on their own abort paths --
                # ModelingStage does it after logging "Enriched data not found.
                # Skipping Modeling Stage." -- and _execute_stage then reported
                # {'status': 'success'} and carried on with the previous
                # stage's outputs. A run that trained nothing still ended with
                # "Pipeline execution completed successfully".
                raise DataProcessingError(
                    f"Stage {stage_name} produced no output; the pipeline "
                    f"cannot continue on the previous stage's data. Check that "
                    f"stage's log for the reason it aborted."
                )
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
