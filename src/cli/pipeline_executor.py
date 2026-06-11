"""
Pipeline execution utilities for hybrid pipeline.
"""
import functools
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

def profile_execution(func):
    """Decorator to log execution time of async functions."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"⏱️ {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

FEATURES_FILE = 'features.parquet'
TARGETS_FILE = 'targets.parquet'


class PipelineExecutor:
    """Handles pipeline execution for different modes."""

    @staticmethod
    @profile_execution
    async def execute_local_mode(orchestrator, tickers: list, timeframes: list
        ):
        """Execute local pipeline stages only."""
        logger.info('Running local pipeline (stages 0-3)...')
        return await orchestrator.run_local_pipeline(tickers=tickers,
            timeframes=timeframes)

    @staticmethod
    @profile_execution
    async def execute_light_mode(orchestrator, tickers: list, timeframes: list
        ):
        """Execute light models training only."""
        logger.info('Running light models training...')
        return await orchestrator.run_light_models(tickers=tickers)

    @staticmethod
    @profile_execution
    async def execute_prepare_mode(orchestrator, tickers: list, timeframes:
        list, **kwargs):
        """Execute preparation for Colab (stages 0-3 + packaging)."""
        logger.info('Preparing data for Colab training...')

        tracker = PipelineExecutor._enable_lineage_tracking_for_run()

        features_df, targets_df = await PipelineExecutor._run_local_pipeline_and_extract_data(
            orchestrator, tickers, timeframes
        )

        PipelineExecutor._capture_final_features(tracker, features_df)

        result = await orchestrator.prepare_colab_data(tickers=tickers,
            timeframes=timeframes, features_df=features_df,
            targets_df=targets_df, **kwargs)

        PipelineExecutor._mark_model_input(tracker, features_df)

        PipelineExecutor._disable_lineage_tracking()
        return result

    @staticmethod
    def _enable_lineage_tracking_for_run():
        """Enable lineage tracking for Colab preparation run."""
        from src.features.feature_orchestrator import enable_lineage_tracking
        return enable_lineage_tracking("diagnostic_reports/feature_lineage_report.json")

    @staticmethod
    def _disable_lineage_tracking():
        """Disable lineage tracking and save report."""
        from src.features.feature_orchestrator import disable_lineage_tracking
        disable_lineage_tracking()

    @staticmethod
    async def _run_local_pipeline_and_extract_data(orchestrator, tickers: list, timeframes: list):
        """Run local pipeline and extract features/targets dataframes."""
        local_results = await orchestrator.run_local_pipeline(tickers=tickers,
            timeframes=timeframes)
        results_data = local_results.get('results', {})
        features_df = results_data.get('features_df', pd.DataFrame())
        targets_df = results_data.get('targets_df', pd.DataFrame())
        logger.info(f'Local pipeline complete: features={features_df.shape}, targets={targets_df.shape}')
        return features_df, targets_df

    @staticmethod
    def _capture_final_features(tracker, features_df):
        """Captures final features for lineage report."""
        if tracker is not None and not features_df.empty:
            try:
                tracker.capture_step("final_features", features_df)
                tracker.mark_model_input(features_df)
            except Exception as e:
                logger.warning(f"[Lineage] Could not capture step: {e}")

    @staticmethod
    def _mark_model_input(tracker, features_df):
        """Marks final features as model input."""
        if tracker is not None:
            try:
                final_features = features_df
                if final_features.empty:
                    final_features = PipelineExecutor._load_features_from_parquet()

                if not final_features.empty:
                    tracker.capture_step("model_input", final_features)
                    tracker.mark_model_input(final_features)
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.debug(f"[Lineage] Could not mark model input: {e}")

    @staticmethod
    def _load_features_from_parquet():
        """Loads features from Parquet if features_df is empty."""
        features_path = Path("data/processed/features/features.parquet")
        if not features_path.exists():
            colab_dirs = sorted(Path("data/colab/accumulated").glob("*/features.parquet"))
            if colab_dirs:
                features_path = colab_dirs[-1]
        
        if features_path.exists():
            final_features = pd.read_parquet(features_path)
            logger.info(f"[Lineage] Loaded features from Parquet: {final_features.shape}")
            return final_features
        return pd.DataFrame()

    @staticmethod
    @profile_execution
    async def execute_full_mode(orchestrator, tickers: list, timeframes: list):
        """Execute the full hybrid preparation flow and pause for Colab."""
        logger.info('Running full hybrid pipeline...')
        from src.pipeline.hybrid_orchestrator import HybridPipelineRequest
        return await orchestrator.run_full_hybrid_pipeline(
            HybridPipelineRequest(tickers=tickers, timeframes=timeframes,
            accumulate=True))

    @staticmethod
    @profile_execution
    async def execute_continue_mode(orchestrator, args):
        """Execute the continue mode after Colab results are ready."""
        batch_name_sanitized = PipelineExecutor._sanitize(getattr(args, 'batch_name', 'unknown'))
        logger.info(f"Running continue mode for batch: {batch_name_sanitized}...")

        # 1. Contract validation
        val_report = PipelineExecutor._validate_batch_contract(orchestrator)
        if not val_report['valid']:
            return {'status': 'failed', 'reason': 'contract_validation_failed', 'errors': val_report['errors']}

        manifest = val_report['manifest']
        PipelineExecutor._log_manifest_details(manifest)

        # 2. Load and validate data
        continue_data = PipelineExecutor._load_continue_data(orchestrator, args)
        (features_df, targets_df, colab_results, news_data, economic_data) = continue_data

        validation_error = PipelineExecutor._validate_continue_inputs(
            features_df, targets_df, colab_results, getattr(args, 'batch_name', 'unknown')
        )
        if validation_error:
            return validation_error

        # 3. Resolve tickers and run light training
        tickers = PipelineExecutor._resolve_tickers(args, colab_results, features_df)
        light_results = await PipelineExecutor._run_light_training_for_continue(
            orchestrator, features_df, targets_df, tickers, args
        )

        # 4. Run final stages
        return await PipelineExecutor._run_final_stages_for_continue(
            orchestrator, features_df, targets_df, colab_results, light_results,
            tickers, manifest, news_data, economic_data, args
        )

    @staticmethod
    async def _run_light_training_for_continue(orchestrator, features_df, targets_df, tickers, args):
        """Run light models training for continue mode."""
        return await orchestrator.run_light_models(
            features_df=features_df,
            targets_df=targets_df,
            tickers=tickers,
            test_ticker=getattr(args, 'test_ticker', None),
            test_target=getattr(args, 'test_target', None),
            batch_name=getattr(args, 'batch_name', None)
        )

    @staticmethod
    async def _run_final_stages_for_continue(orchestrator, features_df, targets_df, colab_results, light_results, tickers, manifest, news_data, economic_data, args):
        """Run final stages for continue mode."""
        logger.info('Running final stages...')
        final_request = {
            'features_df': features_df,
            'targets_df': targets_df,
            'colab_results': colab_results,
            'light_results': light_results,
            'tickers': tickers,
            'timeframes': manifest.get('timeframes', ['15m', '60m', '1d']),
            'batch_name': getattr(args, 'batch_name', None),
            'news_data': news_data,
            'economic_data': economic_data,
            'stages_to_run': getattr(args, 'stages', None)
        }
        return await orchestrator.run_final_stages(final_request)

    @staticmethod
    def _validate_batch_contract(orchestrator) -> dict:
        """Validates the batch directory contract."""
        from src.validation.pipeline_schemas import validate_batch_dir
        return validate_batch_dir(orchestrator.config.output_dir)

    @staticmethod
    def _log_manifest_details(manifest: dict):
        """Logs details from the batch manifest with sanitization."""
        logger.info('✨ Explicit local-Colab contract verified successfully! Manifest details:')
        
        # CWE-117: Sanitize user-controlled data before logging
        def sanitize(val):
            if val is None: return "None"
            return str(val).replace('\r', '\\r').replace('\n', '\\n')

        logger.info(f"   - Batch Name: {sanitize(manifest.get('batch_name'))}")
        logger.info(f"   - Created At: {sanitize(manifest.get('timestamp'))}")
        logger.info(f"   - Tickers: {sanitize(manifest.get('tickers'))}")
        logger.info(f"   - Timeframes: {sanitize(manifest.get('timeframes'))}")

    @staticmethod
    def _sanitize(val: Any) -> str:
        """Utility to sanitize values for logging to prevent CRLF injection."""
        if val is None:
            return "None"
        return str(val).replace('\r', '\\r').replace('\n', '\\n')

    @staticmethod
    def _load_continue_data(orchestrator, args):
        """
        Orchestrates loading of all data required for continue mode.
        """
        features_df, targets_df, colab_results = PipelineExecutor._load_core_continue_data(orchestrator, args)

        if PipelineExecutor._is_error_result(colab_results):
            return PipelineExecutor._return_error_for_invalid_colab_results(args, colab_results)

        news_data, economic_data = PipelineExecutor._load_extra_continue_data(orchestrator, args)
        return features_df, targets_df, colab_results, news_data, economic_data

    @staticmethod
    def _return_error_for_invalid_colab_results(args, colab_results):
        """Return error tuple when Colab results are invalid."""
        batch_name = PipelineExecutor._sanitize(getattr(args, 'batch_name', 'unknown'))
        logger.error(f"No valid Colab results found for batch: {batch_name}")
        return None, None, colab_results, None, None

    @staticmethod
    def _load_core_continue_data(orchestrator, args):
        """Loads Colab results, features and targets dataframes."""
        batch_dir = orchestrator.config.output_dir
        batch_name = getattr(args, 'batch_name', 'unknown')
        colab_results = orchestrator.load_colab_results(batch_name)

        if PipelineExecutor._is_error_result(colab_results):
            return None, None, colab_results

        features_path = batch_dir / FEATURES_FILE
        targets_path = batch_dir / TARGETS_FILE

        features_df = PipelineExecutor._safe_load_parquet(features_path, "Features")
        targets_df = PipelineExecutor._safe_load_parquet(targets_path, "Targets")

        return features_df, targets_df, colab_results

    @staticmethod
    def _load_extra_continue_data(orchestrator, args):
        """Loads or reconstructs news and economic data."""
        batch_dir = orchestrator.config.output_dir

        news_data = PipelineExecutor._load_news_data_with_fallbacks(batch_dir)
        economic_data = PipelineExecutor._load_economic_data_with_fallbacks(batch_dir)

        if news_data is None or economic_data is None:
            news_data, economic_data = PipelineExecutor._reconstruct_data_from_db(orchestrator, news_data, economic_data)

        return news_data, economic_data

    @staticmethod
    def _load_news_data_with_fallbacks(batch_dir):
        """Load news data with batch directory and persistent fallbacks."""
        news_data = PipelineExecutor._safe_load_parquet(batch_dir / 'news_data.parquet', "News (Batch)", silent=True)
        if news_data is None:
            news_data = PipelineExecutor._safe_load_parquet(Path('data/processed/features/news_data.parquet'), "News (Persistent)", silent=True)
        return news_data

    @staticmethod
    def _load_economic_data_with_fallbacks(batch_dir):
        """Load economic data with batch directory and persistent fallbacks."""
        economic_data = PipelineExecutor._safe_load_parquet(batch_dir / 'economic_data.parquet', "Economic (Batch)", silent=True)
        if economic_data is None:
            economic_data = PipelineExecutor._safe_load_parquet(Path('data/processed/features/macro_data.parquet'), "Macro (Persistent)", silent=True)
            if economic_data is None:
                economic_data = PipelineExecutor._safe_load_parquet(Path('data/processed/features/economic_data.parquet'), "Economic (Persistent)", silent=True)
        return economic_data

    @staticmethod
    def _reconstruct_data_from_db(orchestrator, current_news, current_econ):
        """Reconstructs missing news/economic data from database tables."""
        try:
            from src.data.management.data_manager import DataManager
            from src.processing.deduplication_utils import deduplicate_dataframe

            db_manager, collector_configs = PipelineExecutor._initialize_db_reconstruction(orchestrator)
            table_names = db_manager.get_all_table_names()

            news_dfs, macro_dfs = PipelineExecutor._process_tables(db_manager, table_names, collector_configs)

            reconstructed_news = PipelineExecutor._reconstruct_category(current_news, news_dfs, "news", deduplicate_dataframe)
            reconstructed_econ = PipelineExecutor._reconstruct_category(current_econ, macro_dfs, "economic", deduplicate_dataframe)

            return reconstructed_news, reconstructed_econ

        except (pd.errors.EmptyDataError, ValueError, KeyError, ImportError) as ex:
            logger.exception(f"⚠️ Failure reconstructing data from database: {ex}")
            raise

    @staticmethod
    def _initialize_db_reconstruction(orchestrator):
        """Initialize database manager and collector configs for reconstruction."""
        from src.data.management.data_manager import DataManager
        db_manager = DataManager(orchestrator.config_manager)
        collector_configs = orchestrator.config_manager.get_config('collectors', {})
        return db_manager, collector_configs

    @staticmethod
    def _process_tables(db_manager, table_names, collector_configs):
        """Processes all database tables and categorizes them."""
        news_dfs, macro_dfs = [], []
        skipped_tables = {'cache_metadata', 'huggingface_data', 'enriched_features', 'experience_diary', 'market_data'}

        for table_name in table_names:
            if table_name in skipped_tables:
                continue

            df = db_manager.fetch_data_from_table(table_name)
            if df is None or df.empty:
                continue

            data_type = PipelineExecutor._identify_table_data_type(table_name, collector_configs)
            news_dfs, macro_dfs = PipelineExecutor._categorize_dataframe_by_type(df, data_type, news_dfs, macro_dfs)

        return news_dfs, macro_dfs

    @staticmethod
    def _categorize_dataframe_by_type(df, data_type, news_dfs, macro_dfs):
        """Categorize dataframe by data type and append to appropriate list."""
        if data_type == 'news':
            news_dfs.append(df)
        elif data_type == 'macro':
            macro_dfs.append(df)
        return news_dfs, macro_dfs

    @staticmethod
    def _reconstruct_category(current_data, dfs, label, deduplicate_func):
        """Reconstructs a category of data."""
        if current_data is None and dfs:
            reconstructed, _ = deduplicate_func(pd.concat(dfs, ignore_index=True), subset_cols=['timestamp'])
            logger.info(f"✅ Reconstructed {label} data from DB: {reconstructed.shape}")
            return reconstructed
        return current_data

    @staticmethod
    def _identify_table_data_type(table_name: str, collector_configs: dict) -> str:
        """Identifies if a table contains news or macro data."""
        # Check by config
        for config in collector_configs.values():
            if config.get('table_name') == table_name:
                dt = config.get('data_type')
                if dt == 'news':
                    return 'news'
                if dt == 'macro_data':
                    return 'macro'

        # Check by name
        name_lower = table_name.lower()
        if 'fred' in name_lower or 'macro' in name_lower:
            return 'macro'
        return 'unknown'

    @staticmethod
    def _safe_load_parquet(path: Path, label: str, silent: bool = False) -> Any:
        """Safely loads a parquet file, logging success or failure."""
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if not silent:
                    logger.info(f"Loaded {label}: {df.shape}")
                return df
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.exception(f"Failed to load {label} from {path}: {e}")
        elif not silent:
            label_sanitized = PipelineExecutor._sanitize(label)
            logger.error(f"{label_sanitized} file not found: {path}")
        return None

    @staticmethod
    def _is_error_result(result: Any) ->bool:
        """Return True when a loader result represents a missing or failed artifact."""
        if not result:
            return True
        if isinstance(result, dict):
            return bool(result.get('error') or result.get('status') == 'error')
        return False

    @staticmethod
    def _validate_continue_inputs(features_df, targets_df, colab_results,
        batch_name: str):
        """Validate continue-mode inputs before starting local training."""
        bn_sanitized = PipelineExecutor._sanitize(batch_name)

        if PipelineExecutor._is_error_result(colab_results):
            return PipelineExecutor._return_validation_error(bn_sanitized, 'missing_colab_results', 'Colab results are missing or invalid')

        if features_df is None or getattr(features_df, 'empty', True):
            return PipelineExecutor._return_validation_error(bn_sanitized, 'missing_features', 'features.parquet is missing or empty')

        if targets_df is None or getattr(targets_df, 'empty', True):
            return PipelineExecutor._return_validation_error(bn_sanitized, 'missing_targets', 'targets.parquet is missing or empty')

        target_cols = PipelineExecutor._extract_target_columns(targets_df)
        if not target_cols:
            return PipelineExecutor._return_validation_error(batch_name, 'missing_target_columns', 'targets.parquet has no target_* columns')

        return None

    @staticmethod
    def _return_validation_error(batch_name, reason, message):
        """Return validation error dictionary with logging."""
        logger.error(f"Cannot continue batch '{batch_name}': {message}")
        return {'status': 'failed', 'reason': reason}

    @staticmethod
    def _extract_target_columns(targets_df):
        """Extract target columns from targets dataframe."""
        # audit-ignore: ARCHITECTURAL_USAGE
        return [col for col in targets_df.columns if str(col).startswith('target_')]

    @staticmethod
    def _resolve_tickers(args, colab_results, features_df):
        """Resolve tickers for continue mode."""
        tickers = PipelineExecutor._get_tickers_from_args_or_colab(args, colab_results)
        tickers = PipelineExecutor._fallback_to_features_tickers(tickers, features_df)

        tickers_sanitized = PipelineExecutor._sanitize(tickers)
        logger.info(f'Resolved tickers for continue mode: {tickers_sanitized}')
        return tickers

    @staticmethod
    def _get_tickers_from_args_or_colab(args, colab_results):
        """Get tickers from args or colab results."""
        if args.test_ticker:
            return [args.test_ticker]
        return list(colab_results.get('ticker_results', {}).keys())

    @staticmethod
    def _fallback_to_features_tickers(tickers, features_df):
        """Fallback to features dataframe tickers if no tickers found."""
        if tickers or features_df is None:
            return tickers

        if isinstance(features_df.index, pd.MultiIndex):
            return list(features_df.index.get_level_values('ticker').unique())
        elif 'ticker' in features_df.columns:
            return list(features_df['ticker'].unique())
        return tickers

    @staticmethod
    def _merge_results_data(colab_results, light_results):
        """Merge colab results with light models results."""
        merged_results = dict(colab_results)
        if light_results.get('models_metadata'):
            merged_results.setdefault('models_metadata', {}).update(
                light_results['models_metadata'])
        logger.info('Merged Colab results with light models')
        return merged_results

    @staticmethod
    def resolve_tickers_and_timeframes(args, config_manager) -> tuple[list, list]:
        """Resolve tickers and timeframes from args or config."""
        tickers = PipelineExecutor._get_tickers(args, config_manager)
        timeframes = PipelineExecutor._get_timeframes(config_manager)
        
        tickers_final_sanitized = PipelineExecutor._sanitize(tickers)
        timeframes_sanitized = PipelineExecutor._sanitize(timeframes)
        logger.info(f'Final tickers: {tickers_final_sanitized}')
        logger.info(f'Using timeframes: {timeframes_sanitized}')
        return tickers, timeframes

    @staticmethod
    def _get_tickers(args, config_manager) -> list:
        """Resolves tickers from arguments or config."""
        tickers = PipelineExecutor._get_tickers_from_args_or_config(args, config_manager)
        tickers = PipelineExecutor._apply_test_ticker_if_needed(args, tickers)
        return tickers

    @staticmethod
    def _get_tickers_from_args_or_config(args, config_manager):
        """Get tickers from args or config manager."""
        if args.tickers is not None:
            logger.info(f'Using explicitly provided tickers: {PipelineExecutor._sanitize(args.tickers)}')
            return args.tickers

        assets_config = config_manager.get_config('assets') or {}
        sectors = assets_config.get('sectors', {})
        return PipelineExecutor._load_tickers_from_sectors(sectors)

    @staticmethod
    def _apply_test_ticker_if_needed(args, tickers):
        """Apply test ticker if specified in args."""
        if args.test_ticker:
            logger.info(f'Using test ticker: {PipelineExecutor._sanitize(args.test_ticker)}')
            return [args.test_ticker]
        return tickers

    @staticmethod
    def _load_tickers_from_sectors(sectors: dict) -> list:
        """Loads tickers from sector configuration."""
        all_tickers = PipelineExecutor._collect_tickers_from_sectors(sectors)
        tickers = sorted(all_tickers)
        logger.info(f'Loaded {len(tickers)} unique tickers from {len(sectors)} sectors: {PipelineExecutor._sanitize(tickers)}')
        return tickers

    @staticmethod
    def _collect_tickers_from_sectors(sectors: dict):
        """Collect all tickers from sectors configuration."""
        all_tickers = set()
        for sector_name, sector_config in sectors.items():
            sector_assets = sector_config.get('assets', [])
            all_tickers.update(sector_assets)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Sector '{PipelineExecutor._sanitize(sector_name)}': {len(sector_assets)} tickers: {PipelineExecutor._sanitize(sector_assets)}")
        return all_tickers

    @staticmethod
    def _get_timeframes(config_manager) -> list:
        """Resolves timeframes from config."""
        collectors = config_manager.get_config('collectors') or {}
        yf_timeframes = collectors.get('yahoo_finance', {}).get('timeframes', {})
        return list(yf_timeframes.keys()) if yf_timeframes else ['15m', '60m', '1d']

    @staticmethod
    def log_test_mode_info(args) ->None:
        """Log test mode information."""
        if PipelineExecutor._is_test_mode_active(args):
            PipelineExecutor._log_test_mode_details(args)

    @staticmethod
    def _is_test_mode_active(args) ->bool:
        """Check if test mode is active."""
        return any([args.test_ticker, args.test_target, args.test_model])

    @staticmethod
    def _log_test_mode_details(args) ->None:
        """Log test mode details."""
        logger.info('TEST MODE ACTIVATED:')
        if args.test_ticker:
            tt_sanitized = PipelineExecutor._sanitize(args.test_ticker)
            logger.info(f'   Ticker: {tt_sanitized}')
        if args.test_target:
            ttg_sanitized = PipelineExecutor._sanitize(args.test_target)
            logger.info(f'   Target: {ttg_sanitized}')
        if args.test_model:
            tm_sanitized = PipelineExecutor._sanitize(args.test_model)
            logger.info(f'   Model: {tm_sanitized}')
        logger.info(f'   Iterations: {args.max_iterations}')

    @staticmethod
    async def execute_calibrate_mode(orchestrator, args):
        """Execute calibration mode for DEAN hyperparameter tuning."""
        logger.info('Running DEAN calibration...')
        n_trials = getattr(args, 'n_trials', 50)
        results = await orchestrator.run_calibration(test_ticker=getattr(
            args, 'test_ticker', None), test_target=getattr(args,
            'test_target', None), n_trials=n_trials)
        if results.get('status') == 'success':
            logger.info('Calibration successful!')
            metric_sanitized = PipelineExecutor._sanitize(results.get('metric'))
            logger.info(
                f"   Best {metric_sanitized}: {results['best_value']:.4f}")
            
            params_sanitized = PipelineExecutor._sanitize(results.get('best_params'))
            logger.info(f"   Best hyperparameters: {params_sanitized}")
        else:
            reason_sanitized = PipelineExecutor._sanitize(results.get('reason'))
            logger.error(f"Calibration failed: {reason_sanitized}")
        return results
