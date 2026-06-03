"""
Pipeline execution utilities for hybrid pipeline.
"""
import time
import functools
import logging
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
        local_results = await orchestrator.run_local_pipeline(tickers=tickers,
            timeframes=timeframes)
        results_data = local_results.get('results', {})
        features_df = results_data.get('features_df', pd.DataFrame())
        targets_df = results_data.get('targets_df', pd.DataFrame())
        logger.info(f'Local pipeline complete: features={features_df.shape}, targets={targets_df.shape}')
        return await orchestrator.prepare_colab_data(tickers=tickers,
            timeframes=timeframes, features_df=features_df,
            targets_df=targets_df, **kwargs)

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
        logger.info('Running continue mode...')
        
        # 1. Contract validation
        val_report = PipelineExecutor._validate_batch_contract(orchestrator)
        if not val_report['valid']:
            return {'status': 'failed', 'reason': 'contract_validation_failed', 'errors': val_report['errors']}
            
        manifest = val_report['manifest']
        PipelineExecutor._log_manifest_details(manifest)
        
        # 2. Load data
        continue_data = PipelineExecutor._load_continue_data(orchestrator, args)
        (features_df, targets_df, colab_results, news_data, economic_data) = continue_data
        
        # 3. Validation
        validation_error = PipelineExecutor._validate_continue_inputs(features_df, targets_df, colab_results, args.batch_name)
        if validation_error:
            return validation_error
            
        # 4. Resolve tickers
        tickers = PipelineExecutor._resolve_tickers(args, colab_results, features_df)
        
        # 5. Local light training
        light_results = await orchestrator.run_light_models(
            features_df=features_df,
            targets_df=targets_df,
            tickers=tickers, 
            test_ticker=getattr(args, 'test_ticker', None),
            test_target=getattr(args, 'test_target', None),
            batch_name=args.batch_name
        )
        
        # 6. Final stages
        logger.info('Running final stages...')
        final_request = {
            'features_df': features_df,
            'targets_df': targets_df,
            'colab_results': colab_results,
            'light_results': light_results,
            'tickers': tickers,
            'timeframes': manifest.get('timeframes', ['15m', '60m', '1d']),
            'batch_name': args.batch_name,
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
        """Logs details from the batch manifest."""
        logger.info('✨ Explicit local-Colab contract verified successfully! Manifest details:')
        logger.info(f"   - Batch Name: {manifest.get('batch_name')}")
        logger.info(f"   - Created At: {manifest.get('timestamp')}")
        logger.info(f"   - Tickers: {manifest.get('tickers')}")
        logger.info(f"   - Timeframes: {manifest.get('timeframes')}")

    @staticmethod
    def _load_continue_data(orchestrator, args):
        """
        Orchestrates loading of all data required for continue mode.
        """
        # 1. Load core components (Colab results, Features, Targets)
        features_df, targets_df, colab_results = PipelineExecutor._load_core_continue_data(orchestrator, args)
        
        if PipelineExecutor._is_error_result(colab_results):
            logger.error(f"No valid Colab results found for batch: {args.batch_name}")
            return None, None, colab_results, None, None
            
        # 2. Load auxiliary data (News, Economic)
        news_data, economic_data = PipelineExecutor._load_extra_continue_data(orchestrator, args)
        
        return features_df, targets_df, colab_results, news_data, economic_data

    @staticmethod
    def _load_core_continue_data(orchestrator, args):
        """Loads Colab results, features and targets dataframes."""
        batch_dir = orchestrator.config.output_dir
        colab_results = orchestrator.load_colab_results(args.batch_name)
        
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
        
        # Try batch directory first
        news_data = PipelineExecutor._safe_load_parquet(batch_dir / 'news_data.parquet', "News (Batch)", silent=True)
        economic_data = PipelineExecutor._safe_load_parquet(batch_dir / 'economic_data.parquet', "Economic (Batch)", silent=True)
        
        # Try persistent fallbacks
        if news_data is None:
            news_data = PipelineExecutor._safe_load_parquet(Path('data/processed/features/news_data.parquet'), "News (Persistent)", silent=True)
            
        if economic_data is None:
            economic_data = PipelineExecutor._safe_load_parquet(Path('data/processed/features/macro_data.parquet'), "Macro (Persistent)", silent=True)
            if economic_data is None:
                economic_data = PipelineExecutor._safe_load_parquet(Path('data/processed/features/economic_data.parquet'), "Economic (Persistent)", silent=True)

        # Reconstruct from DB if still missing
        if news_data is None or economic_data is None:
            news_data, economic_data = PipelineExecutor._reconstruct_data_from_db(orchestrator, news_data, economic_data)
            
        return news_data, economic_data

    @staticmethod
    def _reconstruct_data_from_db(orchestrator, current_news, current_econ):
        """Reconstructs missing news/economic data from database tables."""
        try:
            from src.data.management.data_manager import DataManager
            from src.processing.deduplication_utils import deduplicate_dataframe
            
            db_manager = DataManager(orchestrator.config_manager)
            table_names = db_manager.get_all_table_names()
            collector_configs = orchestrator.config_manager.get_config('collectors', {})
            
            news_dfs, macro_dfs = [], []
            skipped_tables = {'cache_metadata', 'huggingface_data', 'enriched_features', 'experience_diary', 'market_data'}
            
            for table_name in table_names:
                if table_name in skipped_tables:
                    continue
                    
                df = db_manager.fetch_data_from_table(table_name)
                if df is None or df.empty:
                    continue
                    
                # Identify data type
                data_type = PipelineExecutor._identify_table_data_type(table_name, collector_configs)
                
                if data_type == 'news':
                    news_dfs.append(df)
                elif data_type == 'macro':
                    macro_dfs.append(df)
            
            # Final reconstruction
            reconstructed_news = current_news
            if current_news is None and news_dfs:
                reconstructed_news, _ = deduplicate_dataframe(pd.concat(news_dfs, ignore_index=True))
                logger.info(f"✅ Reconstructed news data from DB: {reconstructed_news.shape}")
                
            reconstructed_econ = current_econ
            if current_econ is None and macro_dfs:
                reconstructed_econ, _ = deduplicate_dataframe(pd.concat(macro_dfs, ignore_index=True))
                logger.info(f"✅ Reconstructed economic data from DB: {reconstructed_econ.shape}")
                
            return reconstructed_news, reconstructed_econ
            
        except Exception as ex:
            logger.error(f"⚠️ Critical failure reconstructing data from database: {ex}", exc_info=True)
            raise

    @staticmethod
    def _identify_table_data_type(table_name: str, collector_configs: dict) -> str:
        """Identifies if a table contains news or macro data."""
        # Check by config
        for config in collector_configs.values():
            if config.get('table_name') == table_name:
                dt = config.get('data_type')
                if dt == 'news': return 'news'
                if dt == 'macro_data': return 'macro'
        
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
            except Exception as e:
                logger.warning(f"Failed to load {label} from {path}: {e}")
        elif not silent:
            logger.error(f"{label} file not found: {path}")
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
        if PipelineExecutor._is_error_result(colab_results):
            logger.error(
                f"Cannot continue batch '{batch_name}': Colab results are missing or invalid."
                )
            return {'status': 'failed', 'reason': 'missing_colab_results'}
        if features_df is None or getattr(features_df, 'empty', True):
            logger.error(
                f"Cannot continue batch '{batch_name}': features.parquet is missing or empty."
                )
            return {'status': 'failed', 'reason': 'missing_features'}
        if targets_df is None or getattr(targets_df, 'empty', True):
            logger.error(
                f"Cannot continue batch '{batch_name}': targets.parquet is missing or empty."
                )
            return {'status': 'failed', 'reason': 'missing_targets'}
        target_cols = [col for col in targets_df.columns if str(col).
            startswith('target_')]
        if not target_cols:
            logger.error(
                f"Cannot continue batch '{batch_name}': targets.parquet has no target_* columns."
                )
            return {'status': 'failed', 'reason': 'missing_target_columns'}
        return None

    @staticmethod
    def _resolve_tickers(args, colab_results, features_df):
        """Resolve tickers for continue mode."""
        tickers = [args.test_ticker] if args.test_ticker else list(
            colab_results.get('ticker_results', {}).keys())
        if not tickers and features_df is not None:
            if isinstance(features_df.index, pd.MultiIndex):
                tickers = list(features_df.index.get_level_values('ticker')
                    .unique())
            elif 'ticker' in features_df.columns:
                tickers = list(features_df['ticker'].unique())
        logger.info(f'Resolved tickers for continue mode: {tickers}')
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
    def resolve_tickers_and_timeframes(args, config_manager) ->tuple[list, list
        ]:
        """Resolve tickers and timeframes from args or config."""
        if args.tickers is not None:
            tickers = args.tickers
            logger.info(f'Using explicitly provided tickers: {tickers}')
        else:
            assets_config = config_manager.get_config('assets') or {}
            sectors = assets_config.get('sectors', {})
            all_tickers = set()
            for sector_name, sector_config in sectors.items():
                sector_assets = sector_config.get('assets', [])
                all_tickers.update(sector_assets)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Sector '{sector_name}': {len(sector_assets)} tickers: {sector_assets}"
                        )
            tickers = sorted(all_tickers)
            logger.info(
                f'Loaded {len(tickers)} unique tickers from all sectors: {tickers}'
                )
            logger.info(f'Processed sectors: {list(sectors.keys())}')
        if args.test_ticker:
            tickers = [args.test_ticker]
            logger.info(f'Using test ticker: {args.test_ticker}')
        collectors = config_manager.get_config('collectors') or {}
        yf_timeframes = collectors.get('yahoo_finance', {}).get('timeframes',
            {})
        timeframes = list(yf_timeframes.keys()) if yf_timeframes else ['15m',
            '60m', '1d']
        logger.info(f'Final tickers: {tickers}')
        logger.info(f'Using timeframes: {timeframes}')
        return tickers, timeframes

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
            logger.info(f'   Ticker: {args.test_ticker}')
        if args.test_target:
            logger.info(f'   Target: {args.test_target}')
        if args.test_model:
            logger.info(f'   Model: {args.test_model}')
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
            logger.info(
                f"   Best {results['metric']}: {results['best_value']:.4f}")
            logger.info(f"   Best hyperparameters: {results['best_params']}")
        else:
            logger.error(f"Calibration failed: {results.get('reason')}")
        return results
