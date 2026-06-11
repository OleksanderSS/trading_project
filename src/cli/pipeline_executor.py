"""
Pipeline execution utilities for hybrid pipeline.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

# Constants for file names
FEATURES_FILE = "features.parquet"
TARGETS_FILE = "targets.parquet"


class PipelineExecutor:
    """Handles pipeline execution for different modes."""

    @staticmethod
    async def execute_local_mode(orchestrator, tickers: list, timeframes: list):
        """Execute local pipeline stages only."""
        logger.info("Running local pipeline (stages 0-3)...")
        return await orchestrator.run_local_pipeline(
            tickers=tickers,
            timeframes=timeframes,
        )

    @staticmethod
    async def execute_light_mode(orchestrator, tickers: list, timeframes: list):
        """Execute light models training only."""
        logger.info("Running light models training...")
        return await orchestrator.run_light_models(
            tickers=tickers,
        )

    @staticmethod
    async def execute_prepare_mode(orchestrator, tickers: list, timeframes: list, **kwargs):
        """Execute preparation for Colab."""
        logger.info("Preparing data for Colab training...")
        return await orchestrator.prepare_colab_data(
            tickers=tickers,
            timeframes=timeframes,
            **kwargs,
        )

    @staticmethod
    async def execute_full_mode(orchestrator, tickers: list, timeframes: list):
        """Execute the full hybrid preparation flow and pause for Colab."""
        logger.info("Running full hybrid pipeline...")
        from src.pipeline.hybrid_orchestrator import HybridPipelineRequest

        return await orchestrator.run_full_hybrid_pipeline(HybridPipelineRequest(
            tickers=tickers,
            timeframes=timeframes,
            accumulate=True,
        ))

    @staticmethod
    async def execute_continue_mode(orchestrator, args):
        """Execute continue mode."""
        logger.info("Running continue mode...")

        # Explicit Local-Colab Contract Validation
        from src.validation.pipeline_schemas import validate_batch_dir
        val_report = validate_batch_dir(orchestrator.config.output_dir)
        if not val_report["valid"]:
            logger.error(f"❌ Batch directory contract validation failed: {val_report['errors']}")
            return {'status': 'failed', 'reason': 'contract_validation_failed', 'errors': val_report['errors']}
        
        logger.info("✨ Explicit local-Colab contract verified successfully! Manifest details:")
        manifest = val_report["manifest"]
        logger.info(f"   - Batch Name: {manifest.get('batch_name')}")
        logger.info(f"   - Created At: {manifest.get('timestamp')}")
        logger.info(f"   - Tickers: {manifest.get('tickers')}")
        logger.info(f"   - Timeframes: {manifest.get('timeframes')}")

        continue_data = PipelineExecutor._load_continue_data(orchestrator, args)
        features_df, targets_df, colab_results, news_data, economic_data = continue_data

        validation_error = PipelineExecutor._validate_continue_inputs(
            features_df, targets_df, colab_results, args.batch_name
        )
        if validation_error:
            return validation_error

        tickers = PipelineExecutor._resolve_tickers(args, colab_results, features_df)

        light_results = await orchestrator.run_light_models(
            tickers=tickers,
            test_ticker=getattr(args, 'test_ticker', None),
            test_target=getattr(args, 'test_target', None),
            batch_name=args.batch_name,
        )

        logger.info("Running final stages...")
        final_request = {
            'features_df': features_df,
            'targets_df': targets_df,
            'colab_results': colab_results,
            'light_results': light_results,
            'tickers': tickers,
            'timeframes': orchestrator.config.timeframes,
            'batch_name': args.batch_name,
            'news_data': news_data,
            'economic_data': economic_data,
            'stages_to_run': getattr(args, 'stages', None),
        }

        return await orchestrator.run_final_stages(final_request)

    @staticmethod
    def _load_continue_data(orchestrator, args):
        """Load data for continue mode."""
        batch_dir = orchestrator.config.output_dir
        colab_results = orchestrator.load_colab_results(args.batch_name)

        if PipelineExecutor._is_error_result(colab_results):
            logger.error(f"No Colab results found for batch: {args.batch_name}")
            return None, None, colab_results, None, None

        features_path = batch_dir / FEATURES_FILE
        targets_path = batch_dir / TARGETS_FILE
        news_path = batch_dir / "news_data.parquet"
        econ_path = batch_dir / "economic_data.parquet"

        features_df = None
        targets_df = None
        news_data = None
        economic_data = None

        if features_path.exists():
            features_df = pd.read_parquet(features_path)
            logger.info(f"Loaded features: {features_df.shape}")
        else:
            logger.error(f"Features file not found: {features_path}")

        if targets_path.exists():
            targets_df = pd.read_parquet(targets_path)
            logger.info(f"Loaded targets: {targets_df.shape}")
        else:
            logger.error(f"Targets file not found: {targets_path}")

        # Triple-redundant loader for news_data and economic_data
        # 1. Check batch_dir
        if news_path.exists():
            news_data = pd.read_parquet(news_path)
            logger.info(f"Loaded news data from batch: {news_data.shape}")
        # 2. Check persistent_dir fallback
        else:
            persistent_news = Path("data/processed/features/news_data.parquet")
            if persistent_news.exists():
                news_data = pd.read_parquet(persistent_news)
                logger.info(f"Loaded news data from persistent fallback: {news_data.shape}")

        # 1. Check batch_dir
        if econ_path.exists():
            economic_data = pd.read_parquet(econ_path)
            logger.info(f"Loaded economic data from batch: {economic_data.shape}")
        # 2. Check persistent_dir fallbacks (macro_data or economic_data)
        else:
            persistent_econ = Path("data/processed/features/macro_data.parquet")
            if not persistent_econ.exists():
                persistent_econ = Path("data/processed/features/economic_data.parquet")
            if persistent_econ.exists():
                economic_data = pd.read_parquet(persistent_econ)
                logger.info(f"Loaded economic data from persistent fallback: {economic_data.shape}")

        # 3. DuckDB Database Fallback
        if news_data is None or economic_data is None:
            try:
                from src.data.management.data_manager import DataManager
                from src.processing.deduplication_utils import deduplicate_dataframe
                db_manager = DataManager(orchestrator.config_manager)
                table_names = db_manager.get_all_table_names()
                
                collector_configs = orchestrator.config_manager.get_config('collectors', {})
                all_news_dfs = []
                macro_dfs = []
                
                for table_name in table_names:
                    # Skip irrelevant/too large tables
                    if table_name in ['cache_metadata', 'huggingface_data', 'enriched_features', 'experience_diary', 'market_data']:
                        continue
                    
                    df = db_manager.fetch_data_from_table(table_name)
                    if df is None or df.empty:
                        continue
                        
                    collector_info = {}
                    for config in collector_configs.values():
                        if config.get('table_name') == table_name:
                            collector_info = config
                            break
                    if not collector_info:
                        collector_info = collector_configs.get(table_name, {})
                        
                    data_type = collector_info.get('data_type')
                    if data_type == 'news':
                        all_news_dfs.append(df)
                    elif 'fred' in table_name.lower() or 'macro' in table_name.lower() or data_type == 'macro_data':
                        macro_dfs.append(df)
                
                if news_data is None and all_news_dfs:
                    news_df = pd.concat(all_news_dfs, ignore_index=True)
                    hashable_cols = [col for col in news_df.columns
                                     if news_df[col].apply(lambda x: isinstance(x, (str, int, float, bool, type(None)))).all()]
                    news_df, _ = deduplicate_dataframe(news_df, hashable_cols)
                    news_data = news_df
                    logger.info(f"✅ Reconstructed news data from database fallback: {news_data.shape}")
                    
                if economic_data is None and macro_dfs:
                    econ_df = pd.concat(macro_dfs, ignore_index=True)
                    hashable_cols = [col for col in econ_df.columns
                                     if econ_df[col].apply(lambda x: isinstance(x, (str, int, float, bool, type(None)))).all()]
                    econ_df, _ = deduplicate_dataframe(econ_df, hashable_cols)
                    economic_data = econ_df
                    logger.info(f"✅ Reconstructed economic data from database fallback: {economic_data.shape}")
            except Exception as ex:
                logger.warning(f"⚠️ Failed to load news/macro fallback from database: {ex}")

        return features_df, targets_df, colab_results, news_data, economic_data

    @staticmethod
    def _is_error_result(result: Any) -> bool:
        """Return True when a loader result represents a missing or failed artifact."""
        if not result:
            return True
        if isinstance(result, dict):
            return bool(result.get('error') or result.get('status') == 'error')
        return False

    @staticmethod
    def _validate_continue_inputs(features_df, targets_df, colab_results, batch_name: str):
        """Validate continue-mode inputs before starting local training."""
        if PipelineExecutor._is_error_result(colab_results):
            logger.error(f"Cannot continue batch '{batch_name}': Colab results are missing or invalid.")
            return {'status': 'failed', 'reason': 'missing_colab_results'}

        if features_df is None or getattr(features_df, 'empty', True):
            logger.error(f"Cannot continue batch '{batch_name}': features.parquet is missing or empty.")
            return {'status': 'failed', 'reason': 'missing_features'}

        if targets_df is None or getattr(targets_df, 'empty', True):
            logger.error(f"Cannot continue batch '{batch_name}': targets.parquet is missing or empty.")
            return {'status': 'failed', 'reason': 'missing_targets'}

        target_cols = [col for col in targets_df.columns if str(col).startswith('target_')]
        if not target_cols:
            logger.error(f"Cannot continue batch '{batch_name}': targets.parquet has no target_* columns.")
            return {'status': 'failed', 'reason': 'missing_target_columns'}

        return None

    @staticmethod
    def _resolve_tickers(args, colab_results, features_df):
        """Resolve tickers for continue mode."""
        tickers = [args.test_ticker] if args.test_ticker else list(
            colab_results.get('ticker_results', {}).keys()
        )

        if not tickers and features_df is not None:
            if isinstance(features_df.index, pd.MultiIndex):
                tickers = list(features_df.index.get_level_values('ticker').unique())
            elif 'ticker' in features_df.columns:
                tickers = list(features_df['ticker'].unique())

        logger.info(f"Resolved tickers for continue mode: {tickers}")
        return tickers

    @staticmethod
    def _merge_results_data(colab_results, light_results):
        """Merge colab results with light models results."""
        merged_results = dict(colab_results)
        if light_results.get('models_metadata'):
            merged_results.setdefault('models_metadata', {}).update(light_results['models_metadata'])

        logger.info("Merged Colab results with light models")
        return merged_results

    @staticmethod
    def resolve_tickers_and_timeframes(args, config_manager) -> tuple[list, list]:
        """Resolve tickers and timeframes from args or config."""
        if args.tickers is not None:
            tickers = args.tickers
            logger.info(f"Using explicitly provided tickers: {tickers}")
        else:
            assets_config = config_manager.get_config('assets') or {}
            sectors = assets_config.get('sectors', {})

            all_tickers = set()
            for sector_name, sector_config in sectors.items():
                sector_assets = sector_config.get('assets', [])
                all_tickers.update(sector_assets)
                logger.debug(f"Sector '{sector_name}': {len(sector_assets)} tickers: {sector_assets}")

            tickers = sorted(all_tickers)
            logger.info(f"Loaded {len(tickers)} unique tickers from all sectors: {tickers}")
            logger.info(f"Processed sectors: {list(sectors.keys())}")

        if args.test_ticker:
            tickers = [args.test_ticker]
            logger.info(f"Using test ticker: {args.test_ticker}")

        collectors = config_manager.get_config('collectors') or {}
        yf_timeframes = collectors.get('yahoo_finance', {}).get('timeframes', {})
        timeframes = list(yf_timeframes.keys()) if yf_timeframes else ['15m', '60m', '1d']

        logger.info(f"Final tickers: {tickers}")
        logger.info(f"Using timeframes: {timeframes}")

        return tickers, timeframes

    @staticmethod
    def log_test_mode_info(args) -> None:
        """Log test mode information."""
        if PipelineExecutor._is_test_mode_active(args):
            PipelineExecutor._log_test_mode_details(args)

    @staticmethod
    def _is_test_mode_active(args) -> bool:
        """Check if test mode is active."""
        return any([args.test_ticker, args.test_target, args.test_model])

    @staticmethod
    def _log_test_mode_details(args) -> None:
        """Log test mode details."""
        logger.info("TEST MODE ACTIVATED:")
        if args.test_ticker:
            logger.info(f"   Ticker: {args.test_ticker}")
        if args.test_target:
            logger.info(f"   Target: {args.test_target}")
        if args.test_model:
            logger.info(f"   Model: {args.test_model}")
        logger.info(f"   Iterations: {args.max_iterations}")

    @staticmethod
    async def execute_calibrate_mode(orchestrator, args):
        """Execute calibration mode for DEAN hyperparameter tuning."""
        logger.info("Running DEAN calibration...")

        n_trials = getattr(args, 'n_trials', 50)
        results = await orchestrator.run_calibration(
            test_ticker=getattr(args, 'test_ticker', None),
            test_target=getattr(args, 'test_target', None),
            n_trials=n_trials,
        )

        if results.get('status') == 'success':
            logger.info("Calibration successful!")
            logger.info(f"   Best {results['metric']}: {results['best_value']:.4f}")
            logger.info(f"   Best hyperparameters: {results['best_params']}")
        else:
            logger.error(f"Calibration failed: {results.get('reason')}")

        return results
