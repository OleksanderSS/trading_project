"""
Pipeline execution utilities for hybrid pipeline.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
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
        logger.info("🏠 Running local pipeline (stages 0-3)...")
        return await orchestrator.run_local_pipeline(
            tickers=tickers,
            timeframes=timeframes
        )

    @staticmethod
    async def execute_light_mode(orchestrator, tickers: list, timeframes: list):
        """Execute light models training only."""
        logger.info("💡 Running light models training...")
        return await orchestrator.run_light_models(
            tickers=tickers
        )

    @staticmethod
    async def execute_prepare_mode(orchestrator, tickers: list, timeframes: list, **kwargs):
        """Execute preparation for Colab."""
        logger.info("📦 Preparing data for Colab training...")
        return await orchestrator.prepare_colab_data(
            tickers=tickers,
            timeframes=timeframes,
            **kwargs
        )

    @staticmethod
    async def execute_full_mode(orchestrator, tickers: list, timeframes: list):
        """Execute full pipeline."""
        logger.info("🚀 Running full pipeline...")
        return await orchestrator.run_local_pipeline(
            tickers=tickers,
            timeframes=timeframes
        )

    @staticmethod
    async def execute_continue_mode(orchestrator, args):
        """Execute continue mode."""
        logger.info("🔄 Running continue mode...")
        
        # Load data for continue mode
        features_df, _targets_df, colab_results = PipelineExecutor._load_continue_data(orchestrator, args)
        
        # Resolve tickers
        tickers = PipelineExecutor._resolve_tickers(args, colab_results, features_df)
        
        # Run light models with test parameters
        light_results = await orchestrator.run_light_models(
            tickers=tickers,
            test_ticker=getattr(args, 'test_ticker', None),
            test_target=getattr(args, 'test_target', None),
            batch_name=args.batch_name
        )
        
        # Run final stages (5-7)
        logger.info("🏁 Running final stages (5-7)...")
        from src.pipeline.hybrid_orchestrator import FinalStagesRequest
        
        final_request = FinalStagesRequest(
            features_df=features_df,
            targets_df=_targets_df,
            colab_results=colab_results,
            light_results=light_results,
            tickers=tickers,
            timeframes=['15m', '60m', '1d'], # Use default timeframes
            batch_name=args.batch_name
        )
        
        return await orchestrator.run_final_stages(final_request)

    @staticmethod
    def _load_continue_data(orchestrator, args):
        """Load data for continue mode."""
        # Note: orchestrator.config.output_dir already includes batch_name!
        # Don't add batch_name again
        batch_dir = orchestrator.config.output_dir
        colab_results = orchestrator.load_colab_results(args.batch_name)
        
        if not colab_results:
            logger.error(f"❌ No Colab results found for batch: {args.batch_name}")
            return None, None, None
        
        # Load features and targets
        features_path = batch_dir / FEATURES_FILE
        targets_path = batch_dir / TARGETS_FILE
        
        features_df = None
        targets_df = None
        
        if features_path.exists():
            features_df = pd.read_parquet(features_path)
            logger.info(f"✅ Loaded features: {features_df.shape}")
        else:
            logger.warning(f"⚠️ Features file not found: {features_path}")
        
        if targets_path.exists():
            targets_df = pd.read_parquet(targets_path)
            logger.info(f"✅ Loaded targets: {targets_df.shape}")
        else:
            logger.warning(f"⚠️ Targets file not found: {targets_path}")
        
        return features_df, targets_df, colab_results

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
        
        logger.info(f"📈 Resolved tickers for continue mode: {tickers}")
        return tickers

    @staticmethod
    def _merge_results_data(colab_results, light_results):
        """Merge colab results with light models results."""
        merged_results = dict(colab_results)
        if light_results.get('models_metadata'):
            merged_results['models_metadata'].update(light_results['models_metadata'])
        
        logger.info("✅ Merged Colab results with light models")
        return merged_results

    @staticmethod
    def resolve_tickers_and_timeframes(args, config_manager) -> Tuple[list, list]:
        """Resolve tickers and timeframes from args or config."""
        # If tickers are explicitly provided in args, use them
        if args.tickers is not None:
            tickers = args.tickers
            logger.info(f"📊 Using explicitly provided tickers: {tickers}")
        else:
            # Load all unique tickers from all sectors in assets configuration
            assets_config = config_manager.get_config('assets') or {}
            sectors = assets_config.get('sectors', {})
            
            # Collect all unique tickers from all sectors
            all_tickers = set()
            for sector_name, sector_config in sectors.items():
                sector_assets = sector_config.get('assets', [])
                all_tickers.update(sector_assets)
                logger.debug(f"📂 Sector '{sector_name}': {len(sector_assets)} tickers: {sector_assets}")
            
            tickers = sorted(all_tickers)
            logger.info(f"📊 Loaded {len(tickers)} unique tickers from all sectors: {tickers}")
            logger.info(f"📂 Processed sectors: {list(sectors.keys())}")
        
        logger.info(f"🔍 DEBUG: args.test_ticker = {repr(args.test_ticker)}, bool = {bool(args.test_ticker)}")
        if args.test_ticker:
            tickers = [args.test_ticker]
            logger.info(f"🧪 Using test ticker: {args.test_ticker}")
        
        timeframes = ['15m', '60m', '1d']
        
        logger.info(f"📈 Final tickers: {tickers}")
        logger.info(f"⏰ Using timeframes: {timeframes}")
        
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
