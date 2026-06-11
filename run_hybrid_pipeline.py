#!/usr/bin/env python3
"""
Script for running the hybrid pipeline.

Usage:
    python run_hybrid_pipeline.py --mode local      # Local stages only (0-3)
    python run_hybrid_pipeline.py --mode full       # Full pipeline execution
    python run_hybrid_pipeline.py --mode prepare    # Preparation for Colab
    python run_hybrid_pipeline.py --mode light      # Light models training only
    python run_hybrid_pipeline.py --mode continue   # Continue after Colab results
    python run_hybrid_pipeline.py --mode calibrate  # DEAN hyperparameter calibration
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import aiofiles

from src.cli.argument_parser import create_argument_parser
from src.cli.argument_validator import ArgumentValidator
from src.cli.batch_manager import BatchManager
from src.cli.pipeline_executor import PipelineExecutor
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.pipeline.hybrid_orchestrator import HybridOrchestrator
from src.models.persistent_pool import PersistentModelPool
from src.models.quality.controller import ModelQualityController

# Configure console encoding for Windows
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[union-attr]
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')  # type: ignore[union-attr]

logger = ProjectLogger.get_logger(__name__)


def _check_versions(_config_manager) -> None:
    """Check version requirements."""
    try:
        from src.core.version_checker import VersionChecker
        checker = VersionChecker(_config_manager)
        checker.check_all()
    except Exception as e:
        logger.warning(f"⚠️ Could not verify versions: {e}")


async def _save_runtime_params(args, batch_name: str) -> None:
    """Save runtime parameters to file ONLY for test mode.

    Гнучка система:
    - Якщо є test параметри → створюється runtime_params.json → тестовий режим
    - Якщо НЕ має test параметрів → НЕ створюється runtime_params.json → повноцінний режим
    """
    # Перевіряємо, чи це тестовий режим
    has_test_params = args.test_ticker or args.test_target or args.test_model or getattr(args, 'epochs', None) or getattr(args, 'max_iterations', None)

    if not has_test_params:
        logger.info("📊 Full mode: NOT creating runtime_params.json")
        return  # ← Не створюємо файл для повноцінного режиму

    # Тільки для тестового# Only for test mode:
    runtime_params = {
        'mode': args.mode,
        'test_ticker': args.test_ticker,
        'test_target': args.test_target,
        'test_model': args.test_model,
        'test_mode': {
            'test_ticker': args.test_ticker,
            'test_target': args.test_target,
            'test_model': args.test_model,
        },
        'epochs': getattr(args, 'epochs', None),  # Only for test mode
        'max_iterations': getattr(args, 'max_iterations', None),  # Only for test mode
        'timeframes': ['15m', '1h', '1d'],  # Повноцінні таймфрейми!
        'tickers': 'all',  # Всі тікери з assets.yaml!
        'batch': {
            'batch_name': batch_name,
            'created_at': datetime.now().isoformat()
        }
    }

    # Save to colab data directory (same location as features/targets)
    colab_data_dir = Path("data/colab/accumulated")
    colab_data_dir.mkdir(parents=True, exist_ok=True)

    batch_dir = colab_data_dir / batch_name
    batch_dir.mkdir(exist_ok=True)

    runtime_params_path = batch_dir / "runtime_params.json"

    async with aiofiles.open(runtime_params_path, 'w') as f:
        await f.write(json.dumps(runtime_params, indent=2))

    logger.info(f"🧪 Test mode: runtime_params.json created at {runtime_params_path}")


# Message templates to avoid complexity
TEST_MODE_MESSAGE = "🧪 TEST MODE"
FULL_MODE_MESSAGE = "📦 FULL MODE"
FULL_MODE_SUFFIX = " (all metrics)"

def _log_batch_mode(args):
    """Log batch mode information."""
    batch_name = args.batch_name

    # Select message components
    if _is_test_mode(args):
        mode_prefix = TEST_MODE_MESSAGE
        mode_suffix = ""
    else:
        mode_prefix = FULL_MODE_MESSAGE
        mode_suffix = FULL_MODE_SUFFIX

    # Log the message
    message = f"{mode_prefix}: batch_name = {batch_name}{mode_suffix}"
    logger.info(message)

def _is_test_mode(args):
    """Check if running in test mode."""
    test_conditions = [args.test_ticker, args.test_target, args.test_model]
    return any(test_conditions)


async def main():
    """Main entry point with improved cohesion."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Initialize Config Manager
    config_manager = UnifiedConfigManager()

    # Check versions
    _check_versions(config_manager)

    # Validate arguments
    ArgumentValidator.validate_arguments(args, config_manager)

    # Generate batch name if not provided
    if not args.batch_name:
        args.batch_name = BatchManager.generate_batch_name(args)
        _log_batch_mode(args)

    # Save runtime parameters
    await _save_runtime_params(args, args.batch_name)

    # Log test mode info
    PipelineExecutor.log_test_mode_info(args)

    # Set environment variables
    if args.max_iterations != 100:
        os.environ['MAX_ITERATIONS'] = str(args.max_iterations)
        logger.info(f"⚡ MAX_ITERATIONS: {args.max_iterations} (default: 100)")

    # Initialize enhanced components
    logger.info("🔧 Initializing enhanced components...")
    model_pool = PersistentModelPool(
        max_models=50,
        cache_dir=".model_cache"
    )
    quality_controller = ModelQualityController(
        drift_threshold=0.3
    )
    logger.info("✅ Enhanced components initialized")

    # Initialize orchestrator
    logger.info(f"🚀 Launching hybrid pipeline (batch: {args.batch_name})...")
    orchestrator = HybridOrchestrator(config_manager, batch_name=args.batch_name)

    # Resolve tickers and timeframes
    tickers, timeframes = PipelineExecutor.resolve_tickers_and_timeframes(args, config_manager)

    # Execute based on mode
    results = None
    if args.mode == 'local':
        results = await PipelineExecutor.execute_local_mode(orchestrator, tickers, timeframes)
    elif args.mode == 'light':
        results = await PipelineExecutor.execute_light_mode(orchestrator, tickers, timeframes)
    elif args.mode == 'prepare':
        results = await PipelineExecutor.execute_prepare_mode(
            orchestrator, tickers, timeframes,
            test_ticker=args.test_ticker,
            test_target=args.test_target,
            test_model=args.test_model,
            epochs=getattr(args, 'epochs', None),
            max_iterations=getattr(args, 'max_iterations', None)
        )
    elif args.mode == 'full':
        results = await PipelineExecutor.execute_full_mode(orchestrator, tickers, timeframes)
    elif args.mode == 'continue':
        results = await PipelineExecutor.execute_continue_mode(orchestrator, args)
    elif args.mode == 'calibrate':
        results = await PipelineExecutor.execute_calibrate_mode(orchestrator, args)

    # Log enhanced component statistics
    if model_pool:
        pool_stats = model_pool.get_enhanced_stats()
        logger.info(f"📊 Model Pool Stats: hits={pool_stats['hits']}, "
                   f"hit_rate={pool_stats['hit_rate']:.1f}%, "
                   f"avg_quality={pool_stats['avg_quality']:.2f}")
    
    if quality_controller:
        quality_report = quality_controller.generate_report()
        logger.info(f"✅ Quality Report: {quality_report['total_baselines']} baselines tracked")

    # Log completion
    failed = (
        not results or
        (isinstance(results, dict) and results.get('status') in {'error', 'failed'})
    )
    if not failed:
        logger.info(f"✅ Pipeline completed successfully for batch: {args.batch_name}")
    else:
        logger.error(f"❌ Pipeline failed for batch: {args.batch_name}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
