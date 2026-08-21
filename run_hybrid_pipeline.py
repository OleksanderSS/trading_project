#!/usr/bin/env python3
"""
Script for running the hybrid pipeline.

Usage:
    python run_hybrid_pipeline.py --mode local      # Local stages only (0-3)
    python run_hybrid_pipeline.py --mode full       # Full pipeline execution
    python run_hybrid_pipeline.py --mode prepare    # Preparation for Colab
    python run_hybrid_pipeline.py --mode light      # Light models training only
    python run_hybrid_pipeline.py --mode continue   # Continue after Colab results
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import aiofiles

from src.cli.argument_parser import create_argument_parser
from src.core.logging.logger import ProjectLogger

# Everything else is imported inside main(), AFTER the arguments are parsed.
#
# `--help` exits inside parse_args and needs none of it, but these were at
# module level, so printing a help message loaded the entire pipeline. From
# `python -X importtime run_hybrid_pipeline.py --help`:
#
#     src.pipeline.hybrid_orchestrator    the whole stage chain
#     sklearn                             via normalization_manager
#     evidently                           13.2 s, until it was made lazy today
#
# The smoke test that runs `--help` has a 30-second timeout and the command
# took 15 to 44 seconds depending on machine load, so it passed or failed on
# weather rather than on anything about the code.
#
# A parse error or a bad argument now also fails in a second instead of after
# half a minute of imports it was never going to use.

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
    #
    # --max-iterations has a DEFAULT of 100, so `or args.max_iterations` was
    # always truthy and has_test_params was always True. The entire
    # "full mode" branch below -- the one that deletes a stale
    # runtime_params.json so a real run is not silently trained with
    # leftover test epochs -- was unreachable, and every run wrote the file.
    # Colab's ConfigLoader only checks that the file EXISTS, so a full run
    # was being treated as a test run: exactly the failure the branch was
    # written to prevent.
    #
    # It counts as explicitly set only when it differs from the default,
    # which is the same test main() already applies at `args.max_iterations != 100`.
    # None means "not supplied at all", which is not an override either.
    max_iterations = getattr(args, 'max_iterations', None)
    iterations_overridden = max_iterations is not None and max_iterations != 100
    has_test_params = bool(
        args.test_ticker
        or args.test_target
        or args.test_model
        or getattr(args, 'epochs', None) is not None
        or iterations_overridden
    )

    if not has_test_params:
        # A runtime_params.json left over from an earlier --epochs/--test-*
        # invocation must not silently survive into a full-mode run: Colab's
        # ConfigLoader only checks whether the file *exists*, not whether
        # this run is actually in test mode, so a stale file forces every
        # model to train with that old (often epochs=1) test configuration
        # while batch_metadata.json still reports test_mode: false.
        stale_path = Path("data/colab/accumulated") / batch_name / "runtime_params.json"
        if stale_path.exists():
            stale_path.unlink()
            logger.warning(
                f"📊 Full mode: removed stale test-mode runtime_params.json "
                f"at {stale_path} so this run isn't silently trained with "
                f"leftover test epochs/iterations."
            )
        else:
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
    # Parse arguments FIRST: --help and argument errors exit here, before any
    # of the pipeline is loaded.
    parser = create_argument_parser()
    args = parser.parse_args()

    from src.cli.argument_validator import ArgumentValidator
    from src.cli.batch_manager import BatchManager
    from src.config.unified_config_manager import UnifiedConfigManager

    # Initialize Config Manager
    config_manager = UnifiedConfigManager()

    # Check versions
    _check_versions(config_manager)

    # Say which numbers in this run are placeholders rather than choices. A
    # value nobody selected is indistinguishable from one that was, and this
    # pipeline has been caught by that repeatedly -- a 0.5% trade cost copied
    # into five files, a CIK resolving to the wrong company, a risk gate
    # reporting "checks passed" from two invented inputs.
    try:
        from src.config.pending_decisions import log_pending_decisions
        log_pending_decisions(logger)
    except Exception as exc:  # noqa: BLE001 - never let a notice end a run
        logger.warning("Could not read pending decisions: %s", exc)

    # Validate arguments
    ArgumentValidator.validate_arguments(args, config_manager)

    # Generate batch name if not provided
    if not args.batch_name:
        args.batch_name = BatchManager.generate_batch_name(args)
        _log_batch_mode(args)

    # Save runtime parameters
    await _save_runtime_params(args, args.batch_name)

    # The pipeline itself, loaded only once there is a run to make.
    from src.cli.pipeline_executor import PipelineExecutor
    from src.pipeline.hybrid_orchestrator import HybridOrchestrator

    # Log test mode info
    PipelineExecutor.log_test_mode_info(args)

    # Set environment variables
    if args.max_iterations != 100:
        os.environ['MAX_ITERATIONS'] = str(args.max_iterations)
        logger.info(f"⚡ MAX_ITERATIONS: {args.max_iterations} (default: 100)")

    # Two "enhanced components" were constructed here -- a PersistentModelPool
    # and a ModelQualityController -- and handed to nobody. Neither was passed
    # to HybridOrchestrator or to any stage, so nothing ever populated them,
    # and the summary below reported their empty state as a result:
    #     📊 Model Pool Stats: hits=0, hit_rate=0.0%, avg_quality=0.00
    #     ✅ Quality Report: 0 baselines tracked
    # in every run in logs/. A zero next to a ✅ reads as "measured, nothing
    # wrong" when it means "never ran", which is the more expensive of the two
    # mistakes. Construction and report removed; the pool class stays in
    # src/models/persistent_pool.py should caching ever be wired for real.

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
    else:
        # `--mode calibrate` sat in the parser's choices and was dispatched
        # here to PipelineExecutor.execute_calibrate_mode, which has never
        # existed -- so the advertised mode raised AttributeError on every
        # invocation. A mode that reaches this branch is advertised without an
        # implementation; say so instead of falling through with results=None,
        # which the reporting below reads as an empty successful run.
        raise NotImplementedError(
            f"Mode '{args.mode}' is accepted by the CLI but has no executor."
        )

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


def _run() -> int:
    """Run the pipeline and return a truthful exit code.

    A stage that fails must never leave a zero behind. Twice in two days a
    run ended with `RuntimeError: Stage FeatureEngineeringStage execution
    failed` and neither "Pipeline completed" nor "Pipeline failed" reached
    the log, so main() had not got as far as reporting -- and one of those
    runs still returned 0 to the shell while the other returned 1, from the
    same invocation shape.

    I could not reproduce which path produces which, and the honest response
    to an unreproduced mechanism is a guarantee rather than a diagnosis:
    anything escaping main() is reported here and exits non-zero, whatever
    the cause turns out to be. The cost of the alternative is measured -- a
    rebuild that "succeeded" left the batch untouched and the next step was
    planned on data that had never changed.

    SystemExit is re-raised untouched so `sys.exit(1)` from the reporting
    block keeps its own code.
    """
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - the whole point is breadth
        logger.critical(
            "Pipeline aborted with %s: %s", type(exc).__name__, exc,
            exc_info=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_run())
