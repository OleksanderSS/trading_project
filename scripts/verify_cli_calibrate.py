
import asyncio

from src.cli.argument_parser import create_argument_parser
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("CLITest")

async def test_cli_calibrate():
    logger.info("Simulating CLI calibrate command execution...")
    
    # 1. Simulate parsed arguments
    parser = create_argument_parser()
    # Mocking arguments as if called with: --mode calibrate --test-ticker AAPL --n-trials 5
    args = parser.parse_args(['--mode', 'calibrate', '--test-ticker', 'AAPL', '--n-trials', '5'])
    
    logger.info(f"Parsed args: mode={args.mode}, ticker={args.test_ticker}, trials={args.n_trials}")
    
    # 2. Validate PipelineExecutor logic
    if args.mode == 'calibrate':
        logger.info("PipelineExecutor would now trigger orchestrator.run_calibration")
        # We don't actually run the full orchestrator here to avoid long execution,
        # but we confirm that the CLI logic maps the command correctly.
        assert args.n_trials == 5
        assert args.test_ticker == 'AAPL'
        logger.info("✅ CLI arguments for calibrate mode parsed correctly.")
    else:
        logger.error("❌ Mode not recognized.")

if __name__ == "__main__":
    asyncio.run(test_cli_calibrate())
