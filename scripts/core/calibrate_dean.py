#!/usr/bin/env python3
"""
Standalone script for DEAN hyperparameter calibration.

Usage:
    python scripts/calibrate_dean.py
    python scripts/calibrate_dean.py --ticker AMD
    python scripts/calibrate_dean.py --ticker AMD --target target_return_1d --trials 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.calibration import CalibrationEngine
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DEAN Hyperparameter Calibration')
    parser.add_argument(
        '--ticker',
        help='Test ticker for calibration'
    )
    parser.add_argument(
        '--target',
        help='Test target for calibration'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of Optuna trials (default: 50)'
    )
    parser.add_argument(
        '--metric',
        default='sharpe_ratio',
        choices=['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor', 'calmar_ratio'],
        help='Primary optimization metric (default: sharpe_ratio)'
    )
    parser.add_argument(
        '--batch-name',
        default='calibration',
        help='Batch name for outputs (default: calibration)'
    )
    parser.add_argument(
        '--real-data-path',
        default='data/duckdb/trading.db',
        help='Path to DuckDB database (default: data/duckdb/trading.db)'
    )
    parser.add_argument(
        '--synthetic-data-path',
        default='data/synthetic/',
        help='Path to synthetic scenarios (default: data/synthetic/)'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("🎯 DEAN Calibration Script")
    logger.info(f"   Ticker: {args.ticker or 'all'}")
    logger.info(f"   Target: {args.target or 'all'}")
    logger.info(f"   Trials: {args.trials}")
    logger.info(f"   Metric: {args.metric}")
    logger.info(f"   Batch: {args.batch_name}")

    # Initialize calibration engine
    engine = CalibrationEngine(
        real_data_path=args.real_data_path,
        synthetic_data_path=args.synthetic_data_path,
        n_trials=args.trials,
        metric=args.metric,
        batch_name=args.batch_name
    )

    # Run calibration
    results = engine.run_calibration(
        test_ticker=args.ticker,
        test_target=args.target
    )

    # Print results
    if results.get('status') == 'success':
        logger.info("✅ Calibration completed successfully!")
        logger.info(f"   Best {results['metric']}: {results['best_value']:.4f}")
        logger.info(f"   Best hyperparameters:")
        for param, value in results['best_params'].items():
            logger.info(f"      {param}: {value}")
        logger.info(f"   Results saved to: results/calibration/{args.batch_name}/calibration_results.json")
    else:
        logger.error(f"❌ Calibration failed: {results.get('reason')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
