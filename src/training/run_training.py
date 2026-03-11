#!/usr/bin/env python
# src/training/run_training.py
"""
Main entry point for the adaptive training pipeline.

This script initializes the training managers, generates an adaptive training plan,
and orchestrates the execution of the training process based on the strategy
defined in the `adaptive_training_manager`.
"""

import argparse
import logging
from typing import List

# Ensure the script can find other modules in the src directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.adaptive_training_manager import AdaptiveTrainingManager, AdaptiveTrainingConfig, TrainingMode
from config.tickers import get_tickers


def setup_logging():
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def main():
    """Main function to run the adaptive training process."""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run the Adaptive Training Pipeline.")
    parser.add_argument(
        "--tickers",
        type=str,
        default="core",
        help="A list of tickers (comma-separated), or a predefined category like 'core', 'all'."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=[mode.value for mode in TrainingMode],
        help="The training mode which influences target selection and model strategy."
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="If set, the manager will only analyze the ticker set and print the plan without executing it."
    )

    args = parser.parse_args()

    logger.info(f"Starting adaptive training process with mode: {args.mode}")

    # --- 1. Get Ticker List ---
    try:
        if "," in args.tickers:
            ticker_list = [t.strip().upper() for t in args.tickers.split(",")]
        else:
            # Fetch tickers from a predefined category
            ticker_list = get_tickers(args.tickers)
            # Limit for safety during testing/development
            if args.tickers == 'all':
                logger.warning("Running with 'all' tickers, limiting to first 50 for this run.")
                ticker_list = ticker_list[:50]

    except Exception as e:
        logger.error(f"Could not resolve tickers for '{args.tickers}'. Using a default list. Error: {e}")
        ticker_list = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL"]

    logger.info(f"Resolved {len(ticker_list)} tickers for training: {ticker_list[:10]}...")

    # --- 2. Initialize the Manager ---
    # The manager will create a training plan based on its analysis
    config = AdaptiveTrainingConfig(mode=TrainingMode(args.mode))
    manager = AdaptiveTrainingManager(config)

    # --- 3. Create or Execute Plan ---
    if args.analyze_only:
        logger.info("Running in 'analyze-only' mode.")
        plan = manager.create_adaptive_training_plan(ticker_list)
        plan_file = manager._save_adaptive_plan(plan) # Using internal method for convenience
        logger.info(f"Full adaptive training plan has been generated and saved to {plan_file}")
        # Here you could add more detailed printouts of the plan if desired
    else:
        logger.info("Executing full training and evaluation cycle...")
        # This will internally create a plan and then execute it.
        # Currently, execution is a simulation as noted in the README.
        results = manager.execute_adaptive_training(ticker_list)
        logger.info("Adaptive training execution finished.")
        summary = results.get("execution_summary", {})
        logger.info(f"Execution Summary: {summary}")

    logger.info("Script finished.")

if __name__ == "__main__":
    main()
