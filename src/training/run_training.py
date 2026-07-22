#!/usr/bin/env python
# src/training/run_training.py
"""
Main entry point for the adaptive training pipeline.

This script initializes the training managers, generates an adaptive training plan,
and orchestrates the execution of the training process based on the strategy
defined in the `adaptive_training_manager`.
"""

import argparse

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.training.adaptive_training_manager import AdaptiveTrainingManager, TrainingMode
from src.training.base_trainer import TrainerConfig

ProjectLogger.setup_logging()
logger = ProjectLogger.get_logger("TrainingRunner")

def main():
    """Main function to run the adaptive training process."""

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
            # Fetch tickers from unified config manager
            cfg_mgr = UnifiedConfigManager()
            assets_cfg = cfg_mgr.get_specific_config("assets")
            preset_name = args.tickers
            
            # First try presets
            preset = assets_cfg.get("presets", {}).get(preset_name, {})
            if preset and "tickers" in preset:
                ticker_list = preset["tickers"]
            else:
                # Then try sectors
                sector = assets_cfg.get("sectors", {}).get(preset_name, {})
                if sector and "assets" in sector:
                    ticker_list = sector["assets"]
                elif preset_name == "all":
                    # Collect all tickers from all sectors
                    all_tickers = set()
                    for sect_data in assets_cfg.get("sectors", {}).values():
                        all_tickers.update(sect_data.get("assets", []))
                    ticker_list = list(all_tickers)
                else:
                    logger.error(f"Ticker category '{preset_name}' not found in assets.yaml")
                    ticker_list = []

            # Limit for safety during testing/development
            if args.tickers == 'all':
                logger.warning("Running with 'all' tickers, limiting to first 50 for this run.")
                ticker_list = ticker_list[:50]

    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.error(f"Could not resolve tickers for '{args.tickers}'. Using a default list. Error: {e}")
        ticker_list = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL"]

    logger.info(f"Resolved {len(ticker_list)} tickers for training: {ticker_list[:10]}...")

    # --- 2. Initialize the Manager ---
    # The manager will create a training plan based on its analysis
    config = TrainerConfig(mode=TrainingMode(args.mode))
    manager = AdaptiveTrainingManager(config)

    # --- 3. Create or Execute Plan ---
    if args.analyze_only:
        logger.info("Running in 'analyze-only' mode.")
        plan = manager.create_adaptive_training_plan(ticker_list)
        plan_file = manager._save_adaptive_report(plan, "plan") # Updated to use the correct method name from the manager
        logger.info(f"Full adaptive training plan has been generated and saved to {plan_file}")
    else:
        logger.info("Executing full training cycle...")
        # This will internally create a plan and then execute it.
        # Currently, execution is a simulation as noted in the module documentation.
        results = manager.execute_adaptive_training(ticker_list)
        logger.info("Adaptive training execution finished.")
        summary = results.get("execution_summary", {})
        if summary:
            logger.info(f"Execution Summary: {summary}")

    logger.info("Script execution completed.")

if __name__ == "__main__":
    main()
