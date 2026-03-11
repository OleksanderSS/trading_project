
# main.py

import asyncio
import argparse
import os
import sys

# Додаємо кореневий каталог проєкту до sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.main.system_orchestrator import SystemOrchestrator

async def main():
    """Asynchronous entry point of the application."""
    # Setup argparse
    parser = argparse.ArgumentParser(description="Run the data processing pipeline.")
    parser.add_argument('--stages', nargs='*', type=int, help='Optional list of stage indices to run.')
    parser.add_argument("--mode", type=str, default="training_data_pipeline", help="Operating mode (e.g., train, predict, backtest, training_data_pipeline)")
    parser.add_argument("--tickers", type=str, nargs='+', help="List of tickers to process")
    args = parser.parse_args()

    # Initialize logging and configuration with DEBUG level
    ProjectLogger.setup_logging(level="DEBUG")
    logger = ProjectLogger.get_logger(__name__)
    logger.info(f"Запуск програми в режимі: {args.mode}")

    try:
        # Створення та запуск оркестратора
        orchestrator = SystemOrchestrator()
        await orchestrator.run_mode(mode=args.mode, tickers=args.tickers)
        logger.info("System run completed successfully.")

    except Exception as e:
        # Централізована обробка винятків на найвищому рівні
        logger.critical(f"A critical error occurred in the system: {e}", exc_info=True)
        sys.exit(1) # Вихід з помилкою

if __name__ == "__main__":
    asyncio.run(main())
