"""
Головний модуль запуску системи.

Цей скрипт відповідає за:
- Ініціалізацію конфігурації.
- Обробку аргументів командного рядка.
- Створення та запуск SystemOrchestrator.

Використання:
  python launcher.py --mode train --tickers BTC-USD
  python launcher.py --help
"""

import argparse
import logging
import sys
import asyncio

# Додаємо шлях до src для правильного імпорту
# sys.path.insert(0, './src')

from src.main.system_orchestrator import SystemOrchestrator
from src.core.logging.logger import ProjectLogger

async def main():
    """Головна функція запуску."""
    # 1. Налаштування логування
    ProjectLogger.setup_logging()
    logger = logging.getLogger(__name__)
    
    # 2. Обробка аргументів командного рядка
    parser = argparse.ArgumentParser(description="DEAN Trading System Launcher")
    parser.add_argument("--stages", type=int, nargs='+', help="List of stages to run (e.g., 0 1 2)")
    parser.add_argument("--mode", type=str, default="default", help="Operating mode (e.g., train, predict, backtest)")
    parser.add_argument("--tickers", type=str, nargs='+', help="List of tickers to process")
    # Додайте інші аргументи, які можуть знадобитися вашим режимам чи етапам

    args = parser.parse_args()
    
    logger.info(f"Launcher started with arguments: {args}")

    try:
        # 3. Створення та запуск оркестратора
        orchestrator = SystemOrchestrator()
        await orchestrator.run_mode(mode=args.mode, tickers=args.tickers, stages=args.stages)
        logger.info("System run completed successfully.")

    except Exception as e:
        # Централізована обробка винятків на найвищому рівні
        logger.critical(f"A critical error occurred in the system: {e}", exc_info=True)
        sys.exit(1) # Вихід з помилкою

if __name__ == "__main__":
    asyncio.run(main())
