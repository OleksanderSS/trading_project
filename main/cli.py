#!/usr/bin/env python3
"""
CLI interface for trading system
"""

import argparse
import logging
from typing import List, Optional
from pathlib import Path

from config.trading_config import get_config
from .orchestrator import TradingOrchestrator


def create_cli_parser() -> argparse.ArgumentParser:
    """Створення CLI парсера"""
    examples = [
        'python main.py --mode train',
        'python main.py --mode analyze --tickers TSLA,NVDA',
        'python main.py --mode backtest --tickers TSLA,NVDA,AAPL --initial-capital 100000',
        'python main.py --mode comprehensive-backtest --tickers TSLA,NVDA,AAPL --initial-capital 100000',
        'python main.py --mode optimized-backtest --tickers TSLA,NVDA,AAPL --initial-capital 100000',
        'python main.py --mode integrated-backtest --tickers TSLA,NVDA,AAPL --initial-capital 100000',
        'python main.py --mode real-data-backtest --tickers TSLA,NVDA,AAPL --initial-capital 100000',
        'python main.py --mode web-ui --port 8080'
    ]
    parser = argparse.ArgumentParser(
        description="AI Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  {'\n  '.join(examples)}
        """
    )
    
    # Основні режими
    parser.add_argument(
        '--mode',
        choices=['train', 'analyze', 'batch-training', 'progressive', 'monster-test', 
                'backtest', 'comprehensive-backtest', 'optimized-backtest', 'integrated-backtest', 'real-data-backtest', 'web-ui',
                'intelligent'],  # Додано інтелектуальний режим
        default='train',
        help='Режим роботи системи (train, analyze, batch-training, progressive, monster-test, backtest, comprehensive-backtest, optimized-backtest, integrated-backtest, real-data-backtest, web-ui, intelligent)'
    )
    
    # Тікери
    parser.add_argument(
        '--tickers',
        type=str,
        help='Список тікерів через кому (напр: TSLA,NVDA,AAPL)'
    )
    
    # Таймфрейми
    parser.add_argument(
        '--timeframes',
        type=str,
        help='Список таймфреймів через кому (напр: 15m,1h,1d)'
    )
    
    # Risk management
    parser.add_argument(
        '--risk-per-trade',
        type=float,
        help='Ризик на одну торгівлю (0.01 = 1%)'
    )
    
    parser.add_argument(
        '--max-positions',
        type=int,
        help='Максимальна кількість позицій'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Початковий капітал для бектестингу (default: 100000)'
    )
    
    # Web UI options
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port для Web UI (default: 8080)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host для Web UI (default: localhost)'
    )
    
    # Логування
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Рівень логування'
    )
    
    # Конфігурація
    parser.add_argument(
        '--config-file',
        type=Path,
        help='Шлях до файлу конфігурації'
    )
    
    # Вихідні дані
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Директорія для результатів'
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Валідація аргументів"""
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
        if len(tickers) > 50:
            raise ValueError("Занадто багато тікерів (максимум 50)")
    
    if args.timeframes:
        timeframes = [t.strip() for t in args.timeframes.split(',')]
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        for tf in timeframes:
            if tf not in valid_timeframes:
                raise ValueError(f"Невалідний таймфрейм: {tf}. Доступні: {valid_timeframes}")
    
    if args.risk_per_trade is not None:
        if args.risk_per_trade <= 0 or args.risk_per_trade > 0.1:
            raise ValueError("Risk per trade повинен бути між 0 і 0.1")
    
    if args.max_positions is not None:
        if args.max_positions <= 0 or args.max_positions > 100:
            raise ValueError("Max positions повинен бути між 1 і 100")


def setup_logging(args: argparse.Namespace) -> None:
    """Налаштування логування"""
    config = get_config()
    
    log_level = getattr(logging, args.log_level)
    
    # Створюємо директорію для логів
    config.logging.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Налаштування форматування
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(config.logging.logs_dir / config.logging.log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def run_cli(args: Optional[List[str]] = None) -> int:
    """Основна функція CLI"""
    try:
        # Парсинг аргументів
        parser = create_cli_parser()
        parsed_args = parser.parse_args(args)
        
        # Валідація
        validate_args(parsed_args)
        
        # Налаштування логування
        setup_logging(parsed_args)
        
        # Створення orchestrator
        orchestrator = TradingOrchestrator(parsed_args)
        
        # Запуск відповідного режиму
        result = orchestrator.run_mode(parsed_args.mode)
        
        # Вивід результатів
        print(f"\n{'='*60}")
        print(f"RESULTS: {parsed_args.mode.upper()}")
        print(f"{'='*60}")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Message: {result.get('message', 'No message')}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    exit(run_cli())
