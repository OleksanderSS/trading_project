#!/usr/bin/env python3
"""
Automatic Intraday Data Accumulator Script
Автоматичний накопичувач даних для коротких свічок (5m, 15m, 1h, 1d)
Уніфікована версія з використанням UnifiedConfigManager та DataManager.
Діє як 'Integrity Guard' для забезпечення цілісності 4-х таймфреймів.
"""

import sys
import argparse
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

# Додавання кореня проекту до шляху
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config
from src.data.management.data_manager import DataManager
from src.core.error_handling.error_handler import ErrorHandler
from src.data.collector_factory import create_all_collectors
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.asset_manager import AssetUniverseManager, TradingStyle, MarketFocus

# Налаштування логування через ProjectLogger
logger = ProjectLogger.get_logger("AutoAccumulatorGuard")

class AutoAccumulatorGuard:
    """
    Клас-охоронець цілісності даних (Integrity Guard) для 4-х таймфреймів.
    Перевіряє наявність дірок та дозавантажує пропущені періоди.
    """
    def __init__(self):
        self.config_manager = get_current_config()
        self.error_handler = ErrorHandler()
        self.db_manager = DataManager(self.config_manager, self.error_handler)
        self.http_factory = HttpClientFactory(self.config_manager, self.error_handler)
        
        # --- INTEGRATION OF AssetUniverseManager ---
        self.asset_manager = AssetUniverseManager(self.config_manager.get_config('asset_universe', {}))
        # For example, using the 'day_trading_tech' preset.
        # This can be made configurable later.
        trading_config = self.asset_manager.get_preset("day_trading_tech")
        self.active_tickers = trading_config.tickers
        logger.info(f"Loaded {len(self.active_tickers)} tickers from AssetUniverseManager preset: 'day_trading_tech'")
        # --- END INTEGRATION ---
        
        # Стандартні ліміти для 4-х таймфреймів згідно з архітектурою та обмеженнями yfinance
        self.timeframes = self.config_manager.get_config('assets', {}).get('timeframes', {
            '5m': {'period': '60d', 'minutes': 5},
            '15m': {'period': '60d', 'minutes': 15},
            '1h': {'period': '730d', 'minutes': 60},
            '1d': {'period': 'max', 'minutes': 1440}
        })
        
        logger.info(f"Integrity Guard initialized for {len(self.active_tickers)} tickers and {len(self.timeframes)} timeframes.")

    def check_integrity_gaps(self, ticker: str, interval: str) -> bool:
        """
        Перевіряє наявність значних часових дірок у даних для конкретного тікера та інтервалу.
        """
        try:
            query = f"""
            SELECT datetime FROM market_data 
            WHERE ticker = '{ticker}' AND interval = '{interval}' 
            ORDER BY datetime DESC LIMIT 100
            """
            df = self.db_manager.con.execute(query).fetchdf()
            
            if df.empty:
                logger.warning(f"Дані для {ticker} ({interval}) відсутні в базі. Потрібне повне завантаження.")
                return True
            
            # Перевірка останньої свічки на актуальність
            df['datetime'] = pd.to_datetime(df['datetime'])
            latest_time = df['datetime'].max()
            now = pd.Timestamp.now(tz=latest_time.tz)
            
            interval_min = self.timeframes.get(interval, {}).get('minutes', 5)
            diff_minutes = (now - latest_time).total_seconds() / 60
            
            # Якщо лаг більше ніж 3 інтервали - вважаємо це діркою
            if diff_minutes > (interval_min * 3):
                logger.info(f"Виявлено лаг для {ticker} ({interval}): {diff_minutes:.1f} хв. Запуск дозавантаження.")
                return True
                
            return False
        except Exception as e:
            logger.error(f"Помилка перевірки цілісності для {ticker} ({interval}): {e}")
            return True

    async def _run_collection_task(self, tickers: List[str], timeframes: Dict[str, Any]):
        """Запускає колектори для конкретних тікерів та таймфреймів."""
        collectors_config = self.config_manager.get_config('collectors', {})
        
        # Динамічне налаштування конфігурації для Yahoo Finance
        if 'yahoo_finance' in collectors_config:
            collectors_config['yahoo_finance']['enabled'] = True
            collectors_config['yahoo_finance']['tickers'] = tickers
            collectors_config['yahoo_finance']['timeframes'] = timeframes

        collectors = create_all_collectors(
            collectors_config=collectors_config,
            db_manager=self.db_manager,
            http_client_factory=self.http_factory,
            error_handler=self.error_handler,
            normalizer=None
        )
        
        if not collectors:
            logger.error("Не вдалося ініціалізувати колектори для дозавантаження.")
            return

        tasks = [c.run() for c in collectors]
        await asyncio.gather(*tasks)

    def run_guard_cycle(self):
        """Проводить повну перевірку та виправлення дірок для всіх активів."""
        logger.info("--- Початок циклу Integrity Guard ---")
        
        needed_updates = {}
        
        for ticker in self.active_tickers:
            ticker_tf_needed = {}
            for interval, params in self.timeframes.items():
                if self.check_integrity_gaps(ticker, interval):
                    ticker_tf_needed[interval] = params
            
            if ticker_tf_needed:
                needed_updates[ticker] = ticker_tf_needed

        if not needed_updates:
            logger.info("Усі 4 таймфрейми для всіх тікерів цілісні. Втручання не потрібне.")
            return True

        logger.info(f"Потрібне дозавантаження для {len(needed_updates)} тікерів.")
        
        # Групуємо оновлення по інтервалах для ефективності колектора
        unique_intervals = set()
        for tf_dict in needed_updates.values():
            unique_intervals.update(tf_dict.keys())
            
        for interval in unique_intervals:
            tickers_for_interval = [t for t, tfs in needed_updates.items() if interval in tfs]
            interval_params = {interval: self.timeframes[interval]}
            
            logger.info(f"Дозавантаження '{interval}' для: {tickers_for_interval}")
            asyncio.run(self._run_collection_task(tickers_for_interval, interval_params))
            
        logger.info("--- Цикл Integrity Guard завершено ---")
        return True

    def get_db_report(self):
        """Друкує короткий звіт про стан бази даних."""
        try:
            tables = self.db_manager.get_all_tables()
            self.logger.info(f"\n[Database Status: {self.db_manager.db_path}]")
            if 'market_data' in tables:
                query = "SELECT interval, COUNT(*) as count, MIN(datetime) as start, MAX(datetime) as end FROM market_data GROUP BY interval"
                df = self.db_manager.con.execute(query).fetchdf()
                self.logger.info(f"\n{df.to_string(index=False)}")
            else:
                self.logger.warning("Table 'market_data' not found.")
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")

def main():
    parser = argparse.ArgumentParser(description='Integrity Guard for 5m, 15m, 1h, 1d Timeframes')
    parser.add_argument('--mode', default='once', 
                       choices=['once', 'cycle'],
                       help='Режим роботи: once (одноразово перед пайплайном) або cycle (постійний моніторинг)')
    parser.add_argument('--interval', type=int, default=15,
                       help='Інтервал перевірки в хвилинах для режиму cycle')
    parser.add_argument('--report', action='store_true', help='Показати звіт бази даних')
    
    args = parser.parse_args()
    guard = AutoAccumulatorGuard()
    
    if args.report:
        guard.get_db_report()
        return

    if args.mode == 'once':
        guard.run_guard_cycle()
    elif args.mode == 'cycle':
        logger.info(f"Запуск в режимі моніторингу кожні {args.interval} хв.")
        while True:
            try:
                guard.run_guard_cycle()
                logger.info(f"Очікування {args.interval} хв до наступної перевірки...")
                time.sleep(args.interval * 60)
            except KeyboardInterrupt:
                logger.info("Моніторинг зупинено користувачем.")
                break
            except Exception as e:
                logger.error(f"Критична помилка в циклі моніторингу: {e}")
                time.sleep(60)

if __name__ == "__main__":
    main()