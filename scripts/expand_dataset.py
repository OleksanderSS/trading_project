#!/usr/bin/env python3
"""
Скрипт для розширення датасету.

Розширює:
1. Кількість тікерів (з 1 на 15+)
2. Період даних (з 30 днів на 1+ рік)
3. Частоту даних (1 день, 1 година, 15 хвилин)

Тікери для розширення:
- Tech: AAPL, MSFT, GOOGL, NVDA, AMD
- Finance: JPM, BAC, GS, MS
- Healthcare: JNJ, PFE, UNH, ABBV
- Energy: XOM, CVX, COP
- Consumer: AMZN, WMT, HD
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Додаємо workspace root до path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.data.collection.market_data_collector import MarketDataCollector
from src.data.management.data_manager import DataManager

class DatasetExpander:
    """Розширює датасет для тренування"""
    
    def __init__(self):
        self.config = UnifiedConfigManager()
        self.logger = ProjectLogger.get_logger("DatasetExpander")
        self.collector = MarketDataCollector()
        self.data_manager = DataManager()
        
        # Розширені тікери
        self.tickers = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD'],
            'finance': ['JPM', 'BAC', 'GS', 'MS'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV'],
            'energy': ['XOM', 'CVX', 'COP'],
            'consumer': ['AMZN', 'WMT', 'HD'],
        }
        
        self.all_tickers = []
        for sector_tickers in self.tickers.values():
            self.all_tickers.extend(sector_tickers)
    
    def _collect_ticker_data(self, ticker, days, frequencies):
        """Collect data for a single ticker across all frequencies."""
        self.logger.info(f"\n📥 Збір даних для {ticker}...")
        ticker_data = {}
        
        for freq in frequencies:
            self.logger.info(f"   ⏱️ Частота: {freq}")
            
            try:
                # Collect data
                data = self.collector.collect_market_data(
                    ticker=ticker,
                    days=days,
                    interval=freq
                )
                
                if data is not None and len(data) > 0:
                    ticker_data[freq] = {
                        'rows': len(data),
                        'columns': len(data.columns),
                        'date_range': f"{data.index[0]} to {data.index[-1]}"
                    }
                    
                    # Store in DuckDB
                    self.data_manager.upsert_data(
                        table_name=f"raw_data_{ticker}_{freq}",
                        data=data
                    )
                    
                    self.logger.info(f"      ✅ Збрано {len(data)} рядків")
                
                else:
                    self.logger.warning(f"      ⚠️ Немає даних для {ticker} ({freq})")
                
            except Exception as e:
                self.logger.error(f"      ❌ Помилка збору {ticker} ({freq}): {e}")
                return None
        
        return ticker_data

    def _log_expansion_start(self, all_tickers, days, frequencies):
        """Log the start of dataset expansion."""
        self.logger.info("🚀 Початок розширення датасету")
        self.logger.info(f"   📊 Тікери: {len(all_tickers)} ({', '.join(all_tickers[:5])}...)")
        self.logger.info(f"   📅 Період: {days} днів")
        self.logger.info(f"   ⏱️ Частоти: {frequencies}")

    def _initialize_results(self, all_tickers, days, frequencies):
        """Initialize results dictionary."""
        return {
            'tickers': all_tickers,
            'days': days,
            'frequencies': frequencies,
            'collected_data': {},
            'errors': []
        }

    def _process_ticker(self, ticker, results, days, frequencies):
        """Process a single ticker and update results."""
        ticker_data = self._collect_ticker_data(ticker, days, frequencies)
        
        if ticker_data:
            results['collected_data'][ticker] = ticker_data
            self.logger.info(f"   ✅ {ticker}: {len(ticker_data)} частот")
        else:
            self.logger.error(f"❌ Помилка обробки {ticker}: No data collected")
            results['errors'].append(f"{ticker}: No data collected")

    def expand_dataset(self, 
                      days: int = 365,
                      frequencies: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Розширює датасет
        
        Args:
            days: Кількість днів для збору (за замовчуванням 365)
            frequencies: Частоти даних (за замовчуванням ['1d', '1h', '15m'])
        
        Returns:
            Dict з результатами розширення
        """
        if frequencies is None:
            frequencies = ['1d', '1h', '15m']
        
        # Log expansion start
        self._log_expansion_start(self.all_tickers, days, frequencies)
        
        # Initialize results
        results = self._initialize_results(self.all_tickers, days, frequencies)
        
        # Process each ticker
        for ticker in self.all_tickers:
            try:
                self._process_ticker(ticker, results, days, frequencies)
            except Exception as e:
                self.logger.error(f"❌ Помилка обробки {ticker}: {e}")
                results['errors'].append(f"{ticker}: {str(e)}")
        
        self.print_summary(results)
        return results
    
    def print_summary(self, results: Dict):
        """Виводить звіт про розширення"""
        print("\n" + "="*60)
        print("📊 ЗВІТ ПРО РОЗШИРЕННЯ ДАТАСЕТУ")
        print("="*60)
        
        print(f"\n✅ Тікери: {len(results['tickers'])}")
        for sector, tickers in self.tickers.items():
            print(f"   {sector.upper()}: {', '.join(tickers)}")
        
        print(f"\n📅 Період: {results['days']} днів")
        print(f"⏱️ Частоти: {', '.join(results['frequencies'])}")
        
        print("\n📥 Зібрано даних:")
        total_rows = 0
        for ticker, data in results['collected_data'].items():
            print(f"   {ticker}:")
            for freq, info in data.items():
                print(f"      {freq}: {info['rows']} рядків ({info['date_range']})")
                total_rows += info['rows']
        
        print(f"\n📈 Всього рядків: {total_rows}")
        
        if results['errors']:
            print(f"\n⚠️ Помилок: {len(results['errors'])}")
            for error in results['errors'][:5]:
                print(f"   - {error}")
        
        print("\n✨ Розширення завершено!")

if __name__ == "__main__":
    expander = DatasetExpander()
    
    # Розширюємо датасет на 1 рік з 3 частотами
    results = expander.expand_dataset(
        days=365,
        frequencies=['1d', '1h', '15m']
    )
