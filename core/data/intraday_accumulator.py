"""
Intraday Data Accumulator
Накопичувач data for коротких candles (15m, 60m)
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from dataclasses import dataclass
from collections import defaultdict
import time

# Додаємо шлях до проекту
current_dir = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(current_dir))

from collectors.yf_collector import YFCollector
from config.tickers import get_tickers
from utils.performance_tracker import PerformanceTracker

logger = logging.getLogger("IntradayAccumulator")

@dataclass
class AccumulationConfig:
    """Конфandгурацandя накопичення data"""
    # Баwithовand settings
    db_path: str = "data/databases/intraday_accumulated.db"
    backup_path: str = "data/backup/intraday"
    max_days_per_ticker: int = 365  # Максимально днandв for withберandгання
    
    # Налаштування накопичення
    accumulation_interval_hours: int = 6  # Інтервал накопичення (години)
    batch_size: int = 20  # Роwithмandр батчу for обробки
    enable_compression: bool = True  # Стисnotння data
    
    # Якandсть data
    enable_validation: bool = True
    min_data_quality_threshold: float = 0.8
    max_missing_percentage: float = 5.0
    
    # Монandторинг
    enable_monitoring: bool = True
    save_statistics: bool = True
    alert_on_errors: bool = True

class IntradayAccumulator:
    """Накопичувач data for коротких candles"""
    
    def __init__(self, config: AccumulationConfig = None):
        self.config = config or AccumulationConfig()
        self.logger = logging.getLogger("IntradayAccumulator")
        self.performance_tracker = PerformanceTracker()
        
        # Створюємо директорandї
        Path(self.config.backup_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Інandцandалandwithуємо колектор
        self.collector = YFCollector()
        
        # Інandцandалandwithуємо баwithу data
        self._init_database()
        
        # Сandтистика накопичення
        self.accumulation_stats = {
            "total_tickers": 0,
            "total_records": 0,
            "last_accumulation": None,
            "accumulation_history": [],
            "data_quality_scores": {}
        }
    
    def _init_database(self):
        """Інandцandалandwithувати баwithу data"""
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблиця for накопичених data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intraday_accumulated (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    data_source TEXT DEFAULT 'yahoo_finance',
                    quality_score REAL,
                    UNIQUE(ticker, interval, timestamp)
                )
            """)
            
            # Інwhereкси for quicklyго доступу
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_interval 
                ON intraday_accumulated(ticker, interval)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON intraday_accumulated(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON intraday_accumulated(created_at)
            """)
            
            # Таблиця for сandтистики
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accumulation_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    accumulation_date DATETIME NOT NULL,
                    ticker TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    records_added INTEGER,
                    records_updated INTEGER,
                    data_quality_score REAL,
                    processing_time_seconds REAL,
                    errors TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            self.logger.info("Database initialized successfully")
    
    def accumulate_ticker_data(self, ticker: str, interval: str = "15m", 
                              days_back: int = 60) -> Dict[str, Any]:
        """
        Накопичити данand for одного тandкера
        
        Args:
            ticker: Символ тandкера
            interval: Інтервал (15m, 60m)
            days_back: Кandлькandсть днandв for withбору
            
        Returns:
            Dict[str, Any]: Реwithульandти накопичення
        """
        start_time = time.time()
        self.logger.info(f"Starting accumulation for {ticker} {interval}")
        
        try:
            # Збираємо данand
            data = self.collector.collect_data([ticker], interval, f"{days_back}d")
            
            if data.empty:
                self.logger.warning(f"No data collected for {ticker} {interval}")
                return {
                    "status": "no_data",
                    "ticker": ticker,
                    "interval": interval,
                    "records": 0
                }
            
            # Обробляємо данand
            processed_data = self._process_data(data, ticker, interval)
            
            # Зберandгаємо в баwithу data
            save_result = self._save_to_database(processed_data, ticker, interval)
            
            # Оновлюємо сandтистику
            processing_time = time.time() - start_time
            self._update_statistics(ticker, interval, save_result, processing_time)
            
            # Створюємо бекап
            if self.config.enable_compression:
                self._create_backup(ticker, interval, processed_data)
            
            result = {
                "status": "success",
                "ticker": ticker,
                "interval": interval,
                "records_collected": len(data),
                "records_saved": save_result["records_saved"],
                "records_updated": save_result["records_updated"],
                "processing_time": processing_time,
                "data_quality": save_result["quality_score"]
            }
            
            self.logger.info(f"Accumulation completed for {ticker} {interval}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error accumulating {ticker} {interval}: {e}")
            return {
                "status": "error",
                "ticker": ticker,
                "interval": interval,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _process_data(self, data: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
        """Обробити данand перед withбереженням"""
        # Перейменовуємо колонки якщо потрandбно
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in data.columns]
        
        # Перевandряємо наявнandсть notобхandдних колонок
        required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}")
            # Спробуємо альтернативнand наwithви
            col_mapping = {
                'Date': 'Datetime',
                'date': 'Datetime',
                'datetime': 'Datetime',
                'Open_': 'Open',
                'High_': 'High',
                'Low_': 'Low',
                'Close_': 'Close',
                'Volume_': 'Volume'
            }
            
            for missing_col in missing_cols:
                for alt_name, target_name in col_mapping.items():
                    if alt_name in data.columns:
                        data = data.rename(columns={alt_name: target_name})
                        break
        
        # Перевandряємо якandсть data
        quality_score = self._calculate_data_quality(data)
        
        # Додаємо меandданand
        data['ticker'] = ticker
        data['interval'] = interval
        data['quality_score'] = quality_score
        
        # Очищення data
        if self.config.enable_validation:
            data = self._validate_data(data)
        
        return data
    
    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Роwithрахувати якandсть data"""
        quality_score = 1.0
        
        # Перевandряємо пропуски
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                missing_pct = data[col].isna().sum() / len(data) * 100
                if missing_pct > self.config.max_missing_percentage:
                    quality_score -= 0.2
                else:
                    quality_score -= missing_pct / 100 * 0.2
        
        # Перевandряємо логandчну послandдовнandсть OHLC
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High >= Low
            invalid_ohlc = (data['High'] < data['Low']).sum()
            if invalid_ohlc > 0:
                quality_score -= min(invalid_ohlc / len(data) * 0.5, 0.3)
            
            # High >= Open, Close
            invalid_hl = ((data['High'] < data['Open']) | (data['High'] < data['Close'])).sum()
            if invalid_hl > 0:
                quality_score -= min(invalid_hl / len(data) * 0.3, 0.2)
            
            # Low <= Open, Close
            invalid_ll = ((data['Low'] > data['Open']) | (data['Low'] > data['Close'])).sum()
            if invalid_ll > 0:
                quality_score -= min(invalid_ll / len(data) * 0.3, 0.2)
        
        # Перевandряємо об'єм
        if 'Volume' in data.columns:
            zero_volume = (data['Volume'] <= 0).sum()
            if zero_volume > 0:
                quality_score -= min(zero_volume / len(data) * 0.1, 0.1)
        
        return max(0.0, quality_score)
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Валandдацandя and очищення data"""
        # Видаляємо рядки with критичними errorми
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # Видаляємо рядки where High < Low
            data = data[data['High'] >= data['Low']]
            
            # Коригуємо notwithначнand помилки
            data['High'] = np.maximum(data['High'], data['Open'])
            data['High'] = np.maximum(data['High'], data['Close'])
            data['Low'] = np.minimum(data['Low'], data['Open'])
            data['Low'] = np.minimum(data['Low'], data['Close'])
        
        # Видаляємо рядки with нульовим об'ємом (якщо це not вихandдний)
        if 'Volume' in data.columns:
            data = data[data['Volume'] > 0]
        
        # Сортуємо for часом
        if 'Datetime' in data.columns:
            data = data.sort_values('Datetime')
        
        return data
    
    def _save_to_database(self, data: pd.DataFrame, ticker: str, interval: str) -> Dict[str, Any]:
        """Зберегти данand в баwithу data"""
        records_saved = 0
        records_updated = 0
        quality_score = 0.0
        
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            
            for _, row in data.iterrows():
                try:
                    # Пandдготовка data
                    timestamp = row['Datetime'] if 'Datetime' in row else row.get('timestamp', row.get('date'))
                    
                    # Перевandряємо чи andснує forпис
                    cursor.execute("""
                        SELECT id FROM intraday_accumulated 
                        WHERE ticker = ? AND interval = ? AND timestamp = ?
                    """, (ticker, interval, timestamp))
                    
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Оновлюємо andснуючий forпис
                        cursor.execute("""
                            UPDATE intraday_accumulated 
                            SET open = ?, high = ?, low = ?, close = ?, volume = ?, 
                                quality_score = ?, created_at = CURRENT_TIMESTAMP
                            WHERE ticker = ? AND interval = ? AND timestamp = ?
                        """, (
                            float(row.get('Open', 0)),
                            float(row.get('High', 0)),
                            float(row.get('Low', 0)),
                            float(row.get('Close', 0)),
                            int(row.get('Volume', 0)),
                            float(row.get('quality_score', 1.0)),
                            ticker, interval, timestamp
                        ))
                        records_updated += 1
                    else:
                        # Всandвляємо новий forпис
                        cursor.execute("""
                            INSERT INTO intraday_accumulated 
                            (ticker, interval, timestamp, open, high, low, close, volume, quality_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            ticker, interval, timestamp,
                            float(row.get('Open', 0)),
                            float(row.get('High', 0)),
                            float(row.get('Low', 0)),
                            float(row.get('Close', 0)),
                            int(row.get('Volume', 0)),
                            float(row.get('quality_score', 1.0))
                        ))
                        records_saved += 1
                    
                    # Накоплюємо якandсть
                    quality_score += float(row.get('quality_score', 1.0))
                    
                except Exception as e:
                    self.logger.warning(f"Error saving row for {ticker}: {e}")
                    continue
            
            conn.commit()
        
        # Середня якandсть
        avg_quality = quality_score / len(data) if len(data) > 0 else 0.0
        
        return {
            "records_saved": records_saved,
            "records_updated": records_updated,
            "quality_score": avg_quality
        }
    
    def _create_backup(self, ticker: str, interval: str, data: pd.DataFrame):
        """Create бекап data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_{interval}_{timestamp}.parquet"
            filepath = Path(self.config.backup_path) / filename
            
            # Зберandгаємо данand
            data.to_parquet(filepath)
            
            # Обмежуємо кandлькandсть бекапandв
            self._cleanup_old_backups(ticker, interval)
            
            self.logger.info(f"Backup created: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error creating backup for {ticker} {interval}: {e}")
    
    def _cleanup_old_backups(self, ticker: str, interval: str, max_backups: int = 10):
        """Очистити сandрand бекапи"""
        try:
            backup_pattern = f"{ticker}_{interval}_*.parquet"
            backup_dir = Path(self.config.backup_path)
            
            backup_files = list(backup_dir.glob(backup_pattern))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Видаляємо сandрand fileи
            for old_backup in backup_files[max_backups:]:
                old_backup.unlink()
                self.logger.debug(f"Removed old backup: {old_backup}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up backups: {e}")
    
    def _update_statistics(self, ticker: str, interval: str, result: Dict[str, Any], processing_time: float):
        """Оновити сandтистику накопичення"""
        # Додаємо forпис в andсторandю
        self.accumulation_stats["accumulation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "interval": interval,
            "status": result["status"],
            "records": result.get("records_saved", 0),
            "processing_time": processing_time,
            "quality_score": result.get("data_quality", 0.0)
        })
        
        # Обмежуємо andсторandю
        if len(self.accumulation_stats["accumulation_history"]) > 1000:
            self.accumulation_stats["accumulation_history"] = self.accumulation_stats["accumulation_history"][-1000:]
        
        # Оновлюємо forгальнand покаwithники
        self.accumulation_stats["last_accumulation"] = datetime.now().isoformat()
        
        # Зберandгаємо в баwithу data
        if self.config.save_statistics:
            self._save_statistics_to_db(ticker, interval, result, processing_time)
    
    def _save_statistics_to_db(self, ticker: str, interval: str, result: Dict[str, Any], processing_time: float):
        """Зберегти сandтистику в баwithу data"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO accumulation_statistics 
                    (accumulation_date, ticker, interval, records_added, records_updated, 
                     data_quality_score, processing_time_seconds, errors)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    ticker,
                    interval,
                    result.get("records_saved", 0),
                    result.get("records_updated", 0),
                    result.get("data_quality", 0.0),
                    processing_time,
                    result.get("error") if result["status"] == "error" else None
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving statistics: {e}")
    
    def accumulate_multiple_tickers(self, tickers: List[str], intervals: List[str] = None) -> Dict[str, Any]:
        """
        Накопичити данand for кandлькох тandкерandв
        
        Args:
            tickers: Список тandкерandв
            intervals: Список andнтервалandв
            
        Returns:
            Dict[str, Any]: Реwithульandти накопичення
        """
        if intervals is None:
            intervals = ["15m", "60m"]
        
        self.logger.info(f"Starting accumulation for {len(tickers)} tickers, intervals: {intervals}")
        
        results = {}
        total_start_time = time.time()
        
        for ticker in tickers:
            for interval in intervals:
                key = f"{ticker}_{interval}"
                
                try:
                    result = self.accumulate_ticker_data(ticker, interval)
                    results[key] = result
                    
                    # Невелика forтримка мandж forпиandми
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error in batch accumulation for {key}: {e}")
                    results[key] = {
                        "status": "error",
                        "ticker": ticker,
                        "interval": interval,
                        "error": str(e)
                    }
        
        # Пandдсумок
        total_time = time.time() - total_start_time
        successful = sum(1 for r in results.values() if r["status"] == "success")
        failed = len(results) - successful
        
        summary = {
            "total_requests": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(results) if results else 0,
            "total_time": total_time,
            "average_time_per_request": total_time / len(results) if results else 0,
            "results": results
        }
        
        self.logger.info(f"Batch accumulation completed: {successful}/{len(results)} successful")
        
        return summary
    
    def get_accumulation_status(self) -> Dict[str, Any]:
        """Отримати сandтус накопичення"""
        with sqlite3.connect(self.config.db_path) as conn:
            cursor = conn.cursor()
            
            # Загальна сandтистика
            cursor.execute("""
                SELECT COUNT(*) as total_records,
                       COUNT(DISTINCT ticker) as unique_tickers,
                       COUNT(DISTINCT interval) as unique_intervals,
                       MIN(timestamp) as earliest_date,
                       MAX(timestamp) as latest_date,
                       AVG(quality_score) as avg_quality
                FROM intraday_accumulated
            """)
            
            stats = cursor.fetchone()
            
            # Сandтистика по тandкерах
            cursor.execute("""
                SELECT ticker, interval, COUNT(*) as records,
                       AVG(quality_score) as avg_quality,
                       MIN(timestamp) as earliest,
                       MAX(timestamp) as latest
                FROM intraday_accumulated
                GROUP BY ticker, interval
                ORDER BY ticker, interval
            """)
            
            ticker_stats = cursor.fetchall()
            
            # Сandтистика накопичення
            cursor.execute("""
                SELECT DATE(accumulation_date) as date,
                       COUNT(*) as accumulations,
                       SUM(records_added) as total_records,
                       AVG(data_quality_score) as avg_quality
                FROM accumulation_statistics
                WHERE accumulation_date >= DATE('now', '-30 days')
                GROUP BY DATE(accumulation_date)
                ORDER BY date DESC
            """)
            
            recent_stats = cursor.fetchall()
        
        return {
            "database_stats": {
                "total_records": stats[0] if stats else 0,
                "unique_tickers": stats[1] if stats else 0,
                "unique_intervals": stats[2] if stats else 0,
                "earliest_date": stats[3] if stats else None,
                "latest_date": stats[4] if stats else None,
                "average_quality": stats[5] if stats else 0
            },
            "ticker_statistics": [
                {
                    "ticker": row[0],
                    "interval": row[1],
                    "records": row[2],
                    "avg_quality": row[3],
                    "earliest": row[4],
                    "latest": row[5]
                }
                for row in ticker_stats
            ],
            "recent_accumulations": [
                {
                    "date": row[0],
                    "accumulations": row[1],
                    "total_records": row[2],
                    "avg_quality": row[3]
                }
                for row in recent_stats
            ]
        }
    
    def export_accumulated_data(self, ticker: str = None, interval: str = None, 
                              output_path: str = None) -> str:
        """
        Експортувати накопиченand данand
        
        Args:
            ticker: Фandльтр по тandкеру
            interval: Фandльтр по andнтервалу
            output_path: Шлях for експорту
            
        Returns:
            str: Шлях до експортованого fileу
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accumulated_data_{timestamp}.parquet"
            output_path = f"data/export/{filename}"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Формуємо SQL forпит
        query = "SELECT * FROM intraday_accumulated WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        if interval:
            query += " AND interval = ?"
            params.append(interval)
        
        query += " ORDER BY ticker, interval, timestamp"
        
        # Заванandжуємо данand
        with sqlite3.connect(self.config.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        # Експортуємо
        df.to_parquet(output_path)
        
        self.logger.info(f"Exported {len(df)} records to {output_path}")
        
        return output_path

def main():
    """Основна функцandя for тестування"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Intraday Data Accumulator')
    parser.add_argument('--tickers', default='core', help='Ticker category or list')
    parser.add_argument('--intervals', nargs='+', default=['15m', '60m'], help='Intervals')
    parser.add_argument('--status', action='store_true', help='Show accumulation status')
    parser.add_argument('--export', help='Export data to file')
    parser.add_argument('--ticker', help='Specific ticker for export')
    parser.add_argument('--interval', help='Specific interval for export')
    
    args = parser.parse_args()
    
    # Налаштування logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Отримуємо тandкери
    try:
        from config.tickers import get_tickers
        if args.tickers == 'core':
            tickers = get_tickers('core')
        elif args.tickers == 'all':
            tickers = get_tickers('all')[:10]  # Обмежуємо for тесту
        else:
            tickers = get_tickers(args.tickers)
    except ImportError:
        tickers = ['SPY', 'QQQ', 'NVDA']
    
    # Створюємо накопичувач
    accumulator = IntradayAccumulator()
    
    if args.status:
        status = accumulator.get_accumulation_status()
        print(f"\n=== Accumulation Status ===")
        print(f"Total records: {status['database_stats']['total_records']}")
        print(f"Unique tickers: {status['database_stats']['unique_tickers']}")
        print(f"Unique intervals: {status['database_stats']['unique_intervals']}")
        print(f"Date range: {status['database_stats']['earliest_date']} to {status['database_stats']['latest_date']}")
        print(f"Average quality: {status['database_stats']['average_quality']:.3f}")
        
        print(f"\n=== Ticker Statistics ===")
        for ticker_stat in status['ticker_statistics'][:10]:  # Покаwithуємо першand 10
            print(f"{ticker_stat['ticker']} {ticker_stat['interval']}: {ticker_stat['records']} records")
        
        return
    
    if args.export:
        output_path = accumulator.export_accumulated_data(args.ticker, args.interval, args.export)
        print(f"Data exported to: {output_path}")
        return
    
    # Накопичуємо данand
    print(f"Starting accumulation for {len(tickers)} tickers: {tickers}")
    print(f"Intervals: {args.intervals}")
    
    results = accumulator.accumulate_multiple_tickers(tickers, args.intervals)
    
    print(f"\n=== Accumulation Results ===")
    print(f"Total requests: {results['total_requests']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Average time per request: {results['average_time_per_request']:.2f} seconds")
    
    # Покаwithуємо whereandльнand реwithульandти
    print(f"\n=== Detailed Results ===")
    for key, result in results['results'].items():
        status_emoji = "[OK]" if result['status'] == 'success' else "[ERROR]"
        print(f"{status_emoji} {key}: {result.get('records_saved', 0)} records saved")

if __name__ == "__main__":
    main()
