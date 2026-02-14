# core/data_accumulator.py

import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

from config.config import PATHS, TICKERS
import logging

logger = logging.getLogger(__name__)

class DataAccumulator:
    """Manages data collection and storage for the trading system"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(PATHS["data"], "trading_data.db")
        self.ensure_database()
    
    def ensure_database(self):
        """Create database and tables if they don't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Prices table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, interval, datetime)
                )
            """)
            
            # News table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    published_at TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    source TEXT,
                    url TEXT UNIQUE,
                    ticker TEXT,
                    sentiment_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Macro data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    value REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, series_id)
                )
            """)
            
            # Features table (processed features)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    date TEXT NOT NULL,
                    feature_data TEXT,  -- JSON string of features
                    target_direction INTEGER,
                    target_pct_change REAL,
                    target_close REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, interval, date)
                )
            """)
            
            # Model results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    mae REAL,
                    r2 REAL,
                    accuracy REAL,
                    f1_score REAL,
                    trained_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info(f"[DataAccumulator] Database initialized: {self.db_path}")
    
    def get_last_price_date(self, ticker: str, interval: str) -> Optional[datetime]:
        """Get the last date we have price data for"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT MAX(datetime) FROM prices WHERE ticker = ? AND interval = ?",
                (ticker, interval)
            )
            result = cursor.fetchone()[0]
            return datetime.fromisoformat(result) if result else None
    
    def get_last_news_date(self) -> Optional[datetime]:
        """Get the last date we have news data for"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT MAX(published_at) FROM news")
            result = cursor.fetchone()[0]
            return datetime.fromisoformat(result) if result else None
    
    def store_prices(self, df: pd.DataFrame):
        """Store price data in database"""
        if df.empty:
            return
        
        # Ensure required columns
        required_cols = ['ticker', 'interval', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"[DataAccumulator] Missing columns in price data: {missing_cols}")
            return
        
        with sqlite3.connect(self.db_path) as conn:
            df[required_cols].to_sql('prices', conn, if_exists='append', index=False)
            logger.info(f"[DataAccumulator] Stored {len(df)} price records")
    
    def store_news(self, df: pd.DataFrame):
        """Store news data in database"""
        if df.empty:
            return
        
        required_cols = ['published_at', 'title', 'description', 'source', 'url']
        df_clean = df[required_cols + [col for col in ['ticker', 'sentiment_score'] if col in df.columns]]
        
        with sqlite3.connect(self.db_path) as conn:
            df_clean.to_sql('news', conn, if_exists='append', index=False)
            logger.info(f"[DataAccumulator] Stored {len(df_clean)} news records")
    
    def store_macro_data(self, df: pd.DataFrame):
        """Store macro economic data"""
        if df.empty:
            return
        
        # Convert wide format to long format
        if 'date' in df.columns:
            df_long = df.melt(id_vars=['date'], var_name='series_id', value_name='value')
            df_long = df_long.dropna(subset=['value'])
            
            with sqlite3.connect(self.db_path) as conn:
                df_long.to_sql('macro_data', conn, if_exists='append', index=False)
                logger.info(f"[DataAccumulator] Stored {len(df_long)} macro records")
    
    def store_features(self, df: pd.DataFrame):
        """Store processed features"""
        if df.empty:
            return
        
        import json
        
        # Extract target columns
        target_cols = [col for col in df.columns if col.startswith('target_')]
        feature_cols = [col for col in df.columns if not col.startswith('target_') 
                       and col not in ['ticker', 'interval', 'date', 'trade_date']]
        
        records = []
        for _, row in df.iterrows():
            # Create feature JSON
            features = {col: row[col] for col in feature_cols if pd.notna(row[col])}
            
            record = {
                'ticker': row.get('ticker', 'GENERAL'),
                'interval': row.get('interval', '1d'),
                'date': row.get('trade_date', row.get('date', datetime.now())).isoformat(),
                'feature_data': json.dumps(features),
                'target_direction': row.get('target_direction_spy_1d'),
                'target_pct_change': row.get('target_pct_change_spy_1d'),
                'target_close': row.get('target_close_spy_1d')
            }
            records.append(record)
        
        if records:
            with sqlite3.connect(self.db_path) as conn:
                pd.DataFrame(records).to_sql('features', conn, if_exists='append', index=False)
                logger.info(f"[DataAccumulator] Stored {len(records)} feature records")
    
    def store_model_results(self, model_name: str, ticker: str, interval: str, 
                          target_type: str, metrics: Dict):
        """Store model training results"""
        record = {
            'model_name': model_name,
            'ticker': ticker,
            'interval': interval,
            'target_type': target_type,
            'mae': metrics.get('mae'),
            'r2': metrics.get('r2'),
            'accuracy': metrics.get('accuracy'),
            'f1_score': metrics.get('F1')
        }
        
        with sqlite3.connect(self.db_path) as conn:
            pd.DataFrame([record]).to_sql('model_results', conn, if_exists='append', index=False)
            logger.info(f"[DataAccumulator] Stored model result: {model_name} on {ticker}")
    
    def get_data_summary(self) -> Dict:
        """Get summary of stored data"""
        with sqlite3.connect(self.db_path) as conn:
            summary = {}
            
            # Price data summary
            cursor = conn.execute("""
                SELECT ticker, interval, COUNT(*) as count, 
                       MIN(datetime) as first_date, MAX(datetime) as last_date
                FROM prices GROUP BY ticker, interval
            """)
            summary['prices'] = cursor.fetchall()
            
            # News summary
            cursor = conn.execute("""
                SELECT COUNT(*) as total_news, 
                       MIN(published_at) as first_news, MAX(published_at) as last_news
                FROM news
            """)
            summary['news'] = cursor.fetchone()
            
            # Features summary
            cursor = conn.execute("""
                SELECT ticker, interval, COUNT(*) as count
                FROM features GROUP BY ticker, interval
            """)
            summary['features'] = cursor.fetchall()
            
            return summary
    
    def needs_data_update(self, ticker: str, interval: str) -> bool:
        """Check if we need to update data for a ticker/interval"""
        last_date = self.get_last_price_date(ticker, interval)
        if not last_date:
            return True
        
        # Check if data is older than update frequency (default 1h)
        threshold = datetime.now() - timedelta(hours=1)
        
        return last_date < threshold

# Global instance
data_accumulator = DataAccumulator()