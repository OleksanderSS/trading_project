# utils/data_viewer.py

import pandas as pd
from typing import Dict, List, Optional
try:
    from core.data_accumulator import data_accumulator
except ImportError:
    data_accumulator = None
from utils.logger import ProjectLogger
import logging


# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger = ProjectLogger.get_logger("DataViewer")

class DataViewer:
    """Utility for viewing and analyzing accumulated data"""
    
    def __init__(self):
        self.accumulator = data_accumulator
    
    def show_summary(self) -> Dict:
        """Show summary of all stored data"""
        summary = self.accumulator.get_data_summary()
        
        print("\n" + "="*50)
        print("[DATA] DATA SUMMARY")
        print("="*50)
        
        # Price data
        print("\n[MONEY] PRICE DATA:")
        if summary.get('prices'):
            for ticker, interval, count, first, last in summary['prices']:
                print(f"  {ticker} ({interval}): {count:,} records from {first} to {last}")
        else:
            print("  No price data found")
        
        # News data
        print("\n NEWS DATA:")
        news_info = summary.get('news')
        if news_info and news_info[0]:
            total, first, last = news_info
            print(f"  Total: {total:,} articles from {first} to {last}")
        else:
            print("  No news data found")
        
        # Features data
        print("\n[TOOL] FEATURES DATA:")
        if summary.get('features'):
            for ticker, interval, count in summary['features']:
                print(f"  {ticker} ({interval}): {count:,} feature records")
        else:
            print("  No features data found")
        
        print("="*50)
        return summary
    
    def get_model_performance(self, limit: int = 10) -> pd.DataFrame:
        """Get recent model performance results"""
        import sqlite3
        
        with sqlite3.connect(self.accumulator.db_path) as conn:
            query = """
                SELECT model_name, ticker, interval, target_type,
                       mae, r2, accuracy, f1_score, trained_at
                FROM model_results 
                ORDER BY trained_at DESC 
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
            
        if not df.empty:
            print(f"\n RECENT MODEL RESULTS (last {len(df)}):")
            print("-" * 80)
            for _, row in df.iterrows():
                metrics = []
                if pd.notna(row['mae']): metrics.append(f"MAE: {row['mae']:.4f}")
                if pd.notna(row['r2']): metrics.append(f"R2: {row['r2']:.4f}")
                if pd.notna(row['accuracy']): metrics.append(f"Acc: {row['accuracy']:.4f}")
                if pd.notna(row['f1_score']): metrics.append(f"F1: {row['f1_score']:.4f}")
                
                print(f"{row['model_name']:>8} | {row['ticker']:>4} | {row['interval']:>3} | "
                      f"{row['target_type']:>12} | {' | '.join(metrics)}")
        else:
            print("\n No model results found")
        
        return df
    
    def get_best_models(self) -> pd.DataFrame:
        """Get best performing models by ticker/interval"""
        import sqlite3
        
        with sqlite3.connect(self.accumulator.db_path) as conn:
            # For classification tasks (use F1 score)
            query_class = """
                SELECT ticker, interval, model_name, target_type, f1_score as score, 'f1' as metric
                FROM model_results 
                WHERE target_type = 'classification' AND f1_score IS NOT NULL
                ORDER BY ticker, interval, f1_score DESC
            """
            
            # For regression tasks (use R2 score)
            query_reg = """
                SELECT ticker, interval, model_name, target_type, r2 as score, 'r2' as metric
                FROM model_results 
                WHERE target_type = 'regression' AND r2 IS NOT NULL
                ORDER BY ticker, interval, r2 DESC
            """
            
            df_class = pd.read_sql_query(query_class, conn)
            df_reg = pd.read_sql_query(query_reg, conn)
            
            # Get best model for each ticker/interval combination
            best_models = []
            
            for df, task_type in [(df_class, 'classification'), (df_reg, 'regression')]:
                if not df.empty:
                    best = df.groupby(['ticker', 'interval']).first().reset_index()
                    best_models.append(best)
            
            if best_models:
                result = pd.concat(best_models, ignore_index=True)
                
                print(f"\n[BEST] BEST MODELS BY TICKER/INTERVAL:")
                print("-" * 60)
                for _, row in result.iterrows():
                    print(f"{row['ticker']:>4} | {row['interval']:>3} | {row['model_name']:>12} | "
                          f"{row['target_type']:>12} | {row['metric'].upper()}: {row['score']:.4f}")
                
                return result
            else:
                print("\n[BEST] No model results available for comparison")
                return pd.DataFrame()
    
    def get_data_gaps(self) -> Dict:
        """Identify gaps in data collection"""
        import sqlite3
        from datetime import datetime, timedelta
        
        gaps = {}
        
        with sqlite3.connect(self.accumulator.db_path) as conn:
            # Check for missing recent data
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            query = """
                SELECT ticker, interval, MAX(datetime) as last_date
                FROM prices 
                GROUP BY ticker, interval
                HAVING last_date < ?
            """
            
            cursor = conn.execute(query, (yesterday,))
            stale_data = cursor.fetchall()
            
            if stale_data:
                gaps['stale_price_data'] = stale_data
                print(f"\n[WARN]  STALE PRICE DATA (older than {yesterday}):")
                for ticker, interval, last_date in stale_data:
                    print(f"  {ticker} ({interval}): last update {last_date}")
            
            # Check for missing tickers
            from config.config import TICKERS
            query = "SELECT DISTINCT ticker FROM prices"
            cursor = conn.execute(query)
            stored_tickers = {row[0] for row in cursor.fetchall()}
            missing_tickers = set(TICKERS.keys()) - stored_tickers
            
            if missing_tickers:
                gaps['missing_tickers'] = list(missing_tickers)
                print(f"\n[WARN]  MISSING TICKERS: {missing_tickers}")
        
        return gaps

# Global instance
data_viewer = DataViewer()

def show_data_status():
    """Quick function to show data status"""
    try:
        data_viewer.show_summary()
        data_viewer.get_model_performance(5)
        data_viewer.get_best_models()
        data_viewer.get_data_gaps()
    except Exception as e:
        # Fallback simple status
        import os
        print("\n[DATA] SIMPLE DATA STATUS:")
        data_files = {
            "Trading Data": "data/trading_data.db",
            "News Data": "data/news.db", 
            "Features": "data/stage3_features.parquet",
            "Models": "models/trained"
        }
        
        for name, path in data_files.items():
            if os.path.exists(path):
                if os.path.isdir(path):
                    count = len([f for f in os.listdir(path) if f.endswith('.pkl')])
                    print(f"  [OK] {name}: {count} files")
                else:
                    size = os.path.getsize(path) / (1024*1024)
                    print(f"  [OK] {name}: {size:.1f} MB")
            else:
                print(f"  [ERROR] {name}: Missing")
        
        logger.warning(f"Data viewer fallback used: {e}")

if __name__ == "__main__":
    show_data_status()