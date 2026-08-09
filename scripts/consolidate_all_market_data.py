#!/usr/bin/env python3
"""
Consolidate, Backfill, and Audit All Market Data
Covers 24 tickers across 3 timeframes (15m, 1h, 1d) into trading_data.duckdb.
"""

import sys
import os
import glob
import duckdb
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Windows console encoding configuration
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

TICKERS = [
    'AAPL', 'AMD', 'AMZN', 'BAC', 'GOOGL', 'GS', 'INTC', 'IWM', 'JPM', 'KO', 
    'MOO', 'MSFT', 'NVDA', 'QQQ', 'SPY', 'TSLA', 'TSM', 'WMT', 'XHB', 'XLE', 
    'XLF', 'XLK', 'XLV', 'XOM'
]

TIMEFRAMES = ['15m', '1h', '1d']
MAIN_DB = 'data/trading_data.duckdb'

BACKUP_DBS = [
    'data/trading_data.duckdb',
    'data/raw_data.duckdb.backup',
    'data/trading_data.duckdb.backup',
    'data/trading_data_snapshot.duckdb'
]

def load_existing_db_records() -> pd.DataFrame:
    """Consolidate market data from all DuckDB files and standardize 60m -> 1h."""
    frames = []
    
    for db_path in BACKUP_DBS:
        if not os.path.exists(db_path):
            continue
        try:
            conn = duckdb.connect(db_path, read_only=True)
            tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
            
            for t in tables:
                if 'market_data' in t:
                    cols = [c[0] for c in conn.execute(f"PRAGMA table_info('{t}')").fetchall()]
                    date_col = 'datetime' if 'datetime' in cols else ('timestamp' if 'timestamp' in cols else None)
                    
                    if date_col and 'ticker' in cols and 'interval' in cols:
                        query = f"""
                        SELECT ticker, 
                               CASE WHEN interval = '60m' THEN '1h' ELSE interval END as interval,
                               {date_col} as datetime,
                               open, high, low, close, volume
                        FROM '{t}'
                        WHERE ticker IN ({','.join(f"'{tk}'" for tk in TICKERS)})
                        """
                        df = conn.execute(query).df()
                        if not df.empty:
                            frames.append(df)
            conn.close()
        except Exception as e:
            print(f"⚠️ Notice reading {db_path}: {e}")
            
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        return combined
    return pd.DataFrame()

def backfill_via_yfinance() -> pd.DataFrame:
    """Backfill 15m (60d), 1h (730d), and 1d (max) for all 24 tickers."""
    downloaded = []
    print("\n📥 Fetching fresh backfill data for 24 tickers from yfinance...")
    
    tf_params = [
        ('15m', '60d'),
        ('1h', '730d'),
        ('1d', 'max')
    ]
    
    for tf, period in tf_params:
        print(f"   Fetching timeframe: {tf} (period: {period})...")
        for ticker in TICKERS:
            try:
                df = yf.download(ticker, period=period, interval=tf, progress=False)
                if df.empty:
                    continue
                
                # Flatten multi-index columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = [col.lower() for col in df.columns]
                    
                df = df.reset_index()
                date_col = 'Datetime' if 'Datetime' in df.columns else ('Date' if 'Date' in df.columns else df.columns[0])
                df = df.rename(columns={date_col: 'datetime'})
                
                df['ticker'] = ticker
                df['interval'] = tf
                
                # Ensure essential columns exist
                required = ['ticker', 'interval', 'datetime', 'open', 'high', 'low', 'close', 'volume']
                df = df[required]
                downloaded.append(df)
            except Exception as e:
                print(f"     ⚠️ Fetch error for {ticker} [{tf}]: {e}")
                
    if downloaded:
        return pd.concat(downloaded, ignore_index=True)
    return pd.DataFrame()

def save_to_duckdb(df: pd.DataFrame):
    """Write cleaned, deduplicated dataset to primary DuckDB database."""
    print(f"\n💾 Saving {len(df)} consolidated rows into {MAIN_DB}...")
    
    conn = duckdb.connect(MAIN_DB)
    
    # Create temp table
    conn.register('temp_consolidated', df)
    
    # Ensure market_data_raw table structure exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS market_data_raw (
        ticker VARCHAR,
        interval VARCHAR,
        datetime TIMESTAMPTZ,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        PRIMARY KEY (ticker, interval, datetime)
    );
    """)
    
    # Insert or replace
    conn.execute("""
    INSERT OR REPLACE INTO market_data_raw (ticker, interval, datetime, open, high, low, close, volume)
    SELECT ticker, interval, datetime, open, high, low, close, volume
    FROM temp_consolidated;
    """)
    
    conn.close()
    print("✅ Successfully updated market_data_raw in DuckDB.")

def audit_dataset_health():
    """Print comprehensive summary and health metrics of the consolidated dataset."""
    conn = duckdb.connect(MAIN_DB, read_only=True)
    
    query = """
    SELECT 
        ticker,
        interval,
        COUNT(*) as total_candles,
        MIN(datetime) as start_date,
        MAX(datetime) as end_date,
        SUM(CASE WHEN volume = 0 THEN 1 ELSE 0 END) as zero_volume_count,
        SUM(CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL THEN 1 ELSE 0 END) as null_price_count
    FROM market_data_raw
    GROUP BY ticker, interval
    ORDER BY interval, ticker;
    """
    
    summary_df = conn.execute(query).df()
    conn.close()
    
    print("\n" + "=" * 80)
    print("📊 DATA CONSOLIDATION & QUALITY HEALTH REPORT")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    # Overall statistics
    total_candles = summary_df['total_candles'].sum()
    print(f"\n📈 GRAND TOTAL: {total_candles:,} candles consolidated across 24 tickers.")
    print("=" * 80)

def main():
    print("=" * 80)
    print("🚀 MARKET DATA CONSOLIDATION & BACKFILL PIPELINE")
    print("=" * 80)
    
    # 1. Load existing historical data
    existing_df = load_existing_db_records()
    print(f"📦 Existing historical records loaded: {len(existing_df):,} rows.")
    
    # 2. Fetch fresh backfill data
    backfill_df = backfill_via_yfinance()
    print(f"🌐 Fresh backfill records fetched: {len(backfill_df):,} rows.")
    
    # 3. Combine and deduplicate
    combined_df = pd.concat([existing_df, backfill_df], ignore_index=True)
    
    if combined_df.empty:
        print("❌ No data available to consolidate.")
        return

    # Clean datetime timezone
    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'], utc=True)
    
    # Deduplicate by (ticker, interval, datetime)
    dedup_df = combined_df.drop_duplicates(subset=['ticker', 'interval', 'datetime'], keep='last')
    print(f"✨ Deduplicated dataset size: {len(dedup_df):,} rows.")
    
    # 4. Save to DuckDB
    save_to_duckdb(dedup_df)
    
    # 5. Audit final health
    audit_dataset_health()

if __name__ == '__main__':
    main()
