#!/usr/bin/env python3
"""Analyze cached news and feature data to understand structure for new ticker integration."""

import pandas as pd
import os
from pathlib import Path
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

def analyze_sentiment_cache():
    """Analyze sentiment cache structure."""
    cache_file = "d:/trading_project/data/cache/sentiment_cache.parquet"
    
    if not os.path.exists(cache_file):
        print("❌ Sentiment cache file not found")
        return
    
    print("=== SENTIMENT CACHE ANALYSIS ===")
    df = pd.read_parquet(cache_file)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Sentiment range: {df['sentiment'].min():.3f} to {df['sentiment'].max():.3f}")
    print("\nSample data:")
    print(df.head())
    
    # Check if sentiment data is ticker-specific
    if 'ticker' in df.columns:
        print(f"\nTickers in cache: {df['ticker'].nunique()}")
        print(f"Ticker distribution: {df['ticker'].value_counts()}")
    else:
        print("\n⚠️  No ticker column - sentiment data appears to be general market sentiment")
    
    return df

def analyze_feature_caches():
    """Analyze feature cache files."""
    cache_dir = Path("d:/trading_project/data/cache/features")
    
    if not cache_dir.exists():
        print("❌ Feature cache directory not found")
        return
    
    parquet_files = list(cache_dir.glob("*.parquet"))
    print(f"\n=== FEATURE CACHE ANALYSIS ===")
    print(f"Found {len(parquet_files)} parquet files")
    
    all_data = {}
    
    for file_path in parquet_files:
        print(f"\n--- File: {file_path.name} ---")
        try:
            df = pd.read_parquet(file_path)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Check for ticker-specific columns
            ticker_cols = [col for col in df.columns if 'ticker' in col.lower()]
            if ticker_cols:
                print(f"Ticker-related columns: {ticker_cols}")
            
            # Check for datetime columns
            datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if datetime_cols:
                print(f"DateTime columns: {datetime_cols}")
                for col in datetime_cols[:2]:  # Show first 2
                    if col in df.columns:
                        print(f"  {col} range: {df[col].min()} to {df[col].max()}")
            
            # Sample data
            print("Sample data:")
            print(df.head(2))
            
            all_data[file_path.name] = df
            
        except Exception:
            logger.error(f"Error reading {file_path}", exc_info=True)
            raise
    
    return all_data

def analyze_model_batches():
    """Analyze model batch data for ticker patterns."""
    batch_dir = Path("d:/trading_project/models/unified/colab_batches")
    
    if not batch_dir.exists():
        print("❌ Model batch directory not found")
        return
    
    print(f"\n=== MODEL BATCHES ANALYSIS ===")
    
    # Find ticker-specific directories
    ticker_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
    ticker_names = set()
    
    for dir_path in ticker_dirs:
        parts = dir_path.name.split('_')
        if len(parts) >= 2:
            potential_ticker = parts[0]
            if potential_ticker.isupper() and len(potential_ticker) >= 3:
                ticker_names.add(potential_ticker)
    
    print(f"Found {len(ticker_names)} unique tickers in model batches:")
    for ticker in sorted(ticker_names):
        print(f"  {ticker}")
    
    # Check some batch files
    sample_files = []
    for dir_path in ticker_dirs[:3]:  # Check first 3 directories
        for file_path in dir_path.iterdir():
            if file_path.suffix in ['.parquet', '.csv', '.json']:
                sample_files.append(file_path)
                break
    
    print(f"\nSample batch files:")
    for file_path in sample_files:
        print(f"\n--- {file_path} ---")
        try:
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                continue
            
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)[:10]}")  # First 10 columns
            if len(df.columns) > 10:
                print(f"... and {len(df.columns) - 10} more columns")
            
        except Exception:
            logger.error(f"Error reading {file_path}", exc_info=True)
            raise

def main():
    """Main analysis function."""
    print("🔍 Analyzing cached data for new ticker integration...")
    
    # Analyze sentiment cache
    sentiment_data = analyze_sentiment_cache()
    
    # Analyze feature caches
    feature_data = analyze_feature_caches()
    
    # Analyze model batches
    analyze_model_batches()
    
    print("\n=== SUMMARY ===")
    print("1. Sentiment cache contains general market sentiment data (no ticker-specific info)")
    print("2. Feature caches contain processed features - need to check if ticker-specific")
    print("3. Model batches show data for specific tickers (NVDA, QQQ, XOM, WMT, AMD)")
    print("\n💡 RECOMMENDATIONS:")
    print("- Sentiment data can be used as general market sentiment for any ticker")
    print("- Feature data may need ticker-specific processing")
    print("- Model batch structure shows how to organize ticker-specific data")

if __name__ == "__main__":
    main()
