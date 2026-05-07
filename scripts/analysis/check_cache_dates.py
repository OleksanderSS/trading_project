#!/usr/bin/env python3
"""Check date ranges of cached data to understand temporal alignment issues."""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_sentiment_cache():
    """Check sentiment cache date range."""
    cache_file = "d:/trading_project/data/cache/sentiment_cache.parquet"
    
    if not Path(cache_file).exists():
        logger.warning(f"Sentiment cache not found: {cache_file}")
        return None
    
    df = pd.read_parquet(cache_file)
    logger.info("=== SENTIMENT CACHE ANALYSIS ===")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Count: {len(df)}")
    logger.info(f"Unique dates: {df['timestamp'].dt.date.nunique()}")
    logger.info(f"Time span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    return df

def check_feature_caches():
    """Check feature cache date ranges."""
    cache_dir = Path("d:/trading_project/data/cache/features")
    
    if not cache_dir.exists():
        logger.warning(f"Feature cache directory not found: {cache_dir}")
        return {}
    
    parquet_files = list(cache_dir.glob("*.parquet"))
    logger.info("\n=== FEATURE CACHE ANALYSIS ===")
    
    results = {}
    
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            
            if 'datetime' in df.columns:
                date_range = f"{df['datetime'].min()} to {df['datetime'].max()}"
                time_span = (df['datetime'].max() - df['datetime'].min()).days
                unique_dates = df['datetime'].dt.date.nunique()
            else:
                date_range = "No datetime column"
                time_span = 0
                unique_dates = 0
            
            logger.info(f"File: {file_path.name}")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Date range: {date_range}")
            logger.info(f"  Time span: {time_span} days")
            logger.info(f"  Unique dates: {unique_dates}")
            
            if 'ticker' in df.columns:
                tickers = df['ticker'].unique()
                logger.info(f"  Tickers: {list(tickers)}")
            
            logger.info("")
            
            results[file_path.name] = {
                'shape': df.shape,
                'date_range': date_range,
                'time_span': time_span,
                'unique_dates': unique_dates,
                'tickers': df['ticker'].unique().tolist() if 'ticker' in df.columns else []
            }
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    
    return results

def check_model_batches():
    """Check model batch date ranges."""
    batch_dir = Path("d:/trading_project/models/unified/colab_batches")
    
    if not batch_dir.exists():
        logger.warning(f"Model batch directory not found: {batch_dir}")
        return {}
    
    logger.info("\n=== MODEL BATCHES ANALYSIS ===")
    
    results = {}
    batch_dirs = [d for d in batch_dir.iterdir() if d.is_dir()]
    
    for batch_dir_path in batch_dirs[:5]:  # Check first 5
        try:
            # Look for parquet files
            parquet_files = list(batch_dir_path.glob("*.parquet"))
            
            if parquet_files:
                for file_path in parquet_files:
                    df = pd.read_parquet(file_path)
                    
                    if 'datetime' in df.columns:
                        date_range = f"{df['datetime'].min()} to {df['datetime'].max()}"
                        time_span = (df['datetime'].max() - df['datetime'].min()).days
                    else:
                        date_range = "No datetime column"
                        time_span = 0
                    
                    logger.info(f"Batch: {batch_dir_path.name}")
                    logger.info(f"  File: {file_path.name}")
                    logger.info(f"  Shape: {df.shape}")
                    logger.info(f"  Date range: {date_range}")
                    logger.info(f"  Time span: {time_span} days")
                    
                    results[batch_dir_path.name] = {
                        'file': file_path.name,
                        'shape': df.shape,
                        'date_range': date_range,
                        'time_span': time_span
                    }
                    break  # Just check first file per batch
            
        except Exception as e:
            logger.error(f"Error reading {batch_dir_path}: {e}")
    
    return results

def analyze_temporal_alignment():
    """Analyze temporal alignment between different data sources."""
    logger.info("\n=== TEMPORAL ALIGNMENT ANALYSIS ===")
    
    # Get data from all sources
    sentiment_df = check_sentiment_cache()
    feature_results = check_feature_caches()
    batch_results = check_model_batches()
    
    # Analyze overlaps
    logger.info("\n=== TEMPORAL OVERLAP ANALYSIS ===")
    
    if sentiment_df is not None:
        sentiment_start = sentiment_df['timestamp'].min()
        sentiment_end = sentiment_df['timestamp'].max()
        
        logger.info(f"Sentiment data: {sentiment_start.date()} to {sentiment_end.date()}")
        
        # Check overlap with feature caches
        for file_name, info in feature_results.items():
            if info['time_span'] > 0:
                # Extract dates from date_range string
                try:
                    date_parts = info['date_range'].split(' to ')
                    if len(date_parts) == 2:
                        feature_start = pd.to_datetime(date_parts[0].strip())
                        feature_end = pd.to_datetime(date_parts[1].strip())
                        
                        # Calculate overlap
                        overlap_start = max(sentiment_start, feature_start)
                        overlap_end = min(sentiment_end, feature_end)
                        
                        if overlap_start <= overlap_end:
                            overlap_days = (overlap_end - overlap_start).days + 1
                            logger.info(f"  Overlap with {file_name}: {overlap_days} days ({overlap_start.date()} to {overlap_end.date()})")
                        else:
                            logger.info(f"  No overlap with {file_name}")
                            
                except Exception as e:
                    logger.warning(f"Could not parse date range for {file_name}: {e}")
    
    logger.info("\n=== RECOMMENDATIONS ===")
    logger.info("1. Check if sentiment data covers the same period as price data")
    logger.info("2. Verify macro data availability for target periods")
    logger.info("3. Consider temporal alignment strategies for new tickers")
    logger.info("4. May need to collect new data for missing periods")

def main():
    """Main analysis function."""
    logger.info("🔍 Analyzing temporal alignment of cached data...")
    
    analyze_temporal_alignment()
    
    logger.info("\n=== SUMMARY ===")
    logger.info("This analysis helps understand:")
    logger.info("1. What time periods are covered by cached data")
    logger.info("2. Whether different data sources overlap temporally")
    logger.info("3. Gaps that need to be filled for new ticker integration")
    logger.info("4. Feasibility of using cached data for new tickers")

if __name__ == "__main__":
    main()
