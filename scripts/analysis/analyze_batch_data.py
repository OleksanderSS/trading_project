#!/usr/bin/env python3
"""
Analyze batch data to understand what was collected.
"""

import pandas as pd
from pathlib import Path
import sys

def analyze_batch(batch_path):
    """Analyze batch data."""
    batch_dir = Path(batch_path)
    
    if not batch_dir.exists():
        print(f"❌ Batch directory not found: {batch_dir}")
        return
    
    print("=" * 80)
    print(f"📊 ANALYZING BATCH: {batch_dir.name}")
    print("=" * 80)
    
    # Load features
    features_file = batch_dir / "features.parquet"
    if not features_file.exists():
        print("❌ features.parquet not found")
        return
    
    df = pd.read_parquet(features_file)
    
    print(f"\n📈 FEATURES:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    # Analyze by interval
    if 'interval' in df.columns:
        print(f"\n⏰ BY INTERVAL:")
        interval_counts = df['interval'].value_counts()
        for interval, count in interval_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {interval:>5}: {count:>6} rows ({pct:>5.1f}%)")
    
    # Analyze by ticker
    if 'ticker' in df.columns:
        print(f"\n🎯 BY TICKER:")
        ticker_counts = df['ticker'].value_counts()
        print(f"   Total tickers: {len(ticker_counts)}")
        print(f"   Tickers: {sorted(ticker_counts.index.tolist())}")
        
        # Show distribution
        print(f"\n   Distribution:")
        for ticker in sorted(ticker_counts.index):
            count = ticker_counts[ticker]
            pct = (count / len(df)) * 100
            print(f"   {ticker:>6}: {count:>6} rows ({pct:>5.1f}%)")
    
    # Analyze 15m specifically
    if 'interval' in df.columns:
        df_15m = df[df['interval'] == '15m']
        if len(df_15m) > 0:
            print(f"\n🔍 15M ANALYSIS:")
            print(f"   Total 15m rows: {len(df_15m)}")
            
            if 'ticker' in df_15m.columns:
                print(f"   15m by ticker:")
                for ticker in sorted(df_15m['ticker'].unique()):
                    ticker_15m = df_15m[df_15m['ticker'] == ticker]
                    print(f"      {ticker:>6}: {len(ticker_15m)} rows")
            
            if 'datetime' in df_15m.columns:
                print(f"\n   15m datetime range:")
                print(f"      Min: {df_15m['datetime'].min()}")
                print(f"      Max: {df_15m['datetime'].max()}")
    
    # Check datetime column
    if 'datetime' in df.columns:
        print(f"\n📅 DATETIME:")
        print(f"   Column exists: ✅")
        print(f"   Type: {df['datetime'].dtype}")
        print(f"   Range: {df['datetime'].min()} → {df['datetime'].max()}")
    else:
        print(f"\n📅 DATETIME:")
        print(f"   Column exists: ❌")
        print(f"   Index type: {type(df.index)}")
    
    # Load targets
    targets_file = batch_dir / "targets.parquet"
    if targets_file.exists():
        df_targets = pd.read_parquet(targets_file)
        target_cols = [c for c in df_targets.columns if c.startswith('target_')]
        print(f"\n🎯 TARGETS:")
        print(f"   Shape: {df_targets.shape}")
        print(f"   Target columns: {len(target_cols)}")
        print(f"   Targets: {target_cols[:10]}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        batch_path = sys.argv[1]
    else:
        # Default to latest batch
        batch_path = "data/colab/accumulated/full_pipeline_trading/full_pipeline_trading"
    
    analyze_batch(batch_path)
