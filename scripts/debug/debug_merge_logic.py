#!/usr/bin/env python3
"""
Debug the merge logic in stage 2
"""
import pandas as pd
import os

def debug_merge_logic():
    """Debug merge logic step by step"""
    print("=" * 80)
    print("DEBUGGING MERGE LOGIC")
    print("=" * 80)
    
    # Load price data
    price_files = {
        '15m': "c:/trading_project/data/stages/prices_15m.parquet",
        '60m': "c:/trading_project/data/stages/prices_60m.parquet"
    }
    
    for interval, file_path in price_files.items():
        print(f"\n{interval.upper()} PRICE DATA:")
        
        if not os.path.exists(file_path):
            print(f"  File not found: {file_path}")
            continue
        
        df = pd.read_parquet(file_path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)[:10]}...")
        
        # Check if we have interval column
        if 'interval' in df.columns:
            print(f"  Interval column: YES")
            print(f"  Unique intervals: {df['interval'].unique()}")
        else:
            print(f"  Interval column: NO")
        
        # Check if we have ticker column
        if 'ticker' in df.columns:
            print(f"  Ticker column: YES")
            print(f"  Unique tickers: {df['ticker'].unique()}")
        else:
            print(f"  Ticker column: NO")
        
        # Check date column
        if 'date' in df.columns:
            print(f"  Date column: YES")
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        else:
            print(f"  Date column: NO")
        
        # Simulate the merge logic
        print(f"\n  Simulating merge logic for {interval}:")
        
        # Step 1: Filter by interval
        if 'interval' in df.columns:
            df_filtered = df[df["interval"] == interval].copy()
            print(f"    After interval filter: {df_filtered.shape}")
        else:
            df_filtered = df.copy()
            print(f"    No interval filter: {df_filtered.shape}")
        
        # Step 2: Check base metrics
        base_metrics = ["open", "high", "low", "close", "volume"]
        tech_metrics = [c for c in df_filtered.columns if c not in ["date",
            "ticker",
            "interval"] and c not in base_metrics]
        value_cols = base_metrics + tech_metrics
        
        print(f"    Base metrics found: {[c for c in base_metrics if c in df_filtered.columns]}")
        print(f"    Tech metrics found: {tech_metrics[:5]}...")
        print(f"    Value cols: {len(value_cols)}")
        
        # Step 3: Keep only needed columns
        keep = ["date","ticker"] + value_cols
        available_keep = [c for c in keep if c in df_filtered.columns]
        df_filtered = df_filtered[available_keep]
        print(f"    After column filter: {df_filtered.shape}")
        
        # Step 4: Try pivot
        if 'date' in df_filtered.columns and 'ticker' in df_filtered.columns and value_cols:
            try:
                pivot = df_filtered.pivot(index="date", columns="ticker", values=value_cols)
                print(f"    Pivot successful: {pivot.shape}")
                
                # Check column names
                print(f"    Pivot columns sample: {list(pivot.columns)[:5]}")
                
                # Flatten column names
                pivot.columns = [f"{interval}_{metric}_{ticker.lower()}" for metric, ticker in pivot.columns]
                print(f"    Flattened columns sample: {list(pivot.columns)[:5]}")
                
                # Check if we have the expected columns
                expected_cols = [f"{interval}_close_nvda", f"{interval}_close_qqq"]
                found_cols = [col for col in expected_cols if col in pivot.columns]
                print(f"    Expected columns found: {found_cols}")
                
            except Exception as e:
                print(f"    Pivot failed: {e}")
        else:
            print(f"    Cannot pivot - missing required columns")
            print(f"    Has date: {'date' in df_filtered.columns}")
            print(f"    Has ticker: {'ticker' in df_filtered.columns}")
            print(f"    Has value_cols: {len(value_cols) > 0}")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    debug_merge_logic()
