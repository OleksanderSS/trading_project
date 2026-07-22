import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def clean_file(path: Path):
    print(f"Cleaning {path.name}...")
    df = pd.read_parquet(path)
    
    was_index = False
    index_name = df.index.name
    if index_name == 'timestamp' or 'timestamp' in df.index.names:
        df = df.reset_index()
        was_index = True
    elif 'timestamp' not in df.columns:
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        else:
            df = df.reset_index(names=['timestamp'])
            index_name = 'timestamp'
            was_index = True
            
    ticker_col = "ticker" if "ticker" in df.columns else "symbol" if "symbol" in df.columns else None
    
    if ticker_col:
        cleaned_groups = []
        for ticker, group in df.groupby(ticker_col):
            group = group.sort_values("timestamp").copy()
            # Fast vectorized outlier detection using rolling median
            rolling_med = group['close'].rolling(window=21, center=True, min_periods=1).median()
            deviation = (group['close'] - rolling_med).abs() / rolling_med
            mask = deviation > 0.35
            
            if mask.any():
                total_cleaned = mask.sum()
                group.loc[mask, "close"] = pd.NA
                group["close"] = group["close"].ffill().bfill()
                print(f"  [{ticker}] Cleaned {total_cleaned} anomalous price points.")
                
            cleaned_groups.append(group)
            
        df_clean = pd.concat(cleaned_groups).sort_index()
    else:
        df_clean = df.sort_values("timestamp").copy()
        rolling_med = df_clean['close'].rolling(window=21, center=True, min_periods=1).median()
        deviation = (df_clean['close'] - rolling_med).abs() / rolling_med
        mask = deviation > 0.35
        
        if mask.any():
            total_cleaned = mask.sum()
            df_clean.loc[mask, "close"] = pd.NA
            df_clean["close"] = df_clean["close"].ffill().bfill()
            print(f"  Cleaned {total_cleaned} anomalous price points.")
            
    if was_index:
        df_clean = df_clean.set_index(index_name)
        
    df_clean.to_parquet(path)
    print(f"Done with {path.name}\n")

processed_dir = Path("data/processed")
for file_path in processed_dir.glob("prices_*.parquet"):
    clean_file(file_path)
