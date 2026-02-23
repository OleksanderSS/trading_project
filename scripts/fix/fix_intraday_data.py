#!/usr/bin/env python3
"""
Fix intraday data collection issues
"""
import pandas as pd
import os
import yfinance as yf
from datetime import datetime, timedelta

def fix_intraday_data():
    """Fix intraday data collection for 15m and 60m"""
    print("=" * 60)
    print("FIXING INTRADAY DATA COLLECTION")
    print("=" * 60)
    
    tickers = ['NVDA', 'QQQ', 'SPY', 'TSLA']
    intervals = ['15m', '60m']
    
    # Check current intraday files
    price_files = {
        '15m': "c:/trading_project/data/stages/prices_15m.parquet",
        '60m': "c:/trading_project/data/stages/prices_60m.parquet"
    }
    
    print("CURRENT INTRADAY FILES STATUS:")
    for interval, file_path in price_files.items():
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            print(f"  {interval}: {len(df)} rows, {len(df.columns)} columns")
            print(f"    Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
            
            if not df.empty:
                # Check date column name
                date_col = None
                for col in ['date', 'Datetime', 'Date']:
                    if col in df.columns:
                        date_col = col
                        break
                
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    date_range = f"{df[date_col].min()} to {df[date_col].max()}"
                    print(f"    Date range: {date_range}")
                else:
                    print(f"    No date column found!")
                
                # Check tickers
                tickers_in_data = df['ticker'].unique() if 'ticker' in df.columns else []
                print(f"    Tickers: {tickers_in_data}")
        else:
            print(f"  {interval}: FILE NOT FOUND")
    
    print("\nDOWNLOADING MISSING INTRADAY DATA...")
    
    # Download missing intraday data
    for interval in intervals:
        file_path = price_files[interval]
        
        # Check if we need to download
        need_download = False
        if not os.path.exists(file_path):
            need_download = True
            print(f"  {interval}: File missing - downloading...")
        else:
            df = pd.read_parquet(file_path)
            if df.empty or len(df) < 1000:  # Too little data
                need_download = True
                print(f"  {interval}: Insufficient data ({len(df)} rows) - downloading...")
        
        if need_download:
            try:
                # Download last 60 days of intraday data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)
                
                print(f"    Downloading {interval} data from {start_date.date()} to {end_date.date()}...")
                
                all_data = []
                for ticker in tickers:
                    try:
                        # Download data
                        data = yf.download(
                            ticker, 
                            start=start_date, 
                            end=end_date, 
                            interval=interval,
                            progress=False
                        )
                        
                        if not data.empty:
                            data.reset_index(inplace=True)
                            data['ticker'] = ticker
                            data['interval'] = interval
                            
                            # Rename columns
                            data = data.rename(columns={
                                'Datetime': 'date',
                                'Date': 'date',
                                'Open': 'open',
                                'High': 'high', 
                                'Low': 'low',
                                'Close': 'close',
                                'Volume': 'volume'
                            })
                            
                            all_data.append(data)
                            print(f"      {ticker}: {len(data)} rows")
                        else:
                            print(f"      {ticker}: No data")
                            
                    except Exception as e:
                        print(f"      {ticker}: Error - {e}")
                
                if all_data:
                    # Combine all data
                    combined_df = pd.concat(all_data, ignore_index=True)
                    
                    # Ensure proper column names
                    if 'date' not in combined_df.columns and 'Datetime' in combined_df.columns:
                        combined_df = combined_df.rename(columns={'Datetime': 'date'})
                    
                    # Save to file
                    combined_df.to_parquet(file_path, index=False)
                    print(f"    [OK] Saved {len(combined_df)} rows to {file_path}")
                else:
                    print(f"    [FAIL] No data downloaded for {interval}")
                    
            except Exception as e:
                print(f"    [FAIL] Failed to download {interval}: {e}")
    
    print("\n" + "=" * 60)
    print("INTRADAY DATA FIX COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    fix_intraday_data()
