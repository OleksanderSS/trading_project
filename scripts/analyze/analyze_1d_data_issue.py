import pandas as pd
import os

print('=== DETAILED 1D DATA ANALYSIS ===')

# Check stages 1D data
stages_1d = pd.read_parquet('c:/trading_project/data/stages/prices_1d.parquet')
print(f'1D data in stages: {stages_1d.shape}')
print(f'Columns: {list(stages_1d.columns)}')
print(f'Date range: {stages_1d["date"].min()} to {stages_1d["date"].max()}')

if 'ticker' in stages_1d.columns:
    print(f'\nTickers: {stages_1d["ticker"].unique()}')
    for ticker in stages_1d['ticker'].unique():
        ticker_data = stages_1d[stages_1d['ticker'] == ticker]
        print(f'\n{ticker}:')
        print(f'  Rows: {len(ticker_data)}')
        print(f'  Date range: {ticker_data["date"].min()} to {ticker_data["date"].max()}')
        
        # Check for missing days
        if len(ticker_data) > 1:
            dates = sorted(ticker_data['date'].unique())
            date_range = pd.date_range(dates[0], dates[-1], freq='D')
            missing_days = date_range.difference(dates)
            print(f'  Missing days: {len(missing_days)}')
            if len(missing_days) > 0:
                print(f'  First missing: {missing_days[0]}')
                print(f'  Last missing: {missing_days[-1]}')

# Check news data
news_df = pd.read_parquet('c:/trading_project/data/stages/merged_full_clean.parquet')
news_start = news_df['published_at'].min()
news_end = news_df['published_at'].max()

print(f'\n=== COVERAGE CHECK ===')
print(f'News period: {news_start} to {news_end}')
print(f'1D data: {stages_1d["date"].min()} to {stages_1d["date"].max()}')

# Check if 1D data has enough history for ATR (needs 14 days)
if 'ticker' in stages_1d.columns:
    for ticker in stages_1d['ticker'].unique():
        ticker_data = stages_1d[stages_1d['ticker'] == ticker]
        ticker_start = ticker_data['date'].min()
        
        # First possible ATR calculation date
        first_atr_date = ticker_start + pd.Timedelta(days=14)
        print(f'\n{ticker}:')
        print(f'  Data starts: {ticker_start}')
        print(f'  First ATR possible: {first_atr_date}')
        print(f'  News starts: {news_start}')
        print(f'  Can calculate ATR for news: {first_atr_date <= news_start}')
        
        if first_atr_date > news_start:
            print(f'  PROBLEM: Not enough history before news!')
            days_missing = (first_atr_date - news_start).days
            print(f'  Missing {days_missing} days of ATR data')

# Check if we need to extend 1D data backwards
print(f'\n=== SOLUTION ANALYSIS ===')
if 'ticker' in stages_1d.columns:
    for ticker in stages_1d['ticker'].unique():
        ticker_data = stages_1d[stages_1d['ticker'] == ticker]
        current_start = ticker_data['date'].min()
        
        # We need at least 14 days before first news
        required_start = news_start - pd.Timedelta(days=14)
        
        print(f'\n{ticker}:')
        print(f'  Current start: {current_start}')
        print(f'  Required start: {required_start}')
        print(f'  Need to extend back: {(required_start - current_start).days} days')
        
        if current_start > required_start:
            print(f'  ACTION: Need to collect historical 1D data before {current_start}')
