"""
Перевірка покриття даних - які тікери мають дані для яких періодів
"""
import sys
from pathlib import Path
import pandas as pd
import duckdb

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_data_coverage():
    """Перевірити покриття даних"""
    print("=" * 80)
    print("DATA COVERAGE CHECK")
    print("=" * 80)
    
    db_path = project_root / 'data' / 'raw_data.duckdb'
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = duckdb.connect(str(db_path))
    
    # Завантажити дані
    market_df = conn.execute("""
        SELECT * FROM market_data_raw 
        WHERE datetime >= '2026-04-20' 
        ORDER BY datetime
    """).fetchdf()
    
    print(f"\nTotal rows: {len(market_df)}")
    print(f"Date range: {market_df['datetime'].min()} to {market_df['datetime'].max()}")
    
    # Перевірити покриття по тікерам
    print("\n" + "=" * 80)
    print("TICKER COVERAGE")
    print("=" * 80)
    
    for interval in ['15m', '60m', '1d']:
        print(f"\n{interval} timeframe:")
        interval_df = market_df[market_df['interval'] == interval]
        
        ticker_counts = interval_df.groupby('ticker').size().sort_values(ascending=False)
        
        print(f"  Total rows: {len(interval_df)}")
        print(f"  Tickers: {len(ticker_counts)}")
        print(f"\n  Rows per ticker:")
        for ticker, count in ticker_counts.items():
            date_range = interval_df[interval_df['ticker'] == ticker]['datetime']
            print(f"    {ticker:6s}: {count:5d} rows  ({date_range.min()} to {date_range.max()})")
    
    # Перевірити gaps (пропуски в даних)
    print("\n" + "=" * 80)
    print("DATA GAPS CHECK")
    print("=" * 80)
    
    for interval in ['15m', '60m', '1d']:
        print(f"\n{interval} timeframe:")
        interval_df = market_df[market_df['interval'] == interval]
        
        for ticker in interval_df['ticker'].unique()[:5]:  # Перші 5 тікерів
            ticker_df = interval_df[interval_df['ticker'] == ticker].sort_values('datetime')
            
            if len(ticker_df) < 2:
                continue
            
            # Розрахувати очікувану різницю між свічками
            if interval == '15m':
                expected_diff = pd.Timedelta(minutes=15)
            elif interval == '60m':
                expected_diff = pd.Timedelta(hours=1)
            else:  # 1d
                expected_diff = pd.Timedelta(days=1)
            
            # Знайти gaps
            ticker_df['time_diff'] = ticker_df['datetime'].diff()
            gaps = ticker_df[ticker_df['time_diff'] > expected_diff * 1.5]
            
            if not gaps.empty:
                print(f"  {ticker}: {len(gaps)} gaps found")
                for idx, row in gaps.head(3).iterrows():
                    print(f"    Gap at {row['datetime']}: {row['time_diff']}")
            else:
                print(f"  {ticker}: No gaps")
    
    conn.close()


if __name__ == '__main__':
    check_data_coverage()
