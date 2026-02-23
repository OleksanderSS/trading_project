import pandas as pd
import numpy as np

# Заванandжуємо данand
price_data = pd.read_parquet('c:/trading_project/data/stages/stage1_price_data.parquet')

print("=== PRICE DATA OVERVIEW ===")
print(f"Total records: {len(price_data)}")
print(f"Columns: {list(price_data.columns)}")

if len(price_data) > 0:
    print(f"Unique tickers: {price_data['ticker'].unique()}")
    print(f"Unique intervals: {price_data['interval'].unique()}")
    
    # Перевandряємо наявнandсть data for кожної комбandнацandї
    for ticker in price_data['ticker'].unique():
        for interval in price_data['interval'].unique():
            subset = price_data[(price_data['ticker'] == ticker) & (price_data['interval'] == interval)]
            print(f"{ticker} {interval}: {len(subset)} records")
            if len(subset) > 0:
                subset['date'] = pd.to_datetime(subset['date'])
                print(f"  Date range: {subset['date'].min()} to {subset['date'].max()}")
else:
    print("NO PRICE DATA FOUND!")

# Перевandряємо чи є andншand fileи with даними
import os
data_dir = 'c:/trading_project/data/stages'
print("\n=== FILES IN DATA/STAGES ===")
for file in os.listdir(data_dir):
    if file.endswith('.parquet'):
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_parquet(file_path)
            print(f"{file}: {df.shape}")
        except Exception as e:
            print(f"{file}: ERROR - {e}")
