import pandas as pd

# Check features
df = pd.read_parquet('data/colab/accumulated/main_database/features.parquet')
print(f'Shape: {df.shape}')
print(f'Columns (first 20): {df.columns.tolist()[:20]}')
print(f'Has datetime: {"datetime" in df.columns}')

if 'datetime' in df.columns:
    print(f'Datetime dtype: {df["datetime"].dtype}')
    print(f'NaT count: {df["datetime"].isna().sum()}')
    print(f'Date range: {df["datetime"].min()} to {df["datetime"].max()}')
    
    # Check for future dates
    from datetime import datetime
    now = pd.Timestamp.now()
    future_count = (df["datetime"] > now).sum()
    print(f'Future dates count: {future_count}')
    
    if future_count > 0:
        print(f'Future dates sample: {df[df["datetime"] > now]["datetime"].head()}')
else:
    print('❌ NO DATETIME COLUMN!')

# Check targets
print('\n--- TARGETS ---')
df_targets = pd.read_parquet('data/colab/accumulated/main_database/targets.parquet')
print(f'Shape: {df_targets.shape}')
print(f'Columns: {df_targets.columns.tolist()}')
print(f'Has datetime: {"datetime" in df_targets.columns}')

if 'datetime' in df_targets.columns:
    print(f'Datetime dtype: {df_targets["datetime"].dtype}')
    print(f'NaT count: {df_targets["datetime"].isna().sum()}')
    print(f'Date range: {df_targets["datetime"].min()} to {df_targets["datetime"].max()}')
