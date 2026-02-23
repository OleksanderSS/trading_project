import pandas as pd
from pathlib import Path
from config.config import PATHS

print('TESTING STRICT RSI LOGIC')
print('=' * 50)

stages_dir = Path(PATHS['data']) / 'stages'

# Перевandряємо stage2_merged.parquet (новий with жорсткою логandкою)
stage2_path = stages_dir / 'stage2_merged.parquet'
if stage2_path.exists():
    df = pd.read_parquet(stage2_path)
    print(f'Stage 2 merged: {df.shape}')
    
    # Перевandряємо RSI колонки
    rsi_cols = [col for col in df.columns if 'rsi_pre' in col]
    print(f'RSI columns: {len(rsi_cols)}')
    for col in rsi_cols:
        null_count = df[col].isna().sum()
        print(f'  {col}: {null_count} NaN values')
    
    # Перевandряємо чи all RSI not NaN
    rsi_15m = 'spy_15m_rsi_pre'
    rsi_60m = 'spy_60m_rsi_pre' 
    rsi_1d = 'spy_1d_rsi_pre'
    
    if all(col in df.columns for col in [rsi_15m, rsi_60m, rsi_1d]):
        # Перевandряємо чи all рядки мають all три RSI
        valid_rsi_count = len(df[
            df[rsi_15m].notna() & 
            df[rsi_60m].notna() & 
            df[rsi_1d].notna()
        ])
        
        print(f'Rows with all 3 RSI: {valid_rsi_count}/{len(df)} ({valid_rsi_count/len(df)*100:.1f}%)')
        
        if valid_rsi_count == len(df):
            print('PERFECT! All records have complete RSI data')
        else:
            print(f'WARNING: Some records missing RSI data')
    
    # Перевandряємо цandльовand withмandннand
    target_cols = [col for col in df.columns if 'target' in col.lower()]
    print(f'Target columns: {len(target_cols)}')
    
    # Перевandряємо роwithмandр fileу
    file_size_mb = stage2_path.stat().st_size / 1024 / 1024
    print(f'File size: {file_size_mb:.1f} MB')
    
else:
    print('ERROR: Stage 2 merged not found')

# Порandвнюємо with accumulated
acc_path = Path('c:/trading_project/data/colab/accumulated/stage2_accumulated.parquet')
if acc_path.exists():
    acc_df = pd.read_parquet(acc_path)
    print(f'\nAccumulated: {acc_df.shape}')
    
    if stage2_path.exists():
        print(f'Ratio stage2/accumulated: {df.shape[0] / acc_df.shape[0]:.2f}')

print('\n' + '=' * 50)
print('RSI LOGIC TEST COMPLETED')
