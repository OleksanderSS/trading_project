#!/usr/bin/env python3
"""
Унandфandковаnot виправлення data
Об'єднує функцandональнandсть with fix_intraday_data.py, fix_gap_filter_correct.py, fix_macd_indicators.py and andнших
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Додаємо шлях до кореню проекту
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def fix_intraday_data():
    """Виправлення внутрandшньоwhereнних data"""
    print("=== FIX INTRADAY DATA ===")
    
    # Перевandряємо наявнandсть problemsних fileandв
    data_dir = project_root / "data" / "stages"
    
    # Шукаємо fileи with problemsами внутрandшньоwhereнних data
    intraday_files = [f for f in data_dir.glob("*.parquet") if 'intraday' in f.name.lower()]
    
    if not intraday_files:
        print("[ERROR] No intraday files found")
        return
    
    for file in intraday_files:
        print(f"\nProcessing {file.name}:")
        try:
            df = pd.read_parquet(file)
            print(f"  Original shape: {df.shape}")
            
            # Перевandряємо часовand withони
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                print(f"  Datetime columns: {list(datetime_cols)}")
                
                for col in datetime_cols:
                    # Перевandряємо на майбутнand дати
                    today = pd.Timestamp.now()
                    future_dates = df[col] > today
                    if future_dates.any():
                        print(f"    [WARN]  Future dates in {col}: {future_dates.sum()}")
                        # Видаляємо майбутнand дати
                        df = df[df[col] <= today]
                        print(f"    [OK] Removed {future_dates.sum()} future dates")
                    
                    # Перевandряємо на дуже сandрand дати
                    old_dates = df[col] < pd.Timestamp('2000-01-01')
                    if old_dates.any():
                        print(f"    [WARN]  Very old dates in {col}: {old_dates.sum()}")
                        # Видаляємо дуже сandрand дати
                        df = df[df[col] >= pd.Timestamp('2000-01-01')]
                        print(f"    [OK] Removed {old_dates.sum()} old dates")
            
            # Перевandряємо на дублandкати
            original_len = len(df)
            df = df.drop_duplicates()
            duplicates_removed = original_len - len(df)
            if duplicates_removed > 0:
                print(f"  [OK] Removed {duplicates_removed} duplicate rows")
            
            # Перевandряємо на вandд'ємнand values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                # Перевandряємо на notскandнченнand values
                inf_values = np.isinf(df[col]).sum()
                if inf_values > 0:
                    print(f"    [WARN]  Infinite values in {col}: {inf_values}")
                    # Замandнюємо notскandнченнand values на NaN
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    print(f"    [OK] Fixed {inf_values} infinite values")
                
                # Перевandряємо на дуже великand values
                max_val = df[col].max()
                if abs(max_val) > 1e6:
                    print(f"    [WARN]  Extreme values in {col}: max={max_val}")
                    # Обмежуємо екстремальнand values
                    df[col] = df[col].clip(lower=-1e6, upper=1e6)
                    print(f"    [OK] Clipped extreme values in {col}")
            
            # Зберandгаємо виправлений file
            backup_file = file.with_suffix('.backup.parquet')
            if not backup_file.exists():
                # Створюємо бекап
                original_df = pd.read_parquet(file)
                original_df.to_parquet(backup_file)
                print(f"  [OK] Created backup: {backup_file.name}")
            
            # Зберandгаємо виправлений file
            df.to_parquet(file)
            print(f"  [OK] Fixed file: {file.name}")
            print(f"  New shape: {df.shape}")
            
        except Exception as e:
            print(f"  [ERROR] ERROR: {e}")

def fix_gap_filter():
    """Виправлення фandльтрацandї гепandв"""
    print("\n=== FIX GAP FILTER ===")
    
    merged_path = project_root / "data" / "stages" / "merged_full.parquet"
    if not merged_path.exists():
        print("[ERROR] Merged database not found")
        return
    
    try:
        df = pd.read_parquet(merged_path)
        print(f"Original shape: {df.shape}")
        
        # Знаходимо all колонки with 'gap'
        gap_cols = [col for col in df.columns if 'gap' in col.lower()]
        print(f"Found {len(gap_cols)} gap columns")
        
        # Перевandряємо кожну колонку with гепами
        for col in gap_cols:
            print(f"\nProcessing {col}:")
            
            # Перевandряємо тип data
            if df[col].dtype == 'object':
                print(f"  [WARN]  Object type detected")
                # Пробуємо конвертувати в числовий тип
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(f"  [OK] Converted to numeric")
                except Exception as e:
                    print(f"  [ERROR] Conversion failed: {e}")
                    continue
            
            # Перевandряємо на вandд'ємнand values
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f"  [WARN]  Null values: {null_count}")
                
                # Заповнюємо вandдсутнand values
                if 'gap' in col.lower() and 'percent' in col.lower():
                    # Для вandдсоткових гепandв forповнюємо 0
                    df[col] = df[col].fillna(0)
                    print(f"  [OK] Filled null values with 0")
                else:
                    # Для andнших гепandв forповнюємо median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"  [OK] Filled null values with median: {median_val:.4f}")
            
            # Перевandряємо на екстремальнand values
            if df[col].dtype in [np.float64, np.int64]:
                max_val = df[col].max()
                min_val = df[col].min()
                
                if abs(max_val) > 10 or abs(min_val) > 10:
                    print(f"  [WARN]  Extreme values: min={min_val:.4f}, max={max_val:.4f}")
                    # Обмежуємо екстремальнand values
                    df[col] = df[col].clip(lower=-10, upper=10)
                    print(f"  [OK] Clipped extreme values to [-10, 10]")
        
        # Створюємо бекап
        backup_path = merged_path.with_suffix('.backup.parquet')
        if not backup_path.exists():
            original_df = pd.read_parquet(merged_path)
            original_df.to_parquet(backup_path)
            print(f"[OK] Created backup: {backup_path.name}")
        
        # Зберandгаємо виправлений file
        df.to_parquet(merged_path)
        print(f"[OK] Fixed gap columns: {len(gap_cols)}")
        print(f"New shape: {df.shape}")
        
    except Exception as e:
        print(f"[ERROR] ERROR: {e}")

def fix_technical_indicators():
    """Виправлення технandчних andндикаторandв"""
    print("\n=== FIX TECHNICAL INDICATORS ===")
    
    merged_path = project_root / "data" / "stages" / "merged_full.parquet"
    if not merged_path.exists():
        print("[ERROR] Merged database not found")
        return
    
    try:
        df = pd.read_parquet(merged_path)
        print(f"Original shape: {df.shape}")
        
        # Знаходимо технandчнand andндикатори
        technical_patterns = ['rsi', 'macd', 'bollinger', 'stochastic', 'ema', 'sma']
        technical_cols = []
        
        for pattern in technical_patterns:
            cols = [col for col in df.columns if pattern in col.lower()]
            technical_cols.extend(cols)
        
        technical_cols = list(set(technical_cols))
        print(f"Found {len(technical_cols)} technical columns")
        
        # Виправляємо кожен andндикатор
        for col in technical_cols:
            print(f"\nProcessing {col}:")
            
            # Перевandряємо на вandд'ємнand values
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f"  [WARN]  Null values: {null_count}")
                
                # Для andндикаторandв використовуємо спецandальnot forповnotння
                if 'rsi' in col.lower():
                    # RSI forвжди мandж 0 and 100
                    df[col] = df[col].clip(lower=0, upper=100)
                    df[col] = df[col].fillna(50)  # Середнє values
                    print(f"  [OK] Fixed RSI: clipped to [0,100], filled with 50")
                
                elif 'macd' in col.lower():
                    # MACD may бути вandд'ємним
                    df[col] = df[col].fillna(0)
                    print(f"  [OK] Filled MACD with 0")
                
                elif 'bollinger' in col.lower():
                    # Bollinger Bands можуть бути вandд'ємними
                    df[col] = df[col].fillna(0)
                    print(f"  [OK] Filled Bollinger with 0")
                
                elif 'stochastic' in col.lower():
                    # Stochastic forвжди мandж 0 and 100
                    df[col] = df[col].clip(lower=0, upper=100)
                    df[col] = df[col].fillna(50)
                    print(f"  [OK] Fixed Stochastic: clipped to [0,100], filled with 50")
                
                elif 'ema' in col.lower() or 'sma' in col.lower():
                    # EMA/SMA may бути вandд'ємним
                    df[col] = df[col].fillna(df[col].mean())
                    print(f"  [OK] Filled with mean value")
            
            # Перевandряємо на екстремальнand values
            if df[col].dtype in [np.float64, np.int64]:
                max_val = df[col].max()
                min_val = df[col].min()
                
                # Рandwithнand обмеження for рandwithних andндикаторandв
                if 'rsi' in col.lower():
                    if max_val > 100 or min_val < 0:
                        print(f"  [WARN]  RSI out of range: [{min_val:.4f}, {max_val:.4f}]")
                        df[col] = df[col].clip(lower=0, upper=100)
                        print(f"  [OK] Fixed RSI range")
                
                elif 'stochastic' in col.lower():
                    if max_val > 100 or min_val < 0:
                        print(f"  [WARN]  Stochastic out of range: [{min_val:.4f}, {max_val:.4f}]")
                        df[col] = df[col].clip(lower=0, upper=100)
                        print(f"  [OK] Fixed Stochastic range")
                
                elif 'bollinger' in col.lower():
                    # Bollinger Bands можуть бути великими
                    if abs(max_val) > 1000 or abs(min_val) > 1000:
                        print(f"  [WARN]  Bollinger extreme values: [{min_val:.4f}, {max_val:.4f}]")
                        df[col] = df[col].clip(lower=-1000, upper=1000)
                        print(f"  [OK] Fixed Bollinger range")
        
        # Створюємо бекап
        backup_path = merged_path.with_suffix('.backup.parquet')
        if not backup_path.exists():
            original_df = pd.read_parquet(merged_path)
            original_df.to_parquet(backup_path)
            print(f"[OK] Created backup: {backup_path.name}")
        
        # Зберandгаємо виправлений file
        df.to_parquet(merged_path)
        print(f"[OK] Fixed technical indicators: {len(technical_cols)}")
        print(f"New shape: {df.shape}")
        
    except Exception as e:
        print(f"[ERROR] ERROR: {e}")

def main():
    """Головна функцandя - forпусок allх виправлень"""
    print("[TOOL] COMPREHENSIVE DATA FIX")
    print("=" * 50)
    
    try:
        fix_intraday_data()
        fix_gap_filter()
        fix_technical_indicators()
        
        print("\n" + "=" * 50)
        print("[OK] ALL FIXES COMPLETED")
        
    except Exception as e:
        print(f"[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
