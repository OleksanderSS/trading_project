#!/usr/bin/env python3
"""
Унandфandкований whereбагandнг об'єднання data
Об'єднує функцandональнandсть with debug_merge_logic.py, debug_merge_detailed.py and andнших
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Додаємо шлях до кореню проекту
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def debug_merge_logic():
    """Дебагandнг логandки об'єднання data"""
    print("=== DEBUG MERGE LOGIC ===")
    
    # Перевandряємо еandпи об'єднання
    stages = ['stage1', 'stage2', 'stage3', 'stage4']
    data_dir = project_root / "data" / "stages"
    
    for stage in stages:
        print(f"\n{stage.upper()} STAGE:")
        stage_files = list(data_dir.glob(f"{stage}*.parquet"))
        
        for file in stage_files:
            try:
                df = pd.read_parquet(file)
                print(f"  {file.name}: {df.shape}")
                
                # Перевandряємо ключовand колонки
                key_cols = ['published_at', 'ticker']
                missing_keys = [col for col in key_cols if col not in df.columns]
                if missing_keys:
                    print(f"    [ERROR] Missing keys: {missing_keys}")
                else:
                    print(f"    [OK] All keys present")
                
                # Перевandряємо типи data
                if 'published_at' in df.columns:
                    print(f"    published_at type: {df['published_at'].dtype}")
                    
            except Exception as e:
                print(f"  [ERROR] ERROR: {e}")

def debug_merge_detailed():
    """Деandльний whereбагandнг об'єднання"""
    print("\n=== DEBUG MERGE DETAILED ===")
    
    # Перевandряємо об'єднання data по часових промandжках
    data_dir = project_root / "data" / "stages"
    
    # Заванandжуємо основнand данand
    try:
        stage1_data = pd.read_parquet(data_dir / "stage1_price_data.parquet")
        print(f"Stage1 data: {stage1_data.shape}")
        
        # Перевandряємо часовand промandжки
        if 'date' in stage1_data.columns:
            stage1_data['date'] = pd.to_datetime(stage1_data['date'], errors='coerce')
            date_range = f"{stage1_data['date'].min()} to {stage1_data['date'].max()}"
            print(f"Date range: {date_range}")
            
            # Перевandряємо унandкальнand часовand промandжки
            unique_intervals = stage1_data['interval'].unique()
            print(f"Intervals: {list(unique_intervals)}")
            
            # Перевandряємо кожен часовий промandжок
            for interval in unique_intervals:
                interval_data = stage1_data[stage1_data['interval'] == interval]
                print(f"\n{interval} interval:")
                print(f"  Records: {len(interval_data)}")
                print(f"  Tickers: {list(interval_data['ticker'].unique())}")
                
                # Перевandряємо послandдовнandсть data
                sorted_data = interval_data.sort_values('date')
                date_gaps = sorted_data['date'].diff().dt.days.dropna()
                max_gap = date_gaps.max()
                print(f"  Max date gap: {max_gap} days")
                
                if max_gap > 7:
                    print(f"  [WARN]  Large gap detected!")
                    gap_dates = sorted_data[date_gaps == max_gap]
                    print(f"  Gap dates: {gap_dates.index}")
        
    except Exception as e:
        print(f"[ERROR] ERROR: {e}")

def debug_data_types():
    """Дебагandнг типandв data"""
    print("\n=== DEBUG DATA TYPES ===")
    
    merged_path = project_root / "data" / "stages" / "merged_full.parquet"
    if merged_path.exists():
        df = pd.read_parquet(merged_path)
        
        print(f"Merged data shape: {df.shape}")
        
        # Аналandwith типandв data
        dtypes = df.dtypes
        print("\nData types:")
        for dtype, count in dtypes.value_counts().items():
            print(f"  {dtype}: {count} columns")
        
        # Перевandряємо problemsнand типи
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            print(f"\nObject columns ({len(object_cols)}):")
            for col in object_cols[:10]:  # Покаwithуємо першand 10
                unique_vals = df[col].nunique()
                print(f"  {col}: {unique_vals} unique values")
                if unique_vals < 20:
                    print(f"    Values: {list(df[col].unique())}")
        
        # Перевandряємо числовand колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric columns ({len(numeric_cols)}):")
            for col in numeric_cols[:10]:  # Покаwithуємо першand 10
                min_val = df[col].min()
                max_val = df[col].max()
                print(f"  {col}: [{min_val:.4f}, {max_val:.4f}]")
                
                # Перевandряємо на екстремальнand values
                if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                    print(f"    [WARN]  Extreme values detected!")
        
        # Перевandряємо часовand колонки
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            print(f"\nDatetime columns ({len(datetime_cols)}):")
            for col in datetime_cols:
                min_date = df[col].min()
                max_date = df[col].max()
                print(f"  {col}: {min_date} to {max_date}")
                
                # Перевandряємо на майбутнand дати
                today = pd.Timestamp.now()
                future_dates = df[col] > today
                if future_dates.any():
                    print(f"    [WARN]  Future dates detected: {future_dates.sum()}")
                
                # Перевandряємо на дуже сandрand дати
                old_dates = df[col] < pd.Timestamp('2000-01-01')
                if old_dates.any():
                    print(f"    [WARN]  Very old dates detected: {old_dates.sum()}")
    else:
        print("[ERROR] Merged database not found")

def debug_column_consistency():
    """Дебагandнг уwithгодженостand колонок"""
    print("\n=== DEBUG COLUMN CONSISTENCY ===")
    
    data_dir = project_root / "data" / "stages"
    
    # Збираємо all fileи
    all_files = list(data_dir.glob("*.parquet"))
    
    if not all_files:
        print("[ERROR] No parquet files found")
        return
    
    # Аналandwithуємо колонки в кожному fileand
    file_columns = {}
    for file in all_files:
        try:
            df = pd.read_parquet(file)
            file_columns[file.name] = set(df.columns)
        except Exception as e:
            print(f"[ERROR] Error reading {file.name}: {e}")
    
    # Знаходимо спandльнand колонки
    if file_columns:
        all_columns = set()
        for cols in file_columns.values():
            all_columns.update(cols)
        
        print(f"Total unique columns: {len(all_columns)}")
        
        # Перевandряємо якand колонки є в усandх fileах
        common_columns = set.intersection(*file_columns.values())
        print(f"Common columns: {len(common_columns)}")
        
        if len(common_columns) > 0:
            print(f"Common columns: {list(common_columns)[:20]}")
        
        # Перевandряємо унandкальнand колонки
        for file_name, cols in file_columns.items():
            unique_cols = cols - common_columns
            if len(unique_cols) > 0:
                print(f"\n{file_name} unique columns ({len(unique_cols)}):")
                for col in list(unique_cols)[:10]:
                    print(f"  {col}")
        
        # Перевandряємо вandдсутнand ключовand колонки
        key_columns = ['published_at', 'ticker', 'close', 'volume']
        missing_in_files = {}
        for file_name, cols in file_columns.items():
            missing = [col for col in key_columns if col not in cols]
            if missing:
                missing_in_files[file_name] = missing
        
        if missing_in_files:
            print(f"\nMissing key columns:")
            for file_name, missing in missing_in_files.items():
                print(f"  {file_name}: {missing}")

def main():
    """Головна функцandя - forпусок allх whereбагandвгandв"""
    print(" COMPREHENSIVE MERGE DEBUG")
    print("=" * 50)
    
    try:
        debug_merge_logic()
        debug_merge_detailed()
        debug_data_types()
        debug_column_consistency()
        
        print("\n" + "=" * 50)
        print("[OK] ALL DEBUG CHECKS COMPLETED")
        
    except Exception as e:
        print(f"[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
