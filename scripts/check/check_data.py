#!/usr/bin/env python3
"""
Унandфandкована перевandрка data
Об'єднує функцandональнandсть with check_1d.py, check_15m_data.py, check_all_data.py and andнших
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Додаємо шлях до кореню проекту
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def check_1d_data():
    """Перевandрка 1D data"""
    print("=== CHECK 1D DATA ===")
    
    price_file = project_root / "data" / "stages" / "prices_1d.parquet"
    if price_file.exists():
        df = pd.read_parquet(price_file)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"Sample:\n{df.head(10)}")
            
            # Перевandряємо чи є multi-index
            if hasattr(df.columns, 'levels'):
                print(f"Column levels: {df.columns.levels}")
                
            # Перевandряємо першand рядки
            print(f"First row keys: {df.iloc[0].to_dict() if len(df) > 0 else 'Empty'}")
    else:
        print("[ERROR] 1D price file not found")

def check_15m_data():
    """Перевandрка 15м data"""
    print("\n=== CHECK 15M DATA ===")
    
    df = pd.read_parquet(project_root / "data" / "stages" / "merged_full.parquet")
    
    # Check 15m close prices
    close_15m_cols = [col for col in df.columns if '15m' in col and 'close' in col]
    print(f"15m close columns: {close_15m_cols}")
    
    if close_15m_cols:
        for col in close_15m_cols[:3]:  # Check first 3
            non_null = df[col].notna().sum()
            unique_vals = df[col].nunique()
            print(f"  {col}: {non_null}/{len(df)} non-null, {unique_vals} unique values")
            print(f"    Sample values: {df[col].dropna().head(5).tolist()}")

def check_all_data():
    """Перевandрка allх data"""
    print("\n=== CHECK ALL DATA ===")
    
    price_data = pd.read_parquet(project_root / "data" / "stages" / "stage1_price_data.parquet")
    
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
        print("[ERROR] NO PRICE DATA FOUND!")

def check_data_quality():
    """Перевandрка якостand data"""
    print("\n=== CHECK DATA QUALITY ===")
    
    # Перевandряємо основнand fileи data
    data_dir = project_root / "data" / "stages"
    print(f"Files in {data_dir}:")
    
    for file in data_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(file)
            null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            print(f"  {file.name}: {df.shape}, nulls: {null_percentage:.1f}%")
        except Exception as e:
            print(f"  {file.name}: ERROR - {e}")

def check_database_structure():
    """Перевandрка структури баwithи data"""
    print("\n=== CHECK DATABASE STRUCTURE ===")
    
    merged_path = project_root / "data" / "stages" / "merged_full.parquet"
    if merged_path.exists():
        df = pd.read_parquet(merged_path)
        print(f"Merged database: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Перевandряємо типи колонок
        dtypes = df.dtypes.value_counts()
        print(f"Data types:\n{dtypes}")
        
        # Перевandряємо наявнandсть ключових колонок
        key_columns = ['published_at', 'ticker', 'close', 'volume']
        missing_keys = [col for col in key_columns if col not in df.columns]
        if missing_keys:
            print(f"[ERROR] Missing key columns: {missing_keys}")
        else:
            print("[OK] All key columns present")
    else:
        print("[ERROR] Merged database not found")

def check_gaps():
    """Перевandрка гепandв"""
    print("\n=== CHECK GAPS ===")
    
    merged_path = project_root / "data" / "stages" / "merged_full.parquet"
    if merged_path.exists():
        df = pd.read_parquet(merged_path)
        
        # Всand колонки with 'gap'
        gap_cols = [col for col in df.columns if 'gap' in col.lower()]
        print(f"Gap columns: {len(gap_cols)}")
        
        # Групуємо for префandксом
        for tf in ['15m', '60m', '1d']:
            tf_cols = [col for col in gap_cols if col.startswith(tf)]
            print(f"\n{tf}: {len(tf_cols)} columns")
            for col in tf_cols[:5]:  # Покаwithуємо першand 5
                non_null = df[col].notna().sum()
                print(f"  {col}: {non_null:,} non-null values")
    else:
        print("[ERROR] Merged database not found")

def main():
    """Головна функцandя - forпуска allх перевandрок"""
    print("[SEARCH] COMPREHENSIVE DATA CHECK")
    print("=" * 50)
    
    try:
        check_1d_data()
        check_15m_data()
        check_all_data()
        check_data_quality()
        check_database_structure()
        check_gaps()
        
        print("\n" + "=" * 50)
        print("[OK] ALL CHECKS COMPLETED")
        
    except Exception as e:
        print(f"[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
