#!/usr/bin/env python3
"""
Унandфandкований аналandwith пайплайну
Об'єднує функцandональнandсть with analyze_pipeline_logic.py, analyze_data_quality.py and andнших
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

def analyze_pipeline_logic():
    """Аналandwith логandки пайплайну"""
    print("=== ANALYZE PIPELINE LOGIC ===")
    
    # Перевandряємо основнand еandпи пайплайну
    stages = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']
    data_dir = project_root / "data" / "stages"
    
    for stage in stages:
        stage_files = list(data_dir.glob(f"{stage}*.parquet"))
        print(f"\n{stage.upper()}:")
        print(f"  Files: {len(stage_files)}")
        
        for file in stage_files:
            try:
                df = pd.read_parquet(file)
                print(f"    {file.name}: {df.shape}")
                
                # Перевandряємо якandсть data
                null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                print(f"      Nulls: {null_percentage:.1f}%")
                
            except Exception as e:
                print(f"    {file.name}: ERROR - {e}")

def analyze_data_quality():
    """Аналandwith якостand data"""
    print("\n=== ANALYZE DATA QUALITY ===")
    
    merged_path = project_root / "data" / "stages" / "merged_full.parquet"
    if merged_path.exists():
        df = pd.read_parquet(merged_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Date range: {df['published_at'].min()} to {df['published_at'].max()}")
        
        # Аналandwith вandдсутнandх data
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data / len(df) * 100).sort_values(ascending=False)
        
        print("\nTop 10 columns with most missing data:")
        for col, pct in missing_percentage.head(10):
            print(f"  {col}: {pct:.1f}%")
        
        # Аналandwith дублandкатandв
        duplicates = df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates} ({duplicates/len(df)*100:.1f}%)")
        
        # Аналandwith унandкальних withначень
        print("\nUnique values in key columns:")
        key_cols = ['ticker', 'interval', 'published_at']
        for col in key_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count}")
    else:
        print("[ERROR] Merged database not found")

def analyze_feature_importance():
    """Аналandwith важливостand фandчей"""
    print("\n=== ANALYZE FEATURE IMPORTANCE ===")
    
    # Перевandряємо наявнandсть fileandв with важливandстю фandчей
    importance_files = list(project_root.glob("feature_importance*.csv"))
    
    if importance_files:
        for file in importance_files:
            print(f"\n{file.name}:")
            try:
                df = pd.read_csv(file)
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                
                if len(df) > 0:
                    # Покаwithуємо топ фandчей
                    if 'feature' in df.columns and 'importance' in df.columns:
                        top_features = df.nlargest(10, 'importance')
                        print("  Top features:")
                        for _, row in top_features.iterrows():
                            print(f"    {row['feature']}: {row['importance']:.4f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        print("[ERROR] No feature importance files found")

def analyze_missing_data():
    """Аналandwith вandдсутнandх data"""
    print("\n=== ANALYZE MISSING DATA ===")
    
    merged_path = project_root / "data" / "stages" / "merged_full.parquet"
    if merged_path.exists():
        df = pd.read_parquet(merged_path)
        
        # Аналandwith вandдсутнandх data по часу
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        
        # Групуємо по днях
        daily_counts = df.groupby(df['published_at'].dt.date).size()
        
        print(f"Data by day:")
        print(f"  Date range: {daily_counts.index.min()} to {daily_counts.index.max()}")
        print(f"  Average records per day: {daily_counts.mean():.1f}")
        print(f"  Min records per day: {daily_counts.min()}")
        print(f"  Max records per day: {daily_counts.max()}")
        
        # Знаходимо днand with малою кandлькandстю data
        low_data_days = daily_counts[daily_counts < daily_counts.quantile(0.1)]
        print(f"\nDays with low data (<10th percentile): {len(low_data_days)}")
        
        if len(low_data_days) > 0:
            print("  Sample dates:")
            for date in low_data_days.head(5).index:
                print(f"    {date}: {daily_counts[date]} records")
    else:
        print("[ERROR] Merged database not found")

def analyze_performance():
    """Аналandwith продуктивностand"""
    print("\n=== ANALYZE PERFORMANCE ===")
    
    # Перевandряємо роwithмandри fileandв
    data_dir = project_root / "data" / "stages"
    files = list(data_dir.glob("*.parquet"))
    
    if files:
        print(f"Analyzing {len(files)} files...")
        
        file_sizes = []
        for file in files:
            try:
                size_mb = file.stat().st_size / (1024 * 1024)
                file_sizes.append((file.name, size_mb))
            except Exception as e:
                print(f"  Error getting size for {file.name}: {e}")
        
        # Сортуємо for роwithмandром
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        
        print("\nLargest files:")
        for name, size in file_sizes[:10]:
            print(f"  {name}: {size:.1f} MB")
        
        total_size = sum(size for _, size in file_sizes)
        print(f"\nTotal size: {total_size:.1f} MB")
        
        # Перевandряємо forгальний роwithмandр data
        try:
            merged_path = project_root / "data" / "stages" / "merged_full.parquet"
            if merged_path.exists():
                merged_size = merged_path.stat().st_size / (1024 * 1024)
                print(f"Merged database size: {merged_size:.1f} MB")
        except Exception as e:
            print(f"Error checking merged database: {e}")
    else:
        print("[ERROR] No parquet files found")

def main():
    """Головна функцandя - forпуска allх аналandwithandв"""
    print("[SEARCH] COMPREHENSIVE PIPELINE ANALYSIS")
    print("=" * 50)
    
    try:
        analyze_pipeline_logic()
        analyze_data_quality()
        analyze_feature_importance()
        analyze_missing_data()
        analyze_performance()
        
        print("\n" + "=" * 50)
        print("[OK] ALL ANALYSES COMPLETED")
        
    except Exception as e:
        print(f"[ERROR] ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
