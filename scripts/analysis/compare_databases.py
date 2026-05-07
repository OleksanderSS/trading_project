#!/usr/bin/env python3
"""
Порівняння двох баз даних.
"""

import pandas as pd
import json
from pathlib import Path

print("=" * 100)
print("📊 ПОРІВНЯННЯ БАЗ ДАНИХ")
print("=" * 100)

# main_database
print("\n" + "=" * 100)
print("📁 MAIN_DATABASE")
print("=" * 100)

main_db_path = Path('data/colab/accumulated/main_database')
if main_db_path.exists():
    try:
        features_main = pd.read_parquet(main_db_path / 'features.parquet')
        targets_main = pd.read_parquet(main_db_path / 'targets.parquet')
        
        with open(main_db_path / 'batch_metadata.json', 'r') as f:
            meta_main = json.load(f)
        
        print(f"\n✅ Features: {features_main.shape}")
        print(f"   Columns: {len(features_main.columns)}")
        if 'ticker' in features_main.columns:
            print(f"   Tickers: {features_main['ticker'].nunique()}")
            print(f"   Tickers list: {sorted(features_main['ticker'].unique())}")
        if 'interval' in features_main.columns:
            print(f"   Timeframes: {sorted(features_main['interval'].unique())}")
        
        print(f"\n✅ Targets: {targets_main.shape}")
        print(f"   Columns: {len(targets_main.columns)}")
        
        print(f"\n✅ Metadata:")
        print(f"   Batch name: {meta_main.get('batch_name')}")
        print(f"   Tickers: {len(meta_main.get('tickers', []))}")
        print(f"   Timeframes: {meta_main.get('timeframes')}")
        
        print(f"\n✅ Quality:")
        nan_pct = (features_main.isna().sum().sum() / (len(features_main) * len(features_main.columns))) * 100
        print(f"   NaN: {nan_pct:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ Directory not found")

# full_pipeline_trading
print("\n" + "=" * 100)
print("📁 FULL_PIPELINE_TRADING")
print("=" * 100)

full_db_path = Path('data/colab/accumulated/full_pipeline_trading/full_pipeline_trading')
if full_db_path.exists():
    try:
        features_full = pd.read_parquet(full_db_path / 'features.parquet')
        targets_full = pd.read_parquet(full_db_path / 'targets.parquet')
        
        with open(full_db_path / 'batch_metadata.json', 'r') as f:
            meta_full = json.load(f)
        
        print(f"\n✅ Features: {features_full.shape}")
        print(f"   Columns: {len(features_full.columns)}")
        if 'ticker' in features_full.columns:
            print(f"   Tickers: {features_full['ticker'].nunique()}")
            print(f"   Tickers list: {sorted(features_full['ticker'].unique())}")
        if 'interval' in features_full.columns:
            print(f"   Timeframes: {sorted(features_full['interval'].unique())}")
        
        print(f"\n✅ Targets: {targets_full.shape}")
        print(f"   Columns: {len(targets_full.columns)}")
        
        print(f"\n✅ Metadata:")
        print(f"   Batch name: {meta_full.get('batch_name')}")
        print(f"   Tickers: {len(meta_full.get('tickers', []))}")
        print(f"   Timeframes: {meta_full.get('timeframes')}")
        
        print(f"\n✅ Quality:")
        nan_pct = (features_full.isna().sum().sum() / (len(features_full) * len(features_full.columns))) * 100
        print(f"   NaN: {nan_pct:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ Directory not found")

# Порівняння
print("\n" + "=" * 100)
print("📊 ПОРІВНЯННЯ")
print("=" * 100)

comparison = {
    'Параметр': ['Розмір (MB)', 'Рядків', 'Колонок', 'Тікерів', 'Таймфреймів', 'NaN %', 'Статус'],
    'main_database': [
        '0.1',
        f'{len(features_main)}' if 'features_main' in locals() else '?',
        f'{len(features_main.columns)}' if 'features_main' in locals() else '?',
        f'{features_main["ticker"].nunique()}' if 'features_main' in locals() and 'ticker' in features_main.columns else '?',
        f'{len(features_main["interval"].unique())}' if 'features_main' in locals() and 'interval' in features_main.columns else '?',
        f'{nan_pct:.2f}' if 'nan_pct' in locals() else '?',
        '❌ МАЛЕНЬКИЙ' if 'features_main' in locals() and len(features_main) < 10000 else '?'
    ],
    'full_pipeline_trading': [
        '25.4',
        f'{len(features_full)}' if 'features_full' in locals() else '?',
        f'{len(features_full.columns)}' if 'features_full' in locals() else '?',
        f'{features_full["ticker"].nunique()}' if 'features_full' in locals() and 'ticker' in features_full.columns else '?',
        f'{len(features_full["interval"].unique())}' if 'features_full' in locals() and 'interval' in features_full.columns else '?',
        f'{nan_pct:.2f}' if 'nan_pct' in locals() else '?',
        '✅ ПОВНОЦІННИЙ' if 'features_full' in locals() and len(features_full) > 50000 else '?'
    ]
}

print("\n")
for i, param in enumerate(comparison['Параметр']):
    main_val = comparison['main_database'][i]
    full_val = comparison['full_pipeline_trading'][i]
    print(f"{param:20} | main_database: {main_val:15} | full_pipeline_trading: {full_val:15}")

# Рекомендація
print("\n" + "=" * 100)
print("🎯 РЕКОМЕНДАЦІЯ")
print("=" * 100)

if 'features_full' in locals() and len(features_full) > 50000:
    print("\n✅ ПЕРЕНОСИТИ: full_pipeline_trading")
    print(f"\n   Розмір: 25.4 MB")
    print(f"   Рядків: {len(features_full)}")
    print(f"   Колонок: {len(features_full.columns)}")
    print(f"   Тікерів: {features_full['ticker'].nunique()}")
    print(f"   Таймфреймів: {len(features_full['interval'].unique())}")
    print(f"   NaN: {nan_pct:.2f}%")
    print(f"\n   Команда:")
    print(f"   cd data/colab/accumulated/full_pipeline_trading/")
    print(f"   zip -r full_pipeline_trading.zip full_pipeline_trading/")
    print(f"\n   В Colab:")
    print(f"   !unzip full_pipeline_trading.zip")
    print(f"   !python colab_clean_cell.py")
else:
    print("\n❌ main_database занадто маленький")
    print(f"   Розмір: 0.1 MB")
    print(f"   Рядків: {len(features_main) if 'features_main' in locals() else '?'}")
    print(f"\n   Це старі тестові дані, не використовувати!")

print("\n" + "=" * 100)
