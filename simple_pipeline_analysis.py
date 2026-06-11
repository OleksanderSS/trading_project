#!/usr/bin/env python3
"""
Простий скрипт для аналізу існуючих даних пайплайну.

Перевіряє наявність даних на кожному етапі та аналізує їх структуру.
"""

import sys
from pathlib import Path

import pandas as pd

# Налаштування кодування для Windows
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')


def print_section_header(title: str):
    """Вивести заголовок секції."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def analyze_dataframe(df: pd.DataFrame, name: str):
    """Проаналізувати DataFrame."""
    if df is None or df.empty:
        print(f"⚠️  {name}: DataFrame порожній або None")
        return
    
    print(f"📊 {name}:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"   Columns: {', '.join(df.columns[:10])}")
    if len(df.columns) > 10:
        print(f"   ... and {len(df.columns) - 10} more")
    
    if not df.empty:
        print(f"\n   Sample data (first 3 rows):")
        for i, row in enumerate(df.head(3).to_dict(orient='records'), 1):
            print(f"   Row {i}: {dict(list(row.items())[:5])}")
    
    print()


def check_stage_files():
    """Перевірити наявність файлів на кожному етапі."""
    print_section_header("PIPELINE DATA ANALYSIS")
    
    # Stage 1: Raw Data
    print_section_header("STAGE 1: RAW DATA")
    raw_data_paths = [
        "data/raw/market_data.parquet",
        "data/raw/news_data.parquet",
        "data/raw/economic_data.parquet",
    ]
    
    for path in raw_data_paths:
        file_path = Path(path)
        if file_path.exists():
            print(f"✅ Found: {path}")
            try:
                df = pd.read_parquet(file_path)
                analyze_dataframe(df, path)
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
        else:
            print(f"❌ Not found: {path}")
    
    # Stage 2: Processed Data
    print_section_header("STAGE 2: PROCESSED DATA")
    processed_data_paths = [
        "data/processed/market_data.parquet",
        "data/processed/cleaned_data.parquet",
    ]
    
    for path in processed_data_paths:
        file_path = Path(path)
        if file_path.exists():
            print(f"✅ Found: {path}")
            try:
                df = pd.read_parquet(file_path)
                analyze_dataframe(df, path)
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
        else:
            print(f"❌ Not found: {path}")
    
    # Stage 3: Features
    print_section_header("STAGE 3: FEATURES")
    features_paths = [
        "data/processed/features/features.parquet",
        "data/processed/features/targets.parquet",
    ]
    
    for path in features_paths:
        file_path = Path(path)
        if file_path.exists():
            print(f"✅ Found: {path}")
            try:
                df = pd.read_parquet(file_path)
                analyze_dataframe(df, path)
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
        else:
            print(f"❌ Not found: {path}")
    
    # Stage 4: Models
    print_section_header("STAGE 4: MODELS")
    models_paths = [
        "data/models/light_models/",
        "data/models/heavy_models/",
    ]
    
    for path in models_paths:
        file_path = Path(path)
        if file_path.exists() and file_path.is_dir():
            print(f"✅ Found directory: {path}")
            files = list(file_path.rglob("*.pkl")) + list(file_path.rglob("*.joblib"))
            print(f"   Model files: {len(files)}")
            for f in files[:5]:
                print(f"   - {f.name}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")
        else:
            print(f"❌ Not found: {path}")
    
    # Stage 5: Predictions
    print_section_header("STAGE 5: PREDICTIONS")
    predictions_paths = [
        "data/predictions/predictions.parquet",
    ]
    
    for path in predictions_paths:
        file_path = Path(path)
        if file_path.exists():
            print(f"✅ Found: {path}")
            try:
                df = pd.read_parquet(file_path)
                analyze_dataframe(df, path)
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
        else:
            print(f"❌ Not found: {path}")
    
    # Stage 6: Trading
    print_section_header("STAGE 6: TRADING")
    trading_paths = [
        "data/trading/portfolio_history.parquet",
        "data/trading/trades.parquet",
    ]
    
    for path in trading_paths:
        file_path = Path(path)
        if file_path.exists():
            print(f"✅ Found: {path}")
            try:
                df = pd.read_parquet(file_path)
                analyze_dataframe(df, path)
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
        else:
            print(f"❌ Not found: {path}")
    
    # Stage 7: Evaluation
    print_section_header("STAGE 7: EVALUATION")
    evaluation_paths = [
        "data/evaluation/metrics.json",
        "data/evaluation/report.json",
    ]
    
    for path in evaluation_paths:
        file_path = Path(path)
        if file_path.exists():
            print(f"✅ Found: {path}")
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   Keys: {list(data.keys())}")
            except Exception as e:
                print(f"❌ Error reading {path}: {e}")
        else:
            print(f"❌ Not found: {path}")
    
    # Colab Data
    print_section_header("COLAB DATA")
    colab_paths = [
        "data/colab/accumulated/",
    ]
    
    for path in colab_paths:
        file_path = Path(path)
        if file_path.exists() and file_path.is_dir():
            print(f"✅ Found directory: {path}")
            batches = list(file_path.iterdir())
            print(f"   Batches: {len(batches)}")
            for batch in batches[:3]:
                print(f"   - {batch.name}")
                # Check for features/targets in batch
                features = batch / "features.parquet"
                targets = batch / "targets.parquet"
                if features.exists():
                    print(f"     ✅ features.parquet")
                if targets.exists():
                    print(f"     ✅ targets.parquet")
            if len(batches) > 3:
                print(f"   ... and {len(batches) - 3} more batches")
        else:
            print(f"❌ Not found: {path}")


def main():
    """Головна функція."""
    print("🔍 Pipeline Data Analysis")
    print("=" * 80)
    
    check_stage_files()
    
    print_section_header("SUMMARY")
    print("Аналіз завершено. Перевірте вивід вище для детальної інформації про наявність даних на кожному етапі.")


if __name__ == "__main__":
    main()
