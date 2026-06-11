#!/usr/bin/env python3
"""
Аналіз якості новинних даних після обробки.
Перевіряє скільки новин було відфільтровано і чому.
"""

import sys
from pathlib import Path
import pandas as pd
import json
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def analyze_news_quality():
    """Аналізує якість новинних даних."""
    
    print("=" * 80)
    print("NEWS DATA QUALITY ANALYSIS")
    print("=" * 80)
    
    # Шляхи до даних
    batch_dir = project_root / "data" / "colab" / "accumulated" / "main_database"
    
    if not batch_dir.exists():
        print(f"\nError: Batch directory not found: {batch_dir}")
        return
    
    # 1. Перевірити metadata
    metadata_path = batch_dir / "batch_metadata.json"
    if metadata_path.exists():
        print("\n1. BATCH METADATA")
        print("-" * 80)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Batch name: {metadata.get('batch_name', 'N/A')}")
        print(f"Created at: {metadata.get('created_at', 'N/A')}")
        print(f"Features shape: {metadata.get('features_shape', 'N/A')}")
        print(f"Targets shape: {metadata.get('targets_shape', 'N/A')}")
    
    # 2. Перевірити features
    features_path = batch_dir / "features.parquet"
    if features_path.exists():
        print("\n2. FEATURES DATASET")
        print("-" * 80)
        features_df = pd.read_parquet(features_path)
        
        print(f"Shape: {features_df.shape}")
        print(f"Columns: {len(features_df.columns)}")
        print(f"Rows: {len(features_df)}")
        
        # Перевірити новинні колонки
        news_cols = [col for col in features_df.columns if 'news' in col.lower() or 'sentiment' in col.lower()]
        print(f"\nNews-related columns: {len(news_cols)}")
        
        if news_cols:
            print("\nNews columns sample:")
            for col in news_cols[:10]:
                non_null = features_df[col].notna().sum()
                null_pct = (features_df[col].isna().sum() / len(features_df)) * 100
                print(f"  {col:40s}: {non_null:6d} non-null ({null_pct:5.1f}% null)")
        
        # Перевірити context_map
        if 'context_map' in features_df.columns:
            print(f"\nContext map:")
            print(f"  Unique contexts: {features_df['context_map'].nunique()}")
            print(f"  Non-null: {features_df['context_map'].notna().sum()}")
    
    # 3. Перевірити targets
    targets_path = batch_dir / "targets.parquet"
    if targets_path.exists():
        print("\n3. TARGETS DATASET")
        print("-" * 80)
        targets_df = pd.read_parquet(targets_path)
        
        print(f"Shape: {targets_df.shape}")
        print(f"Columns: {list(targets_df.columns)}")
        print(f"Rows: {len(targets_df)}")
    
    # 4. Перевірити логи на помилки
    print("\n4. ERROR ANALYSIS")
    print("-" * 80)
    
    log_dir = project_root / "logs"
    if log_dir.exists():
        # Знайти останній лог файл
        log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if log_files:
            latest_log = log_files[0]
            print(f"Analyzing: {latest_log.name}")
            
            with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
            
            # Рахуємо помилки
            news_errors = log_content.count("Error processing news")
            datetime_errors = log_content.count("Invalid comparison between dtype=datetime")
            
            print(f"\nError counts:")
            print(f"  News processing errors: {news_errors}")
            print(f"  Datetime comparison errors: {datetime_errors}")
            
            # Шукаємо статистику фільтрації
            if "Intelligent news filtering:" in log_content:
                import re
                pattern = r"Intelligent news filtering: (\d+) → (\d+) \(removed (\d+)\)"
                matches = re.findall(pattern, log_content)
                
                if matches:
                    print(f"\nNews filtering statistics:")
                    for before, after, removed in matches[-3:]:  # Останні 3
                        removal_pct = (int(removed) / int(before)) * 100 if int(before) > 0 else 0
                        print(f"  {before} → {after} (removed {removed}, {removal_pct:.1f}%)")
    
    # 5. Рекомендації
    print("\n5. RECOMMENDATIONS")
    print("-" * 80)
    
    if news_errors > 0:
        print(f"⚠️  Found {news_errors} news processing errors")
        print("   → Check datetime timezone handling in NewsContextDatasetBuilder")
        print("   → Verify published_date format in news data")
    
    if datetime_errors > 0:
        print(f"⚠️  Found {datetime_errors} datetime comparison errors")
        print("   → Fixed: Use replace(tzinfo=None) instead of tz_localize(None)")
        print("   → Re-run pipeline to apply fix")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    try:
        analyze_news_quality()
    except Exception:
        logger.error("Error in analyze_news_quality", exc_info=True)
        sys.exit(1)
