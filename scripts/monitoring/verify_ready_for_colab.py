#!/usr/bin/env python3
"""
Швидка перевірка готовності до переносу в Colab.
"""

import pandas as pd
import json
from pathlib import Path
import sys

def check_ready_for_colab():
    """Перевірити готовність до переносу в Colab."""
    
    print("=" * 80)
    print("🔍 ПЕРЕВІРКА ГОТОВНОСТІ ДО COLAB")
    print("=" * 80)
    
    batch_dir = Path('data/colab/accumulated/full_pipeline_trading/full_pipeline_trading')
    
    # Check if directory exists
    if not batch_dir.exists():
        print("\n❌ ПОМИЛКА: Директорія не знайдена!")
        print(f"   Шлях: {batch_dir}")
        return False
    
    print(f"\n✅ Директорія знайдена: {batch_dir}")
    
    # Check files
    required_files = {
        'features.parquet': None,
        'targets.parquet': None,
        'batch_metadata.json': None
    }
    
    print("\n📦 ПЕРЕВІРКА ФАЙЛІВ:")
    all_files_exist = True
    
    for filename in required_files.keys():
        filepath = batch_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024**2
            required_files[filename] = size_mb
            print(f"   ✅ {filename}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {filename}: НЕ ЗНАЙДЕНО")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ ПОМИЛКА: Не всі файли присутні!")
        return False
    
    # Load and check data
    print("\n📊 ПЕРЕВІРКА ДАНИХ:")
    
    try:
        features = pd.read_parquet(batch_dir / 'features.parquet')
        targets = pd.read_parquet(batch_dir / 'targets.parquet')
        
        with open(batch_dir / 'batch_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Check shapes
        print(f"   Features shape: {features.shape}")
        print(f"   Targets shape: {targets.shape}")
        
        # Check tickers
        tickers = sorted(features['ticker'].unique())
        print(f"   Tickers ({len(tickers)}): {', '.join(tickers)}")
        
        # Check timeframes
        if 'interval' in features.columns:
            timeframes = sorted(features['interval'].unique())
            print(f"   Timeframes ({len(timeframes)}): {', '.join(timeframes)}")
        
        # Check NaN
        nan_pct = (features.isna().sum().sum() / (len(features) * len(features.columns))) * 100
        print(f"   NaN percentage: {nan_pct:.2f}%")
        
        # Check datetime
        if 'datetime' in features.columns:
            nat_count = features['datetime'].isna().sum()
            print(f"   NaT count: {nat_count}")
        
        # Check context_fingerprint
        has_context = 'context_fingerprint' in features.columns
        print(f"   context_fingerprint: {'✅ Присутній' if has_context else '❌ Відсутній'}")
        
    except Exception as e:
        print(f"\n❌ ПОМИЛКА при завантаженні даних: {e}")
        return False
    
    # Validation checks
    print("\n✅ ВАЛІДАЦІЯ:")
    
    checks = []
    
    # Check 1: Minimum rows
    if len(features) >= 50000:
        print(f"   ✅ Достатньо рядків: {len(features)}")
        checks.append(True)
    else:
        print(f"   ❌ Недостатньо рядків: {len(features)} (мінімум 50,000)")
        checks.append(False)
    
    # Check 2: Minimum tickers
    if len(tickers) >= 10:
        print(f"   ✅ Достатньо тікерів: {len(tickers)}")
        checks.append(True)
    else:
        print(f"   ❌ Недостатньо тікерів: {len(tickers)} (мінімум 10)")
        checks.append(False)
    
    # Check 3: NaN percentage
    if nan_pct < 5:
        print(f"   ✅ NaN прийнятний: {nan_pct:.2f}%")
        checks.append(True)
    else:
        print(f"   ⚠️ Високий NaN: {nan_pct:.2f}% (рекомендовано < 5%)")
        checks.append(False)
    
    # Check 4: No NaT in datetime
    if nat_count == 0:
        print("   ✅ Немає NaT в datetime")
        checks.append(True)
    else:
        print(f"   ❌ Є NaT в datetime: {nat_count}")
        checks.append(False)
    
    # Check 5: context_fingerprint
    if has_context:
        print("   ✅ context_fingerprint присутній")
        checks.append(True)
    else:
        print("   ⚠️ context_fingerprint відсутній (буде обчислено в Colab)")
        checks.append(True)  # Not critical
    
    # Check 6: Minimum features
    if features.shape[1] >= 100:
        print(f"   ✅ Достатньо фіч: {features.shape[1]}")
        checks.append(True)
    else:
        print(f"   ❌ Недостатньо фіч: {features.shape[1]} (мінімум 100)")
        checks.append(False)
    
    # Check 7: Minimum targets
    target_cols = [c for c in targets.columns if c.startswith('target_')]
    if len(target_cols) >= 10:
        print(f"   ✅ Достатньо таргетів: {len(target_cols)}")
        checks.append(True)
    else:
        print(f"   ❌ Недостатньо таргетів: {len(target_cols)} (мінімум 10)")
        checks.append(False)
    
    # Final result
    print("\n" + "=" * 80)
    
    if all(checks):
        print("✅ ВСЕ ПЕРЕВІРКИ ПРОЙДЕНО - ГОТОВО ДО ПЕРЕНОСУ В COLAB!")
        print("=" * 80)
        
        # Print transfer instructions
        print("\n🚀 НАСТУПНІ КРОКИ:")
        print("\n1. Запакувати дані:")
        print("   cd data/colab/accumulated/")
        print("   zip -r full_pipeline_trading.zip full_pipeline_trading/")
        print("\n2. Завантажити в Colab:")
        print("   from google.colab import files")
        print("   uploaded = files.upload()")
        print("   !unzip full_pipeline_trading.zip")
        print("\n3. Запустити тренування:")
        print("   !python colab_clean_cell.py")
        print("\n📚 Детальна інструкція: COLAB_TRANSFER_CHECKLIST.md")
        
        return True
    else:
        failed_count = len([c for c in checks if not c])
        print(f"❌ ПЕРЕВІРКИ НЕ ПРОЙДЕНО: {failed_count}/{len(checks)} помилок")
        print("=" * 80)
        print("\n⚠️ Виправте помилки перед переносом в Colab!")
        print("📚 Детальна інформація: TROUBLESHOOTING.md")
        
        return False

if __name__ == '__main__':
    success = check_ready_for_colab()
    sys.exit(0 if success else 1)
