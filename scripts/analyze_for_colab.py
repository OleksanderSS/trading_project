#!/usr/bin/env python3
"""
Детальний аналіз даних для Colab.
"""

import pandas as pd
import json
from pathlib import Path

batch_dir = Path('data/colab/accumulated/full_pipeline_trading/full_pipeline_trading')

print("=" * 80)
print("📊 АНАЛІЗ ДАНИХ ДЛЯ COLAB")
print("=" * 80)

# Load data
features = pd.read_parquet(batch_dir / 'features.parquet')
targets = pd.read_parquet(batch_dir / 'targets.parquet')

print("\n📈 FEATURES:")
print(f"   Shape: {features.shape}")
print(f"   Memory: {features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Analyze columns
print(f"\n📋 COLUMN TYPES:")
base_cols = ['ticker', 'interval', 'datetime', 'open', 'high', 'low', 'close', 'volume']
tech_cols = [c for c in features.columns if any(x in c.lower() for x in ['rsi', 'sma', 'ema', 'macd', 'bb', 'atr'])]
vol_cols = [c for c in features.columns if 'vol' in c.lower() and c not in base_cols]
context_cols = [c for c in features.columns if any(x in c.lower() for x in ['context', 'regime', 'phase'])]
news_cols = [c for c in features.columns if 'news' in c.lower()]
macro_cols = [c for c in features.columns if any(x in c.lower() for x in ['macro', 'fed', 'gdp', 'inflation'])]

print(f"   Base columns: {len([c for c in base_cols if c in features.columns])}")
print(f"   Technical indicators: {len(tech_cols)}")
print(f"   Volatility features: {len(vol_cols)}")
print(f"   Context features: {len(context_cols)}")
print(f"   News features: {len(news_cols)}")
print(f"   Macro features: {len(macro_cols)}")

# Check for NaN
print(f"\n🔍 DATA QUALITY:")
nan_pct = (features.isna().sum().sum() / (len(features) * len(features.columns))) * 100
print(f"   NaN percentage: {nan_pct:.2f}%")

# Check datetime
print(f"\n📅 DATETIME:")
print(f"   Column exists: {'datetime' in features.columns}")
if 'datetime' in features.columns:
    print(f"   Type: {features['datetime'].dtype}")
    print(f"   Range: {features['datetime'].min()} → {features['datetime'].max()}")
    print(f"   NaT count: {features['datetime'].isna().sum()}")

# Targets
print(f"\n🎯 TARGETS:")
print(f"   Shape: {targets.shape}")
print(f"   Memory: {targets.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

target_cols = [c for c in targets.columns if c.startswith('target_')]
print(f"   Target columns ({len(target_cols)}):")
for tc in target_cols:
    non_null = targets[tc].notna().sum()
    pct = (non_null / len(targets)) * 100
    print(f"      {tc}: {non_null} non-null ({pct:.1f}%)")

# Metadata
print(f"\n📋 METADATA:")
if (batch_dir / 'batch_metadata.json').exists():
    with open(batch_dir / 'batch_metadata.json', 'r') as f:
        meta = json.load(f)
    print(f"   Batch name: {meta.get('batch_name')}")
    print(f"   Timestamp: {meta.get('timestamp')}")
    print(f"   Tickers: {meta.get('tickers')}")
    print(f"   Timeframes: {meta.get('timeframes')}")
    print(f"   Test mode: {meta.get('test_mode')}")
    print(f"   Accumulated: {meta.get('accumulated')}")

# Check for enrichment
print(f"\n🔧 ENRICHMENT STATUS:")
enricher_indicators = {
    'Technical': ['rsi', 'sma', 'ema', 'macd'],
    'Volatility': ['atr', 'bb', 'volatility'],
    'Volume': ['volume_ratio', 'obv', 'vwap'],
    'Context': ['context_fingerprint', 'regime', 'phase'],
    'News': ['news_sentiment', 'news_impact'],
    'Macro': ['fed', 'gdp', 'inflation']
}

for enricher, indicators in enricher_indicators.items():
    found = [ind for ind in indicators if any(ind in c.lower() for c in features.columns)]
    status = "✅" if found else "❌"
    print(f"   {status} {enricher}: {len(found)}/{len(indicators)} indicators")

# Files to transfer
print(f"\n📦 FILES TO TRANSFER TO COLAB:")
files = [
    batch_dir / 'features.parquet',
    batch_dir / 'targets.parquet',
    batch_dir / 'batch_metadata.json'
]

total_size = 0
for file in files:
    if file.exists():
        size_mb = file.stat().st_size / 1024**2
        total_size += size_mb
        print(f"   ✅ {file.name}: {size_mb:.1f} MB")
    else:
        print(f"   ❌ {file.name}: NOT FOUND")

print(f"\n   Total size: {total_size:.1f} MB")

# Recommendations
print(f"\n💡 RECOMMENDATIONS:")
if nan_pct > 10:
    print(f"   ⚠️ High NaN percentage ({nan_pct:.1f}%) - consider imputation")
else:
    print(f"   ✅ NaN percentage acceptable ({nan_pct:.1f}%)")

if 'context_fingerprint' not in features.columns:
    print(f"   ⚠️ context_fingerprint missing - will be computed in Colab")
else:
    print(f"   ✅ context_fingerprint present")

if total_size > 100:
    print(f"   ⚠️ Large dataset ({total_size:.1f} MB) - may take time to upload")
else:
    print(f"   ✅ Dataset size manageable ({total_size:.1f} MB)")

print("\n" + "=" * 80)
print("✅ READY FOR COLAB TRANSFER")
print("=" * 80)
