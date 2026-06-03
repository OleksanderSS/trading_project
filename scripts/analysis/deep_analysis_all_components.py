#!/usr/bin/env python3
"""
Глибокий аналіз всіх компонентів: таргети, колектори, макро-фічі.
"""

import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

def analyze_all_components():
    """Детальний аналіз всіх компонентів."""
    
    print("=" * 100)
    print("🔍 ГЛИБОКИЙ АНАЛІЗ ВСІХ КОМПОНЕНТІВ")
    print("=" * 100)
    
    batch_dir = Path('data/colab/accumulated/full_pipeline_trading/full_pipeline_trading')
    
    # Load data
    features = pd.read_parquet(batch_dir / 'features.parquet')
    targets = pd.read_parquet(batch_dir / 'targets.parquet')
    
    print(f"\n📊 ЗАГАЛЬНА ІНФОРМАЦІЯ:")
    print(f"   Features: {features.shape}")
    print(f"   Targets: {targets.shape}")
    
    # ============================================================================
    # 1. АНАЛІЗ ТАРГЕТІВ ПО ТАЙМФРЕЙМАМ
    # ============================================================================
    print("\n" + "=" * 100)
    print("🎯 АНАЛІЗ ТАРГЕТІВ ПО ТАЙМФРЕЙМАМ")
    print("=" * 100)
    
    target_cols = [c for c in targets.columns if c.startswith('target_')]
    
    if 'interval' in targets.columns:
        timeframes = sorted(targets['interval'].unique())
        
        for tf in timeframes:
            tf_data = targets[targets['interval'] == tf]
            print(f"\n📈 Таймфрейм: {tf}")
            print(f"   Рядків: {len(tf_data)}")
            
            for target_col in target_cols:
                if target_col in tf_data.columns:
                    non_null = tf_data[target_col].notna().sum()
                    pct = (non_null / len(tf_data)) * 100 if len(tf_data) > 0 else 0
                    
                    status = "✅" if pct > 90 else "⚠️" if pct > 50 else "❌"
                    print(f"      {status} {target_col}: {non_null}/{len(tf_data)} ({pct:.1f}%)")
    else:
        print("\n⚠️ Колонка 'interval' не знайдена в targets")
    
    # ============================================================================
    # 2. АНАЛІЗ КОЛЕКТОРІВ (по колонках features)
    # ============================================================================
    print("\n" + "=" * 100)
    print("📡 АНАЛІЗ КОЛЕКТОРІВ")
    print("=" * 100)
    
    # Identify collector sources by column patterns
    collectors = {
        'YFinance (OHLCV)': ['open', 'high', 'low', 'close', 'volume'],
        'Technical Indicators': ['rsi', 'sma', 'ema', 'macd', 'bb', 'atr', 'stoch', 'adx', 'cci', 'roc', 'willr'],
        'Volatility': ['volatility', 'atr', 'bb_width', 'keltner', 'donchian'],
        'Volume': ['volume_ratio', 'obv', 'vwap', 'mfi', 'ad', 'cmf'],
        'Context': ['context', 'regime', 'phase', 'trend', 'cycle'],
        'News': ['news', 'sentiment', 'impact', 'headline'],
        'Macro': ['macro', 'fed', 'gdp', 'inflation', 'unemployment', 'cpi', 'ppi', 'retail', 'housing', 'manufacturing']
    }
    
    feature_cols = [c for c in features.columns if c not in ['ticker', 'interval', 'datetime']]
    
    for collector_name, patterns in collectors.items():
        matching_cols = []
        for col in feature_cols:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in patterns):
                matching_cols.append(col)
        
        status = "✅" if len(matching_cols) > 0 else "❌"
        print(f"\n{status} {collector_name}: {len(matching_cols)} колонок")
        
        if len(matching_cols) > 0:
            # Show first 10 columns
            for col in matching_cols[:10]:
                non_null = features[col].notna().sum()
                pct = (non_null / len(features)) * 100
                print(f"      {col}: {pct:.1f}% non-null")
            
            if len(matching_cols) > 10:
                print(f"      ... та ще {len(matching_cols) - 10} колонок")
    
    # ============================================================================
    # 3. ДЕТАЛЬНИЙ АНАЛІЗ МАКРО-ФІЧ
    # ============================================================================
    print("\n" + "=" * 100)
    print("📊 ДЕТАЛЬНИЙ АНАЛІЗ МАКРО-ФІЧ")
    print("=" * 100)
    
    macro_patterns = ['macro', 'fed', 'gdp', 'inflation', 'unemployment', 'cpi', 'ppi', 
                     'retail', 'housing', 'manufacturing', 'durable', 'consumer', 
                     'industrial', 'capacity', 'sentiment', 'confidence', 'pmi',
                     'treasury', 'yield', 'rate', 'bond']
    
    macro_cols = []
    for col in feature_cols:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in macro_patterns):
            macro_cols.append(col)
    
    print(f"\n📈 Знайдено макро-колонок: {len(macro_cols)}")
    
    if len(macro_cols) > 0:
        print("\n📋 Список всіх макро-колонок:")
        for i, col in enumerate(macro_cols, 1):
            non_null = features[col].notna().sum()
            pct = (non_null / len(features)) * 100
            unique_vals = features[col].nunique()
            
            status = "✅" if pct > 90 else "⚠️" if pct > 50 else "❌"
            print(f"   {i:2d}. {status} {col}")
            print(f"       Non-null: {non_null}/{len(features)} ({pct:.1f}%)")
            print(f"       Unique values: {unique_vals}")
            
            # Show sample values
            if non_null > 0:
                sample = features[col].dropna().head(3).tolist()
                print(f"       Sample: {sample}")
    else:
        print("\n⚠️ МАКРО-КОЛОНКИ НЕ ЗНАЙДЕНО!")
        print("   Можливі причини:")
        print("   - MacroDataCollector не запустився")
        print("   - Макро-дані не додалися до features")
        print("   - Назви колонок не відповідають патернам")
    
    # ============================================================================
    # 4. АНАЛІЗ ПО ТІКЕРАМ
    # ============================================================================
    print("\n" + "=" * 100)
    print("🏢 АНАЛІЗ ПО ТІКЕРАМ")
    print("=" * 100)
    
    if 'ticker' in features.columns:
        tickers = sorted(features['ticker'].unique())
        print(f"\n📊 Всього тікерів: {len(tickers)}")
        
        for ticker in tickers:
            ticker_data = features[features['ticker'] == ticker]
            ticker_targets = targets[targets['ticker'] == ticker] if 'ticker' in targets.columns else pd.DataFrame()
            
            print(f"\n   {ticker}:")
            print(f"      Features: {len(ticker_data)} рядків")
            if not ticker_targets.empty:
                print(f"      Targets: {len(ticker_targets)} рядків")
            
            # Check timeframes
            if 'interval' in ticker_data.columns:
                tf_counts = ticker_data['interval'].value_counts().to_dict()
                for tf, count in sorted(tf_counts.items()):
                    print(f"         {tf}: {count} рядків")
    
    # ============================================================================
    # 5. ПЕРЕВІРКА ENRICHERS
    # ============================================================================
    print("\n" + "=" * 100)
    print("🔧 ПЕРЕВІРКА ENRICHERS")
    print("=" * 100)
    
    enrichers = {
        'TechnicalEnricher': ['rsi', 'sma', 'ema', 'macd', 'bb', 'atr'],
        'VolatilityEnricher': ['volatility', 'atr', 'bb_width'],
        'VolumeEnricher': ['volume_ratio', 'obv', 'vwap'],
        'ContextMapEnricher': ['context_fingerprint', 'regime', 'phase'],
        'MacroEnricher': ['macro', 'fed', 'gdp', 'inflation'],
        'NewsEnricher': ['news_sentiment', 'news_impact']
    }
    
    for enricher_name, indicators in enrichers.items():
        found_indicators = []
        for indicator in indicators:
            matching = [c for c in feature_cols if indicator in c.lower()]
            if matching:
                found_indicators.extend(matching)
        
        status = "✅" if len(found_indicators) > 0 else "❌"
        coverage = len(found_indicators) / len(indicators) * 100 if indicators else 0
        
        print(f"\n{status} {enricher_name}: {len(found_indicators)}/{len(indicators)} індикаторів ({coverage:.0f}%)")
        
        if found_indicators:
            for col in found_indicators[:5]:
                non_null = features[col].notna().sum()
                pct = (non_null / len(features)) * 100
                print(f"      {col}: {pct:.1f}% non-null")
            if len(found_indicators) > 5:
                print(f"      ... та ще {len(found_indicators) - 5}")
    
    # ============================================================================
    # 6. SUMMARY
    # ============================================================================
    print("\n" + "=" * 100)
    print("📋 ПІДСУМОК")
    print("=" * 100)
    
    # Count working components
    working_targets = sum(1 for col in target_cols if targets[col].notna().sum() / len(targets) > 0.9)
    working_collectors = sum(1 for name, patterns in collectors.items() 
                            if any(any(p in c.lower() for p in patterns) for c in feature_cols))
    working_enrichers = sum(1 for name, indicators in enrichers.items()
                           if any(any(ind in c.lower() for c in feature_cols) for ind in indicators))
    
    print(f"\n✅ Працюючі таргети: {working_targets}/{len(target_cols)}")
    print(f"✅ Працюючі колектори: {working_collectors}/{len(collectors)}")
    print(f"✅ Працюючі enrichers: {working_enrichers}/{len(enrichers)}")
    print(f"✅ Макро-колонок: {len(macro_cols)}")
    
    # Critical issues
    print("\n⚠️ КРИТИЧНІ ПРОБЛЕМИ:")
    issues = []
    
    if working_targets < len(target_cols) * 0.8:
        issues.append(f"Тільки {working_targets}/{len(target_cols)} таргетів працюють")
    
    if len(macro_cols) == 0:
        issues.append("Макро-колонки відсутні")
    
    if working_collectors < len(collectors) * 0.7:
        issues.append(f"Тільки {working_collectors}/{len(collectors)} колекторів працюють")
    
    if issues:
        for issue in issues:
            print(f"   ❌ {issue}")
    else:
        print("   ✅ Критичних проблем не виявлено")
    
    print("\n" + "=" * 100)

if __name__ == '__main__':
    analyze_all_components()
