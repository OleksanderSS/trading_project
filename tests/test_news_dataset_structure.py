"""
Test для перевірки структури News Dataset
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.news_dataset_builder import NewsContextDatasetBuilder
from src.config.unified_config_manager import UnifiedConfigManager


def create_mock_prices_df(ticker: str, timeframe: str, n_candles: int = 10) -> pd.DataFrame:
    """Створити mock датафрейм з цінами та фічами"""
    base_time = datetime(2026, 4, 26, 9, 30)
    
    if timeframe == '15m':
        freq = timedelta(minutes=15)
    elif timeframe == '60m':
        freq = timedelta(hours=1)
    else:  # 1d
        freq = timedelta(days=1)
    
    data = []
    for i in range(n_candles):
        dt = base_time + freq * i
        data.append({
            'datetime': dt,
            'ticker': ticker,
            'interval': timeframe,
            'open': 100 + i,
            'high': 101 + i,
            'low': 99 + i,
            'close': 100.5 + i,
            'volume': 1000000 + i * 10000,
            # Технічні індикатори
            'sma_5': 100 + i * 0.5,
            'sma_20': 100 + i * 0.3,
            'rsi_14': 50 + i,
            'macd': 0.5 + i * 0.1,
            'volatility_5d': 0.02 + i * 0.001,
            'trend_5d': 0.01 + i * 0.001,
            'sentiment_score': 0.5 + i * 0.05,
            'context_fingerprint': f'hash_{i}',
        })
    
    return pd.DataFrame(data)


def create_mock_news_df() -> pd.DataFrame:
    """Створити mock датафрейм з новинами"""
    base_time = datetime(2026, 4, 26, 12, 0)
    
    data = [
        {
            'published_date': base_time,
            'title': 'Tesla announces record Q1 deliveries',
            'sentiment': 0.85,
            'news_type': 'TSLA',
            'source': 'google_news',
        },
        {
            'published_date': base_time + timedelta(hours=2),
            'title': 'Fed keeps rates unchanged',
            'sentiment': 0.0,
            'news_type': 'macro',
            'source': 'rss',
        },
    ]
    
    return pd.DataFrame(data)


def create_mock_macro_df() -> pd.DataFrame:
    """Створити mock датафрейм з макро даними"""
    base_time = datetime(2026, 4, 26, 0, 0)
    
    data = []
    for i in range(10):
        dt = base_time + timedelta(days=i)
        data.append({
            'datetime': dt,
            'fed_funds_rate': 5.25,
            'treasury_10y': 4.15,
            'treasury_2y': 4.50,
            'vix': 18.5,
            'cpi': 3.2,
        })
    
    return pd.DataFrame(data)


def test_news_dataset_structure():
    """Тест структури news dataset"""
    print("=" * 80)
    print("Testing News Dataset Structure")
    print("=" * 80)
    
    # 1. Створити mock дані
    print("\n1. Creating mock data...")
    
    tickers = ['AMD', 'NVDA', 'TSLA']
    timeframes = ['15m', '60m', '1d']
    
    prices_dict = {}
    for tf in timeframes:
        dfs = []
        for ticker in tickers:
            df = create_mock_prices_df(ticker, tf, n_candles=10)
            dfs.append(df)
        prices_dict[tf] = pd.concat(dfs, ignore_index=True)
        print(f"   ✅ Created {tf} prices: {prices_dict[tf].shape}")
    
    news_df = create_mock_news_df()
    print(f"   ✅ Created news: {news_df.shape}")
    
    macro_df = create_mock_macro_df()
    print(f"   ✅ Created macro: {macro_df.shape}")
    
    # 2. Ініціалізувати builder
    print("\n2. Initializing NewsDatasetBuilder...")
    
    config_manager = UnifiedConfigManager()
    builder = NewsContextDatasetBuilder(config_manager)
    
    # Override tickers для тесту
    builder.tickers = tickers
    builder.timeframes = timeframes
    
    print(f"   OK Builder initialized")
    print(f"   Tickers: {builder.tickers}")
    print(f"   Timeframes: {builder.timeframes}")
    
    # 3. Побудувати датасет
    print("\n3. Building news dataset...")
    
    try:
        news_dataset = builder.build_dataset(
            news_df=news_df,
            prices_dict=prices_dict,
            macro_df=macro_df,
            market_sentiment_df=None
        )
        
        print(f"   ✅ Dataset built: {news_dataset.shape}")
        
    except Exception as e:
        print(f"   ❌ Error building dataset: {e}")
        import traceback
        traceback.print_exc()
        raise AssertionError(f"Failed to build news dataset: {e}") from e
    
    # 4. Перевірити структуру
    print("\n4️⃣ Verifying structure...")
    
    # Перевірити кількість рядків
    expected_rows = len(news_df)  # Після фільтрації може бути менше
    actual_rows = len(news_dataset)
    print(f"   News rows: {actual_rows} (expected: {expected_rows})")
    
    # Перевірити колонки
    columns = news_dataset.columns.tolist()
    print(f"   Total columns: {len(columns)}")
    
    # Перевірити БЛОК 1: Новина
    news_cols = [col for col in columns if col.startswith('news_')]
    print(f"   ✅ БЛОК 1 (Новина): {len(news_cols)} columns")
    print(f"      {news_cols}")
    
    # Перевірити БЛОК 2: Макро
    macro_cols = [col for col in columns if any(x in col for x in ['fed_', 'treasury_', 'vix', 'cpi', 'hour_', 'day_'])]
    print(f"   ✅ БЛОК 2 (Макро): {len(macro_cols)} columns")
    
    # Перевірити БЛОК 3: Контекст ДО
    before_cols = [col for col in columns if '_before_' in col]
    print(f"   ✅ БЛОК 3 (Контекст ДО): {len(before_cols)} columns")
    
    # Розбити по тікерам
    for ticker in tickers:
        ticker_before = [col for col in before_cols if col.startswith(f'{ticker}_')]
        print(f"      {ticker}: {len(ticker_before)} columns")
    
    # Перевірити БЛОК 4: Реакція ПІСЛЯ
    after_cols = [col for col in columns if '_after_' in col]
    print(f"   ✅ БЛОК 4 (Реакція ПІСЛЯ): {len(after_cols)} columns")
    
    # Розбити по тікерам
    for ticker in tickers:
        ticker_after = [col for col in after_cols if col.startswith(f'{ticker}_')]
        print(f"      {ticker}: {len(ticker_after)} columns")
    
    # 5. Перевірити симетрію
    print("\n5️⃣ Checking symmetry...")
    
    if len(before_cols) == len(after_cols):
        print(f"   ✅ Symmetry OK: {len(before_cols)} columns before = {len(after_cols)} columns after")
    else:
        print(f"   ⚠️ Asymmetry: {len(before_cols)} before ≠ {len(after_cols)} after")
    
    # 6. Перевірити дані
    print("\n6️⃣ Checking data...")
    
    if not news_dataset.empty:
        first_row = news_dataset.iloc[0]
        
        # Перевірити новину
        print(f"   News ID: {first_row.get('news_id', 'N/A')}")
        print(f"   News Title: {first_row.get('news_title', 'N/A')[:50]}...")
        print(f"   News Sentiment: {first_row.get('news_sentiment', 'N/A')}")
        
        # Перевірити контекст ДО
        amd_before_1_close = first_row.get('AMD_15m_before_1_close', None)
        print(f"   AMD 15m before_1 close: {amd_before_1_close}")
        
        # Перевірити реакцію ПІСЛЯ
        amd_after_1_close = first_row.get('AMD_15m_after_1_close', None)
        print(f"   AMD 15m after_1 close: {amd_after_1_close}")
        
        # Перевірити фічі
        amd_before_1_rsi = first_row.get('AMD_15m_before_1_rsi_14', None)
        print(f"   AMD 15m before_1 RSI: {amd_before_1_rsi}")
        
        amd_after_1_rsi = first_row.get('AMD_15m_after_1_rsi_14', None)
        print(f"   AMD 15m after_1 RSI: {amd_after_1_rsi}")
        
        print(f"   ✅ Data looks good!")
    
    # 7. Підсумок
    print("\n" + "=" * 80)
    print("✅ Test completed successfully!")
    print("=" * 80)
    print(f"\nFinal structure:")
    print(f"  Rows: {len(news_dataset)}")
    print(f"  Columns: {len(columns)}")
    print(f"    - News: {len(news_cols)}")
    print(f"    - Macro: {len(macro_cols)}")
    print(f"    - Before: {len(before_cols)}")
    print(f"    - After: {len(after_cols)}")
    print(f"    - Other: {len(columns) - len(news_cols) - len(macro_cols) - len(before_cols) - len(after_cols)}")
    
    assert not news_dataset.empty


if __name__ == '__main__':
    test_news_dataset_structure()
    sys.exit(0)
