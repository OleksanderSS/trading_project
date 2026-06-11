import pandas as pd
from datetime import datetime
from src.features.builders.news_event_dataset_builder import NewsEventDatasetBuilder
from src.utils.trading_calendar import TradingCalendar

def test_instantiation():
    calendar = TradingCalendar()
    builder = NewsEventDatasetBuilder(calendar)
    print("✅ NewsEventDatasetBuilder instantiated successfully")
    
    # Mock data
    news_df = pd.DataFrame({
        'published_at': [datetime(2026, 4, 26, 12, 0)],
        'title': ['Test News'],
        'sentiment': [0.5],
        'hash': ['h1']
    })
    
    price_df = pd.DataFrame({
        'datetime': [datetime(2026, 4, 26, 11, 45), datetime(2026, 4, 26, 12, 15), datetime(2026, 4, 26, 12, 30)],
        'open': [100, 101, 102],
        'high': [101, 102, 103],
        'low': [99, 100, 101],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000, 1100, 1200],
        'ticker': ['AMD', 'AMD', 'AMD']
    })
    
    macro_df = pd.DataFrame({
        'date': [datetime(2026, 4, 26)],
        'VIXCLS': [18.5],
        'DGS10': [4.15],
        'FEDFUNDS': [5.25],
        'CPIAUCSL': [3.2],
        'UNRATE': [4.0]
    })
    
    # We need candle features in the mock price_df
    for feature in ['RSI_14', 'SMA_20', 'EMA_20', 'MACD', 'ATR_14', 'BB_upper', 'BB_lower', 'Stoch_K', 'Stoch_D']:
        price_df[feature] = 50.0
    
    price_data = {'15m': price_df, '60m': price_df, '1d': price_df}
    
    try:
        dataset = builder.build_dataset(news_df, price_data, macro_df, ['AMD'])
        print(f"✅ build_dataset result shape: {dataset.shape}")
        if not dataset.empty:
            print("✅ Dataset columns:", dataset.columns.tolist()[:10], "...")
    except Exception as e:
        print(f"❌ build_dataset failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_instantiation()
