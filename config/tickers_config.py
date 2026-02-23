# config/tickers_config.py

# Список тикерів для аналізу, згрупованих за секторами для диверсифікації
TICKERS = {
    # Технологічний сектор
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "TSLA": "Tesla, Inc.",
    "NVDA": "NVIDIA Corporation",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "META": "Meta Platforms, Inc.",

    # Криптовалюти
    "BTC-USD": "Bitcoin",
}

# Ключові слова для пошуку новин
KEYWORDS = {
    "tickers": list(TICKERS.keys()),
    "general": [
        "stock market", "investing", "economy", "federal reserve", "interest rates",
        "inflation", "recession", "bull market", "bear market", "S&P 500", "NASDAQ", "Dow Jones"
    ]
}
