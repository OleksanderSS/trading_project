# config/tickers_config.py

TICKERS = {
    # Market Indices
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "IWM": "iShares Russell 2000 ETF", # Small-Cap
    "DIA": "SPDR Dow Jones Industrial Average ETF Trust", # 30 Large-Cap

    # Volatile Tech
    "TSLA": "Tesla, Inc.",
    "NVDA": "NVIDIA Corporation",
    "AMD": "Advanced Micro Devices, Inc.",
    "COIN": "Coinbase Global, Inc.",

    # Sector Representatives
    "JPM": "JPMorgan Chase & Co.", # Financials
    "JNJ": "Johnson & Johnson",   # Healthcare
    "XOM": "Exxon Mobil Corporation", # Energy
    "WMT": "Walmart Inc.",          # Consumer Staples
    "CAT": "Caterpillar Inc.",      # Industrials

    # International Markets
    "EEM": "iShares MSCI Emerging Markets ETF",
    "FXI": "iShares China Large-Cap ETF",

    # Currencies & Commodities
    "EURUSD=X": "EUR/USD", 
    "JPY=X": "USD/JPY",
    "GC=F": "Gold",
    "CL=F": "Crude Oil",

    # Crypto
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}

# --- Keyword Dictionary for News Filtering and Ticker Association ---
KEYWORDS = {
    # 1. Direct Ticker Mentions (Highest Priority)
    'tickers': list(TICKERS.keys()),

    # 2. General Market & Macroeconomic Terms (Associated with SPY)
    'macro_spy': [
        "stock market", "investing", "economy", "federal reserve", "fed", "fomc", "interest rates",
        "inflation", "cpi", "ppi", "recession", "economic growth", "gdp", "unemployment",
        "consumer confidence", "monetary policy", "fiscal policy", "market sentiment",
        "s&p 500", "dow jones", "market futures"
    ],

    # 3. Technology Sector Terms (Associated with QQQ)
    'tech_qqq': [
        "tech stocks", "nasdaq", "software", "semiconductor", "ai", "cloud computing",
        "innovation", "big data", "cybersecurity", "hardware", "chip maker"
    ],

    # 4. Crypto-Specific Terms
    'crypto_btc': [
        "bitcoin", "btc"
    ],
    'crypto_eth': [
        "ethereum", "eth"
    ],
    'crypto_general': [
        "crypto", "cryptocurrency", "blockchain", "coinbase", "binance", "sec", "etf",
        "digital asset", "altcoin"
    ],
    
    # 5. Commodity & Forex Terms
    'commodities_general': [
        "commodities", "raw materials", "energy crisis"
    ],
    'oil': [
        "oil", "crude", "opec", "wti", "brent"
    ],
    'gold': [
        "gold", "xau", "precious metal", "gold prices"
    ],
    'forex': [
        "forex", "currency", "dollar", "eur", "jpy", "gbp", "foreign exchange" 
    ]
}
