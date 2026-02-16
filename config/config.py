
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Завантаження .env з батьківської папки
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- Lookback Periods in Days ---
FINANCIAL_LOOKBACK_DAYS = 365 * 4
NEWS_LOOKBACK_DAYS = 90
MACRO_LOOKBACK_DAYS = 365 * 10 # 10 years for macroeconomic data

# --- Date Range Calculation ---
def get_date_range(range_type: str) -> (str, str):
    """
    Returns the start and end dates for data collection based on a type string.
    e.g., "financial", "news", "macro"
    """
    lookback_map = {
        'financial': FINANCIAL_LOOKBACK_DAYS,
        'news': NEWS_LOOKBACK_DAYS,
        'macro': MACRO_LOOKBACK_DAYS,
    }

    # Get the number of days to look back
    # Default to 5 years if the type is not found
    lookback_days = lookback_map.get(range_type, 365 * 5)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


# --- Specific Date Overrides ---
# These can be used to override the dynamic date ranges calculated above
START_FINANCIAL, END_FINANCIAL = get_date_range('financial')
START_NEWS_INTRADAY, END_NEWS_INTRADAY = get_date_range('news')


# API ключі та налаштування
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# --- Налаштування бази даних ---
USE_MEMORY_DB = True
DB_PATH = ":memory:" if USE_MEMORY_DB else "data/trading_data.db"

# --- Шляхи до директорій ---
PATHS = {
    'data': 'data',
    'logs': 'logs',
    'stages': 'data/stages',
    'results': 'results',
    'output': 'output',
    'db': DB_PATH
}


# --- Налаштування збору даних ---
# Modified by Gemini: Diversified tickers across sectors
TICKERS = {
    # Indices
    'SPY': 'SPDR S&P 500 ETF Trust',
    'QQQ': 'Invesco QQQ Trust',
    # Technology
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'NVDA': 'NVIDIA Corporation',
    # 'GOOGL': 'Alphabet Inc.',
    # 'AMZN': 'Amazon.com, Inc.',
    'META': 'Meta Platforms, Inc.',
    # Finance
    'JPM': 'JPMorgan Chase & Co.',
    'GS': 'The Goldman Sachs Group, Inc.',
    # Health Care
    'JNJ': 'Johnson & Johnson',
    'PFE': 'Pfizer Inc.',
    # Industrials
    'CAT': 'Caterpillar Inc.',
    # Energy
    'XOM': 'Exxon Mobil Corporation',
    # Consumer Discretionary
    'TSLA': 'Tesla, Inc.',
    # Crypto
    'BTC-USD': 'Bitcoin USD'
}


# Yahoo Finance Intervals
DATA_INTERVALS = {
    '5m': '5m',
    '15m': '15m',
    '60m': '60m',
    '1d': '1d'
}

YF_MAX_PERIODS = {
    '1m': '7d',
    '5m': '60d',
    '15m': '60d',
    '30m': '60d',
    '60m': '730d',
    '1h': '730d',
    '1d': 'max',
    '1wk': 'max',
    '1mo': 'max'
}

# FRED Economic Data Series IDs
FRED_SERIES = {
    'FEDFUNDS': 'FEDFUNDS', # Federal Funds Rate
    'T10Y2Y': 'T10Y2Y', # 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
    'UNRATE': 'UNRATE', # Unemployment Rate
    'GS10': 'GS10', # 10-Year Treasury Constant Maturity Rate
    'GS2': 'GS2', # 2-Year Treasury Constant Maturity Rate
    'CPI': 'CPIAUCSL', # Consumer Price Index for All Urban Consumers
    'VIX': 'VIXCLS', # CBOE Volatility Index
    'DGS10': 'DGS10', # 10-Year Treasury
    'GDP': 'GDP',
    'INDUSTRIAL_PRODUCTION': 'INDPRO',
    'CAPACITY_UTIL': 'CAPUTLB50001SQ',
    'NONFARM_PAYROLLS': 'PAYEMS',
    'CONSUMER_SENTIMENT': 'UMCSENT',
    'HOUSING_STARTS': 'HOUST',
    'BUILDING_PERMITS': 'PERMIT',
    # 'TED_SPREAD': 'TEDRATE', # Disabled by Gemini - Data source is often unavailable
    'HIGH_YIELD_SPREAD': 'BAMLH0A0HYM2',
    'USD_EUR': 'DEXUSEU',
    'WTI_OIL': 'DCOILWTICO',
    'USD_CNY': 'DEXCHUS',
    'REAL_DISPOSABLE_INCOME': 'DSPIC96',
    'RETAIL_SALES': 'RSAFS',
    'TOTAL_VEHICLE_SALES': 'TOTALSA'
}

RSS_FEEDS = {
    "investing_com": {"url": "https://www.investing.com/rss/news.rss", "relevance_threshold": 0.5},
    "seeking_alpha": {"url": "https://seekingalpha.com/feed.xml", "relevance_threshold": 0.6},
    "benzinga": {"url": "https://www.benzinga.com/feed", "relevance_threshold": 0.5},
    "business_insider": {"url": "https://markets.businessinsider.com/rss/news", "relevance_threshold": 0.5},
    "ft_markets": {"url": "https://www.ft.com/markets?format=rss", "relevance_threshold": 0.7},
    "wsj_markets": {"url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "relevance_threshold": 0.7},
    # "zerohedge": {"url": "https://www.zerohedge.com/feed", "relevance_threshold": 0.4}, # Disabled - feed is not valid XML
    "cointelegraph_crypto": {"url": "https://cointelegraph.com/rss", "relevance_threshold": 0.5},
    # "npr_business": {"url": "https://feeds.npr.org/1004/feed.json", "relevance_threshold": 0.4} # Disabled - feed is not valid XML
}

# --- Налаштування обробки подій ---
# Вікно даних (у хвилинах), необхідне *до* події для розрахунку технічних індикаторів
REQUIRED_TA_WINDOW = {
    '5m': 1000,   # 200 свічок * 5 хв
    '15m': 3000,  # 200 свічок * 15 хв
    '60m': 12000, # 200 свічок * 60 хв
    '1d': 288000 # 200 свічок * 24 год * 60 хв
}

# Кількість свічок *після* події, необхідних для визначення результату
POST_EVENT_HORIZON = 2

# Часові рамки та їх властивості (у хвилинах)
TIME_FRAMES = {
    '5m': {'minutes': 5},
    '15m': {'minutes': 15},
    '60m': {'minutes': 60},
    '1d': {'minutes': 1440}
}

# --- Налаштування моделі ---
# (тут можна додати гіперпараметри моделі, шляхи до файлів тощо)
