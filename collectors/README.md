# Collectors Module Structure - Unified Data Collection Architecture

## 📁 New Collectors Organization

```
collectors/
├── __init__.py
├── base/                         # Base classes and interfaces
│   ├── __init__.py
│   ├── base_collector.py         # Base collector class
│   └── collector_interface.py    # Collector interface
│
├── market/                       # Market data collectors
│   ├── __init__.py
│   ├── yf_collector.py           # Yahoo Finance collector
│   ├── crypto_price_collector.py # Crypto prices
│   └── custom_csv_collector.py   # Custom CSV data
│
├── news/                         # News collectors
│   ├── __init__.py
│   ├── news_collector.py         # Main news collector
│   ├── rss_collector.py          # RSS news collector
│   ├── google_news_collector.py  # Google News collector
│   └── newsapi_collector.py      # NewsAPI collector
│
├── economic/                     # Economic data collectors
│   ├── __init__.py
│   ├── fred_collector.py         # FRED economic data
│   └── economic_calendar_collector.py # Economic calendar
│
├── alternative/                  # Alternative data collectors
│   ├── __init__.py
│   ├── insider_collector.py      # Insider trading data
│   ├── sec_filings_collector.py  # SEC filings
│   ├── free_google_trends_collector.py # Google Trends
│   └── hf_collector.py           # Hedge fund data
│
├── config/                       # Configuration and management
│   ├── __init__.py
│   ├── config_manager.py         # Configuration management
│   ├── collectors_config.json    # Main configuration
│   ├── error_handler.py          # Error handling
│   └── retry_manager.py          # Retry logic
│
└── utils/                        # Collector utilities
    ├── __init__.py
    └── cache_utils.py            # Caching utilities
```

## 🎯 Key Changes Made

### ✅ Removed Duplicates
- **enhanced_newsapi_collector.py** → integrated into **news/newsapi_collector.py**
- **collectors_config_backup.json** → removed (use main config)

### ✅ Organized by Data Type
- **Base**: Base classes and interfaces
- **Market**: Market data (Yahoo Finance, Crypto, CSV)
- **News**: News data (RSS, Google News, NewsAPI)
- **Economic**: Economic data (FRED, Calendar)
- **Alternative**: Alternative data (Insider, SEC, Trends)

### ✅ Unified Configuration
- **Single config file**: `config/collectors_config.json`
- **Centralized error handling**: `config/error_handler.py`
- **Unified retry logic**: `config/retry_manager.py`

## 🚀 Usage Examples

### Market Data Collection
```python
from collectors.market.yf_collector import YFCollector

collector = YFCollector(lookback_days=30)
data = collector.collect_data("SPY", period="1mo")
```

### News Collection
```python
from collectors.news.rss_collector import RSSCollector
from collectors.news.newsapi_collector import NewsAPICollector

rss_collector = RSSCollector()
newsapi_collector = NewsAPICollector()

rss_news = rss_collector.collect_news("SPY")
api_news = newsapi_collector.collect_news("SPY")
```

### Economic Data
```python
from collectors.economic.fred_collector import FREDCollector
from collectors.economic.economic_calendar_collector import EconomicCalendarCollector

fred = FREDCollector()
calendar = EconomicCalendarCollector()

gdp_data = fred.get_series("GDP")
events = calendar.get_events()
```

### Alternative Data
```python
from collectors.alternative.insider_collector import InsiderCollector
from collectors.alternative.sec_filings_collector import SECFilingsCollector

insider = InsiderCollector()
sec = SECFilingsCollector()

insider_data = insider.get_insider_data("AAPL")
filings = sec.get_filings("AAPL")
```

## 🔄 Migration Guide

### Old → New Paths
```python
# Old duplicate news collector
collectors/enhanced_newsapi_collector.py → collectors/news/newsapi_collector.py

# Existing collectors (organized)
collectors/yf_collector.py → collectors/market/yf_collector.py
collectors/rss_collector.py → collectors/news/rss_collector.py
collectors/fred_collector.py → collectors/economic/fred_collector.py
collectors/insider_collector.py → collectors/alternative/insider_collector.py

# Configuration
collectors/collectors_config_backup.json → removed
```

## 📊 Configuration Structure

### Main Configuration
```json
{
  "collectors": {
    "yf": {
      "enabled": true,
      "lookback_days": 30,
      "intervals": ["1m", "5m", "15m", "1h", "1d"]
    },
    "news": {
      "rss": {
        "enabled": true,
        "sources": ["reuters", "bloomberg", "cnbc"]
      },
      "newsapi": {
        "enabled": true,
        "api_key": "${NEWS_API_KEY}",
        "sources": ["wsj", "ft", "bloomberg"]
      }
    },
    "fred": {
      "enabled": true,
      "api_key": "${FRED_API_KEY}"
    }
  }
}
```

### Error Handling
```python
# collectors/config/error_handler.py
class CollectorErrorHandler:
    def handle_error(self, error, collector_name):
        """Unified error handling for all collectors"""
        pass
```

### Retry Logic
```python
# collectors/config/retry_manager.py
class RetryManager:
    def retry_with_backoff(self, func, max_retries=3):
        """Unified retry logic for all collectors"""
        pass
```

## 🎯 Best Practices

### 1. Collector Registration
```python
# collectors/__init__.py
COLLECTOR_REGISTRY = {
    'market': {
        'yf': YFCollector,
        'crypto': CryptoPriceCollector,
        'csv': CustomCSVCollector,
    },
    'news': {
        'rss': RSSCollector,
        'google_news': GoogleNewsCollector,
        'newsapi': NewsAPICollector,
    },
    'economic': {
        'fred': FREDCollector,
        'calendar': EconomicCalendarCollector,
    },
    'alternative': {
        'insider': InsiderCollector,
        'sec': SECFilingsCollector,
        'trends': GoogleTrendsCollector,
    }
}
```

### 2. Unified Interface
```python
# collectors/base/collector_interface.py
class CollectorInterface:
    def collect_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Standard interface for all collectors"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Standard data validation"""
        pass
```

### 3. Configuration Management
```python
# collectors/config/config_manager.py
class CollectorConfigManager:
    def get_config(self, collector_name: str) -> dict:
        """Get configuration for specific collector"""
        pass
    
    def update_config(self, collector_name: str, config: dict):
        """Update configuration for specific collector"""
        pass
```

## 📈 Performance Benefits

### 🎯 Organization
- **Clear categorization** by data type
- **Easy discovery** of collectors
- **Consistent interfaces** across collectors

### 🚀 Maintainability
- **Single source of truth** for each data type
- **Unified error handling** and retry logic
- **Centralized configuration** management

### 📊 Scalability
- **Easy to add new collectors**
- **Clear extension points**
- **Modular architecture**

---

**Status**: Collectors structure unified and organized
**Files Removed**: 2 duplicates
**Structure**: Organized by data type
**Next**: Final project structure validation
