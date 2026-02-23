"""
Collectors Package - Пакет withбирачandв data with єдиною архandтектурою
"""

# Існуючand колектори (основні версії)
from .base_collector import BaseCollector
from .google_news_collector import GoogleNewsCollector
from .newsapi_collector import NewsAPICollector
from .news_collector import NewsCollector
from .rss_collector import RSSCollector
from .fred_collector import FREDCollector
from .yf_collector import YFCollector
from .hf_collector import HFCollector
from .insider_collector import InsiderCollector
from .custom_csv_collector import CustomCSVCollector
from .free_google_trends_collector import FreeGoogleTrendsCollector

# НОВІ БЕЗКОШТОВНІ КОЛЕКТОРИ
from .economic_calendar_collector import EconomicCalendarCollector
from .crypto_price_collector import CryptoPriceCollector
from .sec_filings_collector import SECFilingsCollector

# Інтерфейси
from .collector_interface import (
    CollectorInterface, CollectorStatus, CollectorType, CollectionResult,
    CollectorError, APIError, ConfigurationError, DataValidationError
)

__all__ = [
    # Існуючі колектори (основні версії)
    'BaseCollector', 'GoogleNewsCollector', 'NewsAPICollector', 'NewsCollector',
    'RSSCollector', 'FREDCollector', 'YFCollector', 'HFCollector', 
    'InsiderCollector', 'CustomCSVCollector', 'FreeGoogleTrendsCollector',
    
    # 🆕 Нові withoutкоштовні колектори
    'EconomicCalendarCollector', 'CryptoPriceCollector', 'SECFilingsCollector',
    
    # Інтерфейси та компоненти архітектури
    'CollectorInterface', 'CollectorStatus', 'CollectorType', 'CollectionResult',
    'CollectorError', 'APIError', 'ConfigurationError', 'DataValidationError',
]

__version__ = "1.0.0"
__author__ = "Trading System Collectors Team"
