"""
Unified Collectors Configuration - Єдина конфandгурацandя for allх колекторandв
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    """Баwithова конфandгурацandя колектора"""
    name: str
    type: str
    enabled: bool = True
    max_retries: int = 3
    timeout: int = 30
    rate_limit: int = 100
    cache_enabled: bool = True
    cache_ttl: int = 3600
    batch_size: int = 100
    validate_data: bool = True
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThresholdConfig:
    """Конфandгурацandя порогandв"""
    base_threshold: int = 10
    regional_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "US": 5, "EU": 5, "CN": 5, "JP": 5,
        "UK": 7, "CA": 7, "AU": 7,
        "DE": 7, "FR": 7, "IT": 7, "ES": 7
    })
    event_type_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "041": 5, "042": 5, "043": 5,      # Economic cooperation
        "071": 7, "072": 7, "073": 7, "074": 7,  # Economic aid
        "081": 15, "082": 15, "083": 15, "084": 15,  # Economic sanctions
        "091": 12, "092": 12, "093": 12, "094": 12,  # Economic relations
        "05": 20  # Conflicts
    })
    time_decay_thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "recent_hours": 24,
        "recent_threshold": 5,
        "normal_threshold": 10
    })


@dataclass
class StorageConfig:
    """Конфandгурацandя сховища"""
    max_storage_gb: float = 1000.0
    data_retention_days: int = 365
    compression_enabled: bool = True
    backup_enabled: bool = True
    backup_retention_days: int = 30


class UnifiedCollectorsConfig:
    """
    Унandфandкована конфandгурацandя колекторandв
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Інandцandалandforцandя унandфandкованої конфandгурацandї
        
        Args:
            config_path: Шлях до fileу конфandгурацandї
        """
        self.config_path = config_path or "config/unified_collectors_config.yaml"
        self.collectors: Dict[str, CollectorConfig] = {}
        self.thresholds = ThresholdConfig()
        self.storage = StorageConfig()
        self.global_config: Dict[str, Any] = {}
        
        # Заванandження конфandгурацandї
        self.load_config()
    
    def load_config(self):
        """Заванandження конфandгурацandї with fileу"""
        try:
            config_file = Path(self.config_path)
            
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    if config_file.suffix.lower() == '.yaml':
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                self._parse_config(config_data)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                # Створення конфandгурацandї for forмовчуванням
                self._create_default_config()
                self.save_config()
                logger.info(f"Created default configuration at {self.config_path}")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_config()
    
    def _parse_config(self, config_data: Dict[str, Any]):
        """Парсинг конфandгурацandї"""
        # Глобальнand settings
        self.global_config = config_data.get("global", {})
        
        # Конфandгурацandя сховища
        storage_data = config_data.get("storage", {})
        self.storage = StorageConfig(**storage_data)
        
        # Конфandгурацandя порогandв
        thresholds_data = config_data.get("thresholds", {})
        self.thresholds = ThresholdConfig(**thresholds_data)
        
        # Конфandгурацandї колекторandв
        collectors_data = config_data.get("collectors", {})
        for name, collector_data in collectors_data.items():
            self.collectors[name] = CollectorConfig(**collector_data)
    
    def _create_default_config(self):
        """Створення конфandгурацandї for forмовчуванням"""
        # Глобальнand settings
        self.global_config = {
            "log_level": "INFO",
            "log_file": "collectors.log",
            "max_concurrent_collectors": 5,
            "default_timeout": 30,
            "error_log_file": "collectors_errors.log",
            "metrics_enabled": True,
            "cache_dir": "cache"
        }
        
        # Конфandгурацandї колекторandв
        self.collectors = {
            "google_news": CollectorConfig(
                name="Google News Collector",
                type="news",
                enabled=True,
                max_retries=3,
                timeout=30,
                rate_limit=100,
                cache_ttl=3600,
                batch_size=100,
                additional_params={
                    "similarity_threshold": 0.85,
                    "delay_range": [1.0, 3.0],
                    "days_back": 60
                }
            ),
            "newsapi": CollectorConfig(
                name="NewsAPI Collector",
                type="news",
                enabled=True,
                max_retries=3,
                timeout=30,
                rate_limit=1000,
                cache_ttl=1800,
                batch_size=100,
                additional_params={
                    "api_key_env": "NEWSAPI_KEY",
                    "sentiment_analysis": True,
                    "keyword_extraction": True,
                    "expand_to_tickers": True,
                    "max_articles_per_request": 100
                }
            ),
            "fred": CollectorConfig(
                name="FRED Collector",
                type="economic",
                enabled=True,
                max_retries=3,
                timeout=30,
                rate_limit=120,
                cache_ttl=86400,
                batch_size=100,
                additional_params={
                    "api_key_env": "FRED_API_KEY",
                    "default_frequency": "d",
                    "default_indicators": [
                        "GDP", "UNRATE", "CPIAUCSL", "PCEPI", "FEDFUNDS",
                        "DGS10", "DGS2", "DGS30", "DEXUSEU", "DEXJPUS",
                        "DEXUSUK", "DEXCHUS", "DEXCAUS", "DEXMXUS"
                    ],
                    "enable_new_indicators": True,
                    "macro_enrichment": True
                }
            ),
            "yahoo_finance": CollectorConfig(
                name="Yahoo Finance Collector",
                type="financial",
                enabled=True,
                max_retries=3,
                timeout=30,
                rate_limit=2000,
                cache_ttl=300,
                batch_size=100,
                additional_params={
                    "default_interval": "1d",
                    "default_period": "1y",
                    "default_tickers": [
                        "SPY", "QQQ", "IWM", "DIA",
                        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                        "JPM", "BAC", "WFC", "GS", "MS",
                        "XOM", "CVX", "COP", "BP",
                        "JNJ", "PFE", "UNH", "ABT",
                        "PG", "KO", "WMT", "HD",
                        "CAT", "DE", "GE", "MMM",
                        "GLD", "SLV", "USO", "TLT"
                    ],
                    "technical_indicators": True,
                    "sector_classification": True,
                    "market_cap_analysis": True
                }
            ),
            "rss": CollectorConfig(
                name="RSS Collector",
                type="news",
                enabled=True,
                max_retries=3,
                timeout=30,
                rate_limit=100,
                cache_ttl=3600,
                batch_size=100,
                additional_params={
                    "default_feeds": [
                        "https://feeds.reuters.com/reuters/topNews",
                        "https://feeds.bloomberg.com/markets/news.rss",
                        "https://feeds.feedburner.com/oreilly/radar",
                        "https://feeds.bbci.co.uk/news/rss.xml",
                        "https://rss.cnn.com/rss/edition.rss",
                        "https://feeds.finance.yahoo.com/rss/2.0/headline",
                        "https://feeds.marketwatch.com/marketwatch/topstories",
                        "https://feeds.seekingalpha.com/feed",
                        "https://feeds.feedburner.com/TheStreet",
                        "https://feeds.macrumors.com/articles",
                        "https://feeds.techcrunch.com/TechCrunch/",
                        "https://feeds.wired.com/wired/index",
                        "https://feeds.arstechnica.com/arstechnica/index",
                        "https://feeds.feedburner.com/Forbes/technology",
                        "https://feeds.feedburner.com/HarvardBusiness",
                        "https://feeds.feedburner.com/entrepreneur/latest",
                        "https://feeds.finviz.com/rss.ashx",
                        "https://feeds.feedburner.com/Moneycontrol/marketnews"
                    ],
                    "similarity_threshold": 0.85,
                    "delay_range": [1.0, 3.0],
                    "max_articles_per_feed": 100,
                    "sentiment_analysis": True,
                    "source_classification": True
                }
            ),
            "gdelt": CollectorConfig(
                name="GDELT Collector",
                type="events",
                enabled=True,
                max_retries=3,
                timeout=60,
                rate_limit=1000,
                cache_ttl=7200,
                batch_size=1000,
                additional_params={
                    "default_languages": ["en"],
                    "default_topics": ["Economy", "Finance", "Trade"],
                    "bigquery_project_env": "GDELT_PROJECT_ID",
                    "bigquery_dataset": "gdelt-bq.gdeltv2",
                    "enable_event_filtering": True,
                    "use_json_api": True,
                    "json_api_key_env": "GDELT_API_KEY",
                    "use_bigquery": True,
                    "service_account_path": "config/google_key.json",
                    "max_events_per_request": 500,
                    "focus_indicators": [
                        "GLOBALEVENTID", "EVENTROOTCODE", "EVENTCODE", 
                        "ACTIONGEOCODE", "NUMARTICLES"
                    ],
                    "economical_mode": True,
                    "threshold_strategy": "tiered",
                    "min_mentions_threshold": 3,
                    "economic_event_codes": [
                        "04", "05", "071", "072", "073", "074", 
                        "081", "082", "083", "084", "091", "092", "093", "094"
                    ]
                }
            ),
            "hf": CollectorConfig(
                name="HuggingFace Collector",
                type="ai",
                enabled=True,
                max_retries=3,
                timeout=60,
                rate_limit=100,
                cache_ttl=3600,
                batch_size=50,
                additional_params={
                    "api_key_env": "HF_API_KEY",
                    "default_models": ["distilbert-base-uncased", "bert-base-uncased"],
                    "enable_text_classification": True,
                    "use_free_models": True,
                    "max_requests_per_hour": 100,
                    "focus_tasks": [
                        "sentiment-analysis", "text-classification", "feature-extraction"
                    ],
                    "fallback_to_local": True
                }
            ),
            "insider": CollectorConfig(
                name="Insider Trading Collector",
                type="financial",
                enabled=False,
                max_retries=3,
                timeout=30,
                rate_limit=100,
                cache_ttl=86400,
                batch_size=100,
                additional_params={
                    "api_key_env": "INSIDER_API_KEY",
                    "default_filings": ["4", "3", "5"],
                    "enable_sentiment_analysis": True
                }
            ),
            "telegram": CollectorConfig(
                name="Telegram Collector",
                type="social",
                enabled=False,
                max_retries=3,
                timeout=30,
                rate_limit=30,
                cache_ttl=1800,
                batch_size=100,
                additional_params={
                    "api_id_env": "TELEGRAM_API_ID",
                    "api_hash_env": "TELEGRAM_API_HASH",
                    "default_channels": [],
                    "enable_message_filtering": True
                }
            ),
            "aaii": CollectorConfig(
                name="AAII Collector",
                type="sentiment",
                enabled=False,
                max_retries=3,
                timeout=30,
                rate_limit=100,
                cache_ttl=86400,
                batch_size=100,
                additional_params={
                    "api_key_env": "AAII_API_KEY",
                    "default_surveys": ["investor_sentiment", "bull_bear"],
                    "enable_historical_data": True
                }
            ),
            "custom_csv": CollectorConfig(
                name="Custom CSV Collector",
                type="data",
                enabled=False,
                max_retries=1,
                timeout=30,
                rate_limit=10,
                cache_enabled=False,
                cache_ttl=0,
                batch_size=1000,
                additional_params={
                    "default_file_path": "data/custom/custom_data.csv",
                    "default_delimiter": ",",
                    "enable_data_validation": True,
                    "required_columns": ["date", "value"]
                }
            )
        }
    
    def save_config(self):
        """Збереження конфandгурацandї у file"""
        try:
            config_data = {
                "global": self.global_config,
                "storage": {
                    "max_storage_gb": self.storage.max_storage_gb,
                    "data_retention_days": self.storage.data_retention_days,
                    "compression_enabled": self.storage.compression_enabled,
                    "backup_enabled": self.storage.backup_enabled,
                    "backup_retention_days": self.storage.backup_retention_days
                },
                "thresholds": {
                    "base_threshold": self.thresholds.base_threshold,
                    "regional_thresholds": self.thresholds.regional_thresholds,
                    "event_type_thresholds": self.thresholds.event_type_thresholds,
                    "time_decay_thresholds": self.thresholds.time_decay_thresholds
                },
                "collectors": {}
            }
            
            # Конверandцandя колекторandв
            for name, collector in self.collectors.items():
                config_data["collectors"][name] = {
                    "name": collector.name,
                    "type": collector.type,
                    "enabled": collector.enabled,
                    "max_retries": collector.max_retries,
                    "timeout": collector.timeout,
                    "rate_limit": collector.rate_limit,
                    "cache_enabled": collector.cache_enabled,
                    "cache_ttl": collector.cache_ttl,
                    "batch_size": collector.batch_size,
                    "validate_data": collector.validate_data,
                    "additional_params": collector.additional_params
                }
            
            # Збереження у file
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                if config_file.suffix.lower() == '.yaml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_collector_config(self, name: str) -> Optional[CollectorConfig]:
        """
        Отримання конфandгурацandї колектора
        
        Args:
            name: Наwithва колектора
            
        Returns:
            Optional[CollectorConfig]: Конфandгурацandя колектора
        """
        return self.collectors.get(name)
    
    def get_enabled_collectors(self) -> List[str]:
        """
        Отримання списку увandмкnotних колекторandв
        
        Returns:
            List[str]: Список наwithв увandмкnotних колекторandв
        """
        return [name for name, config in self.collectors.items() if config.enabled]
    
    def get_collectors_by_type(self, collector_type: str) -> List[str]:
        """
        Отримання колекторandв for типом
        
        Args:
            collector_type: Тип колектора
            
        Returns:
            List[str]: Список наwithв колекторandв
        """
        return [
            name for name, config in self.collectors.items() 
            if config.type == collector_type and config.enabled
        ]
    
    def update_collector_config(self, name: str, **kwargs):
        """
        Оновлення конфandгурацandї колектора
        
        Args:
            name: Наwithва колектора
            **kwargs: Параметри for оновлення
        """
        if name in self.collectors:
            collector = self.collectors[name]
            for key, value in kwargs.items():
                if hasattr(collector, key):
                    setattr(collector, key, value)
                elif key == "additional_params":
                    collector.additional_params.update(value)
            
            logger.info(f"Updated configuration for collector: {name}")
        else:
            logger.warning(f"Collector not found: {name}")
    
    def enable_collector(self, name: str):
        """Увandмкnotння колектора"""
        self.update_collector_config(name, enabled=True)
    
    def disable_collector(self, name: str):
        """Вимкnotння колектора"""
        self.update_collector_config(name, enabled=False)
    
    def validate_config(self) -> List[str]:
        """
        Валandдацandя конфandгурацandї
        
        Returns:
            List[str]: Список errors валandдацandї
        """
        errors = []
        
        # Валandдацandя глобальних налаштувань
        if not self.global_config:
            errors.append("Global configuration is empty")
        
        # Валandдацandя колекторandв
        for name, collector in self.collectors.items():
            if not collector.name:
                errors.append(f"Collector {name}: name is empty")
            
            if not collector.type:
                errors.append(f"Collector {name}: type is empty")
            
            if collector.timeout <= 0:
                errors.append(f"Collector {name}: timeout must be positive")
            
            if collector.rate_limit <= 0:
                errors.append(f"Collector {name}: rate_limit must be positive")
            
            if collector.batch_size <= 0:
                errors.append(f"Collector {name}: batch_size must be positive")
        
        # Валandдацandя порогandв
        if self.thresholds.base_threshold <= 0:
            errors.append("Base threshold must be positive")
        
        # Валandдацandя сховища
        if self.storage.max_storage_gb <= 0:
            errors.append("Max storage GB must be positive")
        
        if self.storage.data_retention_days <= 0:
            errors.append("Data retention days must be positive")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Отримання пandдсумку конфandгурацandї
        
        Returns:
            Dict[str, Any]: Пandдсумок конфandгурацandї
        """
        enabled_collectors = self.get_enabled_collectors()
        collectors_by_type = {}
        
        for collector_type in ["news", "economic", "financial", "events", "ai", "social", "sentiment", "data"]:
            collectors_by_type[collector_type] = self.get_collectors_by_type(collector_type)
        
        return {
            "total_collectors": len(self.collectors),
            "enabled_collectors": len(enabled_collectors),
            "collectors_by_type": collectors_by_type,
            "storage_config": {
                "max_storage_gb": self.storage.max_storage_gb,
                "data_retention_days": self.storage.data_retention_days,
                "compression_enabled": self.storage.compression_enabled
            },
            "threshold_config": {
                "base_threshold": self.thresholds.base_threshold,
                "regional_thresholds_count": len(self.thresholds.regional_thresholds),
                "event_type_thresholds_count": len(self.thresholds.event_type_thresholds)
            },
            "validation_errors": self.validate_config()
        }


# Глобальний екwithемпляр конфandгурацandї
_unified_config = None


def get_unified_config() -> UnifiedCollectorsConfig:
    """
    Отримання глобального екwithемпляра унandфandкованої конфandгурацandї
    
    Returns:
        UnifiedCollectorsConfig: Екwithемпляр конфandгурацandї
    """
    global _unified_config
    if _unified_config is None:
        _unified_config = UnifiedCollectorsConfig()
    return _unified_config


def reload_config():
    """Переforванandження конфandгурацandї"""
    global _unified_config
    _unified_config = UnifiedCollectorsConfig()
    logger.info("Configuration reloaded")


# Зручнand функцandї for доступу
def get_collector_config(name: str) -> Optional[CollectorConfig]:
    """Отримання конфandгурацandї колектора"""
    return get_unified_config().get_collector_config(name)


def get_enabled_collectors() -> List[str]:
    """Отримання списку увandмкnotних колекторandв"""
    return get_unified_config().get_enabled_collectors()


def get_collectors_by_type(collector_type: str) -> List[str]:
    """Отримання колекторandв for типом"""
    return get_unified_config().get_collectors_by_type(collector_type)
