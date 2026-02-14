#!/usr/bin/env python3
"""
Модуль конфігурації trading системи
Розбитий на логічні компоненти для кращої підтримки
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Any
from pathlib import Path
from enum import Enum
import os


class LogLevel(str, Enum):
    """Рівні логування"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Timeframe(str, Enum):
    """Доступні таймфрейми"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "60m"
    DAY_1 = "1d"


@dataclass
class DataConfig:
    """Конфігурація data"""
    # Тікери
    tickers: List[str] = field(default_factory=lambda: [
        # Екстремальні волатильні (10)
        'TSLA', 'NVDA', 'AMD', 'COIN', 'MARA', 'RIOT', 'PLTR', 'GME', 'SNAP', 'ROKU',
        # Tech Mega Cap (8)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'CRM', 'ADBE',
        # Напівпровідники (6)
        'INTC', 'MU', 'SOXX', 'SMH', 'TSM', 'ASML',
        # Крипто-пов'язані (5)
        'SQ', 'PYPL', 'MSTR', 'BKNG', 'EBAY',
        # Енергетика (6)
        'XOM', 'CVX', 'CLF', 'HAL', 'SLB', 'COP',
        # Фінанси (5)
        'JPM', 'BAC', 'WFC', 'GS', 'V',
        # ETF (10)
        'QQQ', 'SPY', 'IWM', 'GLD', 'TLT', 'XLE', 'XLK', 'ARKK', 'SOXX', 'SMH'
    ])
    
    # Таймфрейми
    timeframes: List[Timeframe] = field(default_factory=lambda: [
        Timeframe.MINUTE_5, Timeframe.MINUTE_15, Timeframe.HOUR_1, 
        Timeframe.HOUR_4, Timeframe.DAY_1, Timeframe.WEEK_1
    ])
    
    # Налаштування data
    data_retention_days: int = 365
    cache_ttl_minutes: int = 15
    max_memory_usage_gb: float = 4.0
    
    # Шляхи
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    
    def get_tickers_by_category(self) -> Dict[str, List[str]]:
        """Розподіл тікерів по категоріях"""
        return {
            'extreme_volatility': ['TSLA', 'NVDA', 'AMD', 'COIN', 'MARA', 'RIOT', 'PLTR', 'GME', 'SNAP', 'ROKU'],
            'tech_mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'CRM', 'ADBE'],
            'semiconductors': ['INTC', 'MU', 'SOXX', 'SMH', 'TSM', 'ASML', 'NVDA', 'AMD'],
            'crypto_related': ['SQ', 'PYPL', 'MSTR', 'BKNG', 'EBAY', 'COIN', 'MARA', 'RIOT'],
            'energy': ['XOM', 'CVX', 'CLF', 'HAL', 'SLB', 'COP'],
            'finance': ['JPM', 'BAC', 'WFC', 'GS', 'V'],
            'etf_broad': ['QQQ', 'SPY', 'IWM', 'GLD', 'TLT'],
            'etf_sector': ['XLE', 'XLK', 'ARKK', 'SOXX', 'SMH']
        }
    
    def validate(self) -> List[str]:
        """Валідація конфігурації data"""
        errors = []
        
        if not self.tickers:
            errors.append("Tickers list cannot be empty")
        
        if not self.timeframes:
            errors.append("Timeframes list cannot be empty")
        
        if self.data_retention_days <= 0:
            errors.append("Data retention days must be positive")
        
        if self.cache_ttl_minutes <= 0:
            errors.append("Cache TTL must be positive")
        
        if self.max_memory_usage_gb <= 0:
            errors.append("Max memory usage must be positive")
        
        # Перевірка форматів тікерів
        for ticker in self.tickers:
            if not ticker.isalpha() or len(ticker) > 5:
                errors.append(f"Invalid ticker format: {ticker}")
        
        return errors


@dataclass
class RiskConfig:
    """Конфігурація ризиків"""
    # Позиції
    max_positions: int = 10
    risk_per_trade: float = 0.02  # 2% ризик на торгівлю
    max_portfolio_risk: float = 0.2  # 20% максимальний ризик портфеля
    
    # Капітал
    initial_capital: float = 100000.0
    
    # Кореляція
    correlation_limit: float = 0.7
    
    # Stop-loss та take-profit
    stop_loss_pct: float = 0.05  # 5% stop-loss
    take_profit_pct: float = 0.10  # 10% take-profit
    
    # Розмір позиції
    default_position_size: float = 0.1  # 10% від капіталу
    max_position_size: float = 0.25  # 25% максимум
    
    # Trading
    enable_real_trading: bool = False
    trading_hours: int = 4
    
    def validate(self) -> List[str]:
        """Валідація конфігурації ризиків"""
        errors = []
        
        if self.max_positions <= 0:
            errors.append("Max positions must be positive")
        
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            errors.append("Risk per trade should be between 0 and 0.1 (10%)")
        
        if self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1.0:
            errors.append("Max portfolio risk should be between 0 and 1")
        
        if self.correlation_limit <= 0 or self.correlation_limit > 1.0:
            errors.append("Correlation limit should be between 0 and 1")
        
        if self.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 1.0:
            errors.append("Stop loss percentage should be between 0 and 1")
        
        if self.take_profit_pct <= 0 or self.take_profit_pct > 1.0:
            errors.append("Take profit percentage should be between 0 and 1")
        
        if self.default_position_size <= 0 or self.default_position_size > 1.0:
            errors.append("Default position size should be between 0 and 1")
        
        if self.max_position_size <= 0 or self.max_position_size > 1.0:
            errors.append("Max position size should be between 0 and 1")
        
        if self.default_position_size > self.max_position_size:
            errors.append("Default position size cannot be greater than max position size")
        
        return errors


@dataclass
class ModelConfig:
    """Конфігурація моделей"""
    # Типи моделей
    model_types: List[str] = field(default_factory=lambda: ['lgbm', 'xgboost', 'rf'])
    
    # Enhanced Models
    enable_enhanced_models: bool = True
    enable_colab_integration: bool = True
    enable_dean_models: bool = True
    enable_sentiment_models: bool = True
    
    # Параметри навчання
    feature_importance_threshold: float = 0.01
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Параметри оптимізації
    max_trials: int = 100
    timeout_seconds: int = 300
    
    # Шляхи
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "models")
    
    def validate(self) -> List[str]:
        """Валідація конфігурації моделей"""
        errors = []
        
        if not self.model_types:
            errors.append("Model types list cannot be empty")
        
        if self.feature_importance_threshold < 0 or self.feature_importance_threshold > 1.0:
            errors.append("Feature importance threshold should be between 0 and 1")
        
        if self.cross_validation_folds <= 1:
            errors.append("Cross validation folds must be greater than 1")
        
        if self.test_size <= 0 or self.test_size >= 1.0:
            errors.append("Test size should be between 0 and 1")
        
        if self.max_trials <= 0:
            errors.append("Max trials must be positive")
        
        if self.timeout_seconds <= 0:
            errors.append("Timeout seconds must be positive")
        
        return errors


@dataclass
class NewsConfig:
    """Конфігурація новин"""
    # Джерела новин
    news_sources_limit: int = 100
    sentiment_threshold: float = 0.1
    news_update_interval_minutes: int = 30
    
    # Фільтрація
    min_relevance_score: float = 0.5
    max_news_age_hours: int = 24
    
    # Обробка
    enable_sentiment_analysis: bool = True
    enable_keyword_extraction: bool = True
    
    def validate(self) -> List[str]:
        """Валідація конфігурації новин"""
        errors = []
        
        if self.news_sources_limit <= 0:
            errors.append("News sources limit must be positive")
        
        if self.sentiment_threshold < 0 or self.sentiment_threshold > 1.0:
            errors.append("Sentiment threshold should be between 0 and 1")
        
        if self.news_update_interval_minutes <= 0:
            errors.append("News update interval must be positive")
        
        if self.min_relevance_score < 0 or self.min_relevance_score > 1.0:
            errors.append("Min relevance score should be between 0 and 1")
        
        if self.max_news_age_hours <= 0:
            errors.append("Max news age hours must be positive")
        
        return errors


@dataclass
class LoggingConfig:
    """Конфігурація логування"""
    log_level: LogLevel = LogLevel.INFO
    log_file: str = "trading_system.log"
    max_log_size_mb: int = 100
    log_backup_count: int = 5
    
    # Шлях до логів
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")
    
    # Форматування
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    def validate(self) -> List[str]:
        """Валідація конфігурації логування"""
        errors = []
        
        if self.max_log_size_mb <= 0:
            errors.append("Max log size must be positive")
        
        if self.log_backup_count < 0:
            errors.append("Log backup count cannot be negative")
        
        return errors


@dataclass
class APIConfig:
    """Конфігурація API"""
    # Ліміти
    rate_limit_requests_per_minute: int = 60
    retry_attempts: int = 3
    timeout_seconds: int = 30
    
    # API keys (з environment variables)
    alpha_vantage_key: Optional[str] = field(default_factory=lambda: os.getenv('ALPHA_VANTAGE_KEY'))
    news_api_key: Optional[str] = field(default_factory=lambda: os.getenv('NEWS_API_KEY'))
    
    def validate(self) -> List[str]:
        """Валідація конфігурації API"""
        errors = []
        
        if self.rate_limit_requests_per_minute <= 0:
            errors.append("Rate limit must be positive")
        
        if self.retry_attempts < 0:
            errors.append("Retry attempts cannot be negative")
        
        if self.timeout_seconds <= 0:
            errors.append("Timeout seconds must be positive")
        
        return errors


@dataclass
class BacktestConfig:
    """Конфігурація бектестингу"""
    # Период
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    
    # Параметри
    commission: float = 0.001  # 0.1% комісія
    slippage: float = 0.0001  # 0.01% slippage
    
    # Результати
    save_results: bool = True
    generate_plots: bool = True
    
    # Шлях до результатів
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data" / "results")
    
    def validate(self) -> List[str]:
        """Валідація конфігурації бектестингу"""
        errors = []
        
        # Перевірка формату дати
        try:
            from datetime import datetime
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError:
            errors.append("Date format should be YYYY-MM-DD")
        
        if self.commission < 0 or self.commission > 0.1:
            errors.append("Commission should be between 0 and 0.1 (10%)")
        
        if self.slippage < 0 or self.slippage > 0.01:
            errors.append("Slippage should be between 0 and 0.01 (1%)")
        
        return errors


@dataclass
class TradingConfig:
    """Основна конфігурація trading системи"""
    # Компоненти конфігурації
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    def __post_init__(self):
        """Ініціалізація після створення"""
        # Створюємо директорії
        for dir_path in [
            self.data.data_dir,
            self.models.models_dir,
            self.logging.logs_dir,
            self.backtest.results_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'TradingConfig':
        """Створення конфігурації з environment variables"""
        return cls(
            data=DataConfig(
                tickers=os.getenv('TRADING_TICKERS', '').split(',') if os.getenv('TRADING_TICKERS') else DataConfig().tickers,
                data_retention_days=int(os.getenv('DATA_RETENTION_DAYS', str(DataConfig().data_retention_days))),
                cache_ttl_minutes=int(os.getenv('CACHE_TTL_MINUTES', str(DataConfig().cache_ttl_minutes))),
            ),
            risk=RiskConfig(
                max_positions=int(os.getenv('MAX_POSITIONS', str(RiskConfig().max_positions))),
                risk_per_trade=float(os.getenv('RISK_PER_TRADE', str(RiskConfig().risk_per_trade))),
                initial_capital=float(os.getenv('INITIAL_CAPITAL', str(RiskConfig().initial_capital))),
            ),
            logging=LoggingConfig(
                log_level=LogLevel(os.getenv('LOG_LEVEL', LoggingConfig().log_level.value)),
            )
        )
    
    def validate(self) -> bool:
        """Валідація всієї конфігурації"""
        all_errors = []
        
        # Валідація кожного компонента
        all_errors.extend(self.data.validate())
        all_errors.extend(self.risk.validate())
        all_errors.extend(self.models.validate())
        all_errors.extend(self.news.validate())
        all_errors.extend(self.logging.validate())
        all_errors.extend(self.api.validate())
        all_errors.extend(self.backtest.validate())
        
        if all_errors:
            raise ValueError(f"Configuration errors: {all_errors}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник для серіалізації"""
        return {
            'data': self.data.__dict__,
            'risk': self.risk.__dict__,
            'models': self.models.__dict__,
            'news': self.news.__dict__,
            'logging': self.logging.__dict__,
            'api': self.api.__dict__,
            'backtest': self.backtest.__dict__
        }


# Глобальна конфігурація
_config = None

def get_config() -> TradingConfig:
    """Отримати глобальну конфігурацію"""
    global _config
    if _config is None:
        _config = TradingConfig.from_env()
        _config.validate()
    return _config

def reload_config() -> TradingConfig:
    """Перезавантажити конфігурацію"""
    global _config
    _config = TradingConfig.from_env()
    _config.validate()
    return _config
