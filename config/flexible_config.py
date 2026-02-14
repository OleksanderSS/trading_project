#!/usr/bin/env python3
"""
FLEXIBLE CONFIGURATION - ГНУЧКА СИСТЕМА ВИБОРУ ТІКЕРІВ ТА ТАЙМФРЕЙМІВ
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TradingStyle(Enum):
    """Стилі торгівлі"""
    SCALPING = "scalping"
    DAY_TRADING = "day_trading"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"
    INVESTING = "investing"

class MarketFocus(Enum):
    """Фокус ринку"""
    TECH = "tech"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    ENERGY = "energy"
    CRYPTO = "crypto"
    COMMODITIES = "commodities"
    ALL = "all"

@dataclass
class TradingConfig:
    """Конфігурація торгівлі"""
    style: TradingStyle
    market_focus: MarketFocus
    timeframes: List[str]
    tickers: List[str]
    max_tickers: int = 25
    risk_level: str = "medium"  # low, medium, high, extreme

class FlexibleConfig:
    """Гнучка конфігурація системи"""
    
    # [TARGET] ОСНОВНІ ТАЙМФРЕЙМИ
    AVAILABLE_TIMEFRAMES = {
        "5m": {
            "description": "5 хвилин - ультра швидкість",
            "style": [TradingStyle.SCALPING],
            "data_points_per_day": 288,
            "cache_ttl": 3,
            "rsi_threshold": (20, 80),
            "recommended": True
        },
        "15m": {
            "description": "15 хвилин - швидка торгівля",
            "style": [TradingStyle.SCALPING, TradingStyle.DAY_TRADING],
            "data_points_per_day": 96,
            "cache_ttl": 5,
            "rsi_threshold": (20, 80),
            "recommended": True
        },
        "60m": {
            "description": "1 година - денна торгівля",
            "style": [TradingStyle.DAY_TRADING, TradingStyle.SWING_TRADING],
            "data_points_per_day": 24,
            "cache_ttl": 10,
            "rsi_threshold": (25, 75),
            "recommended": True
        },
        "1d": {
            "description": "1 день - довгострокова торгівля",
            "style": [TradingStyle.SWING_TRADING, TradingStyle.POSITION_TRADING, TradingStyle.INVESTING],
            "data_points_per_day": 1,
            "cache_ttl": 90,
            "rsi_threshold": (30, 70),
            "recommended": True
        }
    }
    
    # [TARGET] ТІКЕРИ ЗА РИНКАМИ
    MARKET_TICKERS = {
        MarketFocus.TECH: [
            # Mega Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            # Tech Growth
            'AMD', 'INTC', 'CSCO', 'ADBE', 'CRM', 'PYPL', 'NFLX', 'UBER',
            # Semiconductors
            'MU', 'LRCX', 'ASML', 'TSM', 'QCOM', 'MRVL',
            # Software
            'ORCL', 'SAP', 'INTU', 'NOW', 'TEAM'
        ],
        
        MarketFocus.FINANCE: [
            # Banks
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP',
            # Insurance
            'BRK-B', 'UNH', 'PGR', 'AIG', 'MET', 'PRU',
            # Financial Services
            'V', 'MA', 'PYPL', 'SQ', 'COIN', 'MSTR'
        ],
        
        MarketFocus.HEALTHCARE: [
            # Pharma
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT',
            # Biotech
            'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'MRNA',
            # Medical Devices
            'MDT', 'ISRG', 'SYK', 'BSX', 'ZBH'
        ],
        
        MarketFocus.ENERGY: [
            # Oil & Gas
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'BKR',
            # Renewable Energy
            'NEE', 'ENPH', 'SOL', 'FSLR', 'PLUG', 'BE', 'TSLA',
            # Energy Services
            'KMI', 'WMB', 'ET', 'EPD'
        ],
        
        MarketFocus.CRYPTO: [
            # Crypto Exchanges
            'COIN', 'MARA', 'RIOT', 'MSTR', 'SQ',
            # Blockchain
            'PYPL', 'V', 'MA'
        ],
        
        MarketFocus.COMMODITIES: [
            # Gold
            'GLD', 'GDX', 'GOLD', 'NEM', 'AEM',
            # Silver
            'SLV', 'PSLV', 'WPM', 'FRES',
            # Oil
            'USO', 'XLE', 'XOP', 'OIL'
        ]
    }
    
    # [TARGET] ETF ТА ІНДЕКСИ
    ETF_TICKERS = {
        "broad_market": ['SPY', 'QQQ', 'VTI', 'VOO', 'IVV'],
        "sector": ['XLF', 'XLK', 'XLV', 'XLE', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLRE'],
        "international": ['EFA', 'EEM', 'VWO', 'IEFA', 'VXUS'],
        "bond": ['AGG', 'BND', 'TLT', 'IEF', 'SHY', 'LQD'],
        "commodity": ['GLD', 'SLV', 'USO', 'DBA', 'DBB', 'DBC']
    }
    
    # [TARGET] ПРЕСЕТИ КОНФІГУРАЦІЙ
    PRESET_CONFIGS = {
        "scalping_aggressive": TradingConfig(
            style=TradingStyle.SCALPING,
            market_focus=MarketFocus.TECH,
            timeframes=["5m", "15m"],
            tickers=['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'META', 'AMZN'],
            max_tickers=8,
            risk_level="high"
        ),
        
        "day_trading_balanced": TradingConfig(
            style=TradingStyle.DAY_TRADING,
            market_focus=MarketFocus.ALL,
            timeframes=["5m", "15m", "60m"],
            tickers=['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'BAC', 'XOM'],
            max_tickers=10,
            risk_level="medium"
        ),
        
        "swing_trading_comprehensive": TradingConfig(
            style=TradingStyle.SWING_TRADING,
            market_focus=MarketFocus.ALL,
            timeframes=["15m", "60m", "1d"],
            tickers=['SPY', 'QQQ', 'VTI', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'XOM'],
            max_tickers=12,
            risk_level="medium"
        ),
        
        "position_trading_conservative": TradingConfig(
            style=TradingStyle.POSITION_TRADING,
            market_focus=MarketFocus.ALL,
            timeframes=["60m", "1d"],
            tickers=['SPY', 'QQQ', 'VTI', 'BND', 'GLD', 'AAPL', 'MSFT', 'JNJ', 'JPM', 'XOM'],
            max_tickers=10,
            risk_level="low"
        ),
        
        "crypto_focused": TradingConfig(
            style=TradingStyle.DAY_TRADING,
            market_focus=MarketFocus.CRYPTO,
            timeframes=["5m", "15m", "60m"],
            tickers=['COIN', 'MARA', 'RIOT', 'MSTR', 'SQ', 'PYPL', 'V', 'MA'],
            max_tickers=7,
            risk_level="extreme"
        )
    }
    
    @classmethod
    def get_timeframes_for_style(cls, style: TradingStyle) -> List[str]:
        """Отримати таймфрейми для стилю торгівлі"""
        suitable_tfs = []
        for tf, config in cls.AVAILABLE_TIMEFRAMES.items():
            if style in config["style"] and config["recommended"]:
                suitable_tfs.append(tf)
        return suitable_tfs
    
    @classmethod
    def get_tickers_for_market(cls, market_focus: MarketFocus, max_tickers: int = None) -> List[str]:
        """Отримати тікери для ринку"""
        if market_focus == MarketFocus.ALL:
            # Об'єднуємо всі ринки
            all_tickers = []
            for market in [MarketFocus.TECH, MarketFocus.FINANCE, MarketFocus.HEALTHCARE, MarketFocus.ENERGY]:
                all_tickers.extend(cls.MARKET_TICKERS.get(market, []))
            # Додаємо ETF
            all_tickers.extend(cls.ETF_TICKERS["broad_market"])
            
            # Унікальні тікери
            unique_tickers = list(set(all_tickers))
            return unique_tickers[:max_tickers] if max_tickers else unique_tickers
        else:
            tickers = cls.MARKET_TICKERS.get(market_focus, [])
            # Додаємо основні ETF
            tickers.extend(cls.ETF_TICKERS["broad_market"][:2])
            return tickers[:max_tickers] if max_tickers else tickers
    
    @classmethod
    def create_custom_config(cls, 
                           style: TradingStyle,
                           market_focus: MarketFocus,
                           custom_tickers: Optional[List[str]] = None,
                           custom_timeframes: Optional[List[str]] = None,
                           max_tickers: int = 25,
                           risk_level: str = "medium") -> TradingConfig:
        """Створити кастомну конфігурацію"""
        
        # Автоматично підбираємо таймфрейми
        if not custom_timeframes:
            timeframes = cls.get_timeframes_for_style(style)
        else:
            # Валідуємо таймфрейми
            valid_tfs = []
            for tf in custom_timeframes:
                if tf in cls.AVAILABLE_TIMEFRAMES:
                    valid_tfs.append(tf)
                else:
                    logger.warning(f"Invalid timeframe: {tf}")
            timeframes = valid_tfs
        
        # Автоматично підбираємо тікери
        if not custom_tickers:
            tickers = cls.get_tickers_for_market(market_focus, max_tickers)
        else:
            # Валідуємо тікери
            valid_tickers = []
            for ticker in custom_tickers:
                if isinstance(ticker, str) and ticker.strip():
                    valid_tickers.append(ticker.strip().upper())
            tickers = valid_tickers[:max_tickers]
        
        return TradingConfig(
            style=style,
            market_focus=market_focus,
            timeframes=timeframes,
            tickers=tickers,
            max_tickers=max_tickers,
            risk_level=risk_level
        )
    
    @classmethod
    def get_preset(cls, preset_name: str) -> TradingConfig:
        """Отримати пресет конфігурації"""
        if preset_name not in cls.PRESET_CONFIGS:
            raise ValueError(f"Unknown preset: {preset_name}")
        return cls.PRESET_CONFIGS[preset_name]
    
    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """Список доступних пресетів"""
        descriptions = {
            "scalping_aggressive": "Агресивний скальпінг на tech акціях",
            "day_trading_balanced": "Балансована денна торгівля",
            "swing_trading_comprehensive": "Комплексна свінг-торгівля",
            "position_trading_conservative": "Консервативна позиційна торгівля",
            "crypto_focused": "Фокус на криптовалютах"
        }
        return descriptions
    
    @classmethod
    def validate_config(cls, config: TradingConfig) -> Dict[str, Any]:
        """Валідація конфігурації"""
        issues = []
        
        # Перевіряємо таймфрейми
        for tf in config.timeframes:
            if tf not in cls.AVAILABLE_TIMEFRAMES:
                issues.append(f"Invalid timeframe: {tf}")
        
        # Перевіряємо відповідність таймфреймів стилю
        suitable_tfs = cls.get_timeframes_for_style(config.style)
        for tf in config.timeframes:
            if tf not in suitable_tfs:
                issues.append(f"Timeframe {tf} not suitable for {config.style.value}")
        
        # Перевіряємо кількість тікерів
        if len(config.tickers) > config.max_tickers:
            issues.append(f"Too many tickers: {len(config.tickers)} > {config.max_tickers}")
        
        # Перевіряємо ризик-менеджмент
        valid_risk_levels = ["low", "medium", "high", "extreme"]
        if config.risk_level not in valid_risk_levels:
            issues.append(f"Invalid risk level: {config.risk_level}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config_summary": {
                "style": config.style.value,
                "market_focus": config.market_focus.value,
                "timeframes": config.timeframes,
                "tickers_count": len(config.tickers),
                "max_tickers": config.max_tickers,
                "risk_level": config.risk_level
            }
        }
    
    @classmethod
    def get_config_summary(cls, config: TradingConfig) -> str:
        """Отримати підсумок конфігурації"""
        validation = cls.validate_config(config)
        
        summary = f"""
[TARGET] TRADING CONFIGURATION SUMMARY
{'='*50}

Style: {config.style.value.upper()}
Market Focus: {config.market_focus.value.upper()}
Risk Level: {config.risk_level.upper()}

Timeframes: {', '.join(config.timeframes)}
Tickers: {len(config.tickers)} (max: {config.max_tickers})

Timeframe Details:
"""
        
        for tf in config.timeframes:
            tf_config = cls.AVAILABLE_TIMEFRAMES[tf]
            summary += f"  {tf}: {tf_config['description']}\n"
        
        summary += f"\nTop Tickers: {', '.join(config.tickers[:5])}"
        if len(config.tickers) > 5:
            summary += f" ... and {len(config.tickers) - 5} more"
        
        summary += f"\n\nValidation: {'[OK] PASSED' if validation['valid'] else '[ERROR] FAILED'}"
        
        if validation['issues']:
            summary += "\nIssues:\n"
            for issue in validation['issues']:
                summary += f"  - {issue}"
        
        return summary

# Глобальні функції для зручності
def get_scalping_config(market_focus: MarketFocus = MarketFocus.TECH, 
                       tickers: Optional[List[str]] = None) -> TradingConfig:
    """Отримати конфігурацію для скальпінгу"""
    return FlexibleConfig.create_custom_config(
        style=TradingStyle.SCALPING,
        market_focus=market_focus,
        custom_tickers=tickers,
        risk_level="high"
    )

def get_day_trading_config(market_focus: MarketFocus = MarketFocus.ALL,
                          tickers: Optional[List[str]] = None) -> TradingConfig:
    """Отримати конфігурацію для денної торгівлі"""
    return FlexibleConfig.create_custom_config(
        style=TradingStyle.DAY_TRADING,
        market_focus=market_focus,
        custom_tickers=tickers,
        risk_level="medium"
    )

def get_swing_trading_config(market_focus: MarketFocus = MarketFocus.ALL,
                           tickers: Optional[List[str]] = None) -> TradingConfig:
    """Отримати конфігурацію для свінг-торгівлі"""
    return FlexibleConfig.create_custom_config(
        style=TradingStyle.SWING_TRADING,
        market_focus=market_focus,
        custom_tickers=tickers,
        risk_level="medium"
    )

if __name__ == "__main__":
    # Демонстрація
    print("=== FLEXIBLE CONFIGURATION DEMO ===")
    
    # Приклад 1: Агресивний скальпінг на tech
    config1 = get_scalping_config(MarketFocus.TECH)
    print(FlexibleConfig.get_config_summary(config1))
    
    # Приклад 2: Кастомна конфігурація
    config2 = FlexibleConfig.create_custom_config(
        style=TradingStyle.DAY_TRADING,
        market_focus=MarketFocus.ALL,
        custom_tickers=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
        custom_timeframes=['5m', '15m', '60m'],
        risk_level="high"
    )
    print(FlexibleConfig.get_config_summary(config2))
    
    # Приклад 3: Пресет
    config3 = FlexibleConfig.get_preset("day_trading_balanced")
    print(FlexibleConfig.get_config_summary(config3))
