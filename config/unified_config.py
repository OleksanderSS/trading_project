#!/usr/bin/env python3
"""
UNIFIED CONFIGURATION - ЄДИНИЙ ЦЕНТР КОНФІГУРАЦІЇ
"""

from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class UnifiedConfig:
    """
    Єдина конфігурація для всієї системи
    """
    
    # [TARGET] ОСНОВНІ ТАЙМФРЕЙМИ (оптимізовано для скальпінгу)
    TIMEFRAMES = {
        "5m": {
            "period": "60d",
            "interval": "5m",
            "description": "5 хвилин - ультра швидкість для скальпінгу",
            "rsi_threshold": (20, 80),
            "cache_ttl": 3,  # години
            "priority": "high"
        },
        "15m": {
            "period": "60d", 
            "interval": "15m",
            "description": "15 хвилин - основний таймфрейм",
            "rsi_threshold": (20, 80),
            "cache_ttl": 5,
            "priority": "high"
        },
        "60m": {
            "period": "60d",
            "interval": "60m", 
            "description": "1 година - підтвердження тренду",
            "rsi_threshold": (25, 75),
            "cache_ttl": 10,
            "priority": "medium"
        },
        "1d": {
            "period": "2y",
            "interval": "1d",
            "description": "1 день - ринковий контекст",
            "rsi_threshold": (30, 70),
            "cache_ttl": 90,
            "priority": "low"
        }
    }
    
    # 🗑️ LEGACY ТАЙМФРЕЙМИ (не використовуються)
    LEGACY_TIMEFRAMES = {
        "1m": {"period": "30d", "reason": "забагато шуму"},
        "30m": {"period": "60d", "reason": "не needed"},
        "4h": {"period": "120d", "reason": "занадто довго для скальпінгу"},
        "1w": {"period": "5y", "reason": "не для скальпінгу"}
    }
    
    # [DATA] ПРЕСЕТИ ТАЙМФРЕЙМІВ
    TIMEFRAME_PRESETS = {
        "default": ["5m", "15m", "60m", "1d"],
        "intraday": ["5m", "15m", "60m"],
        "daily": ["1d"],
        "scalping": ["5m", "15m"],
        "swing": ["60m", "1d"],
        "all": list(TIMEFRAMES.keys())
    }
    
    # [TARGET] ОСНОВНІ ТІКЕРИ
    TICKER_PRESETS = {
        "core": [
            # Tech Mega Cap (8)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK-B',
            # ETF (4)
            'SPY', 'QQQ', 'IWM', 'VTI'
        ],
        "tech": [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 
            'INTC', 'CSCO', 'ADBE', 'CRM', 'PYPL', 'NFLX', 'UBER'
        ],
        "etfs": [
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'IVV', 'GLD', 'SLV', 'TLT', 'HYG'
        ],
        "crypto": [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD'
        ],
        "all": [
            # Екстремальні волатильні (10)
            'TSLA', 'NVDA', 'AMD', 'COIN', 'MARA', 'RIOT', 'PLTR', 'GME', 'SNAP', 'ROKU',
            # Tech Mega Cap (8)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JPM', 'JNJ',
            # ETF (4)
            'SPY', 'QQQ', 'IWM', 'VTI'
        ]
    }
    
    # [UP] ПОРОГИ ТА ІНДИКАТОРИ
    INDICATORS = {
        "rsi": {
            "thresholds": TIMEFRAMES,  # Використовуємо таймфрейм-специфічні пороги
            "overbought": 70,
            "oversold": 30,
            "extreme_overbought": 80,
            "extreme_oversold": 20
        },
        "macd": {
            "fast": 12,
            "slow": 26,
            "signal": 9,
            "threshold": 0.01
        },
        "bollinger": {
            "period": 20,
            "std": 2,
            "threshold": 0.02
        },
        "volume": {
            "spike_threshold": 2.0,  # 2x середній обсяг
            "low_threshold": 0.5     # 0.5x середній обсяг
        }
    }
    
    # 📰 НАЛАШТУВАННЯ НОВИН
    NEWS_CONFIG = {
        "sources": [
            "newsapi", "rss", "google_news", "huggingface"
        ],
        "cache_ttl": 1,  # години
        "max_items": 1000,
        "sentiment_threshold": 0.2,
        "keywords": {
            "bullish": ["bullish", "buy", "up", "gain", "profit", "growth", "positive"],
            "bearish": ["bearish", "sell", "down", "loss", "decline", "negative", "risk"]
        }
    }
    
    # 🤖 НАЛАШТУВАННЯ МОДЕЛЕЙ
    MODEL_CONFIG = {
        "enabled_models": ["lgbm", "rf", "xgb"],
        "default_model": "lgbm",
        "ensemble_weights": {
            "lgbm": 0.4,
            "rf": 0.3,
            "xgb": 0.3
        },
        "validation": {
            "method": "time_series_split",
            "n_splits": 5,
            "test_size": 0.2
        }
    }
    
    # [SAVE] НАЛАШТУВАННЯ КЕШУ
    CACHE_CONFIG = {
        "enabled": True,
        "base_path": "cache",
        "ttl_hours": {
            "prices": 24,
            "news": 1,
            "indicators": 6,
            "models": 168  # 7 днів
        },
        "max_memory_gb": 4
    }
    
    # [RESTART] НАЛАШТУВАННЯ PIPELINE
    PIPELINE_CONFIG = {
        "stages": {
            "1": {"name": "Data Collection", "timeout": 300},
            "2": {"name": "Data Enrichment", "timeout": 600},
            "3": {"name": "Feature Engineering", "timeout": 300},
            "4": {"name": "Model Training", "timeout": 600},
            "5": {"name": "Analysis & Signals", "timeout": 300}
        },
        "retry_attempts": 3,
        "parallel_processing": True,
        "batch_size": 50
    }
    
    # [DATA] НАЛАШТУВАННЯ ТОРГІВЛІ
    TRADING_CONFIG = {
        "risk_management": {
            "max_position_size": 0.1,  # 10% портфеля
            "stop_loss": 0.02,         # 2%
            "take_profit": 0.04,       # 4%
            "max_drawdown": 0.15       # 15%
        },
        "signals": {
            "min_confidence": 0.7,
            "cooldown_minutes": 30,
            "max_signals_per_day": 10
        }
    }
    
    @classmethod
    def get_timeframes(cls, preset: str = "default") -> List[str]:
        """Отримати таймфрейми за пресетом"""
        return cls.TIMEFRAME_PRESETS.get(preset, cls.TIMEFRAME_PRESETS["default"])
    
    @classmethod
    def get_tickers(cls, preset: str = "core") -> List[str]:
        """Отримати тікери за пресетом"""
        return cls.TICKER_PRESETS.get(preset, cls.TICKER_PRESETS["core"])
    
    @classmethod
    def get_timeframe_config(cls, timeframe: str) -> Dict[str, Any]:
        """Отримати конфігурацію таймфрейму"""
        return cls.TIMEFRAMES.get(timeframe, {})
    
    @classmethod
    def get_rsi_threshold(cls, timeframe: str) -> Tuple[int, int]:
        """Отримати RSI пороги для таймфрейму"""
        tf_config = cls.TIMEFRAMES.get(timeframe, {})
        return tf_config.get("rsi_threshold", (30, 70))
    
    @classmethod
    def is_valid_timeframe(cls, timeframe: str) -> bool:
        """Перевірити чи таймфрейм валідний"""
        return timeframe in cls.TIMEFRAMES
    
    @classmethod
    def get_cache_ttl(cls, data_type: str, timeframe: str = None) -> int:
        """Отримати TTL для кешу"""
        if timeframe and timeframe in cls.TIMEFRAMES:
            return cls.TIMEFRAMES[timeframe].get("cache_ttl", 24)
        return cls.CACHE_CONFIG["ttl_hours"].get(data_type, 24)
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Валідація конфігурації"""
        issues = []
        
        # Перевіряємо таймфрейми
        for tf in cls.TIMEFRAMES:
            if not cls.TIMEFRAMES[tf].get("period"):
                issues.append(f"Missing period for timeframe {tf}")
            if not cls.TIMEFRAMES[tf].get("interval"):
                issues.append(f"Missing interval for timeframe {tf}")
        
        # Перевіряємо пресети
        for preset_name, tfs in cls.TIMEFRAME_PRESETS.items():
            for tf in tfs:
                if tf not in cls.TIMEFRAMES:
                    issues.append(f"Invalid timeframe {tf} in preset {preset_name}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "timeframes_count": len(cls.TIMEFRAMES),
            "tickers_count": len(cls.TICKER_PRESETS.get("all", []))
        }
    
    @classmethod
    def get_summary(cls) -> str:
        """Отримати підсумок конфігурації"""
        validation = cls.validate_config()
        
        summary = f"""
[TARGET] UNIFIED CONFIGURATION SUMMARY
{'='*50}

[OK] Timeframes: {len(cls.TIMEFRAMES)} active
[DATA] Presets: {len(cls.TIMEFRAME_PRESETS)}
[TARGET] Tickers: {len(cls.TICKER_PRESETS.get('all', []))} total
[UP] Indicators: {len(cls.INDICATORS)}
📰 News Sources: {len(cls.NEWS_CONFIG['sources'])}
🤖 Models: {len(cls.MODEL_CONFIG['enabled_models'])}

[SEARCH] Validation: {'[OK] PASSED' if validation['valid'] else '[ERROR] FAILED'}
{'[WARN] Issues: ' + str(len(validation['issues'])) if validation['issues'] else '[OK] No issues'}

[TARGET] Active Timeframes:
"""
        
        for tf, config in cls.TIMEFRAMES.items():
            priority = config.get('priority', 'unknown')
            summary += f"  {tf}: {config['description']} [{priority}]\n"
        
        return summary

# Глобальний екземпляр
config = UnifiedConfig()

# Для зворотної сумісності
TIME_FRAMES = config.TIMEFRAMES
LEGACY_TIME_FRAMES = config.LEGACY_TIMEFRAMES
YF_MAX_PERIODS = {tf: cfg["period"] for tf, cfg in config.TIMEFRAMES.items()}
DATA_INTERVALS = {tf: cfg["interval"] for tf, cfg in config.TIMEFRAMES.items()}

if __name__ == "__main__":
    print(config.get_summary())
    validation = config.validate_config()
    if not validation["valid"]:
        print("\n[ERROR] CONFIGURATION ISSUES:")
        for issue in validation["issues"]:
            print(f"  - {issue}")
