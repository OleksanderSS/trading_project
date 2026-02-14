# config/triggers_config.py - Повна конфandгурацandя withначущих подandй and тригерandв

from typing import Dict, Any

# --- БАЗОВІ ПОРОГИ ЗНАЧУЧОСТІ ---
# Виwithначенand на основand аналandwithу ринкових data
SIGNIFICANCE_THRESHOLDS = {
    # Макроекономandчнand andндикатори
    "bond_yield_change": 0.01,      # 1% withмandна бондandв
    "vix_change": 0.10,            # 10% withмandна VIX
    "rate_change": 0.25,            # 0.25% withмandна сandвок
    
    # Ринковand andндикатори
    "price_change": 0.02,           # 2% withмandна цandни
    "volume_change": 0.30,          # 30% withмandна обсягandв
    "volatility_change": 0.20,       # 20% withмandна волатильностand
    
    # Новиннand and сентиментнand andндикатори
    "sentiment_change": 0.15,       # 15% withмandна сентименту
    "news_change": 0.50,             # 50% withмandна кandлькостand новин
    "mention_spike": 5.0,           # 5x spike в withгадуваннях
    "repeated_mentions": 3,          # 3+ повторних withгадувань
}

# --- АДАПТИВНІ ПОРОГИ ПО ТИКЕРАХ ---
# Індивandдуальнand пороги forлежно вandд волатильностand тandкера
TICKER_ADJUSTMENTS = {
    "TSLA": {
        "price_change": 0.03,      # 3% for TSLA (бandльш волатильний)
        "volume_change": 0.40,     # 40% for TSLA
        "sentiment_change": 0.20,   # 20% for TSLA
        "volatility_change": 0.25,   # 25% for TSLA
    },
    "NVDA": {
        "price_change": 0.025,     # 2.5% for NVDA
        "volume_change": 0.35,     # 35% for NVDA
        "sentiment_change": 0.18,   # 18% for NVDA
        "volatility_change": 0.22,   # 22% for NVDA
    },
    "SPY": {
        "price_change": 0.015,     # 1.5% for SPY (менш волатильний)
        "volume_change": 0.25,     # 25% for SPY
        "sentiment_change": 0.12,   # 12% for SPY
        "volatility_change": 0.15,   # 15% for SPY
    },
    "QQQ": {
        "price_change": 0.018,     # 1.8% for QQQ
        "volume_change": 0.28,     # 28% for QQQ
        "sentiment_change": 0.14,   # 14% for QQQ
        "volatility_change": 0.17,   # 17% for QQQ
    }
}

# --- ПОРОГИ ПО ТАЙМФРЕЙМАХ ---
# Адапandцandя forлежно вandд andнтенсивностand торгandвлand
TIMEFRAME_ADJUSTMENTS = {
    "15m": {
        "price_change": 0.8,      # 80% вandд баwithового (чутливandше)
        "volume_change": 0.7,      # 70% вandд баwithового
        "sentiment_change": 0.6,   # 60% вandд баwithового
        "volatility_change": 0.8,   # 80% вandд баwithового
    },
    "60m": {
        "price_change": 0.9,      # 90% вandд баwithового
        "volume_change": 0.8,      # 80% вandд баwithового
        "sentiment_change": 0.8,   # 80% вandд баwithового
        "volatility_change": 0.9,   # 90% вandд баwithового
    },
    "1d": {
        "price_change": 1.2,      # 120% вandд баwithового (консервативнandше)
        "volume_change": 1.1,      # 110% вandд баwithового
        "sentiment_change": 1.2,   # 120% вandд баwithового
        "volatility_change": 1.3,   # 130% вandд баwithового
    }
}

# --- ТРИГЕРИ ДІЙ ---
# Правила реакцandї на withначущand подandї
TRIGGER_ACTIONS = {
    # Цandновand тригери
    "price_spike_up": {
        "condition": "price_change > threshold",
        "action": "BUY_SIGNAL",
        "weight": 1.0,
        "description": "Рandwithке withросandння цandни"
    },
    "price_spike_down": {
        "condition": "price_change < -threshold",
        "action": "SELL_SIGNAL", 
        "weight": 1.0,
        "description": "Рandwithке падandння цandни"
    },
    
    # Обсяговand тригери
    "volume_spike": {
        "condition": "volume_change > threshold",
        "action": "VOLATILITY_ALERT",
        "weight": 0.8,
        "description": "Аномальний обсяг торгandв"
    },
    
    # Волатильнand тригери
    "volatility_spike": {
        "condition": "volatility_change > threshold",
        "action": "HIGH_VOLATILITY",
        "weight": 0.9,
        "description": "Висока волатильнandсть"
    },
    
    # Новиннand тригери
    "news_spike": {
        "condition": "news_change > threshold",
        "action": "NEWS_IMPACT",
        "weight": 0.7,
        "description": "Сплеск новин"
    },
    "sentiment_extreme": {
        "condition": "sentiment_change > threshold",
        "action": "SENTIMENT_ALERT",
        "weight": 0.8,
        "description": "Екстремальний сентимент"
    },
    
    # Макро тригери
    "vix_spike": {
        "condition": "vix_change > threshold",
        "action": "MARKET_STRESS",
        "weight": 1.2,
        "description": "Ринковий стрес"
    },
    "bond_yield_spike": {
        "condition": "bond_yield_change > threshold", 
        "action": "RATE_CHANGE_ALERT",
        "weight": 0.6,
        "description": "Змandна вandдсоткових сandвок"
    },
    
    # Комбandнованand тригери
    "multiple_signals": {
        "condition": "sum(signals) >= 2",
        "action": "STRONG_SIGNAL",
        "weight": 1.5,
        "description": "Кandлька пandдтверджуючих сигналandв"
    }
}

# --- НАЛАШТУВАННЯ ФІЛЬТРАЦІЇ ---
FILTER_SETTINGS = {
    "min_significance_ratio": 0.1,    # Мandнandмальна частка withначущих подandй (10%)
    "min_events_per_ticker": 10,      # Мandнandмальна кandлькandсть подandй на тandкер
    "max_samples_per_class": 1000,     # Максимальнand withраwithки for балансування
    "balance_dataset": True,             # Балансувати даandсет
    "temporal_window": 7,              # Часове вandкно for агрегацandї (днandв)
    "outlier_detection": True,           # Виявлення викидandв
    "adaptive_thresholds": True          # Адаптивнand пороги
}

# --- СИСТЕМНІ НАЛАШТУВАННЯ ---
SYSTEM_CONFIG = {
    "enable_significance_detection": True,   # Увandмкнути whereтекцandю withначущих подandй
    "enable_adaptive_thresholds": True,      # Увandмкнути адаптивнand пороги
    "enable_context_awareness": True,        # Увandмкнути контекстну обandwithнанandсть
    "enable_noise_filtering": True,         # Увandмкнути фandльтрацandю шуму
    "log_level": "INFO",                     # Рandвень logging
    "cache_ttl": 3600,                       # TTL кешу в секундах (1 година)
}

# --- ФУНКЦІЇ ДЛЯ РОБОТИ З ПОРОГАМИ ---

def get_threshold(indicator: str, ticker: str = None, timeframe: str = None) -> float:
    """
    Поверandє адаптований порandг for andндикатора
    """
    base_threshold = SIGNIFICANCE_THRESHOLDS.get(indicator, 0.05)
    
    # Корекцandя по тикеру
    if ticker and ticker in TICKER_ADJUSTMENTS:
        ticker_threshold = TICKER_ADJUSTMENTS[ticker].get(indicator)
        if ticker_threshold is not None:
            base_threshold = ticker_threshold
    
    # Корекцandя по andймфрейму
    if timeframe and timeframe in TIMEFRAME_ADJUSTMENTS:
        tf_multiplier = TIMEFRAME_ADJUSTMENTS[timeframe].get(indicator, 1.0)
        base_threshold *= tf_multiplier
    
    return base_threshold

def get_trigger_config(trigger_name: str) -> Dict[str, Any]:
    """
    Поверandє конфandгурацandю тригера
    """
    return TRIGGER_ACTIONS.get(trigger_name, {})

def get_filter_settings() -> Dict[str, Any]:
    """
    Поверandє settings фandльтрацandї
    """
    return FILTER_SETTINGS

def is_system_enabled(feature: str) -> bool:
    """
    Перевandряє чи увandмкnotна функцandя system
    """
    return SYSTEM_CONFIG.get(feature, False)

def get_all_ticker_thresholds(ticker: str) -> Dict[str, float]:
    """
    Поверandє all пороги for тandкера
    """
    if ticker not in TICKER_ADJUSTMENTS:
        return {}
    
    thresholds = {}
    base_thresholds = TICKER_ADJUSTMENTS[ticker]
    
    for indicator, base_value in base_thresholds.items():
        if indicator in SIGNIFICANCE_THRESHOLDS:
            thresholds[indicator] = base_value
    
    return thresholds

def get_timeframe_multipliers() -> Dict[str, Dict[str, float]]:
    """
    Поверandє множники for andймфреймandв
    """
    return TIMEFRAME_ADJUSTMENTS

# --- ВАЛІДАЦІЯ КОНФІГУРАЦІЇ ---
def validate_config() -> Dict[str, Any]:
    """
    Валandдує конфandгурацandю and поверandє сandтистику
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Перевandрка баwithових порогandв
    for indicator, threshold in SIGNIFICANCE_THRESHOLDS.items():
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            validation_results["errors"].append(f"Invalid threshold for {indicator}: {threshold}")
            validation_results["valid"] = False
    
    # Перевandрка тandкерних налаштувань
    for ticker, adjustments in TICKER_ADJUSTMENTS.items():
        for indicator, value in adjustments.items():
            if not isinstance(value, (int, float)) or value <= 0:
                validation_results["errors"].append(f"Invalid ticker adjustment for {ticker}.{indicator}: {value}")
                validation_results["valid"] = False
    
    # Перевandрка andймфреймних налаштувань
    for timeframe, adjustments in TIMEFRAME_ADJUSTMENTS.items():
        for indicator, multiplier in adjustments.items():
            if not isinstance(multiplier, (int, float)) or multiplier <= 0:
                validation_results["errors"].append(f"Invalid timeframe adjustment for {timeframe}.{indicator}: {multiplier}")
                validation_results["valid"] = False
    
    # Сandтистика
    validation_results["stats"] = {
        "total_indicators": len(SIGNIFICANCE_THRESHOLDS),
        "total_tickers": len(TICKER_ADJUSTMENTS),
        "total_timeframes": len(TIMEFRAME_ADJUSTMENTS),
        "total_triggers": len(TRIGGER_ACTIONS),
        "system_features": len(SYSTEM_CONFIG)
    }
    
    return validation_results

if __name__ == "__main__":
    # Тест валandдацandї
    validation = validate_config()
    print("[SEARCH] Валandдацandя конфandгурацandї тригерandв:")
    print(f"  - Валandднandсть: {'[OK]' if validation['valid'] else '[ERROR]'}")
    print(f"  - Помилки: {len(validation['errors'])}")
    print(f"  - Попередження: {len(validation['warnings'])}")
    print(f"  - Сandтистика: {validation['stats']}")
