# config/thresholds.py - ПОРОГИ (ЗАСТАРІЛИЙ)
# 
# [WARN] ЦЕЙ ФАЙЛ ЗАСТАРІВ! 
# Використовуйте config/unified_config.py для всіх нових розробок
#
# Пороги тепер в UnifiedConfig.INDICATORS

from typing import Dict, Tuple, Any
import warnings
import logging

# Імпортуємо з unified_config
from config.unified_config import config, UnifiedConfig

# Logger для зворотної сумісності
logger = logging.getLogger(__name__)

# Для зворотної сумісності
RSI_THRESHOLDS = {tf: cfg.get("rsi_threshold", (30, 70)) for tf, cfg in config.TIMEFRAMES.items()}

BASE_FORECAST_THRESHOLDS = {
    "bullish": 0.01,
    "bearish": -0.01
}

BASE_SENTIMENT_THRESHOLDS = {
    "positive": 0.2,
    "negative": -0.2
}

BASE_INSIDER_THRESHOLDS = {
    "buy": (5.0, float("inf")),
    "sell": (float("-inf"), -5.0)
}

# Попередження
warnings.warn(
    "config/thresholds.py is deprecated. Use config.unified_config.INDICATORS instead.",
    DeprecationWarning,
    stacklevel=2
)

# --- ФУНКЦІЇ ДЛЯ ЗВОРОТНОЇ СУМІСНОСТІ ---

def get_all_thresholds(ticker: str = None, interval: str = None) -> Dict[str, Any]:
    """
    Отримати всі пороги для тікера та інтервалу (зворотна сумісність)
    """
    try:
        # Спробуємо використовувати unified_config
        rsi_thresh = UnifiedConfig.get_rsi_threshold(interval or "60m")
        
        return {
            "rsi_overbought": rsi_thresh[1],
            "rsi_oversold": rsi_thresh[0],
            "forecast_bullish": BASE_FORECAST_THRESHOLDS["bullish"],
            "forecast_bearish": BASE_FORECAST_THRESHOLDS["bearish"],
            "sentiment_positive": BASE_SENTIMENT_THRESHOLDS["positive"],
            "sentiment_negative": BASE_SENTIMENT_THRESHOLDS["negative"],
            "insider_buy": BASE_INSIDER_THRESHOLDS["buy"][0],
            "insider_sell": BASE_INSIDER_THRESHOLDS["sell"][1],
        }
    except Exception as e:
        # Fallback до базових значень
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "forecast_bullish": 0.01,
            "forecast_bearish": -0.01,
            "sentiment_positive": 0.2,
            "sentiment_negative": -0.2,
            "insider_buy": 5.0,
            "insider_sell": -5.0,
        }

def get_rsi_threshold(ticker: str, interval: str) -> Tuple[int, int]:
    """Отримати RSI пороги (зворотна сумісність)"""
    return UnifiedConfig.get_rsi_threshold(interval or "60m")

# --- ПОРОГИ ЗНАЧУЩОСТІ ТА ШУМУ ---

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

# --- ТАЙМФРЕЙМНІ КОРЕКЦІЇ ---

TIMEFRAME_ADJUSTMENTS = {
    "15m": {"multiplier": 0.8},    # Меншand пороги for коротких andнтервалandв
    "60m": {"multiplier": 1.0},    # Баwithовand пороги
    "1d": {"multiplier": 1.2}      # Бandльшand пороги for whereнних графandкandв
}

# --- ТИКЕРНІ КОРЕКЦІЇ ДЛЯ РОЗШИРЕНИХ ПОРОГІВ ---

TICKER_RSI_ADJUSTMENTS = {
    "TSLA": {"15m": (15, 85), "30m": (18, 82), "60m": (20, 80), "1h": (20, 80), "1d": (30, 70)},
    "NVDA": {"15m": (15, 85), "30m": (18, 82), "60m": (22, 78), "1h": (22, 78), "1d": (30, 70)},
    "SPY":  {"15m": (20, 80), "30m": (20, 80), "60m": (25, 75), "1h": (25, 75), "1d": (30, 70)},
    "QQQ":  {"15m": (20, 80), "30m": (20, 80), "60m": (25, 75), "1h": (25, 75), "1d": (30, 70)}
}

TICKER_FORECAST_ADJUSTMENTS = {
    "TSLA": {"bullish": 0.015, "bearish": -0.015},
    "NVDA": {"bullish": 0.013, "bearish": -0.013},
    "SPY":  {"bullish": 0.008, "bearish": -0.008},
    "QQQ":  {"bullish": 0.009, "bearish": -0.009}
}

TICKER_SENTIMENT_ADJUSTMENTS = {
    "TSLA": {"positive": 0.25, "negative": -0.25},
    "NVDA": {"positive": 0.23, "negative": -0.23},
    "SPY":  {"positive": 0.18, "negative": -0.18},
    "QQQ":  {"positive": 0.19, "negative": -0.19}
}

TICKER_INSIDER_ADJUSTMENTS = {
    "TSLA": {"buy": (4.0, float("inf")), "sell": (float("-inf"), -4.0)},
    "NVDA": {"buy": (6.0, float("inf")), "sell": (float("-inf"), -6.0)},
    "SPY":  {"buy": (5.0, float("inf")), "sell": (float("-inf"), -5.0)},
    "QQQ":  {"buy": (5.0, float("inf")), "sell": (float("-inf"), -5.0)}
}

# --- СИСТЕМНІ НАЛАШТУВАННЯ ---

SYSTEM_CONFIG = {
    "significance_detection": True,
    "adaptive_thresholds": True,
    "noise_filtering": True,
    "ticker_adjustments": True,
    "timeframe_adjustments": True
}

FILTER_SETTINGS = {
    "min_confidence": 0.7,
    "max_signals_per_minute": 10,
    "cooldown_period_seconds": 30
}

TRIGGER_ACTIONS = {
    "price_spike": {
        "description": "Рandwithкий стрибок цandни",
        "action": "analyze_volume",
        "priority": "high"
    },
    "volume_surge": {
        "description": "Сплеск обсягandв",
        "action": "check_news",
        "priority": "medium"
    },
    "sentiment_shift": {
        "description": "Змandна сентименту",
        "action": "analyze_context",
        "priority": "medium"
    }
}

# --- TTL ДЛЯ КЕШУ ---

CACHE_TTL: Dict[str, int] = {
    "15m": 3,
    "30m": 5,
    "60m": 10,
    "1h": 14,
    "1d": 90,
    "news": 1
}

# --- ОСНОВНІ ФУНКЦІЇ ---

def get_rsi_threshold(ticker: str, interval: str) -> Tuple[int, int]:
    """Повертає RSI пороги для тікера та інтервалу"""
    thresholds = TICKER_RSI_ADJUSTMENTS.get(ticker, {}).get(interval)
    if thresholds:
        logger.debug(f"[Thresholds] RSI {ticker} {interval}  {thresholds}")
        return thresholds
    logger.info(f"[Thresholds] Використовуємо базовий RSI поріг for {ticker} {interval}")
    return RSI_THRESHOLDS.get(interval, (30, 70))

def get_forecast_threshold(ticker: str, direction: str) -> float:
    """Поверandє порandг прогноwithу for тandкера and напрямку"""
    if direction not in {"bullish", "bearish"}:
        logger.error(f"[Thresholds] [ERROR] Невandдомий напрямок прогноwithу: {direction}")
        raise ValueError(f"Невandдомий напрямок прогноwithу: {direction}")
    val = TICKER_FORECAST_ADJUSTMENTS.get(ticker, {}).get(direction, BASE_FORECAST_THRESHOLDS[direction])
    logger.debug(f"[Thresholds] Forecast {ticker} {direction}  {val}")
    return val

def get_sentiment_threshold(ticker: str, sentiment_type: str) -> float:
    """Поверandє порandг сентименту for тandкера and типу"""
    if sentiment_type not in {"positive", "negative"}:
        logger.error(f"[Thresholds] [ERROR] Невandдомий тип сентименту: {sentiment_type}")
        raise ValueError(f"Невandдомий тип сентименту: {sentiment_type}")
    val = TICKER_SENTIMENT_ADJUSTMENTS.get(ticker, {}).get(sentiment_type, BASE_SENTIMENT_THRESHOLDS[sentiment_type])
    logger.debug(f"[Thresholds] Sentiment {ticker} {sentiment_type}  {val}")
    return val

def get_insider_threshold(ticker: str, action: str) -> Tuple[float, float]:
    """Поверandє порandг andнсайwhereрських дandй for тandкера and дandї"""
    if action not in {"buy", "sell"}:
        logger.error(f"[Thresholds] [ERROR] Невandдомий тип дandї: {action}")
        raise ValueError(f"Невandдомий тип дandї: {action}")
    val = TICKER_INSIDER_ADJUSTMENTS.get(ticker, {}).get(action, BASE_INSIDER_THRESHOLDS[action])
    logger.debug(f"[Thresholds] Insider {ticker} {action}  {val}")
    return val

def get_noise_thresholds() -> Dict[str, float]:
    """Поверandє пороги вandд шуму for аналandwithу"""
    logger.info(f"[Thresholds] Отримано пороги вandд шуму: {len(SIGNIFICANCE_THRESHOLDS)} andндикаторandв")
    return SIGNIFICANCE_THRESHOLDS.copy()

def get_threshold(indicator: str, ticker: str = None, timeframe: str = None) -> float:
    """Поверandє адаптований порandг for andндикатора"""
    base_threshold = SIGNIFICANCE_THRESHOLDS.get(indicator)
    if not base_threshold:
        logger.warning(f"[Thresholds] Невandдомий andндикатор: {indicator}")
        return 0.0
    
    # Тandкернand корекцandї
    if ticker and ticker in TICKER_ADJUSTMENTS:
        ticker_adj = TICKER_ADJUSTMENTS[ticker].get(indicator)
        if ticker_adj:
            base_threshold = ticker_adj
    
    # Таймфреймнand корекцandї
    if timeframe and timeframe in TIMEFRAME_ADJUSTMENTS:
        multiplier = TIMEFRAME_ADJUSTMENTS[timeframe].get("multiplier", 1.0)
        base_threshold *= multiplier
    
    return base_threshold

def get_all_ticker_thresholds(ticker: str) -> Dict[str, float]:
    """Поверandє all пороги for тandкера"""
    if ticker not in TICKER_ADJUSTMENTS:
        return SIGNIFICANCE_THRESHOLDS.copy()
    
    thresholds = SIGNIFICANCE_THRESHOLDS.copy()
    thresholds.update(TICKER_ADJUSTMENTS[ticker])
    return thresholds

def get_timeframe_multipliers(timeframe: str = None) -> Dict[str, float]:
    """Поверandє мультиплandкатори for andймфрейму"""
    if timeframe:
        return TIMEFRAME_ADJUSTMENTS.get(timeframe, {"multiplier": 1.0})
    return {tf: adj["multiplier"] for tf, adj in TIMEFRAME_ADJUSTMENTS.items()}

def get_trigger_config(trigger_name: str) -> Dict[str, Any]:
    """Поверandє конфandгурацandю тригера"""
    return TRIGGER_ACTIONS.get(trigger_name, {})

def get_filter_settings() -> Dict[str, Any]:
    """Поверandє settings фandльтрацandї"""
    return FILTER_SETTINGS.copy()

def is_system_enabled(feature: str) -> bool:
    """Перевandряє чи увandмкnotна системна функцandя"""
    return SYSTEM_CONFIG.get(feature, False)

def validate_config() -> Dict[str, Any]:
    """Валandдує конфandгурацandю"""
    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Перевірка базових порогів
    if not RSI_THRESHOLDS:
        validation["errors"].append("Відсутні базові RSI пороги")
        validation["valid"] = False
    
    # Перевandрка порогandв withначущостand
    if not SIGNIFICANCE_THRESHOLDS:
        validation["errors"].append("Вandдсутнand пороги withначущостand")
        validation["valid"] = False
    
    # Перевірка тикерних корекцій
    for ticker, adjustments in TICKER_ADJUSTMENTS.items():
        for indicator, value in adjustments.items():
            if indicator not in SIGNIFICANCE_THRESHOLDS:
                validation["warnings"].append(f"Невідомий індикатор for {ticker}: {indicator}")
    
    return validation

# --- УТИЛІТИ ---

def print_thresholds_summary():
    """Друкує пandдсумок allх порогandв"""
    print("="*80)
    print("[TOOL] ЄДИНИЙ ЦЕНТР УПРАВЛІННЯ ПОРОГАМИ")
    print("="*80)
    
    print(f"\n[DATA] Баwithовand пороги withначущостand: {len(SIGNIFICANCE_THRESHOLDS)}")
    for indicator, threshold in SIGNIFICANCE_THRESHOLDS.items():
        print(f"  {indicator}: {threshold*100:.1f}%")
    
    print(f"\n[TARGET] Тandкернand корекцandї: {len(TICKER_ADJUSTMENTS)}")
    for ticker, adjustments in TICKER_ADJUSTMENTS.items():
        print(f"  {ticker}: {len(adjustments)} корекцandй")
    
    print(f"\n Таймфреймнand корекцandї: {len(TIMEFRAME_ADJUSTMENTS)}")
    for timeframe, adjustments in TIMEFRAME_ADJUSTMENTS.items():
        print(f"  {timeframe}: {adjustments}")
    
    print(f"\n[TOOL] Тригери: {len(TRIGGER_ACTIONS)}")
    for trigger, config in TRIGGER_ACTIONS.items():
        print(f"  {trigger}: {config.get('description', 'No description')}")
    
    print(f"\n Системнand settings: {len(SYSTEM_CONFIG)}")
    for feature, enabled in SYSTEM_CONFIG.items():
        status = "[OK]" if enabled else "[ERROR]"
        print(f"  {feature}: {status}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print_thresholds_summary()
