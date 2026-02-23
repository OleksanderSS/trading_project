# config/feature_config.py

import logging
from config.macro_config import build_macro_layers

logger = logging.getLogger(__name__)

# 🔹 Локальні фічі (свічка)
LOCAL_FEATURES = [
    # PRE phase features (before news)
    "close_pre", "open_pre", "high_pre", "low_pre", "volume_pre",
    "vol_rel_pre", "atr_rel_pre", "dist_to_ema_pre", "rsi_pre",
    
    # POST phase features (after news)
    "gap_percent", "impact_1_pct", "vol_impact_1", "shadow_ratio_1",
    "impact_2_pct", "reversal_score", "vol_trend",
    
    # Standard OHLCV
    "open", "high", "low", "close", "volume",
    "gap_percent", "price_change_pct", "return",
    "vol_delta", "weekday", "is_earnings_day"
]

# 🔹 EVENT фічі (події)
EVENT_FEATURES = [
    # Time-based event features
    "is_pre_market", "is_during_market", "is_post_market",
    "is_earnings_day", "is_fomc_day", "is_quarter_end",
    
    # Price-based event features  
    "is_gap_up", "is_gap_down", "is_large_gap",
    "is_breakout_up", "is_breakout_down",
    "is_volume_spike", "is_volume_crush",
    
    # Sentiment-based event features
    "is_high_sentiment", "is_low_sentiment",
    "is_sentiment_change", "is_extreme_sentiment",
    
    # Technical event features
    "is_rsi_oversold", "is_rsi_overbought",
    "is_macd_bullish", "is_macd_bearish",
    "is_volatility_spike", "is_trend_change"
]

# 🔹 Alias для FRED серій (розширено глибокими індикаторами)
FRED_ALIAS = {
    # Основні макроіндикатори
    "FEDFUNDS": "FEDFUNDS",
    "T10Y2Y": "T10Y2Y",
    "UNRATE": "UNRATE",
    "GS10": "GS10",
    "GS2": "GS2",
    "CPIAUCSL": "CPI",
    "VIXCLS": "VIX",
    "DGS10": "DGS10",
    "GDP": "GDP",
    
    # 🏦 Промислові індикатори (випереджаючі)
    "INDPRO": "INDUSTRIAL_PRODUCTION",      # Промислове виробництво
    "CAPUTLB50001SQ": "CAPACITY_UTIL",      # Завантаженість потужностей
    "PAYEMS": "NONFARM_PAYROLLS",           # Зайнятість
    
    # 📈 Випереджаючі індикатори
    "UMCSENT": "CONSUMER_SENTIMENT",        # Настрої споживачів (Мічиган)
    "HOUST": "HOUSING_STARTS",              # Початок будівництва житла
    "PERMIT": "BUILDING_PERMITS",           # Дозволи на будівництво
    
    # 💰 Глибокі фінансові індикатори
    "TEDRATE": "TED_SPREAD",                # TED спред (ранній індикатор кризи)
    "BAMLH0A0HYM2": "HIGH_YIELD_SPREAD",    # Спред високодохідних облігацій
    "DEXUSEU": "USD_EUR",                   # Курс долара до євро
    
    # 🌍 Товарні ринки (інфляційні сигнали)
    "DCOILWTICO": "WTI_OIL",               # Нафта WTI
    # "GOLDAMGBD228NLBM": "GOLD_PRICE",      # Золото (помилка 400)
    "DEXCHUS": "USD_CNY",                   # Юань (торгові війни)
    
    # 📊 Високочастотні індикатори
    "DSPIC96": "REAL_DISPOSABLE_INCOME",    # Реальний дохід
    "RSAFS": "RETAIL_SALES",               # Роздрібні продажі
    "TOTALSA": "TOTAL_VEHICLE_SALES"        # Продажі авто (споживчі витрати)
}
FRED_SERIES = list(FRED_ALIAS.keys())

# 🔹 Макро фічі
MACRO_FEATURES = [
    "FEDFUNDS_WEIGHTED", "T10Y2Y_WEIGHTED", "CPI_WEIGHTED",
    "UNRATE_WEIGHTED", "GS10_WEIGHTED", "GS2_WEIGHTED",
    "VIX_WEIGHTED", "DGS10_WEIGHTED", "GDP_WEIGHTED",
    "cpi_surprise", "gdp_surprise",
    "FEDFUNDS_LAG_7d", "CPI_LAG_7d",
    "FEDFUNDS_change", "UNRATE_diff", "CPI_inflation",
    "macro_bias", "macro_volatility",
    "macro_sentiment_interaction", "macro_vix_interaction",
    "sentiment_vix_interaction"
]

# 🔒 Безпечні для заповнення фічі
SAFE_FILL_FEATURES = [
    "adjusted_score", "avg_news_lag", "impact_score",
    "reaction_strength", "impact_score_minus_adjusted",
    "news_score"
]

# 📈 Теханаліз (Оновлено) - Єдине джерело правди для всіх технічних індикаторів
TA_FEATURES = [
    # --- Moving Averages & Trend ---
    "SMA_5", "SMA_10", "SMA_20", "SMA_30", "SMA_50", "SMA_200",
    "EMA_10", "EMA_20", "EMA_50", "EMA_200",
    "MACD", "MACD_signal", "MACD_diff",
    "ADX", "DI_plus", "DI_minus",
    "Vortex_Plus", "Vortex_Minus",
    "KST", "KST_Signal",
    "TRIX",
    "Mass_Index",
    "Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_Senkou_A", "Ichimoku_Senkou_B", "Ichimoku_Chikou",

    # --- Oscillators & Momentum ---
    "RSI_14",
    "Stoch_K", "Stoch_D",
    "CCI",
    "Williams_R",
    "Ultimate_Osc",
    "DPO", # Detrended Price Oscillator

    # --- Volume Indicators ---
    "OBV", # On-Balance Volume
    "MFI", # Money Flow Index
    "VPT", # Volume Price Trend

    # --- Volatility Indicators ---
    "ATR_14", # Average True Range
    "BB_upper", "BB_middle", "BB_lower", # Bollinger Bands
    "KC_Upper", "KC_Middle", "KC_Lower", # Keltner Channels
    "Donchian_Upper", "Donchian_Middle", "Donchian_Lower", # Donchian Channels

    # --- Other Features ---
    "gap_size", "gap_size_pct", "gap_signal",
    "Fib_236", "Fib_382", "Fib_50", "Fib_618", # Fibonacci Retracements
    "HA_Open", "HA_High", "HA_Low", "HA_Close" # Heikin-Ashi
]


# 🌍 Макроекономічний фон
MACRO_LAYERS = build_macro_layers()
MACRO_BACKGROUND_FEATURES = MACRO_LAYERS["background"]
MACRO_TREND_FEATURES = MACRO_LAYERS["trend"]
MACRO_SIGNAL_FEATURES = MACRO_LAYERS["signal"]
MACRO_CONTEXT_FEATURES = MACRO_TREND_FEATURES + MACRO_SIGNAL_FEATURES + [
    "cpi_surprise", "gdp_surprise",
    "FEDFUNDS_LAG_7d", "CPI_LAG_7d",
    "FEDFUNDS_change", "UNRATE_diff", "CPI_inflation",
    "macro_bias", "macro_volatility",
    "macro_sentiment_interaction", "macro_vix_interaction",
    "sentiment_vix_interaction"
]

# 📰 Новини (розширені з агрегатами)
NEWS_FEATURES = [
    "news_count",
    "sentiment_score",
    "sentiment_label_encoded"
]

ENRICHMENT_FEATURES = [
    "summary_length",
    "keyword_count"
]

MARKET_NEWS_CONTEXT_FEATURES = [
    "avg_daily_sentiment",
    "news_flow_rate"
]

NEWS_CONTEXT_FEATURES = NEWS_FEATURES + ENRICHMENT_FEATURES + [
    "hour_of_day", "is_pre_market", "is_after_hours", "time_to_open",
    "daily_sentiment", "news_count", "avg_impact", "gdelt_daily",
    "daily_sentiment_lag1", "daily_sentiment_lag3", "avg_impact_lag1",
    "source_diversity", "avg_sentiment_lag7", "news_volatility",
    "phase_weighted_score", "signal_strength",
    "trend_label", "trend_boost_factor"
]

# 🧠 Реверсивний вплив новин
REVERSE_IMPACT_FEATURES = [
    "reaction_strength",
    "sentiment_miss",
    "impact_score_minus_adjusted",
    "reaction_category",
    "impact_ratio",
    "phase_spike_alignment",
    "context_alignment_score"
]

# 🕯 Свічкові патерни
CANDLE_FEATURES = [
    "doji", "hammer", "shooting_star", "engulfing_bullish", "engulfing_bearish",
    "morning_star", "evening_star", "piercing_pattern", "dark_cloud_cover"
]

# 📅 Сезонність
CALENDAR_FEATURES = [
    "weekday", "month", "quarter", "is_earnings_day",
    "hour_of_day", "is_pre_market", "is_after_hours",
    "vol_std_7", "vol_std_14", "vol_var_7", "vol_var_14",
    "avg_news_lag", "macro_event_intensity",
    "is_month_end", "is_quarter_end", "is_year_end",
    "days_to_next_holiday"
]

# 📊 Ліквідність та альтернативні дані (розширено)
LIQUIDITY_FEATURES = [
    "bid_ask_spread", "order_book_depth", "avg_trade_size",
    # Нові альтернативні індикатори
    "fear_greed_index", "aaii_sentiment_spread", "cftc_positioning",
    "ny_fed_wei", "philly_fed_ads", "google_trends_recession"
]

# 🔗 Крос‑активні зв’язки
CROSS_ASSET_FEATURES = [
    "spy_vs_qqq_corr", "tsla_vs_nvda_corr"
]

# 🧩 Інсайдери
INSIDER_FEATURES = [
    "insider_buy_pressure", "insider_sell_pressure", "insider_net_activity"
]

# 🔑 ОНОВЛЕНО: Шар для мульти-ТФ аналізу (з додаванням 1m та 5m)
def build_multi_tf_features(ticker: str) -> list:
    tf_pairs = [("15m", "1h"), ("1h", "1d")]
    features = [f"pct_growing_candles_{ticker.lower()}_{tf1}_{tf2}" for tf1, tf2 in tf_pairs]
    features.append(f"tf_momentum_score_{ticker.lower()}")
    return features

# Утиліти
UTILITY_FEATURES = ["ae_error"]

# Похідні
DERIVED_FEATURES = [
    "sentiment_score_abs",
    "sentiment_trend",
    "close_ma5",
    "momentum_5",
    "sentiment_vix_interaction"
]

TICKER_TARGET_MAP = {
    "NVDA": "target_close_nvda_15m",
    "SPY": "target_close_spy_1d",
    "QQQ": "target_close_qqq_1d",
    "TSLA": "target_close_tsla_15m"
}

TECHNICAL_FEATURES = TA_FEATURES # Тепер вказує на єдине джерело

# 🔹 Контекстні фічі (навколо новини)
CONTEXT_FEATURES = [
    # Sentiment context
    "general_sentiment_score", "sentiment_volatility", "sentiment_trend_3d",
    
    # Macro context (placeholder - requires VIX data)
    "vix_level", "vix_change", "rate_environment",
    "macro_event_intensity", "fear_greed_index",
    
    # Sector context (placeholder)
    "sector_momentum", "sector_rotation_score",
    
    # News context
    "news_density", "news_frequency_score", "breaking_news_flag"
]

# 🔹 Всі фічі
ALL_FEATURES = (
    LOCAL_FEATURES +
    CONTEXT_FEATURES +
    TECHNICAL_FEATURES +
    MACRO_FEATURES +
    CROSS_ASSET_FEATURES +
    INSIDER_FEATURES +
    NEWS_CONTEXT_FEATURES +
    REVERSE_IMPACT_FEATURES +
    CANDLE_FEATURES +
    TA_FEATURES +
    CALENDAR_FEATURES +
    LIQUIDITY_FEATURES +
    UTILITY_FEATURES +
    DERIVED_FEATURES
)

# 🔹 Золотий список фіч (Core + Missing) 
# Імпортуємо з основного конфігу для уникнення дублювання
from config.config import TICKERS, TIME_FRAMES

# Конвертуємо dict в list для зручності
TICKER_LIST = list(TICKERS.keys())
TIMEFRAME_LIST = list(TIME_FRAMES.keys())

# Базові розрахункові фічі (те, що ти затвердив)
BASE_TICKER_FEATURES = [
    "close_pre", "vol_rel_pre", "atr_rel_pre", "dist_to_ema_pre", "rsi_pre",
    "gap_percent", "impact_1_pct", "vol_impact_1", "shadow_ratio_1",
    "impact_2_pct", "reversal_score", "vol_trend"
]

# Глобальні фічі (не залежать від тікера)
GLOBAL_CONTEXT_FEATURES = [
    "weekday", "hour_of_day", "market_session", 
    "is_earnings_day", "vix_level", "sentiment_score", "breaking_news_flag"
]

# Динамічне формування повного списку CORE_FEATURES
CORE_FEATURES = []
for ticker in TICKER_LIST:
    for tf in TIMEFRAME_LIST:
        for feature in BASE_TICKER_FEATURES:
            CORE_FEATURES.append(f"{ticker}_{tf}_{feature}")

CORE_FEATURES.extend(GLOBAL_CONTEXT_FEATURES)

ALL_MODEL_FEATURES = ALL_FEATURES
USE_CORE_FEATURES = True

# 💡 Сила сигналу за шарами (ПОКИ ВСІ = 1.0)
SIGNAL_STRENGTH_BY_LAYER = {
    # Логіка готова, але поки всі шари нейтральні (1.0)
    # Після тюнінгу моделей можна буде змінити:
    # "local": 1.3,      # Підсилити локальні сигнали
    # "news": 0.6,       # Ослабити новинні сигнали
    # "macro": 0.8,      # Помірно ослабити макро
    # "historical": 1.2, # Підсилити історичний контекст
    # "insider": 0.9     # Легко ослабити інсайдерські дані
}

def get_layer_weight(layer_name: str) -> float:
    """Повертає вагу шару.
    
    Якщо шар не вказано в SIGNAL_STRENGTH_BY_LAYER, 
    повертає 1.0 (нейтрально, не впливає на результат)
    """
    return SIGNAL_STRENGTH_BY_LAYER.get(layer_name, 1.0)


FEATURE_LAYERS = {
    "local": LOCAL_FEATURES,
    "candles": CANDLE_FEATURES,
    "ta": TA_FEATURES, # Оновлено
    "macro": MACRO_CONTEXT_FEATURES,
    # 🔑 FIX: Об'єднана, повна версія News
    "news": NEWS_CONTEXT_FEATURES + [
        "sent_neg", "sent_neu", "sent_pos",
        "sentiment_score", "news_score", "summary"
    ],
    "market_news_context": MARKET_NEWS_CONTEXT_FEATURES,
    "reverse_impact": REVERSE_IMPACT_FEATURES,
    "seasonality": CALENDAR_FEATURES,
    "liquidity": LIQUIDITY_FEATURES,
    "cross_asset": CROSS_ASSET_FEATURES,
    "insider": INSIDER_FEATURES,
    "entities": ["entity_count"],
    # 🔑 FIX: Об'єднана версія Utility
    "utility": UTILITY_FEATURES + ["ae_error"],
    "multi_tf_alignment": sum([build_multi_tf_features(t) for t in TICKER_TARGET_MAP.keys()], []),
    "derived" : DERIVED_FEATURES,
    "historical": [
        "crisis_similarity_2008",
        "crisis_similarity_2020", 
        "geopolitical_tension",
        "tech_disruption_level",
        "market_regime_stability"
    ],
    "leading_indicators": [
        "leading_crisis_probability",
        "leading_breakthrough_probability",
        "leading_market_regime_shift",
        "leading_signal_strength",
        "crisis_ted_spread_spike",
        "crisis_yield_curve_inversion",
        "crisis_vix_stress",
        "breakthrough_institutional_interest",
        "breakthrough_momentum_acceleration",
        "breakthrough_sentiment_shift"
    ],
    "contextual_knowledge": [
        "predicted_market_reaction",
        "event_severity_score",
        "recovery_pattern_similarity",
        "sector_rotation_probability",
        "crisis_escalation_risk",
        "paradigm_shift_indicator"
    ],
    "qualitative_events": [
        "financial_crisis_pattern",
        "geopolitical_escalation_pattern", 
        "tech_breakthrough_pattern",
        "fed_policy_shift_pattern",
        "market_panic_indicator",
        "sector_disruption_score",
        "recovery_timeline_estimate",
        "historical_analogy_strength"
    ]
}

def get_features_by_layer(layer_name: str) -> list:
    """Повертає список фічей для заданого шару."""
    if layer_name == "context":
        return [
            "avg_news_score", "avg_sentiment", "avg_reaction",
            "macro_ratio", "source_diversity"
        ]
    features = FEATURE_LAYERS.get(layer_name, [])
    if not features:
        logger.warning(f"[FeatureLayers] ⚠️ Шар '{layer_name}' не знайдено або він порожній")
    else:
        logger.info(f"[FeatureLayers] ✅ Шар '{layer_name}' → {len(features)} фічей (вага={get_layer_weight(layer_name)})")
    return features
