# config/feature_layers.py

import logging
from config.macro_config import build_macro_layers
from config.feature_config import (
    TECHNICAL_FEATURES,
    NEWS_CONTEXT_FEATURES,
    CALENDAR_FEATURES,
    LIQUIDITY_FEATURES,
    UTILITY_FEATURES,
    MARKET_NEWS_CONTEXT_FEATURES,
    TICKER_TARGET_MAP,
    DERIVED_FEATURES,
    get_features_by_layer,
    get_layer_weight
)

logger = logging.getLogger(__name__)

# 🔹 Локальні фічі (свічка)
LOCAL_FEATURES = [
    "open", "high", "low", "close", "volume",
    "gap_percent", "price_change_pct", "return",
    "vol_delta", "weekday", "is_earnings_day"
]

# 🔸 Короткострокові технічні сигнали
SHORT_TERM_FEATURES = [
    "SMA_5", "SMA_10", "SMA_20", "SMA_30",
    "EMA_day", "EMA_10", "EMA_20",
    "RSI_day", "RSI_14", "MACD_day",
    "ATR_day", "ATR_14",
    "mfi", "momentum_3d", "momentum_7d",
    "vol_std_3", "vol_std_7", "vol_std_14",
    "vol_var_3", "vol_var_7", "vol_var_14",
    "vol_sma_3", "vol_sma_7", "vol_sma_14",
    "macd", "macd_signal",
    "close_ma5", "close_ma10", "close_ma20", "momentum_5",
    "gap_positive", "gap_large"
]

# 🔶 Довгостроковий тренд
TREND_CONTEXT_FEATURES = [
    "SMA_50", "SMA_200", "EMA_50", "EMA_200",
    "RSI_14", "MACD_day", "macd_signal",
    "momentum_30d", "ma_diff", "ma_cross"
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
NEWS_CONTEXT_FEATURES = [
    "news_count",
    "sentiment_score",
    "sentiment_label_encoded",
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

# 📈 Теханаліз (розширений набір)
TA_FEATURES = [
    "bollinger_upper", "bollinger_lower", "bollinger_bandwidth",
    "stochastic_k", "stochastic_d",
    "cci", "willr", "obv", "chaikin_oscillator"
]

# 📅 Сезонність
SEASONALITY_FEATURES = [
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

# 🧠 Повна структура шарів
FEATURE_LAYERS = {
    "local": LOCAL_FEATURES,
    "candles": CANDLE_FEATURES,
    "short_term": SHORT_TERM_FEATURES,
    "trend": TREND_CONTEXT_FEATURES,
    "ta": TA_FEATURES,
    "macro": MACRO_CONTEXT_FEATURES,
    # 🔑 FIX: Об'єднана, повна версія News
    "news": NEWS_CONTEXT_FEATURES + [
        "sent_neg", "sent_neu", "sent_pos",
        "sentiment_score", "news_score", "summary"
    ],
    "market_news_context": MARKET_NEWS_CONTEXT_FEATURES,
    "reverse_impact": REVERSE_IMPACT_FEATURES,
    "seasonality": SEASONALITY_FEATURES,
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


def get_layer_weight(layer_name: str) -> float:
    """Повертає вагу шару.
    
    Якщо шар не вказано в SIGNAL_STRENGTH_BY_LAYER, 
    повертає 1.0 (нейтрально, не впливає на результат)
    """
    return SIGNAL_STRENGTH_BY_LAYER.get(layer_name, 1.0)


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
