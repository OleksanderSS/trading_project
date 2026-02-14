# config/model_features.py

MODEL_FEATURES = {
    #  Баwithовand цandновand оwithнаки
    "base": [
        "open", "high", "low", "close", "volume",
        "return", "price_change_pct", "gap_percent"
    ],

    #  Трендовand оwithнаки (тandльки баwithовand)
    "trend": [
        "SMA_5", "EMA_20", "RSI_14", "MACD_day", "macd_signal",
        "ma_diff", "ma_cross"
    ],

    # [DOWN] Волатильнandсть (баwithовand)
    "volatility": [
        "ATR_14", "vol_std_7", "vol_var_7", "vol_delta"
    ],

    # [FAST] Моментум
    "momentum": [
        "momentum_3d", "momentum_7d", "mfi"
    ],

    #  Календарнand оwithнаки
    "calendar": [
        "weekday", "is_earnings_day"
    ],

    #  Новини (тandльки баwithовand)
    "news": [
        "has_news", "sentiment_score", "adjusted_score", "match_count", "avg_news_lag", "news_score"
    ],

    #  Контекст новин
    "market_news_context": [
        "has_general_news", "general_sentiment_score",
        "general_adjusted_score", "general_news_count",
        "macro_event_intensity"
    ],

    #  Макроекономandка (тandльки баwithовand сигнали)
    "macro": [
        "VIX_SIGNAL", "FEDFUNDS_SIGNAL", "T10Y2Y_SIGNAL"
    ],

    #  Інсайwhereри
    "insider": [
        "insider_buy_pressure",
        "insider_sell_pressure",
        "weighted_insider_signal"
    ],

    #  Свandчковand патерни
    "candles": [
        "doji", "hammer", "shooting_star",
        "engulfing_bullish", "engulfing_bearish",
        "morning_star", "evening_star",
        "piercing_pattern", "dark_cloud_cover"
    ],

    # [UP] Теханалandwith (роwithширений набandр)
    "ta": [
        "bollinger_upper", "bollinger_lower", "bollinger_bandwidth",
        "stochastic_k", "stochastic_d",
        "cci", "willr", "obv", "chaikin_oscillator"
    ],

    # [DATA] Сеwithоннandсть
    "seasonality": [
        "month", "quarter", "is_month_end", "is_quarter_end", "is_year_end",
        "days_to_next_holiday"
    ],

    # [DATA] Лandквandднandсть
    "liquidity": [
        "bid_ask_spread", "order_book_depth", "avg_trade_size"
    ],

    #  Кросактивнand withвяwithки
    "cross_asset": [
        "spy_vs_qqq_corr", "tsla_vs_nvda_corr"
    ],

    #  Сутностand
    "entities": [
        "entity_count"
    ],

    # [BRAIN] Utility / Enrichment
    "utility": [
        "ae_error"
    ]
}

#  Повний список оwithнак for моwhereлей (ядро)
ALL_MODEL_FEATURES = sum(MODEL_FEATURES.values(), [])