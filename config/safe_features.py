# config/safe_features.py

"""
Беwithпечний набandр фandчей, якand гарантовано andснують у data
"""

# Баwithовand фandчand, якand forвжди є в price data
SAFE_PRICE_FEATURES = [
    "open", "high", "low", "close", "volume"
]

# Баwithовand технandчнand andндикатори (роwithраховуються forвжди)
SAFE_TECHNICAL_FEATURES = [
    "price_change_pct", "return", "gap_percent",
    "SMA_5", "SMA_10", "SMA_20", 
    "RSI_14", "MACD_day", "ATR_14"
]

# Баwithовand новиннand фandчand (with fallback valuesми)
SAFE_NEWS_FEATURES = [
    "sentiment_score",  # forвжди є (0.0 якщо notмає новин)
    "news_score",       # forвжди є (0.0 якщо notмає новин)
    "match_count",      # forвжди є (0 якщо notмає withгадок)
    "has_news"          # forвжди є (False якщо notмає новин)
]

# Баwithовand макро фandчand (with fallback valuesми)
SAFE_MACRO_FEATURES = [
    "VIX_SIGNAL",       # forвжди є (0 якщо notмає data)
    "FEDFUNDS_SIGNAL",  # forвжди є (0 якщо notмає data)
    "CPI_inflation"     # forвжди є (0.0 якщо notмає data)
]

# Баwithовand календарнand фandчand
SAFE_CALENDAR_FEATURES = [
    "weekday",          # forвжди роwithраховується with дати
    "is_earnings_day"   # forвжди є (False якщо notмає andнфо)
]

# Беwithпечний набandр for моwhereлей (15 фandчей)
SAFE_CORE_FEATURES = (
    SAFE_TECHNICAL_FEATURES +
    SAFE_NEWS_FEATURES + 
    SAFE_MACRO_FEATURES +
    SAFE_CALENDAR_FEATURES
)

def get_available_features(df, feature_list):
    """Поверandє тandльки тand фandчand, якand реально є в DataFrame"""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    
    if missing:
        print(f"[WARN] Вandдсутнand фandчand: {missing}")
    
    return available

def get_safe_features(df):
    """Поверandє беwithпечний набandр фandчей for моwhereлand"""
    return get_available_features(df, SAFE_CORE_FEATURES)