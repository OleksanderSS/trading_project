# config/macro_config.py

"""
Configuration for FRED macroeconomic indicators.
Contains series, intervals, periods and save paths.
"""

from datetime import datetime, timedelta
from config.config_manager import resolve_paths, detect_environment

# FRED series - indicator IDs (роwithширено with final_macro_config.py)
FRED_SERIES = [
    # Оригandнальнand баwithовand andндикатори
    "FEDFUNDS", "T10Y2Y", "CPIAUCSL", "UNRATE", "GS10", "GS2",
    "VIXCLS", "DGS10", "GDP", "PPIACO", "DGORDER", "UMCSENT",
    
    # Високий прandоритет - додано with final_macro_config.py
    "PCEPILFE",    # Core PCE (офandцandйний покаwithник ФРС)
    "CCSA",         # Continuing Jobless Claims
    
    # Середнandй прandоритет - додано with final_macro_config.py
    "PCEPI",        # Загальний PCE
    "WALCL",        # Баланс ФРС (лandквandднandсть)
    "MANEMP"        # Зайнятandсть у виробництвand
]

# Display name mapping for convenience (роwithширено)
FRED_ALIAS = {
    # Оригandнальнand баwithовand andндикатори
    "FEDFUNDS": "FEDFUNDS",
    "T10Y2Y": "T10Y2Y",
    "UNRATE": "UNRATE",
    "GS10": "GS10",
    "GS2": "GS2",
    "CPIAUCSL": "CPI",
    "VIXCLS": "VIX",
    "DGS10": "DGS10",
    "GDP": "GDP",
    "PPIACO": "PPI",
    "DGORDER": "MANUFACTURING",
    "UMCSENT": "CONSUMER_SENTIMENT",
    
    # Новand високопрandоритетнand andндикатори
    "PCEPILFE": "CORE_PCE",           # Core PCE (офandцandйний покаwithник ФРС)
    "CCSA": "CONTINUING_CLAIMS",       # Continuing Jobless Claims
    
    # Новand середньопрandоритетнand andндикатори
    "PCEPI": "TOTAL_PCE",              # Загальний PCE
    "WALCL": "FED_BALANCE",            # Баланс ФРС (лandквandднandсть)
    "MANEMP": "MANUFACTURING_EMP",       # Зайнятandсть у виробництвand
    
    # Додано валютні курси
    "DEXUSEU": "USD_EUR",               # USD/EUR Exchange Rate
    "DEXCHUS": "USD_CNY",               # USD/CNY Exchange Rate
    "DCOILWTICO": "WTI_OIL",            # WTI Oil Price
    "BAMLH0A0HYM2": "HIGH_YIELD_SPREAD",  # High Yield Spread
    "TOTALSA": "TOTAL_VEHICLE_SALES"     # Total Vehicle Sales
}

MACRO_WINDOWS_FREQ = {
    "daily": 30,
    "monthly": 12,
    "quarterly": 8
}

# for enrichment in macro_features.py
MACRO_WINDOWS = {
    "trend": 30,
    "zscore": 180
}

# Intervals for each indicator (роwithширено)
DATA_INTERVALS = {
    # Оригandнальнand andндикатори
    "FEDFUNDS": "monthly",
    "T10Y2Y": "monthly",
    "CPIAUCSL": "monthly",
    "UNRATE": "monthly",
    "GS10": "monthly",
    "GS2": "monthly",
    "VIXCLS": "daily",
    "PPIACO": "monthly",
    "DGORDER": "monthly",
    "UMCSENT": "monthly",
    "DGS10": "daily",
    "GDP": "quarterly",
    
    # Новand високопрandоритетнand andндикатори
    "PCEPILFE": "monthly",          # Core PCE
    "CCSA": "weekly",               # Continuing Jobless Claims
    
    # Новand середньопрandоритетнand andндикатори
    "PCEPI": "monthly",             # Total PCE
    "WALCL": "weekly",             # Fed Balance
    "MANEMP": "monthly",            # Manufacturing Employment
    
    # Додано валютні курси та інші індикатори
    "DEXUSEU": "daily",            # USD/EUR Exchange Rate
    "DEXCHUS": "daily",            # USD/CNY Exchange Rate
    "DCOILWTICO": "daily",         # WTI Oil Price
    "BAMLH0A0HYM2": "daily",      # High Yield Spread
    "TOTALSA": "monthly"           # Total Vehicle Sales
}

# Normalization scales by alias (to make signal strength comparable)
NORMALIZATION_SCALES = {
    # Оригandнальнand andндикатори
    "FEDFUNDS": 10,
    "FEDFUNDS_SIGNAL": 10,
    "FEDFUNDS_WEIGHTED": 10,
    "FEDFUNDS_DAYS_SINCE_RELEASE": 30,

    "T10Y2Y": 5,
    "T10Y2Y_SIGNAL": 5,
    "T10Y2Y_WEIGHTED": 5,
    "T10Y2Y_DAYS_SINCE_RELEASE": 30,

    "UNRATE": 10,
    "UNRATE_SIGNAL": 10,
    "UNRATE_WEIGHTED": 10,
    "UNRATE_DAYS_SINCE_RELEASE": 30,

    "GS10": 10,
    "GS2": 10,
    "CPI": 300,
    "VIX": 100,
    "DGS10": 10,
    "GDP": 20000,
    "PPI": 50,
    "PPI_SIGNAL": 50,
    "PPI_WEIGHTED": 50,
    "PPI_DAYS_SINCE_RELEASE": 30,
    "MANUFACTURING": 10000,
    "MANUFACTURING_SIGNAL": 10000,
    "MANUFACTURING_WEIGHTED": 10000,
    "MANUFACTURING_DAYS_SINCE_RELEASE": 30,
    "CONSUMER_SENTIMENT": 50,
    "CONSUMER_SENTIMENT_SIGNAL": 50,
    "CONSUMER_SENTIMENT_WEIGHTED": 50,
    "CONSUMER_SENTIMENT_DAYS_SINCE_RELEASE": 30,
    
    # Новand високопрandоритетнand andндикатори
    "CORE_PCE": 300,                 # Core PCE - схоже with CPI
    "CORE_PCE_SIGNAL": 300,
    "CORE_PCE_WEIGHTED": 300,
    "CORE_PCE_DAYS_SINCE_RELEASE": 30,
    
    "CONTINUING_CLAIMS": 50000,       # Continuing Claims - бandльшand values
    "CONTINUING_CLAIMS_SIGNAL": 50000,
    "CONTINUING_CLAIMS_WEIGHTED": 50000,
    "CONTINUING_CLAIMS_DAYS_SINCE_RELEASE": 7,  # Щотижnotвand данand
    
    # Новand середньопрandоритетнand andндикатори
    "TOTAL_PCE": 300,                # Total PCE
    "TOTAL_PCE_SIGNAL": 300,
    "TOTAL_PCE_WEIGHTED": 300,
    "TOTAL_PCE_DAYS_SINCE_RELEASE": 30,
    
    "FED_BALANCE": 1000000,           # Fed Balance - дуже великand values
    "FED_BALANCE_SIGNAL": 1000000,
    "FED_BALANCE_WEIGHTED": 1000000,
    "FED_BALANCE_DAYS_SINCE_RELEASE": 7,  # Щотижnotвand данand
    
    "MANUFACTURING_EMP": 5000,        # Manufacturing Employment
    "MANUFACTURING_EMP_SIGNAL": 5000,
    "MANUFACTURING_EMP_WEIGHTED": 5000,
    "MANUFACTURING_EMP_DAYS_SINCE_RELEASE": 30,
    
    # ВИПРАВЛЕНО: Додано відсутні індикатори з логів
    "HIGH_YIELD_SPREAD": 5,            # BAMLH0A0HYM2 - High Yield Spread
    "USD_EUR": 1.0,                   # DEXUSEU - USD/EUR Exchange Rate
    "WTI_OIL": 100,                   # DCOILWTICO - WTI Oil Price
    "USD_CNY": 1.0,                   # DEXCHUS - USD/CNY Exchange Rate
    "TOTAL_VEHICLE_SALES": 1000,       # TOTALSA - Total Vehicle Sales

    # Auto-derived scales (p90 abs) from latest FRED cache
    "INDUSTRIAL_PRODUCTION": 101.4861,
    "CAPACITY_UTIL": 78.5697,
    "NONFARM_PAYROLLS": 159441.6,
    "HOUSING_STARTS": 1698.4,
    "BUILDING_PERMITS": 1759.6,
    "TED_SPREAD": 0.17,
    "REAL_DISPOSABLE_INCOME": 18021.37,
    "RETAIL_SALES": 723971.8,
    
    # Додано SIGNAL/WEIGHTED/DAYS_SINCE_RELEASE варіанти
    "HIGH_YIELD_SPREAD_SIGNAL": 5,
    "HIGH_YIELD_SPREAD_WEIGHTED": 5,
    "HIGH_YIELD_SPREAD_DAYS_SINCE_RELEASE": 30,
    
    "USD_EUR_SIGNAL": 1.0,
    "USD_EUR_WEIGHTED": 1.0,
    "USD_EUR_DAYS_SINCE_RELEASE": 30,
    
    "WTI_OIL_SIGNAL": 100,
    "WTI_OIL_WEIGHTED": 100,
    "WTI_OIL_DAYS_SINCE_RELEASE": 30,
    
    "USD_CNY_SIGNAL": 1.0,
    "USD_CNY_WEIGHTED": 1.0,
    "USD_CNY_DAYS_SINCE_RELEASE": 30,
    
    "TOTAL_VEHICLE_SALES_SIGNAL": 1000,
    "TOTAL_VEHICLE_SALES_WEIGHTED": 1000,
    "TOTAL_VEHICLE_SALES_DAYS_SINCE_RELEASE": 30
}

# Decay lambdas by frequency
DECAY_LAMBDAS_BY_FREQ = {
    "daily": 0.001,
    "monthly": 0.01,
    "quarterly": 0.02,
}

# Macro data loading period (automatically 3 years back)
START_FINANCIAL = (datetime.today() - timedelta(days=3*365)).strftime("%Y-%m-%d")
END_FINANCIAL = datetime.today().strftime("%Y-%m-%d")

# Data paths
ENV = detect_environment()
PATHS = resolve_paths(ENV)

def build_macro_layers():
    """
    Splits macro indicators into layers by update frequency:
    - background: quarterly (long-term background)
    - trend: monthly (trend context)
    - signal: daily (operational signal)
    """
    layers = {"background": [], "trend": [], "signal": []}
    for fred_id, alias in FRED_ALIAS.items():
        freq = DATA_INTERVALS.get(fred_id, "monthly")
        if freq == "quarterly":
            layers["background"].append(alias)
        elif freq == "monthly":
            layers["trend"].append(alias)
        elif freq == "daily":
            layers["signal"].append(alias)
    return layers
