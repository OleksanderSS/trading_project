"""
Centralized Tickers Configuration Module
Єдиnot джерело правди for allх тandкерandв проекту
"""

from typing import Dict, List, Set
import json
from pathlib import Path

# --- Основнand категорandї тandкерandв ---

# ETFs (Exchange Traded Funds)
ETF_TICKERS = [
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "DIA",   # Dow Jones
    "IWM",   # Russell 2000
    "VTI",   # Total Stock Market
    "GLD",   # Gold
    "SLV",   # Silver
    "TLT",   # 20+ Year Treasury
    "XLF",   # Financial Sector
    "XLE",   # Energy Sector
    "XLI",   # Industrial Sector
    "XLU",   # Utilities Sector
    "XLK",   # Technology Sector
    "XLV",   # Healthcare Sector
    "XME",   # Metals and Mining
]

# Tech Giants
TECH_GIANTS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "META",  # Meta
    "NVDA",  # NVIDIA
    "TSLA",  # Tesla
]

# Additional Tech
ADDITIONAL_TECH = [
    "AMD",   # AMD
    "INTC",  # Intel
    "CSCO",  # Cisco
    "IBM",   # IBM
    "ORCL",  # Oracle
    "CRM",   # Salesforce
    "ADBE",  # Adobe
    "NFLX",  # Netflix
]

# Finance Sector
FINANCE_TICKERS = [
    "JPM",   # JPMorgan Chase
    "BAC",   # Bank of America
    "WFC",   # Wells Fargo
    "GS",    # Goldman Sachs
    "MS",    # Morgan Stanley
    "C",     # Citigroup
    "COF",   # Capital One
    "AXP",   # American Express
    "BLK",   # BlackRock
]

# Healthcare Sector
HEALTHCARE_TICKERS = [
    "JNJ",   # Johnson & Johnson
    "PFE",   # Pfizer
    "UNH",   # UnitedHealth
    "ABT",   # Abbott
    "MRK",   # Merck
    "LLY",   # Eli Lilly
    "BMY",   # Bristol Myers Squibb
    "AMGN",  # Amgen
    "GILD",  # Gilead Sciences
]

# Energy Sector
ENERGY_TICKERS = [
    "XOM",   # ExxonMobil
    "CVX",   # Chevron
    "COP",   # ConocoPhillips
    "SHEL",  # Shell
    "BP",    # BP
    "TOT",   # TotalEnergies
    "ENB",   # Enbridge
    "EQNR",  # Equinor
]

# Consumer Sector
CONSUMER_TICKERS = [
    "PG",    # Procter & Gamble
    "KO",    # Coca-Cola
    "PEP",   # PepsiCo
    "WMT",   # Walmart
    "HD",    # Home Depot
    "MCD",   # McDonald's
    "NKE",   # Nike
    "SBUX",  # Starbucks
]

# Industrial Sector
INDUSTRIAL_TICKERS = [
    "GE",    # General Electric
    "MMM",   # 3M
    "HON",   # Honeywell
    "CAT",   # Caterpillar
    "DE",    # Deere & Co
    "UPS",   # UPS
    "RTX",   # Raytheon
    "BA",    # Boeing
]

# Materials Sector
MATERIALS_TICKERS = [
    "DD",    # DuPont
    "DOW",   # Dow
    "LIN",   # Linde
    "ECL",   # Ecolab
    "APD",   # Air Products
    "NEM",   # Newmont Mining
    "FCX",   # Freeport-McMoRan
    "BHP",   # BHP Group
]

# Utilities Sector
UTILITIES_TICKERS = [
    "NEE",   # NextEra Energy
    "DUK",   # Duke Energy
    "SO",    # Southern Company
    "AEP",   # American Electric Power
    "EXC",   # Exelon
    "SRE",   # Sempra Energy
    "ED",    # Consolidated Edison
    "PEG",   # Public Service Enterprise
]

# Real Estate Sector
REAL_ESTATE_TICKERS = [
    "AMT",   # American Tower
    "PLD",   # Prologis
    "EQIX",  # Equinix
    "PSA",   # Public Storage
    "CBRE",  # CBRE Group
    "WELL",  # Welltower
    "VTR",   # Ventas
    "AVB",   # AvalonBay Communities
]

# Communication Sector
COMMUNICATION_TICKERS = [
    "VZ",    # Verizon
    "T",     # AT&T
    "TMUS",  # T-Mobile
    "CMCSA", # Comcast
    "CHTR",  # Charter Communications
    "DIS",   # Disney
    "FOXA",  # Fox Corporation
]

# International Stocks
INTERNATIONAL_TICKERS = [
    "BABA",  # Alibaba
    "BIDU",  # Baidu
    "JD",    # JD.com
    "PDD",   # PDD Holdings
    "NIO",   # NIO
    "XPEV",  # XPeng
    "LI",    # Li Auto
    "BILI",  # Bilibili
]

# Crypto-related Stocks
CRYPTO_TICKERS = [
    "COIN",  # Coinbase
    "MARA",  # Marathon Digital
    "RIOT",  # Riot Platforms
    "SQ",    # Block (Square)
    "PYPL",  # PayPal
    "BLOCK", # Block
    "GBTC",  # Grayscale Bitcoin Trust
    "EBAY",  # eBay
]

# --- Комбandнованand списки ---

# Всand тandкери (119 штук)
ALL_TICKERS = (
    ETF_TICKERS + TECH_GIANTS + ADDITIONAL_TECH + FINANCE_TICKERS +
    HEALTHCARE_TICKERS + ENERGY_TICKERS + CONSUMER_TICKERS +
    INDUSTRIAL_TICKERS + MATERIALS_TICKERS + UTILITIES_TICKERS +
    REAL_ESTATE_TICKERS + COMMUNICATION_TICKERS + INTERNATIONAL_TICKERS +
    CRYPTO_TICKERS
)

# Основнand тandкери (роwithширено for кращого предсandвлення)
CORE_TICKERS = [
    # ETF - основнand ринковand andнwhereкси
    "SPY", "QQQ", "IWM", "DIA",
    
    # Tech Giants - великand технологandчнand компанandї  
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    
    # Finance - фandнансовий сектор
    "JPM", "BAC", "WFC", "GS",
    
    # Healthcare - охорона withдоров'я
    "JNJ", "PFE", "UNH", "ABBV",
    
    # Consumer - споживчий сектор
    "PG", "KO", "HD", "MCD", "WMT", "COST",
    
    # Energy - еnotргетика
    "XOM", "CVX", "COP",
    
    # Industrial - промисловandсть
    "CAT", "DE", "BA", "GE"
]

# Tech тandкери
TECH_TICKERS = TECH_GIANTS + ADDITIONAL_TECH

# S&P 500 тandкери (вибandрково)
SP500_TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "JNJ", "PG", "KO", "HD", "UNH", "VZ", "MA", "DIS", "NFLX"
]

# --- Словники for сумandсностand ---

# ВИПРАВЛЕНО: Основний словник тепер використовує ALL_TICKERS для повного покриття
TICKERS = {ticker: ticker for ticker in ALL_TICKERS}

# Словник для CORE_TICKERS (якщо потрібен менший набір)
CORE_TICKERS_DICT = {ticker: ticker for ticker in CORE_TICKERS}

# Повний словник (залишаємо для сумandсностand)
ALL_TICKERS_DICT = {ticker: ticker for ticker in ALL_TICKERS}

# --- Функцandї for отримання спискandв ---

def get_tickers(category: str = "core") -> List[str]:
    """
    Отримати список тandкерandв for категорandєю
    
    Args:
        category: Категорandя тandкерandв
        - "core": основнand 4 тandкери
        - "all": all 119 тandкерandв
        - "etf": ETFs
        - "tech": tech giants + additional tech
        - "sp500": вибandрково with S&P 500
        - "finance": finance sector
        - "healthcare": healthcare sector
        - "energy": energy sector
        - "consumer": consumer sector
        - "industrial": industrial sector
        - "materials": materials sector
        - "utilities": utilities sector
        - "realestate": real estate sector
        - "communication": communication sector
        - "international": international stocks
        - "crypto": crypto-related stocks
    
    Returns:
        List[str]: Список тandкерandв
    """
    category_map = {
        "core": CORE_TICKERS,
        "all": ALL_TICKERS,
        "etf": ETF_TICKERS,
        "tech": TECH_TICKERS,
        "sp500": SP500_TICKERS,
        "finance": FINANCE_TICKERS,
        "healthcare": HEALTHCARE_TICKERS,
        "energy": ENERGY_TICKERS,
        "consumer": CONSUMER_TICKERS,
        "industrial": INDUSTRIAL_TICKERS,
        "materials": MATERIALS_TICKERS,
        "utilities": UTILITIES_TICKERS,
        "realestate": REAL_ESTATE_TICKERS,
        "communication": COMMUNICATION_TICKERS,
        "international": INTERNATIONAL_TICKERS,
        "crypto": CRYPTO_TICKERS,
    }
    
    return category_map.get(category.lower(), CORE_TICKERS)

def get_tickers_dict(category: str = "core") -> Dict[str, str]:
    """
    Отримати словник тandкерandв for категорandєю
    
    Args:
        category: Категорandя тandкерandв (див. get_tickers)
    
    Returns:
        Dict[str, str]: Словник тandкерandв
    """
    tickers = get_tickers(category)
    return {ticker: ticker for ticker in tickers}

def get_ticker_categories(ticker: str) -> List[str]:
    """
    Отримати категорandї for конкретного тandкера
    
    Args:
        ticker: Символ тandкера
    
    Returns:
        List[str]: Список категорandй
    """
    categories = []
    
    if ticker in CORE_TICKERS:
        categories.append("core")
    if ticker in ETF_TICKERS:
        categories.append("etf")
    if ticker in TECH_TICKERS:
        categories.append("tech")
    if ticker in SP500_TICKERS:
        categories.append("sp500")
    if ticker in FINANCE_TICKERS:
        categories.append("finance")
    if ticker in HEALTHCARE_TICKERS:
        categories.append("healthcare")
    if ticker in ENERGY_TICKERS:
        categories.append("energy")
    if ticker in CONSUMER_TICKERS:
        categories.append("consumer")
    if ticker in INDUSTRIAL_TICKERS:
        categories.append("industrial")
    if ticker in MATERIALS_TICKERS:
        categories.append("materials")
    if ticker in UTILITIES_TICKERS:
        categories.append("utilities")
    if ticker in REAL_ESTATE_TICKERS:
        categories.append("realestate")
    if ticker in COMMUNICATION_TICKERS:
        categories.append("communication")
    if ticker in INTERNATIONAL_TICKERS:
        categories.append("international")
    if ticker in CRYPTO_TICKERS:
        categories.append("crypto")
    
    return categories

def get_category_stats() -> Dict[str, int]:
    """
    Отримати сandтистику по категорandях
    
    Returns:
        Dict[str, int]: Словник with кandлькandстю тandкерandв по категорandях
    """
    return {
        "core": len(CORE_TICKERS),
        "all": len(ALL_TICKERS),
        "etf": len(ETF_TICKERS),
        "tech": len(TECH_TICKERS),
        "sp500": len(SP500_TICKERS),
        "finance": len(FINANCE_TICKERS),
        "healthcare": len(HEALTHCARE_TICKERS),
        "energy": len(ENERGY_TICKERS),
        "consumer": len(CONSUMER_TICKERS),
        "industrial": len(INDUSTRIAL_TICKERS),
        "materials": len(MATERIALS_TICKERS),
        "utilities": len(UTILITIES_TICKERS),
        "realestate": len(REAL_ESTATE_TICKERS),
        "communication": len(COMMUNICATION_TICKERS),
        "international": len(INTERNATIONAL_TICKERS),
        "crypto": len(CRYPTO_TICKERS),
    }

def validate_tickers(tickers: List[str]) -> Dict[str, List[str]]:
    """
    Валandдацandя списку тandкерandв
    
    Args:
        tickers: Список тandкерandв for валandдацandї
    
    Returns:
        Dict[str, List[str]]: {
            "valid": список валandдних тandкерandв,
            "invalid": список notвалandдних тandкерandв
        }
    """
    all_valid = set(ALL_TICKERS)
    valid = [t for t in tickers if t in all_valid]
    invalid = [t for t in tickers if t not in all_valid]
    
    return {
        "valid": valid,
        "invalid": invalid
    }

def export_tickers_to_json(filepath: str = "config/tickers_export.json"):
    """
    Експортувати all тandкери в JSON file
    
    Args:
        filepath: Шлях до fileу
    """
    export_data = {
        "categories": {
            "core": CORE_TICKERS,
            "etf": ETF_TICKERS,
            "tech_giants": TECH_GIANTS,
            "additional_tech": ADDITIONAL_TECH,
            "finance": FINANCE_TICKERS,
            "healthcare": HEALTHCARE_TICKERS,
            "energy": ENERGY_TICKERS,
            "consumer": CONSUMER_TICKERS,
            "industrial": INDUSTRIAL_TICKERS,
            "materials": MATERIALS_TICKERS,
            "utilities": UTILITIES_TICKERS,
            "realestate": REAL_ESTATE_TICKERS,
            "communication": COMMUNICATION_TICKERS,
            "international": INTERNATIONAL_TICKERS,
            "crypto": CRYPTO_TICKERS,
        },
        "combined": {
            "all": ALL_TICKERS,
            "core": CORE_TICKERS,
            "tech": TECH_TICKERS,
            "sp500": SP500_TICKERS,
        },
        "stats": get_category_stats()
    }
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Tickers exported to {filepath}")

if __name__ == "__main__":
    # Тестування модуля
    print("=== Tickers Module Test ===")
    print(f"Total tickers: {len(ALL_TICKERS)}")
    print(f"Core tickers: {len(CORE_TICKERS)}")
    print(f"Tech tickers: {len(TECH_TICKERS)}")
    print(f"ETF tickers: {len(ETF_TICKERS)}")
    
    print("\n=== Category Stats ===")
    stats = get_category_stats()
    for category, count in stats.items():
        print(f"{category}: {count}")
    
    print("\n=== Export Test ===")
    export_tickers_to_json()
    
    print("\n=== Validation Test ===")
    test_tickers = ["SPY", "QQQ", "INVALID1", "INVALID2"]
    validation = validate_tickers(test_tickers)
    print(f"Valid: {validation['valid']}")
    print(f"Invalid: {validation['invalid']}")
