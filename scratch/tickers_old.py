"""
Centralized Tickers Configuration Module
ðäð┤ð©not ð┤ðÂðÁÐÇðÁð╗ð¥ ð┐ÐÇð░ð▓ð┤ð© for allÐà Ðéandð║ðÁÐÇandð▓ ð┐ÐÇð¥ðÁð║ÐéÐâ
"""

from typing import Dict, List, Set
import json
from pathlib import Path

# --- ð×Ðüð¢ð¥ð▓ð¢and ð║ð░ÐéðÁð│ð¥ÐÇandÐù Ðéandð║ðÁÐÇandð▓ ---

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

# --- ðÜð¥ð╝ð▒andð¢ð¥ð▓ð░ð¢and Ðüð┐ð©Ðüð║ð© ---

# ðÆÐüand Ðéandð║ðÁÐÇð© (119 ÐêÐéÐâð║)
ALL_TICKERS = (
    ETF_TICKERS + TECH_GIANTS + ADDITIONAL_TECH + FINANCE_TICKERS +
    HEALTHCARE_TICKERS + ENERGY_TICKERS + CONSUMER_TICKERS +
    INDUSTRIAL_TICKERS + MATERIALS_TICKERS + UTILITIES_TICKERS +
    REAL_ESTATE_TICKERS + COMMUNICATION_TICKERS + INTERNATIONAL_TICKERS +
    CRYPTO_TICKERS
)

# ð×Ðüð¢ð¥ð▓ð¢and Ðéandð║ðÁÐÇð© (ÐÇð¥withÐêð©ÐÇðÁð¢ð¥ for ð║ÐÇð░Ðëð¥ð│ð¥ ð┐ÐÇðÁð┤Ðüandð▓ð╗ðÁð¢ð¢ÐÅ)
CORE_TICKERS = [
    # ETF - ð¥Ðüð¢ð¥ð▓ð¢and ÐÇð©ð¢ð║ð¥ð▓and andð¢whereð║Ðüð©
    "SPY", "QQQ", "IWM", "DIA",
    
    # Tech Giants - ð▓ðÁð╗ð©ð║and ÐéðÁÐàð¢ð¥ð╗ð¥ð│andÐçð¢and ð║ð¥ð╝ð┐ð░ð¢andÐù  
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    
    # Finance - Ðäandð¢ð░ð¢Ðüð¥ð▓ð©ð╣ ÐüðÁð║Ðéð¥ÐÇ
    "JPM", "BAC", "WFC", "GS",
    
    # Healthcare - ð¥Ðàð¥ÐÇð¥ð¢ð░ withð┤ð¥ÐÇð¥ð▓'ÐÅ
    "JNJ", "PFE", "UNH", "ABBV",
    
    # Consumer - Ðüð┐ð¥ðÂð©ð▓Ðçð©ð╣ ÐüðÁð║Ðéð¥ÐÇ
    "PG", "KO", "HD", "MCD", "WMT", "COST",
    
    # Energy - ðÁnotÐÇð│ðÁÐéð©ð║ð░
    "XOM", "CVX", "COP",
    
    # Industrial - ð┐ÐÇð¥ð╝ð©Ðüð╗ð¥ð▓andÐüÐéÐî
    "CAT", "DE", "BA", "GE"
]

# Tech Ðéandð║ðÁÐÇð©
TECH_TICKERS = TECH_GIANTS + ADDITIONAL_TECH

# S&P 500 Ðéandð║ðÁÐÇð© (ð▓ð©ð▒andÐÇð║ð¥ð▓ð¥)
SP500_TICKERS = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "JNJ", "PG", "KO", "HD", "UNH", "VZ", "MA", "DIS", "NFLX"
]

# --- ðíð╗ð¥ð▓ð¢ð©ð║ð© for ÐüÐâð╝andÐüð¢ð¥ÐüÐéand ---

# ðÆðÿðƒðáðÉðÆðøðòðØð×: ð×Ðüð¢ð¥ð▓ð¢ð©ð╣ Ðüð╗ð¥ð▓ð¢ð©ð║ ÐéðÁð┐ðÁÐÇ ð▓ð©ð║ð¥ÐÇð©ÐüÐéð¥ð▓ÐâÐö ALL_TICKERS ð┤ð╗ÐÅ ð┐ð¥ð▓ð¢ð¥ð│ð¥ ð┐ð¥ð║ÐÇð©ÐéÐéÐÅ
TICKERS = {ticker: ticker for ticker in ALL_TICKERS}

# ðíð╗ð¥ð▓ð¢ð©ð║ ð┤ð╗ÐÅ CORE_TICKERS (ÐÅð║Ðëð¥ ð┐ð¥ÐéÐÇÐûð▒ðÁð¢ ð╝ðÁð¢Ðêð©ð╣ ð¢ð░ð▒ÐûÐÇ)
CORE_TICKERS_DICT = {ticker: ticker for ticker in CORE_TICKERS}

# ðƒð¥ð▓ð¢ð©ð╣ Ðüð╗ð¥ð▓ð¢ð©ð║ (ðÀð░ð╗ð©Ðêð░Ðöð╝ð¥ ð┤ð╗ÐÅ ÐüÐâð╝andÐüð¢ð¥ÐüÐéand)
ALL_TICKERS_DICT = {ticker: ticker for ticker in ALL_TICKERS}

# --- ðñÐâð¢ð║ÐåandÐù for ð¥ÐéÐÇð©ð╝ð░ð¢ð¢ÐÅ Ðüð┐ð©Ðüð║andð▓ ---

def get_tickers(category: str = "core") -> List[str]:
    """
    ð×ÐéÐÇð©ð╝ð░Ðéð© Ðüð┐ð©Ðüð¥ð║ Ðéandð║ðÁÐÇandð▓ for ð║ð░ÐéðÁð│ð¥ÐÇandÐöÐÄ
    
    Args:
        category: ðÜð░ÐéðÁð│ð¥ÐÇandÐÅ Ðéandð║ðÁÐÇandð▓
        - "core": ð¥Ðüð¢ð¥ð▓ð¢and 4 Ðéandð║ðÁÐÇð©
        - "all": all 119 Ðéandð║ðÁÐÇandð▓
        - "etf": ETFs
        - "tech": tech giants + additional tech
        - "sp500": ð▓ð©ð▒andÐÇð║ð¥ð▓ð¥ with S&P 500
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
        List[str]: ðíð┐ð©Ðüð¥ð║ Ðéandð║ðÁÐÇandð▓
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
    ð×ÐéÐÇð©ð╝ð░Ðéð© Ðüð╗ð¥ð▓ð¢ð©ð║ Ðéandð║ðÁÐÇandð▓ for ð║ð░ÐéðÁð│ð¥ÐÇandÐöÐÄ
    
    Args:
        category: ðÜð░ÐéðÁð│ð¥ÐÇandÐÅ Ðéandð║ðÁÐÇandð▓ (ð┤ð©ð▓. get_tickers)
    
    Returns:
        Dict[str, str]: ðíð╗ð¥ð▓ð¢ð©ð║ Ðéandð║ðÁÐÇandð▓
    """
    tickers = get_tickers(category)
    return {ticker: ticker for ticker in tickers}

def get_ticker_categories(ticker: str) -> List[str]:
    """
    ð×ÐéÐÇð©ð╝ð░Ðéð© ð║ð░ÐéðÁð│ð¥ÐÇandÐù for ð║ð¥ð¢ð║ÐÇðÁÐéð¢ð¥ð│ð¥ Ðéandð║ðÁÐÇð░
    
    Args:
        ticker: ðíð©ð╝ð▓ð¥ð╗ Ðéandð║ðÁÐÇð░
    
    Returns:
        List[str]: ðíð┐ð©Ðüð¥ð║ ð║ð░ÐéðÁð│ð¥ÐÇandð╣
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
    ð×ÐéÐÇð©ð╝ð░Ðéð© ÐüandÐéð©ÐüÐéð©ð║Ðâ ð┐ð¥ ð║ð░ÐéðÁð│ð¥ÐÇandÐÅÐà
    
    Returns:
        Dict[str, int]: ðíð╗ð¥ð▓ð¢ð©ð║ with ð║andð╗Ðîð║andÐüÐéÐÄ Ðéandð║ðÁÐÇandð▓ ð┐ð¥ ð║ð░ÐéðÁð│ð¥ÐÇandÐÅÐà
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
    ðÆð░ð╗andð┤ð░ÐåandÐÅ Ðüð┐ð©Ðüð║Ðâ Ðéandð║ðÁÐÇandð▓
    
    Args:
        tickers: ðíð┐ð©Ðüð¥ð║ Ðéandð║ðÁÐÇandð▓ for ð▓ð░ð╗andð┤ð░ÐåandÐù
    
    Returns:
        Dict[str, List[str]]: {
            "valid": Ðüð┐ð©Ðüð¥ð║ ð▓ð░ð╗andð┤ð¢ð©Ðà Ðéandð║ðÁÐÇandð▓,
            "invalid": Ðüð┐ð©Ðüð¥ð║ notð▓ð░ð╗andð┤ð¢ð©Ðà Ðéandð║ðÁÐÇandð▓
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
    ðòð║Ðüð┐ð¥ÐÇÐéÐâð▓ð░Ðéð© all Ðéandð║ðÁÐÇð© ð▓ JSON file
    
    Args:
        filepath: ð¿ð╗ÐÅÐà ð┤ð¥ fileÐâ
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
    # ðóðÁÐüÐéÐâð▓ð░ð¢ð¢ÐÅ ð╝ð¥ð┤Ðâð╗ÐÅ
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
