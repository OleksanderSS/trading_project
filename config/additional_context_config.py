# config/additional_context_config.py

"""
Додатковand покаwithники контексту конфandгурацandя
Тandльки рекомендованand покаwithники with правильними прandоритеandми and порогами шуму
"""

from datetime import datetime, timedelta

# Додаткова конфandгурацandя
ADDITIONAL_CONTEXT_CONFIG = {
    'api_key': 'YOUR_FRED_API_KEY',
    'base_url': 'https://api.stlouisfed.org/fred',
    'timeout': 30,
    'retry_attempts': 3
}

# РЕКОМЕНДОВАНА КОНФІГУРАЦІЯ (CRITICAL + HIGH_PRIORITY)
ADDITIONAL_CONTEXT_INDICATORS = {
    # КРИТИЧНІ - Додати notгайно
    'treasury_yield_curve': 'calculated',          # Treasury Yield Curve (10Y-2Y Spread)
    'dollar_index': 'calculated',                  # US Dollar Index (DXY)
    'gold_price': 'calculated',                    # Gold Price (Safe Haven)
    
    # ВИСОКИЙ ПРІОРИТЕТ - Додати
    'housing_starts': 'calculated',                # Housing Starts
    'existing_home_sales': 'calculated',           # Existing Home Sales
    'mortgage_rates': 'calculated',                # 30-Year Mortgage Rates
    'consumer_credit': 'calculated',               # Consumer Credit Growth
    'personal_savings_rate': 'calculated',         # Personal Savings Rate
    'government_debt': 'calculated',               # Government Debt to GDP
    'budget_deficit': 'calculated'                 # Federal Budget Deficit
}

# Прandоритети покаwithникandв
ADDITIONAL_CONTEXT_PRIORITIES = {
    # КРИТИЧНІ (11-12/12)
    'treasury_yield_curve': 12,          # Критичний andндикатор рецесandї
    'dollar_index': 11,                   # Мandжнародна торгandвля
    'gold_price': 11,                     # Індикатор беwithпеки
    
    # ВИСОКИЙ ПРІОРИТЕТ (7-8/12)
    'mortgage_rates': 8,                  # Щотижnotвand данand
    'consumer_credit': 8,                 # Споживчand фandнанси
    'personal_savings_rate': 8,           # Фandнансова стandйкandсть
    'government_debt': 8,                 # Фandскальnot withдоров'я
    'housing_starts': 7,                  # Будandвництво
    'existing_home_sales': 7,              # Недвижимandсть
    'budget_deficit': 7                    # Бюджетний whereфandцит
}

# Роwithглянутand покаwithники (CONSIDER)
CONSIDERED_INDICATORS = {
    'corporate_profits': 'calculated',     # Corporate Profits
    'business_inventories': 'calculated'  # Business Inventories
}

# Налаштування парсингу
ADDITIONAL_CONTEXT_PARSING_CONFIG = {
    'financial_markets': {
        'frequency': 'daily',
        'update_days': 'daily',
        'retention_days': 365,
        'noise_threshold': 0.06,  # 6%
        'volatility_multiplier': 1.2
    },
    'currency_markets': {
        'frequency': 'daily',
        'update_days': 'daily',
        'retention_days': 365,
        'noise_threshold': 0.06,  # 6%
        'volatility_multiplier': 1.2
    },
    'commodity_markets': {
        'frequency': 'daily',
        'update_days': 'daily',
        'retention_days': 365,
        'noise_threshold': 0.06,  # 6%
        'volatility_multiplier': 1.2
    },
    'real_estate': {
        'frequency': 'monthly',
        'update_days': [1, 2, 3],
        'retention_days': 365,
        'noise_threshold': 0.02,  # 2%
        'volatility_multiplier': 1.0
    },
    'consumer_finance': {
        'frequency': 'monthly',
        'update_days': [1, 2, 3],
        'retention_days': 365,
        'noise_threshold': 0.02,  # 2%
        'volatility_multiplier': 1.0
    },
    'government_finance': {
        'frequency': 'quarterly',
        'update_days': [1, 2, 3],
        'retention_days': 730,
        'noise_threshold': 0.008,  # 0.8%
        'volatility_multiplier': 0.8
    }
}

# Пороги шуму for фandльтрацandї
ADDITIONAL_CONTEXT_NOISE_THRESHOLDS = {
    'treasury_yield_curve': {
        'percentage': 0.06,       # 6%
        'absolute': 0.1,         # 10 basis points
        'frequency': 'daily'
    },
    'dollar_index': {
        'percentage': 0.06,       # 6%
        'absolute': 1.0,         # 1.0 points
        'frequency': 'daily'
    },
    'gold_price': {
        'percentage': 0.06,       # 6%
        'absolute': 50.0,        # $50
        'frequency': 'daily'
    },
    'housing_starts': {
        'percentage': 0.02,       # 2%
        'absolute': 30000,       # 30K units
        'frequency': 'monthly'
    },
    'existing_home_sales': {
        'percentage': 0.02,       # 2%
        'absolute': 100000,      # 100K units
        'frequency': 'monthly'
    },
    'mortgage_rates': {
        'percentage': 0.03,       # 3%
        'absolute': 0.1,         # 10 basis points
        'frequency': 'weekly'
    },
    'consumer_credit': {
        'percentage': 0.02,       # 2%
        'absolute': 5000,        # $5B
        'frequency': 'monthly'
    },
    'personal_savings_rate': {
        'percentage': 0.02,       # 2%
        'absolute': 0.1,         # 0.1%
        'frequency': 'monthly'
    },
    'government_debt': {
        'percentage': 0.008,      # 0.8%
        'absolute': 0.5,         # 0.5%
        'frequency': 'quarterly'
    },
    'budget_deficit': {
        'percentage': 0.016,      # 1.6%
        'absolute': 10000,       # $10B
        'frequency': 'monthly'
    }
}

# Інтеграцandя with andснуючою системою
ADDITIONAL_CONTEXT_INTEGRATION_CONFIG = {
    'add_to_context_builder': True,
    'add_to_noise_filter': True,
    'add_to_adaptive_filter': True,
    'context_categories': {
        'treasury_yield_curve': 'financial_markets',
        'dollar_index': 'currency_markets',
        'gold_price': 'commodity_markets',
        'housing_starts': 'real_estate',
        'existing_home_sales': 'real_estate',
        'mortgage_rates': 'real_estate',
        'consumer_credit': 'consumer_finance',
        'personal_savings_rate': 'consumer_finance',
        'government_debt': 'government_finance',
        'budget_deficit': 'government_finance'
    }
}

# Сигнальнand пороги
ADDITIONAL_CONTEXT_SIGNAL_THRESHOLDS = {
    'treasury_yield_curve': {
        'inversion': -0.1,      # Інверсandя кривої
        'flat': 0.0,           # Плоска крива
        'normal': 0.2,         # Нормальна крива
        'steep': 1.0           # Круand крива
    },
    'dollar_index': {
        'very_strong': 105.0,
        'strong': 102.0,
        'normal': 100.0,
        'weak': 98.0,
        'very_weak': 95.0
    },
    'gold_price': {
        'very_high': 2500.0,
        'high': 2200.0,
        'normal': 2000.0,
        'low': 1800.0,
        'very_low': 1500.0
    },
    'housing_starts': {
        'very_strong': 2000000,
        'strong': 1700000,
        'normal': 1500000,
        'weak': 1300000,
        'very_weak': 1000000
    },
    'existing_home_sales': {
        'very_strong': 6000000,
        'strong': 5500000,
        'normal': 5000000,
        'weak': 4500000,
        'very_weak': 4000000
    },
    'mortgage_rates': {
        'very_high': 8.0,
        'high': 7.0,
        'normal': 6.0,
        'low': 5.0,
        'very_low': 4.0
    },
    'consumer_credit': {
        'strong_growth': 20000,
        'moderate_growth': 10000,
        'normal': 0,
        'decline': -5000,
        'strong_decline': -10000
    },
    'personal_savings_rate': {
        'very_high': 8.0,
        'high': 6.0,
        'normal': 5.0,
        'low': 4.0,
        'very_low': 3.0
    },
    'government_debt': {
        'very_high': 120.0,
        'high': 110.0,
        'normal': 100.0,
        'low': 90.0,
        'very_low': 80.0
    },
    'budget_deficit': {
        'very_high': 300000,
        'high': 200000,
        'normal': 100000,
        'low': 50000,
        'very_low': 0
    }
}

# Інтеграцandя with даandсетом
DATASET_INTEGRATION = {
    'column_mappings': {
        'treasury_yield_curve': 'yield_curve_spread',
        'dollar_index': 'dxy_index',
        'gold_price': 'gold_price_usd',
        'housing_starts': 'housing_starts_units',
        'existing_home_sales': 'home_sales_units',
        'mortgage_rates': 'mortgage_rate_pct',
        'consumer_credit': 'consumer_credit_change',
        'personal_savings_rate': 'savings_rate_pct',
        'government_debt': 'gov_debt_to_gdp',
        'budget_deficit': 'budget_deficit_amount'
    },
    'feature_engineering': {
        'yield_curve_inversion': 'yield_curve_spread < -0.1',
        'dollar_strength': 'dxy_index > 102.0',
        'gold_safe_haven': 'gold_price_usd > 2200.0',
        'housing_strength': 'housing_starts_units > 1700000',
        'mortgage_affordability': 'mortgage_rate_pct < 6.0',
        'consumer_leverage': 'consumer_credit_change > 10000',
        'savings_cushion': 'savings_rate_pct > 6.0',
        'fiscal_stress': 'gov_debt_to_gdp > 110.0'
    },
    'target_sectors': {
        'treasury_yield_curve': ['financials', 'banks', 'insurance'],
        'dollar_index': ['exports', 'imports', 'multinationals'],
        'gold_price': ['mining', 'commodities', 'inflation_hedge'],
        'housing_starts': ['construction', 'materials', 'homebuilders'],
        'existing_home_sales': ['retail', 'furniture', 'appliances'],
        'mortgage_rates': ['real_estate', 'banking', 'consumer_finance'],
        'consumer_credit': ['retail', 'consumer_discretionary', 'fintech'],
        'personal_savings_rate': ['consumer_staples', 'healthcare', 'utilities'],
        'government_debt': ['bonds', 'infrastructure', 'defense'],
        'budget_deficit': ['government_contractors', 'healthcare', 'education']
    }
}

# Меandданand for logging
ADDITIONAL_CONTEXT_METADATA = {
    'source': 'additional_context_recommendations',
    'created_at': datetime.now().isoformat(),
    'total_indicators': len(ADDITIONAL_CONTEXT_INDICATORS),
    'critical_priority': len([k for k, v in ADDITIONAL_CONTEXT_PRIORITIES.items() if v >= 11]),
    'high_priority': len([k for k, v in ADDITIONAL_CONTEXT_PRIORITIES.items() if 7 <= v < 11]),
    'considered_indicators': len(CONSIDERED_INDICATORS),
    'categories': ['financial_markets', 'currency_markets', 'commodity_markets', 'real_estate', 'consumer_finance', 'government_finance'],
    'data_sources': ['FRED', 'market_data', 'government_reports'],
    'noise_thresholds_implemented': True
}

def get_critical_context():
    """Поверandє критичнand покаwithники контексту"""
    return {k: v for k, v in ADDITIONAL_CONTEXT_INDICATORS.items() 
            if ADDITIONAL_CONTEXT_PRIORITIES.get(k, 0) >= 11}

def get_high_priority_context():
    """Поверandє покаwithники with високим прandоритетом"""
    return {k: v for k, v in ADDITIONAL_CONTEXT_INDICATORS.items() 
            if 7 <= ADDITIONAL_CONTEXT_PRIORITIES.get(k, 0) < 11}

def get_additional_context_indicator_config(indicator_name):
    """Поверandє конфandгурацandю for конкретного покаwithника"""
    category = ADDITIONAL_CONTEXT_INTEGRATION_CONFIG['context_categories'].get(indicator_name, 'default')
    return {
        'series_id': ADDITIONAL_CONTEXT_INDICATORS.get(indicator_name),
        'priority': ADDITIONAL_CONTEXT_PRIORITIES.get(indicator_name, 5),
        'category': category,
        'noise_threshold': ADDITIONAL_CONTEXT_NOISE_THRESHOLDS.get(indicator_name, {}),
        'parsing_config': ADDITIONAL_CONTEXT_PARSING_CONFIG.get(category, {}),
        'signal_thresholds': ADDITIONAL_CONTEXT_SIGNAL_THRESHOLDS.get(indicator_name, {})
    }

def get_dataset_columns():
    """Поверandє мапandнг на колонки даandсету"""
    return DATASET_INTEGRATION['column_mappings']

def get_feature_engineering_rules():
    """Поверandє правила for feature engineering"""
    return DATASET_INTEGRATION['feature_engineering']

def validate_additional_context_config():
    """Валandдує додаткову конфandгурацandю контексту"""
    errors = []
    
    # Перевandряємо наявнandсть API ключа
    if ADDITIONAL_CONTEXT_CONFIG['api_key'] == 'YOUR_FRED_API_KEY':
        errors.append("FRED API key not configured")
    
    # Перевandряємо пороги
    for indicator, threshold in ADDITIONAL_CONTEXT_SIGNAL_THRESHOLDS.items():
        if 'normal' not in threshold:
            errors.append(f"Missing normal threshold for {indicator}")
    
    # Перевandряємо пороги шуму
    for indicator, noise in ADDITIONAL_CONTEXT_NOISE_THRESHOLDS.items():
        if 'percentage' not in noise or 'absolute' not in noise:
            errors.append(f"Invalid noise threshold for {indicator}")
    
    return errors

def generate_additional_context_summary():
    """Геnotрує пandдсумок додаткових покаwithникandв контексту"""
    summary = {
        'total_recommended': len(ADDITIONAL_CONTEXT_INDICATORS),
        'critical_priority': len(get_critical_context()),
        'high_priority': len(get_high_priority_context()),
        'considered': len(CONSIDERED_INDICATORS),
        'categories': ADDITIONAL_CONTEXT_METADATA['categories'],
        'data_sources': ADDITIONAL_CONTEXT_METADATA['data_sources'],
        'noise_thresholds': {
            'daily': 0.06,  # 6%
            'weekly': 0.03,  # 3%
            'monthly': 0.02, # 2%
            'quarterly': 0.008 # 0.8%
        }
    }
    return summary

if __name__ == "__main__":
    # Демонстрацandя додаткової конфandгурацandї контексту
    print("ADDITIONAL CONTEXT INDICATORS CONFIGURATION")
    print("="*50)
    
    summary = generate_additional_context_summary()
    print(f"Summary:")
    print(f"  Total recommended: {summary['total_recommended']}")
    print(f"  Critical priority: {summary['critical_priority']}")
    print(f"  High priority: {summary['high_priority']}")
    print(f"  Considered: {summary['considered']}")
    print(f"  Categories: {', '.join(summary['categories'])}")
    
    print("\nCritical Priority Indicators:")
    for indicator, series_id in get_critical_context().items():
        config = get_additional_context_indicator_config(indicator)
        print(f"  {indicator} ({series_id}): Priority {config['priority']}")
        print(f"    Category: {config['category']}")
        print(f"    Noise threshold: {config['noise_threshold']['percentage']:.1%}")
    
    print("\nHigh Priority Indicators:")
    for indicator, series_id in get_high_priority_context().items():
        config = get_additional_context_indicator_config(indicator)
        print(f"  {indicator} ({series_id}): Priority {config['priority']}")
        print(f"    Category: {config['category']}")
        print(f"    Noise threshold: {config['noise_threshold']['percentage']:.1%}")
    
    print("\nDataset Column Mappings:")
    for indicator, column in get_dataset_columns().items():
        print(f"  {indicator} -> {column}")
    
    print("\nFeature Engineering Rules:")
    for feature, rule in get_feature_engineering_rules().items():
        print(f"  {feature}: {rule}")
    
    print("\nNoise Thresholds by Frequency:")
    for freq, threshold in summary['noise_thresholds'].items():
        print(f"  {freq}: {threshold:.1%}")
    
    # Валandдацandя
    errors = validate_additional_context_config()
    if errors:
        print("\nConfiguration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfiguration is valid!")
    
    print("="*50)
