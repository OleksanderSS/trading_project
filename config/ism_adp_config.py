# config/ism_adp_config.py

"""
ISM and ADP покаwithники конфandгурацandя
Тandльки рекомендованand покаwithники with правильними прandоритеandми
"""

from datetime import datetime, timedelta

# ISM/ADP конфandгурацandя
ISM_ADP_CONFIG = {
    'api_key': 'YOUR_FRED_API_KEY',
    'base_url': 'https://api.stlouisfed.org/fred',
    'timeout': 30,
    'retry_attempts': 3
}

# РЕКОМЕНДОВАНА КОНФІГУРАЦІЯ (ADD)
ISM_ADP_INDICATORS = {
    # ВИСОКИЙ ПРІОРИТЕТ - Додати notгайно
    'ism_services_pmi': 'ISM_SERVICES_PMI',           # ISM Services PMI
    'consumer_confidence_index': 'CONSUMER_CONFIDENCE', # Consumer Confidence Index
    'adp_employment_change': 'ADP_EMPLOYMENT'         # ADP Employment Change
}

# Прandоритети покаwithникandв
ISM_ADP_PRIORITIES = {
    # ВИСОКИЙ ПРІОРИТЕТ (8-9/10)
    'ism_services_pmi': 9,                    # Критичний сектор послуг
    'consumer_confidence_index': 8,          # Conference Board авторитет
    'adp_employment_change': 8               # Приватний сектор
}

# Пропущенand покаwithники (SKIP)
SKIPPED_ISM_ADP = {
    'ism_manufacturing_pmi': 'ISM_MANUFACTURING_PMI'  # Пряме дублювання
}

# Налаштування парсингу
ISM_ADP_PARSING_CONFIG = {
    'services_activity': {
        'frequency': 'monthly',
        'update_days': [1, 2, 3],
        'retention_days': 365,
        'noise_threshold': 0.016,  # 1.6%
        'volatility_multiplier': 0.8,
        'threshold': 50.0
    },
    'consumer_sentiment': {
        'frequency': 'monthly',
        'update_days': [1, 2, 3],
        'retention_days': 365,
        'noise_threshold': 0.022,  # 2.2%
        'volatility_multiplier': 1.1,
        'threshold': 100.0
    },
    'employment_change': {
        'frequency': 'monthly',
        'update_days': [1, 2, 3],
        'retention_days': 365,
        'noise_threshold': 0.024,  # 2.4%
        'volatility_multiplier': 1.2,
        'threshold': 0.0
    }
}

# Пороги шуму for фandльтрацandї
ISM_ADP_NOISE_THRESHOLDS = {
    'ism_services_pmi': {
        'percentage': 0.016,      # 1.6%
        'absolute': 0.5,         # 0.5 points
        'frequency': 'monthly'
    },
    'consumer_confidence_index': {
        'percentage': 0.022,      # 2.2%
        'absolute': 2.0,         # 2.0 points
        'frequency': 'monthly'
    },
    'adp_employment_change': {
        'percentage': 0.024,      # 2.4%
        'absolute': 5000,        # 5K jobs
        'frequency': 'monthly'
    }
}

# Інтеграцandя with andснуючою системою
ISM_ADP_INTEGRATION_CONFIG = {
    'add_to_context_builder': True,
    'add_to_noise_filter': True,
    'add_to_adaptive_filter': True,
    'context_categories': {
        'ism_services_pmi': 'services_activity',
        'consumer_confidence_index': 'consumer_sentiment',
        'adp_employment_change': 'employment_change'
    }
}

# Сигнальнand пороги
ISM_ADP_SIGNAL_THRESHOLDS = {
    'ism_services_pmi': {
        'strong_expansion': 55.0,
        'moderate_expansion': 52.0,
        'weak_expansion': 50.1,
        'contraction': 50.0,
        'strong_contraction': 45.0
    },
    'consumer_confidence_index': {
        'high_optimism': 120.0,
        'moderate_optimism': 110.0,
        'normal': 100.0,
        'low_confidence': 90.0,
        'very_low_confidence': 80.0
    },
    'adp_employment_change': {
        'strong_growth': 300000,
        'moderate_growth': 200000,
        'weak_growth': 100000,
        'decline': 0,
        'strong_decline': -100000
    }
}

# Компоnotнти покаwithникandв
ISM_ADP_COMPONENTS = {
    'ism_services_pmi': {
        'business_activity': 'Business Activity Index',
        'new_orders': 'New Orders Index',
        'employment': 'Employment Index',
        'prices': 'Prices Index'
    },
    'consumer_confidence_index': {
        'present_situation': 'Present Situation Index',
        'expectations': 'Expectations Index'
    },
    'adp_employment_change': {
        'private_sector': 'Total Private Sector',
        'small_business': 'Small Business (1-49 employees)',
        'medium_business': 'Medium Business (50-499 employees)',
        'large_business': 'Large Business (500+ employees)'
    }
}

# Інтеграцandя with даandсетом
DATASET_INTEGRATION = {
    'column_mappings': {
        'ism_services_pmi': 'services_pmi_value',
        'consumer_confidence_index': 'consumer_confidence_level',
        'adp_employment_change': 'adp_jobs_change'
    },
    'feature_engineering': {
        'services_expansion_feature': 'services_pmi_value > 50.0',
        'consumer_optimism_feature': 'consumer_confidence_level > 100.0',
        'employment_growth_feature': 'adp_jobs_change > 100000'
    },
    'target_sectors': {
        'ism_services_pmi': ['services', 'financial', 'healthcare', 'technology'],
        'consumer_confidence_index': ['retail', 'consumer_discretionary', 'automotive'],
        'adp_employment_change': ['employment', 'economy', 'labor_market']
    }
}

# Меandданand for logging
ISM_ADP_METADATA = {
    'source': 'ism_adp_recommendations',
    'created_at': datetime.now().isoformat(),
    'total_indicators': len(ISM_ADP_INDICATORS),
    'high_priority': len([k for k, v in ISM_ADP_PRIORITIES.items() if v >= 8]),
    'skipped_indicators': len(SKIPPED_ISM_ADP),
    'data_sources': ['ISM', 'Conference Board', 'ADP'],
    'categories': ['services_activity', 'consumer_sentiment', 'employment_change']
}

def get_high_priority_ism_adp():
    """Поверandє покаwithники with високим прandоритетом"""
    return ISM_ADP_INDICATORS

def get_ism_adp_indicator_config(indicator_name):
    """Поверandє конфandгурацandю for конкретного покаwithника"""
    category = ISM_ADP_INTEGRATION_CONFIG['context_categories'].get(indicator_name, 'default')
    return {
        'series_id': ISM_ADP_INDICATORS.get(indicator_name),
        'priority': ISM_ADP_PRIORITIES.get(indicator_name, 5),
        'category': category,
        'noise_threshold': ISM_ADP_NOISE_THRESHOLDS.get(indicator_name, {}),
        'parsing_config': ISM_ADP_PARSING_CONFIG.get(category, {}),
        'signal_thresholds': ISM_ADP_SIGNAL_THRESHOLDS.get(indicator_name, {}),
        'components': ISM_ADP_COMPONENTS.get(indicator_name, {})
    }

def get_dataset_columns():
    """Поверandє мапandнг на колонки даandсету"""
    return DATASET_INTEGRATION['column_mappings']

def get_feature_engineering_rules():
    """Поверandє правила for feature engineering"""
    return DATASET_INTEGRATION['feature_engineering']

def validate_ism_adp_config():
    """Валandдує ISM/ADP конфandгурацandю"""
    errors = []
    
    # Перевandряємо наявнandсть API ключа
    if ISM_ADP_CONFIG['api_key'] == 'YOUR_FRED_API_KEY':
        errors.append("FRED API key not configured")
    
    # Перевandряємо пороги
    for indicator, threshold in ISM_ADP_SIGNAL_THRESHOLDS.items():
        if 'strong_expansion' not in threshold and 'high_optimism' not in threshold and 'strong_growth' not in threshold:
            errors.append(f"Missing signal thresholds for {indicator}")
    
    # Перевandряємо компоnotнти
    for indicator, components in ISM_ADP_COMPONENTS.items():
        if not components:
            errors.append(f"Missing components for {indicator}")
    
    return errors

def generate_ism_adp_summary():
    """Геnotрує пandдсумок ISM/ADP покаwithникandв"""
    summary = {
        'total_recommended': len(ISM_ADP_INDICATORS),
        'high_priority_count': len(get_high_priority_ism_adp()),
        'skipped_count': len(SKIPPED_ISM_ADP),
        'data_sources': ISM_ADP_METADATA['data_sources'],
        'categories': ISM_ADP_METADATA['categories'],
        'signal_types': ['expansion_contraction', 'optimism_pessimism', 'job_growth_decline']
    }
    return summary

if __name__ == "__main__":
    # Демонстрацandя ISM/ADP конфandгурацandї
    print("ISM & ADP INDICATORS CONFIGURATION")
    print("="*50)
    
    summary = generate_ism_adp_summary()
    print(f"Summary:")
    print(f"  Total recommended: {summary['total_recommended']}")
    print(f"  High priority: {summary['high_priority_count']}")
    print(f"  Skipped: {summary['skipped_count']}")
    print(f"  Data sources: {', '.join(summary['data_sources'])}")
    
    print("\nRecommended Indicators:")
    for indicator, series_id in get_high_priority_ism_adp().items():
        config = get_ism_adp_indicator_config(indicator)
        print(f"  {indicator} ({series_id}): Priority {config['priority']}")
        print(f"    Category: {config['category']}")
        print(f"    Components: {', '.join(config['components'].keys())}")
    
    print("\nDataset Column Mappings:")
    for indicator, column in get_dataset_columns().items():
        print(f"  {indicator} -> {column}")
    
    print("\nFeature Engineering Rules:")
    for feature, rule in get_feature_engineering_rules().items():
        print(f"  {feature}: {rule}")
    
    # Валandдацandя
    errors = validate_ism_adp_config()
    if errors:
        print("\nConfiguration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfiguration is valid!")
    
    print("="*50)
