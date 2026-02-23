# config/critical_signals_config.py

"""
Критичнand сигнали конфandгурацandя
Тandльки критичнand покаwithники with правильними прandоритеandми
"""

from datetime import datetime, timedelta

# Критична конфandгурацandя
CRITICAL_SIGNALS_CONFIG = {
    'api_key': 'YOUR_FRED_API_KEY',
    'base_url': 'https://api.stlouisfed.org/fred',
    'timeout': 30,
    'retry_attempts': 3
}

# КРИТИЧНІ СИГНАЛИ - Додати notгайно
CRITICAL_SIGNALS = {
    # КРИТИЧНІ (CRITICAL_ADD)
    'chicago_pmi_black_swan': 'calculated',           # Chicago PMI Black Swan Signal
    'labor_market_fragmentation': 'calculated',         # Labor Market Fragmentation
    'consumer_confidence_breakdown': 'calculated',    # Consumer Confidence Breakdown
    'shutdown_data_lag': 'monitored',                # Data Lag Problem
    
    # ВИСОКИЙ ПРІОРИТЕТ (HIGH_PRIORITY_ADD)
    'ppi_revision_trend': 'calculated'               # PPI Revision Trend
}

# Прandоритети критичних сигналandв
CRITICAL_SIGNALS_PRIORITIES = {
    # КРИТИЧНІ (14-15/15)
    'chicago_pmi_black_swan': 15,          # Black Swan - найвищий прandоритет
    'consumer_confidence_breakdown': 14,     # Psychological breakdown
    
    # КРИТИЧНІ (13/15)
    'labor_market_fragmentation': 13,        # Structural issues
    'shutdown_data_lag': 13,              # Systemic problem
    
    # ВИСОКИЙ ПРІОРИТЕТ (8-12/15)
    'ppi_revision_trend': 8                  # Trend validation
}

# Пороги for критичних сигналandв
CRITICAL_THRESHOLDS = {
    'chicago_pmi_black_swan': {
        'black_swan_level': 45.0,
        'warning_level': 48.0,
        'normal_range': [50.0, 60.0],
        'interpretation': {
            'below_45': 'BLACK_SWAN - Market crash warning',
            '45_48': 'CRITICAL - Severe contraction',
            '48_50': 'WARNING - Contraction risk',
            'above_50': 'NORMAL - Expansion'
        }
    },
    'labor_market_fragmentation': {
        'fragmentation_high': 0.8,
        'fragmentation_medium': 0.6,
        'fragmentation_low': 0.4,
        'interpretation': {
            'above_0.8': 'CRITICAL - High fragmentation',
            '0.6_0.8': 'WARNING - Medium fragmentation',
            '0.4_0.6': 'CAUTION - Low fragmentation',
            'below_0.4': 'NORMAL - Integrated market'
        }
    },
    'consumer_confidence_breakdown': {
        'breakdown_level': 85.0,
        'warning_level': 80.0,
        'normal_range': [70.0, 85.0],
        'interpretation': {
            'above_85': 'BREAKDOWN - Psychological shift',
            '80_85': 'WARNING - High confidence',
            '70_80': 'NORMAL - Healthy confidence',
            'below_70': 'CONCERN - Low confidence'
        }
    },
    'shutdown_data_lag': {
        'lag_threshold_days': 30,
        'severe_lag_days': 60,
        'interpretation': {
            'above_60': 'SEVERE - Major data delays',
            '30_60': 'WARNING - Significant delays',
            'below_30': 'NORMAL - Acceptable delays'
        }
    },
    'ppi_revision_trend': {
        'revision_up_threshold': 0.5,
        'revision_down_threshold': -0.5,
        'interpretation': {
            'above_0.5': 'INFLATIONARY - PPI revised up',
            'below_-0.5': 'DISINFLATIONARY - PPI revised down',
            '-0.5_0.5': 'STABLE - Minor revisions'
        }
    }
}

# Налаштування парсингу
CRITICAL_PARSING_CONFIG = {
    'critical_pmi': {
        'frequency': 'monthly',
        'update_days': [1, 2, 3],
        'retention_days': 365,
        'noise_threshold': 0.014,  # 1.4%
        'volatility_multiplier': 0.7,
        'alert_threshold': 45.0
    },
    'critical_labor': {
        'frequency': 'monthly',
        'update_days': [1, 2, 3],
        'retention_days': 365,
        'noise_threshold': 0.016,  # 1.6%
        'volatility_multiplier': 0.8,
        'alert_threshold': 0.8
    },
    'critical_sentiment': {
        'frequency': 'monthly',
        'update_days': [1, 2, 3],
        'retention_days': 365,
        'noise_threshold': 0.018,  # 1.8%
        'volatility_multiplier': 0.9,
        'alert_threshold': 85.0
    },
    'data_quality': {
        'frequency': 'irregular',
        'update_days': 'daily',
        'retention_days': 730,
        'noise_threshold': 0.005,  # 0.5%
        'volatility_multiplier': 0.5,
        'alert_threshold': 30  # days
    },
    'inflation_validation': {
        'frequency': 'monthly',
        'update_days': [1, 2, 3],
        'retention_days': 365,
        'noise_threshold': 0.016,  # 1.6%
        'volatility_multiplier': 0.8,
        'alert_threshold': 0.5
    }
}

# Пороги шуму for фandльтрацandї
CRITICAL_NOISE_THRESHOLDS = {
    'chicago_pmi_black_swan': {
        'percentage': 0.014,      # 1.4%
        'absolute': 0.5,         # 0.5 points
        'frequency': 'monthly'
    },
    'labor_market_fragmentation': {
        'percentage': 0.016,      # 1.6%
        'absolute': 0.05,        # 5%
        'frequency': 'monthly'
    },
    'consumer_confidence_breakdown': {
        'percentage': 0.018,      # 1.8%
        'absolute': 1.0,         # 1.0 points
        'frequency': 'monthly'
    },
    'shutdown_data_lag': {
        'percentage': 0.005,      # 0.5%
        'absolute': 1.0,         # 1 day
        'frequency': 'irregular'
    },
    'ppi_revision_trend': {
        'percentage': 0.016,      # 1.6%
        'absolute': 0.1,         # 0.1%
        'frequency': 'monthly'
    }
}

# Інтеграцandя with andснуючою системою
CRITICAL_INTEGRATION_CONFIG = {
    'add_to_context_builder': True,
    'add_to_noise_filter': True,
    'add_to_adaptive_filter': True,
    'context_categories': {
        'chicago_pmi_black_swan': 'critical_pmi',
        'labor_market_fragmentation': 'critical_labor',
        'consumer_confidence_breakdown': 'critical_sentiment',
        'shutdown_data_lag': 'data_quality',
        'ppi_revision_trend': 'inflation_validation'
    },
    'alert_system': {
        'enabled': True,
        'critical_threshold': 0.9,
        'high_threshold': 0.7,
        'medium_threshold': 0.5
    }
}

# Система рandшення problemsи лагу data
DATA_LAG_SOLUTION = {
    'problem': 'September data released in January',
    'solution': {
        'separate_pipelines': {
            'operational': {
                'purpose': 'Real-time trading signals',
                'data_sources': ['real_time_data', 'nowcasts', 'high_frequency'],
                'exclude': ['delayed_revisions', 'seasonal_adjustments_late'],
                'update_frequency': 'real_time'
            },
            'learning': {
                'purpose': 'Model training and validation',
                'data_sources': ['historical_revisions', 'complete_data', 'seasonal_patterns'],
                'include': ['delayed_revisions', 'final_revised_data'],
                'update_frequency': 'batch'
            }
        },
        'data_classification': {
            'real_time': {
                'tag': 'real_time',
                'use_case': 'operational_decisions',
                'delay_tolerance': 'hours'
            },
            'revised': {
                'tag': 'revised',
                'use_case': 'model_learning',
                'delay_tolerance': 'months'
            },
            'final': {
                'tag': 'final',
                'use_case': 'backtesting',
                'delay_tolerance': 'years'
            }
        },
        'validation_logic': {
            'trend_validation': 'Use revised data to confirm trends',
            'example': 'PPI up + PMI down = Stagflation confirmation',
            'operational_use': 'Real-time PMI for trading',
            'learning_use': 'Revised PPI for model training'
        }
    },
    'implementation': {
        'data_tagging': 'Tag all data with release_date and revision_status',
        'pipeline_routing': 'Route data to appropriate pipeline',
        'model_integration': 'Use different data for different model components'
    }
}

# Спецandальнand обчислюванand покаwithники
CRITICAL_CALCULATED_INDICATORS = {
    'chicago_pmi_black_swan': {
        'formula': 'chicago_pmi < 45.0',
        'components': ['chicago_pmi'],
        'description': 'Chicago PMI Black Swan Signal',
        'signal_interpretation': {
            'true': 'BLACK_SWAN_ALERT',
            'false': 'NORMAL_CONDITIONS'
        }
    },
    'labor_market_fragmentation': {
        'formula': '(skill_gap_index + wage_polarization_index + sector_mismatch_index) / 3',
        'components': ['skill_gap_index', 'wage_polarization_index', 'sector_mismatch_index'],
        'description': 'Labor Market Fragmentation Index',
        'signal_interpretation': {
            'above_0.8': 'HIGH_FRAGMENTATION',
            '0.6_0.8': 'MEDIUM_FRAGMENTATION',
            'below_0.6': 'LOW_FRAGMENTATION'
        }
    },
    'consumer_confidence_breakdown': {
        'formula': 'consumer_confidence > 85.0',
        'components': ['consumer_confidence'],
        'description': 'Consumer Confidence Breakdown Signal',
        'signal_interpretation': {
            'true': 'PSYCHOLOGICAL_BREAKDOWN',
            'false': 'NORMAL_SENTIMENT'
        }
    },
    'ppi_revision_trend': {
        'formula': 'ppi_revised - ppi_initial',
        'components': ['ppi_revised', 'ppi_initial'],
        'description': 'PPI Revision Trend',
        'signal_interpretation': {
            'above_0.5': 'INFLATIONARY_REVISION',
            'below_-0.5': 'DISINFLATIONARY_REVISION',
            '-0.5_0.5': 'STABLE_REVISION'
        }
    }
}

# Меandданand for logging
CRITICAL_METADATA = {
    'source': 'critical_signals_recommendations',
    'created_at': datetime.now().isoformat(),
    'total_indicators': len(CRITICAL_SIGNALS),
    'critical_priority': len([k for k, v in CRITICAL_SIGNALS_PRIORITIES.items() if v >= 13]),
    'high_priority': len([k for k, v in CRITICAL_SIGNALS_PRIORITIES.items() if 8 <= v < 13]),
    'calculated_indicators': len(CRITICAL_CALCULATED_INDICATORS),
    'alert_system_enabled': True,
    'data_lag_solution_implemented': True
}

def get_critical_signals():
    """Поверandє all критичнand сигнали"""
    return CRITICAL_SIGNALS

def get_high_priority_critical():
    """Поверandє сигнали with високим прandоритетом"""
    return {k: v for k, v in CRITICAL_SIGNALS.items() 
            if CRITICAL_SIGNALS_PRIORITIES.get(k, 0) >= 13}

def get_critical_indicator_config(indicator_name):
    """Поверandє конфandгурацandю for конкретного покаwithника"""
    category = CRITICAL_INTEGRATION_CONFIG['context_categories'].get(indicator_name, 'default')
    return {
        'series_id': CRITICAL_SIGNALS.get(indicator_name),
        'priority': CRITICAL_SIGNALS_PRIORITIES.get(indicator_name, 5),
        'category': category,
        'noise_threshold': CRITICAL_NOISE_THRESHOLDS.get(indicator_name, {}),
        'parsing_config': CRITICAL_PARSING_CONFIG.get(category, {}),
        'thresholds': CRITICAL_THRESHOLDS.get(indicator_name, {}),
        'calculated_info': CRITICAL_CALCULATED_INDICATORS.get(indicator_name, {})
    }

def get_data_lag_solution():
    """Поверandє рandшення for problemsи лагу data"""
    return DATA_LAG_SOLUTION

def validate_critical_config():
    """Валandдує критичну конфandгурацandю"""
    errors = []
    
    # Перевandряємо наявнandсть API ключа
    if CRITICAL_SIGNALS_CONFIG['api_key'] == 'YOUR_FRED_API_KEY':
        errors.append("FRED API key not configured")
    
    # Перевandряємо пороги
    for indicator, threshold in CRITICAL_THRESHOLDS.items():
        if 'interpretation' not in threshold:
            errors.append(f"Missing interpretation for {indicator}")
    
    # Перевandряємо обчислюванand покаwithники
    for indicator, calc_info in CRITICAL_CALCULATED_INDICATORS.items():
        if 'formula' not in calc_info:
            errors.append(f"Missing formula for {indicator}")
    
    return errors

def generate_critical_summary():
    """Геnotрує пandдсумок критичних сигналandв"""
    summary = {
        'total_critical': len(CRITICAL_SIGNALS),
        'critical_priority': len(get_high_priority_critical()),
        'high_priority': len([k for k, v in CRITICAL_SIGNALS_PRIORITIES.items() if 8 <= v < 13]),
        'calculated_indicators': len(CRITICAL_CALCULATED_INDICATORS),
        'categories': {
            'critical_pmi': 1,
            'critical_labor': 1,
            'critical_sentiment': 1,
            'data_quality': 1,
            'inflation_validation': 1
        },
        'signal_types': {
            'black_swan': 1,
            'fragmentation': 1,
            'psychological_breakdown': 1,
            'data_lag': 1,
            'trend_validation': 1
        }
    }
    return summary

def get_critical_signal_config(indicator: str) -> dict:
    """Отримати конфandгурацandю критичного сигналу"""
    return CRITICAL_SIGNALS.get(indicator, {})

# Прandоритети критичних сигналandв
CRITICAL_SIGNALS_PRIORITIES = {
    'critical': [
        'chicago_pmi_black_swan',
        'labor_market_fragmentation',
        'consumer_confidence_breakdown',
        'shutdown_data_lag'
    ],
    'high': [
        'ppi_revision_trend'
    ]
}

if __name__ == "__main__":
    # Демонстрацandя критичної конфandгурацandї
    print("CRITICAL SIGNALS CONFIGURATION")
    print("="*50)
    
    summary = generate_critical_summary()
    print(f"Summary:")
    print(f"  Total critical signals: {summary['total_critical']}")
    print(f"  Critical priority: {summary['critical_priority']}")
    print(f"  High priority: {summary['high_priority']}")
    print(f"  Calculated indicators: {summary['calculated_indicators']}")
    
    print("\nCritical Priority Signals:")
    for indicator, series_id in get_high_priority_critical().items():
        config = get_critical_indicator_config(indicator)
        print(f"  {indicator} ({series_id}): Priority {config['priority']}")
        print(f"    Formula: {config['calculated_info'].get('formula', 'N/A')}")
    
    print("\nData Lag Solution:")
    solution = get_data_lag_solution()
    print(f"  Problem: {solution['problem']}")
    print(f"  Solution: Separate operational vs learning pipelines")
    
    # Валandдацandя
    errors = validate_critical_config()
    if errors:
        print("\nConfiguration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfiguration is valid!")
    
    print("="*50)
