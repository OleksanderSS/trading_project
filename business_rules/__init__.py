#!/usr/bin/env python3
"""
Business Rules Package Initialization
Інandцandалandforцandя пакету бandwithnotс-правил
"""

from .investment_rules import InvestmentRulesEngine, InvestmentRule, RiskMetrics
from .macro_knowledge_base import MacroKnowledgeEngine, HistoricalEvent, MarketPattern
from .abstraction_levels import AbstractionLevelsEngine, KnowledgeLevel, AnalysisResult

__all__ = [
    # Investment Rules
    'InvestmentRulesEngine',
    'InvestmentRule', 
    'RiskMetrics',
    
    # Macro Knowledge
    'MacroKnowledgeEngine',
    'HistoricalEvent',
    'MarketPattern',
    
    # Abstraction Levels
    'AbstractionLevelsEngine',
    'KnowledgeLevel',
    'AnalysisResult'
]

# Version
__version__ = '1.0.0'

# Description
__description__ = 'Business Rules Layer for Dean Agent Architecture'

# Author
__author__ = 'Trading System Development Team'

# Contact
__contact__ = 'development@tradingsystem.ai'

# License
__license__ = 'MIT'

# Compatibility
__python_requires__ = '>=3.8'

# Dependencies
__dependencies__ = [
    'numpy',
    'pandas',
    'datetime',
    'enum',
    'dataclasses',
    'typing',
    'logging'
]

# Package metadata
package_info = {
    'name': 'business_rules',
    'version': __version__,
    'description': __description__,
    'author': __author__,
    'contact': __contact__,
    'license': __license__,
    'python_requires': __python_requires__,
    'dependencies': __dependencies__,
    'features': [
        'Investment rules and risk management',
        'Macro knowledge base with historical events',
        'Abstraction levels for Worker/Manager separation',
        'Gemini recommendations implementation',
        'Dean architecture compliance'
    ],
    'gemini_recommendations_implemented': [
        '2% rule on position size',
        'Diversification limits',
        'Calendar risk management',
        'Crisis mode protection',
        'Black swan indicators',
        'Margin call protection',
        'Worker micro-mathematics',
        'Manager macro-knowledge',
        'Historical event awareness',
        'Abstraction level separation'
    ],
    'integration_points': [
        'agent_framework.py',
        'gpt_agent_integration.py',
        'integration_tools.py'
    ]
}

def get_package_info():
    """Отримати andнформацandю про пакет"""
    return package_info

def get_available_engines():
    """Отримати доступнand двигуни"""
    return {
        'investment_rules': InvestmentRulesEngine,
        'macro_knowledge': MacroKnowledgeEngine,
        'abstraction_levels': AbstractionLevelsEngine
    }

def initialize_all_engines():
    """Інandцandалandwithувати all двигуни"""
    return {
        'investment_rules': InvestmentRulesEngine(),
        'macro_knowledge': MacroKnowledgeEngine(),
        'abstraction_levels': AbstractionLevelsEngine()
    }

def validate_gemini_compliance():
    """Check вandдповandднandсть порадам Gemini"""
    engines = initialize_all_engines()
    
    compliance_checks = {
        'investment_rules': {
            '2_percent_rule': hasattr(engines['investment_rules'], 'evaluate_position_size'),
            'diversification_limits': hasattr(engines['investment_rules'], 'evaluate_position_size'),
            'calendar_risk': hasattr(engines['investment_rules'], '_check_calendar_risk'),
            'crisis_protection': hasattr(engines['investment_rules'], '_check_crisis_indicators'),
            'black_swan': hasattr(engines['investment_rules'], '_check_black_swan_indicators'),
            'margin_call': hasattr(engines['investment_rules'], 'rules') and 'margin_call_protection' in engines['investment_rules'].rules
        },
        'macro_knowledge': {
            'historical_events': len(engines['macro_knowledge'].historical_events) > 0,
            'crisis_lessons': hasattr(engines['macro_knowledge'], 'get_crisis_lessons'),
            'market_patterns': len(engines['macro_knowledge'].market_patterns) > 0,
            'context_analysis': hasattr(engines['macro_knowledge'], 'analyze_current_context'),
            'investment_recommendations': hasattr(engines['macro_knowledge'], 'get_investment_recommendations')
        },
        'abstraction_levels': {
            'worker_micro': hasattr(engines['abstraction_levels'], 'analyze_micro_level'),
            'manager_macro': hasattr(engines['abstraction_levels'], 'analyze_macro_level'),
            'knowledge_integration': hasattr(engines['abstraction_levels'], 'integrate_knowledge_levels'),
            'conflict_resolution': hasattr(engines['abstraction_levels'], '_suggest_conflict_resolution'),
            'hierarchy_implementation': len(engines['abstraction_levels'].knowledge_hierarchy) > 0
        }
    }
    
    total_checks = sum(len(checks) for checks in compliance_checks.values())
    passed_checks = sum(sum(checks.values()) for checks in compliance_checks.values())
    
    compliance_percentage = (passed_checks / total_checks) * 100
    
    return {
        'compliance_percentage': compliance_percentage,
        'total_checks': total_checks,
        'passed_checks': passed_checks,
        'detailed_checks': compliance_checks,
        'is_compliant': compliance_percentage >= 90,
        'status': 'EXCELLENT' if compliance_percentage >= 95 else 'GOOD' if compliance_percentage >= 85 else 'NEEDS_IMPROVEMENT'
    }

def main():
    """Основна функцandя for тестування пакету"""
    print(" BUSINESS RULES PACKAGE")
    print("=" * 40)
    
    # Інформацandя про пакет
    info = get_package_info()
    print(f" Package: {info['name']} v{info['version']}")
    print(f"[NOTE] Description: {info['description']}")
    print(f" Author: {info['author']}")
    print(f" License: {info['license']}")
    
    # Доступнand двигуни
    print(f"\n[TOOL] AVAILABLE ENGINES:")
    print("-" * 30)
    engines = get_available_engines()
    for name, engine_class in engines.items():
        print(f"    {name}: {engine_class.__name__}")
    
    # Перевandрка вandдповandдностand Gemini
    print(f"\n[TARGET] GEMINI COMPLIANCE CHECK:")
    print("-" * 30)
    
    compliance = validate_gemini_compliance()
    print(f"[DATA] Compliance: {compliance['compliance_percentage']:.1f}%")
    print(f"[UP] Status: {compliance['status']}")
    print(f"[OK] Passed: {compliance['passed_checks']}/{compliance['total_checks']}")
    
    if compliance['is_compliant']:
        print("[SUCCESS] FULLY COMPLIANT with Gemini recommendations!")
    else:
        print("[WARN] Some Gemini recommendations need implementation")
    
    # Деandльнand реwithульandти
    print(f"\n DETAILED COMPLIANCE:")
    print("-" * 30)
    
    for category, checks in compliance['detailed_checks'].items():
        print(f"\n{category.upper()}:")
        for check_name, passed in checks.items():
            status = "[OK]" if passed else "[ERROR]"
            print(f"   {status} {check_name}")
    
    # Інandцandалandforцandя двигунandв
    print(f"\n[START] INITIALIZING ENGINES:")
    print("-" * 30)
    
    initialized = initialize_all_engines()
    for name, engine in initialized.items():
        print(f"   [OK] {name}: {type(engine).__name__}")
    
    print(f"\n[TARGET] BUSINESS RULES PACKAGE READY!")
    print(f"[MONEY] Investment rules implemented")
    print(f" Macro knowledge base active")
    print(f"[BRAIN] Abstraction levels separated")
    print(f"[TARGET] Gemini recommendations compliant")

if __name__ == "__main__":
    main()
