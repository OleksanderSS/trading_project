# core/analysis/indicators_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class IndicatorsAnalyzer:
    """
    Аналandwith роwithширених покаwithникandв на дублювання and notобхandднandсть
    """
    
    def __init__(self):
        # Поточнand покаwithники в системand
        self.current_indicators = {
            # Ринок працand
            'unemployment', 'unemployment_rate', 'labor_force', 'employment',
            'payrolls', 'nonfarm_payrolls', 'job_openings', 'hiring_rate',
            'quit_rate', 'layoff_rate', 'labor_quality', 'labor_quantity',
            
            # Секторнand покаwithники
            'healthcare_sector', 'education_sector', 'leisure_sector',
            'hospitality_sector', 'manufacturing_sector', 'retail_sector',
            'cyclical_sectors', 'defensive_sectors', 'sector_payrolls',
            
            # Банкandвськand покаwithники
            'bank_reserves', 'bank_assets', 'reserves_to_assets_ratio',
            'repo_operations', 'reverse_repo', 'rrp', 'fed_balance_sheet',
            'monetary_base', 'central_bank_assets',
            
            # Сентимент and поwithицandонування
            'consumer_sentiment', 'michigan_sentiment', 'consumer_expectations',
            'investor_positioning', 'margin_debt', 'leverage_ratio',
            'debt_to_equity', 'margin_calls', 'trading_volume',
            'sentiment_index'
        }
        
        # Новand роwithширенand покаwithники
        self.new_advanced_indicators = {
            'jolts': {
                'series_id': 'JTSJOL',  # Job Openings and Labor Turnover Survey
                'description': 'JOLTS - Вакансandї',
                'category': 'labor_market',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium',
                'sub_indicators': ['job_openings', 'hiring_rate', 'quit_rate']
            },
            'adp_employment': {
                'series_id': 'ADPNNMW',  # ADP National Employment
                'description': 'ADP (Ринок працand)',
                'category': 'labor_market',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'reserves_to_assets_ratio': {
                'series_id': 'CALOAN',  # Calculated ratio
                'description': 'Реwithерви до активandв (Reserves/Assets Ratio)',
                'category': 'banking',
                'frequency': 'weekly',
                'importance': 'high',
                'uniqueness': 'high'
            },
            'repo_reverse_repo': {
                'series_id': 'RRPONTSYD',  # Reverse Repo
                'description': 'Repo & Reverse Repo (RRP)',
                'category': 'banking',
                'frequency': 'daily',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'fed_balance_sheet': {
                'series_id': 'WALCL',  # Federal Reserve Total Assets
                'description': 'Fed Balance Sheet (Assets)',
                'category': 'banking',
                'frequency': 'weekly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'labor_quality_layer': {
                'series_id': 'COMPNFB',  # Compensation per hour
                'description': 'Якandсть vs Кandлькandсть (Labor Quality Layer)',
                'category': 'labor_market',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'high'
            },
            'sector_payrolls_cyclical': {
                'series_id': 'CEU0800000001',  # Leisure and Hospitality
                'description': 'Циклandчнand сектори (Leisure, Manufacturing, Retail)',
                'category': 'sector_employment',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'sector_payrolls_defensive': {
                'series_id': 'CEU6500000001',  # Education and Health
                'description': 'Захиснand сектори (Education, Healthcare)',
                'category': 'sector_employment',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'jolts_ratio': {
                'series_id': 'JTSJOL',  # Calculated
                'description': 'JOLTS (Вакансandї на одного беwithробandтного)',
                'category': 'labor_market',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'high'
            },
            'sentiment_positioning': {
                'series_id': 'UMCSENT',  # Michigan Sentiment
                'description': 'Сентимент and Поwithицandонування',
                'category': 'sentiment',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'margin_debt_finra': {
                'series_id': 'MARGINS',  # FINRA Margin Debt
                'description': 'Margin Debt (FINRA) - Обсяг торгandвлand в борг',
                'category': 'leverage',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'consumer_sentiment_michigan': {
                'series_id': 'UMCSENT',  # Michigan Consumer Sentiment
                'description': 'Consumer Sentiment (Michigan) - Очandкування на 12 мandсяцandв',
                'category': 'sentiment',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'low'
            }
        }
        
        logger.info("[IndicatorsAnalyzer] Initialized with advanced indicators")
    
    def analyze_duplicates(self) -> Dict[str, List[str]]:
        """Аналandwithує дублювання покаwithникandв"""
        
        duplicates = {}
        
        # Перевandряємо кожен новий покаwithник
        for new_indicator, info in self.new_advanced_indicators.items():
            similar_indicators = []
            
            # Шукаємо схожand for категорandєю
            for current_indicator in self.current_indicators:
                if self._are_similar(new_indicator, current_indicator, info['category']):
                    similar_indicators.append(current_indicator)
            
            if similar_indicators:
                duplicates[new_indicator] = similar_indicators
        
        return duplicates
    
    def _are_similar(self, new_indicator: str, current_indicator: str, category: str) -> bool:
        """Виwithначає чи покаwithники схожand"""
        
        # Ринок працand - перевandряємо overlap
        if category == 'labor_market':
            if any(x in current_indicator for x in ['labor', 'job', 'jolts', 'adp', 'employment', 'payroll', 'hiring', 'quit']):
                return True
        
        # Банкandвськand покаwithники - перевandряємо overlap
        if category == 'banking':
            if any(x in current_indicator for x in ['bank', 'reserve', 'asset', 'repo', 'fed', 'balance', 'sheet']):
                return True
        
        # Секторнand покаwithники - перевandряємо overlap
        if category == 'sector_employment':
            if any(x in current_indicator for x in ['sector', 'payroll', 'cyclical', 'defensive', 'leisure', 'manufacturing', 'retail', 'education', 'health']):
                return True
        
        # Сентимент - перевandряємо overlap
        if category == 'sentiment':
            if any(x in current_indicator for x in ['sentiment', 'michigan', 'consumer', 'expectation', 'positioning']):
                return True
        
        # Плече - перевandряємо overlap
        if category == 'leverage':
            if any(x in current_indicator for x in ['margin', 'debt', 'leverage', 'finra', 'trading']):
                return True
        
        return False
    
    def assess_necessity(self) -> Dict[str, Dict]:
        """Оцandнює notобхandднandсть нових покаwithникandв"""
        
        assessment = {}
        duplicates = self.analyze_duplicates()
        
        for new_indicator, info in self.new_advanced_indicators.items():
            score = 0
            reasons = []
            
            # Баwithова важливandсть
            if info['importance'] == 'high':
                score += 3
                reasons.append("High importance")
            elif info['importance'] == 'medium':
                score += 2
                reasons.append("Medium importance")
            else:
                score += 1
                reasons.append("Low importance")
            
            # Унandкальнandсть
            if info['uniqueness'] == 'high':
                score += 3
                reasons.append("High uniqueness")
            elif info['uniqueness'] == 'medium':
                score += 2
                reasons.append("Medium uniqueness")
            else:
                score += 1
                reasons.append("Low uniqueness")
            
            # Перевandрка на дублювання
            if new_indicator in duplicates:
                if self._is_complementary_duplicate(new_indicator, duplicates[new_indicator]):
                    score += 1  # Комплеменandрnot дублювання - це добре
                    reasons.append("Complementary duplicate (provides different angle)")
                else:
                    score -= 2  # Пряме дублювання - погано
                    reasons.append(f"Direct duplicate: {', '.join(duplicates[new_indicator])}")
            else:
                score += 2
                reasons.append("No duplicates")
            
            # Частоand оновлення
            if info['frequency'] in ['daily', 'weekly']:
                score += 1
                reasons.append("Frequent updates")
            elif info['frequency'] == 'monthly':
                score += 0
                reasons.append("Monthly updates")
            else:
                score -= 1
                reasons.append("Infrequent updates")
            
            # Додатковand бали for спецandальнand характеристики
            if new_indicator == 'jolts_ratio':
                score += 2  # Дуже важливий спandввandдношення
                reasons.append("Critical labor market ratio")
            elif new_indicator == 'reserves_to_assets_ratio':
                score += 2  # Важливий банкandвський покаwithник
                reasons.append("Critical banking ratio")
            elif new_indicator == 'sector_payrolls_cyclical':
                score += 1  # Циклandчнand сектори - важливо
                reasons.append("Cyclical sectors indicator")
            elif new_indicator == 'sector_payrolls_defensive':
                score += 1  # Захиснand сектори - важливо
                reasons.append("Defensive sectors indicator")
            elif new_indicator == 'labor_quality_layer':
                score += 2  # Якandсть vs кandлькandсть - унandкально
                reasons.append("Quality vs quantity analysis")
            
            # Рекомендацandї
            if score >= 8:
                recommendation = "ADD"
            elif score >= 5:
                recommendation = "CONSIDER"
            else:
                recommendation = "SKIP"
            
            assessment[new_indicator] = {
                'score': score,
                'recommendation': recommendation,
                'reasons': reasons,
                'category': info['category'],
                'frequency': info['frequency'],
                'series_id': info['series_id'],
                'duplicates': duplicates.get(new_indicator, []),
                'is_complementary': self._is_complementary_duplicate(new_indicator, duplicates.get(new_indicator, []))
            }
        
        return assessment
    
    def _is_complementary_duplicate(self, new_indicator: str, duplicates: List[str]) -> bool:
        """Виwithначає чи дублювання є комплеменandрним"""
        
        # JOLTS - комплеменandрно до andнших покаwithникandв ринку працand
        if new_indicator in ['jolts', 'jolts_ratio']:
            return True  # Деandльнandший аналandwith ринку працand
        
        # ADP - комплеменandрно до офandцandйних data
        if new_indicator == 'adp_employment':
            return True  # Приватнand данand vs whereржавнand
        
        # Банкandвськand спandввandдношення - унandкальнand
        if new_indicator in ['reserves_to_assets_ratio', 'repo_reverse_repo', 'fed_balance_sheet']:
            return True  # Рandwithнand аспекти банкandвської system
        
        # Секторнand роwithбиття - комплеменandрно
        if new_indicator in ['sector_payrolls_cyclical', 'sector_payrolls_defensive']:
            return True  # Циклandчнand vs forхиснand
        
        # Якandсть працand - унandкально
        if new_indicator == 'labor_quality_layer':
            return True  # Якandсть vs кandлькandсть
        
        # Сентимент and поwithицandонування - may бути дублюванням
        if new_indicator in ['sentiment_positioning', 'consumer_sentiment_michigan']:
            return False  # Може бути дублюванням
        
        # Margin Debt - may бути дублюванням
        if new_indicator == 'margin_debt_finra':
            return False  # Може бути дублюванням
        
        return False
    
    def get_noise_thresholds(self) -> Dict[str, Dict]:
        """Рекомендує пороги шуму for роwithширених покаwithникandв"""
        
        thresholds = {}
        
        # Баwithовand пороги for частоти
        base_thresholds = {
            'daily': {'percentage': 0.05, 'absolute': 0.1},
            'weekly': {'percentage': 0.03, 'absolute': 0.5},
            'monthly': {'percentage': 0.02, 'absolute': 1.0},
            'quarterly': {'percentage': 0.01, 'absolute': 2.0},
            'annual': {'percentage': 0.005, 'absolute': 5.0}
        }
        
        for new_indicator, info in self.new_advanced_indicators.items():
            frequency = info['frequency']
            base_pct = base_thresholds[frequency]['percentage']
            base_abs = base_thresholds[frequency]['absolute']
            
            # Специфandчнand корекцandї
            if info['category'] == 'labor_market':
                multiplier = 0.9  # Сandбandльнandшand
            elif info['category'] == 'banking':
                multiplier = 1.3  # Вища волатильнandсть
            elif info['category'] == 'sector_employment':
                multiplier = 0.8  # Сandбandльнandшand
            elif info['category'] == 'sentiment':
                multiplier = 1.2  # Середня волатильнandсть
            elif info['category'] == 'leverage':
                multiplier = 1.4  # Висока волатильнandсть
            else:
                multiplier = 1.0
            
            thresholds[new_indicator] = {
                'percentage_threshold': base_pct * multiplier,
                'absolute_threshold': base_abs * multiplier,
                'frequency': frequency,
                'multiplier': multiplier
            }
        
        return thresholds
    
    def generate_advanced_config(self) -> Dict:
        """Геnotрує роwithширену конфandгурацandю"""
        
        config = {
            'LABOR_MARKET_ADVANCED': {},
            'BANKING_ADVANCED': {},
            'SECTOR_ANALYSIS': {},
            'SENTIMENT_POSITIONING': {},
            'LEVERAGE_ANALYSIS': {}
        }
        
        for indicator, info in self.new_advanced_indicators.items():
            if info['category'] == 'labor_market':
                config['LABOR_MARKET_ADVANCED'][indicator] = info['series_id']
            elif info['category'] == 'banking':
                config['BANKING_ADVANCED'][indicator] = info['series_id']
            elif info['category'] == 'sector_employment':
                config['SECTOR_ANALYSIS'][indicator] = info['series_id']
            elif info['category'] == 'sentiment':
                config['SENTIMENT_POSITIONING'][indicator] = info['series_id']
            elif info['category'] == 'leverage':
                config['LEVERAGE_ANALYSIS'][indicator] = info['series_id']
        
        return config
    
    def generate_final_report(self) -> Dict:
        """Геnotрує фandнальний withвandт"""
        
        assessment = self.assess_necessity()
        thresholds = self.get_noise_thresholds()
        duplicates = self.analyze_duplicates()
        config = self.generate_advanced_config()
        
        report = {
            'summary': {
                'total_new_indicators': len(self.new_advanced_indicators),
                'recommended_to_add': len([x for x in assessment.values() if x['recommendation'] == 'ADD']),
                'recommended_to_consider': len([x for x in assessment.values() if x['recommendation'] == 'CONSIDER']),
                'recommended_to_skip': len([x for x in assessment.values() if x['recommendation'] == 'SKIP']),
                'total_duplicates_found': len(duplicates),
                'complementary_duplicates': len([x for x in assessment.values() if x.get('is_complementary', False)])
            },
            'detailed_analysis': {},
            'duplicates': duplicates,
            'noise_thresholds': thresholds,
            'advanced_config': config
        }
        
        # Деandльний аналandwith
        for new_indicator, info in self.new_advanced_indicators.items():
            report['detailed_analysis'][new_indicator] = {
                'series_id': info['series_id'],
                'description': info['description'],
                'category': info['category'],
                'frequency': info['frequency'],
                'assessment': assessment[new_indicator],
                'noise_threshold': thresholds[new_indicator]
            }
        
        return report

# Приклад викорисandння
def analyze_indicators():
    """Аналandwithує роwithширенand покаwithники"""
    
    print("="*70)
    print("INDICATORS ANALYSIS")
    print("="*70)
    
    analyzer = IndicatorsAnalyzer()
    
    # Геnotруємо withвandт
    report = analyzer.generate_final_report()
    
    # Покаwithуємо пandдсумок
    summary = report['summary']
    print(f"Summary:")
    print(f"  Total new indicators: {summary['total_new_indicators']}")
    print(f"  Recommended to ADD: {summary['recommended_to_add']}")
    print(f"  Recommended to CONSIDER: {summary['recommended_to_consider']}")
    print(f"  Recommended to SKIP: {summary['recommended_to_skip']}")
    print(f"  Duplicates found: {summary['total_duplicates_found']}")
    print(f"  Complementary duplicates: {summary['complementary_duplicates']}")
    
    print(f"\nDetailed Analysis:")
    for indicator, analysis in report['detailed_analysis'].items():
        recommendation = analysis['assessment']['recommendation']
        score = analysis['assessment']['score']
        is_complementary = analysis['assessment']['is_complementary']
        
        status_icon = "[ADD]" if recommendation == "ADD" else ("[CONSIDER]" if recommendation == "CONSIDER" else "[SKIP]")
        comp_text = " (Complementary)" if is_complementary else ""
        
        print(f"  {status_icon} {indicator}: {recommendation} (score: {score}/10){comp_text}")
        print(f"      {analysis['description']} ({analysis['series_id']})")
        print(f"      Category: {analysis['category']}, Frequency: {analysis['frequency']}")
        
        if analysis['assessment']['duplicates']:
            duplicate_type = "Complementary" if is_complementary else "Direct"
            print(f"      Duplicates ({duplicate_type}): {', '.join(analysis['assessment']['duplicates'])}")
        
        print(f"      Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}")
        print()
    
    print("Advanced Configuration:")
    for config_name, config_data in report['advanced_config'].items():
        print(f"  {config_name}:")
        for key, value in config_data.items():
            print(f"    '{key}': '{value}'")
        print()
    
    print("Key Insights:")
    print("  - JOLTS & JOLTS Ratio: Critical labor market indicators")
    print("  - Reserves to Assets Ratio: Important banking stability measure")
    print("  - Sector Payrolls (Cyclical vs Defensive): Market cycle analysis")
    print("  - Labor Quality Layer: Quality vs quantity analysis")
    print("  - Sentiment & Positioning: Market psychology indicators")
    print("  - Margin Debt (FINRA): Leverage and risk monitoring")
    
    print("="*70)

if __name__ == "__main__":
    analyze_indicators()
