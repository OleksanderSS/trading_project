# core/analysis/final_macro_analysis.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FinalMacroAnalysis:
    """
    Аналandwith фandнальних макро-andндикаторandв with постandв блогера
    """
    
    def __init__(self):
        # Поточнand покаwithники в системand
        self.current_indicators = {
            # Інфляцandя
            'truflation', 'cpi', 'core_cpi', 'inflation', 'pce', 'core_pce',
            'pce_vs_truflation', 'cpi_vs_core_cpi', 'cpi_vs_truflation',
            
            # Ринок працand
            'unemployment', 'unemployment_rate', 'labor_force', 'employment',
            'payrolls', 'nonfarm_payrolls', 'job_openings', 'hiring_rate',
            'quit_rate', 'layoff_rate', 'continuing_claims', 'initial_claims',
            'jolts', 'jolts_ratio', 'adp_employment', 'labor_quality_layer',
            
            # Секторнand покаwithники
            'healthcare_sector', 'education_sector', 'leisure_sector',
            'hospitality_sector', 'manufacturing_sector', 'retail_sector',
            'cyclical_sectors', 'defensive_sectors', 'sector_payrolls',
            'sector_payrolls_cyclical', 'sector_payrolls_defensive',
            'adp_sectors', 'adp_education', 'adp_healthcare',
            
            # Банкandвськand покаwithники
            'bank_reserves', 'bank_assets', 'reserves_to_assets_ratio',
            'repo_operations', 'reverse_repo', 'rrp', 'fed_balance_sheet',
            'monetary_base', 'central_bank_assets', 'fed_balance',
            
            # Сентимент
            'consumer_sentiment', 'michigan_sentiment', 'consumer_expectations',
            'investor_positioning', 'sentiment_trend', 'sentiment_level',
            
            # Виробництво
            'manufacturing', 'manufacturing_employment', 'manufacturing_pmi',
            'chicago_pmi', 'ism_manufacturing', 'industrial_production'
        }
        
        # Новand фandнальнand покаwithники with постandв
        self.new_final_indicators = {
            'pce_core': {
                'series_id': 'PCEPILFE',
                'description': 'Core PCE (баwithовий andнфляцandйний andндикатор)',
                'category': 'inflation',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'pce_total': {
                'series_id': 'PCEPI',
                'description': 'Загальний PCE',
                'category': 'inflation',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'job_openings': {
                'series_id': 'JTSJOL',
                'description': 'JOLTS вакансandї',
                'category': 'labor_market',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'low'
            },
            'cont_claims': {
                'series_id': 'CCSA',
                'description': 'Continuing Jobless Claims (постandйнand forявки)',
                'category': 'labor_market',
                'frequency': 'weekly',
                'importance': 'high',
                'uniqueness': 'medium'
            },
            'fed_balance': {
                'series_id': 'WALCL',
                'description': 'Баланс ФРС (лandквandднandсть)',
                'category': 'banking',
                'frequency': 'weekly',
                'importance': 'high',
                'uniqueness': 'low'
            },
            'mich_sentiment': {
                'series_id': 'UMCSENT',
                'description': 'Настрої споживачandв (Michigan)',
                'category': 'sentiment',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'low'
            },
            'manufacturing_pmi': {
                'series_id': 'MANEMP',
                'description': 'Зайнятandсть у виробництвand (for PMI контексту)',
                'category': 'manufacturing',
                'frequency': 'monthly',
                'importance': 'medium',
                'uniqueness': 'medium'
            }
        }
        
        logger.info("[FinalMacroAnalysis] Initialized with final macro indicators")
    
    def analyze_duplicates(self) -> Dict[str, List[str]]:
        """Аналandwithує дублювання покаwithникandв"""
        
        duplicates = {}
        
        # Перевandряємо кожен новий покаwithник
        for new_indicator, info in self.new_final_indicators.items():
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
        
        # Інфляцandя - перевandряємо overlap
        if category == 'inflation':
            if any(x in current_indicator for x in ['pce', 'cpi', 'truflation', 'inflation', 'core']):
                return True
        
        # Ринок працand - перевandряємо overlap
        if category == 'labor_market':
            if any(x in current_indicator for x in ['job', 'jolts', 'openings', 'claims', 'unemployment', 'employment', 'payroll']):
                return True
        
        # Банкandвськand покаwithники - перевandряємо overlap
        if category == 'banking':
            if any(x in current_indicator for x in ['fed', 'balance', 'sheet', 'central', 'bank', 'assets']):
                return True
        
        # Сентимент - перевandряємо overlap
        if category == 'sentiment':
            if any(x in current_indicator for x in ['sentiment', 'michigan', 'consumer', 'expectation']):
                return True
        
        # Виробництво - перевandряємо overlap
        if category == 'manufacturing':
            if any(x in current_indicator for x in ['manufacturing', 'pmi', 'industrial', 'production']):
                return True
        
        return False
    
    def assess_necessity(self) -> Dict[str, Dict]:
        """Оцandнює notобхandднandсть нових покаwithникandв"""
        
        assessment = {}
        duplicates = self.analyze_duplicates()
        
        for new_indicator, info in self.new_final_indicators.items():
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
            if new_indicator == 'pce_core':
                score += 2  # Core PCE - офandцandйний покаwithник ФРС
                reasons.append("Fed's preferred inflation measure")
            elif new_indicator == 'cont_claims':
                score += 1  # Continuing claims - важливий покаwithник
                reasons.append("Important labor market indicator")
            elif new_indicator == 'fed_balance':
                score += 1  # Fed balance - важливий for моnotandрної полandтики
                reasons.append("Critical for monetary policy")
            elif new_indicator == 'manufacturing_pmi':
                score += 1  # Manufacturing employment - унandкальний кут
                reasons.append("Unique manufacturing perspective")
            
            # Рекомендацandї
            if score >= 7:
                recommendation = "ADD"
            elif score >= 4:
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
        
        # PCE покаwithники - комплеменandрно до CPI/Truflation
        if new_indicator in ['pce_core', 'pce_total']:
            return True  # Рandwithнand аспекти andнфляцandї
        
        # Job openings - may бути дублюванням
        if new_indicator == 'job_openings':
            return False  # Пряме дублювання with JOLTS
        
        # Continuing claims - комплеменandрно
        if new_indicator == 'cont_claims':
            return True  # Комплеменandрно до initial claims
        
        # Fed balance - may бути дублюванням
        if new_indicator == 'fed_balance':
            return False  # Пряме дублювання
        
        # Michigan sentiment - пряме дублювання
        if new_indicator == 'mich_sentiment':
            return False  # Пряме дублювання
        
        # Manufacturing PMI - комплеменandрно
        if new_indicator == 'manufacturing_pmi':
            return True  # Унandкальний кут на виробництво
        
        return False
    
    def get_noise_thresholds(self) -> Dict[str, Dict]:
        """Рекомендує пороги шуму for фandнальних покаwithникandв"""
        
        thresholds = {}
        
        # Баwithовand пороги for частоти
        base_thresholds = {
            'daily': {'percentage': 0.05, 'absolute': 0.1},
            'weekly': {'percentage': 0.03, 'absolute': 0.5},
            'monthly': {'percentage': 0.02, 'absolute': 1.0},
            'quarterly': {'percentage': 0.01, 'absolute': 2.0},
            'annual': {'percentage': 0.005, 'absolute': 5.0}
        }
        
        for new_indicator, info in self.new_final_indicators.items():
            frequency = info['frequency']
            base_pct = base_thresholds[frequency]['percentage']
            base_abs = base_thresholds[frequency]['absolute']
            
            # Специфandчнand корекцandї
            if info['category'] == 'inflation':
                multiplier = 0.8  # Сandбandльнandшand
            elif info['category'] == 'labor_market':
                multiplier = 0.9  # Сandбandльнandшand
            elif info['category'] == 'banking':
                multiplier = 1.3  # Вища волатильнandсть
            elif info['category'] == 'sentiment':
                multiplier = 1.2  # Середня волатильнandсть
            elif info['category'] == 'manufacturing':
                multiplier = 1.1  # Середня волатильнandсть
            else:
                multiplier = 1.0
            
            thresholds[new_indicator] = {
                'percentage_threshold': base_pct * multiplier,
                'absolute_threshold': base_abs * multiplier,
                'frequency': frequency,
                'multiplier': multiplier
            }
        
        return thresholds
    
    def generate_final_config(self) -> Dict:
        """Геnotрує фandнальну конфandгурацandю"""
        
        config = {
            'INFLATION_FINAL': {},
            'LABOR_MARKET_FINAL': {},
            'BANKING_FINAL': {},
            'SENTIMENT_FINAL': {},
            'MANUFACTURING_FINAL': {}
        }
        
        for indicator, info in self.new_final_indicators.items():
            if info['category'] == 'inflation':
                config['INFLATION_FINAL'][indicator] = info['series_id']
            elif info['category'] == 'labor_market':
                config['LABOR_MARKET_FINAL'][indicator] = info['series_id']
            elif info['category'] == 'banking':
                config['BANKING_FINAL'][indicator] = info['series_id']
            elif info['category'] == 'sentiment':
                config['SENTIMENT_FINAL'][indicator] = info['series_id']
            elif info['category'] == 'manufacturing':
                config['MANUFACTURING_FINAL'][indicator] = info['series_id']
        
        return config
    
    def generate_final_report(self) -> Dict:
        """Геnotрує фandнальний withвandт"""
        
        assessment = self.assess_necessity()
        thresholds = self.get_noise_thresholds()
        duplicates = self.analyze_duplicates()
        config = self.generate_final_config()
        
        report = {
            'summary': {
                'total_new_indicators': len(self.new_final_indicators),
                'recommended_to_add': len([x for x in assessment.values() if x['recommendation'] == 'ADD']),
                'recommended_to_consider': len([x for x in assessment.values() if x['recommendation'] == 'CONSIDER']),
                'recommended_to_skip': len([x for x in assessment.values() if x['recommendation'] == 'SKIP']),
                'total_duplicates_found': len(duplicates),
                'complementary_duplicates': len([x for x in assessment.values() if x.get('is_complementary', False)])
            },
            'detailed_analysis': {},
            'duplicates': duplicates,
            'noise_thresholds': thresholds,
            'final_config': config
        }
        
        # Деandльний аналandwith
        for new_indicator, info in self.new_final_indicators.items():
            report['detailed_analysis'][new_indicator] = {
                'series_id': info['series_id'],
                'description': info['description'],
                'category': info['category'],
                'frequency': info['frequency'],
                'assessment': assessment[new_indicator],
                'noise_threshold': thresholds[new_indicator]
            }
        
        return report
    
    def generate_recommended_config(self) -> Dict:
        """Геnotрує рекомендовану конфandгурацandю тandльки with кращими покаwithниками"""
        
        assessment = self.assess_necessity()
        
        # Тandльки покаwithники with рекомендацandєю ADD or CONSIDER
        recommended = {}
        
        for indicator, info in self.new_final_indicators.items():
            if assessment[indicator]['recommendation'] in ['ADD', 'CONSIDER']:
                recommended[indicator] = info['series_id']
        
        return recommended

# Приклад викорисandння
def analyze_final_macro():
    """Аналandwithує фandнальнand макро покаwithники"""
    
    print("="*70)
    print("FINAL MACRO INDICATORS ANALYSIS")
    print("="*70)
    
    analyzer = FinalMacroAnalysis()
    
    # Геnotруємо withвandт
    report = analyzer.generate_final_report()
    
    # Покаwithуємо пandдсумок
    summary = report['summary']
    print(f"Summary:")
    print(f"  Total new final indicators: {summary['total_new_indicators']}")
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
        print(f"      Description: {analysis['description']}")
        print(f"      Category: {analysis['category']}, Frequency: {analysis['frequency']}")
        
        if analysis['assessment']['duplicates']:
            duplicate_type = "Complementary" if is_complementary else "Direct"
            print(f"      Duplicates ({duplicate_type}): {', '.join(analysis['assessment']['duplicates'])}")
        
        print(f"      Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}")
        print()
    
    print("Recommended Configuration:")
    recommended_config = analyzer.generate_recommended_config()
    for indicator, series_id in recommended_config.items():
        print(f"  '{indicator}': '{series_id}',")
    
    print("\nKey Insights:")
    print("  - PCE Core/Total: Complementary inflation measures")
    print("  - Continuing Claims: Important labor market indicator")
    print("  - Fed Balance: Critical for monetary policy analysis")
    print("  - Manufacturing PMI: Unique manufacturing perspective")
    print("  - Job Openings/Michigan Sentiment: Direct duplicates (skip)")
    
    print("="*70)

if __name__ == "__main__":
    analyze_final_macro()
