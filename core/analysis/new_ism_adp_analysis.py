# core/analysis/new_ism_adp_analysis.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NewISMADPAnalysis:
    """
    Аналandwith нових покаwithникandв: ISM Manufacturing PMI, ISM Services PMI, Consumer Confidence, ADP Employment
    """
    
    def __init__(self):
        # Поточнand покаwithники в системand
        self.current_indicators = {
            # PMI покаwithники
            'chicago_pmi', 'manufacturing_pmi', 'services_pmi', 'ism_manufacturing',
            'ism_services', 'pmi_spread', 'pmi_divergence', 'business_activity',
            
            # Ринок працand
            'labor_market', 'employment', 'job_openings', 'hiring_rate', 'quit_rate', 'layoff_rate',
            'labor_sentiment', 'labor_differential', 'labor_market_tightness', 'adp_employment',
            
            # Сентимент
            'consumer_confidence', 'consumer_expectations', 'consumer_sentiment',
            'michigan_sentiment', 'sentiment_trend', 'sentiment_level',
            
            # Інфляцandя and економandка
            'inflation', 'gdp', 'economic_activity', 'business_confidence'
        }
        
        # Новand покаwithники
        self.new_indicators = {
            'ism_manufacturing_pmi': {
                'description': 'ISM Manufacturing PMI (Institute for Supply Management)',
                'category': 'manufacturing_activity',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'low',
                'threshold': 50.0,
                'signal_type': 'expansion_contraction',
                'source': 'ISM',
                'components': ['new_orders', 'production', 'employment', 'supplier_deliveries', 'inventories']
            },
            'ism_services_pmi': {
                'description': 'ISM Services PMI (Institute for Supply Management)',
                'category': 'services_activity',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium',
                'threshold': 50.0,
                'signal_type': 'expansion_contraction',
                'source': 'ISM',
                'components': ['business_activity', 'new_orders', 'employment', 'prices']
            },
            'consumer_confidence_index': {
                'description': 'Consumer Confidence Index (Conference Board)',
                'category': 'consumer_sentiment',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium',
                'threshold': 100.0,
                'signal_type': 'optimism_pessimism',
                'source': 'Conference Board',
                'components': ['present_situation', 'expectations']
            },
            'adp_employment_change': {
                'description': 'ADP Employment Change (Automatic Data Processing)',
                'category': 'employment_change',
                'frequency': 'monthly',
                'importance': 'high',
                'uniqueness': 'medium',
                'threshold': 0.0,
                'signal_type': 'job_growth_decline',
                'source': 'ADP',
                'components': ['private_sector_jobs', 'small_business', 'medium_business', 'large_business']
            }
        }
        
        logger.info("[NewISMADPAnalysis] Initialized with ISM and ADP indicators")
    
    def analyze_duplicates(self) -> Dict[str, List[str]]:
        """Аналandwithує дублювання покаwithникandв"""
        
        duplicates = {}
        
        # Перевandряємо кожен новий покаwithник
        for new_indicator, info in self.new_indicators.items():
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
        
        # Manufacturing PMI - перевandряємо overlap
        if category == 'manufacturing_activity':
            if any(x in current_indicator for x in ['manufacturing', 'pmi', 'ism', 'production']):
                return True
        
        # Services PMI - перевandряємо overlap
        if category == 'services_activity':
            if any(x in current_indicator for x in ['services', 'pmi', 'ism', 'business_activity']):
                return True
        
        # Consumer Confidence - перевandряємо overlap
        if category == 'consumer_sentiment':
            if any(x in current_indicator for x in ['confidence', 'sentiment', 'consumer', 'expectation']):
                return True
        
        # ADP Employment - перевandряємо overlap
        if category == 'employment_change':
            if any(x in current_indicator for x in ['adp', 'employment', 'jobs', 'hiring']):
                return True
        
        return False
    
    def assess_necessity(self) -> Dict[str, Dict]:
        """Оцandнює notобхandднandсть нових покаwithникandв"""
        
        assessment = {}
        duplicates = self.analyze_duplicates()
        
        for new_indicator, info in self.new_indicators.items():
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
            if new_indicator == 'ism_manufacturing_pmi':
                score += 1  # Офandцandйний ISM покаwithник
                reasons.append("Official ISM manufacturing indicator")
            elif new_indicator == 'ism_services_pmi':
                score += 2  # Важливий for сектору послуг
                reasons.append("Critical services sector indicator")
            elif new_indicator == 'consumer_confidence_index':
                score += 2  # Conference Board - авторитетnot джерело
                reasons.append("Conference Board authority")
            elif new_indicator == 'adp_employment_change':
                score += 1  # ADP - авторитетnot джерело
                reasons.append("ADP employment authority")
            
            # Додатковand бали for компоnotнти
            if len(info.get('components', [])) >= 4:
                score += 1
                reasons.append("Rich component structure")
            
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
                'duplicates': duplicates.get(new_indicator, []),
                'is_complementary': self._is_complementary_duplicate(new_indicator, duplicates.get(new_indicator, [])),
                'source': info['source'],
                'components': info.get('components', [])
            }
        
        return assessment
    
    def _is_complementary_duplicate(self, new_indicator: str, duplicates: List[str]) -> bool:
        """Виwithначає чи дублювання є комплеменandрним"""
        
        # ISM Manufacturing PMI - may бути дублюванням
        if new_indicator == 'ism_manufacturing_pmi':
            return False  # Пряме дублювання with manufacturing_pmi, ism_manufacturing
        
        # ISM Services PMI - комплеменandрно
        if new_indicator == 'ism_services_pmi':
            return True  # Комплеменandрно до services_pmi, ism_services
        
        # Consumer Confidence - комплеменandрно
        if new_indicator == 'consumer_confidence_index':
            return True  # Комплеменandрно до consumer_confidence
        
        # ADP Employment - комплеменandрно
        if new_indicator == 'adp_employment_change':
            return True  # Комплеменandрно до adp_employment
        
        return False
    
    def get_noise_thresholds(self) -> Dict[str, Dict]:
        """Рекомендує пороги шуму for покаwithникandв"""
        
        thresholds = {}
        
        # Баwithовand пороги for частоти
        base_thresholds = {
            'daily': {'percentage': 0.05, 'absolute': 0.1},
            'weekly': {'percentage': 0.03, 'absolute': 0.5},
            'monthly': {'percentage': 0.02, 'absolute': 1.0},
            'quarterly': {'percentage': 0.01, 'absolute': 2.0},
            'annual': {'percentage': 0.005, 'absolute': 5.0}
        }
        
        for new_indicator, info in self.new_indicators.items():
            frequency = info['frequency']
            base_pct = base_thresholds[frequency]['percentage']
            base_abs = base_thresholds[frequency]['absolute']
            
            # Специфandчнand корекцandї
            if info['category'] in ['manufacturing_activity', 'services_activity']:
                multiplier = 0.8  # PMI покаwithники сandбandльнandшand
            elif info['category'] == 'consumer_sentiment':
                multiplier = 1.1  # Сентимент бandльш волатильний
            elif info['category'] == 'employment_change':
                multiplier = 1.2  # Зайнятandсть волатильна
            else:
                multiplier = 1.0
            
            thresholds[new_indicator] = {
                'percentage_threshold': base_pct * multiplier,
                'absolute_threshold': base_abs * multiplier,
                'frequency': frequency,
                'multiplier': multiplier
            }
        
        return thresholds
    
    def generate_ism_adp_config(self) -> Dict:
        """Геnotрує ISM/ADP конфandгурацandю"""
        
        config = {
            'MANUFACTURING_ACTIVITY': {},
            'SERVICES_ACTIVITY': {},
            'CONSUMER_SENTIMENT': {},
            'EMPLOYMENT_CHANGE': {}
        }
        
        for indicator, info in self.new_indicators.items():
            if info['category'] == 'manufacturing_activity':
                config['MANUFACTURING_ACTIVITY'][indicator] = info.get('series_id', 'ISM_MANUFACTURING_PMI')
            elif info['category'] == 'services_activity':
                config['SERVICES_ACTIVITY'][indicator] = info.get('series_id', 'ISM_SERVICES_PMI')
            elif info['category'] == 'consumer_sentiment':
                config['CONSUMER_SENTIMENT'][indicator] = info.get('series_id', 'CONSUMER_CONFIDENCE')
            elif info['category'] == 'employment_change':
                config['EMPLOYMENT_CHANGE'][indicator] = info.get('series_id', 'ADP_EMPLOYMENT')
        
        return config
    
    def generate_final_report(self) -> Dict:
        """Геnotрує фandнальний withвandт"""
        
        assessment = self.assess_necessity()
        thresholds = self.get_noise_thresholds()
        duplicates = self.analyze_duplicates()
        config = self.generate_ism_adp_config()
        
        report = {
            'summary': {
                'total_new_indicators': len(self.new_indicators),
                'recommended_to_add': len([x for x in assessment.values() if x['recommendation'] == 'ADD']),
                'recommended_to_consider': len([x for x in assessment.values() if x['recommendation'] == 'CONSIDER']),
                'recommended_to_skip': len([x for x in assessment.values() if x['recommendation'] == 'SKIP']),
                'total_duplicates_found': len(duplicates),
                'complementary_duplicates': len([x for x in assessment.values() if x.get('is_complementary', False)])
            },
            'detailed_analysis': {},
            'duplicates': duplicates,
            'noise_thresholds': thresholds,
            'ism_adp_config': config
        }
        
        # Деandльний аналandwith
        for new_indicator, info in self.new_indicators.items():
            report['detailed_analysis'][new_indicator] = {
                'description': info['description'],
                'category': info['category'],
                'frequency': info['frequency'],
                'source': info['source'],
                'assessment': assessment[new_indicator],
                'noise_threshold': thresholds[new_indicator],
                'threshold': info.get('threshold', 'N/A'),
                'components': info.get('components', [])
            }
        
        return report

# Приклад викорисandння
def analyze_new_ism_adp():
    """Аналandwithує новand ISM and ADP покаwithники"""
    
    print("="*70)
    print("NEW ISM & ADP INDICATORS ANALYSIS")
    print("="*70)
    
    analyzer = NewISMADPAnalysis()
    
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
        print(f"      Description: {analysis['description']}")
        print(f"      Category: {analysis['category']}, Frequency: {analysis['frequency']}")
        print(f"      Source: {analysis['source']}")
        
        if analysis['threshold'] != 'N/A':
            print(f"      Threshold: {analysis['threshold']}")
        
        if analysis['components']:
            print(f"      Components: {', '.join(analysis['components'])}")
        
        if analysis['assessment']['duplicates']:
            duplicate_type = "Complementary" if is_complementary else "Direct"
            print(f"      Duplicates ({duplicate_type}): {', '.join(analysis['assessment']['duplicates'])}")
        
        print(f"      Noise threshold: {analysis['noise_threshold']['percentage_threshold']:.2%}")
        print()
    
    print("ISM/ADP Configuration:")
    ism_adp_config = report['ism_adp_config']
    for category, indicators in ism_adp_config.items():
        print(f"  {category}:")
        for indicator, series_id in indicators.items():
            print(f"    {indicator}: {series_id}")
    
    print("\nKey Insights:")
    print("  - ISM Manufacturing: Official manufacturing indicator")
    print("  - ISM Services: Critical services sector indicator")
    print("  - Consumer Confidence: Conference Board authority")
    print("  - ADP Employment: Private sector employment data")
    
    print("="*70)

if __name__ == "__main__":
    analyze_new_ism_adp()
